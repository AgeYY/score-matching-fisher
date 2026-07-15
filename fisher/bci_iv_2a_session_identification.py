"""Within-recording identification from time-resolved BCI IV-2a RDMs."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from fisher.bci_iv_2a_dataset import BCIIV2aFeatures, CLASS_NAMES
from fisher.flow_matching_skl import build_flow_skl_model, train_flow_skl_model


N_CLASSES = len(CLASS_NAMES)
QUERY_RUNS = (0, 2, 4)
REFERENCE_RUNS = (1, 3, 5)


@dataclass(frozen=True)
class FlowRDMConfig:
    hidden_dim: int = 64
    depth: int = 2
    epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 15
    quadrature_steps: int = 32
    covariance_ode_steps: int = 48
    covariance_ridge: float = 1e-5
    validation_fraction: float = 0.2


def condition_design(labels: np.ndarray, time_centers: np.ndarray) -> np.ndarray:
    """Encode class as one-hot and physical time as one scaled coordinate."""

    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    times = np.asarray(time_centers, dtype=np.float64).reshape(-1)
    if np.any((y < 0) | (y >= N_CLASSES)):
        raise ValueError("Labels must be in [0, 3].")
    scale = float(np.max(np.abs(times)))
    time_scaled = times / max(scale, np.finfo(np.float64).eps)
    one_hot = np.eye(N_CLASSES, dtype=np.float64)[y]
    return np.concatenate([one_hot, time_scaled[:, None]], axis=1)


def select_half(features: BCIIV2aFeatures, run_ids: Iterable[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    runs = np.asarray(tuple(int(value) for value in run_ids), dtype=np.int64)
    mask = np.isin(features.run_ids, runs)
    return features.features[mask], features.labels[mask], features.run_ids[mask]


def per_class_counts(labels: np.ndarray) -> np.ndarray:
    return np.bincount(np.asarray(labels, dtype=np.int64), minlength=N_CLASSES)


def subsample_balanced_trials(labels: np.ndarray, n_per_class: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    for label in range(N_CLASSES):
        candidates = np.flatnonzero(labels == label)
        if candidates.size < int(n_per_class):
            raise ValueError(f"Class {label} has {candidates.size} trials, fewer than n={n_per_class}.")
        selected.append(np.sort(rng.choice(candidates, size=int(n_per_class), replace=False)))
    return np.sort(np.concatenate(selected))


def standardize_trial_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(x, dtype=np.float64)
    flat = values.reshape(-1, values.shape[-1])
    mean = np.mean(flat, axis=0)
    scale = np.std(flat, axis=0, ddof=1)
    scale = np.maximum(scale, 1e-8)
    return (values - mean) / scale, mean, scale


def classical_mahalanobis_rdms(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fit a Ledoit-Wolf pooled within-class covariance at each time window."""

    values, _, _ = standardize_trial_features(x)
    labels = np.asarray(labels, dtype=np.int64)
    n_times, n_features = int(values.shape[1]), int(values.shape[2])
    rdms = np.zeros((n_times, N_CLASSES, N_CLASSES), dtype=np.float64)
    for time_index in range(n_times):
        means = np.stack([np.mean(values[labels == label, time_index], axis=0) for label in range(N_CLASSES)])
        residuals = np.concatenate(
            [values[labels == label, time_index] - means[label] for label in range(N_CLASSES)], axis=0
        )
        precision = LedoitWolf(assume_centered=True).fit(residuals).precision_
        for left in range(N_CLASSES):
            for right in range(left + 1, N_CLASSES):
                delta = means[left] - means[right]
                distance = max(0.0, float(delta @ precision @ delta))
                rdms[time_index, left, right] = distance
                rdms[time_index, right, left] = distance
    return rdms


def _stratified_validation_trials(labels: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    val: list[int] = []
    for label in range(N_CLASSES):
        candidates = np.flatnonzero(labels == label)
        n_val = min(candidates.size - 1, max(1, int(round(float(fraction) * candidates.size))))
        val.extend(int(value) for value in rng.choice(candidates, size=n_val, replace=False))
    val_idx = np.asarray(sorted(val), dtype=np.int64)
    train_idx = np.setdiff1d(np.arange(labels.size, dtype=np.int64), val_idx)
    return train_idx, val_idx


@torch.no_grad()
def _shared_endpoint_covariance(
    model: torch.nn.Module,
    *,
    device: torch.device,
    steps: int,
    ridge: float,
) -> np.ndarray:
    """Integrate dSigma/dt = A Sigma + Sigma A^T from the N(0,I) base."""

    model.eval()
    x_dim = int(getattr(model, "x_dim"))
    sigma = np.eye(x_dim, dtype=np.float64)
    dt = 1.0 / float(steps)
    dtype = next(model.parameters()).dtype
    for step in range(int(steps)):
        t = torch.full((1, 1), (step + 0.5) * dt, dtype=dtype, device=device)
        a = model.A(t).detach().cpu().numpy().reshape(x_dim, x_dim).astype(np.float64)
        sigma = sigma + dt * (a @ sigma + sigma @ a.T)
        sigma = 0.5 * (sigma + sigma.T)
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    eigenvalues = np.maximum(eigenvalues, float(ridge))
    return (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T


@torch.no_grad()
def _time_conditioned_endpoint_covariances(
    model: torch.nn.Module,
    conditions: np.ndarray,
    *,
    device: torch.device,
    steps: int,
    ridge: float,
) -> np.ndarray:
    """Integrate one endpoint covariance per physical EEG time condition."""

    model.eval()
    x_dim = int(getattr(model, "x_dim"))
    theta = torch.from_numpy(np.asarray(conditions, dtype=np.float32)).to(
        device=device, dtype=next(model.parameters()).dtype
    )
    n_conditions = int(theta.shape[0])
    sigma = np.broadcast_to(np.eye(x_dim, dtype=np.float64), (n_conditions, x_dim, x_dim)).copy()
    dt = 1.0 / float(steps)
    for step in range(int(steps)):
        t = torch.full(
            (n_conditions, 1),
            (step + 0.5) * dt,
            dtype=theta.dtype,
            device=device,
        )
        a = model.A(theta, t).detach().cpu().numpy().astype(np.float64)
        sigma = sigma + dt * (
            np.matmul(a, sigma) + np.matmul(sigma, np.swapaxes(a, -1, -2))
        )
        sigma = 0.5 * (sigma + np.swapaxes(sigma, -1, -2))
    stabilized = np.empty_like(sigma)
    for index in range(n_conditions):
        eigenvalues, eigenvectors = np.linalg.eigh(sigma[index])
        eigenvalues = np.maximum(eigenvalues, float(ridge))
        stabilized[index] = (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T
    return stabilized


def _affine_flow_rdms(
    x: np.ndarray,
    labels: np.ndarray,
    time_centers: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    config: FlowRDMConfig,
    velocity_family: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train a centered affine flow over class and physical EEG time."""

    if velocity_family not in {"shared_affine", "covariate_affine"}:
        raise ValueError(f"Unsupported EEG affine family: {velocity_family!r}.")

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    values, _, _ = standardize_trial_features(x)
    labels = np.asarray(labels, dtype=np.int64)
    times = np.asarray(time_centers, dtype=np.float64)
    train_trials, val_trials = _stratified_validation_trials(labels, config.validation_fraction, seed)

    def flatten_trials(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xx = values[indices].reshape(-1, values.shape[-1])
        yy = np.repeat(labels[indices], times.size)
        tt = np.tile(times, indices.size)
        return condition_design(yy, tt), xx

    theta_train, x_train = flatten_trials(train_trials)
    theta_val, x_val = flatten_trials(val_trials)
    build_kwargs: dict[str, Any] = {}
    if velocity_family == "covariate_affine":
        build_kwargs["affine_condition_indices"] = (N_CLASSES,)
    model = build_flow_skl_model(
        velocity_family=velocity_family,
        theta_dim=theta_train.shape[1],
        x_dim=x_train.shape[1],
        hidden_dim=int(config.hidden_dim),
        depth=int(config.depth),
        quadrature_steps=int(config.quadrature_steps),
        path_schedule="cosine",
        divergence_estimator="exact",
        **build_kwargs,
    ).to(device)
    train_meta = train_flow_skl_model(
        model=model,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        device=device,
        velocity_family=velocity_family,
        path_schedule="cosine",
        epochs=int(config.epochs),
        batch_size=int(config.batch_size),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        patience=int(config.patience),
        min_delta=1e-4,
        ema_alpha=0.1,
        max_grad_norm=10.0,
        log_every=max(10, min(500, int(config.epochs) // 20)),
        checkpoint_selection="best",
        fixed_validation=True,
        validation_seed=int(seed) + 10_000,
    )
    if velocity_family == "shared_affine":
        covariance = _shared_endpoint_covariance(
            model,
            device=device,
            steps=int(config.covariance_ode_steps),
            ridge=float(config.covariance_ridge),
        )
        covariances = np.broadcast_to(covariance, (times.size,) + covariance.shape)
        covariance_sharing = "all_classes_and_all_eeg_times"
    else:
        time_conditions = condition_design(np.zeros(times.size, dtype=np.int64), times)
        covariances = _time_conditioned_endpoint_covariances(
            model,
            time_conditions,
            device=device,
            steps=int(config.covariance_ode_steps),
            ridge=float(config.covariance_ridge),
        )
        covariance_sharing = "all_classes_within_each_eeg_time"
    precisions = np.linalg.inv(
        covariances + float(config.covariance_ridge) * np.eye(covariances.shape[-1])[None, :, :]
    )
    grid_labels = np.repeat(np.arange(N_CLASSES, dtype=np.int64), times.size)
    grid_times = np.tile(times, N_CLASSES)
    conditions = condition_design(grid_labels, grid_times)
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        condition_tensor = torch.from_numpy(conditions.astype(np.float32)).to(device=device, dtype=dtype)
        means_flat = model.endpoint_mean(condition_tensor).detach().cpu().numpy().astype(np.float64)
    means = means_flat.reshape(N_CLASSES, times.size, -1).transpose(1, 0, 2)
    rdms = np.zeros((times.size, N_CLASSES, N_CLASSES), dtype=np.float64)
    for time_index in range(times.size):
        for left in range(N_CLASSES):
            for right in range(left + 1, N_CLASSES):
                delta = means[time_index, left] - means[time_index, right]
                distance = max(0.0, float(delta @ precisions[time_index] @ delta))
                rdms[time_index, left, right] = distance
                rdms[time_index, right, left] = distance
    metadata = {
        "seed": int(seed),
        "n_train_trials": int(train_trials.size),
        "n_validation_trials": int(val_trials.size),
        "best_epoch": int(train_meta["best_epoch"]),
        "stopped_epoch": int(train_meta["stopped_epoch"]),
        "best_val_loss": float(train_meta["best_val_loss"]),
        "checkpoint_selection": str(train_meta["checkpoint_selection"]),
        "selected_epoch": int(train_meta["selected_epoch"]),
        "train_losses": np.asarray(train_meta["train_losses"], dtype=np.float64).tolist(),
        "validation_losses": np.asarray(train_meta["val_losses"], dtype=np.float64).tolist(),
        "monitored_validation_losses": np.asarray(
            train_meta["val_monitor_losses"], dtype=np.float64
        ).tolist(),
        "velocity_family": velocity_family,
        "covariance_sharing": covariance_sharing,
        "n_endpoint_covariances": int(covariances.shape[0] if velocity_family == "covariate_affine" else 1),
        "config": asdict(config),
    }
    return rdms, metadata


def shared_affine_flow_rdms(
    x: np.ndarray,
    labels: np.ndarray,
    time_centers: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    config: FlowRDMConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use one flow covariance shared over all classes and EEG times."""

    return _affine_flow_rdms(
        x,
        labels,
        time_centers,
        device=device,
        seed=seed,
        config=config,
        velocity_family="shared_affine",
    )


def time_varying_shared_affine_flow_rdms(
    x: np.ndarray,
    labels: np.ndarray,
    time_centers: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    config: FlowRDMConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use a covariance shared across classes but varying with physical EEG time."""

    return _affine_flow_rdms(
        x,
        labels,
        time_centers,
        device=device,
        seed=seed,
        config=config,
        velocity_family="covariate_affine",
    )


def vectorize_rdms(rdms: np.ndarray, time_centers: np.ndarray, *, interval: tuple[float, float]) -> np.ndarray:
    values = np.asarray(rdms, dtype=np.float64)
    times = np.asarray(time_centers, dtype=np.float64)
    mask = (times >= float(interval[0])) & (times <= float(interval[1]))
    if not np.any(mask):
        raise ValueError(f"No time centers fall in interval {interval}.")
    upper = np.triu_indices(N_CLASSES, k=1)
    return values[mask][:, upper[0], upper[1]].reshape(-1)


def pearson_similarity(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        raise ValueError("RDM vectors must have the same length >= 2.")
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = math.sqrt(float(x @ x) * float(y @ y))
    return 0.0 if denom <= np.finfo(np.float64).eps else float((x @ y) / denom)


def save_rdm_cache(path: str | Path, *, rdms: np.ndarray, metadata: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        rdms=np.asarray(rdms, dtype=np.float64),
        metadata_json=np.asarray([json.dumps(metadata, sort_keys=True)]),
    )
    return output


def load_rdm_cache(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as data:
        return (
            np.asarray(data["rdms"], dtype=np.float64),
            json.loads(str(np.asarray(data["metadata_json"]).reshape(-1)[0])),
        )
