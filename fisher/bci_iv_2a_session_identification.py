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

from global_setting import EARLY_STOPPING_PATIENCE, TRAINING_MAX_EPOCHS
from fisher.bci_iv_2a_dataset import BCIIV2aFeatures, CLASS_NAMES
from fisher.flow_matching_skl import build_flow_skl_model, train_flow_skl_model


N_CLASSES = len(CLASS_NAMES)
QUERY_RUNS = (0, 2, 4)
REFERENCE_RUNS = (1, 3, 5)
RDM_MATCHING_INTERVAL = (0.0, 3.5)


@dataclass(frozen=True)
class FlowRDMConfig:
    hidden_dim: int = 64
    depth: int = 2
    epochs: int = TRAINING_MAX_EPOCHS
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = EARLY_STOPPING_PATIENCE
    quadrature_steps: int = 32
    covariance_ode_steps: int = 48
    covariance_ridge: float = 1e-5
    validation_fraction: float = 0.2
    standardize_features: bool = True
    device_resident_data: bool = True


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


def stratified_mixed_half_split(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split every trial once into class-stratified reference/query halves.

    Trials are pooled across runs before this split.  For odd class counts, the
    reference half receives the extra trial.  The returned indices are sorted
    only to make cached selections easy to inspect and reproduce.
    """

    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if np.any((y < 0) | (y >= N_CLASSES)):
        raise ValueError("Labels must be in [0, 3].")
    rng = np.random.default_rng(int(seed))
    reference: list[np.ndarray] = []
    query: list[np.ndarray] = []
    for label in range(N_CLASSES):
        candidates = np.flatnonzero(y == label)
        if candidates.size < 2:
            raise ValueError(f"Condition {label} needs at least two trials for a half split.")
        shuffled = rng.permutation(candidates)
        n_reference = (candidates.size + 1) // 2
        reference.append(shuffled[:n_reference])
        query.append(shuffled[n_reference:])
    return np.sort(np.concatenate(reference)), np.sort(np.concatenate(query))


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


def classical_mahalanobis_rdms(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    standardize_features: bool = True,
) -> np.ndarray:
    """Fit a Ledoit-Wolf pooled within-class covariance at each time window."""

    means, _, precisions = empirical_gaussian_components(
        x,
        labels,
        standardize_features=standardize_features,
    )
    return rdms_from_means_and_precisions(means, precisions)


def empirical_gaussian_components(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    standardize_features: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time-local class means and pooled Ledoit--Wolf covariance estimates."""

    values = np.asarray(x, dtype=np.float64)
    if standardize_features:
        values, _, _ = standardize_trial_features(values)
    labels = np.asarray(labels, dtype=np.int64)
    n_times, n_features = int(values.shape[1]), int(values.shape[2])
    means = np.empty((n_times, N_CLASSES, n_features), dtype=np.float64)
    covariances = np.empty((n_times, n_features, n_features), dtype=np.float64)
    precisions = np.empty_like(covariances)
    for time_index in range(n_times):
        means[time_index] = np.stack(
            [
                np.mean(values[labels == label, time_index], axis=0)
                for label in range(N_CLASSES)
            ]
        )
        residuals = np.concatenate(
            [
                values[labels == label, time_index] - means[time_index, label]
                for label in range(N_CLASSES)
            ],
            axis=0,
        )
        estimator = LedoitWolf(assume_centered=True).fit(residuals)
        covariances[time_index] = estimator.covariance_
        precisions[time_index] = estimator.precision_
    return means, covariances, precisions


def rdms_from_means_and_precisions(
    means: np.ndarray,
    precisions: np.ndarray,
) -> np.ndarray:
    """Construct squared Mahalanobis RDMs from time-indexed Gaussian components."""

    mean_values = np.asarray(means, dtype=np.float64)
    precision_values = np.asarray(precisions, dtype=np.float64)
    if mean_values.ndim != 3 or mean_values.shape[1] != N_CLASSES:
        raise ValueError("means must have shape [time, class, feature].")
    expected_precision_shape = (
        mean_values.shape[0],
        mean_values.shape[2],
        mean_values.shape[2],
    )
    if precision_values.shape != expected_precision_shape:
        raise ValueError(
            f"precisions must have shape {expected_precision_shape}; "
            f"got {precision_values.shape}."
        )
    rdms = np.zeros((mean_values.shape[0], N_CLASSES, N_CLASSES), dtype=np.float64)
    for time_index in range(mean_values.shape[0]):
        for left in range(N_CLASSES):
            for right in range(left + 1, N_CLASSES):
                delta = mean_values[time_index, left] - mean_values[time_index, right]
                distance = max(
                    0.0,
                    float(delta @ precision_values[time_index] @ delta),
                )
                rdms[time_index, left, right] = distance
                rdms[time_index, right, left] = distance
    return rdms


def squared_euclidean_rdms_from_means(means: np.ndarray) -> np.ndarray:
    """Construct squared Euclidean RDMs from time-indexed condition means."""

    mean_values = np.asarray(means, dtype=np.float64)
    if mean_values.ndim != 3 or mean_values.shape[1] != N_CLASSES:
        raise ValueError("means must have shape [time, class, feature].")
    differences = mean_values[:, :, None, :] - mean_values[:, None, :, :]
    return np.sum(differences * differences, axis=-1)


def empirical_condition_means(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    standardize_features: bool = True,
) -> np.ndarray:
    """Estimate one sample-mean response vector per EEG time and condition."""

    values = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if values.ndim != 3 or values.shape[0] != labels.size:
        raise ValueError("x must have shape [trial, time, feature] aligned with labels.")
    if standardize_features:
        values, _, _ = standardize_trial_features(values)
    means = []
    for label in range(N_CLASSES):
        selected = values[labels == label]
        if selected.shape[0] == 0:
            raise ValueError(f"Condition {label} has no trials.")
        means.append(np.mean(selected, axis=0))
    return np.stack(means, axis=1)


def classical_squared_euclidean_rdms(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    standardize_features: bool = True,
) -> np.ndarray:
    """Estimate per-time squared Euclidean RDMs from sample condition means."""

    means = empirical_condition_means(
        x,
        labels,
        standardize_features=standardize_features,
    )
    return squared_euclidean_rdms_from_means(means)


def empirical_condition_gaussian_components(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    standardize_features: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a distinct Ledoit--Wolf Gaussian for every EEG time and class."""

    values = np.asarray(x, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if values.ndim != 3 or values.shape[0] != y.size:
        raise ValueError("x must have shape [trial, time, feature] aligned with labels.")
    if standardize_features:
        values, _, _ = standardize_trial_features(values)
    n_times, n_features = int(values.shape[1]), int(values.shape[2])
    means = np.empty((n_times, N_CLASSES, n_features), dtype=np.float64)
    covariances = np.empty(
        (n_times, N_CLASSES, n_features, n_features),
        dtype=np.float64,
    )
    for label in range(N_CLASSES):
        selected = values[y == label]
        if selected.shape[0] < 2:
            raise ValueError(f"Condition {label} needs at least two trials for FID.")
        means[:, label] = np.mean(selected, axis=0, dtype=np.float64)
        for time_index in range(n_times):
            covariance = LedoitWolf().fit(selected[:, time_index]).covariance_
            covariances[time_index, label] = 0.5 * (covariance + covariance.T)
    return means, covariances


def gaussian_fid_rdms_from_moments(
    means: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    """Compute time-resolved Gaussian FID RDMs between the four conditions."""

    mean_values = np.asarray(means, dtype=np.float64)
    covariance_values = np.asarray(covariances, dtype=np.float64)
    if mean_values.ndim != 3 or mean_values.shape[1] != N_CLASSES:
        raise ValueError("means must have shape [time, class, feature].")
    expected = (
        mean_values.shape[0],
        N_CLASSES,
        mean_values.shape[2],
        mean_values.shape[2],
    )
    if covariance_values.shape != expected:
        raise ValueError(f"covariances must have shape {expected}; got {covariance_values.shape}.")
    rdms = np.zeros((mean_values.shape[0], N_CLASSES, N_CLASSES), dtype=np.float64)
    for time_index in range(mean_values.shape[0]):
        covariance_roots: list[np.ndarray] = []
        for label in range(N_CLASSES):
            covariance = 0.5 * (
                covariance_values[time_index, label]
                + covariance_values[time_index, label].T
            )
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 0.0)
            covariance_roots.append(
                (eigenvectors * np.sqrt(eigenvalues)[None, :]) @ eigenvectors.T
            )
        for left in range(N_CLASSES):
            for right in range(left + 1, N_CLASSES):
                delta = mean_values[time_index, left] - mean_values[time_index, right]
                middle = (
                    covariance_roots[left]
                    @ covariance_values[time_index, right]
                    @ covariance_roots[left]
                )
                middle = 0.5 * (middle + middle.T)
                middle_eigenvalues = np.maximum(np.linalg.eigvalsh(middle), 0.0)
                distance = (
                    float(delta @ delta)
                    + float(np.trace(covariance_values[time_index, left]))
                    + float(np.trace(covariance_values[time_index, right]))
                    - 2.0 * float(np.sum(np.sqrt(middle_eigenvalues)))
                )
                distance = max(0.0, distance)
                rdms[time_index, left, right] = distance
                rdms[time_index, right, left] = distance
    return rdms


def classical_fid_rdms(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    standardize_features: bool = True,
) -> np.ndarray:
    """Estimate per-time four-condition FID RDMs with Gaussian plug-ins."""

    means, covariances = empirical_condition_gaussian_components(
        x,
        labels,
        standardize_features=standardize_features,
    )
    return gaussian_fid_rdms_from_moments(means, covariances)


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
    sigma = torch.eye(x_dim, dtype=theta.dtype, device=device).expand(
        n_conditions, -1, -1
    ).clone()
    dt = 1.0 / float(steps)
    for step in range(int(steps)):
        t = torch.full(
            (n_conditions, 1),
            (step + 0.5) * dt,
            dtype=theta.dtype,
            device=device,
        )
        a = model.A(theta, t)
        sigma = sigma + dt * (
            torch.matmul(a, sigma) + torch.matmul(sigma, a.transpose(-1, -2))
        )
        sigma = 0.5 * (sigma + sigma.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(sigma)
    eigenvalues = eigenvalues.clamp_min(float(ridge))
    stabilized = (
        eigenvectors * eigenvalues.unsqueeze(-2)
    ) @ eigenvectors.transpose(-1, -2)
    stabilized = 0.5 * (stabilized + stabilized.transpose(-1, -2))
    return stabilized.detach().cpu().numpy().astype(np.float64)


def _affine_flow_rdms(
    x: np.ndarray,
    labels: np.ndarray,
    time_centers: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    config: FlowRDMConfig,
    velocity_family: str,
    return_components: bool = False,
) -> tuple[np.ndarray, dict[str, Any]] | tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Train a centered affine flow over class and physical EEG time."""

    if velocity_family not in {"shared_affine", "covariate_affine"}:
        raise ValueError(f"Unsupported EEG affine family: {velocity_family!r}.")

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    values = np.asarray(x, dtype=np.float64)
    if config.standardize_features:
        values, _, _ = standardize_trial_features(values)
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
        device_resident_data=bool(config.device_resident_data),
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
    distance_covariances = (
        covariances
        + float(config.covariance_ridge)
        * np.eye(covariances.shape[-1], dtype=np.float64)[None, :, :]
    )
    precisions = np.linalg.inv(distance_covariances)
    grid_labels = np.repeat(np.arange(N_CLASSES, dtype=np.int64), times.size)
    grid_times = np.tile(times, N_CLASSES)
    conditions = condition_design(grid_labels, grid_times)
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        condition_tensor = torch.from_numpy(conditions.astype(np.float32)).to(device=device, dtype=dtype)
        means_flat = model.endpoint_mean(condition_tensor).detach().cpu().numpy().astype(np.float64)
    means = means_flat.reshape(N_CLASSES, times.size, -1).transpose(1, 0, 2)
    rdms = rdms_from_means_and_precisions(means, precisions)
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
        "feature_standardization": "per_fit_mean_and_scale" if config.standardize_features else "none",
        "config": asdict(config),
    }
    if return_components:
        components = {
            "flow_means": means,
            "flow_endpoint_covariances": covariances,
            "flow_distance_covariances": distance_covariances,
            "flow_precisions": precisions,
        }
        return rdms, metadata, components
    return rdms, metadata


def translation_flow_squared_euclidean_rdms(
    x: np.ndarray,
    labels: np.ndarray,
    time_centers: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    config: FlowRDMConfig,
    return_means: bool = False,
    checkpoint_path: str | Path | None = None,
    checkpoint_context: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]] | tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """Fit a translation-only flow and read out squared Euclidean mean RDMs.

    When ``checkpoint_path`` is provided, the exact best-validation model state
    restored before the RDM readout is saved atomically with all information
    required to reconstruct the model for evaluation.
    """

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    values = np.asarray(x, dtype=np.float64)
    standardization_mean: np.ndarray | None = None
    standardization_scale: np.ndarray | None = None
    if config.standardize_features:
        values, standardization_mean, standardization_scale = standardize_trial_features(values)
    labels = np.asarray(labels, dtype=np.int64)
    times = np.asarray(time_centers, dtype=np.float64)
    train_trials, val_trials = _stratified_validation_trials(
        labels,
        config.validation_fraction,
        seed,
    )

    def flatten_trials(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xx = values[indices].reshape(-1, values.shape[-1])
        yy = np.repeat(labels[indices], times.size)
        tt = np.tile(times, indices.size)
        return condition_design(yy, tt), xx

    theta_train, x_train = flatten_trials(train_trials)
    theta_val, x_val = flatten_trials(val_trials)
    model = build_flow_skl_model(
        velocity_family="translation",
        theta_dim=theta_train.shape[1],
        x_dim=x_train.shape[1],
        hidden_dim=int(config.hidden_dim),
        depth=int(config.depth),
        quadrature_steps=int(config.quadrature_steps),
        path_schedule="cosine",
        divergence_estimator="exact",
    ).to(device)
    train_meta = train_flow_skl_model(
        model=model,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        device=device,
        velocity_family="translation",
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
        retain_best_state=checkpoint_path is not None,
        device_resident_data=bool(config.device_resident_data),
    )

    grid_labels = np.repeat(np.arange(N_CLASSES, dtype=np.int64), times.size)
    grid_times = np.tile(times, N_CLASSES)
    conditions = condition_design(grid_labels, grid_times)
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        condition_tensor = torch.from_numpy(conditions.astype(np.float32)).to(
            device=device,
            dtype=dtype,
        )
        means_flat = model.endpoint_mean(condition_tensor).detach().cpu().numpy()
    means = means_flat.astype(np.float64).reshape(N_CLASSES, times.size, -1).transpose(1, 0, 2)
    rdms = squared_euclidean_rdms_from_means(means)
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
            train_meta["val_monitor_losses"],
            dtype=np.float64,
        ).tolist(),
        "velocity_family": "translation",
        "distance": "squared_euclidean_between_endpoint_means",
        "feature_standardization": (
            "per_fit_mean_and_scale" if config.standardize_features else "none"
        ),
        "config": asdict(config),
    }
    if checkpoint_path is not None:
        best_state = train_meta.get("best_state_dict")
        if best_state is None:
            raise RuntimeError("Best model state was not retained for checkpoint saving.")
        target = Path(checkpoint_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        payload = {
            "format_version": 1,
            "checkpoint_role": "best_validation_model_used_for_rdm_evaluation",
            "velocity_family": "translation",
            "model_kwargs": {
                "velocity_family": "translation",
                "theta_dim": int(theta_train.shape[1]),
                "x_dim": int(x_train.shape[1]),
                "hidden_dim": int(config.hidden_dim),
                "depth": int(config.depth),
                "quadrature_steps": int(config.quadrature_steps),
                "path_schedule": "cosine",
                "divergence_estimator": "exact",
            },
            "model_state_dict": {
                key: value.detach().cpu().clone() for key, value in best_state.items()
            },
            "training": {
                "seed": int(seed),
                "best_epoch": int(train_meta["best_epoch"]),
                "stopped_epoch": int(train_meta["stopped_epoch"]),
                "best_val_loss": float(train_meta["best_val_loss"]),
                "checkpoint_selection": str(train_meta["checkpoint_selection"]),
                "selected_epoch": int(train_meta["selected_epoch"]),
            },
            "flow_rdm_config": asdict(config),
            "time_centers": torch.from_numpy(times.astype(np.float64)),
            "feature_standardization": {
                "kind": "per_fit_mean_and_scale" if config.standardize_features else "none",
                "mean": (
                    None
                    if standardization_mean is None
                    else torch.from_numpy(standardization_mean.astype(np.float64))
                ),
                "scale": (
                    None
                    if standardization_scale is None
                    else torch.from_numpy(standardization_scale.astype(np.float64))
                ),
            },
            "context": {} if checkpoint_context is None else dict(checkpoint_context),
        }
        torch.save(payload, temporary)
        temporary.replace(target)
        metadata["checkpoint_path"] = str(target.resolve())
        metadata["checkpoint_format_version"] = 1
    if return_means:
        return rdms, metadata, means
    return rdms, metadata


def load_translation_flow_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Reconstruct a saved translation-flow best checkpoint for evaluation."""

    source = Path(checkpoint_path)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if int(payload.get("format_version", -1)) != 1:
        raise ValueError(f"Unsupported translation-flow checkpoint format in {source}.")
    if payload.get("velocity_family") != "translation":
        raise ValueError(f"Checkpoint {source} is not a translation-flow model.")
    model = build_flow_skl_model(**payload["model_kwargs"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def condition_affine_flow_fid_rdms(
    x: np.ndarray,
    labels: np.ndarray,
    time_centers: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    config: FlowRDMConfig,
    return_components: bool = False,
    checkpoint_path: str | Path | None = None,
    checkpoint_context: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]] | tuple[
    np.ndarray,
    dict[str, Any],
    dict[str, np.ndarray],
]:
    """Fit a class-and-time condition-affine flow and read out Gaussian FID.

    Both the endpoint mean and the full endpoint covariance depend on the four
    class indicators and physical EEG time.  A supplied checkpoint path stores
    the exact best-validation state restored before the moment/FID readout.
    """

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    values = np.asarray(x, dtype=np.float64)
    standardization_mean: np.ndarray | None = None
    standardization_scale: np.ndarray | None = None
    if config.standardize_features:
        values, standardization_mean, standardization_scale = standardize_trial_features(values)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    times = np.asarray(time_centers, dtype=np.float64).reshape(-1)
    if values.ndim != 3 or values.shape[0] != y.size or values.shape[1] != times.size:
        raise ValueError("x, labels, and time_centers have incompatible shapes.")
    train_trials, validation_trials = _stratified_validation_trials(
        y,
        config.validation_fraction,
        seed,
    )

    def flatten_trials(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xx = values[indices].reshape(-1, values.shape[-1])
        yy = np.repeat(y[indices], times.size)
        tt = np.tile(times, indices.size)
        return condition_design(yy, tt), xx

    theta_train, x_train = flatten_trials(train_trials)
    theta_validation, x_validation = flatten_trials(validation_trials)
    model_kwargs = {
        "velocity_family": "condition_affine",
        "theta_dim": int(theta_train.shape[1]),
        "x_dim": int(x_train.shape[1]),
        "hidden_dim": int(config.hidden_dim),
        "depth": int(config.depth),
        "quadrature_steps": int(config.quadrature_steps),
        "path_schedule": "cosine",
        "divergence_estimator": "exact",
    }
    model = build_flow_skl_model(**model_kwargs).to(device)
    train_meta = train_flow_skl_model(
        model=model,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_validation,
        x_val=x_validation,
        device=device,
        velocity_family="condition_affine",
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
        retain_best_state=checkpoint_path is not None,
        device_resident_data=bool(config.device_resident_data),
    )

    grid_labels = np.repeat(np.arange(N_CLASSES, dtype=np.int64), times.size)
    grid_times = np.tile(times, N_CLASSES)
    conditions = condition_design(grid_labels, grid_times)
    dtype = next(model.parameters()).dtype
    condition_tensor = torch.from_numpy(conditions.astype(np.float32)).to(
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        means_flat = model.endpoint_mean(condition_tensor).detach().cpu().numpy()
    covariance_flat = _time_conditioned_endpoint_covariances(
        model,
        conditions,
        device=device,
        steps=int(config.covariance_ode_steps),
        ridge=float(config.covariance_ridge),
    )
    means = means_flat.astype(np.float64).reshape(
        N_CLASSES, times.size, -1
    ).transpose(1, 0, 2)
    covariances = covariance_flat.reshape(
        N_CLASSES,
        times.size,
        values.shape[-1],
        values.shape[-1],
    ).transpose(1, 0, 2, 3)
    rdms = gaussian_fid_rdms_from_moments(means, covariances)
    metadata = {
        "seed": int(seed),
        "n_train_trials": int(train_trials.size),
        "n_validation_trials": int(validation_trials.size),
        "best_epoch": int(train_meta["best_epoch"]),
        "stopped_epoch": int(train_meta["stopped_epoch"]),
        "best_val_loss": float(train_meta["best_val_loss"]),
        "checkpoint_selection": str(train_meta["checkpoint_selection"]),
        "selected_epoch": int(train_meta["selected_epoch"]),
        "train_losses": np.asarray(train_meta["train_losses"], dtype=np.float64).tolist(),
        "validation_losses": np.asarray(train_meta["val_losses"], dtype=np.float64).tolist(),
        "monitored_validation_losses": np.asarray(
            train_meta["val_monitor_losses"],
            dtype=np.float64,
        ).tolist(),
        "velocity_family": "condition_affine",
        "distance": "gaussian_fid_with_class_and_time_specific_full_covariance",
        "covariance_sharing": "none_across_class_or_eeg_time",
        "n_endpoint_covariances": int(N_CLASSES * times.size),
        "feature_standardization": (
            "per_fit_mean_and_scale" if config.standardize_features else "none"
        ),
        "config": asdict(config),
    }
    if checkpoint_path is not None:
        best_state = train_meta.get("best_state_dict")
        if best_state is None:
            raise RuntimeError("Best model state was not retained for checkpoint saving.")
        target = Path(checkpoint_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        payload = {
            "format_version": 1,
            "checkpoint_role": "best_validation_model_used_for_rdm_evaluation",
            "velocity_family": "condition_affine",
            "model_kwargs": model_kwargs,
            "model_state_dict": {
                key: value.detach().cpu().clone() for key, value in best_state.items()
            },
            "training": {
                "seed": int(seed),
                "best_epoch": int(train_meta["best_epoch"]),
                "stopped_epoch": int(train_meta["stopped_epoch"]),
                "best_val_loss": float(train_meta["best_val_loss"]),
                "checkpoint_selection": str(train_meta["checkpoint_selection"]),
                "selected_epoch": int(train_meta["selected_epoch"]),
            },
            "flow_rdm_config": asdict(config),
            "time_centers": torch.from_numpy(times.astype(np.float64)),
            "train_trial_indices": torch.from_numpy(train_trials.astype(np.int64)),
            "validation_trial_indices": torch.from_numpy(validation_trials.astype(np.int64)),
            "feature_standardization": {
                "kind": "per_fit_mean_and_scale" if config.standardize_features else "none",
                "mean": (
                    None
                    if standardization_mean is None
                    else torch.from_numpy(standardization_mean.astype(np.float64))
                ),
                "scale": (
                    None
                    if standardization_scale is None
                    else torch.from_numpy(standardization_scale.astype(np.float64))
                ),
            },
            "context": {} if checkpoint_context is None else dict(checkpoint_context),
        }
        torch.save(payload, temporary)
        temporary.replace(target)
        metadata["checkpoint_path"] = str(target.resolve())
        metadata["checkpoint_format_version"] = 1
    if return_components:
        return rdms, metadata, {
            "flow_means": means,
            "flow_endpoint_covariances": covariances,
        }
    return rdms, metadata


def load_condition_affine_flow_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Reconstruct a saved condition-affine best checkpoint for evaluation."""

    source = Path(checkpoint_path)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if int(payload.get("format_version", -1)) != 1:
        raise ValueError(f"Unsupported condition-affine checkpoint format in {source}.")
    if payload.get("velocity_family") != "condition_affine":
        raise ValueError(f"Checkpoint {source} is not a condition-affine model.")
    model = build_flow_skl_model(**payload["model_kwargs"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


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


def time_varying_shared_affine_flow_rdm_components(
    x: np.ndarray,
    labels: np.ndarray,
    time_centers: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    config: FlowRDMConfig,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Fit the time-varying flow and retain its mean and covariance components."""

    result = _affine_flow_rdms(
        x,
        labels,
        time_centers,
        device=device,
        seed=seed,
        config=config,
        velocity_family="covariate_affine",
        return_components=True,
    )
    if len(result) != 3:
        raise RuntimeError("Flow component fit did not return diagnostic components.")
    return result


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
