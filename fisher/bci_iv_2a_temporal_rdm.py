"""Classical time-by-time RDM estimators for BCI Competition IV-2a."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from global_setting import DEFAULT_EARLY_STOPPING_PATIENCE, DEFAULT_TRAINING_MAX_EPOCHS
from fisher.distance_comparison import (
    correlation_distance_matrix,
    cosine_distance_matrix,
    gaussian_fid_matrix,
    squared_euclidean_mean_distance_matrix,
)
from fisher.flow_matching_skl import build_flow_skl_model, train_flow_skl_model


TEMPORAL_RDM_METRICS = ("correlation", "cosine", "euclidean", "fid")


@dataclass(frozen=True)
class ClassicalTemporalRDMResult:
    """Classical temporal RDMs and their fitted Gaussian moments."""

    rdms: dict[str, np.ndarray]
    means: np.ndarray
    covariances: np.ndarray


@dataclass(frozen=True)
class FlowTemporalRDMConfig:
    """Training and readout settings for a native-time affine flow."""

    hidden_dim: int = 64
    depth: int = 2
    epochs: int = DEFAULT_TRAINING_MAX_EPOCHS
    batch_size: int = 4_096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    validation_fraction: float = 0.25
    covariance_steps: int = 48
    covariance_ridge: float = 1e-5
    fid_block_size: int = 128


@dataclass(frozen=True)
class FlowTemporalRDMResult:
    """Dense no-binning flow temporal RDMs and fitted endpoint moments."""

    rdms: dict[str, np.ndarray]
    means: np.ndarray
    covariances: np.ndarray
    train_trial_indices: np.ndarray
    validation_trial_indices: np.ndarray
    x_normalization_mean: np.ndarray
    x_normalization_std: np.ndarray
    condition_scale: float
    train_metadata: dict[str, Any]


def _validate_temporal_samples(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("samples must have shape [n_trials, n_times, n_features].")
    if values.shape[0] < 2:
        raise ValueError("At least two trials are required.")
    if values.shape[1] < 2:
        raise ValueError("At least two time bins are required.")
    if values.shape[2] < 1:
        raise ValueError("At least one feature is required.")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples contain non-finite values.")
    return values


def fit_temporal_gaussians(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit one Ledoit--Wolf Gaussian to the trial distribution in each time bin."""

    values = _validate_temporal_samples(samples)
    means = np.mean(values, axis=0, dtype=np.float64)
    n_times, n_features = int(values.shape[1]), int(values.shape[2])
    covariances = np.empty((n_times, n_features, n_features), dtype=np.float64)
    for time_index in range(n_times):
        covariance = LedoitWolf().fit(values[:, time_index, :]).covariance_
        covariances[time_index] = 0.5 * (covariance + covariance.T)
    return means, covariances


def classical_temporal_rdms(samples: np.ndarray) -> ClassicalTemporalRDMResult:
    """Compute correlation, cosine, Euclidean, and Gaussian FID time RDMs.

    Correlation, cosine, and Euclidean distances compare the trial-mean feature
    vector in each time bin. FID compares Gaussian fits with a distinct
    Ledoit--Wolf covariance for every time bin.
    """

    means, covariances = fit_temporal_gaussians(samples)
    squared_euclidean = squared_euclidean_mean_distance_matrix(means)
    rdms = {
        "correlation": correlation_distance_matrix(means),
        "cosine": cosine_distance_matrix(means),
        "euclidean": np.sqrt(squared_euclidean),
        "fid": gaussian_fid_matrix(means, covariances),
    }
    for name, matrix in rdms.items():
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} temporal RDM contains non-finite values.")
        if np.min(matrix) < -1e-10:
            raise ValueError(f"{name} temporal RDM contains a negative distance.")
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-10, rtol=0.0)
        np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-10, rtol=0.0)
    return ClassicalTemporalRDMResult(rdms=rdms, means=means, covariances=covariances)


@torch.no_grad()
def time_conditioned_affine_endpoint_moments(
    model: torch.nn.Module,
    conditions: np.ndarray,
    *,
    device: torch.device,
    covariance_steps: int,
    covariance_ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate endpoint means and integrate a full covariance at every condition."""

    theta_values = np.asarray(conditions, dtype=np.float64)
    if theta_values.ndim == 1:
        theta_values = theta_values[:, None]
    if theta_values.ndim != 2 or theta_values.shape[0] < 2:
        raise ValueError("conditions must have shape [n_times, theta_dim] with at least two rows.")
    if int(covariance_steps) < 1:
        raise ValueError("covariance_steps must be positive.")
    if float(covariance_ridge) <= 0.0:
        raise ValueError("covariance_ridge must be positive.")

    model.eval()
    dtype = next(model.parameters()).dtype
    theta = torch.from_numpy(theta_values.astype(np.float32)).to(device=device, dtype=dtype)
    means = model.endpoint_mean(theta)
    n_times = int(theta.shape[0])
    x_dim = int(getattr(model, "x_dim"))
    covariance = torch.eye(x_dim, dtype=dtype, device=device).expand(n_times, -1, -1).clone()
    dt = 1.0 / float(covariance_steps)
    for step in range(int(covariance_steps)):
        flow_time = torch.full(
            (n_times, 1),
            (float(step) + 0.5) * dt,
            dtype=dtype,
            device=device,
        )
        matrix = model.A(theta, flow_time)
        covariance = covariance + dt * (
            matrix @ covariance + covariance @ matrix.transpose(-1, -2)
        )
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(float(covariance_ridge))
    covariance = (eigenvectors * eigenvalues.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)
    return (
        means.detach().cpu().numpy().astype(np.float64),
        covariance.detach().cpu().numpy().astype(np.float64),
    )


@torch.no_grad()
def gaussian_fid_matrix_batched(
    means: np.ndarray,
    covariances: np.ndarray,
    *,
    device: torch.device,
    block_size: int = 128,
) -> np.ndarray:
    """Compute a full-covariance Gaussian FID matrix in GPU-friendly blocks."""

    mean_values = np.asarray(means, dtype=np.float64)
    covariance_values = np.asarray(covariances, dtype=np.float64)
    if mean_values.ndim != 2:
        raise ValueError("means must have shape [n_times, n_features].")
    expected = (mean_values.shape[0], mean_values.shape[1], mean_values.shape[1])
    if covariance_values.shape != expected:
        raise ValueError(f"covariances must have shape {expected}.")
    if int(block_size) < 1:
        raise ValueError("block_size must be positive.")

    dtype = torch.float32 if device.type == "cuda" else torch.float64
    mean_tensor = torch.as_tensor(mean_values, dtype=dtype, device=device)
    covariance_tensor = torch.as_tensor(covariance_values, dtype=dtype, device=device)
    covariance_tensor = 0.5 * (covariance_tensor + covariance_tensor.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_tensor)
    eigenvalues = eigenvalues.clamp_min(torch.finfo(dtype).eps)
    covariance_roots = (
        eigenvectors * torch.sqrt(eigenvalues).unsqueeze(-2)
    ) @ eigenvectors.transpose(-1, -2)
    traces = torch.diagonal(covariance_tensor, dim1=-2, dim2=-1).sum(dim=-1)

    n_times = int(mean_values.shape[0])
    output = np.zeros((n_times, n_times), dtype=np.float64)
    block = int(block_size)
    for left_start in range(0, n_times, block):
        left_stop = min(n_times, left_start + block)
        left_root = covariance_roots[left_start:left_stop, None]
        left_mean = mean_tensor[left_start:left_stop, None]
        left_trace = traces[left_start:left_stop, None]
        for right_start in range(left_start, n_times, block):
            right_stop = min(n_times, right_start + block)
            right_covariance = covariance_tensor[None, right_start:right_stop]
            middle = left_root @ right_covariance @ left_root
            middle = 0.5 * (middle + middle.transpose(-1, -2))
            middle_eigenvalues = torch.linalg.eigvalsh(middle).clamp_min(0.0)
            covariance_term = (
                left_trace
                + traces[None, right_start:right_stop]
                - 2.0 * torch.sqrt(middle_eigenvalues).sum(dim=-1)
            )
            mean_term = torch.sum(
                (left_mean - mean_tensor[None, right_start:right_stop]) ** 2,
                dim=-1,
            )
            values = (mean_term + covariance_term).clamp_min(0.0).detach().cpu().numpy()
            output[left_start:left_stop, right_start:right_stop] = values
            if right_start != left_start:
                output[right_start:right_stop, left_start:left_stop] = values.T
    output = 0.5 * (output + output.T)
    np.fill_diagonal(output, 0.0)
    return output


def flow_temporal_rdms_from_moments(
    means: np.ndarray,
    covariances: np.ndarray,
    *,
    device: torch.device,
    fid_block_size: int,
) -> dict[str, np.ndarray]:
    """Read four temporal metrics from one fitted affine-flow distribution."""

    mean_values = np.asarray(means, dtype=np.float64)
    squared_euclidean = squared_euclidean_mean_distance_matrix(mean_values)
    rdms = {
        "correlation": correlation_distance_matrix(mean_values),
        "cosine": cosine_distance_matrix(mean_values),
        "euclidean": np.sqrt(squared_euclidean),
        "fid": gaussian_fid_matrix_batched(
            mean_values,
            covariances,
            device=device,
            block_size=int(fid_block_size),
        ),
    }
    for name, matrix in rdms.items():
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} flow temporal RDM contains non-finite values.")
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-5, rtol=0.0)
        np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-8, rtol=0.0)
    return rdms


def fit_native_time_affine_flow_rdms(
    samples: np.ndarray,
    time_points: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    config: FlowTemporalRDMConfig,
    evaluation_time_points: np.ndarray | None = None,
) -> tuple[FlowTemporalRDMResult, torch.nn.Module]:
    """Fit one continuous physical-time affine flow and evaluate temporal RDMs.

    Training always uses every supplied native-time observation.  By default,
    endpoint moments and RDMs are also evaluated at every native time.  Passing
    ``evaluation_time_points`` evaluates the fitted continuous model on a
    smaller common grid without temporally binning the training data.
    """

    values = _validate_temporal_samples(samples)
    times = np.asarray(time_points, dtype=np.float64).reshape(-1)
    if times.shape != (values.shape[1],):
        raise ValueError("time_points must have one entry per native sample.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("time_points must be strictly increasing.")
    if evaluation_time_points is None:
        evaluation_times = times
    else:
        evaluation_times = np.asarray(evaluation_time_points, dtype=np.float64).reshape(-1)
        if evaluation_times.size < 2:
            raise ValueError("evaluation_time_points must contain at least two times.")
        if np.any(np.diff(evaluation_times) <= 0.0):
            raise ValueError("evaluation_time_points must be strictly increasing.")
        tolerance = 10.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(times))))
        if evaluation_times[0] < times[0] - tolerance or evaluation_times[-1] > times[-1] + tolerance:
            raise ValueError("evaluation_time_points must lie inside the training time range.")
    if not 0.0 < float(config.validation_fraction) < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1).")

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    rng = np.random.default_rng(int(seed))
    n_trials = int(values.shape[0])
    n_validation = min(n_trials - 1, max(1, int(round(config.validation_fraction * n_trials))))
    validation_trials = np.sort(rng.choice(n_trials, size=n_validation, replace=False))
    train_trials = np.setdiff1d(np.arange(n_trials, dtype=np.int64), validation_trials)

    train_flat_native = values[train_trials].reshape(-1, values.shape[-1])
    normalization_mean = np.mean(train_flat_native, axis=0, dtype=np.float64)
    normalization_std = np.std(train_flat_native, axis=0, ddof=1, dtype=np.float64)
    normalization_std = np.maximum(normalization_std, 1e-8)
    normalized = (values - normalization_mean) / normalization_std

    condition_scale = max(float(np.max(np.abs(times))), np.finfo(np.float64).eps)
    time_conditions = (times / condition_scale).reshape(-1, 1)
    evaluation_conditions = (evaluation_times / condition_scale).reshape(-1, 1)

    def flatten_trials(trial_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = np.tile(time_conditions, (int(trial_indices.size), 1))
        x = normalized[trial_indices].reshape(-1, normalized.shape[-1])
        return theta, x

    theta_train, x_train = flatten_trials(train_trials)
    theta_validation, x_validation = flatten_trials(validation_trials)
    model = build_flow_skl_model(
        velocity_family="covariate_affine",
        theta_dim=1,
        x_dim=int(values.shape[-1]),
        hidden_dim=int(config.hidden_dim),
        depth=int(config.depth),
        quadrature_steps=32,
        path_schedule="cosine",
        divergence_estimator="exact",
        affine_condition_indices=(0,),
    ).to(device)
    train_metadata = train_flow_skl_model(
        model=model,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_validation,
        x_val=x_validation,
        device=device,
        velocity_family="covariate_affine",
        path_schedule="cosine",
        epochs=int(config.epochs),
        batch_size=int(config.batch_size),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        patience=int(config.patience),
        min_delta=1e-4,
        ema_alpha=0.1,
        max_grad_norm=10.0,
        log_every=max(10, min(250, int(config.epochs) // 25)),
        checkpoint_selection="best",
        fixed_validation=True,
        validation_seed=int(seed) + 10_000,
    )
    normalized_means, normalized_covariances = time_conditioned_affine_endpoint_moments(
        model,
        evaluation_conditions,
        device=device,
        covariance_steps=int(config.covariance_steps),
        covariance_ridge=float(config.covariance_ridge),
    )
    means = normalization_mean[None, :] + normalized_means * normalization_std[None, :]
    covariances = (
        normalized_covariances
        * normalization_std[None, :, None]
        * normalization_std[None, None, :]
    )
    rdms = flow_temporal_rdms_from_moments(
        means,
        covariances,
        device=device,
        fid_block_size=int(config.fid_block_size),
    )
    result = FlowTemporalRDMResult(
        rdms=rdms,
        means=means,
        covariances=covariances,
        train_trial_indices=train_trials,
        validation_trial_indices=validation_trials,
        x_normalization_mean=normalization_mean,
        x_normalization_std=normalization_std,
        condition_scale=condition_scale,
        train_metadata=train_metadata,
    )
    return result, model
