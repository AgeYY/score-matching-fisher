"""Moment-matched Stringer surrogates with controlled residual non-Gaussianity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PeriodicFourierMoments:
    """Smooth conditional mean and gridded covariance for a periodic condition."""

    period: float
    n_harmonics: int
    mean_coefficients: np.ndarray
    covariance_grid_centers: np.ndarray
    covariance_grid: np.ndarray

    @property
    def x_dim(self) -> int:
        return int(self.mean_coefficients.shape[1])

    def mean(self, theta: np.ndarray) -> np.ndarray:
        return periodic_fourier_features(
            theta, period=self.period, n_harmonics=self.n_harmonics
        ) @ self.mean_coefficients

    def mean_derivative(self, theta: np.ndarray) -> np.ndarray:
        return periodic_fourier_derivative_features(
            theta, period=self.period, n_harmonics=self.n_harmonics
        ) @ self.mean_coefficients

    def covariance_indices(self, theta: np.ndarray) -> np.ndarray:
        values = np.mod(np.asarray(theta, dtype=np.float64).reshape(-1), self.period)
        n_grid = int(self.covariance_grid.shape[0])
        indices = np.floor(values / self.period * n_grid).astype(np.int64)
        return np.clip(indices, 0, n_grid - 1)

    def covariance(self, theta: np.ndarray) -> np.ndarray:
        return self.covariance_grid[self.covariance_indices(theta)]

    def linear_fisher(self, theta: np.ndarray, *, solve_jitter: float = 1e-8) -> np.ndarray:
        derivative = self.mean_derivative(theta)
        covariance = self.covariance(theta)
        eye = np.eye(self.x_dim, dtype=np.float64)
        solved = np.linalg.solve(
            covariance + float(solve_jitter) * eye[None],
            derivative[..., None],
        )[..., 0]
        return np.einsum("ni,ni->n", derivative, solved)


@dataclass(frozen=True)
class StandardizedResidualBank:
    """Empirical residual samples with zero mean and identity covariance per bin."""

    residuals: np.ndarray
    bin_ids: np.ndarray
    n_bins: int
    period: float
    counts: np.ndarray
    mean_norms: np.ndarray
    covariance_errors: np.ndarray

    def bins_for_theta(self, theta: np.ndarray) -> np.ndarray:
        return periodic_bin_ids(theta, n_bins=self.n_bins, period=self.period)


def periodic_fourier_features(
    theta: np.ndarray,
    *,
    period: float,
    n_harmonics: int,
) -> np.ndarray:
    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    if float(period) <= 0.0:
        raise ValueError("period must be positive.")
    if int(n_harmonics) < 1:
        raise ValueError("n_harmonics must be positive.")
    phase = 2.0 * np.pi * values / float(period)
    columns = [np.ones_like(phase)]
    for harmonic in range(1, int(n_harmonics) + 1):
        columns.extend((np.cos(harmonic * phase), np.sin(harmonic * phase)))
    return np.stack(columns, axis=1)


def periodic_fourier_derivative_features(
    theta: np.ndarray,
    *,
    period: float,
    n_harmonics: int,
) -> np.ndarray:
    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    phase = 2.0 * np.pi * values / float(period)
    omega = 2.0 * np.pi / float(period)
    columns = [np.zeros_like(phase)]
    for harmonic in range(1, int(n_harmonics) + 1):
        frequency = float(harmonic) * omega
        columns.extend(
            (
                -frequency * np.sin(harmonic * phase),
                frequency * np.cos(harmonic * phase),
            )
        )
    return np.stack(columns, axis=1)


def periodic_bin_ids(theta: np.ndarray, *, n_bins: int, period: float) -> np.ndarray:
    if int(n_bins) < 1:
        raise ValueError("n_bins must be positive.")
    values = np.mod(np.asarray(theta, dtype=np.float64).reshape(-1), float(period))
    indices = np.floor(values / float(period) * int(n_bins)).astype(np.int64)
    return np.clip(indices, 0, int(n_bins) - 1)


def _ridge_gram(features: np.ndarray, relative_ridge: float) -> np.ndarray:
    if float(relative_ridge) < 0.0:
        raise ValueError("relative_ridge must be non-negative.")
    gram = features.T @ features
    penalty = np.eye(features.shape[1], dtype=np.float64) * (
        float(relative_ridge) * features.shape[0]
    )
    penalty[0, 0] = 0.0
    return gram + penalty


def _project_spd(matrix: np.ndarray, *, eigenvalue_floor: float) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, float(eigenvalue_floor))
    return (eigenvectors * clipped[None, :]) @ eigenvectors.T


def fit_periodic_fourier_moments(
    theta: np.ndarray,
    responses: np.ndarray,
    *,
    period: float,
    n_harmonics: int = 4,
    relative_ridge: float = 1e-3,
    covariance_grid_size: int = 32,
    covariance_shrinkage: float = 0.25,
    eigenvalue_floor_relative: float = 1e-4,
) -> PeriodicFourierMoments:
    """Fit neutral periodic regressions for conditional first and second moments."""
    x = np.asarray(responses, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("responses must have shape [observation, dimension].")
    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    if values.shape[0] != x.shape[0]:
        raise ValueError("theta and responses must contain the same number of observations.")
    if int(covariance_grid_size) < 2:
        raise ValueError("covariance_grid_size must be at least 2.")
    if not 0.0 <= float(covariance_shrinkage) <= 1.0:
        raise ValueError("covariance_shrinkage must be in [0, 1].")

    features = periodic_fourier_features(
        values, period=float(period), n_harmonics=int(n_harmonics)
    )
    gram = _ridge_gram(features, float(relative_ridge))
    mean_coefficients = np.linalg.solve(gram, features.T @ x)
    residuals = x - features @ mean_coefficients
    global_covariance = residuals.T @ residuals / float(residuals.shape[0])

    covariance_rhs = np.stack(
        [
            residuals.T @ (residuals * features[:, feature_index, None])
            for feature_index in range(features.shape[1])
        ],
        axis=0,
    )
    covariance_coefficients = np.linalg.solve(
        gram,
        covariance_rhs.reshape(features.shape[1], -1),
    ).reshape(features.shape[1], x.shape[1], x.shape[1])

    grid_centers = (
        (np.arange(int(covariance_grid_size), dtype=np.float64) + 0.5)
        * float(period)
        / float(covariance_grid_size)
    )
    grid_features = periodic_fourier_features(
        grid_centers, period=float(period), n_harmonics=int(n_harmonics)
    )
    raw_covariances = np.einsum("np,pij->nij", grid_features, covariance_coefficients)
    scale = max(float(np.trace(global_covariance) / x.shape[1]), 1e-8)
    floor = float(eigenvalue_floor_relative) * scale
    covariance_grid = np.empty_like(raw_covariances)
    for grid_index, raw in enumerate(raw_covariances):
        projected = _project_spd(raw, eigenvalue_floor=floor)
        shrunk = (
            (1.0 - float(covariance_shrinkage)) * projected
            + float(covariance_shrinkage) * global_covariance
        )
        covariance_grid[grid_index] = _project_spd(shrunk, eigenvalue_floor=floor)

    return PeriodicFourierMoments(
        period=float(period),
        n_harmonics=int(n_harmonics),
        mean_coefficients=np.asarray(mean_coefficients, dtype=np.float64),
        covariance_grid_centers=grid_centers,
        covariance_grid=covariance_grid,
    )


def fit_standardized_residual_bank(
    theta: np.ndarray,
    responses: np.ndarray,
    moments: PeriodicFourierMoments,
    *,
    n_bins: int = 16,
    eigenvalue_floor_relative: float = 1e-4,
) -> StandardizedResidualBank:
    """Whiten real residuals separately within circular orientation bins."""
    x = np.asarray(responses, dtype=np.float64)
    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != values.shape[0] or x.shape[1] != moments.x_dim:
        raise ValueError("theta and responses do not match the supplied moment model.")
    bin_ids = periodic_bin_ids(values, n_bins=int(n_bins), period=moments.period)
    centered_residuals = x - moments.mean(values)
    standardized = np.empty_like(centered_residuals)
    counts = np.empty(int(n_bins), dtype=np.int64)
    mean_norms = np.empty(int(n_bins), dtype=np.float64)
    covariance_errors = np.empty(int(n_bins), dtype=np.float64)
    identity = np.eye(x.shape[1], dtype=np.float64)

    for bin_index in range(int(n_bins)):
        indices = np.flatnonzero(bin_ids == bin_index)
        if indices.size <= x.shape[1]:
            raise ValueError(
                f"Residual bin {bin_index} has {indices.size} observations; "
                f"need more than x_dim={x.shape[1]}."
            )
        residual = centered_residuals[indices]
        residual = residual - residual.mean(axis=0, keepdims=True)
        covariance = residual.T @ residual / float(indices.size)
        scale = max(float(np.trace(covariance) / x.shape[1]), 1e-8)
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
        eigenvalues = np.maximum(
            eigenvalues, float(eigenvalue_floor_relative) * scale
        )
        inverse_sqrt = (eigenvectors * (1.0 / np.sqrt(eigenvalues))[None, :]) @ eigenvectors.T
        one_bin = residual @ inverse_sqrt
        one_bin = one_bin - one_bin.mean(axis=0, keepdims=True)
        standardized[indices] = one_bin
        empirical_covariance = one_bin.T @ one_bin / float(indices.size)
        counts[bin_index] = int(indices.size)
        mean_norms[bin_index] = float(np.linalg.norm(one_bin.mean(axis=0)))
        covariance_errors[bin_index] = float(
            np.linalg.norm(empirical_covariance - identity, ord="fro") / np.sqrt(x.shape[1])
        )

    return StandardizedResidualBank(
        residuals=standardized,
        bin_ids=bin_ids,
        n_bins=int(n_bins),
        period=float(moments.period),
        counts=counts,
        mean_norms=mean_norms,
        covariance_errors=covariance_errors,
    )


def sample_moment_matched_surrogate(
    theta: np.ndarray,
    moments: PeriodicFourierMoments,
    residual_bank: StandardizedResidualBank,
    *,
    non_gaussian_weight: float,
    seed: int,
) -> np.ndarray:
    """Sample a Gaussian-to-empirical-residual surrogate at fixed conditions."""
    weight = float(non_gaussian_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("non_gaussian_weight must be in [0, 1].")
    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    gaussian = rng.standard_normal((values.shape[0], moments.x_dim))
    empirical = np.empty_like(gaussian)
    output_bins = residual_bank.bins_for_theta(values)
    for bin_index in range(residual_bank.n_bins):
        output_indices = np.flatnonzero(output_bins == bin_index)
        pool = residual_bank.residuals[residual_bank.bin_ids == bin_index]
        sampled = rng.integers(0, pool.shape[0], size=output_indices.size)
        empirical[output_indices] = pool[sampled]
    epsilon = np.sqrt(1.0 - weight) * gaussian + np.sqrt(weight) * empirical

    output = moments.mean(values)
    covariance_indices = moments.covariance_indices(values)
    for grid_index in np.unique(covariance_indices):
        indices = np.flatnonzero(covariance_indices == grid_index)
        cholesky = np.linalg.cholesky(moments.covariance_grid[int(grid_index)])
        output[indices] += epsilon[indices] @ cholesky.T
    return output.astype(np.float64)
