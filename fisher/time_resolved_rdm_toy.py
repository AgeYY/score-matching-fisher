"""Two-class time-resolved Gaussian toy data for RDM estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fisher.data import ToyConditionalGaussianRandampSqrtdDataset
from fisher.dataset_family_recipes import family_recipe_dict


def correlation_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return Pearson correlation distance between two feature vectors."""
    first_values = np.asarray(first, dtype=np.float64).reshape(-1)
    second_values = np.asarray(second, dtype=np.float64).reshape(-1)
    if first_values.shape != second_values.shape:
        raise ValueError("Correlation inputs must have the same shape.")
    first_centered = first_values - np.mean(first_values)
    second_centered = second_values - np.mean(second_values)
    denominator = float(
        np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    )
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("Correlation distance is undefined for a constant vector.")
    correlation = float(np.dot(first_centered, second_centered) / denominator)
    return float(1.0 - np.clip(correlation, -1.0, 1.0))


TIME_RESOLVED_RDM_METRICS = (
    "correlation",
    "cosine",
    "euclidean",
    "mahalanobis_sq",
    "fid",
)


def mean_vector_distance(first: np.ndarray, second: np.ndarray, metric: str) -> float:
    """Return correlation, cosine, or literal Euclidean mean-vector distance."""

    first_values = np.asarray(first, dtype=np.float64).reshape(-1)
    second_values = np.asarray(second, dtype=np.float64).reshape(-1)
    if first_values.shape != second_values.shape:
        raise ValueError("Distance inputs must have the same shape.")
    name = str(metric).strip().lower()
    if name == "correlation":
        return correlation_distance(first_values, second_values)
    if name == "euclidean":
        return float(np.linalg.norm(first_values - second_values))
    if name != "cosine":
        raise ValueError(f"Unsupported mean-vector metric: {metric!r}.")
    denominator = float(np.linalg.norm(first_values) * np.linalg.norm(second_values))
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("Cosine distance is undefined for a zero-norm vector.")
    cosine = float(np.dot(first_values, second_values) / denominator)
    return float(1.0 - np.clip(cosine, -1.0, 1.0))


def squared_mahalanobis_distance(
    first: np.ndarray,
    second: np.ndarray,
    covariance: np.ndarray,
    *,
    ridge: float = 1e-8,
) -> float:
    """Return the squared Mahalanobis distance under one shared covariance."""

    delta = np.asarray(first, dtype=np.float64).reshape(-1) - np.asarray(
        second, dtype=np.float64
    ).reshape(-1)
    covariance_values = np.asarray(covariance, dtype=np.float64)
    if covariance_values.shape != (delta.size, delta.size):
        raise ValueError("covariance has an incompatible shape.")
    covariance_values = 0.5 * (covariance_values + covariance_values.T)
    diagonal = np.diag(covariance_values)
    off_diagonal_squared = float(np.sum(covariance_values**2) - np.sum(diagonal**2))
    if off_diagonal_squared <= 1e-12 * max(float(np.sum(covariance_values**2)), 1.0):
        solved = delta / (diagonal + float(ridge))
    else:
        covariance_values = covariance_values + float(ridge) * np.eye(delta.size)
        solved = np.linalg.solve(covariance_values, delta)
    return max(0.0, float(delta @ solved))


def gaussian_fid_distance(
    first_mean: np.ndarray,
    first_covariance: np.ndarray,
    second_mean: np.ndarray,
    second_covariance: np.ndarray,
) -> float:
    """Return squared Gaussian 2-Wasserstein distance (Gaussian FID)."""

    first = np.asarray(first_mean, dtype=np.float64).reshape(-1)
    second = np.asarray(second_mean, dtype=np.float64).reshape(-1)
    if first.shape != second.shape:
        raise ValueError("FID mean vectors must have the same shape.")
    first_cov = np.asarray(first_covariance, dtype=np.float64)
    second_cov = np.asarray(second_covariance, dtype=np.float64)
    expected = (first.size, first.size)
    if first_cov.shape != expected or second_cov.shape != expected:
        raise ValueError("FID covariances have incompatible shapes.")

    def psd_root(matrix: np.ndarray) -> np.ndarray:
        symmetric = 0.5 * (matrix + matrix.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        return (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))[None, :]) @ eigenvectors.T

    first_cov = 0.5 * (first_cov + first_cov.T)
    second_cov = 0.5 * (second_cov + second_cov.T)
    first_diagonal = np.diag(first_cov)
    second_diagonal = np.diag(second_cov)
    off_diagonal_squared = (
        float(np.sum(first_cov**2) - np.sum(first_diagonal**2))
        + float(np.sum(second_cov**2) - np.sum(second_diagonal**2))
    )
    total_squared = float(np.sum(first_cov**2) + np.sum(second_cov**2))
    delta = first - second
    if off_diagonal_squared <= 1e-12 * max(total_squared, 1.0):
        covariance_term = np.sum(
            first_diagonal
            + second_diagonal
            - 2.0 * np.sqrt(
                np.maximum(first_diagonal, 0.0)
                * np.maximum(second_diagonal, 0.0)
            )
        )
        return max(0.0, float(delta @ delta) + float(covariance_term))
    first_root = psd_root(first_cov)
    middle = first_root @ second_cov @ first_root
    middle_eigenvalues = np.maximum(
        np.linalg.eigvalsh(0.5 * (middle + middle.T)), 0.0
    )
    value = (
        float(delta @ delta)
        + float(np.trace(first_cov))
        + float(np.trace(second_cov))
        - 2.0 * float(np.sum(np.sqrt(middle_eigenvalues)))
    )
    return max(0.0, value)


def population_distance_trajectory(
    class_means: np.ndarray,
    shared_covariances: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Evaluate one requested population distance at every native time."""

    means = np.asarray(class_means, dtype=np.float64)
    covariances = np.asarray(shared_covariances, dtype=np.float64)
    if means.ndim != 3 or means.shape[0] != 2:
        raise ValueError("class_means must have shape (2, time, features).")
    if covariances.shape != (means.shape[1], means.shape[2], means.shape[2]):
        raise ValueError("shared_covariances has an incompatible shape.")
    name = str(metric).strip().lower()
    values = np.empty(means.shape[1], dtype=np.float64)
    for time_index in range(means.shape[1]):
        first, second = means[:, time_index]
        if name in {"correlation", "cosine", "euclidean"}:
            values[time_index] = mean_vector_distance(first, second, name)
        elif name == "mahalanobis_sq":
            values[time_index] = squared_mahalanobis_distance(
                first, second, covariances[time_index], ridge=0.0
            )
        elif name == "fid":
            values[time_index] = gaussian_fid_distance(
                first,
                covariances[time_index],
                second,
                covariances[time_index],
            )
        else:
            raise ValueError(f"Unsupported time-resolved RDM metric: {metric!r}.")
    return values


def estimate_binned_metric_distance(
    responses: np.ndarray,
    labels: np.ndarray,
    time: np.ndarray,
    *,
    bin_width: float,
    metric: str,
) -> dict[str, np.ndarray]:
    """Classical 500-ms-style plug-in estimator after pooling within time bins.

    Mean-only metrics use the two pooled class means. Because the toy
    population covariance is exactly diagonal, squared Mahalanobis uses one
    pooled within-class diagonal variance per bin and FID uses one diagonal
    variance per class and bin. This matches the covariance structure imposed
    on the corresponding flow estimators.
    """

    values = np.asarray(responses, dtype=np.float64)
    class_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    time_values = np.asarray(time, dtype=np.float64).reshape(-1)
    if values.ndim != 3 or values.shape[:2] != (class_labels.size, time_values.size):
        raise ValueError("responses must have shape (trials, time, features).")
    if not np.array_equal(np.unique(class_labels), np.asarray([0, 1])):
        raise ValueError("labels must contain exactly classes 0 and 1.")
    width = float(bin_width)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("bin_width must be finite and positive.")
    if np.any(np.diff(time_values) <= 0.0):
        raise ValueError("time must be strictly increasing.")
    name = str(metric).strip().lower()
    if name not in TIME_RESOLVED_RDM_METRICS:
        raise ValueError(f"Unsupported time-resolved RDM metric: {metric!r}.")

    time_low = float(time_values[0])
    time_high = float(time_values[-1])
    n_bins = max(1, int(np.ceil((time_high - time_low) / width)))
    bin_left = time_low + width * np.arange(n_bins, dtype=np.float64)
    bin_right = np.minimum(bin_left + width, time_high)
    bin_centers = np.empty(n_bins, dtype=np.float64)
    n_time_samples = np.empty(n_bins, dtype=np.int64)
    estimated_means = np.empty((n_bins, 2, values.shape[2]), dtype=np.float64)
    estimated_distance = np.empty(n_bins, dtype=np.float64)
    pooled_covariances = np.full(
        (n_bins, values.shape[2], values.shape[2]), np.nan, dtype=np.float64
    )
    class_covariances = np.full(
        (n_bins, 2, values.shape[2], values.shape[2]), np.nan, dtype=np.float64
    )

    for bin_index, (left, right) in enumerate(zip(bin_left, bin_right, strict=True)):
        if bin_index == n_bins - 1:
            mask = (time_values >= left) & (time_values <= right)
        else:
            mask = (time_values >= left) & (time_values < right)
        if not np.any(mask):
            raise ValueError(f"Time bin [{left}, {right}] contains no sampled time points.")
        bin_centers[bin_index] = float(np.mean(time_values[mask]))
        n_time_samples[bin_index] = int(np.sum(mask))
        class_rows: list[np.ndarray] = []
        for class_index in range(2):
            rows = values[class_labels == class_index][:, mask, :].reshape(-1, values.shape[2])
            class_rows.append(rows)
            estimated_means[bin_index, class_index] = np.mean(rows, axis=0)

        first_mean, second_mean = estimated_means[bin_index]
        if name in {"correlation", "cosine", "euclidean"}:
            estimated_distance[bin_index] = mean_vector_distance(
                first_mean, second_mean, name
            )
        elif name == "mahalanobis_sq":
            residuals = np.concatenate(
                [rows - mean for rows, mean in zip(class_rows, estimated_means[bin_index], strict=True)],
                axis=0,
            )
            pooled_variance = np.mean(residuals**2, axis=0)
            pooled_variance = np.maximum(pooled_variance, 1e-8)
            pooled_covariances[bin_index] = np.diag(pooled_variance)
            estimated_distance[bin_index] = squared_mahalanobis_distance(
                first_mean,
                second_mean,
                pooled_covariances[bin_index],
                ridge=0.0,
            )
        else:
            for class_index, rows in enumerate(class_rows):
                class_variance = np.var(rows, axis=0, ddof=1)
                class_covariances[bin_index, class_index] = np.diag(
                    np.maximum(class_variance, 1e-8)
                )
            estimated_distance[bin_index] = gaussian_fid_distance(
                first_mean,
                class_covariances[bin_index, 0],
                second_mean,
                class_covariances[bin_index, 1],
            )

    return {
        "bin_centers": bin_centers,
        "bin_left": bin_left,
        "bin_right": bin_right,
        "n_time_samples": n_time_samples,
        "estimated_class_means": estimated_means,
        "estimated_distance": estimated_distance,
        "pooled_covariances": pooled_covariances,
        "class_covariances": class_covariances,
        "metric": np.asarray(name),
    }


def estimate_binned_correlation_distance(
    responses: np.ndarray,
    labels: np.ndarray,
    time: np.ndarray,
    *,
    bin_width: float,
    true_class_means: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Estimate a two-class correlation RDM after pooling samples within bins."""
    values = np.asarray(responses, dtype=np.float64)
    class_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    time_values = np.asarray(time, dtype=np.float64).reshape(-1)
    if values.ndim != 3:
        raise ValueError("responses must have shape (trials, time, features).")
    if values.shape[:2] != (class_labels.size, time_values.size):
        raise ValueError("responses, labels, and time dimensions do not agree.")
    if not np.array_equal(np.unique(class_labels), np.asarray([0, 1])):
        raise ValueError("labels must contain exactly classes 0 and 1.")
    width = float(bin_width)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("bin_width must be finite and positive.")
    if np.any(np.diff(time_values) <= 0.0):
        raise ValueError("time must be strictly increasing.")
    truth = None
    if true_class_means is not None:
        truth = np.asarray(true_class_means, dtype=np.float64)
        if truth.shape != (2, time_values.size, values.shape[2]):
            raise ValueError(
                "true_class_means must have shape (2, time, features)."
            )

    time_low = float(time_values[0])
    time_high = float(time_values[-1])
    n_bins = max(1, int(np.ceil((time_high - time_low) / width)))
    bin_left = time_low + width * np.arange(n_bins, dtype=np.float64)
    bin_right = np.minimum(bin_left + width, time_high)
    bin_centers = np.empty(n_bins, dtype=np.float64)
    n_time_samples = np.empty(n_bins, dtype=np.int64)
    estimated_means = np.empty((n_bins, 2, values.shape[2]), dtype=np.float64)
    estimated_distance = np.empty(n_bins, dtype=np.float64)
    true_distance = np.empty(n_bins, dtype=np.float64) if truth is not None else None

    for bin_index, (left, right) in enumerate(
        zip(bin_left, bin_right, strict=True)
    ):
        if bin_index == n_bins - 1:
            mask = (time_values >= left) & (time_values <= right)
        else:
            mask = (time_values >= left) & (time_values < right)
        if not np.any(mask):
            raise ValueError(
                f"Time bin [{left}, {right}] contains no sampled time points."
            )
        bin_centers[bin_index] = float(np.mean(time_values[mask]))
        n_time_samples[bin_index] = int(np.sum(mask))
        for class_index in range(2):
            class_values = values[class_labels == class_index][:, mask, :]
            estimated_means[bin_index, class_index] = np.mean(
                class_values, axis=(0, 1)
            )
        estimated_distance[bin_index] = correlation_distance(
            estimated_means[bin_index, 0], estimated_means[bin_index, 1]
        )
        if truth is not None and true_distance is not None:
            first_truth = np.mean(truth[0, mask], axis=0)
            second_truth = np.mean(truth[1, mask], axis=0)
            true_distance[bin_index] = correlation_distance(
                first_truth, second_truth
            )

    result = {
        "bin_centers": bin_centers,
        "bin_left": bin_left,
        "bin_right": bin_right,
        "n_time_samples": n_time_samples,
        "estimated_class_means": estimated_means,
        "estimated_correlation_distance": estimated_distance,
    }
    if true_distance is not None:
        result["true_correlation_distance"] = true_distance
    return result


@dataclass
class TwoClassTimeResolvedGaussianToy:
    """Two labeled smooth trajectories with shared time-varying Gaussian noise.

    Each trial contains a response vector at every point on a common time grid:

    ``X[c, r, u] ~ N(mu_c(t[u]), Sigma(t[u]))``.

    ``trajectory_mode="scaled"`` retains the original pair ``mu(t)`` and
    ``scale * mu(t)``.  ``trajectory_mode="controlled_rotation"`` rotates the
    centered direction of the second class so its population correlation
    distance follows an analytically specified broad-plus-narrow curve.
    """

    x_dim: int = 40
    n_time_points: int = 301
    time_low: float = -6.0
    time_high: float = 6.0
    trajectory_mode: str = "scaled"
    secondary_trajectory_scale: float = 2.0
    covariance_alpha: float = 0.65
    rotation_distance_baseline: float = 0.05
    rotation_broad_amplitude: float = 0.25
    rotation_broad_center: float = -2.0
    rotation_broad_width: float = 0.8
    rotation_narrow_amplitude: float = 0.55
    rotation_narrow_center: float = 2.0
    rotation_narrow_width: float = 0.45
    rotation_auxiliary_seed_offset: int = 10_000
    seed: int = 7

    def __post_init__(self) -> None:
        if int(self.x_dim) < 2:
            raise ValueError("x_dim must be at least 2.")
        if int(self.n_time_points) < 2:
            raise ValueError("n_time_points must be at least 2.")
        if not float(self.time_low) < float(self.time_high):
            raise ValueError("time_low must be smaller than time_high.")
        self.trajectory_mode = str(self.trajectory_mode).strip().lower()
        if self.trajectory_mode not in {"scaled", "controlled_rotation"}:
            raise ValueError(
                "trajectory_mode must be 'scaled' or 'controlled_rotation'."
            )
        if self.trajectory_mode == "controlled_rotation" and int(self.x_dim) < 3:
            raise ValueError(
                "controlled_rotation requires x_dim >= 3 so the centered feature "
                "space contains an orthogonal direction."
            )
        if not np.isfinite(self.secondary_trajectory_scale):
            raise ValueError("secondary_trajectory_scale must be finite.")
        if float(self.secondary_trajectory_scale) <= 0.0:
            raise ValueError("secondary_trajectory_scale must be positive.")
        if not np.isfinite(self.covariance_alpha) or float(self.covariance_alpha) < 0.0:
            raise ValueError("covariance_alpha must be finite and non-negative.")
        rotation_parameters = (
            self.rotation_distance_baseline,
            self.rotation_broad_amplitude,
            self.rotation_broad_center,
            self.rotation_broad_width,
            self.rotation_narrow_amplitude,
            self.rotation_narrow_center,
            self.rotation_narrow_width,
        )
        if not np.all(np.isfinite(rotation_parameters)):
            raise ValueError("All controlled-rotation parameters must be finite.")
        if float(self.rotation_broad_width) <= 0.0 or float(self.rotation_narrow_width) <= 0.0:
            raise ValueError("Controlled-rotation widths must be positive.")
        if (
            float(self.rotation_distance_baseline) < 0.0
            or float(self.rotation_broad_amplitude) < 0.0
            or float(self.rotation_narrow_amplitude) < 0.0
        ):
            raise ValueError(
                "Controlled-rotation baseline and amplitudes must be non-negative."
            )

        recipe = family_recipe_dict("randamp_gaussian_sqrtd")
        recipe_keys = (
            "sigma_x1",
            "sigma_x2",
            "randamp_mu_low",
            "randamp_mu_high",
            "randamp_kappa",
            "randamp_omega",
        )
        self.base_dataset = ToyConditionalGaussianRandampSqrtdDataset(
            theta_low=float(self.time_low),
            theta_high=float(self.time_high),
            x_dim=int(self.x_dim),
            seed=int(self.seed),
            cov_theta_amp1=float(self.covariance_alpha),
            cov_theta_amp2=float(self.covariance_alpha),
            **{key: recipe[key] for key in recipe_keys},
        )
        self.time = np.linspace(
            float(self.time_low),
            float(self.time_high),
            int(self.n_time_points),
            dtype=np.float64,
        )
        time_column = self.time[:, None]
        self.base_mean = self.base_dataset.tuning_curve(time_column)
        self.class_scales = np.asarray(
            [1.0, float(self.secondary_trajectory_scale)], dtype=np.float64
        )
        if self.trajectory_mode == "scaled":
            self.class_means = (
                self.class_scales[:, None, None] * self.base_mean[None, :, :]
            )
            self.target_correlation_distance = np.zeros_like(self.time)
            self.auxiliary_mean = None
            self.class_names = ["Class 1", "Class 2 (scaled)"]
        else:
            self._set_controlled_rotation_means(recipe, recipe_keys, time_column)
        self.shared_covariances = self.base_dataset.covariance(time_column)

    def _set_controlled_rotation_means(
        self,
        recipe: dict[str, Any],
        recipe_keys: tuple[str, ...],
        time_column: np.ndarray,
    ) -> None:
        auxiliary_dataset = ToyConditionalGaussianRandampSqrtdDataset(
            theta_low=float(self.time_low),
            theta_high=float(self.time_high),
            x_dim=int(self.x_dim),
            seed=int(self.seed) + int(self.rotation_auxiliary_seed_offset),
            cov_theta_amp1=float(self.covariance_alpha),
            cov_theta_amp2=float(self.covariance_alpha),
            **{key: recipe[key] for key in recipe_keys},
        )
        self.auxiliary_mean = auxiliary_dataset.tuning_curve(time_column)

        base_feature_mean = np.mean(self.base_mean, axis=1, keepdims=True)
        base_centered = self.base_mean - base_feature_mean
        base_norm = np.linalg.norm(base_centered, axis=1, keepdims=True)
        if np.any(base_norm <= np.finfo(np.float64).eps):
            raise RuntimeError("The base tuning trajectory contains a constant vector.")
        first_direction = base_centered / base_norm

        auxiliary_centered = self.auxiliary_mean - np.mean(
            self.auxiliary_mean, axis=1, keepdims=True
        )
        projection = np.sum(auxiliary_centered * first_direction, axis=1, keepdims=True)
        orthogonal = auxiliary_centered - projection * first_direction
        orthogonal_norm = np.linalg.norm(orthogonal, axis=1, keepdims=True)
        degenerate = orthogonal_norm[:, 0] <= 1e-10
        if np.any(degenerate):
            centered_basis = np.eye(int(self.x_dim), dtype=np.float64)
            centered_basis -= 1.0 / float(self.x_dim)
            for time_index in np.flatnonzero(degenerate):
                candidates = centered_basis - np.outer(
                    centered_basis @ first_direction[time_index],
                    first_direction[time_index],
                )
                candidate_norms = np.linalg.norm(candidates, axis=1)
                best = int(np.argmax(candidate_norms))
                orthogonal[time_index] = candidates[best]
                orthogonal_norm[time_index, 0] = candidate_norms[best]
        second_basis_direction = orthogonal / orthogonal_norm

        broad = float(self.rotation_broad_amplitude) * np.exp(
            -0.5
            * ((self.time - float(self.rotation_broad_center)) / float(self.rotation_broad_width))
            ** 2
        )
        narrow = float(self.rotation_narrow_amplitude) * np.exp(
            -0.5
            * ((self.time - float(self.rotation_narrow_center)) / float(self.rotation_narrow_width))
            ** 2
        )
        target = float(self.rotation_distance_baseline) + broad + narrow
        if np.any((target < 0.0) | (target > 2.0)):
            raise ValueError(
                "The controlled-rotation target correlation distance must lie in [0, 2]."
            )
        correlation = 1.0 - target
        second_direction = (
            correlation[:, None] * first_direction
            + np.sqrt(np.clip(1.0 - correlation**2, 0.0, None))[:, None]
            * second_basis_direction
        )
        second_mean = (
            base_feature_mean
            + float(self.secondary_trajectory_scale) * base_norm * second_direction
        )
        self.class_means = np.stack([self.base_mean, second_mean], axis=0)
        self.target_correlation_distance = target
        self.class_names = ["Class 1", "Class 2 (rotated)"]

    def sample_trials(
        self,
        n_trials_per_class: int,
        *,
        sample_seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return shuffled ``(trials, time, features)`` responses and class labels."""
        n_trials = int(n_trials_per_class)
        if n_trials < 1:
            raise ValueError("n_trials_per_class must be positive.")
        resolved_sample_seed = (
            int(self.seed) + 1 if sample_seed is None else int(sample_seed)
        )
        rng = np.random.default_rng(resolved_sample_seed)
        cholesky = np.linalg.cholesky(self.shared_covariances)
        responses: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for class_index in range(2):
            noise = rng.standard_normal(
                size=(n_trials, int(self.n_time_points), int(self.x_dim))
            )
            colored_noise = np.einsum(
                "tij,ntj->nti", cholesky, noise, optimize=True
            )
            responses.append(self.class_means[class_index][None, :, :] + colored_noise)
            labels.append(np.full(n_trials, class_index, dtype=np.int64))
        values = np.concatenate(responses, axis=0).astype(np.float64)
        class_labels = np.concatenate(labels, axis=0)
        order = rng.permutation(class_labels.size)
        return values[order], class_labels[order]

    def true_squared_euclidean_distance(self) -> np.ndarray:
        delta = self.class_means[1] - self.class_means[0]
        return np.einsum("td,td->t", delta, delta, optimize=True)

    def true_euclidean_distance(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.true_squared_euclidean_distance(), 0.0))

    def true_squared_mahalanobis_distance(self) -> np.ndarray:
        delta = self.class_means[1] - self.class_means[0]
        diagonal = np.diagonal(self.shared_covariances, axis1=1, axis2=2)
        total_squared_norm = float(np.sum(self.shared_covariances**2))
        diagonal_squared_norm = float(np.sum(diagonal**2))
        if total_squared_norm - diagonal_squared_norm <= 1e-12 * max(
            total_squared_norm, 1.0
        ):
            solved = delta / diagonal
        else:
            solved = np.linalg.solve(self.shared_covariances, delta[..., None])[
                ..., 0
            ]
        return np.einsum("td,td->t", delta, solved, optimize=True)

    def true_correlation_distance(self) -> np.ndarray:
        return np.asarray(
            [
                correlation_distance(first, second)
                for first, second in zip(
                    self.class_means[0], self.class_means[1], strict=True
                )
            ],
            dtype=np.float64,
        )

    def true_cosine_distance(self) -> np.ndarray:
        return population_distance_trajectory(
            self.class_means, self.shared_covariances, "cosine"
        )

    def true_fid_distance(self) -> np.ndarray:
        # Both classes share the same covariance, so the Bures covariance term
        # cancels and population FID equals squared Euclidean distance.
        return self.true_squared_euclidean_distance().copy()

    def metadata(
        self,
        *,
        n_trials_per_class: int,
        sample_seed: int | None = None,
    ) -> dict[str, Any]:
        return {
            "dataset": "two_class_time_resolved_gaussian_tuning_curves",
            "description": (
                "Two smooth class-mean trajectories with shared time-varying "
                f"covariance; trajectory_mode={self.trajectory_mode}."
            ),
            "trajectory_mode": self.trajectory_mode,
            "seed": int(self.seed),
            "sample_seed": (
                int(self.seed) + 1 if sample_seed is None else int(sample_seed)
            ),
            "x_dim": int(self.x_dim),
            "n_time_points": int(self.n_time_points),
            "time_low": float(self.time_low),
            "time_high": float(self.time_high),
            "n_trials_per_class": int(n_trials_per_class),
            "n_classes": 2,
            "class_names": self.class_names,
            "class_scales": self.class_scales.tolist(),
            "target_correlation_distance_min": float(
                np.min(self.target_correlation_distance)
            ),
            "target_correlation_distance_max": float(
                np.max(self.target_correlation_distance)
            ),
            "rotation_distance_parameters": {
                "baseline": float(self.rotation_distance_baseline),
                "broad_amplitude": float(self.rotation_broad_amplitude),
                "broad_center": float(self.rotation_broad_center),
                "broad_width": float(self.rotation_broad_width),
                "narrow_amplitude": float(self.rotation_narrow_amplitude),
                "narrow_center": float(self.rotation_narrow_center),
                "narrow_width": float(self.rotation_narrow_width),
                "auxiliary_seed_offset": int(self.rotation_auxiliary_seed_offset),
            },
            "covariance_alpha": float(self.covariance_alpha),
            "covariance_sharing": "shared across classes, varying with time",
            "time_sampling": "evenly spaced common grid within every trial",
            "response_shape": [
                2 * int(n_trials_per_class),
                int(self.n_time_points),
                int(self.x_dim),
            ],
        }
