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
    """Labeled trajectories ``mu(t)`` and ``scale * mu(t)`` with shared noise.

    Each trial contains a response vector at every point on a common time grid:

    ``X[c, r, u] ~ N(scale[c] * mu(t[u]), Sigma(t[u]))``.

    The two classes share ``Sigma(t)`` exactly.  This isolates recovery of the
    time-varying separation between their means while retaining the smooth
    random-amplitude tuning curves used in the continuous Fisher experiments.
    """

    x_dim: int = 40
    n_time_points: int = 301
    time_low: float = -6.0
    time_high: float = 6.0
    secondary_trajectory_scale: float = 2.0
    covariance_alpha: float = 0.65
    seed: int = 7

    def __post_init__(self) -> None:
        if int(self.x_dim) < 2:
            raise ValueError("x_dim must be at least 2.")
        if int(self.n_time_points) < 2:
            raise ValueError("n_time_points must be at least 2.")
        if not float(self.time_low) < float(self.time_high):
            raise ValueError("time_low must be smaller than time_high.")
        if not np.isfinite(self.secondary_trajectory_scale):
            raise ValueError("secondary_trajectory_scale must be finite.")
        if float(self.secondary_trajectory_scale) <= 0.0:
            raise ValueError("secondary_trajectory_scale must be positive.")
        if not np.isfinite(self.covariance_alpha) or float(self.covariance_alpha) < 0.0:
            raise ValueError("covariance_alpha must be finite and non-negative.")

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
        self.class_means = self.class_scales[:, None, None] * self.base_mean[None, :, :]
        self.shared_covariances = self.base_dataset.covariance(time_column)

    def sample_trials(self, n_trials_per_class: int) -> tuple[np.ndarray, np.ndarray]:
        """Return shuffled ``(trials, time, features)`` responses and class labels."""
        n_trials = int(n_trials_per_class)
        if n_trials < 1:
            raise ValueError("n_trials_per_class must be positive.")
        rng = np.random.default_rng(int(self.seed) + 1)
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

    def true_squared_mahalanobis_distance(self) -> np.ndarray:
        delta = self.class_means[1] - self.class_means[0]
        solved = np.linalg.solve(self.shared_covariances, delta[..., None])[..., 0]
        return np.einsum("td,td->t", delta, solved, optimize=True)

    def metadata(self, *, n_trials_per_class: int) -> dict[str, Any]:
        scale_label = f"{float(self.secondary_trajectory_scale):g}"
        return {
            "dataset": "two_class_time_resolved_gaussian_tuning_curves",
            "description": (
                "Class 0 has mean mu(t); class 1 has mean "
                f"{float(self.secondary_trajectory_scale):g} * mu(t); covariance is "
                "shared across classes and varies with time."
            ),
            "seed": int(self.seed),
            "x_dim": int(self.x_dim),
            "n_time_points": int(self.n_time_points),
            "time_low": float(self.time_low),
            "time_high": float(self.time_high),
            "n_trials_per_class": int(n_trials_per_class),
            "n_classes": 2,
            "class_names": ["Class 1: mu(t)", f"Class 2: {scale_label} mu(t)"],
            "class_scales": self.class_scales.tolist(),
            "covariance_alpha": float(self.covariance_alpha),
            "covariance_sharing": "shared across classes, varying with time",
            "time_sampling": "evenly spaced common grid within every trial",
            "response_shape": [
                2 * int(n_trials_per_class),
                int(self.n_time_points),
                int(self.x_dim),
            ],
        }
