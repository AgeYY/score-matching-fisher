"""Wrap a full-dimensional diagonal Gaussian toy dataset as a marginal on the leading coordinates.

Used when training/evaluating on ``x[:, :k]`` sliced from a higher-dimensional archive while
ground-truth Hellinger MC follows the *marginal* :math:`p(x_1,\\ldots,x_k\\mid\\theta)` implied by
the full diagonal Gaussian model (same per-coordinate means and variances as in the parent).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fisher.data import ToyConditionalGaussianDataset, _theta_col

if TYPE_CHECKING:
    pass


class MarginalLeadingDimsGaussianWrapper:
    """Observation model on the first ``k`` coordinates of ``full`` (diagonal Gaussian branch)."""

    def __init__(self, full: ToyConditionalGaussianDataset, k: int) -> None:
        self._full = full
        self._k = int(k)
        if self._k < 1:
            raise ValueError("k must be >= 1.")
        if self._k > int(full.x_dim):
            raise ValueError(f"k={self._k} exceeds full model x_dim={full.x_dim}.")
        self.x_dim = self._k
        self.theta_low = float(full.theta_low)
        self.theta_high = float(full.theta_high)
        self.rng = full.rng
        self.diagonal_gaussian_observation_noise = bool(
            getattr(type(full), "diagonal_gaussian_observation_noise", True)
        )

    def sample_x(self, theta: np.ndarray) -> np.ndarray:
        x = self._full.sample_x(theta)
        return x[:, : self._k].astype(np.float64, copy=False)

    def tuning_curve(self, theta: np.ndarray) -> np.ndarray:
        mu = self._full.tuning_curve(theta)
        return mu[:, : self._k].astype(np.float64, copy=False)

    def covariance_scales(self, theta: np.ndarray) -> np.ndarray:
        s = self._full.covariance_scales(theta)
        return s[:, : self._k].astype(np.float64, copy=False)

    def log_p_x_given_theta(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        k = self._k
        x = np.asarray(x, dtype=np.float64).reshape(-1, k)
        theta = _theta_col(theta)
        mu = self._full.tuning_curve(theta)
        mu_k = mu[:, :k]
        v = self._full._variance_diag_from_mu(mu)
        vk = v[:, :k]
        delta = x - mu_k
        quad = np.sum((delta**2) / vk, axis=1)
        logdet = np.sum(np.log(vk), axis=1)
        return -0.5 * (float(k) * np.log(2.0 * np.pi) + logdet + quad)


class MarginalFirstDimGaussianWrapper(MarginalLeadingDimsGaussianWrapper):
    """1D marginal on coordinate 0 (backward-compatible alias)."""

    def __init__(self, full: ToyConditionalGaussianDataset) -> None:
        super().__init__(full, 1)
