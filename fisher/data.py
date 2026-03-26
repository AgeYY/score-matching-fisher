from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def set_seed(seed: int) -> None:
    np.random.seed(seed)


@dataclass
class ToyConditionalGaussianDataset:
    theta_low: float = -3.0
    theta_high: float = 3.0
    sigma_x1: float = 0.30
    sigma_x2: float = 0.22
    rho: float = 0.15
    # Theta-dependent covariance parameters.
    cov_theta_amp1: float = 0.35
    cov_theta_amp2: float = 0.30
    cov_theta_amp_rho: float = 0.30
    cov_theta_freq1: float = 0.90
    cov_theta_freq2: float = 0.75
    cov_theta_freq_rho: float = 1.10
    cov_theta_phase1: float = 0.20
    cov_theta_phase2: float = -0.35
    cov_theta_phase_rho: float = 0.40
    rho_clip: float = 0.85
    seed: int = 42

    def __post_init__(self) -> None:
        if not (self.theta_low < self.theta_high):
            raise ValueError("theta_low must be smaller than theta_high.")
        if not (-0.99 < self.rho < 0.99):
            raise ValueError("rho must be in (-0.99, 0.99).")
        if not (0.0 <= self.cov_theta_amp1 < 0.95 and 0.0 <= self.cov_theta_amp2 < 0.95):
            raise ValueError("cov_theta_amp1 and cov_theta_amp2 must be in [0, 0.95).")
        if not (0.0 <= self.cov_theta_amp_rho <= 1.0):
            raise ValueError("cov_theta_amp_rho must be in [0, 1].")
        if not (0.1 <= self.rho_clip <= 0.95):
            raise ValueError("rho_clip must be in [0.1, 0.95].")
        self.rng = np.random.default_rng(self.seed)
        # Kept for backward compatibility with summary/prints as baseline covariance.
        self.cov = np.array(
            [
                [self.sigma_x1**2, self.rho * self.sigma_x1 * self.sigma_x2],
                [self.rho * self.sigma_x1 * self.sigma_x2, self.sigma_x2**2],
            ],
            dtype=np.float64,
        )
        self.cov = self.cov + 1e-8 * np.eye(2, dtype=np.float64)
        self.cov_chol = np.linalg.cholesky(self.cov)

    def sample_theta(self, n: int) -> np.ndarray:
        theta = self.rng.uniform(self.theta_low, self.theta_high, size=(n, 1))
        return theta.astype(np.float64)

    def tuning_curve(self, theta: np.ndarray) -> np.ndarray:
        t = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
        mu1 = 1.10 * np.sin(1.25 * t) + 0.28 * t
        mu2 = 0.85 * np.cos(1.05 * t + 0.30) - 0.12 * (t**2) + 0.05 * t
        return np.concatenate([mu1, mu2], axis=1)

    def tuning_curve_derivative(self, theta: np.ndarray) -> np.ndarray:
        t = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
        dmu1 = 1.10 * 1.25 * np.cos(1.25 * t) + 0.28
        dmu2 = -0.85 * 1.05 * np.sin(1.05 * t + 0.30) - 0.24 * t + 0.05
        return np.concatenate([dmu1, dmu2], axis=1)

    def covariance_components(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
        s1 = self.sigma_x1 * (1.0 + self.cov_theta_amp1 * np.sin(self.cov_theta_freq1 * t + self.cov_theta_phase1))
        s2 = self.sigma_x2 * (1.0 + self.cov_theta_amp2 * np.cos(self.cov_theta_freq2 * t + self.cov_theta_phase2))
        rho_raw = self.rho + self.cov_theta_amp_rho * np.sin(self.cov_theta_freq_rho * t + self.cov_theta_phase_rho)
        rho_t = np.clip(rho_raw, -self.rho_clip, self.rho_clip)
        return s1.reshape(-1), s2.reshape(-1), rho_t.reshape(-1)

    def covariance(self, theta: np.ndarray) -> np.ndarray:
        s1, s2, rho_t = self.covariance_components(theta)
        n = s1.shape[0]
        cov = np.zeros((n, 2, 2), dtype=np.float64)
        cov[:, 0, 0] = s1**2
        cov[:, 1, 1] = s2**2
        cov12 = rho_t * s1 * s2
        cov[:, 0, 1] = cov12
        cov[:, 1, 0] = cov12
        cov[:, 0, 0] += 1e-8
        cov[:, 1, 1] += 1e-8
        return cov

    def covariance_derivative(self, theta: np.ndarray) -> np.ndarray:
        t = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
        s1, s2, rho_t = self.covariance_components(t)
        ds1 = (
            self.sigma_x1
            * self.cov_theta_amp1
            * self.cov_theta_freq1
            * np.cos(self.cov_theta_freq1 * t + self.cov_theta_phase1).reshape(-1)
        )
        ds2 = (
            -self.sigma_x2
            * self.cov_theta_amp2
            * self.cov_theta_freq2
            * np.sin(self.cov_theta_freq2 * t + self.cov_theta_phase2).reshape(-1)
        )

        rho_raw = self.rho + self.cov_theta_amp_rho * np.sin(self.cov_theta_freq_rho * t + self.cov_theta_phase_rho)
        drho_raw = (
            self.cov_theta_amp_rho
            * self.cov_theta_freq_rho
            * np.cos(self.cov_theta_freq_rho * t + self.cov_theta_phase_rho)
        ).reshape(-1)
        unclipped = (rho_raw.reshape(-1) > -self.rho_clip) & (rho_raw.reshape(-1) < self.rho_clip)
        drho = np.where(unclipped, drho_raw, 0.0)

        n = s1.shape[0]
        dcov = np.zeros((n, 2, 2), dtype=np.float64)
        dcov[:, 0, 0] = 2.0 * s1 * ds1
        dcov[:, 1, 1] = 2.0 * s2 * ds2
        d12 = drho * s1 * s2 + rho_t * ds1 * s2 + rho_t * s1 * ds2
        dcov[:, 0, 1] = d12
        dcov[:, 1, 0] = d12
        return dcov

    def sample_x(self, theta: np.ndarray) -> np.ndarray:
        mu = self.tuning_curve(theta)
        eps = self.rng.standard_normal(size=mu.shape)
        cov = self.covariance(theta)
        chol = np.linalg.cholesky(cov)
        x = mu + np.einsum("nij,nj->ni", chol, eps)
        return x.astype(np.float64)

    def sample_joint(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        theta = self.sample_theta(n)
        x = self.sample_x(theta)
        return theta, x


def make_theta_grid(theta_low: float, theta_high: float, eval_margin: float, n_bins: int) -> np.ndarray:
    lo = theta_low + eval_margin
    hi = theta_high - eval_margin
    if lo >= hi:
        raise ValueError("Invalid eval range; increase theta range or decrease eval margin.")
    return np.linspace(lo, hi, n_bins, dtype=np.float64)


def make_local_decoder_data(
    dataset: ToyConditionalGaussianDataset,
    theta0: float,
    epsilon: float,
    n_train_local: int,
    n_eval_local: int,
) -> dict[str, np.ndarray]:
    theta_plus = theta0 + 0.5 * epsilon
    theta_minus = theta0 - 0.5 * epsilon

    t_train_pos = np.full((n_train_local, 1), theta_plus, dtype=np.float64)
    t_train_neg = np.full((n_train_local, 1), theta_minus, dtype=np.float64)
    x_train_pos = dataset.sample_x(t_train_pos)
    x_train_neg = dataset.sample_x(t_train_neg)
    x_train = np.concatenate([x_train_pos, x_train_neg], axis=0)
    y_train = np.concatenate(
        [np.ones((n_train_local,), dtype=np.float64), np.zeros((n_train_local,), dtype=np.float64)], axis=0
    )

    t_eval_pos = np.full((n_eval_local, 1), theta_plus, dtype=np.float64)
    t_eval_neg = np.full((n_eval_local, 1), theta_minus, dtype=np.float64)
    x_eval_pos = dataset.sample_x(t_eval_pos)
    x_eval_neg = dataset.sample_x(t_eval_neg)
    x_eval_mix = np.concatenate([x_eval_pos, x_eval_neg], axis=0)

    return {
        "theta_plus": np.array([[theta_plus]], dtype=np.float64),
        "theta_minus": np.array([[theta_minus]], dtype=np.float64),
        "x_train": x_train,
        "y_train": y_train,
        "x_eval_pos": x_eval_pos,
        "x_eval_neg": x_eval_neg,
        "x_eval_mix": x_eval_mix,
    }
