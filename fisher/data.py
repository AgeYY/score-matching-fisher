from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


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


@dataclass
class ToyConditionalGMMNonGaussianDataset:
    theta_low: float = -3.0
    theta_high: float = 3.0
    sigma_x1: float = 0.30
    sigma_x2: float = 0.22
    rho: float = 0.15
    sep_scale: float = 1.10
    sep_freq: float = 0.85
    sep_phase: float = 0.35
    mix_logit_scale: float = 1.40
    mix_bias: float = 0.00
    mix_freq: float = 0.95
    mix_phase: float = -0.20
    cov1_amp: float = 0.35
    cov2_amp: float = 0.30
    rho_amp1: float = 0.35
    rho_amp2: float = 0.25
    rho_clip: float = 0.85
    seed: int = 42

    def __post_init__(self) -> None:
        if not (self.theta_low < self.theta_high):
            raise ValueError("theta_low must be smaller than theta_high.")
        if not (0.0 < self.sep_scale):
            raise ValueError("sep_scale must be positive.")
        if not (0.1 <= self.rho_clip <= 0.95):
            raise ValueError("rho_clip must be in [0.1, 0.95].")
        self.rng = np.random.default_rng(self.seed)

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

    def _mix_weight(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
        z = self.mix_logit_scale * np.sin(self.mix_freq * t + self.mix_phase) + self.mix_bias
        pi = 1.0 / (1.0 + np.exp(-z))
        pi = np.clip(pi, 1e-4, 1.0 - 1e-4)
        dpi = pi * (1.0 - pi) * self.mix_logit_scale * self.mix_freq * np.cos(self.mix_freq * t + self.mix_phase)
        return pi.reshape(-1), dpi.reshape(-1)

    def _separation(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
        a = self.sep_scale * (1.0 + 0.25 * np.sin(self.sep_freq * t + self.sep_phase))
        da = self.sep_scale * 0.25 * self.sep_freq * np.cos(self.sep_freq * t + self.sep_phase)
        sep = np.concatenate(
            [
                a * np.cos(0.80 * t + 0.10),
                0.85 * a * np.sin(1.10 * t - 0.25),
            ],
            axis=1,
        )
        dsep = np.concatenate(
            [
                da * np.cos(0.80 * t + 0.10) - a * 0.80 * np.sin(0.80 * t + 0.10),
                0.85 * (da * np.sin(1.10 * t - 0.25) + a * 1.10 * np.cos(1.10 * t - 0.25)),
            ],
            axis=1,
        )
        return sep, dsep

    def _cov_from_components(
        self, s1: np.ndarray, s2: np.ndarray, rho_t: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n = s1.shape[0]
        cov = np.zeros((n, 2, 2), dtype=np.float64)
        cov[:, 0, 0] = s1**2 + 1e-8
        cov[:, 1, 1] = s2**2 + 1e-8
        cov12 = rho_t * s1 * s2
        cov[:, 0, 1] = cov12
        cov[:, 1, 0] = cov12
        inv_cov = np.linalg.inv(cov)
        return cov, inv_cov

    def component_means(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        base = self.tuning_curve(theta)
        sep, _ = self._separation(theta)
        return base + sep, base - sep

    def component_covariances(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t = np.asarray(theta, dtype=np.float64).reshape(-1, 1)

        s11 = self.sigma_x1 * (1.0 + self.cov1_amp * np.sin(0.90 * t + 0.20))
        s12 = self.sigma_x2 * (1.0 + self.cov2_amp * np.cos(0.70 * t - 0.15))
        r1_raw = self.rho + self.rho_amp1 * np.sin(1.10 * t + 0.40)
        r1 = np.clip(r1_raw, -self.rho_clip, self.rho_clip)

        s21 = (1.25 * self.sigma_x1) * (1.0 + 0.30 * np.cos(0.65 * t + 0.55))
        s22 = (0.80 * self.sigma_x2) * (1.0 + 0.35 * np.sin(0.95 * t - 0.35))
        r2_raw = -0.25 + self.rho_amp2 * np.cos(0.85 * t + 0.10)
        r2 = np.clip(r2_raw, -self.rho_clip, self.rho_clip)

        cov1, inv1 = self._cov_from_components(s11.reshape(-1), s12.reshape(-1), r1.reshape(-1))
        cov2, inv2 = self._cov_from_components(s21.reshape(-1), s22.reshape(-1), r2.reshape(-1))
        return cov1, cov2, inv1, inv2

    def _component_cov_derivatives(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.asarray(theta, dtype=np.float64).reshape(-1, 1)

        s11 = self.sigma_x1 * (1.0 + self.cov1_amp * np.sin(0.90 * t + 0.20)).reshape(-1)
        ds11 = (self.sigma_x1 * self.cov1_amp * 0.90 * np.cos(0.90 * t + 0.20)).reshape(-1)
        s12 = self.sigma_x2 * (1.0 + self.cov2_amp * np.cos(0.70 * t - 0.15)).reshape(-1)
        ds12 = (-self.sigma_x2 * self.cov2_amp * 0.70 * np.sin(0.70 * t - 0.15)).reshape(-1)
        r1_raw = (self.rho + self.rho_amp1 * np.sin(1.10 * t + 0.40)).reshape(-1)
        dr1_raw = (self.rho_amp1 * 1.10 * np.cos(1.10 * t + 0.40)).reshape(-1)
        r1 = np.clip(r1_raw, -self.rho_clip, self.rho_clip)
        dr1 = np.where((r1_raw > -self.rho_clip) & (r1_raw < self.rho_clip), dr1_raw, 0.0)

        s21 = ((1.25 * self.sigma_x1) * (1.0 + 0.30 * np.cos(0.65 * t + 0.55))).reshape(-1)
        ds21 = (-(1.25 * self.sigma_x1) * 0.30 * 0.65 * np.sin(0.65 * t + 0.55)).reshape(-1)
        s22 = ((0.80 * self.sigma_x2) * (1.0 + 0.35 * np.sin(0.95 * t - 0.35))).reshape(-1)
        ds22 = ((0.80 * self.sigma_x2) * 0.35 * 0.95 * np.cos(0.95 * t - 0.35)).reshape(-1)
        r2_raw = (-0.25 + self.rho_amp2 * np.cos(0.85 * t + 0.10)).reshape(-1)
        dr2_raw = (-self.rho_amp2 * 0.85 * np.sin(0.85 * t + 0.10)).reshape(-1)
        r2 = np.clip(r2_raw, -self.rho_clip, self.rho_clip)
        dr2 = np.where((r2_raw > -self.rho_clip) & (r2_raw < self.rho_clip), dr2_raw, 0.0)

        n = s11.shape[0]
        dcov1 = np.zeros((n, 2, 2), dtype=np.float64)
        dcov1[:, 0, 0] = 2.0 * s11 * ds11
        dcov1[:, 1, 1] = 2.0 * s12 * ds12
        dcov1[:, 0, 1] = dr1 * s11 * s12 + r1 * ds11 * s12 + r1 * s11 * ds12
        dcov1[:, 1, 0] = dcov1[:, 0, 1]

        dcov2 = np.zeros((n, 2, 2), dtype=np.float64)
        dcov2[:, 0, 0] = 2.0 * s21 * ds21
        dcov2[:, 1, 1] = 2.0 * s22 * ds22
        dcov2[:, 0, 1] = dr2 * s21 * s22 + r2 * ds21 * s22 + r2 * s21 * ds22
        dcov2[:, 1, 0] = dcov2[:, 0, 1]
        return dcov1, dcov2

    def sample_x(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
        n = theta.shape[0]
        pi, _ = self._mix_weight(theta)
        mu1, mu2 = self.component_means(theta)
        cov1, cov2, _, _ = self.component_covariances(theta)
        z = self.rng.uniform(size=n) < pi
        eps = self.rng.standard_normal(size=(n, 2))
        chol1 = np.linalg.cholesky(cov1)
        chol2 = np.linalg.cholesky(cov2)
        x1 = mu1 + np.einsum("nij,nj->ni", chol1, eps)
        x2 = mu2 + np.einsum("nij,nj->ni", chol2, eps)
        x = np.where(z[:, None], x1, x2)
        return x.astype(np.float64)

    def sample_joint(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        theta = self.sample_theta(n)
        x = self.sample_x(theta)
        return theta, x

    def _gaussian_logpdf_and_dtheta(
        self, x: np.ndarray, mu: np.ndarray, dmu: np.ndarray, cov: np.ndarray, dcov: np.ndarray, inv_cov: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        delta = x - mu
        quad = np.einsum("ni,nij,nj->n", delta, inv_cov, delta)
        _, logdet = np.linalg.slogdet(cov)
        d = x.shape[1]
        logp = -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)

        term_mean = np.einsum("ni,nij,nj->n", dmu, inv_cov, delta)
        a = np.einsum("nij,njk->nik", inv_cov, dcov)
        term_trace = -0.5 * np.einsum("nii->n", a)
        b = np.einsum("nij,njk->nik", a, inv_cov)  # inv_cov * dcov * inv_cov
        term_quad_cov = 0.5 * np.einsum("ni,nij,nj->n", delta, b, delta)
        dlogp = term_mean + term_trace + term_quad_cov
        return logp, dlogp

    def log_p_x_given_theta(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1, 2)
        theta = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
        pi, _ = self._mix_weight(theta)
        mu1, mu2 = self.component_means(theta)
        cov1, cov2, inv1, inv2 = self.component_covariances(theta)
        dcov1, dcov2 = self._component_cov_derivatives(theta)
        dmu_base = self.tuning_curve_derivative(theta)
        _, dsep = self._separation(theta)
        dmu1 = dmu_base + dsep
        dmu2 = dmu_base - dsep
        logn1, _ = self._gaussian_logpdf_and_dtheta(x, mu1, dmu1, cov1, dcov1, inv1)
        logn2, _ = self._gaussian_logpdf_and_dtheta(x, mu2, dmu2, cov2, dcov2, inv2)
        logmix = np.stack([np.log(pi) + logn1, np.log(1.0 - pi) + logn2], axis=0)
        return logsumexp(logmix, axis=0)

    def score_theta_exact(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1, 2)
        theta = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
        pi, dpi = self._mix_weight(theta)
        mu1, mu2 = self.component_means(theta)
        cov1, cov2, inv1, inv2 = self.component_covariances(theta)
        dcov1, dcov2 = self._component_cov_derivatives(theta)
        dmu_base = self.tuning_curve_derivative(theta)
        _, dsep = self._separation(theta)
        dmu1 = dmu_base + dsep
        dmu2 = dmu_base - dsep

        logn1, dlogn1 = self._gaussian_logpdf_and_dtheta(x, mu1, dmu1, cov1, dcov1, inv1)
        logn2, dlogn2 = self._gaussian_logpdf_and_dtheta(x, mu2, dmu2, cov2, dcov2, inv2)

        loga1 = np.log(pi) + logn1
        loga2 = np.log(1.0 - pi) + logn2
        norm = logsumexp(np.stack([loga1, loga2], axis=0), axis=0)
        w1 = np.exp(loga1 - norm)
        w2 = np.exp(loga2 - norm)

        term1 = dlogn1 + dpi / pi
        term2 = dlogn2 - dpi / (1.0 - pi)
        score = w1 * term1 + w2 * term2
        return score.reshape(-1)


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
