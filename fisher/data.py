from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def _theta_col(theta: np.ndarray) -> np.ndarray:
    return np.asarray(theta, dtype=np.float64).reshape(-1, 1)


@dataclass
class ToyConditionalGaussianDataset:
    theta_low: float = -3.0
    theta_high: float = 3.0
    x_dim: int = 2
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
        if self.x_dim < 2:
            raise ValueError("x_dim must be >= 2.")
        if not (-0.99 < self.rho < 0.99):
            raise ValueError("rho must be in (-0.99, 0.99).")
        if not (0.0 <= self.cov_theta_amp1 < 0.95 and 0.0 <= self.cov_theta_amp2 < 0.95):
            raise ValueError("cov_theta_amp1 and cov_theta_amp2 must be in [0, 0.95).")
        if not (0.0 <= self.cov_theta_amp_rho <= 1.0):
            raise ValueError("cov_theta_amp_rho must be in [0, 1].")
        if not (0.1 <= self.rho_clip <= 0.95):
            raise ValueError("rho_clip must be in [0.1, 0.95].")

        self.rng = np.random.default_rng(self.seed)
        idx = np.arange(1, self.x_dim + 1, dtype=np.float64)

        # Mean function coefficients.
        self._mu_amp_sin = 1.10 / (1.0 + 0.08 * (idx - 1.0))
        self._mu_amp_cos = 0.35 / (1.0 + 0.06 * (idx - 1.0))
        self._mu_freq_sin = 1.25 + 0.07 * (idx - 1.0)
        self._mu_freq_cos = 0.60 + 0.05 * (idx - 1.0)
        self._mu_phase = 0.30 * (idx - 1.0)
        self._mu_lin = 0.08 * (idx / max(self.x_dim, 2) - 0.5)
        self._mu_quad = 0.02 * np.where((idx.astype(np.int64) % 2) == 0, -1.0, 1.0)

        # Theta-dependent per-dimension covariance scales.
        self._sigma_base = np.linspace(self.sigma_x1, self.sigma_x2, self.x_dim, dtype=np.float64)
        self._sigma_amp1 = np.clip(
            self.cov_theta_amp1 * (0.72 + 0.22 * np.sin(0.45 * idx + 0.10)),
            0.0,
            0.95,
        )
        self._sigma_amp2 = np.clip(
            self.cov_theta_amp2 * (0.68 + 0.25 * np.cos(0.40 * idx - 0.20)),
            0.0,
            0.95,
        )
        self._sigma_freq1 = self.cov_theta_freq1 + 0.08 * (idx - 1.0)
        self._sigma_freq2 = self.cov_theta_freq2 + 0.06 * (idx - 1.0)
        self._sigma_phase1 = self.cov_theta_phase1 + 0.13 * (idx - 1.0)
        self._sigma_phase2 = self.cov_theta_phase2 - 0.11 * (idx - 1.0)

        # Kept for backward compatibility with summary/prints as baseline covariance.
        self.cov = np.diag(self._sigma_base**2)
        if self.x_dim == 2:
            off_diag = self.rho * self._sigma_base[0] * self._sigma_base[1]
            self.cov[0, 1] = off_diag
            self.cov[1, 0] = off_diag
        self.cov = self.cov + 1e-8 * np.eye(self.x_dim, dtype=np.float64)
        self.cov_chol = np.linalg.cholesky(self.cov)

    def sample_theta(self, n: int) -> np.ndarray:
        theta = self.rng.uniform(self.theta_low, self.theta_high, size=(n, 1))
        return theta.astype(np.float64)

    def tuning_curve(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_col(theta)
        phase = self._mu_phase.reshape(1, -1)
        mu = (
            self._mu_amp_sin.reshape(1, -1) * np.sin(t * self._mu_freq_sin.reshape(1, -1) + phase)
            + self._mu_amp_cos.reshape(1, -1) * np.cos(t * self._mu_freq_cos.reshape(1, -1) - 0.5 * phase)
            + self._mu_lin.reshape(1, -1) * t
            + self._mu_quad.reshape(1, -1) * (t**2)
        )
        return mu

    def tuning_curve_derivative(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_col(theta)
        phase = self._mu_phase.reshape(1, -1)
        dmu = (
            self._mu_amp_sin.reshape(1, -1)
            * self._mu_freq_sin.reshape(1, -1)
            * np.cos(t * self._mu_freq_sin.reshape(1, -1) + phase)
            - self._mu_amp_cos.reshape(1, -1)
            * self._mu_freq_cos.reshape(1, -1)
            * np.sin(t * self._mu_freq_cos.reshape(1, -1) - 0.5 * phase)
            + self._mu_lin.reshape(1, -1)
            + 2.0 * self._mu_quad.reshape(1, -1) * t
        )
        return dmu

    def covariance_scales(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_col(theta)
        scales = self._sigma_base.reshape(1, -1) * (
            1.0
            + self._sigma_amp1.reshape(1, -1) * np.sin(t * self._sigma_freq1.reshape(1, -1) + self._sigma_phase1)
            + self._sigma_amp2.reshape(1, -1) * np.cos(t * self._sigma_freq2.reshape(1, -1) + self._sigma_phase2)
        )
        return np.maximum(scales, 0.05 * self._sigma_base.reshape(1, -1))

    def covariance_scales_derivative(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_col(theta)
        dscale = self._sigma_base.reshape(1, -1) * (
            self._sigma_amp1.reshape(1, -1)
            * self._sigma_freq1.reshape(1, -1)
            * np.cos(t * self._sigma_freq1.reshape(1, -1) + self._sigma_phase1)
            - self._sigma_amp2.reshape(1, -1)
            * self._sigma_freq2.reshape(1, -1)
            * np.sin(t * self._sigma_freq2.reshape(1, -1) + self._sigma_phase2)
        )
        return dscale

    def covariance_components(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scales = self.covariance_scales(theta)
        t = _theta_col(theta)
        rho_raw = self.rho + self.cov_theta_amp_rho * np.sin(self.cov_theta_freq_rho * t + self.cov_theta_phase_rho)
        rho_t = np.clip(rho_raw, -self.rho_clip, self.rho_clip).reshape(-1)
        return scales[:, 0], scales[:, 1], rho_t

    def covariance(self, theta: np.ndarray) -> np.ndarray:
        scales = self.covariance_scales(theta)
        n = scales.shape[0]
        cov = np.zeros((n, self.x_dim, self.x_dim), dtype=np.float64)
        diag_vals = scales**2 + 1e-8
        for j in range(self.x_dim):
            cov[:, j, j] = diag_vals[:, j]

        # Keep 2D behavior for backward compatibility.
        if self.x_dim == 2:
            _, _, rho_t = self.covariance_components(theta)
            cov12 = rho_t * scales[:, 0] * scales[:, 1]
            cov[:, 0, 1] = cov12
            cov[:, 1, 0] = cov12
        return cov

    def covariance_derivative(self, theta: np.ndarray) -> np.ndarray:
        scales = self.covariance_scales(theta)
        dscale = self.covariance_scales_derivative(theta)
        n = scales.shape[0]
        dcov = np.zeros((n, self.x_dim, self.x_dim), dtype=np.float64)
        for j in range(self.x_dim):
            dcov[:, j, j] = 2.0 * scales[:, j] * dscale[:, j]

        if self.x_dim == 2:
            t = _theta_col(theta)
            _, _, rho_t = self.covariance_components(theta)
            rho_raw = self.rho + self.cov_theta_amp_rho * np.sin(self.cov_theta_freq_rho * t + self.cov_theta_phase_rho)
            drho_raw = (
                self.cov_theta_amp_rho
                * self.cov_theta_freq_rho
                * np.cos(self.cov_theta_freq_rho * t + self.cov_theta_phase_rho)
            ).reshape(-1)
            unclipped = (rho_raw.reshape(-1) > -self.rho_clip) & (rho_raw.reshape(-1) < self.rho_clip)
            drho = np.where(unclipped, drho_raw, 0.0)
            d12 = (
                drho * scales[:, 0] * scales[:, 1]
                + rho_t * dscale[:, 0] * scales[:, 1]
                + rho_t * scales[:, 0] * dscale[:, 1]
            )
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
    x_dim: int = 2
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
        if self.x_dim < 2:
            raise ValueError("x_dim must be >= 2.")
        if not (0.0 < self.sep_scale):
            raise ValueError("sep_scale must be positive.")
        if not (0.1 <= self.rho_clip <= 0.95):
            raise ValueError("rho_clip must be in [0.1, 0.95].")

        self.rng = np.random.default_rng(self.seed)
        idx = np.arange(1, self.x_dim + 1, dtype=np.float64)

        self._mu_amp_sin = 1.05 / (1.0 + 0.07 * (idx - 1.0))
        self._mu_amp_cos = 0.30 / (1.0 + 0.06 * (idx - 1.0))
        self._mu_freq_sin = 1.20 + 0.06 * (idx - 1.0)
        self._mu_freq_cos = 0.65 + 0.05 * (idx - 1.0)
        self._mu_phase = 0.25 * (idx - 1.0)
        self._mu_lin = 0.06 * (idx / max(self.x_dim, 2) - 0.5)
        self._mu_quad = 0.018 * np.where((idx.astype(np.int64) % 2) == 0, -1.0, 1.0)

        self._sep_weight = 0.95 / (1.0 + 0.05 * (idx - 1.0))
        self._sep_freq = 0.72 + 0.09 * (idx - 1.0)
        self._sep_phase = 0.20 * (idx - 1.0)

        self._sigma1_base = np.linspace(self.sigma_x1, self.sigma_x2, self.x_dim, dtype=np.float64)
        self._sigma2_base = np.linspace(1.15 * self.sigma_x1, 0.85 * self.sigma_x2, self.x_dim, dtype=np.float64)
        self._cov1_amp = np.clip(self.cov1_amp * (0.70 + 0.25 * np.sin(0.50 * idx)), 0.0, 0.95)
        self._cov2_amp = np.clip(self.cov2_amp * (0.70 + 0.25 * np.cos(0.40 * idx + 0.15)), 0.0, 0.95)
        self._cov1_freq = 0.90 + 0.05 * (idx - 1.0)
        self._cov2_freq = 0.65 + 0.07 * (idx - 1.0)
        self._cov1_phase = 0.20 + 0.10 * (idx - 1.0)
        self._cov2_phase = -0.15 - 0.12 * (idx - 1.0)

    def sample_theta(self, n: int) -> np.ndarray:
        theta = self.rng.uniform(self.theta_low, self.theta_high, size=(n, 1))
        return theta.astype(np.float64)

    def tuning_curve(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_col(theta)
        phase = self._mu_phase.reshape(1, -1)
        mu = (
            self._mu_amp_sin.reshape(1, -1) * np.sin(t * self._mu_freq_sin.reshape(1, -1) + phase)
            + self._mu_amp_cos.reshape(1, -1) * np.cos(t * self._mu_freq_cos.reshape(1, -1) - 0.5 * phase)
            + self._mu_lin.reshape(1, -1) * t
            + self._mu_quad.reshape(1, -1) * (t**2)
        )
        return mu

    def tuning_curve_derivative(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_col(theta)
        phase = self._mu_phase.reshape(1, -1)
        dmu = (
            self._mu_amp_sin.reshape(1, -1)
            * self._mu_freq_sin.reshape(1, -1)
            * np.cos(t * self._mu_freq_sin.reshape(1, -1) + phase)
            - self._mu_amp_cos.reshape(1, -1)
            * self._mu_freq_cos.reshape(1, -1)
            * np.sin(t * self._mu_freq_cos.reshape(1, -1) - 0.5 * phase)
            + self._mu_lin.reshape(1, -1)
            + 2.0 * self._mu_quad.reshape(1, -1) * t
        )
        return dmu

    def _mix_weight(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = _theta_col(theta)
        z = self.mix_logit_scale * np.sin(self.mix_freq * t + self.mix_phase) + self.mix_bias
        pi = 1.0 / (1.0 + np.exp(-z))
        pi = np.clip(pi, 1e-4, 1.0 - 1e-4)
        dpi = pi * (1.0 - pi) * self.mix_logit_scale * self.mix_freq * np.cos(self.mix_freq * t + self.mix_phase)
        return pi.reshape(-1), dpi.reshape(-1)

    def _separation(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = _theta_col(theta)
        a = self.sep_scale * (1.0 + 0.25 * np.sin(self.sep_freq * t + self.sep_phase))
        da = self.sep_scale * 0.25 * self.sep_freq * np.cos(self.sep_freq * t + self.sep_phase)

        sep = self._sep_weight.reshape(1, -1) * a * np.sin(t * self._sep_freq.reshape(1, -1) + self._sep_phase)
        dsep = self._sep_weight.reshape(1, -1) * (
            da * np.sin(t * self._sep_freq.reshape(1, -1) + self._sep_phase)
            + a * self._sep_freq.reshape(1, -1) * np.cos(t * self._sep_freq.reshape(1, -1) + self._sep_phase)
        )
        return sep, dsep

    def _diag_cov_and_inv(self, scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, d = scales.shape
        cov = np.zeros((n, d, d), dtype=np.float64)
        inv_cov = np.zeros((n, d, d), dtype=np.float64)
        diag = scales**2 + 1e-8
        inv_diag = 1.0 / diag
        for j in range(d):
            cov[:, j, j] = diag[:, j]
            inv_cov[:, j, j] = inv_diag[:, j]
        return cov, inv_cov

    def component_means(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        base = self.tuning_curve(theta)
        sep, _ = self._separation(theta)
        return base + sep, base - sep

    def component_covariances(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t = _theta_col(theta)

        s1 = self._sigma1_base.reshape(1, -1) * (
            1.0 + self._cov1_amp.reshape(1, -1) * np.sin(t * self._cov1_freq.reshape(1, -1) + self._cov1_phase)
        )
        s2 = self._sigma2_base.reshape(1, -1) * (
            1.0 + self._cov2_amp.reshape(1, -1) * np.cos(t * self._cov2_freq.reshape(1, -1) + self._cov2_phase)
        )
        s1 = np.maximum(s1, 0.05 * self._sigma1_base.reshape(1, -1))
        s2 = np.maximum(s2, 0.05 * self._sigma2_base.reshape(1, -1))

        cov1, inv1 = self._diag_cov_and_inv(s1)
        cov2, inv2 = self._diag_cov_and_inv(s2)
        return cov1, cov2, inv1, inv2

    def _component_cov_derivatives(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = _theta_col(theta)

        s1 = self._sigma1_base.reshape(1, -1) * (
            1.0 + self._cov1_amp.reshape(1, -1) * np.sin(t * self._cov1_freq.reshape(1, -1) + self._cov1_phase)
        )
        ds1 = self._sigma1_base.reshape(1, -1) * (
            self._cov1_amp.reshape(1, -1)
            * self._cov1_freq.reshape(1, -1)
            * np.cos(t * self._cov1_freq.reshape(1, -1) + self._cov1_phase)
        )

        s2 = self._sigma2_base.reshape(1, -1) * (
            1.0 + self._cov2_amp.reshape(1, -1) * np.cos(t * self._cov2_freq.reshape(1, -1) + self._cov2_phase)
        )
        ds2 = self._sigma2_base.reshape(1, -1) * (
            -self._cov2_amp.reshape(1, -1)
            * self._cov2_freq.reshape(1, -1)
            * np.sin(t * self._cov2_freq.reshape(1, -1) + self._cov2_phase)
        )

        s1 = np.maximum(s1, 0.05 * self._sigma1_base.reshape(1, -1))
        s2 = np.maximum(s2, 0.05 * self._sigma2_base.reshape(1, -1))

        n, d = s1.shape
        dcov1 = np.zeros((n, d, d), dtype=np.float64)
        dcov2 = np.zeros((n, d, d), dtype=np.float64)
        for j in range(d):
            dcov1[:, j, j] = 2.0 * s1[:, j] * ds1[:, j]
            dcov2[:, j, j] = 2.0 * s2[:, j] * ds2[:, j]
        return dcov1, dcov2

    def sample_x(self, theta: np.ndarray) -> np.ndarray:
        theta = _theta_col(theta)
        n = theta.shape[0]
        pi, _ = self._mix_weight(theta)
        mu1, mu2 = self.component_means(theta)
        cov1, cov2, _, _ = self.component_covariances(theta)
        z = self.rng.uniform(size=n) < pi
        eps = self.rng.standard_normal(size=(n, self.x_dim))
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
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.x_dim)
        theta = _theta_col(theta)
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
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.x_dim)
        theta = _theta_col(theta)
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
    dataset: ToyConditionalGaussianDataset | ToyConditionalGMMNonGaussianDataset,
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
