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
    tuning_curve_family: str = "cosine"  # "cosine" | "von_mises_raw"
    vm_mu_amp: float = 1.0
    vm_kappa: float = 1.0
    vm_omega: float = 1.0
    sigma_x1: float = 0.30
    sigma_x2: float = 0.30
    rho: float = 0.15
    # Activity coupling for diagonal variance: Var_j = sigma_base_j^2 * (1 + alpha_j * |mu_j|).
    # Per-dimension alpha_j interpolates between these endpoints (also used in dataset .npz meta).
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
        if self.tuning_curve_family not in ("cosine", "von_mises_raw"):
            raise ValueError('tuning_curve_family must be "cosine" or "von_mises_raw".')
        if self.tuning_curve_family == "von_mises_raw":
            if self.vm_kappa < 0.0:
                raise ValueError("vm_kappa must be non-negative for von_mises_raw.")
            if self.vm_mu_amp <= 0.0:
                raise ValueError("vm_mu_amp must be positive for von_mises_raw.")

        self.rng = np.random.default_rng(self.seed)

        # Cosine tuning curves: mu_j(theta) = A * cos(omega * theta + phi_j)
        # Von Mises (raw): mu_j(theta) = A * exp(kappa * cos(omega * theta - phi_j))
        self._mu_amp = 1.0
        self._mu_omega = 1.0
        # phi_j = 2*pi*(j-1)/d. For d=2, mu(theta)=mu(-theta) (cosine or von_mises_raw), hence
        # p(x|theta)=p(x|-theta) under diagonal Var_j(|mu|). For d>=3, mu(theta) != mu(-theta) in general.
        self._mu_phases = 2.0 * np.pi * np.arange(self.x_dim, dtype=np.float64) / float(self.x_dim)

        self._sigma_base = np.linspace(self.sigma_x1, self.sigma_x2, self.x_dim, dtype=np.float64)
        self._sigma_activity_alpha = np.linspace(self.cov_theta_amp1, self.cov_theta_amp2, self.x_dim, dtype=np.float64)

        # Kept for backward compatibility with summary/prints as baseline (diagonal) covariance.
        self.cov = np.diag(self._sigma_base**2) + 1e-8 * np.eye(self.x_dim, dtype=np.float64)
        self.cov_chol = np.linalg.cholesky(self.cov)

    def sample_theta(self, n: int) -> np.ndarray:
        theta = self.rng.uniform(self.theta_low, self.theta_high, size=(n, 1))
        return theta.astype(np.float64)

    def tuning_curve(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_col(theta)
        ph = self._mu_phases.reshape(1, -1)
        if self.tuning_curve_family == "cosine":
            return self._mu_amp * np.cos(self._mu_omega * t + ph)
        z = self.vm_omega * t - ph
        return self.vm_mu_amp * np.exp(self.vm_kappa * np.cos(z))

    def tuning_curve_derivative(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_col(theta)
        ph = self._mu_phases.reshape(1, -1)
        if self.tuning_curve_family == "cosine":
            return -self._mu_amp * self._mu_omega * np.sin(self._mu_omega * t + ph)
        z = self.vm_omega * t - ph
        return (
            -self.vm_mu_amp
            * np.exp(self.vm_kappa * np.cos(z))
            * self.vm_kappa
            * self.vm_omega
            * np.sin(z)
        )

    def _variance_diag_from_mu(self, mu: np.ndarray) -> np.ndarray:
        """Var_j = sigma_base_j^2 * (1 + alpha_j * |mu_j|) + eps (diagonal Gaussian noise)."""
        mu = np.asarray(mu, dtype=np.float64)
        sb = self._sigma_base.reshape(1, -1)
        alpha = self._sigma_activity_alpha.reshape(1, -1)
        return sb**2 * (1.0 + alpha * np.abs(mu)) + 1e-8

    def covariance_scales(self, theta: np.ndarray) -> np.ndarray:
        """Per-dimension standard deviations sqrt(Var_j)."""
        mu = self.tuning_curve(theta)
        v = self._variance_diag_from_mu(mu)
        return np.sqrt(np.maximum(v, 1e-12))

    def covariance_scales_derivative(self, theta: np.ndarray) -> np.ndarray:
        """d(sigma_j)/dtheta where sigma_j = sqrt(Var_j)."""
        mu = self.tuning_curve(theta)
        dmu = self.tuning_curve_derivative(theta)
        sb = self._sigma_base.reshape(1, -1)
        alpha = self._sigma_activity_alpha.reshape(1, -1)
        v = self._variance_diag_from_mu(mu)
        sgn = np.sign(mu)
        dv = sb**2 * alpha * sgn * dmu
        return dv / (2.0 * np.sqrt(np.maximum(v, 1e-12)))

    def covariance(self, theta: np.ndarray) -> np.ndarray:
        mu = self.tuning_curve(theta)
        v = self._variance_diag_from_mu(mu)
        n = v.shape[0]
        cov = np.zeros((n, self.x_dim, self.x_dim), dtype=np.float64)
        for j in range(self.x_dim):
            cov[:, j, j] = v[:, j]
        return cov

    def covariance_derivative(self, theta: np.ndarray) -> np.ndarray:
        mu = self.tuning_curve(theta)
        dmu = self.tuning_curve_derivative(theta)
        sb = self._sigma_base.reshape(1, -1)
        alpha = self._sigma_activity_alpha.reshape(1, -1)
        sgn = np.sign(mu)
        dv = sb**2 * alpha * sgn * dmu
        n = dv.shape[0]
        dcov = np.zeros((n, self.x_dim, self.x_dim), dtype=np.float64)
        for j in range(self.x_dim):
            dcov[:, j, j] = dv[:, j]
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
    tuning_curve_family: str = "cosine"  # "cosine" | "von_mises_raw"
    vm_mu_amp: float = 1.0
    vm_kappa: float = 1.0
    vm_omega: float = 1.0
    sigma_x1: float = 0.30
    sigma_x2: float = 0.30
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
        if self.tuning_curve_family not in ("cosine", "von_mises_raw"):
            raise ValueError('tuning_curve_family must be "cosine" or "von_mises_raw".')
        if self.tuning_curve_family == "von_mises_raw":
            if self.vm_kappa < 0.0:
                raise ValueError("vm_kappa must be non-negative for von_mises_raw.")
            if self.vm_mu_amp <= 0.0:
                raise ValueError("vm_mu_amp must be positive for von_mises_raw.")

        self.rng = np.random.default_rng(self.seed)
        idx = np.arange(1, self.x_dim + 1, dtype=np.float64)

        self._mu_amp = 1.0
        self._mu_omega = 1.0
        self._mu_phases = 2.0 * np.pi * np.arange(self.x_dim, dtype=np.float64) / float(self.x_dim)

        self._sep_weight = 0.95 / (1.0 + 0.05 * (idx - 1.0))
        self._sep_freq = 0.72 + 0.09 * (idx - 1.0)
        self._sep_phase = 0.20 * (idx - 1.0)

        self._sigma1_base = np.linspace(self.sigma_x1, self.sigma_x2, self.x_dim, dtype=np.float64)
        self._sigma2_base = np.linspace(1.15 * self.sigma_x1, 0.85 * self.sigma_x2, self.x_dim, dtype=np.float64)
        # Per-dimension activity weights alpha_j for Var_j = sigma_base_j^2 * (1 + alpha_j * |mu_j|).
        self._cov1_amp = np.clip(self.cov1_amp * (0.70 + 0.25 * np.sin(0.50 * idx)), 0.0, 0.95)
        self._cov2_amp = np.clip(self.cov2_amp * (0.70 + 0.25 * np.cos(0.40 * idx + 0.15)), 0.0, 0.95)

    def sample_theta(self, n: int) -> np.ndarray:
        theta = self.rng.uniform(self.theta_low, self.theta_high, size=(n, 1))
        return theta.astype(np.float64)

    def tuning_curve(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_col(theta)
        ph = self._mu_phases.reshape(1, -1)
        if self.tuning_curve_family == "cosine":
            return self._mu_amp * np.cos(self._mu_omega * t + ph)
        z = self.vm_omega * t - ph
        return self.vm_mu_amp * np.exp(self.vm_kappa * np.cos(z))

    def tuning_curve_derivative(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_col(theta)
        ph = self._mu_phases.reshape(1, -1)
        if self.tuning_curve_family == "cosine":
            return -self._mu_amp * self._mu_omega * np.sin(self._mu_omega * t + ph)
        z = self.vm_omega * t - ph
        return (
            -self.vm_mu_amp
            * np.exp(self.vm_kappa * np.cos(z))
            * self.vm_kappa
            * self.vm_omega
            * np.sin(z)
        )

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

    def _component_var_diag(
        self, mu: np.ndarray, sigma_base: np.ndarray, alpha: np.ndarray
    ) -> np.ndarray:
        """Var_j = sigma_base_j^2 * (1 + alpha_j * |mu_j|) + eps (diagonal Gaussian noise)."""
        sb = sigma_base.reshape(1, -1)
        a = alpha.reshape(1, -1)
        return sb**2 * (1.0 + a * np.abs(mu)) + 1e-8

    def component_covariances(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mu1, mu2 = self.component_means(theta)
        v1 = self._component_var_diag(mu1, self._sigma1_base, self._cov1_amp)
        v2 = self._component_var_diag(mu2, self._sigma2_base, self._cov2_amp)
        s1 = np.sqrt(np.maximum(v1, 1e-12))
        s2 = np.sqrt(np.maximum(v2, 1e-12))
        cov1, inv1 = self._diag_cov_and_inv(s1)
        cov2, inv2 = self._diag_cov_and_inv(s2)
        return cov1, cov2, inv1, inv2

    def _component_cov_derivatives(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu1, mu2 = self.component_means(theta)
        dmu_base = self.tuning_curve_derivative(theta)
        _, dsep = self._separation(theta)
        dmu1 = dmu_base + dsep
        dmu2 = dmu_base - dsep

        sb1 = self._sigma1_base.reshape(1, -1)
        a1 = self._cov1_amp.reshape(1, -1)
        dv1 = sb1**2 * a1 * np.sign(mu1) * dmu1

        sb2 = self._sigma2_base.reshape(1, -1)
        a2 = self._cov2_amp.reshape(1, -1)
        dv2 = sb2**2 * a2 * np.sign(mu2) * dmu2

        n, d = dv1.shape
        dcov1 = np.zeros((n, d, d), dtype=np.float64)
        dcov2 = np.zeros((n, d, d), dtype=np.float64)
        for j in range(d):
            dcov1[:, j, j] = dv1[:, j]
            dcov2[:, j, j] = dv2[:, j]
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
