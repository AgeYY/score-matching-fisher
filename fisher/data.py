from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from scipy.special import expit, logsumexp


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def _theta_col(theta: np.ndarray) -> np.ndarray:
    return np.asarray(theta, dtype=np.float64).reshape(-1, 1)


def _theta_2col(theta: np.ndarray) -> np.ndarray:
    arr = np.asarray(theta, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    if arr.ndim != 2 or int(arr.shape[1]) != 2:
        raise ValueError(f"theta must have shape (N, 2); got {arr.shape}.")
    return arr


def _tuning_centers_uniform_theta(theta_low: float, theta_high: float, x_dim: int) -> np.ndarray:
    """Per-dimension centers in theta, uniform on [theta_low, theta_high] (inclusive endpoints)."""
    return np.linspace(theta_low, theta_high, x_dim, dtype=np.float64)


# ``randamp_gaussian_sqrtd`` / ``cosine_gaussian_sqrtd_rand_tune*`` diagonal variance vs |mu|.
RANDAMP_SQRTD_VAR_MU_LAW_ADDITIVE = "additive_abs_mu"
RANDAMP_SQRTD_VAR_MU_LAW_LEGACY = "legacy_multiplicative_sqrtd"


@dataclass
class ToyConditionalGaussianDataset:
    theta_low: float = -6.0
    theta_high: float = 6.0
    x_dim: int = 2
    tuning_curve_family: str = "cosine"  # "cosine" | "von_mises_raw" | "gaussian_raw"
    vm_mu_amp: float = 1.0
    vm_kappa: float = 1.0
    vm_omega: float = 1.0
    gauss_mu_amp: float = 1.0
    gauss_kappa: float = 0.2
    gauss_omega: float = 1.0
    sigma_x1: float = 0.30
    sigma_x2: float = 0.30
    rho: float = 0.15
    # Activity coupling for diagonal variance: Var_j = sigma_base_j^2 * (1 + alpha * |mu_j|).
    # alpha is constant across dimensions: (cov_theta_amp1 + cov_theta_amp2) / 2 (both stored in .npz meta).
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
    # If False, ``log_p_x_given_theta`` uses full ``covariance(theta)`` via
    # ``fisher.evaluation.log_p_gaussian_mvnormal_from_cov`` (O(d³) per batch row).
    # Set to False on subclasses that implement non-diagonal ``covariance``.
    diagonal_gaussian_observation_noise: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if not (self.theta_low < self.theta_high):
            raise ValueError("theta_low must be smaller than theta_high.")
        if self.x_dim < 1:
            raise ValueError("x_dim must be >= 1.")
        if not (-0.99 < self.rho < 0.99):
            raise ValueError("rho must be in (-0.99, 0.99).")
        # Upper bound is permissive: diagonal / sqrt-d observation models only use these as nonnegative
        # activity weights (e.g. additive law V ∝ d*sigma^2 + alpha*|mu|). Legacy cap 0.95 was overly tight.
        _max_cov_theta_amp = 1.0e3
        if not (
            0.0 <= self.cov_theta_amp1 < _max_cov_theta_amp and 0.0 <= self.cov_theta_amp2 < _max_cov_theta_amp
        ):
            raise ValueError(f"cov_theta_amp1 and cov_theta_amp2 must be in [0, {_max_cov_theta_amp}).")
        if not (0.0 <= self.cov_theta_amp_rho <= 1.0):
            raise ValueError("cov_theta_amp_rho must be in [0, 1].")
        if not (0.1 <= self.rho_clip <= 0.95):
            raise ValueError("rho_clip must be in [0.1, 0.95].")
        if self.tuning_curve_family not in ("cosine", "von_mises_raw", "gaussian_raw"):
            raise ValueError('tuning_curve_family must be "cosine", "von_mises_raw", or "gaussian_raw".')
        if self.tuning_curve_family == "von_mises_raw":
            if self.vm_kappa < 0.0:
                raise ValueError("vm_kappa must be non-negative for von_mises_raw.")
            if self.vm_mu_amp <= 0.0:
                raise ValueError("vm_mu_amp must be positive for von_mises_raw.")
        elif self.tuning_curve_family == "gaussian_raw":
            if self.gauss_kappa < 0.0:
                raise ValueError("gauss_kappa must be non-negative for gaussian_raw.")
            if self.gauss_mu_amp <= 0.0:
                raise ValueError("gauss_mu_amp must be positive for gaussian_raw.")

        self.rng = np.random.default_rng(self.seed)

        # Cosine tuning curves: mu_j(theta) = A * cos(omega * theta + phi_j)
        # phi_j = 2*pi*j/d (periodic phases; cosine only).
        self._mu_amp = 1.0
        self._mu_omega = 1.0
        self._mu_phases = 2.0 * np.pi * np.arange(self.x_dim, dtype=np.float64) / float(self.x_dim)
        # Von Mises / Gaussian (raw): peak centers theta_j uniform on [theta_low, theta_high];
        # z_j = omega * (theta - theta_j). Same centers for both families.
        self._tuning_centers_theta = _tuning_centers_uniform_theta(self.theta_low, self.theta_high, self.x_dim)

        self._sigma_base = np.linspace(self.sigma_x1, self.sigma_x2, self.x_dim, dtype=np.float64)
        _alpha = 0.5 * (float(self.cov_theta_amp1) + float(self.cov_theta_amp2))
        self._sigma_activity_alpha = np.full(self.x_dim, _alpha, dtype=np.float64)

        # Kept for backward compatibility with summary/prints as baseline (diagonal) covariance.
        self.cov = np.diag(self._sigma_base**2) + 1e-8 * np.eye(self.x_dim, dtype=np.float64)
        self.cov_chol = np.linalg.cholesky(self.cov)

    def sample_theta(self, n: int) -> np.ndarray:
        theta = self.rng.uniform(self.theta_low, self.theta_high, size=(n, 1))
        return theta.astype(np.float64)

    def tuning_curve(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_col(theta)
        ph = self._mu_phases.reshape(1, -1)
        tc = self._tuning_centers_theta.reshape(1, -1)
        if self.tuning_curve_family == "cosine":
            return self._mu_amp * np.cos(self._mu_omega * t + ph)
        if self.tuning_curve_family == "von_mises_raw":
            z = self.vm_omega * (t - tc)
            return self.vm_mu_amp * np.exp(self.vm_kappa * np.cos(z))
        if self.tuning_curve_family == "gaussian_raw":
            z = self.gauss_omega * (t - tc)
            return self.gauss_mu_amp * np.exp(-self.gauss_kappa * (z**2))
        raise ValueError(f"Unknown tuning_curve_family: {self.tuning_curve_family!r}")

    def tuning_curve_derivative(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_col(theta)
        ph = self._mu_phases.reshape(1, -1)
        tc = self._tuning_centers_theta.reshape(1, -1)
        if self.tuning_curve_family == "cosine":
            return -self._mu_amp * self._mu_omega * np.sin(self._mu_omega * t + ph)
        if self.tuning_curve_family == "von_mises_raw":
            z = self.vm_omega * (t - tc)
            return (
                -self.vm_mu_amp
                * np.exp(self.vm_kappa * np.cos(z))
                * self.vm_kappa
                * self.vm_omega
                * np.sin(z)
            )
        if self.tuning_curve_family == "gaussian_raw":
            z = self.gauss_omega * (t - tc)
            g = self.gauss_mu_amp * np.exp(-self.gauss_kappa * (z**2))
            return -2.0 * self.gauss_kappa * self.gauss_omega * z * g
        raise ValueError(f"Unknown tuning_curve_family: {self.tuning_curve_family!r}")

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

    def log_p_x_given_theta(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Gaussian log-density: diagonal fast path, or full covariance if configured.

        When :attr:`diagonal_gaussian_observation_noise` is True (default), uses per-dimension
        variances from :meth:`_variance_diag_from_mu` (matches diagonal :meth:`covariance`).

        For a future subclass with a full ``covariance(theta)``, set
        ``diagonal_gaussian_observation_noise = False`` on the class (or override this method).
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.x_dim)
        theta = _theta_col(theta)
        mu = self.tuning_curve(theta)
        if not bool(type(self).diagonal_gaussian_observation_noise):
            from fisher.evaluation import log_p_gaussian_mvnormal_from_cov

            cov = self.covariance(theta)
            return log_p_gaussian_mvnormal_from_cov(x, mu, cov)
        v = self._variance_diag_from_mu(mu)
        delta = x - mu
        quad = np.sum((delta**2) / v, axis=1)
        logdet = np.sum(np.log(v), axis=1)
        d = float(self.x_dim)
        return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)


@dataclass
class ToyConditionalGaussianSqrtdDataset(ToyConditionalGaussianDataset):
    """Gaussian observation noise with per-coordinate std scaled by ``sqrt(x_dim)``.

    Same generative structure as :class:`ToyConditionalGaussianDataset` (tuning curve,
    activity-coupled diagonal variance), but each per-coordinate variance is multiplied
    by ``x_dim`` so that noise std scales like ``sqrt(d)`` relative to the base family.
    This avoids pathological high-SNR / near-diagonal distance structure when ``d`` is large.

    Defaults for `sigma_x1` / `sigma_x2` are ``0.5`` (aligned with the ``cosine_gaussian`` recipe
    scale); CLI ``make_dataset.py`` applies the fixed family recipe when sampling.
    """

    sigma_x1: float = 0.5
    sigma_x2: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        d = float(self.x_dim)
        self.cov = np.diag(self._sigma_base**2 * d) + 1e-8 * np.eye(self.x_dim, dtype=np.float64)
        self.cov_chol = np.linalg.cholesky(self.cov)

    def _variance_diag_from_mu(self, mu: np.ndarray) -> np.ndarray:
        v = super()._variance_diag_from_mu(mu)
        return v * float(self.x_dim)


@dataclass
class ToyConditionalGaussianCosineRandampSqrtdDataset(ToyConditionalGaussianSqrtdDataset):
    """``cosine_gaussian_sqrtd``-style observation noise with per-dimension random cosine amplitudes.

    Mean: ``mu_j(theta) = a_j * cos(omega * theta + phi_j)`` where each ``a_j`` is drawn once at
    init, ``a_j ~ Uniform(cosine_tune_amp_low, cosine_tune_amp_high)``, unless
    ``cosine_tune_amp_per_dim`` is supplied (e.g. from saved NPZ meta). When amplitudes are drawn
    (not supplied), the vector is multiplied by ``cosine_tune_amp_scale`` (default ``1.0``) after
    the Uniform draw; supplied ``cosine_tune_amp_per_dim`` is already final and is not scaled again.

    Diagonal variance vs ``|mu|`` (``cosine_sqrtd_obs_var_mu_law``), same two laws as
    :class:`ToyConditionalGaussianRandampSqrtdDataset`:

    - **Legacy** (default): ``V_j = d * sigma_base_j**2 * (1 + alpha_j * |mu_j|) + d*eps`` (same as
      :class:`ToyConditionalGaussianSqrtdDataset` on this mean).
    - **Additive**: ``V_j = d * sigma_base_j**2 + alpha_j * |mu_j| + eps``.
    """

    cosine_tune_amp_low: float = 0.5
    cosine_tune_amp_high: float = 1.5
    cosine_tune_amp_per_dim: np.ndarray | None = field(default=None)
    cosine_tune_amp_scale: float = 1.0
    cosine_sqrtd_obs_var_mu_law: str = RANDAMP_SQRTD_VAR_MU_LAW_LEGACY

    def __post_init__(self) -> None:
        law = str(self.cosine_sqrtd_obs_var_mu_law)
        if law not in (RANDAMP_SQRTD_VAR_MU_LAW_ADDITIVE, RANDAMP_SQRTD_VAR_MU_LAW_LEGACY):
            raise ValueError(
                "cosine_sqrtd_obs_var_mu_law must be "
                f"{RANDAMP_SQRTD_VAR_MU_LAW_ADDITIVE!r} or {RANDAMP_SQRTD_VAR_MU_LAW_LEGACY!r} "
                f"(got {law!r})."
            )
        self.cosine_sqrtd_obs_var_mu_law = law
        if not (self.cosine_tune_amp_low < self.cosine_tune_amp_high):
            raise ValueError("cosine_tune_amp_low must be < cosine_tune_amp_high.")
        super().__post_init__()
        if self.cosine_tune_amp_per_dim is not None:
            self._cosine_tune_amp = np.asarray(self.cosine_tune_amp_per_dim, dtype=np.float64).reshape(-1)
            if self._cosine_tune_amp.shape[0] != self.x_dim:
                raise ValueError(
                    f"cosine_tune_amp_per_dim length {self._cosine_tune_amp.shape[0]} != x_dim {self.x_dim}."
                )
        else:
            self._cosine_tune_amp = self.rng.uniform(
                self.cosine_tune_amp_low, self.cosine_tune_amp_high, size=(self.x_dim,)
            ).astype(np.float64)
            s = float(self.cosine_tune_amp_scale)
            if not math.isfinite(s) or s <= 0.0:
                raise ValueError("cosine_tune_amp_scale must be a finite positive number.")
            self._cosine_tune_amp *= s

    def tuning_curve(self, theta: np.ndarray) -> np.ndarray:
        if self.tuning_curve_family != "cosine":
            return super().tuning_curve(theta)
        t = _theta_col(theta)
        ph = self._mu_phases.reshape(1, -1)
        return self._cosine_tune_amp.reshape(1, -1) * np.cos(self._mu_omega * t + ph)

    def tuning_curve_derivative(self, theta: np.ndarray) -> np.ndarray:
        if self.tuning_curve_family != "cosine":
            return super().tuning_curve_derivative(theta)
        t = _theta_col(theta)
        ph = self._mu_phases.reshape(1, -1)
        return self._cosine_tune_amp.reshape(1, -1) * (-self._mu_omega * np.sin(self._mu_omega * t + ph))

    def _variance_diag_from_mu(self, mu: np.ndarray) -> np.ndarray:
        mu = np.asarray(mu, dtype=np.float64)
        d = float(self.x_dim)
        sb = self._sigma_base.reshape(1, -1)
        alpha = self._sigma_activity_alpha.reshape(1, -1)
        if self.cosine_sqrtd_obs_var_mu_law == RANDAMP_SQRTD_VAR_MU_LAW_ADDITIVE:
            return d * (sb**2) + alpha * np.abs(mu) + 1e-8
        return super()._variance_diag_from_mu(mu)

    def covariance_scales_derivative(self, theta: np.ndarray) -> np.ndarray:
        mu = self.tuning_curve(theta)
        dmu = self.tuning_curve_derivative(theta)
        v = self._variance_diag_from_mu(mu)
        sgn = np.sign(mu)
        sb = self._sigma_base.reshape(1, -1)
        alpha = self._sigma_activity_alpha.reshape(1, -1)
        d = float(self.x_dim)
        if self.cosine_sqrtd_obs_var_mu_law == RANDAMP_SQRTD_VAR_MU_LAW_ADDITIVE:
            dv = alpha * sgn * dmu
        else:
            dv = d * (sb**2) * alpha * sgn * dmu
        return dv / (2.0 * np.sqrt(np.maximum(v, 1e-12)))

    def covariance_derivative(self, theta: np.ndarray) -> np.ndarray:
        mu = self.tuning_curve(theta)
        dmu = self.tuning_curve_derivative(theta)
        sb = self._sigma_base.reshape(1, -1)
        alpha = self._sigma_activity_alpha.reshape(1, -1)
        sgn = np.sign(mu)
        v = self._variance_diag_from_mu(mu)
        d = float(self.x_dim)
        if self.cosine_sqrtd_obs_var_mu_law == RANDAMP_SQRTD_VAR_MU_LAW_ADDITIVE:
            dv = alpha * sgn * dmu
        else:
            dv = d * (sb**2) * alpha * sgn * dmu
        n = dv.shape[0]
        dcov = np.zeros((n, self.x_dim, self.x_dim), dtype=np.float64)
        for j in range(self.x_dim):
            dcov[:, j, j] = dv[:, j]
        return dcov


@dataclass
class ToyConditionalGaussianRandampDataset(ToyConditionalGaussianDataset):
    """Gaussian bump tuning with per-dimension random amplitudes (fixed across samples).

    For each coordinate ``j``, ``a_j ~ Uniform(randamp_mu_low, randamp_mu_high)`` is sampled once
    at initialization (unless ``randamp_mu_amp_per_dim`` is supplied, e.g. from saved NPZ meta).

    Mean: ``mu_j(theta) = a_j * exp(-randamp_kappa * (randamp_omega * (theta - theta_j))^2)``.

    Centers ``theta_j`` are uniform on ``[theta_low, theta_high]`` (same as ``gaussian_raw``).
    Observation noise matches the parent Gaussian diagonal covariance (not ``cosine_gaussian_sqrtd``).
    """

    tuning_curve_family: str = "cosine"  # unused; tuning is overridden below
    randamp_mu_low: float = 0.2
    randamp_mu_high: float = 2.0
    randamp_kappa: float = 0.2
    randamp_omega: float = 1.0
    randamp_mu_amp_per_dim: np.ndarray | None = field(default=None)

    def __post_init__(self) -> None:
        if not (self.randamp_mu_low < self.randamp_mu_high):
            raise ValueError("randamp_mu_low must be < randamp_mu_high.")
        if self.randamp_kappa < 0.0:
            raise ValueError("randamp_kappa must be non-negative.")
        super().__post_init__()
        if self.randamp_mu_amp_per_dim is not None:
            self._randamp_amp = np.asarray(self.randamp_mu_amp_per_dim, dtype=np.float64).reshape(-1)
            if self._randamp_amp.shape[0] != self.x_dim:
                raise ValueError(
                    f"randamp_mu_amp_per_dim length {self._randamp_amp.shape[0]} != x_dim {self.x_dim}."
                )
        else:
            self._randamp_amp = self.rng.uniform(
                self.randamp_mu_low, self.randamp_mu_high, size=(self.x_dim,)
            ).astype(np.float64)

    def tuning_curve(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_col(theta)
        tc = self._tuning_centers_theta.reshape(1, -1)
        z = self.randamp_omega * (t - tc)
        return self._randamp_amp.reshape(1, -1) * np.exp(-self.randamp_kappa * (z**2))

    def tuning_curve_derivative(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_col(theta)
        tc = self._tuning_centers_theta.reshape(1, -1)
        z = self.randamp_omega * (t - tc)
        g = self._randamp_amp.reshape(1, -1) * np.exp(-self.randamp_kappa * (z**2))
        return -2.0 * self.randamp_kappa * self.randamp_omega * z * g


@dataclass
class ToyConditionalGaussianRandampSqrtdDataset(ToyConditionalGaussianRandampDataset):
    """Same tuning as :class:`ToyConditionalGaussianRandampDataset`, but observation variance uses
    sqrt-``x_dim`` scaling (noise std grows like ``sqrt(d)`` in the baseline term).

    Two diagonal variance laws (``randamp_sqrtd_obs_var_mu_law``):

    - **Additive** (default for new datasets): ``V_j = d * sigma_base_j**2 + alpha_j * abs(mu_j) + eps``.
    - **Legacy** (archives without meta key): ``V_j = d * sigma_base_j**2 * (1 + alpha_j * abs(mu_j)) + eps``.

    ``self.cov`` / ``cov_chol`` remain a theta-agnostic snapshot for compatibility; sampling uses
    :meth:`covariance`.

    Baseline ``sigma_x1``/``sigma_x2`` on the dataclass default to ``0.2``; the
    ``randamp_gaussian_sqrtd`` recipe in ``fisher.dataset_family_recipes`` may override them
    (e.g. ``0.2/sqrt(2)`` with stronger ``cov_theta`` amps).
    """

    sigma_x1: float = 0.2
    sigma_x2: float = 0.2
    randamp_sqrtd_obs_var_mu_law: str = RANDAMP_SQRTD_VAR_MU_LAW_ADDITIVE

    def __post_init__(self) -> None:
        law = str(self.randamp_sqrtd_obs_var_mu_law)
        if law not in (RANDAMP_SQRTD_VAR_MU_LAW_ADDITIVE, RANDAMP_SQRTD_VAR_MU_LAW_LEGACY):
            raise ValueError(
                "randamp_sqrtd_obs_var_mu_law must be "
                f"{RANDAMP_SQRTD_VAR_MU_LAW_ADDITIVE!r} or {RANDAMP_SQRTD_VAR_MU_LAW_LEGACY!r} "
                f"(got {law!r})."
            )
        self.randamp_sqrtd_obs_var_mu_law = law
        super().__post_init__()
        d = float(self.x_dim)
        self.cov = np.diag(self._sigma_base**2 * d) + 1e-8 * np.eye(self.x_dim, dtype=np.float64)
        self.cov_chol = np.linalg.cholesky(self.cov)

    def _variance_diag_from_mu(self, mu: np.ndarray) -> np.ndarray:
        mu = np.asarray(mu, dtype=np.float64)
        d = float(self.x_dim)
        sb = self._sigma_base.reshape(1, -1)
        alpha = self._sigma_activity_alpha.reshape(1, -1)
        if self.randamp_sqrtd_obs_var_mu_law == RANDAMP_SQRTD_VAR_MU_LAW_ADDITIVE:
            return d * (sb**2) + alpha * np.abs(mu) + 1e-8
        v = super()._variance_diag_from_mu(mu)
        return v * d

    def covariance_scales_derivative(self, theta: np.ndarray) -> np.ndarray:
        mu = self.tuning_curve(theta)
        dmu = self.tuning_curve_derivative(theta)
        v = self._variance_diag_from_mu(mu)
        sgn = np.sign(mu)
        sb = self._sigma_base.reshape(1, -1)
        alpha = self._sigma_activity_alpha.reshape(1, -1)
        d = float(self.x_dim)
        if self.randamp_sqrtd_obs_var_mu_law == RANDAMP_SQRTD_VAR_MU_LAW_ADDITIVE:
            dv = alpha * sgn * dmu
        else:
            dv = d * (sb**2) * alpha * sgn * dmu
        return dv / (2.0 * np.sqrt(np.maximum(v, 1e-12)))

    def covariance_derivative(self, theta: np.ndarray) -> np.ndarray:
        mu = self.tuning_curve(theta)
        dmu = self.tuning_curve_derivative(theta)
        sb = self._sigma_base.reshape(1, -1)
        alpha = self._sigma_activity_alpha.reshape(1, -1)
        sgn = np.sign(mu)
        v = self._variance_diag_from_mu(mu)
        d = float(self.x_dim)
        if self.randamp_sqrtd_obs_var_mu_law == RANDAMP_SQRTD_VAR_MU_LAW_ADDITIVE:
            dv = alpha * sgn * dmu
        else:
            dv = d * (sb**2) * alpha * sgn * dmu
        n = dv.shape[0]
        dcov = np.zeros((n, self.x_dim, self.x_dim), dtype=np.float64)
        for j in range(self.x_dim):
            dcov[:, j, j] = dv[:, j]
        return dcov


@dataclass
class ToyConditionalGaussianRandamp2DSqrtdDataset(ToyConditionalGaussianRandampSqrtdDataset):
    """Random-amplitude 2D Gaussian bump means with sqrt-d additive diagonal variance."""

    theta_dim: int = 2
    randamp_center_per_dim: np.ndarray | None = field(default=None)

    def __post_init__(self) -> None:
        if int(self.theta_dim) != 2:
            raise ValueError("ToyConditionalGaussianRandamp2DSqrtdDataset requires theta_dim == 2.")
        super().__post_init__()
        if self.randamp_center_per_dim is not None:
            centers = np.asarray(self.randamp_center_per_dim, dtype=np.float64).reshape(self.x_dim, 2)
        else:
            centers = self.rng.uniform(self.theta_low, self.theta_high, size=(self.x_dim, 2)).astype(np.float64)
        self._randamp_centers_2d = centers

    def sample_theta(self, n: int) -> np.ndarray:
        theta = self.rng.uniform(self.theta_low, self.theta_high, size=(n, 2))
        return theta.astype(np.float64)

    def tuning_curve(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_2col(theta)
        diff = t[:, None, :] - self._randamp_centers_2d[None, :, :]
        r2 = np.sum(diff**2, axis=2)
        return self._randamp_amp.reshape(1, -1) * np.exp(-self.randamp_kappa * r2)

    def tuning_curve_derivative(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_2col(theta)
        diff = t[:, None, :] - self._randamp_centers_2d[None, :, :]
        mu = self.tuning_curve(t)
        return -2.0 * self.randamp_kappa * diff * mu[:, :, None]

    def covariance_scales_derivative(self, theta: np.ndarray) -> np.ndarray:
        mu = self.tuning_curve(theta)
        dmu = self.tuning_curve_derivative(theta)
        v = self._variance_diag_from_mu(mu)
        sgn = np.sign(mu)[:, :, None]
        alpha = self._sigma_activity_alpha.reshape(1, -1, 1)
        if self.randamp_sqrtd_obs_var_mu_law == RANDAMP_SQRTD_VAR_MU_LAW_ADDITIVE:
            dv = alpha * sgn * dmu
        else:
            sb = self._sigma_base.reshape(1, -1, 1)
            dv = float(self.x_dim) * (sb**2) * alpha * sgn * dmu
        return dv / (2.0 * np.sqrt(np.maximum(v, 1e-12))[:, :, None])

    def covariance_derivative(self, theta: np.ndarray) -> np.ndarray:
        mu = self.tuning_curve(theta)
        dmu = self.tuning_curve_derivative(theta)
        sgn = np.sign(mu)[:, :, None]
        alpha = self._sigma_activity_alpha.reshape(1, -1, 1)
        if self.randamp_sqrtd_obs_var_mu_law == RANDAMP_SQRTD_VAR_MU_LAW_ADDITIVE:
            dv = alpha * sgn * dmu
        else:
            sb = self._sigma_base.reshape(1, -1, 1)
            dv = float(self.x_dim) * (sb**2) * alpha * sgn * dmu
        n = dv.shape[0]
        dcov = np.zeros((n, self.x_dim, self.x_dim, 2), dtype=np.float64)
        for j in range(self.x_dim):
            dcov[:, j, j, :] = dv[:, j, :]
        return dcov

    def log_p_x_given_theta(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Use `(N, 2)` θ internally; do not apply scalar-only ``_theta_col`` reshaping."""
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.x_dim)
        th = _theta_2col(theta)
        mu = self.tuning_curve(th)
        if not bool(type(self).diagonal_gaussian_observation_noise):
            from fisher.evaluation import log_p_gaussian_mvnormal_from_cov

            cov = self.covariance(th)
            return log_p_gaussian_mvnormal_from_cov(x, mu, cov)
        v = self._variance_diag_from_mu(mu)
        delta = x - mu
        quad = np.sum((delta**2) / v, axis=1)
        logdet = np.sum(np.log(v), axis=1)
        d = float(self.x_dim)
        return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)


@dataclass
class ToyConditionalGaussianGridcos2DSqrtdDataset(ToyConditionalGaussianCosineRandampSqrtdDataset):
    """Three-orientation 2D random grid-cosine means with sqrt-d additive diagonal variance."""

    theta_dim: int = 2
    gridcos_orientation_per_dim: np.ndarray | None = field(default=None)
    gridcos_phase_per_dim: np.ndarray | None = field(default=None)
    gridcos_omega_per_dim: np.ndarray | None = field(default=None)

    def __post_init__(self) -> None:
        if int(self.theta_dim) != 2:
            raise ValueError("ToyConditionalGaussianGridcos2DSqrtdDataset requires theta_dim == 2.")
        super().__post_init__()
        if self.gridcos_orientation_per_dim is not None:
            rho = np.asarray(self.gridcos_orientation_per_dim, dtype=np.float64).reshape(self.x_dim)
        else:
            rho = self.rng.uniform(0.0, np.pi / 3.0, size=(self.x_dim,)).astype(np.float64)
        if self.gridcos_phase_per_dim is not None:
            phi = np.asarray(self.gridcos_phase_per_dim, dtype=np.float64).reshape(self.x_dim, 3)
        else:
            phi = self.rng.uniform(0.0, 2.0 * np.pi, size=(self.x_dim, 3)).astype(np.float64)
        if self.gridcos_omega_per_dim is not None:
            omega = np.asarray(self.gridcos_omega_per_dim, dtype=np.float64).reshape(self.x_dim)
        else:
            omega = np.ones(self.x_dim, dtype=np.float64)
        angles = rho[:, None] + (np.pi / 3.0) * np.arange(3, dtype=np.float64).reshape(1, 3)
        self._gridcos_orientation = rho
        self._gridcos_phase = phi
        self._gridcos_omega = omega
        self._gridcos_k = omega[:, None, None] * np.stack([np.cos(angles), np.sin(angles)], axis=2)

    def sample_theta(self, n: int) -> np.ndarray:
        theta = self.rng.uniform(self.theta_low, self.theta_high, size=(n, 2))
        return theta.astype(np.float64)

    def tuning_curve(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_2col(theta)
        arg = np.einsum("nt,jmt->njm", t, self._gridcos_k) + self._gridcos_phase[None, :, :]
        return self._cosine_tune_amp.reshape(1, -1) * np.mean(np.cos(arg), axis=2)

    def tuning_curve_derivative(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_2col(theta)
        arg = np.einsum("nt,jmt->njm", t, self._gridcos_k) + self._gridcos_phase[None, :, :]
        grad = -np.einsum("njm,jmt->njt", np.sin(arg), self._gridcos_k) / 3.0
        return self._cosine_tune_amp.reshape(1, -1, 1) * grad

    def covariance_scales_derivative(self, theta: np.ndarray) -> np.ndarray:
        mu = self.tuning_curve(theta)
        dmu = self.tuning_curve_derivative(theta)
        v = self._variance_diag_from_mu(mu)
        sgn = np.sign(mu)[:, :, None]
        alpha = self._sigma_activity_alpha.reshape(1, -1, 1)
        if self.cosine_sqrtd_obs_var_mu_law == RANDAMP_SQRTD_VAR_MU_LAW_ADDITIVE:
            dv = alpha * sgn * dmu
        else:
            sb = self._sigma_base.reshape(1, -1, 1)
            dv = float(self.x_dim) * (sb**2) * alpha * sgn * dmu
        return dv / (2.0 * np.sqrt(np.maximum(v, 1e-12))[:, :, None])

    def covariance_derivative(self, theta: np.ndarray) -> np.ndarray:
        mu = self.tuning_curve(theta)
        dmu = self.tuning_curve_derivative(theta)
        sgn = np.sign(mu)[:, :, None]
        alpha = self._sigma_activity_alpha.reshape(1, -1, 1)
        if self.cosine_sqrtd_obs_var_mu_law == RANDAMP_SQRTD_VAR_MU_LAW_ADDITIVE:
            dv = alpha * sgn * dmu
        else:
            sb = self._sigma_base.reshape(1, -1, 1)
            dv = float(self.x_dim) * (sb**2) * alpha * sgn * dmu
        n = dv.shape[0]
        dcov = np.zeros((n, self.x_dim, self.x_dim, 2), dtype=np.float64)
        for j in range(self.x_dim):
            dcov[:, j, j, :] = dv[:, j, :]
        return dcov

    def log_p_x_given_theta(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Use `(N, 2)` θ internally; do not apply scalar-only ``_theta_col`` reshaping."""
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.x_dim)
        th = _theta_2col(theta)
        mu = self.tuning_curve(th)
        if not bool(type(self).diagonal_gaussian_observation_noise):
            from fisher.evaluation import log_p_gaussian_mvnormal_from_cov

            cov = self.covariance(th)
            return log_p_gaussian_mvnormal_from_cov(x, mu, cov)
        v = self._variance_diag_from_mu(mu)
        delta = x - mu
        quad = np.sum((delta**2) / v, axis=1)
        logdet = np.sum(np.log(v), axis=1)
        d = float(self.x_dim)
        return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)


@dataclass
class ToyCosSinPiecewiseNoiseDataset:
    theta_low: float = -6.0
    theta_high: float = 6.0
    x_dim: int = 2
    sigma_piecewise_low: float = 0.1
    sigma_piecewise_high: float = 0.1
    theta_zero_to_low: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if not (self.theta_low < self.theta_high):
            raise ValueError("theta_low must be smaller than theta_high.")
        if self.x_dim != 2:
            raise ValueError("ToyCosSinPiecewiseNoiseDataset requires x_dim == 2.")
        if self.sigma_piecewise_low <= 0.0 or self.sigma_piecewise_high <= 0.0:
            raise ValueError("sigma_piecewise_low and sigma_piecewise_high must be positive.")
        self.rng = np.random.default_rng(self.seed)

    def sample_theta(self, n: int) -> np.ndarray:
        theta = self.rng.uniform(self.theta_low, self.theta_high, size=(n, 1))
        return theta.astype(np.float64)

    def tuning_curve(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_col(theta)
        return np.concatenate([np.cos(t), np.sin(t)], axis=1).astype(np.float64)

    def tuning_curve_derivative(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_col(theta)
        return np.concatenate([-np.sin(t), np.cos(t)], axis=1).astype(np.float64)

    def _sigma_from_theta(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_col(theta).reshape(-1)
        if self.theta_zero_to_low:
            low_mask = t <= 0.0
        else:
            low_mask = t < 0.0
        sigma = np.where(low_mask, self.sigma_piecewise_low, self.sigma_piecewise_high)
        return sigma.astype(np.float64)

    def covariance_scales(self, theta: np.ndarray) -> np.ndarray:
        sigma = self._sigma_from_theta(theta).reshape(-1, 1)
        return np.repeat(sigma, repeats=2, axis=1).astype(np.float64)

    def covariance_scales_derivative(self, theta: np.ndarray) -> np.ndarray:
        # Piecewise-constant sigma(theta): derivative is zero away from the discontinuity at theta=0.
        n = _theta_col(theta).shape[0]
        return np.zeros((n, 2), dtype=np.float64)

    def covariance(self, theta: np.ndarray) -> np.ndarray:
        sigma = self._sigma_from_theta(theta)
        var = sigma**2
        n = var.shape[0]
        cov = np.zeros((n, 2, 2), dtype=np.float64)
        cov[:, 0, 0] = var
        cov[:, 1, 1] = var
        return cov

    def covariance_derivative(self, theta: np.ndarray) -> np.ndarray:
        # Piecewise-constant variance(theta): derivative is zero away from the discontinuity at theta=0.
        n = _theta_col(theta).shape[0]
        return np.zeros((n, 2, 2), dtype=np.float64)

    def sample_x(self, theta: np.ndarray) -> np.ndarray:
        mu = self.tuning_curve(theta)
        sigma = self._sigma_from_theta(theta).reshape(-1, 1)
        eps = self.rng.standard_normal(size=mu.shape)
        x = mu + sigma * eps
        return x.astype(np.float64)

    def sample_joint(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        theta = self.sample_theta(n)
        x = self.sample_x(theta)
        return theta, x


@dataclass
class ToyLinearPiecewiseNoiseDataset:
    """2D observations x = (k*theta, theta) + isotropic noise with std vs theta."""

    theta_low: float = -6.0
    theta_high: float = 6.0
    x_dim: int = 2
    linear_k: float = 1.0
    sigma_piecewise_low: float = 0.1
    sigma_piecewise_high: float = 0.1
    # "linear": sigma linear in theta from low at theta_low to high at theta_high (see theta_zero_to_low).
    # "sigmoid": smooth transition centered at linear_sigma_sigmoid_center.
    linear_sigma_schedule: str = "linear"
    linear_sigma_sigmoid_center: float = 0.0
    linear_sigma_sigmoid_steepness: float = 2.0
    theta_zero_to_low: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if not (self.theta_low < self.theta_high):
            raise ValueError("theta_low must be smaller than theta_high.")
        if self.x_dim != 2:
            raise ValueError("ToyLinearPiecewiseNoiseDataset requires x_dim == 2.")
        if self.sigma_piecewise_low <= 0.0 or self.sigma_piecewise_high <= 0.0:
            raise ValueError("sigma_piecewise_low and sigma_piecewise_high must be positive.")
        _sched = str(self.linear_sigma_schedule).lower()
        if _sched not in ("linear", "sigmoid"):
            raise ValueError('linear_sigma_schedule must be "linear" or "sigmoid".')
        self.linear_sigma_schedule = _sched
        if _sched == "sigmoid" and self.linear_sigma_sigmoid_steepness <= 0.0:
            raise ValueError("linear_sigma_sigmoid_steepness must be positive when linear_sigma_schedule is sigmoid.")
        self.rng = np.random.default_rng(self.seed)

    def sample_theta(self, n: int) -> np.ndarray:
        theta = self.rng.uniform(self.theta_low, self.theta_high, size=(n, 1))
        return theta.astype(np.float64)

    def tuning_curve(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_col(theta)
        return np.concatenate([self.linear_k * t, t], axis=1).astype(np.float64)

    def tuning_curve_derivative(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_col(theta)
        d0 = np.full_like(t, self.linear_k, dtype=np.float64)
        d1 = np.ones_like(t, dtype=np.float64)
        return np.concatenate([d0, d1], axis=1).astype(np.float64)

    def _noise_weight_and_derivative(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Weight w in [0,1] and dw/dtheta for sigma = low + (high-low)*w."""
        if self.linear_sigma_schedule == "linear":
            return self._noise_weight_linear(theta)
        t = _theta_col(theta).reshape(-1)
        z = self.linear_sigma_sigmoid_steepness * (t - self.linear_sigma_sigmoid_center)
        w = expit(z)
        dw = self.linear_sigma_sigmoid_steepness * w * (1.0 - w)
        if not self.theta_zero_to_low:
            w = 1.0 - w
            dw = -dw
        return w.astype(np.float64), dw.astype(np.float64)

    def _noise_weight_linear(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Linear w from theta_low (0) to theta_high (1); optional flip via theta_zero_to_low."""
        t = _theta_col(theta).reshape(-1)
        span = float(self.theta_high - self.theta_low)
        w = (t - float(self.theta_low)) / span
        w = np.clip(w, 0.0, 1.0)
        if not self.theta_zero_to_low:
            w = 1.0 - w
        # dw/dtheta: 1/span on (theta_low, theta_high), 0 when clipped outside.
        inside = (t > float(self.theta_low)) & (t < float(self.theta_high))
        dw = np.where(inside, 1.0 / span, 0.0)
        if not self.theta_zero_to_low:
            dw = -dw
        return w.astype(np.float64), dw.astype(np.float64)

    def _sigma_from_theta(self, theta: np.ndarray) -> np.ndarray:
        w, _ = self._noise_weight_and_derivative(theta)
        sigma = self.sigma_piecewise_low + (self.sigma_piecewise_high - self.sigma_piecewise_low) * w
        return sigma.astype(np.float64)

    def covariance_scales(self, theta: np.ndarray) -> np.ndarray:
        sigma = self._sigma_from_theta(theta).reshape(-1, 1)
        return np.repeat(sigma, repeats=2, axis=1).astype(np.float64)

    def covariance_scales_derivative(self, theta: np.ndarray) -> np.ndarray:
        _, dw = self._noise_weight_and_derivative(theta)
        dsigma = (self.sigma_piecewise_high - self.sigma_piecewise_low) * dw
        dsigma = dsigma.reshape(-1, 1)
        return np.repeat(dsigma, repeats=2, axis=1).astype(np.float64)

    def covariance(self, theta: np.ndarray) -> np.ndarray:
        sigma = self._sigma_from_theta(theta)
        var = sigma**2
        n = var.shape[0]
        cov = np.zeros((n, 2, 2), dtype=np.float64)
        cov[:, 0, 0] = var
        cov[:, 1, 1] = var
        return cov

    def covariance_derivative(self, theta: np.ndarray) -> np.ndarray:
        sigma = self._sigma_from_theta(theta)
        _, dw = self._noise_weight_and_derivative(theta)
        dsigma = (self.sigma_piecewise_high - self.sigma_piecewise_low) * dw
        dvar = 2.0 * sigma * dsigma
        n = dvar.shape[0]
        dcov = np.zeros((n, 2, 2), dtype=np.float64)
        dcov[:, 0, 0] = dvar
        dcov[:, 1, 1] = dvar
        return dcov

    def sample_x(self, theta: np.ndarray) -> np.ndarray:
        mu = self.tuning_curve(theta)
        sigma = self._sigma_from_theta(theta).reshape(-1, 1)
        eps = self.rng.standard_normal(size=mu.shape)
        x = mu + sigma * eps
        return x.astype(np.float64)

    def sample_joint(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        theta = self.sample_theta(n)
        x = self.sample_x(theta)
        return theta, x


@dataclass
class ToyConditionalGMMNonGaussianDataset:
    theta_low: float = -6.0
    theta_high: float = 6.0
    x_dim: int = 2
    tuning_curve_family: str = "cosine"  # "cosine" | "von_mises_raw" | "gaussian_raw"
    vm_mu_amp: float = 1.0
    vm_kappa: float = 1.0
    vm_omega: float = 1.0
    gauss_mu_amp: float = 1.0
    gauss_kappa: float = 0.2
    gauss_omega: float = 1.0
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
        if self.tuning_curve_family not in ("cosine", "von_mises_raw", "gaussian_raw"):
            raise ValueError('tuning_curve_family must be "cosine", "von_mises_raw", or "gaussian_raw".')
        if self.tuning_curve_family == "von_mises_raw":
            if self.vm_kappa < 0.0:
                raise ValueError("vm_kappa must be non-negative for von_mises_raw.")
            if self.vm_mu_amp <= 0.0:
                raise ValueError("vm_mu_amp must be positive for von_mises_raw.")
        elif self.tuning_curve_family == "gaussian_raw":
            if self.gauss_kappa < 0.0:
                raise ValueError("gauss_kappa must be non-negative for gaussian_raw.")
            if self.gauss_mu_amp <= 0.0:
                raise ValueError("gauss_mu_amp must be positive for gaussian_raw.")

        self.rng = np.random.default_rng(self.seed)
        idx = np.arange(1, self.x_dim + 1, dtype=np.float64)

        self._mu_amp = 1.0
        self._mu_omega = 1.0
        self._mu_phases = 2.0 * np.pi * np.arange(self.x_dim, dtype=np.float64) / float(self.x_dim)
        self._tuning_centers_theta = _tuning_centers_uniform_theta(self.theta_low, self.theta_high, self.x_dim)

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
        tc = self._tuning_centers_theta.reshape(1, -1)
        if self.tuning_curve_family == "cosine":
            return self._mu_amp * np.cos(self._mu_omega * t + ph)
        if self.tuning_curve_family == "von_mises_raw":
            z = self.vm_omega * (t - tc)
            return self.vm_mu_amp * np.exp(self.vm_kappa * np.cos(z))
        if self.tuning_curve_family == "gaussian_raw":
            z = self.gauss_omega * (t - tc)
            return self.gauss_mu_amp * np.exp(-self.gauss_kappa * (z**2))
        raise ValueError(f"Unknown tuning_curve_family: {self.tuning_curve_family!r}")

    def tuning_curve_derivative(self, theta: np.ndarray) -> np.ndarray:
        t = _theta_col(theta)
        ph = self._mu_phases.reshape(1, -1)
        tc = self._tuning_centers_theta.reshape(1, -1)
        if self.tuning_curve_family == "cosine":
            return -self._mu_amp * self._mu_omega * np.sin(self._mu_omega * t + ph)
        if self.tuning_curve_family == "von_mises_raw":
            z = self.vm_omega * (t - tc)
            return (
                -self.vm_mu_amp
                * np.exp(self.vm_kappa * np.cos(z))
                * self.vm_kappa
                * self.vm_omega
                * np.sin(z)
            )
        if self.tuning_curve_family == "gaussian_raw":
            z = self.gauss_omega * (t - tc)
            g = self.gauss_mu_amp * np.exp(-self.gauss_kappa * (z**2))
            return -2.0 * self.gauss_kappa * self.gauss_omega * z * g
        raise ValueError(f"Unknown tuning_curve_family: {self.tuning_curve_family!r}")

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
    dataset: ToyConditionalGaussianDataset
    | ToyConditionalGaussianSqrtdDataset
    | ToyConditionalGaussianCosineRandampSqrtdDataset
    | ToyConditionalGaussianRandampDataset
    | ToyConditionalGaussianRandampSqrtdDataset
    | ToyCosSinPiecewiseNoiseDataset
    | ToyLinearPiecewiseNoiseDataset
    | ToyConditionalGMMNonGaussianDataset,
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
