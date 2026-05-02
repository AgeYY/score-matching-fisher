"""Linear x-space flow matching with analytic Gaussian likelihood.

This module trains a time-independent velocity

    v(x, theta) = A x + b_phi(theta)

on a straight noise-to-data bridge.  After training, the induced endpoint
distribution in the normalized x coordinates is

    N(mu(theta), Sigma),  Sigma = exp(A) exp(A)^T,
    A mu(theta) = (exp(A) - I) b_phi(theta).

The variant :class:`ConditionalThetaDiagonalLinearXFlowMLP` uses
``v(x,theta)=diag(a_phi(theta)) x + b_phi(theta)``; the endpoint is a
diagonal Gaussian with
``Sigma_ii=exp(2 a_i)`` and ``mu_i=((e^{a_i}-1)/a_i) b_i`` (elementwise).

:class:`ConditionalDiagonalLinearXFlowFiLMLP` keeps the same global diagonal drift ``A=diag(a)`` as
:class:`ConditionalDiagonalLinearXFlowMLP`, but parameterizes ``b_phi(theta)`` with a trunk whose hidden
layers use FiLM conditioning from raw ``theta`` (no Fourier features).
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fisher.gaussian_x_flow import GaussianAffinePathSchedule
from fisher.model_weight_ema import (
    clone_model_weight_ema,
    evaluate_with_weight_ema,
    init_model_weight_ema,
    load_model_weights_from_ema_state,
    scalar_val_ema_update,
    update_model_weight_ema,
)


def _phi_expm1_div_a(a: torch.Tensor) -> torch.Tensor:
    """Stable elementwise (exp(a)-1)/a for the diagonal linear-flow endpoint mean."""
    eps = 1e-6
    mask = a.abs() < eps
    taylor = 1.0 + a / 2.0 + (a * a) / 6.0 + (a * a * a) / 24.0
    safe_a = torch.where(mask, torch.ones_like(a), a)
    ref = torch.expm1(a) / safe_a
    return torch.where(mask, taylor, ref)


def _as_2d_float64(a: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D.")
    return arr


def gaussian_hellinger_sq_full(
    mu1: np.ndarray,
    cov1: np.ndarray,
    mu2: np.ndarray,
    cov2: np.ndarray,
    *,
    jitter: float = 1e-9,
) -> float:
    """Squared Hellinger distance between two full-covariance Gaussians."""
    m1 = np.asarray(mu1, dtype=np.float64).reshape(-1)
    m2 = np.asarray(mu2, dtype=np.float64).reshape(-1)
    s1 = np.asarray(cov1, dtype=np.float64)
    s2 = np.asarray(cov2, dtype=np.float64)
    if m1.shape != m2.shape:
        raise ValueError("mu1 and mu2 must have the same shape.")
    d = int(m1.size)
    if s1.shape != (d, d) or s2.shape != (d, d):
        raise ValueError("covariance matrices must have shape [D, D].")
    eye = np.eye(d, dtype=np.float64)
    s1j = 0.5 * (s1 + s1.T) + float(jitter) * eye
    s2j = 0.5 * (s2 + s2.T) + float(jitter) * eye
    sbar = 0.5 * (s1j + s2j)
    sign1, logdet1 = np.linalg.slogdet(s1j)
    sign2, logdet2 = np.linalg.slogdet(s2j)
    signb, logdetb = np.linalg.slogdet(sbar)
    if sign1 <= 0 or sign2 <= 0 or signb <= 0:
        raise ValueError("Gaussian covariance must be positive definite.")
    diff = m1 - m2
    quad = float(diff @ np.linalg.solve(sbar, diff))
    log_bc = 0.25 * logdet1 + 0.25 * logdet2 - 0.5 * logdetb - 0.125 * quad
    h2 = 1.0 - float(np.exp(np.clip(log_bc, -745.0, 0.0)))
    return float(np.clip(h2, 0.0, 1.0))


def gaussian_hellinger_sq_diag(
    mu1: np.ndarray,
    var1: np.ndarray,
    mu2: np.ndarray,
    var2: np.ndarray,
    *,
    jitter: float = 1e-12,
) -> float:
    """Squared Hellinger distance between two diagonal Gaussians."""
    m1 = np.asarray(mu1, dtype=np.float64).reshape(-1)
    m2 = np.asarray(mu2, dtype=np.float64).reshape(-1)
    v1 = np.asarray(var1, dtype=np.float64).reshape(-1) + float(jitter)
    v2 = np.asarray(var2, dtype=np.float64).reshape(-1) + float(jitter)
    if m1.shape != m2.shape or v1.shape != m1.shape or v2.shape != m1.shape:
        raise ValueError("mu and variance vectors must have matching shape.")
    if np.any(v1 <= 0.0) or np.any(v2 <= 0.0):
        raise ValueError("diagonal variances must be positive.")
    vbar = 0.5 * (v1 + v2)
    log_bc = 0.25 * np.sum(np.log(v1)) + 0.25 * np.sum(np.log(v2)) - 0.5 * np.sum(np.log(vbar))
    log_bc -= 0.125 * float(np.sum((m1 - m2) ** 2 / vbar))
    h2 = 1.0 - float(np.exp(np.clip(log_bc, -745.0, 0.0)))
    return float(np.clip(h2, 0.0, 1.0))


def gaussian_hellinger_sq_shared_covariance_matrix(
    mu: np.ndarray,
    cov: np.ndarray,
    *,
    jitter: float = 1e-9,
) -> np.ndarray:
    """Pairwise squared Hellinger matrix for Gaussians with shared full covariance."""
    m = _as_2d_float64(mu, name="mu")
    s = np.asarray(cov, dtype=np.float64)
    d = int(m.shape[1])
    if s.shape != (d, d):
        raise ValueError("cov must have shape [D, D].")
    sj = 0.5 * (s + s.T) + float(jitter) * np.eye(d, dtype=np.float64)
    sol = np.linalg.solve(sj, (m[:, None, :] - m[None, :, :]).reshape(-1, d).T).T
    diff = (m[:, None, :] - m[None, :, :]).reshape(-1, d)
    quad = np.sum(diff * sol, axis=1).reshape(m.shape[0], m.shape[0])
    h2 = 1.0 - np.exp(np.clip(-0.125 * quad, -745.0, 0.0))
    h2 = np.clip(h2, 0.0, 1.0)
    np.fill_diagonal(h2, 0.0)
    return 0.5 * (h2 + h2.T)


def gaussian_hellinger_sq_diag_matrix(
    mu: np.ndarray,
    var_diag: np.ndarray,
    *,
    jitter: float = 1e-12,
) -> np.ndarray:
    """Pairwise squared Hellinger matrix for diagonal Gaussian endpoints."""
    m = _as_2d_float64(mu, name="mu")
    v = _as_2d_float64(var_diag, name="var_diag") + float(jitter)
    if v.shape[0] == 1 and m.shape[0] > 1:
        v = np.repeat(v, repeats=m.shape[0], axis=0)
    if v.shape != m.shape:
        raise ValueError("var_diag must have shape [N, D] or [1, D].")
    if np.any(v <= 0.0):
        raise ValueError("diagonal variances must be positive.")
    vbar = 0.5 * (v[:, None, :] + v[None, :, :])
    log_bc = 0.25 * np.sum(np.log(v[:, None, :]), axis=2)
    log_bc = log_bc + 0.25 * np.sum(np.log(v[None, :, :]), axis=2)
    log_bc = log_bc - 0.5 * np.sum(np.log(vbar), axis=2)
    log_bc -= 0.125 * np.sum((m[:, None, :] - m[None, :, :]) ** 2 / vbar, axis=2)
    h2 = 1.0 - np.exp(np.clip(log_bc, -745.0, 0.0))
    h2 = np.clip(h2, 0.0, 1.0)
    np.fill_diagonal(h2, 0.0)
    return 0.5 * (h2 + h2.T)


def linear_x_flow_endpoint_gaussian(
    *,
    model: nn.Module,
    theta_all: np.ndarray,
    device: torch.device,
    solve_jitter: float = 1e-6,
    quadrature_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return endpoint ``(mu, covariance_or_diag, is_diagonal)`` in normalized x-space."""
    theta = _as_2d_float64(theta_all, name="theta_all")
    theta_t = torch.from_numpy(theta.astype(np.float32, copy=False)).to(device)
    model.eval()
    with torch.no_grad():
        if isinstance(model, ConditionalTimeDiagonalLinearXFlowMLP):
            mu_t, var_t = model.endpoint_mean_covariance_diag(
                theta_t,
                solve_jitter=float(solve_jitter),
                quadrature_steps=quadrature_steps,
            )
            return (
                mu_t.detach().cpu().numpy().astype(np.float64),
                var_t.detach().cpu().numpy().astype(np.float64),
                True,
            )
        if hasattr(model, "endpoint_mean_covariance_diag"):
            mu_t, var_t = model.endpoint_mean_covariance_diag(theta_t, solve_jitter=float(solve_jitter))  # type: ignore[attr-defined]
            return (
                mu_t.detach().cpu().numpy().astype(np.float64),
                var_t.detach().cpu().numpy().astype(np.float64),
                True,
            )
        if hasattr(model, "endpoint_mean_covariance"):
            mu_t, cov_t = model.endpoint_mean_covariance(theta_t, solve_jitter=float(solve_jitter))  # type: ignore[attr-defined]
            return (
                mu_t.detach().cpu().numpy().astype(np.float64),
                cov_t.detach().cpu().numpy().astype(np.float64),
                False,
            )
    raise TypeError(f"{type(model).__name__} does not expose a Gaussian endpoint.")


def compute_linear_x_flow_analytic_hellinger_matrix(
    *,
    model: nn.Module,
    theta_all: np.ndarray,
    device: torch.device,
    solve_jitter: float = 1e-6,
    quadrature_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Compute pairwise squared Hellinger distances from learned Gaussian endpoints."""
    mu, cov_or_diag, is_diag = linear_x_flow_endpoint_gaussian(
        model=model,
        theta_all=theta_all,
        device=device,
        solve_jitter=float(solve_jitter),
        quadrature_steps=quadrature_steps,
    )
    if is_diag:
        h = gaussian_hellinger_sq_diag_matrix(mu, cov_or_diag, jitter=float(solve_jitter) * 1e-3)
    else:
        h = gaussian_hellinger_sq_shared_covariance_matrix(mu, cov_or_diag, jitter=float(solve_jitter) * 1e-3)
    return h, mu, cov_or_diag, bool(is_diag)


def _fill_empty_bin_rows_nearest(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    c = np.asarray(counts, dtype=np.int64).reshape(-1)
    nonempty_idx = np.flatnonzero(c > 0)
    if nonempty_idx.size == 0:
        raise ValueError("At least one non-empty theta bin is required.")
    for b in np.flatnonzero(c <= 0):
        nearest = int(nonempty_idx[np.argmin(np.abs(nonempty_idx - int(b)))])
        out[int(b)] = out[nearest]
    return out


class _BaseConditionalLinearXFlowMLP(nn.Module):
    """Shared theta-conditioned offset MLP plus an ``A`` property supplied by subclasses."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if int(theta_dim) < 1:
            raise ValueError("theta_dim must be >= 1.")
        if int(x_dim) < 1:
            raise ValueError("x_dim must be >= 1.")
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be >= 1.")
        if int(depth) < 1:
            raise ValueError("depth must be >= 1.")
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.B = nn.Parameter(1e-3 * torch.eye(self.x_dim))

        layers: list[nn.Module] = []
        in_dim = self.theta_dim
        for _ in range(int(depth)):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.SiLU())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, self.x_dim))
        self.b_net = nn.Sequential(*layers)

    def b(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        return self.b_net(theta)

    @property
    def A(self) -> torch.Tensor:
        raise NotImplementedError

    def regularization_loss(self) -> torch.Tensor | None:
        return None

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        a = self.A
        return x @ a.transpose(0, 1) + self.b(theta)

    def endpoint_mean_covariance(
        self,
        theta: torch.Tensor,
        *,
        solve_jitter: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        b = self.b(theta)
        e_a = torch.linalg.matrix_exp(self.A)
        sigma = e_a @ e_a.transpose(0, 1)
        rhs = b @ (e_a - torch.eye(self.x_dim, dtype=e_a.dtype, device=e_a.device)).transpose(0, 1)
        a = self.A
        try:
            mu = torch.linalg.solve(a, rhs.transpose(0, 1)).transpose(0, 1)
        except RuntimeError:
            eye = torch.eye(self.x_dim, dtype=a.dtype, device=a.device)
            mu = torch.linalg.solve(a + float(solve_jitter) * eye, rhs.transpose(0, 1)).transpose(0, 1)
        if not torch.all(torch.isfinite(mu)):
            eye = torch.eye(self.x_dim, dtype=a.dtype, device=a.device)
            mu = torch.linalg.solve(a + float(solve_jitter) * eye, rhs.transpose(0, 1)).transpose(0, 1)
        return mu, sigma

    def log_prob_normalized(
        self,
        x_norm: torch.Tensor,
        theta: torch.Tensor,
        *,
        solve_jitter: float = 1e-6,
    ) -> torch.Tensor:
        if x_norm.ndim == 1:
            x_norm = x_norm.unsqueeze(0)
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        if x_norm.shape[0] != theta.shape[0]:
            raise ValueError("x and theta batch sizes must match.")
        mu, sigma = self.endpoint_mean_covariance(theta, solve_jitter=solve_jitter)
        d = int(x_norm.shape[1])
        eye = torch.eye(d, dtype=x_norm.dtype, device=x_norm.device)
        l = torch.linalg.cholesky(sigma + float(solve_jitter) * eye)
        diff = x_norm - mu
        z = torch.cholesky_solve(diff.unsqueeze(-1), l).squeeze(-1)
        quad = torch.sum(diff * z, dim=1)
        log_det = 2.0 * torch.sum(torch.log(torch.clamp(torch.diagonal(l), min=1e-12)))
        return -0.5 * (quad + log_det + float(d) * math.log(2.0 * math.pi))

    def log_prob_observed(
        self,
        x_raw: torch.Tensor,
        theta: torch.Tensor,
        *,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        solve_jitter: float = 1e-6,
    ) -> torch.Tensor:
        z = (x_raw - x_mean) / x_std
        logjac = -torch.sum(torch.log(x_std))
        return self.log_prob_normalized(z, theta, solve_jitter=solve_jitter) + logjac


class ConditionalLinearXFlowMLP(_BaseConditionalLinearXFlowMLP):
    """Full symmetric drift ``A=(B+B.T)/2`` plus theta-conditioned offset MLP."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
    ) -> None:
        super().__init__(theta_dim=theta_dim, x_dim=x_dim, hidden_dim=hidden_dim, depth=depth)
        self.B = nn.Parameter(1e-3 * torch.eye(self.x_dim))

    @property
    def A(self) -> torch.Tensor:
        return 0.5 * (self.B + self.B.transpose(0, 1))


class ConditionalScalarLinearXFlowMLP(_BaseConditionalLinearXFlowMLP):
    """Scalar symmetric drift ``A=a I`` plus theta-conditioned offset MLP."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
    ) -> None:
        super().__init__(theta_dim=theta_dim, x_dim=x_dim, hidden_dim=hidden_dim, depth=depth)
        self.a = nn.Parameter(torch.tensor(1e-3, dtype=torch.float32))

    @property
    def A(self) -> torch.Tensor:
        eye = torch.eye(self.x_dim, dtype=self.a.dtype, device=self.a.device)
        return self.a * eye


class ConditionalDiagonalLinearXFlowMLP(_BaseConditionalLinearXFlowMLP):
    """Diagonal symmetric drift ``A=diag(a)`` plus theta-conditioned offset MLP."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
    ) -> None:
        super().__init__(theta_dim=theta_dim, x_dim=x_dim, hidden_dim=hidden_dim, depth=depth)
        self.a = nn.Parameter(torch.full((self.x_dim,), 1e-3, dtype=torch.float32))

    @property
    def A(self) -> torch.Tensor:
        return torch.diag(self.a)


class ConditionalDiagonalLinearXFlowFiLMLP(nn.Module):
    """Diagonal symmetric drift ``A=diag(a)`` plus theta-conditioned FiLM offset for ``b_phi(theta)``.

    Uses ``depth`` hidden stages (aligned with :class:`ConditionalDiagonalLinearXFlowMLP` ``b_net`` depth):
    the first linear maps ``theta -> hidden_dim``; each stage applies ``Linear``, FiLM
    ``h <- (1+gamma(theta)) * h + beta(theta)``, then SiLU. Output linear maps to ``x_dim``.

    FiLM heads are initialized to zero so initially ``(1+gamma) * h + beta = h``.
    """

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if int(theta_dim) < 1:
            raise ValueError("theta_dim must be >= 1.")
        if int(x_dim) < 1:
            raise ValueError("x_dim must be >= 1.")
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be >= 1.")
        if int(depth) < 1:
            raise ValueError("depth must be >= 1.")
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)

        self.layer0 = nn.Linear(self.theta_dim, self.hidden_dim)
        self.mid_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.depth - 1)]
        )
        self.film_layers = nn.ModuleList(
            [nn.Linear(self.theta_dim, 2 * self.hidden_dim) for _ in range(self.depth)]
        )
        self.out = nn.Linear(self.hidden_dim, self.x_dim)
        self.a = nn.Parameter(torch.full((self.x_dim,), 1e-3, dtype=torch.float32))

        for lin in self.film_layers:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        nn.init.xavier_uniform_(self.out.weight, gain=0.01)
        nn.init.zeros_(self.out.bias)

    @property
    def A(self) -> torch.Tensor:
        return torch.diag(self.a)

    def b(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        h = self.layer0(theta)
        gb = self.film_layers[0](theta)
        gamma, beta = gb.chunk(2, dim=-1)
        h = (1.0 + gamma) * h + beta
        h = torch.nn.functional.silu(h)
        for i, lin in enumerate(self.mid_layers):
            h = lin(h)
            gb = self.film_layers[i + 1](theta)
            gamma, beta = gb.chunk(2, dim=-1)
            h = (1.0 + gamma) * h + beta
            h = torch.nn.functional.silu(h)
        return self.out(h)

    def regularization_loss(self) -> torch.Tensor | None:
        return None

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        a_mat = self.A
        return x @ a_mat.transpose(0, 1) + self.b(theta)

    def endpoint_mean_covariance(
        self,
        theta: torch.Tensor,
        *,
        solve_jitter: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        b_theta = self.b(theta)
        e_a = torch.linalg.matrix_exp(self.A)
        sigma = e_a @ e_a.transpose(0, 1)
        rhs = b_theta @ (e_a - torch.eye(self.x_dim, dtype=e_a.dtype, device=e_a.device)).transpose(0, 1)
        a_mat = self.A
        try:
            mu = torch.linalg.solve(a_mat, rhs.transpose(0, 1)).transpose(0, 1)
        except RuntimeError:
            eye = torch.eye(self.x_dim, dtype=a_mat.dtype, device=a_mat.device)
            mu = torch.linalg.solve(a_mat + float(solve_jitter) * eye, rhs.transpose(0, 1)).transpose(0, 1)
        if not torch.all(torch.isfinite(mu)):
            eye = torch.eye(self.x_dim, dtype=a_mat.dtype, device=a_mat.device)
            mu = torch.linalg.solve(a_mat + float(solve_jitter) * eye, rhs.transpose(0, 1)).transpose(0, 1)
        return mu, sigma

    def log_prob_normalized(
        self,
        x_norm: torch.Tensor,
        theta: torch.Tensor,
        *,
        solve_jitter: float = 1e-6,
    ) -> torch.Tensor:
        if x_norm.ndim == 1:
            x_norm = x_norm.unsqueeze(0)
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        if x_norm.shape[0] != theta.shape[0]:
            raise ValueError("x and theta batch sizes must match.")
        mu, sigma = self.endpoint_mean_covariance(theta, solve_jitter=solve_jitter)
        d = int(x_norm.shape[1])
        eye = torch.eye(d, dtype=x_norm.dtype, device=x_norm.device)
        l = torch.linalg.cholesky(sigma + float(solve_jitter) * eye)
        diff = x_norm - mu
        z = torch.cholesky_solve(diff.unsqueeze(-1), l).squeeze(-1)
        quad = torch.sum(diff * z, dim=1)
        log_det = 2.0 * torch.sum(torch.log(torch.clamp(torch.diagonal(l), min=1e-12)))
        return -0.5 * (quad + log_det + float(d) * math.log(2.0 * math.pi))

    def log_prob_observed(
        self,
        x_raw: torch.Tensor,
        theta: torch.Tensor,
        *,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        solve_jitter: float = 1e-6,
    ) -> torch.Tensor:
        z = (x_raw - x_mean) / x_std
        logjac = -torch.sum(torch.log(x_std))
        return self.log_prob_normalized(z, theta, solve_jitter=solve_jitter) + logjac


class ConditionalTimeDiagonalLinearXFlowMLP(nn.Module):
    """Time-dependent diagonal drift ``A(t)=diag(a(t))`` and offset ``b(t, theta)``.

    The endpoint distribution is diagonal Gaussian. Since ``a(t)`` has no theta input,
    the endpoint covariance is shared across theta; the mean is obtained by fixed-grid
    quadrature of the linear ODE solution.
    """

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        quadrature_steps: int = 64,
    ) -> None:
        super().__init__()
        if int(theta_dim) < 1:
            raise ValueError("theta_dim must be >= 1.")
        if int(x_dim) < 1:
            raise ValueError("x_dim must be >= 1.")
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be >= 1.")
        if int(depth) < 1:
            raise ValueError("depth must be >= 1.")
        if int(quadrature_steps) < 2:
            raise ValueError("quadrature_steps must be >= 2.")
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.quadrature_steps = int(quadrature_steps)

        a_layers: list[nn.Module] = []
        in_dim = 1
        for _ in range(int(depth)):
            a_layers.append(nn.Linear(in_dim, int(hidden_dim)))
            a_layers.append(nn.SiLU())
            in_dim = int(hidden_dim)
        a_out = nn.Linear(in_dim, self.x_dim)
        nn.init.zeros_(a_out.weight)
        nn.init.constant_(a_out.bias, 1e-3)
        a_layers.append(a_out)
        self.a_net = nn.Sequential(*a_layers)

        b_layers: list[nn.Module] = []
        in_dim = self.theta_dim + 1
        for _ in range(int(depth)):
            b_layers.append(nn.Linear(in_dim, int(hidden_dim)))
            b_layers.append(nn.SiLU())
            in_dim = int(hidden_dim)
        b_out = nn.Linear(in_dim, self.x_dim)
        nn.init.xavier_uniform_(b_out.weight, gain=0.01)
        nn.init.zeros_(b_out.bias)
        b_layers.append(b_out)
        self.b_net = nn.Sequential(*b_layers)

    def _as_col_t(self, t: torch.Tensor, *, batch: int | None = None) -> torch.Tensor:
        if t.ndim == 0:
            t = t.reshape(1, 1)
        elif t.ndim == 1:
            t = t.unsqueeze(-1)
        if t.ndim != 2 or int(t.shape[1]) != 1:
            raise ValueError("t must have shape [B] or [B, 1].")
        if batch is not None and int(t.shape[0]) == 1 and int(batch) > 1:
            t = t.expand(int(batch), 1)
        return t

    def a(self, t: torch.Tensor) -> torch.Tensor:
        t = self._as_col_t(t)
        return self.a_net(t)

    def b(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        t = self._as_col_t(t, batch=int(theta.shape[0]))
        return self.b_net(torch.cat([t, theta], dim=1))

    def forward(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = self._as_col_t(t, batch=int(x.shape[0]))
        return self.a(t) * x + self.b(theta, t)

    def regularization_loss(self) -> torch.Tensor | None:
        return None

    def endpoint_mean_covariance_diag(
        self,
        theta: torch.Tensor,
        *,
        solve_jitter: float = 1e-6,
        quadrature_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        q = self.quadrature_steps if quadrature_steps is None else int(quadrature_steps)
        if q < 2:
            raise ValueError("quadrature_steps must be >= 2.")
        grid = torch.linspace(0.0, 1.0, q, dtype=theta.dtype, device=theta.device).reshape(q, 1)
        a_grid = self.a(grid)
        s = torch.trapezoid(a_grid, grid.reshape(-1), dim=0)
        tail = torch.zeros_like(a_grid)
        if q > 1:
            dt = grid[1:, 0] - grid[:-1, 0]
            seg = 0.5 * (a_grid[:-1] + a_grid[1:]) * dt.unsqueeze(-1)
            tail[:-1] = torch.flip(torch.cumsum(torch.flip(seg, dims=(0,)), dim=0), dims=(0,))

        b_vals: list[torch.Tensor] = []
        for k in range(q):
            tk = grid[k].reshape(1, 1).expand(int(theta.shape[0]), 1)
            b_vals.append(self.b(theta, tk))
        b_grid = torch.stack(b_vals, dim=0)
        weights = torch.exp(tail).unsqueeze(1)
        integrand = weights * b_grid
        mu = torch.trapezoid(integrand, grid.reshape(-1), dim=0)
        sigma_diag = torch.exp(2.0 * s).reshape(1, -1).expand_as(mu) + float(solve_jitter)
        return mu, sigma_diag

    def log_prob_normalized(
        self,
        x_norm: torch.Tensor,
        theta: torch.Tensor,
        *,
        solve_jitter: float = 1e-6,
        quadrature_steps: int | None = None,
    ) -> torch.Tensor:
        if x_norm.ndim == 1:
            x_norm = x_norm.unsqueeze(0)
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        if x_norm.shape[0] != theta.shape[0]:
            raise ValueError("x and theta batch sizes must match.")
        mu, sigma_diag = self.endpoint_mean_covariance_diag(
            theta,
            solve_jitter=float(solve_jitter),
            quadrature_steps=quadrature_steps,
        )
        d = int(x_norm.shape[1])
        quad = torch.sum((x_norm - mu) ** 2 / sigma_diag, dim=1)
        log_det = torch.sum(torch.log(sigma_diag), dim=1)
        return -0.5 * (quad + log_det + float(d) * math.log(2.0 * math.pi))

    def log_prob_observed(
        self,
        x_raw: torch.Tensor,
        theta: torch.Tensor,
        *,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        solve_jitter: float = 1e-6,
        quadrature_steps: int | None = None,
    ) -> torch.Tensor:
        z = (x_raw - x_mean) / x_std
        logjac = -torch.sum(torch.log(x_std))
        return self.log_prob_normalized(
            z,
            theta,
            solve_jitter=float(solve_jitter),
            quadrature_steps=quadrature_steps,
        ) + logjac


class ConditionalThetaDiagonalLinearXFlowMLP(nn.Module):
    """Diagonal drift ``a_phi(theta)`` and offset ``b_phi(theta)`` both from theta.

    Velocity ``v(x,theta)=a_phi(theta) \\odot x + b_phi(theta)``. Endpoint in normalized
    coordinates is diagonal Gaussian with ``\\Sigma_{ii}=exp(2 a_i)`` and
    ``\\mu_i = ((e^{a_i}-1)/a_i) b_i``.
    """

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if int(theta_dim) < 1:
            raise ValueError("theta_dim must be >= 1.")
        if int(x_dim) < 1:
            raise ValueError("x_dim must be >= 1.")
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be >= 1.")
        if int(depth) < 1:
            raise ValueError("depth must be >= 1.")
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        trunk_layers: list[nn.Module] = []
        in_dim = self.theta_dim
        for _ in range(int(depth)):
            trunk_layers.append(nn.Linear(in_dim, int(hidden_dim)))
            trunk_layers.append(nn.SiLU())
            in_dim = int(hidden_dim)
        self.trunk = nn.Sequential(*trunk_layers)
        self.a_head = nn.Linear(in_dim, self.x_dim)
        self.b_head = nn.Linear(in_dim, self.x_dim)
        nn.init.zeros_(self.a_head.weight)
        nn.init.constant_(self.a_head.bias, 1e-3)
        nn.init.xavier_uniform_(self.b_head.weight, gain=0.01)
        nn.init.zeros_(self.b_head.bias)

    def a_b(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        h = self.trunk(theta)
        return self.a_head(h), self.b_head(h)

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        a_theta, b_theta = self.a_b(theta)
        return a_theta * x + b_theta

    def regularization_loss(self) -> torch.Tensor | None:
        return None

    def endpoint_mean_covariance_diag(
        self,
        theta: torch.Tensor,
        *,
        solve_jitter: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``mu`` [B, D] and diagonal variance ``sigma_diag`` [B, D] in normalized space."""
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        a_theta, b_theta = self.a_b(theta)
        phi = _phi_expm1_div_a(a_theta)
        mu = phi * b_theta
        sigma_diag = torch.exp(2.0 * a_theta) + float(solve_jitter)
        return mu, sigma_diag

    def log_prob_normalized(
        self,
        x_norm: torch.Tensor,
        theta: torch.Tensor,
        *,
        solve_jitter: float = 1e-6,
    ) -> torch.Tensor:
        if x_norm.ndim == 1:
            x_norm = x_norm.unsqueeze(0)
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        if x_norm.shape[0] != theta.shape[0]:
            raise ValueError("x and theta batch sizes must match.")
        mu, sigma_diag = self.endpoint_mean_covariance_diag(theta, solve_jitter=solve_jitter)
        d = int(x_norm.shape[1])
        quad = torch.sum((x_norm - mu) ** 2 / sigma_diag, dim=1)
        log_det = torch.sum(torch.log(sigma_diag), dim=1)
        return -0.5 * (quad + log_det + float(d) * math.log(2.0 * math.pi))

    def log_prob_observed(
        self,
        x_raw: torch.Tensor,
        theta: torch.Tensor,
        *,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        solve_jitter: float = 1e-6,
    ) -> torch.Tensor:
        z = (x_raw - x_mean) / x_std
        logjac = -torch.sum(torch.log(x_std))
        return self.log_prob_normalized(z, theta, solve_jitter=solve_jitter) + logjac


class ConditionalLowRankLinearXFlowMLP(_BaseConditionalLinearXFlowMLP):
    """Low-rank symmetric drift ``A=diag(a)+U diag(s) U.T`` plus offset MLP."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        rank: int,
        hidden_dim: int = 128,
        depth: int = 3,
    ) -> None:
        if int(rank) < 1:
            raise ValueError("rank must be >= 1.")
        if int(rank) > int(x_dim):
            raise ValueError("rank must be <= x_dim.")
        super().__init__(theta_dim=theta_dim, x_dim=x_dim, hidden_dim=hidden_dim, depth=depth)
        self.rank = int(rank)
        self.a = nn.Parameter(torch.full((self.x_dim,), 1e-3, dtype=torch.float32))
        self.U = nn.Parameter(1e-2 * torch.randn(self.x_dim, self.rank))
        self.s = nn.Parameter(torch.full((self.rank,), 1e-3, dtype=torch.float32))

    @property
    def A(self) -> torch.Tensor:
        return torch.diag(self.a) + (self.U * self.s.unsqueeze(0)) @ self.U.transpose(0, 1)


class ConditionalRandomBasisLowRankLinearXFlowMLP(_BaseConditionalLinearXFlowMLP):
    """Random-basis low-rank drift ``A=diag(a)+Q S Q.T`` plus offset MLP."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        rank: int,
        hidden_dim: int = 128,
        depth: int = 3,
        lambda_a: float = 1e-4,
        lambda_s: float = 1e-4,
    ) -> None:
        if int(rank) < 1:
            raise ValueError("rank must be >= 1.")
        if int(rank) > int(x_dim):
            raise ValueError("rank must be <= x_dim.")
        if float(lambda_a) < 0.0:
            raise ValueError("lambda_a must be >= 0.")
        if float(lambda_s) < 0.0:
            raise ValueError("lambda_s must be >= 0.")
        super().__init__(theta_dim=theta_dim, x_dim=x_dim, hidden_dim=hidden_dim, depth=depth)
        self.rank = int(rank)
        self.lambda_a = float(lambda_a)
        self.lambda_s = float(lambda_s)
        q, _ = torch.linalg.qr(torch.randn(self.x_dim, self.rank), mode="reduced")
        self.register_buffer("Q", q)
        self.a = nn.Parameter(torch.zeros(self.x_dim, dtype=torch.float32))
        self.S_raw = nn.Parameter(torch.zeros(self.rank, self.rank, dtype=torch.float32))

    @property
    def S(self) -> torch.Tensor:
        return 0.5 * (self.S_raw + self.S_raw.transpose(0, 1))

    @property
    def A(self) -> torch.Tensor:
        q = self.Q.to(dtype=self.a.dtype, device=self.a.device)
        return torch.diag(self.a) + q @ self.S @ q.transpose(0, 1)

    def regularization_loss(self) -> torch.Tensor:
        return float(self.lambda_a) * torch.sum(self.a**2) + float(self.lambda_s) * torch.sum(self.S**2)


def fit_residual_pca_basis_from_linear_mean(
    *,
    linear_model: _BaseConditionalLinearXFlowMLP,
    theta_train: np.ndarray,
    x_train_norm: np.ndarray,
    pca_dim: int,
    device: torch.device,
    solve_jitter: float = 1e-6,
) -> np.ndarray:
    """Fit a frozen PCA basis from residuals around the trained linear-flow mean."""
    th = _as_2d_float64(theta_train, name="theta_train")
    x = _as_2d_float64(x_train_norm, name="x_train_norm")
    if th.shape[0] != x.shape[0]:
        raise ValueError("theta_train and x_train_norm row counts must match.")
    n, d = int(x.shape[0]), int(x.shape[1])
    k = int(pca_dim)
    if k < 1:
        raise ValueError("pca_dim must be >= 1.")
    max_rank = min(d, max(1, n - 1))
    if k > max_rank:
        raise ValueError(f"pca_dim must be <= min(x_dim, n_train-1)={max_rank}; got {k}.")
    linear_model.eval()
    with torch.no_grad():
        th_t = torch.from_numpy(th.astype(np.float32)).to(device)
        mu_t, _ = linear_model.endpoint_mean_covariance(th_t, solve_jitter=float(solve_jitter))
        mu = mu_t.detach().cpu().numpy().astype(np.float64)
    residual = x - mu
    residual = residual - np.mean(residual, axis=0, keepdims=True, dtype=np.float64)
    _, _, vh = np.linalg.svd(residual, full_matrices=False)
    u = vh[:k].T.astype(np.float32, copy=False)
    return u


class ConditionalPCANonlinearLinearXFlowMLP(nn.Module):
    """Linear x-flow plus a frozen-PCA nonlinear correction in normalized x-space."""

    def __init__(
        self,
        *,
        linear_model: _BaseConditionalLinearXFlowMLP,
        pca_basis: np.ndarray | torch.Tensor,
        hidden_dim: int = 128,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be >= 1.")
        if int(depth) < 1:
            raise ValueError("depth must be >= 1.")
        self.linear_model = linear_model
        self.theta_dim = int(linear_model.theta_dim)
        self.x_dim = int(linear_model.x_dim)
        u = torch.as_tensor(pca_basis, dtype=torch.float32)
        if u.ndim != 2 or int(u.shape[0]) != self.x_dim:
            raise ValueError("pca_basis must have shape [x_dim, k].")
        self.pca_dim = int(u.shape[1])
        if self.pca_dim < 1:
            raise ValueError("pca_basis must have at least one component.")
        self.register_buffer("U", u)
        layers: list[nn.Module] = []
        in_dim = self.pca_dim + 1 + self.theta_dim
        for _ in range(int(depth)):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.SiLU())
            in_dim = int(hidden_dim)
        out = nn.Linear(in_dim, self.pca_dim)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.h_net = nn.Sequential(*layers)

    @property
    def A(self) -> torch.Tensor:
        return self.linear_model.A

    def linear_mean(self, theta: torch.Tensor, *, solve_jitter: float = 1e-6) -> torch.Tensor:
        mu, _ = self.linear_model.endpoint_mean_covariance(theta, solve_jitter=float(solve_jitter))
        return mu

    def h(self, z: torch.Tensor, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return self.h_net(torch.cat([z, t, theta], dim=1))

    def nonlinear_correction(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        theta: torch.Tensor,
        *,
        solve_jitter: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.linear_mean(theta, solve_jitter=float(solve_jitter))
        z = (x - t * mu) @ self.U
        h = self.h(z, t, theta)
        return h @ self.U.transpose(0, 1), z, h

    def forward(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        t: torch.Tensor,
        *,
        solve_jitter: float = 1e-6,
    ) -> torch.Tensor:
        delta, _, _ = self.nonlinear_correction(x, t, theta, solve_jitter=float(solve_jitter))
        return self.linear_model(x, theta) + delta

    def regularization_loss(self, h: torch.Tensor, lambda_h: float) -> torch.Tensor:
        return float(lambda_h) * torch.mean(h**2)

    def divergence(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        with torch.enable_grad():
            mu = self.linear_mean(theta)
            z = ((x - t * mu) @ self.U).detach().requires_grad_(True)
            h = self.h(z, t, theta)
            tr_h = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
            for j in range(self.pca_dim):
                grad_j = torch.autograd.grad(
                    h[:, j].sum(),
                    z,
                    create_graph=False,
                    retain_graph=j < self.pca_dim - 1,
                )[0]
                tr_h = tr_h + grad_j[:, j]
        tr_a = torch.trace(self.A).to(dtype=x.dtype, device=x.device).detach()
        return tr_a + tr_h.detach()

    def log_prob_normalized(
        self,
        x_norm: torch.Tensor,
        theta: torch.Tensor,
        *,
        solve_jitter: float = 1e-6,
        ode_steps: int = 32,
    ) -> torch.Tensor:
        if x_norm.ndim == 1:
            x_norm = x_norm.unsqueeze(0)
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        if x_norm.shape[0] != theta.shape[0]:
            raise ValueError("x and theta batch sizes must match.")
        steps = int(ode_steps)
        if steps < 1:
            raise ValueError("ode_steps must be >= 1.")
        x = x_norm
        div_int = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        dt = 1.0 / float(steps)
        for s in range(steps, 0, -1):
            t = torch.full((x.shape[0], 1), float(s) / float(steps), dtype=x.dtype, device=x.device)
            div = self.divergence(x, theta, t)
            div_int = div_int + dt * div
            with torch.no_grad():
                v = self.forward(x, theta, t, solve_jitter=float(solve_jitter))
                x = x - dt * v
        d = int(x.shape[1])
        base = -0.5 * (torch.sum(x**2, dim=1) + float(d) * math.log(2.0 * math.pi))
        return base - div_int

    def log_prob_observed(
        self,
        x_raw: torch.Tensor,
        theta: torch.Tensor,
        *,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        solve_jitter: float = 1e-6,
        ode_steps: int = 32,
    ) -> torch.Tensor:
        z = (x_raw - x_mean) / x_std
        logjac = -torch.sum(torch.log(x_std))
        return self.log_prob_normalized(
            z,
            theta,
            solve_jitter=float(solve_jitter),
            ode_steps=int(ode_steps),
        ) + logjac


def train_linear_x_flow(
    *,
    model: ConditionalLinearXFlowMLP,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.0,
    t_eps: float = 0.05,
    patience: int = 1000,
    min_delta: float = 1e-4,
    ema_alpha: float = 0.05,
    weight_ema_decay: float = 0.9,
    max_grad_norm: float = 10.0,
    log_every: int = 50,
    restore_best: bool = True,
) -> dict[str, Any]:
    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1.")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1.")
    if float(lr) <= 0.0:
        raise ValueError("lr must be > 0.")
    if int(patience) < 0:
        raise ValueError("patience must be >= 0.")
    if float(min_delta) < 0.0:
        raise ValueError("min_delta must be >= 0.")
    if not (0.0 < float(ema_alpha) <= 1.0):
        raise ValueError("ema_alpha must be in (0, 1].")
    if not np.isfinite(float(weight_ema_decay)) or float(weight_ema_decay) >= 1.0:
        raise ValueError("weight_ema_decay must be finite and < 1.")
    te = float(t_eps)
    if not (0.0 < te < 0.5):
        raise ValueError("t_eps must be in (0, 0.5) so bridge times lie in (t_eps, 1-t_eps).")

    th_tr = _as_2d_float64(theta_train, name="theta_train")
    x_tr = _as_2d_float64(x_train, name="x_train")
    th_va = _as_2d_float64(theta_val, name="theta_val")
    x_va = _as_2d_float64(x_val, name="x_val")
    if th_tr.shape[0] < 1 or th_va.shape[0] < 1:
        raise ValueError("linear_x_flow requires non-empty train and validation splits.")

    x_mean = np.mean(x_tr, axis=0, dtype=np.float64)
    x_std = np.maximum(np.std(x_tr, axis=0, dtype=np.float64), 1e-6)
    x_tr_n = (x_tr - x_mean) / x_std
    x_va_n = (x_va - x_mean) / x_std

    train_ds = TensorDataset(
        torch.from_numpy(th_tr.astype(np.float32)),
        torch.from_numpy(x_tr_n.astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(th_va.astype(np.float32)),
        torch.from_numpy(x_va_n.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_monitor_losses: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    best_eval_state_cpu: dict[str, torch.Tensor] | None = None
    weight_ema_enabled = float(weight_ema_decay) > 0.0
    weight_ema_state = init_model_weight_ema(model) if weight_ema_enabled else None
    patience_counter = 0
    stopped_early = False
    stopped_epoch = int(epochs)
    n_clipped_steps = 0
    n_total_steps = 0
    val_ema: float | None = None
    alpha = float(ema_alpha)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        ep_losses: list[float] = []
        for tb, x1b in train_loader:
            tb = tb.to(device)
            x1b = x1b.to(device)
            bs = int(x1b.shape[0])
            t = te + (1.0 - 2.0 * te) * torch.rand(bs, 1, device=device, dtype=x1b.dtype)
            x0b = torch.randn_like(x1b)
            xt = (1.0 - t) * x0b + t * x1b
            ut = x1b - x0b
            v = model(xt, tb)
            loss = torch.mean((v - ut) ** 2)
            reg_loss = model.regularization_loss()
            if reg_loss is not None:
                loss = loss + reg_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            n_total_steps += 1
            if float(max_grad_norm) > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
                if float(grad_norm) > float(max_grad_norm):
                    n_clipped_steps += 1
            opt.step()
            if weight_ema_state is not None:
                update_model_weight_ema(weight_ema_state, model, decay=float(weight_ema_decay))
            ep_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(ep_losses))
        train_losses.append(train_loss)

        model.eval()
        val_ep: list[float] = []
        ema_ctx = (
            evaluate_with_weight_ema(model, weight_ema_state)
            if weight_ema_state is not None
            else nullcontext()
        )
        with ema_ctx:
            with torch.no_grad():
                for tb, x1b in val_loader:
                    tb = tb.to(device)
                    x1b = x1b.to(device)
                    bs = int(x1b.shape[0])
                    t = te + (1.0 - 2.0 * te) * torch.rand(bs, 1, device=device, dtype=x1b.dtype)
                    x0b = torch.randn_like(x1b)
                    xt = (1.0 - t) * x0b + t * x1b
                    ut = x1b - x0b
                    loss_b = torch.mean((model(xt, tb) - ut) ** 2)
                    reg_loss = model.regularization_loss()
                    if reg_loss is not None:
                        loss_b = loss_b + reg_loss
                    val_ep.append(float(loss_b.detach().cpu()))
        val_raw = float(np.mean(val_ep))
        val_losses.append(val_raw)
        val_ema = scalar_val_ema_update(val_ema, val_raw, alpha)
        val_smooth = float(val_ema)
        val_monitor_losses.append(val_smooth)
        if val_smooth < best_val - float(min_delta):
            best_val = float(val_smooth)
            best_epoch = int(epoch)
            best_eval_state_cpu = (
                clone_model_weight_ema(weight_ema_state)
                if weight_ema_state is not None
                else {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            )
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch == 1 or epoch % max(1, int(log_every)) == 0 or epoch == int(epochs):
            print(
                f"[linear_x_flow {epoch:4d}/{int(epochs)}] train_fm={train_loss:.6f} "
                f"val_fm={val_raw:.6f} val_smooth={val_smooth:.6f} best_monitor={best_val:.6f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
        if int(patience) > 0 and patience_counter >= int(patience):
            stopped_early = True
            stopped_epoch = int(epoch)
            print(
                f"[linear_x_flow early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_monitor={best_val:.6f} patience={int(patience)}",
                flush=True,
            )
            break

    final_eval_weights = "raw"
    if restore_best and best_eval_state_cpu is not None:
        if weight_ema_enabled:
            load_model_weights_from_ema_state(model, best_eval_state_cpu)
            final_eval_weights = "ema"
            print(
                f"[linear_x_flow restore-best] restored EMA eval weights epoch={best_epoch} "
                f"best_monitor={best_val:.6f}",
                flush=True,
            )
        else:
            model.load_state_dict(best_eval_state_cpu)
            final_eval_weights = "raw"
            print(
                f"[linear_x_flow restore-best] restored raw eval weights epoch={best_epoch} "
                f"best_monitor={best_val:.6f}",
                flush=True,
            )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_monitor_losses": val_monitor_losses,
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
        "lr_last": float(opt.param_groups[0]["lr"]),
        "n_clipped_steps": int(n_clipped_steps),
        "n_total_steps": int(n_total_steps),
        "weight_ema_enabled": bool(weight_ema_enabled),
        "weight_ema_decay": float(weight_ema_decay),
        "final_eval_weights": final_eval_weights,
        "x_mean": x_mean.astype(np.float64),
        "x_std": x_std.astype(np.float64),
    }


def train_time_diagonal_linear_x_flow_schedule(
    *,
    model: ConditionalTimeDiagonalLinearXFlowMLP,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    device: torch.device,
    schedule: GaussianAffinePathSchedule,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.0,
    t_eps: float = 0.05,
    patience: int = 1000,
    min_delta: float = 1e-4,
    ema_alpha: float = 0.05,
    weight_ema_decay: float = 0.9,
    max_grad_norm: float = 10.0,
    log_every: int = 50,
    restore_best: bool = True,
) -> dict[str, Any]:
    """Train ``v(x,t,theta)=diag(a(t))x+b(t,theta)`` on a scheduled affine bridge."""
    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1.")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1.")
    if float(lr) <= 0.0:
        raise ValueError("lr must be > 0.")
    if int(patience) < 0:
        raise ValueError("patience must be >= 0.")
    if float(min_delta) < 0.0:
        raise ValueError("min_delta must be >= 0.")
    if not (0.0 < float(ema_alpha) <= 1.0):
        raise ValueError("ema_alpha must be in (0, 1].")
    if not np.isfinite(float(weight_ema_decay)) or float(weight_ema_decay) >= 1.0:
        raise ValueError("weight_ema_decay must be finite and < 1.")
    te = float(t_eps)
    if not (0.0 < te < 0.5):
        raise ValueError("t_eps must be in (0, 0.5).")

    th_tr = _as_2d_float64(theta_train, name="theta_train")
    x_tr = _as_2d_float64(x_train, name="x_train")
    th_va = _as_2d_float64(theta_val, name="theta_val")
    x_va = _as_2d_float64(x_val, name="x_val")
    if th_tr.shape[0] < 1 or th_va.shape[0] < 1:
        raise ValueError("linear_x_flow_diagonal_t requires non-empty train and validation splits.")

    x_mean = np.mean(x_tr, axis=0, dtype=np.float64)
    x_std = np.maximum(np.std(x_tr, axis=0, dtype=np.float64), 1e-6)
    x_tr_n = (x_tr - x_mean) / x_std
    x_va_n = (x_va - x_mean) / x_std

    train_ds = TensorDataset(torch.from_numpy(th_tr.astype(np.float32)), torch.from_numpy(x_tr_n.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(th_va.astype(np.float32)), torch.from_numpy(x_va_n.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_monitor_losses: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    best_eval_state_cpu: dict[str, torch.Tensor] | None = None
    weight_ema_enabled = float(weight_ema_decay) > 0.0
    weight_ema_state = init_model_weight_ema(model) if weight_ema_enabled else None
    patience_counter = 0
    stopped_early = False
    stopped_epoch = int(epochs)
    n_clipped_steps = 0
    n_total_steps = 0
    val_ema: float | None = None
    alpha = float(ema_alpha)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        ep_losses: list[float] = []
        for tb, x1b in train_loader:
            tb = tb.to(device)
            x1b = x1b.to(device)
            bs = int(x1b.shape[0])
            t_raw = torch.rand(bs, 1, device=device, dtype=x1b.dtype)
            t = te + (1.0 - 2.0 * te) * t_raw
            x0b = torch.randn_like(x1b)
            a, bcoef, ad, bd = schedule.ab_ad_bd(t)
            xt = a * x0b + bcoef * x1b
            ut = ad * x0b + bd * x1b
            loss = torch.mean((model(xt, tb, t) - ut) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            n_total_steps += 1
            if float(max_grad_norm) > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
                if float(grad_norm) > float(max_grad_norm):
                    n_clipped_steps += 1
            opt.step()
            if weight_ema_state is not None:
                update_model_weight_ema(weight_ema_state, model, decay=float(weight_ema_decay))
            ep_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(ep_losses))
        train_losses.append(train_loss)

        model.eval()
        val_ep: list[float] = []
        ema_ctx = evaluate_with_weight_ema(model, weight_ema_state) if weight_ema_state is not None else nullcontext()
        with ema_ctx:
            with torch.no_grad():
                for tb, x1b in val_loader:
                    tb = tb.to(device)
                    x1b = x1b.to(device)
                    bs = int(x1b.shape[0])
                    t_raw = torch.rand(bs, 1, device=device, dtype=x1b.dtype)
                    t = te + (1.0 - 2.0 * te) * t_raw
                    x0b = torch.randn_like(x1b)
                    a, bcoef, ad, bd = schedule.ab_ad_bd(t)
                    xt = a * x0b + bcoef * x1b
                    ut = ad * x0b + bd * x1b
                    val_ep.append(float(torch.mean((model(xt, tb, t) - ut) ** 2).detach().cpu()))
        val_raw = float(np.mean(val_ep))
        val_losses.append(val_raw)
        val_ema = scalar_val_ema_update(val_ema, val_raw, alpha)
        val_smooth = float(val_ema)
        val_monitor_losses.append(val_smooth)
        if val_smooth < best_val - float(min_delta):
            best_val = float(val_smooth)
            best_epoch = int(epoch)
            best_eval_state_cpu = (
                clone_model_weight_ema(weight_ema_state)
                if weight_ema_state is not None
                else {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            )
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch == 1 or epoch % max(1, int(log_every)) == 0 or epoch == int(epochs):
            print(
                f"[linear_x_flow_diagonal_t {epoch:4d}/{int(epochs)}] train_fm={train_loss:.6f} "
                f"val_fm={val_raw:.6f} val_smooth={val_smooth:.6f} best_monitor={best_val:.6f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
        if int(patience) > 0 and patience_counter >= int(patience):
            stopped_early = True
            stopped_epoch = int(epoch)
            print(
                f"[linear_x_flow_diagonal_t early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_monitor={best_val:.6f} patience={int(patience)}",
                flush=True,
            )
            break

    final_eval_weights = "raw"
    if restore_best and best_eval_state_cpu is not None:
        if weight_ema_enabled:
            load_model_weights_from_ema_state(model, best_eval_state_cpu)
            final_eval_weights = "ema"
            print(
                f"[linear_x_flow_diagonal_t restore-best] restored EMA eval weights epoch={best_epoch} "
                f"best_monitor={best_val:.6f}",
                flush=True,
            )
        else:
            model.load_state_dict(best_eval_state_cpu)
            print(
                f"[linear_x_flow_diagonal_t restore-best] restored raw eval weights epoch={best_epoch} "
                f"best_monitor={best_val:.6f}",
                flush=True,
            )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_monitor_losses": val_monitor_losses,
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
        "lr_last": float(opt.param_groups[0]["lr"]),
        "n_clipped_steps": int(n_clipped_steps),
        "n_total_steps": int(n_total_steps),
        "weight_ema_enabled": bool(weight_ema_enabled),
        "weight_ema_decay": float(weight_ema_decay),
        "final_eval_weights": final_eval_weights,
        "x_mean": x_mean.astype(np.float64),
        "x_std": x_std.astype(np.float64),
    }


def train_pca_nonlinear_linear_x_flow(
    *,
    model: ConditionalPCANonlinearLinearXFlowMLP,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    device: torch.device,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.0,
    t_eps: float = 0.05,
    lambda_h: float = 0.0,
    freeze_linear: bool = False,
    patience: int = 1000,
    min_delta: float = 1e-4,
    ema_alpha: float = 0.05,
    weight_ema_decay: float = 0.9,
    max_grad_norm: float = 10.0,
    solve_jitter: float = 1e-6,
    log_every: int = 50,
    restore_best: bool = True,
) -> dict[str, Any]:
    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1.")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1.")
    if float(lr) <= 0.0:
        raise ValueError("lr must be > 0.")
    if float(lambda_h) < 0.0:
        raise ValueError("lambda_h must be >= 0.")
    if int(patience) < 0:
        raise ValueError("patience must be >= 0.")
    if float(min_delta) < 0.0:
        raise ValueError("min_delta must be >= 0.")
    if not (0.0 < float(ema_alpha) <= 1.0):
        raise ValueError("ema_alpha must be in (0, 1].")
    if not np.isfinite(float(weight_ema_decay)) or float(weight_ema_decay) >= 1.0:
        raise ValueError("weight_ema_decay must be finite and < 1.")
    te = float(t_eps)
    if not (0.0 < te < 0.5):
        raise ValueError("t_eps must be in (0, 0.5).")

    th_tr = _as_2d_float64(theta_train, name="theta_train")
    x_tr = _as_2d_float64(x_train, name="x_train")
    th_va = _as_2d_float64(theta_val, name="theta_val")
    x_va = _as_2d_float64(x_val, name="x_val")
    xm = np.asarray(x_mean, dtype=np.float64).reshape(1, -1)
    xs = np.asarray(x_std, dtype=np.float64).reshape(1, -1)
    x_tr_n = (x_tr - xm) / xs
    x_va_n = (x_va - xm) / xs

    old_requires_grad = [p.requires_grad for p in model.linear_model.parameters()]
    if bool(freeze_linear):
        for p in model.linear_model.parameters():
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))

    train_ds = TensorDataset(torch.from_numpy(th_tr.astype(np.float32)), torch.from_numpy(x_tr_n.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(th_va.astype(np.float32)), torch.from_numpy(x_va_n.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_monitor_losses: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    best_eval_state_cpu: dict[str, torch.Tensor] | None = None
    weight_ema_enabled = float(weight_ema_decay) > 0.0
    weight_ema_state = init_model_weight_ema(model) if weight_ema_enabled else None
    patience_counter = 0
    stopped_early = False
    stopped_epoch = int(epochs)
    n_clipped_steps = 0
    n_total_steps = 0
    val_ema: float | None = None
    alpha = float(ema_alpha)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        ep_losses: list[float] = []
        for tb, x1b in train_loader:
            tb = tb.to(device)
            x1b = x1b.to(device)
            bs = int(x1b.shape[0])
            t = te + (1.0 - 2.0 * te) * torch.rand(bs, 1, device=device, dtype=x1b.dtype)
            x0b = torch.randn_like(x1b)
            xt = (1.0 - t) * x0b + t * x1b
            ut = x1b - x0b
            v = model(xt, tb, t, solve_jitter=float(solve_jitter))
            _, _, h = model.nonlinear_correction(xt, t, tb, solve_jitter=float(solve_jitter))
            loss = torch.mean((v - ut) ** 2) + model.regularization_loss(h, float(lambda_h))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            n_total_steps += 1
            if float(max_grad_norm) > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm))
                if float(grad_norm) > float(max_grad_norm):
                    n_clipped_steps += 1
            opt.step()
            if weight_ema_state is not None:
                update_model_weight_ema(weight_ema_state, model, decay=float(weight_ema_decay))
            ep_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(ep_losses))
        train_losses.append(train_loss)

        model.eval()
        val_ep: list[float] = []
        ema_ctx = evaluate_with_weight_ema(model, weight_ema_state) if weight_ema_state is not None else nullcontext()
        with ema_ctx:
            with torch.no_grad():
                for tb, x1b in val_loader:
                    tb = tb.to(device)
                    x1b = x1b.to(device)
                    bs = int(x1b.shape[0])
                    t = te + (1.0 - 2.0 * te) * torch.rand(bs, 1, device=device, dtype=x1b.dtype)
                    x0b = torch.randn_like(x1b)
                    xt = (1.0 - t) * x0b + t * x1b
                    ut = x1b - x0b
                    v = model(xt, tb, t, solve_jitter=float(solve_jitter))
                    _, _, h = model.nonlinear_correction(xt, t, tb, solve_jitter=float(solve_jitter))
                    loss_b = torch.mean((v - ut) ** 2) + model.regularization_loss(h, float(lambda_h))
                    val_ep.append(float(loss_b.detach().cpu()))
        val_raw = float(np.mean(val_ep))
        val_losses.append(val_raw)
        val_ema = scalar_val_ema_update(val_ema, val_raw, alpha)
        val_smooth = float(val_ema)
        val_monitor_losses.append(val_smooth)
        if val_smooth < best_val - float(min_delta):
            best_val = float(val_smooth)
            best_epoch = int(epoch)
            best_eval_state_cpu = (
                clone_model_weight_ema(weight_ema_state)
                if weight_ema_state is not None
                else {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            )
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch == 1 or epoch % max(1, int(log_every)) == 0 or epoch == int(epochs):
            print(
                f"[linear_x_flow_nonlinear_pca {epoch:4d}/{int(epochs)}] train_fm={train_loss:.6f} "
                f"val_fm={val_raw:.6f} val_smooth={val_smooth:.6f} best_monitor={best_val:.6f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
        if int(patience) > 0 and patience_counter >= int(patience):
            stopped_early = True
            stopped_epoch = int(epoch)
            print(
                f"[linear_x_flow_nonlinear_pca early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_monitor={best_val:.6f} patience={int(patience)}",
                flush=True,
            )
            break

    final_eval_weights = "raw"
    if restore_best and best_eval_state_cpu is not None:
        if weight_ema_enabled:
            load_model_weights_from_ema_state(model, best_eval_state_cpu)
            final_eval_weights = "ema"
        else:
            model.load_state_dict(best_eval_state_cpu)

    if bool(freeze_linear):
        for p, req in zip(model.linear_model.parameters(), old_requires_grad):
            p.requires_grad_(req)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_monitor_losses": val_monitor_losses,
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
        "lr_last": float(opt.param_groups[0]["lr"]),
        "n_clipped_steps": int(n_clipped_steps),
        "n_total_steps": int(n_total_steps),
        "weight_ema_enabled": bool(weight_ema_enabled),
        "weight_ema_decay": float(weight_ema_decay),
        "final_eval_weights": final_eval_weights,
    }


def compute_linear_x_flow_c_matrix(
    *,
    model: ConditionalLinearXFlowMLP,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    device: torch.device,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    solve_jitter: float = 1e-6,
    pair_batch_size: int = 65536,
) -> np.ndarray:
    theta = _as_2d_float64(theta_all, name="theta_all")
    x = _as_2d_float64(x_all, name="x_all")
    if theta.shape[0] != x.shape[0]:
        raise ValueError("theta_all and x_all row counts must match.")
    n = int(theta.shape[0])
    if int(pair_batch_size) < 1:
        raise ValueError("pair_batch_size must be >= 1.")
    row_block = max(1, int(pair_batch_size) // max(n, 1))
    theta32 = theta.astype(np.float32, copy=False)
    x_mean_t = torch.from_numpy(np.asarray(x_mean, dtype=np.float32)).to(device)
    x_std_t = torch.from_numpy(np.asarray(x_std, dtype=np.float32)).to(device)
    c = np.zeros((n, n), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for i0 in range(0, n, row_block):
            i1 = min(n, i0 + row_block)
            xb = x[i0:i1].astype(np.float32, copy=False)
            b = int(i1 - i0)
            x_rep = np.repeat(xb, repeats=n, axis=0)
            theta_tile = np.tile(theta32, (b, 1))
            x_t = torch.from_numpy(x_rep).to(device)
            theta_t = torch.from_numpy(theta_tile).to(device)
            logp = model.log_prob_observed(
                x_t,
                theta_t,
                x_mean=x_mean_t,
                x_std=x_std_t,
                solve_jitter=float(solve_jitter),
            )
            c[i0:i1, :] = logp.reshape(b, n).detach().cpu().numpy().astype(np.float64)
    return c


def compute_time_diagonal_linear_x_flow_c_matrix(
    *,
    model: ConditionalTimeDiagonalLinearXFlowMLP,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    device: torch.device,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    solve_jitter: float = 1e-6,
    quadrature_steps: int = 64,
    pair_batch_size: int = 65536,
) -> np.ndarray:
    theta = _as_2d_float64(theta_all, name="theta_all")
    x = _as_2d_float64(x_all, name="x_all")
    if theta.shape[0] != x.shape[0]:
        raise ValueError("theta_all and x_all row counts must match.")
    n = int(theta.shape[0])
    if int(pair_batch_size) < 1:
        raise ValueError("pair_batch_size must be >= 1.")
    if int(quadrature_steps) < 2:
        raise ValueError("quadrature_steps must be >= 2.")
    row_block = max(1, int(pair_batch_size) // max(n, 1))
    theta32 = theta.astype(np.float32, copy=False)
    x_mean_t = torch.from_numpy(np.asarray(x_mean, dtype=np.float32)).to(device)
    x_std_t = torch.from_numpy(np.asarray(x_std, dtype=np.float32)).to(device)
    c = np.zeros((n, n), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for i0 in range(0, n, row_block):
            i1 = min(n, i0 + row_block)
            xb = x[i0:i1].astype(np.float32, copy=False)
            b = int(i1 - i0)
            x_rep = np.repeat(xb, repeats=n, axis=0)
            theta_tile = np.tile(theta32, (b, 1))
            x_t = torch.from_numpy(x_rep).to(device)
            theta_t = torch.from_numpy(theta_tile).to(device)
            logp = model.log_prob_observed(
                x_t,
                theta_t,
                x_mean=x_mean_t,
                x_std=x_std_t,
                solve_jitter=float(solve_jitter),
                quadrature_steps=int(quadrature_steps),
            )
            c[i0:i1, :] = logp.reshape(b, n).detach().cpu().numpy().astype(np.float64)
    return c


def compute_pca_nonlinear_linear_x_flow_c_matrix(
    *,
    model: ConditionalPCANonlinearLinearXFlowMLP,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    device: torch.device,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    solve_jitter: float = 1e-6,
    ode_steps: int = 32,
    pair_batch_size: int = 65536,
) -> np.ndarray:
    theta = _as_2d_float64(theta_all, name="theta_all")
    x = _as_2d_float64(x_all, name="x_all")
    if theta.shape[0] != x.shape[0]:
        raise ValueError("theta_all and x_all row counts must match.")
    n = int(theta.shape[0])
    if int(pair_batch_size) < 1:
        raise ValueError("pair_batch_size must be >= 1.")
    row_block = max(1, int(pair_batch_size) // max(n, 1))
    theta32 = theta.astype(np.float32, copy=False)
    x_mean_t = torch.from_numpy(np.asarray(x_mean, dtype=np.float32)).to(device)
    x_std_t = torch.from_numpy(np.asarray(x_std, dtype=np.float32)).to(device)
    c = np.zeros((n, n), dtype=np.float64)
    model.eval()
    for i0 in range(0, n, row_block):
        i1 = min(n, i0 + row_block)
        xb = x[i0:i1].astype(np.float32, copy=False)
        b = int(i1 - i0)
        x_rep = np.repeat(xb, repeats=n, axis=0)
        theta_tile = np.tile(theta32, (b, 1))
        x_t = torch.from_numpy(x_rep).to(device)
        theta_t = torch.from_numpy(theta_tile).to(device)
        logp = model.log_prob_observed(
            x_t,
            theta_t,
            x_mean=x_mean_t,
            x_std=x_std_t,
            solve_jitter=float(solve_jitter),
            ode_steps=int(ode_steps),
        )
        c[i0:i1, :] = logp.detach().cpu().numpy().reshape(b, n).astype(np.float64)
    return c
