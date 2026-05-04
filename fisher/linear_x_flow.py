"""Linear x-space flow matching with analytic Gaussian likelihood.

This module trains supported scheduled linear x-space flow variants on affine
noise-to-data bridges and evaluates their endpoint or ODE likelihoods.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrizations
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


def _as_col_t(t: torch.Tensor, *, batch: int | None = None) -> torch.Tensor:
    if t.ndim == 0:
        t = t.reshape(1, 1)
    elif t.ndim == 1:
        t = t.unsqueeze(-1)
    if t.ndim != 2 or int(t.shape[1]) != 1:
        raise ValueError("t must have shape [B] or [B, 1].")
    if batch is not None and int(t.shape[0]) == 1 and int(batch) > 1:
        t = t.expand(int(batch), 1)
    return t


def _make_mlp(
    *,
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    depth: int,
    final_gain: float = 0.01,
    final_bias: float = 0.0,
) -> nn.Sequential:
    # SiLU follows each hidden Linear; use ReLU-family Xavier gain for hidden weights.
    _gain_hidden = float(nn.init.calculate_gain("relu"))
    layers: list[nn.Module] = []
    cur = int(in_dim)
    for _ in range(int(depth)):
        lin = nn.Linear(cur, int(hidden_dim))
        nn.init.xavier_uniform_(lin.weight, gain=_gain_hidden)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        layers.append(nn.SiLU())
        cur = int(hidden_dim)
    out = nn.Linear(cur, int(out_dim))
    nn.init.xavier_uniform_(out.weight, gain=float(final_gain))
    nn.init.constant_(out.bias, float(final_bias))
    layers.append(out)
    return nn.Sequential(*layers)


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
        if isinstance(
            model,
            (
                ConditionalTimeDiagonalLinearXFlowMLP,
                ConditionalTimeScalarLinearXFlowMLP,
                ConditionalTimeThetaDiagonalLinearXFlowMLP,
            ),
        ):
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
        if isinstance(
            model,
            (
                ConditionalTimeLinearXFlowMLP,
                ConditionalTimeLowRankLinearXFlowMLP,
                ConditionalTimeRandomBasisLowRankLinearXFlowMLP,
            ),
        ):
            mu_t, cov_t = model.endpoint_mean_covariance(
                theta_t,
                solve_jitter=float(solve_jitter),
                quadrature_steps=quadrature_steps,
            )
            cov_np = cov_t.detach().cpu().numpy().astype(np.float64)
            if cov_np.ndim == 3 and cov_np.shape[0] == theta.shape[0]:
                cov0 = cov_np[0]
                if np.allclose(cov_np, cov0.reshape(1, *cov0.shape), rtol=1e-5, atol=1e-7):
                    cov_np = cov0
            return (
                mu_t.detach().cpu().numpy().astype(np.float64),
                cov_np,
                False,
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
    elif cov_or_diag.ndim == 3:
        n = int(mu.shape[0])
        h = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                h_ij = gaussian_hellinger_sq_full(
                    mu[i],
                    cov_or_diag[i],
                    mu[j],
                    cov_or_diag[j],
                    jitter=float(solve_jitter) * 1e-3,
                )
                h[i, j] = h[j, i] = h_ij
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


class _BaseTimeLinearXFlowMLP(nn.Module):
    """Shared utilities for scheduled time-dependent linear X-flow models."""

    endpoint_is_diagonal = False

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
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.quadrature_steps = int(quadrature_steps)
        self.b_net = _make_mlp(
            in_dim=self.theta_dim + 1,
            out_dim=self.x_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.01,
            final_bias=0.0,
        )

    def b(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        t = _as_col_t(t, batch=int(theta.shape[0]))
        return self.b_net(torch.cat([t, theta], dim=1))

    def A(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def regularization_loss(self) -> torch.Tensor | None:
        return None

    def forward(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = _as_col_t(t, batch=int(x.shape[0]))
        a = self.A(t)
        b = self.b(theta, t)
        if a.ndim == 2:
            return x @ a.transpose(0, 1) + b
        return torch.bmm(a, x.unsqueeze(-1)).squeeze(-1) + b

    def endpoint_mean_covariance(
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
        batch = int(theta.shape[0])
        d = int(self.x_dim)
        mu = torch.zeros(batch, d, dtype=theta.dtype, device=theta.device)
        cov = torch.eye(d, dtype=theta.dtype, device=theta.device).reshape(1, d, d).expand(batch, d, d).clone()
        dt = 1.0 / float(q)
        for k in range(q):
            tk = torch.full((batch, 1), (float(k) + 0.5) / float(q), dtype=theta.dtype, device=theta.device)
            a = self.A(tk)
            if a.ndim == 2:
                a = a.unsqueeze(0).expand(batch, d, d)
            b = self.b(theta, tk)
            mu = mu + dt * (torch.bmm(a, mu.unsqueeze(-1)).squeeze(-1) + b)
            cov = cov + dt * (torch.bmm(a, cov) + torch.bmm(cov, a.transpose(1, 2)))
            cov = 0.5 * (cov + cov.transpose(1, 2))
        eye = torch.eye(d, dtype=theta.dtype, device=theta.device).reshape(1, d, d)
        cov = 0.5 * (cov + cov.transpose(1, 2)) + float(solve_jitter) * eye
        return mu, cov

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
        mu, cov = self.endpoint_mean_covariance(
            theta,
            solve_jitter=float(solve_jitter),
            quadrature_steps=quadrature_steps,
        )
        d = int(x_norm.shape[1])
        eye = torch.eye(d, dtype=x_norm.dtype, device=x_norm.device).reshape(1, d, d)
        l = torch.linalg.cholesky(cov + float(solve_jitter) * eye)
        diff = x_norm - mu
        z = torch.cholesky_solve(diff.unsqueeze(-1), l).squeeze(-1)
        quad = torch.sum(diff * z, dim=1)
        log_det = 2.0 * torch.sum(torch.log(torch.clamp(torch.diagonal(l, dim1=-2, dim2=-1), min=1e-12)), dim=1)
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


class ConditionalTimeLinearXFlowMLP(_BaseTimeLinearXFlowMLP):
    """Full symmetric time-dependent drift ``A(t)`` plus offset ``b(t, theta)``."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        quadrature_steps: int = 64,
        a_final_gain: float = 0.01,
        a_final_bias: float = 0.0,
        a_identity_offset: float = 1e-3,
    ) -> None:
        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            quadrature_steps=quadrature_steps,
        )
        self.a_net = _make_mlp(
            in_dim=1,
            out_dim=self.x_dim * self.x_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=float(a_final_gain),
            final_bias=float(a_final_bias),
        )
        self.a_identity_offset = float(a_identity_offset)

    def A(self, t: torch.Tensor) -> torch.Tensor:
        t = _as_col_t(t)
        raw = self.a_net(t).reshape(int(t.shape[0]), self.x_dim, self.x_dim)
        eye = torch.eye(self.x_dim, dtype=raw.dtype, device=raw.device).reshape(1, self.x_dim, self.x_dim)
        return 0.5 * (raw + raw.transpose(1, 2)) + self.a_identity_offset * eye


class ConditionalTimeThetaOnlyLinearXFlowMLP(ConditionalTimeLinearXFlowMLP):
    """Full symmetric time-dependent ``A(t)`` plus offset ``b(theta)`` (no ``t`` in ``b``).

    Same quadrature / Gaussian endpoint machinery as :class:`ConditionalTimeLinearXFlowMLP`, but
    ``b_net`` maps only ``theta`` so ``b(theta, t)`` is constant in ``t``.
    """

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        quadrature_steps: int = 64,
        a_final_gain: float = 0.01,
        a_final_bias: float = 0.0,
        a_identity_offset: float = 1e-3,
    ) -> None:
        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            quadrature_steps=quadrature_steps,
            a_final_gain=float(a_final_gain),
            a_final_bias=float(a_final_bias),
            a_identity_offset=float(a_identity_offset),
        )
        self.b_net = _make_mlp(
            in_dim=self.theta_dim,
            out_dim=self.x_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.01,
            final_bias=0.0,
        )

    def b(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        return self.b_net(theta)


class ConditionalTimeLowRankCorrectionLinearXFlowMLP(nn.Module):
    """Full symmetric scheduled ``A(t)`` linear x-flow plus learnable orthonormal low-rank correction.

    Velocity is ``v(x,t,theta) = A(t) x + b(t,theta) + U h(U^T x, t, theta)`` with ``U`` in ``R^{D x r}``
    having orthonormal columns (``U^T U = I``).  The base ``(A,b)`` matches :class:`ConditionalTimeLinearXFlowMLP`.

    Likelihood uses a reverse-Euler divergence integral because the low-rank correction
    has no closed-form Gaussian endpoint.

    The reduced-space Jacobian trace ``\\mathrm{tr}\\,\\partial h/\\partial z`` (with ``z = U^T x``)
    defaults to a Hutchinson estimator (Rademacher probes); set ``divergence_estimator="exact"`` for the
    rank-loop autograd sum (slower when ``correction_rank`` is large).
    """

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        correction_rank: int,
        hidden_dim: int = 128,
        depth: int = 3,
        quadrature_steps: int = 64,
        divergence_estimator: str = "hutchinson",
        hutchinson_probes: int = 1,
    ) -> None:
        super().__init__()
        if int(correction_rank) < 1:
            raise ValueError("correction_rank must be >= 1.")
        if int(correction_rank) > int(x_dim):
            raise ValueError("correction_rank must be <= x_dim.")
        if int(quadrature_steps) < 2:
            raise ValueError("quadrature_steps must be >= 2.")
        de = str(divergence_estimator).strip().lower()
        if de not in ("hutchinson", "exact"):
            raise ValueError("divergence_estimator must be one of: hutchinson, exact.")
        if int(hutchinson_probes) < 1:
            raise ValueError("hutchinson_probes must be >= 1.")
        self.divergence_estimator = de
        self.hutchinson_probes = int(hutchinson_probes)
        self.linear = ConditionalTimeLinearXFlowMLP(
            theta_dim=int(theta_dim),
            x_dim=int(x_dim),
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            quadrature_steps=int(quadrature_steps),
            a_final_gain=0.0,
            a_final_bias=0.0,
            a_identity_offset=0.0,
        )
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.quadrature_steps = int(quadrature_steps)
        self.correction_rank = int(correction_rank)
        u_lin = nn.Linear(self.correction_rank, self.x_dim, bias=False)
        nn.init.orthogonal_(u_lin.weight)
        self.u_layer = parametrizations.orthogonal(u_lin, "weight", orthogonal_map="householder")
        self.h_net = _make_mlp(
            in_dim=self.correction_rank + 1 + self.theta_dim,
            out_dim=self.correction_rank,
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            final_gain=0.0,
            final_bias=0.0,
        )

    @property
    def U(self) -> torch.Tensor:
        """Orthonormal columns ``[D, r]`` (``nn.Linear(r, D).weight``)."""
        return self.u_layer.weight

    def forward(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        base = self.linear(x, theta, t)
        u_mat = self.U
        z = x @ u_mat
        if theta.ndim == 1:
            theta2 = theta.unsqueeze(-1)
        else:
            theta2 = theta
        tcol = _as_col_t(t, batch=int(x.shape[0]))
        h = self.h_net(torch.cat([z, tcol, theta2], dim=1))
        return base + h @ u_mat.T

    def regularization_loss(self) -> torch.Tensor | None:
        return None

    def _reduced_trace_exact(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """``sum_j \\partial h_j / \\partial z_j`` via one autograd per output component."""
        tr_h = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for j in range(self.correction_rank):
            grad_j = torch.autograd.grad(
                h[:, j].sum(),
                z,
                create_graph=False,
                retain_graph=j < self.correction_rank - 1,
            )[0]
            tr_h = tr_h + grad_j[:, j]
        return tr_h

    def _reduced_trace_hutchinson(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Hutchinson estimate of ``\\mathrm{tr}\\,\\partial h/\\partial z`` with Rademacher probes."""
        b = int(z.shape[0])
        r = int(self.correction_rank)
        n_probe = int(self.hutchinson_probes)
        acc = torch.zeros(b, dtype=z.dtype, device=z.device)
        for pi in range(n_probe):
            probe = torch.empty(b, r, dtype=z.dtype, device=z.device)
            probe.bernoulli_(0.5).mul_(2.0).sub_(1.0)
            dot = (h * probe).sum(dim=1)
            vjp = torch.autograd.grad(
                dot.sum(),
                z,
                create_graph=False,
                retain_graph=pi < n_probe - 1,
            )[0]
            acc = acc + (vjp * probe).sum(dim=-1)
        return acc / float(n_probe)

    def divergence(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        t = _as_col_t(t, batch=int(x.shape[0]))
        a_mat = self.linear.A(t)
        if a_mat.ndim == 2:
            tr_a = torch.trace(a_mat).to(dtype=x.dtype, device=x.device).reshape(1).expand(int(x.shape[0]))
        else:
            tr_a = torch.diagonal(a_mat, dim1=-2, dim2=-1).sum(dim=-1)
        u_mat = self.U
        with torch.enable_grad():
            z = (x @ u_mat).detach().requires_grad_(True)
            tcol = _as_col_t(t, batch=int(z.shape[0]))
            h = self.h_net(torch.cat([z, tcol, theta], dim=1))
            if self.divergence_estimator == "exact":
                tr_h = self._reduced_trace_exact(z, h)
            else:
                tr_h = self._reduced_trace_hutchinson(z, h)
        return tr_a.detach() + tr_h.detach()

    def log_prob_normalized(
        self,
        x_norm: torch.Tensor,
        theta: torch.Tensor,
        *,
        solve_jitter: float = 1e-6,
        quadrature_steps: int | None = None,
        ode_steps: int = 32,
    ) -> torch.Tensor:
        del solve_jitter, quadrature_steps  # API parity with other scheduled LXF likelihoods
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
            div_int = div_int + dt * self.divergence(x, theta, t)
            with torch.no_grad():
                v = self.forward(x, theta, t)
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
        quadrature_steps: int | None = None,
        ode_steps: int = 32,
    ) -> torch.Tensor:
        z = (x_raw - x_mean) / x_std
        logjac = -torch.sum(torch.log(x_std))
        return self.log_prob_normalized(
            z,
            theta,
            solve_jitter=float(solve_jitter),
            quadrature_steps=quadrature_steps,
            ode_steps=int(ode_steps),
        ) + logjac


class ConditionalTimeThetaOnlyBLowRankCorrectionLinearXFlowMLP(nn.Module):
    """Scheduled full ``A(t)`` linear x-flow plus low-rank correction with ``b(theta)`` (no ``t`` in ``b``).

    Velocity is ``v(x,t,theta) = A(t) x + b(theta) + U h(U^T x, t, theta)`` with orthonormal ``U``.
    Otherwise matches :class:`ConditionalTimeLowRankCorrectionLinearXFlowMLP` (ODE likelihood, Hutchinson
    divergence on ``h`` in ``z = U^T x``).
    """

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        correction_rank: int,
        hidden_dim: int = 128,
        depth: int = 3,
        quadrature_steps: int = 64,
        divergence_estimator: str = "hutchinson",
        hutchinson_probes: int = 1,
    ) -> None:
        super().__init__()
        if int(correction_rank) < 1:
            raise ValueError("correction_rank must be >= 1.")
        if int(correction_rank) > int(x_dim):
            raise ValueError("correction_rank must be <= x_dim.")
        if int(quadrature_steps) < 2:
            raise ValueError("quadrature_steps must be >= 2.")
        de = str(divergence_estimator).strip().lower()
        if de not in ("hutchinson", "exact"):
            raise ValueError("divergence_estimator must be one of: hutchinson, exact.")
        if int(hutchinson_probes) < 1:
            raise ValueError("hutchinson_probes must be >= 1.")
        self.divergence_estimator = de
        self.hutchinson_probes = int(hutchinson_probes)
        self.linear = ConditionalTimeThetaOnlyLinearXFlowMLP(
            theta_dim=int(theta_dim),
            x_dim=int(x_dim),
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            quadrature_steps=int(quadrature_steps),
            a_final_gain=0.0,
            a_final_bias=0.0,
            a_identity_offset=0.0,
        )
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.quadrature_steps = int(quadrature_steps)
        self.correction_rank = int(correction_rank)
        u_lin = nn.Linear(self.correction_rank, self.x_dim, bias=False)
        nn.init.orthogonal_(u_lin.weight)
        self.u_layer = parametrizations.orthogonal(u_lin, "weight", orthogonal_map="householder")
        self.h_net = _make_mlp(
            in_dim=self.correction_rank + 1 + self.theta_dim,
            out_dim=self.correction_rank,
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            final_gain=0.0,
            final_bias=0.0,
        )

    @property
    def U(self) -> torch.Tensor:
        """Orthonormal columns ``[D, r]`` (``nn.Linear(r, D).weight``)."""
        return self.u_layer.weight

    def forward(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        base = self.linear(x, theta, t)
        u_mat = self.U
        z = x @ u_mat
        if theta.ndim == 1:
            theta2 = theta.unsqueeze(-1)
        else:
            theta2 = theta
        tcol = _as_col_t(t, batch=int(x.shape[0]))
        h = self.h_net(torch.cat([z, tcol, theta2], dim=1))
        return base + h @ u_mat.T

    def regularization_loss(self) -> torch.Tensor | None:
        return None

    def _reduced_trace_exact(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        tr_h = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for j in range(self.correction_rank):
            grad_j = torch.autograd.grad(
                h[:, j].sum(),
                z,
                create_graph=False,
                retain_graph=j < self.correction_rank - 1,
            )[0]
            tr_h = tr_h + grad_j[:, j]
        return tr_h

    def _reduced_trace_hutchinson(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        b = int(z.shape[0])
        r = int(self.correction_rank)
        n_probe = int(self.hutchinson_probes)
        acc = torch.zeros(b, dtype=z.dtype, device=z.device)
        for pi in range(n_probe):
            probe = torch.empty(b, r, dtype=z.dtype, device=z.device)
            probe.bernoulli_(0.5).mul_(2.0).sub_(1.0)
            dot = (h * probe).sum(dim=1)
            vjp = torch.autograd.grad(
                dot.sum(),
                z,
                create_graph=False,
                retain_graph=pi < n_probe - 1,
            )[0]
            acc = acc + (vjp * probe).sum(dim=-1)
        return acc / float(n_probe)

    def divergence(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        t = _as_col_t(t, batch=int(x.shape[0]))
        a_mat = self.linear.A(t)
        if a_mat.ndim == 2:
            tr_a = torch.trace(a_mat).to(dtype=x.dtype, device=x.device).reshape(1).expand(int(x.shape[0]))
        else:
            tr_a = torch.diagonal(a_mat, dim1=-2, dim2=-1).sum(dim=-1)
        u_mat = self.U
        with torch.enable_grad():
            z = (x @ u_mat).detach().requires_grad_(True)
            tcol = _as_col_t(t, batch=int(z.shape[0]))
            h = self.h_net(torch.cat([z, tcol, theta], dim=1))
            if self.divergence_estimator == "exact":
                tr_h = self._reduced_trace_exact(z, h)
            else:
                tr_h = self._reduced_trace_hutchinson(z, h)
        return tr_a.detach() + tr_h.detach()

    def log_prob_normalized(
        self,
        x_norm: torch.Tensor,
        theta: torch.Tensor,
        *,
        solve_jitter: float = 1e-6,
        quadrature_steps: int | None = None,
        ode_steps: int = 32,
    ) -> torch.Tensor:
        del solve_jitter, quadrature_steps
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
            div_int = div_int + dt * self.divergence(x, theta, t)
            with torch.no_grad():
                v = self.forward(x, theta, t)
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
        quadrature_steps: int | None = None,
        ode_steps: int = 32,
    ) -> torch.Tensor:
        z = (x_raw - x_mean) / x_std
        logjac = -torch.sum(torch.log(x_std))
        return self.log_prob_normalized(
            z,
            theta,
            solve_jitter=float(solve_jitter),
            quadrature_steps=quadrature_steps,
            ode_steps=int(ode_steps),
        ) + logjac


class ConditionalTimeScalarLinearXFlowMLP(_BaseTimeLinearXFlowMLP):
    """Scalar time-dependent drift ``A(t)=a(t) I`` plus offset ``b(t, theta)``."""

    endpoint_is_diagonal = True

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        quadrature_steps: int = 64,
    ) -> None:
        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            quadrature_steps=quadrature_steps,
        )
        self.a_net = _make_mlp(
            in_dim=1,
            out_dim=1,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.01,
            final_bias=1e-3,
        )

    def a(self, t: torch.Tensor) -> torch.Tensor:
        return self.a_net(_as_col_t(t))

    def A(self, t: torch.Tensor) -> torch.Tensor:
        a = self.a(t).reshape(-1)
        eye = torch.eye(self.x_dim, dtype=a.dtype, device=a.device).reshape(1, self.x_dim, self.x_dim)
        mats = a.reshape(-1, 1, 1) * eye
        return mats[0] if int(mats.shape[0]) == 1 else mats

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
        s = torch.trapezoid(a_grid.reshape(q), grid.reshape(-1), dim=0)
        tail = torch.zeros(q, dtype=theta.dtype, device=theta.device)
        if q > 1:
            dt = grid[1:, 0] - grid[:-1, 0]
            seg = 0.5 * (a_grid[:-1, 0] + a_grid[1:, 0]) * dt
            tail[:-1] = torch.flip(torch.cumsum(torch.flip(seg, dims=(0,)), dim=0), dims=(0,))
        b_vals: list[torch.Tensor] = []
        for k in range(q):
            tk = grid[k].reshape(1, 1).expand(int(theta.shape[0]), 1)
            b_vals.append(self.b(theta, tk))
        b_grid = torch.stack(b_vals, dim=0)
        mu = torch.trapezoid(torch.exp(tail).reshape(q, 1, 1) * b_grid, grid.reshape(-1), dim=0)
        var = torch.exp(2.0 * s).reshape(1, 1).expand_as(mu) + float(solve_jitter)
        return mu, var

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
        mu, var = self.endpoint_mean_covariance_diag(
            theta,
            solve_jitter=float(solve_jitter),
            quadrature_steps=quadrature_steps,
        )
        d = int(x_norm.shape[1])
        quad = torch.sum((x_norm - mu) ** 2 / var, dim=1)
        log_det = torch.sum(torch.log(var), dim=1)
        return -0.5 * (quad + log_det + float(d) * math.log(2.0 * math.pi))


class ConditionalTimeThetaDiagonalLinearXFlowMLP(_BaseTimeLinearXFlowMLP):
    """Diagonal drift ``a(t, theta)`` and offset ``b(t, theta)``."""

    endpoint_is_diagonal = True

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        quadrature_steps: int = 64,
    ) -> None:
        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            quadrature_steps=quadrature_steps,
        )
        self.ab_net = _make_mlp(
            in_dim=self.theta_dim + 1,
            out_dim=2 * self.x_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.01,
            final_bias=0.0,
        )

    def a_b(self, theta: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        t = _as_col_t(t, batch=int(theta.shape[0]))
        out = self.ab_net(torch.cat([t, theta], dim=1))
        a, b = out.chunk(2, dim=1)
        return a + 1e-3, b

    def A(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("ConditionalTimeThetaDiagonalLinearXFlowMLP needs theta to build A.")

    def b(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _, b = self.a_b(theta, t)
        return b

    def forward(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a, b = self.a_b(theta, t)
        return a * x + b

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
        mu = torch.zeros(int(theta.shape[0]), self.x_dim, dtype=theta.dtype, device=theta.device)
        var = torch.ones_like(mu)
        dt = 1.0 / float(q)
        for k in range(q):
            tk = torch.full((int(theta.shape[0]), 1), (float(k) + 0.5) / float(q), dtype=theta.dtype, device=theta.device)
            a, b = self.a_b(theta, tk)
            mu = mu + dt * (a * mu + b)
            var = var + dt * (2.0 * a * var)
            var = torch.clamp(var, min=float(solve_jitter))
        return mu, var + float(solve_jitter)

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
        mu, var = self.endpoint_mean_covariance_diag(
            theta,
            solve_jitter=float(solve_jitter),
            quadrature_steps=quadrature_steps,
        )
        d = int(x_norm.shape[1])
        quad = torch.sum((x_norm - mu) ** 2 / var, dim=1)
        log_det = torch.sum(torch.log(var), dim=1)
        return -0.5 * (quad + log_det + float(d) * math.log(2.0 * math.pi))


class ConditionalTimeLowRankLinearXFlowMLP(_BaseTimeLinearXFlowMLP):
    """Time-dependent low-rank symmetric drift ``A(t)=diag(a(t))+U(t)diag(s(t))U(t)^T``.

    This class is not currently exposed by the H-decoding CLIs; it remains as a reusable
    scheduled low-rank Gaussian endpoint component.
    """

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        rank: int,
        hidden_dim: int = 128,
        depth: int = 3,
        quadrature_steps: int = 64,
    ) -> None:
        if int(rank) < 1:
            raise ValueError("rank must be >= 1.")
        if int(rank) > int(x_dim):
            raise ValueError("rank must be <= x_dim.")
        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            quadrature_steps=quadrature_steps,
        )
        self.rank = int(rank)
        self.a_net = _make_mlp(
            in_dim=1,
            out_dim=self.x_dim + self.x_dim * self.rank + self.rank,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.01,
            final_bias=0.0,
        )

    def A(self, t: torch.Tensor) -> torch.Tensor:
        t = _as_col_t(t)
        out = self.a_net(t)
        d = self.x_dim
        r = self.rank
        a = out[:, :d] + 1e-3
        u = out[:, d : d + d * r].reshape(-1, d, r)
        s = out[:, d + d * r :].reshape(-1, r)
        diag = torch.diag_embed(a)
        low = torch.bmm(u * s.unsqueeze(1), u.transpose(1, 2))
        mats = diag + low
        return mats[0] if int(mats.shape[0]) == 1 else mats


class ConditionalTimeRandomBasisLowRankLinearXFlowMLP(_BaseTimeLinearXFlowMLP):
    """Time-dependent random-basis low-rank drift ``A(t)=diag(a(t))+Q S(t) Q.T``."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        rank: int,
        hidden_dim: int = 128,
        depth: int = 3,
        quadrature_steps: int = 64,
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
        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            quadrature_steps=quadrature_steps,
        )
        self.rank = int(rank)
        self.lambda_a = float(lambda_a)
        self.lambda_s = float(lambda_s)
        q, _ = torch.linalg.qr(torch.randn(self.x_dim, self.rank), mode="reduced")
        self.register_buffer("Q", q)
        self.a_net = _make_mlp(
            in_dim=1,
            out_dim=self.x_dim + self.rank * self.rank,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.01,
            final_bias=0.0,
        )

    def A(self, t: torch.Tensor) -> torch.Tensor:
        t = _as_col_t(t)
        out = self.a_net(t)
        d = self.x_dim
        r = self.rank
        a = out[:, :d]
        s_raw = out[:, d:].reshape(-1, r, r)
        s = 0.5 * (s_raw + s_raw.transpose(1, 2))
        q = self.Q.to(dtype=out.dtype, device=out.device).reshape(1, d, r).expand(int(out.shape[0]), d, r)
        mats = torch.diag_embed(a) + torch.bmm(torch.bmm(q, s), q.transpose(1, 2))
        return mats[0] if int(mats.shape[0]) == 1 else mats

    def regularization_loss(self) -> torch.Tensor:
        grid = torch.linspace(0.0, 1.0, 5, dtype=self.Q.dtype, device=self.Q.device).reshape(5, 1)
        out = self.a_net(grid)
        a = out[:, : self.x_dim]
        s_raw = out[:, self.x_dim :].reshape(5, self.rank, self.rank)
        s = 0.5 * (s_raw + s_raw.transpose(1, 2))
        return float(self.lambda_a) * torch.mean(a**2) + float(self.lambda_s) * torch.mean(s**2)


def time_linear_x_flow_trace(model: nn.Module, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Trace of the scheduled linear drift for supported time-linear base families."""
    if theta.ndim == 1:
        theta = theta.unsqueeze(-1)
    t = _as_col_t(t, batch=int(theta.shape[0]))
    if isinstance(model, ConditionalTimeDiagonalLinearXFlowMLP):
        return torch.sum(model.a(t), dim=1)
    if isinstance(model, ConditionalTimeThetaDiagonalLinearXFlowMLP):
        a, _ = model.a_b(theta, t)
        return torch.sum(a, dim=1)
    a = model.A(t)  # type: ignore[attr-defined]
    if a.ndim == 2:
        tr = torch.trace(a).to(dtype=theta.dtype, device=theta.device)
        return tr.reshape(1).expand(int(theta.shape[0]))
    return torch.diagonal(a, dim1=-2, dim2=-1).sum(dim=1)


def _adamw_param_groups_no_wd_on_parametrizations(model: nn.Module, *, weight_decay: float) -> list[dict[str, Any]]:
    """AdamW param groups: disable weight decay on ``torch.nn.utils.parametrizations`` internals.

    Decaying unconstrained ``.original`` tensors breaks orthogonal maps (e.g. collapses ``U`` to zeros).
    """
    wd = float(weight_decay)
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "parametrizations" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": wd})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        raise ValueError("model has no trainable parameters for AdamW.")
    return groups


def train_time_linear_x_flow_schedule(
    *,
    model: nn.Module,
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
    log_name: str = "linear_x_flow_t",
) -> dict[str, Any]:
    """Train ``v(x,t,theta)`` on a scheduled affine bridge."""
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
        raise ValueError(f"{log_name} requires non-empty train and validation splits.")

    x_mean = np.mean(x_tr, axis=0, dtype=np.float64)
    x_std = np.maximum(np.std(x_tr, axis=0, dtype=np.float64), 1e-6)
    x_tr_n = (x_tr - x_mean) / x_std
    x_va_n = (x_va - x_mean) / x_std

    train_ds = TensorDataset(torch.from_numpy(th_tr.astype(np.float32)), torch.from_numpy(x_tr_n.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(th_va.astype(np.float32)), torch.from_numpy(x_va_n.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)
    opt = torch.optim.AdamW(_adamw_param_groups_no_wd_on_parametrizations(model, weight_decay=float(weight_decay)), lr=float(lr))

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
                f"[{log_name} {epoch:4d}/{int(epochs)}] train_fm={train_loss:.6f} "
                f"val_fm={val_raw:.6f} val_smooth={val_smooth:.6f} best_monitor={best_val:.6f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
        if int(patience) > 0 and patience_counter >= int(patience):
            stopped_early = True
            stopped_epoch = int(epoch)
            print(
                f"[{log_name} early-stop] epoch={epoch} best_epoch={best_epoch} "
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
                f"[{log_name} restore-best] restored EMA eval weights epoch={best_epoch} "
                f"best_monitor={best_val:.6f}",
                flush=True,
            )
        else:
            model.load_state_dict(best_eval_state_cpu)
            print(
                f"[{log_name} restore-best] restored raw eval weights epoch={best_epoch} "
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


def train_b_theta_mean_regression_schedule(
    *,
    model: nn.Module,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.0,
    patience: int = 1000,
    min_delta: float = 1e-4,
    ema_alpha: float = 0.05,
    weight_ema_decay: float = 0.9,
    max_grad_norm: float = 10.0,
    log_every: int = 50,
    restore_best: bool = True,
    log_name: str = "b_theta_mean_regression",
) -> dict[str, Any]:
    """Train ``b(theta)`` to match normalized data ``x1`` via mean-squared error (no bridge time)."""
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

    th_tr = _as_2d_float64(theta_train, name="theta_train")
    x_tr = _as_2d_float64(x_train, name="x_train")
    th_va = _as_2d_float64(theta_val, name="theta_val")
    x_va = _as_2d_float64(x_val, name="x_val")
    if th_tr.shape[0] < 1 or th_va.shape[0] < 1:
        raise ValueError(f"{log_name} requires non-empty train and validation splits.")

    x_mean = np.mean(x_tr, axis=0, dtype=np.float64)
    x_std = np.maximum(np.std(x_tr, axis=0, dtype=np.float64), 1e-6)
    x_tr_n = (x_tr - x_mean) / x_std
    x_va_n = (x_va - x_mean) / x_std

    train_ds = TensorDataset(torch.from_numpy(th_tr.astype(np.float32)), torch.from_numpy(x_tr_n.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(th_va.astype(np.float32)), torch.from_numpy(x_va_n.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)
    opt = torch.optim.AdamW(model.linear.b_net.parameters(), lr=float(lr), weight_decay=float(weight_decay))

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
            t0 = torch.zeros(bs, 1, device=device, dtype=x1b.dtype)
            pred = model.linear.b(tb, t0)
            loss = torch.mean((pred - x1b) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            n_total_steps += 1
            if float(max_grad_norm) > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.linear.b_net.parameters(), float(max_grad_norm))
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
                    t0 = torch.zeros(bs, 1, device=device, dtype=x1b.dtype)
                    pred = model.linear.b(tb, t0)
                    val_ep.append(float(torch.mean((pred - x1b) ** 2).detach().cpu()))
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
                f"[{log_name} {epoch:4d}/{int(epochs)}] train_mse={train_loss:.6f} "
                f"val_mse={val_raw:.6f} val_smooth={val_smooth:.6f} best_monitor={best_val:.6f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
        if int(patience) > 0 and patience_counter >= int(patience):
            stopped_early = True
            stopped_epoch = int(epoch)
            print(
                f"[{log_name} early-stop] epoch={epoch} best_epoch={best_epoch} "
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
                f"[{log_name} restore-best] restored EMA eval weights epoch={best_epoch} "
                f"best_monitor={best_val:.6f}",
                flush=True,
            )
        else:
            model.load_state_dict(best_eval_state_cpu)
            print(
                f"[{log_name} restore-best] restored raw eval weights epoch={best_epoch} "
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


def train_low_rank_t_theta_only_b_mean_regression_pretrain_then_freeze_b(
    *,
    model: nn.Module,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    device: torch.device,
    schedule: GaussianAffinePathSchedule,
    warmup_epochs: int,
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
    log_name: str = "linear_x_flow_lr_t_ts",
) -> dict[str, Any]:
    """Pretrain ``b(theta)`` via mean regression, freeze ``b_net``, then scheduled FM on ``A,U,h``."""
    if int(warmup_epochs) < 1:
        raise ValueError("warmup_epochs must be >= 1.")

    original_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
    try:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.linear.b_net.parameters():
            p.requires_grad_(True)

        warmup_out = train_b_theta_mean_regression_schedule(
            model=model,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            device=device,
            epochs=int(warmup_epochs),
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            min_delta=min_delta,
            ema_alpha=ema_alpha,
            weight_ema_decay=weight_ema_decay,
            max_grad_norm=max_grad_norm,
            log_every=log_every,
            restore_best=restore_best,
            log_name=f"{log_name}_warmup",
        )
    finally:
        for name, p in model.named_parameters():
            p.requires_grad_(original_requires_grad.get(name, True))

    for p in model.parameters():
        p.requires_grad_(True)
    for p in model.linear.b_net.parameters():
        p.requires_grad_(False)

    full_out = train_time_linear_x_flow_schedule(
        model=model,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        device=device,
        schedule=schedule,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        t_eps=t_eps,
        patience=patience,
        min_delta=min_delta,
        ema_alpha=ema_alpha,
        weight_ema_decay=weight_ema_decay,
        max_grad_norm=max_grad_norm,
        log_every=log_every,
        restore_best=restore_best,
        log_name=log_name,
    )
    full_out.update(
        {
            "lxf_low_rank_t_warmup_enabled": True,
            "lxf_low_rank_t_warmup_epochs": int(warmup_epochs),
            "lxf_low_rank_t_warmup_objective": "mean_regression",
            "lxf_low_rank_t_second_stage_freeze_b_enabled": True,
            "warmup_train_losses": warmup_out["train_losses"],
            "warmup_val_losses": warmup_out["val_losses"],
            "warmup_val_monitor_losses": warmup_out["val_monitor_losses"],
            "warmup_best_val_loss": float(warmup_out["best_val_loss"]),
            "warmup_best_epoch": int(warmup_out["best_epoch"]),
            "warmup_stopped_epoch": int(warmup_out["stopped_epoch"]),
            "warmup_stopped_early": bool(warmup_out["stopped_early"]),
            "warmup_lr_last": float(warmup_out.get("lr_last", float("nan"))),
            "warmup_n_clipped_steps": int(warmup_out.get("n_clipped_steps", 0)),
            "warmup_n_total_steps": int(warmup_out.get("n_total_steps", 0)),
            "warmup_weight_ema_enabled": bool(warmup_out.get("weight_ema_enabled", False)),
            "warmup_weight_ema_decay": float(warmup_out.get("weight_ema_decay", weight_ema_decay)),
            "warmup_final_eval_weights": str(warmup_out.get("final_eval_weights", "raw")),
        }
    )
    return full_out


def train_low_rank_t_warmup_then_full(
    *,
    model: nn.Module,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    device: torch.device,
    schedule: GaussianAffinePathSchedule,
    warmup_epochs: int,
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
    log_name: str = "linear_x_flow_low_rank_t",
) -> dict[str, Any]:
    """Warm up only ``b(t, theta)``, then run the normal scheduled low-rank-t trainer."""
    if int(warmup_epochs) < 1:
        raise ValueError("warmup_epochs must be >= 1.")

    original_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
    try:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.linear.b_net.parameters():
            p.requires_grad_(True)

        warmup_out = train_time_linear_x_flow_schedule(
            model=model,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            device=device,
            schedule=schedule,
            epochs=int(warmup_epochs),
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            t_eps=t_eps,
            patience=patience,
            min_delta=min_delta,
            ema_alpha=ema_alpha,
            weight_ema_decay=weight_ema_decay,
            max_grad_norm=max_grad_norm,
            log_every=log_every,
            restore_best=restore_best,
            log_name=f"{log_name}_warmup",
        )
    finally:
        for name, p in model.named_parameters():
            p.requires_grad_(original_requires_grad.get(name, True))

    for p in model.parameters():
        p.requires_grad_(True)

    full_out = train_time_linear_x_flow_schedule(
        model=model,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        device=device,
        schedule=schedule,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        t_eps=t_eps,
        patience=patience,
        min_delta=min_delta,
        ema_alpha=ema_alpha,
        weight_ema_decay=weight_ema_decay,
        max_grad_norm=max_grad_norm,
        log_every=log_every,
        restore_best=restore_best,
        log_name=log_name,
    )
    full_out.update(
        {
            "lxf_low_rank_t_warmup_enabled": True,
            "lxf_low_rank_t_warmup_epochs": int(warmup_epochs),
            "warmup_train_losses": warmup_out["train_losses"],
            "warmup_val_losses": warmup_out["val_losses"],
            "warmup_val_monitor_losses": warmup_out["val_monitor_losses"],
            "warmup_best_val_loss": float(warmup_out["best_val_loss"]),
            "warmup_best_epoch": int(warmup_out["best_epoch"]),
            "warmup_stopped_epoch": int(warmup_out["stopped_epoch"]),
            "warmup_stopped_early": bool(warmup_out["stopped_early"]),
            "warmup_lr_last": float(warmup_out.get("lr_last", float("nan"))),
            "warmup_n_clipped_steps": int(warmup_out.get("n_clipped_steps", 0)),
            "warmup_n_total_steps": int(warmup_out.get("n_total_steps", 0)),
            "warmup_weight_ema_enabled": bool(warmup_out.get("weight_ema_enabled", False)),
            "warmup_weight_ema_decay": float(warmup_out.get("weight_ema_decay", weight_ema_decay)),
            "warmup_final_eval_weights": str(warmup_out.get("final_eval_weights", "raw")),
        }
    )
    return full_out


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
    """Compatibility wrapper for the original diagonal-time method."""
    return train_time_linear_x_flow_schedule(
        model=model,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        device=device,
        schedule=schedule,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        t_eps=t_eps,
        patience=patience,
        min_delta=min_delta,
        ema_alpha=ema_alpha,
        weight_ema_decay=weight_ema_decay,
        max_grad_norm=max_grad_norm,
        log_every=log_every,
        restore_best=restore_best,
        log_name="linear_x_flow_diagonal_t",
    )


def compute_time_linear_x_flow_c_matrix(
    *,
    model: nn.Module,
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
    return compute_time_linear_x_flow_c_matrix(
        model=model,
        theta_all=theta_all,
        x_all=x_all,
        device=device,
        x_mean=x_mean,
        x_std=x_std,
        solve_jitter=solve_jitter,
        quadrature_steps=quadrature_steps,
        pair_batch_size=pair_batch_size,
    )


def compute_ode_time_linear_x_flow_c_matrix(
    *,
    model: nn.Module,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    device: torch.device,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    solve_jitter: float = 1e-6,
    quadrature_steps: int = 64,
    ode_steps: int = 32,
    pair_batch_size: int = 65536,
) -> np.ndarray:
    """Pairwise log ``p(x_i | theta_j)`` for ODE-likelihood scheduled LXF models (e.g. PCA correction)."""
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
            quadrature_steps=int(quadrature_steps),
            ode_steps=int(ode_steps),
        )
        c[i0:i1, :] = logp.detach().cpu().numpy().reshape(b, n).astype(np.float64)
    return c


