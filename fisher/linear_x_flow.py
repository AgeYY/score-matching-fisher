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

:class:`ConditionalThetaDiagonalSplineLinearXFlowMLP` is the same diagonal flow, but ``a`` and ``b``
are linear maps of fixed scalar-theta B-spline features (see ``spline_basis_features_normalized``).

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


def open_uniform_clamped_knot_vector(
    num_basis: int,
    degree: int = 3,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Open-uniform clamped B-spline knot vector for ``num_basis`` functions at given ``degree``.

    Returns ``U`` of length ``num_basis + degree + 1`` (Cox--de Boor indexing).
    """
    n = int(num_basis)
    p = int(degree)
    if n < 1:
        raise ValueError("num_basis must be >= 1.")
    if p < 1:
        raise ValueError("degree must be >= 1.")
    if n < p + 1:
        raise ValueError(f"num_basis must be >= degree+1 (need {p + 1}); got {n}.")
    m = n + p  # last knot index; total knots = m + 1
    u = torch.zeros(m + 1, dtype=dtype, device=device)
    u[: p + 1] = 0.0
    u[m - p :] = 1.0
    if n - p - 1 > 0:
        for j in range(1, n - p):
            u[p + j] = float(j) / float(n - p)
    return u


def bspline_basis_phi_batch(
    u01: torch.Tensor,
    knots: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """Evaluate clamped B-spline basis ``N_{i,p}(u)`` for all ``i=0..n-1``.

    ``u01`` has shape ``[B]`` with values in ``[0, 1]``. ``knots`` has length ``n+p+1``.
    Returns ``phi`` of shape ``[B, n]`` where ``n = len(knots) - degree - 1``.
    """
    if u01.ndim != 1:
        raise ValueError("u01 must be 1D [B].")
    p = int(degree)
    dtype = u01.dtype
    device = u01.device
    u = u01.clamp(1e-6, 1.0 - 1e-6)
    u_mat = knots.to(dtype=dtype, device=device)
    m = int(u_mat.shape[0]) - 1
    n = m - p
    if n < 1:
        raise ValueError("Invalid knots/degree: inferred num_basis < 1.")
    bsz = int(u.shape[0])
    ui = u[:, None]
    u_left = u_mat[:-1].unsqueeze(0)
    u_right = u_mat[1:].unsqueeze(0)
    span_len = u_right - u_left
    zero_len = span_len.abs() < 1e-15
    interior = (ui >= u_left) & (ui < u_right)
    n0 = interior & ~zero_len
    n_curr = n0.to(dtype=dtype)
    eps_d = torch.tensor(1e-15, dtype=dtype, device=device)

    for _r in range(1, p + 1):
        m_new = m - _r
        n_next = torch.zeros(bsz, m_new, dtype=dtype, device=device)
        for i in range(m_new):
            u_i = u_mat[i]
            u_ir = u_mat[i + _r]
            denom1 = u_ir - u_i
            t1 = torch.where(
                denom1.abs() > eps_d,
                (u - u_i) / denom1 * n_curr[:, i],
                torch.zeros(bsz, dtype=dtype, device=device),
            )
            u_i1 = u_mat[i + 1]
            u_ir1 = u_mat[i + _r + 1]
            denom2 = u_ir1 - u_i1
            t2 = torch.where(
                denom2.abs() > eps_d,
                (u_ir1 - u) / denom2 * n_curr[:, i + 1],
                torch.zeros(bsz, dtype=dtype, device=device),
            )
            n_next[:, i] = t1 + t2
        n_curr = n_next
    if int(n_curr.shape[1]) != n:
        raise RuntimeError(f"B-spline basis width mismatch: got {n_curr.shape[1]}, expected {n}.")
    return n_curr


def spline_basis_features_normalized(
    theta: torch.Tensor,
    *,
    theta_min: torch.Tensor,
    theta_max: torch.Tensor,
    num_basis: int,
    knots: torch.Tensor,
    degree: int = 3,
) -> torch.Tensor:
    """Map scalar ``theta`` to ``[0,1]`` and return B-spline features ``phi`` of shape ``[B, K]``."""
    if theta.ndim == 1:
        theta = theta.unsqueeze(-1)
    if int(theta.shape[1]) != 1:
        raise ValueError("spline_basis_features_normalized expects theta with last dim 1 (scalar theta).")
    t0 = theta_min.to(dtype=theta.dtype, device=theta.device).reshape(1, 1)
    t1 = theta_max.to(dtype=theta.dtype, device=theta.device).reshape(1, 1)
    den = (t1 - t0).clamp_min(1e-12)
    u01 = ((theta - t0) / den).clamp(0.0, 1.0)
    u_flat = u01.reshape(-1).clamp(1e-6, 1.0 - 1e-6)
    return bspline_basis_phi_batch(u_flat, knots, int(degree))


def _as_2d_float64(a: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D.")
    return arr


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


class ConditionalThetaDiagonalSplineLinearXFlowMLP(nn.Module):
    """Diagonal drift ``a(theta)`` and offset ``b(theta)`` from fixed B-spline features of scalar ``theta``.

    Velocity ``v(x,theta)=a(theta) \\odot x + b(theta)`` with the same diagonal-Gaussian endpoint as
    :class:`ConditionalThetaDiagonalLinearXFlowMLP` (``\\Sigma_{ii}=exp(2 a_i)``,
    ``\\mu_i = ((e^{a_i}-1)/a_i) b_i``).

    ``theta`` is normalized to ``[0,1]`` using fixed ``theta_min``, ``theta_max`` (typically from the
    training set). Only ``theta_dim=1`` is supported.
    """

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        theta_min: float,
        theta_max: float,
        num_basis: int = 5,
        spline_degree: int = 3,
    ) -> None:
        super().__init__()
        if int(theta_dim) != 1:
            raise ValueError(
                "ConditionalThetaDiagonalSplineLinearXFlowMLP requires scalar theta (theta_dim=1)."
            )
        if int(x_dim) < 1:
            raise ValueError("x_dim must be >= 1.")
        self.theta_dim = 1
        self.x_dim = int(x_dim)
        self.num_basis = int(num_basis)
        self.spline_degree = int(spline_degree)
        if self.num_basis < self.spline_degree + 1:
            raise ValueError(
                f"num_basis must be >= spline_degree+1 (need {self.spline_degree + 1}); got {self.num_basis}."
            )
        knots = open_uniform_clamped_knot_vector(
            self.num_basis,
            degree=self.spline_degree,
            dtype=torch.float32,
        )
        self.register_buffer("knots", knots)
        tmn = float(theta_min)
        tmx = float(theta_max)
        if (not np.isfinite(tmn)) or (not np.isfinite(tmx)):
            raise ValueError("theta_min and theta_max must be finite.")
        if tmx <= tmn:
            tmx = tmn + 1e-6
        self.register_buffer("theta_min_buf", torch.tensor([tmn], dtype=torch.float32))
        self.register_buffer("theta_max_buf", torch.tensor([tmx], dtype=torch.float32))
        self.Wa = nn.Parameter(torch.zeros(self.x_dim, self.num_basis))
        self.ca = nn.Parameter(torch.full((self.x_dim,), 1e-3))
        self.Wb = nn.Parameter(torch.zeros(self.x_dim, self.num_basis))
        self.cb = nn.Parameter(torch.zeros(self.x_dim))
        nn.init.normal_(self.Wa, mean=0.0, std=0.02)
        nn.init.normal_(self.Wb, mean=0.0, std=0.02)

    def spline_phi(self, theta: torch.Tensor) -> torch.Tensor:
        return spline_basis_features_normalized(
            theta,
            theta_min=self.theta_min_buf,
            theta_max=self.theta_max_buf,
            num_basis=self.num_basis,
            knots=self.knots,
            degree=self.spline_degree,
        )

    def a_b(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phi = self.spline_phi(theta)
        a_theta = phi @ self.Wa.T + self.ca
        b_theta = phi @ self.Wb.T + self.cb
        return a_theta, b_theta

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
        phi_scale = _phi_expm1_div_a(a_theta)
        mu = phi_scale * b_theta
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


def train_linear_x_flow_schedule(
    *,
    model: ConditionalLinearXFlowMLP,
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
    """Train the same time-independent model on a scheduled affine bridge."""
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
        raise ValueError("linear_x_flow_schedule requires non-empty train and validation splits.")

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
            t_raw = torch.rand(bs, 1, device=device, dtype=x1b.dtype)
            t = te + (1.0 - 2.0 * te) * t_raw
            x0b = torch.randn_like(x1b)
            a, bcoef, ad, bd = schedule.ab_ad_bd(t)
            xt = a * x0b + bcoef * x1b
            ut = ad * x0b + bd * x1b
            loss = torch.mean((model(xt, tb) - ut) ** 2)
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
                    t_raw = torch.rand(bs, 1, device=device, dtype=x1b.dtype)
                    t = te + (1.0 - 2.0 * te) * t_raw
                    x0b = torch.randn_like(x1b)
                    a, bcoef, ad, bd = schedule.ab_ad_bd(t)
                    xt = a * x0b + bcoef * x1b
                    ut = ad * x0b + bd * x1b
                    val_ep.append(float(torch.mean((model(xt, tb) - ut) ** 2).detach().cpu()))
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
                f"[linear_x_flow_schedule {epoch:4d}/{int(epochs)}] train_fm={train_loss:.6f} "
                f"val_fm={val_raw:.6f} val_smooth={val_smooth:.6f} best_monitor={best_val:.6f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
        if int(patience) > 0 and patience_counter >= int(patience):
            stopped_early = True
            stopped_epoch = int(epoch)
            print(
                f"[linear_x_flow_schedule early-stop] epoch={epoch} best_epoch={best_epoch} "
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
                f"[linear_x_flow_schedule restore-best] restored EMA eval weights epoch={best_epoch} "
                f"best_monitor={best_val:.6f}",
                flush=True,
            )
        else:
            model.load_state_dict(best_eval_state_cpu)
            final_eval_weights = "raw"
            print(
                f"[linear_x_flow_schedule restore-best] restored raw eval weights epoch={best_epoch} "
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
