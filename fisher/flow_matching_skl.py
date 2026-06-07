"""Flow-matching endpoint metrics for model-induced symmetric KL.

This module uses the Jeffreys convention
``DSKL(p, q) = KL(p || q) + KL(q || p)``.  The existing
``fisher.llr_divergence.symmetric_kl_gaussian_full_matrix`` helper returns the
half-symmetrized convention used by older LLR code, so Gaussian endpoint calls
are multiplied by two here.  With this convention, adjacent
``DSKL(theta, theta + dtheta) / dtheta**2`` estimates the scalar Fisher
information directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrizations
from torch.utils.data import DataLoader, TensorDataset

from fisher.gaussian_x_flow import GaussianAffinePathSchedule, path_schedule_from_name
from fisher.linear_x_flow import resolve_lxf_low_rank_dim
from fisher.llr_divergence import symmetric_kl_gaussian_full_matrix


VELOCITY_FAMILIES = (
    "translation",
    "translation_fixed_norm",
    "translation_centered_fixed_norm",
    "shared_affine",
    "shared_affine_scalar",
    "shared_affine_diag",
    "condition_affine",
    "condition_affine_scalar",
    "condition_affine_diag",
    "shared_affine_low_rank",
    "shared_affine_low_rank_scalar",
    "shared_affine_low_rank_diag",
    "nonlinear",
)

TRANSLATION_FAMILIES = {
    "translation",
    "translation_fixed_norm",
    "translation_centered_fixed_norm",
}

SHARED_GAUSSIAN_AFFINE_FAMILIES = {
    "shared_affine",
    "shared_affine_scalar",
    "shared_affine_diag",
}

CONDITION_GAUSSIAN_AFFINE_FAMILIES = {
    "condition_affine",
    "condition_affine_scalar",
    "condition_affine_diag",
}

LOW_RANK_AFFINE_FAMILIES = {
    "shared_affine_low_rank",
    "shared_affine_low_rank_scalar",
    "shared_affine_low_rank_diag",
}

GAUSSIAN_ENDPOINT_FAMILIES = (
    TRANSLATION_FAMILIES
    | SHARED_GAUSSIAN_AFFINE_FAMILIES
    | CONDITION_GAUSSIAN_AFFINE_FAMILIES
)

MC_ENDPOINT_FAMILIES = LOW_RANK_AFFINE_FAMILIES | {"nonlinear"}


@dataclass
class FlowSKLResult:
    """Endpoint metric bundle returned by ``estimate_model_symmetric_kl``."""

    symmetric_kl_matrix: np.ndarray
    canonical_metric_matrix: np.ndarray
    canonical_metric_name: str
    endpoint_mean: np.ndarray | None
    endpoint_covariance: np.ndarray | None
    fisher_theta_midpoints: np.ndarray | None = None
    fisher_full: np.ndarray | None = None
    fisher_linear: np.ndarray | None = None
    train_metadata: dict[str, Any] = field(default_factory=dict)
    normalization: dict[str, Any] = field(default_factory=dict)


def _normalize_velocity_family(velocity_family: str) -> str:
    fam = str(velocity_family).strip().lower().replace("-", "_")
    if fam not in VELOCITY_FAMILIES:
        raise ValueError(f"velocity_family must be one of {VELOCITY_FAMILIES}; got {velocity_family!r}.")
    return fam


def _as_2d_float64(a: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D.")
    return arr


def _as_torch_2d(a: np.ndarray, *, device: torch.device) -> torch.Tensor:
    arr = _as_2d_float64(a, name="array")
    return torch.from_numpy(arr.astype(np.float32, copy=False)).to(device)


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
) -> nn.Sequential:
    if int(depth) < 1:
        raise ValueError("depth must be >= 1.")
    layers: list[nn.Module] = []
    cur = int(in_dim)
    gain = float(nn.init.calculate_gain("relu"))
    for _ in range(int(depth)):
        lin = nn.Linear(cur, int(hidden_dim))
        nn.init.xavier_uniform_(lin.weight, gain=gain)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        layers.append(nn.SiLU())
        cur = int(hidden_dim)
    out = nn.Linear(cur, int(out_dim))
    nn.init.xavier_uniform_(out.weight, gain=float(final_gain))
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


def _resolve_path_schedule(
    path_schedule: str | GaussianAffinePathSchedule,
) -> tuple[GaussianAffinePathSchedule, str]:
    if isinstance(path_schedule, str):
        key = str(path_schedule).strip().lower()
        schedule = path_schedule_from_name(key)
        name = "cosine" if key in ("cosine", "cos") else "linear"
        return schedule, name
    if not hasattr(path_schedule, "ab_ad_bd"):
        raise TypeError("path_schedule must be a schedule name or expose ab_ad_bd(t).")
    return path_schedule, type(path_schedule).__name__


def _expand_theta_to_batch(theta: torch.Tensor, *, batch: int) -> torch.Tensor:
    if theta.ndim == 1:
        theta = theta.unsqueeze(-1)
    if theta.ndim != 2:
        raise ValueError("theta must have shape [B] or [B, theta_dim].")
    if int(theta.shape[0]) == 1 and int(batch) > 1:
        theta = theta.expand(int(batch), int(theta.shape[1]))
    if int(theta.shape[0]) != int(batch):
        raise ValueError("x and theta batch sizes must match.")
    return theta


def _apply_matrix(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if a.ndim == 2:
        return x @ a.transpose(0, 1)
    if a.ndim == 3:
        if int(a.shape[0]) == 1 and int(x.shape[0]) > 1:
            a = a.expand(int(x.shape[0]), int(a.shape[1]), int(a.shape[2]))
        if int(a.shape[0]) != int(x.shape[0]):
            raise ValueError("matrix batch size must match x batch size.")
        return torch.bmm(a, x.unsqueeze(-1)).squeeze(-1)
    raise ValueError("matrix must have shape [D, D] or [B, D, D].")


def _trace_matrix_batch(a: torch.Tensor, *, batch: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if a.ndim == 2:
        return torch.trace(a).to(dtype=dtype, device=device).reshape(1).expand(int(batch))
    if a.ndim == 3:
        if int(a.shape[0]) == 1 and int(batch) > 1:
            a = a.expand(int(batch), int(a.shape[1]), int(a.shape[2]))
        return torch.diagonal(a, dim1=-2, dim2=-1).sum(dim=-1).to(dtype=dtype, device=device)
    raise ValueError("matrix must have shape [D, D] or [B, D, D].")


def _scalar_batch_to_matrix(scale: torch.Tensor, *, x_dim: int) -> torch.Tensor:
    if scale.ndim == 1:
        scale = scale.unsqueeze(-1)
    if scale.ndim != 2 or int(scale.shape[1]) != 1:
        raise ValueError("scalar affine output must have shape [B, 1].")
    eye = torch.eye(int(x_dim), dtype=scale.dtype, device=scale.device).reshape(1, int(x_dim), int(x_dim))
    return scale.reshape(int(scale.shape[0]), 1, 1) * eye


def _diag_batch_to_matrix(diag: torch.Tensor, *, x_dim: int) -> torch.Tensor:
    if diag.ndim == 1:
        diag = diag.reshape(1, -1)
    if diag.ndim != 2 or int(diag.shape[1]) != int(x_dim):
        raise ValueError("diagonal affine output must have shape [B, D].")
    return torch.diag_embed(diag)


def row_radius_normalize(x: torch.Tensor, radius: float, *, eps: float = 1e-12) -> torch.Tensor:
    """Normalize rows of ``x`` to the requested radius."""

    r = float(radius)
    if not math.isfinite(r) or r <= 0.0:
        raise ValueError("radius must be finite and positive.")
    norm = torch.linalg.norm(x, dim=1, keepdim=True).clamp_min(float(eps))
    return x * (r / norm)


def centered_radius_normalize(x: torch.Tensor, radius: float, *, eps: float = 1e-12) -> torch.Tensor:
    """Center each row across x dimensions, then normalize to ``radius``."""

    return row_radius_normalize(x - x.mean(dim=1, keepdim=True), radius, eps=eps)


class TranslationFlowSKLModel(nn.Module):
    """Translation endpoint model ``x_1 = x_0 + b(theta)``."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        velocity_family: str = "translation",
        radius: float = 1.0,
        hidden_dim: int = 128,
        depth: int = 3,
        path_schedule: str | GaussianAffinePathSchedule = "cosine",
    ) -> None:
        super().__init__()
        fam = _normalize_velocity_family(velocity_family)
        if fam not in TRANSLATION_FAMILIES:
            raise ValueError(f"TranslationFlowSKLModel does not support family {velocity_family!r}.")
        if int(theta_dim) < 1 or int(x_dim) < 1:
            raise ValueError("theta_dim and x_dim must be >= 1.")
        self.velocity_family = fam
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.radius = float(radius)
        self.set_path_schedule(path_schedule)
        self.mean_net = _make_mlp(
            in_dim=self.theta_dim,
            out_dim=self.x_dim,
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            final_gain=0.01,
        )

    def set_path_schedule(self, path_schedule: str | GaussianAffinePathSchedule) -> None:
        schedule, name = _resolve_path_schedule(path_schedule)
        self.path_schedule = schedule
        self.path_schedule_name = name

    def endpoint_mean(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        raw = self.mean_net(theta)
        if self.velocity_family == "translation_fixed_norm":
            return row_radius_normalize(raw, self.radius)
        if self.velocity_family == "translation_centered_fixed_norm":
            return centered_radius_normalize(raw, self.radius)
        return raw

    def forward(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del x
        t = _as_col_t(t, batch=int(theta.shape[0]))
        _, _, _, beta_dot = self.path_schedule.ab_ad_bd(t)
        return beta_dot * self.endpoint_mean(theta)

    def log_prob_normalized(self, x_norm: torch.Tensor, theta: torch.Tensor, *, solve_jitter: float = 1e-6) -> torch.Tensor:
        del solve_jitter
        if x_norm.ndim == 1:
            x_norm = x_norm.unsqueeze(0)
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        mu = self.endpoint_mean(theta)
        d = int(x_norm.shape[1])
        quad = torch.sum((x_norm - mu) ** 2, dim=1)
        return -0.5 * (quad + float(d) * math.log(2.0 * math.pi))


class ConditionalNonlinearXFlowMLP(nn.Module):
    """Unconstrained conditional velocity with reverse-ODE likelihood."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        divergence_estimator: str = "hutchinson",
        hutchinson_probes: int = 1,
    ) -> None:
        super().__init__()
        de = str(divergence_estimator).strip().lower()
        if de not in ("hutchinson", "exact"):
            raise ValueError("divergence_estimator must be one of: hutchinson, exact.")
        if int(hutchinson_probes) < 1:
            raise ValueError("hutchinson_probes must be >= 1.")
        self.velocity_family = "nonlinear"
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.divergence_estimator = de
        self.hutchinson_probes = int(hutchinson_probes)
        self.net = _make_mlp(
            in_dim=self.x_dim + 1 + self.theta_dim,
            out_dim=self.x_dim,
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            final_gain=0.0,
        )

    def forward(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        t = _as_col_t(t, batch=int(x.shape[0]))
        return self.net(torch.cat([x, t, theta], dim=1))

    def _trace_exact(self, x_req: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        trace = torch.zeros(x_req.shape[0], dtype=x_req.dtype, device=x_req.device)
        for j in range(self.x_dim):
            grad_j = torch.autograd.grad(
                v[:, j].sum(),
                x_req,
                create_graph=False,
                retain_graph=j < self.x_dim - 1,
            )[0]
            trace = trace + grad_j[:, j]
        return trace

    def _trace_hutchinson(self, x_req: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        acc = torch.zeros(x_req.shape[0], dtype=x_req.dtype, device=x_req.device)
        for p in range(self.hutchinson_probes):
            probe = torch.empty_like(x_req)
            probe.bernoulli_(0.5).mul_(2.0).sub_(1.0)
            dot = torch.sum(v * probe, dim=1)
            grad = torch.autograd.grad(
                dot.sum(),
                x_req,
                create_graph=False,
                retain_graph=p < self.hutchinson_probes - 1,
            )[0]
            acc = acc + torch.sum(grad * probe, dim=1)
        return acc / float(self.hutchinson_probes)

    def divergence(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        t = _as_col_t(t, batch=int(x.shape[0]))
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            v = self.forward(x_req, theta, t)
            if self.divergence_estimator == "exact":
                div = self._trace_exact(x_req, v)
            else:
                div = self._trace_hutchinson(x_req, v)
        return div.detach()

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
        if int(x_norm.shape[0]) != int(theta.shape[0]):
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
                x = x - dt * self.forward(x, theta, t)
        d = int(x.shape[1])
        base = -0.5 * (torch.sum(x**2, dim=1) + float(d) * math.log(2.0 * math.pi))
        return base - div_int


class _CenteredAffineFlowSKLBase(nn.Module):
    """Shared endpoint-mean and path-schedule utilities for centered affine SKL models."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        quadrature_steps: int = 64,
        path_schedule: str | GaussianAffinePathSchedule = "cosine",
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
            in_dim=self.theta_dim,
            out_dim=self.x_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.01,
        )
        self.set_path_schedule(path_schedule)

    def set_path_schedule(self, path_schedule: str | GaussianAffinePathSchedule) -> None:
        schedule, name = _resolve_path_schedule(path_schedule)
        self.path_schedule = schedule
        self.path_schedule_name = name

    def _beta_beta_dot(self, t: torch.Tensor, *, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = _as_col_t(t, batch=int(batch))
        _, beta, _, beta_dot = self.path_schedule.ab_ad_bd(t)
        return beta, beta_dot

    def b(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        return self.b_net(theta)

    def endpoint_mean(self, theta: torch.Tensor) -> torch.Tensor:
        return self.b(theta)

    def regularization_loss(self) -> torch.Tensor | None:
        return None

    def _full_gaussian_log_prob(
        self,
        x_norm: torch.Tensor,
        theta: torch.Tensor,
        *,
        solve_jitter: float,
        quadrature_steps: int | None,
    ) -> torch.Tensor:
        if x_norm.ndim == 1:
            x_norm = x_norm.unsqueeze(0)
        theta = _expand_theta_to_batch(theta, batch=int(x_norm.shape[0]))
        mu, cov = self.endpoint_mean_covariance(
            theta,
            solve_jitter=float(solve_jitter),
            quadrature_steps=quadrature_steps,
        )
        d = int(x_norm.shape[1])
        if cov.ndim == 2:
            cov = cov.reshape(1, d, d).expand(int(x_norm.shape[0]), d, d)
        eye = torch.eye(d, dtype=x_norm.dtype, device=x_norm.device).reshape(1, d, d)
        l = torch.linalg.cholesky(cov + float(solve_jitter) * eye)
        diff = x_norm - mu
        z = torch.cholesky_solve(diff.unsqueeze(-1), l).squeeze(-1)
        quad = torch.sum(diff * z, dim=1)
        diag = torch.diagonal(l, dim1=-2, dim2=-1)
        log_det = 2.0 * torch.sum(torch.log(torch.clamp(diag, min=1e-12)), dim=1)
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


class CenteredSharedAffineFlowSKLModel(_CenteredAffineFlowSKLBase):
    """Centered shared-affine velocity ``bdot b(theta) + A(t)(x - beta b(theta))``."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        quadrature_steps: int = 64,
        path_schedule: str | GaussianAffinePathSchedule = "cosine",
    ) -> None:
        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            quadrature_steps=quadrature_steps,
            path_schedule=path_schedule,
        )
        self.velocity_family = "shared_affine"
        self.a_net = _make_mlp(
            in_dim=1,
            out_dim=self.x_dim * self.x_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.01,
        )

    def A(self, t: torch.Tensor) -> torch.Tensor:
        t = _as_col_t(t)
        return self.a_net(t).reshape(int(t.shape[0]), self.x_dim, self.x_dim)

    def forward(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        theta = _expand_theta_to_batch(theta, batch=int(x.shape[0]))
        t = _as_col_t(t, batch=int(x.shape[0]))
        b = self.b(theta)
        beta, beta_dot = self._beta_beta_dot(t, batch=int(x.shape[0]))
        centered = x - beta * b
        return beta_dot * b + _apply_matrix(self.A(t), centered)

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
        d = int(self.x_dim)
        cov = torch.eye(d, dtype=theta.dtype, device=theta.device)
        dt = 1.0 / float(q)
        for k in range(q):
            tk = torch.full((1, 1), (float(k) + 0.5) / float(q), dtype=theta.dtype, device=theta.device)
            a = self.A(tk)
            if a.ndim == 3:
                a = a[0]
            cov = cov + dt * (a @ cov + cov @ a.transpose(0, 1))
            cov = 0.5 * (cov + cov.transpose(0, 1))
        eye = torch.eye(d, dtype=theta.dtype, device=theta.device)
        cov = 0.5 * (cov + cov.transpose(0, 1)) + float(solve_jitter) * eye
        return self.b(theta), cov

    def log_prob_normalized(
        self,
        x_norm: torch.Tensor,
        theta: torch.Tensor,
        *,
        solve_jitter: float = 1e-6,
        quadrature_steps: int | None = None,
    ) -> torch.Tensor:
        return self._full_gaussian_log_prob(
            x_norm,
            theta,
            solve_jitter=float(solve_jitter),
            quadrature_steps=quadrature_steps,
        )


class CenteredSharedAffineScalarFlowSKLModel(CenteredSharedAffineFlowSKLModel):
    """Centered shared-affine velocity with ``A(t) = a(t) I``."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        quadrature_steps: int = 64,
        path_schedule: str | GaussianAffinePathSchedule = "cosine",
    ) -> None:
        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            quadrature_steps=quadrature_steps,
            path_schedule=path_schedule,
        )
        self.velocity_family = "shared_affine_scalar"
        self.a_net = _make_mlp(
            in_dim=1,
            out_dim=1,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.01,
        )

    def A(self, t: torch.Tensor) -> torch.Tensor:
        t = _as_col_t(t)
        return _scalar_batch_to_matrix(self.a_net(t), x_dim=self.x_dim)


class CenteredSharedAffineDiagFlowSKLModel(CenteredSharedAffineFlowSKLModel):
    """Centered shared-affine velocity with ``A(t) = diag(a(t))``."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        quadrature_steps: int = 64,
        path_schedule: str | GaussianAffinePathSchedule = "cosine",
    ) -> None:
        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            quadrature_steps=quadrature_steps,
            path_schedule=path_schedule,
        )
        self.velocity_family = "shared_affine_diag"
        self.a_net = _make_mlp(
            in_dim=1,
            out_dim=self.x_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.01,
        )

    def A(self, t: torch.Tensor) -> torch.Tensor:
        t = _as_col_t(t)
        return _diag_batch_to_matrix(self.a_net(t), x_dim=self.x_dim)


class CenteredConditionAffineFlowSKLModel(_CenteredAffineFlowSKLBase):
    """Centered condition-specific affine velocity with full ``A(theta,t)``."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        quadrature_steps: int = 64,
        path_schedule: str | GaussianAffinePathSchedule = "cosine",
    ) -> None:
        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            quadrature_steps=quadrature_steps,
            path_schedule=path_schedule,
        )
        self.velocity_family = "condition_affine"
        self.a_net = _make_mlp(
            in_dim=self.theta_dim + 1,
            out_dim=self.x_dim * self.x_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.01,
        )

    def A(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        t = _as_col_t(t, batch=int(theta.shape[0]))
        return self.a_net(torch.cat([t, theta], dim=1)).reshape(int(theta.shape[0]), self.x_dim, self.x_dim)

    def forward(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        theta = _expand_theta_to_batch(theta, batch=int(x.shape[0]))
        t = _as_col_t(t, batch=int(x.shape[0]))
        b = self.b(theta)
        beta, beta_dot = self._beta_beta_dot(t, batch=int(x.shape[0]))
        centered = x - beta * b
        return beta_dot * b + _apply_matrix(self.A(theta, t), centered)

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
        cov = torch.eye(d, dtype=theta.dtype, device=theta.device).reshape(1, d, d).expand(batch, d, d).clone()
        dt = 1.0 / float(q)
        for k in range(q):
            tk = torch.full((batch, 1), (float(k) + 0.5) / float(q), dtype=theta.dtype, device=theta.device)
            a = self.A(theta, tk)
            cov = cov + dt * (torch.bmm(a, cov) + torch.bmm(cov, a.transpose(1, 2)))
            cov = 0.5 * (cov + cov.transpose(1, 2))
        eye = torch.eye(d, dtype=theta.dtype, device=theta.device).reshape(1, d, d)
        cov = 0.5 * (cov + cov.transpose(1, 2)) + float(solve_jitter) * eye
        return self.b(theta), cov

    def log_prob_normalized(
        self,
        x_norm: torch.Tensor,
        theta: torch.Tensor,
        *,
        solve_jitter: float = 1e-6,
        quadrature_steps: int | None = None,
    ) -> torch.Tensor:
        return self._full_gaussian_log_prob(
            x_norm,
            theta,
            solve_jitter=float(solve_jitter),
            quadrature_steps=quadrature_steps,
        )


class CenteredConditionAffineScalarFlowSKLModel(CenteredConditionAffineFlowSKLModel):
    """Centered condition-specific affine velocity with ``A(theta,t) = a(theta,t) I``."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        quadrature_steps: int = 64,
        path_schedule: str | GaussianAffinePathSchedule = "cosine",
    ) -> None:
        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            quadrature_steps=quadrature_steps,
            path_schedule=path_schedule,
        )
        self.velocity_family = "condition_affine_scalar"
        self.a_net = _make_mlp(
            in_dim=self.theta_dim + 1,
            out_dim=1,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.01,
        )

    def A(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        t = _as_col_t(t, batch=int(theta.shape[0]))
        return _scalar_batch_to_matrix(self.a_net(torch.cat([t, theta], dim=1)), x_dim=self.x_dim)


class CenteredConditionAffineDiagFlowSKLModel(CenteredConditionAffineFlowSKLModel):
    """Centered condition-specific affine velocity with ``A(theta,t) = diag(a(theta,t))``."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        quadrature_steps: int = 64,
        path_schedule: str | GaussianAffinePathSchedule = "cosine",
    ) -> None:
        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            quadrature_steps=quadrature_steps,
            path_schedule=path_schedule,
        )
        self.velocity_family = "condition_affine_diag"
        self.a_net = _make_mlp(
            in_dim=self.theta_dim + 1,
            out_dim=self.x_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.01,
        )

    def A(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        t = _as_col_t(t, batch=int(theta.shape[0]))
        return _diag_batch_to_matrix(self.a_net(torch.cat([t, theta], dim=1)), x_dim=self.x_dim)


class CenteredSharedAffineLowRankFlowSKLModel(CenteredSharedAffineFlowSKLModel):
    """Centered shared-affine velocity plus ``U h(theta,t,U^T centered_x)``."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        correction_rank: int,
        hidden_dim: int = 128,
        depth: int = 3,
        quadrature_steps: int = 64,
        path_schedule: str | GaussianAffinePathSchedule = "cosine",
        divergence_estimator: str = "hutchinson",
        hutchinson_probes: int = 1,
    ) -> None:
        if int(correction_rank) < 1:
            raise ValueError("correction_rank must be >= 1.")
        if int(correction_rank) > int(x_dim):
            raise ValueError("correction_rank must be <= x_dim.")
        de = str(divergence_estimator).strip().lower()
        if de not in ("hutchinson", "exact"):
            raise ValueError("divergence_estimator must be one of: hutchinson, exact.")
        if int(hutchinson_probes) < 1:
            raise ValueError("hutchinson_probes must be >= 1.")
        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            quadrature_steps=quadrature_steps,
            path_schedule=path_schedule,
        )
        self.velocity_family = "shared_affine_low_rank"
        self.correction_rank = int(correction_rank)
        self.divergence_estimator = de
        self.hutchinson_probes = int(hutchinson_probes)
        u_lin = nn.Linear(self.correction_rank, self.x_dim, bias=False)
        nn.init.orthogonal_(u_lin.weight)
        self.u_layer = parametrizations.orthogonal(u_lin, "weight", orthogonal_map="householder")
        self.h_net = _make_mlp(
            in_dim=self.correction_rank + 1 + self.theta_dim,
            out_dim=self.correction_rank,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.0,
        )

    @property
    def U(self) -> torch.Tensor:
        """Low-rank basis columns with shape ``[D, r]``."""

        return self.u_layer.weight

    def _centered_inputs(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        theta = _expand_theta_to_batch(theta, batch=int(x.shape[0]))
        t = _as_col_t(t, batch=int(x.shape[0]))
        b = self.b(theta)
        beta, beta_dot = self._beta_beta_dot(t, batch=int(x.shape[0]))
        centered = x - beta * b
        return theta, t, b, beta_dot, centered

    def forward(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        theta, t, b, beta_dot, centered = self._centered_inputs(x, theta, t)
        a_part = _apply_matrix(self.A(t), centered)
        u_mat = self.U
        z = centered @ u_mat
        h = self.h_net(torch.cat([z, t, theta], dim=1))
        return beta_dot * b + a_part + h @ u_mat.transpose(0, 1)

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
        bsz = int(z.shape[0])
        rank = int(self.correction_rank)
        acc = torch.zeros(bsz, dtype=z.dtype, device=z.device)
        for p in range(self.hutchinson_probes):
            probe = torch.empty(bsz, rank, dtype=z.dtype, device=z.device)
            probe.bernoulli_(0.5).mul_(2.0).sub_(1.0)
            dot = torch.sum(h * probe, dim=1)
            grad = torch.autograd.grad(
                dot.sum(),
                z,
                create_graph=False,
                retain_graph=p < self.hutchinson_probes - 1,
            )[0]
            acc = acc + torch.sum(grad * probe, dim=1)
        return acc / float(self.hutchinson_probes)

    def divergence(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        theta, t, _, _, centered = self._centered_inputs(x, theta, t)
        tr_a = _trace_matrix_batch(self.A(t), batch=int(x.shape[0]), dtype=x.dtype, device=x.device)
        u_mat = self.U
        with torch.enable_grad():
            z = (centered @ u_mat).detach().requires_grad_(True)
            h = self.h_net(torch.cat([z, t, theta], dim=1))
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
        theta = _expand_theta_to_batch(theta, batch=int(x_norm.shape[0]))
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
                x = x - dt * self.forward(x, theta, t)
        d = int(x.shape[1])
        base = -0.5 * (torch.sum(x**2, dim=1) + float(d) * math.log(2.0 * math.pi))
        return base - div_int


class CenteredSharedAffineLowRankScalarFlowSKLModel(CenteredSharedAffineLowRankFlowSKLModel):
    """Low-rank corrected centered velocity with shared base ``A(t) = a(t) I``."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        correction_rank: int,
        hidden_dim: int = 128,
        depth: int = 3,
        quadrature_steps: int = 64,
        path_schedule: str | GaussianAffinePathSchedule = "cosine",
        divergence_estimator: str = "hutchinson",
        hutchinson_probes: int = 1,
    ) -> None:
        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            correction_rank=correction_rank,
            hidden_dim=hidden_dim,
            depth=depth,
            quadrature_steps=quadrature_steps,
            path_schedule=path_schedule,
            divergence_estimator=divergence_estimator,
            hutchinson_probes=hutchinson_probes,
        )
        self.velocity_family = "shared_affine_low_rank_scalar"
        self.a_net = _make_mlp(
            in_dim=1,
            out_dim=1,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.01,
        )

    def A(self, t: torch.Tensor) -> torch.Tensor:
        t = _as_col_t(t)
        return _scalar_batch_to_matrix(self.a_net(t), x_dim=self.x_dim)


class CenteredSharedAffineLowRankDiagFlowSKLModel(CenteredSharedAffineLowRankFlowSKLModel):
    """Low-rank corrected centered velocity with shared base ``A(t) = diag(a(t))``."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        correction_rank: int,
        hidden_dim: int = 128,
        depth: int = 3,
        quadrature_steps: int = 64,
        path_schedule: str | GaussianAffinePathSchedule = "cosine",
        divergence_estimator: str = "hutchinson",
        hutchinson_probes: int = 1,
    ) -> None:
        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            correction_rank=correction_rank,
            hidden_dim=hidden_dim,
            depth=depth,
            quadrature_steps=quadrature_steps,
            path_schedule=path_schedule,
            divergence_estimator=divergence_estimator,
            hutchinson_probes=hutchinson_probes,
        )
        self.velocity_family = "shared_affine_low_rank_diag"
        self.a_net = _make_mlp(
            in_dim=1,
            out_dim=self.x_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.01,
        )

    def A(self, t: torch.Tensor) -> torch.Tensor:
        t = _as_col_t(t)
        return _diag_batch_to_matrix(self.a_net(t), x_dim=self.x_dim)


def build_flow_skl_model(
    *,
    velocity_family: str,
    theta_dim: int,
    x_dim: int,
    radius: float = 1.0,
    hidden_dim: int = 128,
    depth: int = 3,
    low_rank_dim: int = 4,
    quadrature_steps: int = 64,
    path_schedule: str | GaussianAffinePathSchedule = "cosine",
    divergence_estimator: str = "hutchinson",
    hutchinson_probes: int = 1,
) -> nn.Module:
    """Build a velocity-family model for flow-matching SKL estimation."""

    fam = _normalize_velocity_family(velocity_family)
    common = {
        "theta_dim": int(theta_dim),
        "x_dim": int(x_dim),
        "hidden_dim": int(hidden_dim),
        "depth": int(depth),
    }
    if fam in TRANSLATION_FAMILIES:
        return TranslationFlowSKLModel(
            velocity_family=fam,
            radius=float(radius),
            path_schedule=path_schedule,
            **common,
        )
    shared_affine_classes = {
        "shared_affine": CenteredSharedAffineFlowSKLModel,
        "shared_affine_scalar": CenteredSharedAffineScalarFlowSKLModel,
        "shared_affine_diag": CenteredSharedAffineDiagFlowSKLModel,
    }
    if fam in shared_affine_classes:
        return shared_affine_classes[fam](
            quadrature_steps=int(quadrature_steps),
            path_schedule=path_schedule,
            **common,
        )
    condition_affine_classes = {
        "condition_affine": CenteredConditionAffineFlowSKLModel,
        "condition_affine_scalar": CenteredConditionAffineScalarFlowSKLModel,
        "condition_affine_diag": CenteredConditionAffineDiagFlowSKLModel,
    }
    if fam in condition_affine_classes:
        return condition_affine_classes[fam](
            quadrature_steps=int(quadrature_steps),
            path_schedule=path_schedule,
            **common,
        )
    low_rank_affine_classes = {
        "shared_affine_low_rank": CenteredSharedAffineLowRankFlowSKLModel,
        "shared_affine_low_rank_scalar": CenteredSharedAffineLowRankScalarFlowSKLModel,
        "shared_affine_low_rank_diag": CenteredSharedAffineLowRankDiagFlowSKLModel,
    }
    if fam in low_rank_affine_classes:
        rank = resolve_lxf_low_rank_dim(int(low_rank_dim), int(x_dim), log_prefix="[flow-skl] ")
        return low_rank_affine_classes[fam](
            correction_rank=rank,
            quadrature_steps=int(quadrature_steps),
            path_schedule=path_schedule,
            divergence_estimator=str(divergence_estimator),
            hutchinson_probes=int(hutchinson_probes),
            **common,
        )
    if fam == "nonlinear":
        return ConditionalNonlinearXFlowMLP(
            divergence_estimator=str(divergence_estimator),
            hutchinson_probes=int(hutchinson_probes),
            **common,
        )
    raise AssertionError(f"Unhandled velocity family {fam!r}.")


def _normalization_from_train(x_train: np.ndarray, *, normalize_x: bool) -> tuple[np.ndarray, np.ndarray]:
    x = _as_2d_float64(x_train, name="x_train")
    if bool(normalize_x):
        mean = np.mean(x, axis=0, dtype=np.float64)
        std = np.maximum(np.std(x, axis=0, dtype=np.float64), 1e-6)
    else:
        mean = np.zeros(x.shape[1], dtype=np.float64)
        std = np.ones(x.shape[1], dtype=np.float64)
    return mean, std


def _adamw_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def train_flow_skl_model(
    *,
    model: nn.Module,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray | None,
    x_val: np.ndarray | None,
    device: torch.device,
    velocity_family: str | None = None,
    path_schedule: str | GaussianAffinePathSchedule = "cosine",
    normalize_x: bool = True,
    epochs: int = 1000,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    t_eps: float = 0.05,
    patience: int = 0,
    min_delta: float = 1e-4,
    max_grad_norm: float = 10.0,
    log_every: int = 50,
) -> dict[str, Any]:
    """Train a flow-SKL model and return metadata plus normalization stats."""

    fam = _normalize_velocity_family(velocity_family or getattr(model, "velocity_family", ""))
    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1.")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1.")
    if float(lr) <= 0.0:
        raise ValueError("lr must be > 0.")
    te = float(t_eps)
    if not (0.0 < te < 0.5):
        raise ValueError("t_eps must be in (0, 0.5).")

    th_tr = _as_2d_float64(theta_train, name="theta_train")
    x_tr = _as_2d_float64(x_train, name="x_train")
    if theta_val is None or x_val is None:
        th_va = th_tr
        x_va = x_tr
    else:
        th_va = _as_2d_float64(theta_val, name="theta_val")
        x_va = _as_2d_float64(x_val, name="x_val")
    if th_tr.shape[0] < 1 or x_tr.shape[0] < 1 or th_va.shape[0] < 1 or x_va.shape[0] < 1:
        raise ValueError("train and validation splits must be non-empty.")
    if th_tr.shape[0] != x_tr.shape[0] or th_va.shape[0] != x_va.shape[0]:
        raise ValueError("theta and x split lengths must match.")

    x_mean, x_std = _normalization_from_train(x_tr, normalize_x=bool(normalize_x))
    x_tr_n = (x_tr - x_mean) / x_std
    x_va_n = (x_va - x_mean) / x_std

    train_ds = TensorDataset(torch.from_numpy(th_tr.astype(np.float32)), torch.from_numpy(x_tr_n.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(th_va.astype(np.float32)), torch.from_numpy(x_va_n.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)

    model.to(device)
    schedule, schedule_name = _resolve_path_schedule(path_schedule)
    if hasattr(model, "set_path_schedule"):
        model.set_path_schedule(path_schedule)  # type: ignore[attr-defined]
    opt = torch.optim.AdamW(_adamw_parameters(model), lr=float(lr), weight_decay=float(weight_decay))

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    stopped_early = False
    stopped_epoch = int(epochs)
    n_clipped_steps = 0
    n_total_steps = 0

    is_translation = fam in TRANSLATION_FAMILIES
    for epoch in range(1, int(epochs) + 1):
        model.train()
        ep_losses: list[float] = []
        for tb, x1b in train_loader:
            tb = tb.to(device)
            x1b = x1b.to(device)
            if is_translation:
                if not hasattr(model, "endpoint_mean"):
                    raise TypeError("Translation family model must expose endpoint_mean(theta).")
                pred = model.endpoint_mean(tb)  # type: ignore[attr-defined]
                loss = torch.mean((pred - x1b) ** 2)
            else:
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
                grad_norm = torch.nn.utils.clip_grad_norm_(_adamw_parameters(model), float(max_grad_norm))
                if float(grad_norm) > float(max_grad_norm):
                    n_clipped_steps += 1
            opt.step()
            ep_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(ep_losses))
        train_losses.append(train_loss)

        model.eval()
        val_ep: list[float] = []
        with torch.no_grad():
            for tb, x1b in val_loader:
                tb = tb.to(device)
                x1b = x1b.to(device)
                if is_translation:
                    pred = model.endpoint_mean(tb)  # type: ignore[attr-defined]
                    val_ep.append(float(torch.mean((pred - x1b) ** 2).detach().cpu()))
                else:
                    bs = int(x1b.shape[0])
                    t_raw = torch.rand(bs, 1, device=device, dtype=x1b.dtype)
                    t = te + (1.0 - 2.0 * te) * t_raw
                    x0b = torch.randn_like(x1b)
                    a, bcoef, ad, bd = schedule.ab_ad_bd(t)
                    xt = a * x0b + bcoef * x1b
                    ut = ad * x0b + bd * x1b
                    val_ep.append(float(torch.mean((model(xt, tb, t) - ut) ** 2).detach().cpu()))
        val_loss = float(np.mean(val_ep))
        val_losses.append(val_loss)

        if val_loss < best_val - float(min_delta):
            best_val = val_loss
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % max(1, int(log_every)) == 0 or epoch == int(epochs):
            print(
                f"[flow-skl {fam} {epoch:4d}/{int(epochs)}] train={train_loss:.6f} "
                f"val={val_loss:.6f} best={best_val:.6f} best_epoch={best_epoch}",
                flush=True,
            )
        if int(patience) > 0 and patience_counter >= int(patience):
            stopped_early = True
            stopped_epoch = int(epoch)
            print(
                f"[flow-skl {fam} early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best={best_val:.6f} patience={int(patience)}",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "velocity_family": fam,
        "train_losses": np.asarray(train_losses, dtype=np.float64),
        "val_losses": np.asarray(val_losses, dtype=np.float64),
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
        "n_clipped_steps": int(n_clipped_steps),
        "n_total_steps": int(n_total_steps),
        "x_mean": x_mean.astype(np.float64),
        "x_std": x_std.astype(np.float64),
        "normalize_x": bool(normalize_x),
        "path_schedule": schedule_name,
    }


def _pairwise_squared_euclidean(mu: np.ndarray) -> np.ndarray:
    m = _as_2d_float64(mu, name="mu")
    diff = m[:, None, :] - m[None, :, :]
    out = np.sum(diff * diff, axis=2, dtype=np.float64)
    out = np.maximum(out, 0.0)
    np.fill_diagonal(out, 0.0)
    return out


def _shared_mahalanobis_sq(mu: np.ndarray, cov: np.ndarray, *, jitter: float = 1e-9) -> np.ndarray:
    m = _as_2d_float64(mu, name="mu")
    s = np.asarray(cov, dtype=np.float64)
    d = int(m.shape[1])
    if s.shape != (d, d):
        raise ValueError("shared covariance must have shape [D, D].")
    sj = 0.5 * (s + s.T) + float(jitter) * np.eye(d, dtype=np.float64)
    diff = (m[:, None, :] - m[None, :, :]).reshape(-1, d)
    sol = np.linalg.solve(sj, diff.T).T
    out = np.sum(diff * sol, axis=1).reshape(m.shape[0], m.shape[0])
    out = np.maximum(out, 0.0)
    np.fill_diagonal(out, 0.0)
    return 0.5 * (out + out.T)


def _gaussian_jeffreys_matrix(
    mu: np.ndarray,
    covariance: np.ndarray,
    *,
    jitter: float = 1e-9,
) -> np.ndarray:
    return 2.0 * symmetric_kl_gaussian_full_matrix(
        mu,
        covariance,
        jitter=float(jitter),
    )


def _translation_endpoint(
    model: nn.Module,
    theta_all: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    if not hasattr(model, "endpoint_mean"):
        raise TypeError(f"{type(model).__name__} does not expose endpoint_mean(theta).")
    theta_t = _as_torch_2d(theta_all, device=device)
    model.eval()
    with torch.no_grad():
        mu = model.endpoint_mean(theta_t)  # type: ignore[attr-defined]
    return mu.detach().cpu().numpy().astype(np.float64)


def _endpoint_gaussian(
    *,
    model: nn.Module,
    theta_all: np.ndarray,
    velocity_family: str,
    device: torch.device,
    solve_jitter: float,
    quadrature_steps: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    fam = _normalize_velocity_family(velocity_family)
    if fam in TRANSLATION_FAMILIES:
        mu = _translation_endpoint(model, theta_all, device=device)
        cov = np.eye(int(mu.shape[1]), dtype=np.float64)
        return mu, cov
    theta_t = _as_torch_2d(theta_all, device=device)
    model.eval()
    if fam in SHARED_GAUSSIAN_AFFINE_FAMILIES or fam in CONDITION_GAUSSIAN_AFFINE_FAMILIES:
        if not hasattr(model, "endpoint_mean_covariance"):
            raise TypeError(f"{type(model).__name__} does not expose endpoint_mean_covariance(theta).")
        with torch.no_grad():
            try:
                mu_t, cov_t = model.endpoint_mean_covariance(  # type: ignore[attr-defined]
                    theta_t,
                    solve_jitter=float(solve_jitter),
                    quadrature_steps=quadrature_steps,
                )
            except TypeError:
                mu_t, cov_t = model.endpoint_mean_covariance(  # type: ignore[attr-defined]
                    theta_t,
                    solve_jitter=float(solve_jitter),
                )
        mu = mu_t.detach().cpu().numpy().astype(np.float64)
        cov = cov_t.detach().cpu().numpy().astype(np.float64)
        d = int(mu.shape[1])
        n = int(mu.shape[0])
        if fam in SHARED_GAUSSIAN_AFFINE_FAMILIES:
            if cov.shape == (d, d):
                return mu, cov
            if cov.shape == (n, d, d):
                cov0 = cov[0]
                if np.allclose(cov, cov0.reshape(1, d, d), rtol=1e-5, atol=1e-7):
                    return mu, cov0
            raise ValueError(f"{fam} endpoint covariance must have shape [D, D].")
        if cov.shape == (n, d, d):
            return mu, cov
        if cov.shape == (d, d):
            return mu, np.broadcast_to(cov.reshape(1, d, d), (n, d, d)).copy()
        raise ValueError(f"{fam} endpoint covariance must have shape [N, D, D].")
    raise AssertionError(f"Unhandled Gaussian endpoint family {fam!r}.")


@torch.no_grad()
def sample_flow_endpoint(
    *,
    model: nn.Module,
    theta: np.ndarray,
    n_samples: int,
    device: torch.device,
    ode_steps: int = 64,
) -> torch.Tensor:
    """Sample ``x_1`` by pushing standard-normal samples through the learned ODE."""

    th = _as_torch_2d(theta, device=device)
    if int(th.shape[0]) != 1:
        raise ValueError("theta must contain exactly one endpoint row for sampling.")
    steps = int(ode_steps)
    if steps < 1:
        raise ValueError("ode_steps must be >= 1.")
    x_dim = int(getattr(model, "x_dim"))
    x = torch.randn(int(n_samples), x_dim, dtype=torch.float32, device=device)
    theta_b = th.expand(int(n_samples), int(th.shape[1]))
    dt = 1.0 / float(steps)
    model.eval()
    for s in range(steps):
        t = torch.full((int(n_samples), 1), (float(s) + 0.5) / float(steps), dtype=x.dtype, device=device)
        x = x + dt * model(x, theta_b, t)
    return x


def _log_prob_model(
    *,
    model: nn.Module,
    x: torch.Tensor,
    theta: np.ndarray,
    device: torch.device,
    ode_steps: int,
    batch_size: int,
    solve_jitter: float,
    quadrature_steps: int | None,
) -> np.ndarray:
    if not hasattr(model, "log_prob_normalized"):
        raise TypeError(f"{type(model).__name__} does not expose log_prob_normalized.")
    th = _as_torch_2d(theta, device=device)
    if int(th.shape[0]) != 1:
        raise ValueError("theta must contain one endpoint row.")
    outs: list[np.ndarray] = []
    n = int(x.shape[0])
    for start in range(0, n, int(batch_size)):
        xb = x[start : start + int(batch_size)]
        tb = th.expand(int(xb.shape[0]), int(th.shape[1]))
        logp = model.log_prob_normalized(  # type: ignore[attr-defined]
            xb,
            tb,
            solve_jitter=float(solve_jitter),
            quadrature_steps=quadrature_steps,
            ode_steps=int(ode_steps),
        )
        outs.append(logp.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(outs, axis=0)


def _estimate_mc_jeffreys(
    *,
    model: nn.Module,
    theta_all: np.ndarray,
    device: torch.device,
    mc_samples: int,
    ode_steps: int,
    batch_size: int,
    solve_jitter: float,
    quadrature_steps: int | None,
) -> np.ndarray:
    theta = _as_2d_float64(theta_all, name="theta_all")
    k = int(theta.shape[0])
    directed = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        xi = sample_flow_endpoint(
            model=model,
            theta=theta[i : i + 1],
            n_samples=int(mc_samples),
            device=device,
            ode_steps=int(ode_steps),
        )
        logp_i = _log_prob_model(
            model=model,
            x=xi,
            theta=theta[i : i + 1],
            device=device,
            ode_steps=int(ode_steps),
            batch_size=int(batch_size),
            solve_jitter=float(solve_jitter),
            quadrature_steps=quadrature_steps,
        )
        for j in range(k):
            if i == j:
                continue
            logp_j = _log_prob_model(
                model=model,
                x=xi,
                theta=theta[j : j + 1],
                device=device,
                ode_steps=int(ode_steps),
                batch_size=int(batch_size),
                solve_jitter=float(solve_jitter),
                quadrature_steps=quadrature_steps,
            )
            directed[i, j] = float(np.mean(logp_i - logp_j, dtype=np.float64))
    out = directed + directed.T
    out = np.maximum(out, 0.0)
    np.fill_diagonal(out, 0.0)
    return out


def estimate_model_symmetric_kl(
    *,
    model: nn.Module,
    theta_all: np.ndarray,
    device: torch.device,
    velocity_family: str | None = None,
    radius: float | None = None,
    mc_samples: int = 4096,
    ode_steps: int = 64,
    batch_size: int = 1024,
    solve_jitter: float = 1e-6,
    quadrature_steps: int | None = None,
    fisher_kind: str = "none",
    train_metadata: dict[str, Any] | None = None,
    normalization: dict[str, Any] | None = None,
) -> FlowSKLResult:
    """Estimate the endpoint symmetric KL matrix and canonical report metric."""

    fam = _normalize_velocity_family(velocity_family or getattr(model, "velocity_family", ""))
    theta = _as_2d_float64(theta_all, name="theta_all")
    model.to(device)
    model.eval()

    endpoint_mean: np.ndarray | None = None
    endpoint_covariance: np.ndarray | None = None
    if fam in GAUSSIAN_ENDPOINT_FAMILIES:
        endpoint_mean, endpoint_covariance = _endpoint_gaussian(
            model=model,
            theta_all=theta,
            velocity_family=fam,
            device=device,
            solve_jitter=float(solve_jitter),
            quadrature_steps=quadrature_steps,
        )
        if fam in TRANSLATION_FAMILIES:
            skl = _pairwise_squared_euclidean(endpoint_mean)
            if fam == "translation":
                canonical = skl.copy()
                metric_name = "squared_euclidean"
            elif fam == "translation_fixed_norm":
                r = float(radius if radius is not None else getattr(model, "radius", 1.0))
                canonical = skl / (2.0 * r * r)
                metric_name = "cosine"
            else:
                r = float(radius if radius is not None else getattr(model, "radius", 1.0))
                canonical = skl / (2.0 * r * r)
                metric_name = "correlation"
        elif fam in SHARED_GAUSSIAN_AFFINE_FAMILIES:
            skl = _shared_mahalanobis_sq(endpoint_mean, endpoint_covariance, jitter=float(solve_jitter))
            metric_name = "mahalanobis_sq"
            canonical = skl.copy()
        else:
            skl = _gaussian_jeffreys_matrix(
                endpoint_mean,
                endpoint_covariance,
                jitter=float(solve_jitter),
            )
            canonical = skl.copy()
            metric_name = "gaussian_symmetric_kl"
    else:
        skl = _estimate_mc_jeffreys(
            model=model,
            theta_all=theta,
            device=device,
            mc_samples=int(mc_samples),
            ode_steps=int(ode_steps),
            batch_size=int(batch_size),
            solve_jitter=float(solve_jitter),
            quadrature_steps=quadrature_steps,
        )
        canonical = skl.copy()
        metric_name = "model_symmetric_kl_mc"

    fisher_mode = str(fisher_kind).strip().lower()
    if fisher_mode not in ("none", "full", "linear", "both"):
        raise ValueError("fisher_kind must be one of: none, full, linear, both.")
    fisher_mid: np.ndarray | None = None
    fisher_full: np.ndarray | None = None
    fisher_linear: np.ndarray | None = None
    if fisher_mode in ("full", "both"):
        fd = estimate_scalar_fisher_from_skl(theta, skl)
        fisher_mid = fd["theta_midpoints"]
        fisher_full = fd["fisher"]
    if fisher_mode in ("linear", "both"):
        if fam in MC_ENDPOINT_FAMILIES:
            raise ValueError("linear Fisher is unavailable for nonlinear endpoint families.")
        fd = estimate_scalar_fisher_from_skl(theta, canonical)
        fisher_mid = fd["theta_midpoints"]
        fisher_linear = fd["fisher"]

    return FlowSKLResult(
        symmetric_kl_matrix=skl.astype(np.float64, copy=False),
        canonical_metric_matrix=canonical.astype(np.float64, copy=False),
        canonical_metric_name=metric_name,
        endpoint_mean=None if endpoint_mean is None else endpoint_mean.astype(np.float64, copy=False),
        endpoint_covariance=None
        if endpoint_covariance is None
        else endpoint_covariance.astype(np.float64, copy=False),
        fisher_theta_midpoints=fisher_mid,
        fisher_full=fisher_full,
        fisher_linear=fisher_linear,
        train_metadata={} if train_metadata is None else dict(train_metadata),
        normalization={} if normalization is None else dict(normalization),
    )


def estimate_scalar_fisher_from_skl(theta_all: np.ndarray, symmetric_kl_matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Estimate scalar Fisher by adjacent finite differences of DSKL."""

    theta = _as_2d_float64(theta_all, name="theta_all")
    if int(theta.shape[1]) != 1:
        raise ValueError("Scalar Fisher requires theta_all with one column.")
    skl = np.asarray(symmetric_kl_matrix, dtype=np.float64)
    n = int(theta.shape[0])
    if skl.shape != (n, n):
        raise ValueError(f"symmetric_kl_matrix must have shape ({n}, {n}).")
    order = np.argsort(theta[:, 0], kind="mergesort")
    theta_s = theta[order, 0]
    skl_s = skl[np.ix_(order, order)]
    if n < 2:
        raise ValueError("At least two theta points are required for adjacent finite differences.")
    dtheta = np.diff(theta_s)
    if np.any(dtheta <= 0.0):
        raise ValueError("theta_all must contain strictly increasing unique scalar values after sorting.")
    fisher = np.array(
        [float(skl_s[i, i + 1]) / float(dtheta[i] ** 2) for i in range(n - 1)],
        dtype=np.float64,
    )
    mid = 0.5 * (theta_s[:-1] + theta_s[1:])
    return {
        "theta_midpoints": mid.reshape(-1, 1).astype(np.float64),
        "theta_left": theta_s[:-1].reshape(-1, 1).astype(np.float64),
        "theta_right": theta_s[1:].reshape(-1, 1).astype(np.float64),
        "fisher": fisher,
    }


def flow_skl_result_to_npz_dict(result: FlowSKLResult) -> dict[str, Any]:
    """Convert a result dataclass to ``np.savez`` compatible fields."""

    out: dict[str, Any] = {
        "symmetric_kl_matrix": result.symmetric_kl_matrix,
        "canonical_metric_matrix": result.canonical_metric_matrix,
        "canonical_metric_name": np.asarray([result.canonical_metric_name], dtype=object),
    }
    if result.endpoint_mean is not None:
        out["endpoint_mean"] = result.endpoint_mean
    if result.endpoint_covariance is not None:
        out["endpoint_covariance"] = result.endpoint_covariance
    if result.fisher_theta_midpoints is not None:
        out["fisher_theta_midpoints"] = result.fisher_theta_midpoints
    if result.fisher_full is not None:
        out["fisher_full"] = result.fisher_full
    if result.fisher_linear is not None:
        out["fisher_linear"] = result.fisher_linear
    for key in ("x_mean", "x_std"):
        if key in result.normalization:
            out[key] = np.asarray(result.normalization[key], dtype=np.float64)
    return out
