"""Flow-matching model-likelihood metrics for symmetric KL.

This module uses the Jeffreys convention
``DSKL(p, q) = KL(p || q) + KL(q || p)``.  With this convention,
adjacent ``DSKL(theta, theta + dtheta) / dtheta**2`` estimates the
scalar Fisher information directly.
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
from fisher.model_weight_ema import scalar_val_ema_update


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

LOW_RANK_AFFINE_FAMILIES = {
    "shared_affine_low_rank",
    "shared_affine_low_rank_scalar",
    "shared_affine_low_rank_diag",
}

MC_ENDPOINT_FAMILIES = LOW_RANK_AFFINE_FAMILIES | {"nonlinear"}


@dataclass
class FlowSKLResult:
    """Metric bundle returned by ``estimate_model_symmetric_kl``."""

    symmetric_kl_matrix: np.ndarray
    canonical_metric_matrix: np.ndarray
    canonical_metric_name: str
    fisher_theta_midpoints: np.ndarray | None = None
    fisher_full: np.ndarray | None = None
    fisher_linear: np.ndarray | None = None
    train_metadata: dict[str, Any] = field(default_factory=dict)


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


class _FiLMResidualBlock(nn.Module):
    """Residual hidden block modulated by a condition vector."""

    def __init__(self, *, hidden_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.film = nn.Linear(int(cond_dim), 2 * int(hidden_dim))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.branch = _make_mlp(
            in_dim=int(hidden_dim),
            out_dim=int(hidden_dim),
            hidden_dim=int(hidden_dim),
            depth=1,
            final_gain=0.01,
        )

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(cond).chunk(2, dim=1)
        y = (1.0 + gamma) * self.norm(h) + beta
        return h + self.branch(y)


class _SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for scalar time columns."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if int(dim) < 1:
            raise ValueError("dim must be >= 1.")
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = _as_col_t(t)
        half = self.dim // 2
        if half == 0:
            return t
        freqs = torch.exp(
            torch.arange(half, dtype=t.dtype, device=t.device)
            * (-math.log(10000.0) / max(1, half - 1))
        )
        angles = t * freqs.reshape(1, half)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros(int(t.shape[0]), 1, dtype=t.dtype, device=t.device)], dim=1)
        return emb


class _ConditionedFiLMNet(nn.Module):
    """FiLM residual network conditioned on sinusoidal time and linear theta embeddings."""

    def __init__(
        self,
        *,
        trunk_dim: int,
        theta_dim: int,
        out_dim: int,
        hidden_dim: int,
        depth: int,
        final_gain: float = 0.01,
    ) -> None:
        super().__init__()
        if int(trunk_dim) < 1 or int(theta_dim) < 1 or int(out_dim) < 1 or int(hidden_dim) < 1:
            raise ValueError("trunk_dim, theta_dim, out_dim, and hidden_dim must be >= 1.")
        if int(depth) < 1:
            raise ValueError("depth must be >= 1.")
        self.network_architecture = "film"
        self.trunk_dim = int(trunk_dim)
        self.theta_dim = int(theta_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)

        self.trunk_proj = nn.Linear(self.trunk_dim, self.hidden_dim)
        nn.init.xavier_uniform_(self.trunk_proj.weight, gain=float(nn.init.calculate_gain("relu")))
        nn.init.zeros_(self.trunk_proj.bias)
        self.activation = nn.SiLU()
        self.time_embedding = _SinusoidalTimeEmbedding(self.hidden_dim)
        self.theta_embedding = nn.Linear(self.theta_dim, self.hidden_dim)
        nn.init.xavier_uniform_(self.theta_embedding.weight, gain=1.0)
        nn.init.zeros_(self.theta_embedding.bias)
        self.condition_mlp = _make_mlp(
            in_dim=2 * self.hidden_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            depth=1,
            final_gain=1.0,
        )
        self.blocks = nn.ModuleList(
            [_FiLMResidualBlock(hidden_dim=self.hidden_dim, cond_dim=self.hidden_dim) for _ in range(self.depth)]
        )
        self.out = nn.Linear(self.hidden_dim, self.out_dim)
        nn.init.xavier_uniform_(self.out.weight, gain=float(final_gain))
        nn.init.zeros_(self.out.bias)

    def forward(self, trunk: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if trunk.ndim != 2 or int(trunk.shape[1]) != self.trunk_dim:
            raise ValueError(f"FiLM trunk input must have shape [B, {self.trunk_dim}].")
        theta = _expand_theta_to_batch(theta, batch=int(trunk.shape[0]))
        if int(theta.shape[1]) != self.theta_dim:
            raise ValueError(f"theta must have {self.theta_dim} features.")
        t = _as_col_t(t, batch=int(trunk.shape[0]))
        h = self.activation(self.trunk_proj(trunk))
        cond = self.condition_mlp(torch.cat([self.time_embedding(t), self.theta_embedding(theta)], dim=1))
        for block in self.blocks:
            h = block(h, cond)
        return self.out(h)


def _make_film_net(
    *,
    trunk_dim: int,
    theta_dim: int,
    out_dim: int,
    hidden_dim: int,
    depth: int,
    final_gain: float = 0.01,
) -> _ConditionedFiLMNet:
    return _ConditionedFiLMNet(
        trunk_dim=int(trunk_dim),
        theta_dim=int(theta_dim),
        out_dim=int(out_dim),
        hidden_dim=int(hidden_dim),
        depth=int(depth),
        final_gain=float(final_gain),
    )


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


def _make_flow_matching_affine_path(path_schedule: str | GaussianAffinePathSchedule) -> tuple[Any, str]:
    """Build a ``flow_matching`` affine path matching the local schedule names."""

    if isinstance(path_schedule, str):
        key = str(path_schedule).strip().lower()
    else:
        _, schedule_name = _resolve_path_schedule(path_schedule)
        low = schedule_name.strip().lower()
        if "linear" in low:
            key = "linear"
        elif "cosine" in low:
            key = "cosine"
        else:
            raise TypeError("flow_matching AffineProbPath supports only linear/straight and cosine schedules.")

    try:
        from flow_matching.path import AffineProbPath
        from flow_matching.path.scheduler import CondOTScheduler, CosineScheduler
    except ImportError as e:
        raise ImportError(
            "Flow-SKL training/evaluation requires the `flow_matching` package. "
            "Install it in the geo_diffusion environment."
        ) from e

    if key in ("linear", "straight"):
        return AffineProbPath(scheduler=CondOTScheduler()), "linear"
    if key in ("cosine", "cos"):
        return AffineProbPath(scheduler=CosineScheduler()), "cosine"
    raise ValueError(f"Unknown path schedule: {path_schedule!r}; use linear/straight or cosine/cos.")


def _make_flow_ode_solver(model: nn.Module) -> Any:
    try:
        from flow_matching.solver.ode_solver import ODESolver
    except ImportError as e:
        raise ImportError(
            "Flow-SKL endpoint sampling and likelihood require the `flow_matching` package. "
            "Install it in the geo_diffusion environment."
        ) from e
    return ODESolver(velocity_model=_ThetaCondVelocityAdapter(model))


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


def _resolve_divergence_controls(
    divergence_estimator: str = "exact",
    hutchinson_probes: int = 1,
) -> tuple[str, int]:
    de = str(divergence_estimator).strip().lower()
    if de not in ("hutchinson", "exact"):
        raise ValueError("divergence_estimator must be one of: hutchinson, exact.")
    probes = int(hutchinson_probes)
    if probes < 1:
        raise ValueError("hutchinson_probes must be >= 1.")
    return de, probes


def _set_divergence_controls(
    model: nn.Module,
    *,
    divergence_estimator: str = "exact",
    hutchinson_probes: int = 1,
) -> None:
    de, probes = _resolve_divergence_controls(
        divergence_estimator=divergence_estimator,
        hutchinson_probes=hutchinson_probes,
    )
    model.divergence_estimator = de  # type: ignore[attr-defined]
    model.hutchinson_probes = probes  # type: ignore[attr-defined]


class _ThetaCondVelocityAdapter(nn.Module):
    """Adapt ``model(x, theta, t_col)`` to the ``flow_matching`` ODE API."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras: Any) -> torch.Tensor:
        if "theta_cond" not in extras:
            raise ValueError("theta_cond must be provided in model_extras.")
        theta = extras["theta_cond"]
        if not torch.is_tensor(theta):
            theta = torch.as_tensor(theta, dtype=x.dtype, device=x.device)
        else:
            theta = theta.to(device=x.device, dtype=x.dtype)
        theta_b = _expand_theta_to_batch(theta, batch=int(x.shape[0]))
        t_col = _as_col_t(t.to(device=x.device, dtype=x.dtype), batch=int(x.shape[0]))
        return self.model(x, theta_b, t_col) + 0.0 * x


def _standard_normal_log_prob(x: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(int(x.shape[0]), -1)
    return -0.5 * (torch.sum(flat * flat, dim=1) + float(flat.shape[1]) * math.log(2.0 * math.pi))


def flow_endpoint_log_prob(
    model: nn.Module,
    x_norm: torch.Tensor,
    theta: torch.Tensor,
    *,
    solve_jitter: float = 1e-6,
    quadrature_steps: int | None = None,
    ode_steps: int = 32,
    ode_method: str = "midpoint",
) -> torch.Tensor:
    """Compute endpoint log probability with ``flow_matching`` ODE likelihood."""

    del solve_jitter, quadrature_steps
    if x_norm.ndim == 1:
        x_norm = x_norm.unsqueeze(0)
    steps = int(ode_steps)
    if steps < 1:
        raise ValueError("ode_steps must be >= 1.")
    if not str(ode_method).strip():
        raise ValueError("ode_method must be non-empty.")
    x_eval = x_norm.detach()
    theta = theta.to(device=x_eval.device, dtype=x_eval.dtype)
    theta_b = _expand_theta_to_batch(theta, batch=int(x_eval.shape[0]))
    de, probes = _resolve_divergence_controls(
        divergence_estimator=str(getattr(model, "divergence_estimator", "exact")),
        hutchinson_probes=int(getattr(model, "hutchinson_probes", 1)),
    )
    exact = de == "exact"
    repeats = 1 if exact else probes
    time_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=x_eval.dtype, device=x_eval.device)
    solver = _make_flow_ode_solver(model)
    logps: list[torch.Tensor] = []
    for _ in range(repeats):
        _, logp = solver.compute_likelihood(
            x_1=x_eval,
            log_p0=_standard_normal_log_prob,
            step_size=None,
            method=str(ode_method),
            time_grid=time_grid,
            exact_divergence=exact,
            enable_grad=False,
            theta_cond=theta_b,
        )
        logps.append(logp)
    if len(logps) == 1:
        return logps[0]
    return torch.stack(logps, dim=0).mean(dim=0)


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
        divergence_estimator: str = "exact",
        hutchinson_probes: int = 1,
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
        _set_divergence_controls(
            self,
            divergence_estimator=divergence_estimator,
            hutchinson_probes=int(hutchinson_probes),
        )
        self.set_path_schedule(path_schedule)
        self.network_architecture = "film"
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

    def log_prob_normalized(
        self,
        x_norm: torch.Tensor,
        theta: torch.Tensor,
        *,
        solve_jitter: float = 1e-6,
        quadrature_steps: int | None = None,
        ode_steps: int = 32,
        ode_method: str = "midpoint",
    ) -> torch.Tensor:
        return flow_endpoint_log_prob(
            self,
            x_norm,
            theta,
            solve_jitter=float(solve_jitter),
            quadrature_steps=quadrature_steps,
            ode_steps=int(ode_steps),
            ode_method=str(ode_method),
        )


class ConditionalNonlinearXFlowFiLM(nn.Module):
    """Unconstrained FiLM conditional velocity with reverse-ODE likelihood."""

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
        self.velocity_family = "nonlinear"
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        _set_divergence_controls(
            self,
            divergence_estimator=divergence_estimator,
            hutchinson_probes=int(hutchinson_probes),
        )
        self.network_architecture = "film"
        self.net = _make_film_net(
            trunk_dim=self.x_dim,
            theta_dim=self.theta_dim,
            out_dim=self.x_dim,
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            final_gain=0.0,
        )

    def forward(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        t = _as_col_t(t, batch=int(x.shape[0]))
        return self.net(x, theta, t)

    def log_prob_normalized(
        self,
        x_norm: torch.Tensor,
        theta: torch.Tensor,
        *,
        solve_jitter: float = 1e-6,
        quadrature_steps: int | None = None,
        ode_steps: int = 32,
        ode_method: str = "midpoint",
    ) -> torch.Tensor:
        return flow_endpoint_log_prob(
            self,
            x_norm,
            theta,
            solve_jitter=float(solve_jitter),
            quadrature_steps=quadrature_steps,
            ode_steps=int(ode_steps),
            ode_method=str(ode_method),
        )


ConditionalNonlinearXFlowMLP = ConditionalNonlinearXFlowFiLM


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
        divergence_estimator: str = "exact",
        hutchinson_probes: int = 1,
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
        self.network_architecture = "film"
        _set_divergence_controls(
            self,
            divergence_estimator=divergence_estimator,
            hutchinson_probes=int(hutchinson_probes),
        )
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

class CenteredSharedAffineFlowSKLModel(_CenteredAffineFlowSKLBase):
    """Centered shared-affine velocity with symmetric shared ``A(t)``."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        quadrature_steps: int = 64,
        path_schedule: str | GaussianAffinePathSchedule = "cosine",
        divergence_estimator: str = "exact",
        hutchinson_probes: int = 1,
        a_diag_jitter: float = 1e-3,
    ) -> None:
        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            quadrature_steps=quadrature_steps,
            path_schedule=path_schedule,
            divergence_estimator=divergence_estimator,
            hutchinson_probes=int(hutchinson_probes),
        )
        self.velocity_family = "shared_affine"
        if not math.isfinite(float(a_diag_jitter)):
            raise ValueError("a_diag_jitter must be finite.")
        if float(a_diag_jitter) < 0.0:
            raise ValueError("a_diag_jitter must be nonnegative.")
        self.a_diag_jitter = float(a_diag_jitter)
        self.a_net = _make_mlp(
            in_dim=1,
            out_dim=self.x_dim * self.x_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.01,
        )

    def _add_a_diag_jitter(self, a: torch.Tensor) -> torch.Tensor:
        if self.a_diag_jitter == 0.0:
            return a
        eye = torch.eye(self.x_dim, dtype=a.dtype, device=a.device).reshape(1, self.x_dim, self.x_dim)
        return a + self.a_diag_jitter * eye

    def A(self, t: torch.Tensor) -> torch.Tensor:
        t = _as_col_t(t)
        raw = self.a_net(t).reshape(int(t.shape[0]), self.x_dim, self.x_dim)
        return self._add_a_diag_jitter(0.5 * (raw + raw.transpose(-1, -2)))

    def forward(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        theta = _expand_theta_to_batch(theta, batch=int(x.shape[0]))
        t = _as_col_t(t, batch=int(x.shape[0]))
        b = self.b(theta)
        beta, beta_dot = self._beta_beta_dot(t, batch=int(x.shape[0]))
        centered = x - beta * b
        return beta_dot * b + _apply_matrix(self.A(t), centered)

    def log_prob_normalized(
        self,
        x_norm: torch.Tensor,
        theta: torch.Tensor,
        *,
        solve_jitter: float = 1e-6,
        quadrature_steps: int | None = None,
        ode_steps: int = 32,
        ode_method: str = "midpoint",
    ) -> torch.Tensor:
        return flow_endpoint_log_prob(
            self,
            x_norm,
            theta,
            solve_jitter=float(solve_jitter),
            quadrature_steps=quadrature_steps,
            ode_steps=int(ode_steps),
            ode_method=str(ode_method),
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
        divergence_estimator: str = "exact",
        hutchinson_probes: int = 1,
        a_diag_jitter: float = 1e-3,
    ) -> None:
        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            quadrature_steps=quadrature_steps,
            path_schedule=path_schedule,
            divergence_estimator=divergence_estimator,
            hutchinson_probes=int(hutchinson_probes),
            a_diag_jitter=float(a_diag_jitter),
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
        return self._add_a_diag_jitter(_scalar_batch_to_matrix(self.a_net(t), x_dim=self.x_dim))


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
        divergence_estimator: str = "exact",
        hutchinson_probes: int = 1,
        a_diag_jitter: float = 1e-3,
    ) -> None:
        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            quadrature_steps=quadrature_steps,
            path_schedule=path_schedule,
            divergence_estimator=divergence_estimator,
            hutchinson_probes=int(hutchinson_probes),
            a_diag_jitter=float(a_diag_jitter),
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
        return self._add_a_diag_jitter(_diag_batch_to_matrix(self.a_net(t), x_dim=self.x_dim))


class CenteredConditionAffineFlowSKLModel(_CenteredAffineFlowSKLBase):
    """Centered condition-specific affine velocity with symmetric ``A(theta,t)``."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        quadrature_steps: int = 64,
        path_schedule: str | GaussianAffinePathSchedule = "cosine",
        divergence_estimator: str = "exact",
        hutchinson_probes: int = 1,
    ) -> None:
        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            quadrature_steps=quadrature_steps,
            path_schedule=path_schedule,
            divergence_estimator=divergence_estimator,
            hutchinson_probes=int(hutchinson_probes),
        )
        self.velocity_family = "condition_affine"
        self.a_net = _make_mlp(
            in_dim=1 + self.theta_dim,
            out_dim=self.x_dim * self.x_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.01,
        )

    def A(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        t = _as_col_t(t, batch=int(theta.shape[0]))
        raw = self.a_net(torch.cat([t, theta], dim=1)).reshape(int(theta.shape[0]), self.x_dim, self.x_dim)
        return 0.5 * (raw + raw.transpose(-1, -2))

    def forward(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        theta = _expand_theta_to_batch(theta, batch=int(x.shape[0]))
        t = _as_col_t(t, batch=int(x.shape[0]))
        b = self.b(theta)
        beta, beta_dot = self._beta_beta_dot(t, batch=int(x.shape[0]))
        centered = x - beta * b
        return beta_dot * b + _apply_matrix(self.A(theta, t), centered)

    def log_prob_normalized(
        self,
        x_norm: torch.Tensor,
        theta: torch.Tensor,
        *,
        solve_jitter: float = 1e-6,
        quadrature_steps: int | None = None,
        ode_steps: int = 32,
        ode_method: str = "midpoint",
    ) -> torch.Tensor:
        return flow_endpoint_log_prob(
            self,
            x_norm,
            theta,
            solve_jitter=float(solve_jitter),
            quadrature_steps=quadrature_steps,
            ode_steps=int(ode_steps),
            ode_method=str(ode_method),
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
        divergence_estimator: str = "exact",
        hutchinson_probes: int = 1,
    ) -> None:
        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            quadrature_steps=quadrature_steps,
            path_schedule=path_schedule,
            divergence_estimator=divergence_estimator,
            hutchinson_probes=int(hutchinson_probes),
        )
        self.velocity_family = "condition_affine_scalar"
        self.a_net = _make_mlp(
            in_dim=1 + self.theta_dim,
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
        divergence_estimator: str = "exact",
        hutchinson_probes: int = 1,
    ) -> None:
        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            quadrature_steps=quadrature_steps,
            path_schedule=path_schedule,
            divergence_estimator=divergence_estimator,
            hutchinson_probes=int(hutchinson_probes),
        )
        self.velocity_family = "condition_affine_diag"
        self.a_net = _make_mlp(
            in_dim=1 + self.theta_dim,
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
        a_diag_jitter: float = 1e-3,
        low_rank_basis: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        if int(correction_rank) < 1:
            raise ValueError("correction_rank must be >= 1.")
        if int(correction_rank) > int(x_dim):
            raise ValueError("correction_rank must be <= x_dim.")
        de, probes = _resolve_divergence_controls(
            divergence_estimator=divergence_estimator,
            hutchinson_probes=int(hutchinson_probes),
        )
        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            quadrature_steps=quadrature_steps,
            path_schedule=path_schedule,
            divergence_estimator=de,
            hutchinson_probes=probes,
            a_diag_jitter=float(a_diag_jitter),
        )
        self.velocity_family = "shared_affine_low_rank"
        self.correction_rank = int(correction_rank)
        if low_rank_basis is None:
            self.low_rank_basis_mode = "learned"
            u_lin = nn.Linear(self.correction_rank, self.x_dim, bias=False)
            nn.init.orthogonal_(u_lin.weight)
            self.u_layer = parametrizations.orthogonal(u_lin, "weight", orthogonal_map="householder")
        else:
            basis = torch.as_tensor(low_rank_basis, dtype=torch.float32)
            if basis.ndim != 2 or tuple(basis.shape) != (self.x_dim, self.correction_rank):
                raise ValueError(
                    "low_rank_basis must have shape "
                    f"({self.x_dim}, {self.correction_rank}); got {tuple(basis.shape)}."
                )
            if not torch.isfinite(basis).all():
                raise ValueError("low_rank_basis must contain only finite values.")
            gram = basis.T @ basis
            eye = torch.eye(self.correction_rank, dtype=basis.dtype)
            if not torch.allclose(gram, eye, rtol=1e-4, atol=1e-5):
                raise ValueError("low_rank_basis columns must be orthonormal.")
            self.low_rank_basis_mode = "fixed"
            self.register_buffer("fixed_u", basis.contiguous())
        self.h_net = _make_film_net(
            trunk_dim=self.correction_rank,
            theta_dim=self.theta_dim,
            out_dim=self.correction_rank,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.0,
        )

    @property
    def U(self) -> torch.Tensor:
        """Low-rank basis columns with shape ``[D, r]``."""

        if self.low_rank_basis_mode == "fixed":
            return self.fixed_u
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
        if isinstance(self.h_net, _ConditionedFiLMNet):
            h = self.h_net(z, theta, t)
        else:
            h = self.h_net(torch.cat([z, t, theta], dim=1))
        return beta_dot * b + a_part + h @ u_mat.transpose(0, 1)

    def log_prob_normalized(
        self,
        x_norm: torch.Tensor,
        theta: torch.Tensor,
        *,
        solve_jitter: float = 1e-6,
        quadrature_steps: int | None = None,
        ode_steps: int = 32,
        ode_method: str = "midpoint",
    ) -> torch.Tensor:
        return flow_endpoint_log_prob(
            self,
            x_norm,
            theta,
            solve_jitter=float(solve_jitter),
            quadrature_steps=quadrature_steps,
            ode_steps=int(ode_steps),
            ode_method=str(ode_method),
        )


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
        a_diag_jitter: float = 1e-3,
        low_rank_basis: np.ndarray | torch.Tensor | None = None,
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
            a_diag_jitter=float(a_diag_jitter),
            low_rank_basis=low_rank_basis,
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
        return self._add_a_diag_jitter(_scalar_batch_to_matrix(self.a_net(t), x_dim=self.x_dim))


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
        a_diag_jitter: float = 1e-3,
        low_rank_basis: np.ndarray | torch.Tensor | None = None,
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
            a_diag_jitter=float(a_diag_jitter),
            low_rank_basis=low_rank_basis,
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
        return self._add_a_diag_jitter(_diag_batch_to_matrix(self.a_net(t), x_dim=self.x_dim))


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
    shared_affine_a_diag_jitter: float = 1e-3,
    low_rank_basis: np.ndarray | torch.Tensor | None = None,
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
            divergence_estimator=str(divergence_estimator),
            hutchinson_probes=int(hutchinson_probes),
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
            divergence_estimator=str(divergence_estimator),
            hutchinson_probes=int(hutchinson_probes),
            a_diag_jitter=float(shared_affine_a_diag_jitter),
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
            divergence_estimator=str(divergence_estimator),
            hutchinson_probes=int(hutchinson_probes),
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
            a_diag_jitter=float(shared_affine_a_diag_jitter),
            low_rank_basis=low_rank_basis,
            **common,
        )
    if fam == "nonlinear":
        return ConditionalNonlinearXFlowFiLM(
            divergence_estimator=str(divergence_estimator),
            hutchinson_probes=int(hutchinson_probes),
            **common,
        )
    raise AssertionError(f"Unhandled velocity family {fam!r}.")


def _adamw_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _sample_fm_batch(
    *,
    path: Any,
    model: nn.Module,
    theta: torch.Tensor,
    x1: torch.Tensor,
    t_eps: float,
) -> torch.Tensor:
    bs = int(x1.shape[0])
    t_raw = torch.rand(bs, device=x1.device, dtype=x1.dtype)
    t = float(t_eps) + (1.0 - 2.0 * float(t_eps)) * t_raw
    x0 = torch.randn_like(x1)
    path_sample = path.sample(x_0=x0, x_1=x1, t=t)
    return torch.mean((model(path_sample.x_t, theta, path_sample.t) - path_sample.dx_t) ** 2)


def _endpoint_warmup_loss(model: nn.Module, theta: torch.Tensor, x1: torch.Tensor) -> torch.Tensor | None:
    endpoint_mean = getattr(model, "endpoint_mean", None)
    if not callable(endpoint_mean):
        return None
    pred = endpoint_mean(theta)
    if not torch.is_tensor(pred) or tuple(pred.shape) != tuple(x1.shape):
        return None
    return torch.mean((pred - x1) ** 2)


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
    epochs: int = 1000,
    batch_size: int = 512,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    t_eps: float = 0.0005,
    patience: int = 0,
    min_delta: float = 1e-4,
    ema_alpha: float = 0.05,
    max_grad_norm: float = 10.0,
    log_every: int = 50,
    endpoint_warmup_epochs: int = 0,
    endpoint_warmup_lr: float | None = None,
) -> dict[str, Any]:
    """Train a flow-SKL model and return training metadata."""

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
    alpha = float(ema_alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("ema_alpha must be in (0, 1].")
    if int(endpoint_warmup_epochs) < 0:
        raise ValueError("endpoint_warmup_epochs must be >= 0.")
    warmup_lr = float(lr if endpoint_warmup_lr is None else endpoint_warmup_lr)
    if int(endpoint_warmup_epochs) > 0 and warmup_lr <= 0.0:
        raise ValueError("endpoint_warmup_lr must be > 0 when endpoint_warmup_epochs > 0.")

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

    train_ds = TensorDataset(torch.from_numpy(th_tr.astype(np.float32)), torch.from_numpy(x_tr.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(th_va.astype(np.float32)), torch.from_numpy(x_va.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)

    model.to(device)
    _, schedule_name = _resolve_path_schedule(path_schedule)
    path, path_name = _make_flow_matching_affine_path(path_schedule)
    schedule_name = path_name
    if hasattr(model, "set_path_schedule"):
        model.set_path_schedule(path_schedule)  # type: ignore[attr-defined]
    opt = torch.optim.AdamW(_adamw_parameters(model), lr=float(lr), weight_decay=float(weight_decay))

    endpoint_warmup_losses: list[float] = []
    endpoint_warmup_val_losses: list[float] = []
    if int(endpoint_warmup_epochs) > 0:
        warmup_opt = torch.optim.AdamW(
            _adamw_parameters(model),
            lr=warmup_lr,
            weight_decay=float(weight_decay),
        )
        for warm_epoch in range(1, int(endpoint_warmup_epochs) + 1):
            model.train()
            ep_losses: list[float] = []
            for tb, x1b in train_loader:
                tb = tb.to(device)
                x1b = x1b.to(device)
                loss = _endpoint_warmup_loss(model, tb, x1b)
                if loss is None:
                    ep_losses = []
                    break
                warmup_opt.zero_grad(set_to_none=True)
                loss.backward()
                if float(max_grad_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(_adamw_parameters(model), float(max_grad_norm))
                warmup_opt.step()
                ep_losses.append(float(loss.detach().cpu()))
            if not ep_losses:
                break
            train_warm = float(np.mean(ep_losses))
            endpoint_warmup_losses.append(train_warm)
            model.eval()
            val_warm_losses: list[float] = []
            with torch.no_grad():
                for batch in val_loader:
                    tb, x1b = batch[0].to(device), batch[1].to(device)
                    val_loss = _endpoint_warmup_loss(model, tb, x1b)
                    if val_loss is not None:
                        val_warm_losses.append(float(val_loss.detach().cpu()))
            endpoint_warmup_val_losses.append(float(np.mean(val_warm_losses)) if val_warm_losses else float("nan"))
            if warm_epoch == 1 or warm_epoch == int(endpoint_warmup_epochs) or warm_epoch % max(1, int(log_every)) == 0:
                print(
                    f"[flow-skl {fam} endpoint-warmup {warm_epoch:4d}/{int(endpoint_warmup_epochs)}] "
                    f"train={train_warm:.6f} val={endpoint_warmup_val_losses[-1]:.6f}",
                    flush=True,
                )

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_monitor_losses: list[float] = []
    val_ema: float | None = None
    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    stopped_early = False
    stopped_epoch = int(epochs)
    n_clipped_steps = 0
    n_total_steps = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        ep_losses: list[float] = []
        for tb, x1b in train_loader:
            tb = tb.to(device)
            x1b = x1b.to(device)
            loss = _sample_fm_batch(
                path=path,
                model=model,
                theta=tb,
                x1=x1b,
                t_eps=te,
            )
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
                val_ep.append(
                    float(
                        _sample_fm_batch(
                            path=path,
                            model=model,
                            theta=tb,
                            x1=x1b,
                            t_eps=te,
                        )
                        .detach()
                        .cpu()
                    )
                )
        val_loss = float(np.mean(val_ep))
        val_losses.append(val_loss)
        val_ema = scalar_val_ema_update(val_ema, val_loss, alpha)
        val_smooth = float(val_ema)
        val_monitor_losses.append(val_smooth)

        if val_smooth < best_val - float(min_delta):
            best_val = val_smooth
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % max(1, int(log_every)) == 0 or epoch == int(epochs):
            print(
                f"[flow-skl {fam} {epoch:4d}/{int(epochs)}] train={train_loss:.6f} "
                f"val={val_loss:.6f} val_smooth={val_smooth:.6f} "
                f"best_smooth={best_val:.6f} best_epoch={best_epoch}",
                flush=True,
            )
        if int(patience) > 0 and patience_counter >= int(patience):
            stopped_early = True
            stopped_epoch = int(epoch)
            print(
                f"[flow-skl {fam} early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_smooth={best_val:.6f} patience={int(patience)}",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "velocity_family": fam,
        "network_architecture": str(getattr(model, "network_architecture", "film")),
        "train_losses": np.asarray(train_losses, dtype=np.float64),
        "val_losses": np.asarray(val_losses, dtype=np.float64),
        "val_monitor_losses": np.asarray(val_monitor_losses, dtype=np.float64),
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
        "n_clipped_steps": int(n_clipped_steps),
        "n_total_steps": int(n_total_steps),
        "path_schedule": schedule_name,
        "early_ema_alpha": float(alpha),
        "endpoint_warmup_epochs": int(endpoint_warmup_epochs),
        "endpoint_warmup_lr": warmup_lr,
        "endpoint_warmup_losses": np.asarray(endpoint_warmup_losses, dtype=np.float64),
        "endpoint_warmup_val_losses": np.asarray(endpoint_warmup_val_losses, dtype=np.float64),
    }


@torch.no_grad()
def sample_flow_endpoint(
    *,
    model: nn.Module,
    theta: np.ndarray,
    n_samples: int,
    device: torch.device,
    ode_steps: int = 64,
    ode_method: str = "midpoint",
) -> torch.Tensor:
    """Sample ``x_1`` by pushing standard-normal samples through the learned ODE."""

    th = _as_torch_2d(theta, device=device)
    if int(th.shape[0]) != 1:
        raise ValueError("theta must contain exactly one endpoint row for sampling.")
    steps = int(ode_steps)
    if steps < 1:
        raise ValueError("ode_steps must be >= 1.")
    if not str(ode_method).strip():
        raise ValueError("ode_method must be non-empty.")
    x_dim = int(getattr(model, "x_dim"))
    x = torch.randn(int(n_samples), x_dim, dtype=torch.float32, device=device)
    theta_b = th.expand(int(n_samples), int(th.shape[1]))
    model.eval()
    time_grid = torch.linspace(0.0, 1.0, steps + 1, dtype=x.dtype, device=device)
    solver = _make_flow_ode_solver(model)
    return solver.sample(
        x_init=x,
        step_size=None,
        method=str(ode_method),
        time_grid=time_grid,
        return_intermediates=False,
        enable_grad=False,
        theta_cond=theta_b,
    )


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
    ode_method: str = "midpoint",
) -> np.ndarray:
    th = _as_torch_2d(theta, device=device)
    if int(th.shape[0]) != 1:
        raise ValueError("theta must contain one endpoint row.")
    model_dtype = _model_floating_dtype(model)
    outs: list[np.ndarray] = []
    n = int(x.shape[0])
    for start in range(0, n, int(batch_size)):
        xb = x[start : start + int(batch_size)].to(device=device, dtype=model_dtype)
        tb = th.to(dtype=xb.dtype).expand(int(xb.shape[0]), int(th.shape[1]))
        logp = flow_endpoint_log_prob(
            model,
            xb,
            tb,
            solve_jitter=float(solve_jitter),
            quadrature_steps=quadrature_steps,
            ode_steps=int(ode_steps),
            ode_method=str(ode_method),
        )
        outs.append(logp.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(outs, axis=0)


def _model_floating_dtype(model: nn.Module) -> torch.dtype:
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.is_floating_point():
            return tensor.dtype
    return torch.float32


def _estimate_model_jeffreys(
    *,
    model: nn.Module,
    theta_all: np.ndarray,
    device: torch.device,
    mc_jeffreys_sample: int,
    ode_steps: int,
    batch_size: int,
    solve_jitter: float,
    quadrature_steps: int | None,
    ode_method: str = "midpoint",
) -> np.ndarray:
    theta = _as_2d_float64(theta_all, name="theta_all")
    if int(mc_jeffreys_sample) < 1:
        raise ValueError("mc_jeffreys_sample must be >= 1.")
    k = int(theta.shape[0])
    directed = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        xi = sample_flow_endpoint(
            model=model,
            theta=theta[i : i + 1],
            n_samples=int(mc_jeffreys_sample),
            device=device,
            ode_steps=int(ode_steps),
            ode_method=str(ode_method),
        )
        logp_i = _log_prob_model(
            model=model,
            x=xi,
            theta=theta[i : i + 1],
            device=device,
            ode_steps=int(ode_steps),
            ode_method=str(ode_method),
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
                ode_method=str(ode_method),
                batch_size=int(batch_size),
                solve_jitter=float(solve_jitter),
                quadrature_steps=quadrature_steps,
            )
            directed[i, j] = float(np.mean(logp_i - logp_j, dtype=np.float64))
    out = directed + directed.T
    out = np.maximum(out, 0.0)
    np.fill_diagonal(out, 0.0)
    return out


def estimate_adjacent_model_jeffreys_fisher(
    *,
    model: nn.Module,
    theta_all: np.ndarray,
    device: torch.device,
    mc_jeffreys_sample: int = 4096,
    ode_steps: int = 64,
    ode_method: str = "midpoint",
    batch_size: int = 1024,
    solve_jitter: float = 1e-6,
    quadrature_steps: int | None = None,
) -> dict[str, np.ndarray]:
    """Estimate scalar Fisher from adjacent model Jeffreys sums only.

    The convention is the Jeffreys sum ``KL(theta_i||theta_j)+KL(theta_j||theta_i)``;
    dividing an adjacent pair by ``dtheta**2`` estimates scalar Fisher.
    """

    theta = _as_2d_float64(theta_all, name="theta_all")
    if int(theta.shape[1]) != 1:
        raise ValueError("Scalar Fisher requires theta_all with one column.")
    if int(theta.shape[0]) < 2:
        raise ValueError("At least two theta points are required.")
    order = np.argsort(theta[:, 0], kind="mergesort")
    theta_s = theta[order]
    dtheta = np.diff(theta_s[:, 0])
    if np.any(dtheta <= 0.0):
        raise ValueError("theta_all must contain strictly increasing unique scalar values after sorting.")
    if int(mc_jeffreys_sample) < 1:
        raise ValueError("mc_jeffreys_sample must be >= 1.")

    model.to(device)
    model.eval()
    jeffreys = np.zeros(int(theta_s.shape[0]) - 1, dtype=np.float64)
    for i in range(int(theta_s.shape[0]) - 1):
        directed: list[float] = []
        for a, b in ((i, i + 1), (i + 1, i)):
            xa = sample_flow_endpoint(
                model=model,
                theta=theta_s[a : a + 1],
                n_samples=int(mc_jeffreys_sample),
                device=device,
                ode_steps=int(ode_steps),
                ode_method=str(ode_method),
            )
            logp_a = _log_prob_model(
                model=model,
                x=xa,
                theta=theta_s[a : a + 1],
                device=device,
                ode_steps=int(ode_steps),
                ode_method=str(ode_method),
                batch_size=int(batch_size),
                solve_jitter=float(solve_jitter),
                quadrature_steps=quadrature_steps,
            )
            logp_b = _log_prob_model(
                model=model,
                x=xa,
                theta=theta_s[b : b + 1],
                device=device,
                ode_steps=int(ode_steps),
                ode_method=str(ode_method),
                batch_size=int(batch_size),
                solve_jitter=float(solve_jitter),
                quadrature_steps=quadrature_steps,
            )
            directed.append(float(np.mean(logp_a - logp_b, dtype=np.float64)))
        jeffreys[i] = max(0.0, float(directed[0] + directed[1]))

    return {
        "theta_midpoints": (0.5 * (theta_s[:-1, 0] + theta_s[1:, 0])).reshape(-1, 1).astype(np.float64),
        "theta_left": theta_s[:-1].astype(np.float64),
        "theta_right": theta_s[1:].astype(np.float64),
        "dtheta": dtheta.astype(np.float64),
        "adjacent_jeffreys": jeffreys,
        "fisher": (jeffreys / (dtheta**2)).astype(np.float64),
    }


@torch.no_grad()
def estimate_affine_mixed_symmetric_kl_fisher(
    *,
    model: nn.Module,
    theta_all: np.ndarray,
    device: torch.device,
    ridge: float = 1e-6,
    ode_steps: int = 64,
) -> dict[str, Any]:
    """Estimate linear Fisher from mixed-affine symmetric KL.

    For each adjacent pair, this integrates ``dSigma/dt = A_bar Sigma +
    Sigma A_bar.T`` with ``A_bar(t)=0.5*(A(theta_i,t)+A(theta_j,t))`` and reads
    out the shared-covariance symmetric KL
    ``delta_mu.T inv(Sigma_bar + ridge I) delta_mu``.  Dividing this adjacent
    mixed-affine SKL by ``dtheta**2`` gives the linear Fisher estimate.
    """

    theta = _as_2d_float64(theta_all, name="theta_all")
    if int(theta.shape[1]) != 1:
        raise ValueError("Scalar Fisher requires theta_all with one column.")
    if int(theta.shape[0]) < 2:
        raise ValueError("At least two theta points are required.")
    if not hasattr(model, "endpoint_mean") or not hasattr(model, "A"):
        raise ValueError("Affine mixed-covariance Fisher requires model.endpoint_mean() and model.A().")
    steps = int(ode_steps)
    if steps < 1:
        raise ValueError("ode_steps must be >= 1.")
    rr = float(ridge)
    if rr < 0.0 or not math.isfinite(rr):
        raise ValueError("ridge must be finite and nonnegative.")

    order = np.argsort(theta[:, 0], kind="mergesort")
    theta_s = theta[order]
    dtheta = np.diff(theta_s[:, 0])
    if np.any(dtheta <= 0.0):
        raise ValueError("theta_all must contain strictly increasing unique scalar values after sorting.")

    model.to(device)
    model.eval()
    dtype = _model_floating_dtype(model)
    x_dim = int(getattr(model, "x_dim"))
    n_theta = int(theta_s.shape[0])
    eye_np = np.eye(x_dim, dtype=np.float64)
    fish = np.zeros(n_theta - 1, dtype=np.float64)
    adjacent_skl = np.zeros(n_theta - 1, dtype=np.float64)
    skl_matrix = np.zeros((n_theta, n_theta), dtype=np.float64)
    mid_covs = np.zeros((n_theta - 1, x_dim, x_dim), dtype=np.float64)
    deltas = np.zeros((n_theta - 1, x_dim), dtype=np.float64)

    def _a_for(th: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        try:
            return model.A(th, t)  # condition-specific affine models
        except TypeError:
            return model.A(t)  # shared affine models

    for i in range(n_theta - 1):
        th_l = torch.from_numpy(theta_s[i : i + 1].astype(np.float32)).to(device=device, dtype=dtype)
        th_r = torch.from_numpy(theta_s[i + 1 : i + 2].astype(np.float32)).to(device=device, dtype=dtype)
        mu_l = model.endpoint_mean(th_l).detach().cpu().numpy().reshape(-1).astype(np.float64)
        mu_r = model.endpoint_mean(th_r).detach().cpu().numpy().reshape(-1).astype(np.float64)
        delta = mu_r - mu_l
        sigma = eye_np.copy()
        dt = 1.0 / float(steps)
        for step in range(steps):
            t_val = (float(step) + 0.5) * dt
            tt = torch.full((1, 1), t_val, dtype=dtype, device=device)
            a_l = _a_for(th_l, tt)
            a_r = _a_for(th_r, tt)
            a_bar = (0.5 * (a_l + a_r)).detach().cpu().numpy().reshape(x_dim, x_dim).astype(np.float64)
            sigma = sigma + dt * (a_bar @ sigma + sigma @ a_bar.T)
            sigma = 0.5 * (sigma + sigma.T)
        cov = sigma + rr * eye_np
        skl = max(0.0, float(delta @ np.linalg.solve(cov, delta)))
        adjacent_skl[i] = skl
        skl_matrix[i, i + 1] = skl
        skl_matrix[i + 1, i] = skl
        fish[i] = skl / float(dtheta[i] ** 2)
        mid_covs[i] = sigma
        deltas[i] = delta

    return {
        "theta_midpoints": (0.5 * (theta_s[:-1, 0] + theta_s[1:, 0])).reshape(-1, 1).astype(np.float64),
        "theta_left": theta_s[:-1].astype(np.float64),
        "theta_right": theta_s[1:].astype(np.float64),
        "dtheta": dtheta.astype(np.float64),
        "delta_mu": deltas,
        "mixed_covariance": mid_covs,
        "adjacent_symmetric_kl": adjacent_skl,
        "symmetric_kl_matrix": skl_matrix,
        "canonical_metric_matrix": skl_matrix.copy(),
        "canonical_metric_name": "mixed_affine_symmetric_kl",
        "fisher": fish,
    }


@torch.no_grad()
def estimate_affine_mixed_covariance_fisher(
    *,
    model: nn.Module,
    theta_all: np.ndarray,
    device: torch.device,
    ridge: float = 1e-6,
    ode_steps: int = 64,
) -> dict[str, Any]:
    """Backward-compatible wrapper for mixed-affine SKL linear Fisher."""

    return estimate_affine_mixed_symmetric_kl_fisher(
        model=model,
        theta_all=theta_all,
        device=device,
        ridge=float(ridge),
        ode_steps=int(ode_steps),
    )


def estimate_model_symmetric_kl(
    *,
    model: nn.Module,
    theta_all: np.ndarray,
    device: torch.device,
    velocity_family: str | None = None,
    radius: float | None = None,
    mc_jeffreys_sample: int = 4096,
    ode_steps: int = 64,
    ode_method: str = "midpoint",
    batch_size: int = 1024,
    solve_jitter: float = 1e-6,
    quadrature_steps: int | None = None,
    fisher_kind: str = "none",
    train_metadata: dict[str, Any] | None = None,
) -> FlowSKLResult:
    """Estimate model symmetric KL from model-sampled likelihood ratios."""

    del radius
    fam = _normalize_velocity_family(velocity_family or getattr(model, "velocity_family", ""))
    theta = _as_2d_float64(theta_all, name="theta_all")
    model.to(device)
    model.eval()

    skl = _estimate_model_jeffreys(
        model=model,
        theta_all=theta,
        device=device,
        mc_jeffreys_sample=int(mc_jeffreys_sample),
        ode_steps=int(ode_steps),
        ode_method=str(ode_method),
        batch_size=int(batch_size),
        solve_jitter=float(solve_jitter),
        quadrature_steps=quadrature_steps,
    )
    canonical = skl.copy()
    metric_name = "model_jeffreys_symmetric_kl"

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

    meta = {} if train_metadata is None else dict(train_metadata)
    meta.setdefault("network_architecture", str(getattr(model, "network_architecture", "film")))

    return FlowSKLResult(
        symmetric_kl_matrix=skl.astype(np.float64, copy=False),
        canonical_metric_matrix=canonical.astype(np.float64, copy=False),
        canonical_metric_name=metric_name,
        fisher_theta_midpoints=fisher_mid,
        fisher_full=fisher_full,
        fisher_linear=fisher_linear,
        train_metadata=meta,
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
        "network_architecture": np.asarray(
            [str(result.train_metadata.get("network_architecture", "film"))],
            dtype=object,
        ),
    }
    if result.fisher_theta_midpoints is not None:
        out["fisher_theta_midpoints"] = result.fisher_theta_midpoints
    if result.fisher_full is not None:
        out["fisher_full"] = result.fisher_full
    if result.fisher_linear is not None:
        out["fisher_linear"] = result.fisher_linear
    return out
