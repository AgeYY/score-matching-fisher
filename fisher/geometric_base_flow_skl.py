"""Affine flow matching from geometric bases with smoothed-curve SKL readouts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fisher.flow_matching_skl import (
    CenteredConditionAffineFlowSKLModel,
    FlowSKLResult,
    _apply_matrix,
    _as_2d_float64,
    _as_col_t,
    _expand_theta_to_batch,
    _make_flow_matching_affine_path,
    _make_flow_ode_solver,
    _model_floating_dtype,
    estimate_scalar_fisher_from_skl,
)
from fisher.gaussian_x_flow import GaussianAffinePathSchedule
from fisher.model_weight_ema import scalar_val_ema_update


SMOOTHED_LINE_CURVE_METRIC = "smoothed_line_curve_symmetric_kl"
GEOMETRIC_BASE_VELOCITY_FAMILIES = ("lie_affine_2d", "lie_similarity_2d", "lie_similarity_3d", "centered_affine")
NF_CHECKPOINT_SELECTIONS = ("last", "best")


@dataclass(frozen=True)
class LineSegmentBase:
    """Noiseless line-segment base ``anchor + u * direction``."""

    anchor: np.ndarray | tuple[float, ...] = (0.0, 0.0)
    direction: np.ndarray | tuple[float, ...] = (1.0, 0.0)
    u_low: float = -0.5
    u_high: float = 0.5
    name: str = "line_segment"

    def __post_init__(self) -> None:
        anchor = np.asarray(self.anchor, dtype=np.float64).reshape(-1)
        direction = np.asarray(self.direction, dtype=np.float64).reshape(-1)
        if anchor.ndim != 1 or direction.ndim != 1 or int(anchor.size) < 1:
            raise ValueError("anchor and direction must be one-dimensional and non-empty.")
        if anchor.shape != direction.shape:
            raise ValueError("anchor and direction must have the same shape.")
        if not np.all(np.isfinite(anchor)) or not np.all(np.isfinite(direction)):
            raise ValueError("anchor and direction must be finite.")
        if float(np.linalg.norm(direction)) <= 0.0:
            raise ValueError("direction must be nonzero.")
        if not math.isfinite(float(self.u_low)) or not math.isfinite(float(self.u_high)):
            raise ValueError("u bounds must be finite.")
        if float(self.u_low) >= float(self.u_high):
            raise ValueError("u_low must be < u_high.")
        object.__setattr__(self, "anchor", anchor)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "u_low", float(self.u_low))
        object.__setattr__(self, "u_high", float(self.u_high))

    @property
    def ambient_dim(self) -> int:
        return int(np.asarray(self.anchor).size)

    @property
    def intrinsic_dim(self) -> int:
        return 1

    def sample_u(self, n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        count = int(n)
        if count < 1:
            raise ValueError("n must be >= 1.")
        return self.u_low + (self.u_high - self.u_low) * torch.rand(count, 1, device=device, dtype=dtype)

    def points_from_u(self, u: torch.Tensor) -> torch.Tensor:
        if u.ndim == 1:
            u = u.unsqueeze(-1)
        if u.ndim != 2 or int(u.shape[1]) != 1:
            raise ValueError("u must have shape [N] or [N, 1].")
        anchor = torch.as_tensor(self.anchor, dtype=u.dtype, device=u.device).reshape(1, self.ambient_dim)
        direction = torch.as_tensor(self.direction, dtype=u.dtype, device=u.device).reshape(1, self.ambient_dim)
        return anchor + u * direction

    def sample(self, n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.points_from_u(self.sample_u(int(n), device=device, dtype=dtype))

    def sample_with_u(self, n: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.sample_u(int(n), device=device, dtype=dtype)
        return self.points_from_u(u), u


@dataclass(frozen=True)
class HalfCircleBase:
    """Noiseless upper half-circle base parameterized by ``u in [0, 1]``."""

    center: np.ndarray | tuple[float, float] = (0.0, 0.0)
    radius: float = 1.0
    u_low: float = 0.0
    u_high: float = 1.0
    name: str = "half_circle"

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64).reshape(-1)
        if center.shape != (2,):
            raise ValueError("center must contain exactly two values.")
        if not np.all(np.isfinite(center)):
            raise ValueError("center must be finite.")
        radius = float(self.radius)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be finite and positive.")
        if not math.isfinite(float(self.u_low)) or not math.isfinite(float(self.u_high)):
            raise ValueError("u bounds must be finite.")
        if float(self.u_low) >= float(self.u_high):
            raise ValueError("u_low must be < u_high.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "u_low", float(self.u_low))
        object.__setattr__(self, "u_high", float(self.u_high))

    @property
    def ambient_dim(self) -> int:
        return 2

    @property
    def intrinsic_dim(self) -> int:
        return 1

    def sample_u(self, n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        count = int(n)
        if count < 1:
            raise ValueError("n must be >= 1.")
        return self.u_low + (self.u_high - self.u_low) * torch.rand(count, 1, device=device, dtype=dtype)

    def points_from_u(self, u: torch.Tensor) -> torch.Tensor:
        if u.ndim == 1:
            u = u.unsqueeze(-1)
        if u.ndim != 2 or int(u.shape[1]) != 1:
            raise ValueError("u must have shape [N] or [N, 1].")
        theta = math.pi * u
        radius = float(self.radius)
        center = torch.as_tensor(self.center, dtype=u.dtype, device=u.device).reshape(1, 2)
        return torch.cat([radius * torch.cos(theta), radius * torch.sin(theta)], dim=1) + center

    def sample(self, n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.points_from_u(self.sample_u(int(n), device=device, dtype=dtype))

    def sample_with_u(self, n: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.sample_u(int(n), device=device, dtype=dtype)
        return self.points_from_u(u), u


@dataclass(frozen=True)
class HalfCircle3DBase:
    """Noiseless upper half-circle in the xy-plane, embedded in ``R^3``."""

    center: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 1.0
    u_low: float = 0.0
    u_high: float = 1.0
    name: str = "half_circle_3d"

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64).reshape(-1)
        if center.shape != (3,):
            raise ValueError("center must contain exactly three values.")
        if not np.all(np.isfinite(center)):
            raise ValueError("center must be finite.")
        radius = float(self.radius)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be finite and positive.")
        if not math.isfinite(float(self.u_low)) or not math.isfinite(float(self.u_high)):
            raise ValueError("u bounds must be finite.")
        if float(self.u_low) >= float(self.u_high):
            raise ValueError("u_low must be < u_high.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "u_low", float(self.u_low))
        object.__setattr__(self, "u_high", float(self.u_high))

    @property
    def ambient_dim(self) -> int:
        return 3

    @property
    def intrinsic_dim(self) -> int:
        return 1

    def sample_u(self, n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        count = int(n)
        if count < 1:
            raise ValueError("n must be >= 1.")
        return self.u_low + (self.u_high - self.u_low) * torch.rand(count, 1, device=device, dtype=dtype)

    def points_from_u(self, u: torch.Tensor) -> torch.Tensor:
        if u.ndim == 1:
            u = u.unsqueeze(-1)
        if u.ndim != 2 or int(u.shape[1]) != 1:
            raise ValueError("u must have shape [N] or [N, 1].")
        theta = math.pi * u
        radius = float(self.radius)
        center = torch.as_tensor(self.center, dtype=u.dtype, device=u.device).reshape(1, 3)
        zeros = torch.zeros_like(theta)
        return torch.cat([radius * torch.cos(theta), radius * torch.sin(theta), zeros], dim=1) + center

    def sample(self, n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.points_from_u(self.sample_u(int(n), device=device, dtype=dtype))

    def sample_with_u(self, n: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.sample_u(int(n), device=device, dtype=dtype)
        return self.points_from_u(u), u


@dataclass(frozen=True)
class SquarePerimeterBase:
    """Noiseless square-boundary base parameterized by perimeter coordinate ``u``."""

    center: np.ndarray | tuple[float, float] = (0.0, 0.0)
    side_length: float = 1.0
    u_low: float = 0.0
    u_high: float = 4.0
    name: str = "square_perimeter"

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64).reshape(-1)
        if center.shape != (2,):
            raise ValueError("center must contain exactly two values.")
        if not np.all(np.isfinite(center)):
            raise ValueError("center must be finite.")
        side = float(self.side_length)
        if not math.isfinite(side) or side <= 0.0:
            raise ValueError("side_length must be finite and positive.")
        if not math.isfinite(float(self.u_low)) or not math.isfinite(float(self.u_high)):
            raise ValueError("u bounds must be finite.")
        if float(self.u_low) >= float(self.u_high):
            raise ValueError("u_low must be < u_high.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "side_length", side)
        object.__setattr__(self, "u_low", float(self.u_low))
        object.__setattr__(self, "u_high", float(self.u_high))

    @property
    def ambient_dim(self) -> int:
        return 2

    @property
    def intrinsic_dim(self) -> int:
        return 1

    def sample_u(self, n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        count = int(n)
        if count < 1:
            raise ValueError("n must be >= 1.")
        return self.u_low + (self.u_high - self.u_low) * torch.rand(count, 1, device=device, dtype=dtype)

    def points_from_u(self, u: torch.Tensor) -> torch.Tensor:
        if u.ndim == 1:
            u = u.unsqueeze(-1)
        if u.ndim != 2 or int(u.shape[1]) != 1:
            raise ValueError("u must have shape [N] or [N, 1].")
        s = torch.remainder(u, 4.0)
        h = 0.5 * float(self.side_length)
        side = float(self.side_length)
        x = torch.empty_like(s)
        y = torch.empty_like(s)

        m0 = s < 1.0
        m1 = (s >= 1.0) & (s < 2.0)
        m2 = (s >= 2.0) & (s < 3.0)
        m3 = s >= 3.0
        x[m0] = -h + side * s[m0]
        y[m0] = -h
        x[m1] = h
        y[m1] = -h + side * (s[m1] - 1.0)
        x[m2] = h - side * (s[m2] - 2.0)
        y[m2] = h
        x[m3] = -h
        y[m3] = h - side * (s[m3] - 3.0)

        center = torch.as_tensor(self.center, dtype=u.dtype, device=u.device).reshape(1, 2)
        return torch.cat([x, y], dim=1) + center

    def sample(self, n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.points_from_u(self.sample_u(int(n), device=device, dtype=dtype))

    def sample_with_u(self, n: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.sample_u(int(n), device=device, dtype=dtype)
        return self.points_from_u(u), u


@dataclass(frozen=True)
class NoisyGeometricBase:
    """Add isotropic ambient Gaussian noise to a geometric base sampler."""

    base: Any
    sigma: float = 0.0
    name: str = "noisy_geometric_base"

    def __post_init__(self) -> None:
        sig = float(self.sigma)
        if not math.isfinite(sig) or sig < 0.0:
            raise ValueError("sigma must be finite and nonnegative.")
        if not hasattr(self.base, "sample_u") or not hasattr(self.base, "points_from_u"):
            raise ValueError("base must expose sample_u and points_from_u.")
        object.__setattr__(self, "sigma", sig)
        if self.name == "noisy_geometric_base":
            object.__setattr__(self, "name", f"noisy_{getattr(self.base, 'name', type(self.base).__name__)}")

    @property
    def ambient_dim(self) -> int:
        return int(getattr(self.base, "ambient_dim"))

    @property
    def intrinsic_dim(self) -> int:
        return int(getattr(self.base, "intrinsic_dim", 1))

    @property
    def u_low(self) -> float:
        return float(getattr(self.base, "u_low"))

    @property
    def u_high(self) -> float:
        return float(getattr(self.base, "u_high"))

    def sample_u(self, n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.base.sample_u(int(n), device=device, dtype=dtype)

    def points_from_u(self, u: torch.Tensor) -> torch.Tensor:
        return self.base.points_from_u(u)

    def sample(self, n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        x = self.points_from_u(self.sample_u(int(n), device=device, dtype=dtype))
        if self.sigma == 0.0:
            return x
        return x + self.sigma * torch.randn_like(x)

    def sample_with_u(self, n: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.sample_u(int(n), device=device, dtype=dtype)
        x = self.points_from_u(u)
        if self.sigma > 0.0:
            x = x + self.sigma * torch.randn_like(x)
        return x, u


def _make_mlp(*, in_dim: int, out_dim: int, hidden_dim: int, depth: int, final_gain: float = 0.01) -> nn.Sequential:
    if int(in_dim) < 1 or int(out_dim) < 1 or int(hidden_dim) < 1 or int(depth) < 1:
        raise ValueError("in_dim, out_dim, hidden_dim, and depth must be >= 1.")
    layers: list[nn.Module] = []
    cur = int(in_dim)
    for _ in range(int(depth)):
        lin = nn.Linear(cur, int(hidden_dim))
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
        layers.extend([lin, nn.SiLU()])
        cur = int(hidden_dim)
    out = nn.Linear(cur, int(out_dim))
    nn.init.xavier_uniform_(out.weight, gain=float(final_gain))
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class ConditionTimeAffineVelocity(nn.Module):
    """Full affine velocity ``v(x, theta, t) = A(theta,t)x + b(theta,t)``."""

    velocity_family = "condition_time_affine_geometric_base"
    network_architecture = "mlp"

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if int(theta_dim) < 1 or int(x_dim) < 1:
            raise ValueError("theta_dim and x_dim must be >= 1.")
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.net = _make_mlp(
            in_dim=1 + self.theta_dim,
            out_dim=self.x_dim * self.x_dim + self.x_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.01,
        )

    def affine_params(self, theta: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        t = _as_col_t(t, batch=int(theta.shape[0]))
        out = self.net(torch.cat([t, theta], dim=1))
        a_raw = out[:, : self.x_dim * self.x_dim]
        b = out[:, self.x_dim * self.x_dim :]
        return a_raw.reshape(int(theta.shape[0]), self.x_dim, self.x_dim), b

    def forward(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        theta = _expand_theta_to_batch(theta, batch=int(x.shape[0]))
        a, b = self.affine_params(theta, t)
        return _apply_matrix(a, x) + b


class ConditionTimeLieAffine2DVelocity(nn.Module):
    """2D affine velocity in translation, rotation, scale, and strain coordinates."""

    velocity_family = "lie_affine_2d"
    network_architecture = "mlp_lie_affine_2d"

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if int(theta_dim) < 1:
            raise ValueError("theta_dim must be >= 1.")
        if int(x_dim) != 2:
            raise ValueError("ConditionTimeLieAffine2DVelocity requires x_dim == 2.")
        self.theta_dim = int(theta_dim)
        self.x_dim = 2
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.net = _make_mlp(
            in_dim=1 + self.theta_dim,
            out_dim=8,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.01,
        )

    def lie_params(self, theta: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        t = _as_col_t(t, batch=int(theta.shape[0]))
        out = self.net(torch.cat([t, theta], dim=1))
        v = out[:, 0:2]
        omega = out[:, 2]
        lam = out[:, 3]
        alpha = out[:, 4]
        gamma = out[:, 5]
        c = out[:, 6:8]
        a = torch.empty(int(out.shape[0]), 2, 2, dtype=out.dtype, device=out.device)
        a[:, 0, 0] = lam + alpha
        a[:, 0, 1] = -omega + gamma
        a[:, 1, 0] = omega + gamma
        a[:, 1, 1] = lam - alpha
        return v, a, c

    def affine_params(self, theta: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v, a, c = self.lie_params(theta, t)
        return a, v - _apply_matrix(a, c)

    def forward(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        theta = _expand_theta_to_batch(theta, batch=int(x.shape[0]))
        v, a, c = self.lie_params(theta, t)
        return v + _apply_matrix(a, x - c)


class ConditionTimeLieSimilarity2DVelocity(nn.Module):
    """2D affine velocity with translation, rotation, uniform scaling, and learned center."""

    velocity_family = "lie_similarity_2d"
    network_architecture = "mlp_lie_similarity_2d"

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int = 2,
        hidden_dim: int = 128,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if int(theta_dim) < 1:
            raise ValueError("theta_dim must be >= 1.")
        if int(x_dim) != 2:
            raise ValueError("ConditionTimeLieSimilarity2DVelocity requires x_dim == 2.")
        self.theta_dim = int(theta_dim)
        self.x_dim = 2
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.net = _make_mlp(
            in_dim=1 + self.theta_dim,
            out_dim=6,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.01,
        )

    def lie_params(self, theta: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        t = _as_col_t(t, batch=int(theta.shape[0]))
        out = self.net(torch.cat([t, theta], dim=1))
        v = out[:, 0:2]
        omega = out[:, 2]
        lam = out[:, 3]
        c = out[:, 4:6]
        a = torch.empty(int(out.shape[0]), 2, 2, dtype=out.dtype, device=out.device)
        a[:, 0, 0] = lam
        a[:, 0, 1] = -omega
        a[:, 1, 0] = omega
        a[:, 1, 1] = lam
        return v, a, c

    def affine_params(self, theta: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v, a, c = self.lie_params(theta, t)
        return a, v - _apply_matrix(a, c)

    def forward(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        theta = _expand_theta_to_batch(theta, batch=int(x.shape[0]))
        v, a, c = self.lie_params(theta, t)
        return v + _apply_matrix(a, x - c)


class ConditionTimeLieSimilarity3DVelocity(nn.Module):
    """3D affine velocity with translation, rotation, uniform scaling, and learned center."""

    velocity_family = "lie_similarity_3d"
    network_architecture = "mlp_lie_similarity_3d"

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int = 3,
        hidden_dim: int = 128,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if int(theta_dim) < 1:
            raise ValueError("theta_dim must be >= 1.")
        if int(x_dim) != 3:
            raise ValueError("ConditionTimeLieSimilarity3DVelocity requires x_dim == 3.")
        self.theta_dim = int(theta_dim)
        self.x_dim = 3
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.net = _make_mlp(
            in_dim=1 + self.theta_dim,
            out_dim=10,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            final_gain=0.01,
        )

    def lie_params(self, theta: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        t = _as_col_t(t, batch=int(theta.shape[0]))
        out = self.net(torch.cat([t, theta], dim=1))
        v = out[:, 0:3]
        omega = out[:, 3:6]
        lam = out[:, 6]
        c = out[:, 7:10]
        a = torch.empty(int(out.shape[0]), 3, 3, dtype=out.dtype, device=out.device)
        wx = omega[:, 0]
        wy = omega[:, 1]
        wz = omega[:, 2]
        a[:, 0, 0] = lam
        a[:, 0, 1] = -wz
        a[:, 0, 2] = wy
        a[:, 1, 0] = wz
        a[:, 1, 1] = lam
        a[:, 1, 2] = -wx
        a[:, 2, 0] = -wy
        a[:, 2, 1] = wx
        a[:, 2, 2] = lam
        return v, a, c

    def affine_params(self, theta: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v, a, c = self.lie_params(theta, t)
        return a, v - _apply_matrix(a, c)

    def forward(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        theta = _expand_theta_to_batch(theta, batch=int(x.shape[0]))
        v, a, c = self.lie_params(theta, t)
        return v + _apply_matrix(a, x - c)


def _normalize_geometric_base_velocity_family(value: str) -> str:
    family = str(value).strip().lower().replace("-", "_")
    if family not in GEOMETRIC_BASE_VELOCITY_FAMILIES:
        allowed = ", ".join(GEOMETRIC_BASE_VELOCITY_FAMILIES)
        raise ValueError(f"velocity_family must be one of {allowed}; got {value!r}.")
    return family


def build_geometric_base_velocity_model(
    *,
    velocity_family: str = "lie_affine_2d",
    theta_dim: int,
    x_dim: int,
    hidden_dim: int = 128,
    depth: int = 3,
    path_schedule: str | GaussianAffinePathSchedule = "cosine",
) -> nn.Module:
    """Construct a geometric-base affine velocity model."""

    family = _normalize_geometric_base_velocity_family(velocity_family)
    if family == "lie_affine_2d":
        return ConditionTimeLieAffine2DVelocity(
            theta_dim=int(theta_dim),
            x_dim=int(x_dim),
            hidden_dim=int(hidden_dim),
            depth=int(depth),
        )
    if family == "lie_similarity_2d":
        return ConditionTimeLieSimilarity2DVelocity(
            theta_dim=int(theta_dim),
            x_dim=int(x_dim),
            hidden_dim=int(hidden_dim),
            depth=int(depth),
        )
    if family == "lie_similarity_3d":
        return ConditionTimeLieSimilarity3DVelocity(
            theta_dim=int(theta_dim),
            x_dim=int(x_dim),
            hidden_dim=int(hidden_dim),
            depth=int(depth),
        )
    return CenteredConditionAffineFlowSKLModel(
        theta_dim=int(theta_dim),
        x_dim=int(x_dim),
        hidden_dim=int(hidden_dim),
        depth=int(depth),
        path_schedule=path_schedule,
    )


def _adamw_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _condition_indices_from_rows(theta: np.ndarray, condition_eval: np.ndarray) -> np.ndarray:
    th = _as_2d_float64(theta, name="theta")
    cond = _as_2d_float64(condition_eval, name="condition_eval")
    if th.shape[1] != cond.shape[1]:
        raise ValueError("theta and condition_eval must have the same feature dimension.")
    out = np.empty(int(th.shape[0]), dtype=np.int64)
    for i, row in enumerate(th):
        matches = np.flatnonzero(np.all(np.isclose(cond, row.reshape(1, -1), rtol=1e-8, atol=1e-8), axis=1))
        if int(matches.size) != 1:
            raise ValueError("Each theta row must match exactly one row of condition_eval.")
        out[i] = int(matches[0])
    return out


def _normalize_nf_checkpoint_selection(value: str) -> str:
    selection = str(value).strip().lower().replace("-", "_")
    if selection not in NF_CHECKPOINT_SELECTIONS:
        raise ValueError(f"checkpoint_selection must be one of {NF_CHECKPOINT_SELECTIONS}; got {value!r}.")
    return selection


def _geometric_base_metadata(base: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "base_name": str(getattr(base, "name", type(base).__name__)),
        "base_u_low": float(getattr(base, "u_low")),
        "base_u_high": float(getattr(base, "u_high")),
        "base_ambient_dim": int(getattr(base, "ambient_dim")),
        "base_intrinsic_dim": int(getattr(base, "intrinsic_dim", 1)),
    }
    if isinstance(base, NoisyGeometricBase):
        meta["base_noise_sigma"] = float(base.sigma)
        meta["base_inner_name"] = str(getattr(base.base, "name", type(base.base).__name__))
    if hasattr(base, "anchor"):
        meta["base_anchor"] = np.asarray(getattr(base, "anchor"), dtype=np.float64)
    if hasattr(base, "direction"):
        meta["base_direction"] = np.asarray(getattr(base, "direction"), dtype=np.float64)
    if hasattr(base, "center"):
        meta["base_center"] = np.asarray(getattr(base, "center"), dtype=np.float64)
    if hasattr(base, "side_length"):
        meta["base_side_length"] = float(getattr(base, "side_length"))
    return meta


def _as_torch_x0(x0: np.ndarray | torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(x0):
        out = x0.to(device=device, dtype=dtype)
    else:
        out = torch.from_numpy(_as_2d_float64(np.asarray(x0), name="x0").astype(np.float32)).to(device=device, dtype=dtype)
    if out.ndim == 1:
        out = out.unsqueeze(0)
    if out.ndim != 2:
        raise ValueError("x0 must have shape [N, D].")
    return out


def _require_noisy_geometric_base(base: Any) -> NoisyGeometricBase:
    if not isinstance(base, NoisyGeometricBase):
        raise ValueError("Normalizing-flow likelihood fine-tuning requires NoisyGeometricBase.")
    if float(base.sigma) <= 0.0:
        raise ValueError("Normalizing-flow likelihood fine-tuning requires base.sigma > 0.")
    return base


def _base_u_grid(base: Any, n_points: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    count = int(n_points)
    if count < 1:
        raise ValueError("density_points must be >= 1.")
    return torch.linspace(float(base.u_low), float(base.u_high), count, dtype=dtype, device=device).reshape(-1, 1)


def train_geometric_base_affine_flow(
    *,
    model: nn.Module,
    base: LineSegmentBase,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray | None,
    x_val: np.ndarray | None,
    device: torch.device,
    path_schedule: str | GaussianAffinePathSchedule = "cosine",
    epochs: int = 2000,
    batch_size: int = 512,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    t_eps: float = 0.0005,
    patience: int = 0,
    min_delta: float = 1e-4,
    ema_alpha: float = 0.05,
    max_grad_norm: float = 10.0,
    log_every: int = 50,
) -> dict[str, Any]:
    """Train a conditional affine velocity from a geometric base to endpoint data."""

    if int(base.ambient_dim) != int(getattr(model, "x_dim")):
        raise ValueError("base ambient_dim must match model x_dim.")
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
    if int(x_tr.shape[1]) != int(base.ambient_dim) or int(x_va.shape[1]) != int(base.ambient_dim):
        raise ValueError("x dimensions must match base ambient_dim.")

    train_ds = TensorDataset(torch.from_numpy(th_tr.astype(np.float32)), torch.from_numpy(x_tr.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(th_va.astype(np.float32)), torch.from_numpy(x_va.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)

    path, path_name = _make_flow_matching_affine_path(path_schedule)
    if hasattr(model, "set_path_schedule"):
        model.set_path_schedule(path_schedule)
    model.to(device)
    opt = torch.optim.AdamW(_adamw_parameters(model), lr=float(lr), weight_decay=float(weight_decay))

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
            bs = int(x1b.shape[0])
            t_raw = torch.rand(bs, device=device, dtype=x1b.dtype)
            t = te + (1.0 - 2.0 * te) * t_raw
            x0b = base.sample(bs, device=device, dtype=x1b.dtype)
            path_sample = path.sample(x_0=x0b, x_1=x1b, t=t)
            loss = torch.mean((model(path_sample.x_t, tb, path_sample.t) - path_sample.dx_t) ** 2)
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
                bs = int(x1b.shape[0])
                t_raw = torch.rand(bs, device=device, dtype=x1b.dtype)
                t = te + (1.0 - 2.0 * te) * t_raw
                x0b = base.sample(bs, device=device, dtype=x1b.dtype)
                path_sample = path.sample(x_0=x0b, x_1=x1b, t=t)
                val_ep.append(float(torch.mean((model(path_sample.x_t, tb, path_sample.t) - path_sample.dx_t) ** 2).detach().cpu()))
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
                f"[geometric-base-affine {epoch:4d}/{int(epochs)}] train={train_loss:.6f} "
                f"val={val_loss:.6f} val_smooth={val_smooth:.6f} "
                f"best_smooth={best_val:.6f} best_epoch={best_epoch}",
                flush=True,
            )
        if int(patience) > 0 and patience_counter >= int(patience):
            stopped_early = True
            stopped_epoch = int(epoch)
            print(
                f"[geometric-base-affine early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_smooth={best_val:.6f} patience={int(patience)}",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    meta = {
        "velocity_family": str(getattr(model, "velocity_family", "condition_time_affine_geometric_base")),
        "network_architecture": str(getattr(model, "network_architecture", "mlp")),
        "train_losses": np.asarray(train_losses, dtype=np.float64),
        "val_losses": np.asarray(val_losses, dtype=np.float64),
        "val_monitor_losses": np.asarray(val_monitor_losses, dtype=np.float64),
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
        "n_clipped_steps": int(n_clipped_steps),
        "n_total_steps": int(n_total_steps),
        "path_schedule": path_name,
        "early_ema_alpha": float(alpha),
    }
    meta.update(_geometric_base_metadata(base))
    return meta


def _push_base_curve_ode(
    *,
    model: nn.Module,
    base: LineSegmentBase,
    theta: np.ndarray | torch.Tensor,
    device: torch.device,
    u: np.ndarray | torch.Tensor | None = None,
    n_points: int | None = None,
    ode_steps: int = 64,
    ode_method: str = "midpoint",
    enable_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Push base points through the learned ODE, optionally preserving gradients."""

    steps = int(ode_steps)
    if steps < 1:
        raise ValueError("ode_steps must be >= 1.")
    if not str(ode_method).strip():
        raise ValueError("ode_method must be non-empty.")
    dtype = _model_floating_dtype(model)
    if u is None:
        if n_points is None:
            raise ValueError("Either u or n_points must be supplied.")
        u_t = base.sample_u(int(n_points), device=device, dtype=dtype)
    else:
        u_t = torch.as_tensor(u, dtype=dtype, device=device)
        if u_t.ndim == 1:
            u_t = u_t.unsqueeze(-1)
    x0 = base.points_from_u(u_t)
    if torch.is_tensor(theta):
        th = theta.to(device=device, dtype=dtype)
    else:
        th = torch.from_numpy(_as_2d_float64(np.asarray(theta), name="theta").astype(np.float32)).to(device=device, dtype=dtype)
    if th.ndim == 1:
        th = th.unsqueeze(0)
    if int(th.shape[0]) != 1:
        raise ValueError("theta must contain exactly one endpoint row.")
    theta_b = th.expand(int(x0.shape[0]), int(th.shape[1]))
    model.to(device)
    time_grid = torch.linspace(0.0, 1.0, steps + 1, dtype=dtype, device=device)
    solver = _make_flow_ode_solver(model)
    x1 = solver.sample(
        x_init=x0,
        step_size=None,
        method=str(ode_method),
        time_grid=time_grid,
        return_intermediates=False,
        enable_grad=bool(enable_grad),
        theta_cond=theta_b,
    )
    return x1, u_t


def _velocity_affine_params(model: nn.Module, theta: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "affine_params"):
        return model.affine_params(theta, t)
    if hasattr(model, "A") and hasattr(model, "b") and hasattr(model, "_beta_beta_dot"):
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        t = _as_col_t(t, batch=int(theta.shape[0]))
        try:
            a = model.A(theta, t)
        except TypeError:
            a = model.A(t)
        b_endpoint = model.b(theta)
        beta, beta_dot = model._beta_beta_dot(t, batch=int(theta.shape[0]))
        intercept = beta_dot * b_endpoint - _apply_matrix(a, beta * b_endpoint)
        return a, intercept
    raise ValueError(
        "affine_map NLL endpoint solver requires either affine_params(theta, t) "
        "or the centered affine A/b/_beta_beta_dot interface."
    )


def _affine_velocity_and_divergence(
    model: nn.Module,
    x: torch.Tensor,
    theta: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    a, b = _velocity_affine_params(model, theta, t)
    return _apply_matrix(a, x) + b, torch.diagonal(a, dim1=-2, dim2=-1).sum(dim=1)


def log_noisy_geometric_base_density(
    x: torch.Tensor,
    *,
    base: NoisyGeometricBase,
    support_u: torch.Tensor,
) -> torch.Tensor:
    """Log-density of ``base.points_from_u(U) + Normal(0, sigma^2 I)`` on a fixed support grid."""

    noisy_base = _require_noisy_geometric_base(base)
    if int(support_u.shape[0]) < 1:
        raise ValueError("support_u must contain at least one point.")
    xb = x
    if xb.ndim == 1:
        xb = xb.unsqueeze(0)
    if xb.ndim != 2:
        raise ValueError("x must have shape [N, D].")
    centers = noisy_base.points_from_u(support_u.to(device=xb.device, dtype=xb.dtype))
    if int(xb.shape[1]) != int(centers.shape[1]):
        raise ValueError("x dimension must match base ambient_dim.")
    sigma = torch.as_tensor(float(noisy_base.sigma), dtype=xb.dtype, device=xb.device)
    d = int(xb.shape[1])
    sq = torch.sum((xb[:, None, :] - centers[None, :, :]) ** 2, dim=-1)
    log_norm = -0.5 * float(d) * (math.log(2.0 * math.pi) + 2.0 * torch.log(sigma))
    return torch.logsumexp(log_norm - 0.5 * sq / (sigma * sigma), dim=1) - math.log(float(centers.shape[0]))


def geometric_base_cnf_log_prob(
    *,
    model: nn.Module,
    base: NoisyGeometricBase,
    x: torch.Tensor,
    theta: torch.Tensor,
    support_u: torch.Tensor,
    device: torch.device,
    ode_steps: int = 64,
    ode_method: str = "midpoint",
    enable_grad: bool = True,
) -> torch.Tensor:
    """CNF log-likelihood by integrating target data backward to a noisy geometric base."""

    noisy_base = _require_noisy_geometric_base(base)
    steps = int(ode_steps)
    if steps < 1:
        raise ValueError("ode_steps must be >= 1.")
    method = str(ode_method).strip().lower()
    if method not in ("euler", "midpoint"):
        raise ValueError("CNF likelihood currently supports only euler and midpoint ODE methods.")
    dtype = _model_floating_dtype(model)
    xb = x.to(device=device, dtype=dtype)
    if xb.ndim == 1:
        xb = xb.unsqueeze(0)
    if xb.ndim != 2:
        raise ValueError("x must have shape [N, D].")
    if int(xb.shape[1]) != int(noisy_base.ambient_dim) or int(xb.shape[1]) != int(getattr(model, "x_dim")):
        raise ValueError("x dimension must match base and model x_dim.")
    th = theta.to(device=device, dtype=dtype)
    th = _expand_theta_to_batch(th, batch=int(xb.shape[0]))
    u = support_u.to(device=device, dtype=dtype)
    if u.ndim == 1:
        u = u.unsqueeze(-1)

    model.to(device)
    dt = -1.0 / float(steps)
    with torch.set_grad_enabled(bool(enable_grad)):
        z = xb
        log_det = torch.zeros(int(xb.shape[0]), dtype=dtype, device=device)
        for step in range(steps):
            t0 = torch.full((int(xb.shape[0]), 1), 1.0 + float(step) * dt, dtype=dtype, device=device)
            if method == "euler":
                vel, div = _affine_velocity_and_divergence(model, z, th, t0)
                z = z + dt * vel
                log_det = log_det + dt * div
            else:
                vel0, _div0 = _affine_velocity_and_divergence(model, z, th, t0)
                z_mid = z + 0.5 * dt * vel0
                t_mid = torch.full((int(xb.shape[0]), 1), 1.0 + (float(step) + 0.5) * dt, dtype=dtype, device=device)
                vel_mid, div_mid = _affine_velocity_and_divergence(model, z_mid, th, t_mid)
                z = z + dt * vel_mid
                log_det = log_det + dt * div_mid
        return log_noisy_geometric_base_density(z, base=noisy_base, support_u=u) + log_det


def finetune_geometric_base_cnf_likelihood(
    *,
    model: nn.Module,
    base: NoisyGeometricBase,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray | None,
    x_val: np.ndarray | None,
    condition_eval: np.ndarray,
    device: torch.device,
    epochs: int = 1000,
    batch_size: int = 1024,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    density_points: int = 512,
    ode_steps: int = 64,
    ode_method: str = "midpoint",
    checkpoint_selection: str = "last",
    log_every: int = 100,
) -> dict[str, Any]:
    """Fine-tune a geometric-base flow by exact affine-CNF likelihood."""

    noisy_base = _require_noisy_geometric_base(base)
    if int(noisy_base.ambient_dim) != int(getattr(model, "x_dim")):
        raise ValueError("base ambient_dim must match model x_dim.")
    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1.")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1.")
    if float(lr) <= 0.0:
        raise ValueError("lr must be > 0.")
    if float(weight_decay) < 0.0:
        raise ValueError("weight_decay must be >= 0.")
    if int(density_points) < 1:
        raise ValueError("density_points must be >= 1.")
    selection = _normalize_nf_checkpoint_selection(checkpoint_selection)
    log_interval = max(1, int(log_every))

    cond = _as_2d_float64(condition_eval, name="condition_eval")
    th_tr = _as_2d_float64(theta_train, name="theta_train")
    x_tr = _as_2d_float64(x_train, name="x_train")
    if theta_val is None or x_val is None:
        th_va = th_tr
        x_va = x_tr
    else:
        th_va = _as_2d_float64(theta_val, name="theta_val")
        x_va = _as_2d_float64(x_val, name="x_val")
    if th_tr.shape[0] != x_tr.shape[0] or th_va.shape[0] != x_va.shape[0]:
        raise ValueError("theta and x split lengths must match.")
    if x_tr.shape[1] != noisy_base.ambient_dim or x_va.shape[1] != noisy_base.ambient_dim:
        raise ValueError("x dimensions must match base ambient_dim.")
    tr_idx = _condition_indices_from_rows(th_tr, cond)
    va_idx = _condition_indices_from_rows(th_va, cond)

    train_ds = TensorDataset(torch.from_numpy(tr_idx), torch.from_numpy(x_tr.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(va_idx), torch.from_numpy(x_va.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)

    model.to(device)
    dtype = _model_floating_dtype(model)
    support_u = _base_u_grid(noisy_base, int(density_points), device=device, dtype=dtype).detach()
    cond_t = torch.from_numpy(cond.astype(np.float32)).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(_adamw_parameters(model), lr=float(lr), weight_decay=float(weight_decay))

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    def _batch_nll(cb: torch.Tensor, xb: torch.Tensor, *, enable_grad: bool) -> torch.Tensor:
        cb = cb.to(device=device, dtype=torch.long)
        xb = xb.to(device=device, dtype=dtype)
        theta_b = cond_t.index_select(0, cb)
        log_prob = geometric_base_cnf_log_prob(
            model=model,
            base=noisy_base,
            x=xb,
            theta=theta_b,
            support_u=support_u,
            device=device,
            ode_steps=int(ode_steps),
            ode_method=str(ode_method),
            enable_grad=bool(enable_grad),
        )
        return -torch.mean(log_prob)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        ep_losses: list[float] = []
        for cb, xb in train_loader:
            loss = _batch_nll(cb, xb, enable_grad=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(ep_losses))
        train_losses.append(train_loss)

        model.eval()
        val_ep: list[float] = []
        with torch.no_grad():
            for cb, xb in val_loader:
                val_ep.append(float(_batch_nll(cb, xb, enable_grad=False).detach().cpu()))
        val_loss = float(np.mean(val_ep))
        val_losses.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % log_interval == 0 or epoch == int(epochs):
            print(
                f"[geometric-base-cnf {epoch:4d}/{int(epochs)}] train_nll={train_loss:.6f} "
                f"val_nll={val_loss:.6f} best_val_nll={best_val:.6f} best_epoch={best_epoch}",
                flush=True,
            )

    selected_epoch = int(epochs)
    selected_val = float(val_losses[-1])
    if selection == "best" and best_state is not None:
        model.load_state_dict(best_state)
        selected_epoch = int(best_epoch)
        selected_val = float(best_val)

    return {
        "enabled": True,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "density_points": int(density_points),
        "base_noise_sigma": float(noisy_base.sigma),
        "ode_steps": int(ode_steps),
        "ode_method": str(ode_method),
        "checkpoint_selection": selection,
        "best_epoch": int(best_epoch),
        "best_val_nll": float(best_val),
        "selected_epoch": int(selected_epoch),
        "selected_val_nll": float(selected_val),
        "train_nll_losses": np.asarray(train_losses, dtype=np.float64),
        "val_nll_losses": np.asarray(val_losses, dtype=np.float64),
    }




@torch.no_grad()
def push_base_curve(
    *,
    model: nn.Module,
    base: LineSegmentBase,
    theta: np.ndarray | torch.Tensor,
    device: torch.device,
    u: np.ndarray | torch.Tensor | None = None,
    n_points: int | None = None,
    ode_steps: int = 64,
    ode_method: str = "midpoint",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Push line-base points through the learned conditional ODE."""

    model.to(device)
    model.eval()
    return _push_base_curve_ode(
        model=model,
        base=base,
        theta=theta,
        device=device,
        u=u,
        n_points=n_points,
        ode_steps=int(ode_steps),
        ode_method=str(ode_method),
        enable_grad=False,
    )


@torch.no_grad()
def push_initial_points(
    *,
    model: nn.Module,
    x0: np.ndarray | torch.Tensor,
    theta: np.ndarray | torch.Tensor,
    device: torch.device,
    ode_steps: int = 64,
    ode_method: str = "midpoint",
) -> torch.Tensor:
    """Push arbitrary initial ambient points through the learned conditional ODE."""

    steps = int(ode_steps)
    if steps < 1:
        raise ValueError("ode_steps must be >= 1.")
    method = str(ode_method).strip()
    if not method:
        raise ValueError("ode_method must be non-empty.")
    dtype = _model_floating_dtype(model)
    x0_t = _as_torch_x0(x0, device=device, dtype=dtype)
    if int(x0_t.shape[1]) != int(getattr(model, "x_dim")):
        raise ValueError("x0 dimension must match model x_dim.")
    if torch.is_tensor(theta):
        theta_t = theta.to(device=device, dtype=dtype)
    else:
        theta_t = torch.from_numpy(_as_2d_float64(np.asarray(theta), name="theta").astype(np.float32)).to(device=device, dtype=dtype)
    if theta_t.ndim == 1:
        theta_t = theta_t.unsqueeze(0)
    if theta_t.ndim != 2:
        raise ValueError("theta must have shape [1, theta_dim].")
    if int(theta_t.shape[0]) != 1:
        raise ValueError("theta must contain exactly one endpoint row.")

    model.to(device)
    model.eval()
    theta_b = theta_t.expand(int(x0_t.shape[0]), int(theta_t.shape[1]))
    time_grid = torch.linspace(0.0, 1.0, steps + 1, dtype=dtype, device=device)
    solver = _make_flow_ode_solver(model)
    return solver.sample(
        x_init=x0_t,
        step_size=None,
        method=method,
        time_grid=time_grid,
        return_intermediates=False,
        enable_grad=False,
        theta_cond=theta_b,
    )


@torch.no_grad()
def sample_smoothed_curve(
    *,
    model: nn.Module,
    base: LineSegmentBase,
    theta: np.ndarray | torch.Tensor,
    n_samples: int,
    smooth_sigma: float,
    device: torch.device,
    ode_steps: int = 64,
    ode_method: str = "midpoint",
) -> torch.Tensor:
    """Sample from ``gamma_theta(U) + Normal(0, sigma^2 I)``."""

    sigma = float(smooth_sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("smooth_sigma must be finite and positive.")
    centers, _ = push_base_curve(
        model=model,
        base=base,
        theta=theta,
        device=device,
        n_points=int(n_samples),
        ode_steps=int(ode_steps),
        ode_method=str(ode_method),
    )
    return centers + sigma * torch.randn_like(centers)


@torch.no_grad()
def log_smoothed_curve_density(
    *,
    model: nn.Module,
    base: LineSegmentBase,
    x: torch.Tensor,
    theta: np.ndarray | torch.Tensor,
    smooth_sigma: float,
    density_mc_samples: int,
    device: torch.device,
    ode_steps: int = 64,
    ode_method: str = "midpoint",
    batch_size: int = 1024,
    support_u: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    """Estimate ``log q_theta(x)`` by Monte Carlo integration over line coordinates."""

    sigma = float(smooth_sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("smooth_sigma must be finite and positive.")
    k = int(density_mc_samples)
    if k < 1:
        raise ValueError("density_mc_samples must be >= 1.")
    bs = int(batch_size)
    if bs < 1:
        raise ValueError("batch_size must be >= 1.")
    dtype = _model_floating_dtype(model)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    x_eval = x.to(device=device, dtype=dtype)
    if support_u is None:
        centers, _ = push_base_curve(
            model=model,
            base=base,
            theta=theta,
            device=device,
            n_points=k,
            ode_steps=int(ode_steps),
            ode_method=str(ode_method),
        )
    else:
        centers, _ = push_base_curve(
            model=model,
            base=base,
            theta=theta,
            device=device,
            u=support_u,
            ode_steps=int(ode_steps),
            ode_method=str(ode_method),
        )
        k = int(centers.shape[0])
    centers = centers.to(device=device, dtype=dtype)
    d = int(x_eval.shape[1])
    log_norm = -0.5 * float(d) * math.log(2.0 * math.pi * sigma * sigma)
    outs: list[torch.Tensor] = []
    for start in range(0, int(x_eval.shape[0]), bs):
        xb = x_eval[start : start + bs]
        sq = torch.sum((xb[:, None, :] - centers[None, :, :]) ** 2, dim=-1)
        log_kernel = log_norm - 0.5 * sq / (sigma * sigma)
        outs.append(torch.logsumexp(log_kernel, dim=1) - math.log(float(k)))
    return torch.cat(outs, dim=0)


@torch.no_grad()
def estimate_smoothed_curve_symmetric_kl(
    *,
    model: nn.Module,
    base: LineSegmentBase,
    theta_all: np.ndarray,
    device: torch.device,
    smooth_sigma: float = 0.12,
    mc_skl_samples: int = 4096,
    density_mc_samples: int = 1024,
    ode_steps: int = 64,
    ode_method: str = "midpoint",
    batch_size: int = 1024,
    fisher_kind: str = "none",
    train_metadata: dict[str, Any] | None = None,
) -> FlowSKLResult:
    """Estimate pairwise SKL between smoothed fitted line-curve distributions."""

    theta = _as_2d_float64(theta_all, name="theta_all")
    if int(mc_skl_samples) < 1:
        raise ValueError("mc_skl_samples must be >= 1.")
    if int(density_mc_samples) < 1:
        raise ValueError("density_mc_samples must be >= 1.")
    model.to(device)
    model.eval()
    dtype = _model_floating_dtype(model)
    k_theta = int(theta.shape[0])
    directed = np.zeros((k_theta, k_theta), dtype=np.float64)

    support_u = base.sample_u(int(density_mc_samples), device=device, dtype=dtype)
    for i in range(k_theta):
        xi = sample_smoothed_curve(
            model=model,
            base=base,
            theta=theta[i : i + 1],
            n_samples=int(mc_skl_samples),
            smooth_sigma=float(smooth_sigma),
            device=device,
            ode_steps=int(ode_steps),
            ode_method=str(ode_method),
        )
        logp_i = log_smoothed_curve_density(
            model=model,
            base=base,
            x=xi,
            theta=theta[i : i + 1],
            smooth_sigma=float(smooth_sigma),
            density_mc_samples=int(density_mc_samples),
            device=device,
            ode_steps=int(ode_steps),
            ode_method=str(ode_method),
            batch_size=int(batch_size),
            support_u=support_u,
        )
        for j in range(k_theta):
            if i == j:
                continue
            logp_j = log_smoothed_curve_density(
                model=model,
                base=base,
                x=xi,
                theta=theta[j : j + 1],
                smooth_sigma=float(smooth_sigma),
                density_mc_samples=int(density_mc_samples),
                device=device,
                ode_steps=int(ode_steps),
                ode_method=str(ode_method),
                batch_size=int(batch_size),
                support_u=support_u,
            )
            directed[i, j] = float(torch.mean(logp_i - logp_j).detach().cpu())

    skl = np.maximum(directed + directed.T, 0.0)
    np.fill_diagonal(skl, 0.0)
    fisher_mode = str(fisher_kind).strip().lower()
    if fisher_mode not in ("none", "full", "linear", "both"):
        raise ValueError("fisher_kind must be one of: none, full, linear, both.")
    fisher_mid: np.ndarray | None = None
    fisher_full: np.ndarray | None = None
    fisher_linear: np.ndarray | None = None
    if fisher_mode in ("full", "both", "linear"):
        fd = estimate_scalar_fisher_from_skl(theta, skl)
        fisher_mid = fd["theta_midpoints"]
        if fisher_mode in ("full", "both"):
            fisher_full = fd["fisher"]
        if fisher_mode in ("linear", "both"):
            fisher_linear = fd["fisher"]

    meta = {} if train_metadata is None else dict(train_metadata)
    meta.update(
        {
            "canonical_metric_name": SMOOTHED_LINE_CURVE_METRIC,
            "smooth_sigma": float(smooth_sigma),
            "mc_skl_samples": int(mc_skl_samples),
            "density_mc_samples": int(density_mc_samples),
            "ode_steps": int(ode_steps),
            "ode_method": str(ode_method),
        }
    )
    return FlowSKLResult(
        symmetric_kl_matrix=skl.astype(np.float64, copy=False),
        canonical_metric_matrix=skl.astype(np.float64, copy=True),
        canonical_metric_name=SMOOTHED_LINE_CURVE_METRIC,
        fisher_theta_midpoints=fisher_mid,
        fisher_full=fisher_full,
        fisher_linear=fisher_linear,
        train_metadata=meta,
    )


def geometric_flow_result_to_npz_dict(result: FlowSKLResult) -> dict[str, Any]:
    """Convert a geometric-base result to fields for ``np.savez``."""

    out: dict[str, Any] = {
        "symmetric_kl_matrix": result.symmetric_kl_matrix,
        "canonical_metric_matrix": result.canonical_metric_matrix,
        "canonical_metric_name": np.asarray([result.canonical_metric_name], dtype=object),
        "network_architecture": np.asarray(
            [str(result.train_metadata.get("network_architecture", "mlp"))],
            dtype=object,
        ),
    }
    for key in (
        "base_anchor",
        "base_direction",
        "base_center",
        "train_losses",
        "val_losses",
        "val_monitor_losses",
    ):
        if key in result.train_metadata:
            out[key] = np.asarray(result.train_metadata[key])
    for key in (
        "base_u_low",
        "base_u_high",
        "base_ambient_dim",
        "base_intrinsic_dim",
        "base_side_length",
        "smooth_sigma",
        "mc_skl_samples",
        "density_mc_samples",
        "best_val_loss",
        "best_epoch",
        "stopped_epoch",
        "stopped_early",
        "early_ema_alpha",
    ):
        if key in result.train_metadata:
            out[key] = np.asarray([result.train_metadata[key]])
    if result.fisher_theta_midpoints is not None:
        out["fisher_theta_midpoints"] = result.fisher_theta_midpoints
    if result.fisher_full is not None:
        out["fisher_full"] = result.fisher_full
    if result.fisher_linear is not None:
        out["fisher_linear"] = result.fisher_linear
    return out
