"""Affine flow matching from geometric bases with smoothed-curve SKL readouts."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fisher.flow_matching_skl import (
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


def _adamw_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _inverse_softplus(value: float) -> float:
    target = float(value)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("value must be finite and positive.")
    if target >= 20.0:
        return target
    return float(math.log(math.expm1(target)))


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


def _resolve_source_pairing(source_pairing: str) -> str:
    key = str(source_pairing).strip().lower().replace("-", "_")
    if key in ("random", "independent", "none"):
        return "random"
    if key in ("ot", "ot_cfm", "optimal_transport"):
        return "ot"
    raise ValueError("source_pairing must be one of: random, ot.")


def _resolve_ot_method(ot_method: str) -> str:
    key = str(ot_method).strip().lower().replace("-", "_")
    if key in ("exact", "emd"):
        return "exact"
    if key in ("sinkhorn", "entropic", "torch_sinkhorn", "gpu_sinkhorn"):
        return "torch_sinkhorn"
    if key in ("pot_sinkhorn", "cpu_sinkhorn"):
        return "pot_sinkhorn"
    if key in ("unbalanced", "unbalanced_sinkhorn"):
        return "unbalanced"
    if key in ("partial", "partial_wasserstein"):
        return "partial"
    raise ValueError("ot_method must be one of: exact, sinkhorn, torch_sinkhorn, pot_sinkhorn, unbalanced, partial.")


def _resolve_ot_num_threads(num_threads: int | str) -> int | str:
    if isinstance(num_threads, str):
        text = num_threads.strip().lower()
        if text == "max":
            return "max"
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError("ot_num_threads must be an integer or 'max'.") from exc
    else:
        value = int(num_threads)
    if value < 1:
        raise ValueError("ot_num_threads must be >= 1 or 'max'.")
    return value


class MinibatchOTPlanSampler:
    """TorchCFM-style minibatch OT plan sampler using squared Euclidean costs."""

    def __init__(
        self,
        *,
        method: str = "sinkhorn",
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = False,
        num_threads: int | str = 1,
        sinkhorn_iters: int = 100,
        warn: bool = True,
    ) -> None:
        self.method = _resolve_ot_method(method)
        self.reg = float(reg)
        self.reg_m = float(reg_m)
        self.normalize_cost = bool(normalize_cost)
        self.num_threads = _resolve_ot_num_threads(num_threads)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.warn = bool(warn)
        if self.reg <= 0.0:
            raise ValueError("ot_reg must be > 0.")
        if self.reg_m <= 0.0:
            raise ValueError("ot_reg_m must be > 0.")
        if self.sinkhorn_iters < 1:
            raise ValueError("ot_sinkhorn_iters must be >= 1.")

    def _pot_module(self):
        try:
            import ot as pot  # type: ignore[import-not-found]
        except ImportError:
            return None
        return pot

    def _fallback_exact_map(self, cost: np.ndarray) -> np.ndarray:
        if cost.ndim != 2 or int(cost.shape[0]) != int(cost.shape[1]):
            raise RuntimeError("POT is required for non-square exact minibatch OT plans.")
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:  # pragma: no cover - depends on environment packaging.
            raise RuntimeError("source_pairing='ot' with ot_method='exact' requires POT or scipy.") from exc
        row_ind, col_ind = linear_sum_assignment(cost)
        if int(len(row_ind)) != int(cost.shape[0]):
            raise RuntimeError("Exact OT assignment did not produce a full minibatch matching.")
        pi = np.zeros_like(cost, dtype=np.float64)
        pi[row_ind, col_ind] = 1.0 / float(cost.shape[0])
        return pi

    def _cost_matrix(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        if x0.ndim < 2 or x1.ndim < 2 or int(x0.shape[0]) < 1 or int(x1.shape[0]) < 1:
            raise ValueError("x0 and x1 must have shape [batch, ...] with non-empty batches.")
        x0_flat = x0.reshape(int(x0.shape[0]), -1) if x0.ndim > 2 else x0
        x1_flat = x1.reshape(int(x1.shape[0]), -1) if x1.ndim > 2 else x1
        if int(x0_flat.shape[1]) != int(x1_flat.shape[1]):
            raise ValueError("x0 and x1 flattened feature dimensions must match.")
        with torch.no_grad():
            cost_t = torch.cdist(x0_flat.detach(), x1_flat.detach(), p=2.0).square()
            if self.normalize_cost:
                cost_max = torch.max(cost_t)
                if float(cost_max.detach().cpu()) > 0.0:
                    cost_t = cost_t / cost_max
        return cost_t

    def get_torch_map(self, x0: torch.Tensor, x1: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Compute an entropic balanced OT plan on the input tensor device."""

        cost_t = self._cost_matrix(x0, x1)
        work_dtype = torch.float64 if cost_t.dtype == torch.float64 else torch.float32
        cost = cost_t.to(dtype=work_dtype)
        m, n = int(cost.shape[0]), int(cost.shape[1])
        log_a = torch.full((m,), -math.log(float(m)), device=cost.device, dtype=work_dtype)
        log_b = torch.full((n,), -math.log(float(n)), device=cost.device, dtype=work_dtype)
        log_k = -cost / float(self.reg)
        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)
        for _ in range(self.sinkhorn_iters):
            log_u = log_a - torch.logsumexp(log_k + log_v.reshape(1, n), dim=1)
            log_v = log_b - torch.logsumexp(log_k + log_u.reshape(m, 1), dim=0)
        pi = torch.exp(log_u.reshape(m, 1) + log_k + log_v.reshape(1, n))
        if not bool(torch.all(torch.isfinite(pi)).detach().cpu()):
            if self.warn:
                warnings.warn("Numerical errors in torch Sinkhorn plan, reverting to a uniform plan.", RuntimeWarning)
            pi = torch.full_like(pi, 1.0 / float(m * n))
        total_mass = torch.sum(pi)
        if float(torch.abs(total_mass).detach().cpu()) < 1e-8:
            if self.warn:
                warnings.warn("Torch Sinkhorn plan has near-zero mass, reverting to a uniform plan.", RuntimeWarning)
            pi = torch.full_like(pi, 1.0 / float(m * n))
            total_mass = torch.sum(pi)
        plan_cost = float((torch.sum(pi * cost) / torch.clamp(total_mass, min=1e-12)).detach().cpu())
        return pi, plan_cost

    def get_map(self, x0: torch.Tensor, x1: torch.Tensor) -> tuple[np.ndarray, float]:
        if self.method == "torch_sinkhorn":
            pi_t, plan_cost = self.get_torch_map(x0, x1)
            return pi_t.detach().cpu().numpy().astype(np.float64, copy=False), plan_cost
        cost_t = self._cost_matrix(x0, x1)
        cost = cost_t.detach().cpu().numpy().astype(np.float64, copy=False)

        pot = self._pot_module()
        if pot is None:
            if self.method != "exact":
                raise RuntimeError(
                    "source_pairing='ot' with ot_method other than 'exact' requires the POT package "
                    "(install requirement 'POT')."
                )
            pi = self._fallback_exact_map(cost)
        else:
            a = pot.unif(int(x0.shape[0]))
            b = pot.unif(int(x1.shape[0]))
            if self.method == "exact":
                pi = pot.emd(a, b, cost, numThreads=self.num_threads)
            elif self.method == "pot_sinkhorn":
                pi = pot.sinkhorn(a, b, cost, reg=self.reg)
            elif self.method == "unbalanced":
                pi = pot.unbalanced.sinkhorn_knopp_unbalanced(a, b, cost, reg=self.reg, reg_m=self.reg_m)
            elif self.method == "partial":
                pi = pot.partial.entropic_partial_wasserstein(a, b, cost, reg=self.reg)
            else:  # pragma: no cover - guarded by _resolve_ot_method.
                raise ValueError(f"Unknown OT method: {self.method}")

        pi = np.asarray(pi, dtype=np.float64)
        if not np.all(np.isfinite(pi)):
            raise RuntimeError("OT plan contains non-finite values.")
        if abs(float(pi.sum())) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to a uniform plan.", RuntimeWarning)
            pi = np.ones_like(pi, dtype=np.float64) / float(pi.size)
        plan_cost = float(np.sum(pi * cost) / max(float(np.sum(pi)), 1e-12))
        return pi, plan_cost

    def sample_map(self, pi: np.ndarray, batch_size: int, *, replace: bool = True) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(pi, dtype=np.float64).reshape(-1)
        p = p / float(p.sum())
        choices = np.random.choice(int(pi.shape[0]) * int(pi.shape[1]), p=p, size=int(batch_size), replace=bool(replace))
        return np.divmod(choices, int(pi.shape[1]))

    def sample_torch_map(
        self,
        pi: torch.Tensor,
        batch_size: int,
        *,
        replace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p = pi.reshape(-1)
        p = p / torch.sum(p)
        choices = torch.multinomial(p, num_samples=int(batch_size), replacement=bool(replace))
        rows = torch.div(choices, int(pi.shape[1]), rounding_mode="floor")
        cols = torch.remainder(choices, int(pi.shape[1]))
        return rows.to(dtype=torch.long), cols.to(dtype=torch.long)

    def sample_plan(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        *,
        replace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if self.method == "torch_sinkhorn":
            pi_t, plan_cost = self.get_torch_map(x0, x1)
            i, j = self.sample_torch_map(pi_t, int(x0.shape[0]), replace=bool(replace))
            return x0.index_select(0, i), x1.index_select(0, j), i, j, plan_cost
        pi, plan_cost = self.get_map(x0, x1)
        i_np, j_np = self.sample_map(pi, int(x0.shape[0]), replace=bool(replace))
        i = torch.as_tensor(i_np, device=x0.device, dtype=torch.long)
        j = torch.as_tensor(j_np, device=x1.device, dtype=torch.long)
        return x0.index_select(0, i), x1.index_select(0, j), i, j, plan_cost


def _ot_pair_source_to_target_batch(
    x0: torch.Tensor,
    x1: torch.Tensor,
    *,
    ot_method: str = "torch_sinkhorn",
    ot_reg: float = 0.05,
    ot_reg_m: float = 1.0,
    ot_normalize_cost: bool = False,
    ot_num_threads: int | str = 1,
    ot_sinkhorn_iters: int = 100,
    ot_replace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Sample source-target pairs from a TorchCFM-style minibatch OT plan."""

    if x0.ndim != 2 or x1.ndim != 2 or x0.shape != x1.shape:
        raise ValueError("x0 and x1 must have the same shape [batch, dim].")
    sampler = MinibatchOTPlanSampler(
        method=ot_method,
        reg=float(ot_reg),
        reg_m=float(ot_reg_m),
        normalize_cost=bool(ot_normalize_cost),
        num_threads=ot_num_threads,
        sinkhorn_iters=int(ot_sinkhorn_iters),
    )
    x0_ot, x1_ot, _i, j, plan_cost = sampler.sample_plan(x0, x1, replace=bool(ot_replace))
    return x0_ot, x1_ot, j, plan_cost


def _pair_source_to_target_batch(
    x0: torch.Tensor,
    x1: torch.Tensor,
    *,
    source_pairing: str,
    ot_method: str,
    ot_reg: float,
    ot_reg_m: float,
    ot_normalize_cost: bool,
    ot_num_threads: int | str,
    ot_sinkhorn_iters: int,
    ot_replace: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float | None]:
    pairing = _resolve_source_pairing(source_pairing)
    if pairing == "random":
        return x0, x1, None, None
    x0_ot, x1_ot, target_idx, plan_cost = _ot_pair_source_to_target_batch(
        x0,
        x1,
        ot_method=ot_method,
        ot_reg=float(ot_reg),
        ot_reg_m=float(ot_reg_m),
        ot_normalize_cost=bool(ot_normalize_cost),
        ot_num_threads=ot_num_threads,
        ot_sinkhorn_iters=int(ot_sinkhorn_iters),
        ot_replace=bool(ot_replace),
    )
    return x0_ot, x1_ot, target_idx, plan_cost


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
    source_pairing: str = "random",
    ot_method: str = "torch_sinkhorn",
    ot_reg: float = 0.05,
    ot_reg_m: float = 1.0,
    ot_normalize_cost: bool = False,
    ot_num_threads: int | str = 1,
    ot_sinkhorn_iters: int = 100,
    ot_replace: bool = True,
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
    pairing = _resolve_source_pairing(source_pairing)
    ot_method_resolved = _resolve_ot_method(ot_method)
    ot_num_threads_resolved = _resolve_ot_num_threads(ot_num_threads)
    ot_sinkhorn_iters_resolved = int(ot_sinkhorn_iters)
    if ot_sinkhorn_iters_resolved < 1:
        raise ValueError("ot_sinkhorn_iters must be >= 1.")

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
    train_pairing_costs: list[float] = []
    val_pairing_costs: list[float] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        ep_losses: list[float] = []
        ep_pair_costs: list[float] = []
        for tb, x1b in train_loader:
            tb = tb.to(device)
            x1b = x1b.to(device)
            bs = int(x1b.shape[0])
            t_raw = torch.rand(bs, device=device, dtype=x1b.dtype)
            t = te + (1.0 - 2.0 * te) * t_raw
            x0b = base.sample(bs, device=device, dtype=x1b.dtype)
            x0b, x1b, target_idx, pair_cost = _pair_source_to_target_batch(
                x0b,
                x1b,
                source_pairing=pairing,
                ot_method=ot_method_resolved,
                ot_reg=float(ot_reg),
                ot_reg_m=float(ot_reg_m),
                ot_normalize_cost=bool(ot_normalize_cost),
                ot_num_threads=ot_num_threads_resolved,
                ot_sinkhorn_iters=ot_sinkhorn_iters_resolved,
                ot_replace=bool(ot_replace),
            )
            if target_idx is not None:
                tb = tb.index_select(0, target_idx)
            if pair_cost is not None:
                ep_pair_costs.append(float(pair_cost))
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
        if ep_pair_costs:
            train_pairing_costs.append(float(np.mean(ep_pair_costs)))

        model.eval()
        val_ep: list[float] = []
        val_pair_ep: list[float] = []
        with torch.no_grad():
            for tb, x1b in val_loader:
                tb = tb.to(device)
                x1b = x1b.to(device)
                bs = int(x1b.shape[0])
                t_raw = torch.rand(bs, device=device, dtype=x1b.dtype)
                t = te + (1.0 - 2.0 * te) * t_raw
                x0b = base.sample(bs, device=device, dtype=x1b.dtype)
                x0b, x1b, target_idx, pair_cost = _pair_source_to_target_batch(
                    x0b,
                    x1b,
                    source_pairing=pairing,
                    ot_method=ot_method_resolved,
                    ot_reg=float(ot_reg),
                    ot_reg_m=float(ot_reg_m),
                    ot_normalize_cost=bool(ot_normalize_cost),
                    ot_num_threads=ot_num_threads_resolved,
                    ot_sinkhorn_iters=ot_sinkhorn_iters_resolved,
                    ot_replace=bool(ot_replace),
                )
                if target_idx is not None:
                    tb = tb.index_select(0, target_idx)
                if pair_cost is not None:
                    val_pair_ep.append(float(pair_cost))
                path_sample = path.sample(x_0=x0b, x_1=x1b, t=t)
                val_ep.append(float(torch.mean((model(path_sample.x_t, tb, path_sample.t) - path_sample.dx_t) ** 2).detach().cpu()))
        val_loss = float(np.mean(val_ep))
        val_losses.append(val_loss)
        if val_pair_ep:
            val_pairing_costs.append(float(np.mean(val_pair_ep)))
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
                f"best_smooth={best_val:.6f} best_epoch={best_epoch}"
                + (f" pairing={pairing} ot_method={ot_method_resolved}" if pairing != "random" else ""),
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

    return {
        "velocity_family": str(getattr(model, "velocity_family", "condition_time_affine_geometric_base")),
        "base_name": str(base.name),
        "base_anchor": np.asarray(base.anchor, dtype=np.float64),
        "base_direction": np.asarray(base.direction, dtype=np.float64),
        "base_u_low": float(base.u_low),
        "base_u_high": float(base.u_high),
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
        "source_pairing": pairing,
        "ot_method": ot_method_resolved,
        "ot_reg": float(ot_reg),
        "ot_reg_m": float(ot_reg_m),
        "ot_normalize_cost": bool(ot_normalize_cost),
        "ot_num_threads": ot_num_threads_resolved,
        "ot_sinkhorn_iters": int(ot_sinkhorn_iters_resolved),
        "ot_replace": bool(ot_replace),
        "train_pairing_costs": np.asarray(train_pairing_costs, dtype=np.float64),
        "val_pairing_costs": np.asarray(val_pairing_costs, dtype=np.float64),
    }


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


def geometric_smoothed_curve_nll_loss(
    *,
    model: nn.Module,
    base: LineSegmentBase,
    x: torch.Tensor,
    theta: np.ndarray | torch.Tensor,
    raw_sigma: torch.Tensor,
    sigma_min: float,
    u_grid: torch.Tensor,
    device: torch.device,
    ode_steps: int = 64,
    ode_method: str = "midpoint",
    enable_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Monte Carlo NLL for ``gamma_theta(U) + Normal(0, sigma^2 I)``."""

    sigma_floor = float(sigma_min)
    if not math.isfinite(sigma_floor) or sigma_floor <= 0.0:
        raise ValueError("sigma_min must be finite and positive.")
    if int(u_grid.shape[0]) < 1:
        raise ValueError("u_grid must contain at least one particle.")
    dtype = _model_floating_dtype(model)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    xb = x.to(device=device, dtype=dtype)
    centers, _ = _push_base_curve_ode(
        model=model,
        base=base,
        theta=theta,
        device=device,
        u=u_grid,
        ode_steps=int(ode_steps),
        ode_method=str(ode_method),
        enable_grad=bool(enable_grad),
    )
    sigma = sigma_floor + F.softplus(raw_sigma.to(device=device, dtype=dtype))
    d = int(xb.shape[1])
    sq = torch.sum((xb[:, None, :] - centers[None, :, :]) ** 2, dim=-1)
    log_kernel = -0.5 * sq / (sigma * sigma)
    log_prob = torch.logsumexp(log_kernel, dim=1) - math.log(float(centers.shape[0]))
    log_norm = -0.5 * float(d) * (math.log(2.0 * math.pi) + 2.0 * torch.log(sigma))
    return -(log_norm + log_prob).mean(), sigma


def finetune_geometric_base_nll(
    *,
    model: nn.Module,
    base: LineSegmentBase,
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
    sigma_min: float = 1e-4,
    sigma_init: float = 0.03,
    n_particles: int = 512,
    ode_steps: int = 64,
    ode_method: str = "midpoint",
    checkpoint_selection: str = "last",
    log_every: int = 100,
) -> dict[str, Any]:
    """Fine-tune a geometric-base flow by endpoint mixture NLL."""

    if int(base.ambient_dim) != int(getattr(model, "x_dim")):
        raise ValueError("base ambient_dim must match model x_dim.")
    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1.")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1.")
    if float(lr) <= 0.0:
        raise ValueError("lr must be > 0.")
    if float(weight_decay) < 0.0:
        raise ValueError("weight_decay must be >= 0.")
    if int(n_particles) < 1:
        raise ValueError("n_particles must be >= 1.")
    sigma_floor = float(sigma_min)
    sigma_start = float(sigma_init)
    if not math.isfinite(sigma_floor) or sigma_floor <= 0.0:
        raise ValueError("sigma_min must be finite and positive.")
    if not math.isfinite(sigma_start) or sigma_start <= sigma_floor:
        raise ValueError("sigma_init must be finite and greater than sigma_min.")
    selection = str(checkpoint_selection).strip().lower().replace("-", "_")
    if selection not in ("last", "best"):
        raise ValueError("checkpoint_selection must be one of: last, best.")

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
    if x_tr.shape[1] != base.ambient_dim or x_va.shape[1] != base.ambient_dim:
        raise ValueError("x dimensions must match base ambient_dim.")
    tr_idx = _condition_indices_from_rows(th_tr, cond)
    va_idx = _condition_indices_from_rows(th_va, cond)

    train_ds = TensorDataset(torch.from_numpy(tr_idx), torch.from_numpy(x_tr.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(va_idx), torch.from_numpy(x_va.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)

    model.to(device)
    dtype = _model_floating_dtype(model)
    raw_init = _inverse_softplus(sigma_start - sigma_floor)
    raw_sigma = nn.Parameter(torch.full((int(cond.shape[0]),), raw_init, dtype=dtype, device=device))
    u_grid = base.sample_u(int(n_particles), device=device, dtype=dtype).detach()
    cond_t = torch.from_numpy(cond.astype(np.float32)).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(_adamw_parameters(model) + [raw_sigma], lr=float(lr), weight_decay=float(weight_decay))

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_raw_sigma: torch.Tensor | None = None

    def _batch_nll(cb: torch.Tensor, xb: torch.Tensor, *, enable_grad: bool) -> tuple[torch.Tensor, torch.Tensor]:
        cb = cb.to(device=device, dtype=torch.long)
        xb = xb.to(device=device, dtype=dtype)
        total = torch.zeros((), dtype=dtype, device=device)
        total_count = 0
        sigma_values: list[torch.Tensor] = []
        for cond_idx in torch.unique(cb, sorted=True):
            mask = cb == cond_idx
            count = int(torch.sum(mask).detach().cpu())
            if count < 1:
                continue
            idx = int(cond_idx.detach().cpu())
            loss_c, sigma_c = geometric_smoothed_curve_nll_loss(
                model=model,
                base=base,
                x=xb[mask],
                theta=cond_t[idx : idx + 1],
                raw_sigma=raw_sigma[idx],
                sigma_min=sigma_floor,
                u_grid=u_grid,
                device=device,
                ode_steps=int(ode_steps),
                ode_method=str(ode_method),
                enable_grad=bool(enable_grad),
            )
            total = total + loss_c * float(count)
            total_count += count
            sigma_values.append(sigma_c.detach())
        if total_count < 1:
            raise RuntimeError("Encountered an empty NLL batch.")
        return total / float(total_count), torch.stack(sigma_values) if sigma_values else raw_sigma.detach()

    for epoch in range(1, int(epochs) + 1):
        model.train()
        ep_losses: list[float] = []
        for cb, xb in train_loader:
            loss, _sigmas = _batch_nll(cb, xb, enable_grad=True)
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
                val_loss, _sigmas = _batch_nll(cb, xb, enable_grad=False)
                val_ep.append(float(val_loss.detach().cpu()))
        val_loss_f = float(np.mean(val_ep))
        val_losses.append(val_loss_f)
        if val_loss_f < best_val:
            best_val = val_loss_f
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_raw_sigma = raw_sigma.detach().cpu().clone()

        if epoch == 1 or epoch % max(1, int(log_every)) == 0 or epoch == int(epochs):
            sigmas = (sigma_floor + F.softplus(raw_sigma.detach())).detach().cpu().numpy()
            print(
                f"[geometric-base-nll {epoch:4d}/{int(epochs)}] train_nll={train_loss:.6f} "
                f"val_nll={val_loss_f:.6f} best_val_nll={best_val:.6f} best_epoch={best_epoch} "
                f"sigmas={np.array2string(sigmas, precision=5, separator=',')}",
                flush=True,
            )

    selected_epoch = int(epochs)
    selected_val = float(val_losses[-1])
    if selection == "best" and best_state is not None and best_raw_sigma is not None:
        model.load_state_dict(best_state)
        raw_sigma.data.copy_(best_raw_sigma.to(device=device, dtype=dtype))
        selected_epoch = int(best_epoch)
        selected_val = float(best_val)

    learned_sigmas = (sigma_floor + F.softplus(raw_sigma.detach())).detach().cpu().numpy().astype(np.float64)
    return {
        "enabled": True,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "sigma_min": float(sigma_floor),
        "sigma_init": float(sigma_start),
        "n_particles": int(n_particles),
        "ode_steps": int(ode_steps),
        "ode_method": str(ode_method),
        "checkpoint_selection": selection,
        "best_epoch": int(best_epoch),
        "best_val_nll": float(best_val),
        "selected_epoch": int(selected_epoch),
        "selected_val_nll": float(selected_val),
        "train_nll_losses": np.asarray(train_losses, dtype=np.float64),
        "val_nll_losses": np.asarray(val_losses, dtype=np.float64),
        "learned_sigmas": learned_sigmas,
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
        "train_losses",
        "val_losses",
        "val_monitor_losses",
        "train_pairing_costs",
        "val_pairing_costs",
    ):
        if key in result.train_metadata:
            out[key] = np.asarray(result.train_metadata[key])
    for key in (
        "base_u_low",
        "base_u_high",
        "smooth_sigma",
        "mc_skl_samples",
        "density_mc_samples",
        "best_val_loss",
        "best_epoch",
        "stopped_epoch",
        "stopped_early",
        "early_ema_alpha",
        "ot_reg",
        "ot_reg_m",
        "ot_sinkhorn_iters",
        "ot_normalize_cost",
        "ot_replace",
    ):
        if key in result.train_metadata:
            out[key] = np.asarray([result.train_metadata[key]])
    for key in ("source_pairing", "ot_method", "ot_num_threads"):
        if key in result.train_metadata:
            out[key] = np.asarray([str(result.train_metadata[key])], dtype=object)
    if result.fisher_theta_midpoints is not None:
        out["fisher_theta_midpoints"] = result.fisher_theta_midpoints
    if result.fisher_full is not None:
        out["fisher_full"] = result.fisher_full
    if result.fisher_linear is not None:
        out["fisher_linear"] = result.fisher_linear
    return out
