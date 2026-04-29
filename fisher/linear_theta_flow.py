"""Linear theta-space flow matching with analytic Gaussian-mixture likelihood.

This module trains a time-independent mixture velocity

    v_k(theta, x) = A_k theta + b_k(x)

on a straight noise-to-data bridge in normalized theta coordinates.  The
endpoint density is a conditional Gaussian mixture for p(theta | x).
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fisher.model_weight_ema import (
    clone_model_weight_ema,
    evaluate_with_weight_ema,
    init_model_weight_ema,
    load_model_weights_from_ema_state,
    scalar_val_ema_update,
    update_model_weight_ema,
)


def _as_2d_float64(a: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D.")
    return arr


class ConditionalLinearThetaFlowMixtureMLP(nn.Module):
    """Mixture of symmetric linear theta drifts conditioned on x."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        num_components: int = 3,
        hidden_dim: int = 128,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if int(theta_dim) < 1:
            raise ValueError("theta_dim must be >= 1.")
        if int(x_dim) < 1:
            raise ValueError("x_dim must be >= 1.")
        if int(num_components) < 1:
            raise ValueError("num_components must be >= 1.")
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be >= 1.")
        if int(depth) < 1:
            raise ValueError("depth must be >= 1.")
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.num_components = int(num_components)
        self.B = nn.Parameter(1e-3 * torch.eye(self.theta_dim).repeat(self.num_components, 1, 1))

        layers: list[nn.Module] = []
        in_dim = self.x_dim
        for _ in range(int(depth)):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.SiLU())
            in_dim = int(hidden_dim)
        self.trunk = nn.Sequential(*layers)
        self.b_head = nn.Linear(in_dim, self.num_components * self.theta_dim)
        self.logit_head = nn.Linear(in_dim, self.num_components)

    @property
    def A(self) -> torch.Tensor:
        return 0.5 * (self.B + self.B.transpose(-1, -2))

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.trunk(x)

    def b(self, x: torch.Tensor) -> torch.Tensor:
        h = self._features(x)
        return self.b_head(h).reshape(-1, self.num_components, self.theta_dim)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        h = self._features(x)
        return self.logit_head(h)

    def component_velocities(self, theta: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        a = self.A
        linear = torch.einsum("bd,ked->bke", theta, a)
        return linear + self.b(x), self.logits(x)

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        v, logits = self.component_velocities(theta, x)
        weights = torch.softmax(logits, dim=-1)
        return torch.sum(weights.unsqueeze(-1) * v, dim=1)

    def endpoint_component_mean_precision(self, x: torch.Tensor, *, solve_jitter: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        b = self.b(x)
        a = self.A
        e_a = torch.linalg.matrix_exp(a)
        eye = torch.eye(self.theta_dim, dtype=a.dtype, device=a.device)
        rhs = torch.einsum("bkd,ked->bke", b, e_a - eye)
        mus: list[torch.Tensor] = []
        for k in range(self.num_components):
            ak = a[k]
            rhsk = rhs[:, k, :]
            try:
                muk = torch.linalg.solve(ak, rhsk.transpose(0, 1)).transpose(0, 1)
            except RuntimeError:
                muk = torch.linalg.solve(ak + float(solve_jitter) * eye, rhsk.transpose(0, 1)).transpose(0, 1)
            if not torch.all(torch.isfinite(muk)):
                muk = torch.linalg.solve(ak + float(solve_jitter) * eye, rhsk.transpose(0, 1)).transpose(0, 1)
            mus.append(muk)
        mu = torch.stack(mus, dim=1)
        precision = torch.linalg.matrix_exp(-2.0 * a)
        return mu, precision

    def log_prob_normalized(self, theta_norm: torch.Tensor, x_norm: torch.Tensor, *, solve_jitter: float = 1e-6) -> torch.Tensor:
        if theta_norm.ndim == 1:
            theta_norm = theta_norm.unsqueeze(-1)
        if x_norm.ndim == 1:
            x_norm = x_norm.unsqueeze(0)
        if theta_norm.shape[0] != x_norm.shape[0]:
            raise ValueError("theta and x batch sizes must match.")
        mu, precision = self.endpoint_component_mean_precision(x_norm, solve_jitter=solve_jitter)
        diff = theta_norm.unsqueeze(1) - mu
        quad = torch.einsum("bkd,kde,bke->bk", diff, precision, diff)
        log_det_sigma = 2.0 * torch.diagonal(self.A, dim1=-2, dim2=-1).sum(dim=-1)
        log_gauss = -0.5 * (quad + log_det_sigma.unsqueeze(0) + float(self.theta_dim) * math.log(2.0 * math.pi))
        log_pi = torch.log_softmax(self.logits(x_norm), dim=-1)
        return torch.logsumexp(log_pi + log_gauss, dim=-1)

    def log_prob_observed(
        self,
        theta_raw: torch.Tensor,
        x_raw: torch.Tensor,
        *,
        theta_mean: torch.Tensor,
        theta_std: torch.Tensor,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        solve_jitter: float = 1e-6,
    ) -> torch.Tensor:
        theta_norm = (theta_raw - theta_mean) / theta_std
        x_norm = (x_raw - x_mean) / x_std
        logjac = -torch.sum(torch.log(theta_std))
        return self.log_prob_normalized(theta_norm, x_norm, solve_jitter=solve_jitter) + logjac


def train_linear_theta_flow(
    *,
    model: ConditionalLinearThetaFlowMixtureMLP,
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
    patience: int = 300,
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
        raise ValueError("linear_theta_flow requires non-empty train and validation splits.")

    theta_mean = np.mean(th_tr, axis=0, dtype=np.float64)
    theta_std = np.maximum(np.std(th_tr, axis=0, dtype=np.float64), 1e-6)
    x_mean = np.mean(x_tr, axis=0, dtype=np.float64)
    x_std = np.maximum(np.std(x_tr, axis=0, dtype=np.float64), 1e-6)
    th_tr_n = (th_tr - theta_mean) / theta_std
    th_va_n = (th_va - theta_mean) / theta_std
    x_tr_n = (x_tr - x_mean) / x_std
    x_va_n = (x_va - x_mean) / x_std

    train_ds = TensorDataset(torch.from_numpy(th_tr_n.astype(np.float32)), torch.from_numpy(x_tr_n.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(th_va_n.astype(np.float32)), torch.from_numpy(x_va_n.astype(np.float32)))
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

    def _fm_loss(theta1: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        bs = int(theta1.shape[0])
        t = te + (1.0 - 2.0 * te) * torch.rand(bs, 1, device=device, dtype=theta1.dtype)
        theta0 = torch.randn_like(theta1)
        theta_t = (1.0 - t) * theta0 + t * theta1
        target = theta1 - theta0
        v, logits = model.component_velocities(theta_t, xb)
        per_k = torch.mean((v - target.unsqueeze(1)) ** 2, dim=-1)
        weights = torch.softmax(logits, dim=-1)
        return torch.mean(torch.sum(weights * per_k, dim=-1))

    for epoch in range(1, int(epochs) + 1):
        model.train()
        ep_losses: list[float] = []
        for thb, xb in train_loader:
            thb = thb.to(device)
            xb = xb.to(device)
            loss = _fm_loss(thb, xb)
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
                for thb, xb in val_loader:
                    val_ep.append(float(_fm_loss(thb.to(device), xb.to(device)).detach().cpu()))
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
                f"[linear_theta_flow {epoch:4d}/{int(epochs)}] train_fm={train_loss:.6f} "
                f"val_fm={val_raw:.6f} val_smooth={val_smooth:.6f} best_monitor={best_val:.6f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
        if int(patience) > 0 and patience_counter >= int(patience):
            stopped_early = True
            stopped_epoch = int(epoch)
            print(
                f"[linear_theta_flow early-stop] epoch={epoch} best_epoch={best_epoch} "
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
                f"[linear_theta_flow restore-best] restored EMA eval weights epoch={best_epoch} "
                f"best_monitor={best_val:.6f}",
                flush=True,
            )
        else:
            model.load_state_dict(best_eval_state_cpu)
            final_eval_weights = "raw"
            print(
                f"[linear_theta_flow restore-best] restored raw eval weights epoch={best_epoch} "
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
        "theta_mean": theta_mean.astype(np.float64),
        "theta_std": theta_std.astype(np.float64),
        "x_mean": x_mean.astype(np.float64),
        "x_std": x_std.astype(np.float64),
    }


def compute_linear_theta_flow_c_matrix(
    *,
    model: ConditionalLinearThetaFlowMixtureMLP,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    device: torch.device,
    theta_mean: np.ndarray,
    theta_std: np.ndarray,
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
    theta_mean_t = torch.from_numpy(np.asarray(theta_mean, dtype=np.float32)).to(device)
    theta_std_t = torch.from_numpy(np.asarray(theta_std, dtype=np.float32)).to(device)
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
                theta_t,
                x_t,
                theta_mean=theta_mean_t,
                theta_std=theta_std_t,
                x_mean=x_mean_t,
                x_std=x_std_t,
                solve_jitter=float(solve_jitter),
            )
            c[i0:i1, :] = logp.reshape(b, n).detach().cpu().numpy().astype(np.float64)
    return c
