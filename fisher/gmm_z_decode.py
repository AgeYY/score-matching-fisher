"""Posterior-sufficiency dimension reduction with a GMM theta decoder."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _as_2d_float64(a: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D.")
    return arr


def _require_scalar_theta(theta: np.ndarray, *, name: str) -> np.ndarray:
    arr = _as_2d_float64(theta, name=name)
    if int(arr.shape[1]) != 1:
        raise ValueError(f"gmm-z-decode v1 requires scalar theta; {name} has shape {arr.shape}.")
    return arr


class GMMZDecodeModel(nn.Module):
    """MLP encoder ``x -> z`` with a univariate GMM density head for ``theta | z``."""

    def __init__(
        self,
        *,
        x_dim: int,
        latent_dim: int = 2,
        components: int = 5,
        hidden_dim: int = 128,
        depth: int = 2,
        min_std: float = 1e-3,
    ) -> None:
        super().__init__()
        if int(x_dim) < 1:
            raise ValueError("x_dim must be >= 1.")
        if int(latent_dim) < 1:
            raise ValueError("latent_dim must be >= 1.")
        if int(components) < 1:
            raise ValueError("components must be >= 1.")
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be >= 1.")
        if int(depth) < 1:
            raise ValueError("depth must be >= 1.")
        if not math.isfinite(float(min_std)) or float(min_std) <= 0.0:
            raise ValueError("min_std must be finite and positive.")
        self.x_dim = int(x_dim)
        self.latent_dim = int(latent_dim)
        self.components = int(components)
        self.min_std = float(min_std)

        enc_layers: list[nn.Module] = []
        in_dim = self.x_dim
        for _ in range(int(depth)):
            enc_layers.append(nn.Linear(in_dim, int(hidden_dim)))
            enc_layers.append(nn.SiLU())
            in_dim = int(hidden_dim)
        enc_layers.append(nn.Linear(in_dim, self.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        self.head = nn.Sequential(
            nn.Linear(self.latent_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 3 * self.components),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.encoder(x)

    def mixture_params_from_z(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.head(z)
        k = self.components
        logits = out[:, :k]
        means = out[:, k : 2 * k]
        std = torch.nn.functional.softplus(out[:, 2 * k :]) + self.min_std
        return logits, means, std

    def log_prob_theta_given_z(self, theta: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        if theta.shape[1] != 1:
            raise ValueError("GMMZDecodeModel supports scalar theta only.")
        logits, means, std = self.mixture_params_from_z(z)
        th = theta[:, :1]
        log_comp = -0.5 * ((th - means) / std).pow(2) - torch.log(std) - 0.5 * math.log(2.0 * math.pi)
        return torch.logsumexp(torch.log_softmax(logits, dim=1) + log_comp, dim=1)

    def log_prob_theta_given_x(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.log_prob_theta_given_z(theta, self.encode(x))


def train_gmm_z_decode(
    *,
    model: GMMZDecodeModel,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.0,
    patience: int = 300,
    min_delta: float = 1e-4,
    ema_alpha: float = 0.05,
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
    if float(weight_decay) < 0.0:
        raise ValueError("weight_decay must be >= 0.")
    if int(patience) < 0:
        raise ValueError("patience must be >= 0.")
    if float(min_delta) < 0.0:
        raise ValueError("min_delta must be >= 0.")
    if not (0.0 < float(ema_alpha) <= 1.0):
        raise ValueError("ema_alpha must be in (0, 1].")
    if not math.isfinite(float(max_grad_norm)) or float(max_grad_norm) < 0.0:
        raise ValueError("max_grad_norm must be finite and >= 0.")

    th_tr = _require_scalar_theta(theta_train, name="theta_train")
    th_va = _require_scalar_theta(theta_val, name="theta_val")
    x_tr = _as_2d_float64(x_train, name="x_train")
    x_va = _as_2d_float64(x_val, name="x_val")
    if th_tr.shape[0] < 1 or th_va.shape[0] < 1:
        raise ValueError("gmm-z-decode requires non-empty train and validation splits.")
    if th_tr.shape[0] != x_tr.shape[0] or th_va.shape[0] != x_va.shape[0]:
        raise ValueError("theta/x row count mismatch.")
    if int(x_tr.shape[1]) != model.x_dim or int(x_va.shape[1]) != model.x_dim:
        raise ValueError("x dimension does not match GMMZDecodeModel.x_dim.")

    x_mean = np.mean(x_tr, axis=0, dtype=np.float64)
    x_std = np.maximum(np.std(x_tr, axis=0, dtype=np.float64), 1e-6)
    theta_mean = np.mean(th_tr, axis=0, dtype=np.float64)
    theta_std = np.maximum(np.std(th_tr, axis=0, dtype=np.float64), 1e-6)
    x_tr_n = (x_tr - x_mean) / x_std
    x_va_n = (x_va - x_mean) / x_std
    th_tr_n = (th_tr - theta_mean) / theta_std
    th_va_n = (th_va - theta_mean) / theta_std

    x_tr_t = torch.from_numpy(x_tr_n.astype(np.float32)).to(device)
    th_tr_t = torch.from_numpy(th_tr_n.astype(np.float32)).to(device)
    x_va_t = torch.from_numpy(x_va_n.astype(np.float32)).to(device)
    th_va_t = torch.from_numpy(th_va_n.astype(np.float32)).to(device)
    ntr = int(x_tr_t.shape[0])
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_ema_losses: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_ema = float("inf")
    best_epoch = 0
    bad = 0
    ema: float | None = None
    stopped_early = False
    stopped_epoch = int(epochs)
    n_clipped = 0

    for ep in range(1, int(epochs) + 1):
        model.train()
        idx = torch.randint(0, ntr, (int(batch_size),), device=device)
        loss = -model.log_prob_theta_given_x(th_tr_t[idx], x_tr_t[idx]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if float(max_grad_norm) > 0.0:
            g = torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            if float(g.detach().cpu()) > float(max_grad_norm):
                n_clipped += 1
        opt.step()

        tr = float(loss.detach().cpu().item())
        train_losses.append(tr)

        model.eval()
        with torch.no_grad():
            va = float((-model.log_prob_theta_given_x(th_va_t, x_va_t).mean()).cpu())
        val_losses.append(va)
        ema = va if ema is None else (float(ema_alpha) * va + (1.0 - float(ema_alpha)) * float(ema))
        val_ema_losses.append(float(ema))

        if float(ema) < best_ema - float(min_delta):
            best_ema = float(ema)
            best_epoch = int(ep)
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        if ep == 1 or ep % max(1, int(log_every)) == 0 or ep == int(epochs):
            print(
                f"[gmm_z_decode {ep:4d}/{int(epochs)}] train_nll={tr:.6f} "
                f"val_nll={va:.6f} val_smooth={float(ema):.6f} best_smooth={best_ema:.6f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
        if int(patience) > 0 and bad >= int(patience):
            stopped_early = True
            stopped_epoch = int(ep)
            print(
                f"[gmm_z_decode early-stop] epoch={ep} best_epoch={best_epoch} "
                f"best_smooth={best_ema:.6f} patience={int(patience)}",
                flush=True,
            )
            break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses": np.asarray(train_losses, dtype=np.float64),
        "val_losses": np.asarray(val_losses, dtype=np.float64),
        "val_ema_losses": np.asarray(val_ema_losses, dtype=np.float64),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
        "best_val_ema": float(best_ema),
        "lr_last": float(opt.param_groups[0]["lr"]),
        "n_clipped_steps": int(n_clipped),
        "n_total_steps": int(len(train_losses)),
        "x_mean": x_mean.astype(np.float64),
        "x_std": x_std.astype(np.float64),
        "theta_mean": theta_mean.astype(np.float64),
        "theta_std": theta_std.astype(np.float64),
    }


def encode_gmm_z_decode_z(
    *,
    model: GMMZDecodeModel,
    x_all: np.ndarray,
    device: torch.device,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    batch_size: int = 8192,
) -> np.ndarray:
    x = _as_2d_float64(x_all, name="x_all")
    mean = np.asarray(x_mean, dtype=np.float64).reshape(1, -1)
    std = np.asarray(x_std, dtype=np.float64).reshape(1, -1)
    if mean.shape[1] != x.shape[1] or std.shape[1] != x.shape[1]:
        raise ValueError("x_mean/x_std shape mismatch.")
    z_out = np.empty((x.shape[0], model.latent_dim), dtype=np.float64)
    bs = max(1, int(batch_size))
    model.eval()
    with torch.no_grad():
        for i0 in range(0, x.shape[0], bs):
            i1 = min(x.shape[0], i0 + bs)
            xb = ((x[i0:i1] - mean) / std).astype(np.float32, copy=False)
            z = model.encode(torch.from_numpy(xb).to(device))
            z_out[i0:i1] = z.detach().cpu().numpy().astype(np.float64, copy=False)
    return z_out


def compute_gmm_z_decode_c_matrix(
    *,
    model: GMMZDecodeModel,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    device: torch.device,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    theta_mean: np.ndarray,
    theta_std: np.ndarray,
    pair_batch_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    theta = _require_scalar_theta(theta_all, name="theta_all")
    x = _as_2d_float64(x_all, name="x_all")
    if theta.shape[0] != x.shape[0]:
        raise ValueError("theta_all and x_all row counts must match.")
    n = int(theta.shape[0])
    z_all = encode_gmm_z_decode_z(
        model=model,
        x_all=x,
        device=device,
        x_mean=x_mean,
        x_std=x_std,
        batch_size=max(1, int(pair_batch_size) // max(n, 1)),
    )
    th_mean = np.asarray(theta_mean, dtype=np.float64).reshape(1, 1)
    th_std = np.asarray(theta_std, dtype=np.float64).reshape(1, 1)
    theta_n = (theta - th_mean) / th_std
    max_pairs = max(1, int(pair_batch_size))
    row_bs = max(1, min(n, int(np.sqrt(max_pairs))))
    col_bs = max(1, min(n, max_pairs // row_bs))
    c = np.empty((n, n), dtype=np.float64)
    z_t = torch.from_numpy(z_all.astype(np.float32))
    theta_t = torch.from_numpy(theta_n.astype(np.float32))
    model.eval()
    with torch.no_grad():
        for i0 in range(0, n, row_bs):
            i1 = min(n, i0 + row_bs)
            z_blk = z_t[i0:i1].to(device)
            bi = int(i1 - i0)
            for j0 in range(0, n, col_bs):
                j1 = min(n, j0 + col_bs)
                theta_blk = theta_t[j0:j1].to(device)
                bj = int(j1 - j0)
                z_rep = z_blk.repeat_interleave(bj, dim=0)
                theta_rep = theta_blk.repeat(bi, 1)
                lp_norm = model.log_prob_theta_given_z(theta_rep, z_rep).reshape(bi, bj)
                # Convert density from normalized theta back to original theta units.
                lp = lp_norm - float(np.log(th_std.reshape(-1)[0]))
                c[i0:i1, j0:j1] = lp.detach().cpu().numpy().astype(np.float64, copy=False)
    return c, z_all
