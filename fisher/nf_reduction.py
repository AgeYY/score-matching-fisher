"""Normalizing-flow dimension reduction for H-decoding.

Learns an invertible representation flow ``x -> u=(z, epsilon)`` and models only
``z`` conditionally on theta for likelihood-ratio construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from fisher.nf_hellinger import require_zuko_for_nf, zuko


def _as_2d_float64(a: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D.")
    return arr


class ConditionalZFlow(nn.Module):
    """Conditional NSF model for ``p(z | theta)``."""

    def __init__(
        self,
        *,
        theta_dim: int,
        z_dim: int,
        context_dim: int,
        hidden_dim: int,
        transforms: int,
    ) -> None:
        super().__init__()
        require_zuko_for_nf()
        self.theta_dim = int(theta_dim)
        self.z_dim = int(z_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.theta_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(context_dim)),
        )
        self.flow = zuko.flows.NSF(  # type: ignore[union-attr]
            features=self.z_dim,
            context=int(context_dim),
            transforms=int(transforms),
            hidden_features=[int(hidden_dim), int(hidden_dim)],
        )

    def distribution(self, theta: torch.Tensor) -> torch.distributions.Distribution:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        return self.flow(self.encoder(theta))

    def log_prob(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return self.distribution(theta).log_prob(z)


class NFReductionModel(nn.Module):
    """Invertible ``x -> (z, epsilon)`` reducer with conditional density on ``z``."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        transforms: int = 5,
        context_dim: int = 32,
    ) -> None:
        super().__init__()
        require_zuko_for_nf()
        if int(theta_dim) < 1:
            raise ValueError("theta_dim must be >= 1.")
        if int(x_dim) < 2:
            raise ValueError("x_dim must be >= 2.")
        if int(latent_dim) < 1 or int(latent_dim) >= int(x_dim):
            raise ValueError("latent_dim must satisfy 1 <= latent_dim < x_dim.")
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be >= 1.")
        if int(transforms) < 1:
            raise ValueError("transforms must be >= 1.")
        if int(context_dim) < 1:
            raise ValueError("context_dim must be >= 1.")
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.latent_dim = int(latent_dim)
        self.residual_dim = self.x_dim - self.latent_dim
        self.representation_flow = zuko.flows.NSF(  # type: ignore[union-attr]
            features=self.x_dim,
            transforms=int(transforms),
            hidden_features=[int(hidden_dim), int(hidden_dim)],
        )
        self.z_flow = ConditionalZFlow(
            theta_dim=self.theta_dim,
            z_dim=self.latent_dim,
            context_dim=int(context_dim),
            hidden_dim=int(hidden_dim),
            transforms=int(transforms),
        )

    def encode_normalized(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x_norm.ndim == 1:
            x_norm = x_norm.unsqueeze(0)
        dist = self.representation_flow()
        # zuko's transform maps base -> data, so inverse maps data x -> base u.
        u, logdet = dist.transform.inv.call_and_ladj(x_norm)
        z = u[:, : self.latent_dim]
        eps = u[:, self.latent_dim :]
        return z, eps, logdet

    def log_prob_z_given_theta(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return self.z_flow.log_prob(z, theta)

    def log_prob_epsilon(self, eps: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(eps * eps, dim=1) - 0.5 * float(self.residual_dim) * np.log(2.0 * np.pi)

    def log_prob_normalized_x_given_theta(self, x_norm: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        z, eps, logdet = self.encode_normalized(x_norm)
        return self.log_prob_z_given_theta(z, theta) + self.log_prob_epsilon(eps) + logdet


def train_nf_reduction(
    *,
    model: NFReductionModel,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int = 300,
    min_delta: float = 1e-4,
    ema_alpha: float = 0.05,
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

    th_tr = _as_2d_float64(theta_train, name="theta_train")
    x_tr = _as_2d_float64(x_train, name="x_train")
    th_va = _as_2d_float64(theta_val, name="theta_val")
    x_va = _as_2d_float64(x_val, name="x_val")
    if th_tr.shape[0] < 1 or th_va.shape[0] < 1:
        raise ValueError("nf-reduction requires non-empty train and validation splits.")
    if x_tr.shape[1] != model.x_dim or x_va.shape[1] != model.x_dim:
        raise ValueError("x dimension does not match NFReductionModel.x_dim.")

    x_mean = np.mean(x_tr, axis=0, dtype=np.float64)
    x_std = np.maximum(np.std(x_tr, axis=0, dtype=np.float64), 1e-6)
    norm_logdet = -float(np.sum(np.log(x_std)))
    x_tr_n = (x_tr - x_mean) / x_std
    x_va_n = (x_va - x_mean) / x_std

    th_tr_t = torch.from_numpy(th_tr.astype(np.float32)).to(device)
    x_tr_t = torch.from_numpy(x_tr_n.astype(np.float32)).to(device)
    th_va_t = torch.from_numpy(th_va.astype(np.float32)).to(device)
    x_va_t = torch.from_numpy(x_va_n.astype(np.float32)).to(device)
    ntr = int(x_tr_t.shape[0])
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

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

    for ep in range(1, int(epochs) + 1):
        model.train()
        idx = torch.randint(0, ntr, (int(batch_size),), device=device)
        logp = model.log_prob_normalized_x_given_theta(x_tr_t[idx], th_tr_t[idx]) + norm_logdet
        loss = -logp.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        tr = float(loss.detach().cpu().item())
        train_losses.append(tr)

        model.eval()
        with torch.no_grad():
            va = float((-(model.log_prob_normalized_x_given_theta(x_va_t, th_va_t) + norm_logdet).mean()).cpu())
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
                f"[nf_reduction {ep:4d}/{int(epochs)}] train_nll={tr:.6f} "
                f"val_nll={va:.6f} val_smooth={float(ema):.6f} best_smooth={best_ema:.6f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )

        if int(patience) > 0 and bad >= int(patience):
            stopped_early = True
            stopped_epoch = int(ep)
            print(
                f"[nf_reduction early-stop] epoch={ep} best_epoch={best_epoch} "
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
        "x_mean": x_mean.astype(np.float64),
        "x_std": x_std.astype(np.float64),
    }


def encode_nf_reduction_z(
    *,
    model: NFReductionModel,
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
            z, _, _ = model.encode_normalized(torch.from_numpy(xb).to(device))
            z_out[i0:i1] = z.detach().cpu().numpy().astype(np.float64, copy=False)
    return z_out


def compute_nf_reduction_c_matrix(
    *,
    model: NFReductionModel,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    device: torch.device,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    pair_batch_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    theta = _as_2d_float64(theta_all, name="theta_all")
    x = _as_2d_float64(x_all, name="x_all")
    if theta.shape[0] != x.shape[0]:
        raise ValueError("theta_all and x_all row counts must match.")
    n = int(theta.shape[0])
    z_all = encode_nf_reduction_z(
        model=model,
        x_all=x,
        device=device,
        x_mean=x_mean,
        x_std=x_std,
        batch_size=max(1, int(pair_batch_size) // max(n, 1)),
    )
    max_pairs = max(1, int(pair_batch_size))
    row_bs = max(1, min(n, int(np.sqrt(max_pairs))))
    col_bs = max(1, min(n, max_pairs // row_bs))
    c = np.empty((n, n), dtype=np.float64)
    z_t = torch.from_numpy(z_all.astype(np.float32))
    theta_t = torch.from_numpy(theta.astype(np.float32))
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
                lp = model.log_prob_z_given_theta(z_rep, theta_rep).reshape(bi, bj)
                c[i0:i1, j0:j1] = lp.detach().cpu().numpy().astype(np.float64, copy=False)
    return c, z_all
