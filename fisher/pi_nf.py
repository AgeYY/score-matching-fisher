"""Flow-style pi-VAE / pi-NF dimension reduction for H-decoding."""

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


class ConditionalDiagonalGaussianZ(nn.Module):
    """Diagonal Gaussian ``p(z | theta)`` with MLP mean and scale."""

    def __init__(
        self,
        *,
        theta_dim: int,
        z_dim: int,
        hidden_dim: int,
        min_std: float,
    ) -> None:
        super().__init__()
        if int(theta_dim) < 1:
            raise ValueError("theta_dim must be >= 1.")
        if int(z_dim) < 1:
            raise ValueError("z_dim must be >= 1.")
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be >= 1.")
        if float(min_std) <= 0.0:
            raise ValueError("min_std must be > 0.")
        self.theta_dim = int(theta_dim)
        self.z_dim = int(z_dim)
        self.min_std = float(min_std)
        self.net = nn.Sequential(
            nn.Linear(self.theta_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 2 * self.z_dim),
        )

    def forward(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        out = self.net(theta)
        mu = out[:, : self.z_dim]
        std = torch.nn.functional.softplus(out[:, self.z_dim :]) + self.min_std
        return mu, std

    def log_prob(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if z.ndim == 1:
            z = z.unsqueeze(0)
        mu, std = self.forward(theta)
        diff = (z - mu) / std
        return -0.5 * torch.sum(diff * diff, dim=1) - torch.sum(torch.log(std), dim=1) - 0.5 * self.z_dim * np.log(2.0 * np.pi)


class PiNFModel(nn.Module):
    """Invertible ``x -> (z, r)`` model with diagonal Gaussian ``p(z | theta)`` and standard normal residual."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        transforms: int = 5,
        min_std: float = 1e-3,
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
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.latent_dim = int(latent_dim)
        self.residual_dim = self.x_dim - self.latent_dim
        self.representation_flow = zuko.flows.NSF(  # type: ignore[union-attr]
            features=self.x_dim,
            transforms=int(transforms),
            hidden_features=[int(hidden_dim), int(hidden_dim)],
        )
        self.z_density = ConditionalDiagonalGaussianZ(
            theta_dim=self.theta_dim,
            z_dim=self.latent_dim,
            hidden_dim=int(hidden_dim),
            min_std=float(min_std),
        )

    def encode_normalized(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x_norm.ndim == 1:
            x_norm = x_norm.unsqueeze(0)
        dist = self.representation_flow()
        u, logdet = dist.transform.inv.call_and_ladj(x_norm)
        z = u[:, : self.latent_dim]
        r = u[:, self.latent_dim :]
        return z, r, logdet

    def decode_normalized(self, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if z.ndim == 1:
            z = z.unsqueeze(0)
        if r.ndim == 1:
            r = r.unsqueeze(0)
        if z.shape[0] != r.shape[0]:
            raise ValueError("z and r batch sizes must match.")
        if z.shape[1] != self.latent_dim or r.shape[1] != self.residual_dim:
            raise ValueError("z/r dimensions do not match PiNFModel latent/residual dimensions.")
        u = torch.cat([z, r], dim=1)
        return self.representation_flow().transform(u)

    def reconstruction_mse_with_sampled_residual(
        self,
        x_norm: torch.Tensor,
        *,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z, _, _ = self.encode_normalized(x_norm)
        if residual is None:
            residual = torch.randn(z.shape[0], self.residual_dim, dtype=z.dtype, device=z.device)
        x_recon = self.decode_normalized(z, residual)
        return torch.mean((x_recon - x_norm) ** 2)

    def log_prob_z_given_theta(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return self.z_density.log_prob(z, theta)

    def log_prob_residual(self, r: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(r * r, dim=1) - 0.5 * float(self.residual_dim) * np.log(2.0 * np.pi)

    def log_prob_normalized_x_given_theta(self, x_norm: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        z, r, logdet = self.encode_normalized(x_norm)
        return self.log_prob_z_given_theta(z, theta) + self.log_prob_residual(r) + logdet


def _ridge_r2(features: np.ndarray, target: np.ndarray, alpha: float = 1e-3) -> float:
    x = _as_2d_float64(features, name="features")
    y = _as_2d_float64(target, name="target")
    if x.shape[0] != y.shape[0] or x.shape[0] < 3:
        return float("nan")
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    reg = float(alpha) * np.eye(x_aug.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)
    pred = x_aug @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y, axis=0, keepdims=True)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def train_pi_nf(
    *,
    model: PiNFModel,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.0,
    recon_weight: float = 1.0,
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
    if float(weight_decay) < 0.0:
        raise ValueError("weight_decay must be >= 0.")
    if not np.isfinite(float(recon_weight)) or float(recon_weight) < 0.0:
        raise ValueError("recon_weight must be finite and >= 0.")
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
        raise ValueError("pi-nf requires non-empty train and validation splits.")
    if th_tr.shape[0] != x_tr.shape[0] or th_va.shape[0] != x_va.shape[0]:
        raise ValueError("theta/x row count mismatch.")
    if x_tr.shape[1] != model.x_dim or x_va.shape[1] != model.x_dim:
        raise ValueError("x dimension does not match PiNFModel.x_dim.")
    if th_tr.shape[1] != model.theta_dim or th_va.shape[1] != model.theta_dim:
        raise ValueError("theta dimension does not match PiNFModel.theta_dim.")

    x_mean = np.mean(x_tr, axis=0, dtype=np.float64)
    x_std = np.maximum(np.std(x_tr, axis=0, dtype=np.float64), 1e-6)
    theta_mean = np.mean(th_tr, axis=0, dtype=np.float64)
    theta_std = np.maximum(np.std(th_tr, axis=0, dtype=np.float64), 1e-6)
    norm_logdet = -float(np.sum(np.log(x_std)))
    x_tr_n = (x_tr - x_mean) / x_std
    x_va_n = (x_va - x_mean) / x_std
    th_tr_n = (th_tr - theta_mean) / theta_std
    th_va_n = (th_va - theta_mean) / theta_std

    th_tr_t = torch.from_numpy(th_tr_n.astype(np.float32)).to(device)
    x_tr_t = torch.from_numpy(x_tr_n.astype(np.float32)).to(device)
    th_va_t = torch.from_numpy(th_va_n.astype(np.float32)).to(device)
    x_va_t = torch.from_numpy(x_va_n.astype(np.float32)).to(device)
    ntr = int(x_tr_t.shape[0])
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    val_residual_t = torch.randn(
        x_va_t.shape[0],
        model.residual_dim,
        dtype=x_va_t.dtype,
        device=device,
    )

    train_losses: list[float] = []
    train_nll_losses: list[float] = []
    train_recon_losses: list[float] = []
    val_losses: list[float] = []
    val_nll_losses: list[float] = []
    val_recon_losses: list[float] = []
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
        xb = x_tr_t[idx]
        thb = th_tr_t[idx]
        logp = model.log_prob_normalized_x_given_theta(xb, thb) + norm_logdet
        nll_loss = -logp.mean()
        recon_loss = model.reconstruction_mse_with_sampled_residual(xb)
        loss = nll_loss + float(recon_weight) * recon_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        tr = float(loss.detach().cpu().item())
        train_losses.append(tr)
        train_nll_losses.append(float(nll_loss.detach().cpu().item()))
        train_recon_losses.append(float(recon_loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_nll = -(model.log_prob_normalized_x_given_theta(x_va_t, th_va_t) + norm_logdet).mean()
            val_recon = model.reconstruction_mse_with_sampled_residual(x_va_t, residual=val_residual_t)
            val_total = val_nll + float(recon_weight) * val_recon
            va = float(val_total.cpu())
        val_losses.append(va)
        val_nll_losses.append(float(val_nll.cpu()))
        val_recon_losses.append(float(val_recon.cpu()))
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
                f"[pi_nf {ep:4d}/{int(epochs)}] train_total={tr:.6f} "
                f"train_nll={train_nll_losses[-1]:.6f} train_recon={train_recon_losses[-1]:.6f} "
                f"val_total={va:.6f} val_nll={val_nll_losses[-1]:.6f} val_recon={val_recon_losses[-1]:.6f} "
                f"val_smooth={float(ema):.6f} best_smooth={best_ema:.6f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )

        if int(patience) > 0 and bad >= int(patience):
            stopped_early = True
            stopped_epoch = int(ep)
            print(
                f"[pi_nf early-stop] epoch={ep} best_epoch={best_epoch} "
                f"best_smooth={best_ema:.6f} patience={int(patience)}",
                flush=True,
            )
            break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses": np.asarray(train_losses, dtype=np.float64),
        "train_total_losses": np.asarray(train_losses, dtype=np.float64),
        "train_nll_losses": np.asarray(train_nll_losses, dtype=np.float64),
        "train_recon_losses": np.asarray(train_recon_losses, dtype=np.float64),
        "val_losses": np.asarray(val_losses, dtype=np.float64),
        "val_total_losses": np.asarray(val_losses, dtype=np.float64),
        "val_nll_losses": np.asarray(val_nll_losses, dtype=np.float64),
        "val_recon_losses": np.asarray(val_recon_losses, dtype=np.float64),
        "val_ema_losses": np.asarray(val_ema_losses, dtype=np.float64),
        "recon_weight": float(recon_weight),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
        "best_val_ema": float(best_ema),
        "lr_last": float(opt.param_groups[0]["lr"]),
        "x_mean": x_mean.astype(np.float64),
        "x_std": x_std.astype(np.float64),
        "theta_mean": theta_mean.astype(np.float64),
        "theta_std": theta_std.astype(np.float64),
    }


def encode_pi_nf(
    *,
    model: PiNFModel,
    x_all: np.ndarray,
    device: torch.device,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    x = _as_2d_float64(x_all, name="x_all")
    mean = np.asarray(x_mean, dtype=np.float64).reshape(1, -1)
    std = np.asarray(x_std, dtype=np.float64).reshape(1, -1)
    if mean.shape[1] != x.shape[1] or std.shape[1] != x.shape[1]:
        raise ValueError("x_mean/x_std shape mismatch.")
    z_out = np.empty((x.shape[0], model.latent_dim), dtype=np.float64)
    r_out = np.empty((x.shape[0], model.residual_dim), dtype=np.float64)
    bs = max(1, int(batch_size))
    model.eval()
    with torch.no_grad():
        for i0 in range(0, x.shape[0], bs):
            i1 = min(x.shape[0], i0 + bs)
            xb = ((x[i0:i1] - mean) / std).astype(np.float32, copy=False)
            z, r, _ = model.encode_normalized(torch.from_numpy(xb).to(device))
            z_out[i0:i1] = z.detach().cpu().numpy().astype(np.float64, copy=False)
            r_out[i0:i1] = r.detach().cpu().numpy().astype(np.float64, copy=False)
    return z_out, r_out


def compute_pi_nf_c_matrix(
    *,
    model: PiNFModel,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    device: torch.device,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    theta_mean: np.ndarray,
    theta_std: np.ndarray,
    pair_batch_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = _as_2d_float64(theta_all, name="theta_all")
    x = _as_2d_float64(x_all, name="x_all")
    if theta.shape[0] != x.shape[0]:
        raise ValueError("theta_all and x_all row counts must match.")
    if theta.shape[1] != model.theta_dim:
        raise ValueError("theta dimension does not match PiNFModel.theta_dim.")
    n = int(theta.shape[0])
    z_all, r_all = encode_pi_nf(
        model=model,
        x_all=x,
        device=device,
        x_mean=x_mean,
        x_std=x_std,
        batch_size=max(1, int(pair_batch_size) // max(n, 1)),
    )
    th_mean = np.asarray(theta_mean, dtype=np.float64).reshape(1, -1)
    th_std = np.asarray(theta_std, dtype=np.float64).reshape(1, -1)
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
                lp = model.log_prob_z_given_theta(z_rep, theta_rep).reshape(bi, bj)
                c[i0:i1, j0:j1] = lp.detach().cpu().numpy().astype(np.float64, copy=False)
    return c, z_all, r_all


def pi_nf_diagnostics(*, z_all: np.ndarray, r_all: np.ndarray, theta_all: np.ndarray) -> dict[str, float]:
    theta = _as_2d_float64(theta_all, name="theta_all")
    return {
        "pinf_z_to_theta_r2": float(_ridge_r2(z_all, theta)),
        "pinf_r_to_theta_r2": float(_ridge_r2(r_all, theta)),
    }
