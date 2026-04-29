from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ConditionalGaussianPrecisionMLP(nn.Module):
    """Conditional Gaussian p(x|theta) with an MLP mean and precision Cholesky factor."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        diag_floor: float = 1e-4,
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
        if float(diag_floor) <= 0.0 or not math.isfinite(float(diag_floor)):
            raise ValueError("diag_floor must be finite and positive.")
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.diag_floor = float(diag_floor)
        n_tri = self.x_dim * (self.x_dim + 1) // 2
        layers: list[nn.Module] = []
        in_dim = self.theta_dim
        for _ in range(int(depth)):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.SiLU())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, self.x_dim + n_tri))
        self.net = nn.Sequential(*layers)
        tri = torch.tril_indices(row=self.x_dim, col=self.x_dim, offset=0)
        self.register_buffer("_tri_i", tri[0], persistent=False)
        self.register_buffer("_tri_j", tri[1], persistent=False)
        self._diag_positions = [
            int(k)
            for k, (i, j) in enumerate(zip(tri[0].tolist(), tri[1].tolist()))
            if int(i) == int(j)
        ]

    def forward(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        out = self.net(theta)
        mu = out[:, : self.x_dim]
        raw = out[:, self.x_dim :]
        batch = int(theta.shape[0])
        l = torch.zeros(batch, self.x_dim, self.x_dim, dtype=raw.dtype, device=raw.device)
        l[:, self._tri_i, self._tri_j] = raw
        diag_raw = raw[:, self._diag_positions]
        diag = torch.nn.functional.softplus(diag_raw) + self.diag_floor
        d = torch.arange(self.x_dim, device=raw.device)
        l[:, d, d] = diag
        return mu, l

    def log_prob(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        if x.shape[0] != theta.shape[0]:
            raise ValueError("x and theta batch sizes must match.")
        mu, l = self.forward(theta)
        diff = x - mu
        z = torch.bmm(l.transpose(1, 2), diff.unsqueeze(-1)).squeeze(-1)
        quad = torch.sum(z * z, dim=1)
        diag = torch.diagonal(l, dim1=1, dim2=2)
        logdet_precision = 2.0 * torch.sum(torch.log(torch.clamp(diag, min=1e-12)), dim=1)
        return 0.5 * logdet_precision - 0.5 * quad - 0.5 * self.x_dim * math.log(2.0 * math.pi)


class ConditionalDiagonalGaussianPrecisionMLP(nn.Module):
    """Conditional Gaussian p(x|theta) with diagonal precision Cholesky factor."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        diag_floor: float = 1e-4,
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
        if float(diag_floor) <= 0.0 or not math.isfinite(float(diag_floor)):
            raise ValueError("diag_floor must be finite and positive.")
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.diag_floor = float(diag_floor)
        layers: list[nn.Module] = []
        in_dim = self.theta_dim
        for _ in range(int(depth)):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.SiLU())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 2 * self.x_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        out = self.net(theta)
        mu = out[:, : self.x_dim]
        diag = torch.nn.functional.softplus(out[:, self.x_dim :]) + self.diag_floor
        return mu, diag

    def log_prob(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        if x.shape[0] != theta.shape[0]:
            raise ValueError("x and theta batch sizes must match.")
        mu, diag = self.forward(theta)
        diff = x - mu
        z = diag * diff
        quad = torch.sum(z * z, dim=1)
        logdet_precision = 2.0 * torch.sum(torch.log(torch.clamp(diag, min=1e-12)), dim=1)
        return 0.5 * logdet_precision - 0.5 * quad - 0.5 * self.x_dim * math.log(2.0 * math.pi)


class ConditionalLowRankGaussianCovarianceMLP(nn.Module):
    """Conditional Gaussian with low-rank covariance plus learned diagonal residual noise."""

    def __init__(
        self,
        *,
        theta_dim: int,
        x_dim: int,
        rank: int,
        hidden_dim: int = 128,
        depth: int = 3,
        diag_floor: float = 1e-4,
        psi_floor: float = 1e-6,
    ) -> None:
        super().__init__()
        if int(theta_dim) < 1:
            raise ValueError("theta_dim must be >= 1.")
        if int(x_dim) < 1:
            raise ValueError("x_dim must be >= 1.")
        if int(rank) < 1:
            raise ValueError("rank must be >= 1.")
        if int(rank) > int(x_dim):
            raise ValueError("rank must be <= x_dim.")
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be >= 1.")
        if int(depth) < 1:
            raise ValueError("depth must be >= 1.")
        if float(diag_floor) <= 0.0 or not math.isfinite(float(diag_floor)):
            raise ValueError("diag_floor must be finite and positive.")
        if float(psi_floor) <= 0.0 or not math.isfinite(float(psi_floor)):
            raise ValueError("psi_floor must be finite and positive.")
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.rank = int(rank)
        self.diag_floor = float(diag_floor)
        self.psi_floor = float(psi_floor)
        n_tri = self.rank * (self.rank + 1) // 2
        layers: list[nn.Module] = []
        in_dim = self.theta_dim
        for _ in range(int(depth)):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.SiLU())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, self.x_dim + n_tri))
        self.net = nn.Sequential(*layers)
        self.A = nn.Parameter(torch.randn(self.x_dim, self.rank) / math.sqrt(float(self.rank)))
        self.raw_psi = nn.Parameter(torch.zeros(self.x_dim))
        tri = torch.tril_indices(row=self.rank, col=self.rank, offset=0)
        self.register_buffer("_tri_i", tri[0], persistent=False)
        self.register_buffer("_tri_j", tri[1], persistent=False)
        self._diag_positions = [
            int(k)
            for k, (i, j) in enumerate(zip(tri[0].tolist(), tri[1].tolist()))
            if int(i) == int(j)
        ]

    def forward(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        out = self.net(theta)
        mu_x = out[:, : self.x_dim]
        raw = out[:, self.x_dim :]
        batch = int(theta.shape[0])
        l = torch.zeros(batch, self.rank, self.rank, dtype=raw.dtype, device=raw.device)
        l[:, self._tri_i, self._tri_j] = raw
        diag_raw = raw[:, self._diag_positions]
        diag = torch.nn.functional.softplus(diag_raw) + self.diag_floor
        d = torch.arange(self.rank, device=raw.device)
        l[:, d, d] = diag
        return mu_x, l

    def residual_variance(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_psi) + self.psi_floor

    @staticmethod
    def _safe_cholesky(mat: torch.Tensor, *, base_jitter: float = 1e-8, max_tries: int = 8) -> torch.Tensor:
        sym = 0.5 * (mat + mat.transpose(-1, -2))
        sym = torch.nan_to_num(sym, nan=0.0, posinf=1e12, neginf=-1e12)
        eye = torch.eye(sym.shape[-1], dtype=sym.dtype, device=sym.device).expand_as(sym)
        jitter = float(base_jitter)
        last_info: torch.Tensor | None = None
        for _ in range(int(max_tries)):
            chol, info = torch.linalg.cholesky_ex(sym + jitter * eye)
            if bool(torch.all(info == 0)):
                return chol
            last_info = info
            jitter *= 10.0
        evals, evecs = torch.linalg.eigh(sym)
        floor = torch.as_tensor(max(float(base_jitter), 1e-8), dtype=sym.dtype, device=sym.device)
        repaired = (evecs * torch.clamp(evals, min=floor).unsqueeze(-2)) @ evecs.transpose(-1, -2)
        chol, info = torch.linalg.cholesky_ex(0.5 * (repaired + repaired.transpose(-1, -2)))
        if bool(torch.all(info == 0)):
            return chol
        diag = torch.clamp(torch.diagonal(sym, dim1=-2, dim2=-1), min=floor)
        return torch.diag_embed(torch.sqrt(diag))

    def log_prob(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        if x.shape[0] != theta.shape[0]:
            raise ValueError("x and theta batch sizes must match.")
        out_dtype = x.dtype
        # Schur complement + batched Cholesky are fragile in float32 when filling the C matrix
        # (large cross-theta batches in ``compute_gaussian_network_c_matrix``).
        comp_dtype = torch.float64 if x.dtype == torch.float32 else x.dtype
        mu_x, l = self.forward(theta)
        mu_x = mu_x.to(comp_dtype)
        l = l.to(comp_dtype)
        A = self.A.to(comp_dtype)
        x_comp = x.to(comp_dtype)
        y = x_comp - mu_x

        psi_var = self.residual_variance().to(comp_dtype)
        psi_inv = torch.reciprocal(torch.clamp(psi_var, min=1e-12))
        a_psi = A * psi_inv.unsqueeze(-1)
        at_psi_a = A.T @ a_psi

        eye = torch.eye(self.rank, dtype=comp_dtype, device=x.device).expand(x.shape[0], self.rank, self.rank)
        c_inv = torch.cholesky_solve(eye, l)
        s = c_inv + at_psi_a.unsqueeze(0)
        s_chol = self._safe_cholesky(s, base_jitter=max(float(self.psi_floor), 1e-8))

        psi_inv_y = y * psi_inv.unsqueeze(0)
        v = psi_inv_y @ A
        s_inv_v = torch.cholesky_solve(v.unsqueeze(-1), s_chol).squeeze(-1)
        q1 = torch.sum(y * psi_inv_y, dim=1)
        q2 = torch.sum(v * s_inv_v, dim=1)
        quad = torch.clamp(q1 - q2, min=0.0)

        logdet_psi = torch.sum(torch.log(torch.clamp(psi_var, min=1e-12)))
        logdet_c = 2.0 * torch.sum(torch.log(torch.clamp(torch.diagonal(l, dim1=1, dim2=2), min=1e-12)), dim=1)
        logdet_s = 2.0 * torch.sum(torch.log(torch.clamp(torch.diagonal(s_chol, dim1=1, dim2=2), min=1e-12)), dim=1)
        logdet = logdet_psi + logdet_c + logdet_s
        out = -0.5 * (quad + logdet + self.x_dim * math.log(2.0 * math.pi))
        return out.to(out_dtype)


class ObservationAutoencoder(nn.Module):
    """Plain autoencoder for compressing observations before Gaussian likelihood fitting."""

    def __init__(
        self,
        *,
        x_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        depth: int = 2,
    ) -> None:
        super().__init__()
        if int(x_dim) < 1:
            raise ValueError("x_dim must be >= 1.")
        if int(latent_dim) < 1:
            raise ValueError("latent_dim must be >= 1.")
        if int(latent_dim) > int(x_dim):
            raise ValueError("latent_dim must be <= x_dim.")
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be >= 1.")
        if int(depth) < 1:
            raise ValueError("depth must be >= 1.")
        self.x_dim = int(x_dim)
        self.latent_dim = int(latent_dim)

        enc_layers: list[nn.Module] = []
        in_dim = self.x_dim
        for _ in range(int(depth)):
            enc_layers.append(nn.Linear(in_dim, int(hidden_dim)))
            enc_layers.append(nn.SiLU())
            in_dim = int(hidden_dim)
        enc_layers.append(nn.Linear(in_dim, self.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = []
        in_dim = self.latent_dim
        for _ in range(int(depth)):
            dec_layers.append(nn.Linear(in_dim, int(hidden_dim)))
            dec_layers.append(nn.SiLU())
            in_dim = int(hidden_dim)
        dec_layers.append(nn.Linear(in_dim, self.x_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decoder(z)
        return z, x_hat


def _as_2d_float64(a: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D.")
    return arr


def train_observation_autoencoder(
    *,
    model: ObservationAutoencoder,
    x_train: np.ndarray,
    x_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.0,
    patience: int = 200,
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
    if int(patience) < 0:
        raise ValueError("patience must be >= 0.")
    if float(min_delta) < 0.0:
        raise ValueError("min_delta must be >= 0.")
    if not (0.0 < float(ema_alpha) <= 1.0):
        raise ValueError("ema_alpha must be in (0, 1].")

    x_tr = _as_2d_float64(x_train, name="x_train")
    x_va = _as_2d_float64(x_val, name="x_val")
    if x_tr.shape[0] < 1 or x_va.shape[0] < 1:
        raise ValueError("autoencoder requires non-empty train and validation splits.")
    if x_tr.shape[1] != model.x_dim or x_va.shape[1] != model.x_dim:
        raise ValueError("autoencoder x dimension mismatch.")

    train_ds = TensorDataset(torch.from_numpy(x_tr.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(x_va.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_monitor_losses: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    val_ema: float | None = None
    patience_counter = 0
    stopped_early = False
    stopped_epoch = int(epochs)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        ep_losses: list[float] = []
        for (xb,) in train_loader:
            xb = xb.to(device)
            _, x_hat = model(xb)
            loss = torch.mean((x_hat - xb) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(ep_losses))
        train_losses.append(train_loss)

        model.eval()
        val_ep: list[float] = []
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                _, x_hat = model(xb)
                val_ep.append(float(torch.mean((x_hat - xb) ** 2).detach().cpu()))
        val_loss = float(np.mean(val_ep))
        val_losses.append(val_loss)
        val_ema = val_loss if val_ema is None else float(ema_alpha) * val_loss + (1.0 - float(ema_alpha)) * val_ema
        val_monitor_losses.append(float(val_ema))
        if val_ema < best_val - float(min_delta):
            best_val = float(val_ema)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch == 1 or epoch % max(1, int(log_every)) == 0 or epoch == int(epochs):
            print(
                f"[gaussian_network_ae {epoch:4d}/{int(epochs)}] train_mse={train_loss:.6f} "
                f"val_mse={val_loss:.6f} val_smooth={val_ema:.6f} best_smooth={best_val:.6f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
        if int(patience) > 0 and patience_counter >= int(patience):
            stopped_early = True
            stopped_epoch = int(epoch)
            print(
                f"[gaussian_network_ae early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_smooth={best_val:.6f} patience={int(patience)}",
                flush=True,
            )
            break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[gaussian_network_ae restore-best] restored epoch={best_epoch} val_smooth={best_val:.6f}", flush=True)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_monitor_losses": val_monitor_losses,
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
        "lr_last": float(opt.param_groups[0]["lr"]),
    }


def encode_observations(
    *,
    model: ObservationAutoencoder,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    arr = _as_2d_float64(x, name="x")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1.")
    out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, arr.shape[0], int(batch_size)):
            xb = torch.from_numpy(arr[i : i + int(batch_size)].astype(np.float32)).to(device)
            out.append(model.encode(xb).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(out, axis=0) if out else np.zeros((0, model.latent_dim), dtype=np.float64)


def train_gaussian_network(
    *,
    model: ConditionalGaussianPrecisionMLP
    | ConditionalDiagonalGaussianPrecisionMLP
    | ConditionalLowRankGaussianCovarianceMLP,
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
    if int(patience) < 0:
        raise ValueError("patience must be >= 0.")
    if float(min_delta) < 0.0:
        raise ValueError("min_delta must be >= 0.")
    if not (0.0 < float(ema_alpha) <= 1.0):
        raise ValueError("ema_alpha must be in (0, 1].")
    if float(weight_decay) < 0.0:
        raise ValueError("weight_decay must be >= 0.")

    th_tr = _as_2d_float64(theta_train, name="theta_train")
    x_tr = _as_2d_float64(x_train, name="x_train")
    th_va = _as_2d_float64(theta_val, name="theta_val")
    x_va = _as_2d_float64(x_val, name="x_val")
    if th_tr.shape[0] < 1 or th_va.shape[0] < 1:
        raise ValueError("gaussian_network requires non-empty train and validation splits.")
    if x_tr.shape[0] != th_tr.shape[0] or x_va.shape[0] != th_va.shape[0]:
        raise ValueError("theta and x row counts must match.")

    train_ds = TensorDataset(
        torch.from_numpy(th_tr.astype(np.float32)),
        torch.from_numpy(x_tr.astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(th_va.astype(np.float32)),
        torch.from_numpy(x_va.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_monitor_losses: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    val_ema: float | None = None
    patience_counter = 0
    stopped_early = False
    stopped_epoch = int(epochs)
    grad_norms: list[float] = []
    n_clipped = 0
    n_steps = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        ep_losses: list[float] = []
        for tb, xb in train_loader:
            tb = tb.to(device)
            xb = xb.to(device)
            loss = -torch.mean(model.log_prob(xb, tb))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(max_grad_norm) > 0.0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
                grad_norm = float(gn.detach().cpu()) if torch.is_tensor(gn) else float(gn)
                if grad_norm > float(max_grad_norm):
                    n_clipped += 1
            else:
                grad_norm_sq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        g = float(torch.linalg.vector_norm(p.grad.detach()).cpu())
                        grad_norm_sq += g * g
                grad_norm = float(math.sqrt(grad_norm_sq))
            grad_norms.append(grad_norm)
            n_steps += 1
            opt.step()
            ep_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(ep_losses))
        train_losses.append(train_loss)

        model.eval()
        val_ep: list[float] = []
        with torch.no_grad():
            for tb, xb in val_loader:
                tb = tb.to(device)
                xb = xb.to(device)
                val_ep.append(float((-torch.mean(model.log_prob(xb, tb))).detach().cpu()))
        val_loss = float(np.mean(val_ep))
        val_losses.append(val_loss)
        val_ema = val_loss if val_ema is None else float(ema_alpha) * val_loss + (1.0 - float(ema_alpha)) * val_ema
        val_monitor_losses.append(float(val_ema))
        if val_ema < best_val - float(min_delta):
            best_val = float(val_ema)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch == 1 or epoch % max(1, int(log_every)) == 0 or epoch == int(epochs):
            print(
                f"[gaussian_network {epoch:4d}/{int(epochs)}] train_nll={train_loss:.6f} "
                f"val_nll={val_loss:.6f} val_smooth={val_ema:.6f} best_smooth={best_val:.6f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
        if int(patience) > 0 and patience_counter >= int(patience):
            stopped_early = True
            stopped_epoch = int(epoch)
            print(
                f"[gaussian_network early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_smooth={best_val:.6f} patience={int(patience)}",
                flush=True,
            )
            break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[gaussian_network restore-best] restored epoch={best_epoch} val_smooth={best_val:.6f}", flush=True)

    param_norm_sq = 0.0
    with torch.no_grad():
        for p in model.parameters():
            v = float(torch.linalg.vector_norm(p.detach()).cpu())
            param_norm_sq += v * v
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_monitor_losses": val_monitor_losses,
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
        "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else float("nan"),
        "grad_norm_max": float(np.max(grad_norms)) if grad_norms else float("nan"),
        "param_norm_final": float(math.sqrt(param_norm_sq)),
        "n_clipped_steps": int(n_clipped),
        "n_total_steps": int(n_steps),
        "lr_last": float(opt.param_groups[0]["lr"]),
    }


def compute_gaussian_network_c_matrix(
    *,
    model: ConditionalGaussianPrecisionMLP
    | ConditionalDiagonalGaussianPrecisionMLP
    | ConditionalLowRankGaussianCovarianceMLP,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    device: torch.device,
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
            logp = model.log_prob(x_t, theta_t)
            c[i0:i1, :] = logp.reshape(b, n).detach().cpu().numpy().astype(np.float64)
    return c
