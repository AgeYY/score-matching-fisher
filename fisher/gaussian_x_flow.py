r"""Gaussian-parameterized conditional density trained via analytic flow-matching velocity.

Trains \(p_\phi(x\mid\theta)=\mathcal N(\mu_\phi(\theta), L_\phi(\theta)L_\phi(\theta)^\top)\)
with lower-triangular covariance Cholesky \(L_\phi\) using the FM objective between
noise \(x_0\sim\mathcal N(0,I)\) and data \(x_1\), with pluggable affine probability paths
\(x_t=a_t x_0 + b_t x_1\).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_2d_float64(a: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D.")
    return arr


class GaussianAffinePathSchedule(ABC):
    r"""Affine bridge \(x_t = a(t) x_0 + b(t) x_1\) with scalar \(t\in[0,1]\)."""

    @abstractmethod
    def ab_ad_bd(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``a, b, adot, bdot`` broadcastable with ``t`` shape ``(B,1)`` or ``(B,)``."""


class LinearAffinePathSchedule(GaussianAffinePathSchedule):
    r"""\(a=1-t,\; b=t\) (straight bridge)."""

    def ab_ad_bd(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        one = torch.ones_like(t)
        a = one - t
        b = t
        ad = -one
        bd = one
        return a, b, ad, bd


class CosineAffinePathSchedule(GaussianAffinePathSchedule):
    r"""\(a=\cos(\pi t/2),\; b=\sin(\pi t/2)\)."""

    def ab_ad_bd(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        half_pi = 0.5 * math.pi
        a = torch.cos(half_pi * t)
        b = torch.sin(half_pi * t)
        ad = -half_pi * torch.sin(half_pi * t)
        bd = half_pi * torch.cos(half_pi * t)
        return a, b, ad, bd


def path_schedule_from_name(name: str) -> GaussianAffinePathSchedule:
    k = str(name).strip().lower()
    if k in ("linear", "straight"):
        return LinearAffinePathSchedule()
    if k in ("cosine", "cos"):
        return CosineAffinePathSchedule()
    raise ValueError(f"Unknown gxf path schedule: {name!r}; use linear or cosine.")


class ConditionalGaussianCovarianceFMMLP(nn.Module):
    r"""Outputs \(\mu(\theta)\) and lower-triangular covariance Cholesky \(L(\theta)\)."""

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

    def covariance(self, theta: torch.Tensor) -> torch.Tensor:
        _, l = self.forward(theta)
        return torch.bmm(l, l.transpose(1, 2))

    def log_prob_normalized(self, x_norm: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        r"""Log-density for \(x\) in the **normalized** coordinate system used in training."""
        if x_norm.ndim == 1:
            x_norm = x_norm.unsqueeze(0)
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        if x_norm.shape[0] != theta.shape[0]:
            raise ValueError("x and theta batch sizes must match.")
        mu, l = self.forward(theta)
        diff = x_norm - mu
        z = torch.linalg.solve_triangular(l, diff.unsqueeze(-1), upper=False).squeeze(-1)
        quad = torch.sum(z * z, dim=1)
        diag_l = torch.diagonal(l, dim1=-2, dim2=-1)
        log_det_sigma = 2.0 * torch.sum(torch.log(torch.clamp(diag_l, min=1e-12)), dim=1)
        return -0.5 * quad - 0.5 * log_det_sigma - 0.5 * float(self.x_dim) * math.log(2.0 * math.pi)

    def log_prob_observed(
        self,
        x_raw: torch.Tensor,
        theta: torch.Tensor,
        *,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
    ) -> torch.Tensor:
        r"""Log \(p(x_{\mathrm{raw}}\mid\theta)\) with affine normalization \(z=(x-\mu_{\mathrm{stat}})/\sigma_{\mathrm{stat}}\)."""
        z = (x_raw - x_mean) / x_std
        logjac = -torch.sum(torch.log(x_std))
        return self.log_prob_normalized(z, theta) + logjac


class ConditionalDiagonalGaussianCovarianceFMMLP(nn.Module):
    r"""Outputs \(\mu(\theta)\) and **diagonal** covariance Cholesky \(L(\theta)=\mathrm{diag}(\sigma(\cdot)+\texttt{diag\_floor})\).

    Uses the same FM objective and likelihood interface as :class:`ConditionalGaussianCovarianceFMMLP`, but
    parameterizes only \(d\) diagonal entries per batch (full \(L\) is stored as a batched matrix with zeros off-diagonal).
    """

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
        raw_diag = out[:, self.x_dim :]
        diag = torch.nn.functional.softplus(raw_diag) + self.diag_floor
        batch = int(theta.shape[0])
        l = torch.zeros(batch, self.x_dim, self.x_dim, dtype=out.dtype, device=out.device)
        idx = torch.arange(self.x_dim, device=out.device)
        l[:, idx, idx] = diag
        return mu, l

    def covariance(self, theta: torch.Tensor) -> torch.Tensor:
        _, l = self.forward(theta)
        return torch.bmm(l, l.transpose(1, 2))

    def log_prob_normalized(self, x_norm: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        r"""Log-density for \(x\) in the **normalized** coordinate system used in training."""
        if x_norm.ndim == 1:
            x_norm = x_norm.unsqueeze(0)
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        if x_norm.shape[0] != theta.shape[0]:
            raise ValueError("x and theta batch sizes must match.")
        mu, l = self.forward(theta)
        diff = x_norm - mu
        z = torch.linalg.solve_triangular(l, diff.unsqueeze(-1), upper=False).squeeze(-1)
        quad = torch.sum(z * z, dim=1)
        diag_l = torch.diagonal(l, dim1=-2, dim2=-1)
        log_det_sigma = 2.0 * torch.sum(torch.log(torch.clamp(diag_l, min=1e-12)), dim=1)
        return -0.5 * quad - 0.5 * log_det_sigma - 0.5 * float(self.x_dim) * math.log(2.0 * math.pi)

    def log_prob_observed(
        self,
        x_raw: torch.Tensor,
        theta: torch.Tensor,
        *,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
    ) -> torch.Tensor:
        r"""Log \(p(x_{\mathrm{raw}}\mid\theta)\) with affine normalization \(z=(x-\mu_{\mathrm{stat}})/\sigma_{\mathrm{stat}}\)."""
        z = (x_raw - x_mean) / x_std
        logjac = -torch.sum(torch.log(x_std))
        return self.log_prob_normalized(z, theta) + logjac


GaussianXFlowModel = Union[ConditionalGaussianCovarianceFMMLP, ConditionalDiagonalGaussianCovarianceFMMLP]


def analytic_gaussian_fm_velocity(
    *,
    xt: torch.Tensor,
    t: torch.Tensor,
    mu: torch.Tensor,
    l_cov: torch.Tensor,
    schedule: GaussianAffinePathSchedule,
    cov_jitter: float,
) -> torch.Tensor:
    r"""Analytic velocity \(v_\phi(x_t,t,\theta)\) for Gaussian marginal path (batched)."""
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    b, d = xt.shape
    a, bcoef, ad, bd = schedule.ab_ad_bd(t)
    if a.ndim == 1:
        a = a.unsqueeze(-1)
        bcoef = bcoef.unsqueeze(-1)
        ad = ad.unsqueeze(-1)
        bd = bd.unsqueeze(-1)
    sigma = torch.bmm(l_cov, l_cov.transpose(1, 2))
    mt = bcoef * mu
    zvec = xt - mt
    aa = a * ad
    bb = bcoef * bd
    eye = torch.eye(d, dtype=xt.dtype, device=xt.device).unsqueeze(0).expand(b, d, d)
    c_t = (a * a).view(b, 1, 1) * eye + (bcoef * bcoef).view(b, 1, 1) * sigma + float(cov_jitter) * eye
    r = torch.linalg.cholesky(c_t)
    y = torch.cholesky_solve(zvec.unsqueeze(-1), r).squeeze(-1)
    sig_y = torch.bmm(sigma, y.unsqueeze(-1)).squeeze(-1)
    v = bd * mu + aa * y + bb * sig_y
    return v


def train_gaussian_x_flow(
    *,
    model: GaussianXFlowModel,
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
    cov_jitter: float = 1e-4,
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
    te = float(t_eps)
    if not (0.0 < te < 0.5):
        raise ValueError("t_eps must be in (0, 0.5).")
    if float(cov_jitter) <= 0.0:
        raise ValueError("cov_jitter must be > 0.")

    th_tr = _as_2d_float64(theta_train, name="theta_train")
    x_tr = _as_2d_float64(x_train, name="x_train")
    th_va = _as_2d_float64(theta_val, name="theta_val")
    x_va = _as_2d_float64(x_val, name="x_val")
    if th_tr.shape[0] < 1 or th_va.shape[0] < 1:
        raise ValueError("gaussian_x_flow requires non-empty train and validation splits.")
    d = int(x_tr.shape[1])
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
    best_state: dict[str, torch.Tensor] | None = None
    val_ema: float | None = None
    patience_counter = 0
    stopped_early = False
    stopped_epoch = int(epochs)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        ep_losses: list[float] = []
        for tb, x1b in train_loader:
            tb = tb.to(device)
            x1b = x1b.to(device)
            bs = int(x1b.shape[0])
            t_raw = torch.rand(bs, 1, device=device, dtype=x1b.dtype)
            t = float(te) + (1.0 - 2.0 * float(te)) * t_raw
            x0b = torch.randn_like(x1b)
            a, bcoef, ad, bd = schedule.ab_ad_bd(t)
            xt = a * x0b + bcoef * x1b
            ut = ad * x0b + bd * x1b
            mu, l_cov = model(tb)
            v = analytic_gaussian_fm_velocity(
                xt=xt,
                t=t,
                mu=mu,
                l_cov=l_cov,
                schedule=schedule,
                cov_jitter=float(cov_jitter),
            )
            loss = torch.mean(torch.sum((v - ut) ** 2, dim=1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(max_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
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
                t_raw = torch.rand(bs, 1, device=device, dtype=x1b.dtype)
                t = float(te) + (1.0 - 2.0 * float(te)) * t_raw
                x0b = torch.randn_like(x1b)
                a, bcoef, ad, bd = schedule.ab_ad_bd(t)
                xt = a * x0b + bcoef * x1b
                ut = ad * x0b + bd * x1b
                mu, l_cov = model(tb)
                v = analytic_gaussian_fm_velocity(
                    xt=xt,
                    t=t,
                    mu=mu,
                    l_cov=l_cov,
                    schedule=schedule,
                    cov_jitter=float(cov_jitter),
                )
                loss_b = torch.mean(torch.sum((v - ut) ** 2, dim=1))
                val_ep.append(float(loss_b.detach().cpu()))
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
                f"[gaussian_x_flow {epoch:4d}/{int(epochs)}] train_fm={train_loss:.6f} "
                f"val_fm={val_loss:.6f} val_smooth={val_ema:.6f} best_smooth={best_val:.6f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
        if int(patience) > 0 and patience_counter >= int(patience):
            stopped_early = True
            stopped_epoch = int(epoch)
            print(
                f"[gaussian_x_flow early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_smooth={best_val:.6f} patience={int(patience)}",
                flush=True,
            )
            break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[gaussian_x_flow restore-best] restored epoch={best_epoch} val_smooth={best_val:.6f}", flush=True)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_monitor_losses": val_monitor_losses,
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
        "lr_last": float(opt.param_groups[0]["lr"]),
        "x_mean": x_mean.astype(np.float64),
        "x_std": x_std.astype(np.float64),
    }


def compute_gaussian_x_flow_c_matrix(
    *,
    model: GaussianXFlowModel,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    device: torch.device,
    x_mean: np.ndarray,
    x_std: np.ndarray,
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
            logp = model.log_prob_observed(x_t, theta_t, x_mean=x_mean_t, x_std=x_std_t)
            c[i0:i1, :] = logp.reshape(b, n).detach().cpu().numpy().astype(np.float64)
    return c
