"""Normalizing-flow helpers for Hellinger-style H-matrix estimation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    import zuko  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - exercised by runtime environments without zuko
    zuko = None


def require_zuko_for_nf() -> None:
    """Raise a clear error when NF support dependency is unavailable."""
    if zuko is None:
        raise RuntimeError(
            "Normalizing-flow method requires the optional dependency 'zuko'. "
            "Install it in the active environment (e.g., `mamba run -n geo_diffusion pip install zuko`)."
        )


class ConditionalThetaNF(nn.Module):
    """Conditional NSF model for p(theta | x)."""

    def __init__(
        self,
        *,
        x_dim: int,
        context_dim: int,
        hidden_dim: int,
        transforms: int,
    ) -> None:
        super().__init__()
        require_zuko_for_nf()
        self.encoder = nn.Sequential(
            nn.Linear(int(x_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(context_dim)),
        )
        self.flow = zuko.flows.NSF(  # type: ignore[union-attr]
            features=1,
            context=int(context_dim),
            transforms=int(transforms),
            hidden_features=[int(hidden_dim), int(hidden_dim)],
        )

    def distribution(self, x: torch.Tensor) -> torch.distributions.Distribution:
        return self.flow(self.encoder(x))

    def log_prob(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.distribution(x).log_prob(theta.reshape(-1, 1))


def train_conditional_nf(
    *,
    model: ConditionalThetaNF,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    min_delta: float,
    ema_alpha: float,
) -> dict[str, Any]:
    """Train NF by conditional NLL with EMA-monitored early stopping."""
    xtr = torch.from_numpy(np.asarray(x_train, dtype=np.float32)).to(device)
    ttr = torch.from_numpy(np.asarray(theta_train, dtype=np.float32).reshape(-1, 1)).to(device)
    xva = torch.from_numpy(np.asarray(x_val, dtype=np.float32)).to(device)
    tva = torch.from_numpy(np.asarray(theta_val, dtype=np.float32).reshape(-1, 1)).to(device)
    ntr = int(xtr.shape[0])
    if ntr < 1:
        raise ValueError("NF training requires at least one training sample.")
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_ema_losses: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_ema = float("inf")
    best_epoch = 0
    bad = 0
    ema: float | None = None

    for ep in range(1, int(epochs) + 1):
        model.train()
        idx = torch.randint(0, ntr, (int(batch_size),), device=device)
        loss = -model.log_prob(ttr[idx], xtr[idx]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        tr = float(loss.detach().cpu().item())
        train_losses.append(tr)

        model.eval()
        with torch.no_grad():
            va = float((-model.log_prob(tva, xva).mean()).detach().cpu().item())
        val_losses.append(va)
        ema = va if ema is None else (float(ema_alpha) * va + (1.0 - float(ema_alpha)) * float(ema))
        val_ema_losses.append(float(ema))

        if float(ema) < (best_ema - float(min_delta)):
            best_ema = float(ema)
            best_epoch = ep
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        if bad >= int(patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses": np.asarray(train_losses, dtype=np.float64),
        "val_losses": np.asarray(val_losses, dtype=np.float64),
        "val_ema_losses": np.asarray(val_ema_losses, dtype=np.float64),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(len(train_losses)),
        "best_val_ema": float(best_ema),
    }


def compute_c_matrix_nf(
    *,
    model: ConditionalThetaNF,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    device: torch.device,
    pair_batch_size: int,
) -> np.ndarray:
    """Compute C[i, j] = log p(theta_j | x_i) in manageable blocks."""
    theta_vec = np.asarray(theta_all, dtype=np.float64).reshape(-1)
    x_mat = np.asarray(x_all, dtype=np.float64)
    n = int(theta_vec.shape[0])
    if x_mat.shape[0] != n:
        raise ValueError(f"theta/x size mismatch for NF C-matrix: {n} vs {x_mat.shape[0]}.")
    max_pairs = max(1, int(pair_batch_size))
    row_bs = max(1, min(n, int(np.sqrt(max_pairs))))
    col_bs = max(1, min(n, max_pairs // row_bs))

    c = np.empty((n, n), dtype=np.float64)
    theta_t = torch.from_numpy(theta_vec.astype(np.float32))
    x_t = torch.from_numpy(x_mat.astype(np.float32))

    model.eval()
    with torch.no_grad():
        for i0 in range(0, n, row_bs):
            i1 = min(n, i0 + row_bs)
            x_blk = x_t[i0:i1].to(device)
            bi = int(i1 - i0)
            for j0 in range(0, n, col_bs):
                j1 = min(n, j0 + col_bs)
                theta_blk = theta_t[j0:j1].to(device)
                bj = int(j1 - j0)
                x_rep = x_blk.repeat_interleave(bj, dim=0)
                theta_rep = theta_blk.repeat(bi).reshape(-1, 1)
                lp = model.log_prob(theta_rep, x_rep).reshape(bi, bj)
                c[i0:i1, j0:j1] = lp.detach().cpu().numpy().astype(np.float64, copy=False)
    return c


def compute_delta_l(c_matrix: np.ndarray) -> np.ndarray:
    c = np.asarray(c_matrix, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("C matrix must be square.")
    d = np.diag(c).reshape(-1, 1)
    return c - d


def compute_h_directed(delta_l: np.ndarray) -> np.ndarray:
    dl = np.asarray(delta_l, dtype=np.float64)
    h = 1.0 - (1.0 / np.cosh(0.5 * dl))
    np.fill_diagonal(h, 0.0)
    return np.clip(h, 0.0, 1.0)


def symmetrize(h_directed: np.ndarray) -> np.ndarray:
    h = np.asarray(h_directed, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("H matrix must be square.")
    return 0.5 * (h + h.T)
