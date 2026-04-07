"""Kernel-smoothed stimulus decoder (SSSD): soft bin targets and sigma-conditioned decoder.

Gaussian kernel on linear theta: q_sigma(b|theta) = Phi((e_{b+1}-theta)/sigma) - Phi((e_b-theta)/sigma).

Symmetric discrimination matrix (paper):
  M_ij(sigma) = 1/2 E_{x|bin i}[log f(i|x,s)/f(j|x,s)] + 1/2 E_{x|bin j}[log f(j|x,s)/f(i|x,s)].
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Bin geometry (linear) — matches visualize_*_binned scripts
# ---------------------------------------------------------------------------


def theta_bin_edges_from_arrays(
    theta_used: np.ndarray,
    meta: dict,
    n_bins: int,
    mode: str,
) -> tuple[np.ndarray, float, float]:
    """Equal-width edges on [lo, hi]; mode 'range' or 'meta_range'."""
    th = np.asarray(theta_used, dtype=np.float64).reshape(-1)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")
    if mode == "range":
        lo = float(np.min(th))
        hi = float(np.max(th))
    elif mode == "meta_range":
        lo = float(meta["theta_low"])
        hi = float(meta["theta_high"])
    else:
        raise ValueError(f"Unknown theta-bin mode: {mode}")
    if hi <= lo:
        raise ValueError(f"Invalid theta range for binning: [{lo}, {hi}]")
    edges = np.linspace(lo, hi, n_bins + 1, dtype=np.float64)
    return edges, lo, hi


def theta_to_bin_index(theta: np.ndarray, edges: np.ndarray, n_bins: int) -> np.ndarray:
    th = np.asarray(theta, dtype=np.float64).reshape(-1)
    idx = np.searchsorted(edges, th, side="right") - 1
    return np.clip(idx, 0, n_bins - 1).astype(np.int64)


def default_sigma_training_range_from_theta(theta: np.ndarray) -> tuple[float, float]:
    """Default ``(sigma_min, sigma_max)`` for SSSD training-σ sampling.

    ``sigma_max`` is the empirical standard deviation of the stimulus/label ``theta``;
    ``sigma_min`` is ``sigma_max / 8``. If ``std(theta)`` is zero or non-finite, falls back
    to a small scale derived from the range of ``theta``.
    """
    th = np.asarray(theta, dtype=np.float64).reshape(-1)
    if th.size == 0:
        raise ValueError("theta is empty.")
    theta_std = float(np.std(th))
    if not np.isfinite(theta_std) or theta_std <= 0.0:
        span = float(np.ptp(th))
        theta_std = max(span / max(len(th), 1), 1e-8)
    smax = theta_std
    smin = smax / 8.0
    return smin, smax


# ---------------------------------------------------------------------------
# Soft targets q_sigma(b | theta) — Gaussian kernel on R
# ---------------------------------------------------------------------------


def soft_bin_probs_gaussian_numpy(
    theta: np.ndarray,
    edges: np.ndarray,
    sigma: float | np.ndarray,
) -> np.ndarray:
    """Soft bin masses under N(theta, sigma^2) integrated on each bin interval.

    Parameters
    ----------
    theta : (N,)
    edges : (B+1,)
    sigma : positive scalar or (N,) per-sample bandwidth

    Returns
    -------
    q : (N, B) row-stochastic
    """
    from scipy.stats import norm

    th = np.asarray(theta, dtype=np.float64).reshape(-1)
    e = np.asarray(edges, dtype=np.float64).reshape(-1)
    b = int(e.size) - 1
    if b < 1:
        raise ValueError("edges must have length >= 2.")
    sig = np.asarray(sigma, dtype=np.float64)
    if sig.ndim == 0:
        sig = np.full(th.shape[0], float(sig), dtype=np.float64)
    else:
        sig = sig.reshape(-1)
        if sig.shape[0] != th.shape[0]:
            raise ValueError("sigma must be scalar or same length as theta.")
    if np.any(sig <= 0.0):
        raise ValueError("sigma must be positive.")

    e_left = e[:-1].reshape(1, -1)
    e_right = e[1:].reshape(1, -1)
    th_col = th.reshape(-1, 1)
    sig_col = sig.reshape(-1, 1)
    z_r = (e_right - th_col) / sig_col
    z_l = (e_left - th_col) / sig_col
    q = norm.cdf(z_r) - norm.cdf(z_l)
    # numerical cleanup
    q = np.clip(q, 0.0, 1.0)
    s = np.sum(q, axis=1, keepdims=True)
    s = np.maximum(s, 1e-12)
    q = q / s
    return q.astype(np.float64)


def soft_bin_probs_gaussian_torch(
    theta: torch.Tensor,
    edges: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Vectorized soft bin probs; theta (N,), edges (B+1), sigma (N,) or scalar tensor."""
    from torch.distributions import Normal

    n = int(theta.shape[0])
    e = edges.reshape(-1)
    b = int(e.numel()) - 1
    if b < 1:
        raise ValueError("edges must have length >= 2.")
    if sigma.numel() == 1:
        sig = sigma.reshape(1).expand(n)
    else:
        sig = sigma.reshape(-1)
        if sig.shape[0] != n:
            raise ValueError("sigma must be scalar or length N.")
    std = Normal(
        loc=torch.zeros((), device=theta.device, dtype=theta.dtype),
        scale=torch.ones((), device=theta.device, dtype=theta.dtype),
    )
    e_left = e[:-1].unsqueeze(0).expand(n, -1)
    e_right = e[1:].unsqueeze(0).expand(n, -1)
    th = theta.reshape(-1, 1).expand(n, b)
    sig_e = sig.reshape(-1, 1).clamp(min=1e-12)
    z_r = (e_right - th) / sig_e
    z_l = (e_left - th) / sig_e
    q = std.cdf(z_r) - std.cdf(z_l)
    q = q.clamp(min=0.0)
    s = q.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return q / s


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Mean over batch of -sum_b q_b log pi_b."""
    log_pi = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_pi).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Sigma-conditioned decoder f_phi(b | x, sigma)
# ---------------------------------------------------------------------------


class SigmaConditionedBinDecoder(nn.Module):
    """MLP: [x, encoded(log sigma)] -> logits over B bins."""

    def __init__(
        self,
        x_dim: int,
        n_bins: int,
        hidden_dim: int = 128,
        depth: int = 3,
        sigma_embed_dim: int = 16,
    ) -> None:
        super().__init__()
        if x_dim < 1:
            raise ValueError("x_dim must be >= 1.")
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2 for multiclass decoder.")
        self.n_bins = int(n_bins)
        self.sigma_encoder = nn.Sequential(
            nn.Linear(1, sigma_embed_dim),
            nn.SiLU(),
            nn.Linear(sigma_embed_dim, sigma_embed_dim),
            nn.SiLU(),
        )
        in_dim = x_dim + sigma_embed_dim
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(int(depth)):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.SiLU())
            d = hidden_dim
        layers.append(nn.Linear(d, n_bins))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """x: (N, x_dim), log_sigma: (N, 1) natural log of sigma."""
        z = self.sigma_encoder(log_sigma)
        h = torch.cat([x, z], dim=-1)
        return self.net(h)


def forward_logits(
    model: SigmaConditionedBinDecoder,
    x: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """sigma: (N,) or (N,1) positive. Not under no_grad — used for training."""
    if sigma.dim() == 1:
        sigma = sigma.unsqueeze(-1)
    log_s = torch.log(sigma.clamp(min=1e-12))
    return model(x, log_s)


def sample_training_sigmas(
    batch_size: int,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
) -> torch.Tensor:
    """Log-uniform in [sigma_min, sigma_max], shape (batch_size, 1)."""
    if sigma_min <= 0.0 or sigma_max <= 0.0:
        raise ValueError("sigma_min and sigma_max must be positive.")
    lo = min(float(sigma_min), float(sigma_max))
    hi = max(float(sigma_min), float(sigma_max))
    u = torch.rand((batch_size, 1), device=device)
    log_sigma = math.log(lo) + u * (math.log(hi) - math.log(lo))
    return torch.exp(log_sigma)


@dataclass
class SSSDTrainResult:
    train_losses: list[float]
    best_state_dict: dict
    best_epoch: int
    stopped_epoch: int
    stopped_early: bool


def train_sssd_decoder(
    theta: np.ndarray,
    x: np.ndarray,
    edges: np.ndarray,
    *,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
    epochs: int = 10000,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    depth: int = 3,
    val_frac: float = 0.1,
    patience: int = 1000,
    seed: int = 0,
    log_every: int = 20,
) -> tuple[SigmaConditionedBinDecoder, SSSDTrainResult]:
    """Train sigma-conditioned decoder with soft targets and sampled sigma.

    Optimization: AdamW on soft cross-entropy between Gaussian bin targets q_σ(b|θ) and softmax
    outputs. Each minibatch draws σ log-uniformly in [sigma_min, sigma_max].

    A random fraction ``val_frac`` of samples is held out for validation loss (same objective,
    no dropout). **Early stopping:** if ``patience`` > 0 and a validation set exists, training
    stops when validation loss does not improve for ``patience`` consecutive epochs (measured
    against the best val so far). At the end, weights are restored to the epoch with **lowest**
    validation loss (``best_epoch``). If ``patience`` <= 0 or there is no val split, training runs
    for the full ``epochs``.
    """
    th = np.asarray(theta, dtype=np.float64).reshape(-1)
    x2 = np.asarray(x, dtype=np.float64)
    if x2.ndim != 2 or x2.shape[0] != th.shape[0]:
        raise ValueError("theta and x row count mismatch.")
    n_bins = int(len(edges) - 1)
    if n_bins < 2:
        raise ValueError("Need at least 2 bins.")

    n = int(th.shape[0])
    rng = np.random.default_rng(seed)
    if n < 2:
        fit_idx = np.arange(n, dtype=np.int64)
        val_idx = np.array([], dtype=np.int64)
    else:
        perm = rng.permutation(n)
        n_val = max(1, int(round(float(val_frac) * n)))
        n_val = min(n_val, n - 1)
        val_idx = perm[:n_val]
        fit_idx = perm[n_val:]

    theta_fit = th[fit_idx]
    x_fit = x2[fit_idx]
    theta_val = th[val_idx] if val_idx.size > 0 else np.zeros(0, dtype=np.float64)
    x_val = x2[val_idx] if val_idx.size > 0 else np.zeros((0, x2.shape[1]), dtype=np.float64)
    n_val = int(theta_val.shape[0])

    edges_t = torch.from_numpy(edges.astype(np.float32)).to(device)

    model = SigmaConditionedBinDecoder(
        x_dim=int(x2.shape[1]),
        n_bins=n_bins,
        hidden_dim=hidden_dim,
        depth=depth,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def run_epoch(
        theta_np: np.ndarray,
        x_np: np.ndarray,
        train: bool,
    ) -> float:
        model.train(train)
        n_loc = int(theta_np.shape[0])
        if n_loc == 0:
            return float("nan")
        order = rng.permutation(n_loc) if train else np.arange(n_loc)
        losses: list[float] = []
        for start in range(0, n_loc, batch_size):
            idx = order[start : start + batch_size]
            tb = torch.from_numpy(theta_np[idx].astype(np.float32)).to(device)
            xb = torch.from_numpy(x_np[idx].astype(np.float32)).to(device)
            bs = int(tb.shape[0])
            sig = sample_training_sigmas(bs, sigma_min, sigma_max, device).squeeze(-1)
            q = soft_bin_probs_gaussian_torch(tb, edges_t, sig)
            logits = forward_logits(model, xb, sig)
            loss = soft_cross_entropy(logits, q)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(losses)) if losses else float("nan")

    train_losses: list[float] = []
    best_val = float("inf")
    best_state: dict = {}
    best_ep = 0
    val_patience = int(patience)
    no_improve = 0
    stopped_early = False
    stopped_at = int(epochs)

    for ep in range(1, int(epochs) + 1):
        tr = run_epoch(theta_fit, x_fit, train=True)
        train_losses.append(tr)
        va = float("nan")
        if n_val > 0 and theta_val.shape[0] > 0:
            with torch.no_grad():
                va = run_epoch(theta_val, x_val, train=False)
            if va < best_val:
                best_val = va
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_ep = ep
                no_improve = 0
            else:
                if val_patience > 0:
                    no_improve += 1
        else:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep

        if log_every > 0 and (ep % log_every == 0 or ep == 1):
            msg = f"[sssd] epoch {ep}/{epochs} train_loss={tr:.6f}"
            if np.isfinite(va):
                msg += f" val_loss={va:.6f}"
            if n_val > 0 and val_patience > 0:
                msg += f" val_no_improve={no_improve}/{val_patience}"
            print(msg)

        if (
            n_val > 0
            and theta_val.shape[0] > 0
            and val_patience > 0
            and no_improve >= val_patience
        ):
            stopped_early = True
            stopped_at = ep
            print(f"[sssd] early stopping at epoch {ep} (val plateau {val_patience} epochs).")
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, SSSDTrainResult(
        train_losses=train_losses,
        best_state_dict=best_state,
        best_epoch=best_ep,
        stopped_epoch=stopped_at,
        stopped_early=stopped_early,
    )


@torch.no_grad()
def decoder_log_probs(
    model: SigmaConditionedBinDecoder,
    x: np.ndarray,
    sigma: float,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """log f(b|x,sigma) for all rows, shape (N, B)."""
    x2 = np.asarray(x, dtype=np.float64)
    n = int(x2.shape[0])
    model.eval()
    out: list[np.ndarray] = []
    sig_t = torch.full((1, 1), float(sigma), device=device)
    for start in range(0, n, batch_size):
        xb = torch.from_numpy(x2[start : start + batch_size].astype(np.float32)).to(device)
        m = xb.shape[0]
        sig_batch = sig_t.expand(m, 1)
        logits = forward_logits(model, xb, sig_batch)
        lp = F.log_softmax(logits, dim=-1)
        out.append(lp.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


def symmetric_discrimination_matrix_M(
    log_probs: np.ndarray,
    bin_idx: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """M_ij from log_probs (N,B) and hard bin_idx (N,). Diagonal NaN."""
    lp = np.asarray(log_probs, dtype=np.float64)
    bi = np.asarray(bin_idx, dtype=np.int64).reshape(-1)
    if lp.ndim != 2 or lp.shape[1] != n_bins:
        raise ValueError("log_probs must be (N, n_bins).")
    if bi.shape[0] != lp.shape[0]:
        raise ValueError("bin_idx length must match log_probs rows.")

    m = np.full((n_bins, n_bins), np.nan, dtype=np.float64)
    for i in range(n_bins):
        for j in range(n_bins):
            if i == j:
                m[i, j] = np.nan
                continue
            mi = np.flatnonzero(bi == i)
            mj = np.flatnonzero(bi == j)
            if mi.size == 0 or mj.size == 0:
                continue
            # E_{x|i}[log f(i)/f(j)]
            lpi_i = lp[mi, i] - lp[mi, j]
            # E_{x|j}[log f(j)/f(i)]
            lpj_j = lp[mj, j] - lp[mj, i]
            m[i, j] = 0.5 * (float(np.mean(lpi_i)) + float(np.mean(lpj_j)))
    return m


def symmetric_lr_accuracy_matrix(
    log_probs: np.ndarray,
    bin_idx: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Symmetric pairwise accuracy from LR sign: same structure as M_ij but in [0, 1].

    For i != j: among x with true bin i, accuracy vs j is mean 1[log f(i|x) > log f(j|x)].
    Among x with true bin j, accuracy vs i is mean 1[log f(j|x) > log f(i|x)].
    Report A_ij = 0.5 * (acc_from_bin_i + acc_from_bin_j). Diagonal NaN.

    In-sample (full batch); ties (log f(i) == log f(j)) count as incorrect for both strict >.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    bi = np.asarray(bin_idx, dtype=np.int64).reshape(-1)
    if lp.ndim != 2 or lp.shape[1] != n_bins:
        raise ValueError("log_probs must be (N, n_bins).")
    if bi.shape[0] != lp.shape[0]:
        raise ValueError("bin_idx length must match log_probs rows.")

    a = np.full((n_bins, n_bins), np.nan, dtype=np.float64)
    for i in range(n_bins):
        for j in range(n_bins):
            if i == j:
                a[i, j] = np.nan
                continue
            mi = np.flatnonzero(bi == i)
            mj = np.flatnonzero(bi == j)
            if mi.size == 0 or mj.size == 0:
                continue
            correct_i = (lp[mi, i] > lp[mi, j]).astype(np.float64)
            correct_j = (lp[mj, j] > lp[mj, i]).astype(np.float64)
            a[i, j] = 0.5 * (float(np.mean(correct_i)) + float(np.mean(correct_j)))
    return a


def add_sssd_cli_arguments(p: argparse.ArgumentParser) -> None:
    """Shared CLI flags for SSSD (kernel-smoothed decoder + pairwise metrics)."""
    p.add_argument(
        "--no-sssd",
        action="store_true",
        default=False,
        help="Skip kernel-smoothed decoder (SSSD) training and M_ij(sigma) outputs.",
    )
    p.add_argument(
        "--sssd-sigma-min",
        type=float,
        default=None,
        help=(
            "SSSD: minimum kernel bandwidth σ for training sampling. "
            "Default: σ_max/8 with σ_max = std(theta). If only --sssd-sigma-max is set, min = max/8."
        ),
    )
    p.add_argument(
        "--sssd-sigma-max",
        type=float,
        default=None,
        help=(
            "SSSD: maximum σ for training sampling and auto eval grid. "
            "Default: std(theta). If only --sssd-sigma-min is set, max = 8*min."
        ),
    )
    p.add_argument(
        "--sssd-sigmas",
        type=str,
        default="auto",
        help="Comma-separated evaluation sigmas for SSSD metrics, or 'auto' for log-spaced grid.",
    )
    p.add_argument(
        "--sssd-n-sigmas",
        type=int,
        default=5,
        help="When --sssd-sigmas=auto, number of evaluation sigmas between min and max.",
    )
    p.add_argument(
        "--sssd-epochs",
        type=int,
        default=10000,
        help="SSSD maximum training epochs (default 10000).",
    )
    p.add_argument(
        "--sssd-patience",
        type=int,
        default=1000,
        help="SSSD early stopping: stop if validation loss does not improve for this many epochs "
        "(default 1000). 0 disables early stopping.",
    )
    p.add_argument("--sssd-batch-size", type=int, default=256, help="SSSD batch size.")
    p.add_argument("--sssd-lr", type=float, default=1e-3, help="SSSD AdamW learning rate.")
    p.add_argument("--sssd-hidden-dim", type=int, default=128, help="SSSD MLP hidden width.")
    p.add_argument("--sssd-depth", type=int, default=3, help="SSSD MLP depth (Linear+SiLU blocks).")
    p.add_argument(
        "--sssd-val-frac",
        type=float,
        default=0.1,
        help="SSSD held-out validation fraction for early model selection.",
    )
    p.add_argument(
        "--sssd-seed",
        type=int,
        default=-1,
        help="SSSD RNG seed; -1 uses dataset seed from NPZ.",
    )
    p.add_argument(
        "--sssd-log-every",
        type=int,
        default=50,
        help="Print SSSD training every N epochs (0 to silence).",
    )


def parse_sigma_list(s: str) -> list[float]:
    """Comma-separated positive floats, e.g. '0.02,0.05,0.1'."""
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    out: list[float] = []
    for p in parts:
        v = float(p)
        if v <= 0.0:
            raise ValueError(f"sigma must be positive, got {v}")
        out.append(v)
    if not out:
        raise ValueError("Empty sigma list.")
    return out


def sanity_check_soft_targets(
    theta: np.ndarray,
    edges: np.ndarray,
    sigma_small: float,
    sigma_large: float,
) -> None:
    """Assert q sums to 1 and small-sigma concentrates near hard bin."""
    th = np.asarray(theta, dtype=np.float64).reshape(-1)
    n_bins = len(edges) - 1
    q_small = soft_bin_probs_gaussian_numpy(th[: min(50, th.shape[0])], edges, sigma_small)
    assert np.allclose(q_small.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)
    assert np.all(q_small >= -1e-9)
    hard = theta_to_bin_index(th[: q_small.shape[0]], edges, n_bins)
    mass_on_hard = q_small[np.arange(q_small.shape[0]), hard]
    # very small sigma -> most mass on own bin
    assert float(np.mean(mass_on_hard)) > 0.5, "expected small sigma to put mass on hard bin"
    q_big = soft_bin_probs_gaussian_numpy(th[: q_small.shape[0]], edges, sigma_large)
    assert np.allclose(q_big.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)
