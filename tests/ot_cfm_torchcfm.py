#!/usr/bin/env python3
"""2D bimodal unconditional CFM vs OT-CFM (TorchCFM) benchmark — standalone, no pytest.

Trains a velocity field with ``ConditionalFlowMatcher`` (independent pairing) and
``ExactOptimalTransportConditionalFlowMatcher`` (minibatch exact OT) under matched
settings, then samples via forward Euler and writes metrics + figures.

Dependencies: ``torch``, ``torchcfm`` (``pip install torch torchcfm``), ``matplotlib``,
optional ``tqdm`` for a progress bar.

Default device is **CUDA** (``--device cuda``). Use ``--device cpu`` only when you intend CPU.

A **train+val** i.i.d. pool (size ``--target-pool-size``) is split into train and validation only; a **separately drawn** i.i.d. test set of fixed size (``--test-size``, default 1000) is used for plots, ``bimodal`` ``target`` metrics, and MMD. The model trains on a fixed set (shuffled in epochs, no replacement per epoch). **Val** is only for EMA early stopping when ``--early-stopping-patience > 0``; if patience is 0, train and val are merged for optimization (the **test** set remains independent for evaluation).

Default artifacts: ``$DATAROOT/tests/ot_cfm_torchcfm/`` (see ``global_setting.DATAROOT``).

Figures: combined + per-method training loss, target scatter, scatter vs generated models,
2D count heatmaps (``distribution_grid``, ``target_distribution``, ``cfm_distribution``, ``ot_cfm_distribution``);
single-page ``combined_report`` (loss + training / test / generated scatters; heatmaps remain in ``distribution_*``); paths in ``summary.json`` under ``figures``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec as mgs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import DATAROOT

try:
    from torchcfm.conditional_flow_matching import (
        ConditionalFlowMatcher,
        ExactOptimalTransportConditionalFlowMatcher,
    )
except ImportError as _e:  # pragma: no cover
    raise SystemExit(
        "The `torchcfm` package is required for this script. Install with:\n"
        "  pip install torchcfm\n"
        f"Import error: {_e}"
    ) from _e

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(x: Any, **_: Any) -> Any:
        return x


@dataclass
class TrainStats:
    """Per-method training / validation monitor (validation only if early stopping is on)."""

    train_losses: list[float]
    val_step: list[int]
    val_mse: list[float]
    val_smoothed: list[float]
    n_steps: int
    best_step: int
    best_val_smoothed: float
    early_stopped: bool
    n_train_points: int
    n_epochs: int
    cumulative_target_points_seen: int
    train_sampler: str


@dataclass
class RunConfig:
    mode1: tuple[float, float]
    mode2: tuple[float, float]
    mode_std: float
    batch_size: int
    train_steps: int
    learning_rate: float
    weight_decay: float
    sigma: float
    hidden_dim: int
    n_mlp_blocks: int
    time_frequencies: int
    ode_steps: int
    n_gen: int
    n_eval: int
    seed: int
    middle_band: float
    device: str


def set_seed(seed: int, device: torch.device) -> torch.Generator:
    s = int(seed)
    np.random.seed(s)
    torch.manual_seed(s)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    # `torch.randint(..., device=cuda, generator=...)` requires a CUDA generator
    g = torch.Generator(device=device)
    g.manual_seed(s)
    return g


def _default_out_dir() -> Path:
    return Path(DATAROOT) / "tests" / "ot_cfm_torchcfm"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2D bimodal CFM vs OT-CFM (TorchCFM) benchmark.")
    p.add_argument("--output-dir", type=str, default=str(_default_out_dir()), help="Directory for all outputs.")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Use cuda (default) or cpu.")
    p.add_argument("--seed", type=int, default=7, help="Random seed.")
    p.add_argument(
        "--mode1",
        type=float,
        nargs=2,
        default=[-2.0, 0.0],
        metavar=("X", "Y"),
        help="First Gaussian mode mean (2D).",
    )
    p.add_argument(
        "--mode2",
        type=float,
        nargs=2,
        default=[2.0, 0.0],
        metavar=("X", "Y"),
        help="Second Gaussian mode mean (2D).",
    )
    p.add_argument(
        "--mode-std",
        type=float,
        default=0.75,
        help="Isotropic std per mode (default: 0.75 = 3× the prior 0.25 setting).",
    )
    p.add_argument("--batch-size", type=int, default=256, help="Minibatch size for training.")
    p.add_argument("--train-steps", type=int, default=4000, help="Optimization steps per method.")
    p.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate.")
    p.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    p.add_argument(
        "--sigma",
        type=float,
        default=0.4,
        help="Path noise for TorchCFM Gaussian bridge (higher = blurrier targets).",
    )
    p.add_argument("--hidden-dim", type=int, default=128, help="MLP hidden size.")
    p.add_argument("--n-mlp-blocks", type=int, default=3, help="Number of MLP hidden blocks.")
    p.add_argument(
        "--time-frequencies",
        type=int,
        default=8,
        help="Number of log-spaced sin/cos frequencies for t embedding (output dim 4*f).",
    )
    p.add_argument("--ode-steps", type=int, default=200, help="Euler steps for generation (t: 0->1).")
    p.add_argument("--n-gen", type=int, default=4096, help="Number of generated samples for metrics/plots.")
    p.add_argument(
        "--target-pool-size",
        type=int,
        default=20,
        help="Size of the train+val i.i.d. pool only; split into train and val (N>=2).",
    )
    p.add_argument(
        "--test-size",
        type=int,
        default=1000,
        help="Size of a separate i.i.d. test draw for MMD and target visualizations (not taken from the train+val pool).",
    )
    p.add_argument(
        "--n-eval",
        type=int,
        default=10000,
        help="Cap on how many test points to use for target visuals (min with --test-size).",
    )
    p.add_argument(
        "--middle-band",
        type=float,
        default=0.4,
        help="Band |x0| < this counts as 'middle' occupancy (artifact diagnostic).",
    )
    p.add_argument(
        "--run",
        type=str,
        default="both",
        choices=["both", "cfm", "ot_cfm"],
        help="Which matcher(s) to train.",
    )
    p.add_argument(
        "--dist-bins",
        type=int,
        default=80,
        help="2D histogram bin count per axis for distribution heatmaps.",
    )
    p.add_argument(
        "--plot-max-points",
        type=int,
        default=20_000,
        help="Max points per series used for density histograms (subsampled for speed).",
    )
    p.add_argument(
        "--n-mmd",
        type=int,
        default=2048,
        help="Subsample size per cloud for RBF MMD² vs. test (combined subtitle + summary).",
    )
    p.add_argument(
        "--train-fraction",
        type=float,
        default=0.6,
        help="Weight for splitting the train+val pool (normalized against --val-fraction).",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Weight for splitting the train+val pool (normalized against --train-fraction; test is a separate i.i.d. draw).",
    )
    p.add_argument(
        "--val-every",
        type=int,
        default=5,
        help="Run a full validation pass every this many training steps (EMA / early stopping).",
    )
    p.add_argument(
        "--val-ema-beta",
        type=float,
        default=0.95,
        help="EMA for validation MSE: ema = beta*ema + (1-beta)*val_mse.",
    )
    p.add_argument(
        "--val-min-delta",
        type=float,
        default=0.0,
        help="Min improvement in smoothed val MSE to reset early-stopping patience (>= 0).",
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=200,
        help="Stop after this many val checks with no EMA val improvement. 0 = off: train+val merged for optimization; test still held out for target visuals/metrics.",
    )
    return p.parse_args()


def split_train_val(
    pool: torch.Tensor,
    train_fraction: float,
    val_fraction: float,
    g: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """One random permutation, then consecutive slices: train, val (uses full pool). Weights are normalized."""
    n = int(pool.shape[0])
    if n < 2:
        raise ValueError(
            f"Need at least 2 i.i.d. points in the train+val pool; got n={n} (increase --target-pool-size)."
        )
    ft = float(train_fraction)
    fv = float(val_fraction)
    s = abs(ft) + abs(fv)
    if s <= 0.0 or not (math.isfinite(ft) and math.isfinite(fv)):
        raise ValueError(
            f"--train-fraction and --val-fraction must be non-zero with at least one positive; got {ft}, {fv}."
        )
    w_tr, w_va = abs(ft) / s, abs(fv) / s
    n_tr = max(1, int(round(n * w_tr)))
    n_va = n - n_tr
    if n_va < 1:
        n_va = 1
        n_tr = n - 1
    if n_tr < 1:
        n_tr = 1
        n_va = n - 1
    if n_tr < 1 or n_va < 1 or n_tr + n_va != n:
        raise ValueError("Split failed: need positive train and val sizes covering the whole pool.")
    perm = torch.randperm(n, device=device, generator=g)
    sh = pool[perm]
    tr = sh[:n_tr]
    va = sh[n_tr : n_tr + n_va]
    return tr, va, n_tr, n_va


@torch.no_grad()
def _eval_val_flow_mse(
    model: VelocityMLP,
    matcher: ConditionalFlowMatcher,
    val_pool: torch.Tensor,
    batch_size: int,
    g: torch.Generator,
    device: torch.device,
) -> float:
    """Mean MSE( pred, u_t ) over the val pool (one flow draw per x1, same as training)."""
    n = int(val_pool.shape[0])
    if n < 1:
        return float("nan")
    model.eval()
    tot = 0.0
    c = 0
    bs = max(1, int(batch_size))
    for s in range(0, n, bs):
        e = min(s + bs, n)
        b = e - s
        x1 = val_pool[s:e]
        x0 = torch.randn(b, 2, generator=g, device=device, dtype=x1.dtype)
        t, xt, ut = matcher.sample_location_and_conditional_flow(x0, x1)
        pred = model(xt, t)
        loss = F.mse_loss(pred, ut, reduction="sum")
        tot += float(loss.detach().cpu().item())
        c += b
    model.train()
    return tot / float(max(c, 1))


def train_velocity(
    name: str,
    matcher: ConditionalFlowMatcher,
    model: VelocityMLP,
    train_data: torch.Tensor,
    val_pool: torch.Tensor | None,
    cfg: RunConfig,
    g: torch.Generator,
    device: torch.device,
    *,
    max_steps: int,
    use_validation: bool,
    val_every: int,
    patience: int,
    val_ema_beta: float,
    val_min_delta: float,
) -> TrainStats:
    """Train on fixed `train_data` with epoch shuffles and no replacement; val for EMA early stop only."""
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    train_losses: list[float] = []
    val_step: list[int] = []
    val_mse: list[float] = []
    val_smoothed: list[float] = []
    n_data = int(train_data.shape[0])
    if n_data < 1:
        raise ValueError(f"[{name}] training data is empty")
    bsz = int(cfg.batch_size)
    vpool = val_pool
    do_val = bool(use_validation) and vpool is not None and int(vpool.shape[0]) >= 1
    if use_validation and not do_val:
        raise ValueError(
            f"[{name}] early stopping requested but validation pool is empty; "
            "increase --target-pool-size or adjust --train-fraction / --val-fraction."
        )
    val_ema: float | None = None
    best_ema = float("inf")
    best_state: dict[str, torch.Tensor] = {}
    bad = 0
    n_steps = 0
    best_step = 0
    early_stopped = False
    p = int(max(0, int(patience)))
    ve = max(1, int(val_every))
    min_delta = max(0.0, float(val_min_delta))
    beta = float(val_ema_beta)
    if beta < 0.0 or beta >= 1.0:
        raise ValueError("--val-ema-beta must be in [0, 1).")
    max_steps = int(max_steps)
    n_epochs = 0
    cumulative_target_points_seen = 0
    pbar = tqdm(total=max_steps, desc=f"train[{name}]", mininterval=0.5)
    while n_steps < max_steps:
        n_epochs += 1
        perm = torch.randperm(n_data, device=device, generator=g)
        for s in range(0, n_data, bsz):
            if n_steps >= max_steps:
                break
            e = min(s + bsz, n_data)
            B = e - s
            idx = perm[s:e]
            x1 = train_data.index_select(0, idx)
            x0 = torch.randn(B, 2, generator=g, device=device, dtype=x1.dtype)
            t, xt, ut = matcher.sample_location_and_conditional_flow(x0, x1)
            pred = model(xt, t)
            loss = F.mse_loss(pred, ut)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            v = float(loss.detach().cpu().item())
            train_losses.append(v)
            n_steps += 1
            cumulative_target_points_seen += B
            pbar.update(1)
            pbar.set_postfix(loss=f"{v:.2e}", refresh=False)
            if do_val and p > 0 and n_steps % ve == 0:
                v_m = _eval_val_flow_mse(model, matcher, vpool, bsz, g, device)
                if val_ema is None:
                    val_ema = v_m
                else:
                    val_ema = beta * val_ema + (1.0 - beta) * v_m
                val_step.append(n_steps)
                val_mse.append(v_m)
                val_smoothed.append(float(val_ema))
                if val_ema < best_ema - min_delta:
                    best_ema = float(val_ema)
                    best_step = n_steps
                    bad = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= p:
                        early_stopped = True
                        break
            if early_stopped:
                break
        if early_stopped or n_steps >= max_steps:
            break
    pbar.close()
    sm = "epoch_shuffle_no_replace"
    if not do_val or p <= 0:
        return TrainStats(
            train_losses=train_losses,
            val_step=[],
            val_mse=[],
            val_smoothed=[],
            n_steps=n_steps,
            best_step=n_steps,
            best_val_smoothed=float("nan"),
            early_stopped=False,
            n_train_points=n_data,
            n_epochs=n_epochs,
            cumulative_target_points_seen=cumulative_target_points_seen,
            train_sampler=sm,
        )
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        out_step = int(best_step)
        out_ema = float(best_ema)
    else:
        out_step = int(n_steps)
        out_ema = float("nan")
    return TrainStats(
        train_losses=train_losses,
        val_step=val_step,
        val_mse=val_mse,
        val_smoothed=val_smoothed,
        n_steps=n_steps,
        best_step=out_step,
        best_val_smoothed=out_ema,
        early_stopped=early_stopped,
        n_train_points=n_data,
        n_epochs=n_epochs,
        cumulative_target_points_seen=cumulative_target_points_seen,
        train_sampler=sm,
    )


def sinusoidal_time_features(t: torch.Tensor, n_freq: int) -> torch.Tensor:
    """t: (B,) in [0,1] -> (B, 4*n_freq) sin/cos features."""
    t = t.float().unsqueeze(1) * 2.0 * math.pi
    freqs = 2.0 ** torch.arange(int(n_freq), device=t.device, dtype=t.dtype)
    ang = t * freqs.unsqueeze(0)
    return torch.cat([torch.sin(ang), torch.cos(ang), torch.sin(2.0 * ang), torch.cos(2.0 * ang)], dim=1)


class VelocityMLP(nn.Module):
    """Velocity v_theta(x, t) for 2D data."""

    def __init__(
        self,
        *,
        data_dim: int,
        time_freqs: int,
        hidden: int,
        n_blocks: int,
    ) -> None:
        super().__init__()
        t_in = 4 * int(time_freqs)
        self._time_freqs = int(time_freqs)
        in_d = int(data_dim) + t_in
        layers: list[nn.Module] = [nn.Linear(in_d, hidden), nn.SiLU()]
        for _ in range(max(0, int(n_blocks) - 1)):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, int(data_dim))]
        self._net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.view(1).expand(x.shape[0])
        te = sinusoidal_time_features(t, self._time_freqs)
        z = torch.cat([x, te], dim=1)
        return self._net(z)


@torch.no_grad()
def sample_euler(
    model: VelocityMLP,
    n: int,
    ode_steps: int,
    g: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    x = torch.randn(int(n), 2, generator=g, device=device)
    for k in range(int(ode_steps)):
        t0 = float(k) / float(ode_steps)
        t1 = float(k + 1) / float(ode_steps)
        tb = torch.full((x.shape[0],), t0, device=device, dtype=x.dtype)
        v = model(x, tb)
        x = x + (t1 - t0) * v
    return x


def bimodal_metrics(
    x: np.ndarray,
    true_means: np.ndarray,
    middle_band: float,
) -> dict[str, Any]:
    """Per-mode mass, middle-band fraction, and mean distance to true centers (by sign split)."""
    # Assign by nearest true mean (2 modes)
    d0 = ((x - true_means[0:1]) ** 2).sum(axis=1)
    d1 = ((x - true_means[1:2]) ** 2).sum(axis=1)
    ass = (d0 > d1).astype(np.int64)
    n0 = int((ass == 0).sum())
    n1 = int((ass == 1).sum())
    mid = float((np.abs(x[:, 0]) < float(middle_band)).mean())
    c0 = x[ass == 0].mean(axis=0) if n0 else np.full(2, np.nan, dtype=np.float64)
    c1 = x[ass == 1].mean(axis=0) if n1 else np.full(2, np.nan, dtype=np.float64)
    err0 = float(np.linalg.norm(c0 - true_means[0])) if n0 else float("nan")
    err1 = float(np.linalg.norm(c1 - true_means[1])) if n1 else float("nan")
    return {
        "n_mode0": n0,
        "n_mode1": n1,
        "p_mid": mid,
        "est_center0": c0.tolist(),
        "est_center1": c1.tolist(),
        "center_error0": err0,
        "center_error1": err1,
    }


def rbf_mmd2(
    x_ref: np.ndarray,
    x_gen: np.ndarray,
    *,
    n_mmd: int,
    rng: np.random.Generator,
) -> tuple[float, dict[str, float]]:
    """Biased MMD² (RBF) between samples x_ref and x_gen. Lower is a better match to the ref cloud.

    Bandwidth: median of pairwise **Euclidean** distances on the pooled (subsampled) union.
    """
    if x_ref.ndim != 2 or x_gen.ndim != 2 or int(x_ref.shape[1]) != int(x_gen.shape[1]):
        return float("nan"), {}
    n0, n1 = int(x_ref.shape[0]), int(x_gen.shape[0])
    if n0 < 2 or n1 < 2:
        return float("nan"), {"n_ref": float(n0), "n_gen": float(n1)}

    cap = max(2, int(n_mmd))

    def _ss(x: np.ndarray) -> np.ndarray:
        n = int(x.shape[0])
        m = min(n, cap)
        if m >= n:
            return np.asarray(x, dtype=np.float64)
        idx = rng.permutation(n)[:m]
        return np.asarray(x[idx], dtype=np.float64)

    X = _ss(x_ref)
    Y = _ss(x_gen)
    nx, ny = int(X.shape[0]), int(Y.shape[0])

    Z = np.vstack([X, Y])
    nz = int(Z.shape[0])
    d2_zz = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=-1)
    iu = np.triu_indices(nz, k=1)
    dist_e = np.sqrt(np.maximum(d2_zz[iu], 0.0))
    sigma = max(float(np.median(dist_e)), 1e-8)
    sig2 = 2.0 * sigma**2

    d2_xx = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1)
    d2_yy = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(axis=-1)
    d2_xy = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=-1)
    kxx = np.exp(-d2_xx / sig2)
    kyy = np.exp(-d2_yy / sig2)
    kxy = np.exp(-d2_xy / sig2)
    txx = float(kxx.sum()) / float(nx * nx)
    tyy = float(kyy.sum()) / float(ny * ny)
    txy = float(kxy.sum()) / float(nx * ny)
    mmd2 = txx + tyy - 2.0 * txy
    if not math.isfinite(mmd2):
        mmd2 = float("nan")
    else:
        mmd2 = max(0.0, mmd2)

    meta: dict[str, float] = {
        "mmd2_n_ref": float(nx),
        "mmd2_n_gen": float(ny),
        "mmd2_rbf_sigma": float(sigma),
    }
    return mmd2, meta


def _plot_losses(
    cfm_losses: list[float] | None,
    ot_losses: list[float] | None,
    out_path: Path,
    *,
    cfm_val: tuple[list[int], list[float]] | None = None,
    ot_val: tuple[list[int], list[float]] | None = None,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0), dpi=120, layout="tight")
    if cfm_losses:
        ax.plot(
            np.arange(1, len(cfm_losses) + 1),
            cfm_losses,
            color="#1f77b4",
            lw=1.1,
            label="CFM train (batch MSE)",
        )
    if cfm_val and cfm_val[0]:
        ax.plot(
            cfm_val[0],
            cfm_val[1],
            color="#6baed6",
            lw=1.0,
            ls="--",
            label="CFM val (EMA on full val pass)",
        )
    if ot_losses:
        ax.plot(
            np.arange(1, len(ot_losses) + 1),
            ot_losses,
            color="#d62728",
            lw=1.1,
            label="OT-CFM train (batch MSE)",
        )
    if ot_val and ot_val[0]:
        ax.plot(
            ot_val[0],
            ot_val[1],
            color="#f28d8d",
            lw=1.0,
            ls="--",
            label="OT-CFM val (EMA on full val pass)",
        )
    ax.set_xlabel("training step (epoch = one optimizer step here)")
    ax.set_ylabel("MSE (velocity error)")
    ax.set_yscale("log")
    ax.set_title("CFM and OT-CFM training (same architecture / hyperparams)")
    ax.legend(loc="best", fontsize=7)
    _save_fig_png_svg(fig, out_path, dpi=140)
    plt.close(fig)


def _plot_loss_single(
    losses: list[float],
    out_stem: Path,
    *,
    color: str,
    method_title: str,
    val: tuple[list[int], list[float]] | None = None,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.3), dpi=120, layout="tight")
    ax.plot(
        np.arange(1, len(losses) + 1), losses, color=color, lw=1.1, label="train (batch MSE)"
    )
    if val and val[0]:
        ax.plot(
            val[0],
            val[1],
            color="#888888",
            lw=1.0,
            ls="--",
            label="val EMA (full val pass)",
        )
    ax.set_xlabel("training step (epoch = one optimizer step here)")
    ax.set_ylabel("MSE (velocity error)")
    ax.set_yscale("log")
    ax.set_title(f"{method_title} — training loss (MSE vs step)")
    ax.legend(loc="best", fontsize=7)
    _save_fig_png_svg(fig, out_stem, dpi=140)
    plt.close(fig)


def _save_fig_png_svg(fig: plt.Figure, out_stem: Path, *, dpi: int = 150) -> None:
    p = out_stem
    if p.suffix in (".png", ".svg"):
        p = p.with_suffix("")
    fig.savefig(str(p.with_suffix(".png")), dpi=dpi)
    fig.savefig(str(p.with_suffix(".svg")))


def _hist2d_density(
    xy: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n_bins: int,
) -> np.ndarray:
    """2D count histogram with shape (n_bins, n_bins) for imshow (rows = y, cols = x)."""
    h, _, _ = np.histogram2d(
        xy[:, 0],
        xy[:, 1],
        bins=(int(n_bins), int(n_bins)),
        range=((x_min, x_max), (y_min, y_max)),
    )
    return h.T


def _subsample_xy(x: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    n = int(x.shape[0])
    if n <= int(max_n):
        return x
    idx = rng.choice(n, size=int(max_n), replace=False)
    return x[idx]


def _plot_distribution_heatmap(
    title: str,
    xy: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n_bins: int,
    ax: Any,
    vmax: float,
) -> Any:
    h = _hist2d_density(
        xy, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, n_bins=n_bins
    )
    im = ax.imshow(
        h,
        origin="lower",
        extent=(x_min, x_max, y_min, y_max),
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=float(vmax),
    )
    ax.set_xlabel(r"$x_0$", fontsize=8)
    ax.set_ylabel(r"$x_1$", fontsize=8)
    ax.set_title(title, fontsize=9)
    return im


def _shared_extent_2d(arrays: list[np.ndarray], pad: float = 0.4) -> tuple[float, float, float, float]:
    a = np.concatenate(arrays, axis=0)
    return (
        float(a[:, 0].min() - pad),
        float(a[:, 0].max() + pad),
        float(a[:, 1].min() - pad),
        float(a[:, 1].max() + pad),
    )


def _vmax_from_histograms(
    arrs: list[np.ndarray],
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n_bins: int,
) -> float:
    vm = 0.0
    for xy in arrs:
        h = _hist2d_density(xy, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, n_bins=n_bins)
        vm = max(vm, float(h.max()))
    return max(vm, 1.0)


def _plot_distribution_panels(
    panels: list[tuple[str, np.ndarray]],
    means: np.ndarray,
    out_stem: Path,
    *,
    n_bins: int,
    max_points: int,
    rng: np.random.Generator,
) -> None:
    """2D bin-count heatmaps with shared axis limits and shared vmax."""
    if not panels:
        return
    arrs = [_subsample_xy(a, int(max_points), rng) for _, a in panels]
    x_min, x_max, y_min, y_max = _shared_extent_2d(arrs)
    vmax = _vmax_from_histograms(
        arrs, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, n_bins=n_bins
    )
    n = len(panels)
    fig, axes = plt.subplots(
        1, n, figsize=(3.0 * n + 0.6, 3.2), dpi=120, constrained_layout=True
    )
    if n == 1:
        axes = [axes]
    last_im: Any = None
    for ax, (title, _orig), a_sub in zip(axes, panels, arrs):
        last_im = _plot_distribution_heatmap(
            title,
            a_sub,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            n_bins=n_bins,
            ax=ax,
            vmax=vmax,
        )
        ax.scatter(means[:, 0], means[:, 1], c="w", s=8, zorder=5, edgecolors="C3", linewidths=0.5)
    cbar = fig.colorbar(last_im, ax=axes, orientation="vertical", fraction=0.04, pad=0.04)
    cbar.set_label("count / bin", fontsize=7)
    _save_fig_png_svg(fig, out_stem, dpi=150)
    plt.close(fig)


def _scatter_on_ax(
    ax: Any,
    d: np.ndarray | None,
    title: str,
    means: np.ndarray,
    *,
    max_show: int,
    subtitle: str | None = None,
) -> None:
    if d is None or d.size == 0:
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"{title}\n(not run)", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        return
    dd = d[: int(max_show)]
    ax.scatter(dd[:, 0], dd[:, 1], s=1.2, c="#444444", alpha=0.32, rasterized=True)
    ax.axhline(0.0, color="k", lw=0.2, alpha=0.2)
    ax.axvline(0.0, color="k", lw=0.2, alpha=0.2)
    ax.scatter(means[:, 0], means[:, 1], c="C3", s=14, zorder=3, label="mode means")
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=8.5, pad=3)
    else:
        ax.set_title(title, fontsize=9)
    ax.set_xlabel(r"$x_0$", fontsize=8)
    ax.set_ylabel(r"$x_1$", fontsize=8)
    ax.set_aspect("equal", adjustable="box")


def _ax_placeholder(ax: Any, text: str) -> None:
    ax.set_axis_off()
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=8, color="0.45")


def _plot_combined_report(
    *,
    cfm_losses: list[float] | None,
    ot_losses: list[float] | None,
    train_viz: np.ndarray,
    target_viz: np.ndarray,
    cfm_gen_np: np.ndarray | None,
    ot_gen_np: np.ndarray | None,
    means: np.ndarray,
    out_stem: Path,
    scatter_max: int,
    n_train_val_pool: int,
    n_test_total: int,
    batch_size: int,
    cfm_subtitle: str | None = None,
    ot_subtitle: str | None = None,
    cfm_val: tuple[list[int], list[float]] | None = None,
    ot_val: tuple[list[int], list[float]] | None = None,
) -> None:
    """One figure: training loss and four scatter columns: train, test, CFM, OT (no separate density row)."""
    fig = plt.figure(figsize=(14.0, 8.0), dpi=120, constrained_layout=False)
    grid = mgs.GridSpec(2, 4, figure=fig, height_ratios=[0.9, 1.0], hspace=0.48, wspace=0.28)
    fig.subplots_adjust(left=0.06, right=0.95, top=0.9, bottom=0.1)

    ax_loss = fig.add_subplot(grid[0, :])
    if cfm_losses:
        ax_loss.plot(
            np.arange(1, len(cfm_losses) + 1),
            cfm_losses,
            color="#1f77b4",
            lw=1.2,
            label="CFM train",
        )
    if cfm_val and cfm_val[0]:
        ax_loss.plot(
            cfm_val[0],
            cfm_val[1],
            color="#6baed6",
            lw=1.0,
            ls="--",
            label="CFM val (EMA)",
        )
    if ot_losses:
        ax_loss.plot(
            np.arange(1, len(ot_losses) + 1),
            ot_losses,
            color="#d62728",
            lw=1.2,
            label="OT-CFM train",
        )
    if ot_val and ot_val[0]:
        ax_loss.plot(
            ot_val[0],
            ot_val[1],
            color="#f28d8d",
            lw=1.0,
            ls="--",
            label="OT val (EMA)",
        )
    if not cfm_losses and not ot_losses:
        _ax_placeholder(ax_loss, "No training losses (nothing to plot).")
    else:
        ax_loss.set_xlabel("training step (one opt step = one step here)")
        ax_loss.set_ylabel("MSE (velocity error)")
        ax_loss.set_yscale("log")
        ax_loss.legend(loc="best", fontsize=7)
    ax_loss.set_title("Training + validation (EMA) loss", fontsize=10)

    titles_sc = (
        "Training (x₁ for loss)",
        r"Test  $p_{data}$  (hold-out)",
        "CFM generated",
        "OT-CFM generated",
    )
    datas_sc: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None] = (
        train_viz,
        target_viz,
        cfm_gen_np,
        ot_gen_np,
    )
    ax0 = fig.add_subplot(grid[1, 0])
    axs_sc: list[Any] = [ax0]
    for j in range(1, 4):
        axs_sc.append(fig.add_subplot(grid[1, j], sharex=ax0, sharey=ax0))
    subs: tuple[str | None, str | None, str | None, str | None] = (None, None, cfm_subtitle, ot_subtitle)
    for j, (tit, d, sub) in enumerate(zip(titles_sc, datas_sc, subs)):
        _scatter_on_ax(axs_sc[j], d, tit, means, max_show=scatter_max, subtitle=sub)

    # One global (x, y) box for all scatter and density panels
    for_xy: list[np.ndarray] = [train_viz, target_viz]
    if cfm_gen_np is not None and cfm_gen_np.size:
        for_xy.append(cfm_gen_np)
    if ot_gen_np is not None and ot_gen_np.size:
        for_xy.append(ot_gen_np)
    sx0, sx1, sy0, sy1 = _shared_extent_2d(for_xy)
    for ax in axs_sc:
        ax.set_xlim(sx0, sx1)
        ax.set_ylim(sy0, sy1)

    n_train_shown = min(int(scatter_max), int(train_viz.shape[0]))
    n_shown = int(target_viz.shape[0])
    n_trv = int(n_train_val_pool)
    n_tte = int(n_test_total)
    b = int(batch_size)
    n_c = len(cfm_losses) if cfm_losses else 0
    n_o = len(ot_losses) if ot_losses else 0
    fig.suptitle(
        "OT-CFM toy: training loss and samples",
        fontsize=12,
        y=0.98,
    )
    cap = (
        f"Train+val pool: {n_trv:,} i.i.d. (2-mode mix), split train/val. "
        f"Test: {n_tte:,} i.i.d. hold-out. "
        f"Scatter columns: training x₁ (n={n_train_shown} shown), then test (n={n_shown}, cap n-eval). "
        f"Minibatch B={b}; opt steps: CFM {n_c}, OT {n_o} (cumulative x1-uses in summary). "
        f"Densities: distribution_grid / target_ / cfm_ / ot_cfm_*. "
    )
    fig.text(0.5, 0.02, cap, ha="center", va="bottom", fontsize=7.5, color="0.25")

    _save_fig_png_svg(fig, out_stem, dpi=160)
    plt.close(fig)


def _plot_scatter_grid(
    train_np: np.ndarray,
    target_np: np.ndarray,
    cfm_np: np.ndarray | None,
    ot_np: np.ndarray | None,
    means: np.ndarray,
    out_path: Path,
    *,
    max_show: int = 4000,
) -> None:
    ncols = 2 + (1 if cfm_np is not None else 0) + (1 if ot_np is not None else 0)
    fig, ax_arr = plt.subplots(1, ncols, figsize=(3.0 * ncols, 3.1), dpi=120, sharex=True, sharey=True, layout="tight")
    ax_list: list[Any] = [ax_arr] if ncols == 1 else [np.ravel(ax_arr)[j] for j in range(ncols)]
    def _sc(ax: Any, d: np.ndarray, title: str) -> None:
        d = d[: int(max_show)]
        ax.scatter(d[:, 0], d[:, 1], s=2, c="#444444", alpha=0.35, rasterized=True)
        ax.axhline(0.0, color="k", lw=0.2, alpha=0.2)
        ax.axvline(0.0, color="k", lw=0.2, alpha=0.2)
        ax.scatter(means[:, 0], means[:, 1], c="C3", s=20, zorder=3, label="true modes")
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal", adjustable="box")
    _sc(ax_list[0], train_np, "Training (x₁)")
    _sc(ax_list[1], target_np, "Test (hold-out)")
    k = 2
    if cfm_np is not None:
        _sc(ax_list[k], cfm_np, "CFM gen.")
        k += 1
    if ot_np is not None:
        _sc(ax_list[k], ot_np, "OT-CFM gen.")
    for ax in ax_list:
        ax.set_xlabel(r"$x_0$", fontsize=8)
        ax.set_ylabel(r"$x_1$", fontsize=8)
    _save_fig_png_svg(fig, out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Pass --device cpu or set up CUDA.")
    pool_size = int(args.target_pool_size)
    if pool_size < 2:
        raise ValueError(f"--target-pool-size must be >= 2, got {pool_size}.")
    test_n = int(args.test_size)
    if test_n < 1:
        raise ValueError(f"--test-size must be >= 1, got {test_n}.")

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    g = set_seed(int(args.seed), device)

    means = torch.tensor([args.mode1, args.mode2], dtype=torch.float32, device=device)
    # Train+val pool: i.i.d. mixture, split into train/val only; then a separate i.i.d. test draw of fixed size.
    idx = torch.randint(0, 2, (pool_size,), generator=g, device=device)
    noise = torch.randn(pool_size, 2, generator=g, device=device) * float(args.mode_std)
    target_pool = means[idx] + noise
    t_idx = torch.randint(0, 2, (test_n,), generator=g, device=device)
    t_noise = torch.randn(test_n, 2, generator=g, device=device) * float(args.mode_std)
    test_pool = means[t_idx] + t_noise

    cfg = RunConfig(
        mode1=(float(args.mode1[0]), float(args.mode1[1])),
        mode2=(float(args.mode2[0]), float(args.mode2[1])),
        mode_std=float(args.mode_std),
        batch_size=int(args.batch_size),
        train_steps=int(args.train_steps),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        sigma=float(args.sigma),
        hidden_dim=int(args.hidden_dim),
        n_mlp_blocks=int(args.n_mlp_blocks),
        time_frequencies=int(args.time_frequencies),
        ode_steps=int(args.ode_steps),
        n_gen=int(args.n_gen),
        n_eval=int(args.n_eval),
        seed=int(args.seed),
        middle_band=float(args.middle_band),
        device=str(args.device),
    )

    true_means = np.array([cfg.mode1, cfg.mode2], dtype=np.float64)
    train_t, val_t, n_tr, n_va = split_train_val(
        target_pool,
        float(args.train_fraction),
        float(args.val_fraction),
        g,
        device,
    )
    n_te = int(test_pool.shape[0])
    es = int(args.early_stopping_patience) > 0
    if es:
        train_data = train_t
        val_pool: torch.Tensor | None = val_t
    else:
        train_data = torch.cat([train_t, val_t], dim=0)
        val_pool = None
    n_train_opt = int(train_data.shape[0])
    target_viz = test_pool[: min(int(args.n_eval), n_te)].detach().cpu().numpy()
    train_viz = train_data.detach().cpu().numpy()

    cfm_losses: list[float] | None = None
    ot_losses: list[float] | None = None
    cfm_st: TrainStats | None = None
    ot_st: TrainStats | None = None
    cfm_state: dict[str, Any] | None = None
    ot_state: dict[str, Any] | None = None

    max_steps = int(args.train_steps)
    es_kw = dict(
        max_steps=max_steps,
        use_validation=es,
        val_every=int(args.val_every),
        patience=int(args.early_stopping_patience),
        val_ema_beta=float(args.val_ema_beta),
        val_min_delta=float(args.val_min_delta),
    )
    if args.run in ("both", "cfm"):
        model_c = VelocityMLP(
            data_dim=2,
            time_freqs=cfg.time_frequencies,
            hidden=cfg.hidden_dim,
            n_blocks=cfg.n_mlp_blocks,
        ).to(device)
        fm_c = ConditionalFlowMatcher(sigma=cfg.sigma)
        cfm_st = train_velocity("cfm", fm_c, model_c, train_data, val_pool, cfg, g, device, **es_kw)
        cfm_losses = cfm_st.train_losses
        cfm_state = {
            "model": model_c.state_dict(),
            "losses": cfm_losses,
            "train_stats": asdict(cfm_st),
        }
        torch.save(
            {**cfm_state, "config": asdict(cfg)},
            out_dir / "cfm_checkpoint.pt",
        )
        del fm_c
    if args.run in ("both", "ot_cfm"):
        model_o = VelocityMLP(
            data_dim=2,
            time_freqs=cfg.time_frequencies,
            hidden=cfg.hidden_dim,
            n_blocks=cfg.n_mlp_blocks,
        ).to(device)
        fm_o = ExactOptimalTransportConditionalFlowMatcher(sigma=cfg.sigma)
        ot_st = train_velocity("ot_cfm", fm_o, model_o, train_data, val_pool, cfg, g, device, **es_kw)
        ot_losses = ot_st.train_losses
        ot_state = {
            "model": model_o.state_dict(),
            "losses": ot_losses,
            "train_stats": asdict(ot_st),
        }
        torch.save(
            {**ot_state, "config": asdict(cfg)},
            out_dir / "ot_cfm_checkpoint.pt",
        )
        del fm_o

    cfm_val_tup = (
        (cfm_st.val_step, cfm_st.val_smoothed) if cfm_st and cfm_st.val_step else None
    )
    ot_val_tup = (ot_st.val_step, ot_st.val_smoothed) if ot_st and ot_st.val_step else None

    _plot_losses(
        cfm_losses,
        ot_losses,
        out_dir / "training_loss",
        cfm_val=cfm_val_tup,
        ot_val=ot_val_tup,
    )
    if cfm_losses is not None:
        _plot_loss_single(
            cfm_losses,
            out_dir / "cfm_loss",
            color="#1f77b4",
            method_title="CFM (independent x0, x1 pairing)",
            val=cfm_val_tup,
        )
    if ot_losses is not None:
        _plot_loss_single(
            ot_losses,
            out_dir / "ot_cfm_loss",
            color="#d62728",
            method_title="OT-CFM (minibatch exact OT plan)",
            val=ot_val_tup,
        )

    cfm_gen_np: np.ndarray | None = None
    ot_gen_np: np.ndarray | None = None

    if cfm_state is not None:
        m = VelocityMLP(
            data_dim=2, time_freqs=cfg.time_frequencies, hidden=cfg.hidden_dim, n_blocks=cfg.n_mlp_blocks
        ).to(device)
        m.load_state_dict(cfm_state["model"])
        m.eval()
        cfm_gen = sample_euler(m, int(cfg.n_gen), int(cfg.ode_steps), g, device)
        cfm_gen_np = cfm_gen.detach().cpu().numpy()
    if ot_state is not None:
        m = VelocityMLP(
            data_dim=2, time_freqs=cfg.time_frequencies, hidden=cfg.hidden_dim, n_blocks=cfg.n_mlp_blocks
        ).to(device)
        m.load_state_dict(ot_state["model"])
        m.eval()
        ot_gen = sample_euler(m, int(cfg.n_gen), int(cfg.ode_steps), g, device)
        ot_gen_np = ot_gen.detach().cpu().numpy()

    m_target = bimodal_metrics(target_viz, true_means, cfg.middle_band)

    def _jsonify(a: Any) -> Any:
        if isinstance(a, (str, int, bool)) or a is None:
            return a
        if isinstance(a, float):
            if not math.isfinite(a):
                return None
            return a
        if is_dataclass(a) and not isinstance(a, type):
            return _jsonify(asdict(a))
        if isinstance(a, Path):
            return str(a)
        if isinstance(a, np.ndarray):
            return a.tolist()
        if isinstance(a, (list, tuple)):
            return [_jsonify(x) for x in a]
        if isinstance(a, dict):
            return {str(k): _jsonify(v) for k, v in a.items()}
        return str(a)

    n_c = len(cfm_losses) if cfm_losses else 0
    n_o = len(ot_losses) if ot_losses else 0
    bsz_d = int(args.batch_size)
    cum_c = int(cfm_st.cumulative_target_points_seen) if cfm_st is not None else 0
    cum_o = int(ot_st.cumulative_target_points_seen) if ot_st is not None else 0
    draw_sum = cum_c + cum_o
    tr_val_block: dict[str, Any] = {
        "n_train_val_pool": int(pool_size),
        "n_train_split": n_tr,
        "n_val_split": n_va,
        "n_test_iid": int(n_te),
        "n_test_split": int(n_te),
        "n_train_for_optimizer": n_train_opt,
        "train_fraction": float(args.train_fraction),
        "val_fraction": float(args.val_fraction),
        "train_val_merged_for_optimizer": not es,
        "target_plots_from": "independent_iid_test_draw",
        "early_stopping": es,
    }
    if es:
        tr_val_block["early_stopping_patience"] = int(args.early_stopping_patience)
        tr_val_block["val_every"] = int(args.val_every)
        tr_val_block["val_ema_beta"] = float(args.val_ema_beta)
        tr_val_block["val_min_delta"] = float(args.val_min_delta)
    summary: dict[str, Any] = {
        "config": asdict(cfg),
        "args": _jsonify(dict(vars(args))),
        "data": {
            "n_target_pool": int(pool_size),
            "n_test_size": int(test_n),
            "n_target_points_plotted": int(target_viz.shape[0]),
            "n_eval_cap": int(args.n_eval),
            "target_minibatch_size": int(args.batch_size),
            "target_train_val": tr_val_block,
            "n_target_train_points_used_in_loss": {
                "cfm": cum_c,
                "ot_cfm": cum_o,
            },
            "n_target_train_points_used_total": int(draw_sum),
            "train_sampler": "epoch_shuffle_no_replace",
            "mmd2": {
                "n_mmd": int(args.n_mmd),
                "subsample_seed_salt_cfm": 8001,
                "subsample_seed_salt_ot": 8002,
            },
        },
        "target": m_target,
    }
    cfm_mmd_sub: str | None = None
    ot_mmd_sub: str | None = None
    if cfm_gen_np is not None:
        m_c = bimodal_metrics(cfm_gen_np, true_means, cfg.middle_band)
        m2, mmeta = rbf_mmd2(
            target_viz,
            cfm_gen_np,
            n_mmd=int(args.n_mmd),
            rng=np.random.default_rng(int(args.seed) + 8001),
        )
        summary["cfm"] = {**m_c, "mmd2_to_target": m2, **mmeta}
        if math.isfinite(m2):
            cfm_mmd_sub = f"RBF MMD²={m2:.3e} (lower is better)"
        if cfm_st is not None:
            summary["cfm"]["train_run"] = asdict(cfm_st)
    if ot_gen_np is not None:
        m_o = bimodal_metrics(ot_gen_np, true_means, cfg.middle_band)
        m2o, mmeta_o = rbf_mmd2(
            target_viz,
            ot_gen_np,
            n_mmd=int(args.n_mmd),
            rng=np.random.default_rng(int(args.seed) + 8002),
        )
        summary["ot_cfm"] = {**m_o, "mmd2_to_target": m2o, **mmeta_o}
        if math.isfinite(m2o):
            ot_mmd_sub = f"RBF MMD²={m2o:.3e} (lower is better)"
        if ot_st is not None:
            summary["ot_cfm"]["train_run"] = asdict(ot_st)

    if cfm_gen_np is not None and ot_gen_np is not None:
        summary["deltas"] = {
            "p_mid_ot_minus_cfm": float(summary["ot_cfm"]["p_mid"] - summary["cfm"]["p_mid"]),
            "mean_center_err_cfm": float(
                0.5
                * (summary["cfm"]["center_error0"] + summary["cfm"]["center_error1"])
            ),
            "mean_center_err_ot": float(
                0.5
                * (summary["ot_cfm"]["center_error0"] + summary["ot_cfm"]["center_error1"])
            ),
        }

    plot_rng = np.random.default_rng(int(args.seed) + 911)

    # --- train + test scatters; generated columns when available ---
    _plot_scatter_grid(
        train_viz, target_viz, None, None, true_means, out_dir / "target_dataset_scatter"
    )

    _plot_scatter_grid(
        train_viz, target_viz, cfm_gen_np, ot_gen_np, true_means, out_dir / "scatter_target_vs_generated"
    )

    # --- 2D histogram "distribution" heatmaps (shared scale within each figure) ---
    dist_panels: list[tuple[str, np.ndarray]] = [
        ("Training (x₁ for loss)", train_viz),
        ("Test (p_data), hold-out", target_viz),
    ]
    if cfm_gen_np is not None:
        dist_panels.append(("CFM generated", cfm_gen_np))
    if ot_gen_np is not None:
        dist_panels.append(("OT-CFM generated", ot_gen_np))
    _plot_distribution_panels(
        dist_panels,
        true_means,
        out_dir / "distribution_grid",
        n_bins=int(args.dist_bins),
        max_points=int(args.plot_max_points),
        rng=plot_rng,
    )
    _plot_distribution_panels(
        [("Test (p_data), hold-out", target_viz)],
        true_means,
        out_dir / "target_distribution",
        n_bins=int(args.dist_bins),
        max_points=int(args.plot_max_points),
        rng=plot_rng,
    )
    if cfm_gen_np is not None:
        _plot_distribution_panels(
            [("CFM generated", cfm_gen_np)],
            true_means,
            out_dir / "cfm_distribution",
            n_bins=int(args.dist_bins),
            max_points=int(args.plot_max_points),
            rng=plot_rng,
        )
    if ot_gen_np is not None:
        _plot_distribution_panels(
            [("OT-CFM generated", ot_gen_np)],
            true_means,
            out_dir / "ot_cfm_distribution",
            n_bins=int(args.dist_bins),
            max_points=int(args.plot_max_points),
            rng=plot_rng,
        )

    sc_max = min(
        4000,
        max(1, int(train_viz.shape[0]), int(target_viz.shape[0])),
    )
    _plot_combined_report(
        cfm_losses=cfm_losses,
        ot_losses=ot_losses,
        train_viz=train_viz,
        target_viz=target_viz,
        cfm_gen_np=cfm_gen_np,
        ot_gen_np=ot_gen_np,
        means=true_means,
        out_stem=out_dir / "combined_report",
        scatter_max=sc_max,
        n_train_val_pool=int(pool_size),
        n_test_total=int(n_te),
        batch_size=int(args.batch_size),
        cfm_subtitle=cfm_mmd_sub,
        ot_subtitle=ot_mmd_sub,
        cfm_val=cfm_val_tup,
        ot_val=ot_val_tup,
    )

    def _art(stem: str) -> list[str]:
        return [f"{stem}.png", f"{stem}.svg"]

    summary["figures"] = {
        "output_dir": str(out_dir),
        "training_loss": _art("training_loss"),
        "cfm_loss": _art("cfm_loss") if cfm_losses is not None else None,
        "ot_cfm_loss": _art("ot_cfm_loss") if ot_losses is not None else None,
        "target_dataset_scatter": _art("target_dataset_scatter"),
        "scatter_target_vs_generated": _art("scatter_target_vs_generated"),
        "distribution_grid": _art("distribution_grid"),
        "target_distribution": _art("target_distribution"),
        "cfm_distribution": _art("cfm_distribution") if cfm_gen_np is not None else None,
        "ot_cfm_distribution": _art("ot_cfm_distribution") if ot_gen_np is not None else None,
        "combined_report": _art("combined_report"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(_jsonify(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with (out_dir / "summary_pretty.md").open("w", encoding="utf-8") as f:
        f.write("# OT-CFM run summary\n\n")
        f.write("```json\n")
        f.write(json.dumps(_jsonify(summary), indent=2, allow_nan=False))
        f.write("\n```\n")

    print(f"[ot_cfm_torchcfm] Wrote outputs under: {out_dir}", flush=True)
    print(
        f"  Train+val pool: N={int(pool_size):,}  |  train={n_tr}, val={n_va}  |  "
        f"independent test draw: {n_te:,} i.i.d.  |  "
        f"optimizer train set: {n_train_opt} (train+val merged if ES off)  |  "
        f"test pts in target plots: {int(target_viz.shape[0]):,} (min --test-size, --n-eval)  |  "
        f"per-mode std σ={float(args.mode_std):g}",
        flush=True,
    )
    if es:
        print(
            f"  Early stopping: patience={int(args.early_stopping_patience)} val checks, "
            f"val every {int(args.val_every)} steps, val EMA β={float(args.val_ema_beta):g}",
            flush=True,
        )
    else:
        print(
            "  Early stopping: off (train+val merged for optimization; test held out for plots/metrics).",
            flush=True,
        )
    if cfm_st is not None and es and math.isfinite(cfm_st.best_val_smoothed):
        print(
            f"  CFM  best EMA val MSE: {float(cfm_st.best_val_smoothed):.6e}  (step {int(cfm_st.best_step)})  |  "
            f"train steps: {int(cfm_st.n_steps)}" + ("  (early stop)" if cfm_st.early_stopped else ""),
            flush=True,
        )
    if ot_st is not None and es and math.isfinite(ot_st.best_val_smoothed):
        print(
            f"  OT   best EMA val MSE: {float(ot_st.best_val_smoothed):.6e}  (step {int(ot_st.best_step)})  |  "
            f"train steps: {int(ot_st.n_steps)}" + ("  (early stop)" if ot_st.early_stopped else ""),
            flush=True,
        )
    if cfm_losses is not None:
        print(f"  CFM  final MSE: {float(cfm_losses[-1]):.6e}  (mean last 10: {float(np.mean(cfm_losses[-10:])):.6e})", flush=True)
    if cfm_gen_np is not None and "cfm" in summary and "mmd2_to_target" in summary["cfm"]:
        m = float(summary["cfm"]["mmd2_to_target"])
        if math.isfinite(m):
            print(f"  CFM  RBF MMD² to test: {m:.6e}  (lower is better)", flush=True)
    if ot_losses is not None:
        print(f"  OT   final MSE: {float(ot_losses[-1]):.6e}  (mean last 10: {float(np.mean(ot_losses[-10:])):.6e})", flush=True)
    if ot_gen_np is not None and "ot_cfm" in summary and "mmd2_to_target" in summary["ot_cfm"]:
        m = float(summary["ot_cfm"]["mmd2_to_target"])
        if math.isfinite(m):
            print(f"  OT   RBF MMD² to test: {m:.6e}  (lower is better)", flush=True)
    if "deltas" in summary:
        print(
            f"  Δ p(middle) (OT - CFM): {summary['deltas']['p_mid_ot_minus_cfm']:.4f}  (negative is better for reducing middle fill)",
            flush=True,
        )


if __name__ == "__main__":
    main()
