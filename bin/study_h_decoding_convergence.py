#!/usr/bin/env python3
"""Convergence of binned H and pairwise decoding vs references.

**Binned H:** off-diagonal Pearson r vs a **generative ground-truth** Hellinger matrix
estimated by Monte Carlo from the known toy likelihood (see ``fisher/hellinger_gt.py`` and
``report/notes/hellinger_idea.tex``). The MC routine returns squared Hellinger **H^2**; this
script compares and visualizes **elementwise square roots** ``sqrt(H^2)`` (GT) and
``sqrt(H^sym)`` (learned binned symmetric ``h_sym``), both clipped to ``[0, 1]`` before the
square root. With ``n_bins`` = ``--num-theta-bins``, the MC count per
bin row is ``n_mc = n_ref // n_bins`` (integer floor); ``n_bins * n_mc`` may be less than ``n_ref``.
Nested subset training for each ``n`` in ``--n-list`` uses up to ``n`` samples; the ``n_ref`` column does not train a model.

**Pairwise decoding:** off-diagonal Pearson r vs the decoding matrix from the ``--n-ref``
subset (same bin edges as GT), unchanged.

**Dataset:** the loaded NPZ must match ``--dataset-family`` (default ``randamp_gaussian_sqrtd``:
random-amplitude Gaussian bumps plus ``sqrt(x_dim)`` observation-noise scaling; see ``make_dataset.py``).
Regenerate the NPZ if the family does not match.

For each ``n`` in ``--n-list``, the H matrix is computed from trained **posterior** and **prior**
models (``--theta-field-method dsm``, ``flow``, or ``flow_likelihood``). In ``flow`` mode, H uses
the **flow-derived score** (velocity-to-epsilon conversion and ``s = -eps/sigma_t``), not raw
velocity; ``flow_likelihood`` uses direct ODE likelihood ratios.
For **DSM** (``--theta-field-method dsm``), this script defaults to **multi-layer FiLM** for the
posterior score (``--score-arch film``, ``--score-depth 3``) and a **3-layer MLP** prior score
(``--prior-score-arch mlp``). For **flow** / **flow_likelihood**, theta velocity nets default to
**MLP** (``--flow-score-arch mlp``, ``--flow-prior-arch mlp``); pass ``film`` to either flag for FiLM.
The **reference column** (``n_ref``) does **not** run learned H
training: the
matrix-panel top row shows **MC generative** ``sqrt(H^2)`` (same as the H correlation
target), while the bottom row still shows pairwise decoding on the ``n_ref`` data subset.

**Visualization-only:** Pass ``--visualization-only`` to reload ``h_decoding_convergence_results.npz``
from ``--output-dir`` and regenerate figures/CSV/summary without retraining (requires a prior full run
and ``training_losses/n_*.npz`` for the loss panel).

**NPZ semantics:** Arrays ``h_binned_columns``, ``h_binned_ref``, and the key ``hellinger_gt_sq_mc``
hold **square-root** matrices for this study (legacy key name ``hellinger_gt_sq_mc``; values are
``sqrt(H^2)``, not ``H^2``).
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict

_repo_root = Path(__file__).resolve().parent.parent
_bin_dir = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_bin_dir) not in sys.path:
    sys.path.insert(0, str(_bin_dir))

# Matplotlib rcParams (tick size, spines) apply when ``global_setting`` is imported — before pyplot.
from global_setting import DATA_DIR

import matplotlib.pyplot as plt
import numpy as np
import torch

import visualize_h_matrix_binned as vhb
from fisher.cli_shared_fisher import add_estimation_arguments
from fisher.hellinger_gt import bin_centers_from_edges, estimate_hellinger_sq_one_sided_mc
from fisher.shared_dataset_io import SharedDatasetBundle, load_shared_dataset_npz
from fisher.shared_fisher_est import (
    build_dataset_from_meta,
    merge_meta_into_args,
    require_device,
    validate_estimation_args,
)


# Isolated ``h_decoding_convergence.{png,svg}`` size; combined figure uses the same width/height
# ratio for the right-hand curve column (column height matches the matrix panel height).
_H_DECODING_CURVE_FIGSIZE_IN: tuple[float, float] = (3.5, 3.5)


def _sqrt_h_like(mat: np.ndarray) -> np.ndarray:
    """Elementwise sqrt for H^2- or H_sym-like matrices in [0, 1]; preserves NaN positions."""
    h = np.asarray(mat, dtype=np.float64)
    out = np.full_like(h, np.nan, dtype=np.float64)
    finite = np.isfinite(h)
    out[finite] = np.sqrt(np.clip(h[finite], 0.0, 1.0))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Load a shared dataset .npz, train score models for each n in --n-list, then compare "
            "sqrt(binned H_sym) to sqrt(MC generative H^2) and pairwise decoding to the n_ref-subset decoding matrix. "
            "The n_ref matrix-panel column uses MC GT sqrt(H^2) for the top row (no n_ref model training). "
            "Also writes h_decoding_convergence_combined.{png,svg} (line plot + matrix panel side-by-side) "
            "and h_decoding_training_losses_panel.{png,svg} (score/prior loss vs epoch, one column per n)."
        )
    )
    p.add_argument(
        "--dataset-npz",
        type=str,
        required=True,
        help="Path to shared dataset .npz from make_dataset.py (must match --dataset-family).",
    )
    p.add_argument(
        "--dataset-family",
        type=str,
        default="randamp_gaussian_sqrtd",
        choices=[
            "cosine_gaussian",
            "cosine_gaussian_sqrtd",
            "randamp_gaussian",
            "randamp_gaussian_sqrtd",
            "cosine_gmm",
            "cos_sin_piecewise",
            "linear_piecewise",
        ],
        help=(
            "Expected generative family stored in the NPZ meta; must match make_dataset.py "
            "when the archive was created. Default: randamp_gaussian_sqrtd (random-amplitude "
            "Gaussian bumps + sqrt(x_dim) observation-noise scaling)."
        ),
    )
    p.add_argument(
        "--n-ref",
        type=int,
        default=5000,
        help=(
            "Reference subset size (default 5000). Requires len(dataset) >= n_ref and max(n-list) <= n_ref. "
            "GT Hellinger MC uses n_mc = n_ref // num_theta_bins samples per bin row (floor division)."
        ),
    )
    p.add_argument(
        "--n-list",
        type=str,
        default="80,200,400,600",
        help=(
            "Comma-separated nested subset sizes n to compare to the reference (default: "
            "80,160,240,320,400 — min 80, max 400, four equal steps of 80). Each n must be <= --n-ref."
        ),
    )
    p.add_argument(
        "--subset-seed-offset",
        type=int,
        default=0,
        help=(
            "Added to the convergence base seed for the global permutation (default 0). "
            "The base seed is --run-seed when set, otherwise the dataset meta seed from the NPZ."
        ),
    )
    p.add_argument(
        "--run-seed",
        type=int,
        default=None,
        help=(
            "If set, use this as the base RNG seed for this study (global permutation uses "
            "run_seed + --subset-seed-offset; shared Fisher training uses this via merged args.seed; "
            "default clf / GT Hellinger MC seeds use this when their dedicated flags are -1). "
            "Omit to keep the dataset meta seed from the NPZ (same behavior as before)."
        ),
    )
    p.add_argument(
        "--num-theta-bins",
        type=int,
        default=10,
        help=(
            "Number of equal-width theta bins (default 10). GT Hellinger MC uses "
            "n_mc = n_ref // num_theta_bins (floor) samples per bin row."
        ),
    )
    p.add_argument(
        "--clf-test-frac",
        type=float,
        default=0.3,
        help="Held-out fraction for pairwise bin-vs-bin logistic regression (default 0.3).",
    )
    p.add_argument(
        "--clf-min-class-count",
        type=int,
        default=5,
        help="Minimum samples per bin class for pairwise classifiers (default 5).",
    )
    p.add_argument(
        "--clf-random-state",
        type=int,
        default=-1,
        help="Random seed for train/test split; -1 uses dataset seed from NPZ meta.",
    )
    p.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep per-n training output directories under output-dir/sweep_runs/ (default: temp dirs).",
    )
    p.add_argument(
        "--visualization-only",
        action="store_true",
        help=(
            "Skip dataset sweep, GT Hellinger MC, and per-n training. Load "
            "h_decoding_convergence_results.npz from --output-dir and regenerate figures, CSV, "
            "and summary only. Requires a prior full run in that directory; "
            "training_losses/n_*.npz must exist for the loss panel."
        ),
    )
    p.add_argument(
        "--gt-hellinger-seed",
        type=int,
        default=-1,
        help="RNG seed for GT Hellinger MC; -1 uses dataset meta seed.",
    )
    p.add_argument(
        "--gt-hellinger-symmetrize",
        action="store_true",
        help="If set, symmetrize GT H^2 matrix as (H^2 + H^2.T) / 2 after one-sided MC estimation.",
    )
    add_estimation_arguments(p)
    p.set_defaults(
        output_dir=str(Path(DATA_DIR) / "h_decoding_convergence"),
        # DSM: FiLM posterior score + MLP prior score (see --score-arch / --prior-score-arch).
        score_arch="film",
        score_depth=3,
        prior_depth=3,
        # Flow / flow_likelihood: MLP velocity nets unless user passes --flow-score-arch film, etc.
        flow_score_arch="mlp",
        flow_prior_arch="mlp",
        score_early_ema_alpha=0.05,
        prior_early_ema_alpha=0.05,
        score_early_ema_warmup_epochs=0,
        prior_early_ema_warmup_epochs=0,
    )
    return p


def _parse_n_list(s: str) -> list[int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if not parts:
        raise ValueError("--n-list must contain at least one integer.")
    return [int(x) for x in parts]


def _validate_cli(args: argparse.Namespace) -> None:
    validate_estimation_args(args)
    if int(args.num_theta_bins) < 1:
        raise ValueError("--num-theta-bins must be >= 1.")
    if not (0.0 < float(args.clf_test_frac) < 1.0):
        raise ValueError("--clf-test-frac must be in (0, 1).")
    if int(args.clf_min_class_count) < 1:
        raise ValueError("--clf-min-class-count must be >= 1.")
    n_ref = int(args.n_ref)
    n_bins_cli = int(args.num_theta_bins)
    if n_ref < 2:
        raise ValueError("--n-ref must be >= 2.")
    if n_ref // n_bins_cli < 1:
        raise ValueError(
            "GT Hellinger requires n_mc = n_ref // num_theta_bins >= 1 "
            f"(got n_ref={n_ref} num_theta_bins={n_bins_cli})."
        )
    _parse_n_list(args.n_list)  # syntax check only; pool size checked in main


def _subset_bundle(
    bundle: SharedDatasetBundle,
    perm: np.ndarray,
    n: int,
    meta: dict,
) -> SharedDatasetBundle:
    """First n indices in perm order (nested subsets). Train/eval split matches make_dataset."""
    n = int(n)
    sub_perm = perm[:n]
    # Keep theta shape (N,1) like make_dataset / SharedDatasetBundle — 1D theta + (B,1)
    # sigma in training would broadcast to (B,B) and break FiLM/MLP.
    theta_all = np.asarray(bundle.theta_all[sub_perm], dtype=np.float64)
    if theta_all.ndim == 1:
        theta_all = theta_all.reshape(-1, 1)
    elif theta_all.ndim != 2 or theta_all.shape[1] != 1:
        theta_all = theta_all.reshape(-1, 1)
    x_all = np.asarray(bundle.x_all[sub_perm], dtype=np.float64)
    tf = float(meta["train_frac"])
    if tf >= 1.0:
        n_train = n
    else:
        n_train = int(tf * n)
        n_train = min(max(n_train, 1), n - 1)
    theta_train = theta_all[:n_train]
    x_train = x_all[:n_train]
    theta_eval = theta_all[n_train:]
    x_eval = x_all[n_train:]
    train_idx = np.arange(n_train, dtype=np.int64)
    eval_idx = np.arange(n_train, n, dtype=np.int64)
    return SharedDatasetBundle(
        meta=bundle.meta,
        theta_all=theta_all,
        x_all=x_all,
        train_idx=train_idx,
        eval_idx=eval_idx,
        theta_train=theta_train,
        x_train=x_train,
        theta_eval=theta_eval,
        x_eval=x_eval,
    )


def _make_full_args(args: argparse.Namespace, meta: dict) -> SimpleNamespace:
    full_args = merge_meta_into_args(meta, args)
    rs = getattr(args, "run_seed", None)
    if rs is not None:
        setattr(full_args, "seed", int(rs))
    setattr(full_args, "compute_h_matrix", True)
    setattr(full_args, "h_restore_original_order", True)
    setattr(full_args, "skip_shared_fisher_gt_compare", True)
    validate_estimation_args(full_args)
    return full_args


def _run_ctx_for_bundle(
    args: argparse.Namespace,
    meta: dict,
    bundle: SharedDatasetBundle,
    full_args: SimpleNamespace,
    n_bins: int,
) -> vhb.RunContext:
    run_seed = int(getattr(full_args, "seed", meta["seed"]))
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    rng = np.random.default_rng(run_seed)
    dev = require_device(str(full_args.device))
    dataset = build_dataset_from_meta(meta)
    cfg = vhb.BinnedVizConfig(args=args, dataset_npz=str(args.dataset_npz), n_bins=n_bins, h_only=False)
    return vhb.RunContext(
        args=args,
        config=cfg,
        bundle=bundle,
        meta=meta,
        full_args=full_args,
        dataset=dataset,
        rng=rng,
        device=dev,
    )


def _metrics_fixed_edges(
    loaded: vhb.LoadedHMatrix,
    x_aligned: np.ndarray,
    edges: np.ndarray,
    n_bins: int,
    clf_test_frac: float,
    clf_min_class_count: int,
    clf_random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    bin_idx = vhb.theta_to_bin_index(loaded.theta_used, edges, n_bins)
    h_binned, _ = vhb.average_matrix_by_bins(loaded.h_sym, bin_idx, n_bins)
    clf_acc, _, _, _ = vhb.pairwise_bin_logistic_accuracy_matrix(
        x_aligned,
        bin_idx,
        n_bins,
        test_frac=float(clf_test_frac),
        min_class_count=int(clf_min_class_count),
        random_state=int(clf_random_state),
    )
    return h_binned, clf_acc


def _pairwise_clf_from_bundle(
    *,
    args: argparse.Namespace,
    meta: dict,
    bundle: SharedDatasetBundle,
    output_dir: str,
    edges: np.ndarray,
    n_bins: int,
    clf_test_frac: float,
    clf_min_class_count: int,
    clf_random_state: int,
) -> np.ndarray:
    """Pairwise bin decoding matrix without training or loading an H-matrix (same split as H path)."""
    d = vars(args).copy()
    d.setdefault("h_matrix_npz", None)
    d.setdefault("h_only", False)
    args2 = argparse.Namespace(**d)
    args2.output_dir = output_dir
    full_args = _make_full_args(args2, meta)
    theta = vhb.theta_for_h_matrix_alignment(bundle, full_args)
    x = vhb.x_for_h_matrix_alignment(bundle, full_args)
    bin_idx = vhb.theta_to_bin_index(theta, edges, n_bins)
    clf_acc, _, _, _ = vhb.pairwise_bin_logistic_accuracy_matrix(
        x,
        bin_idx,
        n_bins,
        test_frac=float(clf_test_frac),
        min_class_count=int(clf_min_class_count),
        random_state=int(clf_random_state),
    )
    return clf_acc


def _estimate_one(
    *,
    args: argparse.Namespace,
    meta: dict,
    bundle: SharedDatasetBundle,
    output_dir: str,
    n_bins: int,
) -> tuple[vhb.LoadedHMatrix, np.ndarray, torch.device]:
    """Train (unless h-only), load H, return loaded H, x_aligned, and device."""
    d = vars(args).copy()
    d.setdefault("h_matrix_npz", None)
    d.setdefault("h_only", False)
    args2 = argparse.Namespace(**d)
    args2.output_dir = output_dir
    full_args = _make_full_args(args2, meta)
    ctx = _run_ctx_for_bundle(args2, meta, bundle, full_args, n_bins)
    vhb.run_h_estimation_if_needed(ctx)
    loaded = vhb.load_h_matrix(ctx)
    theta_chk = vhb.theta_for_h_matrix_alignment(ctx.bundle, ctx.full_args)
    if theta_chk.shape[0] != loaded.theta_used.shape[0]:
        raise ValueError(
            f"theta/H row mismatch: theta_chk={theta_chk.shape[0]} theta_used={loaded.theta_used.shape[0]}"
        )
    if not np.allclose(theta_chk, loaded.theta_used, rtol=0.0, atol=1e-5):
        raise ValueError(
            "theta_used from H-matrix npz does not match dataset theta for score_fisher_eval_data split."
        )
    x_aligned = vhb.x_for_h_matrix_alignment(ctx.bundle, ctx.full_args)
    if x_aligned.shape[0] != loaded.theta_used.shape[0]:
        raise ValueError(
            f"x/H row mismatch: x_aligned={x_aligned.shape[0]} theta_used={loaded.theta_used.shape[0]}"
        )
    return loaded, x_aligned, ctx.device


def _save_figure_png_svg(fig: plt.Figure, path_png: str, *, dpi: int = 160) -> str:
    """Write PNG and a sibling .svg (vector) for the same figure."""
    fig.savefig(path_png, dpi=dpi)
    path_svg = str(Path(path_png).with_suffix(".svg"))
    fig.savefig(path_svg)
    return path_svg


def _load_per_n_training_loss_npz(path: str) -> dict[str, Any]:
    """Load arrays from ``score_prior_training_losses.npz`` (DSM or flow)."""
    # Object arrays (e.g. ``theta_field_method``) require allow_pickle=True.
    z = np.load(path, allow_pickle=True)

    def _arr(name: str) -> np.ndarray:
        if name not in z.files:
            return np.asarray([], dtype=np.float64)
        return np.asarray(z[name], dtype=np.float64).ravel()

    prior_enable = True
    if "prior_enable" in z.files:
        pv = z["prior_enable"]
        prior_enable = bool(np.asarray(pv).reshape(-1)[0])

    tfm = "dsm"
    if "theta_field_method" in z.files:
        raw = z["theta_field_method"]
        if raw.dtype == object or raw.dtype.kind in ("U", "S"):
            tfm = str(np.asarray(raw).reshape(-1)[0])
        else:
            tfm = str(raw)

    return {
        "theta_field_method": tfm,
        "prior_enable": prior_enable,
        "score_train_losses": _arr("score_train_losses"),
        "score_val_losses": _arr("score_val_losses"),
        "score_val_monitor_losses": _arr("score_val_monitor_losses"),
        "prior_train_losses": _arr("prior_train_losses"),
        "prior_val_losses": _arr("prior_val_losses"),
        "prior_val_monitor_losses": _arr("prior_val_monitor_losses"),
    }


def _plot_loss_triplet(
    ax: plt.Axes,
    train: np.ndarray,
    val: np.ndarray,
    ema: np.ndarray,
    *,
    ylabel: str,
    title: str | None,
    show_legend: bool,
    score_like: bool,
) -> None:
    """Plot train / val / EMA monitor curves on one axis."""
    train = np.asarray(train, dtype=np.float64).ravel()
    val = np.asarray(val, dtype=np.float64).ravel()
    ema = np.asarray(ema, dtype=np.float64).ravel()
    lengths = [s.size for s in (train, val, ema) if s.size > 0]
    if not lengths:
        ax.text(0.5, 0.5, "no loss data", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_axis_off()
        return
    n_epoch = int(max(lengths))
    epochs = np.arange(1, n_epoch + 1, dtype=np.float64)
    if train.size > 0:
        m = min(train.size, n_epoch)
        ax.plot(epochs[:m], train[:m], color="#1f77b4", linewidth=1.6, label="train")
    if val.size > 0 and np.any(np.isfinite(val)):
        m = min(val.size, n_epoch)
        ax.plot(epochs[:m], val[:m], color="#d62728", linewidth=1.6, label="val")
    if ema.size > 0 and np.any(np.isfinite(ema)):
        m = min(ema.size, n_epoch)
        label = "val EMA (early-stop)" if score_like else "prior val EMA"
        ax.plot(epochs[:m], ema[:m], color="#ff7f0e", linewidth=1.4, linestyle="--", label=label)
    ax.set_xlabel("epoch", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.35)
    if show_legend:
        ax.legend(loc="upper right", fontsize=7)


def _render_training_losses_panel(
    *,
    ns: list[int],
    loss_dir: str,
    out_png_path: str,
    dpi: int = 160,
) -> str:
    """Two rows (score, prior), one column per ``n``; save PNG + SVG."""
    n_cols = len(ns)
    if n_cols < 1:
        raise ValueError("n-list must be non-empty for training loss panel.")
    w = max(2.6 * n_cols, 6.0)
    h = 5.8
    fig, axes = plt.subplots(2, n_cols, figsize=(w, h), squeeze=False, sharex="col")

    row0_ylabel = "score / posterior loss"
    row1_ylabel = "prior loss"

    for j, n in enumerate(ns):
        path = os.path.join(loss_dir, f"n_{int(n):06d}.npz")
        if not os.path.isfile(path):
            for r in (0, 1):
                axes[r, j].text(
                    0.5,
                    0.5,
                    f"missing\n{path}",
                    ha="center",
                    va="center",
                    transform=axes[r, j].transAxes,
                    fontsize=8,
                    color="crimson",
                )
                axes[r, j].set_axis_off()
            continue

        try:
            bundle = _load_per_n_training_loss_npz(path)
        except Exception as e:
            for r in (0, 1):
                axes[r, j].text(
                    0.5,
                    0.5,
                    f"load error:\n{e!s}"[:200],
                    ha="center",
                    va="center",
                    transform=axes[r, j].transAxes,
                    fontsize=7,
                    color="crimson",
                )
                axes[r, j].set_axis_off()
            continue

        tfm = str(bundle.get("theta_field_method", "dsm")).strip().lower()
        if tfm == "flow":
            post_lab = "theta-flow score"
        elif tfm == "flow_likelihood":
            post_lab = "theta-flow direct likelihood"
        else:
            post_lab = "score (DSM)"
        _plot_loss_triplet(
            axes[0, j],
            bundle["score_train_losses"],
            bundle["score_val_losses"],
            bundle["score_val_monitor_losses"],
            ylabel=row0_ylabel if j == 0 else "",
            title=f"n={n} ({post_lab})",
            show_legend=(j == 0),
            score_like=True,
        )

        if bundle["prior_enable"]:
            _plot_loss_triplet(
                axes[1, j],
                bundle["prior_train_losses"],
                bundle["prior_val_losses"],
                bundle["prior_val_monitor_losses"],
                ylabel=row1_ylabel if j == 0 else "",
                title=None,
                show_legend=(j == 0),
                score_like=False,
            )
        else:
            axes[1, j].text(
                0.5,
                0.5,
                "prior disabled",
                ha="center",
                va="center",
                transform=axes[1, j].transAxes,
                fontsize=10,
            )
            axes[1, j].set_axis_off()

    fig.suptitle(
        "Training loss vs epoch (top: posterior; bottom: prior). Columns: nested subset sizes n.",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    svg = _save_figure_png_svg(fig, out_png_path, dpi=dpi)
    plt.close(fig)
    return svg


def _save_combined_convergence_figure(
    *,
    h_mats: list[np.ndarray],
    clf_mats: list[np.ndarray],
    col_labels: list[str],
    n_bins: int,
    theta_centers: np.ndarray,
    ns: list[int],
    corr_h: np.ndarray,
    corr_clf: np.ndarray,
    out_png_path: str,
    dpi: int = 160,
) -> str:
    """Side-by-side matrix panel and correlation curves using one matplotlib figure.

    PNG is raster as usual. SVG keeps the right-hand curve as vector paths (not a single
    embedded screenshot); heatmaps still use matplotlib's normal SVG image handling for ``imshow``.

    Figure height matches the matrix panel (``m_h``). The right column width is chosen so its
    aspect ratio matches ``_H_DECODING_CURVE_FIGSIZE_IN`` (default square 3.5×3.5 like the
    isolated curve figure).
    """
    crv_w, crv_h = _H_DECODING_CURVE_FIGSIZE_IN
    if crv_h <= 0:
        raise ValueError("_H_DECODING_CURVE_FIGSIZE_IN height must be > 0.")
    n_cols = len(h_mats)
    m_w, m_h = 2.8 * n_cols, 5.0
    H = float(m_h)
    Wm = m_w * H / m_h
    Wc = H * (float(crv_w) / float(crv_h))
    # Constrained layout: colorbars make tight_layout warn and mis-place panels.
    fig = plt.figure(figsize=(Wm + Wc, H), dpi=dpi, layout="constrained")
    gs0 = fig.add_gridspec(1, 2, width_ratios=[Wm, Wc])
    gs_left = gs0[0, 0].subgridspec(2, n_cols)
    axes_m = np.empty((2, n_cols), dtype=object)
    for c in range(n_cols):
        axes_m[0, c] = fig.add_subplot(gs_left[0, c])
        axes_m[1, c] = fig.add_subplot(gs_left[1, c])
    _populate_matrix_panel_axes(
        axes_m,
        h_mats=h_mats,
        clf_mats=clf_mats,
        col_labels=col_labels,
        n_bins=n_bins,
        theta_centers=theta_centers,
    )
    ax_c = fig.add_subplot(gs0[0, 1])
    _populate_convergence_curve_ax(
        ax_c,
        list(ns),
        corr_h,
        corr_clf,
        tick_labelsize=13.0,
        axis_labelsize=13.0,
        legend_fontsize=10.0,
    )
    path_svg = _save_figure_png_svg(fig, out_png_path, dpi=dpi)
    plt.close(fig)
    return path_svg


def _finite_min_max(matrices: list[np.ndarray]) -> tuple[float, float]:
    """Shared color scale from stacked finite values (ignores NaN)."""
    parts: list[np.ndarray] = []
    for m in matrices:
        a = np.asarray(m, dtype=np.float64).ravel()
        parts.append(a[np.isfinite(a)])
    if not parts:
        return 0.0, 1.0
    v = np.concatenate(parts)
    if v.size == 0:
        return 0.0, 1.0
    return float(np.min(v)), float(np.max(v))


def _format_theta_tick_labels(theta_centers: np.ndarray) -> list[str]:
    """String tick labels from bin-center θ values (matches ``imshow`` index ticks 0..n_bins-1)."""
    tc = np.asarray(theta_centers, dtype=np.float64).ravel()
    return [f"{float(v):.1f}" for v in tc]


def _matrix_panel_tick_indices(n_bins: int, *, max_ticks: int = 5) -> np.ndarray:
    """Sparse integer bin indices (inclusive ends) so matrix panels show fewer ticks."""
    nb = int(n_bins)
    if nb < 1:
        raise ValueError("n_bins must be >= 1.")
    k = min(max(2, int(max_ticks)), nb)
    if nb <= k:
        return np.arange(nb, dtype=np.int64)
    return np.unique(np.round(np.linspace(0, nb - 1, k)).astype(np.int64))


def _matrix_axes_show_top_right_spines(ax: Any) -> None:
    """Override global_setting spine defaults for matrix heatmaps."""
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)


def _populate_matrix_panel_axes(
    axes: np.ndarray,
    *,
    h_mats: list[np.ndarray],
    clf_mats: list[np.ndarray],
    col_labels: list[str],
    n_bins: int,
    theta_centers: np.ndarray,
) -> None:
    """Draw 2 × n_cols heatmaps on existing axes (shared with standalone matrix panel + combined figure)."""
    tc = np.asarray(theta_centers, dtype=np.float64).ravel()
    if int(tc.size) != int(n_bins):
        raise ValueError(f"theta_centers length {tc.size} must match n_bins={n_bins}.")
    tick_labs_full = _format_theta_tick_labels(tc)
    tick_idx = _matrix_panel_tick_indices(n_bins, max_ticks=5)
    tick_pos = tick_idx.tolist()
    tick_labs = [tick_labs_full[int(i)] for i in tick_idx]
    x_rot = 45 if len(tick_pos) > 6 else 0
    _matrix_tick_labelsize = 11
    _matrix_colorbar_tick_labelsize = 11
    n_cols = len(h_mats)
    if n_cols != len(clf_mats) or n_cols != len(col_labels):
        raise ValueError("h_mats, clf_mats, col_labels length mismatch.")
    if axes.shape != (2, n_cols):
        raise ValueError(f"axes must be shape (2, {n_cols}); got {axes.shape}.")

    vmin_h, vmax_h = 0.0, 1.0
    vmin_c, vmax_c = _finite_min_max(clf_mats)
    if vmin_c >= vmax_c:
        vmax_c = vmin_c + 1e-12
    cmap = "viridis"

    for c in range(n_cols):
        ax0 = axes[0, c]
        im0 = ax0.imshow(
            h_mats[c],
            vmin=vmin_h,
            vmax=vmax_h,
            cmap=cmap,
            aspect="equal",
            origin="lower",
        )
        ax0.set_title(col_labels[c], fontsize=10)
        ax0.set_xticks(tick_pos)
        ax0.set_xticklabels(tick_labs, rotation=x_rot, ha="right" if x_rot else "center", fontsize=_matrix_tick_labelsize)
        ax0.set_yticks(tick_pos)
        ax0.set_yticklabels(tick_labs, fontsize=_matrix_tick_labelsize)
        ax0.tick_params(axis="both", labelsize=_matrix_tick_labelsize)
        _matrix_axes_show_top_right_spines(ax0)
        if c == 0:
            ax0.set_ylabel(r"$\sqrt{H^2}$", fontsize=11)
        _cb0 = plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
        _cb0.ax.tick_params(labelsize=_matrix_colorbar_tick_labelsize)

        ax1 = axes[1, c]
        im1 = ax1.imshow(
            clf_mats[c],
            vmin=vmin_c,
            vmax=vmax_c,
            cmap=cmap,
            aspect="equal",
            origin="lower",
        )
        ax1.set_xticks(tick_pos)
        ax1.set_xticklabels(tick_labs, rotation=x_rot, ha="right" if x_rot else "center", fontsize=_matrix_tick_labelsize)
        ax1.set_yticks(tick_pos)
        ax1.set_yticklabels(tick_labs, fontsize=_matrix_tick_labelsize)
        ax1.tick_params(axis="both", labelsize=_matrix_tick_labelsize)
        _matrix_axes_show_top_right_spines(ax1)
        if c == 0:
            ax1.set_ylabel("Pairwise decoding", fontsize=11)
        ax1.set_xlabel(r"$\theta$", fontsize=11)
        _cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        _cb1.ax.tick_params(labelsize=_matrix_colorbar_tick_labelsize)


def _populate_convergence_curve_ax(
    ax: plt.Axes,
    ns: list[int],
    corr_h: np.ndarray,
    corr_clf: np.ndarray,
    *,
    tick_labelsize: float = 11.0,
    axis_labelsize: float = 12.0,
    legend_fontsize: float = 9.0,
) -> None:
    """Pearson r vs n on a single axis (shared with standalone curve figure + combined figure).

    Tick sizes default to ``global_setting``-style values (11 pt ticks, 12 pt axis labels).
    The combined figure passes larger values so the right panel stays readable at reduced width.
    """
    ns_list = [int(x) for x in ns]
    ch = np.asarray(corr_h, dtype=np.float64).ravel()
    cc = np.asarray(corr_clf, dtype=np.float64).ravel()
    ax.plot(
        ns_list,
        ch,
        color="#1f77b4",
        linewidth=1.8,
        marker="o",
        markersize=6,
        label="H matrix",
    )
    ax.plot(
        ns_list,
        cc,
        color="#d62728",
        linewidth=1.8,
        marker="s",
        markersize=5,
        label="decoding acc matrix",
    )
    ax.axhline(
        1.0,
        color="0.45",
        linestyle="--",
        linewidth=1.0,
        zorder=0,
        clip_on=False,
    )
    ax.set_xlabel("dataset size n", fontsize=axis_labelsize)
    ax.set_ylabel("Pearson r (off-diagonal estimated vs off-diagonal approx GT)", fontsize=axis_labelsize)
    ax.set_xticks(ns_list)
    ax.tick_params(axis="both", labelsize=tick_labelsize)
    ax.legend(loc="best", fontsize=legend_fontsize)


def _render_matrix_panel(
    *,
    h_mats: list[np.ndarray],
    clf_mats: list[np.ndarray],
    col_labels: list[str],
    out_path: str,
    n_bins: int,
    theta_centers: np.ndarray,
) -> None:
    """Two rows: sqrt(binned H_sym) vs sqrt(GT H^2), pairwise decoding."""
    n_cols = len(h_mats)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.8 * n_cols, 5.0), squeeze=False)
    _populate_matrix_panel_axes(
        axes,
        h_mats=h_mats,
        clf_mats=clf_mats,
        col_labels=col_labels,
        n_bins=n_bins,
        theta_centers=theta_centers,
    )
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.12)
    _save_figure_png_svg(fig, out_path, dpi=160)
    plt.close(fig)


def _write_summary(
    path: str,
    args: argparse.Namespace,
    meta: dict,
    perm_seed: int,
    n_pool: int,
    ref_dir: str,
    paths_out: dict[str, str],
    per_n_loss_rows: list[dict[str, str]],
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("study_h_decoding_convergence\n")
        f.write(f"dataset_npz: {args.dataset_npz}\n")
        f.write(f"dataset_family: {meta.get('dataset_family')}  # matches --dataset-family={args.dataset_family}\n")
        f.write(f"output_dir: {args.output_dir}\n")
        f.write(f"n_ref: {args.n_ref}  n_list: {args.n_list}\n")
        f.write(f"num_theta_bins: {args.num_theta_bins}\n")
        f.write(f"subset_seed_offset: {args.subset_seed_offset}\n")
        f.write(f"permutation rng seed: {perm_seed}\n")
        f.write(f"dataset pool size: {n_pool}\n")
        f.write(f"reference run dir: {ref_dir}\n")
        f.write(f"meta seed: {meta.get('seed')}\n")
        for k, v in paths_out.items():
            f.write(f"{k}: {v}\n")
        f.write("\n# Per-n training-loss artifacts\n")
        for row in per_n_loss_rows:
            f.write(
                "n={n} status={status} run_dir={run_dir} src={src} dst={dst} note={note}\n".format(
                    n=row["n"],
                    status=row["status"],
                    run_dir=row["run_dir"],
                    src=row["src"],
                    dst=row["dst"],
                    note=row["note"],
                )
            )


class CachedConvergenceBundle(TypedDict):
    """Arrays loaded from ``h_decoding_convergence_results.npz`` for visualization-only mode."""

    n: np.ndarray
    corr_h: np.ndarray
    corr_clf: np.ndarray
    wall_s: np.ndarray
    h_cols: np.ndarray
    clf_cols: np.ndarray
    n_ref: int
    perm_seed: int
    base_seed: int
    meta_seed: int
    edges: np.ndarray
    centers: np.ndarray
    gt_n_mc: int
    gt_seed: int
    gt_symmetrize: bool
    out_npz: str


def _load_cached_convergence_results(output_dir: str) -> CachedConvergenceBundle:
    """Load metrics and matrices from a prior full run."""
    out_npz = os.path.join(output_dir, "h_decoding_convergence_results.npz")
    if not os.path.isfile(out_npz):
        raise FileNotFoundError(
            f"Visualization-only mode requires prior results; missing file:\n  {out_npz}\n"
            "Run once without --visualization-only to generate h_decoding_convergence_results.npz."
        )
    z = np.load(out_npz, allow_pickle=True)
    required = (
        "n",
        "corr_h_binned_vs_gt_mc",
        "corr_clf_vs_ref",
        "wall_seconds",
        "h_binned_columns",
        "clf_acc_columns",
        "n_ref",
        "theta_bin_edges",
    )
    missing = [k for k in required if k not in z.files]
    if missing:
        raise KeyError(f"{out_npz} missing keys: {missing}")

    h_cols = np.asarray(z["h_binned_columns"], dtype=np.float64)
    clf_cols = np.asarray(z["clf_acc_columns"], dtype=np.float64)
    if h_cols.ndim != 3 or clf_cols.ndim != 3:
        raise ValueError("h_binned_columns / clf_acc_columns must be 3D (columns, bins, bins).")
    if h_cols.shape != clf_cols.shape:
        raise ValueError("h_binned_columns and clf_acc_columns shape mismatch.")

    edges = np.asarray(z["theta_bin_edges"], dtype=np.float64).ravel()
    n_bins_file = int(h_cols.shape[1])
    if edges.size != n_bins_file + 1:
        raise ValueError(
            f"theta_bin_edges length {edges.size} inconsistent with matrix dim {n_bins_file} (expected n_bins+1 edges)."
        )

    if "theta_bin_centers" in z.files:
        centers = np.asarray(z["theta_bin_centers"], dtype=np.float64).ravel()
    else:
        centers = bin_centers_from_edges(edges)

    gt_n_mc = int(np.asarray(z["gt_hellinger_n_mc"]).reshape(-1)[0]) if "gt_hellinger_n_mc" in z.files else 0
    gt_seed = int(np.asarray(z["gt_hellinger_seed"]).reshape(-1)[0]) if "gt_hellinger_seed" in z.files else 0
    sym_raw = z["gt_hellinger_symmetrize"] if "gt_hellinger_symmetrize" in z.files else np.int32(0)
    gt_symmetrize = bool(int(np.asarray(sym_raw).reshape(-1)[0]))

    return CachedConvergenceBundle(
        n=np.asarray(z["n"], dtype=np.int64).ravel(),
        corr_h=np.asarray(z["corr_h_binned_vs_gt_mc"], dtype=np.float64).ravel(),
        corr_clf=np.asarray(z["corr_clf_vs_ref"], dtype=np.float64).ravel(),
        wall_s=np.asarray(z["wall_seconds"], dtype=np.float64).ravel(),
        h_cols=h_cols,
        clf_cols=clf_cols,
        n_ref=int(np.asarray(z["n_ref"]).reshape(-1)[0]),
        perm_seed=int(np.asarray(z["perm_seed"]).reshape(-1)[0]) if "perm_seed" in z.files else 0,
        base_seed=int(np.asarray(z["convergence_base_seed"]).reshape(-1)[0]) if "convergence_base_seed" in z.files else 0,
        meta_seed=int(np.asarray(z["dataset_meta_seed"]).reshape(-1)[0]) if "dataset_meta_seed" in z.files else 0,
        edges=np.asarray(z["theta_bin_edges"], dtype=np.float64),
        centers=np.asarray(centers, dtype=np.float64).ravel(),
        gt_n_mc=gt_n_mc,
        gt_seed=gt_seed,
        gt_symmetrize=gt_symmetrize,
        out_npz=os.path.abspath(out_npz),
    )


def _validate_cached_matches_cli(args: argparse.Namespace, cached: CachedConvergenceBundle, ns: list[int]) -> None:
    n_arr = np.asarray(cached["n"], dtype=np.int64).ravel()
    if n_arr.size != len(ns) or not np.array_equal(n_arr, np.asarray(ns, dtype=np.int64)):
        raise ValueError(
            f"--n-list {ns} does not match cached results n={n_arr.tolist()}. "
            "Use the same --n-list as the run that produced h_decoding_convergence_results.npz."
        )
    if int(cached["n_ref"]) != int(args.n_ref):
        raise ValueError(
            f"Cached n_ref={cached['n_ref']} does not match --n-ref={int(args.n_ref)}."
        )
    n_bins_h = int(cached["h_cols"].shape[1])
    if n_bins_h != int(args.num_theta_bins):
        raise ValueError(
            f"Cached matrices imply num_theta_bins={n_bins_h} but CLI has --num-theta-bins={args.num_theta_bins}."
        )
    n_cols = int(cached["h_cols"].shape[0])
    if n_cols != len(ns) + 1:
        raise ValueError(
            f"Expected h_binned_columns to have {len(ns) + 1} columns (n sweep + n_ref); got {n_cols}."
        )


def _render_convergence_figures_and_summary(
    *,
    args: argparse.Namespace,
    meta: dict,
    perm_seed: int,
    n_pool: int,
    ref_dir: str,
    ns: list[int],
    n_bins: int,
    theta_centers: np.ndarray,
    gt_n_mc: int,
    gt_seed: int,
    gt_symmetrize_effective: bool,
    corr_h: np.ndarray,
    corr_clf: np.ndarray,
    wall_s: np.ndarray,
    h_cols: np.ndarray,
    clf_cols: np.ndarray,
    out_npz: str,
    per_n_loss_rows: list[dict[str, str]],
    err_msg: list[str],
    visualization_only: bool,
) -> None:
    """Write CSV, figures, manifest, training-loss panel, summary, and print artifact paths."""
    csv_path = os.path.join(args.output_dir, "h_decoding_convergence_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "n",
                "corr_h_binned_vs_gt_mc",
                "corr_clf_vs_ref",
                "wall_seconds",
            ]
        )
        for i, n in enumerate(ns):
            w.writerow([n, corr_h[i], corr_clf[i], wall_s[i]])

    fig_path = os.path.join(args.output_dir, "h_decoding_convergence.png")
    fig, ax = plt.subplots(1, 1, figsize=_H_DECODING_CURVE_FIGSIZE_IN)
    _populate_convergence_curve_ax(ax, list(ns), corr_h, corr_clf)
    fig.tight_layout()
    conv_svg = _save_figure_png_svg(fig, fig_path, dpi=160)
    plt.close(fig)

    matrix_panel_path = os.path.join(args.output_dir, "h_decoding_matrices_panel.png")
    col_labels = [f"n={n}" for n in ns] + [f"Approx GT, n_ref={int(args.n_ref)}"]
    _render_matrix_panel(
        h_mats=list(h_cols),
        clf_mats=list(clf_cols),
        col_labels=col_labels,
        out_path=matrix_panel_path,
        n_bins=n_bins,
        theta_centers=theta_centers,
    )

    combined_path = os.path.join(args.output_dir, "h_decoding_convergence_combined.png")
    combined_svg = _save_combined_convergence_figure(
        h_mats=list(h_cols),
        clf_mats=list(clf_cols),
        col_labels=col_labels,
        n_bins=n_bins,
        theta_centers=theta_centers,
        ns=list(ns),
        corr_h=corr_h,
        corr_clf=corr_clf,
        out_png_path=combined_path,
        dpi=160,
    )

    loss_dir = os.path.join(args.output_dir, "training_losses")
    os.makedirs(loss_dir, exist_ok=True)
    manifest_path = os.path.join(loss_dir, "manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        mf.write("# n\tstatus\trun_dir\tsrc_loss_npz\tdst_loss_npz\tnote\n")
        for row in per_n_loss_rows:
            mf.write(
                "{n}\t{status}\t{run_dir}\t{src}\t{dst}\t{note}\n".format(
                    n=row["n"],
                    status=row["status"],
                    run_dir=row["run_dir"],
                    src=row["src"],
                    dst=row["dst"],
                    note=row["note"],
                )
            )

    loss_panel_png = os.path.join(args.output_dir, "h_decoding_training_losses_panel.png")
    loss_panel_svg = _render_training_losses_panel(
        ns=list(ns),
        loss_dir=loss_dir,
        out_png_path=loss_panel_png,
        dpi=160,
    )

    summary_path = os.path.join(args.output_dir, "h_decoding_convergence_summary.txt")
    paths_out = {
        "results_npz": out_npz,
        "results_csv": csv_path,
        "figure": fig_path,
        "figure_svg": conv_svg,
        "matrix_panel": matrix_panel_path,
        "matrix_panel_svg": str(Path(matrix_panel_path).with_suffix(".svg")),
        "combined_figure": combined_path,
        "combined_figure_svg": combined_svg,
        "training_losses_panel": loss_panel_png,
        "training_losses_panel_svg": loss_panel_svg,
        "reference_npz": os.path.join(args.output_dir, "h_decoding_convergence_reference.npz"),
        "training_losses_dir": loss_dir,
        "training_losses_manifest": manifest_path,
    }
    if visualization_only:
        paths_out["mode"] = "visualization-only (figures/csv/summary regenerated from cached NPZ)"
    if err_msg:
        paths_out["errors"] = "; ".join(err_msg[:20])
    _write_summary(summary_path, args, meta, perm_seed, n_pool, ref_dir, paths_out, per_n_loss_rows)
    with open(summary_path, "a", encoding="utf-8") as sf:
        sf.write("\n# Correlation targets (Pearson r off-diagonal; matrix_corr_offdiag_pearson)\n")
        sf.write(
            "# corr_h_binned_vs_gt_mc: Pearson r, off-diagonal binned sqrt(H_sym) vs sqrt(generative GT H^2) (MC one-sided Hellinger).\n"
        )
        sf.write(
            "# corr_clf_vs_ref: Pearson r, off-diagonal pairwise decoding vs n_ref subset decoding matrix (same bin edges).\n"
        )
        sf.write(
            "# h_binned_columns last column: MC GT sqrt(H^2) (not DSM/flow); hellinger_gt_sq_mc key stores sqrt(H^2).\n"
        )
        sf.write(
            f"gt_hellinger_n_mc: {int(gt_n_mc)}  # MC samples per bin row; floor(n_ref / num_theta_bins)\n"
        )
        sf.write(
            f"gt_hellinger_n_ref_budget: {int(args.n_ref)}  # reference training subset size (--n-ref)\n"
        )
        sf.write(
            f"gt_hellinger_n_bins_times_n_mc: {int(n_bins * gt_n_mc)}  # n_bins * n_mc (may be < n_ref)\n"
        )
        sf.write(f"gt_hellinger_seed: {int(gt_seed)}\n")
        sf.write(f"gt_hellinger_symmetrize: {bool(gt_symmetrize_effective)}\n")

    tag = "[convergence] Saved (visualization-only):" if visualization_only else "[convergence] Saved:"
    print(tag)
    print(f"  - {out_npz}")
    print(f"  - {csv_path}")
    print(f"  - {fig_path}")
    print(f"  - {conv_svg}")
    print(f"  - {matrix_panel_path}")
    print(f"  - {paths_out['matrix_panel_svg']}")
    print(f"  - {combined_path}")
    print(f"  - {combined_svg}")
    print(f"  - {loss_panel_png}")
    print(f"  - {loss_panel_svg}")
    print(f"  - {loss_dir}/ (per-n training loss .npz + manifest.txt)")
    print(f"  - {summary_path}")


def main(argv: list[str] | None = None) -> None:
    # When stdout is redirected (e.g. nohup), default block buffering delays run.log updates.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass
    p = build_parser()
    args = p.parse_args(argv)
    args.output_dir = os.path.abspath(str(args.output_dir))
    args.dataset_npz = os.path.abspath(str(args.dataset_npz))
    _validate_cli(args)
    ns = _parse_n_list(args.n_list)

    os.makedirs(args.output_dir, exist_ok=True)
    bundle = load_shared_dataset_npz(args.dataset_npz)
    meta = bundle.meta
    meta_family = str(meta.get("dataset_family", ""))
    if meta_family != str(args.dataset_family):
        raise ValueError(
            f"NPZ meta dataset_family={meta_family!r} does not match --dataset-family={str(args.dataset_family)!r}. "
            "Regenerate with matching make_dataset.py --dataset-family, or pass --dataset-family to match the NPZ."
        )
    n_pool = int(bundle.theta_all.shape[0])
    need = max(int(args.n_ref), max(ns))
    if n_pool < need:
        raise ValueError(
            f"Dataset has n_total={n_pool} but need at least max(n_ref, max(n_list))={need}. "
            "Regenerate with make_dataset.py --n-total >= that value."
        )
    for n in ns:
        if n < 1:
            raise ValueError(f"Each n in --n-list must be >= 1; got {n}.")
        if n > n_pool:
            raise ValueError(f"Each n in --n-list must be <= n_total={n_pool}; got n={n}.")
    if max(ns) > int(args.n_ref):
        raise ValueError(
            f"Require max(n-list) <= n-ref for nested subsets; got max(n_list)={max(ns)} n_ref={args.n_ref}."
        )

    if bool(getattr(args, "visualization_only", False)):
        cached = _load_cached_convergence_results(args.output_dir)
        _validate_cached_matches_cli(args, cached, ns)
        if int(cached["meta_seed"]) != 0 and int(meta["seed"]) != int(cached["meta_seed"]):
            raise ValueError(
                f"Dataset NPZ meta seed={meta['seed']} does not match cached results dataset_meta_seed={cached['meta_seed']}. "
                "Use the same --dataset-npz as the run that produced the cached results."
            )
        ref_dir_v = os.path.join(args.output_dir, "reference")
        n_bins_v = int(args.num_theta_bins)
        per_n_loss_rows_viz: list[dict[str, str]] = []
        for n in ns:
            loss_npz = os.path.join(args.output_dir, "training_losses", f"n_{int(n):06d}.npz")
            if not os.path.isfile(loss_npz):
                raise FileNotFoundError(
                    f"Visualization-only mode requires per-n training loss NPZ (for loss panel):\n  {loss_npz}\n"
                    "Run a full study first so training_losses/n_*.npz exists, or copy artifacts from a prior run."
                )
            loss_abs = os.path.abspath(loss_npz)
            per_n_loss_rows_viz.append(
                {
                    "n": str(n),
                    "status": "cached",
                    "run_dir": "",
                    "src": loss_abs,
                    "dst": loss_abs,
                    "note": "visualization-only (no re-training)",
                }
            )
        print(
            "[convergence] --visualization-only: skipping GT MC, reference sweep, and per-n training; "
            "regenerating figures from cached NPZ.",
            flush=True,
        )
        _render_convergence_figures_and_summary(
            args=args,
            meta=meta,
            perm_seed=int(cached["perm_seed"]),
            n_pool=n_pool,
            ref_dir=ref_dir_v,
            ns=ns,
            n_bins=n_bins_v,
            theta_centers=cached["centers"],
            gt_n_mc=int(cached["gt_n_mc"]),
            gt_seed=int(cached["gt_seed"]),
            gt_symmetrize_effective=bool(cached["gt_symmetrize"]),
            corr_h=cached["corr_h"],
            corr_clf=cached["corr_clf"],
            wall_s=cached["wall_s"],
            h_cols=cached["h_cols"],
            clf_cols=cached["clf_cols"],
            out_npz=cached["out_npz"],
            per_n_loss_rows=per_n_loss_rows_viz,
            err_msg=[],
            visualization_only=True,
        )
        return

    n_bins = int(args.num_theta_bins)
    base_seed = int(args.run_seed) if args.run_seed is not None else int(meta["seed"])
    perm_seed = base_seed + int(args.subset_seed_offset)
    rng_perm = np.random.default_rng(perm_seed)
    perm = rng_perm.permutation(n_pool)
    if args.run_seed is not None:
        print(
            f"[convergence] --run-seed={int(args.run_seed)} "
            f"(dataset meta seed={int(meta['seed'])}; perm_seed={perm_seed})",
            flush=True,
        )

    theta_ref = np.asarray(bundle.theta_all[perm[: int(args.n_ref)]], dtype=np.float64).reshape(-1)
    edges, edge_lo, edge_hi = vhb.theta_bin_edges(theta_ref, n_bins)

    clf_rs = base_seed if int(args.clf_random_state) < 0 else int(args.clf_random_state)

    dataset_for_gt = build_dataset_from_meta(meta)
    gt_seed = base_seed if int(args.gt_hellinger_seed) < 0 else int(args.gt_hellinger_seed)
    if hasattr(dataset_for_gt, "rng"):
        dataset_for_gt.rng = np.random.default_rng(gt_seed)
    centers = bin_centers_from_edges(edges)
    gt_n_mc = int(args.n_ref) // n_bins
    t_gt0 = time.time()
    h_gt_mc = estimate_hellinger_sq_one_sided_mc(
        dataset_for_gt,
        centers,
        n_mc=gt_n_mc,
        symmetrize=bool(args.gt_hellinger_symmetrize),
    )
    h_gt_sqrt = _sqrt_h_like(h_gt_mc)
    print(
        f"[convergence] GT Hellinger (MC likelihood) n_bins={n_bins} n_mc={gt_n_mc} "
        f"(n_bins*n_mc={n_bins * gt_n_mc} <= n_ref={int(args.n_ref)}) wall time: {time.time() - t_gt0:.1f}s "
        f"(H track uses sqrt(H^2) vs sqrt(binned h_sym))",
        flush=True,
    )

    ref_dir = os.path.join(args.output_dir, "reference")
    os.makedirs(ref_dir, exist_ok=True)
    tfm = str(getattr(args, "theta_field_method", "dsm")).strip().lower()
    if tfm == "flow":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(flow_score_arch={getattr(args, 'flow_score_arch', 'mlp')}, "
            f"flow_prior_arch={getattr(args, 'flow_prior_arch', 'mlp')}; "
            f"DSM --score-arch/--prior-score-arch do not select theta-flow velocity nets)",
            flush=True,
        )
        print(
            "[convergence] flow mode uses score-from-velocity conversion "
            "(path.velocity_to_epsilon then s=-eps/sigma_t).",
            flush=True,
        )
    elif tfm == "flow_likelihood":
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(flow_score_arch={getattr(args, 'flow_score_arch', 'mlp')}, "
            f"flow_prior_arch={getattr(args, 'flow_prior_arch', 'mlp')})",
            flush=True,
        )
        print(
            "[convergence] flow_likelihood mode uses direct ODE likelihood ratios "
            "(no velocity-to-score theta integration).",
            flush=True,
        )
    else:
        print(
            f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
            f"(score_arch={getattr(args, 'score_arch', 'mlp')}, "
            f"prior_arch={getattr(args, 'prior_score_arch', 'mlp')})",
            flush=True,
        )
    print(
        "[convergence] n_ref reference: no learned H training at n_ref; matrix-panel top row = MC GT sqrt(H^2); "
        "pairwise decoding from n_ref subset only.",
        flush=True,
    )
    print(f"[convergence] reference dir (decoding-only artifacts) n={args.n_ref} -> {ref_dir}", flush=True)
    print(f"[convergence] n_list={ns}", flush=True)
    t0 = time.time()
    bundle_ref = _subset_bundle(bundle, perm, int(args.n_ref), meta)
    h_ref = np.asarray(h_gt_sqrt, dtype=np.float64)
    clf_ref = _pairwise_clf_from_bundle(
        args=args,
        meta=meta,
        bundle=bundle_ref,
        output_dir=ref_dir,
        edges=edges,
        n_bins=n_bins,
        clf_test_frac=float(args.clf_test_frac),
        clf_min_class_count=int(args.clf_min_class_count),
        clf_random_state=clf_rs,
    )
    print(f"[convergence] reference (GT + decoding) wall time: {time.time() - t0:.1f}s")

    loss_dir = os.path.join(args.output_dir, "training_losses")
    os.makedirs(loss_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(args.output_dir, "h_decoding_convergence_reference.npz"),
        h_binned_ref=h_ref,
        clf_acc_ref=clf_ref,
        # Legacy key name: stores sqrt(H^2) from MC (not raw H^2); see module docstring.
        hellinger_gt_sq_mc=h_gt_sqrt,
        h_binned_ref_is_gt_mc=np.int32(1),
        theta_bin_centers=centers,
        gt_hellinger_n_mc=np.int64(gt_n_mc),
        gt_hellinger_n_ref_budget=np.int64(args.n_ref),
        gt_hellinger_seed=np.int64(gt_seed),
        gt_hellinger_symmetrize=np.int32(1 if bool(args.gt_hellinger_symmetrize) else 0),
        theta_bin_edges=edges,
        edge_lo=np.float64(edge_lo),
        edge_hi=np.float64(edge_hi),
        perm_seed=np.int64(perm_seed),
        convergence_base_seed=np.int64(base_seed),
        dataset_meta_seed=np.int64(meta["seed"]),
        n_ref=np.int64(args.n_ref),
    )

    corr_h = np.full(len(ns), np.nan, dtype=np.float64)
    corr_clf = np.full(len(ns), np.nan, dtype=np.float64)
    wall_s = np.full(len(ns), np.nan, dtype=np.float64)
    err_msg: list[str] = []
    h_sweep: list[np.ndarray] = []
    clf_sweep: list[np.ndarray] = []
    per_n_loss_rows: list[dict[str, str]] = []

    sweep_root = os.path.join(args.output_dir, "sweep_runs")
    if bool(args.keep_intermediate):
        os.makedirs(sweep_root, exist_ok=True)
    print(
        "[convergence] per-n training sweep is enabled; collecting training_losses from each run artifact.",
        flush=True,
    )

    for k, n in enumerate(ns):
        run_dir: str
        tmp_ctx: tempfile.TemporaryDirectory[str] | None = None
        if bool(args.keep_intermediate):
            run_dir = os.path.join(sweep_root, f"n_{n:06d}")
            os.makedirs(run_dir, exist_ok=True)
        else:
            tmp_ctx = tempfile.TemporaryDirectory(prefix=f"h_conv_n{n}_", dir=args.output_dir)
            run_dir = tmp_ctx.name

        try:
            t1 = time.time()
            bundle_n = _subset_bundle(bundle, perm, int(n), meta)
            loaded_n, x_n, _ = _estimate_one(
                args=args,
                meta=meta,
                bundle=bundle_n,
                output_dir=run_dir,
                n_bins=n_bins,
            )
            h_n, clf_n = _metrics_fixed_edges(
                loaded_n,
                x_n,
                edges,
                n_bins,
                float(args.clf_test_frac),
                int(args.clf_min_class_count),
                clf_rs,
            )
            h_n_sqrt = _sqrt_h_like(h_n)
            corr_h[k] = vhb.matrix_corr_offdiag_pearson(h_n_sqrt, h_gt_sqrt)
            corr_clf[k] = vhb.matrix_corr_offdiag_pearson(clf_n, clf_ref)
            wall_s[k] = time.time() - t1
            h_sweep.append(np.asarray(h_n_sqrt, dtype=np.float64))
            clf_sweep.append(np.asarray(clf_n, dtype=np.float64))
            print(
                f"[convergence] n={n}  corr_h={corr_h[k]:.4f}  corr_clf={corr_clf[k]:.4f}  "
                f"wall={wall_s[k]:.1f}s",
                flush=True,
            )
            run_loss_npz = os.path.join(run_dir, "score_prior_training_losses.npz")
            run_loss_npz_abs = os.path.abspath(run_loss_npz)
            if not os.path.isfile(run_loss_npz_abs):
                raise FileNotFoundError(
                    f"Expected per-n training loss artifact is missing: {run_loss_npz_abs}"
                )
            dst_loss_npz_abs = os.path.abspath(os.path.join(loss_dir, f"n_{n:06d}.npz"))
            shutil.copy2(run_loss_npz_abs, dst_loss_npz_abs)
            per_n_loss_rows.append(
                {
                    "n": str(n),
                    "status": "copied",
                    "run_dir": os.path.abspath(run_dir),
                    "src": run_loss_npz_abs,
                    "dst": dst_loss_npz_abs,
                    "note": "from per-n training sweep run",
                }
            )
            print(
                f"[convergence] n={n} training_loss copied -> {dst_loss_npz_abs}",
                flush=True,
            )
        except Exception as e:
            err_msg.append(f"n={n}: {e!r}")
            print(f"[convergence] ERROR n={n}: {e}")
            per_n_loss_rows.append(
                {
                    "n": str(n),
                    "status": "error",
                    "run_dir": os.path.abspath(run_dir),
                    "src": os.path.abspath(os.path.join(run_dir, "score_prior_training_losses.npz")),
                    "dst": os.path.abspath(os.path.join(loss_dir, f"n_{n:06d}.npz")),
                    "note": repr(e),
                }
            )
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if err_msg:
        msg = "\n".join([f"  - {m}" for m in err_msg[:20]])
        raise RuntimeError(
            "Per-n sweep failed (including required training-loss collection).\n"
            f"{msg}"
        )
    if len(h_sweep) != len(ns) or len(clf_sweep) != len(ns):
        raise RuntimeError(
            "Missing binned matrices for some n (partial failures). "
            "Fix errors above or re-run with a smaller n-list."
        )
    h_cols = np.stack(h_sweep + [h_ref], axis=0)
    clf_cols = np.stack(clf_sweep + [clf_ref], axis=0)
    column_n = np.asarray(list(ns) + [int(args.n_ref)], dtype=np.int64)

    out_npz = os.path.join(args.output_dir, "h_decoding_convergence_results.npz")
    np.savez_compressed(
        out_npz,
        n=np.asarray(ns, dtype=np.int64),
        corr_h_binned_vs_gt_mc=corr_h,
        corr_clf_vs_ref=corr_clf,
        wall_seconds=wall_s,
        n_ref=np.int64(args.n_ref),
        perm_seed=np.int64(perm_seed),
        convergence_base_seed=np.int64(base_seed),
        dataset_meta_seed=np.int64(meta["seed"]),
        theta_bin_edges=edges,
        theta_bin_centers=centers,
        # Legacy key name: sqrt(H^2) from MC; see module docstring.
        hellinger_gt_sq_mc=h_gt_sqrt,
        gt_hellinger_n_mc=np.int64(gt_n_mc),
        gt_hellinger_n_ref_budget=np.int64(args.n_ref),
        gt_hellinger_seed=np.int64(gt_seed),
        gt_hellinger_symmetrize=np.int32(1 if bool(args.gt_hellinger_symmetrize) else 0),
        h_binned_ref_is_gt_mc=np.int32(1),
        h_binned_columns=h_cols,
        clf_acc_columns=clf_cols,
        column_n=column_n,
    )

    _render_convergence_figures_and_summary(
        args=args,
        meta=meta,
        perm_seed=int(perm_seed),
        n_pool=n_pool,
        ref_dir=ref_dir,
        ns=ns,
        n_bins=n_bins,
        theta_centers=centers,
        gt_n_mc=int(gt_n_mc),
        gt_seed=int(gt_seed),
        gt_symmetrize_effective=bool(args.gt_hellinger_symmetrize),
        corr_h=corr_h,
        corr_clf=corr_clf,
        wall_s=wall_s,
        h_cols=h_cols,
        clf_cols=clf_cols,
        out_npz=out_npz,
        per_n_loss_rows=per_n_loss_rows,
        err_msg=err_msg,
        visualization_only=False,
    )


if __name__ == "__main__":
    main()
