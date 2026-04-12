#!/usr/bin/env python3
"""Convergence of binned H and pairwise decoding vs references.

**Binned H:** off-diagonal correlation vs a **generative ground-truth** Hellinger matrix
estimated by Monte Carlo from the known toy likelihood (see ``fisher/hellinger_gt.py`` and
``report/notes/hellinger_idea.tex``). The MC routine returns squared Hellinger **H^2**; this
script compares and visualizes **elementwise square roots** ``sqrt(H^2)`` (GT) and
``sqrt(H^sym)`` (learned binned symmetric ``h_sym``), both clipped to ``[0, 1]`` before the
square root. With ``n_bins`` = ``--num-theta-bins``, the MC count per
bin row is ``n_mc = n_ref // n_bins`` (integer floor); ``n_bins * n_mc`` may be less than ``n_ref``.
Nested subset training for each ``n`` in ``--n-list`` uses up to ``n`` samples; the ``n_ref`` column does not train a model.

**Pairwise decoding:** off-diagonal correlation vs the decoding matrix from the ``--n-ref``
subset (same bin edges as GT), unchanged.

**Dataset:** the loaded NPZ must match ``--dataset-family`` (default ``gaussian_sqrtd`` with
``gaussian_raw`` tuning from ``make_dataset.py``). Regenerate the NPZ if the family does not match.

For each ``n`` in ``--n-list``, the H matrix is computed from trained **posterior** and **prior**
models (``--theta-field-method dsm`` or ``flow``). In ``flow`` mode, H uses the
**flow-derived score** (velocity-to-epsilon conversion and ``s = -eps/sigma_t``), not raw velocity.
This script defaults to **multi-layer FiLM**
for the posterior (``--score-arch film``, ``--score-depth 3``) and a **3-layer MLP** prior
(``--prior-score-arch mlp``). The **reference column** (``n_ref``) does **not** run DSM/flow: the
matrix-panel top row shows **MC generative** ``sqrt(H^2)`` (same as the H correlation
target), while the bottom row still shows pairwise decoding on the ``n_ref`` data subset.

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
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent
_bin_dir = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_bin_dir) not in sys.path:
    sys.path.insert(0, str(_bin_dir))

import matplotlib.pyplot as plt
import numpy as np
import torch

import visualize_h_matrix_binned as vhb
from global_setting import DATA_DIR
from fisher.cli_shared_fisher import add_estimation_arguments
from fisher.hellinger_gt import bin_centers_from_edges, estimate_hellinger_sq_one_sided_mc
from fisher.shared_dataset_io import SharedDatasetBundle, load_shared_dataset_npz
from fisher.shared_fisher_est import (
    build_dataset_from_meta,
    merge_meta_into_args,
    require_device,
    validate_estimation_args,
)


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
        default="gaussian_sqrtd",
        choices=[
            "gaussian",
            "gaussian_sqrtd",
            "gaussian_randamp",
            "gaussian_randamp_sqrtd",
            "gmm_non_gauss",
            "cos_sin_piecewise_noise",
            "linear_piecewise_noise",
        ],
        help=(
            "Expected generative family stored in the NPZ meta; must match make_dataset.py "
            "when the archive was created. Default: gaussian_sqrtd (Gaussian bump tuning + "
            "sqrt(x_dim) observation-noise scaling)."
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
        default="80,160,240,320,400",
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
        # Posterior: multi-layer FiLM with x trunk and (theta, sigma) additive residual; prior: MLP.
        score_arch="film",
        score_depth=3,
        prior_depth=3,
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
        post_lab = "theta-flow" if tfm == "flow" else "score (DSM)"
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
    left_png_path: str,
    right_png_path: str,
    out_png_path: str,
    *,
    dpi: int = 160,
) -> str:
    """Side-by-side: left = corr vs n, right = matrix panel; same height as the journal combined figure."""
    from PIL import Image

    left = Image.open(left_png_path).convert("RGB")
    right = Image.open(right_png_path).convert("RGB")
    h = max(left.height, right.height)

    def _scale_to_height(im: Image.Image, target_h: int) -> Image.Image:
        w = int(round(im.width * target_h / im.height))
        return im.resize((w, target_h), Image.Resampling.LANCZOS)

    l2 = _scale_to_height(left, h)
    r2 = _scale_to_height(right, h)
    combined = Image.new("RGB", (l2.width + r2.width, h), (255, 255, 255))
    combined.paste(l2, (0, 0))
    combined.paste(r2, (l2.width, 0))
    combined.save(out_png_path, dpi=(dpi, dpi))

    arr = np.asarray(combined)
    fig = plt.figure(figsize=(arr.shape[1] / dpi, arr.shape[0] / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(arr)
    ax.axis("off")
    path_svg = str(Path(out_png_path).with_suffix(".svg"))
    fig.savefig(path_svg, pad_inches=0)
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


def _render_matrix_panel(
    *,
    h_mats: list[np.ndarray],
    clf_mats: list[np.ndarray],
    col_labels: list[str],
    out_path: str,
    n_bins: int,
) -> None:
    """Two rows: sqrt(binned H_sym) vs sqrt(GT H^2), pairwise decoding."""
    n_cols = len(h_mats)
    if n_cols != len(clf_mats) or n_cols != len(col_labels):
        raise ValueError("h_mats, clf_mats, col_labels length mismatch.")
    fig, axes = plt.subplots(2, n_cols, figsize=(2.8 * n_cols, 5.6), squeeze=False)
    # sqrt(H) entries lie in [0, 1]; use [0, 1] for a consistent cross-column color scale.
    vmin_h, vmax_h = 0.0, 1.0
    vmin_c, vmax_c = _finite_min_max(clf_mats)
    if vmin_c >= vmax_c:
        vmax_c = vmin_c + 1e-12
    cmap = "viridis"

    for c in range(n_cols):
        im0 = axes[0, c].imshow(
            h_mats[c],
            vmin=vmin_h,
            vmax=vmax_h,
            cmap=cmap,
            aspect="equal",
            origin="lower",
        )
        axes[0, c].set_title(col_labels[c], fontsize=10)
        axes[0, c].set_xticks(range(n_bins))
        axes[0, c].set_yticks(range(n_bins))
        axes[0, c].tick_params(labelsize=7)
        if c == 0:
            axes[0, c].set_ylabel(r"Binned $\sqrt{H^{\mathrm{sym}}}$ / GT $\sqrt{H^2}$ (n_ref)", fontsize=11)
        plt.colorbar(im0, ax=axes[0, c], fraction=0.046, pad=0.04)

        im1 = axes[1, c].imshow(
            clf_mats[c],
            vmin=vmin_c,
            vmax=vmax_c,
            cmap=cmap,
            aspect="equal",
            origin="lower",
        )
        axes[1, c].set_xticks(range(n_bins))
        axes[1, c].set_yticks(range(n_bins))
        axes[1, c].tick_params(labelsize=7)
        if c == 0:
            axes[1, c].set_ylabel("Pairwise decoding", fontsize=11)
        plt.colorbar(im1, ax=axes[1, c], fraction=0.046, pad=0.04)

    fig.suptitle(
        "Top: sqrt(binned H_sym) for each n; last column n_ref shows MC GT sqrt(H^2). "
        "Bottom: pairwise decoding (rows 1–2) by nested subset size.",
        fontsize=10,
    )
    fig.tight_layout()
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
    print(
        f"[convergence] sweep n in --n-list: --theta-field-method={tfm} "
        f"(score_arch={getattr(args, 'score_arch', 'mlp')}, "
        f"prior_arch={getattr(args, 'prior_score_arch', 'mlp')})",
        flush=True,
    )
    if tfm == "flow":
        print(
            "[convergence] flow mode uses score-from-velocity conversion "
            "(path.velocity_to_epsilon then s=-eps/sigma_t).",
            flush=True,
        )
    print(
        "[convergence] n_ref reference: no DSM/flow training; matrix-panel top row = MC GT sqrt(H^2); "
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
            corr_h[k] = vhb.matrix_corr_offdiag(h_n_sqrt, h_gt_sqrt)
            corr_clf[k] = vhb.matrix_corr_offdiag(clf_n, clf_ref)
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
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 4.8))
    ax.plot(
        ns,
        corr_h,
        color="#1f77b4",
        linewidth=1.8,
        marker="o",
        markersize=6,
        label=r"Binned $\sqrt{H^{\mathrm{sym}}}$ vs GT $\sqrt{H^2}$ (MC likelihood)",
    )
    ax.plot(
        ns,
        corr_clf,
        color="#d62728",
        linewidth=1.8,
        marker="s",
        markersize=5,
        label="Pairwise decoding vs n_ref decoding",
    )
    ax.set_xlabel("dataset size n (nested subset)")
    ax.set_ylabel("corr (off-diag)")
    ax.set_title(
        "Convergence: sqrt(H) vs generative sqrt(H^2); decoding vs n_ref=%d"
        % int(args.n_ref)
    )
    ax.set_xticks(ns)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    conv_svg = _save_figure_png_svg(fig, fig_path, dpi=160)
    plt.close(fig)

    matrix_panel_path = os.path.join(args.output_dir, "h_decoding_matrices_panel.png")
    col_labels = [f"n={n}" for n in ns] + [f"n_ref={int(args.n_ref)} (GT MC sqrt(H^2))"]
    _render_matrix_panel(
        h_mats=list(h_cols),
        clf_mats=list(clf_cols),
        col_labels=col_labels,
        out_path=matrix_panel_path,
        n_bins=n_bins,
    )

    combined_path = os.path.join(args.output_dir, "h_decoding_convergence_combined.png")
    combined_svg = _save_combined_convergence_figure(fig_path, matrix_panel_path, combined_path, dpi=160)

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
    if err_msg:
        paths_out["errors"] = "; ".join(err_msg[:20])
    _write_summary(summary_path, args, meta, perm_seed, n_pool, ref_dir, paths_out, per_n_loss_rows)
    with open(summary_path, "a", encoding="utf-8") as sf:
        sf.write("\n# Correlation targets\n")
        sf.write(
            "# corr_h_binned_vs_gt_mc: binned sqrt(H_sym) vs sqrt(generative GT H^2) (MC one-sided Hellinger).\n"
        )
        sf.write(
            "# corr_clf_vs_ref: pairwise decoding vs n_ref subset decoding matrix (same bin edges).\n"
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
        sf.write(f"gt_hellinger_symmetrize: {bool(args.gt_hellinger_symmetrize)}\n")

    print("[convergence] Saved:")
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


if __name__ == "__main__":
    main()
