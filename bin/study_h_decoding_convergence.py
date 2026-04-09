#!/usr/bin/env python3
"""Convergence of binned H, Hellinger LB from binned H, pairwise decoding, and L/C Bayes accuracy.

Tracks off-diagonal correlation to a reference run for: **binned H**, **Hellinger accuracy lower
bound** (treating binned symmetric H as H², matching ``visualize_h_matrix_binned``),
**pairwise logistic decoding** accuracy, and **pairwise Bayesian-optimal accuracy** estimated from
the score-matching **C-matrix** (integrated log-ratio / L pipeline) via bin-mean differences.

The H matrix is computed from trained **posterior** and **prior** denoising score models
(DSM). This script defaults to **multi-layer FiLM** for the posterior (``--score-arch film``,
``--score-depth 3``) and a **3-layer MLP** prior (``--prior-score-arch mlp``). After training,
``HMatrixEstimator`` combines score evaluations into
the sample H matrix, which is then theta-binned for comparison.
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
from fisher.shared_dataset_io import SharedDatasetBundle, load_shared_dataset_npz
from fisher.shared_fisher_est import (
    build_dataset_from_meta,
    merge_meta_into_args,
    require_device,
    run_shared_fisher_estimation,
    validate_estimation_args,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Load a shared dataset .npz, build a large-n reference (binned H + Hellinger LB + "
            "pairwise logistic + Bayes-opt accuracy from C-matrix bin means), then sweep nested "
            "subset sizes and plot off-diagonal correlation to the reference."
        )
    )
    p.add_argument(
        "--dataset-npz",
        type=str,
        required=True,
        help="Path to shared dataset .npz from make_dataset.py.",
    )
    p.add_argument(
        "--n-ref",
        type=int,
        default=5000,
        help="Reference subset size (default 5000). Requires len(dataset) >= n_ref and max(n-list) <= n_ref.",
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
        help="Added to dataset meta seed for the global permutation (default 0).",
    )
    p.add_argument(
        "--num-theta-bins",
        type=int,
        default=10,
        help="Number of equal-width theta bins (default 10).",
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
    if int(args.n_ref) < 2:
        raise ValueError("--n-ref must be >= 2.")
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
    setattr(full_args, "compute_h_matrix", True)
    setattr(full_args, "h_restore_original_order", True)
    setattr(full_args, "skip_shared_fisher_gt_compare", True)
    # Persist C-matrix for Bayes-opt row (bin-mean Δℓ from score-matching L pipeline).
    setattr(full_args, "h_save_intermediates", True)
    validate_estimation_args(full_args)
    return full_args


def _run_ctx_for_bundle(
    args: argparse.Namespace,
    meta: dict,
    bundle: SharedDatasetBundle,
    full_args: SimpleNamespace,
    n_bins: int,
) -> vhb.RunContext:
    np.random.seed(int(meta["seed"]))
    torch.manual_seed(int(meta["seed"]))
    rng = np.random.default_rng(int(meta["seed"]))
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


def _load_c_matrix_from_h_npz(h_path: str) -> np.ndarray:
    """Load integrated C matrix from h_matrix_results*.npz (requires h_save_intermediates)."""
    z = np.load(h_path, allow_pickle=True)
    if "c_matrix" not in z.files:
        raise FileNotFoundError(
            f"c_matrix not found in {h_path}. "
            "Convergence study forces --h-save-intermediates when estimating H."
        )
    return np.asarray(z["c_matrix"], dtype=np.float64)


def _c_matrix_bayes_opt_accuracy_matrix(
    c_matrix: np.ndarray,
    bin_idx: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Bayes-opt pairwise accuracy from bin-mean C differences (score-matching L pipeline).

    For sample row ``r``, ``mu_b(r) = mean_{k : bin_idx[k] == b} C[r, k]``.
    ``Delta_ij(r) = mu_i(r) - mu_j(r)``.
    ``A*_ij = 0.5 * P_{r in bin i}[Delta_ij > 0] + 0.5 * P_{r in bin j}[Delta_ij < 0]``.
    Diagonal NaN; empty-bin pairs skipped (NaN).
    """
    c = np.asarray(c_matrix, dtype=np.float64)
    bi = np.asarray(bin_idx, dtype=np.int64).reshape(-1)
    n = int(c.shape[0])
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError(f"c_matrix must be square; got {c.shape}")
    if bi.shape[0] != n:
        raise ValueError("bin_idx length must match c_matrix order.")
    if n_bins < 2:
        return np.full((n_bins, n_bins), np.nan, dtype=np.float64)

    cols_in_bin = [np.flatnonzero(bi == b) for b in range(n_bins)]
    a = np.full((n_bins, n_bins), np.nan, dtype=np.float64)
    for i in range(n_bins):
        for j in range(n_bins):
            if i == j:
                continue
            ci, cj = cols_in_bin[i], cols_in_bin[j]
            if ci.size == 0 or cj.size == 0:
                continue
            mu_i = np.mean(c[:, ci], axis=1)
            mu_j = np.mean(c[:, cj], axis=1)
            d = mu_i - mu_j
            ri = np.flatnonzero(bi == i)
            rj = np.flatnonzero(bi == j)
            if ri.size == 0 or rj.size == 0:
                continue
            p_i = float(np.mean(d[ri] > 0.0))
            p_j = float(np.mean(d[rj] < 0.0))
            a[i, j] = 0.5 * (p_i + p_j)
    return a


def _metrics_fixed_edges(
    loaded: vhb.LoadedHMatrix,
    x_aligned: np.ndarray,
    edges: np.ndarray,
    n_bins: int,
    clf_test_frac: float,
    clf_min_class_count: int,
    clf_random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    hell_lb = vhb.hellinger_acc_lb_from_binned_h_squared(h_binned)
    c_mat = _load_c_matrix_from_h_npz(loaded.h_path)
    if c_mat.shape[0] != loaded.theta_used.shape[0]:
        raise ValueError(
            f"c_matrix rows {c_mat.shape[0]} != theta_used {loaded.theta_used.shape[0]}"
        )
    bayes_c = _c_matrix_bayes_opt_accuracy_matrix(c_mat, bin_idx, n_bins)
    return h_binned, clf_acc, hell_lb, bayes_c


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
    hell_mats: list[np.ndarray],
    bayes_c_mats: list[np.ndarray],
    col_labels: list[str],
    out_path: str,
    n_bins: int,
) -> None:
    """Four rows: binned H, pairwise decoding, Hellinger LB, Bayes-opt acc from C-matrix."""
    n_cols = len(h_mats)
    if (
        n_cols != len(clf_mats)
        or n_cols != len(hell_mats)
        or n_cols != len(bayes_c_mats)
        or n_cols != len(col_labels)
    ):
        raise ValueError("h_mats, clf_mats, hell_mats, bayes_c_mats, col_labels length mismatch.")
    fig, axes = plt.subplots(4, n_cols, figsize=(2.8 * n_cols, 10.4), squeeze=False)
    # Binned H is treated on [0, 1] for a consistent cross-column color scale.
    vmin_h, vmax_h = 0.0, 1.0
    vmin_c, vmax_c = _finite_min_max(clf_mats)
    if vmin_c >= vmax_c:
        vmax_c = vmin_c + 1e-12
    vmin_hell, vmax_hell = 0.0, 1.0
    vmin_bayes, vmax_bayes = 0.0, 1.0
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
            axes[0, c].set_ylabel("Binned H", fontsize=11)
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

        im2 = axes[2, c].imshow(
            hell_mats[c],
            vmin=vmin_hell,
            vmax=vmax_hell,
            cmap=cmap,
            aspect="equal",
            origin="lower",
        )
        axes[2, c].set_xticks(range(n_bins))
        axes[2, c].set_yticks(range(n_bins))
        axes[2, c].tick_params(labelsize=7)
        if c == 0:
            axes[2, c].set_ylabel("Hellinger LB (H)", fontsize=11)
        plt.colorbar(im2, ax=axes[2, c], fraction=0.046, pad=0.04)

        im3 = axes[3, c].imshow(
            bayes_c_mats[c],
            vmin=vmin_bayes,
            vmax=vmax_bayes,
            cmap=cmap,
            aspect="equal",
            origin="lower",
        )
        axes[3, c].set_xticks(range(n_bins))
        axes[3, c].set_yticks(range(n_bins))
        axes[3, c].tick_params(labelsize=7)
        if c == 0:
            axes[3, c].set_ylabel("Bayes-opt acc (C / L)", fontsize=11)
        plt.colorbar(im3, ax=axes[3, c], fraction=0.046, pad=0.04)

    fig.suptitle(
        "Binned H, pairwise decoding, Hellinger LB, Bayes-opt acc from C (rows 1–4) by nested subset size",
        fontsize=11,
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
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("study_h_decoding_convergence\n")
        f.write(f"dataset_npz: {args.dataset_npz}\n")
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


def main(argv: list[str] | None = None) -> None:
    # When stdout is redirected (e.g. nohup), default block buffering delays run.log updates.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass
    p = build_parser()
    args = p.parse_args(argv)
    _validate_cli(args)
    ns = _parse_n_list(args.n_list)

    os.makedirs(args.output_dir, exist_ok=True)
    bundle = load_shared_dataset_npz(args.dataset_npz)
    meta = bundle.meta
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
    perm_seed = int(meta["seed"]) + int(args.subset_seed_offset)
    rng_perm = np.random.default_rng(perm_seed)
    perm = rng_perm.permutation(n_pool)

    theta_ref = np.asarray(bundle.theta_all[perm[: int(args.n_ref)]], dtype=np.float64).reshape(-1)
    edges, edge_lo, edge_hi = vhb.theta_bin_edges(theta_ref, n_bins)

    clf_rs = int(meta["seed"]) if int(args.clf_random_state) < 0 else int(args.clf_random_state)

    ref_dir = os.path.join(args.output_dir, "reference")
    os.makedirs(ref_dir, exist_ok=True)
    print(
        "[convergence] H from DSM: posterior + prior score nets "
        f"(score_arch={getattr(args, 'score_arch', 'mlp')}, "
        f"prior_arch={getattr(args, 'prior_score_arch', 'mlp')})",
        flush=True,
    )
    print(f"[convergence] reference n={args.n_ref} -> {ref_dir}", flush=True)
    print(f"[convergence] n_list={ns}", flush=True)
    t0 = time.time()
    bundle_ref = _subset_bundle(bundle, perm, int(args.n_ref), meta)
    loaded_ref, x_ref, _ = _estimate_one(
        args=args,
        meta=meta,
        bundle=bundle_ref,
        output_dir=ref_dir,
        n_bins=n_bins,
    )
    h_ref, clf_ref, hell_ref, bayes_ref = _metrics_fixed_edges(
        loaded_ref,
        x_ref,
        edges,
        n_bins,
        float(args.clf_test_frac),
        int(args.clf_min_class_count),
        clf_rs,
    )
    print(f"[convergence] reference wall time: {time.time() - t0:.1f}s")

    loss_dir = os.path.join(args.output_dir, "training_losses")
    os.makedirs(loss_dir, exist_ok=True)
    ref_loss_npz = os.path.join(ref_dir, "score_prior_training_losses.npz")
    if os.path.isfile(ref_loss_npz):
        shutil.copy2(
            ref_loss_npz,
            os.path.join(loss_dir, f"n_ref_{int(args.n_ref)}.npz"),
        )

    np.savez_compressed(
        os.path.join(args.output_dir, "h_decoding_convergence_reference.npz"),
        h_binned_ref=h_ref,
        clf_acc_ref=clf_ref,
        hellinger_lb_ref=hell_ref,
        bayes_c_acc_ref=bayes_ref,
        theta_bin_edges=edges,
        edge_lo=np.float64(edge_lo),
        edge_hi=np.float64(edge_hi),
        perm_seed=np.int64(perm_seed),
        n_ref=np.int64(args.n_ref),
    )

    corr_h = np.full(len(ns), np.nan, dtype=np.float64)
    corr_clf = np.full(len(ns), np.nan, dtype=np.float64)
    corr_hell_lb = np.full(len(ns), np.nan, dtype=np.float64)
    corr_bayes_c = np.full(len(ns), np.nan, dtype=np.float64)
    wall_s = np.full(len(ns), np.nan, dtype=np.float64)
    err_msg: list[str] = []
    h_sweep: list[np.ndarray] = []
    clf_sweep: list[np.ndarray] = []
    hell_sweep: list[np.ndarray] = []
    bayes_sweep: list[np.ndarray] = []

    sweep_root = os.path.join(args.output_dir, "sweep_runs")
    if bool(args.keep_intermediate):
        os.makedirs(sweep_root, exist_ok=True)

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
            h_n, clf_n, hell_n, bayes_n = _metrics_fixed_edges(
                loaded_n,
                x_n,
                edges,
                n_bins,
                float(args.clf_test_frac),
                int(args.clf_min_class_count),
                clf_rs,
            )
            corr_h[k] = vhb.matrix_corr_offdiag(h_n, h_ref)
            corr_clf[k] = vhb.matrix_corr_offdiag(clf_n, clf_ref)
            corr_hell_lb[k] = vhb.matrix_corr_offdiag(hell_n, hell_ref)
            corr_bayes_c[k] = vhb.matrix_corr_offdiag(bayes_n, bayes_ref)
            wall_s[k] = time.time() - t1
            h_sweep.append(np.asarray(h_n, dtype=np.float64))
            clf_sweep.append(np.asarray(clf_n, dtype=np.float64))
            hell_sweep.append(np.asarray(hell_n, dtype=np.float64))
            bayes_sweep.append(np.asarray(bayes_n, dtype=np.float64))
            print(
                f"[convergence] n={n}  corr_h={corr_h[k]:.4f}  corr_clf={corr_clf[k]:.4f}  "
                f"corr_hell_lb={corr_hell_lb[k]:.4f}  corr_bayes_c={corr_bayes_c[k]:.4f}  "
                f"wall={wall_s[k]:.1f}s",
                flush=True,
            )
            run_loss_npz = os.path.join(run_dir, "score_prior_training_losses.npz")
            if os.path.isfile(run_loss_npz):
                shutil.copy2(run_loss_npz, os.path.join(loss_dir, f"n_{n:06d}.npz"))
        except Exception as e:
            err_msg.append(f"n={n}: {e!r}")
            print(f"[convergence] ERROR n={n}: {e}")
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if (
        len(h_sweep) != len(ns)
        or len(clf_sweep) != len(ns)
        or len(hell_sweep) != len(ns)
        or len(bayes_sweep) != len(ns)
    ):
        raise RuntimeError(
            "Missing binned matrices for some n (partial failures). "
            "Fix errors above or re-run with a smaller n-list."
        )
    h_cols = np.stack(h_sweep + [h_ref], axis=0)
    clf_cols = np.stack(clf_sweep + [clf_ref], axis=0)
    hell_cols = np.stack(hell_sweep + [hell_ref], axis=0)
    bayes_cols = np.stack(bayes_sweep + [bayes_ref], axis=0)
    column_n = np.asarray(list(ns) + [int(args.n_ref)], dtype=np.int64)

    out_npz = os.path.join(args.output_dir, "h_decoding_convergence_results.npz")
    np.savez_compressed(
        out_npz,
        n=np.asarray(ns, dtype=np.int64),
        corr_h_binned_vs_ref=corr_h,
        corr_clf_vs_ref=corr_clf,
        corr_hell_lb_vs_ref=corr_hell_lb,
        corr_bayes_c_vs_ref=corr_bayes_c,
        wall_seconds=wall_s,
        n_ref=np.int64(args.n_ref),
        perm_seed=np.int64(perm_seed),
        theta_bin_edges=edges,
        h_binned_columns=h_cols,
        clf_acc_columns=clf_cols,
        hell_lb_columns=hell_cols,
        bayes_c_acc_columns=bayes_cols,
        column_n=column_n,
    )

    csv_path = os.path.join(args.output_dir, "h_decoding_convergence_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "n",
                "corr_h_binned_vs_ref",
                "corr_clf_vs_ref",
                "corr_hell_lb_vs_ref",
                "corr_bayes_c_vs_ref",
                "wall_seconds",
            ]
        )
        for i, n in enumerate(ns):
            w.writerow([n, corr_h[i], corr_clf[i], corr_hell_lb[i], corr_bayes_c[i], wall_s[i]])

    fig_path = os.path.join(args.output_dir, "h_decoding_convergence.png")
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 4.8))
    ax.plot(
        ns,
        corr_h,
        color="#1f77b4",
        linewidth=1.8,
        marker="o",
        markersize=6,
        label="Binned H vs ref",
    )
    ax.plot(
        ns,
        corr_clf,
        color="#d62728",
        linewidth=1.8,
        marker="s",
        markersize=5,
        label="Pairwise decoding vs ref",
    )
    ax.plot(
        ns,
        corr_hell_lb,
        color="#2ca02c",
        linewidth=1.8,
        marker="^",
        markersize=6,
        label="Hellinger LB (from binned H) vs ref",
    )
    ax.plot(
        ns,
        corr_bayes_c,
        color="#9467bd",
        linewidth=1.8,
        marker="D",
        markersize=5,
        label="Bayes-opt acc (C / L est.) vs ref",
    )
    ax.set_xlabel("dataset size n (nested subset)")
    ax.set_ylabel("corr (off-diag)")
    ax.set_title("Convergence to n_ref=%d reference" % int(args.n_ref))
    ax.set_xticks(ns)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    conv_svg = _save_figure_png_svg(fig, fig_path, dpi=160)
    plt.close(fig)

    matrix_panel_path = os.path.join(args.output_dir, "h_decoding_matrices_panel.png")
    col_labels = [f"n={n}" for n in ns] + [f"n_ref={int(args.n_ref)}"]
    _render_matrix_panel(
        h_mats=list(h_cols),
        clf_mats=list(clf_cols),
        hell_mats=list(hell_cols),
        bayes_c_mats=list(bayes_cols),
        col_labels=col_labels,
        out_path=matrix_panel_path,
        n_bins=n_bins,
    )

    manifest_lines = [
        f"n_ref_{int(args.n_ref)}\ttraining_losses/n_ref_{int(args.n_ref)}.npz",
        *[f"{n}\ttraining_losses/n_{n:06d}.npz" for n in ns],
    ]
    manifest_path = os.path.join(loss_dir, "manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        mf.write("# n_or_nref_label\trelative_path (score_prior_training_losses copies)\n")
        for line in manifest_lines:
            mf.write(line + "\n")

    summary_path = os.path.join(args.output_dir, "h_decoding_convergence_summary.txt")
    paths_out = {
        "results_npz": out_npz,
        "results_csv": csv_path,
        "figure": fig_path,
        "figure_svg": conv_svg,
        "matrix_panel": matrix_panel_path,
        "matrix_panel_svg": str(Path(matrix_panel_path).with_suffix(".svg")),
        "reference_npz": os.path.join(args.output_dir, "h_decoding_convergence_reference.npz"),
        "training_losses_dir": loss_dir,
        "training_losses_manifest": manifest_path,
    }
    if err_msg:
        paths_out["errors"] = "; ".join(err_msg[:20])
    _write_summary(summary_path, args, meta, perm_seed, n_pool, ref_dir, paths_out)

    print("[convergence] Saved:")
    print(f"  - {out_npz}")
    print(f"  - {csv_path}")
    print(f"  - {fig_path}")
    print(f"  - {conv_svg}")
    print(f"  - {matrix_panel_path}")
    print(f"  - {paths_out['matrix_panel_svg']}")
    print(f"  - {loss_dir}/ (per-n training loss .npz + manifest.txt)")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()
