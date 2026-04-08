#!/usr/bin/env python3
"""Load a shared dataset, estimate H-matrix, bin theta, average H_sym by bins, visualize.

Also builds a pairwise theta-bin logistic-regression accuracy matrix on x (same rows as H).

From ``delta_l_matrix`` in the H-matrix NPZ (requires ``--h-save-intermediates`` when training),
bins pairwise log-likelihood ratios ``ΔL_ij``, maps to symmetric accuracy
``0.5*(f(\\Delta L)+f(\\Delta L^T))`` with ``f(z)=\\max(\\sigma(z),1-\\sigma(z))``, and adds a panel to the composite figure.
Panel titles follow ``h_field_method`` in the NPZ (DSM: denoising score / ``\\nabla_\\theta \\log p(\\theta|x)``; flow: velocity field at ``t``).

SSSD (kernel-smoothed decoder): trains a sigma-conditioned softmax decoder with soft bin targets
and reports symmetric pairwise LR decision accuracies A_ij(sigma) (primary); log-ratio M_ij in NPZ.

Additionally computes a ground-truth approximation matrix by applying the same decoding scheme
to a larger sampled dataset (default n=10000), and reports correlations against this matrix.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
from fisher import sssd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Load a shared dataset .npz, run Fisher + H-matrix estimation (or load existing), "
            "bin theta into K bins, average H_sym over bin pairs, save heatmap + NPZ."
        )
    )
    p.add_argument(
        "--dataset-npz",
        type=str,
        required=True,
        help="Path to shared dataset .npz from fisher_make_dataset.py.",
    )
    p.add_argument(
        "--num-theta-bins",
        type=int,
        default=15,
        help="Number of equal-width theta bins (default 15).",
    )
    p.add_argument(
        "--theta-bin-mode",
        type=str,
        default="range",
        choices=["range", "meta_range"],
        help=(
            "How to set theta bin edges: "
            "'range' uses min/max of theta_used from H-matrix; "
            "'meta_range' uses theta_low/theta_high from dataset meta."
        ),
    )
    p.add_argument(
        "--h-only",
        "--mds-only",
        dest="h_only",
        action="store_true",
        default=False,
        help="Skip training; load h_matrix_results*.npz from output-dir (or from --h-matrix-npz).",
    )
    p.add_argument(
        "--h-matrix-npz",
        type=str,
        default=None,
        help=(
            "With --h-only: path to an existing h_matrix_results*.npz. "
            "If omitted, uses output-dir/h_matrix_results{suffix}.npz."
        ),
    )
    p.add_argument(
        "--clf-test-frac",
        type=float,
        default=0.2,
        help="Held-out fraction for pairwise bin-vs-bin logistic regression (default 0.2).",
    )
    p.add_argument(
        "--clf-min-class-count",
        type=int,
        default=5,
        help="Minimum samples per bin class required to train a pairwise classifier (default 5).",
    )
    p.add_argument(
        "--clf-max-iter",
        type=int,
        default=1000,
        help="Max iterations for sklearn LogisticRegression (default 1000).",
    )
    p.add_argument(
        "--clf-random-state",
        type=int,
        default=-1,
        help="Random seed for train/test split; -1 uses dataset seed from NPZ meta.",
    )
    p.add_argument(
        "--gt-approx-n-total",
        type=int,
        default=10000,
        help="Sample count for GT approximation via pairwise decoding (default 10000).",
    )
    p.add_argument(
        "--gt-approx-seed",
        type=int,
        default=-1,
        help="Sampling seed for GT approximation; -1 uses dataset seed + 17.",
    )
    sssd.add_sssd_cli_arguments(p)
    add_estimation_arguments(p)
    p.set_defaults(output_dir=str(Path(DATA_DIR) / "outputs_h_matrix_binned"))
    return p.parse_args()


def theta_bin_edges(
    theta_used: np.ndarray,
    meta: dict,
    n_bins: int,
    mode: str,
) -> tuple[np.ndarray, float, float]:
    """Return (edges length n_bins+1, theta_low_used, theta_high_used)."""
    th = np.asarray(theta_used, dtype=np.float64).reshape(-1)
    if n_bins < 1:
        raise ValueError("--num-theta-bins must be >= 1.")
    if mode == "range":
        lo = float(np.min(th))
        hi = float(np.max(th))
    elif mode == "meta_range":
        lo = float(meta["theta_low"])
        hi = float(meta["theta_high"])
    else:
        raise ValueError(f"Unknown theta-bin-mode: {mode}")
    if hi <= lo:
        raise ValueError(f"Invalid theta range for binning: [{lo}, {hi}]")
    edges = np.linspace(lo, hi, n_bins + 1, dtype=np.float64)
    return edges, lo, hi


def theta_to_bin_index(theta: np.ndarray, edges: np.ndarray, n_bins: int) -> np.ndarray:
    """Map each theta to an integer bin in [0, n_bins-1]."""
    th = np.asarray(theta, dtype=np.float64).reshape(-1)
    # searchsorted(..., side="right") - 1 puts values in [edges[i], edges[i+1]) mostly;
    # clip handles edge cases at boundaries.
    idx = np.searchsorted(edges, th, side="right") - 1
    return np.clip(idx, 0, n_bins - 1).astype(np.int64)


def average_h_by_bins(
    h_sym: np.ndarray,
    bin_idx: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average H_ij over i in bin a, j in bin b. Returns (h_binned, count_matrix)."""
    h = np.asarray(h_sym, dtype=np.float64)
    n = h.shape[0]
    if h.shape != (n, n) or bin_idx.shape[0] != n:
        raise ValueError("h_sym and bin_idx shape mismatch.")
    h_binned = np.full((n_bins, n_bins), np.nan, dtype=np.float64)
    count_matrix = np.zeros((n_bins, n_bins), dtype=np.int64)

    for a in range(n_bins):
        ia = np.flatnonzero(bin_idx == a)
        na = int(ia.size)
        for b in range(n_bins):
            jb = np.flatnonzero(bin_idx == b)
            nb = int(jb.size)
            if na == 0 or nb == 0:
                continue
            sub = h[np.ix_(ia, jb)]
            h_binned[a, b] = float(np.mean(sub))
            count_matrix[a, b] = na * nb

    return h_binned, count_matrix


def average_matrix_by_bins(
    mat: np.ndarray,
    bin_idx: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average arbitrary square matrix entries over theta-bin pairs (a,b). Returns (binned, count_matrix)."""
    m = np.asarray(mat, dtype=np.float64)
    n = m.shape[0]
    if m.ndim != 2 or m.shape != (n, n) or bin_idx.shape[0] != n:
        raise ValueError("mat must be square (N,N) and bin_idx must have length N.")
    out = np.full((n_bins, n_bins), np.nan, dtype=np.float64)
    count_matrix = np.zeros((n_bins, n_bins), dtype=np.int64)

    for a in range(n_bins):
        ia = np.flatnonzero(bin_idx == a)
        na = int(ia.size)
        for b in range(n_bins):
            jb = np.flatnonzero(bin_idx == b)
            nb = int(jb.size)
            if na == 0 or nb == 0:
                continue
            sub = m[np.ix_(ia, jb)]
            out[a, b] = float(np.nanmean(sub))
            count_matrix[a, b] = na * nb

    return out, count_matrix


def lr_symmetric_accuracy_from_binned_delta_l(delta_l_binned: np.ndarray) -> np.ndarray:
    """Symmetric pairwise score from binned log-ratios.

    Uses per-entry confidence ``f(z)=max(σ(z),1-σ(z))`` then
    ``A_ij = 0.5*(f(ΔL_ij)+f(ΔL_ji))``. Diagonal NaN. Values lie in ``[0.5, 1]`` off-diagonal.
    """
    dl = np.asarray(delta_l_binned, dtype=np.float64)
    if dl.ndim != 2 or dl.shape[0] != dl.shape[1]:
        raise ValueError("delta_l_binned must be square.")
    n = int(dl.shape[0])
    z = np.clip(dl, -60.0, 60.0)
    p = 1.0 / (1.0 + np.exp(-z))
    conf = np.maximum(p, 1.0 - p)
    acc = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            acc[i, j] = 0.5 * (float(conf[i, j]) + float(conf[j, i]))
    return acc


def delta_l_figure_labels(h_field_method: str) -> tuple[str, str, str]:
    """Return (panel_title_top, suptitle_delta_l_fragment, summary_line_prefix) for ΔL panel text.

    ``h_field_method`` comes from ``h_matrix_results*.npz`` (``dsm`` vs ``flow``).
    """
    m = str(h_field_method).strip().lower()
    if m == "flow":
        return (
            r"Binned flow $\Delta L_{ij}$ → score",
            r"flow $\Delta L$",
            "Flow velocity field: Delta L (from h_matrix_results*.npz delta_l_matrix): ",
        )
    return (
        r"Binned DSM $\Delta L_{ij}$ → score"
        + "\n"
        + r"(denoising score matching: $\nabla_\theta \log p(\theta \mid x)$)",
        r"DSM $\Delta L$",
        "DSM score-field Delta L (from h_matrix_results*.npz delta_l_matrix; "
        "posterior/prior denoising scores, not flow velocity): ",
    )


def validate_symmetric_loss_matrix(mat: np.ndarray, name: str) -> None:
    m = np.asarray(mat, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"{name} must be square.")
    asym = float(np.max(np.abs(m - m.T)))
    if asym > 1e-8:
        raise ValueError(f"{name} is not symmetric within tolerance: max|m-m.T|={asym}")
    if not np.all(np.isfinite(m)):
        raise ValueError(f"{name} contains non-finite values.")


def theta_for_fisher_alignment(bundle: SharedDatasetBundle, full_args: SimpleNamespace) -> np.ndarray:
    mode = str(getattr(full_args, "score_fisher_eval_data", "full"))
    if mode == "full":
        return np.asarray(bundle.theta_all, dtype=np.float64).reshape(-1)
    if mode == "score_eval":
        return np.asarray(bundle.theta_eval, dtype=np.float64).reshape(-1)
    raise ValueError(f"Unknown score_fisher_eval_data: {mode}")


def x_for_fisher_alignment(bundle: SharedDatasetBundle, full_args: SimpleNamespace) -> np.ndarray:
    mode = str(getattr(full_args, "score_fisher_eval_data", "full"))
    if mode == "full":
        return np.asarray(bundle.x_all, dtype=np.float64)
    if mode == "score_eval":
        return np.asarray(bundle.x_eval, dtype=np.float64)
    raise ValueError(f"Unknown score_fisher_eval_data: {mode}")


def pairwise_bin_logistic_accuracy_matrix(
    x: np.ndarray,
    bin_idx: np.ndarray,
    n_bins: int,
    *,
    test_frac: float,
    min_class_count: int,
    max_iter: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Return (accuracy_matrix, valid_mask, pair_support, diag_stats).

    For each pair of bins (i, j) with i < j, train a binary logistic regression on x
    to distinguish samples from bin i (label 0) vs bin j (label 1). Values are mirrored
    to (j, i). Diagonal is NaN.
    """
    x2 = np.asarray(x, dtype=np.float64)
    if x2.ndim != 2:
        raise ValueError("x must be 2D.")
    bi = np.asarray(bin_idx, dtype=np.int64).reshape(-1)
    if x2.shape[0] != bi.shape[0]:
        raise ValueError("x and bin_idx must have the same number of rows.")
    if not (0.0 < float(test_frac) < 1.0):
        raise ValueError("--clf-test-frac must be in (0, 1).")
    if min_class_count < 1:
        raise ValueError("--clf-min-class-count must be >= 1.")

    acc = np.full((n_bins, n_bins), np.nan, dtype=np.float64)
    valid = np.zeros((n_bins, n_bins), dtype=bool)
    support = np.zeros((n_bins, n_bins), dtype=np.int64)
    stats = {
        "insufficient_counts": 0,
        "split_fail": 0,
        "fit_fail": 0,
        "ok_pairs": 0,
    }

    rs = int(random_state)

    for i in range(n_bins):
        for j in range(i+1, n_bins):
            ia = np.flatnonzero(bi == i)
            jb = np.flatnonzero(bi == j)
            ni, nj = int(ia.size), int(jb.size)
            support[i, j] = ni + nj
            support[j, i] = ni + nj

            if ni < min_class_count or nj < min_class_count:
                stats["insufficient_counts"] += 1
                continue

            X = np.vstack([x2[ia], x2[jb]])
            y = np.concatenate([np.zeros(ni, dtype=np.int64), np.ones(nj, dtype=np.int64)])

            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=float(test_frac), stratify=y, random_state=rs
                )
            except ValueError:
                try:
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y, test_size=float(test_frac), random_state=rs
                    )
                except ValueError:
                    stats["split_fail"] += 1
                    continue

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                stats["split_fail"] += 1
                continue

            try:
                clf = LogisticRegression(
                    max_iter=int(max_iter),
                    solver="lbfgs",
                    random_state=rs,
                )
                clf.fit(X_tr, y_tr)
                score = float(clf.score(X_te, y_te))
            except Exception:
                stats["fit_fail"] += 1
                continue

            acc[i, j] = score
            acc[j, i] = score
            valid[i, j] = True
            valid[j, i] = True
            stats["ok_pairs"] += 1

    for k in range(n_bins):
        acc[k, k] = np.nan

    return acc, valid, support, stats


def matrix_corr_offdiag(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over finite off-diagonal entries; NaN if undefined."""
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape or aa.ndim != 2 or aa.shape[0] != aa.shape[1]:
        raise ValueError("matrix_corr_offdiag requires equal-shape square matrices.")
    n = aa.shape[0]
    off = ~np.eye(n, dtype=bool)
    mask = off & np.isfinite(aa) & np.isfinite(bb)
    if int(np.sum(mask)) < 3:
        return float("nan")
    av = aa[mask]
    bv = bb[mask]
    if float(np.std(av)) <= 0.0 or float(np.std(bv)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def main() -> None:
    args = parse_args()
    dataset_npz = args.dataset_npz
    n_bins = int(args.num_theta_bins)
    bin_mode = str(args.theta_bin_mode)

    validate_estimation_args(args)
    if n_bins < 1:
        raise ValueError("--num-theta-bins must be >= 1.")
    if not (0.0 < float(args.clf_test_frac) < 1.0):
        raise ValueError("--clf-test-frac must be in (0, 1).")
    if int(args.clf_min_class_count) < 1:
        raise ValueError("--clf-min-class-count must be >= 1.")
    if int(args.clf_max_iter) < 1:
        raise ValueError("--clf-max-iter must be >= 1.")
    if int(args.gt_approx_n_total) < 2:
        raise ValueError("--gt-approx-n-total must be >= 2.")
    if int(args.sssd_epochs) < 1:
        raise ValueError("--sssd-epochs must be >= 1.")
    if int(args.sssd_patience) < 0:
        raise ValueError("--sssd-patience must be >= 0 (0 disables early stopping).")
    if int(args.sssd_batch_size) < 1:
        raise ValueError("--sssd-batch-size must be >= 1.")
    if not (0.0 < float(args.sssd_val_frac) < 1.0):
        raise ValueError("--sssd-val-frac must be in (0, 1).")
    if str(args.sssd_sigmas).strip().lower() == "auto" and int(args.sssd_n_sigmas) < 2:
        raise ValueError("--sssd-n-sigmas must be >= 2 when using --sssd-sigmas=auto.")
    bundle = load_shared_dataset_npz(dataset_npz)
    meta = bundle.meta
    full_args = merge_meta_into_args(meta, args)

    setattr(full_args, "compute_h_matrix", True)
    setattr(full_args, "h_restore_original_order", True)
    # Required for binned ΔL → accuracy panel (delta_l_matrix in h_matrix_results*.npz).
    setattr(full_args, "h_save_intermediates", True)

    np.random.seed(int(meta["seed"]))
    torch.manual_seed(int(meta["seed"]))
    rng = np.random.default_rng(int(meta["seed"]))

    dev = require_device(str(full_args.device))

    dataset = build_dataset_from_meta(meta)
    os.makedirs(full_args.output_dir, exist_ok=True)

    print(
        f"[h_binned] dataset_npz={dataset_npz} output_dir={full_args.output_dir} "
        f"num_theta_bins={n_bins} theta_bin_mode={bin_mode} h_only={bool(args.h_only)} "
        f"theta_field_method={str(getattr(full_args, 'theta_field_method', 'dsm'))}"
    )

    if not bool(args.h_only):
        run_shared_fisher_estimation(
            full_args,
            dataset,
            theta_all=bundle.theta_all,
            x_all=bundle.x_all,
            theta_train=bundle.theta_train,
            x_train=bundle.x_train,
            theta_eval=bundle.theta_eval,
            x_eval=bundle.x_eval,
            rng=rng,
        )
    else:
        print("[h_binned] --h-only: skipping Fisher training; using existing h_matrix_results*.npz.")

    suffix = "_non_gauss" if full_args.dataset_family == "gmm_non_gauss" else "_theta_cov"
    if args.h_matrix_npz:
        h_path = os.path.abspath(args.h_matrix_npz)
    else:
        h_path = os.path.join(full_args.output_dir, f"h_matrix_results{suffix}.npz")
    if not os.path.isfile(h_path):
        raise FileNotFoundError(f"Expected H-matrix file not found: {h_path}")

    h_npz = np.load(h_path, allow_pickle=True)
    h_sym = np.asarray(h_npz["h_sym"], dtype=np.float64)
    theta_used = np.asarray(h_npz["theta_used"], dtype=np.float64).reshape(-1)
    h_field_method = str(h_npz["h_field_method"][0]) if "h_field_method" in h_npz.files else "dsm"
    h_eval_scalar_name = (
        str(h_npz["h_eval_scalar_name"][0]) if "h_eval_scalar_name" in h_npz.files else "sigma_eval"
    )
    h_eval_scalar_value = (
        float(np.asarray(h_npz["sigma_eval"], dtype=np.float64).reshape(-1)[0])
        if "sigma_eval" in h_npz.files
        else float("nan")
    )
    dl_panel_title_top, dl_suptitle_delta_l_tag, dl_summary_delta_l_prefix = delta_l_figure_labels(
        h_field_method
    )

    theta_chk = theta_for_fisher_alignment(bundle, full_args)
    if theta_chk.shape[0] != theta_used.shape[0]:
        raise ValueError(
            f"theta/H row mismatch: theta_chk={theta_chk.shape[0]} theta_used={theta_used.shape[0]}"
        )
    if not np.allclose(theta_chk, theta_used, rtol=0.0, atol=1e-5):
        raise ValueError(
            "theta_used from H-matrix npz does not match dataset theta for score_fisher_eval_data split."
        )

    x_chk = x_for_fisher_alignment(bundle, full_args)
    if x_chk.shape[0] != theta_used.shape[0]:
        raise ValueError(
            f"x/H row mismatch: x_aligned={x_chk.shape[0]} theta_used={theta_used.shape[0]}"
        )

    validate_symmetric_loss_matrix(h_sym, "h_sym")

    if "delta_l_matrix" not in h_npz.files:
        raise ValueError(
            "H-matrix NPZ does not contain 'delta_l_matrix'. "
            "Re-run Fisher/H-matrix estimation with --h-save-intermediates so ΔL_ij is saved, "
            "then re-run this script."
        )
    delta_l_matrix = np.asarray(h_npz["delta_l_matrix"], dtype=np.float64)
    if delta_l_matrix.shape != h_sym.shape:
        raise ValueError(
            f"delta_l_matrix shape {delta_l_matrix.shape} does not match h_sym {h_sym.shape}."
        )

    edges, edge_lo, edge_hi = theta_bin_edges(theta_used, meta, n_bins, bin_mode)
    bin_idx = theta_to_bin_index(theta_used, edges, n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    h_binned, count_matrix = average_h_by_bins(h_sym, bin_idx, n_bins)
    h_binned_sqrt = np.sqrt(np.clip(h_binned, 0.0, None))

    clf_rs = int(meta["seed"]) if int(args.clf_random_state) < 0 else int(args.clf_random_state)
    clf_acc, clf_valid, clf_support, clf_stats = pairwise_bin_logistic_accuracy_matrix(
        x_chk,
        bin_idx,
        n_bins,
        test_frac=float(args.clf_test_frac),
        min_class_count=int(args.clf_min_class_count),
        max_iter=int(args.clf_max_iter),
        random_state=clf_rs,
    )
    gt_seed = (int(meta["seed"]) + 17) if int(args.gt_approx_seed) < 0 else int(args.gt_approx_seed)
    theta_gt, x_gt = dataset.sample_joint(int(args.gt_approx_n_total))
    theta_gt = np.asarray(theta_gt, dtype=np.float64).reshape(-1)
    x_gt = np.asarray(x_gt, dtype=np.float64)
    gt_bin_idx = theta_to_bin_index(theta_gt, edges, n_bins)
    gt_acc, gt_valid, gt_support, gt_stats = pairwise_bin_logistic_accuracy_matrix(
        x_gt,
        gt_bin_idx,
        n_bins,
        test_frac=float(args.clf_test_frac),
        min_class_count=int(args.clf_min_class_count),
        max_iter=int(args.clf_max_iter),
        random_state=gt_seed,
    )
    corr_h_vs_gt = matrix_corr_offdiag(h_binned_sqrt, gt_acc)
    corr_clf_vs_gt = matrix_corr_offdiag(clf_acc, gt_acc)

    delta_l_binned, count_matrix_dl = average_matrix_by_bins(delta_l_matrix, bin_idx, n_bins)
    lr_acc_binned_from_delta_l = lr_symmetric_accuracy_from_binned_delta_l(delta_l_binned)
    corr_lr_dl_vs_gt = matrix_corr_offdiag(lr_acc_binned_from_delta_l, gt_acc)

    # --- SSSD: kernel-smoothed decoder + symmetric M_ij(sigma) ---
    sssd_eval_sigmas = np.asarray([], dtype=np.float64)
    sssd_M_stack = np.zeros((0, n_bins, n_bins), dtype=np.float64)
    sssd_gt_M_stack = np.zeros((0, n_bins, n_bins), dtype=np.float64)
    sssd_train_losses = np.asarray([], dtype=np.float64)
    sssd_best_epoch = -1
    sssd_sigma_min_used = float("nan")
    sssd_sigma_max_used = float("nan")
    corr_sssd_M_vs_gt_M = np.asarray([], dtype=np.float64)
    sssd_acc_stack = np.zeros((0, n_bins, n_bins), dtype=np.float64)
    sssd_gt_acc_stack = np.zeros((0, n_bins, n_bins), dtype=np.float64)
    corr_sssd_acc_vs_gt_acc = np.asarray([], dtype=np.float64)
    sssd_res = None

    if not bool(args.no_sssd) and n_bins >= 2:
        span = float(edge_hi - edge_lo)
        if span <= 0.0:
            print("[h_binned] SSSD skipped: non-positive theta span.")
        else:
            smin_def, smax_def = sssd.default_sigma_training_range_from_theta(theta_used)
            if args.sssd_sigma_min is not None and args.sssd_sigma_max is not None:
                smin = float(args.sssd_sigma_min)
                smax = float(args.sssd_sigma_max)
            elif args.sssd_sigma_max is not None:
                smax = float(args.sssd_sigma_max)
                smin = smax / 8.0
            elif args.sssd_sigma_min is not None:
                smin = float(args.sssd_sigma_min)
                smax = smin * 8.0
            else:
                smin, smax = smin_def, smax_def
            if not (0.0 < smin < smax):
                raise ValueError(f"SSSD sigma range invalid: sigma_min={smin} sigma_max={smax}")
            sssd_sigma_min_used = smin
            sssd_sigma_max_used = smax
            sigmas_s = str(args.sssd_sigmas).strip().lower()
            if sigmas_s == "auto":
                n_s = max(2, int(args.sssd_n_sigmas))
                sssd_eval_sigmas = np.geomspace(smin, smax, num=n_s, dtype=np.float64)
            else:
                sssd_eval_sigmas = np.asarray(sssd.parse_sigma_list(str(args.sssd_sigmas)), dtype=np.float64)

            sssd_seed = clf_rs if int(args.sssd_seed) < 0 else int(args.sssd_seed)
            try:
                sssd.sanity_check_soft_targets(
                    theta_used[: min(50, theta_used.shape[0])],
                    edges,
                    sigma_small=max(smin * 0.5, 1e-12),
                    sigma_large=smax,
                )
            except Exception as e:
                print(f"[h_binned] SSSD sanity_check_soft_targets warning: {e}")

            print(
                f"[h_binned] SSSD training: sigma_train=[{smin:g},{smax:g}] "
                f"eval_sigmas={np.array2string(sssd_eval_sigmas, precision=4)} "
                f"max_epochs={int(args.sssd_epochs)} patience={int(args.sssd_patience)}"
            )
            sssd_model, sssd_res = sssd.train_sssd_decoder(
                theta_used,
                x_chk,
                edges,
                sigma_min=smin,
                sigma_max=smax,
                device=dev,
                epochs=int(args.sssd_epochs),
                batch_size=int(args.sssd_batch_size),
                lr=float(args.sssd_lr),
                hidden_dim=int(args.sssd_hidden_dim),
                depth=int(args.sssd_depth),
                val_frac=float(args.sssd_val_frac),
                patience=int(args.sssd_patience),
                seed=sssd_seed,
                log_every=int(args.sssd_log_every),
            )
            sssd_train_losses = np.asarray(sssd_res.train_losses, dtype=np.float64)
            sssd_best_epoch = int(sssd_res.best_epoch)
            es = "stopped early" if sssd_res.stopped_early else "ran full"
            print(
                "[h_binned] SSSD training summary: "
                f"AdamW(lr={float(args.sssd_lr):g}), "
                f"{es} at epoch {sssd_res.stopped_epoch}/{int(args.sssd_epochs)}, "
                f"patience={int(args.sssd_patience)} (0=off), "
                f"val_frac={float(args.sssd_val_frac):g}, "
                f"batch_size={int(args.sssd_batch_size)}, "
                f"checkpoint=best val soft-CE at epoch {sssd_best_epoch}. "
                "Each step samples σ log-uniform in [sigma_min,sigma_max] and fits soft bin targets q_σ(b|θ)."
            )

            S = int(sssd_eval_sigmas.shape[0])
            sssd_M_stack = np.full((S, n_bins, n_bins), np.nan, dtype=np.float64)
            sssd_gt_M_stack = np.full((S, n_bins, n_bins), np.nan, dtype=np.float64)
            corr_sssd_M_vs_gt_M = np.full(S, np.nan, dtype=np.float64)
            sssd_acc_stack = np.full((S, n_bins, n_bins), np.nan, dtype=np.float64)
            sssd_gt_acc_stack = np.full((S, n_bins, n_bins), np.nan, dtype=np.float64)
            corr_sssd_acc_vs_gt_acc = np.full(S, np.nan, dtype=np.float64)

            for si, sig in enumerate(sssd_eval_sigmas):
                sigf = float(sig)
                lp_ds = sssd.decoder_log_probs(
                    sssd_model, x_chk, sigf, dev, batch_size=max(512, int(args.sssd_batch_size))
                )
                lp_gt = sssd.decoder_log_probs(
                    sssd_model, x_gt, sigf, dev, batch_size=max(512, int(args.sssd_batch_size))
                )
                M_ds = sssd.symmetric_discrimination_matrix_M(lp_ds, bin_idx, n_bins)
                M_gt = sssd.symmetric_discrimination_matrix_M(lp_gt, gt_bin_idx, n_bins)
                A_ds = sssd.symmetric_lr_accuracy_matrix(lp_ds, bin_idx, n_bins)
                A_gt = sssd.symmetric_lr_accuracy_matrix(lp_gt, gt_bin_idx, n_bins)
                sssd_M_stack[si] = M_ds
                sssd_gt_M_stack[si] = M_gt
                sssd_acc_stack[si] = A_ds
                sssd_gt_acc_stack[si] = A_gt
                corr_sssd_M_vs_gt_M[si] = matrix_corr_offdiag(M_ds, M_gt)
                corr_sssd_acc_vs_gt_acc[si] = matrix_corr_offdiag(A_ds, A_gt)
                asym = float(np.max(np.abs(M_ds - M_ds.T)))
                if asym > 1e-3:
                    print(f"[h_binned] SSSD M_ij note: max|M-M^T|={asym:.4f} at sigma={sigf:g} (MC asymmetry)")
    else:
        if bool(args.no_sssd):
            print("[h_binned] --no-sssd: skipping kernel-smoothed decoder.")
        elif n_bins < 2:
            print("[h_binned] SSSD requires num_theta_bins >= 2; skipped.")
    out_npz = os.path.join(full_args.output_dir, "h_matrix_binned_results.npz")
    np.savez_compressed(
        out_npz,
        h_binned=h_binned,
        h_binned_sqrt=h_binned_sqrt,
        count_matrix=count_matrix,
        clf_accuracy_binned=clf_acc,
        gt_approx_clf_accuracy_binned=gt_acc,
        gt_approx_valid_mask=gt_valid,
        gt_approx_pair_support=gt_support,
        gt_approx_n_total=np.asarray([int(args.gt_approx_n_total)], dtype=np.int64),
        gt_approx_seed=np.asarray([gt_seed], dtype=np.int64),
        corr_h_binned_vs_gt_approx=np.asarray([corr_h_vs_gt], dtype=np.float64),
        corr_clf_binned_vs_gt_approx=np.asarray([corr_clf_vs_gt], dtype=np.float64),
        delta_l_binned=delta_l_binned,
        lr_acc_binned_from_delta_l=lr_acc_binned_from_delta_l,
        count_matrix_delta_l=count_matrix_dl,
        corr_lr_binned_delta_l_vs_gt_approx=np.asarray([corr_lr_dl_vs_gt], dtype=np.float64),
        clf_valid_mask=clf_valid,
        clf_pair_support=clf_support,
        clf_test_frac=np.asarray([float(args.clf_test_frac)], dtype=np.float64),
        clf_min_class_count=np.asarray([int(args.clf_min_class_count)], dtype=np.int64),
        clf_max_iter=np.asarray([int(args.clf_max_iter)], dtype=np.int64),
        clf_random_state=np.asarray([clf_rs], dtype=np.int64),
        theta_bin_edges=edges,
        theta_bin_centers=centers,
        bin_index_per_sample=bin_idx,
        theta_used=theta_used,
        x_aligned=x_chk,
        num_theta_bins=np.asarray([n_bins], dtype=np.int64),
        theta_bin_mode=np.asarray([bin_mode], dtype=object),
        theta_bin_edge_lo=np.asarray([edge_lo], dtype=np.float64),
        theta_bin_edge_hi=np.asarray([edge_hi], dtype=np.float64),
        h_field_method=np.asarray([h_field_method], dtype=object),
        h_eval_scalar_name=np.asarray([h_eval_scalar_name], dtype=object),
        h_eval_scalar_value=np.asarray([h_eval_scalar_value], dtype=np.float64),
        dataset_npz=np.asarray([os.path.abspath(dataset_npz)], dtype=object),
        sssd_eval_sigmas=sssd_eval_sigmas.astype(np.float64),
        sssd_M_sym_stack=sssd_M_stack.astype(np.float64),
        sssd_gt_M_sym_stack=sssd_gt_M_stack.astype(np.float64),
        sssd_train_losses=sssd_train_losses,
        sssd_best_epoch=np.asarray([sssd_best_epoch], dtype=np.int64),
        sssd_sigma_min_used=np.asarray([sssd_sigma_min_used], dtype=np.float64),
        sssd_sigma_max_used=np.asarray([sssd_sigma_max_used], dtype=np.float64),
        corr_sssd_M_vs_gt_M=corr_sssd_M_vs_gt_M.astype(np.float64),
        sssd_acc_sym_stack=sssd_acc_stack.astype(np.float64),
        sssd_gt_acc_sym_stack=sssd_gt_acc_stack.astype(np.float64),
        corr_sssd_acc_vs_gt_acc=corr_sssd_acc_vs_gt_acc.astype(np.float64),
    )

    fig_path = os.path.join(full_args.output_dir, "h_matrix_binned_heatmap.png")
    plt.figure(figsize=(7.0, 6.0))
    im = plt.imshow(h_binned_sqrt, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04, label=r"mean $\sqrt{H^{\mathrm{sym}}}$ in bin pair")
    plt.xlabel(r"bin $j$")
    plt.ylabel(r"bin $i$")
    plt.title(f"Binned sqrt(H)-matrix ({n_bins} bins, mode={bin_mode})")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

    clf_fig_path = os.path.join(full_args.output_dir, "h_matrix_binned_classifier_heatmap.png")
    plt.figure(figsize=(7.0, 6.0))
    imc = plt.imshow(clf_acc, aspect="auto", origin="lower", interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.colorbar(imc, fraction=0.046, pad=0.04, label="held-out accuracy")
    plt.xlabel(r"bin $j$")
    plt.ylabel(r"bin $i$")
    plt.title("Pairwise bin-vs-bin logistic accuracy (x)")
    plt.tight_layout()
    plt.savefig(clf_fig_path, dpi=180)
    plt.close()

    count_fig_path = os.path.join(full_args.output_dir, "h_matrix_binned_count_heatmap.png")
    plt.figure(figsize=(7.0, 6.0))
    # log1p for visibility when counts vary widely
    log_counts = np.log1p(count_matrix.astype(np.float64))
    im2 = plt.imshow(log_counts, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(im2, fraction=0.046, pad=0.04, label=r"$\log(1 + N_{ij})$ pair count")
    plt.xlabel(r"bin $j$")
    plt.ylabel(r"bin $i$")
    plt.title("Log pair counts per bin pair")
    plt.tight_layout()
    plt.savefig(count_fig_path, dpi=180)
    plt.close()

    combo_fig_path = os.path.join(full_args.output_dir, "h_matrix_binned_and_classifier_panels.png")
    s_combo = int(sssd_acc_stack.shape[0]) if sssd_acc_stack.size else 0

    if s_combo > 0:
        nc = max(4, s_combo)
        fig = plt.figure(figsize=(4.0 * nc, 10.5))
        gs = GridSpec(2, nc, figure=fig, height_ratios=[1.0, 1.0], hspace=0.33, wspace=0.35)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[0, 3])
    else:
        fig, axes = plt.subplots(1, 4, figsize=(24.0, 5.8), layout="constrained")
        ax0, ax1, ax2, ax3 = axes[0], axes[1], axes[2], axes[3]

    im0 = ax0.imshow(h_binned_sqrt, aspect="auto", origin="lower", interpolation="nearest")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label=r"mean $\sqrt{H^{\mathrm{sym}}}$")
    ax0.set_xlabel(r"bin $j$")
    ax0.set_ylabel(r"bin $i$")
    ax0.set_title(f"Binned sqrt(H)-matrix\ncorr vs GT={corr_h_vs_gt:.3f}")

    im1 = ax1.imshow(clf_acc, aspect="auto", origin="lower", interpolation="nearest", vmin=0.0, vmax=1.0)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="held-out accuracy")
    ax1.set_xlabel(r"bin $j$")
    ax1.set_ylabel(r"bin $i$")
    ax1.set_title(f"Pairwise logistic accuracy (x)\ncorr vs GT={corr_clf_vs_gt:.3f}")

    im_gt = ax2.imshow(gt_acc, aspect="auto", origin="lower", interpolation="nearest", vmin=0.0, vmax=1.0)
    fig.colorbar(im_gt, ax=ax2, fraction=0.046, pad=0.04, label="held-out accuracy")
    ax2.set_xlabel(r"bin $j$")
    ax2.set_ylabel(r"bin $i$")
    ax2.set_title(f"GT approx logistic accuracy (n={int(args.gt_approx_n_total)})")

    im_lr = ax3.imshow(
        lr_acc_binned_from_delta_l,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(
        im_lr,
        ax=ax3,
        fraction=0.046,
        pad=0.04,
        label=r"sym. $\max(\sigma,1\!-\!\sigma)$ from $\Delta L$",
    )
    ax3.set_xlabel(r"bin $j$")
    ax3.set_ylabel(r"bin $i$")
    ax3.set_title(
        f"{dl_panel_title_top}\n"
        + rf"$0.5(f(\Delta L)+f(\Delta L^\top))$, $f=\max(\sigma,1-\sigma)$, corr vs GT={corr_lr_dl_vs_gt:.3f}"
    )

    if s_combo > 0:
        for si in range(s_combo):
            ax_s = fig.add_subplot(gs[1, si])
            corr_acc_gt = float(corr_sssd_acc_vs_gt_acc[si])
            sig_val = float(sssd_eval_sigmas[si])
            im_s = ax_s.imshow(
                sssd_acc_stack[si],
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                vmin=0.0,
                vmax=1.0,
            )
            fig.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04, label="in-sample LR acc")
            ax_s.set_xlabel(r"bin $j$")
            ax_s.set_ylabel(r"bin $i$")
            ax_s.set_title(
                rf"SSSD LR acc $\sigma$={sig_val:.3g}" + "\n" + f"corr vs GT acc={corr_acc_gt:.3f}"
            )
        fig.suptitle(
            f"Binned matrices + {dl_suptitle_delta_l_tag} acc + SSSD ({n_bins} bins, mode={bin_mode}; "
            f"{s_combo} evaluation $\\sigma$)",
            fontsize=13,
        )
    else:
        fig.suptitle(
            f"Binned matrices + {dl_suptitle_delta_l_tag} acc ({n_bins} bins, mode={bin_mode})",
            fontsize=13,
        )

    plt.savefig(combo_fig_path, dpi=180, bbox_inches="tight")
    plt.close()

    sssd_acc_panels_path = ""
    sssd_acc_primary_path = ""
    sssd_acc_panels_file = os.path.join(full_args.output_dir, "h_matrix_sssd_acc_panels.png")
    sssd_acc_primary_file = os.path.join(full_args.output_dir, "h_matrix_sssd_acc_primary.png")
    if sssd_acc_stack.size > 0 and sssd_acc_stack.shape[0] > 0:
        S = int(sssd_acc_stack.shape[0])
        ncols = min(5, S)
        nrows = int(np.ceil(S / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 3.5 * nrows), squeeze=False)
        last_im = None
        for si in range(S):
            r, c = divmod(si, ncols)
            ax = axes[r][c]
            last_im = ax.imshow(
                sssd_acc_stack[si],
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                vmin=0.0,
                vmax=1.0,
            )
            ax.set_title(rf"$\sigma$={float(sssd_eval_sigmas[si]):.4g}")
            ax.set_xlabel(r"bin $j$")
            ax.set_ylabel(r"bin $i$")
        for si in range(S, nrows * ncols):
            r, c = divmod(si, ncols)
            axes[r][c].axis("off")
        fig.colorbar(
            last_im,
            ax=axes,
            fraction=0.03,
            pad=0.04,
            label="symmetric LR accuracy",
        )
        fig.suptitle("SSSD symmetric pairwise LR accuracy (kernel-smoothed decoder)", fontsize=13)
        plt.savefig(sssd_acc_panels_file, dpi=180, bbox_inches="tight")
        plt.close()

        mid = S // 2
        plt.figure(figsize=(7.0, 6.0))
        im_p = plt.imshow(
            sssd_acc_stack[mid],
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        plt.colorbar(im_p, fraction=0.046, pad=0.04, label="symmetric LR accuracy")
        plt.xlabel(r"bin $j$")
        plt.ylabel(r"bin $i$")
        plt.title(
            rf"SSSD LR accuracy (primary $\sigma$={float(sssd_eval_sigmas[mid]):.4g})"
        )
        plt.tight_layout()
        plt.savefig(sssd_acc_primary_file, dpi=180)
        plt.close()
        sssd_acc_panels_path = sssd_acc_panels_file
        sssd_acc_primary_path = sssd_acc_primary_file

    summary_path = os.path.join(full_args.output_dir, "h_matrix_binned_summary.txt")
    n_finite = int(np.sum(np.isfinite(h_binned)))
    n_nan = int(np.sum(~np.isfinite(h_binned)))
    clf_finite = int(np.sum(np.isfinite(clf_acc) & ~np.eye(n_bins, dtype=bool)))
    clf_nan_off = int(np.sum(~np.isfinite(clf_acc) & ~np.eye(n_bins, dtype=bool)))
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Theta-binned sqrt(H)-matrix + pairwise classifier summary\n")
        f.write(f"dataset_npz: {dataset_npz}\n")
        f.write(f"output_dir: {full_args.output_dir}\n")
        f.write(f"n_samples: {h_sym.shape[0]}\n")
        f.write(f"score_fisher_eval_data: {getattr(full_args, 'score_fisher_eval_data', '')}\n")
        f.write(f"h_field_method: {h_field_method}\n")
        f.write(f"{h_eval_scalar_name}: {h_eval_scalar_value}\n")
        f.write(f"num_theta_bins: {n_bins}\n")
        f.write(f"theta_bin_mode: {bin_mode}\n")
        f.write(f"theta_bin_edges: [{edge_lo}, {edge_hi}] (mode-dependent)\n")
        f.write(f"h_binned finite cells: {n_finite} nan cells: {n_nan}\n")
        if n_finite > 0:
            f.write(
                f"h_binned min (finite): {float(np.nanmin(h_binned))} "
                f"max (finite): {float(np.nanmax(h_binned))}\n"
            )
            f.write(
                f"h_binned_sqrt min (finite): {float(np.nanmin(h_binned_sqrt))} "
                f"max: {float(np.nanmax(h_binned_sqrt))}\n"
            )
        f.write(
            "classifier: sklearn LogisticRegression(lbfgs), pairwise bin i vs j, held-out accuracy\n"
        )
        f.write(
            f"  clf_test_frac={float(args.clf_test_frac)} clf_min_class_count={int(args.clf_min_class_count)} "
            f"clf_max_iter={int(args.clf_max_iter)} clf_random_state={clf_rs}\n"
        )
        f.write(
            f"  off-diagonal finite: {clf_finite} nan: {clf_nan_off} "
            f"ok_pairs={clf_stats['ok_pairs']} insufficient_counts={clf_stats['insufficient_counts']} "
            f"split_fail={clf_stats['split_fail']} fit_fail={clf_stats['fit_fail']}\n"
        )
        off_mask = ~np.eye(n_bins, dtype=bool)
        sub_acc = clf_acc[off_mask]
        if np.any(np.isfinite(sub_acc)):
            f.write(
                f"  clf accuracy min (finite off-diag): {float(np.nanmin(sub_acc))} "
                f"max: {float(np.nanmax(sub_acc))}\n"
            )
        gt_sub = gt_acc[off_mask]
        f.write(
            f"gt_approx: n_total={int(args.gt_approx_n_total)} seed={gt_seed} "
            f"ok_pairs={gt_stats['ok_pairs']} insufficient_counts={gt_stats['insufficient_counts']} "
            f"split_fail={gt_stats['split_fail']} fit_fail={gt_stats['fit_fail']}\n"
        )
        if np.any(np.isfinite(gt_sub)):
            f.write(
                f"  gt_approx accuracy min (finite off-diag): {float(np.nanmin(gt_sub))} "
                f"max: {float(np.nanmax(gt_sub))}\n"
            )
        f.write(f"correlation_h_binned_sqrt_vs_gt_approx: {corr_h_vs_gt}\n")
        f.write(f"correlation_clf_binned_vs_gt_approx: {corr_clf_vs_gt}\n")
        f.write(
            dl_summary_delta_l_prefix
            + "bin-average Delta L_ij over theta-bin pairs, then f(z)=max(sigmoid(z),1-sigmoid(z)), "
            "A_ij = 0.5*(f(Delta L_ij)+f(Delta L_ji)); diagonal NaN; off-diagonal in [0.5,1].\n"
        )
        if str(h_field_method).strip().lower() != "flow":
            f.write(
                "  DSM note: the trained fields are denoising scores; "
                "∇_θ log p(θ|x) is the standard score target, and "
                "∇_θ p(θ|x) = p(θ|x) ∇_θ log p(θ|x) on the support of p.\n"
            )
        f.write(f"correlation_lr_binned_delta_l_vs_gt_approx: {corr_lr_dl_vs_gt}\n")
        if sssd_acc_stack.size > 0 and sssd_acc_stack.shape[0] > 0:
            f.write(
                "SSSD training (fisher/sssd.train_sssd_decoder): AdamW on soft cross-entropy vs "
                "Gaussian bin targets q_sigma(b|theta); each batch samples sigma log-uniformly in "
                f"[sigma_min,sigma_max] for targets and decoder conditioning. "
                f"max_epochs={int(args.sssd_epochs)}, early_stopping_patience={int(args.sssd_patience)} "
                "(0 disables), batch_size={int(args.sssd_batch_size)}, "
                f"lr={float(args.sssd_lr):g}, val_frac={float(args.sssd_val_frac):g}. "
                "Checkpoint = lowest validation soft-CE (best_epoch). "
                + (
                    f"Stopped at epoch {sssd_res.stopped_epoch} "
                    f"({'early' if sssd_res.stopped_early else 'full budget'}).\n"
                    if sssd_res is not None
                    else "\n"
                )
            )
            f.write(
                "SSSD: sigma-conditioned softmax decoder, Gaussian soft bin targets; "
                "primary metrics are symmetric pairwise LR decision accuracies A_ij(sigma) "
                "(see fisher/sssd.py). Log-ratio matrices M_ij retained in NPZ for diagnostics.\n"
            )
            f.write(
                f"  sigma_train=[{sssd_sigma_min_used:g},{sssd_sigma_max_used:g}] "
                f"best_epoch={sssd_best_epoch}\n"
            )
            for si in range(int(sssd_eval_sigmas.shape[0])):
                f.write(
                    f"  sigma={float(sssd_eval_sigmas[si]):.6g} "
                    f"corr(acc_dataset,acc_gt)={float(corr_sssd_acc_vs_gt_acc[si]):g} "
                    f"corr(M_dataset,M_gt)={float(corr_sssd_M_vs_gt_M[si]):g}\n"
                )
        f.write("artifacts:\n")
        f.write(f"  {out_npz}\n")
        f.write(f"  {fig_path}\n")
        f.write(f"  {clf_fig_path}\n")
        f.write(f"  {count_fig_path}\n")
        f.write(f"  {combo_fig_path}\n")
        if sssd_acc_panels_path:
            f.write(f"  {sssd_acc_panels_path}\n")
        if sssd_acc_primary_path:
            f.write(f"  {sssd_acc_primary_path}\n")
        f.write(f"  {summary_path}\n")

    print("[h_binned] Saved:")
    print(f"  - {out_npz}")
    print(f"  - {fig_path}")
    print(f"  - {clf_fig_path}")
    print(f"  - {count_fig_path}")
    print(f"  - {combo_fig_path}")
    if sssd_acc_panels_path:
        print(f"  - {sssd_acc_panels_path}")
    if sssd_acc_primary_path:
        print(f"  - {sssd_acc_primary_path}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()
