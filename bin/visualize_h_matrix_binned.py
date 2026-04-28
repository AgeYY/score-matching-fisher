#!/usr/bin/env python3
"""Theta-binned H-matrix visualization and diagnostics."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from global_setting import DATA_DIR  # import before pyplot so matplotlib rcParams apply

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from fisher import sssd
from fisher.cli_shared_fisher import add_estimation_arguments
from fisher.shared_dataset_io import SharedDatasetBundle, load_shared_dataset_npz
from fisher.shared_fisher_est import (
    build_dataset_from_meta,
    merge_meta_into_args,
    require_device,
    run_shared_fisher_estimation,
    validate_estimation_args,
)

_SKLEARN_LR_MAX_ITER_DEFAULT = int(LogisticRegression().get_params()["max_iter"])

# Number of Gaussian smoothing strengths for the pairwise logistic-accuracy row in the combo figure.
_CLF_SMOOTH_N_STRENGTHS = 5


def _auto_clf_smooth_sigmas(n_bins: int) -> np.ndarray:
    """Five increasing Gaussian sigma values (in bin-index / pixel units) for smoothing clf_acc."""
    n = float(max(int(n_bins), 2))
    # Scale a fixed template so larger bin grids get proportionally wider kernels.
    scales = np.array([0.15, 0.35, 0.6, 0.95, 1.35], dtype=np.float64)
    return scales * (n / 10.0) ** 0.5


def smooth_pairwise_matrix_gaussian(mat: np.ndarray, sigma: float) -> np.ndarray:
    """2D Gaussian smooth a square matrix with NaNs (e.g. diagonal); NaN-safe via weighted average."""
    a = np.asarray(mat, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("smooth_pairwise_matrix_gaussian expects a square 2D matrix.")
    n = a.shape[0]
    valid = np.isfinite(a)
    w = valid.astype(np.float64)
    a0 = np.where(valid, a, 0.0)
    sig = float(max(sigma, 1e-6))
    num = gaussian_filter(a0 * w, sigma=sig, mode="constant", cval=0.0)
    den = gaussian_filter(w, sigma=sig, mode="constant", cval=0.0)
    out = np.full((n, n), np.nan, dtype=np.float64)
    mask = den > 1e-12
    out[mask] = num[mask] / den[mask]
    np.fill_diagonal(out, np.nan)
    off = ~np.eye(n, dtype=bool)
    finite = off & np.isfinite(out)
    if np.any(finite):
        out[finite] = np.clip(out[finite], 0.0, 1.0)
    return out


@dataclass(frozen=True)
class BinnedVizConfig:
    args: argparse.Namespace
    dataset_npz: str
    n_bins: int
    h_only: bool


@dataclass(frozen=True)
class RunContext:
    args: argparse.Namespace
    config: BinnedVizConfig
    bundle: SharedDatasetBundle
    meta: dict[str, Any]
    full_args: SimpleNamespace
    dataset: Any
    rng: np.random.Generator
    device: torch.device


@dataclass(frozen=True)
class LoadedHMatrix:
    h_path: str
    h_sym: np.ndarray
    theta_used: np.ndarray
    h_field_method: str
    h_eval_scalar_name: str
    h_eval_scalar_value: float
    hell_panel_title_top: str
    hell_suptitle_tag: str
    hell_summary_prefix: str
    flow_scheduler: str | None
    flow_score_mode: str | None


@dataclass(frozen=True)
class BinnedMetrics:
    edges: np.ndarray
    edge_lo: float
    edge_hi: float
    centers: np.ndarray
    bin_idx: np.ndarray
    x_aligned: np.ndarray
    h_binned: np.ndarray
    h_binned_sqrt: np.ndarray
    count_matrix: np.ndarray
    hellinger_acc_lb_binned: np.ndarray
    hellinger_acc_ub_binned: np.ndarray
    clf_rs: int
    clf_acc: np.ndarray
    clf_valid: np.ndarray
    clf_support: np.ndarray
    clf_stats: dict[str, int]
    gt_seed: int
    gt_acc: np.ndarray
    gt_valid: np.ndarray
    gt_support: np.ndarray
    gt_stats: dict[str, int]
    corr_h_vs_gt: float
    corr_clf_vs_gt: float
    corr_hellinger_lb_vs_gt: float
    corr_hellinger_ub_vs_gt: float


@dataclass(frozen=True)
class SSSDMetrics:
    eval_sigmas: np.ndarray
    m_stack: np.ndarray
    gt_m_stack: np.ndarray
    train_losses: np.ndarray
    best_epoch: int
    sigma_min_used: float
    sigma_max_used: float
    corr_m_vs_gt_m: np.ndarray
    acc_stack: np.ndarray
    gt_acc_stack: np.ndarray
    corr_acc_vs_gt_acc: np.ndarray
    train_result: Any | None


@dataclass(frozen=True)
class ArtifactPaths:
    out_npz: str
    fig_path: str
    clf_fig_path: str
    count_fig_path: str
    hell_ub_fig_path: str
    combo_fig_path: str
    summary_path: str
    sssd_acc_panels_path: str
    sssd_acc_primary_path: str


@dataclass(frozen=True)
class BinnedVizResult:
    context: RunContext
    h_matrix: LoadedHMatrix
    binned: BinnedMetrics
    sssd: SSSDMetrics
    artifacts: ArtifactPaths


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Load a shared dataset .npz, run/load H-matrix estimation, "
            "bin theta, aggregate H_sym over bin pairs, and write diagnostic artifacts."
        )
    )
    p.add_argument(
        "--dataset-npz",
        type=str,
        required=True,
        help="Path to shared dataset .npz from make_dataset.py.",
    )
    p.add_argument(
        "--num-theta-bins",
        type=int,
        default=10,
        help="Number of equal-width theta bins (default 10).",
    )
    p.add_argument(
        "--h-only",
        action="store_true",
        default=False,
        help="Skip model training and only load existing h_matrix_results*.npz.",
    )
    p.add_argument(
        "--h-matrix-npz",
        type=str,
        default=None,
        help=(
            "Path to existing h_matrix_results*.npz. "
            "When omitted, uses output-dir/h_matrix_results{suffix}.npz."
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
        help="Minimum samples per bin class required to train pairwise classifiers (default 5).",
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
    sssd.add_sssd_cli_arguments(p)
    add_estimation_arguments(p)
    p.set_defaults(output_dir=str(Path(DATA_DIR) / "outputs_h_matrix_binned"))
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def config_from_args(args: argparse.Namespace) -> BinnedVizConfig:
    dataset_npz = str(args.dataset_npz)
    n_bins = int(args.num_theta_bins)
    return BinnedVizConfig(args=args, dataset_npz=dataset_npz, n_bins=n_bins, h_only=bool(args.h_only))


def theta_bin_edges(
    theta_used: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, float, float]:
    th = np.asarray(theta_used, dtype=np.float64).reshape(-1)
    if n_bins < 1:
        raise ValueError("--num-theta-bins must be >= 1.")
    lo = float(np.min(th))
    hi = float(np.max(th))
    if hi <= lo:
        raise ValueError(f"Invalid theta range for binning: [{lo}, {hi}]")
    edges = np.linspace(lo, hi, n_bins + 1, dtype=np.float64)
    return edges, lo, hi


def theta_to_bin_index(theta: np.ndarray, edges: np.ndarray, n_bins: int) -> np.ndarray:
    th = np.asarray(theta, dtype=np.float64).reshape(-1)
    idx = np.searchsorted(edges, th, side="right") - 1
    return np.clip(idx, 0, n_bins - 1).astype(np.int64)


def theta_segment_ids_equal_width(theta: np.ndarray, n_segments: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-width theta segmentation helper used by segmented theta-flow mode."""
    edges, _, _ = theta_bin_edges(theta, int(n_segments))
    seg_ids = theta_to_bin_index(theta, edges, int(n_segments))
    return seg_ids, edges


def average_matrix_by_bins(
    mat: np.ndarray,
    bin_idx: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
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


def hellinger_acc_lb_from_binned_h_squared(h_squared_binned: np.ndarray) -> np.ndarray:
    h2 = np.asarray(h_squared_binned, dtype=np.float64)
    if h2.ndim != 2 or h2.shape[0] != h2.shape[1]:
        raise ValueError("h_squared_binned must be square.")
    acc = 0.5 * (1.0 + np.clip(h2, 0.0, 1.0))
    np.fill_diagonal(acc, np.nan)
    return acc


def hellinger_acc_ub_from_binned_h_squared(h_squared_binned: np.ndarray) -> np.ndarray:
    h2 = np.asarray(h_squared_binned, dtype=np.float64)
    if h2.ndim != 2 or h2.shape[0] != h2.shape[1]:
        raise ValueError("h_squared_binned must be square.")
    h2c = np.clip(h2, 0.0, 1.0)
    rad = np.clip(2.0 * h2c - h2c * h2c, 0.0, None)
    acc = 0.5 * (1.0 + np.sqrt(rad))
    np.fill_diagonal(acc, np.nan)
    return acc


def hellinger_figure_labels(h_field_method: str) -> tuple[str, str, str]:
    m = str(h_field_method).strip().lower()
    if m == "flow":
        return (
            r"Binned $H_{ij}^2$ (flow-derived score) → Hellinger LB on $A^*_{ij}$",
            r"Hellinger LB ($H^2$)",
            "Flow-derived score field (from velocity): binned symmetric H treated as H^2; ",
        )
    return (
        r"Binned $H_{ij}^2$ → Hellinger LB on $A^*_{ij}$"
        + "\n"
        + r"(denoising score matching: $\nabla_\theta \log p(\theta \mid x)$)",
        r"Hellinger LB ($H^2$)",
        "DSM score-field: binned symmetric H treated as H^2; ",
    )


def theta_for_h_matrix_alignment(bundle: SharedDatasetBundle, full_args: SimpleNamespace) -> np.ndarray:
    mode = str(getattr(full_args, "score_fisher_eval_data", "full"))
    if mode == "full":
        th = np.asarray(bundle.theta_all, dtype=np.float64)
    elif mode == "score_eval":
        th = np.asarray(bundle.theta_eval, dtype=np.float64)
    else:
        raise ValueError(f"Unknown score_fisher_eval_data: {mode}")

    tfm = str(getattr(full_args, "theta_field_method", "")).strip().lower()
    if tfm == "theta_flow_discrete_scaffold":
        from fisher.theta_gaussian_scaffold import ThetaDiscreteScaffold

        out_dir = str(getattr(full_args, "output_dir", "") or "").strip()
        if not out_dir:
            raise ValueError(
                "theta_flow_discrete_scaffold requires full_args.output_dir for theta_discrete_scaffold.npz"
            )
        scaffold_path = os.path.join(out_dir, "theta_discrete_scaffold.npz")
        if not os.path.isfile(scaffold_path):
            raise ValueError(f"Missing discrete scaffold artifact: {scaffold_path}")
        sc = ThetaDiscreteScaffold.from_npz(scaffold_path)
        out = sc.discretize_theta_np(th)
        return np.asarray(out, dtype=np.float64).reshape(-1)

    return np.asarray(th, dtype=np.float64).reshape(-1)


def x_for_h_matrix_alignment(bundle: SharedDatasetBundle, full_args: SimpleNamespace) -> np.ndarray:
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
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
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
    stats = {"insufficient_counts": 0, "split_fail": 0, "fit_fail": 0, "ok_pairs": 0}

    rs = int(random_state)
    for i in range(n_bins):
        for j in range(i + 1, n_bins):
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
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=float(test_frac), random_state=rs)
                except ValueError:
                    stats["split_fail"] += 1
                    continue

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                stats["split_fail"] += 1
                continue

            try:
                clf = LogisticRegression(solver="lbfgs", random_state=rs)
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

    np.fill_diagonal(acc, np.nan)
    return acc, valid, support, stats


def pairwise_bin_logistic_accuracy_train_val(
    x_train: np.ndarray,
    bin_train: np.ndarray,
    x_all: np.ndarray,
    bin_all: np.ndarray,
    n_bins: int,
    *,
    min_class_count: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Train pairwise bin classifiers on the train split, evaluate accuracy on the full pool."""
    x_tr = np.asarray(x_train, dtype=np.float64)
    x_ev = np.asarray(x_all, dtype=np.float64)
    bi_tr = np.asarray(bin_train, dtype=np.int64).reshape(-1)
    bi_ev = np.asarray(bin_all, dtype=np.int64).reshape(-1)
    if x_tr.ndim != 2 or x_ev.ndim != 2 or x_tr.shape[1] != x_ev.shape[1]:
        raise ValueError("x_train and x_all must be 2D with matching feature dim.")
    if x_tr.shape[0] != bi_tr.shape[0] or x_ev.shape[0] != bi_ev.shape[0]:
        raise ValueError("x_* and bin_* must have the same number of rows.")
    if min_class_count < 1:
        raise ValueError("min_class_count must be >= 1.")

    acc = np.full((n_bins, n_bins), np.nan, dtype=np.float64)
    valid = np.zeros((n_bins, n_bins), dtype=bool)
    support = np.zeros((n_bins, n_bins), dtype=np.int64)
    stats = {
        "insufficient_counts": 0,
        "fit_fail": 0,
        "eval_fail": 0,
        "ok_pairs": 0,
    }
    rs = int(random_state)

    for i in range(n_bins):
        for j in range(i + 1, n_bins):
            ia_tr = np.flatnonzero(bi_tr == i)
            jb_tr = np.flatnonzero(bi_tr == j)
            ni_tr, nj_tr = int(ia_tr.size), int(jb_tr.size)
            ia_ev = np.flatnonzero(bi_ev == i)
            jb_ev = np.flatnonzero(bi_ev == j)
            ni_ev, nj_ev = int(ia_ev.size), int(jb_ev.size)
            support[i, j] = ni_ev + nj_ev
            support[j, i] = ni_ev + nj_ev

            if ni_tr < min_class_count or nj_tr < min_class_count:
                stats["insufficient_counts"] += 1
                continue
            if ni_ev < 1 or nj_ev < 1:
                stats["eval_fail"] += 1
                continue

            X_tr = np.vstack([x_tr[ia_tr], x_tr[jb_tr]])
            y_tr = np.concatenate([np.zeros(ni_tr, dtype=np.int64), np.ones(nj_tr, dtype=np.int64)])
            X_ev = np.vstack([x_ev[ia_ev], x_ev[jb_ev]])
            y_ev = np.concatenate([np.zeros(ni_ev, dtype=np.int64), np.ones(nj_ev, dtype=np.int64)])

            if len(np.unique(y_tr)) < 2:
                stats["fit_fail"] += 1
                continue

            try:
                clf = LogisticRegression(solver="lbfgs", random_state=rs)
                clf.fit(X_tr, y_tr)
                if len(np.unique(y_ev)) < 2:
                    stats["eval_fail"] += 1
                    continue
                score = float(clf.score(X_ev, y_ev))
            except Exception:
                stats["fit_fail"] += 1
                continue

            acc[i, j] = score
            acc[j, i] = score
            valid[i, j] = True
            valid[j, i] = True
            stats["ok_pairs"] += 1

    np.fill_diagonal(acc, np.nan)
    return acc, valid, support, stats


def matrix_corr_offdiag(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between vectorized off-diagonal entries (finite pairs only)."""
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
    res = spearmanr(av, bv)
    stat = getattr(res, "statistic", None)
    if stat is None:
        stat = res[0]
    return float(stat)


def matrix_corr_offdiag_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between vectorized off-diagonal entries (finite pairs only)."""
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape or aa.ndim != 2 or aa.shape[0] != aa.shape[1]:
        raise ValueError("matrix_corr_offdiag_pearson requires equal-shape square matrices.")
    n = aa.shape[0]
    off = ~np.eye(n, dtype=bool)
    mask = off & np.isfinite(aa) & np.isfinite(bb)
    if int(np.sum(mask)) < 3:
        return float("nan")
    av = aa[mask]
    bv = bb[mask]
    if float(np.std(av)) <= 0.0 or float(np.std(bv)) <= 0.0:
        return float("nan")
    r = np.corrcoef(av, bv)[0, 1]
    return float(r)


def _validate_args(args: argparse.Namespace, n_bins: int) -> None:
    validate_estimation_args(args)
    if n_bins < 1:
        raise ValueError("--num-theta-bins must be >= 1.")
    if not (0.0 < float(args.clf_test_frac) < 1.0):
        raise ValueError("--clf-test-frac must be in (0, 1).")
    if int(args.clf_min_class_count) < 1:
        raise ValueError("--clf-min-class-count must be >= 1.")
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


def prepare_context(config: BinnedVizConfig) -> RunContext:
    args = config.args
    _validate_args(args, config.n_bins)

    bundle = load_shared_dataset_npz(config.dataset_npz)
    meta = bundle.meta
    full_args = merge_meta_into_args(meta, args)
    setattr(full_args, "compute_h_matrix", True)
    setattr(full_args, "h_restore_original_order", True)
    setattr(full_args, "skip_shared_fisher_gt_compare", True)
    validate_estimation_args(full_args)

    np.random.seed(int(meta["seed"]))
    torch.manual_seed(int(meta["seed"]))
    rng = np.random.default_rng(int(meta["seed"]))

    dev = require_device(str(full_args.device))
    dataset = build_dataset_from_meta(meta)
    os.makedirs(full_args.output_dir, exist_ok=True)

    return RunContext(
        args=args,
        config=config,
        bundle=bundle,
        meta=meta,
        full_args=full_args,
        dataset=dataset,
        rng=rng,
        device=dev,
    )


def run_h_estimation_if_needed(ctx: RunContext) -> None:
    if ctx.config.h_only:
        print("[h_binned] --h-only: skipping Fisher training; using existing h_matrix_results*.npz.")
        return
    run_shared_fisher_estimation(
        ctx.full_args,
        ctx.dataset,
        theta_all=ctx.bundle.theta_all,
        x_all=ctx.bundle.x_all,
        theta_train=ctx.bundle.theta_train,
        x_train=ctx.bundle.x_train,
        theta_validation=ctx.bundle.theta_validation,
        x_validation=ctx.bundle.x_validation,
        rng=ctx.rng,
    )


def load_h_matrix(ctx: RunContext) -> LoadedHMatrix:
    args = ctx.args
    suffix = "_non_gauss" if ctx.full_args.dataset_family == "cosine_gmm" else "_theta_cov"
    if args.h_matrix_npz:
        h_path = os.path.abspath(args.h_matrix_npz)
    else:
        h_path = os.path.join(ctx.full_args.output_dir, f"h_matrix_results{suffix}.npz")
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
    flow_scheduler = str(h_npz["h_flow_scheduler"][0]) if "h_flow_scheduler" in h_npz.files else None
    flow_score_mode = str(h_npz["h_flow_score_mode"][0]) if "h_flow_score_mode" in h_npz.files else None
    hell_panel_title_top, hell_suptitle_tag, hell_summary_prefix = hellinger_figure_labels(h_field_method)
    return LoadedHMatrix(
        h_path=h_path,
        h_sym=h_sym,
        theta_used=theta_used,
        h_field_method=h_field_method,
        h_eval_scalar_name=h_eval_scalar_name,
        h_eval_scalar_value=h_eval_scalar_value,
        hell_panel_title_top=hell_panel_title_top,
        hell_suptitle_tag=hell_suptitle_tag,
        hell_summary_prefix=hell_summary_prefix,
        flow_scheduler=flow_scheduler,
        flow_score_mode=flow_score_mode,
    )


def _validate_alignment(ctx: RunContext, loaded: LoadedHMatrix) -> np.ndarray:
    theta_chk = theta_for_h_matrix_alignment(ctx.bundle, ctx.full_args)
    if theta_chk.shape[0] != loaded.theta_used.shape[0]:
        raise ValueError(
            f"theta/H row mismatch: theta_chk={theta_chk.shape[0]} theta_used={loaded.theta_used.shape[0]}"
        )
    if not np.allclose(theta_chk, loaded.theta_used, rtol=0.0, atol=1e-5):
        raise ValueError(
            "theta_used from H-matrix npz does not match dataset theta for score_fisher_eval_data split."
        )
    x_chk = x_for_h_matrix_alignment(ctx.bundle, ctx.full_args)
    if x_chk.shape[0] != loaded.theta_used.shape[0]:
        raise ValueError(f"x/H row mismatch: x_aligned={x_chk.shape[0]} theta_used={loaded.theta_used.shape[0]}")
    return x_chk


def compute_binned_metrics(ctx: RunContext, loaded: LoadedHMatrix) -> BinnedMetrics:
    n_bins = ctx.config.n_bins
    args = ctx.args
    x_chk = _validate_alignment(ctx, loaded)

    edges, edge_lo, edge_hi = theta_bin_edges(loaded.theta_used, n_bins)
    bin_idx = theta_to_bin_index(loaded.theta_used, edges, n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    h_binned, count_matrix = average_matrix_by_bins(loaded.h_sym, bin_idx, n_bins)
    h_binned_sqrt = np.sqrt(np.clip(h_binned, 0.0, None))
    hellinger_acc_lb_binned = hellinger_acc_lb_from_binned_h_squared(h_binned)
    hellinger_acc_ub_binned = hellinger_acc_ub_from_binned_h_squared(h_binned)

    clf_rs = int(ctx.meta["seed"]) if int(args.clf_random_state) < 0 else int(args.clf_random_state)
    clf_acc, clf_valid, clf_support, clf_stats = pairwise_bin_logistic_accuracy_matrix(
        x_chk,
        bin_idx,
        n_bins,
        test_frac=float(args.clf_test_frac),
        min_class_count=int(args.clf_min_class_count),
        random_state=clf_rs,
    )

    gt_seed = int(ctx.meta["seed"]) + 17
    theta_gt, x_gt = ctx.dataset.sample_joint(int(args.gt_approx_n_total))
    theta_gt = np.asarray(theta_gt, dtype=np.float64).reshape(-1)
    x_gt = np.asarray(x_gt, dtype=np.float64)
    gt_bin_idx = theta_to_bin_index(theta_gt, edges, n_bins)
    gt_acc, gt_valid, gt_support, gt_stats = pairwise_bin_logistic_accuracy_matrix(
        x_gt,
        gt_bin_idx,
        n_bins,
        test_frac=float(args.clf_test_frac),
        min_class_count=int(args.clf_min_class_count),
        random_state=gt_seed,
    )

    corr_h_vs_gt = matrix_corr_offdiag(h_binned_sqrt, gt_acc)
    corr_clf_vs_gt = matrix_corr_offdiag(clf_acc, gt_acc)
    corr_hellinger_lb_vs_gt = matrix_corr_offdiag(hellinger_acc_lb_binned, gt_acc)
    corr_hellinger_ub_vs_gt = matrix_corr_offdiag(hellinger_acc_ub_binned, gt_acc)

    return BinnedMetrics(
        edges=edges,
        edge_lo=edge_lo,
        edge_hi=edge_hi,
        centers=centers,
        bin_idx=bin_idx,
        x_aligned=x_chk,
        h_binned=h_binned,
        h_binned_sqrt=h_binned_sqrt,
        count_matrix=count_matrix,
        hellinger_acc_lb_binned=hellinger_acc_lb_binned,
        hellinger_acc_ub_binned=hellinger_acc_ub_binned,
        clf_rs=clf_rs,
        clf_acc=clf_acc,
        clf_valid=clf_valid,
        clf_support=clf_support,
        clf_stats=clf_stats,
        gt_seed=gt_seed,
        gt_acc=gt_acc,
        gt_valid=gt_valid,
        gt_support=gt_support,
        gt_stats=gt_stats,
        corr_h_vs_gt=corr_h_vs_gt,
        corr_clf_vs_gt=corr_clf_vs_gt,
        corr_hellinger_lb_vs_gt=corr_hellinger_lb_vs_gt,
        corr_hellinger_ub_vs_gt=corr_hellinger_ub_vs_gt,
    )


def _empty_sssd(n_bins: int) -> SSSDMetrics:
    return SSSDMetrics(
        eval_sigmas=np.asarray([], dtype=np.float64),
        m_stack=np.zeros((0, n_bins, n_bins), dtype=np.float64),
        gt_m_stack=np.zeros((0, n_bins, n_bins), dtype=np.float64),
        train_losses=np.asarray([], dtype=np.float64),
        best_epoch=-1,
        sigma_min_used=float("nan"),
        sigma_max_used=float("nan"),
        corr_m_vs_gt_m=np.asarray([], dtype=np.float64),
        acc_stack=np.zeros((0, n_bins, n_bins), dtype=np.float64),
        gt_acc_stack=np.zeros((0, n_bins, n_bins), dtype=np.float64),
        corr_acc_vs_gt_acc=np.asarray([], dtype=np.float64),
        train_result=None,
    )


def run_sssd_analysis(ctx: RunContext, loaded: LoadedHMatrix, metrics: BinnedMetrics) -> SSSDMetrics:
    n_bins = ctx.config.n_bins
    args = ctx.args
    if bool(args.no_sssd):
        print("[h_binned] --no-sssd: skipping kernel-smoothed decoder.")
        return _empty_sssd(n_bins)
    if n_bins < 2:
        print("[h_binned] SSSD requires num_theta_bins >= 2; skipped.")
        return _empty_sssd(n_bins)

    span = float(metrics.edge_hi - metrics.edge_lo)
    if span <= 0.0:
        print("[h_binned] SSSD skipped: non-positive theta span.")
        return _empty_sssd(n_bins)

    smin_def, smax_def = sssd.default_sigma_training_range_from_theta(loaded.theta_used)
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

    sigmas_s = str(args.sssd_sigmas).strip().lower()
    if sigmas_s == "auto":
        n_s = max(2, int(args.sssd_n_sigmas))
        eval_sigmas = np.geomspace(smin, smax, num=n_s, dtype=np.float64)
    else:
        eval_sigmas = np.asarray(sssd.parse_sigma_list(str(args.sssd_sigmas)), dtype=np.float64)

    sssd_seed = metrics.clf_rs if int(args.sssd_seed) < 0 else int(args.sssd_seed)
    try:
        sssd.sanity_check_soft_targets(
            loaded.theta_used[: min(50, loaded.theta_used.shape[0])],
            metrics.edges,
            sigma_small=max(smin * 0.5, 1e-12),
            sigma_large=smax,
        )
    except Exception as e:
        print(f"[h_binned] SSSD sanity_check_soft_targets warning: {e}")

    print(
        f"[h_binned] SSSD training: sigma_train=[{smin:g},{smax:g}] "
        f"eval_sigmas={np.array2string(eval_sigmas, precision=4)} "
        f"max_epochs={int(args.sssd_epochs)} patience={int(args.sssd_patience)}"
    )
    model, train_res = sssd.train_sssd_decoder(
        loaded.theta_used,
        metrics.x_aligned,
        metrics.edges,
        sigma_min=smin,
        sigma_max=smax,
        device=ctx.device,
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

    train_losses = np.asarray(train_res.train_losses, dtype=np.float64)
    best_epoch = int(train_res.best_epoch)
    es = "stopped early" if train_res.stopped_early else "ran full"
    print(
        "[h_binned] SSSD training summary: "
        f"AdamW(lr={float(args.sssd_lr):g}), "
        f"{es} at epoch {train_res.stopped_epoch}/{int(args.sssd_epochs)}, "
        f"patience={int(args.sssd_patience)} (0=off), "
        f"val_frac={float(args.sssd_val_frac):g}, "
        f"batch_size={int(args.sssd_batch_size)}, "
        f"checkpoint=best val soft-CE at epoch {best_epoch}. "
        "Each step samples σ log-uniform in [sigma_min,sigma_max] and fits soft bin targets q_σ(b|θ)."
    )

    theta_gt, x_gt = ctx.dataset.sample_joint(int(args.gt_approx_n_total))
    theta_gt = np.asarray(theta_gt, dtype=np.float64).reshape(-1)
    x_gt = np.asarray(x_gt, dtype=np.float64)
    gt_bin_idx = theta_to_bin_index(theta_gt, metrics.edges, n_bins)

    s = int(eval_sigmas.shape[0])
    m_stack = np.full((s, n_bins, n_bins), np.nan, dtype=np.float64)
    gt_m_stack = np.full((s, n_bins, n_bins), np.nan, dtype=np.float64)
    corr_m_vs_gt_m = np.full(s, np.nan, dtype=np.float64)
    acc_stack = np.full((s, n_bins, n_bins), np.nan, dtype=np.float64)
    gt_acc_stack = np.full((s, n_bins, n_bins), np.nan, dtype=np.float64)
    corr_acc_vs_gt_acc = np.full(s, np.nan, dtype=np.float64)

    for si, sig in enumerate(eval_sigmas):
        sigf = float(sig)
        lp_ds = sssd.decoder_log_probs(model, metrics.x_aligned, sigf, ctx.device, batch_size=max(512, int(args.sssd_batch_size)))
        lp_gt = sssd.decoder_log_probs(model, x_gt, sigf, ctx.device, batch_size=max(512, int(args.sssd_batch_size)))
        m_ds = sssd.symmetric_discrimination_matrix_M(lp_ds, metrics.bin_idx, n_bins)
        m_gt = sssd.symmetric_discrimination_matrix_M(lp_gt, gt_bin_idx, n_bins)
        a_ds = sssd.symmetric_lr_accuracy_matrix(lp_ds, metrics.bin_idx, n_bins)
        a_gt = sssd.symmetric_lr_accuracy_matrix(lp_gt, gt_bin_idx, n_bins)

        m_stack[si] = m_ds
        gt_m_stack[si] = m_gt
        acc_stack[si] = a_ds
        gt_acc_stack[si] = a_gt
        corr_m_vs_gt_m[si] = matrix_corr_offdiag(m_ds, m_gt)
        corr_acc_vs_gt_acc[si] = matrix_corr_offdiag(a_ds, a_gt)

        asym = float(np.max(np.abs(m_ds - m_ds.T)))
        if asym > 1e-3:
            print(f"[h_binned] SSSD M_ij note: max|M-M^T|={asym:.4f} at sigma={sigf:g} (MC asymmetry)")

    return SSSDMetrics(
        eval_sigmas=eval_sigmas,
        m_stack=m_stack,
        gt_m_stack=gt_m_stack,
        train_losses=train_losses,
        best_epoch=best_epoch,
        sigma_min_used=smin,
        sigma_max_used=smax,
        corr_m_vs_gt_m=corr_m_vs_gt_m,
        acc_stack=acc_stack,
        gt_acc_stack=gt_acc_stack,
        corr_acc_vs_gt_acc=corr_acc_vs_gt_acc,
        train_result=train_res,
    )


def write_results_npz(ctx: RunContext, loaded: LoadedHMatrix, metrics: BinnedMetrics, sssd_metrics: SSSDMetrics) -> str:
    out_npz = os.path.join(ctx.full_args.output_dir, "h_matrix_binned_results.npz")
    np.savez_compressed(
        out_npz,
        h_binned=metrics.h_binned,
        h_binned_sqrt=metrics.h_binned_sqrt,
        count_matrix=metrics.count_matrix,
        clf_accuracy_binned=metrics.clf_acc,
        gt_approx_clf_accuracy_binned=metrics.gt_acc,
        gt_approx_valid_mask=metrics.gt_valid,
        gt_approx_pair_support=metrics.gt_support,
        gt_approx_n_total=np.asarray([int(ctx.args.gt_approx_n_total)], dtype=np.int64),
        gt_approx_seed=np.asarray([metrics.gt_seed], dtype=np.int64),
        corr_h_binned_vs_gt_approx=np.asarray([metrics.corr_h_vs_gt], dtype=np.float64),
        corr_clf_binned_vs_gt_approx=np.asarray([metrics.corr_clf_vs_gt], dtype=np.float64),
        hellinger_acc_lb_binned=metrics.hellinger_acc_lb_binned,
        corr_hellinger_acc_lb_vs_gt_approx=np.asarray([metrics.corr_hellinger_lb_vs_gt], dtype=np.float64),
        hellinger_acc_ub_binned=metrics.hellinger_acc_ub_binned,
        corr_hellinger_acc_ub_vs_gt_approx=np.asarray([metrics.corr_hellinger_ub_vs_gt], dtype=np.float64),
        clf_valid_mask=metrics.clf_valid,
        clf_pair_support=metrics.clf_support,
        clf_test_frac=np.asarray([float(ctx.args.clf_test_frac)], dtype=np.float64),
        clf_min_class_count=np.asarray([int(ctx.args.clf_min_class_count)], dtype=np.int64),
        clf_max_iter=np.asarray([_SKLEARN_LR_MAX_ITER_DEFAULT], dtype=np.int64),
        clf_random_state=np.asarray([metrics.clf_rs], dtype=np.int64),
        theta_bin_edges=metrics.edges,
        theta_bin_centers=metrics.centers,
        bin_index_per_sample=metrics.bin_idx,
        theta_used=loaded.theta_used,
        x_aligned=metrics.x_aligned,
        num_theta_bins=np.asarray([ctx.config.n_bins], dtype=np.int64),
        theta_bin_edge_lo=np.asarray([metrics.edge_lo], dtype=np.float64),
        theta_bin_edge_hi=np.asarray([metrics.edge_hi], dtype=np.float64),
        h_field_method=np.asarray([loaded.h_field_method], dtype=object),
        h_eval_scalar_name=np.asarray([loaded.h_eval_scalar_name], dtype=object),
        h_eval_scalar_value=np.asarray([loaded.h_eval_scalar_value], dtype=np.float64),
        dataset_npz=np.asarray([os.path.abspath(ctx.config.dataset_npz)], dtype=object),
        sssd_eval_sigmas=sssd_metrics.eval_sigmas.astype(np.float64),
        sssd_M_sym_stack=sssd_metrics.m_stack.astype(np.float64),
        sssd_gt_M_sym_stack=sssd_metrics.gt_m_stack.astype(np.float64),
        sssd_train_losses=sssd_metrics.train_losses,
        sssd_best_epoch=np.asarray([sssd_metrics.best_epoch], dtype=np.int64),
        sssd_sigma_min_used=np.asarray([sssd_metrics.sigma_min_used], dtype=np.float64),
        sssd_sigma_max_used=np.asarray([sssd_metrics.sigma_max_used], dtype=np.float64),
        corr_sssd_M_vs_gt_M=sssd_metrics.corr_m_vs_gt_m.astype(np.float64),
        sssd_acc_sym_stack=sssd_metrics.acc_stack.astype(np.float64),
        sssd_gt_acc_sym_stack=sssd_metrics.gt_acc_stack.astype(np.float64),
        corr_sssd_acc_vs_gt_acc=sssd_metrics.corr_acc_vs_gt_acc.astype(np.float64),
    )
    return out_npz


def _render_single_heatmap(
    matrix: np.ndarray,
    path: str,
    title: str,
    colorbar_label: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    plt.figure(figsize=(7.0, 6.0))
    im = plt.imshow(matrix, aspect="auto", origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04, label=colorbar_label)
    plt.xlabel(r"bin $j$")
    plt.ylabel(r"bin $i$")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def render_figures(ctx: RunContext, loaded: LoadedHMatrix, metrics: BinnedMetrics, sssd_metrics: SSSDMetrics) -> dict[str, str]:
    fig_path = os.path.join(ctx.full_args.output_dir, "h_matrix_binned_heatmap.png")
    _render_single_heatmap(
        metrics.h_binned_sqrt,
        fig_path,
        f"Binned sqrt(H)-matrix ({ctx.config.n_bins} bins)",
        r"mean $\sqrt{H^{\mathrm{sym}}}$ in bin pair",
    )

    clf_fig_path = os.path.join(ctx.full_args.output_dir, "h_matrix_binned_classifier_heatmap.png")
    _render_single_heatmap(
        metrics.clf_acc,
        clf_fig_path,
        "Pairwise bin-vs-bin logistic accuracy (x)",
        "held-out accuracy",
        vmin=0.0,
        vmax=1.0,
    )

    count_fig_path = os.path.join(ctx.full_args.output_dir, "h_matrix_binned_count_heatmap.png")
    _render_single_heatmap(
        np.log1p(metrics.count_matrix.astype(np.float64)),
        count_fig_path,
        "Log pair counts per bin pair",
        r"$\log(1 + N_{ij})$ pair count",
    )

    hell_ub_fig_path = os.path.join(ctx.full_args.output_dir, "h_matrix_binned_hellinger_acc_ub_heatmap.png")
    _render_single_heatmap(
        metrics.hellinger_acc_ub_binned,
        hell_ub_fig_path,
        f"Hellinger upper bound on $A^*_{{ij}}$ ({ctx.config.n_bins} bins); corr vs GT={metrics.corr_hellinger_ub_vs_gt:.3f}",
        r"$\frac{1}{2}(1+\sqrt{2H_{ij}^2-H_{ij}^4})$ upper bound on $A^*_{ij}$",
        vmin=0.0,
        vmax=1.0,
    )

    combo_fig_path = os.path.join(ctx.full_args.output_dir, "h_matrix_binned_and_classifier_panels.png")
    s_combo = int(sssd_metrics.acc_stack.shape[0]) if sssd_metrics.acc_stack.size else 0
    if s_combo > 0:
        nc = max(5, s_combo)
        fig = plt.figure(figsize=(4.0 * nc, 15.5))
        gs = GridSpec(3, nc, figure=fig, height_ratios=[1.0, 1.0, 1.0], hspace=0.33, wspace=0.35)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[0, 3])
        ax4 = fig.add_subplot(gs[0, 4])
    else:
        fig, axes = plt.subplots(1, 5, figsize=(30.0, 5.8), layout="constrained")
        ax0, ax1, ax2, ax3, ax4 = axes[0], axes[1], axes[2], axes[3], axes[4]

    im0 = ax0.imshow(metrics.h_binned_sqrt, aspect="auto", origin="lower", interpolation="nearest")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label=r"mean $\sqrt{H^{\mathrm{sym}}}$")
    ax0.set_xlabel(r"bin $j$")
    ax0.set_ylabel(r"bin $i$")
    ax0.set_title(f"Binned sqrt(H)-matrix\ncorr vs GT={metrics.corr_h_vs_gt:.3f}")

    im_hell_lb = ax1.imshow(
        metrics.hellinger_acc_lb_binned,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(
        im_hell_lb,
        ax=ax1,
        fraction=0.046,
        pad=0.04,
        label=r"$\frac{1}{2}(1+H_{ij}^2)$ lower bound on $A^*_{ij}$",
    )
    ax1.set_xlabel(r"bin $j$")
    ax1.set_ylabel(r"bin $i$")
    ax1.set_title(
        f"{loaded.hell_panel_title_top} (lower)\n"
        + rf"$A^{{lb}}_{{ij}}=\frac{{1}}{{2}}(1+H_{{ij}}^2)$, corr vs GT={metrics.corr_hellinger_lb_vs_gt:.3f}"
    )

    im_hell_ub = ax2.imshow(
        metrics.hellinger_acc_ub_binned,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(
        im_hell_ub,
        ax=ax2,
        fraction=0.046,
        pad=0.04,
        label=r"$\frac{1}{2}(1+\sqrt{2H_{ij}^2-H_{ij}^4})$ upper bound on $A^*_{ij}$",
    )
    ax2.set_xlabel(r"bin $j$")
    ax2.set_ylabel(r"bin $i$")
    ax2.set_title(
        f"{loaded.hell_panel_title_top} (upper)\n"
        + rf"$A^{{ub}}_{{ij}}=\frac{{1}}{{2}}(1+\sqrt{{2H_{{ij}}^2-H_{{ij}}^4}})$, corr vs GT={metrics.corr_hellinger_ub_vs_gt:.3f}"
    )

    im_clf = ax3.imshow(metrics.clf_acc, aspect="auto", origin="lower", interpolation="nearest", vmin=0.0, vmax=1.0)
    fig.colorbar(im_clf, ax=ax3, fraction=0.046, pad=0.04, label="held-out accuracy")
    ax3.set_xlabel(r"bin $j$")
    ax3.set_ylabel(r"bin $i$")
    ax3.set_title(f"Pairwise logistic accuracy (x)\ncorr vs GT={metrics.corr_clf_vs_gt:.3f}")

    im_gt = ax4.imshow(metrics.gt_acc, aspect="auto", origin="lower", interpolation="nearest", vmin=0.0, vmax=1.0)
    fig.colorbar(im_gt, ax=ax4, fraction=0.046, pad=0.04, label="held-out accuracy")
    ax4.set_xlabel(r"bin $j$")
    ax4.set_ylabel(r"bin $i$")
    ax4.set_title(f"GT approx logistic accuracy (n={int(ctx.args.gt_approx_n_total)})")

    if s_combo > 0:
        smooth_sigmas = _auto_clf_smooth_sigmas(ctx.config.n_bins)
        assert int(smooth_sigmas.shape[0]) == _CLF_SMOOTH_N_STRENGTHS
        for k in range(_CLF_SMOOTH_N_STRENGTHS):
            ax_sm = fig.add_subplot(gs[1, k])
            smat = smooth_pairwise_matrix_gaussian(metrics.clf_acc, float(smooth_sigmas[k]))
            corr_sm_gt = matrix_corr_offdiag(smat, metrics.gt_acc)
            im_sm = ax_sm.imshow(
                smat, aspect="auto", origin="lower", interpolation="nearest", vmin=0.0, vmax=1.0
            )
            fig.colorbar(im_sm, ax=ax_sm, fraction=0.046, pad=0.04, label="smoothed held-out acc")
            ax_sm.set_xlabel(r"bin $j$")
            ax_sm.set_ylabel(r"bin $i$")
            ax_sm.set_title(
                "Smoothed pairwise logistic acc (x)\n"
                + rf"$\sigma_{{\mathrm{{smooth}}}}$={float(smooth_sigmas[k]):.3g}"
                + f", corr vs GT={corr_sm_gt:.3f}"
            )
        for si in range(s_combo):
            ax_s = fig.add_subplot(gs[2, si])
            corr_acc_gt = float(sssd_metrics.corr_acc_vs_gt_acc[si])
            sig_val = float(sssd_metrics.eval_sigmas[si])
            im_s = ax_s.imshow(
                sssd_metrics.acc_stack[si], aspect="auto", origin="lower", interpolation="nearest", vmin=0.0, vmax=1.0
            )
            fig.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04, label="in-sample LR acc")
            ax_s.set_xlabel(r"bin $j$")
            ax_s.set_ylabel(r"bin $i$")
            ax_s.set_title(rf"SSSD LR acc $\sigma$={sig_val:.3g}" + "\n" + f"corr vs GT acc={corr_acc_gt:.3f}")
        fig.suptitle(
            f"Binned matrices + {loaded.hell_suptitle_tag} + smoothed clf ({_CLF_SMOOTH_N_STRENGTHS} "
            + rf"$\sigma_{{\mathrm{{smooth}}}}$) + SSSD ({ctx.config.n_bins} bins; {s_combo} evaluation $\sigma$)",
            fontsize=13,
        )
    else:
        fig.suptitle(f"Binned matrices + {loaded.hell_suptitle_tag} ({ctx.config.n_bins} bins)", fontsize=13)

    plt.savefig(combo_fig_path, dpi=180, bbox_inches="tight")
    plt.close()

    sssd_acc_panels_path = ""
    sssd_acc_primary_path = ""
    sssd_acc_panels_file = os.path.join(ctx.full_args.output_dir, "h_matrix_sssd_acc_panels.png")
    sssd_acc_primary_file = os.path.join(ctx.full_args.output_dir, "h_matrix_sssd_acc_primary.png")
    if sssd_metrics.acc_stack.size > 0 and sssd_metrics.acc_stack.shape[0] > 0:
        s = int(sssd_metrics.acc_stack.shape[0])
        ncols = min(5, s)
        nrows = int(np.ceil(s / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 3.5 * nrows), squeeze=False)
        last_im = None
        for si in range(s):
            r, c = divmod(si, ncols)
            ax = axes[r][c]
            last_im = ax.imshow(
                sssd_metrics.acc_stack[si],
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                vmin=0.0,
                vmax=1.0,
            )
            ax.set_title(rf"$\sigma$={float(sssd_metrics.eval_sigmas[si]):.4g}")
            ax.set_xlabel(r"bin $j$")
            ax.set_ylabel(r"bin $i$")
        for si in range(s, nrows * ncols):
            r, c = divmod(si, ncols)
            axes[r][c].axis("off")
        fig.colorbar(last_im, ax=axes, fraction=0.03, pad=0.04, label="symmetric LR accuracy")
        fig.suptitle("SSSD symmetric pairwise LR accuracy (kernel-smoothed decoder)", fontsize=13)
        plt.savefig(sssd_acc_panels_file, dpi=180, bbox_inches="tight")
        plt.close()

        mid = s // 2
        _render_single_heatmap(
            sssd_metrics.acc_stack[mid],
            sssd_acc_primary_file,
            rf"SSSD LR accuracy (primary $\sigma$={float(sssd_metrics.eval_sigmas[mid]):.4g})",
            "symmetric LR accuracy",
            vmin=0.0,
            vmax=1.0,
        )
        sssd_acc_panels_path = sssd_acc_panels_file
        sssd_acc_primary_path = sssd_acc_primary_file

    return {
        "fig_path": fig_path,
        "clf_fig_path": clf_fig_path,
        "count_fig_path": count_fig_path,
        "hell_ub_fig_path": hell_ub_fig_path,
        "combo_fig_path": combo_fig_path,
        "sssd_acc_panels_path": sssd_acc_panels_path,
        "sssd_acc_primary_path": sssd_acc_primary_path,
    }


def write_summary(
    ctx: RunContext,
    loaded: LoadedHMatrix,
    metrics: BinnedMetrics,
    sssd_metrics: SSSDMetrics,
    paths: ArtifactPaths,
) -> None:
    n_bins = ctx.config.n_bins
    n_finite = int(np.sum(np.isfinite(metrics.h_binned)))
    n_nan = int(np.sum(~np.isfinite(metrics.h_binned)))
    clf_finite = int(np.sum(np.isfinite(metrics.clf_acc) & ~np.eye(n_bins, dtype=bool)))
    clf_nan_off = int(np.sum(~np.isfinite(metrics.clf_acc) & ~np.eye(n_bins, dtype=bool)))

    with open(paths.summary_path, "w", encoding="utf-8") as f:
        f.write("Theta-binned sqrt(H)-matrix + pairwise classifier summary\n")
        f.write(f"dataset_npz: {ctx.config.dataset_npz}\n")
        f.write(f"output_dir: {ctx.full_args.output_dir}\n")
        f.write(f"n_samples: {loaded.h_sym.shape[0]}\n")
        f.write(f"score_fisher_eval_data: {getattr(ctx.full_args, 'score_fisher_eval_data', '')}\n")
        f.write(f"h_field_method: {loaded.h_field_method}\n")
        f.write(f"{loaded.h_eval_scalar_name}: {loaded.h_eval_scalar_value}\n")
        f.write(f"num_theta_bins: {n_bins}\n")
        f.write(f"theta_bin_edges: [{metrics.edge_lo}, {metrics.edge_hi}] (min/max of theta_used from H-matrix)\n")
        f.write(f"h_binned finite cells: {n_finite} nan cells: {n_nan}\n")
        if n_finite > 0:
            f.write(
                f"h_binned min (finite): {float(np.nanmin(metrics.h_binned))} max (finite): {float(np.nanmax(metrics.h_binned))}\n"
            )
            f.write(
                f"h_binned_sqrt min (finite): {float(np.nanmin(metrics.h_binned_sqrt))} max: {float(np.nanmax(metrics.h_binned_sqrt))}\n"
            )

        f.write("classifier: sklearn LogisticRegression(lbfgs), pairwise bin i vs j, held-out accuracy\n")
        f.write(
            f"  clf_test_frac={float(ctx.args.clf_test_frac)} clf_min_class_count={int(ctx.args.clf_min_class_count)} "
            f"clf_max_iter={_SKLEARN_LR_MAX_ITER_DEFAULT} (sklearn LogisticRegression default) "
            f"clf_random_state={metrics.clf_rs}\n"
        )
        f.write(
            f"  off-diagonal finite: {clf_finite} nan: {clf_nan_off} "
            f"ok_pairs={metrics.clf_stats['ok_pairs']} insufficient_counts={metrics.clf_stats['insufficient_counts']} "
            f"split_fail={metrics.clf_stats['split_fail']} fit_fail={metrics.clf_stats['fit_fail']}\n"
        )

        off_mask = ~np.eye(n_bins, dtype=bool)
        sub_acc = metrics.clf_acc[off_mask]
        if np.any(np.isfinite(sub_acc)):
            f.write(
                f"  clf accuracy min (finite off-diag): {float(np.nanmin(sub_acc))} max: {float(np.nanmax(sub_acc))}\n"
            )
        gt_sub = metrics.gt_acc[off_mask]
        f.write(
            f"gt_approx: n_total={int(ctx.args.gt_approx_n_total)} seed={metrics.gt_seed} "
            f"ok_pairs={metrics.gt_stats['ok_pairs']} insufficient_counts={metrics.gt_stats['insufficient_counts']} "
            f"split_fail={metrics.gt_stats['split_fail']} fit_fail={metrics.gt_stats['fit_fail']}\n"
        )
        if np.any(np.isfinite(gt_sub)):
            f.write(
                f"  gt_approx accuracy min (finite off-diag): {float(np.nanmin(gt_sub))} max: {float(np.nanmax(gt_sub))}\n"
            )

        f.write(f"correlation_h_binned_sqrt_vs_gt_approx: {metrics.corr_h_vs_gt}\n")
        f.write(f"correlation_clf_binned_vs_gt_approx: {metrics.corr_clf_vs_gt}\n")
        f.write(
            loaded.hell_summary_prefix
            + "bin-average symmetric H_ij over theta-bin pairs, treated as H_ij^2; "
            "bounds on Bayes-optimal accuracy A*_ij: "
            "A^lb_ij = 0.5*(1+H_ij^2) <= A*_ij <= A^ub_ij = 0.5*(1+sqrt(2*H_ij^2-H_ij^4)); "
            "diagonal NaN; off-diagonal in [0.5,1] when H_ij^2 in [0,1].\n"
        )
        if str(loaded.h_field_method).strip().lower() != "flow":
            f.write(
                "  DSM note: the trained fields are denoising scores; "
                "∇_θ log p(θ|x) is the standard score target, and "
                "∇_θ p(θ|x) = p(θ|x) ∇_θ log p(θ|x) on the support of p.\n"
            )
        else:
            f.write(
                "  Flow note: H uses score recovered from velocity via the affine path conversion "
                "(path.velocity_to_epsilon followed by s = -eps / sigma_t), "
                "not raw velocity-as-score.\n"
            )
            if loaded.flow_scheduler is not None:
                f.write(f"  flow_scheduler: {loaded.flow_scheduler}\n")
            if loaded.flow_score_mode is not None:
                f.write(f"  flow_score_mode: {loaded.flow_score_mode}\n")
        f.write(f"correlation_hellinger_acc_lb_vs_gt_approx: {metrics.corr_hellinger_lb_vs_gt}\n")
        f.write(f"correlation_hellinger_acc_ub_vs_gt_approx: {metrics.corr_hellinger_ub_vs_gt}\n")

        if sssd_metrics.acc_stack.size > 0 and sssd_metrics.acc_stack.shape[0] > 0:
            tr = sssd_metrics.train_result
            f.write(
                "SSSD training (fisher/sssd.train_sssd_decoder): AdamW on soft cross-entropy vs "
                "Gaussian bin targets q_sigma(b|theta); each batch samples sigma log-uniformly in "
                f"[sigma_min,sigma_max] for targets and decoder conditioning. "
                f"max_epochs={int(ctx.args.sssd_epochs)}, early_stopping_patience={int(ctx.args.sssd_patience)} "
                "(0 disables), "
                f"batch_size={int(ctx.args.sssd_batch_size)}, "
                f"lr={float(ctx.args.sssd_lr):g}, val_frac={float(ctx.args.sssd_val_frac):g}. "
                "Checkpoint = lowest validation soft-CE (best_epoch). "
                + (
                    f"Stopped at epoch {tr.stopped_epoch} ({'early' if tr.stopped_early else 'full budget'}).\n"
                    if tr is not None
                    else "\n"
                )
            )
            f.write(
                "SSSD: sigma-conditioned softmax decoder, Gaussian soft bin targets; "
                "primary metrics are symmetric pairwise LR decision accuracies A_ij(sigma) "
                "(see fisher/sssd.py). Log-ratio matrices M_ij retained in NPZ for diagnostics.\n"
            )
            f.write(
                f"  sigma_train=[{sssd_metrics.sigma_min_used:g},{sssd_metrics.sigma_max_used:g}] best_epoch={sssd_metrics.best_epoch}\n"
            )
            for si in range(int(sssd_metrics.eval_sigmas.shape[0])):
                f.write(
                    f"  sigma={float(sssd_metrics.eval_sigmas[si]):.6g} "
                    f"corr(acc_dataset,acc_gt)={float(sssd_metrics.corr_acc_vs_gt_acc[si]):g} "
                    f"corr(M_dataset,M_gt)={float(sssd_metrics.corr_m_vs_gt_m[si]):g}\n"
                )

        f.write("artifacts:\n")
        f.write(f"  {paths.out_npz}\n")
        f.write(f"  {paths.fig_path}\n")
        f.write(f"  {paths.clf_fig_path}\n")
        f.write(f"  {paths.count_fig_path}\n")
        f.write(f"  {paths.hell_ub_fig_path}\n")
        f.write(f"  {paths.combo_fig_path}\n")
        if paths.sssd_acc_panels_path:
            f.write(f"  {paths.sssd_acc_panels_path}\n")
        if paths.sssd_acc_primary_path:
            f.write(f"  {paths.sssd_acc_primary_path}\n")
        f.write(f"  {paths.summary_path}\n")


def run_binned_visualization(config: BinnedVizConfig) -> BinnedVizResult:
    ctx = prepare_context(config)

    print(
        f"[h_binned] dataset_npz={config.dataset_npz} output_dir={ctx.full_args.output_dir} "
        f"num_theta_bins={config.n_bins} h_only={config.h_only} "
        f"theta_field_method={str(getattr(ctx.full_args, 'theta_field_method', 'dsm'))}"
    )

    run_h_estimation_if_needed(ctx)
    loaded = load_h_matrix(ctx)
    metrics = compute_binned_metrics(ctx, loaded)
    sssd_metrics = run_sssd_analysis(ctx, loaded, metrics)
    out_npz = write_results_npz(ctx, loaded, metrics, sssd_metrics)

    fig_paths = render_figures(ctx, loaded, metrics, sssd_metrics)
    summary_path = os.path.join(ctx.full_args.output_dir, "h_matrix_binned_summary.txt")
    artifacts = ArtifactPaths(
        out_npz=out_npz,
        fig_path=fig_paths["fig_path"],
        clf_fig_path=fig_paths["clf_fig_path"],
        count_fig_path=fig_paths["count_fig_path"],
        hell_ub_fig_path=fig_paths["hell_ub_fig_path"],
        combo_fig_path=fig_paths["combo_fig_path"],
        summary_path=summary_path,
        sssd_acc_panels_path=fig_paths["sssd_acc_panels_path"],
        sssd_acc_primary_path=fig_paths["sssd_acc_primary_path"],
    )
    write_summary(ctx, loaded, metrics, sssd_metrics, artifacts)

    print("[h_binned] Saved:")
    print(f"  - {artifacts.out_npz}")
    print(f"  - {artifacts.fig_path}")
    print(f"  - {artifacts.clf_fig_path}")
    print(f"  - {artifacts.count_fig_path}")
    print(f"  - {artifacts.hell_ub_fig_path}")
    print(f"  - {artifacts.combo_fig_path}")
    if artifacts.sssd_acc_panels_path:
        print(f"  - {artifacts.sssd_acc_panels_path}")
    if artifacts.sssd_acc_primary_path:
        print(f"  - {artifacts.sssd_acc_primary_path}")
    print(f"  - {artifacts.summary_path}")

    return BinnedVizResult(context=ctx, h_matrix=loaded, binned=metrics, sssd=sssd_metrics, artifacts=artifacts)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    run_binned_visualization(config)


if __name__ == "__main__":
    main()
