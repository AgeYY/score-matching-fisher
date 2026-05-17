#!/usr/bin/env python3
"""Nested-n sweep for categorical random MoG: LLR-based category Hellinger vs analytic GT.

Mirrors :mod:`fisher.h_decoding_twofig` layout (merged **sweep+GT** matrix figure, **corr+NMSE** two-panel figure,
**training-loss panel** SVG, NPZ, summary) while using the categorical estimators and directed-then-symmetrized
category aggregation from ``bin/debug_categorical_xflow_llr.py``. Specialized to ``random_mog_categorical`` only.

Use ``--eval-split validation`` to evaluate learned H, GT LLR diagnostics, and pairwise decoding accuracy
on the held-out validation rows of each nested ``n`` subset only. Pairwise classifiers still train on the
nested training slice.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent
_bin_dir = _repo_root / "bin"
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_bin_dir) not in sys.path:
    sys.path.insert(0, str(_bin_dir))

import global_setting  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression

from global_setting import DATA_DIR

from fisher import h_binned_visualization as vhb
from fisher import h_decoding_convergence as conv
from fisher.gaussian_x_flow import path_schedule_from_name
from fisher.hellinger_gt import hellinger_sq_gaussian_diag
from fisher.h_decoding_convergence_methods import (
    SweepSubset,
    _fit_sir_projection,
    prepare_categorical_binning_for_convergence,
)
from fisher.shared_dataset_io import SharedDatasetBundle
from fisher.h_decoding_twofig import (
    _matrix_nmse_offdiag,
    _render_corr_nmse_two_panel,
    _render_method_sweep_panel,
    _render_row_n_training_losses_panel,
    _sanitize_row_label,
    _theta_axis_tick_labels,
)
from fisher.contrastive_llr import (
    ContrastiveAdditiveIndependentScorer,
    ContrastiveNormalizedDotScorer,
    compute_contrastive_soft_categorical_c_matrix,
    train_contrastive_soft_categorical_llr,
)
from fisher.h_matrix import HMatrixEstimator
from fisher.shared_dataset_io import SharedDatasetBundle, load_shared_dataset_npz
from fisher.ctsm_models import ToyPairConditionedTimeScoreNet, ToyPairConditionedTimeScoreNetFiLM
from fisher.linear_x_flow import (
    ConditionalTimeLinearXFlowMLP,
    ConditionalTimeLowRankCorrectionLinearXFlowMLP,
    compute_ode_time_linear_x_flow_c_matrix,
    compute_time_linear_x_flow_c_matrix,
    train_time_linear_x_flow_schedule,
)
from fisher.shared_fisher_est import (
    build_conditional_x_velocity_model,
    build_dataset_from_meta,
    require_device,
    train_pair_conditioned_ctsm_v_model,
)
from fisher.svg_utils import concatenate_svgs_horizontally_to_png
from fisher.models import (
    ConditionalThetaFlowVelocity,
    ConditionalThetaFlowVelocityFiLMPerLayer,
    PriorThetaFlowVelocity,
    PriorThetaFlowVelocityFiLMPerLayer,
)
from fisher.trainers import (
    train_conditional_x_flow_model,
    train_conditional_theta_flow_likelihood_finetune,
    train_conditional_theta_flow_model,
    train_prior_theta_flow_likelihood_finetune,
    train_prior_theta_flow_model,
)

_METHOD_ALIASES: dict[str, str] = {
    "x-flow": "x_flow",
    "x_flow": "x_flow",
    "xflow": "x_flow",
    "binary-classification": "binary_classifier",
    "binary_classification": "binary_classifier",
    "binary-classifier": "binary_classifier",
    "binary_classifier": "binary_classifier",
    "classifier": "binary_classifier",
    "linear-x-flow-t": "linear_x_flow_t",
    "linear_x_flow_t": "linear_x_flow_t",
    "xflow-sir-lrank": "xflow_sir_lrank",
    "xflow_sir_lrank": "xflow_sir_lrank",
    "contrastive-soft-categorical": "contrastive_soft_categorical",
    "contrastive_soft_categorical": "contrastive_soft_categorical",
    "theta-flow-cate": "theta_flow_cate",
    "theta_flow_cate": "theta_flow_cate",
    "thetaflow-cate": "theta_flow_cate",
    "bin_gaussian": "bin_gaussian",
    "bin_gaussian_cate": "bin_gaussian",
    "bin-gaussian": "bin_gaussian",
    "binned_gaussian": "bin_gaussian",
    "binned-gaussian": "bin_gaussian",
    "ctsm-v": "ctsm_v",
    "ctsm_v": "ctsm_v",
    "ctsm": "ctsm_v",
    "lda-ctsm-v": "lda_ctsm_v",
    "lda_ctsm_v": "lda_ctsm_v",
    "ldactsm-v": "lda_ctsm_v",
    "pls-ctsm-v": "pls_ctsm_v",
    "pls_ctsm_v": "pls_ctsm_v",
    "plsctsm-v": "pls_ctsm_v",
    "pca-ctsm-v": "pca_ctsm_v",
    "pca_ctsm_v": "pca_ctsm_v",
    "pcactsm-v": "pca_ctsm_v",
}
_DEFAULT_METHODS = ("x_flow", "binary_classifier", "linear_x_flow_t", "xflow_sir_lrank")
_SUPPORTED_METHODS_HELP = ", ".join(
    _DEFAULT_METHODS
    + (
        "contrastive_soft_categorical",
        "theta_flow_cate",
        "bin_gaussian",
        "ctsm_v",
        "lda_ctsm_v",
        "pls_ctsm_v",
        "pca_ctsm_v",
    )
)
_ALL_COLUMNS_PNG_NAME = "h_decoding_categorical_twofig_all_columns.png"


def _debug_categorical_module() -> Any:
    """Compatibility shim for tests/older call sites that patched the removed bin helper module."""
    return sys.modules[__name__]


def parse_methods(methods: str) -> list[str]:
    toks = [t.strip() for t in str(methods).split(",") if t.strip()]
    if not toks:
        raise ValueError("--methods must contain at least one method.")
    out: list[str] = []
    for tok in toks:
        key = tok.strip().lower()
        norm = _METHOD_ALIASES.get(key)
        if norm is None:
            raise ValueError(f"Unknown method {tok!r}; valid methods: {_SUPPORTED_METHODS_HELP}")
        if norm not in out:
            out.append(norm)
    return out


def _write_all_columns_png(
    output_dir: str | os.PathLike[str],
    *,
    sweep_svg: str,
    corr_nmse_svg: str,
    loss_panel_svg: str,
) -> str:
    """Rasterize sweep, corr/NMSE, and training-loss SVGs into one wide PNG.

    Optional per-method diagnostic SVGs (e.g. ``llr_est_vs_true_all.svg``) are **not** included:
    for larger $K$ and many methods they can contain millions of SVG primitives and make
    ``rsvg-convert`` fail (librsvg element limits); those files remain available standalone.
    """
    out_dir = Path(output_dir)
    source_paths: list[str | Path] = [sweep_svg, corr_nmse_svg, loss_panel_svg]
    return concatenate_svgs_horizontally_to_png(source_paths, out_dir / _ALL_COLUMNS_PNG_NAME, dpi=300)


def _default_dataset_npz(num_categories: int) -> Path:
    return _repo_root / "data" / f"random_mog_categorical_xdim2_k{int(num_categories)}" / "random_mog_categorical.npz"


def _abs_without_resolving_symlinks(path: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return _repo_root / p


def _default_pr_output_npz(native_npz: Path, pr_dim: int) -> Path:
    return _abs_without_resolving_symlinks(native_npz).parent / f"pr_xdim{int(pr_dim)}" / "random_mog_categorical_pr.npz"


def _run(cmd: list[str]) -> None:
    print("[cat-twofig] running: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(_repo_root), check=True)


def _ensure_dataset(args: argparse.Namespace) -> None:
    ds = Path(args.dataset_npz)
    ds.parent.mkdir(parents=True, exist_ok=True)
    if args.force_regenerate or not ds.exists():
        _run(
            [
                sys.executable,
                "bin/make_dataset.py",
                "--dataset-family",
                "random_mog_categorical",
                "--num-categories",
                str(int(args.num_categories)),
                "--n-total",
                str(int(args.n_total)),
                "--output-npz",
                str(ds),
            ]
        )
    else:
        print(f"[cat-twofig] using existing 2D dataset NPZ: {ds}", flush=True)


def _ensure_pr_projected_npz(args: argparse.Namespace, *, native_npz: Path, pr_out: Path) -> None:
    pr_out = Path(pr_out)
    native_npz = Path(native_npz).resolve()
    pr_out.parent.mkdir(parents=True, exist_ok=True)
    if bool(args.force_regenerate) and pr_out.is_file():
        pr_out.unlink()
    if pr_out.is_file():
        print(f"[cat-twofig] using existing PR-projected NPZ: {pr_out}", flush=True)
        return
    cmd: list[str] = [
        sys.executable,
        str(_repo_root / "bin" / "project_dataset_pr_autoencoder.py"),
        "--input-npz",
        str(native_npz),
        "--output-npz",
        str(pr_out.resolve()),
        "--h-dim",
        str(int(args.pr_dim)),
        "--device",
        str(args.device),
        "--allow-non-randamp-sqrtd",
        "--cache-dir",
        str(args.pr_cache_dir),
    ]
    if bool(args.pr_use_cache):
        cmd.append("--use-cache")
    if bool(args.pr_skip_viz):
        cmd.append("--skip-viz")
    if args.pr_hidden1 is not None:
        cmd.extend(["--pr-hidden1", str(int(args.pr_hidden1))])
    if args.pr_hidden2 is not None:
        cmd.extend(["--pr-hidden2", str(int(args.pr_hidden2))])
    if args.pr_train_samples is not None:
        cmd.extend(["--pr-train-samples", str(int(args.pr_train_samples))])
    if args.pr_train_epochs is not None:
        cmd.extend(["--pr-train-epochs", str(int(args.pr_train_epochs))])
    if args.pr_train_batch_size is not None:
        cmd.extend(["--pr-train-batch-size", str(int(args.pr_train_batch_size))])
    if args.pr_train_lr is not None:
        cmd.extend(["--pr-train-lr", str(float(args.pr_train_lr))])
    if args.pr_lambda_pr is not None:
        cmd.extend(["--pr-lambda-pr", str(float(args.pr_lambda_pr))])
    if args.pr_eps is not None:
        cmd.extend(["--pr-eps", str(float(args.pr_eps))])
    print("[cat-twofig] running PR projection: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(_repo_root), check=True)


def _validation_only_work_sweep_subset(subset_w: SweepSubset) -> SweepSubset:
    """Build a :class:`SweepSubset` whose ``bundle.x_all`` / ``bin_all`` are validation rows only.

    Used for ``--eval-split validation`` with :func:`conv._binned_gaussian_hellinger_sq`,
    which reads ``subset.bundle.x_all`` and ``subset.bin_all``.
    """
    b = subset_w.bundle
    tv = np.asarray(b.theta_validation, dtype=np.float64)
    xv = np.asarray(b.x_validation, dtype=np.float64)
    binv = np.asarray(subset_w.bin_validation, dtype=np.int64).reshape(-1)
    if tv.ndim == 1:
        tv = tv.reshape(-1, 1)
    elif tv.ndim != 2:
        raise ValueError("theta_validation must be 1D or 2D.")
    n = int(xv.shape[0])
    if n < 1:
        raise ValueError("validation split is empty.")
    if int(binv.shape[0]) != n:
        raise ValueError("bin_validation length must match x_validation.")
    idx = np.arange(n, dtype=np.int64)
    empty = np.arange(0, dtype=np.int64)
    meta = dict(b.meta)
    new_b = SharedDatasetBundle(
        meta=meta,
        theta_all=tv,
        x_all=xv,
        train_idx=empty,
        validation_idx=idx,
        theta_train=np.empty((0, tv.shape[1]), dtype=np.float64),
        x_train=np.empty((0, xv.shape[1]), dtype=np.float64),
        theta_validation=tv,
        x_validation=xv,
    )
    return SweepSubset(bundle=new_b, bin_all=binv, bin_train=binv, bin_validation=binv)


def compute_true_conditional_loglik_matrix(x_all: np.ndarray, theta_all: np.ndarray, meta: dict) -> np.ndarray:
    gen_ds = build_dataset_from_meta(dict(meta))
    n = int(np.asarray(x_all).shape[0])
    x_all = np.asarray(x_all, dtype=np.float64)
    theta_all = np.asarray(theta_all, dtype=np.float64)
    true_c = np.empty((n, n), dtype=np.float64)
    for j in range(n):
        th_col = np.tile(theta_all[j : j + 1], (n, 1))
        true_c[:, j] = gen_ds.log_p_x_given_theta(x_all, th_col)
    return true_c


def hellinger_gt_sq_category_matrix(gen_ds: object) -> np.ndarray:
    means = np.asarray(getattr(gen_ds, "_mog_means"), dtype=np.float64)
    variances = np.asarray(getattr(gen_ds, "_mog_variances"), dtype=np.float64)
    k = int(means.shape[0])
    if int(variances.shape[0]) != k or int(means.shape[1]) != int(variances.shape[1]):
        raise ValueError("Inconsistent _mog_means / _mog_variances on dataset.")
    h2 = np.zeros((k, k), dtype=np.float64)
    for a in range(k):
        for b in range(k):
            h2[a, b] = hellinger_sq_gaussian_diag(means[a], variances[a], means[b], variances[b])
    np.fill_diagonal(h2, 0.0)
    np.clip(h2, 0.0, 1.0, out=h2)
    return h2


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]
    b = b[finite]
    if a.size < 2:
        return float("nan")
    if float(np.std(a)) < 1e-15 or float(np.std(b)) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def h_sq_directed_from_delta_l(delta_l: np.ndarray) -> np.ndarray:
    return HMatrixEstimator.compute_h_directed(np.asarray(delta_l, dtype=np.float64))


def h_sq_sym_sample_from_delta_l(delta_l: np.ndarray) -> np.ndarray:
    h_dir = h_sq_directed_from_delta_l(delta_l)
    return HMatrixEstimator.symmetrize(h_dir)


def h_sq_category_from_sample_directed(
    h_directed: np.ndarray,
    category_labels: np.ndarray,
    *,
    k_cat: int,
) -> np.ndarray:
    """Aggregate directed $H^2$ to categories, symmetrize, zero diagonal (debug script recipe)."""
    h = np.asarray(h_directed, dtype=np.float64)
    n = int(h.shape[0])
    labs = np.asarray(category_labels, dtype=np.int64).reshape(-1)
    if int(labs.shape[0]) != n:
        raise ValueError("category_labels length must match h_directed.shape[0].")
    col_sum = np.zeros((n, int(k_cat)), dtype=np.float64)
    col_cnt = np.zeros((n, int(k_cat)), dtype=np.float64)
    for j in range(n):
        b = int(labs[j])
        if 0 <= b < int(k_cat):
            col_sum[:, b] += h[:, j]
            col_cnt[:, b] += 1.0
    tilde = col_sum / np.maximum(col_cnt, 1.0)
    row_sum = np.zeros((int(k_cat), int(k_cat)), dtype=np.float64)
    row_cnt = np.zeros((int(k_cat), int(k_cat)), dtype=np.float64)
    for i in range(n):
        a = int(labs[i])
        if 0 <= a < int(k_cat):
            row_sum[a, :] += tilde[i, :]
            row_cnt[a, :] += 1.0
    h_dir_cat = row_sum / np.maximum(row_cnt, 1.0)
    h_sym = 0.5 * (h_dir_cat + h_dir_cat.T)
    np.fill_diagonal(h_sym, 0.0)
    return h_sym


_h_sq_category_from_sample_directed = h_sq_category_from_sample_directed


def _hellinger_cat_offdiag_mask(k: int) -> np.ndarray:
    return np.triu(np.ones((int(k), int(k)), dtype=bool), k=1)


def _hellinger_comparison_metrics_cat(est: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    e = np.asarray(est, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    k = int(e.shape[0]) if e.ndim == 2 else 0
    if g.shape != e.shape or e.ndim != 2 or k < 2:
        return {
            "hellinger_mae_offdiag_cat": float("nan"),
            "hellinger_rmse_offdiag_cat": float("nan"),
            "hellinger_bias_offdiag_cat": float("nan"),
            "hellinger_pearson_r_offdiag_cat": float("nan"),
        }
    m = _hellinger_cat_offdiag_mask(k)
    d = (e - g)[m]
    return {
        "hellinger_mae_offdiag_cat": float(np.nanmean(np.abs(d))),
        "hellinger_rmse_offdiag_cat": float(np.sqrt(np.nanmean(d**2))),
        "hellinger_bias_offdiag_cat": float(np.nanmean(d)),
        "hellinger_pearson_r_offdiag_cat": _pearson_r(e[m], g[m]),
    }


LLR_GT_METRIC_BAND_LO = -8.0
LLR_GT_METRIC_BAND_HI = 8.0


def _llr_comparison_metrics(est_delta: np.ndarray, true_delta: np.ndarray) -> dict[str, float]:
    est = np.asarray(est_delta, dtype=np.float64)
    true_m = np.asarray(true_delta, dtype=np.float64)
    diff = est - true_m
    finite = np.isfinite(diff)
    if np.any(finite):
        all_abs = float(np.nanmean(np.abs(diff)))
        all_rmse = float(np.sqrt(np.nanmean(diff**2)))
        all_bias = float(np.nanmean(diff))
        all_std = float(np.nanstd(diff, ddof=0))
    else:
        all_abs = all_rmse = all_bias = all_std = float("nan")
    out: dict[str, float] = {
        "llr_mae_all": all_abs,
        "llr_rmse_all": all_rmse,
        "llr_bias_all": all_bias,
        "llr_std_err_all": all_std,
        "llr_pearson_r_all": _pearson_r(est, true_m),
    }
    n = int(est.shape[0]) if est.ndim == 2 else 0
    if n > 1 and true_m.shape == est.shape:
        mask = ~np.eye(n, dtype=bool)
        d_off = diff[mask]
        e_off = est[mask]
        t_off = true_m[mask]
        finite_off = np.isfinite(d_off)
        if np.any(finite_off):
            out["llr_mae_offdiag"] = float(np.nanmean(np.abs(d_off)))
            out["llr_rmse_offdiag"] = float(np.sqrt(np.nanmean(d_off**2)))
            out["llr_bias_offdiag"] = float(np.nanmean(d_off))
            out["llr_std_err_offdiag"] = float(np.nanstd(d_off, ddof=0))
        else:
            out["llr_mae_offdiag"] = float("nan")
            out["llr_rmse_offdiag"] = float("nan")
            out["llr_bias_offdiag"] = float("nan")
            out["llr_std_err_offdiag"] = float("nan")
        out["llr_pearson_r_offdiag"] = _pearson_r(e_off, t_off)
        band_m = (t_off >= LLR_GT_METRIC_BAND_LO) & (t_off <= LLR_GT_METRIC_BAND_HI)
        e_b = e_off[band_m]
        t_b = t_off[band_m]
        if int(e_b.size) == 0 or not np.any(np.isfinite(e_b - t_b)):
            out["llr_rmse_offdiag_true_in_m8_p8"] = float("nan")
            out["llr_pearson_r_offdiag_true_in_m8_p8"] = float("nan")
        else:
            d_b = e_b - t_b
            out["llr_rmse_offdiag_true_in_m8_p8"] = float(np.sqrt(np.nanmean(d_b**2)))
            out["llr_pearson_r_offdiag_true_in_m8_p8"] = _pearson_r(e_b, t_b)
    else:
        out["llr_mae_offdiag"] = float("nan")
        out["llr_rmse_offdiag"] = float("nan")
        out["llr_bias_offdiag"] = float("nan")
        out["llr_std_err_offdiag"] = float("nan")
        out["llr_pearson_r_offdiag"] = float("nan")
        out["llr_rmse_offdiag_true_in_m8_p8"] = float("nan")
        out["llr_pearson_r_offdiag_true_in_m8_p8"] = float("nan")
    return out


def _save_llr_est_vs_true_figure(
    method_deltas: dict[str, np.ndarray],
    true_delta: np.ndarray,
    *,
    out_base: Path,
    metrics_by_method: dict[str, dict[str, float]],
) -> None:
    n = int(true_delta.shape[0])
    mask = ~np.eye(n, dtype=bool) if n > 1 else np.zeros_like(true_delta, dtype=bool)
    x = np.asarray(true_delta, dtype=np.float64)[mask].ravel()
    fig, ax = plt.subplots(figsize=(6.2, 5.8), layout="constrained")
    if x.size == 0:
        ax.text(0.5, 0.5, "No off-diagonal LLR pairs", ha="center", va="center", transform=ax.transAxes)
    else:
        ys: list[np.ndarray] = []
        for method_name, est_delta in method_deltas.items():
            y = np.asarray(est_delta, dtype=np.float64)[mask].ravel()
            ys.append(y)
            m = metrics_by_method.get(method_name, {})
            label = (
                f"{method_name} "
                f"(RMSE={m.get('llr_rmse_offdiag', float('nan')):.3g}, "
                f"r={m.get('llr_pearson_r_offdiag', float('nan')):.3g})"
            )
            ax.scatter(x, y, s=8, alpha=0.28, linewidths=0, label=label)
        vals = np.concatenate([x] + ys) if ys else x
        vals = vals[np.isfinite(vals)]
        lo = float(np.min(vals)) if vals.size else -1.0
        hi = float(np.max(vals)) if vals.size else 1.0
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.0, alpha=0.7)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.set_xlabel(r"true $\Delta L$")
    ax.set_ylabel(r"estimated $\Delta L$")
    ax.set_title(r"LLR: estimated vs ground truth")
    ax.grid(True, alpha=0.25)
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_hellinger_est_vs_gt_figure(
    est_cat_by_method: dict[str, np.ndarray],
    hellinger_gt_sq_category: np.ndarray,
    *,
    out_base: Path,
    metrics_by_method: dict[str, dict[str, float]],
) -> None:
    gt = np.asarray(hellinger_gt_sq_category, dtype=np.float64)
    k = int(gt.shape[0])
    mask = _hellinger_cat_offdiag_mask(k) if k > 1 else np.zeros_like(gt, dtype=bool)
    x = gt[mask].ravel()
    fig, ax = plt.subplots(figsize=(6.2, 5.8), layout="constrained")
    if x.size == 0:
        ax.text(0.5, 0.5, "No category off-diagonal pairs", ha="center", va="center", transform=ax.transAxes)
    else:
        cmap = plt.get_cmap("tab10")
        ys: list[np.ndarray] = []
        for idx, (method_name, est) in enumerate(est_cat_by_method.items()):
            y = np.asarray(est, dtype=np.float64)[mask].ravel()
            ys.append(y)
            m = metrics_by_method.get(method_name, {})
            label = (
                f"{method_name} "
                f"(RMSE={m.get('hellinger_rmse_offdiag_cat', float('nan')):.3g}, "
                f"r={m.get('hellinger_pearson_r_offdiag_cat', float('nan')):.3g})"
            )
            ax.scatter(x, y, s=22, alpha=0.55, linewidths=0, color=cmap(idx % 10), label=label)
        vals = np.concatenate([x] + ys) if ys else x
        vals = vals[np.isfinite(vals)]
        lo = float(np.min(vals)) if vals.size else 0.0
        hi = float(np.max(vals)) if vals.size else 1.0
        pad = 0.05 * (hi - lo) if hi > lo else 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.0, alpha=0.7)
        ax.set_xlim(max(0.0, lo - pad), min(1.0, hi + pad))
        ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best", fontsize=7, framealpha=0.9)
    ax.set_xlabel(r"analytic GT $H^2$ (category)")
    ax.set_ylabel(r"LLR-derived est. $H^2$ (category)")
    ax.set_title(r"$H^2$: category-level LLR map vs analytic GT")
    ax.grid(True, alpha=0.25)
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _build_theta_flow_post_model(args: argparse.Namespace, *, x_dim: int, dev: torch.device) -> torch.nn.Module:
    arch = str(getattr(args, "flow_arch", "mlp")).strip().lower()
    if arch == "mlp":
        return ConditionalThetaFlowVelocity(
            x_dim=int(x_dim),
            hidden_dim=int(getattr(args, "flow_hidden_dim", 128)),
            depth=int(getattr(args, "flow_depth", 3)),
            use_logit_time=True,
            theta_dim=1,
        ).to(dev)
    if arch == "film":
        return ConditionalThetaFlowVelocityFiLMPerLayer(
            x_dim=int(x_dim),
            hidden_dim=int(getattr(args, "flow_hidden_dim", 128)),
            depth=int(getattr(args, "flow_depth", 3)),
            use_logit_time=True,
            use_layer_norm=False,
            gated_film=False,
            zero_out_init=False,
            cond_embed_dim=16,
            cond_embed_depth=1,
            cond_embed_act="silu",
        ).to(dev)
    raise ValueError("theta_flow_cate currently supports --flow-arch mlp or film.")


def _build_theta_flow_prior_model(args: argparse.Namespace, *, dev: torch.device) -> torch.nn.Module:
    arch = str(getattr(args, "flow_arch", "mlp")).strip().lower()
    if arch == "mlp":
        return PriorThetaFlowVelocity(
            hidden_dim=int(getattr(args, "prior_hidden_dim", getattr(args, "flow_hidden_dim", 128))),
            depth=int(getattr(args, "prior_depth", getattr(args, "flow_depth", 3))),
            use_logit_time=True,
            theta_dim=1,
        ).to(dev)
    if arch == "film":
        return PriorThetaFlowVelocityFiLMPerLayer(
            hidden_dim=int(getattr(args, "prior_hidden_dim", getattr(args, "flow_hidden_dim", 128))),
            depth=int(getattr(args, "prior_depth", getattr(args, "flow_depth", 3))),
            use_logit_time=True,
            use_layer_norm=False,
            gated_film=False,
            zero_out_init=False,
            cond_embed_dim=16,
            cond_embed_depth=1,
            cond_embed_act="silu",
        ).to(dev)
    raise ValueError("theta_flow_cate currently supports --flow-arch mlp or film.")


def _mean_padded_curves(curves: list[np.ndarray]) -> np.ndarray:
    nonempty = [np.asarray(c, dtype=np.float64).ravel() for c in curves if np.asarray(c).size > 0]
    if not nonempty:
        return np.asarray([], dtype=np.float64)
    width = max(int(c.size) for c in nonempty)
    arr = np.full((len(nonempty), width), np.nan, dtype=np.float64)
    for i, c in enumerate(nonempty):
        arr[i, : c.size] = c
    return np.nanmean(arr, axis=0)


def theta_flow_categorical_hellinger_sqrt(
    args: argparse.Namespace,
    *,
    dev: torch.device,
    x_train: np.ndarray,
    bins_train: np.ndarray,
    x_val: np.ndarray,
    bins_val: np.ndarray,
    x_all: np.ndarray,
    bins_all: np.ndarray,
    k_cat: int,
) -> dict[str, Any]:
    """Pairwise binary theta-flow category estimator returning categorical ``sqrt(H^2)``."""
    x_train = np.asarray(x_train, dtype=np.float64)
    x_val = np.asarray(x_val, dtype=np.float64)
    x_all = np.asarray(x_all, dtype=np.float64)
    bins_train = np.asarray(bins_train, dtype=np.int64).reshape(-1)
    bins_val = np.asarray(bins_val, dtype=np.int64).reshape(-1)
    bins_all = np.asarray(bins_all, dtype=np.int64).reshape(-1)
    min_count = int(getattr(args, "clf_min_class_count", 5))
    h_sqrt = np.full((int(k_cat), int(k_cat)), np.nan, dtype=np.float64)
    np.fill_diagonal(h_sqrt, 0.0)
    valid_pairs: list[tuple[int, int]] = []
    skipped_pairs: list[tuple[int, int, str]] = []
    train_counts = np.zeros((int(k_cat), int(k_cat), 2), dtype=np.int64)
    val_counts = np.zeros((int(k_cat), int(k_cat), 2), dtype=np.int64)
    pair_train_outs: list[dict[str, Any]] = []

    for a in range(int(k_cat)):
        for b in range(a + 1, int(k_cat)):
            tr_mask = (bins_train == a) | (bins_train == b)
            va_mask = (bins_val == a) | (bins_val == b)
            ev_mask = (bins_all == a) | (bins_all == b)
            tr_y = (bins_train[tr_mask] == b).astype(np.float64).reshape(-1, 1)
            va_y = (bins_val[va_mask] == b).astype(np.float64).reshape(-1, 1)
            ev_y = (bins_all[ev_mask] == b).astype(np.float64).reshape(-1, 1)
            tr_counts_pair = np.bincount(tr_y.reshape(-1).astype(np.int64), minlength=2)[:2]
            va_counts_pair = np.bincount(va_y.reshape(-1).astype(np.int64), minlength=2)[:2]
            train_counts[a, b] = train_counts[b, a] = tr_counts_pair
            val_counts[a, b] = val_counts[b, a] = va_counts_pair
            if np.any(tr_counts_pair < min_count):
                skipped_pairs.append((a, b, "min_train_count"))
                continue
            if np.any(va_counts_pair < 1):
                skipped_pairs.append((a, b, "min_validation_count"))
                continue
            if int(ev_y.shape[0]) < 2 or len(np.unique(ev_y.reshape(-1))) < 2:
                skipped_pairs.append((a, b, "missing_eval_class"))
                continue

            print(
                f"[theta_flow_cate] pair=({a},{b}) train_counts={tr_counts_pair.tolist()} "
                f"val_counts={va_counts_pair.tolist()} eval={int(ev_y.shape[0])}",
                flush=True,
            )
            post_model = _build_theta_flow_post_model(args, x_dim=int(x_train.shape[1]), dev=dev)
            post_train_out = train_conditional_theta_flow_model(
                model=post_model,
                theta_train=tr_y,
                x_train=x_train[tr_mask],
                epochs=int(getattr(args, "flow_epochs", 10000)),
                batch_size=int(getattr(args, "flow_batch_size", 256)),
                lr=float(getattr(args, "flow_lr", 1e-3)),
                device=dev,
                log_every=max(1, int(getattr(args, "log_every", 50))),
                theta_val=va_y,
                x_val=x_val[va_mask],
                early_stopping_patience=int(getattr(args, "flow_early_patience", 1000)),
                early_stopping_min_delta=float(getattr(args, "flow_early_min_delta", 1e-4)),
                early_stopping_ema_alpha=float(getattr(args, "flow_early_ema_alpha", 0.05)),
                restore_best=bool(getattr(args, "flow_restore_best", True)),
                scheduler_name=str(getattr(args, "flow_scheduler", "cosine")),
                fm_t_eps=float(getattr(args, "flow_fm_t_eps", 0.05)),
            )
            pair_out: dict[str, Any] = {"pair": (a, b), "post_train_out": post_train_out}
            ft_epochs = int(getattr(args, "flow_likelihood_finetune_epochs", 0))
            ft_batch = int(getattr(args, "flow_likelihood_finetune_batch_size", 0))
            if ft_batch <= 0:
                ft_batch = int(getattr(args, "flow_batch_size", 256))
            if ft_epochs > 0:
                pair_out["post_likelihood_finetune_out"] = train_conditional_theta_flow_likelihood_finetune(
                    model=post_model,
                    theta_train=tr_y,
                    x_train=x_train[tr_mask],
                    epochs=ft_epochs,
                    batch_size=ft_batch,
                    lr=float(getattr(args, "flow_likelihood_finetune_lr", 1e-4)),
                    device=dev,
                    log_every=max(1, int(getattr(args, "log_every", 50))),
                    theta_val=va_y,
                    x_val=x_val[va_mask],
                    early_stopping_patience=int(getattr(args, "flow_likelihood_finetune_patience", 100)),
                    early_stopping_min_delta=float(getattr(args, "flow_likelihood_finetune_min_delta", 1e-4)),
                    early_stopping_ema_alpha=float(getattr(args, "flow_likelihood_finetune_ema_alpha", 0.05)),
                    restore_best=bool(getattr(args, "flow_restore_best", True)),
                    ode_steps=int(getattr(args, "flow_likelihood_finetune_ode_steps", 64)),
                )

            posterior_only = bool(getattr(args, "theta_flow_posterior_only_likelihood", False))
            prior_model = None
            if not posterior_only:
                prior_model = _build_theta_flow_prior_model(args, dev=dev)
                prior_train_out = train_prior_theta_flow_model(
                    model=prior_model,
                    theta_train=tr_y,
                    epochs=int(getattr(args, "prior_epochs", 10000)),
                    batch_size=int(getattr(args, "prior_batch_size", getattr(args, "flow_batch_size", 256))),
                    lr=float(getattr(args, "prior_lr", 1e-3)),
                    device=dev,
                    log_every=max(1, int(getattr(args, "log_every", 50))),
                    theta_val=va_y,
                    early_stopping_patience=int(getattr(args, "prior_early_patience", 1000)),
                    early_stopping_min_delta=float(getattr(args, "prior_early_min_delta", 1e-4)),
                    early_stopping_ema_alpha=float(getattr(args, "prior_early_ema_alpha", 0.05)),
                    restore_best=bool(getattr(args, "prior_restore_best", True)),
                    scheduler_name=str(getattr(args, "flow_scheduler", "cosine")),
                    fm_t_eps=float(getattr(args, "flow_fm_t_eps", 0.05)),
                )
                pair_out["prior_train_out"] = prior_train_out
                if ft_epochs > 0:
                    pair_out["prior_likelihood_finetune_out"] = train_prior_theta_flow_likelihood_finetune(
                        model=prior_model,
                        theta_train=tr_y,
                        epochs=ft_epochs,
                        batch_size=ft_batch,
                        lr=float(getattr(args, "flow_likelihood_finetune_lr", 1e-4)),
                        device=dev,
                        log_every=max(1, int(getattr(args, "log_every", 50))),
                        theta_val=va_y,
                        early_stopping_patience=int(getattr(args, "flow_likelihood_finetune_patience", 100)),
                        early_stopping_min_delta=float(getattr(args, "flow_likelihood_finetune_min_delta", 1e-4)),
                        early_stopping_ema_alpha=float(getattr(args, "flow_likelihood_finetune_ema_alpha", 0.05)),
                        restore_best=bool(getattr(args, "prior_restore_best", True)),
                        ode_steps=int(getattr(args, "flow_likelihood_finetune_ode_steps", 64)),
                    )

            est = HMatrixEstimator(
                model_post=post_model,
                model_prior=prior_model,
                sigma_eval=1.0,
                device=dev,
                pair_batch_size=int(getattr(args, "h_batch_size", 65536)),
                field_method="theta_flow",
                flow_scheduler=str(getattr(args, "flow_scheduler", "cosine")),
                flow_ode_steps=int(getattr(args, "flow_ode_steps", 64)),
                flow_likelihood_exact_divergence=bool(getattr(args, "flow_likelihood_exact_divergence", False)),
                theta_flow_posterior_only_likelihood=posterior_only,
            )
            h_res = est.run(ev_y, x_all[ev_mask], restore_original_order=True)
            h_pair_sq = h_sq_category_from_sample_directed(h_res.h_directed, ev_y.reshape(-1).astype(np.int64), k_cat=2)
            val = float(np.sqrt(np.clip(h_pair_sq[0, 1], 0.0, 1.0)))
            h_sqrt[a, b] = h_sqrt[b, a] = val
            valid_pairs.append((a, b))
            pair_train_outs.append(pair_out)

    post_train_curves = [np.asarray(p["post_train_out"].get("train_losses", []), dtype=np.float64) for p in pair_train_outs]
    post_val_curves = [np.asarray(p["post_train_out"].get("val_losses", []), dtype=np.float64) for p in pair_train_outs]
    post_mon_curves = [np.asarray(p["post_train_out"].get("val_monitor_losses", []), dtype=np.float64) for p in pair_train_outs]
    return {
        "h_sqrt": h_sqrt,
        "h_cat_sq": np.square(h_sqrt),
        "train_out": {
            "train_losses": _mean_padded_curves(post_train_curves),
            "val_losses": _mean_padded_curves(post_val_curves),
            "val_monitor_losses": _mean_padded_curves(post_mon_curves),
        },
        "theta_flow_cate_valid_pairs": np.asarray(valid_pairs, dtype=np.int64).reshape(-1, 2),
        "theta_flow_cate_skipped_pairs": np.asarray(skipped_pairs, dtype=object),
        "theta_flow_cate_train_counts": train_counts,
        "theta_flow_cate_val_counts": val_counts,
        "theta_flow_cate_num_valid_pairs": np.int64(len(valid_pairs)),
        "theta_flow_cate_num_skipped_pairs": np.int64(len(skipped_pairs)),
        "pair_train_outs": pair_train_outs,
    }


def _as_2d(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}.")
    return arr


def _nmse_h_vs_gt(h_sweep: np.ndarray, h_gt_sqrt: np.ndarray) -> np.ndarray:
    h_sw = np.asarray(h_sweep, dtype=np.float64)
    h_gt = np.asarray(h_gt_sqrt, dtype=np.float64)
    if h_sw.ndim != 4:
        raise ValueError(f"h_sweep must be 4D; got {h_sw.shape}.")
    n_m, n_cols = int(h_sw.shape[0]), int(h_sw.shape[1])
    out = np.full((n_m, n_cols), np.nan, dtype=np.float64)
    h_gt_imp = vhb.impute_offdiag_nan_mean(h_gt)
    for i in range(n_m):
        for j in range(n_cols):
            out[i, j] = _matrix_nmse_offdiag(
                vhb.impute_offdiag_nan_mean(np.asarray(h_sw[i, j], dtype=np.float64)),
                h_gt_imp,
            )
    return out


def _nmse_decode_vs_ref(decode_sweep: np.ndarray, decode_ref: np.ndarray) -> np.ndarray:
    d_sw = np.asarray(decode_sweep, dtype=np.float64)
    d_ref = np.asarray(decode_ref, dtype=np.float64)
    if d_sw.ndim != 3:
        raise ValueError(f"decode_sweep must be 3D; got {d_sw.shape}.")
    n_cols = int(d_sw.shape[0])
    out = np.full((n_cols,), np.nan, dtype=np.float64)
    d_ref_imp = vhb.impute_offdiag_nan_mean(d_ref)
    for j in range(n_cols):
        out[j] = _matrix_nmse_offdiag(
            vhb.impute_offdiag_nan_mean(np.asarray(d_sw[j], dtype=np.float64)),
            d_ref_imp,
        )
    return out


def _selected_eval_split(args: argparse.Namespace) -> str:
    split = str(getattr(args, "eval_split", "all")).strip().lower()
    if split not in ("all", "validation"):
        raise ValueError("--eval-split must be 'all' or 'validation'.")
    return split


def _load_cached_results(output_dir: str) -> dict[str, Any]:
    zpath = os.path.join(output_dir, "h_decoding_categorical_twofig_results.npz")
    if not os.path.isfile(zpath):
        raise FileNotFoundError(
            f"Missing {zpath}. Run without --visualization-only first, or pass --output-dir from a prior run."
        )
    with np.load(zpath, allow_pickle=True) as z:
        bundle = {k: z[k] for k in z.files}
    need = (
        "n",
        "n_ref",
        "method_names",
        "num_categories",
        "theta_bin_centers",
        "h_gt_sqrt",
        "decode_ref",
        "h_sqrt_sweep",
        "decode_sweep",
        "corr_h_vs_gt",
        "nmse_h_vs_gt",
        "corr_decode_vs_ref",
        "nmse_decode_vs_ref",
    )
    for k in need:
        if k not in bundle:
            raise KeyError(f"Cached NPZ missing key {k!r}: {zpath}")
    return bundle


def _validate_cached_cli(
    args: argparse.Namespace,
    cached: dict[str, Any],
    methods: list[str],
    ns: list[int],
) -> None:
    n_arr = np.asarray(cached["n"], dtype=np.int64).ravel()
    if n_arr.size != len(ns) or not np.array_equal(n_arr, np.asarray(ns, dtype=np.int64)):
        raise ValueError(
            f"--n-list {ns} does not match cached n={n_arr.tolist()}. "
            "Use the same --n-list as the run that produced the NPZ."
        )
    if int(np.asarray(cached["n_ref"]).reshape(-1)[0]) != int(args.n_ref):
        raise ValueError(
            f"Cached n_ref does not match --n-ref={args.n_ref}."
        )
    if int(np.asarray(cached["num_categories"]).reshape(-1)[0]) != int(args.num_categories):
        raise ValueError("Cached num_categories does not match --num-categories.")
    names = [str(x) for x in np.asarray(cached["method_names"], dtype=object).tolist()]
    if names != methods:
        raise ValueError(f"Cached method_names {names} do not match CLI methods {methods}.")
    if "native_dataset_npz" in cached:
        raw = cached["native_dataset_npz"]
        ds_cached = str(np.asarray(raw, dtype=object).reshape(-1)[0]) if np.size(raw) else None
        if ds_cached:
            want = str(Path(args.dataset_npz).resolve())
            got = str(Path(ds_cached).resolve())
            same = False
            if os.path.isfile(got) and os.path.isfile(want):
                try:
                    same = os.path.samefile(got, want)
                except OSError:
                    same = False
            if not same and got != want:
                raise ValueError(
                    f"--dataset-npz {args.dataset_npz!r} does not match cached native_dataset_npz={ds_cached!r}."
                )
    want_split = _selected_eval_split(args)
    if "eval_split" in cached:
        got_split = str(np.asarray(cached["eval_split"], dtype=object).reshape(-1)[0]).strip().lower()
        if got_split != want_split:
            raise ValueError(
                f"--eval-split {want_split!r} does not match cached eval_split={got_split!r}."
            )
    elif "hellinger_eval_split" in cached:
        legacy_split = str(np.asarray(cached["hellinger_eval_split"], dtype=object).reshape(-1)[0]).strip().lower()
        if want_split != "all" or legacy_split != "all":
            raise ValueError(
                "Cached results NPZ uses legacy hellinger_eval_split metadata and may have old all-pool "
                "decoding semantics. Re-run the study with --eval-split for validation caches."
            )
    elif want_split != "all":
        raise ValueError(
            "Cached results NPZ has no eval_split metadata (older run). "
            "Use --eval-split all with --visualization-only, or re-run the study."
        )


def _run_visualization_only(args: argparse.Namespace, methods: list[str], ns: list[int]) -> None:
    cached = _load_cached_results(args.output_dir)
    _validate_cached_cli(args, cached, methods, ns)
    k_cat = int(np.asarray(cached["num_categories"]).reshape(-1)[0])
    theta_centers = np.asarray(cached["theta_bin_centers"], dtype=np.float64)
    h_gt_sqrt = np.asarray(cached["h_gt_sqrt"], dtype=np.float64)
    decode_ref = np.asarray(cached["decode_ref"], dtype=np.float64)
    h_sw = np.asarray(cached["h_sqrt_sweep"], dtype=np.float64)
    dec_sw = np.asarray(cached["decode_sweep"], dtype=np.float64)
    corr_h = np.asarray(cached["corr_h_vs_gt"], dtype=np.float64)
    corr_d = np.asarray(cached["corr_decode_vs_ref"], dtype=np.float64).ravel()
    nmse_h = np.asarray(cached["nmse_h_vs_gt"], dtype=np.float64)
    nmse_d = np.asarray(cached["nmse_decode_vs_ref"], dtype=np.float64).ravel()
    zdummy = np.zeros_like(corr_h)
    cat_footer = {
        "h_gt_sqrt": h_gt_sqrt,
        "decode_ref": decode_ref,
        "n_ref": int(args.n_ref),
        "decode_sweep_for_decode_limits": dec_sw,
    }
    sweep_svg = _render_method_sweep_panel(
        row_labels=methods,
        h_sweep=h_sw,
        clf_sweep_shared=dec_sw,
        clf_hellinger_sweep_shared=None,
        n_list=ns,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_categorical_twofig_sweep.svg"),
        n_bins=k_cat,
        theta_centers=theta_centers,
        clf_ref_decode_limits=np.asarray(decode_ref, dtype=np.float64),
        category_gt_footer=cat_footer,
    )
    corr_nmse_svg = _render_corr_nmse_two_panel(
        row_labels=methods,
        n_list=ns,
        corr_h=corr_h,
        corr_decode_shared=corr_d,
        corr_decode_hellinger_shared=None,
        corr_hellinger_lb=zdummy,
        corr_hellinger_ub=zdummy,
        show_hellinger_bounds=False,
        nmse_h=nmse_h,
        nmse_decode_shared=nmse_d,
        nmse_decode_hellinger_shared=None,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_categorical_twofig_corr_nmse.svg"),
    )
    loss_root = os.path.join(str(args.output_dir), "training_losses")
    if not os.path.isdir(loss_root):
        raise FileNotFoundError(
            "visualization-only requires per-(method, n) curves under training_losses/ "
            f"(missing {os.path.abspath(loss_root)}). Re-run without --visualization-only first."
        )
    loss_panel_svg = _render_row_n_training_losses_panel(
        row_labels=methods,
        n_list=ns,
        loss_root=loss_root,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_categorical_twofig_training_losses_panel.svg"),
    )
    all_columns_png = _write_all_columns_png(
        args.output_dir,
        sweep_svg=sweep_svg,
        corr_nmse_svg=corr_nmse_svg,
        loss_panel_svg=loss_panel_svg,
    )
    out_npz = os.path.join(args.output_dir, "h_decoding_categorical_twofig_results.npz")
    summary_path = os.path.join(args.output_dir, "h_decoding_categorical_twofig_summary.txt")
    _write_summary(
        summary_path,
        args=args,
        out_npz=os.path.abspath(out_npz),
        sweep_svg=os.path.abspath(sweep_svg),
        corr_nmse_svg=os.path.abspath(corr_nmse_svg),
        visualization_only=True,
        loss_panel_svg=os.path.abspath(loss_panel_svg),
        all_columns_png=os.path.abspath(all_columns_png),
        training_losses_root=os.path.abspath(loss_root),
    )
    print("[cat-twofig] Saved (visualization-only):", flush=True)
    for p in (sweep_svg, corr_nmse_svg, loss_panel_svg, all_columns_png, summary_path):
        print(f"  - {os.path.abspath(p)}", flush=True)


def _write_summary(
    path: str,
    *,
    args: argparse.Namespace,
    out_npz: str,
    sweep_svg: str,
    corr_nmse_svg: str,
    visualization_only: bool,
    loss_panel_svg: str | None = None,
    all_columns_png: str | None = None,
    training_losses_root: str | None = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("study_h_decoding_categorical_twofig\n")
        f.write(f"visualization_only: {bool(visualization_only)}\n")
        f.write(f"output_dir: {os.path.abspath(args.output_dir)}\n")
        f.write(f"dataset_npz: {os.path.abspath(str(args.dataset_npz))}\n")
        f.write(f"num_categories: {int(args.num_categories)}\n")
        f.write(f"n_list: {args.n_list}\n")
        f.write(f"n_ref: {int(args.n_ref)}\n")
        f.write(f"methods: {args.methods}\n")
        f.write(f"pr_project: {bool(args.pr_project)}\n")
        f.write(f"device: {args.device}\n")
        f.write(f"results_npz: {out_npz}\n")
        f.write(f"h_decoding_categorical_twofig_sweep.svg: {sweep_svg}\n")
        f.write(f"h_decoding_categorical_twofig_corr_nmse.svg: {corr_nmse_svg}\n")
        if loss_panel_svg:
            f.write(f"h_decoding_categorical_twofig_training_losses_panel.svg: {loss_panel_svg}\n")
        if all_columns_png:
            f.write(f"{_ALL_COLUMNS_PNG_NAME}: {all_columns_png}\n")
        if training_losses_root:
            f.write(f"training_losses_root: {training_losses_root}\n")
        f.write(f"eval_split: {_selected_eval_split(args)}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-categories", type=int, default=2)
    p.add_argument(
        "--dataset-npz",
        type=Path,
        default=None,
        help="Native 2D NPZ. Default: data/random_mog_categorical_xdim2_kK/random_mog_categorical.npz",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--force-regenerate", action="store_true")
    p.add_argument("--n-total", type=int, default=50000)
    p.add_argument("--n-list", type=str, default="80,200,400,600")
    p.add_argument("--n-ref", type=int, default=10000)
    p.add_argument("--run-seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--methods",
        type=str,
        default=",".join(_DEFAULT_METHODS),
        help=f"Comma-separated methods. Supported: {_SUPPORTED_METHODS_HELP}. "
        "--clf-min-class-count also gates theta_flow_cate pair training.",
    )
    p.add_argument("--visualization-only", action="store_true")
    p.add_argument(
        "--eval-split",
        type=str,
        default="all",
        choices=("all", "validation"),
        help=(
            "For learned H, GT-LLR diagnostics, reference decoding, and sweep decoding: "
            "use all nested rows (default) or validation rows only. Pairwise classifiers still train on "
            "the nested training slice."
        ),
    )
    p.add_argument(
        "--no-scatter-diagnostics",
        action="store_true",
        help="Skip LLR and H^2 scatter figures (llr_est_vs_true_all, hellinger_est_vs_gt_all).",
    )
    p.add_argument("--flow-arch", type=str, default="mlp", choices=["mlp", "film", "film_fourier"])
    p.add_argument("--flow-epochs", type=int, default=10000)
    p.add_argument("--flow-batch-size", type=int, default=256)
    p.add_argument("--flow-lr", type=float, default=1e-3)
    p.add_argument("--flow-hidden-dim", type=int, default=128)
    p.add_argument("--flow-depth", type=int, default=3)
    p.add_argument(
        "--flow-cond-embed-dim",
        type=int,
        default=16,
        help="x_flow + --flow-arch film: per-channel width for theta and time FiLM cond MLPs. Default: 16.",
    )
    p.add_argument(
        "--flow-cond-embed-depth",
        type=int,
        default=1,
        help="x_flow + --flow-arch film: linear layers in each theta/time cond MLP. Default: 1.",
    )
    p.add_argument(
        "--flow-cond-embed-act",
        type=str,
        default="silu",
        choices=["silu", "relu", "tanh"],
        help="x_flow + --flow-arch film: activation between cond MLP layers (not after last). Default: silu.",
    )
    p.add_argument("--flow-scheduler", type=str, default="cosine")
    p.add_argument("--flow-fm-t-eps", type=float, default=0.05)
    p.add_argument("--flow-early-patience", type=int, default=1000)
    p.add_argument("--flow-early-min-delta", type=float, default=1e-4)
    p.add_argument("--flow-early-ema-alpha", type=float, default=0.05)
    p.add_argument("--no-flow-restore-best", dest="flow_restore_best", action="store_false")
    p.set_defaults(flow_restore_best=True)
    p.add_argument("--flow-ode-steps", type=int, default=64)
    p.add_argument("--flow-likelihood-exact-divergence", action="store_true")
    p.add_argument("--flow-likelihood-finetune-epochs", type=int, default=0)
    p.add_argument("--flow-likelihood-finetune-lr", type=float, default=1e-4)
    p.add_argument("--flow-likelihood-finetune-batch-size", type=int, default=0)
    p.add_argument("--flow-likelihood-finetune-ode-steps", type=int, default=64)
    p.add_argument("--flow-likelihood-finetune-patience", type=int, default=100)
    p.add_argument("--flow-likelihood-finetune-min-delta", type=float, default=1e-4)
    p.add_argument("--flow-likelihood-finetune-ema-alpha", type=float, default=0.05)
    p.add_argument("--theta-flow-posterior-only-likelihood", action="store_true", default=False)
    p.add_argument("--prior-epochs", type=int, default=10000)
    p.add_argument("--prior-batch-size", type=int, default=256)
    p.add_argument("--prior-lr", type=float, default=1e-3)
    p.add_argument("--prior-hidden-dim", type=int, default=128)
    p.add_argument("--prior-depth", type=int, default=3)
    p.add_argument("--prior-early-patience", type=int, default=1000)
    p.add_argument("--prior-early-min-delta", type=float, default=1e-4)
    p.add_argument("--prior-early-ema-alpha", type=float, default=0.05)
    p.add_argument("--no-prior-restore-best", dest="prior_restore_best", action="store_false")
    p.set_defaults(prior_restore_best=True)
    p.add_argument("--h-batch-size", type=int, default=65536)
    p.add_argument("--contrastive-epochs", type=int, default=2000)
    p.add_argument("--contrastive-batch-size", type=int, default=256)
    p.add_argument("--contrastive-lr", type=float, default=1e-3)
    p.add_argument("--contrastive-hidden-dim", type=int, default=128)
    p.add_argument("--contrastive-depth", type=int, default=3)
    p.add_argument("--contrastive-weight-decay", type=float, default=0.0)
    p.add_argument("--contrastive-early-patience", type=int, default=300)
    p.add_argument("--contrastive-early-min-delta", type=float, default=1e-4)
    p.add_argument("--contrastive-early-ema-alpha", type=float, default=0.05)
    p.add_argument("--contrastive-max-grad-norm", type=float, default=10.0)
    p.add_argument("--contrastive-pair-batch-size", type=int, default=65536)
    p.add_argument(
        "--contrastive-soft-score-arch",
        type=str,
        default="normalized_dot",
        choices=["normalized_dot", "additive_independent"],
    )
    p.add_argument("--contrastive-soft-dot-dim", type=int, default=10)
    p.add_argument("--contrastive-soft-categorical-beta", type=float, default=0.0)
    p.add_argument("--ctsm-epochs", type=int, default=8000)
    p.add_argument("--ctsm-batch-size", type=int, default=512)
    p.add_argument("--ctsm-lr", type=float, default=2e-3)
    p.add_argument("--ctsm-hidden-dim", type=int, default=256)
    p.add_argument("--ctsm-two-sb-var", type=float, default=2.0)
    p.add_argument("--ctsm-path-schedule", type=str, default="linear", choices=["linear", "cosine"])
    p.add_argument("--ctsm-path-eps", type=float, default=1e-12)
    p.add_argument("--ctsm-factor", type=float, default=1.0)
    p.add_argument("--ctsm-t-eps", type=float, default=1e-5)
    p.add_argument("--ctsm-int-n-time", type=int, default=300)
    p.add_argument("--ctsm-m-scale", type=float, default=1.0)
    p.add_argument("--ctsm-delta-scale", type=float, default=0.5)
    p.add_argument("--ctsm-arch", type=str, default="film", choices=["mlp", "film"])
    p.add_argument("--ctsm-film-depth", type=int, default=3)
    p.add_argument("--ctsm-gated-film", action="store_true", default=False)
    p.add_argument("--ctsm-raw-time", action="store_true", default=False)
    p.add_argument("--ctsm-weight-decay", type=float, default=0.0)
    p.add_argument("--lda-ctsm-shrinkage", type=float, default=1e-3)
    p.add_argument("--lda-ctsm-eps", type=float, default=1e-6)
    p.add_argument(
        "--lda-ctsm-max-dim",
        type=int,
        default=0,
        help="Retained LDA coordinate dimension for lda_ctsm_v; 0 keeps the full x dimension.",
    )
    p.add_argument(
        "--pls-ctsm-max-dim",
        type=int,
        default=0,
        help="Retained PLS component dimension for pls_ctsm_v; 0 uses the largest feasible component count.",
    )
    p.add_argument("--pls-ctsm-eps", type=float, default=1e-6)
    p.add_argument("--pls-ctsm-scale-x", action="store_true", default=False)
    p.add_argument("--pls-ctsm-max-iter", type=int, default=500)
    p.add_argument("--pls-ctsm-tol", type=float, default=1e-6)
    p.add_argument(
        "--pca-ctsm-max-dim",
        type=int,
        default=0,
        help="Retained PCA component dimension for pca_ctsm_v; 0 uses min(x_dim, n_train - 1).",
    )
    p.add_argument("--pca-ctsm-eps", type=float, default=1e-6)
    p.add_argument("--pca-ctsm-scale-x", action="store_true", default=False)
    p.add_argument("--clf-min-class-count", type=int, default=5)
    p.add_argument("--clf-random-state", type=int, default=-1)
    p.add_argument("--clf-max-iter", type=int, default=1000)
    p.add_argument("--lxfs-epochs", type=int, default=2000)
    p.add_argument("--lxfs-batch-size", type=int, default=1024)
    p.add_argument("--lxfs-lr", type=float, default=1e-3)
    p.add_argument("--lxfs-weight-decay", type=float, default=0.0)
    p.add_argument("--lxfs-hidden-dim", type=int, default=128)
    p.add_argument("--lxfs-depth", type=int, default=3)
    p.add_argument("--lxfs-path-schedule", type=str, default="cosine")
    p.add_argument("--lxfs-t-eps", type=float, default=0.05)
    p.add_argument("--lxfs-early-patience", type=int, default=1000)
    p.add_argument("--lxfs-early-min-delta", type=float, default=1e-4)
    p.add_argument("--lxfs-early-ema-alpha", type=float, default=0.05)
    p.add_argument("--lxfs-weight-ema-decay", type=float, default=0.9)
    p.add_argument("--lxfs-max-grad-norm", type=float, default=10.0)
    p.add_argument("--lxfs-quadrature-steps", type=int, default=64)
    p.add_argument("--lxfs-pair-batch-size", type=int, default=65536)
    p.add_argument("--lxfs-solve-jitter", type=float, default=1e-6)
    p.add_argument("--lxf-low-rank-dim", type=int, default=1)
    p.add_argument("--lxf-low-rank-divergence-estimator", type=str, default="hutchinson")
    p.add_argument("--lxf-hutchinson-probes", type=int, default=1)
    p.add_argument("--lxf-nlpca-ode-steps", type=int, default=32)
    p.add_argument("--sir-num-bins", type=int, default=10)
    p.add_argument("--sir-ridge", type=float, default=1e-6)
    p.add_argument("--no-lxf-restore-best", dest="lxf_restore_best", action="store_false")
    p.set_defaults(lxf_restore_best=True)
    p.add_argument("--log-every", type=int, default=50)

    p.add_argument("--pr-project", action="store_true")
    p.add_argument("--pr-dim", type=int, default=10)
    p.add_argument("--pr-output-npz", type=Path, default=None)
    p.add_argument("--pr-use-cache", action="store_true")
    p.add_argument("--pr-cache-dir", type=str, default="data/pr_autoencoder_cache")
    p.add_argument("--pr-train-epochs", type=int, default=None)
    p.add_argument("--pr-train-samples", type=int, default=None)
    p.add_argument("--pr-train-batch-size", type=int, default=None)
    p.add_argument("--pr-train-lr", type=float, default=None)
    p.add_argument("--pr-lambda-pr", type=float, default=None)
    p.add_argument("--pr-eps", type=float, default=None)
    p.add_argument("--pr-hidden1", type=int, default=None)
    p.add_argument("--pr-hidden2", type=int, default=None)
    p.add_argument("--pr-skip-viz", action="store_true")
    p.add_argument(
        "--decode-source-npz",
        type=str,
        default="",
        help=(
            "Deprecated compatibility option. Categorical twofig classifier/decoding paths now use the "
            "working feature space, including PR-embedded features when --pr-project is active."
        ),
    )
    return p


def _save_method_training_loss_npz(out_path: str | Path, *, method_name: str, result: dict[str, Any]) -> None:
    """Write per-(method, n) curves compatible with :func:`conv._load_per_n_training_loss_npz`."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    train_out = result.get("train_out")
    if train_out:
        tr = np.asarray(train_out.get("train_losses", []), dtype=np.float64).ravel()
        va = np.asarray(train_out.get("val_losses", []), dtype=np.float64).ravel()
        em = np.asarray(train_out.get("val_monitor_losses", []), dtype=np.float64).ravel()
    else:
        tr = np.asarray([], dtype=np.float64)
        va = np.asarray([], dtype=np.float64)
        em = np.asarray([], dtype=np.float64)
    extra: dict[str, Any] = {}
    for key in (
        "theta_flow_cate_valid_pairs",
        "theta_flow_cate_skipped_pairs",
        "theta_flow_cate_train_counts",
        "theta_flow_cate_val_counts",
        "theta_flow_cate_num_valid_pairs",
        "theta_flow_cate_num_skipped_pairs",
    ):
        if key in result:
            extra[key] = result[key]
    np.savez_compressed(
        str(p),
        theta_field_method=np.asarray([str(method_name)], dtype=object),
        prior_enable=np.bool_(False),
        score_train_losses=tr,
        score_val_losses=va,
        score_val_monitor_losses=em,
        **extra,
    )


def _train_x_flow_delta(
    args: argparse.Namespace,
    *,
    dev: torch.device,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    theta_all: np.ndarray,
    x_all: np.ndarray,
) -> dict[str, Any]:
    model_args = argparse.Namespace(
        x_dim=int(x_all.shape[1]),
        flow_hidden_dim=int(args.flow_hidden_dim),
        flow_depth=int(args.flow_depth),
        flow_use_layer_norm=False,
        flow_gated_film=False,
        flow_zero_out_init=False,
        flow_cond_embed_dim=int(getattr(args, "flow_cond_embed_dim", 16)),
        flow_cond_embed_depth=int(getattr(args, "flow_cond_embed_depth", 1)),
        flow_cond_embed_act=str(getattr(args, "flow_cond_embed_act", "silu")),
        flow_x_theta_fourier_k=4,
        flow_x_theta_fourier_omega=1.0,
        flow_x_theta_fourier_omega_mode="fixed",
        flow_x_theta_fourier_no_linear=False,
        flow_x_theta_fourier_no_bias=False,
        theta_low=0.0,
        theta_high=1.0,
    )
    print(
        f"[cat-twofig] training x_flow: train={x_train.shape[0]} val={x_val.shape[0]} "
        f"x_dim={x_all.shape[1]} theta_dim={theta_all.shape[1]}",
        flush=True,
    )
    model = build_conditional_x_velocity_model(
        flow_arch=str(args.flow_arch),
        args=model_args,
        device=dev,
        theta_dim=int(theta_all.shape[1]),
    )
    train_out = train_conditional_x_flow_model(
        model=model,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        epochs=int(args.flow_epochs),
        batch_size=int(args.flow_batch_size),
        lr=float(args.flow_lr),
        device=dev,
        log_every=max(1, int(args.log_every)),
        early_stopping_patience=int(args.flow_early_patience),
        early_stopping_min_delta=float(args.flow_early_min_delta),
        early_stopping_ema_alpha=float(args.flow_early_ema_alpha),
        restore_best=bool(args.flow_restore_best),
        scheduler_name=str(args.flow_scheduler),
        fm_t_eps=float(args.flow_fm_t_eps),
    )
    estimator = HMatrixEstimator(
        model_post=model,
        model_prior=None,
        sigma_eval=1.0,
        device=dev,
        pair_batch_size=int(args.h_batch_size),
        field_method="flow_x_likelihood",
        flow_scheduler=str(args.flow_scheduler),
        flow_ode_steps=int(args.flow_ode_steps),
        flow_likelihood_exact_divergence=bool(args.flow_likelihood_exact_divergence),
    )
    c_matrix = estimator.compute_x_conditional_loglik_matrix(theta_all, x_all)
    return {"c_matrix": c_matrix, "delta_l": HMatrixEstimator.compute_delta_l(c_matrix), "train_out": train_out}


def _train_binary_classifier_delta(
    args: argparse.Namespace,
    *,
    x_train: np.ndarray,
    bins_train: np.ndarray,
    x_all: np.ndarray,
    bins_all: np.ndarray,
    k_cat: int,
) -> dict[str, Any]:
    print("[cat-twofig] training binary_classifier pairwise logistic LLRs", flush=True)
    bins_train = np.asarray(bins_train, dtype=np.int64).reshape(-1)
    bins_all = np.asarray(bins_all, dtype=np.int64).reshape(-1)
    n = int(x_all.shape[0])
    delta = np.zeros((n, n), dtype=np.float64)
    valid_pairs = np.zeros((k_cat, k_cat), dtype=bool)
    stats = {"ok_pairs": 0, "insufficient_counts": 0, "fit_fail": 0}
    rs = int(args.clf_random_state)
    if rs < 0:
        rs = int(args.run_seed)
    for a in range(int(k_cat)):
        for b in range(a + 1, int(k_cat)):
            ia = np.flatnonzero(bins_train == a)
            ib = np.flatnonzero(bins_train == b)
            if ia.size < int(args.clf_min_class_count) or ib.size < int(args.clf_min_class_count):
                stats["insufficient_counts"] += 1
                continue
            x_pair = np.vstack([x_train[ia], x_train[ib]])
            y_pair = np.concatenate([np.zeros(ia.size, dtype=np.int64), np.ones(ib.size, dtype=np.int64)])
            try:
                clf = LogisticRegression(solver="lbfgs", random_state=rs, max_iter=int(args.clf_max_iter))
                clf.fit(x_pair, y_pair)
                prior_log_odds = float(np.log(float(ib.size) / float(ia.size)))
                llr_b_minus_a = np.asarray(clf.decision_function(x_all), dtype=np.float64) - prior_log_odds
            except Exception:
                stats["fit_fail"] += 1
                continue
            row_a = np.flatnonzero(bins_all == a)
            row_b = np.flatnonzero(bins_all == b)
            if row_a.size and row_b.size:
                delta[np.ix_(row_a, row_b)] = llr_b_minus_a[row_a].reshape(-1, 1)
                delta[np.ix_(row_b, row_a)] = -llr_b_minus_a[row_b].reshape(-1, 1)
            valid_pairs[a, b] = True
            valid_pairs[b, a] = True
            stats["ok_pairs"] += 1
    return {"c_matrix": None, "delta_l": delta, "classifier_valid_pairs": valid_pairs, "classifier_stats": stats}


def _train_linear_x_flow_delta(
    args: argparse.Namespace,
    *,
    method_name: str,
    dev: torch.device,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    theta_all: np.ndarray,
    x_all: np.ndarray,
) -> dict[str, Any]:
    print(f"[cat-twofig] training {method_name}", flush=True)
    rank = int(args.lxf_low_rank_dim)
    common = dict(
        theta_dim=int(theta_all.shape[1]),
        x_dim=int(x_all.shape[1]),
        hidden_dim=int(args.lxfs_hidden_dim),
        depth=int(args.lxfs_depth),
    )
    sir_meta: dict[str, Any] = {}
    if method_name == "linear_x_flow_t":
        model = ConditionalTimeLinearXFlowMLP(
            **common,
            quadrature_steps=int(args.lxfs_quadrature_steps),
        ).to(dev)
        ode_likelihood = False
    elif method_name == "xflow_sir_lrank":
        if rank > int(x_all.shape[1]):
            raise ValueError(f"--lxf-low-rank-dim must be <= x_dim={x_all.shape[1]}; got {rank}.")
        _, _, _, fitted = _fit_sir_projection(
            x_train=x_train,
            theta_train=theta_train,
            x_val=x_val,
            x_all=x_all,
            sir_dim=rank,
            num_bins=int(args.sir_num_bins),
            ridge=float(args.sir_ridge),
        )
        sir_meta = dict(fitted)
        model = ConditionalTimeLowRankCorrectionLinearXFlowMLP(
            **common,
            correction_rank=rank,
            quadrature_steps=int(args.lxfs_quadrature_steps),
            divergence_estimator=str(args.lxf_low_rank_divergence_estimator).strip().lower(),
            hutchinson_probes=int(args.lxf_hutchinson_probes),
            fixed_u=np.asarray(sir_meta["sir_components"], dtype=np.float64),
        ).to(dev)
        ode_likelihood = True
    else:
        raise ValueError(f"Unsupported linear x-flow method {method_name!r}.")

    train_out = train_time_linear_x_flow_schedule(
        model=model,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        device=dev,
        schedule=path_schedule_from_name(str(args.lxfs_path_schedule)),
        epochs=int(args.lxfs_epochs),
        batch_size=int(args.lxfs_batch_size),
        lr=float(args.lxfs_lr),
        weight_decay=float(args.lxfs_weight_decay),
        t_eps=float(args.lxfs_t_eps),
        patience=int(args.lxfs_early_patience),
        min_delta=float(args.lxfs_early_min_delta),
        ema_alpha=float(args.lxfs_early_ema_alpha),
        weight_ema_decay=float(args.lxfs_weight_ema_decay),
        max_grad_norm=float(args.lxfs_max_grad_norm),
        log_every=max(1, int(args.log_every)),
        restore_best=bool(args.lxf_restore_best),
        log_name=method_name,
    )
    x_mean = np.asarray(train_out["x_mean"], dtype=np.float64)
    x_std = np.asarray(train_out["x_std"], dtype=np.float64)
    if ode_likelihood:
        c_matrix = compute_ode_time_linear_x_flow_c_matrix(
            model=model,
            theta_all=theta_all,
            x_all=x_all,
            device=dev,
            x_mean=x_mean,
            x_std=x_std,
            solve_jitter=float(args.lxfs_solve_jitter),
            quadrature_steps=int(args.lxfs_quadrature_steps),
            ode_steps=int(args.lxf_nlpca_ode_steps),
            pair_batch_size=int(args.lxfs_pair_batch_size),
        )
    else:
        c_matrix = compute_time_linear_x_flow_c_matrix(
            model=model,
            theta_all=theta_all,
            x_all=x_all,
            device=dev,
            x_mean=x_mean,
            x_std=x_std,
            solve_jitter=float(args.lxfs_solve_jitter),
            quadrature_steps=int(args.lxfs_quadrature_steps),
            pair_batch_size=int(args.lxfs_pair_batch_size),
        )
    return {
        "c_matrix": c_matrix,
        "delta_l": HMatrixEstimator.compute_delta_l(c_matrix),
        "train_out": train_out,
        "sir_meta": sir_meta,
    }


def _build_ctsm_v_model(args: argparse.Namespace, *, x_dim: int, theta_dim: int, dev: torch.device) -> torch.nn.Module:
    arch = str(getattr(args, "ctsm_arch", "film")).strip().lower()
    if arch == "film":
        return ToyPairConditionedTimeScoreNetFiLM(
            dim=int(x_dim),
            hidden_dim=int(getattr(args, "ctsm_hidden_dim", 256)),
            depth=int(getattr(args, "ctsm_film_depth", 3)),
            theta_dim=int(theta_dim),
            m_scale=float(getattr(args, "ctsm_m_scale", 1.0)),
            delta_scale=float(getattr(args, "ctsm_delta_scale", 0.5)),
            use_logit_time=not bool(getattr(args, "ctsm_raw_time", False)),
            gated_film=bool(getattr(args, "ctsm_gated_film", False)),
        ).to(dev)
    if arch == "mlp":
        return ToyPairConditionedTimeScoreNet(
            dim=int(x_dim),
            hidden_dim=int(getattr(args, "ctsm_hidden_dim", 256)),
            theta_dim=int(theta_dim),
            m_scale=float(getattr(args, "ctsm_m_scale", 1.0)),
            delta_scale=float(getattr(args, "ctsm_delta_scale", 0.5)),
        ).to(dev)
    raise ValueError("--ctsm-arch must be one of {'mlp','film'}.")


def _train_ctsm_v_delta(
    args: argparse.Namespace,
    *,
    dev: torch.device,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    theta_all: np.ndarray,
    x_all: np.ndarray,
) -> dict[str, Any]:
    theta_train = _as_2d(theta_train)
    theta_val = _as_2d(theta_val)
    theta_all = _as_2d(theta_all)
    theta_dim = int(theta_all.shape[1])
    if theta_train.shape[1] != theta_dim or theta_val.shape[1] != theta_dim:
        raise ValueError("ctsm_v categorical theta dimensions must match across train/validation/eval.")
    print(
        f"[cat-twofig] training ctsm_v: train={x_train.shape[0]} val={x_val.shape[0]} "
        f"eval={x_all.shape[0]} x_dim={x_all.shape[1]} theta_dim={theta_dim} "
        f"arch={getattr(args, 'ctsm_arch', 'film')}",
        flush=True,
    )
    model = _build_ctsm_v_model(args, x_dim=int(x_all.shape[1]), theta_dim=theta_dim, dev=dev)
    train_out = train_pair_conditioned_ctsm_v_model(
        model=model,
        theta_train=theta_train,
        x_train=x_train,
        epochs=int(getattr(args, "ctsm_epochs", 8000)),
        batch_size=int(getattr(args, "ctsm_batch_size", 512)),
        lr=float(getattr(args, "ctsm_lr", 2e-3)),
        weight_decay=float(getattr(args, "ctsm_weight_decay", 0.0)),
        device=dev,
        log_every=max(1, int(getattr(args, "log_every", 50))),
        two_sb_var=float(getattr(args, "ctsm_two_sb_var", 2.0)),
        path_schedule=str(getattr(args, "ctsm_path_schedule", "linear")),
        path_eps=float(getattr(args, "ctsm_path_eps", 1e-12)),
        factor=float(getattr(args, "ctsm_factor", 1.0)),
        t_eps=float(getattr(args, "ctsm_t_eps", 1e-5)),
        theta_val=theta_val,
        x_val=x_val,
        early_stopping_patience=int(getattr(args, "flow_early_patience", 1000)),
        early_stopping_min_delta=float(getattr(args, "flow_early_min_delta", 1e-4)),
        early_stopping_ema_alpha=float(getattr(args, "flow_early_ema_alpha", 0.05)),
        restore_best=bool(getattr(args, "flow_restore_best", True)),
    )
    estimator = HMatrixEstimator(
        model_post=model,
        model_prior=None,
        sigma_eval=float(getattr(args, "ctsm_t_eps", 1e-5)),
        device=dev,
        pair_batch_size=int(getattr(args, "h_batch_size", 65536)),
        field_method="ctsm_v",
        ctsm_int_n_time=int(getattr(args, "ctsm_int_n_time", 300)),
        ctsm_t_eps=float(getattr(args, "ctsm_t_eps", 1e-5)),
    )
    h_res = estimator.run(theta_all, x_all, restore_original_order=True)
    return {
        "c_matrix": h_res.c_matrix,
        "delta_l": h_res.delta_l_matrix,
        "h_directed": h_res.h_directed,
        "h_sym": h_res.h_sym,
        "train_out": train_out,
        "ctsm_theta_encoding": np.asarray(["one_hot"], dtype=object),
    }


def _fit_lda_weighted_projection(
    *,
    x_train: np.ndarray,
    bins_train: np.ndarray,
    x_val: np.ndarray,
    x_all: np.ndarray,
    shrinkage: float,
    eps: float,
    max_dim: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit train-only LDA coordinates and apply q_j = sqrt(w_j) v_j^T x."""
    x_tr = np.asarray(x_train, dtype=np.float64)
    x_va = np.asarray(x_val, dtype=np.float64)
    x_ev = np.asarray(x_all, dtype=np.float64)
    y = np.asarray(bins_train, dtype=np.int64).reshape(-1)
    if x_tr.ndim != 2 or x_va.ndim != 2 or x_ev.ndim != 2:
        raise ValueError("lda_ctsm_v expects x_train, x_val, and x_all to be 2D.")
    if x_tr.shape[0] != y.size:
        raise ValueError("lda_ctsm_v bins_train length must match x_train rows.")
    if x_tr.shape[0] < 2:
        raise ValueError("lda_ctsm_v requires at least two training samples.")
    if x_tr.shape[1] != x_va.shape[1] or x_tr.shape[1] != x_ev.shape[1]:
        raise ValueError("lda_ctsm_v x dimension mismatch across train/validation/eval.")
    if not np.all(np.isfinite(x_tr)) or not np.all(np.isfinite(x_va)) or not np.all(np.isfinite(x_ev)):
        raise ValueError("lda_ctsm_v input features contain non-finite values.")
    if not np.isfinite(float(shrinkage)) or not (0.0 <= float(shrinkage) <= 1.0):
        raise ValueError("--lda-ctsm-shrinkage must be finite and in [0, 1].")
    if not np.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("--lda-ctsm-eps must be finite and > 0.")

    x_dim = int(x_tr.shape[1])
    m = x_dim if int(max_dim) <= 0 else int(max_dim)
    if m < 1 or m > x_dim:
        raise ValueError(f"--lda-ctsm-max-dim must be 0 or in [1, x_dim={x_dim}]; got {max_dim}.")

    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        raise ValueError("lda_ctsm_v requires at least two non-empty training classes.")
    n = float(x_tr.shape[0])
    global_mean = np.mean(x_tr, axis=0, dtype=np.float64)
    sw = np.zeros((x_dim, x_dim), dtype=np.float64)
    sb = np.zeros((x_dim, x_dim), dtype=np.float64)
    class_means = np.zeros((classes.size, x_dim), dtype=np.float64)
    for i, cls in enumerate(classes):
        xk = x_tr[y == int(cls)]
        class_means[i] = np.mean(xk, axis=0, dtype=np.float64)
        centered = xk - class_means[i].reshape(1, -1)
        sw += (centered.T @ centered) / n
        diff = (class_means[i] - global_mean).reshape(-1, 1)
        sb += (float(xk.shape[0]) / n) * (diff @ diff.T)

    tau = float(np.trace(sw) / max(1, x_dim))
    if not np.isfinite(tau) or tau <= 0.0:
        tau = 1.0
    gamma = float(shrinkage)
    sw_reg = (1.0 - gamma) * sw + gamma * tau * np.eye(x_dim, dtype=np.float64)
    sw_reg = 0.5 * (sw_reg + sw_reg.T)
    sb = 0.5 * (sb + sb.T)
    jitter = max(float(eps), 1e-12) * tau
    chol = None
    for attempt in range(6):
        try:
            chol = np.linalg.cholesky(sw_reg + (10.0**attempt) * jitter * np.eye(x_dim, dtype=np.float64))
            break
        except np.linalg.LinAlgError:
            continue
    if chol is None:
        raise ValueError("lda_ctsm_v could not Cholesky-factor the shrinkage within-class scatter.")

    whitened = np.linalg.solve(chol, sb)
    whitened = np.linalg.solve(chol, whitened.T).T
    whitened = 0.5 * (whitened + whitened.T)
    eigvals, eigvecs = np.linalg.eigh(whitened)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.asarray(eigvals[order], dtype=np.float64)
    eigvecs = np.asarray(eigvecs[:, order], dtype=np.float64)
    components_full = np.linalg.solve(chol.T, eigvecs).astype(np.float64, copy=False)
    components = components_full[:, :m]
    eigvals_m = eigvals[:m]
    lam_tilde = np.maximum(eigvals_m, 0.0) + float(eps)
    weights = float(m) * lam_tilde / float(np.sum(lam_tilde))
    scale = np.sqrt(weights).reshape(1, -1)

    z_train = (x_tr @ components) * scale
    z_val = (x_va @ components) * scale
    z_all = (x_ev @ components) * scale
    meta: dict[str, Any] = {
        "lda_ctsm_dim": np.int64(m),
        "lda_ctsm_original_dim": np.int64(x_dim),
        "lda_ctsm_shrinkage": np.float64(gamma),
        "lda_ctsm_eps": np.float64(float(eps)),
        "lda_ctsm_classes": np.asarray(classes, dtype=np.int64),
        "lda_ctsm_class_counts": np.asarray(counts, dtype=np.int64),
        "lda_ctsm_global_mean": global_mean.astype(np.float64, copy=False),
        "lda_ctsm_class_means": class_means.astype(np.float64, copy=False),
        "lda_ctsm_sw_trace": np.float64(float(np.trace(sw))),
        "lda_ctsm_tau": np.float64(tau),
        "lda_ctsm_eigenvalues": eigvals_m.astype(np.float64, copy=False),
        "lda_ctsm_weights": weights.astype(np.float64, copy=False),
        "lda_ctsm_components": components.astype(np.float64, copy=False),
        "lda_ctsm_transform": (components * np.sqrt(weights).reshape(1, -1)).T.astype(np.float64, copy=False),
    }
    return z_train, z_val, z_all, meta


def _train_lda_ctsm_v_delta(
    args: argparse.Namespace,
    *,
    dev: torch.device,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    bins_train: np.ndarray,
) -> dict[str, Any]:
    z_train, z_val, z_all, lda_meta = _fit_lda_weighted_projection(
        x_train=x_train,
        bins_train=bins_train,
        x_val=x_val,
        x_all=x_all,
        shrinkage=float(getattr(args, "lda_ctsm_shrinkage", 1e-3)),
        eps=float(getattr(args, "lda_ctsm_eps", 1e-6)),
        max_dim=int(getattr(args, "lda_ctsm_max_dim", 0)),
    )
    print(
        f"[cat-twofig] training lda_ctsm_v: x_dim={np.asarray(x_train).shape[1]} "
        f"lda_dim={z_train.shape[1]} shrinkage={float(getattr(args, 'lda_ctsm_shrinkage', 1e-3)):.6g} "
        f"eps={float(getattr(args, 'lda_ctsm_eps', 1e-6)):.6g}",
        flush=True,
    )
    result = _train_ctsm_v_delta(
        args,
        dev=dev,
        theta_train=theta_train,
        x_train=z_train,
        theta_val=theta_val,
        x_val=z_val,
        theta_all=theta_all,
        x_all=z_all,
    )
    result.update(lda_meta)
    result["ctsm_theta_encoding"] = np.asarray(["one_hot_lda_weighted_x"], dtype=object)
    return result


def _fit_pls_weighted_projection(
    *,
    x_train: np.ndarray,
    bins_train: np.ndarray,
    x_val: np.ndarray,
    x_all: np.ndarray,
    eps: float,
    max_dim: int = 0,
    scale_x: bool = False,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit train-only PLS-DA coordinates and apply q_j = sqrt(w_j) u_j^T (x - xbar)."""
    x_tr = np.asarray(x_train, dtype=np.float64)
    x_va = np.asarray(x_val, dtype=np.float64)
    x_ev = np.asarray(x_all, dtype=np.float64)
    y = np.asarray(bins_train, dtype=np.int64).reshape(-1)
    if x_tr.ndim != 2 or x_va.ndim != 2 or x_ev.ndim != 2:
        raise ValueError("pls_ctsm_v expects x_train, x_val, and x_all to be 2D.")
    if x_tr.shape[0] != y.size:
        raise ValueError("pls_ctsm_v bins_train length must match x_train rows.")
    if x_tr.shape[0] < 2:
        raise ValueError("pls_ctsm_v requires at least two training samples.")
    if x_tr.shape[1] != x_va.shape[1] or x_tr.shape[1] != x_ev.shape[1]:
        raise ValueError("pls_ctsm_v x dimension mismatch across train/validation/eval.")
    if not np.all(np.isfinite(x_tr)) or not np.all(np.isfinite(x_va)) or not np.all(np.isfinite(x_ev)):
        raise ValueError("pls_ctsm_v input features contain non-finite values.")
    if not np.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("--pls-ctsm-eps must be finite and > 0.")
    if int(max_iter) < 1:
        raise ValueError("--pls-ctsm-max-iter must be >= 1.")
    if not np.isfinite(float(tol)) or float(tol) <= 0.0:
        raise ValueError("--pls-ctsm-tol must be finite and > 0.")

    x_dim = int(x_tr.shape[1])
    classes, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    if classes.size < 2:
        raise ValueError("pls_ctsm_v requires at least two non-empty training classes.")
    feasible_dim = min(x_dim, int(x_tr.shape[0]) - 1)
    p = feasible_dim if int(max_dim) <= 0 else int(max_dim)
    if p < 1 or p > feasible_dim:
        raise ValueError(
            f"--pls-ctsm-max-dim must be 0 or in [1, feasible_dim={feasible_dim}]; got {max_dim}."
        )

    y_onehot = np.eye(int(classes.size), dtype=np.float64)[inverse]
    x_mean = np.mean(x_tr, axis=0, dtype=np.float64)
    x_scale = np.ones(x_dim, dtype=np.float64)
    if bool(scale_x):
        x_scale = np.std(x_tr, axis=0, dtype=np.float64)
        x_scale = np.where(np.isfinite(x_scale) & (x_scale > 0.0), x_scale, 1.0)

    def _prep(x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - x_mean.reshape(1, -1)) / x_scale.reshape(1, -1)

    x_tr_pre = _prep(x_tr)
    x_va_pre = _prep(x_va)
    x_ev_pre = _prep(x_ev)
    y_mean = np.mean(y_onehot, axis=0, dtype=np.float64)
    y_centered = y_onehot - y_mean.reshape(1, -1)

    pls = PLSRegression(
        n_components=p,
        scale=False,
        max_iter=int(max_iter),
        tol=float(tol),
        copy=True,
    )
    pls.fit(x_tr_pre, y_centered)
    rotations = np.asarray(pls.x_rotations_[:, :p], dtype=np.float64)
    scores = x_tr_pre @ rotations
    denom = max(1, int(x_tr_pre.shape[0]) - 1)
    cov_zy = (scores.T @ y_centered) / float(denom)
    importance = np.sum(np.square(cov_zy), axis=1)
    importance_tilde = np.maximum(importance, 0.0) + float(eps)
    weights = float(p) * importance_tilde / float(np.sum(importance_tilde))
    scale = np.sqrt(weights).reshape(1, -1)

    z_train = scores * scale
    z_val = (x_va_pre @ rotations) * scale
    z_all = (x_ev_pre @ rotations) * scale
    transform = (rotations / x_scale.reshape(-1, 1)) * np.sqrt(weights).reshape(1, -1)
    meta: dict[str, Any] = {
        "pls_ctsm_dim": np.int64(p),
        "pls_ctsm_original_dim": np.int64(x_dim),
        "pls_ctsm_eps": np.float64(float(eps)),
        "pls_ctsm_scale_x": np.bool_(bool(scale_x)),
        "pls_ctsm_max_iter": np.int64(int(max_iter)),
        "pls_ctsm_tol": np.float64(float(tol)),
        "pls_ctsm_classes": np.asarray(classes, dtype=np.int64),
        "pls_ctsm_class_counts": np.asarray(counts, dtype=np.int64),
        "pls_ctsm_x_mean": x_mean.astype(np.float64, copy=False),
        "pls_ctsm_x_scale": x_scale.astype(np.float64, copy=False),
        "pls_ctsm_y_mean": y_mean.astype(np.float64, copy=False),
        "pls_ctsm_rotations": rotations.astype(np.float64, copy=False),
        "pls_ctsm_covariance_scores": importance.astype(np.float64, copy=False),
        "pls_ctsm_weights": weights.astype(np.float64, copy=False),
        "pls_ctsm_transform": transform.T.astype(np.float64, copy=False),
    }
    return z_train, z_val, z_all, meta


def _train_pls_ctsm_v_delta(
    args: argparse.Namespace,
    *,
    dev: torch.device,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    bins_train: np.ndarray,
) -> dict[str, Any]:
    z_train, z_val, z_all, pls_meta = _fit_pls_weighted_projection(
        x_train=x_train,
        bins_train=bins_train,
        x_val=x_val,
        x_all=x_all,
        eps=float(getattr(args, "pls_ctsm_eps", 1e-6)),
        max_dim=int(getattr(args, "pls_ctsm_max_dim", 0)),
        scale_x=bool(getattr(args, "pls_ctsm_scale_x", False)),
        max_iter=int(getattr(args, "pls_ctsm_max_iter", 500)),
        tol=float(getattr(args, "pls_ctsm_tol", 1e-6)),
    )
    print(
        f"[cat-twofig] training pls_ctsm_v: x_dim={np.asarray(x_train).shape[1]} "
        f"pls_dim={z_train.shape[1]} scale_x={bool(getattr(args, 'pls_ctsm_scale_x', False))} "
        f"eps={float(getattr(args, 'pls_ctsm_eps', 1e-6)):.6g}",
        flush=True,
    )
    result = _train_ctsm_v_delta(
        args,
        dev=dev,
        theta_train=theta_train,
        x_train=z_train,
        theta_val=theta_val,
        x_val=z_val,
        theta_all=theta_all,
        x_all=z_all,
    )
    result.update(pls_meta)
    result["ctsm_theta_encoding"] = np.asarray(["one_hot_pls_weighted_x"], dtype=object)
    return result


def _fit_pca_weighted_projection(
    *,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_all: np.ndarray,
    eps: float,
    max_dim: int = 0,
    scale_x: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit train-only PCA coordinates and apply q_j = sqrt(w_j) u_j^T (x - xbar)."""
    x_tr = np.asarray(x_train, dtype=np.float64)
    x_va = np.asarray(x_val, dtype=np.float64)
    x_ev = np.asarray(x_all, dtype=np.float64)
    if x_tr.ndim != 2 or x_va.ndim != 2 or x_ev.ndim != 2:
        raise ValueError("pca_ctsm_v expects x_train, x_val, and x_all to be 2D.")
    if x_tr.shape[0] < 2:
        raise ValueError("pca_ctsm_v requires at least two training samples.")
    if x_tr.shape[1] != x_va.shape[1] or x_tr.shape[1] != x_ev.shape[1]:
        raise ValueError("pca_ctsm_v x dimension mismatch across train/validation/eval.")
    if not np.all(np.isfinite(x_tr)) or not np.all(np.isfinite(x_va)) or not np.all(np.isfinite(x_ev)):
        raise ValueError("pca_ctsm_v input features contain non-finite values.")
    if not np.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("--pca-ctsm-eps must be finite and > 0.")

    x_dim = int(x_tr.shape[1])
    n_train = int(x_tr.shape[0])
    feasible_dim = min(x_dim, n_train - 1)
    p = feasible_dim if int(max_dim) <= 0 else int(max_dim)
    if p < 1 or p > feasible_dim:
        raise ValueError(
            f"--pca-ctsm-max-dim must be 0 or in [1, feasible_dim={feasible_dim}]; got {max_dim}."
        )

    x_mean = np.mean(x_tr, axis=0, dtype=np.float64)
    x_scale = np.ones(x_dim, dtype=np.float64)
    if bool(scale_x):
        x_scale = np.std(x_tr, axis=0, dtype=np.float64)
        x_scale = np.where(np.isfinite(x_scale) & (x_scale > 0.0), x_scale, 1.0)

    def _prep(x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - x_mean.reshape(1, -1)) / x_scale.reshape(1, -1)

    x_tr_pre = _prep(x_tr)
    x_va_pre = _prep(x_va)
    x_ev_pre = _prep(x_ev)
    _, singular_values, vt = np.linalg.svd(x_tr_pre, full_matrices=False)
    components = np.asarray(vt[:p].T, dtype=np.float64)
    denom = max(1, n_train - 1)
    explained_full = np.square(np.asarray(singular_values, dtype=np.float64)) / float(denom)
    explained = explained_full[:p]
    importance = np.maximum(explained, 0.0) + float(eps)
    weights = float(p) * importance / float(np.sum(importance))
    scale = np.sqrt(weights).reshape(1, -1)

    z_train = (x_tr_pre @ components) * scale
    z_val = (x_va_pre @ components) * scale
    z_all = (x_ev_pre @ components) * scale
    transform = (components / x_scale.reshape(-1, 1)) * np.sqrt(weights).reshape(1, -1)
    meta: dict[str, Any] = {
        "pca_ctsm_dim": np.int64(p),
        "pca_ctsm_original_dim": np.int64(x_dim),
        "pca_ctsm_eps": np.float64(float(eps)),
        "pca_ctsm_scale_x": np.bool_(bool(scale_x)),
        "pca_ctsm_x_mean": x_mean.astype(np.float64, copy=False),
        "pca_ctsm_x_scale": x_scale.astype(np.float64, copy=False),
        "pca_ctsm_components": components.astype(np.float64, copy=False),
        "pca_ctsm_explained_variance": explained.astype(np.float64, copy=False),
        "pca_ctsm_weights": weights.astype(np.float64, copy=False),
        "pca_ctsm_transform": transform.T.astype(np.float64, copy=False),
    }
    return z_train, z_val, z_all, meta


def _train_pca_ctsm_v_delta(
    args: argparse.Namespace,
    *,
    dev: torch.device,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    theta_all: np.ndarray,
    x_all: np.ndarray,
) -> dict[str, Any]:
    z_train, z_val, z_all, pca_meta = _fit_pca_weighted_projection(
        x_train=x_train,
        x_val=x_val,
        x_all=x_all,
        eps=float(getattr(args, "pca_ctsm_eps", 1e-6)),
        max_dim=int(getattr(args, "pca_ctsm_max_dim", 0)),
        scale_x=bool(getattr(args, "pca_ctsm_scale_x", False)),
    )
    print(
        f"[cat-twofig] training pca_ctsm_v: x_dim={np.asarray(x_train).shape[1]} "
        f"pca_dim={z_train.shape[1]} scale_x={bool(getattr(args, 'pca_ctsm_scale_x', False))} "
        f"eps={float(getattr(args, 'pca_ctsm_eps', 1e-6)):.6g}",
        flush=True,
    )
    result = _train_ctsm_v_delta(
        args,
        dev=dev,
        theta_train=theta_train,
        x_train=z_train,
        theta_val=theta_val,
        x_val=z_val,
        theta_all=theta_all,
        x_all=z_all,
    )
    result.update(pca_meta)
    result["ctsm_theta_encoding"] = np.asarray(["one_hot_pca_weighted_x"], dtype=object)
    return result


def _train_contrastive_soft_categorical_delta(
    args: argparse.Namespace,
    *,
    dev: torch.device,
    x_train: np.ndarray,
    bins_train: np.ndarray,
    x_val: np.ndarray,
    bins_val: np.ndarray,
    x_all: np.ndarray,
    bins_all: np.ndarray,
    k_cat: int,
) -> dict[str, Any]:
    x_train = np.asarray(x_train, dtype=np.float64)
    x_val = np.asarray(x_val, dtype=np.float64)
    x_all = np.asarray(x_all, dtype=np.float64)
    bins_train = np.asarray(bins_train, dtype=np.int64).reshape(-1)
    bins_val = np.asarray(bins_val, dtype=np.int64).reshape(-1)
    bins_all = np.asarray(bins_all, dtype=np.int64).reshape(-1)
    hidden_dim = int(getattr(args, "contrastive_hidden_dim", 128))
    depth = int(getattr(args, "contrastive_depth", 3))
    soft_arch = str(getattr(args, "contrastive_soft_score_arch", "normalized_dot")).strip().lower().replace("-", "_")
    dot_dim = int(getattr(args, "contrastive_soft_dot_dim", 10))
    if soft_arch == "normalized_dot":
        model = ContrastiveNormalizedDotScorer(
            x_dim=int(x_all.shape[1]),
            theta_dim=int(k_cat),
            feature_dim=dot_dim,
            hidden_dim=hidden_dim,
            depth=depth,
        ).to(dev)
    elif soft_arch == "additive_independent":
        model = ContrastiveAdditiveIndependentScorer(
            x_dim=int(x_all.shape[1]),
            theta_dim=int(k_cat),
            feature_dim=dot_dim,
            hidden_dim=hidden_dim,
            depth=depth,
        ).to(dev)
    else:
        raise ValueError("--contrastive-soft-score-arch must be one of {'normalized_dot','additive_independent'}.")
    train_out = train_contrastive_soft_categorical_llr(
        model=model,
        x_train=x_train,
        y_train=bins_train,
        x_val=x_val,
        y_val=bins_val,
        n_classes=int(k_cat),
        device=dev,
        epochs=int(getattr(args, "contrastive_epochs", 2000)),
        batch_size=int(getattr(args, "contrastive_batch_size", 256)),
        lr=float(getattr(args, "contrastive_lr", 1e-3)),
        beta=float(getattr(args, "contrastive_soft_categorical_beta", 0.0)),
        weight_decay=float(getattr(args, "contrastive_weight_decay", 0.0)),
        patience=int(getattr(args, "contrastive_early_patience", 300)),
        min_delta=float(getattr(args, "contrastive_early_min_delta", 1e-4)),
        ema_alpha=float(getattr(args, "contrastive_early_ema_alpha", 0.05)),
        max_grad_norm=float(getattr(args, "contrastive_max_grad_norm", 10.0)),
        log_every=max(1, int(getattr(args, "log_every", 50))),
        restore_best=True,
    )
    c_matrix = compute_contrastive_soft_categorical_c_matrix(
        model=model,
        x_all=x_all,
        y_all=bins_all,
        n_classes=int(k_cat),
        device=dev,
        x_mean=np.asarray(train_out["x_mean"], dtype=np.float64),
        x_std=np.asarray(train_out["x_std"], dtype=np.float64),
        pair_batch_size=int(getattr(args, "contrastive_pair_batch_size", 65536)),
    )
    return {
        "c_matrix": c_matrix,
        "delta_l": HMatrixEstimator.compute_delta_l(c_matrix),
        "train_out": train_out,
        "contrastive_soft_categorical_beta": np.float64(train_out["beta"]),
        "contrastive_class_priors": np.asarray(train_out["class_priors"], dtype=np.float64),
    }


def _train_one_method(
    args: argparse.Namespace,
    *,
    dev: torch.device,
    method_name: str,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    bins_train: np.ndarray,
    bins_val: np.ndarray,
    bins_all: np.ndarray,
    k_cat: int,
) -> dict[str, Any]:
    if method_name == "x_flow":
        return _train_x_flow_delta(
            args,
            dev=dev,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            theta_all=theta_all,
            x_all=x_all,
        )
    if method_name == "binary_classifier":
        return _train_binary_classifier_delta(
            args,
            x_train=x_train,
            bins_train=bins_train,
            x_all=x_all,
            bins_all=bins_all,
            k_cat=k_cat,
        )
    if method_name == "contrastive_soft_categorical":
        return _train_contrastive_soft_categorical_delta(
            args,
            dev=dev,
            x_train=x_train,
            bins_train=bins_train,
            x_val=x_val,
            bins_val=bins_val,
            x_all=x_all,
            bins_all=bins_all,
            k_cat=k_cat,
        )
    if method_name in ("linear_x_flow_t", "xflow_sir_lrank"):
        return _train_linear_x_flow_delta(
            args,
            method_name=method_name,
            dev=dev,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            theta_all=theta_all,
            x_all=x_all,
        )
    if method_name == "theta_flow_cate":
        return theta_flow_categorical_hellinger_sqrt(
            args,
            dev=dev,
            x_train=x_train,
            bins_train=bins_train,
            x_val=x_val,
            bins_val=bins_val,
            x_all=x_all,
            bins_all=bins_all,
            k_cat=k_cat,
        )
    if method_name == "ctsm_v":
        return _train_ctsm_v_delta(
            args,
            dev=dev,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            theta_all=theta_all,
            x_all=x_all,
        )
    if method_name == "lda_ctsm_v":
        return _train_lda_ctsm_v_delta(
            args,
            dev=dev,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            theta_all=theta_all,
            x_all=x_all,
            bins_train=bins_train,
        )
    if method_name == "pls_ctsm_v":
        return _train_pls_ctsm_v_delta(
            args,
            dev=dev,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            theta_all=theta_all,
            x_all=x_all,
            bins_train=bins_train,
        )
    if method_name == "pca_ctsm_v":
        return _train_pca_ctsm_v_delta(
            args,
            dev=dev,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            theta_all=theta_all,
            x_all=x_all,
        )
    raise RuntimeError(f"Unhandled method {method_name!r}")


def main(argv: list[str] | None = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)
    methods = parse_methods(str(args.methods))
    if int(args.num_categories) < 2:
        raise ValueError("--num-categories must be >= 2.")
    if (not bool(args.visualization_only)) and str(args.device).strip().lower() == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use a CUDA machine or pass --device cuda when available.")

    ns = conv._parse_n_list(str(args.n_list))
    if not ns:
        raise ValueError("--n-list produced an empty list.")
    if max(ns) > int(args.n_ref):
        raise ValueError(f"Require max(n-list) <= n-ref; got max(n_list)={max(ns)} n_ref={args.n_ref}.")

    if args.dataset_npz is None:
        args.dataset_npz = _default_dataset_npz(int(args.num_categories))
    args.dataset_npz = _abs_without_resolving_symlinks(Path(args.dataset_npz))
    if args.output_dir is None:
        args.output_dir = Path(DATA_DIR) / "h_decoding_categorical_twofig"
    args.output_dir = Path(args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.visualization_only):
        _run_visualization_only(args, methods, ns)
        return

    _ensure_dataset(args)
    native_npz = Path(args.dataset_npz).resolve()
    native_bundle = load_shared_dataset_npz(native_npz)
    native_meta = dict(native_bundle.meta)
    if str(native_meta.get("dataset_family", "")) != "random_mog_categorical":
        raise ValueError(f"Expected random_mog_categorical NPZ, got {native_meta.get('dataset_family')!r}.")
    if str(native_meta.get("theta_type", "")) != "categorical":
        raise ValueError(f"Expected categorical theta_type, got {native_meta.get('theta_type')!r}.")
    native_x_dim = int(native_bundle.x_all.shape[1])
    if native_x_dim != 2:
        raise ValueError(f"Native x_dim must be 2; got {native_x_dim}.")

    pr_project = bool(args.pr_project)
    pr_out_resolved: Path | None = None
    meta_work = dict(native_bundle.meta)
    if pr_project:
        if int(args.pr_dim) <= native_x_dim:
            raise ValueError(f"--pr-dim must exceed native x_dim={native_x_dim}; got {args.pr_dim}.")
        pr_out = args.pr_output_npz
        if pr_out is None:
            pr_out = _default_pr_output_npz(native_npz, int(args.pr_dim))
        else:
            pr_out = _abs_without_resolving_symlinks(Path(pr_out))
        pr_out_resolved = Path(pr_out)
        _ensure_pr_projected_npz(args, native_npz=native_npz, pr_out=pr_out_resolved)
        work_bundle = load_shared_dataset_npz(pr_out_resolved)
        meta_work = dict(work_bundle.meta)
        if int(work_bundle.x_all.shape[0]) != int(native_bundle.x_all.shape[0]):
            raise ValueError("Native and projected NPZ row counts disagree.")
        nt = np.asarray(native_bundle.theta_all, dtype=np.float64)
        wt = np.asarray(work_bundle.theta_all, dtype=np.float64)
        if nt.shape != wt.shape or float(np.max(np.abs(nt - wt))) > 1e-5:
            raise ValueError("Native vs projected theta_all mismatch.")
    else:
        work_bundle = native_bundle

    n_pool = int(native_bundle.theta_all.shape[0])
    if int(args.n_ref) > n_pool:
        raise ValueError(f"--n-ref={args.n_ref} exceeds n_total={n_pool}.")
    need = max(int(args.n_ref), max(ns))
    if n_pool < need:
        raise ValueError(f"Dataset n_total={n_pool} < max(n_ref, max(n_list))={need}.")

    eval_split = _selected_eval_split(args)
    if eval_split == "validation" and float(meta_work.get("train_frac", 0.7)) >= 1.0:
        raise ValueError(
            "--eval-split validation requires train_frac < 1 in dataset meta; "
            f"got train_frac={meta_work.get('train_frac')!r}."
        )

    k_cat = int(native_meta.get("num_categories", int(args.num_categories)))
    _, _, _, _, _, bin_idx_all = prepare_categorical_binning_for_convergence(native_bundle.theta_all, k_cat)
    theta_centers = np.arange(k_cat, dtype=np.float64).reshape(-1, 1)

    rng = np.random.default_rng(int(args.run_seed))
    perm = rng.permutation(n_pool)

    gen_ds_h = build_dataset_from_meta(dict(native_meta))
    hellinger_gt_sq = hellinger_gt_sq_category_matrix(gen_ds_h)
    h_gt_sqrt = np.sqrt(np.clip(hellinger_gt_sq, 0.0, 1.0))
    np.fill_diagonal(h_gt_sqrt, 0.0)

    clf_rs = int(args.run_seed) if int(args.clf_random_state) < 0 else int(args.clf_random_state)
    subset_ref = conv._subset_bundle(work_bundle, perm, int(args.n_ref), meta_work, bin_idx_all=bin_idx_all)
    decode_ref_x_train = np.asarray(subset_ref.bundle.x_train, dtype=np.float64)
    decode_ref_x_all = np.asarray(subset_ref.bundle.x_all, dtype=np.float64)
    decode_ref_bin_all = subset_ref.bin_all
    if eval_split == "validation":
        decode_ref_bin_all = subset_ref.bin_validation
        decode_ref_x_all = np.asarray(subset_ref.bundle.x_validation, dtype=np.float64)
    decode_ref = conv._pairwise_clf_from_bundle(
        args=args,
        meta=meta_work,
        subset=subset_ref,
        output_dir=str(args.output_dir / "reference"),
        n_bins=k_cat,
        clf_min_class_count=int(args.clf_min_class_count),
        clf_random_state=clf_rs,
        decode_x_train=decode_ref_x_train,
        decode_x_all=decode_ref_x_all,
        decode_bin_all=decode_ref_bin_all,
    )
    n_m = len(methods)
    n_cols = len(ns)
    h_sqrt_sweep = np.full((n_m, n_cols, k_cat, k_cat), np.nan, dtype=np.float64)
    wall_s = np.full((n_m, n_cols), np.nan, dtype=np.float64)
    corr_h = np.full((n_m, n_cols), np.nan, dtype=np.float64)
    nmse_h = np.full((n_m, n_cols), np.nan, dtype=np.float64)
    llr_pearson_offdiag = np.full((n_m, n_cols), np.nan, dtype=np.float64)
    hellinger_pearson_offdiag_cat = np.full((n_m, n_cols), np.nan, dtype=np.float64)
    theta_flow_cate_num_valid_pairs = np.full((n_m, n_cols), -1, dtype=np.int64)
    theta_flow_cate_num_skipped_pairs = np.full((n_m, n_cols), -1, dtype=np.int64)
    theta_flow_cate_train_counts = np.full((n_m, n_cols, k_cat, k_cat, 2), -1, dtype=np.int64)
    theta_flow_cate_val_counts = np.full((n_m, n_cols, k_cat, k_cat, 2), -1, dtype=np.int64)
    decode_sweep_list: list[np.ndarray] = []

    dev = require_device(str(args.device))
    torch.manual_seed(int(args.run_seed))
    np.random.seed(int(args.run_seed))

    dbg = _debug_categorical_module()
    scatter_deltas: dict[str, np.ndarray] = {}
    scatter_cat_h2: dict[str, np.ndarray] = {}
    scatter_true_delta: np.ndarray | None = None
    n_scatter = max(ns)
    out_dir = str(args.output_dir)
    loss_root = os.path.join(out_dir, "training_losses")

    for j, n in enumerate(ns):
        subset_w = conv._subset_bundle(work_bundle, perm, int(n), meta_work, bin_idx_all=bin_idx_all)
        subset_n = conv._subset_bundle(native_bundle, perm, int(n), meta_work, bin_idx_all=bin_idx_all)
        bins_n = np.asarray(subset_w.bin_all, dtype=np.int64).reshape(-1)
        x_native_all = np.asarray(subset_n.bundle.x_all, dtype=np.float64)
        train_theta = _as_2d(subset_w.bundle.theta_train)
        val_theta = _as_2d(subset_w.bundle.theta_validation)
        x_train = np.asarray(subset_w.bundle.x_train, dtype=np.float64)
        x_val = np.asarray(subset_w.bundle.x_validation, dtype=np.float64)
        x_all = np.asarray(subset_w.bundle.x_all, dtype=np.float64)
        bins_train = np.asarray(subset_w.bin_train, dtype=np.int64).reshape(-1)
        bins_val = np.asarray(subset_w.bin_validation, dtype=np.int64).reshape(-1)

        val_h = eval_split == "validation"
        if val_h:
            n_val = int(subset_w.bundle.x_validation.shape[0])
            if n_val < 1:
                raise ValueError(f"n={int(n)}: empty validation split; cannot use --eval-split validation.")
            theta_h = val_theta
            x_h = x_val
            x_native_h = np.asarray(subset_n.bundle.x_validation, dtype=np.float64)
            bins_h = np.asarray(subset_w.bin_validation, dtype=np.int64).reshape(-1)
        else:
            theta_h = _as_2d(subset_w.bundle.theta_all)
            x_h = x_all
            x_native_h = x_native_all
            bins_h = bins_n

        true_c = compute_true_conditional_loglik_matrix(x_native_h, theta_h, native_meta)
        true_delta_l = HMatrixEstimator.compute_delta_l(true_c)

        decode_eval_x_all = x_h
        decode_eval_bins_all = bins_h
        clf_n = conv._pairwise_clf_from_bundle(
            args=args,
            meta=meta_work,
            subset=subset_w,
            output_dir=str(args.output_dir / "decode_shared" / f"n_{int(n):06d}"),
            n_bins=k_cat,
            clf_min_class_count=int(args.clf_min_class_count),
            clf_random_state=clf_rs,
            decode_x_train=x_train,
            decode_x_all=decode_eval_x_all,
            decode_bin_all=decode_eval_bins_all,
        )
        decode_sweep_list.append(np.asarray(clf_n, dtype=np.float64))

        for i, method_name in enumerate(methods):
            t0 = time.time()
            row_loss_dir = os.path.join(loss_root, _sanitize_row_label(method_name))
            os.makedirs(row_loss_dir, exist_ok=True)
            loss_npz_path = os.path.join(row_loss_dir, f"n_{int(n):06d}.npz")

            if method_name == "bin_gaussian":
                vf = float(
                    getattr(
                        args,
                        "flow_theta_reg_variance_floor",
                        getattr(args, "flow_x_reg_variance_floor", 1e-6),
                    )
                )
                subset_bg = _validation_only_work_sweep_subset(subset_w) if val_h else subset_w
                bg_h2 = conv._binned_gaussian_hellinger_sq(subset_bg, k_cat, variance_floor=vf)
                h_sqrt_sweep[i, j] = np.asarray(conv._sqrt_h_like(np.asarray(bg_h2, dtype=np.float64)), dtype=np.float64)
                np.fill_diagonal(h_sqrt_sweep[i, j], 0.0)
                wall_s[i, j] = time.time() - t0
                result = {"train_out": None}
                nan_delta = np.full(np.asarray(true_delta_l, dtype=np.float64).shape, np.nan, dtype=np.float64)
                m_llr = dbg._llr_comparison_metrics(nan_delta, true_delta_l)
                llr_pearson_offdiag[i, j] = float(m_llr["llr_pearson_r_offdiag"])
                m_h = dbg._hellinger_comparison_metrics_cat(bg_h2, hellinger_gt_sq)
                hellinger_pearson_offdiag_cat[i, j] = float(m_h["hellinger_pearson_r_offdiag_cat"])
                _save_method_training_loss_npz(loss_npz_path, method_name=method_name, result=result)
            else:
                call_x_all = x_h
                call_bins_all = bins_h
                result = _train_one_method(
                    args,
                    dev=dev,
                    method_name=method_name,
                    theta_train=train_theta,
                    x_train=x_train,
                    theta_val=val_theta,
                    x_val=x_val,
                    theta_all=theta_h,
                    x_all=call_x_all,
                    bins_train=bins_train,
                    bins_val=bins_val,
                    bins_all=call_bins_all,
                    k_cat=k_cat,
                )
                _save_method_training_loss_npz(loss_npz_path, method_name=method_name, result=result)
                if "h_sqrt" in result:
                    h_sqrt_sweep[i, j] = np.asarray(result["h_sqrt"], dtype=np.float64)
                    np.fill_diagonal(h_sqrt_sweep[i, j], 0.0)
                    h_cat_sq = np.asarray(result.get("h_cat_sq", np.square(h_sqrt_sweep[i, j])), dtype=np.float64)
                    delta = np.full(np.asarray(true_delta_l, dtype=np.float64).shape, np.nan, dtype=np.float64)
                    theta_flow_cate_num_valid_pairs[i, j] = int(result.get("theta_flow_cate_num_valid_pairs", -1))
                    theta_flow_cate_num_skipped_pairs[i, j] = int(result.get("theta_flow_cate_num_skipped_pairs", -1))
                    if "theta_flow_cate_train_counts" in result:
                        theta_flow_cate_train_counts[i, j] = np.asarray(
                            result["theta_flow_cate_train_counts"], dtype=np.int64
                        )
                    if "theta_flow_cate_val_counts" in result:
                        theta_flow_cate_val_counts[i, j] = np.asarray(
                            result["theta_flow_cate_val_counts"], dtype=np.int64
                        )
                else:
                    delta = np.asarray(result["delta_l"], dtype=np.float64)
                    h_dir = h_sq_directed_from_delta_l(delta)
                    h_cat_sq = h_sq_category_from_sample_directed(h_dir, bins_h, k_cat=k_cat)
                    h_sqrt_sweep[i, j] = np.sqrt(np.clip(h_cat_sq, 0.0, 1.0))
                    np.fill_diagonal(h_sqrt_sweep[i, j], 0.0)
                wall_s[i, j] = time.time() - t0

                m_llr = dbg._llr_comparison_metrics(delta, true_delta_l)
                llr_pearson_offdiag[i, j] = float(m_llr["llr_pearson_r_offdiag"])
                m_h = dbg._hellinger_comparison_metrics_cat(h_cat_sq, hellinger_gt_sq)
                hellinger_pearson_offdiag_cat[i, j] = float(m_h["hellinger_pearson_r_offdiag_cat"])

            corr_h[i, j] = vhb.matrix_corr_offdiag_pearson(
                vhb.impute_offdiag_nan_mean(h_sqrt_sweep[i, j]),
                vhb.impute_offdiag_nan_mean(np.asarray(h_gt_sqrt, dtype=np.float64)),
            )
            nmse_h[i, j] = _matrix_nmse_offdiag(
                vhb.impute_offdiag_nan_mean(h_sqrt_sweep[i, j]),
                vhb.impute_offdiag_nan_mean(np.asarray(h_gt_sqrt, dtype=np.float64)),
            )

            if int(n) == int(n_scatter):
                scatter_true_delta = np.asarray(true_delta_l, dtype=np.float64).copy()
                if method_name == "bin_gaussian":
                    scatter_cat_h2[method_name] = np.asarray(bg_h2, dtype=np.float64).copy()
                elif "h_sqrt" in result:
                    scatter_cat_h2[method_name] = np.asarray(h_cat_sq, dtype=np.float64).copy()
                else:
                    scatter_deltas[method_name] = np.asarray(delta, dtype=np.float64)
                    scatter_cat_h2[method_name] = np.asarray(h_cat_sq, dtype=np.float64).copy()

    decode_sweep = np.stack(decode_sweep_list, axis=0)
    corr_decode = np.full((n_cols,), np.nan, dtype=np.float64)
    nmse_decode = np.full((n_cols,), np.nan, dtype=np.float64)
    decode_ref_imp = vhb.impute_offdiag_nan_mean(np.asarray(decode_ref, dtype=np.float64))
    for j in range(n_cols):
        corr_decode[j] = vhb.matrix_corr_offdiag_pearson(
            vhb.impute_offdiag_nan_mean(np.asarray(decode_sweep[j], dtype=np.float64)),
            decode_ref_imp,
        )
    nmse_decode = _nmse_decode_vs_ref(decode_sweep, np.asarray(decode_ref, dtype=np.float64))

    cat_footer = {
        "h_gt_sqrt": h_gt_sqrt,
        "decode_ref": decode_ref,
        "n_ref": int(args.n_ref),
        "decode_sweep_for_decode_limits": decode_sweep,
    }
    sweep_svg = _render_method_sweep_panel(
        row_labels=methods,
        h_sweep=h_sqrt_sweep,
        clf_sweep_shared=decode_sweep,
        clf_hellinger_sweep_shared=None,
        n_list=ns,
        out_svg_path=os.path.join(out_dir, "h_decoding_categorical_twofig_sweep.svg"),
        n_bins=k_cat,
        theta_centers=theta_centers,
        clf_ref_decode_limits=np.asarray(decode_ref, dtype=np.float64),
        category_gt_footer=cat_footer,
    )
    zdummy = np.zeros_like(corr_h)
    corr_nmse_svg = _render_corr_nmse_two_panel(
        row_labels=methods,
        n_list=ns,
        corr_h=corr_h,
        corr_decode_shared=corr_decode,
        corr_decode_hellinger_shared=None,
        corr_hellinger_lb=zdummy,
        corr_hellinger_ub=zdummy,
        show_hellinger_bounds=False,
        nmse_h=nmse_h,
        nmse_decode_shared=nmse_decode,
        nmse_decode_hellinger_shared=None,
        out_svg_path=os.path.join(out_dir, "h_decoding_categorical_twofig_corr_nmse.svg"),
    )
    loss_panel_svg = _render_row_n_training_losses_panel(
        row_labels=methods,
        n_list=ns,
        loss_root=loss_root,
        out_svg_path=os.path.join(out_dir, "h_decoding_categorical_twofig_training_losses_panel.svg"),
    )

    if not bool(args.no_scatter_diagnostics) and scatter_true_delta is not None and scatter_cat_h2:
        metrics_by_method: dict[str, dict[str, float]] = {}
        hell_cat: dict[str, np.ndarray] = {}
        for name in methods:
            if name not in scatter_cat_h2:
                continue
            hell_cat[name] = scatter_cat_h2[name]
            if name == "bin_gaussian":
                nan_d = np.full_like(scatter_true_delta, np.nan, dtype=np.float64)
                metrics_by_method[name] = dbg._llr_comparison_metrics(nan_d, scatter_true_delta)
            elif name not in scatter_deltas:
                nan_d = np.full_like(scatter_true_delta, np.nan, dtype=np.float64)
                metrics_by_method[name] = dbg._llr_comparison_metrics(nan_d, scatter_true_delta)
            else:
                dlt = scatter_deltas[name]
                metrics_by_method[name] = dbg._llr_comparison_metrics(dlt, scatter_true_delta)
            metrics_by_method[name].update(
                dbg._hellinger_comparison_metrics_cat(scatter_cat_h2[name], hellinger_gt_sq)
            )
        dbg._save_llr_est_vs_true_figure(
            scatter_deltas,
            scatter_true_delta,
            out_base=Path(out_dir) / "llr_est_vs_true_all",
            metrics_by_method=metrics_by_method,
        )
        dbg._save_hellinger_est_vs_gt_figure(
            hell_cat,
            hellinger_gt_sq,
            out_base=Path(out_dir) / "hellinger_est_vs_gt_all",
            metrics_by_method=metrics_by_method,
        )

    source_indices = np.asarray(perm[: int(args.n_ref)], dtype=np.int64)
    out_npz = os.path.join(out_dir, "h_decoding_categorical_twofig_results.npz")
    np.savez_compressed(
        out_npz,
        n=np.asarray(ns, dtype=np.int64),
        n_ref=np.int64(int(args.n_ref)),
        num_categories=np.int64(k_cat),
        method_names=np.asarray(methods, dtype=object),
        theta_bin_centers=np.asarray(theta_centers, dtype=np.float64),
        perm_seed=np.int64(int(args.run_seed)),
        h_gt_sqrt=np.asarray(h_gt_sqrt, dtype=np.float64),
        hellinger_gt_sq_category=np.asarray(hellinger_gt_sq, dtype=np.float64),
        decode_ref=np.asarray(decode_ref, dtype=np.float64),
        decode_sweep=np.asarray(decode_sweep, dtype=np.float64),
        h_sqrt_sweep=np.asarray(h_sqrt_sweep, dtype=np.float64),
        corr_h_vs_gt=np.asarray(corr_h, dtype=np.float64),
        nmse_h_vs_gt=np.asarray(nmse_h, dtype=np.float64),
        corr_decode_vs_ref=np.asarray(corr_decode, dtype=np.float64),
        nmse_decode_vs_ref=np.asarray(nmse_decode, dtype=np.float64),
        llr_pearson_offdiag=np.asarray(llr_pearson_offdiag, dtype=np.float64),
        hellinger_pearson_offdiag_cat=np.asarray(hellinger_pearson_offdiag_cat, dtype=np.float64),
        theta_flow_cate_num_valid_pairs=np.asarray(theta_flow_cate_num_valid_pairs, dtype=np.int64),
        theta_flow_cate_num_skipped_pairs=np.asarray(theta_flow_cate_num_skipped_pairs, dtype=np.int64),
        theta_flow_cate_train_counts=np.asarray(theta_flow_cate_train_counts, dtype=np.int64),
        theta_flow_cate_val_counts=np.asarray(theta_flow_cate_val_counts, dtype=np.int64),
        wall_seconds=np.asarray(wall_s, dtype=np.float64),
        source_indices=source_indices,
        native_dataset_npz=np.asarray([str(native_npz)], dtype=object),
        work_dataset_npz=np.asarray(
            [str(pr_out_resolved) if pr_out_resolved is not None else str(native_npz)], dtype=object
        ),
        pr_projected=np.bool_(pr_project),
        pr_dim=np.int64(int(args.pr_dim) if pr_project else native_x_dim),
        training_losses_root=np.asarray([os.path.abspath(loss_root)], dtype=object),
        eval_split=np.asarray([eval_split], dtype=object),
    )

    summary_path = os.path.join(out_dir, "h_decoding_categorical_twofig_summary.txt")
    all_columns_png: str | None = None
    try:
        all_columns_png = _write_all_columns_png(
            out_dir,
            sweep_svg=sweep_svg,
            corr_nmse_svg=corr_nmse_svg,
            loss_panel_svg=loss_panel_svg,
        )
    except Exception as exc:
        print(
            f"[cat-twofig] WARNING: all-columns PNG failed ({type(exc).__name__}: {exc}); "
            "sweep / corr_nmse / loss SVGs and results NPZ are still saved.",
            flush=True,
        )

    _write_summary(
        summary_path,
        args=args,
        out_npz=os.path.abspath(out_npz),
        sweep_svg=os.path.abspath(sweep_svg),
        corr_nmse_svg=os.path.abspath(corr_nmse_svg),
        visualization_only=False,
        loss_panel_svg=os.path.abspath(loss_panel_svg),
        all_columns_png=os.path.abspath(all_columns_png) if all_columns_png else None,
        training_losses_root=os.path.abspath(loss_root),
    )

    print("[cat-twofig] Saved:", flush=True)
    print(f"  - {os.path.abspath(out_npz)}", flush=True)
    print(f"  - {os.path.abspath(sweep_svg)}", flush=True)
    print(f"  - {os.path.abspath(corr_nmse_svg)}", flush=True)
    print(f"  - {os.path.abspath(loss_panel_svg)}", flush=True)
    if all_columns_png:
        print(f"  - {os.path.abspath(all_columns_png)}", flush=True)
    print(f"  - {os.path.abspath(summary_path)}", flush=True)


if __name__ == "__main__":
    main()
