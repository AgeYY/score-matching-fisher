#!/usr/bin/env python3
"""Categorical benchmark diagnostic for log-likelihood-ratio estimators.

Loads the native 2D random_mog_categorical NPZ (no PR autoencoder by default). Default is
``K=2`` mixture components; data live under
``data/random_mog_categorical_xdim2_k2/`` unless ``--dataset-npz`` is set. Optional ``--pr-project``
embeds ``x`` through a PR autoencoder to ``--pr-dim`` (default 10) while ground-truth LLR and
analytic category Hellinger remain on the native 2D MoG. Saves
a 2D scatter of native observations colored by category. Compares estimated pairwise
LLRs to exact MoG ``log p(x|theta)`` from NPZ metadata (same
``ToyCategoricalRandomMoGDataset`` recipe). Supported estimators include
nonlinear x-flow, pairwise binary classifiers, scheduled linear x-flow, and
SIR-basis low-rank scheduled x-flow.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import global_setting  # noqa: F401  # matplotlib rc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from fisher.gaussian_x_flow import path_schedule_from_name
from fisher.h_decoding_convergence_methods import _fit_sir_projection, prepare_categorical_binning_for_convergence
from fisher.h_matrix import HMatrixEstimator
from fisher.hellinger_gt import hellinger_sq_gaussian_diag
from fisher.linear_x_flow import (
    ConditionalTimeLinearXFlowMLP,
    ConditionalTimeLowRankCorrectionLinearXFlowMLP,
    compute_ode_time_linear_x_flow_c_matrix,
    compute_time_linear_x_flow_c_matrix,
    train_time_linear_x_flow_schedule,
)
from fisher.lxf_bin_likelihood_hellinger import lxf_bin_likelihood_hellinger
from fisher.shared_dataset_io import SharedDatasetBundle, load_shared_dataset_npz
from fisher.shared_fisher_est import (
    build_conditional_x_velocity_model,
    build_dataset_from_meta,
    require_device,
)
from fisher.trainers import train_conditional_x_flow_model


_METHOD_ALIASES = {
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
}
_DEFAULT_METHODS = ("x_flow", "binary_classifier", "linear_x_flow_t", "xflow_sir_lrank")


def _default_dataset_npz(num_categories: int) -> Path:
    return (
        _repo_root
        / "data"
        / f"random_mog_categorical_xdim2_k{int(num_categories)}"
        / "random_mog_categorical.npz"
    )


def _abs_without_resolving_symlinks(path: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return _repo_root / p


def _run(cmd: list[str]) -> None:
    print("[debug-xflow] running: " + " ".join(cmd), flush=True)
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
                str(args.n_total),
                "--output-npz",
                str(ds),
            ]
        )
    else:
        print(f"[debug-xflow] using existing 2D dataset NPZ: {ds}", flush=True)


def _default_debug_dir(args: argparse.Namespace) -> Path:
    if args.output_npz is not None:
        return Path(args.output_npz).resolve().parent
    return _abs_without_resolving_symlinks(Path(args.dataset_npz)).parent / "xflow_llr_debug"


def _default_pr_output_npz(native_npz: Path, pr_dim: int) -> Path:
    return _abs_without_resolving_symlinks(native_npz).parent / f"pr_xdim{int(pr_dim)}" / "random_mog_categorical_pr.npz"


def _ensure_pr_projected_npz(args: argparse.Namespace, *, native_npz: Path, pr_out: Path) -> None:
    """Run ``project_dataset_pr_autoencoder.py`` unless ``pr_out`` exists (unless ``--force-regenerate``)."""
    pr_out = Path(pr_out)
    native_npz = Path(native_npz).resolve()
    pr_out.parent.mkdir(parents=True, exist_ok=True)
    if bool(args.force_regenerate) and pr_out.is_file():
        pr_out.unlink()
    if pr_out.is_file():
        print(f"[debug-xflow] using existing PR-projected NPZ: {pr_out}", flush=True)
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
    print("[debug-xflow] running PR projection: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(_repo_root), check=True)


def _save_2d_dataset_scatter(
    x: np.ndarray,
    bin_idx: np.ndarray,
    *,
    k_cat: int,
    out_path: Path,
    rng: np.random.Generator,
    max_points: int,
    title: str | None = None,
) -> None:
    x = np.asarray(x, dtype=np.float64)
    bin_idx = np.asarray(bin_idx, dtype=np.int64).reshape(-1)
    if x.shape[1] != 2:
        raise ValueError(f"Expected x_dim=2 for scatter plot, got {x.shape[1]}.")
    n = int(x.shape[0])
    if n > int(max_points):
        pick = np.sort(rng.choice(n, size=int(max_points), replace=False))
        x = x[pick]
        bin_idx = bin_idx[pick]
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6.0, 5.5), layout="constrained")
    for k in range(int(k_cat)):
        m = bin_idx == k
        if not np.any(m):
            continue
        ax.scatter(
            x[m, 0],
            x[m, 1],
            s=6,
            alpha=0.55,
            c=[cmap(k % 10)],
            label=f"category {k}",
            linewidths=0,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(title if title is not None else f"random_mog_categorical, $K={k_cat}$ (2D)")
    ax.legend(loc="best", fontsize=8, markerscale=2.0, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _compute_true_conditional_loglik_matrix(
    x_all: np.ndarray,
    theta_all: np.ndarray,
    meta: dict,
) -> np.ndarray:
    """Exact ``c_matrix[i,j] = log p(x_i | theta_j)`` for categorical MoG from NPZ meta."""
    gen_ds = build_dataset_from_meta(dict(meta))
    n = int(np.asarray(x_all).shape[0])
    x_all = np.asarray(x_all, dtype=np.float64)
    theta_all = np.asarray(theta_all, dtype=np.float64)
    true_c = np.empty((n, n), dtype=np.float64)
    for j in range(n):
        th_col = np.tile(theta_all[j : j + 1], (n, 1))
        true_c[:, j] = gen_ds.log_p_x_given_theta(x_all, th_col)
    return true_c


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2:
        return float("nan")
    if float(np.std(a)) < 1e-15 or float(np.std(b)) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _hellinger_gt_sq_category_matrix(gen_ds: object) -> np.ndarray:
    """Analytic category-to-category squared Hellinger from stored MoG means and variances."""
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


def _h_sq_directed_from_delta_l(delta_l: np.ndarray) -> np.ndarray:
    """Directed sample-level $H^2$ from $\\Delta L$ (matches ``HMatrixEstimator.compute_h_directed``)."""
    return HMatrixEstimator.compute_h_directed(np.asarray(delta_l, dtype=np.float64))


def _h_sq_sym_sample_from_delta_l(delta_l: np.ndarray) -> np.ndarray:
    """Symmetric sample-level $H^2$ matrix from $\\Delta L$."""
    h_dir = _h_sq_directed_from_delta_l(delta_l)
    return HMatrixEstimator.symmetrize(h_dir)


def _h_sq_category_from_sample_directed(
    h_directed: np.ndarray,
    category_labels: np.ndarray,
    *,
    k_cat: int,
) -> np.ndarray:
    """Aggregate directed $H^2$ to categories, symmetrize, zero diagonal (plan spec)."""
    h = np.asarray(h_directed, dtype=np.float64)
    n = int(h.shape[0])
    labs = np.asarray(category_labels, dtype=np.int64).reshape(-1)
    if int(labs.shape[0]) != n:
        raise ValueError("category_labels length must match h_directed.shape[0].")
    # tilde_H[i, b] = mean_{j : label(j)=b} H_directed[i, j]
    col_sum = np.zeros((n, int(k_cat)), dtype=np.float64)
    col_cnt = np.zeros((n, int(k_cat)), dtype=np.float64)
    for j in range(n):
        b = int(labs[j])
        if 0 <= b < int(k_cat):
            col_sum[:, b] += h[:, j]
            col_cnt[:, b] += 1.0
    denom = np.maximum(col_cnt, 1.0)
    tilde = col_sum / denom
    # H_dir[a, b] = mean_{i : label(i)=a} tilde_H[i, b]
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


def _hellinger_cat_offdiag_mask(k: int) -> np.ndarray:
    return np.triu(np.ones((int(k), int(k)), dtype=bool), k=1)


def _hellinger_comparison_metrics_cat(est: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """MAE/RMSE/bias and Pearson on upper-triangular off-diagonal category pairs (unique)."""
    e = np.asarray(est, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    k = int(e.shape[0])
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
        "hellinger_mae_offdiag_cat": float(np.mean(np.abs(d))),
        "hellinger_rmse_offdiag_cat": float(np.sqrt(np.mean(d**2))),
        "hellinger_bias_offdiag_cat": float(np.mean(d)),
        "hellinger_pearson_r_offdiag_cat": _pearson_r(e[m], g[m]),
    }


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
        ax.text(
            0.5,
            0.5,
            "No category off-diagonal pairs\n(K ≤ 1)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_xlabel(r"analytic GT $H^2$ (category)")
        ax.set_ylabel(r"LLR-derived est. $H^2$ (category)")
        ax.set_title(r"Squared Hellinger: estimated vs analytic GT")
    else:
        cmap = plt.get_cmap("tab10")
        for idx, (method_name, est) in enumerate(est_cat_by_method.items()):
            y = np.asarray(est, dtype=np.float64)[mask].ravel()
            m = metrics_by_method[method_name]
            label = (
                f"{method_name} "
                f"(RMSE={m.get('hellinger_rmse_offdiag_cat', float('nan')):.3g}, "
                f"r={m.get('hellinger_pearson_r_offdiag_cat', float('nan')):.3g})"
            )
            ax.scatter(x, y, s=22, alpha=0.55, linewidths=0, color=cmap(idx % 10), label=label)
        y_all = np.concatenate(
            [np.asarray(est_cat_by_method[m], dtype=np.float64)[mask].ravel() for m in est_cat_by_method]
        )
        finite = np.isfinite(np.concatenate([x, y_all]))
        vals = np.concatenate([x, y_all])[finite]
        lo = float(np.min(vals)) if vals.size else 0.0
        hi = float(np.max(vals)) if vals.size else 1.0
        pad = 0.05 * (hi - lo) if hi > lo else 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.0, alpha=0.7, label="identity")
        ax.set_xlim(max(0.0, lo - pad), min(1.0, hi + pad))
        ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"analytic GT $H^2$ (category)")
        ax.set_ylabel(r"LLR-derived est. $H^2$ (category)")
        ax.set_title(r"$H^2$: category-level LLR map vs analytic diagonal Gaussian GT")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=7, framealpha=0.9)
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# Ground-truth ΔL band for auxiliary off-diagonal LLR metrics (scatter legends / NPZ).
LLR_GT_METRIC_BAND_LO = -8.0
LLR_GT_METRIC_BAND_HI = 8.0


def _llr_comparison_metrics(est_delta: np.ndarray, true_delta: np.ndarray) -> dict[str, float]:
    """RMSE/MAE/bias/std and Pearson for all entries and off-diagonal only."""
    est = np.asarray(est_delta, dtype=np.float64)
    true_m = np.asarray(true_delta, dtype=np.float64)
    diff = est - true_m
    n = int(est.shape[0])
    out: dict[str, float] = {
        "llr_mae_all": float(np.mean(np.abs(diff))),
        "llr_rmse_all": float(np.sqrt(np.mean(diff**2))),
        "llr_bias_all": float(np.mean(diff)),
        "llr_std_err_all": float(np.std(diff, ddof=0)),
        "llr_pearson_r_all": _pearson_r(est, true_m),
    }
    if n > 1:
        mask = ~np.eye(n, dtype=bool)
        d_off = diff[mask]
        e_off = est[mask]
        t_off = true_m[mask]
        out["llr_mae_offdiag"] = float(np.mean(np.abs(d_off)))
        out["llr_rmse_offdiag"] = float(np.sqrt(np.mean(d_off**2)))
        out["llr_bias_offdiag"] = float(np.mean(d_off))
        out["llr_std_err_offdiag"] = float(np.std(d_off, ddof=0))
        out["llr_pearson_r_offdiag"] = _pearson_r(e_off, t_off)
        band_m = (t_off >= LLR_GT_METRIC_BAND_LO) & (t_off <= LLR_GT_METRIC_BAND_HI)
        e_b = e_off[band_m]
        t_b = t_off[band_m]
        if int(e_b.size) == 0:
            out["llr_rmse_offdiag_true_in_m8_p8"] = float("nan")
            out["llr_pearson_r_offdiag_true_in_m8_p8"] = float("nan")
        else:
            d_b = e_b - t_b
            out["llr_rmse_offdiag_true_in_m8_p8"] = float(np.sqrt(np.mean(d_b**2)))
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


def _parse_methods(methods: str) -> list[str]:
    toks = [t.strip() for t in str(methods).split(",") if t.strip()]
    if not toks:
        raise ValueError("--methods must contain at least one method.")
    out: list[str] = []
    for tok in toks:
        key = tok.strip().lower()
        norm = _METHOD_ALIASES.get(key)
        if norm is None:
            raise ValueError(f"Unknown method {tok!r}; valid methods: {', '.join(_DEFAULT_METHODS)}")
        if norm not in out:
            out.append(norm)
    return out


def _save_llr_est_vs_true_figure(
    method_deltas: dict[str, np.ndarray],
    true_delta: np.ndarray,
    *,
    out_base: Path,
    metrics_by_method: dict[str, dict[str, float]],
) -> None:
    n = int(true_delta.shape[0])
    if n > 1:
        mask = ~np.eye(n, dtype=bool)
        x = np.asarray(true_delta, dtype=np.float64)[mask].ravel()
    else:
        mask = np.zeros_like(true_delta, dtype=bool)
        x = np.array([], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.2, 5.8), layout="constrained")
    if x.size == 0:
        ax.text(
            0.5,
            0.5,
            "No off-diagonal LLR pairs\n(n_ref ≤ 1)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_xlabel(r"true $\Delta L$")
        ax.set_ylabel(r"estimated $\Delta L$")
        ax.set_title(r"LLR: estimated vs ground truth")
    else:
        ys: list[np.ndarray] = []
        for method_name, est_delta in method_deltas.items():
            y = np.asarray(est_delta, dtype=np.float64)[mask].ravel()
            ys.append(y)
            m = metrics_by_method[method_name]
            label = (
                f"{method_name} "
                f"(RMSE={m.get('llr_rmse_offdiag', float('nan')):.3g}, "
                f"r={m.get('llr_pearson_r_offdiag', float('nan')):.3g}; "
                f"GT∈[{LLR_GT_METRIC_BAND_LO:g},{LLR_GT_METRIC_BAND_HI:g}]: "
                f"RMSE={m.get('llr_rmse_offdiag_true_in_m8_p8', float('nan')):.3g}, "
                f"r={m.get('llr_pearson_r_offdiag_true_in_m8_p8', float('nan')):.3g})"
            )
            ax.scatter(x, y, s=8, alpha=0.28, linewidths=0, label=label)
        y_all = np.concatenate(ys) if ys else x
        finite = np.isfinite(np.concatenate([x, y_all]))
        vals = np.concatenate([x, y_all])[finite]
        lo = float(np.min(vals)) if vals.size else -1.0
        hi = float(np.max(vals)) if vals.size else 1.0
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.0, alpha=0.7)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"true $\Delta L$")
        ax.set_ylabel(r"estimated $\Delta L$")
        ax.set_title(r"LLR: estimated vs ground truth (off-diagonal)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _subset_bundle(
    bundle: SharedDatasetBundle,
    *,
    perm: np.ndarray,
    n_ref: int,
    bin_idx_all: np.ndarray,
) -> tuple[SharedDatasetBundle, np.ndarray, np.ndarray]:
    idx = np.asarray(perm[: int(n_ref)], dtype=np.int64)
    theta_all = np.asarray(bundle.theta_all[idx], dtype=np.float64)
    x_all = np.asarray(bundle.x_all[idx], dtype=np.float64)
    bins = np.asarray(bin_idx_all[idx], dtype=np.int64).reshape(-1)
    if theta_all.ndim == 1:
        theta_all = theta_all.reshape(-1, 1)
    if x_all.ndim != 2:
        raise ValueError("x_all must be 2D.")

    train_frac = float(bundle.meta.get("train_frac", 0.7))
    if train_frac >= 1.0:
        n_train = int(n_ref)
    else:
        n_train = int(train_frac * int(n_ref))
        n_train = min(max(n_train, 1), int(n_ref) - 1)

    sub = SharedDatasetBundle(
        meta=bundle.meta,
        theta_all=theta_all,
        x_all=x_all,
        train_idx=np.arange(n_train, dtype=np.int64),
        validation_idx=np.arange(n_train, int(n_ref), dtype=np.int64),
        theta_train=theta_all[:n_train],
        x_train=x_all[:n_train],
        theta_validation=theta_all[n_train:],
        x_validation=x_all[n_train:],
    )
    return sub, bins, idx


def _as_2d(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}.")
    return arr


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
) -> dict[str, object]:
    model_args = SimpleNamespace(
        x_dim=int(x_all.shape[1]),
        flow_hidden_dim=int(args.flow_hidden_dim),
        flow_depth=int(args.flow_depth),
        flow_use_layer_norm=False,
        flow_gated_film=False,
        flow_zero_out_init=False,
        flow_cond_embed_dim=16,
        flow_cond_embed_depth=1,
        flow_cond_embed_act="silu",
        flow_x_theta_fourier_k=4,
        flow_x_theta_fourier_omega=1.0,
        flow_x_theta_fourier_no_linear=False,
        flow_x_theta_fourier_no_bias=False,
    )
    print(
        f"[debug-xflow] training x_flow: n_ref={args.n_ref} train={x_train.shape[0]} "
        f"val={x_val.shape[0]} x_dim={x_all.shape[1]} theta_dim={theta_all.shape[1]}",
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

    print("[debug-xflow] computing x_flow c_matrix = log p(x_i | theta_j)", flush=True)
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
) -> dict[str, object]:
    print("[debug-xflow] training binary_classifier pairwise logistic LLRs", flush=True)
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
            col_a = row_a
            col_b = row_b
            if row_a.size and col_b.size:
                delta[np.ix_(row_a, col_b)] = llr_b_minus_a[row_a].reshape(-1, 1)
            if row_b.size and col_a.size:
                delta[np.ix_(row_b, col_a)] = -llr_b_minus_a[row_b].reshape(-1, 1)
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
) -> dict[str, object]:
    print(f"[debug-xflow] training {method_name}", flush=True)
    rank = int(args.lxf_low_rank_dim)
    common = dict(
        theta_dim=int(theta_all.shape[1]),
        x_dim=int(x_all.shape[1]),
        hidden_dim=int(args.lxfs_hidden_dim),
        depth=int(args.lxfs_depth),
    )
    sir_meta: dict[str, object] = {}
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2D random_mog_categorical: visualize data and compare LLR estimators to exact MoG ground truth."
    )
    p.add_argument(
        "--num-categories",
        type=int,
        default=2,
        help="Mixture cardinality K for random_mog_categorical (passed to make_dataset when generating).",
    )
    p.add_argument(
        "--dataset-npz",
        type=Path,
        default=None,
        help="Native 2D NPZ (x_dim=2). Default: data/random_mog_categorical_xdim2_kK/random_mog_categorical.npz.",
    )
    p.add_argument("--output-npz", type=Path, default=None)
    p.add_argument("--force-regenerate", action="store_true")
    p.add_argument("--n-total", type=int, default=50000)
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving the 2D dataset scatter figure (SVG+PNG).",
    )
    p.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Base path for figures (extensions .svg/.png added). Default: under xflow_llr_debug/.",
    )
    p.add_argument("--plot-max-points", type=int, default=12000)
    p.add_argument("--n-ref", type=int, default=600)
    p.add_argument("--run-seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--methods",
        type=str,
        default=",".join(_DEFAULT_METHODS),
        help="Comma-separated estimators: x_flow,binary_classifier,linear_x_flow_t,xflow_sir_lrank.",
    )

    p.add_argument("--flow-arch", type=str, default="mlp", choices=["mlp", "film", "film_fourier"])
    p.add_argument("--flow-epochs", type=int, default=10000)
    p.add_argument("--flow-batch-size", type=int, default=256)
    p.add_argument("--flow-lr", type=float, default=1e-3)
    p.add_argument("--flow-hidden-dim", type=int, default=128)
    p.add_argument("--flow-depth", type=int, default=3)
    p.add_argument("--flow-scheduler", type=str, default="cosine")
    p.add_argument("--flow-fm-t-eps", type=float, default=0.05)
    p.add_argument("--flow-early-patience", type=int, default=1000)
    p.add_argument("--flow-early-min-delta", type=float, default=1e-4)
    p.add_argument("--flow-early-ema-alpha", type=float, default=0.05)
    p.add_argument("--no-flow-restore-best", dest="flow_restore_best", action="store_false")
    p.set_defaults(flow_restore_best=True)
    p.add_argument("--flow-ode-steps", type=int, default=64)
    p.add_argument("--flow-likelihood-exact-divergence", action="store_true")
    p.add_argument("--h-batch-size", type=int, default=65536)
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

    p.add_argument(
        "--pr-project",
        action="store_true",
        help="Embed native x through a PR autoencoder; estimators use projected x, GT LLR/Hellinger stay native 2D MoG.",
    )
    p.add_argument(
        "--pr-dim",
        type=int,
        default=10,
        help="Target observation dimension after PR embedding (must exceed native x_dim, default 2).",
    )
    p.add_argument(
        "--pr-output-npz",
        type=Path,
        default=None,
        help="Projected dataset path. Default: <native_dir>/pr_xdim<d>/random_mog_categorical_pr.npz.",
    )
    p.add_argument(
        "--pr-use-cache",
        action="store_true",
        help="Forward to project_dataset_pr_autoencoder.py --use-cache.",
    )
    p.add_argument(
        "--pr-cache-dir",
        type=str,
        default="data/pr_autoencoder_cache",
        help="PR autoencoder checkpoint cache directory (forwarded to projector).",
    )
    p.add_argument("--pr-train-epochs", type=int, default=None)
    p.add_argument("--pr-train-samples", type=int, default=None)
    p.add_argument("--pr-train-batch-size", type=int, default=None)
    p.add_argument("--pr-train-lr", type=float, default=None)
    p.add_argument("--pr-lambda-pr", type=float, default=None)
    p.add_argument("--pr-eps", type=float, default=None)
    p.add_argument("--pr-hidden1", type=int, default=None)
    p.add_argument("--pr-hidden2", type=int, default=None)
    p.add_argument(
        "--pr-skip-viz",
        action="store_true",
        help="Forward --skip-viz to project_dataset_pr_autoencoder.py (no pr_projection_summary figures).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    methods = _parse_methods(str(args.methods))
    if int(args.num_categories) < 2:
        raise ValueError("--num-categories must be >= 2.")
    if args.dataset_npz is None:
        args.dataset_npz = _default_dataset_npz(int(args.num_categories))
    if str(args.device).strip().lower() == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this project requires --device cuda for this run.")

    t0 = time.time()
    _ensure_dataset(args)

    native_npz = _abs_without_resolving_symlinks(Path(args.dataset_npz))
    native_bundle = load_shared_dataset_npz(native_npz)
    native_meta = dict(native_bundle.meta)
    if str(native_meta.get("dataset_family", "")) != "random_mog_categorical":
        raise ValueError(f"Expected random_mog_categorical NPZ, got {native_meta.get('dataset_family')!r}.")
    if str(native_meta.get("theta_type", "")) != "categorical":
        raise ValueError(f"Expected categorical theta_type, got {native_meta.get('theta_type')!r}.")
    native_x_dim = int(native_bundle.x_all.shape[1])
    if native_x_dim != 2:
        raise ValueError(
            f"This script expects native 2D observations (x_dim=2); got x_dim={native_x_dim}."
        )

    pr_project = bool(args.pr_project)
    pr_out_resolved: Path | None = None
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
        if int(work_bundle.x_all.shape[0]) != int(native_bundle.x_all.shape[0]):
            raise ValueError(
                f"Native and projected NPZ row counts disagree: native={native_bundle.x_all.shape[0]} "
                f"projected={work_bundle.x_all.shape[0]}."
            )
        if int(work_bundle.x_all.shape[1]) != int(args.pr_dim):
            raise ValueError(
                f"Projected x_dim mismatch: expected pr_dim={args.pr_dim}, got {work_bundle.x_all.shape[1]}."
            )
        nt = np.asarray(native_bundle.theta_all, dtype=np.float64)
        wt = np.asarray(work_bundle.theta_all, dtype=np.float64)
        if nt.shape != wt.shape or float(np.max(np.abs(nt - wt))) > 1e-5:
            raise ValueError("Native vs projected theta_all mismatch (expect identical rows).")
    else:
        work_bundle = native_bundle

    n_pool = int(native_bundle.theta_all.shape[0])
    if int(args.n_ref) > n_pool:
        raise ValueError(f"--n-ref={args.n_ref} exceeds dataset rows n_total={n_pool}.")

    k_cat = int(native_meta.get("num_categories", 5))
    _, _, _, _, _, bin_idx_all = prepare_categorical_binning_for_convergence(native_bundle.theta_all, k_cat)
    rng = np.random.default_rng(int(args.run_seed))
    if not args.no_plot:
        dbg_dir = _default_debug_dir(args)
        plot_base = args.plot_path
        if plot_base is None:
            plot_base = dbg_dir / "dataset_2d_scatter"
        else:
            plot_base = _abs_without_resolving_symlinks(Path(plot_base))
        plot_rng = np.random.default_rng(int(args.run_seed) + 913_409)
        scatter_title = (
            rf"random_mog_categorical, $K={k_cat}$ — native 2D source ($n={n_pool}$)"
            if pr_project
            else None
        )
        _save_2d_dataset_scatter(
            native_bundle.x_all,
            bin_idx_all,
            k_cat=k_cat,
            out_path=Path(plot_base),
            rng=plot_rng,
            max_points=int(args.plot_max_points),
            title=scatter_title,
        )
        print(f"[debug-xflow] saved dataset scatter {plot_base}.svg / .png", flush=True)
    perm = rng.permutation(n_pool)
    sub_native, bins, source_indices = _subset_bundle(
        native_bundle, perm=perm, n_ref=int(args.n_ref), bin_idx_all=bin_idx_all
    )
    sub_work, _, _ = _subset_bundle(
        work_bundle, perm=perm, n_ref=int(args.n_ref), bin_idx_all=bin_idx_all
    )

    dev = require_device(str(args.device))
    torch.manual_seed(int(args.run_seed))
    np.random.seed(int(args.run_seed))

    train_theta = _as_2d(sub_work.theta_train)
    val_theta = _as_2d(sub_work.theta_validation)
    theta_all = _as_2d(sub_work.theta_all)
    x_train = np.asarray(sub_work.x_train, dtype=np.float64)
    x_val = np.asarray(sub_work.x_validation, dtype=np.float64)
    x_all = np.asarray(sub_work.x_all, dtype=np.float64)
    x_native_all = np.asarray(sub_native.x_all, dtype=np.float64)

    print("[debug-xflow] computing ground-truth c_matrix from dataset meta", flush=True)
    true_c_matrix = _compute_true_conditional_loglik_matrix(x_native_all, theta_all, native_meta)
    true_delta_l = HMatrixEstimator.compute_delta_l(true_c_matrix)

    gen_ds_h = build_dataset_from_meta(dict(native_meta))
    hellinger_gt_sq_category = _hellinger_gt_sq_category_matrix(gen_ds_h)

    results: dict[str, dict[str, object]] = {}
    metrics_by_method: dict[str, dict[str, float]] = {}
    bins_train = np.asarray(bins[: int(sub_work.theta_train.shape[0])], dtype=np.int64)
    for method_name in methods:
        if method_name == "x_flow":
            result = _train_x_flow_delta(
                args,
                dev=dev,
                theta_train=train_theta,
                x_train=x_train,
                theta_val=val_theta,
                x_val=x_val,
                theta_all=theta_all,
                x_all=x_all,
            )
        elif method_name == "binary_classifier":
            result = _train_binary_classifier_delta(
                args,
                x_train=x_train,
                bins_train=bins_train,
                x_all=x_all,
                bins_all=bins,
                k_cat=k_cat,
            )
        elif method_name in ("linear_x_flow_t", "xflow_sir_lrank"):
            result = _train_linear_x_flow_delta(
                args,
                method_name=method_name,
                dev=dev,
                theta_train=train_theta,
                x_train=x_train,
                theta_val=val_theta,
                x_val=x_val,
                theta_all=theta_all,
                x_all=x_all,
            )
        else:
            raise RuntimeError(f"Unhandled method {method_name!r}.")
        delta = np.asarray(result["delta_l"], dtype=np.float64)
        results[method_name] = result
        metrics_by_method[method_name] = _llr_comparison_metrics(delta, true_delta_l)

    hellinger_cat_by_method: dict[str, np.ndarray] = {}
    hellinger_sample_by_method: dict[str, np.ndarray] = {}
    for method_name in methods:
        delta_m = np.asarray(results[method_name]["delta_l"], dtype=np.float64)
        h_dir_m = _h_sq_directed_from_delta_l(delta_m)
        hellinger_sample_by_method[method_name] = _h_sq_sym_sample_from_delta_l(delta_m)
        hellinger_cat_by_method[method_name] = _h_sq_category_from_sample_directed(
            h_dir_m, bins, k_cat=int(k_cat)
        )
        metrics_by_method[method_name].update(
            _hellinger_comparison_metrics_cat(hellinger_cat_by_method[method_name], hellinger_gt_sq_category)
        )

    out_path = args.output_npz
    if out_path is None:
        if pr_project:
            out_path = _default_debug_dir(args) / f"llr_debug_pr{int(args.pr_dim)}_n{int(args.n_ref)}.npz"
        else:
            out_path = _default_debug_dir(args) / f"llr_debug_n{int(args.n_ref)}.npz"
    out_path = Path(out_path)
    out_path_display = _abs_without_resolving_symlinks(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    llr_fig_base = out_path.parent / "llr_est_vs_true_all"
    _save_llr_est_vs_true_figure(
        {name: np.asarray(res["delta_l"], dtype=np.float64) for name, res in results.items()},
        true_delta_l,
        out_base=llr_fig_base,
        metrics_by_method=metrics_by_method,
    )
    print(f"[debug-xflow] saved LLR comparison {llr_fig_base}.svg / .png", flush=True)

    if not args.no_plot:
        h2_fig_base = out_path.parent / "hellinger_est_vs_gt_all"
        _save_hellinger_est_vs_gt_figure(
            hellinger_cat_by_method,
            hellinger_gt_sq_category,
            out_base=h2_fig_base,
            metrics_by_method=metrics_by_method,
        )
        print(f"[debug-xflow] saved Hellinger comparison {h2_fig_base}.svg / .png", flush=True)

    method_names = np.asarray(list(results.keys()), dtype=object)
    delta_stack = np.stack([np.asarray(results[m]["delta_l"], dtype=np.float64) for m in method_names], axis=0)
    c_stack = np.stack(
        [
            np.asarray(results[m]["c_matrix"], dtype=np.float64)
            if results[m].get("c_matrix") is not None
            else np.full_like(true_c_matrix, np.nan, dtype=np.float64)
            for m in method_names
        ],
        axis=0,
    )
    save_payload: dict[str, object] = dict(
        theta_used=np.asarray(theta_all, dtype=np.float64),
        category_labels=np.asarray(bins, dtype=np.int64),
        source_indices=np.asarray(source_indices, dtype=np.int64),
        native_dataset_npz=np.asarray([str(native_npz)], dtype=object),
        work_dataset_npz=np.asarray(
            [str(pr_out_resolved) if pr_out_resolved is not None else str(native_npz)], dtype=object
        ),
        pr_projected=np.bool_(pr_project),
        pr_dim=np.int64(int(args.pr_dim) if pr_project else int(native_x_dim)),
        x_used=np.asarray(x_all, dtype=np.float64),
        x_native_for_gt=np.asarray(x_native_all, dtype=np.float64),
        method_names=method_names,
        c_matrices=c_stack,
        delta_l_matrices=delta_stack,
        true_c_matrix=np.asarray(true_c_matrix, dtype=np.float64),
        true_delta_l_matrix=np.asarray(true_delta_l, dtype=np.float64),
        llr_mae_all_by_method=np.asarray([metrics_by_method[m]["llr_mae_all"] for m in method_names], dtype=np.float64),
        llr_rmse_all_by_method=np.asarray([metrics_by_method[m]["llr_rmse_all"] for m in method_names], dtype=np.float64),
        llr_bias_all_by_method=np.asarray([metrics_by_method[m]["llr_bias_all"] for m in method_names], dtype=np.float64),
        llr_std_err_all_by_method=np.asarray([metrics_by_method[m]["llr_std_err_all"] for m in method_names], dtype=np.float64),
        llr_pearson_r_all_by_method=np.asarray([metrics_by_method[m]["llr_pearson_r_all"] for m in method_names], dtype=np.float64),
        llr_mae_offdiag_by_method=np.asarray([metrics_by_method[m]["llr_mae_offdiag"] for m in method_names], dtype=np.float64),
        llr_rmse_offdiag_by_method=np.asarray([metrics_by_method[m]["llr_rmse_offdiag"] for m in method_names], dtype=np.float64),
        llr_bias_offdiag_by_method=np.asarray([metrics_by_method[m]["llr_bias_offdiag"] for m in method_names], dtype=np.float64),
        llr_std_err_offdiag_by_method=np.asarray([metrics_by_method[m]["llr_std_err_offdiag"] for m in method_names], dtype=np.float64),
        llr_pearson_r_offdiag_by_method=np.asarray([metrics_by_method[m]["llr_pearson_r_offdiag"] for m in method_names], dtype=np.float64),
        llr_rmse_offdiag_true_in_m8_p8_by_method=np.asarray(
            [metrics_by_method[m]["llr_rmse_offdiag_true_in_m8_p8"] for m in method_names], dtype=np.float64
        ),
        llr_pearson_r_offdiag_true_in_m8_p8_by_method=np.asarray(
            [metrics_by_method[m]["llr_pearson_r_offdiag_true_in_m8_p8"] for m in method_names], dtype=np.float64
        ),
        hellinger_gt_sq_category=np.asarray(hellinger_gt_sq_category, dtype=np.float64),
        hellinger_est_sq_category_by_method=np.stack(
            [np.asarray(hellinger_cat_by_method[str(m)], dtype=np.float64) for m in method_names], axis=0
        ),
        hellinger_est_sq_sample_by_method=np.stack(
            [np.asarray(hellinger_sample_by_method[str(m)], dtype=np.float64) for m in method_names], axis=0
        ),
        hellinger_mae_offdiag_cat_by_method=np.asarray(
            [metrics_by_method[m]["hellinger_mae_offdiag_cat"] for m in method_names], dtype=np.float64
        ),
        hellinger_rmse_offdiag_cat_by_method=np.asarray(
            [metrics_by_method[m]["hellinger_rmse_offdiag_cat"] for m in method_names], dtype=np.float64
        ),
        hellinger_bias_offdiag_cat_by_method=np.asarray(
            [metrics_by_method[m]["hellinger_bias_offdiag_cat"] for m in method_names], dtype=np.float64
        ),
        hellinger_pearson_r_offdiag_cat_by_method=np.asarray(
            [metrics_by_method[m]["hellinger_pearson_r_offdiag_cat"] for m in method_names], dtype=np.float64
        ),
        dataset_npz=np.asarray([str(native_npz)], dtype=object),
        method=method_names,
        flow_arch=np.asarray([str(args.flow_arch)], dtype=object),
        flow_scheduler=np.asarray([str(args.flow_scheduler)], dtype=object),
        flow_ode_steps=np.int64(int(args.flow_ode_steps)),
        flow_likelihood_exact_divergence=np.bool_(args.flow_likelihood_exact_divergence),
        n_ref=np.int64(int(args.n_ref)),
        run_seed=np.int64(int(args.run_seed)),
        wall_seconds=np.float64(time.time() - t0),
    )
    if "x_flow" in results:
        xflow = results["x_flow"]
        xflow_c = np.asarray(xflow["c_matrix"], dtype=np.float64)
        xflow_delta = np.asarray(xflow["delta_l"], dtype=np.float64)
        bin_ll = lxf_bin_likelihood_hellinger(xflow_c, bins, k_cat)
        train_out = xflow.get("train_out", {})
        m = metrics_by_method["x_flow"]
        save_payload.update(
            c_matrix=xflow_c,
            delta_l_matrix=xflow_delta,
            bin_log_likelihood=np.asarray(bin_ll["bin_log_likelihood"], dtype=np.float64),
            bin_delta_l_matrix=np.asarray(bin_ll["bin_delta_l"], dtype=np.float64),
            bin_counts=np.asarray(bin_ll["bin_counts"], dtype=np.int64),
            score_train_losses=np.asarray(train_out.get("train_losses", []), dtype=np.float64),
            score_val_losses=np.asarray(train_out.get("val_losses", []), dtype=np.float64),
            score_val_monitor_losses=np.asarray(train_out.get("val_monitor_losses", []), dtype=np.float64),
            score_best_epoch=np.int64(train_out.get("best_epoch", -1)),
            score_stopped_epoch=np.int64(train_out.get("stopped_epoch", -1)),
            score_stopped_early=np.bool_(train_out.get("stopped_early", False)),
            score_best_val_smooth=np.float64(train_out.get("best_val_loss", np.nan)),
            llr_mae_all=np.float64(m["llr_mae_all"]),
            llr_rmse_all=np.float64(m["llr_rmse_all"]),
            llr_bias_all=np.float64(m["llr_bias_all"]),
            llr_std_err_all=np.float64(m["llr_std_err_all"]),
            llr_pearson_r_all=np.float64(m["llr_pearson_r_all"]),
            llr_mae_offdiag=np.float64(m["llr_mae_offdiag"]),
            llr_rmse_offdiag=np.float64(m["llr_rmse_offdiag"]),
            llr_bias_offdiag=np.float64(m["llr_bias_offdiag"]),
            llr_std_err_offdiag=np.float64(m["llr_std_err_offdiag"]),
            llr_pearson_r_offdiag=np.float64(m["llr_pearson_r_offdiag"]),
            llr_rmse_offdiag_true_in_m8_p8=np.float64(m["llr_rmse_offdiag_true_in_m8_p8"]),
            llr_pearson_r_offdiag_true_in_m8_p8=np.float64(m["llr_pearson_r_offdiag_true_in_m8_p8"]),
        )
    if "binary_classifier" in results:
        clf_res = results["binary_classifier"]
        save_payload.update(
            classifier_valid_pairs=np.asarray(clf_res["classifier_valid_pairs"], dtype=bool),
            classifier_stats=np.asarray([str(clf_res["classifier_stats"])], dtype=object),
        )
    if "xflow_sir_lrank" in results:
        sir_meta = results["xflow_sir_lrank"].get("sir_meta", {})
        if sir_meta:
            save_payload.update(
                sir_dim=np.int64(sir_meta["sir_dim"]),
                sir_num_bins=np.int64(sir_meta["sir_num_bins"]),
                sir_ridge=np.float64(sir_meta["sir_ridge"]),
                sir_components=np.asarray(sir_meta["sir_components"], dtype=np.float64),
                sir_x_mean=np.asarray(sir_meta["sir_x_mean"], dtype=np.float64),
                sir_eigenvalues=np.asarray(sir_meta["sir_eigenvalues"], dtype=np.float64),
                sir_bin_counts=np.asarray(sir_meta["sir_bin_counts"], dtype=np.int64),
                sir_theta_edges=np.asarray(sir_meta["sir_theta_edges"], dtype=np.float64),
            )
    np.savez_compressed(out_path, **save_payload)
    print(f"[debug-xflow] saved {out_path_display}", flush=True)


if __name__ == "__main__":
    main()
