#!/usr/bin/env python3
"""Diagnose the KNN diagonal Gaussian x-prior used by ``x_flow_reg`` regularization.

This script isolates the first part of the regularizer used by
``train_conditional_x_flow_model``: estimate ``p_KNN(x | theta)`` from training
data, where the mean is a theta-KNN Gaussian-kernel average and the diagonal
variance is the global training residual variance around that KNN mean.

It intentionally does not train a flow model. The goal is to inspect the
distribution that a large regularization weight (for example lambda=1000)
pushes the velocity field toward.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from global_setting import DATA_DIR
from fisher.cli_shared_fisher import add_dataset_arguments
from fisher.dataset_family_recipes import (
    assert_no_legacy_dataset_cli_flags,
    format_resolved_family_summary,
)
from fisher.shared_dataset_io import SharedDatasetBundle, load_shared_dataset_npz, meta_dict_from_args
from fisher.shared_fisher_est import build_dataset_from_args, build_dataset_from_meta, require_device
from fisher.trainers import KnnDiagGaussianXPrior


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Isolate and visualize the KNN diagonal Gaussian x-prior used by x_flow_reg regularization."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_dataset_arguments(p)
    p.add_argument(
        "--dataset-npz",
        type=str,
        default="",
        help="Optional shared dataset NPZ to load. If omitted, a dataset is generated from dataset CLI args.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(DATA_DIR) / "knn_x_prior_diagnostic"),
        help="Directory for diagnostic plots and tables.",
    )
    p.add_argument("--device", type=str, default="cuda", help="Device for the KNN query tensors.")
    p.add_argument("--knn-k", type=int, default=32, help="KNN k used by the x_flow_reg prior.")
    p.add_argument(
        "--bandwidth-floor",
        type=float,
        default=1e-6,
        help="Minimum KNN Gaussian-kernel bandwidth in theta space.",
    )
    p.add_argument(
        "--variance-floor",
        type=float,
        default=1e-6,
        help="Minimum global diagonal residual variance.",
    )
    p.add_argument(
        "--no-weighted-var-correction",
        action="store_true",
        help=(
            "Mirror the trainer flag shape. Currently the trainer stores this value but uses global "
            "residual variance either way."
        ),
    )
    p.add_argument(
        "--eval-grid-size",
        type=int,
        default=300,
        help="Number of theta grid points where the estimated prior is evaluated.",
    )
    p.add_argument(
        "--plot-dims",
        type=str,
        default="0,1,2,3",
        help="Comma-separated x dimensions to plot individually.",
    )
    p.add_argument(
        "--max-train",
        type=int,
        default=0,
        help="Optional cap on training rows used by the estimator; 0 means use the full train split.",
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        default=-1,
        help="If --max-train is used, seed for selecting the capped training subset; -1 uses --seed.",
    )
    p.add_argument(
        "--mean-all-dims-panels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write a single figure with one subplot per x dimension: true tuning-curve mean vs KNN mean "
            "as a function of theta."
        ),
    )
    p.add_argument(
        "--mean-panels-ncols",
        type=int,
        default=10,
        help="Number of columns in the all-dimensions mean grid (rows follow from x_dim / ncols).",
    )
    p.add_argument(
        "--binned-compare",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare KNN against an equal-width theta-bin mean estimator with shared residual variance.",
    )
    p.add_argument(
        "--bin-mean-n-bins",
        type=int,
        default=10,
        help="Number of equal-width theta bins for the binned mean estimator.",
    )
    return p.parse_args(argv)


def _make_generated_bundle(args: argparse.Namespace) -> tuple[SharedDatasetBundle, Any]:
    assert_no_legacy_dataset_cli_flags(sys.argv[1:])
    # ``validate_dataset_sample_args`` mutates args by applying the family recipe.
    from fisher.shared_fisher_est import validate_dataset_sample_args

    validate_dataset_sample_args(args)
    print("[data] Resolved family configuration:")
    print(format_resolved_family_summary(args))

    np.random.seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    dataset = build_dataset_from_args(args)
    theta_all, x_all = dataset.sample_joint(int(args.n_total))
    perm = rng.permutation(int(args.n_total))
    train_frac = float(args.train_frac)
    if train_frac >= 1.0:
        n_train = int(args.n_total)
    else:
        n_train = int(train_frac * int(args.n_total))
        n_train = min(max(n_train, 1), int(args.n_total) - 1)
    tr_idx = perm[:n_train]
    val_idx = perm[n_train:]
    meta = meta_dict_from_args(args)
    bundle = SharedDatasetBundle(
        meta=meta,
        theta_all=np.asarray(theta_all, dtype=np.float64),
        x_all=np.asarray(x_all, dtype=np.float64),
        train_idx=np.asarray(tr_idx, dtype=np.int64),
        validation_idx=np.asarray(val_idx, dtype=np.int64),
        theta_train=np.asarray(theta_all[tr_idx], dtype=np.float64),
        x_train=np.asarray(x_all[tr_idx], dtype=np.float64),
        theta_validation=np.asarray(theta_all[val_idx], dtype=np.float64),
        x_validation=np.asarray(x_all[val_idx], dtype=np.float64),
    )
    return bundle, dataset


def _load_or_generate(args: argparse.Namespace) -> tuple[SharedDatasetBundle, Any]:
    if str(args.dataset_npz).strip():
        bundle = load_shared_dataset_npz(args.dataset_npz)
        dataset = build_dataset_from_meta(bundle.meta)
        return bundle, dataset
    return _make_generated_bundle(args)


def _cap_train_split(
    theta_train: np.ndarray,
    x_train: np.ndarray,
    *,
    max_train: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(theta_train.shape[0])
    if int(max_train) <= 0 or int(max_train) >= n:
        return theta_train, x_train
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(n, size=int(max_train), replace=False))
    return theta_train[idx], x_train[idx]


def _parse_dims(s: str, x_dim: int) -> list[int]:
    dims: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        d = int(part)
        if d < 0 or d >= int(x_dim):
            raise ValueError(f"plot dim {d} is outside [0, {int(x_dim) - 1}]")
        dims.append(d)
    return dims or [0]


def _query_prior(
    prior: KnnDiagGaussianXPrior,
    theta_eval: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    mu_parts: list[np.ndarray] = []
    var_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(theta_eval.shape[0]), int(batch_size)):
            th = torch.from_numpy(theta_eval[start : start + batch_size].astype(np.float32)).to(device)
            mu, var = prior.query(th)
            mu_parts.append(mu.detach().cpu().numpy().astype(np.float64))
            var_parts.append(var.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(mu_parts, axis=0), np.concatenate(var_parts, axis=0)


def _true_diag_variance(dataset: Any, theta: np.ndarray) -> np.ndarray | None:
    if not hasattr(dataset, "covariance"):
        return None
    cov = np.asarray(dataset.covariance(theta), dtype=np.float64)
    if cov.ndim != 3:
        return None
    return np.diagonal(cov, axis1=1, axis2=2)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if int(mask.sum()) < 2:
        return float("nan")
    aa = aa[mask]
    bb = bb[mask]
    if float(np.std(aa)) <= 0.0 or float(np.std(bb)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _theta_bin_indices(theta: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    theta_flat = np.asarray(theta, dtype=np.float64).reshape(-1)
    edges = np.asarray(bin_edges, dtype=np.float64).reshape(-1)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bin_edges must be one-dimensional with at least two entries.")
    idx = np.searchsorted(edges, theta_flat, side="right") - 1
    return np.clip(idx, 0, int(edges.size) - 2).astype(np.int64)


def _fill_empty_bin_means(mu_bin: np.ndarray, counts: np.ndarray) -> np.ndarray:
    filled = np.asarray(mu_bin, dtype=np.float64).copy()
    counts_arr = np.asarray(counts, dtype=np.int64).reshape(-1)
    nonempty = np.flatnonzero(counts_arr > 0)
    if nonempty.size == 0:
        raise ValueError("Cannot fill binned means: every theta bin is empty.")
    for b in np.flatnonzero(counts_arr <= 0):
        nearest = int(nonempty[np.argmin(np.abs(nonempty - int(b)))])
        filled[int(b)] = filled[nearest]
    return filled


def _estimate_binned_x_prior(
    *,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_eval: np.ndarray,
    theta_low: float,
    theta_high: float,
    n_bins: int,
    variance_floor: float,
) -> dict[str, np.ndarray | int]:
    """Equal-width theta-bin mean with one shared diagonal residual variance."""
    if int(n_bins) < 1:
        raise ValueError("--bin-mean-n-bins must be >= 1.")
    theta_np = np.asarray(theta_train, dtype=np.float64).reshape(-1, 1)
    x_np = np.asarray(x_train, dtype=np.float64)
    if x_np.ndim != 2 or x_np.shape[0] != theta_np.shape[0]:
        raise ValueError("Binned estimator expects theta_train length N and x_train shape (N, d).")
    if not float(theta_high) > float(theta_low):
        raise ValueError("theta_high must be greater than theta_low for equal-width bins.")

    bin_edges = np.linspace(float(theta_low), float(theta_high), int(n_bins) + 1, dtype=np.float64)
    train_bins = _theta_bin_indices(theta_np, bin_edges)
    x_dim = int(x_np.shape[1])
    sums = np.zeros((int(n_bins), x_dim), dtype=np.float64)
    counts = np.bincount(train_bins, minlength=int(n_bins)).astype(np.int64)
    np.add.at(sums, train_bins, x_np)

    mu_bin = np.zeros_like(sums)
    nonempty = counts > 0
    mu_bin[nonempty] = sums[nonempty] / counts[nonempty, None]
    empty_bin_count = int(np.sum(~nonempty))
    mu_bin_filled = _fill_empty_bin_means(mu_bin, counts)

    train_mu = mu_bin_filled[train_bins]
    residual = x_np - train_mu
    global_var = np.maximum(np.mean(residual**2, axis=0), float(variance_floor))

    eval_bins = _theta_bin_indices(theta_eval, bin_edges)
    mu_eval = mu_bin_filled[eval_bins]
    var_eval = np.broadcast_to(global_var.reshape(1, -1), mu_eval.shape).copy()
    return {
        "bin_edges": bin_edges,
        "bin_counts": counts,
        "train_bin_idx": train_bins,
        "eval_bin_idx": eval_bins,
        "mu_bin": mu_bin_filled,
        "mu_bin_raw": mu_bin,
        "mu_eval": mu_eval,
        "var_eval": var_eval,
        "global_var": global_var,
        "empty_bin_count": empty_bin_count,
    }


def _mean_var_metrics(
    *,
    prefix: str,
    mu_est: np.ndarray,
    var_est: np.ndarray,
    mu_true: np.ndarray,
    var_true: np.ndarray | None,
) -> dict[str, float]:
    mean_err = np.asarray(mu_est, dtype=np.float64) - np.asarray(mu_true, dtype=np.float64)
    var_row = np.asarray(var_est, dtype=np.float64)[0]
    metrics: dict[str, float] = {
        f"{prefix}_mean_rmse_all": float(np.sqrt(np.mean(mean_err**2))),
        f"{prefix}_mean_mae_all": float(np.mean(np.abs(mean_err))),
        f"{prefix}_mean_corr_all": _safe_corr(mu_est, mu_true),
        f"{prefix}_global_var_mean": float(np.mean(var_row)),
        f"{prefix}_global_var_min": float(np.min(var_row)),
        f"{prefix}_global_var_max": float(np.max(var_row)),
    }
    if var_true is not None:
        avg_true_var = np.mean(np.asarray(var_true, dtype=np.float64), axis=0)
        metrics[f"{prefix}_global_var_vs_true_avg_corr"] = _safe_corr(var_row, avg_true_var)
        metrics[f"{prefix}_global_var_vs_true_avg_ratio_mean"] = float(
            np.mean(var_row / np.maximum(avg_true_var, 1e-12))
        )
    return metrics


def _write_summary(
    *,
    path: str,
    args: argparse.Namespace,
    bundle: SharedDatasetBundle,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    mu_est: np.ndarray,
    var_est: np.ndarray,
    mu_true: np.ndarray,
    var_true: np.ndarray | None,
) -> dict[str, float]:
    mean_err = mu_est - mu_true
    metrics: dict[str, float] = {
        "n_total": float(bundle.theta_all.shape[0]),
        "n_train_used": float(theta_train.shape[0]),
        "x_dim": float(x_train.shape[1]),
        "knn_k_effective": float(min(int(args.knn_k), int(theta_train.shape[0]))),
        "mean_rmse_all": float(np.sqrt(np.mean(mean_err**2))),
        "mean_mae_all": float(np.mean(np.abs(mean_err))),
        "mean_corr_all": _safe_corr(mu_est, mu_true),
        "est_global_var_mean": float(np.mean(var_est[0])),
        "est_global_var_min": float(np.min(var_est[0])),
        "est_global_var_max": float(np.max(var_est[0])),
    }
    if var_true is not None:
        avg_true_var = np.mean(var_true, axis=0)
        metrics.update(
            {
                "true_avg_var_mean": float(np.mean(avg_true_var)),
                "true_avg_var_min": float(np.min(avg_true_var)),
                "true_avg_var_max": float(np.max(avg_true_var)),
                "global_var_vs_true_avg_corr": _safe_corr(var_est[0], avg_true_var),
                "global_var_vs_true_avg_ratio_mean": float(np.mean(var_est[0] / np.maximum(avg_true_var, 1e-12))),
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("diagnose_knn_x_prior\n")
        f.write(f"dataset_npz: {str(args.dataset_npz).strip() or '<generated in-memory>'}\n")
        f.write(f"dataset_family: {bundle.meta.get('dataset_family')}\n")
        f.write(f"theta_low: {bundle.meta.get('theta_low')}  theta_high: {bundle.meta.get('theta_high')}\n")
        f.write(f"train_frac: {bundle.meta.get('train_frac')}\n")
        f.write(f"knn_k_requested: {int(args.knn_k)}\n")
        f.write(f"bandwidth_floor: {float(args.bandwidth_floor)}\n")
        f.write(f"variance_floor: {float(args.variance_floor)}\n")
        f.write(f"weighted_var_correction: {not bool(args.no_weighted_var_correction)}\n")
        f.write("\n# Metrics\n")
        for key in sorted(metrics):
            f.write(f"{key}: {metrics[key]:.12g}\n")
    return metrics


def _save_eval_csv(
    *,
    path: str,
    theta_eval: np.ndarray,
    mu_est: np.ndarray,
    var_est: np.ndarray,
    mu_true: np.ndarray,
    var_true: np.ndarray | None,
    dims: list[int],
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["theta"]
        for d in dims:
            header.extend([f"mu_est_dim{d}", f"mu_true_dim{d}", f"var_est_dim{d}"])
            if var_true is not None:
                header.append(f"var_true_dim{d}")
        writer.writerow(header)
        for i in range(theta_eval.shape[0]):
            row: list[float] = [float(theta_eval[i, 0])]
            for d in dims:
                row.extend([float(mu_est[i, d]), float(mu_true[i, d]), float(var_est[i, d])])
                if var_true is not None:
                    row.append(float(var_true[i, d]))
            writer.writerow(row)


def _save_plots(
    *,
    output_dir: str,
    theta_eval: np.ndarray,
    theta_train: np.ndarray,
    mu_est: np.ndarray,
    var_est: np.ndarray,
    mu_true: np.ndarray,
    var_true: np.ndarray | None,
    dims: list[int],
) -> None:
    theta_flat = theta_eval.reshape(-1)
    train_theta_flat = theta_train.reshape(-1)
    order = np.argsort(theta_flat)
    theta_sorted = theta_flat[order]

    fig, axes = plt.subplots(len(dims), 2, figsize=(11.5, max(3.2, 2.7 * len(dims))), squeeze=False)
    for r, d in enumerate(dims):
        ax = axes[r, 0]
        ax.plot(theta_sorted, mu_true[order, d], color="#222222", linewidth=2.0, label="true mean")
        ax.plot(theta_sorted, mu_est[order, d], color="#1f77b4", linewidth=1.6, label="KNN mean")
        ax.scatter(
            train_theta_flat,
            np.zeros_like(train_theta_flat),
            s=4,
            color="0.6",
            alpha=0.25,
            label="train theta locations" if r == 0 else None,
        )
        ax.set_title(f"mean dim {d}")
        ax.set_xlabel("theta")
        ax.set_ylabel("x mean")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
        ax.legend(fontsize=8)

        axv = axes[r, 1]
        axv.plot(theta_sorted, var_est[order, d], color="#ff7f0e", linewidth=1.8, label="KNN global var")
        if var_true is not None:
            axv.plot(theta_sorted, var_true[order, d], color="#222222", linewidth=1.8, label="true var(theta)")
        axv.set_title(f"variance dim {d}")
        axv.set_xlabel("theta")
        axv.set_ylabel("diag variance")
        axv.grid(alpha=0.25, linestyle="--", linewidth=0.8)
        axv.legend(fontsize=8)
    fig.tight_layout()
    out = Path(output_dir) / "knn_prior_mean_variance_by_theta.png"
    fig.savefig(out, dpi=170)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    ax.scatter(mu_true.reshape(-1), mu_est.reshape(-1), s=5, alpha=0.25)
    lo = float(min(np.min(mu_true), np.min(mu_est)))
    hi = float(max(np.max(mu_true), np.max(mu_est)))
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("true mean")
    ax.set_ylabel("KNN estimated mean")
    ax.set_title("KNN prior mean vs true tuning curve")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    fig.tight_layout()
    out = Path(output_dir) / "knn_prior_mean_scatter.png"
    fig.savefig(out, dpi=170)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    dims_all = np.arange(var_est.shape[1])
    ax.plot(dims_all, var_est[0], marker="o", markersize=3, linewidth=1.2, label="KNN global residual var")
    if var_true is not None:
        ax.plot(
            dims_all,
            np.mean(var_true, axis=0),
            marker="o",
            markersize=3,
            linewidth=1.2,
            label="mean true var(theta)",
        )
    ax.set_xlabel("x dimension")
    ax.set_ylabel("diag variance")
    ax.set_title("Global diagonal variance used by regularizer")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    out = Path(output_dir) / "knn_prior_global_variance_by_dim.png"
    fig.savefig(out, dpi=170)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def _save_mean_all_dims_panels(
    *,
    output_dir: str,
    theta_eval: np.ndarray,
    mu_est: np.ndarray,
    mu_true: np.ndarray,
    ncols: int,
    est_label: str = "KNN",
    est_color: str = "#1f77b4",
    out_basename: str = "knn_prior_mean_all_dims_panels",
    title: str = "KNN mean vs true mean (per dimension, vs theta)",
) -> None:
    """One subplot per dimension: estimated mean vs true mean as functions of sorted theta."""
    theta_flat = theta_eval.reshape(-1)
    order = np.argsort(theta_flat)
    theta_s = theta_flat[order]
    mu_e = np.asarray(mu_est, dtype=np.float64)[order]
    mu_t = np.asarray(mu_true, dtype=np.float64)[order]
    x_dim = int(mu_e.shape[1])
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(x_dim / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(1.9 * ncols, 1.7 * nrows),
        sharex=True,
        squeeze=False,
    )
    for d in range(x_dim):
        r, c = divmod(d, ncols)
        ax = axes[r, c]
        r_dim = _safe_corr(mu_e[:, d], mu_t[:, d])
        rmse = float(np.sqrt(np.mean((mu_e[:, d] - mu_t[:, d]) ** 2)))
        ax.plot(theta_s, mu_t[:, d], color="#222222", linewidth=1.0, label="true" if d == 0 else None)
        ax.plot(theta_s, mu_e[:, d], color=est_color, linewidth=1.0, label=est_label if d == 0 else None)
        ax.set_title(f"dim {d}  r={r_dim:.2f}  rmse={rmse:.2f}", fontsize=6.5)
        ax.tick_params(axis="both", labelsize=5.5)
        ax.grid(alpha=0.2, linewidth=0.6)
    for d in range(x_dim, nrows * ncols):
        r, c = divmod(d, ncols)
        axes[r, c].set_visible(False)
    if x_dim > 0:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", fontsize=7, framealpha=0.9)
    fig.suptitle(title, fontsize=11, y=1.002)
    fig.tight_layout()
    out = Path(output_dir) / f"{out_basename}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _save_knn_vs_binned_mean_panels(
    *,
    output_dir: str,
    theta_eval: np.ndarray,
    mu_knn: np.ndarray,
    mu_binned: np.ndarray,
    mu_true: np.ndarray,
    ncols: int,
) -> None:
    theta_flat = theta_eval.reshape(-1)
    order = np.argsort(theta_flat)
    theta_s = theta_flat[order]
    knn = np.asarray(mu_knn, dtype=np.float64)[order]
    binned = np.asarray(mu_binned, dtype=np.float64)[order]
    true = np.asarray(mu_true, dtype=np.float64)[order]
    x_dim = int(true.shape[1])
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(x_dim / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.05 * ncols, 1.8 * nrows),
        sharex=True,
        squeeze=False,
    )
    for d in range(x_dim):
        r, c = divmod(d, ncols)
        ax = axes[r, c]
        rmse_knn = float(np.sqrt(np.mean((knn[:, d] - true[:, d]) ** 2)))
        rmse_bin = float(np.sqrt(np.mean((binned[:, d] - true[:, d]) ** 2)))
        ax.plot(theta_s, true[:, d], color="#222222", linewidth=1.0, label="true" if d == 0 else None)
        ax.plot(theta_s, knn[:, d], color="#1f77b4", linewidth=0.9, label="KNN" if d == 0 else None)
        ax.plot(theta_s, binned[:, d], color="#d62728", linewidth=0.9, label="binned" if d == 0 else None)
        ax.set_title(f"dim {d}  rmse K={rmse_knn:.2f} B={rmse_bin:.2f}", fontsize=6.5)
        ax.tick_params(axis="both", labelsize=5.5)
        ax.grid(alpha=0.2, linewidth=0.6)
    for d in range(x_dim, nrows * ncols):
        r, c = divmod(d, ncols)
        axes[r, c].set_visible(False)
    if x_dim > 0:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", fontsize=7, framealpha=0.9)
    fig.suptitle("KNN vs binned mean vs true mean (per dimension, vs theta)", fontsize=11, y=1.002)
    fig.tight_layout()
    out = Path(output_dir) / "knn_vs_binned_mean_all_dims_panels.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _save_knn_vs_binned_scatter(
    *,
    output_dir: str,
    mu_knn: np.ndarray,
    mu_binned: np.ndarray,
    mu_true: np.ndarray,
) -> None:
    true = np.asarray(mu_true, dtype=np.float64).reshape(-1)
    knn = np.asarray(mu_knn, dtype=np.float64).reshape(-1)
    binned = np.asarray(mu_binned, dtype=np.float64).reshape(-1)
    lo = float(min(np.min(true), np.min(knn), np.min(binned)))
    hi = float(max(np.max(true), np.max(knn), np.max(binned)))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), sharex=True, sharey=True)
    axes[0].scatter(true, knn, s=5, alpha=0.22, color="#1f77b4")
    axes[0].plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title("KNN mean")
    axes[0].set_xlabel("true mean")
    axes[0].set_ylabel("estimated mean")
    axes[0].grid(alpha=0.25, linestyle="--", linewidth=0.8)
    axes[1].scatter(true, binned, s=5, alpha=0.22, color="#d62728")
    axes[1].plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("binned mean")
    axes[1].set_xlabel("true mean")
    axes[1].grid(alpha=0.25, linestyle="--", linewidth=0.8)
    fig.suptitle("Estimated mean vs true tuning curve")
    fig.tight_layout()
    out = Path(output_dir) / "knn_vs_binned_mean_scatter.png"
    fig.savefig(out, dpi=180)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def _write_knn_vs_binned_summary(
    *,
    path: str,
    args: argparse.Namespace,
    bundle: SharedDatasetBundle,
    theta_train: np.ndarray,
    knn_metrics: dict[str, float],
    binned_metrics: dict[str, float],
    binned_info: dict[str, np.ndarray | int],
) -> dict[str, float]:
    metrics = dict(knn_metrics)
    metrics.update(binned_metrics)
    metrics["binned_n_bins"] = float(int(args.bin_mean_n_bins))
    metrics["binned_empty_bin_count"] = float(int(binned_info["empty_bin_count"]))
    counts = np.asarray(binned_info["bin_counts"], dtype=np.int64)
    nonempty_counts = counts[counts > 0]
    metrics["binned_min_nonempty_count"] = float(np.min(nonempty_counts)) if nonempty_counts.size else float("nan")
    metrics["binned_max_count"] = float(np.max(counts)) if counts.size else float("nan")
    with open(path, "w", encoding="utf-8") as f:
        f.write("knn_vs_binned_x_prior\n")
        f.write(f"dataset_npz: {str(args.dataset_npz).strip() or '<generated in-memory>'}\n")
        f.write(f"dataset_family: {bundle.meta.get('dataset_family')}\n")
        f.write(f"n_train_used: {int(theta_train.shape[0])}\n")
        f.write(f"knn_k_requested: {int(args.knn_k)}\n")
        f.write(f"bin_mean_n_bins: {int(args.bin_mean_n_bins)}\n")
        f.write(f"binned_empty_bin_count: {int(binned_info['empty_bin_count'])}\n")
        f.write("\n# Metrics\n")
        for key in sorted(metrics):
            f.write(f"{key}: {metrics[key]:.12g}\n")
    return metrics


def main() -> None:
    args = parse_args()
    device = require_device(str(args.device))
    if int(args.knn_k) < 1:
        raise ValueError("--knn-k must be >= 1.")
    if int(args.eval_grid_size) < 2:
        raise ValueError("--eval-grid-size must be >= 2.")
    if bool(args.binned_compare) and int(args.bin_mean_n_bins) < 1:
        raise ValueError("--bin-mean-n-bins must be >= 1.")
    os.makedirs(args.output_dir, exist_ok=True)

    bundle, dataset = _load_or_generate(args)
    theta_train = np.asarray(bundle.theta_train, dtype=np.float64)
    x_train = np.asarray(bundle.x_train, dtype=np.float64)
    subset_seed = int(args.seed if int(args.sample_seed) < 0 else args.sample_seed)
    theta_train, x_train = _cap_train_split(theta_train, x_train, max_train=int(args.max_train), seed=subset_seed)

    if theta_train.shape[0] < 1:
        raise ValueError("KNN prior diagnostic requires at least one training row.")
    if x_train.ndim != 2:
        raise ValueError("x_train must be two-dimensional.")

    theta_low = float(bundle.meta.get("theta_low", np.min(bundle.theta_all)))
    theta_high = float(bundle.meta.get("theta_high", np.max(bundle.theta_all)))
    theta_eval = np.linspace(theta_low, theta_high, int(args.eval_grid_size), dtype=np.float64).reshape(-1, 1)
    dims = _parse_dims(args.plot_dims, int(x_train.shape[1]))

    prior = KnnDiagGaussianXPrior(
        theta_train=theta_train,
        x_train=x_train,
        k=int(args.knn_k),
        bandwidth_floor=float(args.bandwidth_floor),
        variance_floor=float(args.variance_floor),
        weighted_var_correction=not bool(args.no_weighted_var_correction),
        device=device,
    )
    mu_est, var_est = _query_prior(prior, theta_eval, device=device)
    mu_true = np.asarray(dataset.tuning_curve(theta_eval), dtype=np.float64)
    var_true = _true_diag_variance(dataset, theta_eval)
    binned_info: dict[str, np.ndarray | int] | None = None
    if bool(args.binned_compare):
        binned_info = _estimate_binned_x_prior(
            theta_train=theta_train,
            x_train=x_train,
            theta_eval=theta_eval,
            theta_low=theta_low,
            theta_high=theta_high,
            n_bins=int(args.bin_mean_n_bins),
            variance_floor=float(args.variance_floor),
        )

    out_dir = Path(args.output_dir)
    metrics = _write_summary(
        path=str(out_dir / "knn_x_prior_summary.txt"),
        args=args,
        bundle=bundle,
        theta_train=theta_train,
        x_train=x_train,
        mu_est=mu_est,
        var_est=var_est,
        mu_true=mu_true,
        var_true=var_true,
    )
    _save_eval_csv(
        path=str(out_dir / "knn_x_prior_eval_grid.csv"),
        theta_eval=theta_eval,
        mu_est=mu_est,
        var_est=var_est,
        mu_true=mu_true,
        var_true=var_true,
        dims=dims,
    )
    np.savez_compressed(
        out_dir / "knn_x_prior_diagnostic.npz",
        theta_eval=theta_eval,
        theta_train=theta_train,
        x_train=x_train,
        mu_est=mu_est,
        var_est=var_est,
        mu_true=mu_true,
        var_true=np.asarray([] if var_true is None else var_true, dtype=np.float64),
        plot_dims=np.asarray(dims, dtype=np.int64),
        metrics_keys=np.asarray(sorted(metrics), dtype=object),
        metrics_values=np.asarray([metrics[k] for k in sorted(metrics)], dtype=np.float64),
        binned_enabled=np.bool_(bool(args.binned_compare)),
        binned_bin_edges=np.asarray([] if binned_info is None else binned_info["bin_edges"], dtype=np.float64),
        binned_bin_counts=np.asarray([] if binned_info is None else binned_info["bin_counts"], dtype=np.int64),
        binned_train_bin_idx=np.asarray([] if binned_info is None else binned_info["train_bin_idx"], dtype=np.int64),
        binned_eval_bin_idx=np.asarray([] if binned_info is None else binned_info["eval_bin_idx"], dtype=np.int64),
        binned_mu_bin=np.asarray([] if binned_info is None else binned_info["mu_bin"], dtype=np.float64),
        binned_mu_bin_raw=np.asarray([] if binned_info is None else binned_info["mu_bin_raw"], dtype=np.float64),
        binned_mu_eval=np.asarray([] if binned_info is None else binned_info["mu_eval"], dtype=np.float64),
        binned_var_eval=np.asarray([] if binned_info is None else binned_info["var_eval"], dtype=np.float64),
        binned_global_var=np.asarray([] if binned_info is None else binned_info["global_var"], dtype=np.float64),
        binned_empty_bin_count=np.int64(0 if binned_info is None else int(binned_info["empty_bin_count"])),
    )
    _save_plots(
        output_dir=str(out_dir),
        theta_eval=theta_eval,
        theta_train=theta_train,
        mu_est=mu_est,
        var_est=var_est,
        mu_true=mu_true,
        var_true=var_true,
        dims=dims,
    )
    if bool(getattr(args, "mean_all_dims_panels", True)):
        if int(args.mean_panels_ncols) < 1:
            raise ValueError("--mean-panels-ncols must be >= 1.")
        _save_mean_all_dims_panels(
            output_dir=str(out_dir),
            theta_eval=theta_eval,
            mu_est=mu_est,
            mu_true=mu_true,
            ncols=int(args.mean_panels_ncols),
        )
    comparison_metrics: dict[str, float] | None = None
    if binned_info is not None:
        mu_binned = np.asarray(binned_info["mu_eval"], dtype=np.float64)
        var_binned = np.asarray(binned_info["var_eval"], dtype=np.float64)
        _save_mean_all_dims_panels(
            output_dir=str(out_dir),
            theta_eval=theta_eval,
            mu_est=mu_binned,
            mu_true=mu_true,
            ncols=int(args.mean_panels_ncols),
            est_label="binned",
            est_color="#d62728",
            out_basename="binned_prior_mean_all_dims_panels",
            title="Binned mean vs true mean (per dimension, vs theta)",
        )
        _save_knn_vs_binned_mean_panels(
            output_dir=str(out_dir),
            theta_eval=theta_eval,
            mu_knn=mu_est,
            mu_binned=mu_binned,
            mu_true=mu_true,
            ncols=int(args.mean_panels_ncols),
        )
        _save_knn_vs_binned_scatter(
            output_dir=str(out_dir),
            mu_knn=mu_est,
            mu_binned=mu_binned,
            mu_true=mu_true,
        )
        comparison_metrics = _write_knn_vs_binned_summary(
            path=str(out_dir / "knn_vs_binned_summary.txt"),
            args=args,
            bundle=bundle,
            theta_train=theta_train,
            knn_metrics=_mean_var_metrics(
                prefix="knn",
                mu_est=mu_est,
                var_est=var_est,
                mu_true=mu_true,
                var_true=var_true,
            ),
            binned_metrics=_mean_var_metrics(
                prefix="binned",
                mu_est=mu_binned,
                var_est=var_binned,
                mu_true=mu_true,
                var_true=var_true,
            ),
            binned_info=binned_info,
        )
        np.savez_compressed(
            out_dir / "binned_x_prior_diagnostic.npz",
            theta_eval=theta_eval,
            theta_train=theta_train,
            x_train=x_train,
            mu_true=mu_true,
            var_true=np.asarray([] if var_true is None else var_true, dtype=np.float64),
            bin_edges=np.asarray(binned_info["bin_edges"], dtype=np.float64),
            bin_counts=np.asarray(binned_info["bin_counts"], dtype=np.int64),
            train_bin_idx=np.asarray(binned_info["train_bin_idx"], dtype=np.int64),
            eval_bin_idx=np.asarray(binned_info["eval_bin_idx"], dtype=np.int64),
            mu_bin=np.asarray(binned_info["mu_bin"], dtype=np.float64),
            mu_bin_raw=np.asarray(binned_info["mu_bin_raw"], dtype=np.float64),
            mu_binned_eval=mu_binned,
            var_binned_eval=var_binned,
            global_var=np.asarray(binned_info["global_var"], dtype=np.float64),
            empty_bin_count=np.int64(int(binned_info["empty_bin_count"])),
            metrics_keys=np.asarray(sorted(comparison_metrics), dtype=object),
            metrics_values=np.asarray([comparison_metrics[k] for k in sorted(comparison_metrics)], dtype=np.float64),
        )

    print("[knn_x_prior] Saved:")
    for name in (
        "knn_x_prior_summary.txt",
        "knn_x_prior_eval_grid.csv",
        "knn_x_prior_diagnostic.npz",
        "knn_prior_mean_variance_by_theta.png",
        "knn_prior_mean_variance_by_theta.svg",
        "knn_prior_mean_scatter.png",
        "knn_prior_mean_scatter.svg",
        "knn_prior_global_variance_by_dim.png",
        "knn_prior_global_variance_by_dim.svg",
    ):
        print(f"  - {out_dir / name}")
    if bool(getattr(args, "mean_all_dims_panels", True)):
        print(f"  - {out_dir / 'knn_prior_mean_all_dims_panels.png'}")
        print(f"  - {out_dir / 'knn_prior_mean_all_dims_panels.svg'}")
    if binned_info is not None:
        for name in (
            "binned_x_prior_diagnostic.npz",
            "knn_vs_binned_summary.txt",
            "binned_prior_mean_all_dims_panels.png",
            "binned_prior_mean_all_dims_panels.svg",
            "knn_vs_binned_mean_all_dims_panels.png",
            "knn_vs_binned_mean_all_dims_panels.svg",
            "knn_vs_binned_mean_scatter.png",
            "knn_vs_binned_mean_scatter.svg",
        ):
            print(f"  - {out_dir / name}")
    print(
        "[knn_x_prior] "
        f"mean_rmse_all={metrics['mean_rmse_all']:.6g} "
        f"mean_corr_all={metrics['mean_corr_all']:.6g} "
        f"est_global_var_mean={metrics['est_global_var_mean']:.6g}"
    )
    if comparison_metrics is not None:
        print(
            "[knn_vs_binned] "
            f"knn_rmse={comparison_metrics['knn_mean_rmse_all']:.6g} "
            f"binned_rmse={comparison_metrics['binned_mean_rmse_all']:.6g} "
            f"knn_corr={comparison_metrics['knn_mean_corr_all']:.6g} "
            f"binned_corr={comparison_metrics['binned_mean_corr_all']:.6g} "
            f"empty_bins={int(comparison_metrics['binned_empty_bin_count'])}"
        )


if __name__ == "__main__":
    main()
