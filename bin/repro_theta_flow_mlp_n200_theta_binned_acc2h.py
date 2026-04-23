#!/usr/bin/env python3
"""Minimal reproducer for theta-flow/NF convergence at fixed n=200 with theta bin-centering.

This variant adds an experiment path that converts pairwise decoding accuracy
matrices into Hellinger-distance lower-bound matrices via:
H_lb = sqrt(max(0, 2 * Acc - 1)).

This script intentionally exposes only a tiny CLI surface and fixes the rest:
- dataset_family = --dataset-family (default randamp_gaussian; also cosine_gaussian_sqrtd)
- x_dim = --x-dim (default 2)
- obs_noise_scale = 0.5 (TEMPORARY: half the family baseline noise; restore to 1.0 for default)
- n_total = 3000, train_frac = 0.7, seed = 7
- theta in [theta-low, theta-high] (default -6, 6), then snapped to 10 equal-width bin centers
- theta_field_method = theta_flow or nf
- flow_arch = mlp or soft_moe (theta_flow only)
- theta_flow auxiliary conditional likelihood loss can be controlled via
  --flow-endpoint-loss-weight and --flow-endpoint-steps (theta_flow only); study default
  weight is 0 (flow matching only) unless you pass --flow-endpoint-loss-weight here
- n_ref = 1000
- n in --n (default 200) as the sole --n-list value; --n-ref (default 1000) for reference subset
- num_theta_bins = 10
- optional ``--theta-filter-union`` (default off): when enabled, keep only
  theta in [-3.5,-2.5] U [-3.5+2pi,-2.5+2pi] and resample until n_total.
  Default behavior uses unfiltered uniform [theta-low, theta-high].

It creates a shared dataset NPZ, then runs ``bin/study_h_decoding_convergence.py``
with fixed settings and prints the resulting metrics. Method selection supports
theta-flow only, NF only, or both (into separate subdirectories).

Convergence outputs include ``h_decoding_convergence_combined.{png,svg}`` and a fixed-$x$
posterior + GT tuning diagnostic at
``sweep_runs/n_000200/diagnostics/theta_flow_single_x_posterior_hist.{png,svg}`` (embedded in the
combined figure). Additional acc->H artifacts are written to:
``h_from_accuracy_convergence_results.{csv,npz}`` and
``h_from_accuracy_matrix_n_<n>.{png,svg}``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from fisher.cli_shared_fisher import add_dataset_arguments
from fisher.shared_dataset_io import meta_dict_from_args, save_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_args

import matplotlib.pyplot as plt
import numpy as np
import torch

import visualize_h_matrix_binned as vhb

# TEMPORARY: <1.0 reduces Gaussian observation noise (see --obs-noise-scale in make_dataset.py).
# Restore to 1.0 when the low-noise experiment is done.
_TEMP_OBS_NOISE_SCALE = 0.5
# Total joint (theta, x) rows written by this repro; study requires n_pool >= max(--n, --n-ref).
_DATASET_N_TOTAL = 3000
# Fixed union filter (used when --theta-filter-union): two windows one period 2pi apart.
_THETA_UNION_A = -3.5
_THETA_UNION_B = -2.5
# Proposed joint rows per resampling batch when filtering (wide vs rare acceptance).
_THETA_UNION_CHUNK = 16_384
_THETA_UNION_MAX_ROUNDS = 100_000
_THETA_NUM_BINS = 10


def _default_output_dir(
    x_dim: int,
    dataset_family: str,
    *,
    theta_low: float,
    theta_high: float,
    n_sweep: int,
    theta_filter_union: bool,
) -> str:
    th = f"th{float(theta_low):g}_{float(theta_high):g}".replace(".", "p")
    tail = (
        f"repro_theta_flow_mlp_n{int(n_sweep)}_{dataset_family}_xdim{x_dim}_obsnoise0p5_"
        f"{th}_thetabin{_THETA_NUM_BINS}_acc2h"
    )
    if theta_filter_union:
        tail = f"{tail}_thunion2pi"
    return str(Path("data") / tail)


def _theta_snap_to_bin_centers(theta: np.ndarray, *, theta_low: float, theta_high: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Snap scalar theta values to centers of equal-width bins on [theta_low, theta_high]."""
    if float(theta_high) <= float(theta_low):
        raise ValueError(f"Invalid theta range for binning: [{theta_low}, {theta_high}]")
    edges = np.linspace(float(theta_low), float(theta_high), _THETA_NUM_BINS + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    t = np.asarray(theta, dtype=np.float64).reshape(-1)
    idx = np.searchsorted(edges, t, side="right") - 1
    idx = np.clip(idx, 0, _THETA_NUM_BINS - 1).astype(np.int64, copy=False)
    snapped = centers[idx].reshape(-1, 1)
    return snapped, edges, centers


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Minimal fixed reproducer for theta-flow/NF on randamp_gaussian or "
            "cosine_gaussian_sqrtd; convergence uses a single --n-list value from --n."
        )
    )
    p.add_argument(
        "--n",
        type=int,
        default=200,
        help="Training subset size (sole entry in --n-list for study_h_decoding; default: 200).",
    )
    p.add_argument(
        "--n-ref",
        type=int,
        default=1000,
        help="Reference subset size; must be >= --n and <= shared dataset n_total (default: 1000).",
    )
    p.add_argument(
        "--dataset-family",
        type=str,
        default="randamp_gaussian",
        choices=["randamp_gaussian", "cosine_gaussian_sqrtd"],
        help="Generative family for the shared NPZ and study (default: randamp_gaussian).",
    )
    p.add_argument(
        "--x-dim",
        type=int,
        default=2,
        help="Observation dimension x ∈ R^{d} (default 2).",
    )
    p.add_argument(
        "--theta-low",
        type=float,
        default=-6.0,
        help="Uniform theta support lower bound (default: -6).",
    )
    p.add_argument(
        "--theta-high",
        type=float,
        default=6.0,
        help="Uniform theta support upper bound (default: 6).",
    )
    p.add_argument(
        "--theta-filter-union",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Restrict rows to theta in [-3.5,-2.5] U [-3.5+2pi,-2.5+2pi], resampling joint draws "
            "until n_total (default: off; full [theta-low, theta-high])."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for generated dataset + study artifacts. "
            "If omitted, uses data/repro_theta_flow_mlp_n{n}_{dataset_family}_xdim{d}_obsnoise0p5_th{lo}_{hi}_thetabin10_acc2h "
            "with optional _thunion2pi when --theta-filter-union is enabled."
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda"],
        help="Execution device. Per repo policy this reproducer requires CUDA.",
    )
    p.add_argument(
        "--method",
        type=str,
        default="both",
        choices=["theta-flow", "nf", "both"],
        help="Which method branch to run: theta-flow, nf, or both.",
    )
    p.add_argument("--flow-epochs", type=int, default=None, help="Optional theta-flow override: --flow-epochs.")
    p.add_argument("--prior-epochs", type=int, default=None, help="Optional theta-flow override: --prior-epochs.")
    p.add_argument(
        "--flow-arch",
        type=str,
        default="mlp",
        choices=["mlp", "soft_moe"],
        help="Theta-flow posterior architecture override (default: mlp).",
    )
    p.add_argument(
        "--flow-batch-size",
        type=int,
        default=None,
        help="Optional theta-flow override: --flow-batch-size.",
    )
    p.add_argument(
        "--prior-batch-size",
        type=int,
        default=None,
        help="Optional theta-flow override: --prior-batch-size.",
    )
    p.add_argument(
        "--flow-endpoint-loss-weight",
        type=float,
        default=None,
        help=(
            "Optional theta-flow override for study_h_decoding_convergence: "
            "--flow-endpoint-loss-weight (0 disables auxiliary -mean log p(theta|x) term)."
        ),
    )
    p.add_argument(
        "--flow-endpoint-steps",
        type=int,
        default=None,
        help=(
            "Optional theta-flow override for study_h_decoding_convergence: "
            "--flow-endpoint-steps (ODE steps for auxiliary likelihood term)."
        ),
    )
    p.add_argument(
        "--flow-depth",
        type=int,
        default=None,
        help="Optional theta-flow override: --flow-depth (MLP hidden layers; 1 = single hidden block).",
    )
    p.add_argument(
        "--prior-depth",
        type=int,
        default=None,
        help="Optional theta-flow override: --prior-depth (prior flow MLP depth).",
    )
    p.add_argument(
        "--flow-moe-num-experts",
        type=int,
        default=None,
        help="Theta-flow soft_moe only: optional override for --flow-moe-num-experts.",
    )
    p.add_argument(
        "--flow-moe-router-temperature",
        type=float,
        default=None,
        help="Theta-flow soft_moe only: optional override for --flow-moe-router-temperature.",
    )
    p.add_argument("--nf-epochs", type=int, default=2000, help="NF method only: training epochs.")
    p.add_argument("--nf-batch-size", type=int, default=256, help="NF method only: batch size.")
    p.add_argument("--nf-lr", type=float, default=1e-3, help="NF method only: learning rate.")
    p.add_argument("--nf-hidden-dim", type=int, default=128, help="NF method only: hidden size.")
    p.add_argument("--nf-context-dim", type=int, default=32, help="NF method only: context size.")
    p.add_argument("--nf-transforms", type=int, default=5, help="NF method only: spline transforms.")
    p.add_argument(
        "--nf-pair-batch-size",
        type=int,
        default=65536,
        help="NF method only: pair budget per C-matrix block.",
    )
    p.add_argument("--nf-early-patience", type=int, default=300, help="NF method only: early-stop patience.")
    p.add_argument("--nf-early-min-delta", type=float, default=1e-4, help="NF method only: early min delta.")
    p.add_argument("--nf-early-ema-alpha", type=float, default=0.05, help="NF method only: EMA alpha.")
    p.add_argument(
        "--nf-prior-epochs",
        type=int,
        default=None,
        help="NF method only: optional prior-NF override for --nf-prior-epochs.",
    )
    p.add_argument(
        "--nf-prior-batch-size",
        type=int,
        default=None,
        help="NF method only: optional prior-NF override for --nf-prior-batch-size.",
    )
    p.add_argument(
        "--nf-prior-lr",
        type=float,
        default=None,
        help="NF method only: optional prior-NF override for --nf-prior-lr.",
    )
    p.add_argument(
        "--nf-prior-hidden-dim",
        type=int,
        default=None,
        help="NF method only: optional prior-NF override for --nf-prior-hidden-dim.",
    )
    p.add_argument(
        "--nf-prior-transforms",
        type=int,
        default=None,
        help="NF method only: optional prior-NF override for --nf-prior-transforms.",
    )
    p.add_argument(
        "--nf-prior-early-patience",
        type=int,
        default=None,
        help="NF method only: optional prior-NF override for --nf-prior-early-patience.",
    )
    p.add_argument(
        "--nf-prior-early-min-delta",
        type=float,
        default=None,
        help="NF method only: optional prior-NF override for --nf-prior-early-min-delta.",
    )
    p.add_argument(
        "--nf-prior-early-ema-alpha",
        type=float,
        default=None,
        help="NF method only: optional prior-NF override for --nf-prior-early-ema-alpha.",
    )
    return p


def _normalize_output_dir(raw: str) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    # Keep repo-relative paths stable for human-facing output (e.g., ./data/...).
    return _repo_root / p


def _theta_in_union_m2pi(theta: np.ndarray) -> np.ndarray:
    """True where theta is in [a,b] or [a+2pi, b+2pi] with a=-3.5, b=-2.5."""
    t = np.asarray(theta, dtype=np.float64).reshape(-1)
    tau = 2.0 * math.pi
    a, b = _THETA_UNION_A, _THETA_UNION_B
    return ((t >= a) & (t <= b)) | ((t >= a + tau) & (t <= b + tau))


def _sample_joint_respecting_theta_union(
    dataset: Any,
    n_target: int,
    *,
    chunk_size: int = _THETA_UNION_CHUNK,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw joint samples, keeping only rows whose theta lies in the fixed two-window union.

    The underlying generative is still p(x|theta) with uniform theta on [theta_low, theta_high]
    in the sampler, but we reject-and-resample on theta until we have n_target accepted rows
    (each accepted row is a valid draw from the conditional at that theta).
    """
    th_list: list[np.ndarray] = []
    x_list: list[np.ndarray] = []
    got = 0
    rounds = 0
    while got < n_target:
        rounds += 1
        if rounds > _THETA_UNION_MAX_ROUNDS:
            raise RuntimeError(
                f"theta union filter: could not collect n={n_target} joint rows after {rounds} "
                f"proposals of size {chunk_size} each; ensure [--theta-low,--theta-high] "
                f"overlaps the filter intervals or use --no-theta-filter-union."
            )
        th, xx = dataset.sample_joint(int(chunk_size))
        m = _theta_in_union_m2pi(th)
        if not np.any(m):
            continue
        th_list.append(np.asarray(th[m], dtype=np.float64, copy=False))
        x_list.append(np.asarray(xx[m], dtype=np.float64, copy=False))
        got += int(m.sum())
    theta_all = np.vstack(th_list)
    x_all = np.vstack(x_list)
    if theta_all.shape[0] < n_target:
        raise RuntimeError("internal: union filter returned fewer than n_target rows")
    return theta_all[:n_target], x_all[:n_target]


def _write_dataset(
    dataset_npz: Path,
    *,
    x_dim: int,
    dataset_family: str,
    theta_low: float,
    theta_high: float,
    theta_filter_union: bool,
) -> None:
    # Build namespace via the shared dataset parser to stay aligned with family recipes.
    ds_parser = argparse.ArgumentParser(add_help=False)
    add_dataset_arguments(ds_parser)
    ds_args = ds_parser.parse_args([])
    ds_args.dataset_family = str(dataset_family)
    ds_args.x_dim = int(x_dim)
    ds_args.theta_low = float(theta_low)
    ds_args.theta_high = float(theta_high)
    if float(ds_args.theta_high) <= float(ds_args.theta_low):
        raise ValueError(
            f"require --theta-high > --theta-low; got [{ds_args.theta_low}, {ds_args.theta_high}]."
        )
    ds_args.obs_noise_scale = float(_TEMP_OBS_NOISE_SCALE)
    ds_args.n_total = int(_DATASET_N_TOTAL)
    ds_args.train_frac = 0.7
    ds_args.seed = 7

    np.random.seed(int(ds_args.seed))
    rng = np.random.default_rng(int(ds_args.seed))
    dataset = build_dataset_from_args(ds_args)
    n_total = int(ds_args.n_total)
    if theta_filter_union:
        theta_all, x_all = _sample_joint_respecting_theta_union(dataset, n_total)
    else:
        theta_all, x_all = dataset.sample_joint(n_total)
    theta_all, theta_edges, theta_centers = _theta_snap_to_bin_centers(
        theta_all, theta_low=float(ds_args.theta_low), theta_high=float(ds_args.theta_high)
    )

    perm = rng.permutation(n_total)
    n_train = int(float(ds_args.train_frac) * n_total)
    n_train = min(max(n_train, 1), n_total - 1)
    tr_idx = perm[:n_train].astype(np.int64, copy=False)
    va_idx = perm[n_train:].astype(np.int64, copy=False)

    meta = meta_dict_from_args(ds_args)
    if hasattr(dataset, "_randamp_amp"):
        meta["randamp_mu_amp_per_dim"] = dataset._randamp_amp.tolist()
    if theta_filter_union:
        tau = 2.0 * math.pi
        meta["repro_theta_filter_union"] = True
        meta["repro_theta_filter_intervals"] = [
            [float(_THETA_UNION_A), float(_THETA_UNION_B)],
            [float(_THETA_UNION_A + tau), float(_THETA_UNION_B + tau)],
        ]
    meta["repro_theta_binned"] = True
    meta["repro_theta_num_bins"] = int(_THETA_NUM_BINS)
    meta["repro_theta_bin_edges"] = theta_edges.tolist()
    meta["repro_theta_bin_centers"] = theta_centers.tolist()
    meta["repro_theta_binning_range"] = [float(ds_args.theta_low), float(ds_args.theta_high)]
    save_shared_dataset_npz(
        str(dataset_npz),
        meta=meta,
        theta_all=theta_all,
        x_all=x_all,
        train_idx=tr_idx,
        validation_idx=va_idx,
        theta_train=theta_all[tr_idx],
        x_train=x_all[tr_idx],
        theta_validation=theta_all[va_idx],
        x_validation=x_all[va_idx],
    )


def _parse_metrics(results_csv: Path) -> tuple[int, float, float]:
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one row in {results_csv}, found {len(rows)}.")
    row = rows[0]
    n = int(row["n"])
    corr_h = float(row["corr_h_binned_vs_gt_mc"])
    corr_clf = float(row["corr_clf_vs_ref"])
    return n, corr_h, corr_clf


def _accuracy_matrix_to_h_lower_bound(acc: np.ndarray) -> np.ndarray:
    """Map pairwise accuracy to H lower bound via explicit clipping: A_clip in [0.5,1], H_lb=sqrt(2A_clip-1)."""
    a = np.asarray(acc, dtype=np.float64)
    out = np.full_like(a, np.nan, dtype=np.float64)
    finite = np.isfinite(a)
    a_clip = np.clip(a[finite], 0.5, 1.0)
    out[finite] = np.sqrt(np.clip(2.0 * a_clip - 1.0, 0.0, 1.0))
    if out.ndim == 2 and out.shape[0] == out.shape[1]:
        np.fill_diagonal(out, 0.0)
    return out


def _matrix_tick_indices(n_bins: int, *, max_ticks: int = 5) -> np.ndarray:
    if n_bins <= max_ticks:
        return np.arange(n_bins, dtype=np.int64)
    return np.unique(np.round(np.linspace(0, n_bins - 1, num=max_ticks)).astype(np.int64))


def _matrix_tick_labels(tick_idx: np.ndarray, theta_edges: np.ndarray | None) -> list[str]:
    if theta_edges is None or theta_edges.size < 2:
        return [str(int(i)) for i in tick_idx.tolist()]
    centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    return [f"{float(centers[int(i)]):.2f}" for i in tick_idx.tolist()]


def _render_acc_h_matrix_figure(
    *,
    out_dir: Path,
    n_value: int,
    h_from_acc: np.ndarray,
    h_gt_sqrt: np.ndarray,
    corr_h_from_acc: float,
    theta_edges: np.ndarray | None,
) -> tuple[Path, Path]:
    n_bins = int(h_from_acc.shape[0])
    tick_idx = _matrix_tick_indices(n_bins, max_ticks=5)
    tick_labs = _matrix_tick_labels(tick_idx, theta_edges)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), constrained_layout=True)
    vmax = float(np.nanmax(np.asarray([h_from_acc, h_gt_sqrt], dtype=np.float64)))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    im0 = axes[0].imshow(h_from_acc, origin="lower", vmin=0.0, vmax=vmax, cmap="viridis", interpolation="nearest")
    axes[0].set_title(rf"Acc-derived $\sqrt{{H^2}}$ (n={int(n_value)})", fontsize=10)
    axes[0].set_xlabel(r"$\theta$ bin", fontsize=9)
    axes[0].set_ylabel(r"$\theta$ bin", fontsize=9)
    axes[0].set_xticks(tick_idx)
    axes[0].set_yticks(tick_idx)
    axes[0].set_xticklabels(tick_labs, rotation=20, ha="right")
    axes[0].set_yticklabels(tick_labs)
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    cb0.set_label(r"$\sqrt{H^2}$", fontsize=8)

    im1 = axes[1].imshow(h_gt_sqrt, origin="lower", vmin=0.0, vmax=vmax, cmap="viridis", interpolation="nearest")
    axes[1].set_title(rf"GT MC $\sqrt{{H^2}}$ (r={float(corr_h_from_acc):.4f})", fontsize=10)
    axes[1].set_xlabel(r"$\theta$ bin", fontsize=9)
    axes[1].set_ylabel(r"$\theta$ bin", fontsize=9)
    axes[1].set_xticks(tick_idx)
    axes[1].set_yticks(tick_idx)
    axes[1].set_xticklabels(tick_labs, rotation=20, ha="right")
    axes[1].set_yticklabels(tick_labs)
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    cb1.set_label(r"$\sqrt{H^2}$", fontsize=8)

    png_path = out_dir / f"h_from_accuracy_matrix_n_{int(n_value):06d}.png"
    svg_path = png_path.with_suffix(".svg")
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def _write_accuracy_to_h_artifacts(
    *,
    out_dir: Path,
    n_target: int,
) -> tuple[Path, Path, Path, Path, int, float]:
    """Create acc->H artifacts from convergence outputs and return the n_target correlation."""
    results_npz = out_dir / "h_decoding_convergence_results.npz"
    if not results_npz.is_file():
        raise FileNotFoundError(f"Missing expected convergence NPZ: {results_npz}")
    z = np.load(results_npz, allow_pickle=True)
    required = ("clf_acc_columns", "hellinger_gt_sq_mc", "corr_h_binned_vs_gt_mc", "corr_clf_vs_ref")
    missing = [k for k in required if k not in z.files]
    if missing:
        raise KeyError(f"{results_npz} missing required keys: {missing}")
    if "n_values" in z.files:
        n_values = np.asarray(z["n_values"], dtype=np.int64).reshape(-1)
    elif "n" in z.files:
        n_values = np.asarray(z["n"], dtype=np.int64).reshape(-1)
    else:
        raise KeyError(f"{results_npz} missing both n_values and n keys.")
    clf_cols = np.asarray(z["clf_acc_columns"], dtype=np.float64)
    h_gt_sqrt = np.asarray(z["hellinger_gt_sq_mc"], dtype=np.float64)
    corr_h = np.asarray(z["corr_h_binned_vs_gt_mc"], dtype=np.float64).reshape(-1)
    corr_clf = np.asarray(z["corr_clf_vs_ref"], dtype=np.float64).reshape(-1)
    theta_edges = np.asarray(z["theta_bin_edges"], dtype=np.float64).reshape(-1) if "theta_bin_edges" in z.files else None

    if clf_cols.ndim != 3:
        raise ValueError(f"Expected clf_acc_columns to be 3D, got shape {clf_cols.shape}.")
    if clf_cols.shape[0] < n_values.size:
        raise ValueError(
            f"Need at least len(n_values) columns in clf_acc_columns, got {clf_cols.shape[0]} vs {n_values.size}."
        )
    if corr_h.size != n_values.size or corr_clf.size != n_values.size:
        raise ValueError(
            f"Correlation vector sizes must match n_values: {corr_h.size}, {corr_clf.size}, {n_values.size}."
        )

    h_from_acc_cols = np.stack([_accuracy_matrix_to_h_lower_bound(clf_cols[k]) for k in range(clf_cols.shape[0])], axis=0)
    corr_h_from_acc = np.asarray(
        [vhb.matrix_corr_offdiag_pearson(h_from_acc_cols[k], h_gt_sqrt) for k in range(n_values.size)],
        dtype=np.float64,
    )

    csv_path = out_dir / "h_from_accuracy_convergence_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "n",
                "corr_h_from_acc_lb_vs_gt_mc",
                "corr_h_binned_vs_gt_mc",
                "corr_clf_vs_ref",
            ],
        )
        w.writeheader()
        for k, n in enumerate(n_values.tolist()):
            w.writerow(
                {
                    "n": int(n),
                    "corr_h_from_acc_lb_vs_gt_mc": float(corr_h_from_acc[k]),
                    "corr_h_binned_vs_gt_mc": float(corr_h[k]),
                    "corr_clf_vs_ref": float(corr_clf[k]),
                }
            )

    npz_path = out_dir / "h_from_accuracy_convergence_results.npz"
    np.savez(
        npz_path,
        n_values=np.asarray(n_values, dtype=np.int64),
        h_from_accuracy_lb_columns=np.asarray(h_from_acc_cols, dtype=np.float64),
        corr_h_from_acc_lb_vs_gt_mc=np.asarray(corr_h_from_acc, dtype=np.float64),
        corr_h_binned_vs_gt_mc=np.asarray(corr_h, dtype=np.float64),
        corr_clf_vs_ref=np.asarray(corr_clf, dtype=np.float64),
        hellinger_gt_sq_mc=np.asarray(h_gt_sqrt, dtype=np.float64),
    )

    where = np.where(n_values == int(n_target))[0]
    if where.size < 1:
        raise ValueError(f"n_target={n_target} not found in n_values={n_values.tolist()}.")
    idx = int(where[0])
    fig_png, fig_svg = _render_acc_h_matrix_figure(
        out_dir=out_dir,
        n_value=int(n_values[idx]),
        h_from_acc=np.asarray(h_from_acc_cols[idx], dtype=np.float64),
        h_gt_sqrt=np.asarray(h_gt_sqrt, dtype=np.float64),
        corr_h_from_acc=float(corr_h_from_acc[idx]),
        theta_edges=theta_edges,
    )
    return csv_path, npz_path, fig_png, fig_svg, int(n_values[idx]), float(corr_h_from_acc[idx])


def _run_convergence(
    *,
    method: str,
    out_dir: Path,
    dataset_npz: Path,
    dataset_family: str,
    n: int,
    n_ref: int,
    args: argparse.Namespace,
) -> tuple[Path, int, float, float]:
    study_py = _repo_root / "bin" / "study_h_decoding_convergence.py"
    if not study_py.is_file():
        raise FileNotFoundError(f"Missing study script: {study_py}")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable,
        str(study_py),
        "--dataset-npz",
        str(dataset_npz),
        "--dataset-family",
        str(dataset_family),
        "--output-dir",
        str(out_dir),
        "--theta-field-method",
        str(method),
        "--n-ref",
        str(n_ref),
        "--n-list",
        str(n),
        "--num-theta-bins",
        "10",
        "--keep-intermediate",
        "--run-seed",
        "7",
        "--device",
        args.device,
    ]
    if str(method) == "theta_flow":
        cmd += ["--flow-arch", str(args.flow_arch)]
        if args.flow_epochs is not None:
            cmd += ["--flow-epochs", str(int(args.flow_epochs))]
        if args.prior_epochs is not None:
            cmd += ["--prior-epochs", str(int(args.prior_epochs))]
        if args.flow_batch_size is not None:
            cmd += ["--flow-batch-size", str(int(args.flow_batch_size))]
        if args.prior_batch_size is not None:
            cmd += ["--prior-batch-size", str(int(args.prior_batch_size))]
        if args.flow_endpoint_loss_weight is not None:
            cmd += ["--flow-endpoint-loss-weight", str(float(args.flow_endpoint_loss_weight))]
        if args.flow_endpoint_steps is not None:
            cmd += ["--flow-endpoint-steps", str(int(args.flow_endpoint_steps))]
        if args.flow_depth is not None:
            cmd += ["--flow-depth", str(int(args.flow_depth))]
        if args.prior_depth is not None:
            cmd += ["--prior-depth", str(int(args.prior_depth))]
        if args.flow_moe_num_experts is not None:
            cmd += ["--flow-moe-num-experts", str(int(args.flow_moe_num_experts))]
        if args.flow_moe_router_temperature is not None:
            cmd += ["--flow-moe-router-temperature", str(float(args.flow_moe_router_temperature))]
    else:
        cmd += [
            "--nf-epochs",
            str(int(args.nf_epochs)),
            "--nf-batch-size",
            str(int(args.nf_batch_size)),
            "--nf-lr",
            str(float(args.nf_lr)),
            "--nf-hidden-dim",
            str(int(args.nf_hidden_dim)),
            "--nf-context-dim",
            str(int(args.nf_context_dim)),
            "--nf-transforms",
            str(int(args.nf_transforms)),
            "--nf-pair-batch-size",
            str(int(args.nf_pair_batch_size)),
            "--nf-early-patience",
            str(int(args.nf_early_patience)),
            "--nf-early-min-delta",
            str(float(args.nf_early_min_delta)),
            "--nf-early-ema-alpha",
            str(float(args.nf_early_ema_alpha)),
        ]
        if args.nf_prior_epochs is not None:
            cmd += ["--nf-prior-epochs", str(int(args.nf_prior_epochs))]
        if args.nf_prior_batch_size is not None:
            cmd += ["--nf-prior-batch-size", str(int(args.nf_prior_batch_size))]
        if args.nf_prior_lr is not None:
            cmd += ["--nf-prior-lr", str(float(args.nf_prior_lr))]
        if args.nf_prior_hidden_dim is not None:
            cmd += ["--nf-prior-hidden-dim", str(int(args.nf_prior_hidden_dim))]
        if args.nf_prior_transforms is not None:
            cmd += ["--nf-prior-transforms", str(int(args.nf_prior_transforms))]
        if args.nf_prior_early_patience is not None:
            cmd += ["--nf-prior-early-patience", str(int(args.nf_prior_early_patience))]
        if args.nf_prior_early_min_delta is not None:
            cmd += ["--nf-prior-early-min-delta", str(float(args.nf_prior_early_min_delta))]
        if args.nf_prior_early_ema_alpha is not None:
            cmd += ["--nf-prior-early-ema-alpha", str(float(args.nf_prior_early_ema_alpha))]
    print(f"[repro] running convergence ({method})...", flush=True)
    result = subprocess.run(cmd, cwd=str(_repo_root), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"study_h_decoding_convergence ({method}) failed with code {result.returncode}.")
    results_csv = out_dir / "h_decoding_convergence_results.csv"
    if not results_csv.is_file():
        raise FileNotFoundError(f"Missing expected results CSV: {results_csv}")
    n_row, corr_h, corr_clf = _parse_metrics(results_csv)
    return results_csv, n_row, corr_h, corr_clf


def main() -> None:
    args = _build_parser().parse_args()
    x_dim = int(args.x_dim)
    if x_dim < 2:
        raise ValueError(f"--x-dim must be >= 2, got {x_dim}.")
    dataset_family = str(args.dataset_family)
    theta_lo = float(args.theta_low)
    theta_hi = float(args.theta_high)
    n_sweep = int(args.n)
    n_ref = int(args.n_ref)
    if n_sweep < 1:
        raise ValueError(f"--n must be >= 1, got {n_sweep}.")
    if n_ref < n_sweep:
        raise ValueError(f"require --n-ref >= --n; got n_ref={n_ref} and n={n_sweep}.")
    need = max(n_ref, n_sweep)
    if need > int(_DATASET_N_TOTAL):
        raise ValueError(
            f"max(--n, --n-ref)={need} but shared dataset n_total={_DATASET_N_TOTAL}; "
            f"raise _DATASET_N_TOTAL in {Path(__file__).name} or use smaller --n / --n-ref."
        )
    theta_filter = bool(getattr(args, "theta_filter_union", False))
    out_raw = args.output_dir if args.output_dir is not None else _default_output_dir(
        x_dim,
        dataset_family,
        theta_low=theta_lo,
        theta_high=theta_hi,
        n_sweep=n_sweep,
        theta_filter_union=theta_filter,
    )
    out_dir = _normalize_output_dir(out_raw)
    os.makedirs(out_dir, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Per repo policy, do not fallback silently.")

    n = n_sweep
    dataset_npz = out_dir / "shared_dataset.npz"
    _write_dataset(
        dataset_npz,
        x_dim=x_dim,
        dataset_family=dataset_family,
        theta_low=theta_lo,
        theta_high=theta_hi,
        theta_filter_union=theta_filter,
    )
    print(
        f"[repro] dataset_family={dataset_family} x_dim={x_dim} "
        f"theta_range=[{theta_lo}, {theta_hi}] theta_filter_union={theta_filter} "
        f"dataset_npz={dataset_npz}",
        flush=True,
    )
    method = str(args.method).strip().lower()
    run_theta = method in ("theta-flow", "both")
    run_nf = method in ("nf", "both")

    results: list[tuple[str, Path, Path, Path, Path, Path, int, float, float, float]] = []
    if run_theta:
        theta_dir = out_dir if method != "both" else out_dir / "theta_flow"
        csv_path, n_row, corr_h, corr_clf = _run_convergence(
            method="theta_flow",
            out_dir=theta_dir,
            dataset_npz=dataset_npz,
            dataset_family=dataset_family,
            n=n,
            n_ref=n_ref,
            args=args,
        )
        acc_h_csv, acc_h_npz, acc_h_fig_png, acc_h_fig_svg, _, corr_h_from_acc = _write_accuracy_to_h_artifacts(
            out_dir=theta_dir, n_target=n
        )
        results.append(
            ("theta_flow", csv_path, acc_h_csv, acc_h_npz, acc_h_fig_png, acc_h_fig_svg, n_row, corr_h, corr_clf, corr_h_from_acc)
        )
    if run_nf:
        nf_dir = out_dir if method != "both" else out_dir / "nf"
        csv_path, n_row, corr_h, corr_clf = _run_convergence(
            method="nf",
            out_dir=nf_dir,
            dataset_npz=dataset_npz,
            dataset_family=dataset_family,
            n=n,
            n_ref=n_ref,
            args=args,
        )
        acc_h_csv, acc_h_npz, acc_h_fig_png, acc_h_fig_svg, _, corr_h_from_acc = _write_accuracy_to_h_artifacts(
            out_dir=nf_dir, n_target=n
        )
        results.append(
            ("nf", csv_path, acc_h_csv, acc_h_npz, acc_h_fig_png, acc_h_fig_svg, n_row, corr_h, corr_clf, corr_h_from_acc)
        )

    print("[repro] completed.", flush=True)
    print(f"[repro] output_dir={out_dir}", flush=True)
    print(f"[repro] method={method}", flush=True)
    for tag, results_csv, acc_h_csv, acc_h_npz, acc_h_fig_png, acc_h_fig_svg, n_row, corr_h, corr_clf, corr_h_from_acc in results:
        gap = corr_h - corr_clf
        gap_acc_h = corr_h_from_acc - corr_h
        print(f"[repro][{tag}] results_csv={results_csv}", flush=True)
        print(f"[repro][{tag}] h_from_accuracy_csv={acc_h_csv}", flush=True)
        print(f"[repro][{tag}] h_from_accuracy_npz={acc_h_npz}", flush=True)
        print(f"[repro][{tag}] h_from_accuracy_matrix_png={acc_h_fig_png}", flush=True)
        print(f"[repro][{tag}] h_from_accuracy_matrix_svg={acc_h_fig_svg}", flush=True)
        print(
            "[repro][{}] n={} corr_h_binned_vs_gt_mc={:.6f} corr_clf_vs_ref={:.6f} gap(h-clf)={:.6f}".format(
                tag,
                n_row,
                corr_h,
                corr_clf,
                gap,
            ),
            flush=True,
        )
        print(
            "[repro][{}] n={} corr_h_from_acc_lb_vs_gt_mc={:.6f} delta(acc_h - h_binned)={:.6f}".format(
                tag,
                n_row,
                corr_h_from_acc,
                gap_acc_h,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
