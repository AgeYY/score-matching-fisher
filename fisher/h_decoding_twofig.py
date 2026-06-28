#!/usr/bin/env python3
"""Two-figure H/decoding convergence study.

Reuses the full-compute pipeline from ``bin/study_h_decoding_convergence.py``
but emits compact figure artifacts:

1) ``h_decoding_twofig_sweep.svg``: columns over ``--n-list`` only
   (one row per method for estimated sqrt(H)-like binned matrices, plus one
   shared bottom row for decoding) plus a footer row with GT matrices.
2) ``h_decoding_twofig_corr_nmse.svg``: correlation and normalized-MSE curves
   stacked vertically.

The sweep footer shows left = approximate GT sqrt(H^2) matrix (MC likelihood),
right = decoding matrix from the ``n_ref`` subset.
   For PR-embedded archives, that decoding GT uses native (pre-projection) ``x``
   from the source NPZ; estimated rows and sweep decoding still use embedded ``x``.

Also writes a training-loss SVG. Pass
``--visualization-only`` with the same ``--output-dir`` as a prior full run to
regenerate figures from ``h_decoding_twofig_results.npz`` without retraining.

Off-diagonal Pearson correlations and off-diagonal NMSE use
``fisher.h_binned_visualization.impute_offdiag_nan_mean`` on both matrices in each
pair (sweep vs GT / MC reference, and decoding sweep vs shared reference) before
the finite-pair mask inside each metric.

For PR-embedded archives, optional decoding correlation/NMSE vs GT compares embedded
sweep decoding matrices to the native-x GT reference (different feature spaces by design).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any, TypedDict, cast

_repo_root = Path(__file__).resolve().parent.parent
_bin_dir = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_bin_dir) not in sys.path:
    sys.path.insert(0, str(_bin_dir))

from global_setting import DATA_DIR

import matplotlib.pyplot as plt
import numpy as np

from fisher import h_decoding_convergence as conv
from fisher.hellinger_gt import (
    bin_centers_from_edges,
    estimate_hellinger_sq_grid_centers_analytic,
    estimate_hellinger_sq_grid_centers_mc,
    estimate_hellinger_sq_one_sided_mc,
    theta_centers_for_analytic_gt,
)
from fisher.decoding_native_x import decoding_x_train_all_from_native, load_native_bundle_for_pr_gt_decoding
from fisher.shared_dataset_io import SharedDatasetBundle, load_shared_dataset_npz
from fisher.h_decoding_convergence_methods import SweepSubset, _fit_sir_projection
from fisher.shared_fisher_est import build_dataset_from_meta, normalize_flow_arch, normalize_theta_field_method
from fisher.vae_ctsm_v import _vae_payload, prepare_vae_mean_features

# Valid row choices for --theta-field-rows:
# - theta_flow:mlp
# - theta_flow:film
# - theta_flow:film_fourier
# - theta_path_integral:mlp
# - theta_path_integral:film
# - theta_path_integral:film_fourier
# - x_flow:mlp
# - x_flow:film
# - x_flow:film_fourier
# - ctsm_v
# - nf
# - bin_gaussian
# - contrastive_soft / contrastive_soft_categorical (aliases per convergence)
# - linear_x_flow_t,
#   linear_x_flow_scalar_t,
#   linear_x_flow_diagonal_t,
#   linear_x_flow_diagonal_theta_t,
#   linear_x_flow_low_rank_t (full A(t) + learnable U h(U^T x) correction;
#     static orthonormal U; divergence default: --lxf-low-rank-divergence-estimator hutchinson, --lxf-hutchinson-probes 1),
#   xflow_sir_lrank (same full-x low-rank correction, but fixed raw SIR directions for U;
#     rank from --lxf-low-rank-dim, SIR bins/ridge from --sir-num-bins/--sir-ridge),
#   xflow_sir_lrank_dia (same fixed raw SIR directions, but diagonal A(t)),
#   xflow_sir_lrank_dia_theta (same fixed raw SIR directions, but diagonal A(theta,t)),
#   xflow_sir_lrank_scalar (same fixed raw SIR directions, but scalar A(t)=a(t)I),
#   xflow_sir_lrank_scalar_theta (same fixed raw SIR directions, but scalar A(theta,t)=a(theta,t)I),
#   xflow_sir_pure_lrank (fixed raw SIR directions for U; velocity U h(U^T x) only, no A or b),
#   linear_x_flow_pure_low_rank_t (velocity U h(U^T x) only; same divergence / ODE likelihood path),
#   linear_x_flow_pure_cond_low_rank_t (U(theta,t) from MLP; tr((U^T U) dh/dz) divergence; same ODE likelihood path),
#   linear_x_flow_lr_t_ts (same scheduled low-rank correction but b(theta) only; mean-regression pretrain then freeze b),
#   linear_x_flow_low_rank_randb_t,
#   sir_xflow_lrank_t (SIR preprocessing followed by linear_x_flow_low_rank_t in projected space)
#   sir_xflow (SIR preprocessing followed by x_flow on projected z)
#   sir_thetaflow (SIR preprocessing followed by theta_flow conditioning on projected z)
_FLOW_BASED_METHODS = {"theta_flow", "theta_path_integral", "x_flow", "sir_xflow", "sir_thetaflow"}
# Row specs allow ``method:arch`` when the trained estimator ultimately uses ``--flow-arch`` (includes staged embeddings).
_FLOW_ROW_ARCH_METHODS = _FLOW_BASED_METHODS
_NO_TRAIN_METHODS = {"bin_gaussian"}
_VAE_WRAPPED_METHODS = {
    "vae_x_flow": "x_flow",
    "vae_xflow_sir_lrank": "xflow_sir_lrank",
    "vae_bin_gaussian": "bin_gaussian",
    "vae_ctsm_v": "ctsm_v",
}


class CachedTwofigBundle(TypedDict, total=False):
    """Arrays and metadata loaded from ``h_decoding_twofig_results.npz`` for visualization-only reruns."""

    n: np.ndarray
    n_ref: np.ndarray
    theta_field_rows: np.ndarray
    theta_field_row_methods: np.ndarray
    theta_field_row_arches: np.ndarray
    theta_bin_centers: np.ndarray
    h_gt_sqrt: np.ndarray
    decode_ref: np.ndarray
    h_binned_sweep: np.ndarray
    decode_sweep: np.ndarray
    corr_h_binned_vs_gt_mc: np.ndarray
    nmse_h_binned_vs_gt_mc: np.ndarray
    hellinger_acc_lb_sweep: np.ndarray
    hellinger_acc_ub_sweep: np.ndarray
    corr_hellinger_lb_vs_decode_shared: np.ndarray
    corr_hellinger_ub_vs_decode_shared: np.ndarray
    corr_decode_vs_ref_shared: np.ndarray
    nmse_decode_vs_ref_shared: np.ndarray
    wall_seconds: np.ndarray
    perm_seed: np.ndarray
    convergence_base_seed: np.ndarray
    dataset_meta_seed: np.ndarray
    training_losses_root: np.ndarray
    dataset_npz: np.ndarray
    dataset_family: np.ndarray
    dataset_pool_size: np.ndarray


SIR_FIRST_DEFAULT_ROWS = "theta_path_integral,theta_flow,x_flow,linear_x_flow_t,contrastive_soft,bin_gaussian"


def _matrix_nmse_offdiag(est: np.ndarray, gt: np.ndarray) -> float:
    """Off-diagonal normalized MSE: mean((est - gt)^2) / mean(gt^2); finite pairs only.

    Uses the same off-diagonal mask as ``visualize_h_matrix_binned.matrix_corr_offdiag_pearson``.
    Twofig callers pass matrices already imputed with ``impute_offdiag_nan_mean``
    so NaN cells do not drop pairs asymmetrically.
    """
    aa = np.asarray(est, dtype=np.float64)
    bb = np.asarray(gt, dtype=np.float64)
    if aa.shape != bb.shape or aa.ndim != 2 or aa.shape[0] != aa.shape[1]:
        raise ValueError("_matrix_nmse_offdiag requires equal-shape square matrices.")
    n = aa.shape[0]
    off = ~np.eye(n, dtype=bool)
    mask = off & np.isfinite(aa) & np.isfinite(bb)
    if int(np.sum(mask)) < 1:
        return float("nan")
    ev = aa[mask]
    gv = bb[mask]
    denom = float(np.mean(gv * gv))
    if denom <= 0.0 or not np.isfinite(denom):
        return float("nan")
    mse = float(np.mean((ev - gv) ** 2))
    return float(mse / denom)


def _decode_accuracy_color_limits(*arrays: np.ndarray) -> tuple[float, float]:
    """Shared color scale for pairwise decoding accuracy heatmaps.

    Uses ``vmin = min(data_min, 0.5)`` and ``vmax = 1``, where ``data_min`` is the
    smallest finite entry across all passed matrices (e.g. sweep columns plus reference).
    """
    chunks: list[np.ndarray] = []
    for arr in arrays:
        v = np.asarray(arr, dtype=np.float64)
        flat = v[np.isfinite(v)]
        if flat.size:
            chunks.append(flat.ravel())
    if not chunks:
        return 0.5, 1.0
    data_min = float(np.min(np.concatenate(chunks)))
    vmin = min(data_min, 0.5)
    vmax = 1.0
    if vmin >= vmax:
        vmin = float(max(0.0, vmax - 1e-12))
    return vmin, vmax


def _theta_binning_mode(args: argparse.Namespace) -> str:
    return str(getattr(args, "theta_binning_mode", "theta1")).strip().lower()


def _num_theta_bins_y(args: argparse.Namespace) -> int:
    return int(getattr(args, "num_theta_bins_y", 0)) or int(args.num_theta_bins)


def _total_theta_bins_from_args(args: argparse.Namespace) -> int:
    if _theta_binning_mode(args) == "theta2_grid":
        return int(args.num_theta_bins) * _num_theta_bins_y(args)
    return int(args.num_theta_bins)


def _theta_center_array_for_axis(theta_centers: np.ndarray, n_bins: int) -> np.ndarray:
    tc = np.asarray(theta_centers, dtype=np.float64)
    if tc.ndim == 1:
        if int(tc.size) != int(n_bins):
            raise ValueError(f"theta_centers length {tc.size} must match n_bins={n_bins}.")
        return tc.reshape(-1, 1)
    if tc.ndim == 2 and int(tc.shape[0]) == int(n_bins):
        return tc
    raise ValueError(f"theta_centers shape {tc.shape} incompatible with n_bins={n_bins}.")


def _theta_axis_tick_labels(theta_centers: np.ndarray, n_bins: int) -> tuple[list[int], list[str], str]:
    tc = _theta_center_array_for_axis(theta_centers, n_bins)
    tick_idx = conv._matrix_panel_tick_indices(int(n_bins), max_ticks=5)
    tick_pos = tick_idx.tolist()
    if int(tc.shape[1]) == 1:
        labs_full = conv._format_theta_tick_labels(tc[:, 0])
        return tick_pos, [labs_full[int(i)] for i in tick_idx], r"$\theta$"
    labs = [f"{int(i)}\n({tc[int(i), 0]:.2g},{tc[int(i), 1]:.2g})" for i in tick_idx]
    return tick_pos, labs, r"flat $(\theta_1,\theta_2)$ bin"


def _nmse_h_binned_vs_gt_mc(h_sweep_arr: np.ndarray, h_gt_sqrt: np.ndarray) -> np.ndarray:
    """Shape (n_methods, n_cols): NMSE of each estimated H matrix vs GT for each n column."""
    h_sw = np.asarray(h_sweep_arr, dtype=np.float64)
    h_gt = np.asarray(h_gt_sqrt, dtype=np.float64)
    if h_sw.ndim != 4:
        raise ValueError(f"h_sweep_arr must be 4D; got {h_sw.shape}.")
    n_m, n_cols = int(h_sw.shape[0]), int(h_sw.shape[1])
    out = np.full((n_m, n_cols), np.nan, dtype=np.float64)
    h_gt_imp = conv.vhb.impute_offdiag_nan_mean(h_gt)
    for i in range(n_m):
        for j in range(n_cols):
            out[i, j] = _matrix_nmse_offdiag(
                conv.vhb.impute_offdiag_nan_mean(np.asarray(h_sw[i, j], dtype=np.float64)),
                h_gt_imp,
            )
    return out


def _nmse_decode_vs_ref_shared(decode_sweep: np.ndarray, decode_ref: np.ndarray) -> np.ndarray:
    """Shape (n_cols,): off-diagonal NMSE of shared decoding matrix vs reference for each n."""
    d_sw = np.asarray(decode_sweep, dtype=np.float64)
    d_ref = np.asarray(decode_ref, dtype=np.float64)
    if d_sw.ndim != 3:
        raise ValueError(f"decode_sweep must be 3D (n_cols, n_bins, n_bins); got {d_sw.shape}.")
    n_cols = int(d_sw.shape[0])
    out = np.full((n_cols,), np.nan, dtype=np.float64)
    d_ref_imp = conv.vhb.impute_offdiag_nan_mean(d_ref)
    for j in range(n_cols):
        out[j] = _matrix_nmse_offdiag(conv.vhb.impute_offdiag_nan_mean(np.asarray(d_sw[j], dtype=np.float64)), d_ref_imp)
    return out


def _hellinger_sqrt_to_accuracy_bounds(h_sqrt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert stored sqrt(H^2)-like matrices to Bayes accuracy lower/upper bounds."""
    h = np.asarray(h_sqrt, dtype=np.float64)
    h2 = np.clip(h * h, 0.0, 1.0)
    lb = 0.5 * (1.0 + h2)
    ub = 0.5 * (1.0 + np.sqrt(np.clip(2.0 * h2 - h2 * h2, 0.0, 1.0)))
    if h.ndim < 2 or h.shape[-1] != h.shape[-2]:
        raise ValueError(f"h_sqrt must have square matrix trailing dimensions; got {h.shape}.")
    diag = np.arange(int(h.shape[-1]))
    lb = np.asarray(lb, dtype=np.float64).copy()
    ub = np.asarray(ub, dtype=np.float64).copy()
    lb[..., diag, diag] = np.nan
    ub[..., diag, diag] = np.nan
    return lb, ub


def _corr_hellinger_bounds_vs_decode_shared(
    hellinger_acc_bound_sweep: np.ndarray,
    decode_sweep: np.ndarray,
) -> np.ndarray:
    """Shape (n_methods, n_cols): bound matrix at n correlated with same-n decoding."""
    b_sw = np.asarray(hellinger_acc_bound_sweep, dtype=np.float64)
    d_sw = np.asarray(decode_sweep, dtype=np.float64)
    if b_sw.ndim != 4:
        raise ValueError(
            f"hellinger_acc_bound_sweep must be 4D (methods, n_cols, n_bins, n_bins); got {b_sw.shape}."
        )
    if d_sw.ndim != 3:
        raise ValueError(f"decode_sweep must be 3D (n_cols, n_bins, n_bins); got {d_sw.shape}.")
    n_methods, n_cols, n_bins, n_bins2 = (int(x) for x in b_sw.shape)
    if n_bins != n_bins2:
        raise ValueError(f"hellinger_acc_bound_sweep trailing dims must be square; got {b_sw.shape}.")
    if d_sw.shape != (n_cols, n_bins, n_bins):
        raise ValueError(f"decode_sweep shape {d_sw.shape} expected ({n_cols}, {n_bins}, {n_bins}).")

    out = np.full((n_methods, n_cols), np.nan, dtype=np.float64)
    for i in range(n_methods):
        for j in range(n_cols):
            out[i, j] = conv.vhb.matrix_corr_offdiag_pearson(
                conv.vhb.impute_offdiag_nan_mean(np.asarray(b_sw[i, j], dtype=np.float64)),
                conv.vhb.impute_offdiag_nan_mean(np.asarray(d_sw[j], dtype=np.float64)),
            )
    return out


def _npz_str_field(z: Any, key: str) -> str | None:
    if key not in z.files:
        return None
    raw = z[key]
    arr = np.asarray(raw)
    if arr.dtype.kind in ("S", "U", "O"):
        flat = arr.reshape(-1)
        if flat.size == 0:
            return None
        x = flat[0]
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="replace")
        return str(x)
    return str(arr.reshape(-1)[0])


def _theta_field_row_labels_from_array(arr: np.ndarray) -> list[str]:
    raw = np.asarray(arr).reshape(-1)
    out: list[str] = []
    for x in raw:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(bytes(x).decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return out


def _theta_field_row_labels_from_npz(z: Any, *, key: str = "theta_field_rows") -> list[str]:
    return _theta_field_row_labels_from_array(np.asarray(z[key]))


def _load_cached_twofig_results(output_dir: str) -> CachedTwofigBundle:
    out_npz = os.path.join(output_dir, "h_decoding_twofig_results.npz")
    if not os.path.isfile(out_npz):
        raise FileNotFoundError(
            "Visualization-only mode requires prior results; missing file:\n"
            f"  {out_npz}\n"
            "Run once without --visualization-only to generate h_decoding_twofig_results.npz."
        )
    required = (
        "n",
        "n_ref",
        "theta_field_rows",
        "theta_bin_centers",
        "h_gt_sqrt",
        "decode_ref",
        "h_binned_sweep",
        "decode_sweep",
        "corr_h_binned_vs_gt_mc",
        "corr_decode_vs_ref_shared",
        "wall_seconds",
    )
    with np.load(out_npz, allow_pickle=True) as z:
        missing = [k for k in required if k not in z.files]
        if missing:
            raise KeyError(f"{out_npz} missing keys: {missing}")

        n_arr = np.asarray(z["n"], dtype=np.int64).ravel()
        centers = np.asarray(z["theta_bin_centers"], dtype=np.float64)
        h_gt = np.asarray(z["h_gt_sqrt"], dtype=np.float64)
        dec_ref = np.asarray(z["decode_ref"], dtype=np.float64)
        h_sw = np.asarray(z["h_binned_sweep"], dtype=np.float64)
        dec_sw = np.asarray(z["decode_sweep"], dtype=np.float64)
        corr_h = np.asarray(z["corr_h_binned_vs_gt_mc"], dtype=np.float64)
        corr_d = np.asarray(z["corr_decode_vs_ref_shared"], dtype=np.float64).ravel()
        wall = np.asarray(z["wall_seconds"], dtype=np.float64)

        n_bins = int(h_gt.shape[0])
        if h_gt.ndim != 2 or h_gt.shape[0] != h_gt.shape[1]:
            raise ValueError(f"h_gt_sqrt must be square 2D; got shape {h_gt.shape}.")
        if (centers.ndim == 1 and int(centers.size) != n_bins) or (
            centers.ndim == 2 and int(centers.shape[0]) != n_bins
        ):
            raise ValueError(
                f"theta_bin_centers shape {centers.shape} inconsistent with h_gt_sqrt dim {n_bins}."
            )
        if centers.ndim not in (1, 2):
            raise ValueError(f"theta_bin_centers must be 1D or 2D; got shape {centers.shape}.")
        if dec_ref.shape != (n_bins, n_bins):
            raise ValueError(f"decode_ref shape {dec_ref.shape} expected ({n_bins}, {n_bins}).")

        n_cols = int(n_arr.size)
        if h_sw.ndim != 4:
            raise ValueError(f"h_binned_sweep must be 4D (methods, n_cols, n_bins, n_bins); got {h_sw.shape}.")
        n_methods = int(h_sw.shape[0])
        if int(h_sw.shape[1]) != n_cols or int(h_sw.shape[2]) != n_bins or int(h_sw.shape[3]) != n_bins:
            raise ValueError(
                f"h_binned_sweep shape {h_sw.shape} inconsistent with n columns={n_cols}, n_bins={n_bins}."
            )
        if dec_sw.shape != (n_cols, n_bins, n_bins):
            raise ValueError(f"decode_sweep shape {dec_sw.shape} expected ({n_cols}, {n_bins}, {n_bins}).")

        row_labels = _theta_field_row_labels_from_npz(z, key="theta_field_rows")
        if len(row_labels) != n_methods:
            raise ValueError(
                f"theta_field_rows count {len(row_labels)} != h_binned_sweep methods dim {n_methods}."
            )
        if corr_h.shape != (n_methods, n_cols):
            raise ValueError(f"corr_h_binned_vs_gt_mc shape {corr_h.shape} expected ({n_methods}, {n_cols}).")
        if corr_d.shape != (n_cols,):
            raise ValueError(f"corr_decode_vs_ref_shared shape {corr_d.shape} expected ({n_cols},).")
        if wall.shape != (n_methods, n_cols):
            raise ValueError(f"wall_seconds shape {wall.shape} expected ({n_methods}, {n_cols}).")

        bundle = {k: np.asarray(z[k]) for k in z.files}

    if "hellinger_acc_lb_sweep" not in bundle or "hellinger_acc_ub_sweep" not in bundle:
        lb_sw, ub_sw = _hellinger_sqrt_to_accuracy_bounds(h_sw)
        bundle["hellinger_acc_lb_sweep"] = lb_sw
        bundle["hellinger_acc_ub_sweep"] = ub_sw
    else:
        lb_sw = np.asarray(bundle["hellinger_acc_lb_sweep"], dtype=np.float64)
        ub_sw = np.asarray(bundle["hellinger_acc_ub_sweep"], dtype=np.float64)
    if lb_sw.shape != h_sw.shape:
        raise ValueError(f"hellinger_acc_lb_sweep shape {lb_sw.shape} expected {h_sw.shape}.")
    if ub_sw.shape != h_sw.shape:
        raise ValueError(f"hellinger_acc_ub_sweep shape {ub_sw.shape} expected {h_sw.shape}.")

    if "corr_hellinger_lb_vs_decode_shared" not in bundle:
        bundle["corr_hellinger_lb_vs_decode_shared"] = _corr_hellinger_bounds_vs_decode_shared(lb_sw, dec_sw)
    if "corr_hellinger_ub_vs_decode_shared" not in bundle:
        bundle["corr_hellinger_ub_vs_decode_shared"] = _corr_hellinger_bounds_vs_decode_shared(ub_sw, dec_sw)
    corr_lb = np.asarray(bundle["corr_hellinger_lb_vs_decode_shared"], dtype=np.float64)
    corr_ub = np.asarray(bundle["corr_hellinger_ub_vs_decode_shared"], dtype=np.float64)
    if corr_lb.shape != (n_methods, n_cols):
        raise ValueError(
            f"corr_hellinger_lb_vs_decode_shared shape {corr_lb.shape} expected ({n_methods}, {n_cols})."
        )
    if corr_ub.shape != (n_methods, n_cols):
        raise ValueError(
            f"corr_hellinger_ub_vs_decode_shared shape {corr_ub.shape} expected ({n_methods}, {n_cols})."
        )

    return cast(CachedTwofigBundle, bundle)


def _validate_cached_twofig_cli(
    args: argparse.Namespace,
    cached: CachedTwofigBundle,
    ns: list[int],
    row_labels_cached: list[str],
    row_labels_cli: list[str],
) -> None:
    n_arr = np.asarray(cached["n"], dtype=np.int64).ravel()
    if n_arr.size != len(ns) or not np.array_equal(n_arr, np.asarray(ns, dtype=np.int64)):
        raise ValueError(
            f"--n-list {ns} does not match cached results n={n_arr.tolist()}. "
            "Use the same --n-list as the run that produced h_decoding_twofig_results.npz."
        )
    if int(np.asarray(cached["n_ref"]).reshape(-1)[0]) != int(args.n_ref):
        raise ValueError(
            f"Cached n_ref={int(np.asarray(cached['n_ref']).reshape(-1)[0])} does not match --n-ref={int(args.n_ref)}."
        )
    h_gt = np.asarray(cached["h_gt_sqrt"], dtype=np.float64)
    n_bins_file = int(h_gt.shape[0])
    n_bins_cli = _total_theta_bins_from_args(args)
    if n_bins_file != n_bins_cli:
        raise ValueError(
            f"Cached matrices imply total theta bins={n_bins_file} but CLI implies {n_bins_cli}."
        )

    if row_labels_cli != row_labels_cached:
        raise ValueError(
            "Cached theta_field_rows do not match CLI-resolved rows.\n"
            f"  cached: {row_labels_cached}\n"
            f"  cli:    {row_labels_cli}\n"
            "Pass the same --theta-field-method / --theta-field-methods / --theta-field-rows as the prior run."
        )

    z_path = os.path.join(args.output_dir, "h_decoding_twofig_results.npz")
    with np.load(z_path, allow_pickle=True) as z2:
        ds_cached = _npz_str_field(z2, "dataset_npz")
        fam_cached = _npz_str_field(z2, "dataset_family")
        mode_cached = _npz_str_field(z2, "theta_binning_mode")
    if ds_cached is not None and os.path.abspath(str(args.dataset_npz)) != os.path.abspath(ds_cached):
        raise ValueError(
            f"--dataset-npz {args.dataset_npz!r} does not match cached dataset_npz={ds_cached!r}. "
            "Use the same dataset as the run that produced the cache."
        )
    if fam_cached is not None and str(args.dataset_family) != str(fam_cached):
        raise ValueError(
            f"--dataset-family {args.dataset_family!r} does not match cached dataset_family={fam_cached!r}."
        )
    if mode_cached is not None and _theta_binning_mode(args) != str(mode_cached):
        raise ValueError(
            f"--theta-binning-mode {_theta_binning_mode(args)!r} does not match cached theta_binning_mode={mode_cached!r}."
        )


def _run_twofig_visualization_only(
    args: argparse.Namespace,
    *,
    row_labels: list[str],
    ns: list[int],
    meta: dict[str, Any],
    n_pool: int,
    perm_seed: int,
    cached: CachedTwofigBundle,
) -> None:
    n_bins = int(np.asarray(cached["h_gt_sqrt"], dtype=np.float64).shape[0])
    centers = np.asarray(cached["theta_bin_centers"], dtype=np.float64)
    h_sweep_arr = np.asarray(cached["h_binned_sweep"], dtype=np.float64)
    clf_sweep_arr = np.asarray(cached["decode_sweep"], dtype=np.float64)
    h_gt_sqrt = np.asarray(cached["h_gt_sqrt"], dtype=np.float64)
    clf_ref = np.asarray(cached["decode_ref"], dtype=np.float64)
    corr_h_binned_vs_gt_mc = np.asarray(cached["corr_h_binned_vs_gt_mc"], dtype=np.float64)
    corr_decode_vs_ref_shared = np.asarray(cached["corr_decode_vs_ref_shared"], dtype=np.float64)
    hellinger_acc_lb_sweep = np.asarray(cached["hellinger_acc_lb_sweep"], dtype=np.float64)
    hellinger_acc_ub_sweep = np.asarray(cached["hellinger_acc_ub_sweep"], dtype=np.float64)
    corr_hellinger_lb_vs_decode_shared = np.asarray(cached["corr_hellinger_lb_vs_decode_shared"], dtype=np.float64)
    corr_hellinger_ub_vs_decode_shared = np.asarray(cached["corr_hellinger_ub_vs_decode_shared"], dtype=np.float64)
    nmse_h_binned_vs_gt_mc = _nmse_h_binned_vs_gt_mc(h_sweep_arr, h_gt_sqrt)
    nmse_decode_vs_ref_shared = _nmse_decode_vs_ref_shared(clf_sweep_arr, clf_ref)

    z_path = os.path.join(args.output_dir, "h_decoding_twofig_results.npz")
    with np.load(z_path, allow_pickle=True) as z_meta:
        loss_root_str = _npz_str_field(z_meta, "training_losses_root")
        fourier_on = bool(np.asarray(z_meta.get("theta_flow_fourier_state", False)).reshape(-1)[0])
        _tf_d = z_meta.get("theta_fourier_state_dim")
        theta_fourier_feature_dim_viz = (
            int(np.asarray(_tf_d).reshape(-1)[0]) if fourier_on and _tf_d is not None else None
        )
    loss_root = (
        loss_root_str
        if loss_root_str and os.path.isdir(loss_root_str)
        else os.path.join(args.output_dir, "training_losses")
    )
    continuous_footer = {
        "h_gt_sqrt": h_gt_sqrt,
        "decode_ref": clf_ref,
        "n_ref": int(args.n_ref),
        "decode_sweep_for_decode_limits": clf_sweep_arr,
    }

    sweep_svg = _render_method_sweep_panel(
        row_labels=row_labels,
        h_sweep=h_sweep_arr,
        clf_sweep_shared=clf_sweep_arr,
        n_list=ns,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_twofig_sweep.svg"),
        n_bins=n_bins,
        theta_centers=centers,
        clf_ref_decode_limits=np.asarray(clf_ref, dtype=np.float64),
        continuous_gt_footer=continuous_footer,
    )
    corr_nmse_svg = _render_corr_nmse_two_panel(
        row_labels=row_labels,
        n_list=ns,
        corr_h=corr_h_binned_vs_gt_mc,
        corr_decode_shared=corr_decode_vs_ref_shared,
        corr_hellinger_lb=corr_hellinger_lb_vs_decode_shared,
        corr_hellinger_ub=corr_hellinger_ub_vs_decode_shared,
        show_hellinger_bounds=bool(getattr(args, "show_hellinger_bound_corr", False)),
        corr_decode_hellinger_shared=None,
        nmse_h=nmse_h_binned_vs_gt_mc,
        nmse_decode_shared=nmse_decode_vs_ref_shared,
        nmse_decode_hellinger_shared=None,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_twofig_corr_nmse.svg"),
    )
    loss_panel_svg = _render_row_n_training_losses_panel(
        row_labels=row_labels,
        n_list=ns,
        loss_root=loss_root,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_twofig_training_losses_panel.svg"),
    )

    out_npz = os.path.join(args.output_dir, "h_decoding_twofig_results.npz")
    wall_shape = tuple(int(x) for x in np.asarray(cached["wall_seconds"], dtype=np.float64).shape)
    summary_path = os.path.join(args.output_dir, "h_decoding_twofig_summary.txt")
    _write_summary(
        summary_path,
        args=args,
        meta=meta,
        n_pool=n_pool,
        perm_seed=int(np.asarray(cached.get("perm_seed", 0)).reshape(-1)[0]),
        out_npz=os.path.abspath(out_npz),
        sweep_svg=os.path.abspath(sweep_svg),
        corr_nmse_svg=os.path.abspath(corr_nmse_svg),
        loss_panel_svg=os.path.abspath(loss_panel_svg),
        training_losses_root=os.path.abspath(loss_root),
        h_sweep_shape=tuple(int(x) for x in h_sweep_arr.shape),
        decode_sweep_shape=tuple(int(x) for x in clf_sweep_arr.shape),
        hellinger_acc_lb_shape=tuple(int(x) for x in hellinger_acc_lb_sweep.shape),
        hellinger_acc_ub_shape=tuple(int(x) for x in hellinger_acc_ub_sweep.shape),
        corr_h_shape=tuple(int(x) for x in corr_h_binned_vs_gt_mc.shape),
        nmse_h_shape=tuple(int(x) for x in nmse_h_binned_vs_gt_mc.shape),
        corr_hellinger_lb_shape=tuple(int(x) for x in corr_hellinger_lb_vs_decode_shared.shape),
        corr_hellinger_ub_shape=tuple(int(x) for x in corr_hellinger_ub_vs_decode_shared.shape),
        show_hellinger_bound_corr=bool(getattr(args, "show_hellinger_bound_corr", False)),
        corr_decode_shape=tuple(int(x) for x in corr_decode_vs_ref_shared.shape),
        nmse_decode_shape=tuple(int(x) for x in nmse_decode_vs_ref_shared.shape),
        wall_seconds_shape=wall_shape,
        visualization_only=True,
        theta_fourier_feature_dim=theta_fourier_feature_dim_viz,
    )

    print("[twofig] Saved (visualization-only):", flush=True)
    print(f"  - {os.path.abspath(sweep_svg)}", flush=True)
    print(f"  - {os.path.abspath(corr_nmse_svg)}", flush=True)
    print(f"  - {os.path.abspath(loss_panel_svg)}", flush=True)
    print(f"  - {os.path.abspath(loss_root)}/", flush=True)
    print(f"  - {os.path.abspath(out_npz)} (unchanged)", flush=True)
    print(f"  - {os.path.abspath(summary_path)}", flush=True)


def _normalize_theta_field_method_local(method: str) -> str:
    m = str(method).strip().lower()
    if m in {"vae-x-flow", "vae_x_flow"}:
        return "vae_x_flow"
    if m in {"vae-xflow-sir-lrank", "vae_xflow_sir_lrank"}:
        return "vae_xflow_sir_lrank"
    if m in {"vae-bin-gaussian", "vae_bin_gaussian"}:
        return "vae_bin_gaussian"
    if m in {"vae-ctsm-v", "vae_ctsm_v"}:
        return "vae_ctsm_v"
    if m in {"bin_gaussian", "binned_gaussian", "binned-gaussian", "bin-gaussian"}:
        return "bin_gaussian"
    if m == "nf":
        return "nf"
    sir = conv._normalize_sir_xflow_method(m)
    if sir is not None:
        return str(sir)
    lxf = conv._normalize_linear_x_flow_method(m)
    if lxf is not None:
        return str(lxf)
    ctr = conv._normalize_contrastive_method(m)
    if ctr is not None:
        return str(ctr)
    return normalize_theta_field_method(m)


@dataclass(frozen=True)
class ThetaFieldRowSpec:
    method: str
    arch: str | None
    label: str


def build_parser() -> argparse.ArgumentParser:
    p = conv.build_parser()
    p.description = (
        "Load a shared dataset .npz, run full-compute H/decoding estimation for each n in --n-list, "
        "and save two figures only: sweep matrices over n-list and GT/reference matrices "
        "(plus corr vs n, NMSE vs n, and training-loss panel). "
        "Use --visualization-only to rerender figures from an existing h_decoding_twofig_results.npz "
        "in --output-dir without retraining."
    )
    p.set_defaults(output_dir=str(Path(DATA_DIR) / "h_decoding_twofig"))
    p.add_argument(
        "--theta-field-methods",
        type=str,
        default="",
        help=(
            "Comma-separated theta-field methods to sweep in one run. "
            "Overrides --theta-field-method when non-empty. "
            "Supported values: theta_flow, theta_path_integral, x_flow, ctsm_v, nf, bin_gaussian, "
            "vae_x_flow, vae_xflow_sir_lrank, vae_bin_gaussian, vae_ctsm_v, "
            "contrastive_soft / contrastive_soft_categorical "
            "(same aliases as study_h_decoding_convergence.py), "
            "and supported scheduled linear_x_flow variants including linear_x_flow_diagonal_t "
            "and xflow_sir_lrank / xflow_sir_lrank_dia / xflow_sir_lrank_dia_theta / xflow_sir_lrank_scalar / xflow_sir_lrank_scalar_theta / xflow_sir_pure_lrank, "
            "plus SIR wrappers sir_xflow_lrank_t, sir_xflow, sir_thetaflow."
        ),
    )
    p.add_argument(
        "--theta-field-rows",
        type=str,
        default="",
        help=(
            "Comma-separated theta-field row specs, highest precedence over --theta-field-methods and "
            "--theta-field-method. Tokens are method or method:arch, e.g. "
            "theta_flow:mlp,theta_flow:film,x_flow:film_fourier,ctsm_v,bin_gaussian,"
            "linear_x_flow_low_rank_t,linear_x_flow_pure_low_rank_t,linear_x_flow_pure_cond_low_rank_t,linear_x_flow_diagonal_t. "
            "For linear_x_flow_t and xflow_sir_lrank, --theta-flow-fourier-state can replace scalar theta "
            "with the shared Fourier theta state. "
            "For low-rank linear_x_flow rows use --lxf-low-rank-dim (default 3 for non-SIR low-rank rows). "
            "For xflow_sir_lrank and xflow_sir_pure_lrank variants, omitting --lxf-low-rank-dim selects "
            "the raw SIR U rank automatically from >=90%% inverse-regression eigenvalue mass plus one; when set, "
            "--lxf-low-rank-dim is the manual raw SIR U rank and "
            "--sir-num-bins/--sir-ridge control SIR. "
            "For SIR preprocessing use sir_xflow_lrank_t, sir_xflow, or sir_thetaflow with --sir-dim and --sir-num-bins "
            "(sir_xflow_lrank_t also needs --lxf-low-rank-dim <= --sir-dim)."
        ),
    )
    p.add_argument(
        "--decode-source-npz",
        type=str,
        default="",
        help=(
            "Optional path to the low-dimensional (pre-PR) shared dataset .npz. "
            "When set, overrides meta pr_autoencoder_source_npz for GT decoding only."
        ),
    )
    p.add_argument(
        "--decode-gt-fallback-embedded",
        action="store_true",
        help=(
            "If PR-embedded but the native source NPZ cannot be resolved, use embedded x for "
            "GT decoding instead of failing (not recommended)."
        ),
    )
    p.add_argument(
        "--show-hellinger-bound-corr",
        action="store_true",
        help=(
            "Include optional Hellinger-derived lower/upper accuracy-bound vs decoding "
            "correlation curves in h_decoding_twofig_corr_nmse.svg. By default these "
            "bound comparisons are computed/saved but not plotted."
        ),
    )
    p.add_argument(
        "--sir-first",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--vae-latent-dim", type=int, default=0, help="VAE wrapper latent dimension. 0 uses min(5, x_dim).")
    p.add_argument("--vae-hidden-dim", type=int, default=128)
    p.add_argument("--vae-depth", type=int, default=4)
    p.add_argument("--vae-epochs", type=int, default=5000)
    p.add_argument("--vae-batch-size", type=int, default=256)
    p.add_argument("--vae-lr", type=float, default=1e-3)
    p.add_argument("--vae-weight-decay", type=float, default=0.0)
    p.add_argument("--vae-early-patience", type=int, default=500)
    p.add_argument("--vae-early-min-delta", type=float, default=1e-4)
    p.add_argument("--vae-early-ema-alpha", type=float, default=0.05)
    p.add_argument("--vae-kl-weight", type=float, default=0.01)
    return p


def build_sir_first_parser() -> argparse.ArgumentParser:
    p = build_parser()
    p.description = (
        "SIR-first variant of study_h_decoding_twofig.py. For each n in --n-list, "
        "Sliced Inverse Regression is fit only on that nested subset's training split, "
        "then train/validation/all x arrays are projected before any estimator or sweep "
        "decoder sees the data. The n_ref GT decoding panel keeps the native/pre-PR "
        "twofig convention."
    )
    p.epilog = (
        "Benchmark-1D PR-30D examples:\n"
        "  mamba run -n geo_diffusion python bin/study_h_decoding_twofig_sir.py "
        "--dataset-npz data/randamp_gaussian_sqrtd_xdim5/randamp_gaussian_sqrtd_xdim5_pr30d.npz "
        "--dataset-family randamp_gaussian_sqrtd "
        f"--theta-field-rows {SIR_FIRST_DEFAULT_ROWS} --n-list 80,200,400,600 "
        "--device cuda:1 --output-dir data/experiments/h_decoding_twofig_sir_pr30d_linearbench_<TAG>\n"
        "  mamba run -n geo_diffusion python bin/study_h_decoding_twofig_sir.py "
        "--dataset-npz data/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha4x/"
        "cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha4x_pr30d.npz "
        "--dataset-family cosine_gaussian_sqrtd_rand_tune_additive "
        f"--theta-field-rows {SIR_FIRST_DEFAULT_ROWS} --n-list 80,200,400,600 "
        "--device cuda:1 --output-dir data/experiments/h_decoding_twofig_sir_pr30d_cosinebench_noise2x_alpha4x_<TAG>"
    )
    p.formatter_class = argparse.RawDescriptionHelpFormatter
    p.set_defaults(
        output_dir=str(Path(DATA_DIR) / "h_decoding_twofig_sir"),
        theta_field_rows=SIR_FIRST_DEFAULT_ROWS,
        n_list="80,200,400,600",
        sir_first=True,
    )
    return p


def _parse_theta_field_methods(args: argparse.Namespace) -> list[str]:
    # Legacy parser used when --theta-field-rows is not set.
    raw = str(getattr(args, "theta_field_methods", "") or "").strip()
    if raw:
        toks = [t.strip() for t in raw.split(",") if t.strip()]
        if not toks:
            raise ValueError("--theta-field-methods is provided but no method tokens were found.")
    else:
        toks = [str(getattr(args, "theta_field_method", "theta_flow"))]
    methods: list[str] = []
    seen: set[str] = set()
    for tok in toks:
        m = _normalize_theta_field_method_local(tok)
        if m not in seen:
            seen.add(m)
            methods.append(m)
    if not methods:
        raise ValueError("No theta-field methods resolved from CLI arguments.")
    return methods


def _row_label(method: str, arch: str | None) -> str:
    return str(method) if arch is None else f"{method}:{arch}"


def _parse_theta_field_rows(args: argparse.Namespace) -> list[ThetaFieldRowSpec]:
    raw_rows = str(getattr(args, "theta_field_rows", "") or "").strip()
    if raw_rows:
        toks = [t.strip() for t in raw_rows.split(",") if t.strip()]
        if not toks:
            raise ValueError("--theta-field-rows is provided but no row tokens were found.")
    else:
        methods = _parse_theta_field_methods(args)
        toks = methods

    rows: list[ThetaFieldRowSpec] = []
    seen: set[tuple[str, str | None]] = set()
    for tok in toks:
        parts = [p.strip() for p in str(tok).split(":")]
        if len(parts) > 2:
            raise ValueError(
                f"Invalid --theta-field-rows token {tok!r}; expected method or method:arch."
            )
        method = _normalize_theta_field_method_local(parts[0])
        arch: str | None = None
        if len(parts) == 2:
            raw_arch = parts[1]
            if not raw_arch:
                raise ValueError(
                    f"Invalid --theta-field-rows token {tok!r}; empty arch after ':'."
                )
            arch = normalize_flow_arch(argparse.Namespace(flow_arch=raw_arch))
            if method not in _FLOW_ROW_ARCH_METHODS:
                raise ValueError(
                    f"Invalid --theta-field-rows token {tok!r}; arch suffix is only allowed for "
                    "flow-based rows (theta_flow, theta_path_integral, x_flow, SIR flow wrappers)."
                )
        key = (method, arch)
        if key in seen:
            continue
        seen.add(key)
        rows.append(ThetaFieldRowSpec(method=method, arch=arch, label=_row_label(method, arch)))
    if not rows:
        raise ValueError("No theta-field rows resolved from CLI arguments.")
    return rows


def _validate_cli_for_rows(args: argparse.Namespace, rows: list[ThetaFieldRowSpec]) -> None:
    for row in rows:
        base_method = _VAE_WRAPPED_METHODS.get(row.method, row.method)
        if row.method in _NO_TRAIN_METHODS or base_method in _NO_TRAIN_METHODS:
            continue
        args_r = deepcopy(args)
        setattr(args_r, "theta_field_method", base_method)
        if row.arch is not None:
            setattr(args_r, "flow_arch", row.arch)
        try:
            conv._validate_cli(args_r)
        except Exception as exc:
            raise ValueError(f"row={row.label}: {exc}") from exc


def _validate_categorical_rows(meta: dict, rows: list[ThetaFieldRowSpec]) -> None:
    if str(meta.get("theta_type", "")) != "categorical":
        return
    unsupported = {
        "theta_flow",
        "theta_path_integral",
        "ctsm_v",
        "nf",
        "linear_theta_flow",
        "sir_thetaflow",
    }
    bad = [row.label for row in rows if row.method in unsupported]
    if bad:
        raise ValueError(
            "Categorical datasets do not support continuous theta-density rows: "
            f"{', '.join(bad)}. Use category-native rows such as bin_gaussian or x_flow/linear_x_flow variants."
        )


def _project_sir_first_subset(
    *,
    subset: SweepSubset,
    theta_fit_subset: SweepSubset,
    args: argparse.Namespace,
    n: int,
    sir_projection_root: str,
) -> tuple[SweepSubset, dict[str, np.ndarray | int | float], str]:
    z_train, z_val, z_all, sir_meta = _fit_sir_projection(
        x_train=subset.bundle.x_train,
        theta_train=theta_fit_subset.bundle.theta_train,
        x_val=subset.bundle.x_validation,
        x_all=subset.bundle.x_all,
        sir_dim=int(args.sir_dim),
        num_bins=int(args.sir_num_bins),
        ridge=float(args.sir_ridge),
    )
    meta_proj = dict(subset.bundle.meta)
    meta_proj["x_dim"] = int(args.sir_dim)
    meta_proj["sir_enabled"] = True
    meta_proj["sir_dim"] = int(args.sir_dim)
    meta_proj["sir_num_bins"] = int(args.sir_num_bins)
    meta_proj["sir_ridge"] = float(args.sir_ridge)
    bundle_proj = SharedDatasetBundle(
        meta=meta_proj,
        theta_all=subset.bundle.theta_all,
        x_all=np.asarray(z_all, dtype=np.float64),
        train_idx=subset.bundle.train_idx,
        validation_idx=subset.bundle.validation_idx,
        theta_train=subset.bundle.theta_train,
        x_train=np.asarray(z_train, dtype=np.float64),
        theta_validation=subset.bundle.theta_validation,
        x_validation=np.asarray(z_val, dtype=np.float64),
    )
    projected = SweepSubset(
        bundle=bundle_proj,
        bin_all=subset.bin_all,
        bin_train=subset.bin_train,
        bin_validation=subset.bin_validation,
    )
    os.makedirs(sir_projection_root, exist_ok=True)
    sir_path = os.path.abspath(os.path.join(sir_projection_root, f"n_{int(n):06d}.npz"))
    np.savez_compressed(
        sir_path,
        n=np.int64(n),
        sir_enabled=np.bool_(True),
        sir_dim=np.int64(args.sir_dim),
        sir_num_bins=np.int64(args.sir_num_bins),
        sir_ridge=np.float64(args.sir_ridge),
        sir_components=np.asarray(sir_meta["sir_components"], dtype=np.float64),
        sir_x_mean=np.asarray(sir_meta["sir_x_mean"], dtype=np.float64),
        sir_eigenvalues=np.asarray(sir_meta["sir_eigenvalues"], dtype=np.float64),
        sir_eigenvalues_all=np.asarray(sir_meta.get("sir_eigenvalues_all", sir_meta["sir_eigenvalues"]), dtype=np.float64),
        sir_rank_mode=np.asarray(sir_meta.get("sir_rank_mode", ["manual"]), dtype=object),
        sir_rank_requested=np.asarray(sir_meta.get("sir_rank_requested", args.sir_dim)),
        sir_rank_auto_threshold=np.asarray(sir_meta.get("sir_rank_auto_threshold", 0.90)),
        sir_bin_counts=np.asarray(sir_meta["sir_bin_counts"], dtype=np.int64),
        sir_nonempty_bin_ids=np.asarray(sir_meta["sir_nonempty_bin_ids"], dtype=np.int64),
        sir_slice_means=np.asarray(sir_meta["sir_slice_means"], dtype=np.float64),
        sir_theta_edges=np.asarray(sir_meta["sir_theta_edges"], dtype=np.float64),
        train_idx=np.asarray(subset.bundle.train_idx, dtype=np.int64),
        validation_idx=np.asarray(subset.bundle.validation_idx, dtype=np.int64),
        bin_train=np.asarray(subset.bin_train, dtype=np.int64),
        bin_validation=np.asarray(subset.bin_validation, dtype=np.int64),
        bin_all=np.asarray(subset.bin_all, dtype=np.int64),
    )
    return projected, sir_meta, sir_path


def _subset_with_vae_mean_x(
    *,
    subset: SweepSubset,
    args: argparse.Namespace,
    dev: Any,
) -> tuple[SweepSubset, dict[str, Any]]:
    vae = prepare_vae_mean_features(
        args,
        device=dev,
        x_train=subset.bundle.x_train,
        x_val=subset.bundle.x_validation,
        x_eval=subset.bundle.x_all,
    )
    meta = dict(subset.bundle.meta)
    meta["x_dim"] = int(np.asarray(vae["x_eval"], dtype=np.float64).shape[1])
    meta["vae_preprocessed"] = True
    bundle = SharedDatasetBundle(
        meta=meta,
        theta_all=subset.bundle.theta_all,
        x_all=np.asarray(vae["x_eval"], dtype=np.float64),
        train_idx=subset.bundle.train_idx,
        validation_idx=subset.bundle.validation_idx,
        theta_train=subset.bundle.theta_train,
        x_train=np.asarray(vae["x_train"], dtype=np.float64),
        theta_validation=subset.bundle.theta_validation,
        x_validation=np.asarray(vae["x_validation"], dtype=np.float64),
    )
    return (
        SweepSubset(
            bundle=bundle,
            bin_all=subset.bin_all,
            bin_train=subset.bin_train,
            bin_validation=subset.bin_validation,
        ),
        _vae_payload(vae, args),
    )


def _save_vae_wrapper_loss_npz(path: str, *, method_name: str, vae_payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tr = np.asarray(vae_payload.get("train_losses", []), dtype=np.float64).ravel()
    va = np.asarray(vae_payload.get("val_losses", []), dtype=np.float64).ravel()
    em = np.asarray(vae_payload.get("val_monitor_losses", []), dtype=np.float64).ravel()
    extra = {f"vae_{k}": v for k, v in vae_payload.items()}
    np.savez_compressed(
        path,
        theta_field_method=np.asarray([str(method_name)], dtype=object),
        prior_enable=np.bool_(False),
        score_train_losses=tr,
        score_val_losses=va,
        score_val_monitor_losses=em,
        **extra,
    )


def _annotate_npz_with_sir(path: str, sir_meta: dict[str, Any], sir_path: str) -> None:
    if not os.path.isfile(path):
        return
    with np.load(path, allow_pickle=True) as data:
        payload = {k: np.asarray(data[k]) for k in data.files}
    payload.update(
        sir_enabled=np.bool_(True),
        sir_projection_npz=np.asarray(os.path.abspath(sir_path), dtype=np.str_),
        sir_dim=np.asarray(sir_meta["sir_dim"]),
        sir_num_bins=np.asarray(sir_meta["sir_num_bins"]),
        sir_ridge=np.asarray(sir_meta["sir_ridge"]),
        sir_components=np.asarray(sir_meta["sir_components"], dtype=np.float64),
        sir_x_mean=np.asarray(sir_meta["sir_x_mean"], dtype=np.float64),
        sir_eigenvalues=np.asarray(sir_meta["sir_eigenvalues"], dtype=np.float64),
        sir_eigenvalues_all=np.asarray(sir_meta.get("sir_eigenvalues_all", sir_meta["sir_eigenvalues"]), dtype=np.float64),
        sir_rank_mode=np.asarray(sir_meta.get("sir_rank_mode", ["manual"]), dtype=object),
        sir_rank_requested=np.asarray(sir_meta.get("sir_rank_requested", sir_meta["sir_dim"])),
        sir_rank_auto_threshold=np.asarray(sir_meta.get("sir_rank_auto_threshold", 0.90)),
        sir_bin_counts=np.asarray(sir_meta["sir_bin_counts"], dtype=np.int64),
        sir_theta_edges=np.asarray(sir_meta["sir_theta_edges"], dtype=np.float64),
    )
    np.savez_compressed(path, **payload)


def _draw_categorical_gt_footer_row(
    ax_gt: Any,
    ax_dec: Any,
    *,
    h_gt_sqrt: np.ndarray,
    decode_ref: np.ndarray,
    n_ref: int,
    n_bins: int,
    theta_centers: np.ndarray,
    decode_sweep_for_decode_limits: np.ndarray | None,
) -> None:
    """Left/right heatmaps matching categorical GT panel (analytic + pairwise decoding at n_ref)."""
    tick_pos, tick_labs, axis_label = _theta_axis_tick_labels(theta_centers, int(n_bins))
    x_rot = 45 if len(tick_pos) > 6 else 0

    def _one(ax: Any, mat: np.ndarray, title: str, vmin: float, vmax: float) -> None:
        im = ax.imshow(
            np.asarray(mat, dtype=np.float64),
            vmin=float(vmin),
            vmax=float(vmax),
            cmap="viridis",
            aspect="equal",
            origin="lower",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labs, rotation=x_rot, ha="right" if x_rot else "center", fontsize=11)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_labs, fontsize=11)
        ax.set_xlabel(axis_label, fontsize=11)
        conv._matrix_axes_show_top_right_spines(ax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=11)

    _one(ax_gt, h_gt_sqrt, r"Analytic GT $\sqrt{H^2}$ (category)", 0.0, 1.0)
    ax_gt.set_ylabel("category", fontsize=11)
    lim_arrays: list[np.ndarray] = [np.asarray(decode_ref, dtype=np.float64)]
    if decode_sweep_for_decode_limits is not None:
        dsw = np.asarray(decode_sweep_for_decode_limits, dtype=np.float64)
        for j in range(int(dsw.shape[0])):
            lim_arrays.append(np.asarray(dsw[j], dtype=np.float64))
    vmin_c, vmax_c = _decode_accuracy_color_limits(*lim_arrays)
    _one(ax_dec, decode_ref, f"Pairwise decoding (n_ref={int(n_ref)})", vmin_c, vmax_c)


def _draw_continuous_gt_footer_row(
    ax_gt: Any,
    ax_dec: Any,
    *,
    h_gt_sqrt: np.ndarray,
    decode_ref: np.ndarray,
    n_ref: int,
    n_bins: int,
    theta_centers: np.ndarray,
    decode_sweep_for_decode_limits: np.ndarray | None,
) -> None:
    """Left/right heatmaps matching continuous GT panel (MC Hellinger + n_ref decoding)."""
    _draw_single_heatmap(
        ax_gt,
        h_gt_sqrt,
        n_bins=n_bins,
        theta_centers=theta_centers,
        title="Approx GT H matrix",
        vmin=0.0,
        vmax=1.0,
    )
    ax_gt.set_ylabel(r"$\theta$", fontsize=11)
    lim_parts: list[np.ndarray] = [np.asarray(decode_ref, dtype=np.float64)]
    if decode_sweep_for_decode_limits is not None:
        dsw = np.asarray(decode_sweep_for_decode_limits, dtype=np.float64)
        for j in range(int(dsw.shape[0])):
            lim_parts.append(np.asarray(dsw[j], dtype=np.float64))
    vmin_c, vmax_c = _decode_accuracy_color_limits(*lim_parts)
    _draw_single_heatmap(
        ax_dec,
        decode_ref,
        n_bins=n_bins,
        theta_centers=theta_centers,
        title=f"Approx GT decoding (n_ref={int(n_ref)})",
        vmin=vmin_c,
        vmax=vmax_c,
    )


def _render_method_sweep_panel(
    *,
    row_labels: list[str],
    h_sweep: np.ndarray,
    clf_sweep_shared: np.ndarray,
    n_list: list[int],
    out_svg_path: str,
    n_bins: int,
    theta_centers: np.ndarray,
    clf_hellinger_sweep_shared: np.ndarray | None = None,
    clf_ref_decode_limits: np.ndarray | None = None,
    category_gt_footer: dict[str, Any] | None = None,
    continuous_gt_footer: dict[str, Any] | None = None,
) -> str:
    if category_gt_footer is not None and continuous_gt_footer is not None:
        raise ValueError("Pass at most one of category_gt_footer or continuous_gt_footer.")
    n_methods = len(row_labels)
    n_cols = len(n_list)
    if h_sweep.shape[:2] != (n_methods, n_cols):
        raise ValueError(
            f"h_sweep shape mismatch: expected leading dims {(n_methods, n_cols)}, got {h_sweep.shape}."
        )
    if clf_sweep_shared.shape[:1] != (n_cols,):
        raise ValueError(
            f"decode sweep shape mismatch: expected leading dims {(n_cols,)}, got {clf_sweep_shared.shape}."
        )
    clf_h_shared = None
    if clf_hellinger_sweep_shared is not None:
        clf_h_shared = np.asarray(clf_hellinger_sweep_shared, dtype=np.float64)
        if clf_h_shared.shape[:1] != (n_cols,):
            raise ValueError(
                f"classifier Hellinger sweep shape mismatch: expected leading dims {(n_cols,)}, "
                f"got {clf_h_shared.shape}."
            )
    _theta_center_array_for_axis(theta_centers, n_bins)

    n_extra = 2 if clf_h_shared is not None else 1
    n_rows = n_methods + n_extra
    decode_row = n_methods + (1 if clf_h_shared is not None else 0)

    tick_pos, tick_labs, axis_label = _theta_axis_tick_labels(theta_centers, n_bins)
    x_rot = 45 if len(tick_pos) > 6 else 0

    vmin_h, vmax_h = 0.0, 1.0
    decode_lim_parts: list[np.ndarray] = [
        np.asarray(clf_sweep_shared[c], dtype=np.float64) for c in range(n_cols)
    ]
    if clf_ref_decode_limits is not None:
        decode_lim_parts.append(np.asarray(clf_ref_decode_limits, dtype=np.float64))
    vmin_c, vmax_c = _decode_accuracy_color_limits(*decode_lim_parts)

    def _draw_sweep_cells(axes: np.ndarray) -> None:
        for m_idx, label in enumerate(row_labels):
            for c_idx, n in enumerate(n_list):
                ax_h = axes[m_idx, c_idx]
                im_h = ax_h.imshow(
                    np.asarray(h_sweep[m_idx, c_idx], dtype=np.float64),
                    vmin=vmin_h,
                    vmax=vmax_h,
                    cmap="viridis",
                    aspect="equal",
                    origin="lower",
                )
                if m_idx == 0:
                    ax_h.set_title(f"n={int(n)}", fontsize=10)
                ax_h.set_xticks(tick_pos)
                ax_h.set_xticklabels(tick_labs, rotation=x_rot, ha="right" if x_rot else "center", fontsize=11)
                ax_h.set_yticks(tick_pos)
                ax_h.set_yticklabels(tick_labs, fontsize=11)
                conv._matrix_axes_show_top_right_spines(ax_h)
                if c_idx == 0:
                    ax_h.set_ylabel(f"{label} | sqrt(H^2)", fontsize=11)
                if m_idx == (n_methods - 1):
                    ax_h.set_xlabel(axis_label, fontsize=11)
                if c_idx == (n_cols - 1):
                    cb_h = plt.colorbar(im_h, ax=ax_h, fraction=0.046, pad=0.04)
                    cb_h.ax.tick_params(labelsize=11)

        if clf_h_shared is not None:
            clf_h_row = n_methods
            for c_idx, n in enumerate(n_list):
                ax_ch = axes[clf_h_row, c_idx]
                im_ch = ax_ch.imshow(
                    np.asarray(clf_h_shared[c_idx], dtype=np.float64),
                    vmin=vmin_h,
                    vmax=vmax_h,
                    cmap="viridis",
                    aspect="equal",
                    origin="lower",
                )
                ax_ch.set_xticks(tick_pos)
                ax_ch.set_xticklabels(tick_labs, rotation=x_rot, ha="right" if x_rot else "center", fontsize=11)
                ax_ch.set_yticks(tick_pos)
                ax_ch.set_yticklabels(tick_labs, fontsize=11)
                conv._matrix_axes_show_top_right_spines(ax_ch)
                if c_idx == 0:
                    ax_ch.set_ylabel("classifier LLR | sqrt(H^2)", fontsize=11)
                ax_ch.set_xlabel(axis_label, fontsize=11)
                if c_idx == (n_cols - 1):
                    cb_ch = plt.colorbar(im_ch, ax=ax_ch, fraction=0.046, pad=0.04)
                    cb_ch.ax.tick_params(labelsize=11)

        for c_idx, n in enumerate(n_list):
            ax_c = axes[decode_row, c_idx]
            im_c = ax_c.imshow(
                np.asarray(clf_sweep_shared[c_idx], dtype=np.float64),
                vmin=vmin_c,
                vmax=vmax_c,
                cmap="viridis",
                aspect="equal",
                origin="lower",
            )
            ax_c.set_xticks(tick_pos)
            ax_c.set_xticklabels(tick_labs, rotation=x_rot, ha="right" if x_rot else "center", fontsize=11)
            ax_c.set_yticks(tick_pos)
            ax_c.set_yticklabels(tick_labs, fontsize=11)
            conv._matrix_axes_show_top_right_spines(ax_c)
            if c_idx == 0:
                ax_c.set_ylabel("decoding", fontsize=11)
            ax_c.set_xlabel(axis_label, fontsize=11)
            if c_idx == (n_cols - 1):
                cb_c = plt.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
                cb_c.ax.tick_params(labelsize=11)

    footer = category_gt_footer if category_gt_footer is not None else continuous_gt_footer
    if footer is None:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.5 * n_rows), squeeze=False)
        _draw_sweep_cells(np.asarray(axes, dtype=object))
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.12)
    else:
        foot = dict(footer)
        need = ("h_gt_sqrt", "decode_ref", "n_ref")
        for k in need:
            if k not in foot:
                footer_name = "category_gt_footer" if category_gt_footer is not None else "continuous_gt_footer"
                raise ValueError(f"{footer_name} missing required key {k!r}")
        fig_h = 2.5 * (n_rows + 1)
        fig = plt.figure(figsize=(2.8 * n_cols, fig_h))
        gs = fig.add_gridspec(
            n_rows + 1,
            n_cols,
            height_ratios=[1.0] * n_rows + [1.0],
            hspace=0.12,
        )
        axes = np.empty((n_rows, n_cols), dtype=object)
        for i in range(n_rows):
            for j in range(n_cols):
                axes[i, j] = fig.add_subplot(gs[i, j])
        _draw_sweep_cells(axes)
        # n_cols==1: footer still uses two panels side-by-side spanning the full width.
        sub_gs = gs[n_rows, :].subgridspec(1, 2, wspace=0.35)
        ax_gt = fig.add_subplot(sub_gs[0, 0])
        ax_dec = fig.add_subplot(sub_gs[0, 1])
        dlim = foot.get("decode_sweep_for_decode_limits")
        dlim_arr = None if dlim is None else np.asarray(dlim, dtype=np.float64)
        if category_gt_footer is not None:
            _draw_categorical_gt_footer_row(
                ax_gt,
                ax_dec,
                h_gt_sqrt=np.asarray(foot["h_gt_sqrt"], dtype=np.float64),
                decode_ref=np.asarray(foot["decode_ref"], dtype=np.float64),
                n_ref=int(foot["n_ref"]),
                n_bins=int(n_bins),
                theta_centers=theta_centers,
                decode_sweep_for_decode_limits=dlim_arr,
            )
        else:
            _draw_continuous_gt_footer_row(
                ax_gt,
                ax_dec,
                h_gt_sqrt=np.asarray(foot["h_gt_sqrt"], dtype=np.float64),
                decode_ref=np.asarray(foot["decode_ref"], dtype=np.float64),
                n_ref=int(foot["n_ref"]),
                n_bins=int(n_bins),
                theta_centers=theta_centers,
                decode_sweep_for_decode_limits=dlim_arr,
            )
        fig.subplots_adjust(hspace=0.12)

    svg = _save_figure_svg(fig, out_svg_path)
    plt.close(fig)
    return svg


def _save_figure_svg(fig: plt.Figure, path_svg: str) -> str:
    fig.savefig(path_svg)
    return path_svg


def _draw_single_heatmap(
    ax: Any,
    mat: np.ndarray,
    *,
    n_bins: int,
    theta_centers: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    tick_pos, tick_labs, axis_label = _theta_axis_tick_labels(theta_centers, int(n_bins))
    x_rot = 45 if len(tick_pos) > 6 else 0

    im = ax.imshow(
        np.asarray(mat, dtype=np.float64),
        vmin=float(vmin),
        vmax=float(vmax),
        cmap="viridis",
        aspect="equal",
        origin="lower",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labs, rotation=x_rot, ha="right" if x_rot else "center", fontsize=11)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labs, fontsize=11)
    ax.set_xlabel(axis_label, fontsize=11)
    conv._matrix_axes_show_top_right_spines(ax)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=11)


def _render_gt_panel(
    *,
    h_gt_sqrt: np.ndarray,
    clf_ref: np.ndarray,
    n_ref: int,
    n_bins: int,
    theta_centers: np.ndarray,
    out_svg_path: str,
    clf_sweep_shared_for_decode_limits: np.ndarray | None = None,
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.2), squeeze=False)
    _draw_continuous_gt_footer_row(
        axes[0, 0],
        axes[0, 1],
        h_gt_sqrt=h_gt_sqrt,
        decode_ref=clf_ref,
        n_ref=n_ref,
        n_bins=n_bins,
        theta_centers=theta_centers,
        decode_sweep_for_decode_limits=clf_sweep_shared_for_decode_limits,
    )
    fig.tight_layout()
    svg = _save_figure_svg(fig, out_svg_path)
    plt.close(fig)
    return svg


def _write_summary(
    path: str,
    *,
    args: argparse.Namespace,
    meta: dict,
    n_pool: int,
    perm_seed: int,
    out_npz: str,
    sweep_svg: str,
    corr_nmse_svg: str,
    loss_panel_svg: str,
    training_losses_root: str,
    h_sweep_shape: tuple[int, ...],
    decode_sweep_shape: tuple[int, ...],
    hellinger_acc_lb_shape: tuple[int, ...],
    hellinger_acc_ub_shape: tuple[int, ...],
    corr_h_shape: tuple[int, ...],
    nmse_h_shape: tuple[int, ...],
    corr_hellinger_lb_shape: tuple[int, ...],
    corr_hellinger_ub_shape: tuple[int, ...],
    show_hellinger_bound_corr: bool,
    corr_decode_shape: tuple[int, ...],
    nmse_decode_shape: tuple[int, ...],
    wall_seconds_shape: tuple[int, ...],
    visualization_only: bool = False,
    theta_fourier_feature_dim: int | None = None,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("study_h_decoding_twofig\n")
        if visualization_only:
            f.write("visualization_only: True\n")
        f.write(f"dataset_npz: {args.dataset_npz}\n")
        f.write(f"dataset_family: {meta.get('dataset_family')}\n")
        f.write(f"output_dir: {args.output_dir}\n")
        f.write(f"n_ref: {int(args.n_ref)}\n")
        f.write(f"theta_field_rows: {','.join(getattr(args, 'theta_field_rows_resolved', []))}\n")
        f.write(f"theta_field_row_methods: {','.join(getattr(args, 'theta_field_row_methods_resolved', []))}\n")
        f.write(f"theta_field_row_arches: {','.join(getattr(args, 'theta_field_row_arches_resolved', []))}\n")
        f.write(f"n_list: {args.n_list}\n")
        f.write(f"num_theta_bins: {int(args.num_theta_bins)}\n")
        f.write(f"theta_binning_mode: {_theta_binning_mode(args)}\n")
        if _theta_binning_mode(args) == "theta2_grid":
            f.write(f"num_theta_bins_y: {_num_theta_bins_y(args)}\n")
            f.write(f"total_theta_bins: {_total_theta_bins_from_args(args)}\n")
        theta_fourier_enabled = bool(getattr(args, "theta_flow_fourier_state", False))
        f.write(f"theta_flow_fourier_state: {theta_fourier_enabled}\n")
        if theta_fourier_enabled:
            k_fourier = int(getattr(args, "theta_flow_fourier_k", 0))
            include_linear = bool(getattr(args, "theta_flow_fourier_include_linear", False))
            f.write(f"theta_flow_fourier_k: {k_fourier}\n")
            f.write(f"theta_flow_fourier_period_mult: {float(getattr(args, 'theta_flow_fourier_period_mult', float('nan'))):.17g}\n")
            f.write(f"theta_flow_fourier_include_linear: {include_linear}\n")
            if theta_fourier_feature_dim is not None:
                f.write(f"theta_fourier_state_dim: {int(theta_fourier_feature_dim)}\n")
            else:
                f.write(f"theta_fourier_state_dim: {2 * k_fourier + (1 if include_linear else 0)}\n")
        f.write(f"dataset_pool_size: {int(n_pool)}\n")
        f.write(f"dataset_meta_seed: {int(meta.get('seed', 0))}\n")
        f.write(f"perm_seed: {int(perm_seed)}\n")
        sir_enabled = bool(getattr(args, "sir_first", False))
        f.write(f"sir_enabled: {sir_enabled}\n")
        if sir_enabled:
            f.write(f"sir_dim: {int(args.sir_dim)}\n")
            f.write(f"sir_num_bins: {int(args.sir_num_bins)}\n")
            f.write(f"sir_ridge: {float(args.sir_ridge):.17g}\n")
            f.write(f"sir_projection_root: {os.path.abspath(os.path.join(args.output_dir, 'sir_projections'))}\n")
        f.write(f"results_npz: {out_npz}\n")
        f.write(f"h_binned_sweep_shape: {h_sweep_shape}\n")
        f.write(f"decode_sweep_shape: {decode_sweep_shape}\n")
        f.write(f"hellinger_acc_lb_sweep_shape: {hellinger_acc_lb_shape}\n")
        f.write(f"hellinger_acc_ub_sweep_shape: {hellinger_acc_ub_shape}\n")
        f.write(f"corr_h_binned_vs_gt_mc_shape: {corr_h_shape}\n")
        f.write(f"nmse_h_binned_vs_gt_mc_shape: {nmse_h_shape}\n")
        f.write(f"corr_hellinger_lb_vs_decode_shared_shape: {corr_hellinger_lb_shape}\n")
        f.write(f"corr_hellinger_ub_vs_decode_shared_shape: {corr_hellinger_ub_shape}\n")
        f.write(f"show_hellinger_bound_corr: {bool(show_hellinger_bound_corr)}\n")
        f.write(f"corr_decode_vs_ref_shared_shape: {corr_decode_shape}\n")
        f.write(f"nmse_decode_vs_ref_shared_shape: {nmse_decode_shape}\n")
        f.write("hellinger_accuracy_bounds: h2=clip(h_sqrt**2,0,1); lb=0.5*(1+h2); ub=0.5*(1+sqrt(2*h2-h2**2)); diagonal=NaN\n")
        f.write("decode_sweep_semantics: shared_across_methods\n")
        f.write(f"wall_seconds_shape: {wall_seconds_shape}\n")
        f.write(f"figure_sweep_svg: {sweep_svg}\n")
        f.write(f"figure_corr_nmse_svg: {corr_nmse_svg}\n")
        f.write(f"figure_training_losses_panel_svg: {loss_panel_svg}\n")
        f.write(f"training_losses_root: {training_losses_root}\n")


def _sanitize_row_label(label: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label).strip())
    return out.strip("_") or "row"


def _render_row_n_training_losses_panel(
    *,
    row_labels: list[str],
    n_list: list[int],
    loss_root: str,
    out_svg_path: str,
) -> str:
    n_rows = len(row_labels)
    n_cols = len(n_list)
    if n_rows < 1 or n_cols < 1:
        raise ValueError("row labels and n-list must be non-empty for loss panel.")
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(3.0 * n_cols, 7.0), max(2.5 * n_rows, 3.0)),
        squeeze=False,
        sharex=False,
    )

    for i, row_label in enumerate(row_labels):
        row_dir = os.path.join(loss_root, _sanitize_row_label(row_label))
        for j, n in enumerate(n_list):
            ax = axes[i, j]
            loss_npz = os.path.join(row_dir, f"n_{int(n):06d}.npz")
            title = f"{row_label} | n={int(n)}"
            if not os.path.isfile(loss_npz):
                ax.text(
                    0.5,
                    0.5,
                    f"missing\n{loss_npz}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="crimson",
                )
                ax.set_title(title, fontsize=9)
                ax.set_axis_off()
                continue

            try:
                bundle = conv._load_per_n_training_loss_npz(loss_npz)
            except Exception as e:
                ax.text(
                    0.5,
                    0.5,
                    f"load error:\n{e!s}"[:220],
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=7,
                    color="crimson",
                )
                ax.set_title(title, fontsize=9)
                ax.set_axis_off()
                continue

            conv._plot_loss_triplet(
                ax,
                bundle["score_train_losses"],
                bundle["score_val_losses"],
                bundle["score_val_monitor_losses"],
                ylabel="loss" if j == 0 else "",
                title=title,
                show_legend=(i == 0 and j == 0),
                score_like=True,
            )
            if not bool(bundle.get("prior_enable", True)):
                ax.text(
                    0.02,
                    0.02,
                    "prior disabled",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="#444444",
                    ha="left",
                    va="bottom",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "#f2f2f2", "edgecolor": "#bdbdbd"},
                )

    fig.tight_layout()
    svg = _save_figure_svg(fig, out_svg_path)
    plt.close(fig)
    return svg


def _plot_nmse_vs_n_on_ax(
    ax: Any,
    *,
    row_labels: list[str],
    n_list: list[int],
    nmse_h: np.ndarray,
    nmse_decode_shared: np.ndarray,
    nmse_decode_hellinger_shared: np.ndarray | None,
) -> None:
    nmse_arr = np.asarray(nmse_h, dtype=np.float64)
    nmse_decode_arr = np.asarray(nmse_decode_shared, dtype=np.float64).ravel()
    nmse_decode_h_arr = None
    if nmse_decode_hellinger_shared is not None:
        nmse_decode_h_arr = np.asarray(nmse_decode_hellinger_shared, dtype=np.float64).ravel()
    n_arr = np.asarray(n_list, dtype=np.float64).ravel()
    if nmse_arr.shape != (len(row_labels), len(n_list)):
        raise ValueError(
            f"nmse_h shape mismatch: expected {(len(row_labels), len(n_list))}, got {nmse_arr.shape}."
        )
    if nmse_decode_arr.shape != (len(n_list),):
        raise ValueError(
            f"nmse_decode_shared shape mismatch: expected {(len(n_list),)}, got {nmse_decode_arr.shape}."
        )
    if nmse_decode_h_arr is not None and nmse_decode_h_arr.shape != (len(n_list),):
        raise ValueError(
            f"nmse_decode_hellinger_shared shape mismatch: expected {(len(n_list),)}, "
            f"got {nmse_decode_h_arr.shape}."
        )

    for i, label in enumerate(row_labels):
        ax.plot(
            n_arr,
            nmse_arr[i],
            marker="o",
            linewidth=1.8,
            markersize=4.0,
            label=f"{label} (H vs GT)",
        )
    ax.plot(
        n_arr,
        nmse_decode_arr,
        color="black",
        linestyle="--",
        marker="s",
        linewidth=1.6,
        markersize=3.5,
        label="decoding (shared)",
    )
    if nmse_decode_h_arr is not None:
        ax.plot(
            n_arr,
            nmse_decode_h_arr,
            color="dimgray",
            linestyle=":",
            marker="D",
            linewidth=1.6,
            markersize=3.5,
            label="classifier LLR sqrt(H^2) vs analytic GT (shared)",
        )
    finite_h = nmse_arr[np.isfinite(nmse_arr)]
    finite_d = nmse_decode_arr[np.isfinite(nmse_decode_arr)]
    finite_parts = [finite_h.ravel(), finite_d.ravel()]
    if nmse_decode_h_arr is not None:
        finite_parts.append(nmse_decode_h_arr[np.isfinite(nmse_decode_h_arr)].ravel())
    finite = np.concatenate([x for x in finite_parts if x.size]) if any(x.size for x in finite_parts) else np.array([])
    if finite.size > 0 and np.all(finite > 0):
        ax.set_yscale("log")
    ax.set_xlabel("dataset size n", fontsize=10)
    ax.set_ylabel("normalized MSE (off-diagonal, vs GT)", fontsize=10)
    ax.set_title("NMSE vs n", fontsize=11)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=8)


def _render_nmse_vs_n_panel(
    *,
    row_labels: list[str],
    n_list: list[int],
    nmse_h: np.ndarray,
    nmse_decode_shared: np.ndarray,
    out_svg_path: str,
    nmse_decode_hellinger_shared: np.ndarray | None = None,
) -> str:
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    _plot_nmse_vs_n_on_ax(
        ax,
        row_labels=row_labels,
        n_list=n_list,
        nmse_h=nmse_h,
        nmse_decode_shared=nmse_decode_shared,
        nmse_decode_hellinger_shared=nmse_decode_hellinger_shared,
    )
    fig.tight_layout()
    svg = _save_figure_svg(fig, out_svg_path)
    plt.close(fig)
    return svg


def _plot_corr_vs_n_on_ax(
    ax: Any,
    *,
    row_labels: list[str],
    n_list: list[int],
    corr_h: np.ndarray,
    corr_decode_shared: np.ndarray,
    corr_hellinger_lb: np.ndarray,
    corr_hellinger_ub: np.ndarray,
    show_hellinger_bounds: bool,
    corr_decode_hellinger_shared: np.ndarray | None,
    with_xlabel: bool = True,
) -> None:
    corr_h_arr = np.asarray(corr_h, dtype=np.float64)
    corr_decode_arr = np.asarray(corr_decode_shared, dtype=np.float64).ravel()
    corr_decode_h_arr = None
    if corr_decode_hellinger_shared is not None:
        corr_decode_h_arr = np.asarray(corr_decode_hellinger_shared, dtype=np.float64).ravel()
    corr_lb_arr = np.asarray(corr_hellinger_lb, dtype=np.float64)
    corr_ub_arr = np.asarray(corr_hellinger_ub, dtype=np.float64)
    n_arr = np.asarray(n_list, dtype=np.float64).ravel()
    if corr_h_arr.shape != (len(row_labels), len(n_list)):
        raise ValueError(
            f"corr_h shape mismatch: expected {(len(row_labels), len(n_list))}, got {corr_h_arr.shape}."
        )
    if bool(show_hellinger_bounds):
        if corr_lb_arr.shape != (len(row_labels), len(n_list)):
            raise ValueError(
                f"corr_hellinger_lb shape mismatch: expected {(len(row_labels), len(n_list))}, got {corr_lb_arr.shape}."
            )
        if corr_ub_arr.shape != (len(row_labels), len(n_list)):
            raise ValueError(
                f"corr_hellinger_ub shape mismatch: expected {(len(row_labels), len(n_list))}, got {corr_ub_arr.shape}."
            )
    if corr_decode_arr.shape != (len(n_list),):
        raise ValueError(f"corr_decode shape mismatch: expected {(len(n_list),)}, got {corr_decode_arr.shape}.")
    if corr_decode_h_arr is not None and corr_decode_h_arr.shape != (len(n_list),):
        raise ValueError(
            f"corr_decode_hellinger shape mismatch: expected {(len(n_list),)}, got {corr_decode_h_arr.shape}."
        )

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for i, label in enumerate(row_labels):
        color = colors[i % len(colors)] if colors else None
        ax.plot(
            n_arr,
            corr_h_arr[i],
            color=color,
            marker="o",
            linewidth=1.8,
            markersize=4.0,
            label=f"{label} (H vs GT)",
        )
        if bool(show_hellinger_bounds):
            ax.plot(
                n_arr,
                corr_lb_arr[i],
                color=color,
                linestyle=":",
                marker="^",
                linewidth=1.3,
                markersize=3.2,
                label=f"{label} (H LB vs decoding)",
            )
            ax.plot(
                n_arr,
                corr_ub_arr[i],
                color=color,
                linestyle="-.",
                marker="v",
                linewidth=1.3,
                markersize=3.2,
                label=f"{label} (H UB vs decoding)",
            )
    ax.plot(
        n_arr,
        corr_decode_arr,
        color="black",
        linestyle="--",
        marker="s",
        linewidth=1.6,
        markersize=3.5,
        label="decoding (shared)",
    )
    if corr_decode_h_arr is not None:
        ax.plot(
            n_arr,
            corr_decode_h_arr,
            color="dimgray",
            linestyle=":",
            marker="D",
            linewidth=1.6,
            markersize=3.5,
            label="classifier LLR sqrt(H^2) vs analytic GT (shared)",
        )
    if with_xlabel:
        ax.set_xlabel("dataset size n", fontsize=10)
    ax.set_ylabel("correlation (off-diagonal Pearson r)", fontsize=10)
    ax.set_title("Correlation vs n", fontsize=11)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=8)


def _render_corr_vs_n_panel(
    *,
    row_labels: list[str],
    n_list: list[int],
    corr_h: np.ndarray,
    corr_decode_shared: np.ndarray,
    corr_hellinger_lb: np.ndarray,
    corr_hellinger_ub: np.ndarray,
    show_hellinger_bounds: bool,
    out_svg_path: str,
    corr_decode_hellinger_shared: np.ndarray | None = None,
) -> str:
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.8))
    _plot_corr_vs_n_on_ax(
        ax,
        row_labels=row_labels,
        n_list=n_list,
        corr_h=corr_h,
        corr_decode_shared=corr_decode_shared,
        corr_hellinger_lb=corr_hellinger_lb,
        corr_hellinger_ub=corr_hellinger_ub,
        show_hellinger_bounds=show_hellinger_bounds,
        corr_decode_hellinger_shared=corr_decode_hellinger_shared,
        with_xlabel=True,
    )
    fig.tight_layout()
    svg = _save_figure_svg(fig, out_svg_path)
    plt.close(fig)
    return svg


def _render_corr_nmse_two_panel(
    *,
    row_labels: list[str],
    n_list: list[int],
    corr_h: np.ndarray,
    corr_decode_shared: np.ndarray,
    corr_hellinger_lb: np.ndarray,
    corr_hellinger_ub: np.ndarray,
    show_hellinger_bounds: bool,
    corr_decode_hellinger_shared: np.ndarray | None,
    nmse_h: np.ndarray,
    nmse_decode_shared: np.ndarray,
    nmse_decode_hellinger_shared: np.ndarray | None,
    out_svg_path: str,
) -> str:
    fig, axes2 = plt.subplots(2, 1, figsize=(4.8, 6.8), sharex=True, squeeze=False, layout="constrained")
    ax_corr = axes2[0, 0]
    ax_nmse = axes2[1, 0]
    _plot_corr_vs_n_on_ax(
        ax_corr,
        row_labels=row_labels,
        n_list=n_list,
        corr_h=corr_h,
        corr_decode_shared=corr_decode_shared,
        corr_hellinger_lb=corr_hellinger_lb,
        corr_hellinger_ub=corr_hellinger_ub,
        show_hellinger_bounds=show_hellinger_bounds,
        corr_decode_hellinger_shared=corr_decode_hellinger_shared,
        with_xlabel=False,
    )
    _plot_nmse_vs_n_on_ax(
        ax_nmse,
        row_labels=row_labels,
        n_list=n_list,
        nmse_h=nmse_h,
        nmse_decode_shared=nmse_decode_shared,
        nmse_decode_hellinger_shared=nmse_decode_hellinger_shared,
    )
    svg = _save_figure_svg(fig, out_svg_path)
    plt.close(fig)
    return svg


def main(argv: list[str] | None = None, *, sir_first_default: bool = False) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    p = build_sir_first_parser() if sir_first_default else build_parser()
    args = p.parse_args(argv)
    if sir_first_default:
        args.sir_first = True
    args.output_dir = os.path.abspath(str(args.output_dir))
    args.dataset_npz = os.path.abspath(str(args.dataset_npz))

    row_specs = _parse_theta_field_rows(args)
    row_labels = [r.label for r in row_specs]
    row_methods = [r.method for r in row_specs]
    row_arches = [("" if r.arch is None else r.arch) for r in row_specs]
    setattr(args, "theta_field_rows_resolved", row_labels)
    setattr(args, "theta_field_row_methods_resolved", row_methods)
    setattr(args, "theta_field_row_arches_resolved", row_arches)
    validation_row = next((r for r in row_specs if r.method not in _NO_TRAIN_METHODS), row_specs[0])
    validation_method = _VAE_WRAPPED_METHODS.get(validation_row.method, validation_row.method)
    args.theta_field_method = "theta_flow" if validation_method in _NO_TRAIN_METHODS else validation_method
    if validation_row.arch is not None:
        args.flow_arch = validation_row.arch
    conv._validate_cli(args)
    _validate_cli_for_rows(args, row_specs)
    ns = conv._parse_n_list(args.n_list)

    if bool(getattr(args, "visualization_only", False)):
        os.makedirs(args.output_dir, exist_ok=True)
        cached = _load_cached_twofig_results(args.output_dir)
        row_labels_cached = _theta_field_row_labels_from_array(np.asarray(cached["theta_field_rows"]))
        _validate_cached_twofig_cli(args, cached, ns, row_labels_cached, row_labels)
        bundle = load_shared_dataset_npz(args.dataset_npz)
        meta = bundle.meta
        meta_family = str(meta.get("dataset_family", ""))
        if meta_family != str(args.dataset_family):
            raise ValueError(
                f"NPZ meta dataset_family={meta_family!r} does not match --dataset-family={str(args.dataset_family)!r}."
            )
        cached_seed = int(np.asarray(cached["dataset_meta_seed"]).reshape(-1)[0])
        if int(meta.get("seed", -1)) != cached_seed:
            raise ValueError(
                f"Dataset NPZ meta seed={int(meta.get('seed', -1))} does not match cached dataset_meta_seed={cached_seed}. "
                "Use the same --dataset-npz as the run that produced h_decoding_twofig_results.npz."
            )
        n_pool = int(bundle.theta_all.shape[0])
        perm_seed_v = int(np.asarray(cached.get("perm_seed", np.int64(0))).reshape(-1)[0])
        _run_twofig_visualization_only(
            args,
            row_labels=row_labels_cached,
            ns=ns,
            meta=meta,
            n_pool=n_pool,
            perm_seed=perm_seed_v,
            cached=cached,
        )
        return

    os.makedirs(args.output_dir, exist_ok=True)
    bundle = load_shared_dataset_npz(args.dataset_npz)
    meta = bundle.meta
    meta_family = str(meta.get("dataset_family", ""))
    if meta_family != str(args.dataset_family):
        raise ValueError(
            f"NPZ meta dataset_family={meta_family!r} does not match --dataset-family={str(args.dataset_family)!r}. "
            "Regenerate with matching make_dataset.py --dataset-family, or pass --dataset-family to match the NPZ."
        )
    _validate_categorical_rows(meta, row_specs)

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

    theta_binning_mode = _theta_binning_mode(args)
    n_bins_x = int(args.num_theta_bins)
    n_bins_y = _num_theta_bins_y(args)
    n_bins = n_bins_x
    base_seed = int(args.run_seed) if args.run_seed is not None else int(meta["seed"])
    perm_seed = base_seed + int(args.subset_seed_offset)
    rng_perm = np.random.default_rng(perm_seed)
    perm = rng_perm.permutation(n_pool)

    theta_raw_all = np.asarray(bundle.theta_all, dtype=np.float64)
    theta_grid_edges0 = np.asarray([], dtype=np.float64)
    theta_grid_edges1 = np.asarray([], dtype=np.float64)
    theta_grid_shape = np.asarray([], dtype=np.int64)
    if str(meta.get("theta_type", "")) == "categorical":
        if theta_binning_mode != "theta1":
            raise ValueError("Categorical datasets require --theta-binning-mode theta1.")
        k_cat = int(meta.get("num_categories", n_bins_x))
        if n_bins_x != k_cat:
            raise ValueError(
                f"Categorical dataset requires --num-theta-bins == num_categories ({k_cat}); got {n_bins_x}."
            )
        theta_scalar_all, theta_ref, edges, _, _, bin_idx_all = conv.prepare_categorical_binning_for_convergence(
            theta_raw_all,
            k_cat,
        )
        centers = bin_centers_from_edges(edges)
        n_bins = k_cat
    elif theta_binning_mode == "theta2_grid":
        grid = conv.prepare_theta2_grid_binning_for_convergence(
            theta_raw_all,
            perm,
            int(args.n_ref),
            n_bins_x,
            n_bins_y,
        )
        theta_scalar_all = grid.theta_scalar_all
        theta_ref = grid.theta_ref
        edges = grid.edges0
        bin_idx_all = grid.bin_idx_all
        centers = grid.centers
        n_bins = int(grid.grid_shape[0] * grid.grid_shape[1])
        theta_grid_edges0 = grid.edges0
        theta_grid_edges1 = grid.edges1
        theta_grid_shape = np.asarray(grid.grid_shape, dtype=np.int64)
    else:
        theta_scalar_all, theta_ref, edges, _, _, bin_idx_all = conv.prepare_theta_binning_for_convergence(
            theta_raw_all,
            perm,
            int(args.n_ref),
            n_bins,
        )
        centers = bin_centers_from_edges(edges)

    theta_state_all: np.ndarray | None = None
    theta_fourier_ref_range: np.ndarray | float = float("nan")
    theta_fourier_period: np.ndarray | float = float("nan")
    theta_fourier_center: np.ndarray | float = float("nan")
    if bool(getattr(args, "theta_flow_onehot_state", False)):
        theta_state_all = np.eye(n_bins, dtype=np.float64)[bin_idx_all]
        print(
            f"[twofig] theta_flow one-hot state enabled: theta -> one_hot(bin(theta), K={n_bins})",
            flush=True,
        )
    elif bool(getattr(args, "theta_flow_fourier_state", False)):
        if theta_binning_mode == "theta2_grid":
            theta_fourier_src = theta_raw_all
            theta_fourier_ref_rows = theta_ref
        else:
            theta_fourier_src, theta_fourier_ref_rows = conv.theta_phys_rows_and_ref_for_fourier(
                theta_raw_all,
                perm,
                int(args.n_ref),
            )
        theta_state_all, theta_fourier_ref_range, theta_fourier_period, theta_fourier_center = conv._build_theta_fourier_state(
            theta_fourier_src,
            theta_ref=theta_fourier_ref_rows,
            k=int(args.theta_flow_fourier_k),
            period_mult=float(args.theta_flow_fourier_period_mult),
            include_linear=bool(args.theta_flow_fourier_include_linear),
        )
        print(
            conv.format_theta_fourier_state_log_message(
                tag="[twofig]",
                state_dim=int(theta_state_all.shape[1]),
                k=int(args.theta_flow_fourier_k),
                ref_range_vec=np.asarray(theta_fourier_ref_range, dtype=np.float64),
                period_vec=np.asarray(theta_fourier_period, dtype=np.float64),
                center_vec=np.asarray(theta_fourier_center, dtype=np.float64),
                period_mult=float(args.theta_flow_fourier_period_mult),
                include_linear=bool(args.theta_flow_fourier_include_linear),
            ),
            flush=True,
        )

    clf_rs = base_seed if int(args.clf_random_state) < 0 else int(args.clf_random_state)

    dataset_for_gt = build_dataset_from_meta(meta)
    gt_seed = base_seed if int(args.gt_hellinger_seed) < 0 else int(args.gt_hellinger_seed)
    if hasattr(dataset_for_gt, "rng"):
        dataset_for_gt.rng = np.random.default_rng(gt_seed)
    gt_n_mc = int(args.n_ref) // n_bins
    t_gt0 = time.time()
    gt_method = "analytic_gaussian_centers"
    gt_theta_centers = theta_centers_for_analytic_gt(dataset_for_gt, centers)
    try:
        h_gt_mc = estimate_hellinger_sq_grid_centers_analytic(
            dataset_for_gt,
            gt_theta_centers,
            symmetrize=bool(args.gt_hellinger_symmetrize),
        )
    except TypeError:
        gt_method = "mc_likelihood"
        if theta_binning_mode == "theta2_grid":
            h_gt_mc = estimate_hellinger_sq_grid_centers_mc(
                dataset_for_gt,
                centers,
                n_mc=gt_n_mc,
                symmetrize=bool(args.gt_hellinger_symmetrize),
            )
        else:
            h_gt_mc = estimate_hellinger_sq_one_sided_mc(
                dataset_for_gt,
                centers,
                n_mc=gt_n_mc,
                symmetrize=bool(args.gt_hellinger_symmetrize),
            )
    h_gt_sqrt = conv._sqrt_h_like(h_gt_mc)
    print(
        f"[twofig] GT Hellinger ({gt_method}) theta_binning_mode={theta_binning_mode} "
        f"n_bins={n_bins} center_shape={gt_theta_centers.shape} "
        f"legacy_n_mc={gt_n_mc} wall time: {time.time() - t_gt0:.1f}s",
        flush=True,
    )

    ref_dir = os.path.join(args.output_dir, "reference")
    os.makedirs(ref_dir, exist_ok=True)
    subset_ref = conv._subset_bundle(
        bundle,
        perm,
        int(args.n_ref),
        meta,
        bin_idx_all=bin_idx_all,
        theta_state_all=theta_state_all,
    )
    decode_gt_x_train: np.ndarray | None = None
    decode_gt_x_all: np.ndarray | None = None
    if bool(meta.get("pr_autoencoder_embedded")):
        override_npz = str(getattr(args, "decode_source_npz", "") or "").strip() or None
        fallback_emb = bool(getattr(args, "decode_gt_fallback_embedded", False))
        native_gt, tried_gt = load_native_bundle_for_pr_gt_decoding(
            bundle,
            meta,
            args.dataset_npz,
            override_npz,
        )
        if native_gt is not None:
            decode_gt_x_train, decode_gt_x_all = decoding_x_train_all_from_native(
                native_gt,
                perm,
                int(args.n_ref),
                meta,
            )
            print(
                "[twofig] GT decoding pairwise logistic regression uses native (pre-PR) x "
                f"(dim={int(decode_gt_x_all.shape[1])}); sweep decoding uses embedded x.",
                flush=True,
            )
        elif fallback_emb:
            msg_tried = "\n  ".join(tried_gt) if tried_gt else "(no candidate paths)"
            print(
                "[twofig] WARN: native NPZ missing for GT decoding; tried:\n  "
                f"{msg_tried}\n"
                "  Using embedded x for GT decoding (--decode-gt-fallback-embedded).",
                flush=True,
            )
        else:
            msg_tried = "\n  ".join(tried_gt) if tried_gt else "(no candidate paths)"
            raise FileNotFoundError(
                "PR-embedded dataset: could not load native source NPZ for GT decoding (pre-projection x). "
                "Pass --decode-source-npz PATH to the low-dimensional archive, or "
                "--decode-gt-fallback-embedded to use embedded x. Tried:\n  "
                f"{msg_tried}"
            )
    clf_ref = conv._pairwise_clf_from_bundle(
        args=args,
        meta=meta,
        subset=subset_ref,
        output_dir=ref_dir,
        n_bins=n_bins,
        clf_min_class_count=int(args.clf_min_class_count),
        clf_random_state=clf_rs,
        decode_x_train=decode_gt_x_train,
        decode_x_all=decode_gt_x_all,
    )

    h_sweep_by_method: list[np.ndarray] = []
    clf_sweep_shared: list[np.ndarray] = []
    wall_s = np.full((len(row_specs), len(ns)), np.nan, dtype=np.float64)

    sweep_root = os.path.join(args.output_dir, "sweep_runs")
    if bool(args.keep_intermediate):
        os.makedirs(sweep_root, exist_ok=True)
    loss_root = os.path.join(args.output_dir, "training_losses")
    os.makedirs(loss_root, exist_ok=True)
    dev = conv.require_device(str(args.device))

    decode_dir = os.path.join(args.output_dir, "decode_shared")
    os.makedirs(decode_dir, exist_ok=True)
    sir_first = bool(getattr(args, "sir_first", False))
    sir_projection_root = os.path.abspath(os.path.join(args.output_dir, "sir_projections"))
    sir_subset_cache: dict[int, SweepSubset] = {}
    sir_meta_cache: dict[int, dict[str, np.ndarray | int | float]] = {}
    sir_path_cache: dict[int, str] = {}
    for n in ns:
        subset_n = conv._subset_bundle(
            bundle,
            perm,
            int(n),
            meta,
            bin_idx_all=bin_idx_all,
            theta_state_all=theta_state_all,
        )
        if sir_first:
            theta_fit_subset_n = conv._subset_bundle(
                bundle,
                perm,
                int(n),
                meta,
                bin_idx_all=bin_idx_all,
                theta_state_all=None,
            )
            subset_n, sir_meta_n, sir_path_n = _project_sir_first_subset(
                subset=subset_n,
                theta_fit_subset=theta_fit_subset_n,
                args=args,
                n=int(n),
                sir_projection_root=sir_projection_root,
            )
            sir_subset_cache[int(n)] = subset_n
            sir_meta_cache[int(n)] = sir_meta_n
            sir_path_cache[int(n)] = sir_path_n
        clf_n = conv._pairwise_clf_from_bundle(
            args=args,
            meta=subset_n.bundle.meta if sir_first else meta,
            subset=subset_n,
            output_dir=os.path.join(decode_dir, f"n_{int(n):06d}"),
            n_bins=n_bins,
            clf_min_class_count=int(args.clf_min_class_count),
            clf_random_state=clf_rs,
        )
        clf_sweep_shared.append(np.asarray(clf_n, dtype=np.float64))

    for m_idx, row in enumerate(row_specs):
        method_h: list[np.ndarray] = []
        print(f"[twofig] row={row.label} start", flush=True)
        for k, n in enumerate(ns):
            t1 = time.time()
            tmp_ctx: tempfile.TemporaryDirectory[str] | None = None
            args_method = deepcopy(args)
            base_method = _VAE_WRAPPED_METHODS.get(row.method, row.method)
            args_method.theta_field_method = base_method
            if row.arch is not None:
                args_method.flow_arch = row.arch
            try:
                if sir_first:
                    subset_n = sir_subset_cache[int(n)]
                    sir_meta_n = sir_meta_cache[int(n)]
                    sir_path_n = sir_path_cache[int(n)]
                else:
                    subset_n = conv._subset_bundle(
                        bundle,
                        perm,
                        int(n),
                        meta,
                        bin_idx_all=bin_idx_all,
                        theta_state_all=theta_state_all,
                    )
                    sir_meta_n = {}
                    sir_path_n = ""
                vae_payload_n: dict[str, Any] | None = None
                if row.method in _VAE_WRAPPED_METHODS:
                    subset_n, vae_payload_n = _subset_with_vae_mean_x(
                        subset=subset_n,
                        args=args,
                        dev=dev,
                    )
                if base_method == "bin_gaussian":
                    variance_floor = float(
                        getattr(args, "flow_theta_reg_variance_floor", getattr(args, "flow_x_reg_variance_floor", 1e-6))
                    )
                    bg_h2 = conv._binned_gaussian_hellinger_sq(
                        subset_n,
                        n_bins,
                        variance_floor=variance_floor,
                    )
                    method_h.append(np.asarray(conv._sqrt_h_like(bg_h2), dtype=np.float64))
                    wall_s[m_idx, k] = time.time() - t1
                    if vae_payload_n is not None:
                        row_loss_dir = os.path.join(loss_root, _sanitize_row_label(row.label))
                        os.makedirs(row_loss_dir, exist_ok=True)
                        _save_vae_wrapper_loss_npz(
                            os.path.abspath(os.path.join(row_loss_dir, f"n_{int(n):06d}.npz")),
                            method_name=row.label,
                            vae_payload=vae_payload_n,
                        )
                    print(
                        f"[twofig] row={row.label} n={n} done in {wall_s[m_idx, k]:.1f}s "
                        f"(binned Gaussian, variance_floor={variance_floor:g})",
                        flush=True,
                    )
                    continue

                if bool(args.keep_intermediate):
                    row_dir = row.label.replace(":", "__")
                    run_dir = os.path.join(sweep_root, row_dir, f"n_{n:06d}")
                    os.makedirs(run_dir, exist_ok=True)
                else:
                    row_prefix = row.label.replace(":", "__")
                    tmp_ctx = tempfile.TemporaryDirectory(prefix=f"h_twofig_{row_prefix}_n{n}_", dir=args.output_dir)
                    run_dir = tmp_ctx.name

                loaded_n, _, _ = conv._estimate_one(
                    args=args_method,
                    meta=subset_n.bundle.meta,
                    bundle=subset_n.bundle,
                    output_dir=run_dir,
                    n_bins=n_bins,
                    bin_train=subset_n.bin_train,
                    bin_validation=subset_n.bin_validation,
                    bin_all=subset_n.bin_all,
                )
                src_loss_npz = os.path.abspath(os.path.join(run_dir, "score_prior_training_losses.npz"))
                if not os.path.isfile(src_loss_npz):
                    raise FileNotFoundError(
                        f"Expected per-run training loss artifact is missing: {src_loss_npz}"
                    )
                row_loss_dir = os.path.join(loss_root, _sanitize_row_label(row.label))
                os.makedirs(row_loss_dir, exist_ok=True)
                dst_loss_npz = os.path.abspath(os.path.join(row_loss_dir, f"n_{int(n):06d}.npz"))
                shutil.copy2(src_loss_npz, dst_loss_npz)
                if vae_payload_n is not None:
                    with np.load(dst_loss_npz, allow_pickle=True) as z_loss:
                        payload_loss = {k2: np.asarray(z_loss[k2]) for k2 in z_loss.files}
                    payload_loss.update({f"vae_{k2}": v2 for k2, v2 in vae_payload_n.items()})
                    np.savez_compressed(dst_loss_npz, **payload_loss)
                if sir_first:
                    _annotate_npz_with_sir(dst_loss_npz, sir_meta_n, sir_path_n)
                    for h_name in ("h_matrix_results.npz", "h_matrix_results_theta_cov.npz"):
                        _annotate_npz_with_sir(os.path.abspath(os.path.join(run_dir, h_name)), sir_meta_n, sir_path_n)
                if loaded_n.h_sym.shape[0] != subset_n.bin_all.shape[0]:
                    raise ValueError(
                        f"h_sym rows {loaded_n.h_sym.shape[0]} do not match subset bins length {subset_n.bin_all.shape[0]}."
                    )
                h_n, _ = conv.vhb.average_matrix_by_bins(loaded_n.h_sym, subset_n.bin_all, n_bins)
                method_h.append(np.asarray(conv._sqrt_h_like(h_n), dtype=np.float64))
                wall_s[m_idx, k] = time.time() - t1
                print(f"[twofig] row={row.label} n={n} done in {wall_s[m_idx, k]:.1f}s", flush=True)
            finally:
                if tmp_ctx is not None:
                    tmp_ctx.cleanup()
        h_sweep_by_method.append(np.stack(method_h, axis=0))

    h_sweep_arr = np.stack(h_sweep_by_method, axis=0)
    clf_sweep_arr = np.stack(clf_sweep_shared, axis=0)
    h_gt_sqrt_imp = conv.vhb.impute_offdiag_nan_mean(np.asarray(h_gt_sqrt, dtype=np.float64))
    clf_ref_imp = conv.vhb.impute_offdiag_nan_mean(np.asarray(clf_ref, dtype=np.float64))
    corr_h_binned_vs_gt_mc = np.full((len(row_specs), len(ns)), np.nan, dtype=np.float64)
    for i in range(len(row_specs)):
        for j in range(len(ns)):
            corr_h_binned_vs_gt_mc[i, j] = conv.vhb.matrix_corr_offdiag_pearson(
                conv.vhb.impute_offdiag_nan_mean(np.asarray(h_sweep_arr[i, j], dtype=np.float64)),
                h_gt_sqrt_imp,
            )
    corr_decode_vs_ref_shared = np.full((len(ns),), np.nan, dtype=np.float64)
    for j in range(len(ns)):
        corr_decode_vs_ref_shared[j] = conv.vhb.matrix_corr_offdiag_pearson(
            conv.vhb.impute_offdiag_nan_mean(np.asarray(clf_sweep_arr[j], dtype=np.float64)),
            clf_ref_imp,
        )
    hellinger_acc_lb_sweep, hellinger_acc_ub_sweep = _hellinger_sqrt_to_accuracy_bounds(h_sweep_arr)
    corr_hellinger_lb_vs_decode_shared = _corr_hellinger_bounds_vs_decode_shared(
        hellinger_acc_lb_sweep,
        clf_sweep_arr,
    )
    corr_hellinger_ub_vs_decode_shared = _corr_hellinger_bounds_vs_decode_shared(
        hellinger_acc_ub_sweep,
        clf_sweep_arr,
    )
    nmse_h_binned_vs_gt_mc = _nmse_h_binned_vs_gt_mc(h_sweep_arr, h_gt_sqrt)
    nmse_decode_vs_ref_shared = _nmse_decode_vs_ref_shared(clf_sweep_arr, clf_ref)
    continuous_footer = {
        "h_gt_sqrt": h_gt_sqrt,
        "decode_ref": clf_ref,
        "n_ref": int(args.n_ref),
        "decode_sweep_for_decode_limits": clf_sweep_arr,
    }

    sweep_svg = _render_method_sweep_panel(
        row_labels=row_labels,
        h_sweep=h_sweep_arr,
        clf_sweep_shared=clf_sweep_arr,
        n_list=ns,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_twofig_sweep.svg"),
        n_bins=n_bins,
        theta_centers=centers,
        clf_ref_decode_limits=np.asarray(clf_ref, dtype=np.float64),
        continuous_gt_footer=continuous_footer,
    )
    corr_nmse_svg = _render_corr_nmse_two_panel(
        row_labels=row_labels,
        n_list=ns,
        corr_h=corr_h_binned_vs_gt_mc,
        corr_decode_shared=corr_decode_vs_ref_shared,
        corr_hellinger_lb=corr_hellinger_lb_vs_decode_shared,
        corr_hellinger_ub=corr_hellinger_ub_vs_decode_shared,
        show_hellinger_bounds=bool(getattr(args, "show_hellinger_bound_corr", False)),
        corr_decode_hellinger_shared=None,
        nmse_h=nmse_h_binned_vs_gt_mc,
        nmse_decode_shared=nmse_decode_vs_ref_shared,
        nmse_decode_hellinger_shared=None,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_twofig_corr_nmse.svg"),
    )
    loss_panel_svg = _render_row_n_training_losses_panel(
        row_labels=row_labels,
        n_list=ns,
        loss_root=loss_root,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_twofig_training_losses_panel.svg"),
    )

    if theta_state_all is None:
        tf_rr_save = np.float64(np.nan)
        tf_per_save = np.float64(np.nan)
        tf_cen_save = np.float64(np.nan)
    else:
        tf_rr_save = np.asarray(theta_fourier_ref_range, dtype=np.float64)
        tf_per_save = np.asarray(theta_fourier_period, dtype=np.float64)
        tf_cen_save = np.asarray(theta_fourier_center, dtype=np.float64)

    out_npz = os.path.join(args.output_dir, "h_decoding_twofig_results.npz")
    np.savez_compressed(
        out_npz,
        n=np.asarray(ns, dtype=np.int64),
        wall_seconds=np.asarray(wall_s, dtype=np.float64),
        n_ref=np.int64(args.n_ref),
        theta_field_methods=np.asarray(row_methods, dtype=np.str_),
        theta_field_rows=np.asarray(row_labels, dtype=np.str_),
        theta_field_row_methods=np.asarray(row_methods, dtype=np.str_),
        theta_field_row_arches=np.asarray(row_arches, dtype=np.str_),
        perm_seed=np.int64(perm_seed),
        convergence_base_seed=np.int64(base_seed),
        dataset_meta_seed=np.int64(meta["seed"]),
        theta_flow_fourier_state=np.bool_(bool(getattr(args, "theta_flow_fourier_state", False))),
        theta_flow_fourier_k=np.int64(getattr(args, "theta_flow_fourier_k", 0)),
        theta_flow_fourier_period_mult=np.float64(getattr(args, "theta_flow_fourier_period_mult", np.nan)),
        theta_flow_fourier_include_linear=np.bool_(bool(getattr(args, "theta_flow_fourier_include_linear", False))),
        theta_fourier_state_dim=np.int64(0 if theta_state_all is None else int(theta_state_all.shape[1])),
        theta_fourier_ref_range=tf_rr_save,
        theta_fourier_period=tf_per_save,
        theta_fourier_center=tf_cen_save,
        theta_binning_mode=np.asarray([theta_binning_mode], dtype=object),
        theta_grid_shape=np.asarray(theta_grid_shape, dtype=np.int64),
        theta_bin_edges=np.asarray(edges, dtype=np.float64),
        theta_bin_centers=np.asarray(centers, dtype=np.float64),
        theta_grid_edges_0=np.asarray(theta_grid_edges0, dtype=np.float64),
        theta_grid_edges_1=np.asarray(theta_grid_edges1, dtype=np.float64),
        theta_grid_centers=np.asarray(centers if theta_binning_mode == "theta2_grid" else [], dtype=np.float64),
        gt_hellinger_method=np.asarray([gt_method], dtype=object),
        gt_hellinger_theta_centers=np.asarray(gt_theta_centers, dtype=np.float64),
        gt_hellinger_n_mc=np.int64(gt_n_mc),
        gt_hellinger_seed=np.int64(gt_seed),
        gt_hellinger_symmetrize=np.int32(1 if bool(args.gt_hellinger_symmetrize) else 0),
        h_gt_sqrt=np.asarray(h_gt_sqrt, dtype=np.float64),
        decode_ref=np.asarray(clf_ref, dtype=np.float64),
        h_binned_sweep=np.asarray(h_sweep_arr, dtype=np.float64),
        decode_sweep=np.asarray(clf_sweep_arr, dtype=np.float64),
        hellinger_acc_lb_sweep=np.asarray(hellinger_acc_lb_sweep, dtype=np.float64),
        hellinger_acc_ub_sweep=np.asarray(hellinger_acc_ub_sweep, dtype=np.float64),
        corr_h_binned_vs_gt_mc=np.asarray(corr_h_binned_vs_gt_mc, dtype=np.float64),
        nmse_h_binned_vs_gt_mc=np.asarray(nmse_h_binned_vs_gt_mc, dtype=np.float64),
        corr_hellinger_lb_vs_decode_shared=np.asarray(corr_hellinger_lb_vs_decode_shared, dtype=np.float64),
        corr_hellinger_ub_vs_decode_shared=np.asarray(corr_hellinger_ub_vs_decode_shared, dtype=np.float64),
        corr_decode_vs_ref_shared=np.asarray(corr_decode_vs_ref_shared, dtype=np.float64),
        nmse_decode_vs_ref_shared=np.asarray(nmse_decode_vs_ref_shared, dtype=np.float64),
        column_n=np.asarray(ns, dtype=np.int64),
        corr_curve_svg=np.asarray(os.path.abspath(corr_nmse_svg), dtype=np.str_),
        nmse_curve_svg=np.asarray(os.path.abspath(corr_nmse_svg), dtype=np.str_),
        corr_nmse_curve_svg=np.asarray(os.path.abspath(corr_nmse_svg), dtype=np.str_),
        training_losses_root=np.asarray(os.path.abspath(loss_root), dtype=np.str_),
        training_losses_panel_svg=np.asarray(os.path.abspath(loss_panel_svg), dtype=np.str_),
        dataset_npz=np.asarray(os.path.abspath(str(args.dataset_npz)), dtype=np.str_),
        dataset_family=np.asarray(str(meta.get("dataset_family", "")), dtype=np.str_),
        dataset_pool_size=np.int64(n_pool),
        sir_enabled=np.bool_(sir_first),
        sir_dim=np.int64(getattr(args, "sir_dim", 0) if sir_first else 0),
        sir_num_bins=np.int64(getattr(args, "sir_num_bins", 0) if sir_first else 0),
        sir_ridge=np.float64(getattr(args, "sir_ridge", np.nan) if sir_first else np.nan),
        sir_projection_root=np.asarray(os.path.abspath(sir_projection_root) if sir_first else "", dtype=np.str_),
        sir_projection_npz_by_n=np.asarray([sir_path_cache.get(int(n), "") for n in ns], dtype=np.str_),
    )

    summary_path = os.path.join(args.output_dir, "h_decoding_twofig_summary.txt")
    _write_summary(
        summary_path,
        args=args,
        meta=meta,
        n_pool=n_pool,
        perm_seed=perm_seed,
        out_npz=os.path.abspath(out_npz),
        sweep_svg=os.path.abspath(sweep_svg),
        corr_nmse_svg=os.path.abspath(corr_nmse_svg),
        loss_panel_svg=os.path.abspath(loss_panel_svg),
        training_losses_root=os.path.abspath(loss_root),
        h_sweep_shape=tuple(int(x) for x in h_sweep_arr.shape),
        decode_sweep_shape=tuple(int(x) for x in clf_sweep_arr.shape),
        hellinger_acc_lb_shape=tuple(int(x) for x in hellinger_acc_lb_sweep.shape),
        hellinger_acc_ub_shape=tuple(int(x) for x in hellinger_acc_ub_sweep.shape),
        corr_h_shape=tuple(int(x) for x in corr_h_binned_vs_gt_mc.shape),
        nmse_h_shape=tuple(int(x) for x in nmse_h_binned_vs_gt_mc.shape),
        corr_hellinger_lb_shape=tuple(int(x) for x in corr_hellinger_lb_vs_decode_shared.shape),
        corr_hellinger_ub_shape=tuple(int(x) for x in corr_hellinger_ub_vs_decode_shared.shape),
        show_hellinger_bound_corr=bool(getattr(args, "show_hellinger_bound_corr", False)),
        corr_decode_shape=tuple(int(x) for x in corr_decode_vs_ref_shared.shape),
        nmse_decode_shape=tuple(int(x) for x in nmse_decode_vs_ref_shared.shape),
        wall_seconds_shape=tuple(int(x) for x in wall_s.shape),
        theta_fourier_feature_dim=(None if theta_state_all is None else int(theta_state_all.shape[1])),
    )

    print("[twofig] Saved:", flush=True)
    print(f"  - {os.path.abspath(sweep_svg)}", flush=True)
    print(f"  - {os.path.abspath(corr_nmse_svg)}", flush=True)
    print(f"  - {os.path.abspath(loss_panel_svg)}", flush=True)
    print(f"  - {os.path.abspath(loss_root)}/", flush=True)
    print(f"  - {os.path.abspath(out_npz)}", flush=True)
    print(f"  - {os.path.abspath(summary_path)}", flush=True)


if __name__ == "__main__":
    main()
