#!/usr/bin/env python3
"""Two-figure H/decoding convergence study.

Reuses the full-compute pipeline from ``bin/study_h_decoding_convergence.py``
but emits only two matrix-figure artifacts:

1) ``h_decoding_twofig_sweep.svg``: columns over ``--n-list`` only
   (one row per method for estimated sqrt(H)-like binned matrices, plus one
   shared bottom row for decoding).
2) ``h_decoding_twofig_gt.svg``: left = approximate GT sqrt(H^2) matrix
   (MC likelihood), right = decoding matrix from the ``n_ref`` subset.

Also writes correlation, normalized-MSE-vs-n, and training-loss SVGs. Pass
``--visualization-only`` with the same ``--output-dir`` as a prior full run to
regenerate figures from ``h_decoding_twofig_results.npz`` without retraining.

Off-diagonal Pearson correlations and off-diagonal NMSE use
``visualize_h_matrix_binned.impute_offdiag_nan_mean`` on both matrices in each
pair (sweep vs GT / MC reference, and decoding sweep vs shared reference) before
the finite-pair mask inside each metric.
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

import study_h_decoding_convergence as conv
from fisher.hellinger_gt import (
    bin_centers_from_edges,
    estimate_hellinger_sq_grid_centers_mc,
    estimate_hellinger_sq_one_sided_mc,
)
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_meta, normalize_flow_arch, normalize_theta_field_method

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
# - linear_x_flow_t, linear_x_flow_scalar_t,
#   linear_x_flow_diagonal_t,
#   linear_x_flow_diagonal_theta_t,
#   linear_x_flow_low_rank_t (full A(t) + learnable U h(U^T x) correction;
#     static orthonormal U; divergence default: --lxf-low-rank-divergence-estimator hutchinson, --lxf-hutchinson-probes 1),
#   linear_x_flow_lr_t_ts (same scheduled low-rank correction but b(theta) only; mean-regression pretrain then freeze b),
#   linear_x_flow_low_rank_randb_t
_FLOW_BASED_METHODS = {"theta_flow", "theta_path_integral", "x_flow"}
_NO_TRAIN_METHODS = {"bin_gaussian"}


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
    nmse_h_binned_vs_gt_mc = _nmse_h_binned_vs_gt_mc(h_sweep_arr, h_gt_sqrt)
    nmse_decode_vs_ref_shared = _nmse_decode_vs_ref_shared(clf_sweep_arr, clf_ref)

    z_path = os.path.join(args.output_dir, "h_decoding_twofig_results.npz")
    with np.load(z_path, allow_pickle=True) as z_meta:
        loss_root_str = _npz_str_field(z_meta, "training_losses_root")
    loss_root = (
        loss_root_str
        if loss_root_str and os.path.isdir(loss_root_str)
        else os.path.join(args.output_dir, "training_losses")
    )

    sweep_svg = _render_method_sweep_panel(
        row_labels=row_labels,
        h_sweep=h_sweep_arr,
        clf_sweep_shared=clf_sweep_arr,
        n_list=ns,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_twofig_sweep.svg"),
        n_bins=n_bins,
        theta_centers=centers,
    )
    gt_svg = _render_gt_panel(
        h_gt_sqrt=h_gt_sqrt,
        clf_ref=clf_ref,
        n_ref=int(args.n_ref),
        n_bins=n_bins,
        theta_centers=centers,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_twofig_gt.svg"),
    )
    corr_svg = _render_corr_vs_n_panel(
        row_labels=row_labels,
        n_list=ns,
        corr_h=corr_h_binned_vs_gt_mc,
        corr_decode_shared=corr_decode_vs_ref_shared,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_twofig_corr_vs_n.svg"),
    )
    nmse_svg = _render_nmse_vs_n_panel(
        row_labels=row_labels,
        n_list=ns,
        nmse_h=nmse_h_binned_vs_gt_mc,
        nmse_decode_shared=nmse_decode_vs_ref_shared,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_twofig_nmse_vs_n.svg"),
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
        gt_svg=os.path.abspath(gt_svg),
        corr_svg=os.path.abspath(corr_svg),
        nmse_svg=os.path.abspath(nmse_svg),
        loss_panel_svg=os.path.abspath(loss_panel_svg),
        training_losses_root=os.path.abspath(loss_root),
        h_sweep_shape=tuple(int(x) for x in h_sweep_arr.shape),
        decode_sweep_shape=tuple(int(x) for x in clf_sweep_arr.shape),
        corr_h_shape=tuple(int(x) for x in corr_h_binned_vs_gt_mc.shape),
        nmse_h_shape=tuple(int(x) for x in nmse_h_binned_vs_gt_mc.shape),
        corr_decode_shape=tuple(int(x) for x in corr_decode_vs_ref_shared.shape),
        nmse_decode_shape=tuple(int(x) for x in nmse_decode_vs_ref_shared.shape),
        wall_seconds_shape=wall_shape,
        visualization_only=True,
    )

    print("[twofig] Saved (visualization-only):", flush=True)
    print(f"  - {os.path.abspath(sweep_svg)}", flush=True)
    print(f"  - {os.path.abspath(gt_svg)}", flush=True)
    print(f"  - {os.path.abspath(corr_svg)}", flush=True)
    print(f"  - {os.path.abspath(nmse_svg)}", flush=True)
    print(f"  - {os.path.abspath(loss_panel_svg)}", flush=True)
    print(f"  - {os.path.abspath(loss_root)}/", flush=True)
    print(f"  - {os.path.abspath(out_npz)} (unchanged)", flush=True)
    print(f"  - {os.path.abspath(summary_path)}", flush=True)


def _normalize_theta_field_method_local(method: str) -> str:
    m = str(method).strip().lower()
    if m in {"bin_gaussian", "binned_gaussian", "binned-gaussian", "bin-gaussian"}:
        return "bin_gaussian"
    if m == "nf":
        return "nf"
    lxf = conv._normalize_linear_x_flow_method(m)
    if lxf is not None:
        return str(lxf)
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
            "and supported scheduled linear_x_flow variants including linear_x_flow_diagonal_t "
            "(see study_h_decoding_convergence._normalize_linear_x_flow_method)."
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
            "linear_x_flow_low_rank_t,linear_x_flow_diagonal_t. "
            "For low-rank linear_x_flow rows use --lxf-low-rank-dim."
        ),
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
            if method not in _FLOW_BASED_METHODS:
                raise ValueError(
                    f"Invalid --theta-field-rows token {tok!r}; arch suffix is only allowed for "
                    "flow methods {theta_flow, theta_path_integral, x_flow}."
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
        if row.method in _NO_TRAIN_METHODS:
            continue
        args_r = deepcopy(args)
        setattr(args_r, "theta_field_method", row.method)
        if row.arch is not None:
            setattr(args_r, "flow_arch", row.arch)
        try:
            conv._validate_cli(args_r)
        except Exception as exc:
            raise ValueError(f"row={row.label}: {exc}") from exc


def _render_method_sweep_panel(
    *,
    row_labels: list[str],
    h_sweep: np.ndarray,
    clf_sweep_shared: np.ndarray,
    n_list: list[int],
    out_svg_path: str,
    n_bins: int,
    theta_centers: np.ndarray,
) -> str:
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
    _theta_center_array_for_axis(theta_centers, n_bins)

    n_rows = n_methods + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.5 * n_rows), squeeze=False)

    tick_pos, tick_labs, axis_label = _theta_axis_tick_labels(theta_centers, n_bins)
    x_rot = 45 if len(tick_pos) > 6 else 0

    vmin_h, vmax_h = 0.0, 1.0
    vmin_c, vmax_c = conv._finite_min_max([np.asarray(clf_sweep_shared[c], dtype=np.float64) for c in range(n_cols)])
    if vmin_c >= vmax_c:
        vmax_c = vmin_c + 1e-12

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

    decode_row = n_methods
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

    fig.tight_layout()
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
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.2), squeeze=False)
    ax_h = axes[0, 0]
    ax_c = axes[0, 1]
    _draw_single_heatmap(
        ax_h,
        h_gt_sqrt,
        n_bins=n_bins,
        theta_centers=theta_centers,
        title="Approx GT H matrix",
        vmin=0.0,
        vmax=1.0,
    )
    ax_h.set_ylabel(r"$\theta$", fontsize=11)
    vmin_c, vmax_c = conv._finite_min_max([clf_ref])
    if vmin_c >= vmax_c:
        vmax_c = vmin_c + 1e-12
    _draw_single_heatmap(
        ax_c,
        clf_ref,
        n_bins=n_bins,
        theta_centers=theta_centers,
        title=f"Approx GT decoding (n_ref={int(n_ref)})",
        vmin=vmin_c,
        vmax=vmax_c,
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
    gt_svg: str,
    corr_svg: str,
    nmse_svg: str,
    loss_panel_svg: str,
    training_losses_root: str,
    h_sweep_shape: tuple[int, ...],
    decode_sweep_shape: tuple[int, ...],
    corr_h_shape: tuple[int, ...],
    nmse_h_shape: tuple[int, ...],
    corr_decode_shape: tuple[int, ...],
    nmse_decode_shape: tuple[int, ...],
    wall_seconds_shape: tuple[int, ...],
    visualization_only: bool = False,
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
        f.write(f"dataset_pool_size: {int(n_pool)}\n")
        f.write(f"dataset_meta_seed: {int(meta.get('seed', 0))}\n")
        f.write(f"perm_seed: {int(perm_seed)}\n")
        f.write(f"results_npz: {out_npz}\n")
        f.write(f"h_binned_sweep_shape: {h_sweep_shape}\n")
        f.write(f"decode_sweep_shape: {decode_sweep_shape}\n")
        f.write(f"corr_h_binned_vs_gt_mc_shape: {corr_h_shape}\n")
        f.write(f"nmse_h_binned_vs_gt_mc_shape: {nmse_h_shape}\n")
        f.write(f"corr_decode_vs_ref_shared_shape: {corr_decode_shape}\n")
        f.write(f"nmse_decode_vs_ref_shared_shape: {nmse_decode_shape}\n")
        f.write("decode_sweep_semantics: shared_across_methods\n")
        f.write(f"wall_seconds_shape: {wall_seconds_shape}\n")
        f.write(f"figure_sweep_svg: {sweep_svg}\n")
        f.write(f"figure_gt_svg: {gt_svg}\n")
        f.write(f"figure_corr_vs_n_svg: {corr_svg}\n")
        f.write(f"figure_nmse_vs_n_svg: {nmse_svg}\n")
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


def _render_nmse_vs_n_panel(
    *,
    row_labels: list[str],
    n_list: list[int],
    nmse_h: np.ndarray,
    nmse_decode_shared: np.ndarray,
    out_svg_path: str,
) -> str:
    nmse_arr = np.asarray(nmse_h, dtype=np.float64)
    nmse_decode_arr = np.asarray(nmse_decode_shared, dtype=np.float64).ravel()
    n_arr = np.asarray(n_list, dtype=np.float64).ravel()
    if nmse_arr.shape != (len(row_labels), len(n_list)):
        raise ValueError(
            f"nmse_h shape mismatch: expected {(len(row_labels), len(n_list))}, got {nmse_arr.shape}."
        )
    if nmse_decode_arr.shape != (len(n_list),):
        raise ValueError(
            f"nmse_decode_shared shape mismatch: expected {(len(n_list),)}, got {nmse_decode_arr.shape}."
        )

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
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
    finite_h = nmse_arr[np.isfinite(nmse_arr)]
    finite_d = nmse_decode_arr[np.isfinite(nmse_decode_arr)]
    finite = np.concatenate([finite_h.ravel(), finite_d.ravel()]) if finite_d.size else finite_h.ravel()
    if finite.size > 0 and np.all(finite > 0):
        ax.set_yscale("log")
    ax.set_xlabel("dataset size n", fontsize=10)
    ax.set_ylabel("normalized MSE (off-diagonal, vs GT)", fontsize=10)
    ax.set_title("NMSE vs n", fontsize=11)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    svg = _save_figure_svg(fig, out_svg_path)
    plt.close(fig)
    return svg


def _render_corr_vs_n_panel(
    *,
    row_labels: list[str],
    n_list: list[int],
    corr_h: np.ndarray,
    corr_decode_shared: np.ndarray,
    out_svg_path: str,
) -> str:
    corr_h_arr = np.asarray(corr_h, dtype=np.float64)
    corr_decode_arr = np.asarray(corr_decode_shared, dtype=np.float64).ravel()
    n_arr = np.asarray(n_list, dtype=np.float64).ravel()
    if corr_h_arr.shape != (len(row_labels), len(n_list)):
        raise ValueError(
            f"corr_h shape mismatch: expected {(len(row_labels), len(n_list))}, got {corr_h_arr.shape}."
        )
    if corr_decode_arr.shape != (len(n_list),):
        raise ValueError(f"corr_decode shape mismatch: expected {(len(n_list),)}, got {corr_decode_arr.shape}.")

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    for i, label in enumerate(row_labels):
        ax.plot(
            n_arr,
            corr_h_arr[i],
            marker="o",
            linewidth=1.8,
            markersize=4.0,
            label=f"{label} (H vs GT)",
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
    ax.set_xlabel("dataset size n", fontsize=10)
    ax.set_ylabel("correlation (off-diagonal Pearson r)", fontsize=10)
    ax.set_title("Correlation vs n", fontsize=11)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    svg = _save_figure_svg(fig, out_svg_path)
    plt.close(fig)
    return svg


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    p = build_parser()
    args = p.parse_args(argv)
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
    args.theta_field_method = "theta_flow" if validation_row.method in _NO_TRAIN_METHODS else validation_row.method
    if validation_row.arch is not None:
        args.flow_arch = validation_row.arch
    conv._validate_cli(args)
    _validate_cli_for_rows(args, row_specs)
    if _theta_binning_mode(args) == "theta2_grid" and bool(getattr(args, "theta_flow_fourier_state", False)):
        raise ValueError("--theta-binning-mode theta2_grid does not support --theta-flow-fourier-state.")
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
    if theta_binning_mode == "theta2_grid":
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
    if bool(getattr(args, "theta_flow_onehot_state", False)):
        theta_state_all = np.eye(n_bins, dtype=np.float64)[bin_idx_all]
        print(
            f"[twofig] theta_flow one-hot state enabled: theta -> one_hot(bin(theta), K={n_bins})",
            flush=True,
        )
    elif bool(getattr(args, "theta_flow_fourier_state", False)):
        theta_state_all, theta_fourier_ref_range, theta_fourier_period, theta_fourier_center = conv._build_theta_fourier_state(
            theta_scalar_all,
            theta_ref=theta_ref,
            k=int(args.theta_flow_fourier_k),
            period_mult=float(args.theta_flow_fourier_period_mult),
            include_linear=bool(args.theta_flow_fourier_include_linear),
        )
        print(
            "[twofig] theta_flow Fourier state enabled: "
            f"dim={theta_state_all.shape[1]} K={int(args.theta_flow_fourier_k)} "
            f"period={theta_fourier_period:.6g} "
            f"(mult={float(args.theta_flow_fourier_period_mult):.3g}, ref_range={theta_fourier_ref_range:.6g}, "
            f"center={theta_fourier_center:.6g}, include_linear={bool(args.theta_flow_fourier_include_linear)})",
            flush=True,
        )

    clf_rs = base_seed if int(args.clf_random_state) < 0 else int(args.clf_random_state)

    dataset_for_gt = build_dataset_from_meta(meta)
    gt_seed = base_seed if int(args.gt_hellinger_seed) < 0 else int(args.gt_hellinger_seed)
    if hasattr(dataset_for_gt, "rng"):
        dataset_for_gt.rng = np.random.default_rng(gt_seed)
    gt_n_mc = int(args.n_ref) // n_bins
    t_gt0 = time.time()
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
        f"[twofig] GT Hellinger (MC likelihood) theta_binning_mode={theta_binning_mode} "
        f"n_bins={n_bins} n_mc={gt_n_mc} "
        f"(n_bins*n_mc={n_bins * gt_n_mc} <= n_ref={int(args.n_ref)}) wall time: {time.time() - t_gt0:.1f}s",
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
    clf_ref = conv._pairwise_clf_from_bundle(
        args=args,
        meta=meta,
        subset=subset_ref,
        output_dir=ref_dir,
        n_bins=n_bins,
        clf_min_class_count=int(args.clf_min_class_count),
        clf_random_state=clf_rs,
    )

    h_sweep_by_method: list[np.ndarray] = []
    clf_sweep_shared: list[np.ndarray] = []
    wall_s = np.full((len(row_specs), len(ns)), np.nan, dtype=np.float64)

    sweep_root = os.path.join(args.output_dir, "sweep_runs")
    if bool(args.keep_intermediate):
        os.makedirs(sweep_root, exist_ok=True)
    loss_root = os.path.join(args.output_dir, "training_losses")
    os.makedirs(loss_root, exist_ok=True)

    decode_dir = os.path.join(args.output_dir, "decode_shared")
    os.makedirs(decode_dir, exist_ok=True)
    for n in ns:
        subset_n = conv._subset_bundle(
            bundle,
            perm,
            int(n),
            meta,
            bin_idx_all=bin_idx_all,
            theta_state_all=theta_state_all,
        )
        clf_n = conv._pairwise_clf_from_bundle(
            args=args,
            meta=meta,
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
            args_method.theta_field_method = row.method
            if row.arch is not None:
                args_method.flow_arch = row.arch
            try:
                subset_n = conv._subset_bundle(
                    bundle,
                    perm,
                    int(n),
                    meta,
                    bin_idx_all=bin_idx_all,
                    theta_state_all=theta_state_all,
                )
                if row.method == "bin_gaussian":
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
                    meta=meta,
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
    nmse_h_binned_vs_gt_mc = _nmse_h_binned_vs_gt_mc(h_sweep_arr, h_gt_sqrt)
    nmse_decode_vs_ref_shared = _nmse_decode_vs_ref_shared(clf_sweep_arr, clf_ref)

    sweep_svg = _render_method_sweep_panel(
        row_labels=row_labels,
        h_sweep=h_sweep_arr,
        clf_sweep_shared=clf_sweep_arr,
        n_list=ns,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_twofig_sweep.svg"),
        n_bins=n_bins,
        theta_centers=centers,
    )

    gt_svg = _render_gt_panel(
        h_gt_sqrt=h_gt_sqrt,
        clf_ref=clf_ref,
        n_ref=int(args.n_ref),
        n_bins=n_bins,
        theta_centers=centers,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_twofig_gt.svg"),
    )
    corr_svg = _render_corr_vs_n_panel(
        row_labels=row_labels,
        n_list=ns,
        corr_h=corr_h_binned_vs_gt_mc,
        corr_decode_shared=corr_decode_vs_ref_shared,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_twofig_corr_vs_n.svg"),
    )
    nmse_svg = _render_nmse_vs_n_panel(
        row_labels=row_labels,
        n_list=ns,
        nmse_h=nmse_h_binned_vs_gt_mc,
        nmse_decode_shared=nmse_decode_vs_ref_shared,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_twofig_nmse_vs_n.svg"),
    )
    loss_panel_svg = _render_row_n_training_losses_panel(
        row_labels=row_labels,
        n_list=ns,
        loss_root=loss_root,
        out_svg_path=os.path.join(args.output_dir, "h_decoding_twofig_training_losses_panel.svg"),
    )

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
        theta_binning_mode=np.asarray([theta_binning_mode], dtype=object),
        theta_grid_shape=np.asarray(theta_grid_shape, dtype=np.int64),
        theta_bin_edges=np.asarray(edges, dtype=np.float64),
        theta_bin_centers=np.asarray(centers, dtype=np.float64),
        theta_grid_edges_0=np.asarray(theta_grid_edges0, dtype=np.float64),
        theta_grid_edges_1=np.asarray(theta_grid_edges1, dtype=np.float64),
        theta_grid_centers=np.asarray(centers if theta_binning_mode == "theta2_grid" else [], dtype=np.float64),
        gt_hellinger_n_mc=np.int64(gt_n_mc),
        gt_hellinger_seed=np.int64(gt_seed),
        gt_hellinger_symmetrize=np.int32(1 if bool(args.gt_hellinger_symmetrize) else 0),
        h_gt_sqrt=np.asarray(h_gt_sqrt, dtype=np.float64),
        decode_ref=np.asarray(clf_ref, dtype=np.float64),
        h_binned_sweep=np.asarray(h_sweep_arr, dtype=np.float64),
        decode_sweep=np.asarray(clf_sweep_arr, dtype=np.float64),
        corr_h_binned_vs_gt_mc=np.asarray(corr_h_binned_vs_gt_mc, dtype=np.float64),
        nmse_h_binned_vs_gt_mc=np.asarray(nmse_h_binned_vs_gt_mc, dtype=np.float64),
        corr_decode_vs_ref_shared=np.asarray(corr_decode_vs_ref_shared, dtype=np.float64),
        nmse_decode_vs_ref_shared=np.asarray(nmse_decode_vs_ref_shared, dtype=np.float64),
        column_n=np.asarray(ns, dtype=np.int64),
        corr_curve_svg=np.asarray(os.path.abspath(corr_svg), dtype=np.str_),
        nmse_curve_svg=np.asarray(os.path.abspath(nmse_svg), dtype=np.str_),
        training_losses_root=np.asarray(os.path.abspath(loss_root), dtype=np.str_),
        training_losses_panel_svg=np.asarray(os.path.abspath(loss_panel_svg), dtype=np.str_),
        dataset_npz=np.asarray(os.path.abspath(str(args.dataset_npz)), dtype=np.str_),
        dataset_family=np.asarray(str(meta.get("dataset_family", "")), dtype=np.str_),
        dataset_pool_size=np.int64(n_pool),
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
        gt_svg=os.path.abspath(gt_svg),
        corr_svg=os.path.abspath(corr_svg),
        nmse_svg=os.path.abspath(nmse_svg),
        loss_panel_svg=os.path.abspath(loss_panel_svg),
        training_losses_root=os.path.abspath(loss_root),
        h_sweep_shape=tuple(int(x) for x in h_sweep_arr.shape),
        decode_sweep_shape=tuple(int(x) for x in clf_sweep_arr.shape),
        corr_h_shape=tuple(int(x) for x in corr_h_binned_vs_gt_mc.shape),
        nmse_h_shape=tuple(int(x) for x in nmse_h_binned_vs_gt_mc.shape),
        corr_decode_shape=tuple(int(x) for x in corr_decode_vs_ref_shared.shape),
        nmse_decode_shape=tuple(int(x) for x in nmse_decode_vs_ref_shared.shape),
        wall_seconds_shape=tuple(int(x) for x in wall_s.shape),
    )

    print("[twofig] Saved:", flush=True)
    print(f"  - {os.path.abspath(sweep_svg)}", flush=True)
    print(f"  - {os.path.abspath(gt_svg)}", flush=True)
    print(f"  - {os.path.abspath(corr_svg)}", flush=True)
    print(f"  - {os.path.abspath(nmse_svg)}", flush=True)
    print(f"  - {os.path.abspath(loss_panel_svg)}", flush=True)
    print(f"  - {os.path.abspath(loss_root)}/", flush=True)
    print(f"  - {os.path.abspath(out_npz)}", flush=True)
    print(f"  - {os.path.abspath(summary_path)}", flush=True)


if __name__ == "__main__":
    main()
