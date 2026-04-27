#!/usr/bin/env python3
"""Two-figure H/decoding convergence study.

Reuses the full-compute pipeline from ``bin/study_h_decoding_convergence.py``
but emits only two matrix-figure artifacts:

1) ``h_decoding_twofig_sweep.svg``: columns over ``--n-list`` only
   (one row per method for estimated sqrt(H)-like binned matrices, plus one
   shared bottom row for decoding).
2) ``h_decoding_twofig_gt.svg``: left = approximate GT sqrt(H^2) matrix
   (MC likelihood), right = decoding matrix from the ``n_ref`` subset.
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
from typing import Any

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
from fisher.hellinger_gt import bin_centers_from_edges, estimate_hellinger_sq_one_sided_mc
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_meta, normalize_flow_arch, normalize_theta_field_method

# Valid row choices for --theta-field-rows:
# - theta_flow:mlp
# - theta_flow:film
# - theta_flow:film_fourier
# - theta_flow_gaussian_scaffold:mlp
# - theta_flow_gaussian_scaffold:film
# - theta_flow_gaussian_scaffold:film_fourier
# - theta_path_integral:mlp
# - theta_path_integral:film
# - theta_path_integral:film_fourier
# - x_flow:mlp
# - x_flow:film
# - x_flow:film_fourier
# - ctsm_v
# - nf
_FLOW_BASED_METHODS = {"theta_flow", "theta_flow_gaussian_scaffold", "theta_path_integral", "x_flow"}


def _normalize_theta_field_method_local(method: str) -> str:
    m = str(method).strip().lower()
    if m == "nf":
        return "nf"
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
        "and save two figures only: sweep matrices over n-list and GT/reference matrices."
    )
    p.set_defaults(output_dir=str(Path(DATA_DIR) / "h_decoding_twofig"))
    p.add_argument(
        "--theta-field-methods",
        type=str,
        default="",
        help=(
            "Comma-separated theta-field methods to sweep in one run. "
            "Overrides --theta-field-method when non-empty. "
            "Supported values: theta_flow, theta_flow_gaussian_scaffold, theta_path_integral, x_flow, ctsm_v, nf."
        ),
    )
    p.add_argument(
        "--theta-field-rows",
        type=str,
        default="",
        help=(
            "Comma-separated theta-field row specs, highest precedence over --theta-field-methods and "
            "--theta-field-method. Tokens are method or method:arch, e.g. "
            "theta_flow:mlp,theta_flow:film,x_flow:film_fourier,ctsm_v."
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
                    "flow methods {theta_flow, theta_flow_gaussian_scaffold, theta_path_integral, x_flow}."
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
    tc = np.asarray(theta_centers, dtype=np.float64).ravel()
    if int(tc.size) != int(n_bins):
        raise ValueError(f"theta_centers length {tc.size} must match n_bins={n_bins}.")

    n_rows = n_methods + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.5 * n_rows), squeeze=False)

    tick_labs_full = conv._format_theta_tick_labels(tc)
    tick_idx = conv._matrix_panel_tick_indices(n_bins, max_ticks=5)
    tick_pos = tick_idx.tolist()
    tick_labs = [tick_labs_full[int(i)] for i in tick_idx]
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
                ax_h.set_xlabel(r"$\theta$", fontsize=11)
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
        ax_c.set_xlabel(r"$\theta$", fontsize=11)
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
    tick_labs_full = conv._format_theta_tick_labels(np.asarray(theta_centers, dtype=np.float64).ravel())
    tick_idx = conv._matrix_panel_tick_indices(int(n_bins), max_ticks=5)
    tick_pos = tick_idx.tolist()
    tick_labs = [tick_labs_full[int(i)] for i in tick_idx]
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
    ax.set_xlabel(r"$\theta$", fontsize=11)
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
    loss_panel_svg: str,
    training_losses_root: str,
    h_sweep_shape: tuple[int, ...],
    decode_sweep_shape: tuple[int, ...],
    corr_h_shape: tuple[int, ...],
    corr_decode_shape: tuple[int, ...],
    wall_seconds_shape: tuple[int, ...],
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("study_h_decoding_twofig\n")
        f.write(f"dataset_npz: {args.dataset_npz}\n")
        f.write(f"dataset_family: {meta.get('dataset_family')}\n")
        f.write(f"output_dir: {args.output_dir}\n")
        f.write(f"n_ref: {int(args.n_ref)}\n")
        f.write(f"theta_field_rows: {','.join(getattr(args, 'theta_field_rows_resolved', []))}\n")
        f.write(f"theta_field_row_methods: {','.join(getattr(args, 'theta_field_row_methods_resolved', []))}\n")
        f.write(f"theta_field_row_arches: {','.join(getattr(args, 'theta_field_row_arches_resolved', []))}\n")
        f.write(f"n_list: {args.n_list}\n")
        f.write(f"num_theta_bins: {int(args.num_theta_bins)}\n")
        f.write(f"dataset_pool_size: {int(n_pool)}\n")
        f.write(f"dataset_meta_seed: {int(meta.get('seed', 0))}\n")
        f.write(f"perm_seed: {int(perm_seed)}\n")
        f.write(f"results_npz: {out_npz}\n")
        f.write(f"h_binned_sweep_shape: {h_sweep_shape}\n")
        f.write(f"decode_sweep_shape: {decode_sweep_shape}\n")
        f.write(f"corr_h_binned_vs_gt_mc_shape: {corr_h_shape}\n")
        f.write(f"corr_decode_vs_ref_shared_shape: {corr_decode_shape}\n")
        f.write("decode_sweep_semantics: shared_across_methods\n")
        f.write(f"wall_seconds_shape: {wall_seconds_shape}\n")
        f.write(f"figure_sweep_svg: {sweep_svg}\n")
        f.write(f"figure_gt_svg: {gt_svg}\n")
        f.write(f"figure_corr_vs_n_svg: {corr_svg}\n")
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

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.2))
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

    if bool(getattr(args, "visualization_only", False)):
        raise ValueError("study_h_decoding_twofig.py does not support --visualization-only; run full compute mode.")

    row_specs = _parse_theta_field_rows(args)
    row_labels = [r.label for r in row_specs]
    row_methods = [r.method for r in row_specs]
    row_arches = [("" if r.arch is None else r.arch) for r in row_specs]
    setattr(args, "theta_field_rows_resolved", row_labels)
    setattr(args, "theta_field_row_methods_resolved", row_methods)
    setattr(args, "theta_field_row_arches_resolved", row_arches)
    args.theta_field_method = row_specs[0].method
    if row_specs[0].arch is not None:
        args.flow_arch = row_specs[0].arch
    conv._validate_cli(args)
    _validate_cli_for_rows(args, row_specs)
    ns = conv._parse_n_list(args.n_list)

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

    theta_raw_all = np.asarray(bundle.theta_all, dtype=np.float64)
    if theta_raw_all.ndim == 2 and int(theta_raw_all.shape[1]) != 1:
        raise ValueError(
            "Convergence binning requires scalar theta in dataset bundle; "
            f"got theta_all shape={theta_raw_all.shape}."
        )
    theta_scalar_all = theta_raw_all.reshape(-1)
    theta_ref = np.asarray(theta_scalar_all[perm[: int(args.n_ref)]], dtype=np.float64).reshape(-1)
    edges, _, _ = conv.vhb.theta_bin_edges(theta_ref, n_bins)
    centers = bin_centers_from_edges(edges)
    bin_idx_all = conv.vhb.theta_to_bin_index(theta_scalar_all, edges, n_bins)

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
    h_gt_mc = estimate_hellinger_sq_one_sided_mc(
        dataset_for_gt,
        centers,
        n_mc=gt_n_mc,
        symmetrize=bool(args.gt_hellinger_symmetrize),
    )
    h_gt_sqrt = conv._sqrt_h_like(h_gt_mc)
    print(
        f"[twofig] GT Hellinger (MC likelihood) n_bins={n_bins} n_mc={gt_n_mc} "
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
                if bool(args.keep_intermediate):
                    row_dir = row.label.replace(":", "__")
                    run_dir = os.path.join(sweep_root, row_dir, f"n_{n:06d}")
                    os.makedirs(run_dir, exist_ok=True)
                else:
                    row_prefix = row.label.replace(":", "__")
                    tmp_ctx = tempfile.TemporaryDirectory(prefix=f"h_twofig_{row_prefix}_n{n}_", dir=args.output_dir)
                    run_dir = tmp_ctx.name

                subset_n = conv._subset_bundle(
                    bundle,
                    perm,
                    int(n),
                    meta,
                    bin_idx_all=bin_idx_all,
                    theta_state_all=theta_state_all,
                )
                loaded_n, _, _ = conv._estimate_one(
                    args=args_method,
                    meta=meta,
                    bundle=subset_n.bundle,
                    output_dir=run_dir,
                    n_bins=n_bins,
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
    corr_h_binned_vs_gt_mc = np.full((len(row_specs), len(ns)), np.nan, dtype=np.float64)
    for i in range(len(row_specs)):
        for j in range(len(ns)):
            corr_h_binned_vs_gt_mc[i, j] = conv.vhb.matrix_corr_offdiag_pearson(
                np.asarray(h_sweep_arr[i, j], dtype=np.float64),
                np.asarray(h_gt_sqrt, dtype=np.float64),
            )
    corr_decode_vs_ref_shared = np.full((len(ns),), np.nan, dtype=np.float64)
    for j in range(len(ns)):
        corr_decode_vs_ref_shared[j] = conv.vhb.matrix_corr_offdiag_pearson(
            np.asarray(clf_sweep_arr[j], dtype=np.float64),
            np.asarray(clf_ref, dtype=np.float64),
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
        theta_bin_edges=np.asarray(edges, dtype=np.float64),
        theta_bin_centers=np.asarray(centers, dtype=np.float64),
        gt_hellinger_n_mc=np.int64(gt_n_mc),
        gt_hellinger_seed=np.int64(gt_seed),
        gt_hellinger_symmetrize=np.int32(1 if bool(args.gt_hellinger_symmetrize) else 0),
        h_gt_sqrt=np.asarray(h_gt_sqrt, dtype=np.float64),
        decode_ref=np.asarray(clf_ref, dtype=np.float64),
        h_binned_sweep=np.asarray(h_sweep_arr, dtype=np.float64),
        decode_sweep=np.asarray(clf_sweep_arr, dtype=np.float64),
        corr_h_binned_vs_gt_mc=np.asarray(corr_h_binned_vs_gt_mc, dtype=np.float64),
        corr_decode_vs_ref_shared=np.asarray(corr_decode_vs_ref_shared, dtype=np.float64),
        column_n=np.asarray(ns, dtype=np.int64),
        corr_curve_svg=np.asarray(os.path.abspath(corr_svg), dtype=np.str_),
        training_losses_root=np.asarray(os.path.abspath(loss_root), dtype=np.str_),
        training_losses_panel_svg=np.asarray(os.path.abspath(loss_panel_svg), dtype=np.str_),
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
        loss_panel_svg=os.path.abspath(loss_panel_svg),
        training_losses_root=os.path.abspath(loss_root),
        h_sweep_shape=tuple(int(x) for x in h_sweep_arr.shape),
        decode_sweep_shape=tuple(int(x) for x in clf_sweep_arr.shape),
        corr_h_shape=tuple(int(x) for x in corr_h_binned_vs_gt_mc.shape),
        corr_decode_shape=tuple(int(x) for x in corr_decode_vs_ref_shared.shape),
        wall_seconds_shape=tuple(int(x) for x in wall_s.shape),
    )

    print("[twofig] Saved:", flush=True)
    print(f"  - {os.path.abspath(sweep_svg)}", flush=True)
    print(f"  - {os.path.abspath(gt_svg)}", flush=True)
    print(f"  - {os.path.abspath(corr_svg)}", flush=True)
    print(f"  - {os.path.abspath(loss_panel_svg)}", flush=True)
    print(f"  - {os.path.abspath(loss_root)}/", flush=True)
    print(f"  - {os.path.abspath(out_npz)}", flush=True)
    print(f"  - {os.path.abspath(summary_path)}", flush=True)


if __name__ == "__main__":
    main()
