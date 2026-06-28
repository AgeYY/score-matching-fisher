#!/usr/bin/env python3
"""LLR-only categorical random-MoG study.

This script isolates the estimated-vs-ground-truth log-likelihood-ratio
diagnostic from ``fisher.h_decoding_categorical_twofig``. It intentionally does
not compute Hellinger distances, decoding accuracy, or two-figure sweep panels.
"""

from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import global_setting  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fisher.dataset_visualization import pca_project
from fisher.h_decoding_categorical_twofig import (
    _abs_without_resolving_symlinks,
    _default_pr_output_npz,
    _ensure_dataset,
    _ensure_pr_projected_npz,
    _llr_comparison_metrics,
    _render_row_n_training_losses_panel,
    _sanitize_row_label,
    _save_method_training_loss_npz,
    _selected_eval_split,
    _train_one_method,
    build_parser as build_twofig_parser,
    compute_true_conditional_loglik_matrix,
    parse_methods,
)
from fisher.h_decoding_convergence_methods import prepare_categorical_binning_for_convergence
from fisher.h_matrix import HMatrixEstimator
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import require_device
from fisher.svg_utils import (
    concatenate_pngs_horizontally,
    concatenate_svgs_horizontally,
    concatenate_svgs_horizontally_to_png,
    svg_viewbox_size,
)
from fisher import h_decoding_convergence as conv


_DEFAULT_LLR_METHODS = (
    "x_flow",
    "binary_classifier",
    "linear_x_flow_t",
    "xflow_sir_lrank",
    "contrastive_soft_categorical",
    "ctsm_v",
    "ctsm_v_binary",
    "lda_ctsm_v",
    "pls_ctsm_v",
    "pca_ctsm_v",
)
_NON_LLR_METHODS = {"bin_gaussian", "theta_flow_cate"}

# Canonical mog-2pr5 bundle (K=2 native 2D x, PR-embedded 5D x); see .cursor/skills/mog-2pr5/SKILL.md.
_MOG2PR5_DIR = _repo_root / "data" / "mog_2pr5_default"
_MOG2PR5_NATIVE_NPZ = _MOG2PR5_DIR / "random_mog_categorical.npz"
_MOG2PR5_PR5_NPZ = _MOG2PR5_DIR / "random_mog_categorical_pr5.npz"
_MOG2PR5_OUTPUT_DIR = _MOG2PR5_DIR / "h_decoding_categorical_llr_scatter"


def build_parser():
    p = build_twofig_parser()
    p.description = __doc__
    p.set_defaults(
        methods=",".join(_DEFAULT_LLR_METHODS),
        n_list="",
        num_categories=2,
        dataset_npz=str(_MOG2PR5_NATIVE_NPZ),
        output_dir=str(_MOG2PR5_OUTPUT_DIR),
        pr_project=True,
        pr_dim=5,
        pr_output_npz=str(_MOG2PR5_PR5_NPZ),
        no_scatter_diagnostics=False,
        visualization_only=False,
        eval_split="all",
    )
    for action in p._actions:
        if action.dest == "methods":
            action.help = (
                "Comma-separated LLR-returning methods. Supported here: "
                + ", ".join(_DEFAULT_LLR_METHODS)
                + "."
            )
        elif action.dest == "eval_split":
            action.help = (
                "Rows used for LLR metrics and scatter: 'all' (default) uses every sampled row; "
                "'validation' uses only the held-out validation slice of the n-eval subset "
                "(methods still train on the nested train split)."
            )
        elif action.dest == "visualization_only":
            action.help = "Inherited compatibility flag; not used by this LLR-only script."
        elif action.dest == "no_scatter_diagnostics":
            action.help = "Inherited compatibility flag; not used because this script always writes the LLR scatter."
    p.add_argument(
        "--n-eval",
        type=int,
        default=600,
        help="Number of sampled rows used for fitting/evaluating the LLR scatter comparison.",
    )
    return p


def _metric_table(metrics_by_method: dict[str, dict[str, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metric_names = sorted({k for metrics in metrics_by_method.values() for k in metrics})
    method_names = list(metrics_by_method)
    table = np.full((len(method_names), len(metric_names)), np.nan, dtype=np.float64)
    for i, name in enumerate(method_names):
        metrics = metrics_by_method[name]
        for j, metric_name in enumerate(metric_names):
            table[i, j] = float(metrics.get(metric_name, np.nan))
    return (
        np.asarray(method_names, dtype=object),
        np.asarray(metric_names, dtype=object),
        table,
    )


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(np.count_nonzero(mask)) < 2:
        return float("nan")
    xm = x_arr[mask] - float(np.mean(x_arr[mask]))
    ym = y_arr[mask] - float(np.mean(y_arr[mask]))
    denom = float(np.sqrt(np.sum(xm**2) * np.sum(ym**2)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(xm * ym) / denom)


def _binary_delta_l_to_raw_llr_1_minus_0(delta_l: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Reconstruct row-wise raw binary LLR r(x)=log p1(x)-log p0(x)."""
    delta = np.asarray(delta_l, dtype=np.float64)
    labels = np.asarray(bins, dtype=np.int64).reshape(-1)
    if delta.ndim != 2 or delta.shape[0] != delta.shape[1]:
        raise ValueError(f"Binary raw LLR reconstruction expects a square matrix; got shape {delta.shape}.")
    n = int(delta.shape[0])
    if int(labels.shape[0]) != n:
        raise ValueError(f"Binary raw LLR reconstruction label length {labels.shape[0]} does not match n={n}.")
    unique = set(np.unique(labels).tolist())
    if not unique.issubset({0, 1}) or unique != {0, 1}:
        raise ValueError("Raw binary LLR reconstruction requires both binary labels 0 and 1.")

    out = np.full((n,), np.nan, dtype=np.float64)
    for i, yi in enumerate(labels):
        if int(yi) == 0:
            vals = delta[i, labels == 1]
        else:
            vals = -delta[i, labels == 0]
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            out[i] = float(np.mean(vals))
    return out


def _raw_llr_metrics(est: np.ndarray, true: np.ndarray) -> dict[str, float]:
    est_arr = np.asarray(est, dtype=np.float64).reshape(-1)
    true_arr = np.asarray(true, dtype=np.float64).reshape(-1)
    if est_arr.shape != true_arr.shape:
        raise ValueError(f"Raw LLR metric shapes must match; got {est_arr.shape} and {true_arr.shape}.")
    mask = np.isfinite(est_arr) & np.isfinite(true_arr)
    if not np.any(mask):
        return {
            "llr_raw_rmse": float("nan"),
            "llr_raw_mae": float("nan"),
            "llr_raw_bias": float("nan"),
            "llr_raw_pearson_r": float("nan"),
        }
    diff = est_arr[mask] - true_arr[mask]
    return {
        "llr_raw_rmse": float(np.sqrt(np.mean(diff**2))),
        "llr_raw_mae": float(np.mean(np.abs(diff))),
        "llr_raw_bias": float(np.mean(diff)),
        "llr_raw_pearson_r": _pearson_r(est_arr[mask], true_arr[mask]),
    }


def _save_raw_binary_llr_est_vs_true_figure(
    method_llrs: dict[str, np.ndarray],
    true_llr: np.ndarray,
    bins_eval: np.ndarray,
    *,
    out_base: Path,
    metrics_by_method: dict[str, dict[str, float]],
) -> None:
    """Save raw binary LLR scatter colored by row class."""
    true_arr = np.asarray(true_llr, dtype=np.float64).reshape(-1)
    labels = np.asarray(bins_eval, dtype=np.int64).reshape(-1)
    if true_arr.shape[0] != labels.shape[0]:
        raise ValueError(f"Raw LLR label length {labels.shape[0]} does not match values {true_arr.shape[0]}.")

    fig, ax = plt.subplots(figsize=(6.4, 5.8), layout="constrained")
    markers = {0: "o", 1: "^"}
    cmap = plt.get_cmap("tab10")
    ys: list[np.ndarray] = []
    if true_arr.size == 0:
        ax.text(0.5, 0.5, "No raw binary LLR rows", ha="center", va="center", transform=ax.transAxes)
    else:
        for method_idx, (method_name, est_llr) in enumerate(method_llrs.items()):
            y = np.asarray(est_llr, dtype=np.float64).reshape(-1)
            if y.shape != true_arr.shape:
                raise ValueError(f"Method {method_name!r} raw LLR shape {y.shape} does not match {true_arr.shape}.")
            ys.append(y)
            color = cmap(method_idx % 10)
            m = metrics_by_method.get(method_name, {})
            label_prefix = (
                f"{method_name} "
                f"(RMSE={m.get('llr_raw_rmse', float('nan')):.3g}, "
                f"r={m.get('llr_raw_pearson_r', float('nan')):.3g})"
            )
            for cls in (0, 1):
                mask = (labels == cls) & np.isfinite(true_arr) & np.isfinite(y)
                if not np.any(mask):
                    continue
                ax.scatter(
                    true_arr[mask],
                    y[mask],
                    s=18,
                    alpha=0.45,
                    linewidths=0,
                    marker=markers[cls],
                    color=color,
                    label=f"{label_prefix}, class {cls}",
                )
        vals = np.concatenate([true_arr] + ys) if ys else true_arr
        vals = vals[np.isfinite(vals)]
        lo = float(np.min(vals)) if vals.size else -1.0
        hi = float(np.max(vals)) if vals.size else 1.0
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.0, alpha=0.7)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best", fontsize=7, framealpha=0.9)
    ax.set_xlabel(r"true log $p_1(x)$ - log $p_0(x)$")
    ax.set_ylabel(r"estimated log $p_1(x)$ - log $p_0(x)$")
    ax.set_title("Raw binary LLR: estimated vs ground truth")
    ax.grid(True, alpha=0.25)
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    path: Path,
    *,
    args: Any,
    methods: list[str],
    eval_split: str,
    n_eval_matrix: int,
    dataset_npz: Path,
    work_dataset_npz: Path,
    results_npz: Path,
    llr_svg: Path,
    llr_png: Path,
    training_losses_root: Path,
    loss_panel_svg: Path,
    combined_svg: Path,
    combined_png: Path | None,
    dataset_pca_svg: Path,
    dataset_pca_png: Path,
    dataset_pca_llr_mse_svg: Path | None,
    dataset_pca_llr_mse_png: Path | None,
    combined_with_dataset_pca_svg: Path,
    combined_with_dataset_pca_png: Path | None,
    wall_seconds: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("study_h_decoding_categorical_llr_scatter\n")
        f.write(f"output_dir: {Path(args.output_dir).resolve()}\n")
        f.write(f"dataset_npz: {dataset_npz}\n")
        f.write(f"work_dataset_npz: {work_dataset_npz}\n")
        f.write(f"num_categories: {int(args.num_categories)}\n")
        f.write(f"n_eval: {int(args.n_eval)}\n")
        f.write(f"eval_split: {eval_split}\n")
        f.write(f"n_eval_matrix: {int(n_eval_matrix)}\n")
        f.write(f"methods: {','.join(methods)}\n")
        f.write(f"pr_project: {bool(args.pr_project)}\n")
        f.write(f"device: {args.device}\n")
        f.write(f"results_npz: {results_npz}\n")
        f.write(f"llr_est_vs_true_all.svg: {llr_svg}\n")
        f.write(f"llr_est_vs_true_all.png: {llr_png}\n")
        f.write(f"training_losses_root: {training_losses_root}\n")
        f.write(f"h_decoding_categorical_llr_scatter_training_losses_panel.svg: {loss_panel_svg}\n")
        f.write(f"llr_est_vs_true_all_with_losses.svg: {combined_svg}\n")
        if combined_png is not None:
            f.write(f"llr_est_vs_true_all_with_losses.png: {combined_png}\n")
        f.write(f"dataset_pca_projection.svg: {dataset_pca_svg}\n")
        f.write(f"dataset_pca_projection.png: {dataset_pca_png}\n")
        if dataset_pca_llr_mse_svg is not None:
            f.write(f"dataset_pca_llr_mse_ctsm_v_binary.svg: {dataset_pca_llr_mse_svg}\n")
        if dataset_pca_llr_mse_png is not None:
            f.write(f"dataset_pca_llr_mse_ctsm_v_binary.png: {dataset_pca_llr_mse_png}\n")
        f.write(f"llr_est_vs_true_all_with_losses_and_dataset_pca.svg: {combined_with_dataset_pca_svg}\n")
        if combined_with_dataset_pca_png is not None:
            f.write(f"llr_est_vs_true_all_with_losses_and_dataset_pca.png: {combined_with_dataset_pca_png}\n")
        for name, wall_s in zip(methods, wall_seconds):
            f.write(f"wall_seconds.{name}: {float(wall_s):.6g}\n")


def _save_dataset_pca_projection_figure(
    x_eval: np.ndarray,
    bins_eval: np.ndarray,
    *,
    k_cat: int,
    out_base: Path,
) -> tuple[Path, Path]:
    """Save PCA projection of the evaluated working feature rows."""
    x = np.asarray(x_eval, dtype=np.float64)
    labels = np.asarray(bins_eval, dtype=np.int64).reshape(-1)
    if x.ndim != 2:
        raise ValueError(f"Dataset PCA expects 2D x_eval; got shape {x.shape}.")
    if int(x.shape[0]) != int(labels.shape[0]):
        raise ValueError(f"Dataset PCA label length {labels.shape[0]} does not match x rows {x.shape[0]}.")
    if int(x.shape[0]) < 2:
        raise ValueError("Dataset PCA requires at least two evaluated rows.")
    if int(x.shape[1]) < 2:
        raise ValueError("Dataset PCA requires x_dim >= 2.")

    proj, _, _ = pca_project(x, n_components=2)
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    svg_path = out_base.with_suffix(".svg").resolve()
    png_path = out_base.with_suffix(".png").resolve()

    fig, ax = plt.subplots(figsize=(5.4, 5.8), layout="constrained")
    cmap = plt.get_cmap("tab10")
    for c in range(int(k_cat)):
        mask = labels == c
        if not np.any(mask):
            continue
        ax.scatter(
            proj[mask, 0],
            proj[mask, 1],
            s=12,
            alpha=0.55,
            linewidths=0,
            color=cmap(c % 10),
            label=f"category {c}",
        )
    if not any(np.any(labels == c) for c in range(int(k_cat))):
        ax.text(0.5, 0.5, "No categorical labels", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Dataset PCA projection")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


def _per_sample_llr_mse(delta_est: np.ndarray, delta_true: np.ndarray) -> np.ndarray:
    """Aggregate pairwise LLR squared error to evaluated samples."""
    est = np.asarray(delta_est, dtype=np.float64)
    true = np.asarray(delta_true, dtype=np.float64)
    if est.shape != true.shape:
        raise ValueError(f"delta_est shape {est.shape} must match delta_true shape {true.shape}.")
    if est.ndim != 2 or est.shape[0] != est.shape[1]:
        raise ValueError(f"LLR matrices must be square 2D arrays; got shape {est.shape}.")
    n = int(est.shape[0])
    sq = (est - true) ** 2
    valid = np.isfinite(sq)
    valid[np.eye(n, dtype=bool)] = False
    sums = np.where(valid, sq, 0.0).sum(axis=1) + np.where(valid, sq, 0.0).sum(axis=0)
    counts = valid.sum(axis=1) + valid.sum(axis=0)
    out = np.full((n,), np.nan, dtype=np.float64)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def _save_dataset_pca_llr_mse_projection_figure(
    x_eval: np.ndarray,
    llr_mse: np.ndarray,
    *,
    method_label: str,
    out_base: Path,
) -> tuple[Path, Path]:
    """Save PCA projection of evaluated rows colored by per-sample LLR MSE."""
    x = np.asarray(x_eval, dtype=np.float64)
    values = np.asarray(llr_mse, dtype=np.float64).reshape(-1)
    if x.ndim != 2:
        raise ValueError(f"Dataset PCA LLR-MSE expects 2D x_eval; got shape {x.shape}.")
    if int(x.shape[0]) != int(values.shape[0]):
        raise ValueError(f"LLR-MSE length {values.shape[0]} does not match x rows {x.shape[0]}.")
    if int(x.shape[0]) < 2:
        raise ValueError("Dataset PCA LLR-MSE requires at least two evaluated rows.")
    if int(x.shape[1]) < 2:
        raise ValueError("Dataset PCA LLR-MSE requires x_dim >= 2.")

    proj, _, _ = pca_project(x, n_components=2)
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    svg_path = out_base.with_suffix(".svg").resolve()
    png_path = out_base.with_suffix(".png").resolve()

    fig, ax = plt.subplots(figsize=(5.8, 5.8), layout="constrained")
    finite = np.isfinite(values)
    if np.any(finite):
        sc = ax.scatter(
            proj[finite, 0],
            proj[finite, 1],
            c=values[finite],
            s=16,
            alpha=0.82,
            linewidths=0,
            cmap="viridis",
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean squared LLR error")
    if np.any(~finite):
        ax.scatter(
            proj[~finite, 0],
            proj[~finite, 1],
            s=16,
            alpha=0.45,
            linewidths=0,
            color="0.7",
            label="non-finite MSE",
        )
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Dataset PCA colored by LLR MSE ({method_label})")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


def _write_combined_llr_loss_outputs(
    *,
    llr_svg: Path,
    llr_png: Path,
    loss_panel_svg: Path,
    combined_svg: Path,
    combined_png: Path,
) -> tuple[Path, Path | None]:
    """Write combined LLR/loss SVG and best-effort PNG."""
    _, loss_height = svg_viewbox_size(loss_panel_svg)
    concatenate_svgs_horizontally(
        [llr_svg, loss_panel_svg],
        combined_svg,
        target_height=loss_height,
        valign="center",
    )
    combined_png_out: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="llr_loss_png_") as td:
            loss_png = Path(td) / "loss_panel.png"
            concatenate_svgs_horizontally_to_png([loss_panel_svg], loss_png, dpi=300)
            from PIL import Image

            with Image.open(loss_png) as im:
                loss_png_height = int(im.height)
            png = concatenate_pngs_horizontally(
                [llr_png, loss_png],
                combined_png,
                spacing=int(round(24.0 / 72.0 * 300.0)),
                target_height=loss_png_height,
                valign="center",
            )
        combined_png_out = Path(png).resolve()
    except Exception as exc:
        print(
            f"[llr-scatter] WARNING: combined PNG failed ({type(exc).__name__}: {exc}); "
            "combined SVG, loss SVG, and results NPZ are still saved.",
            flush=True,
        )
    return combined_svg.resolve(), combined_png_out


def _write_combined_llr_loss_dataset_pca_outputs(
    *,
    llr_svg: Path,
    llr_png: Path,
    loss_panel_svg: Path,
    dataset_pca_svg: Path,
    dataset_pca_png: Path,
    combined_svg: Path,
    combined_png: Path,
    dataset_pca_llr_mse_svg: Path | None = None,
    dataset_pca_llr_mse_png: Path | None = None,
) -> tuple[Path, Path | None]:
    """Write combined LLR/loss/dataset-PCA SVG and best-effort PNG."""
    sources = [dataset_pca_svg]
    png_sources = [dataset_pca_png]
    if dataset_pca_llr_mse_svg is not None and dataset_pca_llr_mse_png is not None:
        sources.append(dataset_pca_llr_mse_svg)
        png_sources.append(dataset_pca_llr_mse_png)
    sources.extend([loss_panel_svg, llr_svg])
    _, loss_height = svg_viewbox_size(loss_panel_svg)
    concatenate_svgs_horizontally(sources, combined_svg, target_height=loss_height, valign="center")
    combined_png_out: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="llr_loss_pca_png_") as td:
            loss_png = Path(td) / "loss_panel.png"
            concatenate_svgs_horizontally_to_png([loss_panel_svg], loss_png, dpi=300)
            from PIL import Image

            with Image.open(loss_png) as im:
                loss_png_height = int(im.height)
            png_sources.extend([loss_png, llr_png])
            png = concatenate_pngs_horizontally(
                png_sources,
                combined_png,
                spacing=int(round(24.0 / 72.0 * 300.0)),
                target_height=loss_png_height,
                valign="center",
            )
        combined_png_out = Path(png).resolve()
    except Exception as exc:
        print(
            f"[llr-scatter] WARNING: combined LLR/loss/dataset-PCA PNG failed "
            f"({type(exc).__name__}: {exc}); combined SVG, component SVGs, and results NPZ are still saved.",
            flush=True,
        )
    return combined_svg.resolve(), combined_png_out


def main(argv: list[str] | None = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)
    methods = parse_methods(str(args.methods))
    bad = [m for m in methods if m in _NON_LLR_METHODS]
    if bad:
        raise ValueError(
            "This LLR-only script supports methods that return a full delta_l matrix; "
            f"unsupported here: {bad}."
        )
    if int(args.num_categories) != 2:
        raise ValueError("Raw binary LLR plotting requires exactly two categories.")
    if int(args.n_eval) < 2:
        raise ValueError("--n-eval must be >= 2.")
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use a CUDA machine or pass a CUDA --device when available.")

    if args.dataset_npz is None:
        args.dataset_npz = _MOG2PR5_NATIVE_NPZ
    args.dataset_npz = _abs_without_resolving_symlinks(Path(args.dataset_npz))
    if args.output_dir is None:
        args.output_dir = _MOG2PR5_OUTPUT_DIR
    args.output_dir = Path(args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
        if int(args.pr_dim) < native_x_dim:
            raise ValueError(f"--pr-dim must be >= native x_dim={native_x_dim}; got {args.pr_dim}.")
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
    if int(args.n_eval) > n_pool:
        raise ValueError(f"--n-eval={args.n_eval} exceeds n_total={n_pool}.")

    k_cat = int(native_meta.get("num_categories", int(args.num_categories)))
    if k_cat != 2:
        raise ValueError("Raw binary LLR plotting requires exactly two categories.")
    _, _, _, _, _, bin_idx_all = prepare_categorical_binning_for_convergence(native_bundle.theta_all, k_cat)

    rng = np.random.default_rng(int(args.run_seed))
    perm = rng.permutation(n_pool)
    subset_w = conv._subset_bundle(work_bundle, perm, int(args.n_eval), meta_work, bin_idx_all=bin_idx_all)
    subset_n = conv._subset_bundle(native_bundle, perm, int(args.n_eval), meta_work, bin_idx_all=bin_idx_all)

    eval_split = _selected_eval_split(args)
    if eval_split == "validation" and float(meta_work.get("train_frac", 0.7)) >= 1.0:
        raise ValueError(
            "--eval-split validation requires train_frac < 1 in dataset meta; "
            f"got train_frac={meta_work.get('train_frac')!r}."
        )

    train_theta = np.asarray(subset_w.bundle.theta_train, dtype=np.float64)
    val_theta = np.asarray(subset_w.bundle.theta_validation, dtype=np.float64)
    theta_all = np.asarray(subset_w.bundle.theta_all, dtype=np.float64)
    if train_theta.ndim == 1:
        train_theta = train_theta.reshape(-1, 1)
    if val_theta.ndim == 1:
        val_theta = val_theta.reshape(-1, 1)
    if theta_all.ndim == 1:
        theta_all = theta_all.reshape(-1, 1)

    x_train = np.asarray(subset_w.bundle.x_train, dtype=np.float64)
    x_val = np.asarray(subset_w.bundle.x_validation, dtype=np.float64)
    x_all = np.asarray(subset_w.bundle.x_all, dtype=np.float64)
    x_native_all = np.asarray(subset_n.bundle.x_all, dtype=np.float64)
    bins_train = np.asarray(subset_w.bin_train, dtype=np.int64).reshape(-1)
    bins_val = np.asarray(subset_w.bin_validation, dtype=np.int64).reshape(-1)
    bins_all = np.asarray(subset_w.bin_all, dtype=np.int64).reshape(-1)

    n_train = int(train_theta.shape[0])
    if eval_split == "validation":
        n_val = int(val_theta.shape[0])
        if n_val < 1:
            raise ValueError(
                f"--eval-split validation: empty validation split for n_eval={int(args.n_eval)}."
            )
        theta_eval = val_theta
        x_eval = x_val
        x_native_eval = np.asarray(subset_n.bundle.x_validation, dtype=np.float64)
        bins_eval = bins_val
        source_indices = np.asarray(perm[n_train : int(args.n_eval)], dtype=np.int64)
    else:
        theta_eval = theta_all
        x_eval = x_all
        x_native_eval = x_native_all
        bins_eval = bins_all
        source_indices = np.asarray(perm[: int(args.n_eval)], dtype=np.int64)

    n_eval_matrix = int(theta_eval.shape[0])
    true_c = compute_true_conditional_loglik_matrix(x_native_eval, theta_eval, native_meta)
    true_delta_l = HMatrixEstimator.compute_delta_l(true_c)
    true_llr_1_minus_0 = _binary_delta_l_to_raw_llr_1_minus_0(true_delta_l, bins_eval)

    dev = require_device(str(args.device))
    torch.manual_seed(int(args.run_seed))
    np.random.seed(int(args.run_seed))

    method_deltas: dict[str, np.ndarray] = {}
    method_raw_llrs: dict[str, np.ndarray] = {}
    metrics_by_method: dict[str, dict[str, float]] = {}
    raw_metrics_by_method: dict[str, dict[str, float]] = {}
    wall_s = np.full((len(methods),), np.nan, dtype=np.float64)
    out_dir = Path(args.output_dir)
    loss_root = (out_dir / "training_losses").resolve()
    for i, method_name in enumerate(methods):
        t0 = time.time()
        result = _train_one_method(
            args,
            dev=dev,
            method_name=method_name,
            theta_train=train_theta,
            x_train=x_train,
            theta_val=val_theta,
            x_val=x_val,
            theta_all=theta_eval,
            x_all=x_eval,
            bins_train=bins_train,
            bins_val=bins_val,
            bins_all=bins_eval,
            k_cat=k_cat,
        )
        loss_npz = loss_root / _sanitize_row_label(method_name) / f"n_{int(args.n_eval):06d}.npz"
        _save_method_training_loss_npz(loss_npz, method_name=method_name, result=result)
        if "delta_l" not in result:
            raise RuntimeError(f"Method {method_name!r} did not return delta_l.")
        delta = np.asarray(result["delta_l"], dtype=np.float64)
        if delta.shape != true_delta_l.shape:
            raise ValueError(
                f"Method {method_name!r} returned delta_l shape {delta.shape}, "
                f"expected {true_delta_l.shape}."
            )
        method_deltas[method_name] = delta
        metrics_by_method[method_name] = _llr_comparison_metrics(delta, true_delta_l)
        reconstructed_llr = _binary_delta_l_to_raw_llr_1_minus_0(delta, bins_eval)
        if method_name == "ctsm_v_binary" and "ctsm_binary_llr_1_minus_0" in result:
            direct_llr = np.asarray(result["ctsm_binary_llr_1_minus_0"], dtype=np.float64).reshape(-1)
            if direct_llr.shape != reconstructed_llr.shape:
                raise ValueError(
                    "ctsm_v_binary returned ctsm_binary_llr_1_minus_0 shape "
                    f"{direct_llr.shape}, expected {reconstructed_llr.shape}."
                )
            if not np.allclose(direct_llr, reconstructed_llr, rtol=1e-5, atol=1e-7, equal_nan=True):
                max_abs = float(np.nanmax(np.abs(direct_llr - reconstructed_llr)))
                raise ValueError(
                    "ctsm_v_binary ctsm_binary_llr_1_minus_0 does not match "
                    f"the reconstructed vector from delta_l (max_abs={max_abs:.6g})."
                )
            raw_llr = direct_llr
        else:
            raw_llr = reconstructed_llr
        method_raw_llrs[method_name] = raw_llr
        raw_metrics_by_method[method_name] = _raw_llr_metrics(raw_llr, true_llr_1_minus_0)
        wall_s[i] = time.time() - t0

    out_base = out_dir / "llr_est_vs_true_all"
    _save_raw_binary_llr_est_vs_true_figure(
        method_raw_llrs,
        true_llr_1_minus_0,
        bins_eval,
        out_base=out_base,
        metrics_by_method=raw_metrics_by_method,
    )
    llr_svg = out_base.with_suffix(".svg").resolve()
    llr_png = out_base.with_suffix(".png").resolve()

    loss_panel_svg = Path(
        _render_row_n_training_losses_panel(
            row_labels=methods,
            n_list=[int(args.n_eval)],
            loss_root=str(loss_root),
            out_svg_path=out_dir / "h_decoding_categorical_llr_scatter_training_losses_panel.svg",
        )
    ).resolve()
    combined_svg, combined_png = _write_combined_llr_loss_outputs(
        llr_svg=llr_svg,
        llr_png=llr_png,
        loss_panel_svg=loss_panel_svg,
        combined_svg=out_dir / "llr_est_vs_true_all_with_losses.svg",
        combined_png=out_dir / "llr_est_vs_true_all_with_losses.png",
    )
    dataset_pca_svg, dataset_pca_png = _save_dataset_pca_projection_figure(
        x_eval,
        bins_eval,
        k_cat=k_cat,
        out_base=out_dir / "dataset_pca_projection",
    )
    dataset_pca_llr_mse_svg: Path | None = None
    dataset_pca_llr_mse_png: Path | None = None
    ctsm_v_binary_sample_mse: np.ndarray | None = None
    if "ctsm_v_binary" in method_deltas:
        ctsm_v_binary_sample_mse = _per_sample_llr_mse(method_deltas["ctsm_v_binary"], true_delta_l)
        dataset_pca_llr_mse_svg, dataset_pca_llr_mse_png = _save_dataset_pca_llr_mse_projection_figure(
            x_eval,
            ctsm_v_binary_sample_mse,
            method_label="ctsm_v_binary",
            out_base=out_dir / "dataset_pca_llr_mse_ctsm_v_binary",
        )
    else:
        print(
            "[llr-scatter] WARNING: ctsm_v_binary is not in --methods; "
            "skipping dataset PCA panel colored by LLR MSE.",
            flush=True,
        )
    combined_with_dataset_pca_svg, combined_with_dataset_pca_png = _write_combined_llr_loss_dataset_pca_outputs(
        llr_svg=llr_svg,
        llr_png=llr_png,
        loss_panel_svg=loss_panel_svg,
        dataset_pca_svg=dataset_pca_svg,
        dataset_pca_png=dataset_pca_png,
        dataset_pca_llr_mse_svg=dataset_pca_llr_mse_svg,
        dataset_pca_llr_mse_png=dataset_pca_llr_mse_png,
        combined_svg=out_dir / "llr_est_vs_true_all_with_losses_and_dataset_pca.svg",
        combined_png=out_dir / "llr_est_vs_true_all_with_losses_and_dataset_pca.png",
    )

    metric_method_names, metric_names, metric_values = _metric_table(metrics_by_method)
    raw_metric_method_names, raw_metric_names, raw_metric_values = _metric_table(raw_metrics_by_method)
    delta_stack = np.stack([method_deltas[m] for m in methods], axis=0)
    raw_llr_stack = np.stack([method_raw_llrs[m] for m in methods], axis=0)
    work_npz = pr_out_resolved.resolve() if pr_out_resolved is not None else native_npz
    results_npz = (out_dir / "llr_scatter_results.npz").resolve()
    result_payload: dict[str, Any] = dict(
        n_eval=np.int64(int(args.n_eval)),
        n_eval_matrix=np.int64(n_eval_matrix),
        eval_split=np.asarray([eval_split], dtype=object),
        num_categories=np.int64(k_cat),
        method_names=np.asarray(methods, dtype=object),
        true_delta_l=np.asarray(true_delta_l, dtype=np.float64),
        delta_l_est=np.asarray(delta_stack, dtype=np.float64),
        true_llr_1_minus_0=np.asarray(true_llr_1_minus_0, dtype=np.float64),
        llr_1_minus_0_est=np.asarray(raw_llr_stack, dtype=np.float64),
        llr_metric_method_names=metric_method_names,
        llr_metric_names=metric_names,
        llr_metric_values=metric_values,
        llr_raw_metric_method_names=raw_metric_method_names,
        llr_raw_metric_names=raw_metric_names,
        llr_raw_metric_values=raw_metric_values,
        wall_seconds=np.asarray(wall_s, dtype=np.float64),
        source_indices=source_indices,
        native_dataset_npz=np.asarray([str(native_npz)], dtype=object),
        work_dataset_npz=np.asarray([str(work_npz)], dtype=object),
        pr_projected=np.bool_(pr_project),
        pr_dim=np.int64(int(args.pr_dim) if pr_project else native_x_dim),
        run_seed=np.int64(int(args.run_seed)),
        training_losses_root=np.asarray([str(loss_root)], dtype=object),
        loss_panel_svg=np.asarray([str(loss_panel_svg)], dtype=object),
        combined_svg=np.asarray([str(combined_svg)], dtype=object),
        dataset_pca_svg=np.asarray([str(dataset_pca_svg)], dtype=object),
        dataset_pca_png=np.asarray([str(dataset_pca_png)], dtype=object),
        combined_with_dataset_pca_svg=np.asarray([str(combined_with_dataset_pca_svg)], dtype=object),
    )
    if ctsm_v_binary_sample_mse is not None:
        result_payload["ctsm_v_binary_per_sample_llr_mse"] = np.asarray(
            ctsm_v_binary_sample_mse,
            dtype=np.float64,
        )
    if dataset_pca_llr_mse_svg is not None:
        result_payload["dataset_pca_llr_mse_svg"] = np.asarray([str(dataset_pca_llr_mse_svg)], dtype=object)
    if dataset_pca_llr_mse_png is not None:
        result_payload["dataset_pca_llr_mse_png"] = np.asarray([str(dataset_pca_llr_mse_png)], dtype=object)
    if combined_png is not None:
        result_payload["combined_png"] = np.asarray([str(combined_png)], dtype=object)
    if combined_with_dataset_pca_png is not None:
        result_payload["combined_with_dataset_pca_png"] = np.asarray(
            [str(combined_with_dataset_pca_png)], dtype=object
        )
    np.savez_compressed(results_npz, **result_payload)

    summary_path = (out_dir / "llr_scatter_summary.txt").resolve()
    _write_summary(
        summary_path,
        args=args,
        methods=methods,
        eval_split=eval_split,
        n_eval_matrix=n_eval_matrix,
        dataset_npz=native_npz,
        work_dataset_npz=work_npz,
        results_npz=results_npz,
        llr_svg=llr_svg,
        llr_png=llr_png,
        training_losses_root=loss_root,
        loss_panel_svg=loss_panel_svg,
        combined_svg=combined_svg,
        combined_png=combined_png,
        dataset_pca_svg=dataset_pca_svg,
        dataset_pca_png=dataset_pca_png,
        dataset_pca_llr_mse_svg=dataset_pca_llr_mse_svg,
        dataset_pca_llr_mse_png=dataset_pca_llr_mse_png,
        combined_with_dataset_pca_svg=combined_with_dataset_pca_svg,
        combined_with_dataset_pca_png=combined_with_dataset_pca_png,
        wall_seconds=wall_s,
    )

    print("[llr-scatter] Saved:", flush=True)
    saved_paths = [
        results_npz,
        llr_svg,
        llr_png,
        loss_panel_svg,
        combined_svg,
        dataset_pca_svg,
        dataset_pca_png,
        combined_with_dataset_pca_svg,
    ]
    if dataset_pca_llr_mse_svg is not None:
        saved_paths.append(dataset_pca_llr_mse_svg)
    if dataset_pca_llr_mse_png is not None:
        saved_paths.append(dataset_pca_llr_mse_png)
    if combined_png is not None:
        saved_paths.append(combined_png)
    if combined_with_dataset_pca_png is not None:
        saved_paths.append(combined_with_dataset_pca_png)
    saved_paths.append(summary_path)
    for path in saved_paths:
        print(f"  - {path}", flush=True)


if __name__ == "__main__":
    main()
