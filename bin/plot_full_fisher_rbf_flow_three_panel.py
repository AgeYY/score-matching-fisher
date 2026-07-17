#!/usr/bin/env python3
"""Evaluate cached RBF affine flows and plot full Fisher scaling."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.continuous_fisher_comparison import (  # noqa: E402
    METHOD_GT_NATIVE_FULL,
    native_ground_truth_curves,
    theta_grid_from_meta,
    theta_midpoints,
)
from fisher.flow_matching_skl import (  # noqa: E402
    build_flow_skl_model,
    estimate_affine_gaussian_jeffreys_fisher,
)
from fisher.shared_dataset_io import load_shared_dataset_npz  # noqa: E402
from fisher.shared_fisher_est import require_device  # noqa: E402


DEFAULT_N_LIST = (500, 1_000, 3_000, 5_000, 10_000)
DEFAULT_DIMENSIONS = (3, 10, 30, 50, 70, 90, 110)
DEFAULT_SEEDS = (7, 8, 9, 10, 11)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "full_fisher_rbf_flow_three_panel_summary"
HIDDEN_DIM = 256
DEPTH = 5
QUADRATURE_STEPS = 64
PATH_SCHEDULE = "cosine"
ODE_STEPS = 64
RIDGE = 1e-6
THETA_RBF_CENTERS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--sample-x-dim", type=int, default=50)
    parser.add_argument("--sample-n-list", type=int, nargs="+", default=list(DEFAULT_N_LIST))
    parser.add_argument("--dimension-n-total", type=int, default=3_000)
    parser.add_argument(
        "--dimension-list", type=int, nargs="+", default=list(DEFAULT_DIMENSIONS)
    )
    parser.add_argument("--seed-list", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--representative-n", type=int, default=5_000)
    parser.add_argument("--theta-spacing", type=float, default=0.2)
    parser.add_argument("--ode-steps", type=int, default=ODE_STEPS)
    parser.add_argument("--ridge", type=float, default=RIDGE)
    parser.add_argument("--case-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _case_dir(case_root: Path, *, x_dim: int, n_total: int, seed: int) -> Path:
    suffix = "" if int(seed) == 7 else f"_datasetseed{int(seed)}"
    return case_root / f"gkr_fixed_xdim{int(x_dim)}_n{int(n_total)}_linear{suffix}"


def _checkpoint_path(case_dir: Path, *, seed: int) -> Path:
    suffix = "" if int(seed) == 7 else f"_trainseed{int(seed)}"
    return case_dir / f"flow_linear_rbf8_hauto{suffix}_selected_model.pt"


def _theta_grid(meta: dict[str, object], spacing: float) -> np.ndarray:
    span = float(meta["theta_high"]) - float(meta["theta_low"])
    n_intervals = int(round(span / float(spacing)))
    if n_intervals < 1 or not np.isclose(span / n_intervals, spacing):
        raise ValueError("--theta-spacing must evenly divide the condition range.")
    return theta_grid_from_meta(meta, theta_grid_size=n_intervals + 1)


def _evaluate_case(
    *,
    case_root: Path,
    x_dim: int,
    n_total: int,
    seed: int,
    theta_spacing: float,
    ode_steps: int,
    ridge: float,
    device: torch.device,
) -> dict[str, object]:
    case_dir = _case_dir(case_root, x_dim=x_dim, n_total=n_total, seed=seed)
    dataset_path = case_dir / f"randamp_gaussian_sqrtd_xdim{x_dim}_n{n_total}.npz"
    checkpoint_path = _checkpoint_path(case_dir, seed=seed)
    if not dataset_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing cached dataset or RBF-flow checkpoint: {dataset_path}, {checkpoint_path}"
        )

    bundle = load_shared_dataset_npz(dataset_path)
    grid = _theta_grid(dict(bundle.meta), float(theta_spacing))
    midpoints = theta_midpoints(grid)
    truth = np.asarray(
        native_ground_truth_curves(midpoints, dict(bundle.meta))[METHOD_GT_NATIVE_FULL],
        dtype=np.float64,
    )
    model = build_flow_skl_model(
        velocity_family="condition_affine",
        theta_dim=1,
        x_dim=int(x_dim),
        hidden_dim=HIDDEN_DIM,
        depth=DEPTH,
        quadrature_steps=QUADRATURE_STEPS,
        path_schedule=PATH_SCHEDULE,
        divergence_estimator="exact",
        hutchinson_probes=1,
        theta_embedding="gaussian_rbf",
        theta_rbf_num_centers=THETA_RBF_CENTERS,
        theta_rbf_lower=float(bundle.meta["theta_low"]),
        theta_rbf_upper=float(bundle.meta["theta_high"]),
        theta_rbf_bandwidth=None,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    estimate = estimate_affine_gaussian_jeffreys_fisher(
        model=model,
        theta_all=grid,
        device=device,
        ridge=float(ridge),
        ode_steps=int(ode_steps),
    )
    flow = np.asarray(estimate["fisher"], dtype=np.float64)
    if flow.shape != truth.shape:
        raise ValueError(f"Full-Fisher shape mismatch for d={x_dim}, N={n_total}, seed={seed}.")
    if not np.all(np.isfinite(flow)):
        raise FloatingPointError(f"Nonfinite full Fisher for d={x_dim}, N={n_total}, seed={seed}.")
    result = {
        "x_dim": int(x_dim),
        "n_total": int(n_total),
        "seed": int(seed),
        "theta": midpoints[:, 0],
        "ground_truth": truth,
        "flow": flow,
        "linear_component": np.asarray(estimate["linear_fisher"], dtype=np.float64),
        "covariance_component": np.asarray(
            estimate["covariance_fisher"], dtype=np.float64
        ),
        "mae": float(np.mean(np.abs(flow - truth))),
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(checkpoint_path),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _style_axis(axis: plt.Axes, *, grid: bool) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.8)
    axis.spines["bottom"].set_linewidth(1.8)
    axis.tick_params(width=1.8, labelsize=16)
    axis.set_axisbelow(True)
    if grid:
        axis.grid(axis="y", color="0.82", linewidth=0.8)
    else:
        axis.grid(False)


def _plot(
    *,
    sample_n: np.ndarray,
    sample_theta: np.ndarray,
    sample_truth: np.ndarray,
    sample_flow: np.ndarray,
    sample_mae: np.ndarray,
    dimensions: np.ndarray,
    dimension_mae: np.ndarray,
    representative_n: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    matches = np.flatnonzero(sample_n == int(representative_n))
    if matches.size != 1:
        raise ValueError("--representative-n must occur exactly once in --sample-n-list.")
    representative = int(matches[0])
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 13,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True)
    curve_axis, sample_axis, dimension_axis = axes

    curve_axis.plot(
        sample_theta[representative, 0],
        sample_truth[representative, 0],
        color="black",
        linestyle="--",
        linewidth=2.3,
        label="Ground truth",
    )
    curve_axis.plot(
        sample_theta[representative, 0],
        sample_flow[representative, 0],
        color="C0",
        linewidth=2.2,
        label="Flow matching",
    )
    curve_axis.set_xlabel(r"$\theta$")
    curve_axis.set_ylabel("Full Fisher information")
    curve_axis.set_title(rf"$N={int(representative_n):,}$")
    curve_axis.legend(frameon=False, loc="best")
    _style_axis(curve_axis, grid=False)

    sample_axis.errorbar(
        sample_n,
        np.mean(sample_mae, axis=1),
        yerr=np.std(sample_mae, axis=1, ddof=1),
        color="C0",
        marker="o",
        markersize=6,
        linewidth=2.2,
        capsize=3,
    )
    sample_axis.set_xscale("log")
    sample_axis.set_xticks(sample_n)
    sample_axis.set_xticklabels(("500", "1k", "3k", "5k", "10k"))
    sample_axis.set_ylim(bottom=0.0)
    sample_axis.set_xlabel("Total samples")
    sample_axis.set_ylabel("Mean absolute error")
    sample_axis.set_title("Error versus sample size")
    _style_axis(sample_axis, grid=True)

    dimension_axis.errorbar(
        dimensions,
        np.mean(dimension_mae, axis=1),
        yerr=np.std(dimension_mae, axis=1, ddof=1),
        color="C0",
        marker="o",
        markersize=6,
        linewidth=2.2,
        capsize=3,
    )
    dimension_axis.set_ylim(bottom=0.0)
    dimension_axis.set_xticks(dimensions)
    dimension_axis.set_xticklabels([str(value) for value in dimensions])
    tick_labels = dimension_axis.get_xticklabels()
    if 3 in dimensions and 10 in dimensions:
        tick_labels[dimensions.tolist().index(3)].set_horizontalalignment("right")
        tick_labels[dimensions.tolist().index(10)].set_horizontalalignment("left")
    dimension_axis.set_xlabel("Response dimension")
    dimension_axis.set_ylabel("Mean absolute error")
    dimension_axis.set_title("Error versus dimension")
    _style_axis(dimension_axis, grid=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "full_fisher_rbf_flow_curves_sample_and_dimension_errors"
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def main() -> None:
    args = parse_args()
    sample_n = np.asarray(sorted(set(int(value) for value in args.sample_n_list)))
    dimensions = np.asarray(sorted(set(int(value) for value in args.dimension_list)))
    seeds = np.asarray(list(dict.fromkeys(int(value) for value in args.seed_list)))
    if sample_n.size < 1 or dimensions.size < 1 or seeds.size < 2:
        raise ValueError("Sample sizes and dimensions must be nonempty; at least two seeds are required.")
    if int(args.representative_n) not in sample_n:
        raise ValueError("--representative-n must occur in --sample-n-list.")
    device = require_device(str(args.device))
    case_root = args.case_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    started = time.perf_counter()

    sample_cases: list[list[dict[str, object]]] = []
    for n_total in sample_n:
        repeats = []
        for seed in seeds:
            print(f"[full-fisher] sample d={args.sample_x_dim} N={n_total} seed={seed}", flush=True)
            repeats.append(
                _evaluate_case(
                    case_root=case_root,
                    x_dim=int(args.sample_x_dim),
                    n_total=int(n_total),
                    seed=int(seed),
                    theta_spacing=float(args.theta_spacing),
                    ode_steps=int(args.ode_steps),
                    ridge=float(args.ridge),
                    device=device,
                )
            )
        sample_cases.append(repeats)

    dimension_cases: list[list[dict[str, object]]] = []
    for x_dim in dimensions:
        repeats = []
        for seed in seeds:
            print(f"[full-fisher] dimension d={x_dim} N={args.dimension_n_total} seed={seed}", flush=True)
            repeats.append(
                _evaluate_case(
                    case_root=case_root,
                    x_dim=int(x_dim),
                    n_total=int(args.dimension_n_total),
                    seed=int(seed),
                    theta_spacing=float(args.theta_spacing),
                    ode_steps=int(args.ode_steps),
                    ridge=float(args.ridge),
                    device=device,
                )
            )
        dimension_cases.append(repeats)

    def stack(cases, key):
        return np.stack(
            [np.stack([np.asarray(repeat[key]) for repeat in repeats]) for repeats in cases]
        )

    sample_theta = stack(sample_cases, "theta")
    sample_truth = stack(sample_cases, "ground_truth")
    sample_flow = stack(sample_cases, "flow")
    sample_linear = stack(sample_cases, "linear_component")
    sample_covariance = stack(sample_cases, "covariance_component")
    sample_mae = np.asarray(
        [[float(repeat["mae"]) for repeat in repeats] for repeats in sample_cases]
    )
    dimension_theta = stack(dimension_cases, "theta")
    dimension_truth = stack(dimension_cases, "ground_truth")
    dimension_flow = stack(dimension_cases, "flow")
    dimension_linear = stack(dimension_cases, "linear_component")
    dimension_covariance = stack(dimension_cases, "covariance_component")
    dimension_mae = np.asarray(
        [[float(repeat["mae"]) for repeat in repeats] for repeats in dimension_cases]
    )

    figure_png, figure_svg = _plot(
        sample_n=sample_n,
        sample_theta=sample_theta,
        sample_truth=sample_truth,
        sample_flow=sample_flow,
        sample_mae=sample_mae,
        dimensions=dimensions,
        dimension_mae=dimension_mae,
        representative_n=int(args.representative_n),
        output_dir=output_dir,
    )
    results_path = output_dir / "full_fisher_rbf_flow_results.npz"
    np.savez_compressed(
        results_path,
        sample_n_values=sample_n,
        dimensions=dimensions,
        seeds=seeds,
        sample_theta=sample_theta,
        sample_ground_truth=sample_truth,
        sample_flow=sample_flow,
        sample_linear_component=sample_linear,
        sample_covariance_component=sample_covariance,
        sample_flow_mae=sample_mae,
        dimension_theta=dimension_theta,
        dimension_ground_truth=dimension_truth,
        dimension_flow=dimension_flow,
        dimension_linear_component=dimension_linear,
        dimension_covariance_component=dimension_covariance,
        dimension_flow_mae=dimension_mae,
    )
    elapsed = time.perf_counter() - started
    summary = {
        "method": "RBF-conditioned affine flow full Fisher via adjacent Gaussian Jeffreys",
        "sample_x_dim": int(args.sample_x_dim),
        "sample_n_values": sample_n.tolist(),
        "dimension_n_total": int(args.dimension_n_total),
        "dimensions": dimensions.tolist(),
        "seeds": seeds.tolist(),
        "theta_spacing": float(args.theta_spacing),
        "ode_steps": int(args.ode_steps),
        "ridge": float(args.ridge),
        "error_bars": "sample standard deviation across seeds",
        "sample_flow_mae_mean": np.mean(sample_mae, axis=1).tolist(),
        "sample_flow_mae_std": np.std(sample_mae, axis=1, ddof=1).tolist(),
        "dimension_flow_mae_mean": np.mean(dimension_mae, axis=1).tolist(),
        "dimension_flow_mae_std": np.std(dimension_mae, axis=1, ddof=1).tolist(),
        "elapsed_seconds": float(elapsed),
        "results_npz": str(results_path),
        "figure_png": str(figure_png),
        "figure_svg": str(figure_svg),
    }
    summary_path = output_dir / "full_fisher_rbf_flow_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
