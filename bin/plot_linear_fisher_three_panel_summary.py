#!/usr/bin/env python3
"""Combine Fisher curves, sample-size errors, and dimension errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_RESULTS = (
    REPO_ROOT
    / "data"
    / "linear_fisher_xdim50_gkr_classical_flow_n500_1000_3000_5000_10000_r5"
    / "linear_fisher_gkr_classical_flow_results.npz"
)
DEFAULT_DIMENSION_RESULTS = (
    REPO_ROOT
    / "data"
    / "linear_fisher_n3000_dimension_sweep_xdim3_10_30_50_70_90_110_r5"
    / "linear_fisher_dimension_sweep_results.npz"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "linear_fisher_three_panel_summary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--sample-results", type=Path, default=DEFAULT_SAMPLE_RESULTS)
    parser.add_argument(
        "--dimension-results", type=Path, default=DEFAULT_DIMENSION_RESULTS
    )
    parser.add_argument("--representative-n", type=int, default=5_000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _style_main_axis(axis: plt.Axes, *, grid: bool) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.8)
    axis.spines["bottom"].set_linewidth(1.8)
    axis.tick_params(width=1.8, labelsize=16)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="0.82", linewidth=0.8) if grid else axis.grid(False)


def _style_inset(axis: plt.Axes) -> None:
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
    axis.tick_params(width=1.5, length=3.0, labelsize=12)


def _plot_sample_inset(
    axis: plt.Axes,
    *,
    n_values: np.ndarray,
    flow: np.ndarray,
    gkr: np.ndarray,
    classical: np.ndarray,
    flow_error: np.ndarray | None = None,
    gkr_error: np.ndarray | None = None,
    classical_error: np.ndarray | None = None,
) -> None:
    for values, error, color, marker in (
        (flow, flow_error, "C0", "o"),
        (gkr, gkr_error, "C2", "^"),
        (classical, classical_error, "C1", "s"),
    ):
        axis.errorbar(
            n_values,
            values,
            yerr=error,
            color=color,
            marker=marker,
            markersize=3.5,
            linewidth=1.6,
            capsize=2,
        )
    axis.set_xscale("log")
    axis.set_ylim(bottom=0.0)
    axis.set_xticks((500, 3_000, 10_000))
    axis.set_xticklabels(("500", "3k", "10k"))
    axis.set_yticks((0.0, 100.0, 200.0))
    axis.set_title("Classical + LW", color="C1", fontsize=11, pad=3)
    _style_inset(axis)


def _plot_dimension_inset(
    axis: plt.Axes,
    *,
    dimensions: np.ndarray,
    flow: np.ndarray,
    gkr: np.ndarray,
    classical: np.ndarray,
) -> None:
    for values, color, marker in (
        (flow, "C0", "o"),
        (gkr, "C2", "^"),
        (classical, "C1", "s"),
    ):
        errors = np.std(values, axis=1, ddof=1) if values.shape[1] > 1 else None
        axis.errorbar(
            dimensions,
            np.mean(values, axis=1),
            yerr=errors,
            color=color,
            marker=marker,
            markersize=3.5,
            linewidth=1.6,
            capsize=2.0,
        )
    axis.set_ylim(bottom=0.0)
    axis.set_xticks((3, 50, 110))
    axis.set_yticks((0.0, 50.0, 100.0))
    axis.set_title("Classical + LW", color="C1", fontsize=11, pad=3)
    _style_inset(axis)


def main() -> None:
    args = parse_args()
    sample_path = args.sample_results.expanduser().resolve()
    dimension_path = args.dimension_results.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not sample_path.is_file():
        raise FileNotFoundError(sample_path)
    if not dimension_path.is_file():
        raise FileNotFoundError(dimension_path)

    with np.load(sample_path, allow_pickle=False) as result:
        n_values = np.asarray(result["n_values"], dtype=np.int64)
        theta = np.asarray(result["theta"], dtype=np.float64)
        ground_truth = np.asarray(result["ground_truth"], dtype=np.float64)
        sample_flow = np.asarray(result["flow"], dtype=np.float64)
        sample_gkr = np.asarray(result["gkr"], dtype=np.float64)
        sample_classical = np.asarray(
            result["classical_ledoit_wolf"], dtype=np.float64
        )
        sample_flow_mae = np.asarray(result["flow_mae"], dtype=np.float64)
        sample_gkr_mae = np.asarray(result["gkr_mae"], dtype=np.float64)
        sample_classical_mae = np.asarray(
            result["classical_ledoit_wolf_mae"], dtype=np.float64
        )
    representative_matches = np.flatnonzero(n_values == int(args.representative_n))
    if representative_matches.size != 1:
        raise ValueError("--representative-n must occur exactly once in sample results.")
    representative_index = int(representative_matches[0])
    if sample_flow_mae.ndim == 1:
        sample_flow_mean, sample_flow_error = sample_flow_mae, None
        sample_gkr_mean, sample_gkr_error = sample_gkr_mae, None
        sample_classical_mean, sample_classical_error = sample_classical_mae, None
    elif sample_flow_mae.ndim == 2:
        sample_flow_mean = np.mean(sample_flow_mae, axis=1)
        sample_gkr_mean = np.mean(sample_gkr_mae, axis=1)
        sample_classical_mean = np.mean(sample_classical_mae, axis=1)
        if sample_flow_mae.shape[1] > 1:
            sample_flow_error = np.std(sample_flow_mae, axis=1, ddof=1)
            sample_gkr_error = np.std(sample_gkr_mae, axis=1, ddof=1)
            sample_classical_error = np.std(sample_classical_mae, axis=1, ddof=1)
        else:
            sample_flow_error = None
            sample_gkr_error = None
            sample_classical_error = None
    else:
        raise ValueError("Sample-size MAE arrays must have one or two dimensions.")
    if theta.ndim == 3:
        theta = theta[:, 0]
        ground_truth = ground_truth[:, 0]
        sample_flow = sample_flow[:, 0]
        sample_gkr = sample_gkr[:, 0]

    with np.load(dimension_path, allow_pickle=False) as result:
        dimensions = np.asarray(result["x_dims"], dtype=np.int64)
        dimension_flow_mae = np.asarray(result["flow_mae"], dtype=np.float64)
        dimension_gkr_mae = np.asarray(result["gkr_mae"], dtype=np.float64)
        dimension_classical_mae = np.asarray(
            result["classical_ledoit_wolf_mae"], dtype=np.float64
        )

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
        theta[representative_index],
        ground_truth[representative_index],
        color="black",
        linestyle="--",
        linewidth=2.3,
        label="Ground truth",
    )
    curve_axis.plot(
        theta[representative_index],
        sample_flow[representative_index],
        color="C0",
        linewidth=2.2,
        label="Flow matching",
    )
    curve_axis.plot(
        theta[representative_index],
        sample_gkr[representative_index],
        color="C2",
        linewidth=2.2,
        label="GKR",
    )
    curve_axis.set_xlabel(r"$\theta$")
    curve_axis.set_ylabel("Linear Fisher information")
    curve_axis.set_title(rf"$N={int(args.representative_n):,}$")
    curve_axis.legend(frameon=False, loc="lower center", fontsize=13)
    _style_main_axis(curve_axis, grid=False)

    for values, error, color, marker in (
        (sample_flow_mean, sample_flow_error, "C0", "o"),
        (sample_gkr_mean, sample_gkr_error, "C2", "^"),
    ):
        sample_axis.errorbar(
            n_values,
            values,
            yerr=error,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2.2,
            capsize=3,
        )
    sample_axis.set_xscale("log")
    sample_axis.set_xticks(n_values)
    sample_axis.set_xticklabels(("500", "1k", "3k", "5k", "10k"))
    sample_axis.set_ylim(bottom=0.0)
    sample_axis.set_xlabel("Total samples")
    sample_axis.set_ylabel("Mean absolute error")
    sample_axis.set_title("Error versus sample size")
    _style_main_axis(sample_axis, grid=True)
    sample_inset = sample_axis.inset_axes((0.62, 0.54, 0.32, 0.37))
    _plot_sample_inset(
        sample_inset,
        n_values=n_values,
        flow=sample_flow_mean,
        gkr=sample_gkr_mean,
        classical=sample_classical_mean,
        flow_error=sample_flow_error,
        gkr_error=sample_gkr_error,
        classical_error=sample_classical_error,
    )

    for values, color, marker in (
        (dimension_flow_mae, "C0", "o"),
        (dimension_gkr_mae, "C2", "^"),
    ):
        errors = np.std(values, axis=1, ddof=1) if values.shape[1] > 1 else None
        dimension_axis.errorbar(
            dimensions,
            np.mean(values, axis=1),
            yerr=errors,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2.2,
            capsize=3,
        )
    dimension_axis.set_ylim(bottom=0.0)
    dimension_axis.set_xticks(dimensions)
    dimension_axis.set_xticklabels([str(value) for value in dimensions])
    dimension_ticks = dimension_axis.get_xticklabels()
    dimension_ticks[dimensions.tolist().index(3)].set_horizontalalignment("right")
    dimension_ticks[dimensions.tolist().index(10)].set_horizontalalignment("left")
    dimension_axis.set_xlabel("Response dimension")
    dimension_axis.set_ylabel("Mean absolute error")
    dimension_axis.set_title("Error versus dimension")
    _style_main_axis(dimension_axis, grid=True)
    dimension_inset = dimension_axis.inset_axes((0.10, 0.53, 0.32, 0.37))
    _plot_dimension_inset(
        dimension_inset,
        dimensions=dimensions,
        flow=dimension_flow_mae,
        gkr=dimension_gkr_mae,
        classical=dimension_classical_mae,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / "linear_fisher_curves_sample_and_dimension_errors"
    figure_png = output_stem.with_suffix(".png")
    figure_svg = output_stem.with_suffix(".svg")
    fig.savefig(figure_png, dpi=300)
    fig.savefig(figure_svg)
    plt.close(fig)

    summary = {
        "sample_results": str(sample_path),
        "dimension_results": str(dimension_path),
        "representative_n": int(args.representative_n),
        "figure_png": str(figure_png),
        "figure_svg": str(figure_svg),
    }
    summary_path = output_dir / "linear_fisher_three_panel_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
