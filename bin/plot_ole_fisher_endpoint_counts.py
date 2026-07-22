#!/usr/bin/env python3
"""Align a representative cross-fitted OLE curve with its endpoint counts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = (
    REPO_ROOT
    / "data"
    / "linear_fisher_xdim50_gkr_classical_flow_n500_1000_3000_5000_10000_r5"
    / "linear_fisher_gkr_classical_flow_results.npz"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "linear_fisher_three_panel_summary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--n-total", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _unique_index(values: np.ndarray, target: int, name: str) -> int:
    matches = np.flatnonzero(np.asarray(values, dtype=np.int64) == int(target))
    if matches.size != 1:
        raise ValueError(f"{name}={target} must occur exactly once in the archive.")
    return int(matches[0])


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if np.std(x_array) == 0.0 or np.std(y_array) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_array, y_array)[0, 1])


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.8)
    axis.spines["bottom"].set_linewidth(1.8)
    axis.tick_params(width=1.8)
    axis.grid(False)


def main() -> None:
    args = parse_args()
    results_path = args.results.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not results_path.is_file():
        raise FileNotFoundError(results_path)

    required = (
        "n_values",
        "seeds",
        "ole_theta",
        "ole_ground_truth",
        "ole_crossfit",
        "ole_crossfit_n_left",
        "ole_crossfit_n_right",
    )
    with np.load(results_path, allow_pickle=False) as result:
        missing = [key for key in required if key not in result.files]
        if missing:
            raise KeyError(f"Results archive is missing: {', '.join(missing)}")
        n_index = _unique_index(result["n_values"], int(args.n_total), "n_total")
        seed_index = _unique_index(result["seeds"], int(args.seed), "seed")
        theta = np.asarray(result["ole_theta"][n_index, seed_index], dtype=np.float64)
        truth = np.asarray(
            result["ole_ground_truth"][n_index, seed_index], dtype=np.float64
        )
        ole = np.asarray(result["ole_crossfit"][n_index, seed_index], dtype=np.float64)
        n_left = np.asarray(
            result["ole_crossfit_n_left"][n_index, seed_index], dtype=np.int64
        )
        n_right = np.asarray(
            result["ole_crossfit_n_right"][n_index, seed_index], dtype=np.int64
        )
        ole_spacing = float(
            np.asarray(result["ole_theta_spacing"])
            if "ole_theta_spacing" in result.files
            else np.median(np.diff(theta))
        )

    if not theta.shape == truth.shape == ole.shape == n_left.shape == n_right.shape:
        raise ValueError("OLE curves and endpoint-count arrays must have matching shapes.")
    if np.any(n_left <= 0) or np.any(n_right <= 0):
        raise ValueError("Endpoint-window counts must be positive.")

    absolute_error = np.abs(ole - truth)
    minimum_count = np.minimum(n_left, n_right)
    total_count = n_left + n_right

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 12,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, (fisher_axis, count_axis) = plt.subplots(
        2,
        1,
        figsize=(6.0, 5.5),
        sharex=True,
        gridspec_kw={"height_ratios": (1.45, 1.0)},
        constrained_layout=True,
    )
    fisher_axis.plot(
        theta,
        truth,
        color="black",
        linestyle="--",
        linewidth=2.3,
        label="Ground truth",
    )
    fisher_axis.plot(
        theta,
        ole,
        color="C1",
        marker="s",
        markersize=3.5,
        linewidth=2.0,
        label="OLE (cross-fit)",
    )
    fisher_axis.set_ylabel("Linear Fisher\ninformation")
    fisher_axis.set_title(rf"$N={int(args.n_total):,}$, seed {int(args.seed)}")
    fisher_axis.legend(frameon=False, loc="upper right", ncol=2)
    _style_axis(fisher_axis)

    count_axis.plot(
        theta,
        n_left,
        color="C0",
        marker="o",
        markersize=3.5,
        linewidth=1.9,
        label="Left window",
    )
    count_axis.plot(
        theta,
        n_right,
        color="C2",
        marker="^",
        markersize=4.0,
        linewidth=1.9,
        label="Right window",
    )
    count_axis.set_xlabel(r"$\theta$ midpoint")
    count_axis.set_ylabel("Data points")
    count_axis.legend(frameon=False, loc="lower center", ncol=2)
    _style_axis(count_axis)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"ole_fisher_endpoint_counts_n{int(args.n_total)}_seed{int(args.seed)}"
    figure_png = stem.with_suffix(".png")
    figure_svg = stem.with_suffix(".svg")
    fig.savefig(figure_png, dpi=300)
    fig.savefig(figure_svg)
    plt.close(fig)

    csv_path = stem.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "theta_midpoint",
                "ole_linear_fisher",
                "ground_truth_linear_fisher",
                "absolute_error",
                "n_left",
                "n_right",
                "n_min",
                "n_total",
            ),
        )
        writer.writeheader()
        for values in zip(
            theta,
            ole,
            truth,
            absolute_error,
            n_left,
            n_right,
            minimum_count,
            total_count,
            strict=True,
        ):
            writer.writerow(dict(zip(writer.fieldnames, values, strict=True)))

    summary = {
        "source_results": str(results_path),
        "n_total": int(args.n_total),
        "seed": int(args.seed),
        "ole_theta_spacing": ole_spacing,
        "count_definition": (
            "Selected disjoint left and right endpoint-window responses; each "
            "response is evaluated once held out across the cross-fitting folds."
        ),
        "n_left_range": [int(np.min(n_left)), int(np.max(n_left))],
        "n_right_range": [int(np.min(n_right)), int(np.max(n_right))],
        "n_total_range": [int(np.min(total_count)), int(np.max(total_count))],
        "pearson_abs_error_vs_min_endpoint_count": _pearson(
            absolute_error, minimum_count
        ),
        "pearson_abs_error_vs_total_count": _pearson(absolute_error, total_count),
        "figure_png": str(figure_png),
        "figure_svg": str(figure_svg),
        "aligned_values_csv": str(csv_path),
    }
    summary_path = stem.with_suffix(".json")
    summary["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
