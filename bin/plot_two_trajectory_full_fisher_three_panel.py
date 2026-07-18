#!/usr/bin/env python3
"""Plot full-Fisher curve and repeated sample/dimension scaling sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_N_LIST = (500, 1_000, 3_000, 5_000, 10_000)
DEFAULT_DIMENSIONS = (3, 10, 30, 50, 70, 90, 110)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--sample-x-dim", type=int, default=50)
    parser.add_argument("--sample-n-list", type=int, nargs="+", default=list(DEFAULT_N_LIST))
    parser.add_argument("--dimension-n-total", type=int, default=3_000)
    parser.add_argument("--dimension-list", type=int, nargs="+", default=list(DEFAULT_DIMENSIONS))
    parser.add_argument("--representative-n", type=int, default=5_000)
    parser.add_argument(
        "--representative-case-dir",
        type=Path,
        default=None,
        help="Optional result directory for the representative case.",
    )
    return parser.parse_args()


def load_case(case_dir: Path) -> dict[str, np.ndarray | float]:
    result_path = case_dir / "two_trajectory_full_fisher_results.npz"
    tre_result_path = case_dir / "two_trajectory_binned_tre_full_fisher_results.npz"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    with np.load(result_path, allow_pickle=False) as result:
        theta = np.asarray(result["theta_midpoints"], dtype=np.float64).reshape(-1)
        truth = np.asarray(result["ground_truth_full_fisher"], dtype=np.float64)
        flow = np.asarray(result["flow_full_fisher"], dtype=np.float64)
    if not tre_result_path.is_file():
        raise FileNotFoundError(tre_result_path)
    with np.load(tre_result_path, allow_pickle=False) as result:
        tre = np.asarray(result["tre_full_fisher"], dtype=np.float64)
    if theta.shape != truth.shape or truth.shape != flow.shape or flow.shape != tre.shape:
        raise ValueError(f"Curve shape mismatch in {result_path}.")
    return {
        "theta": theta,
        "truth": truth,
        "flow": flow,
        "tre": tre,
        "flow_mae": float(np.mean(np.abs(flow - truth))),
        "tre_mae": float(np.mean(np.abs(tre - truth))),
    }


def case_dir(case_root: Path, *, seed: int, x_dim: int, n_total: int) -> Path:
    return case_root / f"seed{int(seed)}" / f"xdim{int(x_dim)}_n{int(n_total)}"


def style_axis(axis: plt.Axes, *, grid: bool) -> None:
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


def main() -> int:
    args = parse_args()
    case_root = args.case_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    seeds = np.asarray(list(dict.fromkeys(int(seed) for seed in args.seeds)), dtype=np.int64)
    if seeds.size < 1:
        raise ValueError("--seeds must contain at least one seed.")
    n_values = np.asarray(sorted(set(int(value) for value in args.sample_n_list)), dtype=np.int64)
    dimensions = np.asarray(sorted(set(int(value) for value in args.dimension_list)), dtype=np.int64)
    representative_n = int(args.representative_n)
    if representative_n not in n_values:
        raise ValueError("--representative-n must occur in --sample-n-list.")

    sample_mae_by_repeat = np.asarray(
        [
            [
                load_case(
                    case_dir(
                        case_root,
                        seed=int(seed),
                        x_dim=int(args.sample_x_dim),
                        n_total=int(n_total),
                    )
                )["flow_mae"]
                for n_total in n_values
            ]
            for seed in seeds
        ],
        dtype=np.float64,
    )
    dimension_mae_by_repeat = np.asarray(
        [
            [
                load_case(
                    case_dir(
                        case_root,
                        seed=int(seed),
                        x_dim=int(x_dim),
                        n_total=int(args.dimension_n_total),
                    )
                )["flow_mae"]
                for x_dim in dimensions
            ]
            for seed in seeds
        ],
        dtype=np.float64,
    )
    sample_tre_mae_by_repeat = np.asarray(
        [
            [
                load_case(
                    case_dir(
                        case_root,
                        seed=int(seed),
                        x_dim=int(args.sample_x_dim),
                        n_total=int(n_total),
                    )
                )["tre_mae"]
                for n_total in n_values
            ]
            for seed in seeds
        ],
        dtype=np.float64,
    )
    dimension_tre_mae_by_repeat = np.asarray(
        [
            [
                load_case(
                    case_dir(
                        case_root,
                        seed=int(seed),
                        x_dim=int(x_dim),
                        n_total=int(args.dimension_n_total),
                    )
                )["tre_mae"]
                for x_dim in dimensions
            ]
            for seed in seeds
        ],
        dtype=np.float64,
    )
    representative_dir = (
        args.representative_case_dir.expanduser().resolve()
        if args.representative_case_dir is not None
        else case_dir(
            case_root,
            seed=int(seeds[0]),
            x_dim=int(args.sample_x_dim),
            n_total=representative_n,
        )
    )
    representative = load_case(representative_dir)
    sample_mae = sample_mae_by_repeat.mean(axis=0)
    dimension_mae = dimension_mae_by_repeat.mean(axis=0)
    ddof = 1 if seeds.size > 1 else 0
    sample_std = sample_mae_by_repeat.std(axis=0, ddof=ddof)
    dimension_std = dimension_mae_by_repeat.std(axis=0, ddof=ddof)
    sample_tre_mae = sample_tre_mae_by_repeat.mean(axis=0)
    dimension_tre_mae = dimension_tre_mae_by_repeat.mean(axis=0)
    sample_tre_std = sample_tre_mae_by_repeat.std(axis=0, ddof=ddof)
    dimension_tre_std = dimension_tre_mae_by_repeat.std(axis=0, ddof=ddof)

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
        representative["theta"],
        representative["truth"],
        color="black",
        linestyle="--",
        linewidth=2.3,
        label="Ground truth",
    )
    curve_axis.plot(
        representative["theta"],
        representative["flow"],
        color="C0",
        linewidth=2.2,
        label="Flow matching",
    )
    curve_axis.plot(
        representative["theta"],
        representative["tre"],
        color="C3",
        linewidth=2.2,
        label="Binned TRE-8",
    )
    curve_axis.set_xlabel(r"$\theta$")
    curve_axis.set_ylabel("Full Fisher information")
    representative_label = f"{representative_n:,}".replace(",", "{,}")
    curve_axis.set_title(rf"$N={representative_label}$")
    curve_axis.legend(frameon=False, loc="best")
    style_axis(curve_axis, grid=False)

    sample_axis.errorbar(
        n_values,
        sample_mae,
        yerr=sample_std,
        color="C0",
        marker="o",
        markersize=6,
        linewidth=2.2,
        capsize=3,
        label="Flow matching",
    )
    sample_axis.errorbar(
        n_values,
        sample_tre_mae,
        yerr=sample_tre_std,
        color="C3",
        marker="s",
        markersize=6,
        linewidth=2.2,
        capsize=3,
        label="Binned TRE-8",
    )
    sample_axis.set_xscale("log")
    sample_axis.set_xticks(n_values)
    sample_axis.set_xticklabels([f"{value // 1000}k" if value >= 1000 else str(value) for value in n_values])
    sample_axis.margins(y=0.08)
    sample_axis.set_xlabel("Total samples")
    sample_axis.set_ylabel("Mean absolute error")
    sample_axis.set_yscale("log")
    sample_axis.set_title("Error versus sample size")
    style_axis(sample_axis, grid=True)

    dimension_axis.errorbar(
        dimensions,
        dimension_mae,
        yerr=dimension_std,
        color="C0",
        marker="o",
        markersize=6,
        linewidth=2.2,
        capsize=3,
        label="Flow matching",
    )
    dimension_axis.errorbar(
        dimensions,
        dimension_tre_mae,
        yerr=dimension_tre_std,
        color="C3",
        marker="s",
        markersize=6,
        linewidth=2.2,
        capsize=3,
        label="Binned TRE-8",
    )
    dimension_axis.set_xticks(dimensions)
    dimension_axis.set_xticklabels([str(value) for value in dimensions])
    dimension_tick_labels = dimension_axis.get_xticklabels()
    if 3 in dimensions and 10 in dimensions:
        dimension_tick_labels[dimensions.tolist().index(3)].set_horizontalalignment("right")
        dimension_tick_labels[dimensions.tolist().index(10)].set_horizontalalignment("left")
    dimension_axis.margins(y=0.08)
    dimension_axis.set_xlabel("Response dimension")
    dimension_axis.set_ylabel("Mean absolute error")
    dimension_axis.set_yscale("log")
    dimension_axis.set_title("Error versus dimension")
    style_axis(dimension_axis, grid=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "two_trajectory_full_fisher_curve_sample_dimension_errors"
    figure_png = stem.with_suffix(".png")
    figure_svg = stem.with_suffix(".svg")
    fig.savefig(figure_png, dpi=300)
    fig.savefig(figure_svg)
    plt.close(fig)
    results_path = output_dir / "two_trajectory_full_fisher_sweep_results.npz"
    np.savez_compressed(
        results_path,
        sample_n=n_values,
        sample_mae=sample_mae,
        sample_mae_std=sample_std,
        sample_mae_by_repeat=sample_mae_by_repeat,
        sample_tre_mae=sample_tre_mae,
        sample_tre_mae_std=sample_tre_std,
        sample_tre_mae_by_repeat=sample_tre_mae_by_repeat,
        dimensions=dimensions,
        dimension_mae=dimension_mae,
        dimension_mae_std=dimension_std,
        dimension_mae_by_repeat=dimension_mae_by_repeat,
        dimension_tre_mae=dimension_tre_mae,
        dimension_tre_mae_std=dimension_tre_std,
        dimension_tre_mae_by_repeat=dimension_tre_mae_by_repeat,
        seeds=seeds,
        representative_theta=np.asarray(representative["theta"]),
        representative_truth=np.asarray(representative["truth"]),
        representative_flow=np.asarray(representative["flow"]),
        representative_tre=np.asarray(representative["tre"]),
    )
    summary = {
        "sample_x_dim": int(args.sample_x_dim),
        "sample_n_list": n_values.tolist(),
        "sample_mae": sample_mae.tolist(),
        "sample_mae_std": sample_std.tolist(),
        "sample_mae_by_repeat": sample_mae_by_repeat.tolist(),
        "sample_tre_mae": sample_tre_mae.tolist(),
        "sample_tre_mae_std": sample_tre_std.tolist(),
        "sample_tre_mae_by_repeat": sample_tre_mae_by_repeat.tolist(),
        "dimension_n_total": int(args.dimension_n_total),
        "dimension_list": dimensions.tolist(),
        "dimension_mae": dimension_mae.tolist(),
        "dimension_mae_std": dimension_std.tolist(),
        "dimension_mae_by_repeat": dimension_mae_by_repeat.tolist(),
        "dimension_tre_mae": dimension_tre_mae.tolist(),
        "dimension_tre_mae_std": dimension_tre_std.tolist(),
        "dimension_tre_mae_by_repeat": dimension_tre_mae_by_repeat.tolist(),
        "representative_n": representative_n,
        "representative_case_dir": str(representative_dir),
        "error_metric": "mean absolute error over theta midpoints",
        "n_repeats": int(seeds.size),
        "seeds": seeds.tolist(),
        "error_bars": "sample standard deviation across repeats",
        "theta_spacing": 0.4,
        "divergence_estimator": "four-probe Hutchinson",
        "baseline_estimator": "adjacent-bin TRE with 8 bridges",
        "y_scale": {"fisher_curve": "linear", "error_panels": "log"},
        "figure_png": str(figure_png),
        "figure_svg": str(figure_svg),
        "results_npz": str(results_path),
    }
    summary_path = output_dir / "two_trajectory_full_fisher_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
