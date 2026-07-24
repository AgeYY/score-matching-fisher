#!/usr/bin/env python3
"""Plot Gaussian-mixture linear Fisher curves and scaling errors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


METHOD_KEYS = {
    "Flow Matching": "flow_fisher",
    "GKR": "gkr_fisher",
    "OLE (cross-fit)": "ole_fisher",
}
COLORS = {
    "Flow Matching": "C0",
    "GKR": "C2",
    "OLE (cross-fit)": "C1",
}
MARKERS = {
    "Flow Matching": "o",
    "GKR": "^",
    "OLE (cross-fit)": "s",
}
DEFAULT_SAMPLE_ROOT = (
    REPO_ROOT / "data" / "toy_linear_fisher_density_xdim50_r5"
)
DEFAULT_DIMENSION_ROOT = (
    REPO_ROOT / "data" / "gaussian_mixture_linear_fisher_dimension_n3000_r5"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "data" / "gaussian_mixture_linear_fisher_three_panel"
)


def _csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("Expected a comma-separated integer list.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument(
        "--dimension-root", type=Path, default=DEFAULT_DIMENSION_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--n-list",
        type=_csv_ints,
        default=[100, 200, 300, 500, 700, 1000, 3000, 5000, 10000],
    )
    parser.add_argument(
        "--dimension-list",
        type=_csv_ints,
        default=[3, 10, 30, 50, 70, 90, 110],
    )
    parser.add_argument(
        "--seeds", type=_csv_ints, default=[7, 8, 9, 10, 11]
    )
    parser.add_argument("--representative-n", type=int, default=5000)
    parser.add_argument("--representative-seed", type=int, default=7)
    return parser.parse_args()


def case_result(
    *,
    sample_root: Path,
    dimension_root: Path,
    x_dim: int,
    n_total: int,
    seed: int,
) -> Path:
    if int(x_dim) == 50:
        return (
            sample_root
            / "cosine_gmm"
            / f"seed{int(seed)}"
            / f"n{int(n_total)}"
            / "linear_fisher_density_result.npz"
        )
    return (
        dimension_root
        / f"xdim{int(x_dim)}"
        / "cosine_gmm"
        / f"seed{int(seed)}"
        / f"n{int(n_total)}"
        / "linear_fisher_density_result.npz"
    )


def load_case(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as source:
        required = {
            "theta_midpoints",
            "ground_truth",
            *METHOD_KEYS.values(),
        }
        missing = required - set(source.files)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}.")
        return {key: np.asarray(source[key]) for key in source.files}


def mae(case: dict[str, np.ndarray], method: str) -> float:
    return float(
        np.mean(
            np.abs(
                np.asarray(case[METHOD_KEYS[method]], dtype=np.float64)
                - np.asarray(case["ground_truth"], dtype=np.float64)
            )
        )
    )


def collect(args: argparse.Namespace) -> dict[str, object]:
    sample_root = args.sample_root.expanduser().resolve()
    dimension_root = args.dimension_root.expanduser().resolve()
    n_values = np.asarray(args.n_list, dtype=np.int64)
    dimensions = np.asarray(args.dimension_list, dtype=np.int64)
    seeds = np.asarray(args.seeds, dtype=np.int64)

    sample_mae = {
        method: np.empty((len(n_values), len(seeds)), dtype=np.float64)
        for method in METHOD_KEYS
    }
    dimension_mae = {
        method: np.empty((len(dimensions), len(seeds)), dtype=np.float64)
        for method in METHOD_KEYS
    }
    for i, n_total in enumerate(n_values):
        for j, seed in enumerate(seeds):
            case = load_case(
                case_result(
                    sample_root=sample_root,
                    dimension_root=dimension_root,
                    x_dim=50,
                    n_total=int(n_total),
                    seed=int(seed),
                )
            )
            for method in METHOD_KEYS:
                sample_mae[method][i, j] = mae(case, method)
    for i, x_dim in enumerate(dimensions):
        for j, seed in enumerate(seeds):
            case = load_case(
                case_result(
                    sample_root=sample_root,
                    dimension_root=dimension_root,
                    x_dim=int(x_dim),
                    n_total=3000,
                    seed=int(seed),
                )
            )
            for method in METHOD_KEYS:
                dimension_mae[method][i, j] = mae(case, method)

    representative = load_case(
        case_result(
            sample_root=sample_root,
            dimension_root=dimension_root,
            x_dim=50,
            n_total=int(args.representative_n),
            seed=int(args.representative_seed),
        )
    )
    return {
        "n_values": n_values,
        "dimensions": dimensions,
        "seeds": seeds,
        "sample_mae": sample_mae,
        "dimension_mae": dimension_mae,
        "representative": representative,
    }


def plot(results: dict[str, object], output_dir: Path) -> tuple[Path, Path]:
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
    fig, axes = plt.subplots(
        1, 3, figsize=(11.0, 3.5), constrained_layout=True
    )
    representative = results["representative"]
    assert isinstance(representative, dict)
    theta = np.asarray(
        representative["theta_midpoints"], dtype=np.float64
    ).reshape(-1)
    truth = np.asarray(representative["ground_truth"], dtype=np.float64)
    axes[0].plot(
        theta,
        truth,
        color="black",
        linestyle="--",
        linewidth=2.3,
        label="Ground truth",
    )
    for method, key in METHOD_KEYS.items():
        axes[0].plot(
            theta,
            np.asarray(representative[key], dtype=np.float64),
            color=COLORS[method],
            linewidth=2.2,
            label=method,
        )
    axes[0].set_title(r"$d=50,\ N=5{,}000$")
    axes[0].set_xlabel(r"$\theta$")
    axes[0].set_ylabel("Linear Fisher information")

    def add_scaling(
        axis: plt.Axes,
        x: np.ndarray,
        values: dict[str, np.ndarray],
    ) -> None:
        for method in METHOD_KEYS:
            matrix = np.asarray(values[method], dtype=np.float64)
            axis.errorbar(
                x,
                matrix.mean(axis=1),
                yerr=matrix.std(axis=1, ddof=1),
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=5.5,
                linewidth=2.2,
                capsize=3.0,
            )
        axis.set_ylabel("Mean absolute error")

    sample_mae = results["sample_mae"]
    dimension_mae = results["dimension_mae"]
    assert isinstance(sample_mae, dict) and isinstance(dimension_mae, dict)
    n_values = np.asarray(results["n_values"], dtype=np.int64)
    dimensions = np.asarray(results["dimensions"], dtype=np.int64)
    add_scaling(axes[1], n_values, sample_mae)
    axes[1].set_title("Error versus sample size")
    axes[1].set_xlabel("Total samples")
    axes[1].set_xscale("log")
    preferred_ticks = {100, 300, 1000, 3000, 10000}
    sample_ticks = np.asarray(
        [value for value in n_values if int(value) in preferred_ticks],
        dtype=np.int64,
    )
    if sample_ticks.size == 0:
        sample_ticks = n_values
    axes[1].set_xticks(sample_ticks)
    axes[1].set_xticklabels(
        [
            f"{value // 1000}k" if value >= 1000 else str(value)
            for value in sample_ticks
        ]
    )
    add_scaling(axes[2], dimensions, dimension_mae)
    axes[2].set_title("Error versus dimension")
    axes[2].set_xlabel("Response dimension")
    axes[2].set_xticks(dimensions)
    dimension_ticklabels = axes[2].set_xticklabels(
        [str(value) for value in dimensions]
    )
    dimension_ticklabels[0].set_horizontalalignment("right")
    dimension_ticklabels[1].set_horizontalalignment("left")

    axes[1].legend(
        *axes[0].get_legend_handles_labels(),
        frameon=False,
        loc="center right",
        bbox_to_anchor=(0.99, 0.58),
        handlelength=1.8,
        labelspacing=0.25,
        borderaxespad=0.0,
    )
    for axis in axes:
        axis.set_axisbelow(True)
        axis.yaxis.grid(True, color="0.88", linewidth=0.8)
        axis.xaxis.grid(False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines["left"].set_linewidth(1.8)
        axis.spines["bottom"].set_linewidth(1.8)
        axis.tick_params(width=1.8)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "gaussian_mixture_linear_fisher_scaling"
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def save_results(
    results: dict[str, object],
    output_dir: Path,
    png: Path,
    svg: Path,
) -> tuple[Path, Path]:
    sample_mae = results["sample_mae"]
    dimension_mae = results["dimension_mae"]
    representative = results["representative"]
    assert isinstance(sample_mae, dict)
    assert isinstance(dimension_mae, dict)
    assert isinstance(representative, dict)
    npz = output_dir / "gaussian_mixture_linear_fisher_scaling_results.npz"
    np.savez_compressed(
        npz,
        n_values=np.asarray(results["n_values"]),
        dimensions=np.asarray(results["dimensions"]),
        seeds=np.asarray(results["seeds"]),
        theta_midpoints=np.asarray(representative["theta_midpoints"]),
        ground_truth=np.asarray(representative["ground_truth"]),
        **{
            f"representative_{key}": np.asarray(representative[key])
            for key in METHOD_KEYS.values()
        },
        **{
            f"sample_mae_{method.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}": values
            for method, values in sample_mae.items()
        },
        **{
            f"dimension_mae_{method.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}": values
            for method, values in dimension_mae.items()
        },
    )
    summary = {
        "dataset": "cosine_gmm",
        "target": "linear Fisher of exact marginal mean and covariance",
        "condition_grid": {
            "range": [-6.0, 6.0],
            "n_endpoints": 31,
            "spacing": 0.4,
        },
        "n_values": np.asarray(results["n_values"]).tolist(),
        "dimensions": np.asarray(results["dimensions"]).tolist(),
        "seeds": np.asarray(results["seeds"]).tolist(),
        "methods": {
            method: {
                "sample_mae_mean": values.mean(axis=1).tolist(),
                "sample_mae_std": values.std(axis=1, ddof=1).tolist(),
                "dimension_mae_mean": np.asarray(
                    dimension_mae[method]
                ).mean(axis=1).tolist(),
                "dimension_mae_std": np.asarray(
                    dimension_mae[method]
                ).std(axis=1, ddof=1).tolist(),
            }
            for method, values in sample_mae.items()
        },
        "figure_png": str(png),
        "figure_svg": str(svg),
        "results_npz": str(npz),
    }
    summary_path = (
        output_dir / "gaussian_mixture_linear_fisher_scaling_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return npz, summary_path


def main() -> None:
    args = parse_args()
    if int(args.representative_n) not in args.n_list:
        raise ValueError("--representative-n must appear in --n-list.")
    if int(args.representative_seed) not in args.seeds:
        raise ValueError("--representative-seed must appear in --seeds.")
    results = collect(args)
    output_dir = args.output_dir.expanduser().resolve()
    png, svg = plot(results, output_dir)
    npz, summary = save_results(results, output_dir, png, svg)
    print(
        json.dumps(
            {
                "figure_png": str(png),
                "figure_svg": str(svg),
                "results_npz": str(npz),
                "summary_json": str(summary),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
