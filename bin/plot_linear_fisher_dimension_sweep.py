#!/usr/bin/env python3
"""Plot 3,000-sample linear Fisher error versus response dimension."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import LedoitWolf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.shared_dataset_io import load_shared_dataset_npz


DEFAULT_DIMS = (3, 10, 30, 50, 70, 90, 110)
DEFAULT_SEEDS = (7, 8, 9, 10, 11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-total", type=int, default=3_000)
    parser.add_argument("--x-dim-list", type=int, nargs="+", default=list(DEFAULT_DIMS))
    parser.add_argument("--seed-list", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--theta-spacing", type=float, default=0.2)
    parser.add_argument("--min-endpoint-samples", type=int, default=8)
    parser.add_argument("--case-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "linear_fisher_n3000_dimension_sweep_xdim3_10_30_50_70_90_110_r5",
    )
    return parser.parse_args()


def _spacing_suffix(spacing: float) -> str:
    if np.isclose(float(spacing), 0.4):
        return ""
    return "_h" + f"{float(spacing):g}".replace(".", "p")


def _case_paths(
    *, case_root: Path, x_dim: int, n_total: int, theta_spacing: float, seed: int
) -> tuple[Path, Path]:
    seed_suffix = "" if seed == 7 else f"_datasetseed{seed}"
    case_dir = case_root / f"gkr_fixed_xdim{x_dim}_n{n_total}_linear{seed_suffix}"
    dataset_path = case_dir / f"randamp_gaussian_sqrtd_xdim{x_dim}_n{n_total}.npz"
    train_seed_suffix = "" if seed == 7 else f"_trainseed{seed}"
    result_path = case_dir / (
        f"gkr_flow_fixed_xdim{x_dim}_n{n_total}_linear"
        f"{_spacing_suffix(theta_spacing)}_rbf8_hauto{train_seed_suffix}_results.npz"
    )
    return dataset_path, result_path


def _classical_ledoit_wolf_linear_fisher(
    *,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    theta_grid: np.ndarray,
    min_endpoint_samples: int,
) -> np.ndarray:
    theta = np.asarray(theta_all, dtype=np.float64).reshape(-1)
    x = np.asarray(x_all, dtype=np.float64)
    grid = np.asarray(theta_grid, dtype=np.float64).reshape(-1)
    radius = 0.5 * float(np.min(np.diff(grid)))

    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    for value in grid:
        indices = np.flatnonzero(np.abs(theta - value) <= radius + 1e-12)
        if int(indices.size) < int(min_endpoint_samples):
            indices = np.argsort(np.abs(theta - value), kind="mergesort")[
                : int(min_endpoint_samples)
            ]
        fitted = LedoitWolf().fit(x[indices])
        means.append(np.asarray(fitted.location_, dtype=np.float64))
        covariances.append(np.asarray(fitted.covariance_, dtype=np.float64))

    local_means = np.asarray(means, dtype=np.float64)
    local_covariances = np.asarray(covariances, dtype=np.float64)
    fisher = np.empty(grid.shape[0] - 1, dtype=np.float64)
    for index, delta in enumerate(np.diff(grid)):
        mean_delta = local_means[index + 1] - local_means[index]
        covariance = 0.5 * (
            local_covariances[index] + local_covariances[index + 1]
        )
        fisher[index] = float(
            mean_delta @ np.linalg.solve(covariance, mean_delta) / float(delta) ** 2
        )
    return fisher


def _mae(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(estimate) - np.asarray(truth))))


def _plot(cases: list[dict[str, object]], output_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 12,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axis = plt.subplots(figsize=(4.0, 3.5))
    dimensions = np.asarray([int(case["x_dim"]) for case in cases], dtype=np.int64)

    def add_errorbar(
        target_axis: plt.Axes,
        key: str,
        label: str,
        color: str,
        marker: str,
        *,
        markersize: float = 6.0,
        linewidth: float = 2.2,
        capsize: float = 3.0,
    ) -> None:
        values = np.asarray([case[key] for case in cases], dtype=np.float64)
        means = np.mean(values, axis=1)
        errors = np.std(values, axis=1, ddof=1)
        target_axis.errorbar(
            dimensions,
            means,
            yerr=errors,
            color=color,
            marker=marker,
            markersize=markersize,
            linewidth=linewidth,
            capsize=capsize,
            label=label,
        )

    add_errorbar(axis, "flow_mae", "Flow matching", "C0", "o")
    add_errorbar(axis, "gkr_mae", "GKR", "C2", "^")
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="0.82", linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(2.2)
    axis.spines["bottom"].set_linewidth(2.2)
    axis.tick_params(width=1.8, labelsize=16)
    axis.set_ylim(0.0, 1.55)
    axis.set_yticks((0.0, 0.5, 1.0, 1.5))
    axis.set_xticks(dimensions)
    axis.set_xticklabels([str(value) for value in dimensions])
    tick_labels = axis.get_xticklabels()
    tick_labels[dimensions.tolist().index(3)].set_horizontalalignment("right")
    tick_labels[dimensions.tolist().index(10)].set_horizontalalignment("left")
    axis.set_xlabel("Response dimension")
    axis.set_ylabel("Mean absolute error")

    inset_axis = axis.inset_axes((0.15, 0.39, 0.27, 0.301))
    for key, label, color, marker in (
        ("flow_mae", "Flow matching", "C0", "o"),
        ("gkr_mae", "GKR", "C2", "^"),
        ("classical_mae", "Classical + LW", "C1", "s"),
    ):
        add_errorbar(
            inset_axis,
            key,
            label,
            color,
            marker,
            markersize=4.0,
            linewidth=1.8,
            capsize=2.5,
        )
    inset_axis.set_xlim(axis.get_xlim())
    inset_axis.set_ylim(0.0, 125.0)
    inset_axis.set_yticks((0.0, 50.0, 100.0))
    inset_axis.set_xticks((10, 50, 90))
    inset_axis.grid(False)
    for spine in inset_axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
    inset_axis.tick_params(width=1.5, length=3.0, labelsize=14)
    inset_axis.set_title("Classical + LW", color="C1", fontsize=10, pad=2)
    axis.legend(
        frameon=False,
        loc="upper left",
        fontsize=13,
        ncol=1,
        handletextpad=0.6,
    )
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.18, top=0.98)

    output_stem = output_dir / "linear_fisher_error_vs_dimension_n3000"
    png = output_stem.with_suffix(".png")
    svg = output_stem.with_suffix(".svg")
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def main() -> None:
    args = parse_args()
    dimensions = sorted({int(value) for value in args.x_dim_list})
    seeds = list(dict.fromkeys(int(value) for value in args.seed_list))
    if int(args.n_total) < 2:
        raise ValueError("--n-total must be >= 2.")
    if not dimensions or any(value < 1 for value in dimensions):
        raise ValueError("--x-dim-list values must be >= 1.")
    if len(seeds) < 2:
        raise ValueError("--seed-list must contain at least two distinct seeds.")
    if float(args.theta_spacing) <= 0.0:
        raise ValueError("--theta-spacing must be positive.")
    if int(args.min_endpoint_samples) < 2:
        raise ValueError("--min-endpoint-samples must be >= 2.")

    case_root = args.case_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []

    for x_dim in dimensions:
        repeats: list[dict[str, object]] = []
        for seed in seeds:
            dataset_path, result_path = _case_paths(
                case_root=case_root,
                x_dim=x_dim,
                n_total=int(args.n_total),
                theta_spacing=float(args.theta_spacing),
                seed=seed,
            )
            if not dataset_path.is_file() or not result_path.is_file():
                raise FileNotFoundError(
                    f"Missing cached inputs for x_dim={x_dim}, seed={seed}: "
                    f"{dataset_path} or {result_path}"
                )
            bundle = load_shared_dataset_npz(dataset_path)
            with np.load(result_path, allow_pickle=False) as result:
                theta_grid = np.asarray(result["theta_grid"], dtype=np.float64)
                theta = np.asarray(result["theta_midpoints"], dtype=np.float64).reshape(-1)
                truth = np.asarray(result["ground_truth_linear_fisher"], dtype=np.float64)
                flow = np.asarray(result["flow_linear_fisher"], dtype=np.float64)
                gkr = np.asarray(result["gkr_linear_fisher"], dtype=np.float64)
            classical = _classical_ledoit_wolf_linear_fisher(
                theta_all=bundle.theta_all,
                x_all=bundle.x_all,
                theta_grid=theta_grid,
                min_endpoint_samples=int(args.min_endpoint_samples),
            )
            if not (
                theta.shape == truth.shape == flow.shape == gkr.shape == classical.shape
            ):
                raise ValueError(f"Curve shape mismatch for x_dim={x_dim}, seed={seed}.")
            repeats.append(
                {
                    "seed": seed,
                    "dataset_path": str(dataset_path),
                    "result_path": str(result_path),
                    "theta": theta,
                    "ground_truth": truth,
                    "flow": flow,
                    "classical": classical,
                    "gkr": gkr,
                    "flow_mae": _mae(flow, truth),
                    "classical_mae": _mae(classical, truth),
                    "gkr_mae": _mae(gkr, truth),
                }
            )
        cases.append(
            {
                "x_dim": x_dim,
                "repeats": repeats,
                "flow_mae": [float(repeat["flow_mae"]) for repeat in repeats],
                "classical_mae": [float(repeat["classical_mae"]) for repeat in repeats],
                "gkr_mae": [float(repeat["gkr_mae"]) for repeat in repeats],
            }
        )

    figure_png, figure_svg = _plot(cases, output_dir)
    results_npz = output_dir / "linear_fisher_dimension_sweep_results.npz"
    np.savez_compressed(
        results_npz,
        x_dims=np.asarray(dimensions, dtype=np.int64),
        seeds=np.asarray(seeds, dtype=np.int64),
        theta=np.stack(
            [np.stack([np.asarray(repeat["theta"]) for repeat in case["repeats"]]) for case in cases]
        ),
        ground_truth=np.stack(
            [
                np.stack([np.asarray(repeat["ground_truth"]) for repeat in case["repeats"]])
                for case in cases
            ]
        ),
        flow=np.stack(
            [np.stack([np.asarray(repeat["flow"]) for repeat in case["repeats"]]) for case in cases]
        ),
        classical_ledoit_wolf=np.stack(
            [
                np.stack([np.asarray(repeat["classical"]) for repeat in case["repeats"]])
                for case in cases
            ]
        ),
        gkr=np.stack(
            [np.stack([np.asarray(repeat["gkr"]) for repeat in case["repeats"]]) for case in cases]
        ),
        flow_mae=np.asarray([case["flow_mae"] for case in cases], dtype=np.float64),
        classical_ledoit_wolf_mae=np.asarray(
            [case["classical_mae"] for case in cases], dtype=np.float64
        ),
        gkr_mae=np.asarray([case["gkr_mae"] for case in cases], dtype=np.float64),
    )

    csv_path = output_dir / "linear_fisher_dimension_sweep_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("x_dim", "seed", "method", "mae"))
        writer.writeheader()
        for case in cases:
            for repeat in case["repeats"]:
                for key, method in (
                    ("flow_mae", "Flow matching"),
                    ("classical_mae", "Classical + LW"),
                    ("gkr_mae", "GKR"),
                ):
                    writer.writerow(
                        {
                            "x_dim": int(case["x_dim"]),
                            "seed": int(repeat["seed"]),
                            "method": method,
                            "mae": float(repeat[key]),
                        }
                    )

    summary = {
        "n_total": int(args.n_total),
        "x_dim_list": dimensions,
        "n_repeats": len(seeds),
        "seeds": seeds,
        "population_note": (
            "Each seed and dimensionality has a separate population draw; dimensions "
            "and repeats are not nested coordinate subsets."
        ),
        "theta_spacing": float(args.theta_spacing),
        "error_metric": "mean absolute error over theta midpoints",
        "classical_estimator": "local finite-difference mean with Ledoit-Wolf covariance",
        "flow_condition_embedding": "eight-center Gaussian RBF",
        "error_bars": "sample standard deviation across seeds",
        "cases": [
            {
                "x_dim": int(case["x_dim"]),
                **{
                    f"{key}_{statistic}": float(function(np.asarray(case[key], dtype=np.float64)))
                    for key in ("flow_mae", "classical_mae", "gkr_mae")
                    for statistic, function in (("mean", np.mean), ("std", lambda x: np.std(x, ddof=1)))
                },
                "repeats": [
                    {
                        key: value
                        for key, value in repeat.items()
                        if key not in {"theta", "ground_truth", "flow", "classical", "gkr"}
                    }
                    for repeat in case["repeats"]
                ],
            }
            for case in cases
        ],
        "figure_png": str(figure_png),
        "figure_svg": str(figure_svg),
        "results_npz": str(results_npz),
        "errors_csv": str(csv_path),
    }
    summary_path = output_dir / "linear_fisher_dimension_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
