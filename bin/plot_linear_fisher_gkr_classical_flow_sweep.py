#!/usr/bin/env python3
"""Plot linear Fisher curves and sample-size errors for cached 50D runs."""

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


DEFAULT_N_LIST = (500, 1_000, 3_000, 5_000, 10_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--x-dim", type=int, default=50)
    parser.add_argument("--n-list", type=int, nargs="+", default=list(DEFAULT_N_LIST))
    parser.add_argument("--representative-n", type=int, default=5_000)
    parser.add_argument("--theta-spacing", type=float, default=0.2)
    parser.add_argument("--min-endpoint-samples", type=int, default=8)
    parser.add_argument("--case-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "linear_fisher_xdim50_gkr_classical_flow_n500_1000_3000_5000_10000_r1",
    )
    return parser.parse_args()


def _spacing_suffix(spacing: float) -> str:
    if np.isclose(float(spacing), 0.4):
        return ""
    return "_h" + f"{float(spacing):g}".replace(".", "p")


def _case_paths(
    *, case_root: Path, x_dim: int, n_total: int, theta_spacing: float
) -> tuple[Path, Path]:
    case_dir = case_root / f"gkr_fixed_xdim{x_dim}_n{n_total}_linear"
    dataset_path = case_dir / f"randamp_gaussian_sqrtd_xdim{x_dim}_n{n_total}.npz"
    result_path = case_dir / (
        f"gkr_flow_fixed_xdim{x_dim}_n{n_total}_linear"
        f"{_spacing_suffix(theta_spacing)}_rbf8_hauto_results.npz"
    )
    return dataset_path, result_path


def _classical_ledoit_wolf_linear_fisher(
    *,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    theta_grid: np.ndarray,
    min_endpoint_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta_all, dtype=np.float64).reshape(-1)
    x = np.asarray(x_all, dtype=np.float64)
    grid = np.asarray(theta_grid, dtype=np.float64).reshape(-1)
    radius = 0.5 * float(np.min(np.diff(grid)))

    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    counts: list[int] = []
    for value in grid:
        indices = np.flatnonzero(np.abs(theta - value) <= radius + 1e-12)
        if int(indices.size) < int(min_endpoint_samples):
            indices = np.argsort(np.abs(theta - value), kind="mergesort")[
                : int(min_endpoint_samples)
            ]
        fitted = LedoitWolf().fit(x[indices])
        means.append(np.asarray(fitted.location_, dtype=np.float64))
        covariances.append(np.asarray(fitted.covariance_, dtype=np.float64))
        counts.append(int(indices.size))

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
    return fisher, np.asarray(counts, dtype=np.int64)


def _mae(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(
                np.asarray(estimate, dtype=np.float64)
                - np.asarray(truth, dtype=np.float64)
            )
        )
    )


def _plot(cases: list[dict[str, object]], representative_n: int, output_dir: Path) -> tuple[Path, Path]:
    representative = next(
        case for case in cases if int(case["n_total"]) == int(representative_n)
    )
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), constrained_layout=True)

    curve_axis = axes[0]
    theta = np.asarray(representative["theta"], dtype=np.float64)
    curve_axis.plot(
        theta,
        np.asarray(representative["ground_truth"]),
        color="black",
        linestyle="--",
        linewidth=2.3,
        label="Ground truth",
    )
    curve_axis.plot(
        theta,
        np.asarray(representative["flow"]),
        color="C0",
        linewidth=2.2,
        label="Flow matching",
    )
    curve_axis.plot(
        theta,
        np.asarray(representative["gkr"]),
        color="C2",
        linewidth=2.2,
        label="GKR",
    )
    curve_axis.set_yscale("log")
    curve_axis.set_xlabel(r"$\theta$")
    curve_axis.set_ylabel("Linear Fisher information")
    curve_axis.set_title(rf"$N={int(representative_n):,}$")
    curve_axis.legend(frameon=False, loc="best")

    error_axis = axes[1]
    n_values = np.asarray([int(case["n_total"]) for case in cases], dtype=np.int64)
    for key, label, color, marker in (
        ("flow_mae", "Flow matching", "C0", "o"),
        ("classical_mae", "Classical + LW", "C1", "s"),
        ("gkr_mae", "GKR", "C2", "^"),
    ):
        error_axis.plot(
            n_values,
            np.asarray([float(case[key]) for case in cases]),
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2.2,
            label=label,
        )
    error_axis.set_xscale("log")
    error_axis.set_yscale("log")
    error_axis.set_xticks(n_values)
    error_axis.set_xticklabels(["500", "1k", "3k", "5k", "10k"])
    error_axis.set_xlabel("Total samples")
    error_axis.set_ylabel("Mean absolute error")
    error_axis.set_title("Error versus sample size")

    for axis in axes:
        for spine in axis.spines.values():
            spine.set_linewidth(1.8)
        axis.tick_params(width=1.8)

    output_stem = output_dir / "linear_fisher_gkr_classical_flow_curves_and_error"
    png = output_stem.with_suffix(".png")
    svg = output_stem.with_suffix(".svg")
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def main() -> None:
    args = parse_args()
    n_list = sorted({int(value) for value in args.n_list})
    if int(args.x_dim) < 1:
        raise ValueError("--x-dim must be >= 1.")
    if not n_list or any(value < 2 for value in n_list):
        raise ValueError("--n-list values must be >= 2.")
    if int(args.representative_n) not in n_list:
        raise ValueError("--representative-n must be included in --n-list.")
    if float(args.theta_spacing) <= 0.0:
        raise ValueError("--theta-spacing must be positive.")
    if int(args.min_endpoint_samples) < 2:
        raise ValueError("--min-endpoint-samples must be >= 2.")

    case_root = args.case_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []

    for n_total in n_list:
        dataset_path, result_path = _case_paths(
            case_root=case_root,
            x_dim=int(args.x_dim),
            n_total=n_total,
            theta_spacing=float(args.theta_spacing),
        )
        if not dataset_path.is_file() or not result_path.is_file():
            raise FileNotFoundError(
                f"Missing cached inputs for N={n_total}: {dataset_path} or {result_path}"
            )
        bundle = load_shared_dataset_npz(dataset_path)
        with np.load(result_path, allow_pickle=False) as result:
            theta_grid = np.asarray(result["theta_grid"], dtype=np.float64)
            theta = np.asarray(result["theta_midpoints"], dtype=np.float64).reshape(-1)
            truth = np.asarray(result["ground_truth_linear_fisher"], dtype=np.float64)
            flow = np.asarray(result["flow_linear_fisher"], dtype=np.float64)
            gkr = np.asarray(result["gkr_linear_fisher"], dtype=np.float64)
        classical, endpoint_counts = _classical_ledoit_wolf_linear_fisher(
            theta_all=bundle.theta_all,
            x_all=bundle.x_all,
            theta_grid=theta_grid,
            min_endpoint_samples=int(args.min_endpoint_samples),
        )
        if not (truth.shape == flow.shape == gkr.shape == classical.shape == theta.shape):
            raise ValueError(f"Curve shape mismatch for N={n_total}.")
        cases.append(
            {
                "n_total": n_total,
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
                "endpoint_count_min": int(endpoint_counts.min()),
                "endpoint_count_max": int(endpoint_counts.max()),
            }
        )

    figure_png, figure_svg = _plot(cases, int(args.representative_n), output_dir)
    results_npz = output_dir / "linear_fisher_gkr_classical_flow_results.npz"
    np.savez_compressed(
        results_npz,
        n_values=np.asarray(n_list, dtype=np.int64),
        theta=np.stack([np.asarray(case["theta"]) for case in cases]),
        ground_truth=np.stack([np.asarray(case["ground_truth"]) for case in cases]),
        flow=np.stack([np.asarray(case["flow"]) for case in cases]),
        classical_ledoit_wolf=np.stack(
            [np.asarray(case["classical"]) for case in cases]
        ),
        gkr=np.stack([np.asarray(case["gkr"]) for case in cases]),
        flow_mae=np.asarray([float(case["flow_mae"]) for case in cases]),
        classical_ledoit_wolf_mae=np.asarray(
            [float(case["classical_mae"]) for case in cases]
        ),
        gkr_mae=np.asarray([float(case["gkr_mae"]) for case in cases]),
    )

    csv_path = output_dir / "linear_fisher_gkr_classical_flow_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("n_total", "method", "mae"))
        writer.writeheader()
        for case in cases:
            for key, method in (
                ("flow_mae", "Flow matching"),
                ("classical_mae", "Classical + LW"),
                ("gkr_mae", "GKR"),
            ):
                writer.writerow(
                    {
                        "n_total": int(case["n_total"]),
                        "method": method,
                        "mae": float(case[key]),
                    }
                )

    summary = {
        "x_dim": int(args.x_dim),
        "n_list": n_list,
        "n_repeats": 1,
        "dataset_seed": 7,
        "theta_spacing": float(args.theta_spacing),
        "representative_n": int(args.representative_n),
        "error_metric": "mean absolute error over theta midpoints",
        "classical_estimator": {
            "covariance": "Ledoit-Wolf shrinkage",
            "mean_derivative": "adjacent local-window finite difference",
            "uses_all_samples": True,
            "min_endpoint_samples": int(args.min_endpoint_samples),
        },
        "flow_estimator": {
            "condition_embedding": "eight-center Gaussian RBF",
            "uses_training_fraction": 0.8,
        },
        "gkr_estimator": {"uses_training_fraction": 0.8},
        "cases": [
            {
                key: value
                for key, value in case.items()
                if key not in {"theta", "ground_truth", "flow", "classical", "gkr"}
            }
            for case in cases
        ],
        "figure_png": str(figure_png),
        "figure_svg": str(figure_svg),
        "results_npz": str(results_npz),
        "errors_csv": str(csv_path),
    }
    summary_path = output_dir / "linear_fisher_gkr_classical_flow_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
