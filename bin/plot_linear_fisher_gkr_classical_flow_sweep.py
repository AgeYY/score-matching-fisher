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

from fisher.optimal_linear_estimator import (
    cross_fitted_ole_linear_fisher,
    optimal_linear_estimator,
)
from fisher.continuous_fisher_comparison import native_linear_fisher_curve
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_meta


DEFAULT_N_LIST = (500, 1_000, 3_000, 5_000, 10_000)
DEFAULT_SEEDS = (7, 8, 9, 10, 11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--x-dim", type=int, default=50)
    parser.add_argument("--n-list", type=int, nargs="+", default=list(DEFAULT_N_LIST))
    parser.add_argument("--representative-n", type=int, default=5_000)
    parser.add_argument("--seed-list", type=int, nargs="+", default=[7])
    parser.add_argument("--theta-spacing", type=float, default=0.2)
    parser.add_argument("--min-endpoint-samples", type=int, default=8)
    parser.add_argument("--ole-crossfit-folds", type=int, default=5)
    parser.add_argument("--ole-crossfit-seed", type=int, default=20260721)
    parser.add_argument("--ole-theta-spacing", type=float, default=0.4)
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
    *, case_root: Path, x_dim: int, n_total: int, theta_spacing: float, seed: int
) -> tuple[Path, Path]:
    seed_suffix = "" if int(seed) == 7 else f"_datasetseed{int(seed)}"
    case_dir = case_root / f"gkr_fixed_xdim{x_dim}_n{n_total}_linear{seed_suffix}"
    dataset_path = case_dir / f"randamp_gaussian_sqrtd_xdim{x_dim}_n{n_total}.npz"
    train_seed_suffix = "" if int(seed) == 7 else f"_trainseed{int(seed)}"
    result_path = case_dir / (
        f"gkr_flow_fixed_xdim{x_dim}_n{n_total}_linear"
        f"{_spacing_suffix(theta_spacing)}_rbf8_hauto{train_seed_suffix}_results.npz"
    )
    return dataset_path, result_path


def _grid_with_spacing(reference_grid: np.ndarray, spacing: float) -> np.ndarray:
    reference = np.asarray(reference_grid, dtype=np.float64).reshape(-1)
    span = float(reference[-1] - reference[0])
    n_intervals = int(round(span / float(spacing)))
    if n_intervals < 1 or not np.isclose(
        n_intervals * float(spacing), span, rtol=1e-10, atol=1e-10
    ):
        raise ValueError("OLE spacing must evenly divide the condition range.")
    return np.linspace(reference[0], reference[-1], n_intervals + 1)


def _local_plugin_ledoit_wolf_linear_fisher(
    *,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    theta_grid: np.ndarray,
    min_endpoint_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate an OLE curve from local empirical means and covariances."""
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
    mean_derivatives = np.empty(
        (grid.shape[0] - 1, x.shape[1]), dtype=np.float64
    )
    pooled_covariances = np.empty(
        (grid.shape[0] - 1, x.shape[1], x.shape[1]), dtype=np.float64
    )
    for index, delta in enumerate(np.diff(grid)):
        mean_derivatives[index] = (
            local_means[index + 1] - local_means[index]
        ) / float(delta)
        pooled_covariances[index] = 0.5 * (
            local_covariances[index] + local_covariances[index + 1]
        )
    estimate = optimal_linear_estimator(mean_derivatives, pooled_covariances)
    return (
        estimate.linear_fisher,
        estimate.weights,
        estimate.variance,
        np.asarray(counts, dtype=np.int64),
    )


def _mae(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(
                np.asarray(estimate, dtype=np.float64)
                - np.asarray(truth, dtype=np.float64)
            )
        )
    )


def _sample_count_label(value: int) -> str:
    if int(value) >= 1_000 and int(value) % 1_000 == 0:
        return f"{int(value) // 1_000}k"
    return str(int(value))


def _plot(
    cases: list[dict[str, object]],
    representative_n: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    representative = next(
        case for case in cases if int(case["n_total"]) == int(representative_n)
    )
    representative_repeat = representative["repeats"][0]
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
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), constrained_layout=True)

    curve_axis = axes[0]
    theta = np.asarray(representative_repeat["theta"], dtype=np.float64)
    ole_theta = np.asarray(representative_repeat["ole_theta"], dtype=np.float64)
    curve_axis.plot(
        theta,
        np.asarray(representative_repeat["ground_truth"]),
        color="black",
        linestyle="--",
        linewidth=2.3,
        label="Ground truth",
    )
    curve_axis.plot(
        theta,
        np.asarray(representative_repeat["flow"]),
        color="C0",
        linewidth=2.2,
        label="Flow matching",
    )
    curve_axis.plot(
        theta,
        np.asarray(representative_repeat["gkr"]),
        color="C2",
        linewidth=2.2,
        label="GKR",
    )
    curve_axis.plot(
        ole_theta,
        np.asarray(representative_repeat["ole"]),
        color="C1",
        linewidth=2.2,
        label="OLE (cross-fit)",
    )
    curve_axis.set_xlabel(r"$\theta$")
    curve_axis.set_ylabel("Linear Fisher information")
    curve_axis.set_title(rf"$N={int(representative_n):,}$")
    curve_axis.legend(frameon=False, loc="best")

    error_axis = axes[1]
    n_values = np.asarray([int(case["n_total"]) for case in cases], dtype=np.int64)
    for key, label, color, marker in (
        ("flow_mae", "Flow matching", "C0", "o"),
        ("gkr_mae", "GKR", "C2", "^"),
        ("ole_mae", "OLE (cross-fit)", "C1", "s"),
    ):
        values = np.asarray([case[key] for case in cases], dtype=np.float64)
        error_axis.errorbar(
            n_values,
            np.mean(values, axis=1),
            yerr=(np.std(values, axis=1, ddof=1) if values.shape[1] > 1 else None),
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2.2,
            capsize=3,
            label=label,
        )
    error_axis.set_xscale("log")
    error_axis.set_xticks(n_values)
    error_axis.set_xticklabels([_sample_count_label(value) for value in n_values])
    error_axis.set_xlabel("Total samples")
    error_axis.set_ylabel("Mean absolute error")
    error_axis.set_title("Error versus sample size")
    error_axis.set_ylim(bottom=0.0)
    error_axis.set_axisbelow(True)
    error_axis.grid(axis="y", color="0.82", linewidth=0.8)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(1.8)
        axis.spines["bottom"].set_linewidth(1.8)
        axis.tick_params(width=1.8, labelsize=16)

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
    seeds = list(dict.fromkeys(int(value) for value in args.seed_list))
    if int(args.x_dim) < 1:
        raise ValueError("--x-dim must be >= 1.")
    if not n_list or any(value < 2 for value in n_list):
        raise ValueError("--n-list values must be >= 2.")
    if int(args.representative_n) not in n_list:
        raise ValueError("--representative-n must be included in --n-list.")
    if not seeds:
        raise ValueError("--seed-list must contain at least one seed.")
    if float(args.theta_spacing) <= 0.0:
        raise ValueError("--theta-spacing must be positive.")
    if int(args.min_endpoint_samples) < 2:
        raise ValueError("--min-endpoint-samples must be >= 2.")
    if int(args.ole_crossfit_folds) < 2:
        raise ValueError("--ole-crossfit-folds must be >= 2.")
    if int(args.min_endpoint_samples) < int(args.ole_crossfit_folds):
        raise ValueError("--min-endpoint-samples must be >= --ole-crossfit-folds.")
    if float(args.ole_theta_spacing) <= 0.0:
        raise ValueError("--ole-theta-spacing must be positive.")

    case_root = args.case_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []

    for n_total in n_list:
        repeats: list[dict[str, object]] = []
        for seed in seeds:
            dataset_path, result_path = _case_paths(
                case_root=case_root,
                x_dim=int(args.x_dim),
                n_total=n_total,
                theta_spacing=float(args.theta_spacing),
                seed=seed,
            )
            if not dataset_path.is_file() or not result_path.is_file():
                raise FileNotFoundError(
                    f"Missing cached inputs for N={n_total}, seed={seed}: "
                    f"{dataset_path} or {result_path}"
                )
            bundle = load_shared_dataset_npz(dataset_path)
            with np.load(result_path, allow_pickle=False) as result:
                theta_grid = np.asarray(result["theta_grid"], dtype=np.float64)
                theta = np.asarray(result["theta_midpoints"], dtype=np.float64).reshape(-1)
                truth = np.asarray(result["ground_truth_linear_fisher"], dtype=np.float64)
                flow = np.asarray(result["flow_linear_fisher"], dtype=np.float64)
                gkr = np.asarray(result["gkr_linear_fisher"], dtype=np.float64)
            ole_grid = _grid_with_spacing(theta_grid, float(args.ole_theta_spacing))
            ole_theta = 0.5 * (ole_grid[:-1] + ole_grid[1:])
            population = build_dataset_from_meta(bundle.meta)
            ole_truth = native_linear_fisher_curve(ole_theta, population)
            plugin, plugin_weights, plugin_variance, endpoint_counts = (
                _local_plugin_ledoit_wolf_linear_fisher(
                    theta_all=bundle.theta_all,
                    x_all=bundle.x_all,
                    theta_grid=ole_grid,
                    min_endpoint_samples=int(args.min_endpoint_samples),
                )
            )
            crossfit = cross_fitted_ole_linear_fisher(
                bundle.theta_all,
                bundle.x_all,
                ole_grid,
                n_splits=int(args.ole_crossfit_folds),
                seed=int(args.ole_crossfit_seed) + int(seed),
                min_endpoint_samples=int(args.min_endpoint_samples),
            )
            ole = crossfit.linear_fisher
            if not truth.shape == flow.shape == gkr.shape == theta.shape:
                raise ValueError(f"Curve shape mismatch for N={n_total}, seed={seed}.")
            if not ole.shape == ole_truth.shape == ole_theta.shape:
                raise ValueError(f"OLE curve shape mismatch for N={n_total}, seed={seed}.")
            repeats.append(
                {
                    "seed": seed,
                    "dataset_path": str(dataset_path),
                    "result_path": str(result_path),
                    "theta": theta,
                    "ole_theta": ole_theta,
                    "ground_truth": truth,
                    "ole_ground_truth": ole_truth,
                    "flow": flow,
                    "ole": ole,
                    "ole_raw": crossfit.linear_fisher_raw,
                    "ole_fold_weights": crossfit.fold_weights,
                    "ole_fold_intercepts": crossfit.fold_intercepts,
                    "ole_projected_mean_left": crossfit.projected_mean_left,
                    "ole_projected_mean_right": crossfit.projected_mean_right,
                    "ole_projected_variance_left": crossfit.projected_variance_left,
                    "ole_projected_variance_right": crossfit.projected_variance_right,
                    "ole_n_left": crossfit.n_left,
                    "ole_n_right": crossfit.n_right,
                    "plugin": plugin,
                    "plugin_weights": plugin_weights,
                    "plugin_variance": plugin_variance,
                    "gkr": gkr,
                    "flow_mae": _mae(flow, truth),
                    "ole_mae": _mae(ole, ole_truth),
                    "ole_raw_mae": _mae(crossfit.linear_fisher_raw, ole_truth),
                    "plugin_mae": _mae(plugin, ole_truth),
                    "gkr_mae": _mae(gkr, truth),
                    "endpoint_count_min": int(endpoint_counts.min()),
                    "endpoint_count_max": int(endpoint_counts.max()),
                }
            )
        cases.append(
            {
                "n_total": n_total,
                "repeats": repeats,
                "flow_mae": [float(repeat["flow_mae"]) for repeat in repeats],
                "ole_mae": [float(repeat["ole_mae"]) for repeat in repeats],
                "ole_raw_mae": [float(repeat["ole_raw_mae"]) for repeat in repeats],
                "plugin_mae": [float(repeat["plugin_mae"]) for repeat in repeats],
                "gkr_mae": [float(repeat["gkr_mae"]) for repeat in repeats],
            }
        )

    figure_png, figure_svg = _plot(
        cases,
        int(args.representative_n),
        output_dir,
    )
    results_npz = output_dir / "linear_fisher_gkr_classical_flow_results.npz"
    np.savez_compressed(
        results_npz,
        n_values=np.asarray(n_list, dtype=np.int64),
        seeds=np.asarray(seeds, dtype=np.int64),
        ole_theta_spacing=np.asarray(float(args.ole_theta_spacing), dtype=np.float64),
        theta=np.stack([np.stack([np.asarray(r["theta"]) for r in case["repeats"]]) for case in cases]),
        ole_theta=np.stack(
            [np.stack([np.asarray(r["ole_theta"]) for r in case["repeats"]]) for case in cases]
        ),
        ground_truth=np.stack([np.stack([np.asarray(r["ground_truth"]) for r in case["repeats"]]) for case in cases]),
        ole_ground_truth=np.stack(
            [np.stack([np.asarray(r["ole_ground_truth"]) for r in case["repeats"]]) for case in cases]
        ),
        flow=np.stack([np.stack([np.asarray(r["flow"]) for r in case["repeats"]]) for case in cases]),
        ole=np.stack(
            [np.stack([np.asarray(r["ole"]) for r in case["repeats"]]) for case in cases]
        ),
        ole_crossfit=np.stack(
            [np.stack([np.asarray(r["ole"]) for r in case["repeats"]]) for case in cases]
        ),
        ole_crossfit_raw=np.stack(
            [np.stack([np.asarray(r["ole_raw"]) for r in case["repeats"]]) for case in cases]
        ),
        ole_crossfit_fold_weights=np.stack(
            [np.stack([np.asarray(r["ole_fold_weights"]) for r in case["repeats"]]) for case in cases]
        ),
        ole_crossfit_fold_intercepts=np.stack(
            [np.stack([np.asarray(r["ole_fold_intercepts"]) for r in case["repeats"]]) for case in cases]
        ),
        ole_crossfit_projected_mean_left=np.stack(
            [np.stack([np.asarray(r["ole_projected_mean_left"]) for r in case["repeats"]]) for case in cases]
        ),
        ole_crossfit_projected_mean_right=np.stack(
            [np.stack([np.asarray(r["ole_projected_mean_right"]) for r in case["repeats"]]) for case in cases]
        ),
        ole_crossfit_projected_variance_left=np.stack(
            [np.stack([np.asarray(r["ole_projected_variance_left"]) for r in case["repeats"]]) for case in cases]
        ),
        ole_crossfit_projected_variance_right=np.stack(
            [np.stack([np.asarray(r["ole_projected_variance_right"]) for r in case["repeats"]]) for case in cases]
        ),
        ole_crossfit_n_left=np.stack(
            [np.stack([np.asarray(r["ole_n_left"]) for r in case["repeats"]]) for case in cases]
        ),
        ole_crossfit_n_right=np.stack(
            [np.stack([np.asarray(r["ole_n_right"]) for r in case["repeats"]]) for case in cases]
        ),
        ole_plugin=np.stack(
            [np.stack([np.asarray(r["plugin"]) for r in case["repeats"]]) for case in cases]
        ),
        ole_plugin_weights=np.stack(
            [np.stack([np.asarray(r["plugin_weights"]) for r in case["repeats"]]) for case in cases]
        ),
        ole_plugin_variance=np.stack(
            [np.stack([np.asarray(r["plugin_variance"]) for r in case["repeats"]]) for case in cases]
        ),
        classical_ledoit_wolf=np.stack(
            [np.stack([np.asarray(r["plugin"]) for r in case["repeats"]]) for case in cases]
        ),
        gkr=np.stack([np.stack([np.asarray(r["gkr"]) for r in case["repeats"]]) for case in cases]),
        flow_mae=np.asarray([case["flow_mae"] for case in cases], dtype=np.float64),
        ole_mae=np.asarray([case["ole_mae"] for case in cases], dtype=np.float64),
        ole_crossfit_mae=np.asarray([case["ole_mae"] for case in cases], dtype=np.float64),
        ole_crossfit_raw_mae=np.asarray(
            [case["ole_raw_mae"] for case in cases], dtype=np.float64
        ),
        ole_plugin_mae=np.asarray([case["plugin_mae"] for case in cases], dtype=np.float64),
        classical_ledoit_wolf_mae=np.asarray(
            [case["plugin_mae"] for case in cases], dtype=np.float64
        ),
        gkr_mae=np.asarray([case["gkr_mae"] for case in cases], dtype=np.float64),
    )

    csv_path = output_dir / "linear_fisher_gkr_classical_flow_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("n_total", "seed", "method", "mae")
        )
        writer.writeheader()
        for case in cases:
            for repeat in case["repeats"]:
                for key, method in (
                    ("flow_mae", "Flow matching"),
                    ("ole_mae", "OLE (cross-fitted held-out)"),
                    ("plugin_mae", "Local plug-in + LW"),
                    ("gkr_mae", "GKR"),
                ):
                    writer.writerow(
                        {
                            "n_total": int(case["n_total"]),
                            "seed": int(repeat["seed"]),
                            "method": method,
                            "mae": float(repeat[key]),
                        }
                    )

    summary = {
        "x_dim": int(args.x_dim),
        "n_list": n_list,
        "n_repeats": len(seeds),
        "seeds": seeds,
        "theta_spacing": float(args.theta_spacing),
        "ole_theta_spacing": float(args.ole_theta_spacing),
        "representative_n": int(args.representative_n),
        "error_metric": "mean absolute error over theta midpoints",
        "ole_estimator": {
            "definition": "cross-fitted locally unbiased linear decoder",
            "fit": "training-fold endpoint means and Ledoit-Wolf covariances",
            "evaluation": "bias-reduced achieved information from pooled held-out projections",
            "n_splits": int(args.ole_crossfit_folds),
            "crossfit_seed_base": int(args.ole_crossfit_seed),
            "adaptive_endpoint_fallback": "nearest disjoint 2*min_endpoint_samples block",
            "uses_all_samples": True,
            "min_endpoint_samples": int(args.min_endpoint_samples),
        },
        "flow_estimator": {
            "condition_embedding": "eight-center Gaussian RBF",
            "uses_training_fraction": 0.8,
        },
        "gkr_estimator": {"uses_training_fraction": 0.8},
        "error_bars": "sample standard deviation across seeds",
        "cases": [
            {
                "n_total": int(case["n_total"]),
                **{
                    f"{key}_{stat}": float(fn(np.asarray(case[key], dtype=np.float64)))
                    for key in ("flow_mae", "ole_mae", "gkr_mae")
                    for stat, fn in (
                        ("mean", np.mean),
                        (
                            "std",
                            lambda x: np.std(x, ddof=1) if x.size > 1 else 0.0,
                        ),
                    )
                },
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
