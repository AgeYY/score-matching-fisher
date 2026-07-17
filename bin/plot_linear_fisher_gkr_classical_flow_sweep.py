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
    curve_axis.set_xlabel(r"$\theta$")
    curve_axis.set_ylabel("Linear Fisher information")
    curve_axis.set_title(rf"$N={int(representative_n):,}$")
    curve_axis.legend(frameon=False, loc="best")

    error_axis = axes[1]
    n_values = np.asarray([int(case["n_total"]) for case in cases], dtype=np.int64)
    for key, label, color, marker in (
        ("flow_mae", "Flow matching", "C0", "o"),
        ("gkr_mae", "GKR", "C2", "^"),
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
    error_axis.set_xticklabels(["500", "1k", "3k", "5k", "10k"])
    error_axis.set_xlabel("Total samples")
    error_axis.set_ylabel("Mean absolute error")
    error_axis.set_title("Error versus sample size")
    error_axis.set_ylim(bottom=0.0)
    error_axis.set_axisbelow(True)
    error_axis.grid(axis="y", color="0.82", linewidth=0.8)

    inset_axis = error_axis.inset_axes((0.62, 0.54, 0.32, 0.37))
    for key, color, marker in (
        ("flow_mae", "C0", "o"),
        ("gkr_mae", "C2", "^"),
        ("classical_mae", "C1", "s"),
    ):
        values = np.asarray([case[key] for case in cases], dtype=np.float64)
        inset_axis.errorbar(
            n_values,
            np.mean(values, axis=1),
            yerr=(np.std(values, axis=1, ddof=1) if values.shape[1] > 1 else None),
            color=color,
            marker=marker,
            markersize=3.5,
            linewidth=1.6,
            capsize=2,
        )
    inset_axis.set_xscale("log")
    inset_axis.set_ylim(bottom=0.0)
    inset_axis.set_xticks((500, 3_000, 10_000))
    inset_axis.set_xticklabels(("500", "3k", "10k"))
    inset_axis.set_yticks((0.0, 100.0, 200.0))
    inset_axis.grid(False)
    for spine in inset_axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
    inset_axis.tick_params(width=1.5, length=3.0, labelsize=12)
    inset_axis.set_title(
        "Classical + LW",
        color="C1",
        fontsize=11,
        pad=3,
    )

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
            classical, endpoint_counts = _classical_ledoit_wolf_linear_fisher(
                theta_all=bundle.theta_all,
                x_all=bundle.x_all,
                theta_grid=theta_grid,
                min_endpoint_samples=int(args.min_endpoint_samples),
            )
            if not (
                truth.shape == flow.shape == gkr.shape == classical.shape == theta.shape
            ):
                raise ValueError(f"Curve shape mismatch for N={n_total}, seed={seed}.")
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
                    "endpoint_count_min": int(endpoint_counts.min()),
                    "endpoint_count_max": int(endpoint_counts.max()),
                }
            )
        cases.append(
            {
                "n_total": n_total,
                "repeats": repeats,
                "flow_mae": [float(repeat["flow_mae"]) for repeat in repeats],
                "classical_mae": [float(repeat["classical_mae"]) for repeat in repeats],
                "gkr_mae": [float(repeat["gkr_mae"]) for repeat in repeats],
            }
        )

    figure_png, figure_svg = _plot(cases, int(args.representative_n), output_dir)
    results_npz = output_dir / "linear_fisher_gkr_classical_flow_results.npz"
    np.savez_compressed(
        results_npz,
        n_values=np.asarray(n_list, dtype=np.int64),
        seeds=np.asarray(seeds, dtype=np.int64),
        theta=np.stack([np.stack([np.asarray(r["theta"]) for r in case["repeats"]]) for case in cases]),
        ground_truth=np.stack([np.stack([np.asarray(r["ground_truth"]) for r in case["repeats"]]) for case in cases]),
        flow=np.stack([np.stack([np.asarray(r["flow"]) for r in case["repeats"]]) for case in cases]),
        classical_ledoit_wolf=np.stack(
            [np.stack([np.asarray(r["classical"]) for r in case["repeats"]]) for case in cases]
        ),
        gkr=np.stack([np.stack([np.asarray(r["gkr"]) for r in case["repeats"]]) for case in cases]),
        flow_mae=np.asarray([case["flow_mae"] for case in cases], dtype=np.float64),
        classical_ledoit_wolf_mae=np.asarray(
            [case["classical_mae"] for case in cases], dtype=np.float64
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
                    ("classical_mae", "Classical + LW"),
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
        "error_bars": "sample standard deviation across seeds",
        "cases": [
            {
                "n_total": int(case["n_total"]),
                **{
                    f"{key}_{stat}": float(fn(np.asarray(case[key], dtype=np.float64)))
                    for key in ("flow_mae", "classical_mae", "gkr_mae")
                    for stat, fn in (("mean", np.mean), ("std", lambda x: np.std(x, ddof=1) if x.size > 1 else 0.0))
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
