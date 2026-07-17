#!/usr/bin/env python3
"""Aggregate repeated continuous-Fisher runs with a Ledoit-Wolf baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import LedoitWolf


METHODS = ("Classical + LW", "Flow matching", "GKR")
FAMILIES = ("linear", "full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-list", type=int, nargs="+", required=True)
    parser.add_argument("--population-seed", type=int, default=7)
    parser.add_argument("--repeat-seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--result-filename",
        type=str,
        default="continuous_pr_fisher_results.npz",
        help="Per-case result archive to aggregate.",
    )
    parser.add_argument(
        "--reuse-lw",
        action="store_true",
        help="Reuse existing per-case Ledoit-Wolf archives instead of refitting.",
    )
    return parser.parse_args()


def case_paths(
    repo_root: Path,
    case_root: Path,
    n_total: int,
    repeat_idx: int,
    result_filename: str = "continuous_pr_fisher_results.npz",
) -> tuple[Path, Path, Path]:
    if repeat_idx == 0:
        case_dir = repo_root / f"data/randamp_gaussian_sqrtd_xdim100_native_n{n_total}"
    else:
        case_dir = case_root / f"n{n_total}" / f"repeat_{repeat_idx:02d}"
    dataset_path = case_dir / "randamp_gaussian_sqrtd_xdim100_native.npz"
    result_path = case_dir / "continuous_pr_fisher" / result_filename
    lw_path = case_dir / "continuous_pr_fisher/classical_ledoit_wolf_fisher.npz"
    return dataset_path, result_path, lw_path


def fit_ledoit_wolf(
    dataset_path: Path,
    result_path: Path,
    output_path: Path,
) -> dict[str, np.ndarray]:
    with np.load(dataset_path, allow_pickle=False) as data, np.load(
        result_path, allow_pickle=False
    ) as result:
        theta = np.asarray(data["theta_all"], dtype=np.float64).reshape(-1)
        x = np.asarray(data["x_all"], dtype=np.float64)
        grid = np.asarray(result["theta_grid"], dtype=np.float64).reshape(-1)

    radius = 0.5 * float(np.min(np.diff(grid)))
    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    counts: list[int] = []
    for value in grid:
        indices = np.flatnonzero(np.abs(theta - value) <= radius + 1e-12)
        if indices.size < 8:
            indices = np.argsort(np.abs(theta - value))[:8]
        fitted = LedoitWolf().fit(x[indices])
        means.append(fitted.location_)
        covariances.append(fitted.covariance_)
        counts.append(int(indices.size))

    local_means = np.asarray(means)
    local_covariances = np.asarray(covariances)
    linear_fisher: list[float] = []
    full_fisher: list[float] = []
    dimension = x.shape[1]
    for index, delta in enumerate(np.diff(grid)):
        mean_delta = local_means[index + 1] - local_means[index]
        average_covariance = 0.5 * (
            local_covariances[index] + local_covariances[index + 1]
        )
        linear_fisher.append(
            float(
                mean_delta @ np.linalg.solve(average_covariance, mean_delta)
                / delta**2
            )
        )

        covariance_0 = local_covariances[index]
        covariance_1 = local_covariances[index + 1]
        mean_term = mean_delta @ (
            np.linalg.solve(covariance_0, mean_delta)
            + np.linalg.solve(covariance_1, mean_delta)
        )
        jeffreys = 0.5 * (
            np.trace(np.linalg.solve(covariance_1, covariance_0))
            + np.trace(np.linalg.solve(covariance_0, covariance_1))
            + mean_term
            - 2 * dimension
        )
        full_fisher.append(float(jeffreys / delta**2))

    arrays = {
        "theta_midpoints": 0.5 * (grid[:-1] + grid[1:]),
        "classical_lw_linear_fisher": np.asarray(linear_fisher),
        "classical_lw_full_fisher": np.asarray(full_fisher),
        "local_window_counts": np.asarray(counts, dtype=np.int64),
        "local_means": local_means,
        "local_covariances": local_covariances,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return arrays


def plot_summary(summary_rows: list[dict[str, object]], n_list: list[int], output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    styles = {
        "Classical + LW": {"color": "C1", "marker": "s"},
        "Flow matching": {"color": "C0", "marker": "o"},
        "GKR": {"color": "C2", "marker": "^"},
    }
    for axis, family, title in zip(
        axes, FAMILIES, ("Linear Fisher", "Full Fisher"), strict=True
    ):
        for method in METHODS:
            selected = sorted(
                (
                    row
                    for row in summary_rows
                    if row["family"] == family and row["method"] == method
                ),
                key=lambda row: int(row["n_total"]),
            )
            means = np.asarray([row["mae_mean"] for row in selected], dtype=float)
            standard_deviations = np.asarray(
                [row["mae_sd"] for row in selected], dtype=float
            )
            axis.errorbar(
                n_list,
                means,
                yerr=standard_deviations,
                label=method,
                linewidth=2.0,
                markersize=7,
                capsize=3.0,
                capthick=1.5,
                elinewidth=1.5,
                **styles[method],
            )
        axis.set_yscale("log")
        axis.set_xticks(n_list)
        axis.set_xticklabels([f"{value / 1000:g}k" for value in n_list])
        axis.set_xlabel("Total samples")
        axis.set_ylabel("Mean absolute error")
        axis.set_title(title)
        for spine in axis.spines.values():
            spine.set_linewidth(1.8)
        axis.tick_params(width=1.8)
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout(w_pad=2.0)
    stem = output_dir / "continuous_fisher_xdim100_classical_lw_error_vs_n_r5"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def plot_curves_and_errors(
    summary_rows: list[dict[str, object]],
    n_list: list[int],
    result_path: Path,
    lw_path: Path,
    representative_n: int,
    output_dir: Path,
) -> None:
    with np.load(result_path, allow_pickle=False) as result, np.load(
        lw_path, allow_pickle=False
    ) as lw:
        theta = np.asarray(result["theta_midpoints"], dtype=float).reshape(-1)
        representative_curves = {
            "linear": {
                "Ground truth": np.asarray(
                    result["ground_truth_native_linear_fisher"], dtype=float
                ),
                "Classical + LW": np.asarray(
                    lw["classical_lw_linear_fisher"], dtype=float
                ),
                "Flow matching": np.asarray(result["flow_linear_fisher"], dtype=float),
                "GKR": np.asarray(result["gkr_linear_fisher"], dtype=float),
            },
            "full": {
                "Ground truth": np.asarray(
                    result["ground_truth_native_full_fisher"], dtype=float
                ),
                "Classical + LW": np.asarray(
                    lw["classical_lw_full_fisher"], dtype=float
                ),
                "Flow matching": np.asarray(result["flow_full_fisher"], dtype=float),
                "GKR": np.asarray(result["gkr_full_fisher"], dtype=float),
            },
        }

    styles = {
        "Ground truth": {"color": "black", "linestyle": "--", "marker": None},
        "Classical + LW": {"color": "C1", "linestyle": "-", "marker": "s"},
        "Flow matching": {"color": "C0", "linestyle": "-", "marker": "o"},
        "GKR": {"color": "C2", "linestyle": "-", "marker": "^"},
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0))
    for axis, family, title in zip(
        axes[0], FAMILIES, ("Linear Fisher", "Full Fisher"), strict=True
    ):
        for method in ("Ground truth", *METHODS):
            axis.plot(
                theta,
                representative_curves[family][method],
                label=method,
                linewidth=2.2 if method == "Ground truth" else 1.8,
                markersize=5,
                markevery=5,
                **styles[method],
            )
        axis.set_yscale("log")
        axis.set_xlabel(r"$\theta$")
        axis.set_ylabel("Fisher information")
        axis.set_title(f"{title}, N={representative_n:,}")
        for spine in axis.spines.values():
            spine.set_linewidth(1.8)
        axis.tick_params(width=1.8)

    error_styles = {
        "Classical + LW": {"color": "C1", "marker": "s"},
        "Flow matching": {"color": "C0", "marker": "o"},
        "GKR": {"color": "C2", "marker": "^"},
    }
    for axis, family, title in zip(
        axes[1], FAMILIES, ("Linear Fisher error", "Full Fisher error"), strict=True
    ):
        for method in METHODS:
            selected = sorted(
                (
                    row
                    for row in summary_rows
                    if row["family"] == family and row["method"] == method
                ),
                key=lambda row: int(row["n_total"]),
            )
            means = np.asarray([row["mae_mean"] for row in selected], dtype=float)
            standard_deviations = np.asarray(
                [row["mae_sd"] for row in selected], dtype=float
            )
            axis.errorbar(
                n_list,
                means,
                yerr=standard_deviations,
                linewidth=2.0,
                markersize=7,
                capsize=3.0,
                capthick=1.5,
                elinewidth=1.5,
                **error_styles[method],
            )
        axis.set_yscale("log")
        axis.set_xticks(n_list)
        axis.set_xticklabels([f"{value / 1000:g}k" for value in n_list])
        axis.set_xlabel("Total samples")
        axis.set_ylabel("Mean absolute error")
        axis.set_title(title)
        for spine in axis.spines.values():
            spine.set_linewidth(1.8)
        axis.tick_params(width=1.8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), h_pad=1.8, w_pad=2.0)
    stem = output_dir / "continuous_fisher_xdim100_curves_and_error_vs_n_r5"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    case_root = args.case_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(args.repeat_seeds) < 2:
        raise ValueError("At least two repeat seeds are required for error bars.")

    metric_rows: list[dict[str, object]] = []
    reference_ground_truth: dict[str, np.ndarray] = {}
    for n_total in args.n_list:
        for repeat_idx, repeat_seed in enumerate(args.repeat_seeds):
            dataset_path, result_path, lw_path = case_paths(
                repo_root,
                case_root,
                n_total,
                repeat_idx,
                args.result_filename,
            )
            if not dataset_path.is_file() or not result_path.is_file():
                raise FileNotFoundError(
                    f"Missing dataset or result for N={n_total}, repeat={repeat_idx}"
                )
            if args.reuse_lw and lw_path.is_file():
                with np.load(lw_path, allow_pickle=False) as saved_lw:
                    lw = {key: saved_lw[key] for key in saved_lw.files}
            else:
                lw = fit_ledoit_wolf(dataset_path, result_path, lw_path)
            with np.load(result_path, allow_pickle=False) as result:
                curves = {
                    ("Classical + LW", "linear"): lw[
                        "classical_lw_linear_fisher"
                    ],
                    ("Classical + LW", "full"): lw["classical_lw_full_fisher"],
                    ("Flow matching", "linear"): result["flow_linear_fisher"],
                    ("Flow matching", "full"): result["flow_full_fisher"],
                    ("GKR", "linear"): result["gkr_linear_fisher"],
                    ("GKR", "full"): result["gkr_full_fisher"],
                }
                ground_truth = {
                    "linear": result["ground_truth_native_linear_fisher"],
                    "full": result["ground_truth_native_full_fisher"],
                }

            for family in FAMILIES:
                if family not in reference_ground_truth:
                    reference_ground_truth[family] = ground_truth[family].copy()
                if not np.allclose(
                    ground_truth[family],
                    reference_ground_truth[family],
                    rtol=1e-10,
                    atol=1e-10,
                ):
                    raise AssertionError(
                        f"Ground truth changed for N={n_total}, repeat={repeat_idx}"
                    )

            for (method, family), estimate in curves.items():
                estimate = np.asarray(estimate)
                if estimate.shape != ground_truth[family].shape or not np.all(
                    np.isfinite(estimate)
                ):
                    raise AssertionError(
                        f"Invalid {method} {family} curve at N={n_total}, "
                        f"repeat={repeat_idx}"
                    )
                error = estimate - ground_truth[family]
                metric_rows.append(
                    {
                        "n_total": n_total,
                        "repeat_idx": repeat_idx,
                        "repeat_seed": repeat_seed,
                        "method": method,
                        "family": family,
                        "mae": float(np.mean(np.abs(error))),
                        "rmse": float(np.sqrt(np.mean(error**2))),
                        "case_result_npz": str(result_path),
                        "classical_lw_npz": str(lw_path),
                    }
                )

    csv_path = output_dir / "continuous_fisher_xdim100_r5_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    summary_rows: list[dict[str, object]] = []
    for n_total in args.n_list:
        for family in FAMILIES:
            for method in METHODS:
                values = np.asarray(
                    [
                        row["mae"]
                        for row in metric_rows
                        if row["n_total"] == n_total
                        and row["family"] == family
                        and row["method"] == method
                    ],
                    dtype=float,
                )
                if values.size != len(args.repeat_seeds):
                    raise AssertionError((n_total, family, method, values.size))
                summary_rows.append(
                    {
                        "n_total": n_total,
                        "family": family,
                        "method": method,
                        "n_repeats": int(values.size),
                        "mae_mean": float(values.mean()),
                        "mae_sd": float(values.std(ddof=1)),
                        "mae_sem": float(values.std(ddof=1) / np.sqrt(values.size)),
                        "mae_values": values.tolist(),
                    }
                )

    summary = {
        "native_x_dim": 100,
        "population_seed": args.population_seed,
        "repeat_seeds": args.repeat_seeds,
        "n_repeats": len(args.repeat_seeds),
        "error_bars": "sample standard deviation across repeats",
        "rows": summary_rows,
    }
    summary_path = output_dir / "continuous_fisher_xdim100_r5_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    plot_summary(summary_rows, args.n_list, output_dir)
    representative_n = max(args.n_list)
    _, representative_result, representative_lw = case_paths(
        repo_root,
        case_root,
        representative_n,
        0,
        args.result_filename,
    )
    plot_curves_and_errors(
        summary_rows,
        args.n_list,
        representative_result,
        representative_lw,
        representative_n,
        output_dir,
    )

    print(f"Saved metrics: {csv_path}")
    print(f"Saved summary: {summary_path}")
    for family in FAMILIES:
        print(f"[{family}]")
        for n_total in args.n_list:
            values = []
            for method in METHODS:
                row = next(
                    row
                    for row in summary_rows
                    if row["n_total"] == n_total
                    and row["family"] == family
                    and row["method"] == method
                )
                values.append(
                    f"{method}={row['mae_mean']:.3f} +/- {row['mae_sd']:.3f}"
                )
            print(f"N={n_total}: " + "; ".join(values))


if __name__ == "__main__":
    main()
