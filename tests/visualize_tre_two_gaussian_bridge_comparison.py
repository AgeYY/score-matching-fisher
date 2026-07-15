#!/usr/bin/env python3
"""Compare classical and TRE bridge-count calibration on the same Gaussian test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import DATA_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--one-bridge-dir",
        type=Path,
        default=Path(DATA_DIR) / "tre_two_gaussian_log_ratio_m1_e10000",
    )
    parser.add_argument(
        "--four-bridge-dir",
        type=Path,
        default=Path(DATA_DIR) / "tre_two_gaussian_log_ratio_m4_e3000",
    )
    parser.add_argument(
        "--eight-bridge-dir",
        type=Path,
        default=Path(DATA_DIR) / "tre_two_gaussian_log_ratio",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "tre_two_gaussian_bridge_comparison",
    )
    parser.add_argument("--max-points", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument(
        "--layout",
        choices=("overlay", "panels"),
        default="overlay",
        help="Overlay every method in one calibration panel or use separate panels.",
    )
    return parser


def _load_run(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    run_dir = path.expanduser().resolve()
    metrics_path = run_dir / "metrics.json"
    arrays_path = run_dir / "tre_two_gaussian_log_ratio.npz"
    if not metrics_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError(f"Missing TRE Gaussian artifacts under {run_dir}.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    return metrics, arrays


def _verify_matched_runs(runs: list[tuple[dict[str, object], dict[str, np.ndarray]]]) -> None:
    reference_metrics, reference = runs[0]
    for metrics, arrays in runs[1:]:
        for key in ("x0_test", "x1_test", "true_log_ratio", "classical_log_ratio"):
            np.testing.assert_allclose(arrays[key], reference[key], rtol=0.0, atol=0.0)
        if metrics["problem"] != reference_metrics["problem"]:
            raise ValueError("Bridge runs do not use the same Gaussian problem.")
        if metrics["sample_sizes_per_gaussian"] != reference_metrics["sample_sizes_per_gaussian"]:
            raise ValueError("Bridge runs do not use the same sample sizes.")


def _calibration_inputs(
    runs: list[tuple[str, dict[str, object], dict[str, np.ndarray]]],
    *,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], float]:
    truth = np.asarray(runs[0][2]["true_log_ratio"], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    count = min(int(max_points), int(truth.size))
    indices = rng.choice(truth.size, size=count, replace=False)
    estimates = [
        np.asarray(
            arrays["classical_log_ratio" if label == "Classical" else "tre_log_ratio"],
            dtype=np.float64,
        )
        for label, _, arrays in runs
    ]
    bound = 1.03 * float(
        max(
            np.max(np.abs(truth)),
            *(np.max(np.abs(estimate)) for estimate in estimates),
        )
    )
    return truth, indices, estimates, bound


def _style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
    ax.tick_params(width=1.8, length=5)


def _plot_panels(
    *,
    runs: list[tuple[str, dict[str, object], dict[str, np.ndarray]]],
    output_dir: Path,
    max_points: int,
    seed: int,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    truth, indices, estimates, bound = _calibration_inputs(
        runs, max_points=max_points, seed=seed
    )

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.5), sharex=True, sharey=True)
    colors = ("C1", "C0", "C2", "C3")
    for panel_index, (ax, (label, metrics, arrays), estimate, color) in enumerate(
        zip(axes, runs, estimates, colors, strict=True)
    ):
        metric_key = "classical" if label == "Classical" else "tre"
        point_metrics = metrics[metric_key]
        ax.scatter(
            truth[indices],
            estimate[indices],
            s=11,
            alpha=0.28,
            linewidths=0.0,
            color=color,
        )
        ax.plot([-bound, bound], [-bound, bound], color="black", linestyle="--", linewidth=1.8, label="Exact")
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(label)
        ax.set_xlabel("Analytic log ratio")
        if panel_index == 0:
            ax.set_ylabel("Estimated log ratio")
            ax.legend(frameon=False, loc="upper left")
        ax.text(
            0.96,
            0.05,
            f"RMSE {float(point_metrics['rmse']):.2f}\nJ {float(point_metrics['jeffreys']):.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=14,
        )
        _style_axis(ax)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "tre_two_gaussian_bridge_comparison.png"
    svg_path = output_dir / "tre_two_gaussian_bridge_comparison.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def _plot_overlay(
    *,
    runs: list[tuple[str, dict[str, object], dict[str, np.ndarray]]],
    output_dir: Path,
    max_points: int,
    seed: int,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 11,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    truth, indices, estimates, bound = _calibration_inputs(
        runs, max_points=max_points, seed=seed
    )
    colors = ("C1", "C0", "C2", "C3")
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    for (label, metrics, _), estimate, color in zip(runs, estimates, colors, strict=True):
        metric_key = "classical" if label == "Classical" else "tre"
        point_metrics = metrics[metric_key]
        ax.scatter(
            truth[indices],
            estimate[indices],
            s=8,
            alpha=0.13,
            linewidths=0.0,
            color=color,
        )
        slope, intercept = np.polyfit(truth, estimate, deg=1)
        line_x = np.array([-bound, bound], dtype=np.float64)
        ax.plot(
            line_x,
            slope * line_x + intercept,
            color=color,
            linewidth=2.4,
            label=(
                f"{label}: RMSE {float(point_metrics['rmse']):.2f}, "
                f"J {float(point_metrics['jeffreys']):.1f}"
            ),
        )
    ax.plot(
        [-bound, bound],
        [-bound, bound],
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="Exact",
    )
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Analytic log ratio")
    ax.set_ylabel("Estimated log ratio")
    ax.legend(frameon=False, loc="upper left")
    _style_axis(ax)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "tre_two_gaussian_bridge_comparison.png"
    svg_path = output_dir / "tre_two_gaussian_bridge_comparison.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    args = build_parser().parse_args()
    one = _load_run(args.one_bridge_dir)
    four = _load_run(args.four_bridge_dir)
    eight = _load_run(args.eight_bridge_dir)
    _verify_matched_runs([one, four, eight])
    classical_metrics = eight[0]
    classical_arrays = eight[1]
    runs = [
        ("Classical", classical_metrics, classical_arrays),
        ("TRE, 1 bridge", one[0], one[1]),
        ("TRE, 4 bridges", four[0], four[1]),
        ("TRE, 8 bridges", eight[0], eight[1]),
    ]
    output_dir = args.output_dir.expanduser().resolve()
    plotter = _plot_overlay if args.layout == "overlay" else _plot_panels
    png_path, svg_path = plotter(
        runs=runs,
        output_dir=output_dir,
        max_points=int(args.max_points),
        seed=int(args.seed),
    )
    summary = {
        label: {
            "rmse": float(metrics["classical" if label == "Classical" else "tre"]["rmse"]),
            "correlation": float(
                metrics["classical" if label == "Classical" else "tre"]["correlation"]
            ),
            "jeffreys": float(metrics["classical" if label == "Classical" else "tre"]["jeffreys"]),
            "best_epoch": (
                None if label == "Classical" else int(metrics["tre_training"]["best_epoch"])
            ),
            "stopped_epoch": (
                None if label == "Classical" else int(metrics["tre_training"]["stopped_epoch"])
            ),
        }
        for label, metrics, _ in runs
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved figure: {png_path}")
    print(f"Saved vector figure: {svg_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
