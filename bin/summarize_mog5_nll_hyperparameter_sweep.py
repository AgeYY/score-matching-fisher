#!/usr/bin/env python3
"""Aggregate and plot the MoG5 CNF-NLL hyperparameter experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_ROOT = _REPO_ROOT / "data"


@dataclass(frozen=True)
class Experiment:
    label: str
    short_label: str
    errors_csv: Path
    n_total: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_DEFAULT_DATA_ROOT,
        help="Repository data directory containing the completed sweep outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Summary output directory. Defaults to <data-root>/mog5_nll_hparam_sweeps/summary.",
    )
    return parser


def _experiments(data_root: Path) -> list[Experiment]:
    sweep_root = data_root / "mog5_nll_hparam_sweeps"
    baseline = data_root / "mog5_native_xdim3_distance_sweeps" / "mog5_pr_distance_sweep_errors.csv"
    return [
        Experiment("baseline", "Baseline", baseline, 1000),
        Experiment(
            "batch256_lr1e-4_steps32",
            "B256\nLR 1e-4",
            sweep_root / "screen_bs256_lr1e4_s32" / "mog5_pr_distance_sweep_errors.csv",
            1000,
        ),
        Experiment(
            "batch256_lr3e-5_steps32",
            "B256\nLR 3e-5",
            sweep_root / "screen_bs256_lr3e5_s32" / "mog5_pr_distance_sweep_errors.csv",
            1000,
        ),
        Experiment(
            "batch256_lr3e-5_steps64",
            "B256\n64 steps",
            sweep_root / "screen_bs256_lr3e5_s64" / "mog5_pr_distance_sweep_errors.csv",
            1000,
        ),
        Experiment(
            "batch2048_lr3e-5_steps32_epochs500",
            "B2048\nLR 3e-5",
            sweep_root / "screen_bs2048_lr3e5_s32_e500" / "mog5_pr_distance_sweep_errors.csv",
            1000,
        ),
        Experiment("baseline", "Baseline", baseline, 3000),
        Experiment(
            "batch2048_lr3e-5_steps32_epochs500",
            "B2048\nLR 3e-5",
            sweep_root
            / "validate_n3000_bs2048_lr3e5_s32_e500"
            / "mog5_pr_distance_sweep_errors.csv",
            3000,
        ),
    ]


def _load_experiment(experiment: Experiment) -> pd.DataFrame:
    if not experiment.errors_csv.is_file():
        raise FileNotFoundError(experiment.errors_csv)
    frame = pd.read_csv(experiment.errors_csv)
    selected = frame[
        (frame["n_total"] == int(experiment.n_total))
        & (frame["estimator"] == "flow_matching_nll_finetuned")
    ].copy()
    if selected.empty:
        raise ValueError(
            f"No NLL-fine-tuned rows for N={experiment.n_total} in {experiment.errors_csv}"
        )
    grouped = (
        selected.groupby("metric", as_index=False)["abs_error"]
        .agg(mean_abs_error="mean", std_abs_error="std")
        .assign(
            config=experiment.label,
            config_label=experiment.short_label.replace("\n", " "),
            n_total=int(experiment.n_total),
            n_repeats=int(selected["repeat_idx"].nunique()),
            source_csv=str(experiment.errors_csv.resolve()),
        )
    )
    return grouped


def aggregate(data_root: Path) -> pd.DataFrame:
    summary = pd.concat([_load_experiment(exp) for exp in _experiments(data_root)], ignore_index=True)
    baseline = (
        summary[summary["config"] == "baseline"]
        .set_index(["n_total", "metric"])["mean_abs_error"]
        .rename("baseline_mean_abs_error")
    )
    summary = summary.join(baseline, on=["n_total", "metric"])
    summary["error_ratio_to_baseline"] = summary["mean_abs_error"] / summary["baseline_mean_abs_error"]
    return summary


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(width=1.8)


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))

    screen = summary[summary["n_total"] == 1000]
    config_order = [
        "baseline",
        "batch256_lr1e-4_steps32",
        "batch256_lr3e-5_steps32",
        "batch256_lr3e-5_steps64",
        "batch2048_lr3e-5_steps32_epochs500",
    ]
    screen_ratio = screen.groupby("config")["error_ratio_to_baseline"].mean().reindex(config_order)
    label_by_config = screen.drop_duplicates("config").set_index("config")["config_label"]
    colors = ["#666666", "#C44E52", "#DD8452", "#8172B3", "#4C72B0"]
    bars = axes[0].bar(
        np.arange(len(config_order)),
        screen_ratio.to_numpy(),
        color=colors,
        width=0.72,
    )
    axes[0].axhline(1.0, color="black", linewidth=1.4, linestyle="--")
    axes[0].set_xticks(np.arange(len(config_order)))
    axes[0].set_xticklabels([label_by_config.loc[key] for key in config_order], rotation=35, ha="right")
    axes[0].set_ylabel("Mean error ratio")
    axes[0].set_title("Screen at N = 1000")
    axes[0].set_ylim(0.90, max(1.15, float(screen_ratio.max()) + 0.04))
    for bar, value in zip(bars, screen_ratio.to_numpy(), strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            float(value) + 0.008,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    selected = summary[summary["config"] == "batch2048_lr3e-5_steps32_epochs500"]
    metric_order = ["correlation", "cosine", "squared_euclidean", "mahalanobis_sq", "symmetric_kl"]
    metric_labels = ["Corr.", "Cosine", "Euclid.$^2$", "Mahal.$^2$", "Jeffreys"]
    x = np.arange(len(metric_order))
    selected_curves: dict[int, np.ndarray] = {}
    for n_total, marker, color in ((1000, "o", "#4C72B0"), (3000, "s", "#55A868")):
        values = (
            selected[selected["n_total"] == n_total]
            .set_index("metric")["error_ratio_to_baseline"]
            .reindex(metric_order)
        )
        selected_curves[n_total] = values.to_numpy()
        axes[1].plot(x, values.to_numpy(), marker=marker, linewidth=2.0, markersize=7, color=color)
    axes[1].axhline(1.0, color="black", linewidth=1.4, linestyle="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metric_labels, rotation=35, ha="right")
    axes[1].set_ylabel("Selected / baseline error")
    axes[1].set_title("Selected configuration")
    axes[1].text(4.12, selected_curves[1000][-1] + 0.006, "N = 1000", color="#4C72B0", fontsize=12)
    axes[1].text(4.12, selected_curves[3000][-1] - 0.006, "N = 3000", color="#55A868", fontsize=12)
    axes[1].set_xlim(-0.2, 5.0)
    axes[1].set_ylim(0.75, 1.10)

    for axis in axes:
        _style_axis(axis)
    fig.tight_layout(w_pad=1.3)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "mog5_nll_hyperparameter_summary.png"
    svg_path = output_dir / "mog5_nll_hyperparameter_summary.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root).expanduser()
    output_dir = (
        data_root / "mog5_nll_hparam_sweeps" / "summary"
        if args.output_dir is None
        else Path(args.output_dir).expanduser()
    )
    summary = aggregate(data_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "mog5_nll_hyperparameter_summary.csv"
    summary.to_csv(csv_path, index=False)
    png_path, svg_path = plot_summary(summary, output_dir)
    print(f"summary_csv: {csv_path.resolve()}")
    print(f"figure_png: {png_path.resolve()}")
    print(f"figure_svg: {svg_path.resolve()}")


if __name__ == "__main__":
    main()
