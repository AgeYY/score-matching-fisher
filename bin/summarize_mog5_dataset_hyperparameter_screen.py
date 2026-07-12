#!/usr/bin/env python3
"""Summarize the MoG5 dataset-hyperparameter screen and seed validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ROOT = _REPO_ROOT / "data" / "mog5_dataset_hparam_screen"
_ROBUST_GROUPS = {
    "baseline_seed7": "baseline",
    "seed19": "baseline",
    "seed31": "baseline",
    "cov050": "cov050",
    "cov050_seed19": "cov050",
    "cov050_seed31": "cov050",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _load_stage(stage_dir: Path, *, stage: str, n_total: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(stage_dir.glob("*/mog5_pr_distance_sweep_errors.csv")):
        config_dir = csv_path.parent
        summary_path = config_dir / "mog5_pr_distance_sweep_summary.json"
        if not summary_path.is_file():
            continue
        config = json.loads(summary_path.read_text(encoding="utf-8"))["config"]
        frame = pd.read_csv(csv_path)
        frame = frame[frame["n_total"] == int(n_total)].copy()
        frame["stage"] = str(stage)
        frame["config_name"] = config_dir.name
        frame["component_seed"] = int(config["seed"])
        frame["obs_noise_scale"] = float(config["dataset_obs_noise_scale"])
        frame["cov_theta_amp_scale"] = float(config["dataset_cov_theta_amp_scale"])
        mean_min_dist = config.get("dataset_mog_mean_min_dist")
        frame["mean_min_dist"] = np.nan if mean_min_dist is None else float(mean_min_dist)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No completed sweep CSVs found under {stage_dir}")
    return pd.concat(frames, ignore_index=True)


def load_results(root: Path) -> pd.DataFrame:
    return pd.concat(
        [
            _load_stage(root / "screen", stage="screen_n1000", n_total=1000),
            _load_stage(root / "validate_n3000", stage="validate_n3000", n_total=3000),
        ],
        ignore_index=True,
    )


def summarize_all(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(
            [
                "stage",
                "n_total",
                "config_name",
                "component_seed",
                "obs_noise_scale",
                "cov_theta_amp_scale",
                "mean_min_dist",
                "metric",
                "estimator",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            mean_abs_error=("abs_error", "mean"),
            mean_relative_error=("rel_error", "mean"),
            median_relative_error=("rel_error", "median"),
            n_pairs=("rel_error", "size"),
        )
    )


def summarize_robust(results: pd.DataFrame) -> pd.DataFrame:
    selected = results[results["config_name"].isin(_ROBUST_GROUPS)].copy()
    selected["dataset_setting"] = selected["config_name"].map(_ROBUST_GROUPS)
    return (
        selected.groupby(
            ["stage", "n_total", "dataset_setting", "metric", "estimator"],
            as_index=False,
        )
        .agg(
            mean_abs_error=("abs_error", "mean"),
            mean_relative_error=("rel_error", "mean"),
            std_relative_error=("rel_error", "std"),
            n_component_seeds=("component_seed", "nunique"),
            n_pair_estimates=("rel_error", "size"),
        )
    )


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(width=1.8)


def plot_summary(robust: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 16,
            "legend.fontsize": 13,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    estimators = ("classical", "flow_matching", "flow_matching_nll_finetuned")
    labels = {"classical": "Classical", "flow_matching": "FM", "flow_matching_nll_finetuned": "FM + NLL"}
    colors = {"classical": "#DD8452", "flow_matching": "#4C72B0", "flow_matching_nll_finetuned": "#55A868"}

    overall = robust.groupby(["n_total", "dataset_setting", "estimator"])["mean_relative_error"].mean()
    x_labels = ("N1k\nBase", "N1k\nCov 0.5", "N3k\nBase", "N3k\nCov 0.5")
    keys = ((1000, "baseline"), (1000, "cov050"), (3000, "baseline"), (3000, "cov050"))
    x = np.arange(len(keys), dtype=np.float64)
    offsets = (-0.22, 0.0, 0.22)
    for estimator, offset in zip(estimators, offsets, strict=True):
        values = [float(overall.loc[n_total, setting, estimator]) for n_total, setting in keys]
        axes[0].bar(x + offset, values, width=0.20, color=colors[estimator], label=labels[estimator])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels)
    axes[0].set_ylabel("Mean relative error")
    axes[0].set_title("Seed-averaged performance")
    axes[0].set_ylim(0.0, 0.19)
    axes[0].legend(frameon=False, loc="upper right")

    validation = robust[robust["n_total"] == 3000]
    metric_order = ("correlation", "cosine", "squared_euclidean", "mahalanobis_sq", "symmetric_kl")
    metric_labels = ("Corr.", "Cosine", "Euclid.$^2$", "Mahal.$^2$", "Jeffreys")
    pivot = validation.pivot_table(
        index=["metric", "estimator"],
        columns="dataset_setting",
        values="mean_relative_error",
    )
    mx = np.arange(len(metric_order), dtype=np.float64)
    for estimator, marker in zip(estimators, ("^", "o", "s"), strict=True):
        ratios = [float(pivot.loc[(metric, estimator), "cov050"] / pivot.loc[(metric, estimator), "baseline"]) for metric in metric_order]
        axes[1].plot(mx, ratios, marker=marker, linewidth=2.0, markersize=7, color=colors[estimator], label=labels[estimator])
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.4)
    axes[1].set_xticks(mx)
    axes[1].set_xticklabels(metric_labels, rotation=35, ha="right")
    axes[1].set_ylabel("Cov 0.5 / baseline error")
    axes[1].set_title("Per metric at N = 3000")
    axes[1].set_ylim(0.72, 1.14)

    for axis in axes:
        _style_axis(axis)
    fig.tight_layout(w_pad=1.3)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "mog5_dataset_hyperparameter_summary.png"
    svg_path = output_dir / "mog5_dataset_hyperparameter_summary.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).expanduser()
    output_dir = root / "summary" if args.output_dir is None else Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(root)
    all_summary = summarize_all(results)
    robust_summary = summarize_robust(results)
    all_path = output_dir / "all_config_metric_summary.csv"
    robust_path = output_dir / "seed_averaged_baseline_vs_cov050.csv"
    all_summary.to_csv(all_path, index=False)
    robust_summary.to_csv(robust_path, index=False)
    png_path, svg_path = plot_summary(robust_summary, output_dir)
    print(f"all_config_summary: {all_path.resolve()}")
    print(f"robust_summary: {robust_path.resolve()}")
    print(f"figure_png: {png_path.resolve()}")
    print(f"figure_svg: {svg_path.resolve()}")


if __name__ == "__main__":
    main()
