#!/usr/bin/env python3
"""Combine toy and Stringer Fisher-validation pilot results into two figures."""

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

from global_setting import DATA_DIR

DATASET_ORDER = ["randamp_gaussian_sqrtd", "cosine_gmm", "Stringer"]
DATASET_LABELS = {
    "randamp_gaussian_sqrtd": "Gaussian toy",
    "cosine_gmm": "Gaussian-mixture toy",
    "Stringer": "Stringer",
}
METHOD_ORDER = ["Flow matching", "GKR", "Oracle"]
COLORS = {"Flow matching": "C0", "GKR": "C2", "Oracle": "0.25"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    root = Path(DATA_DIR) / "fisher_validation_pilot"
    parser.add_argument("--toy-summary", type=Path, default=root / "toy" / "toy_fisher_validation_summary.json")
    parser.add_argument("--stringer-summary", type=Path, default=root / "stringer" / "stringer_fisher_validation_summary.json")
    parser.add_argument("--output-dir", type=Path, default=root / "figures")
    return parser.parse_args()


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.8)
    axis.spines["bottom"].set_linewidth(1.8)
    axis.tick_params(width=1.8)


def _set_data_driven_ylim(axis: plt.Axes, values: list[float], *, bounded_unit: bool = False) -> None:
    data = np.asarray(values, dtype=np.float64)
    low = float(np.min(data))
    high = float(np.max(data))
    span = high - low
    if span <= 1e-12:
        span = max(abs(low), 1.0) * 0.05
    margin = 0.15 * span
    lower = low - margin
    upper = high + margin
    if bounded_unit:
        lower = max(0.0, lower)
        upper = min(1.0, upper)
    axis.set_ylim(lower, upper)


def _heldout_figure(rows: list[dict[str, object]], output_dir: Path) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    metrics = [("mean_achieved_fisher", "Held-out achieved information"), ("mean_auc", "Held-out ROC AUC")]
    for column, dataset in enumerate(DATASET_ORDER):
        for row_index, (key, ylabel) in enumerate(metrics):
            axis = axes[row_index, column]
            methods = [method for method in METHOD_ORDER if any(item["dataset"] == dataset and item["method"] == method for item in rows)]
            panel_values: list[float] = []
            for method_index, method in enumerate(methods):
                values = np.asarray(
                    [float(item[key]) for item in rows if item["dataset"] == dataset and item["method"] == method]
                )
                panel_values.extend(values.tolist())
                axis.bar(method_index, np.mean(values), color=COLORS[method], width=0.68, alpha=0.82)
                axis.scatter(
                    np.full(values.size, method_index) + np.linspace(-0.07, 0.07, values.size),
                    values,
                    color="black",
                    s=24,
                    zorder=3,
                )
            axis.set_xticks(np.arange(len(methods)), [method.replace(" matching", "") for method in methods], rotation=20)
            if column == 0:
                axis.set_ylabel(ylabel)
            if row_index == 0:
                axis.set_title(DATASET_LABELS[dataset])
            _set_data_driven_ylim(axis, panel_values, bounded_unit=key == "mean_auc")
            _style_axis(axis)
    png = output_dir / "heldout_local_decoding.png"
    svg = output_dir / "heldout_local_decoding.svg"
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def _calibration_figure(rows: list[dict[str, object]], output_dir: Path) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), constrained_layout=True)
    for column, dataset in enumerate(DATASET_ORDER):
        axis = axes[column]
        dataset_rows = [item for item in rows if item["dataset"] == dataset]
        if not dataset_rows:
            axis.text(0.5, 0.5, "Not run", ha="center", va="center", transform=axis.transAxes)
            axis.set_title(DATASET_LABELS[dataset])
            _style_axis(axis)
            continue
        truth_values: list[np.ndarray] = []
        estimated_values: list[np.ndarray] = []
        for method in METHOD_ORDER[:2]:
            selected = [item for item in dataset_rows if item["method"] == method]
            if not selected:
                continue
            truth = np.concatenate([np.asarray(item["target"], dtype=np.float64) for item in selected])
            estimate = np.concatenate([np.asarray(item["estimated"], dtype=np.float64) for item in selected])
            truth_values.append(truth)
            estimated_values.append(estimate)
            axis.scatter(
                truth,
                estimate,
                s=15,
                alpha=0.48,
                color=COLORS[method],
                edgecolors="none",
                label=method,
            )
        x_low = min(float(np.min(value)) for value in truth_values)
        x_high = max(float(np.max(value)) for value in truth_values)
        x_margin = 0.05 * max(x_high - x_low, 1e-8)
        y_low = min(float(np.min(value)) for value in estimated_values + truth_values)
        y_high = max(float(np.max(value)) for value in estimated_values + truth_values)
        y_margin = 0.05 * max(y_high - y_low, 1e-8)
        line_low = x_low - x_margin
        line_high = x_high + x_margin
        axis.plot([line_low, line_high], [line_low, line_high], color="black", linestyle="--", linewidth=1.8)
        axis.set_xlim(line_low, line_high)
        axis.set_ylim(y_low - y_margin, y_high + y_margin)
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Known Fisher increment")
        if column == 0:
            axis.set_ylabel("Estimated increment")
            axis.legend(frameon=False)
        _style_axis(axis)
    png = output_dir / "known_fisher_increment_calibration.png"
    svg = output_dir / "known_fisher_increment_calibration.svg"
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def main() -> None:
    args = parse_args()
    toy = json.loads(args.toy_summary.expanduser().resolve().read_text(encoding="utf-8"))
    stringer = json.loads(args.stringer_summary.expanduser().resolve().read_text(encoding="utf-8"))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 16,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    heldout_png, heldout_svg = _heldout_figure(toy["heldout"] + stringer["heldout"], output_dir)
    calibration_png, calibration_svg = _calibration_figure(
        toy["calibration"] + stringer["calibration"], output_dir
    )
    summary_path = output_dir / "figure_manifest.json"
    summary_path.write_text(
        json.dumps(
            {
                "heldout_png": str(heldout_png),
                "heldout_svg": str(heldout_svg),
                "calibration_png": str(calibration_png),
                "calibration_svg": str(calibration_svg),
                "toy_summary": str(args.toy_summary.expanduser().resolve()),
                "stringer_summary": str(args.stringer_summary.expanduser().resolve()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {heldout_svg}", flush=True)
    print(f"Saved: {calibration_svg}", flush=True)


if __name__ == "__main__":
    main()
