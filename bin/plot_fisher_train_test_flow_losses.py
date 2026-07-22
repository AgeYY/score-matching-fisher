#!/usr/bin/env python3
"""Plot FM training diagnostics for the fixed-validation train/test sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from global_setting import DATA_DIR


DATASETS = (
    ("Gaussian toy", "randamp_gaussian_sqrtd"),
    ("Gaussian-mixture toy", "cosine_gmm"),
    ("Stringer", "stringer"),
)
EMA_ALPHA = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(DATA_DIR)
        / "fisher_validation_directions"
        / "cache_train_test_allocation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "fisher_validation_directions" / "figures",
    )
    return parser.parse_args()


def ema(values: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    smoothed = np.empty_like(values, dtype=np.float64)
    smoothed[0] = values[0]
    for index in range(1, values.size):
        smoothed[index] = alpha * values[index] + (1.0 - alpha) * smoothed[index - 1]
    return smoothed


def padded_mean(curves: list[np.ndarray]) -> np.ndarray:
    width = max(curve.size for curve in curves)
    values = np.full((len(curves), width), np.nan, dtype=np.float64)
    for row, curve in enumerate(curves):
        values[row, : curve.size] = curve
    return np.nanmean(values, axis=0)


def load_run(case_dir: Path) -> dict[str, Any]:
    metadata = json.loads((case_dir / "fit" / "metadata.json").read_text())
    evaluation = json.loads((case_dir / "evaluation.json").read_text())
    with np.load(case_dir / "fit" / "estimates.npz") as saved:
        train = np.asarray(saved["flow_train_loss"], dtype=np.float64)
        validation = np.asarray(saved["flow_validation_loss"], dtype=np.float64)
    flow = metadata["flow_training"]
    return {
        "seed": int(metadata["seed"]),
        "test_fraction": float(evaluation["test_fraction"]),
        "train": train,
        "train_ema": ema(train),
        "validation": validation,
        "validation_ema": ema(validation),
        "selected_epoch": int(flow["selected_epoch"]),
        "stopped_epoch": int(flow["stopped_epoch"]),
        "best_validation_loss": float(flow["best_val_loss"]),
    }


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.8)
    axis.spines["bottom"].set_linewidth(1.8)
    axis.tick_params(width=1.8)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 15,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    available_datasets = [
        (label, token)
        for label, token in DATASETS
        if any((input_dir / token).glob("**/evaluation.json"))
    ]
    if not available_datasets:
        raise FileNotFoundError(f"No cached evaluations found under {input_dir}")
    test_fractions = sorted(
        {
            float(json.loads(path.read_text())["test_fraction"])
            for _, token in available_datasets
            for path in (input_dir / token).glob("**/evaluation.json")
        }
    )
    figure, axes = plt.subplots(
        len(available_datasets),
        len(test_fractions),
        figsize=(4.0 * len(test_fractions), 3.5 * len(available_datasets)),
        constrained_layout=True,
        sharey="row",
        squeeze=False,
    )
    records: list[dict[str, Any]] = []

    for row, (dataset_label, dataset_token) in enumerate(available_datasets):
        runs = [load_run(path.parent) for path in sorted((input_dir / dataset_token).glob("**/evaluation.json"))]
        for column, test_fraction in enumerate(test_fractions):
            axis = axes[row, column]
            selected = [run for run in runs if np.isclose(run["test_fraction"], test_fraction)]
            if not selected:
                raise FileNotFoundError(f"No runs for {dataset_token}, test_fraction={test_fraction}")

            train_curves = [run["train_ema"] for run in selected]
            validation_curves = [run["validation_ema"] for run in selected]
            for run in selected:
                train_epoch = np.arange(1, run["train_ema"].size + 1)
                validation_epoch = np.arange(1, run["validation_ema"].size + 1)
                axis.plot(train_epoch, run["train_ema"], color="0.55", linewidth=0.8, alpha=0.2)
                axis.plot(validation_epoch, run["validation_ema"], color="C0", linewidth=0.8, alpha=0.2)
                checkpoint = run["selected_epoch"]
                axis.scatter(
                    checkpoint,
                    run["validation_ema"][checkpoint - 1],
                    marker="x",
                    color="C3",
                    linewidth=1.5,
                    s=28,
                    zorder=4,
                )
                records.append(
                    {
                        "dataset": dataset_label,
                        "test_fraction": test_fraction,
                        "seed": run["seed"],
                        "selected_epoch": checkpoint,
                        "stopped_epoch": run["stopped_epoch"],
                        "best_validation_loss": run["best_validation_loss"],
                    }
                )

            mean_train = padded_mean(train_curves)
            mean_validation = padded_mean(validation_curves)
            axis.plot(np.arange(1, mean_train.size + 1), mean_train, color="0.25", linewidth=2.0)
            axis.plot(np.arange(1, mean_validation.size + 1), mean_validation, color="C0", linewidth=2.2)
            axis.set_title(f"Test {int(round(100 * test_fraction))}%")
            if column == 0:
                axis.set_ylabel(f"{dataset_label}\nFM loss")
            if row == len(available_datasets) - 1:
                axis.set_xlabel("Epoch")
            style_axis(axis)

    handles = (
        Line2D([0], [0], color="0.25", linewidth=2.0, label="Training EMA"),
        Line2D([0], [0], color="C0", linewidth=2.2, label="Validation EMA"),
        Line2D([0], [0], color="C3", marker="x", linestyle="none", markersize=7, label="Selected checkpoints"),
    )
    axes[0, 0].legend(handles=handles, frameon=False)

    png_path = output_dir / "train_test_allocation_flow_losses.png"
    svg_path = output_dir / "train_test_allocation_flow_losses.svg"
    json_path = output_dir / "train_test_allocation_flow_losses.json"
    figure.savefig(png_path, dpi=300)
    figure.savefig(svg_path)
    plt.close(figure)
    json_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
