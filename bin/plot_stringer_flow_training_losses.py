#!/usr/bin/env python3
"""Plot aggregated flow-matching loss histories from a Stringer sweep."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SUBSET_PATTERN = re.compile(r"/subset_a/n_(\d+)/")
GROUP_ORDER = ("Full half", "200", "650", "1100", "1550", "2000")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Stringer session-identification run containing half_curves caches.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=None,
        help="Output path without extension (default: <run-dir>/stringer_flow_training_losses).",
    )
    return parser.parse_args()


def _group_label(path: Path) -> str:
    match = SUBSET_PATTERN.search(path.as_posix())
    return str(int(match.group(1))) if match else "Full half"


def load_histories(run_dir: Path) -> dict[str, list[dict[str, np.ndarray | int]]]:
    groups: dict[str, list[dict[str, np.ndarray | int]]] = defaultdict(list)
    curve_paths = sorted(run_dir.glob("**/half_curves/*.npz"))
    if not curve_paths:
        raise FileNotFoundError(f"No half-curve caches found under {run_dir}")

    for curve_path in curve_paths:
        with np.load(curve_path, allow_pickle=False) as curve_data:
            flow_path = Path(str(curve_data["flow_npz_path"].item()))
        if not flow_path.is_file():
            raise FileNotFoundError(f"Missing flow cache referenced by {curve_path}: {flow_path}")
        with np.load(flow_path, allow_pickle=True) as flow_data:
            groups[_group_label(curve_path)].append(
                {
                    "train": np.asarray(flow_data["train_losses"], dtype=np.float64),
                    "validation": np.asarray(
                        flow_data["val_monitor_losses"], dtype=np.float64
                    ),
                    "best_epoch": int(np.asarray(flow_data["best_epoch"]).item()),
                }
            )
    return dict(groups)


def _common_summary(
    histories: list[dict[str, np.ndarray | int]], key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common_length = min(np.asarray(history[key]).size for history in histories)
    values = np.stack(
        [np.asarray(history[key])[:common_length] for history in histories], axis=0
    )
    return (
        np.median(values, axis=0),
        np.quantile(values, 0.25, axis=0),
        np.quantile(values, 0.75, axis=0),
    )


def plot_histories(
    groups: dict[str, list[dict[str, np.ndarray | int]]], output_stem: Path
) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 13,
            "axes.grid": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), sharex=True)

    labels = [label for label in GROUP_ORDER if label in groups]
    for color_index, label in enumerate(labels):
        histories = groups[label]
        color = f"C{color_index}"
        display_label = label if label == "Full half" else f"N = {label}"
        for axis, key in zip(axes, ("train", "validation"), strict=True):
            median, lower, upper = _common_summary(histories, key)
            epochs = np.arange(1, median.size + 1)
            axis.fill_between(epochs, lower, upper, color=color, alpha=0.14, linewidth=0)
            axis.plot(epochs, median, color=color, linewidth=2.0, label=display_label)

        selected_epoch = int(
            round(np.median([int(history["best_epoch"]) for history in histories]))
        )
        validation_median, _, _ = _common_summary(histories, "validation")
        marker_epoch = min(max(selected_epoch, 1), validation_median.size)
        axes[1].scatter(
            marker_epoch,
            validation_median[marker_epoch - 1],
            color=color,
            edgecolor="white",
            linewidth=0.8,
            s=38,
            zorder=4,
        )

    axes[0].set_title("Training")
    axes[1].set_title("Fixed validation (10-path EMA)")
    axes[0].set_ylabel("Flow-matching loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(1.8)
        axis.spines["bottom"].set_linewidth(1.8)
        axis.tick_params(width=1.8)
    axes[1].legend(frameon=False, ncol=2, columnspacing=0.9, handlelength=1.5)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_stem = (
        args.output_stem.resolve()
        if args.output_stem is not None
        else run_dir / "stringer_flow_training_losses"
    )
    groups = load_histories(run_dir)
    plot_histories(groups, output_stem)
    for label in GROUP_ORDER:
        if label not in groups:
            continue
        histories = groups[label]
        best_epochs = np.asarray(
            [int(history["best_epoch"]) for history in histories], dtype=np.int64
        )
        print(
            f"{label}: fits={len(histories)}, "
            f"best_epoch_median={np.median(best_epochs):.1f}, "
            f"range=[{best_epochs.min()}, {best_epochs.max()}]"
        )
    print(f"Saved: {output_stem.with_suffix('.png')}")
    print(f"Saved: {output_stem.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
