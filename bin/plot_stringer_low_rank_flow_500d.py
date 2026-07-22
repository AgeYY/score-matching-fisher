#!/usr/bin/env python3
"""Compare rank-constrained and dense 500D Stringer Fisher estimators."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--low-rank-summary", type=Path, required=True)
    parser.add_argument("--dense-summary", type=Path, required=True)
    parser.add_argument("--gkr-summary", type=Path, required=True)
    parser.add_argument("--low-rank-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _method_values(path: Path, method: str) -> np.ndarray:
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    selected = sorted(
        (
            (int(row["seed"]), float(row["mean_achieved_fisher"]))
            for row in rows
            if str(row["method"]) == method
        ),
        key=lambda pair: pair[0],
    )
    if not selected:
        raise ValueError(f"No {method!r} rows found in {path}.")
    return np.asarray([value for _, value in selected], dtype=np.float64)


def _loss_histories(path: Path) -> dict[int, dict[str, list[float]]]:
    seed_pattern = re.compile(r"dataset=stringer seed=(\d+)")
    loss_pattern = re.compile(
        r"condition_affine_low_rank\s+(\d+)/\d+\]\s+"
        r"train=([0-9.eE+-]+)\s+val=([0-9.eE+-]+)\s+"
        r"val_smooth=([0-9.eE+-]+)"
    )
    histories: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"epoch": [], "train": [], "validation": []}
    )
    seed: int | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        seed_match = seed_pattern.search(line)
        if seed_match:
            seed = int(seed_match.group(1))
            continue
        loss_match = loss_pattern.search(line)
        if seed is None or loss_match is None:
            continue
        epoch, train, _, validation_smooth = loss_match.groups()
        histories[seed]["epoch"].append(int(epoch))
        histories[seed]["train"].append(float(train))
        histories[seed]["validation"].append(float(validation_smooth))
    if not histories:
        raise ValueError(f"No low-rank FM loss histories found in {path}.")
    return dict(histories)


def main() -> None:
    args = parse_args()
    low_rank = _method_values(args.low_rank_summary, "Flow matching")
    dense = _method_values(args.dense_summary, "Flow matching")
    gkr = _method_values(args.gkr_summary, "GKR")
    histories = _loss_histories(args.low_rank_log)

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 13,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), constrained_layout=True)

    methods = (
        ("Low-rank\nFM", low_rank, "C4"),
        ("Dense\nFM", dense, "C0"),
        ("GKR", gkr, "C2"),
    )
    for index, (label, values, color) in enumerate(methods):
        ci = 1.96 * np.std(values, ddof=1) / np.sqrt(values.size)
        axes[0].errorbar(
            index,
            np.mean(values),
            yerr=ci,
            color=color,
            marker="o",
            markersize=7,
            linewidth=2.2,
            capsize=4,
        )
    axes[0].set_xticks(range(len(methods)), [item[0] for item in methods])
    axes[0].set_ylabel("Held-out achieved information")
    axes[0].set_title("500D whitened PCA")

    for history in histories.values():
        axes[1].plot(
            history["epoch"], history["train"], color="C0", alpha=0.18, linewidth=1.0
        )
        axes[1].plot(
            history["epoch"],
            history["validation"],
            color="C1",
            alpha=0.18,
            linewidth=1.0,
        )
    common_epochs = sorted(
        set.intersection(*(set(history["epoch"]) for history in histories.values()))
    )
    for key, label, color in (
        ("train", "Training", "C0"),
        ("validation", "Validation", "C1"),
    ):
        mean = [
            np.mean(
                [
                    history[key][history["epoch"].index(epoch)]
                    for history in histories.values()
                ]
            )
            for epoch in common_epochs
        ]
        axes[1].plot(common_epochs, mean, color=color, linewidth=2.4, label=label)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Flow-matching loss")
    axes[1].set_title("Rank-82 training")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.set_axisbelow(True)
        axis.yaxis.grid(True, color="0.88", linewidth=0.8)
        axis.xaxis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(1.8)
        axis.spines["bottom"].set_linewidth(1.8)
        axis.tick_params(width=1.8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "stringer_low_rank_flow_500d_rank82_comparison"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    print(f"Saved: {stem.with_suffix('.png').resolve()}")
    print(f"Saved: {stem.with_suffix('.svg').resolve()}")


if __name__ == "__main__":
    main()
