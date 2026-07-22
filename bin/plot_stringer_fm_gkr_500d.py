#!/usr/bin/env python3
"""Plot Stringer held-out achieved information for FM and GKR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("Flow matching", "GKR")
COLORS = {"Flow matching": "C0", "GKR": "C2"}
LABELS = {"Flow matching": "FM", "GKR": "GKR"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--fm-summary", type=Path, required=True)
    parser.add_argument("--gkr-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Stringer, 500D PCA")
    parser.add_argument(
        "--output-stem",
        default="stringer_fm_gkr_500d_test30_achieved_information",
    )
    return parser.parse_args()


def _rows(path: Path) -> list[dict[str, object]]:
    return list(json.loads(path.resolve().read_text(encoding="utf-8"))["rows"])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows(args.fm_summary) + _rows(args.gkr_summary)
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "savefig.bbox": "tight",
        }
    )
    fig, axis = plt.subplots(figsize=(4.0, 3.5), constrained_layout=True)
    for index, method in enumerate(METHODS):
        values = np.asarray(
            [
                float(row["mean_achieved_fisher"])
                for row in rows
                if row["method"] == method
            ],
            dtype=np.float64,
        )
        error = 1.96 * np.std(values, ddof=1) / np.sqrt(values.size)
        axis.errorbar(
            index,
            float(np.mean(values)),
            yerr=float(error),
            color=COLORS[method],
            marker="o",
            markersize=7,
            linewidth=2.2,
            capsize=4,
        )
    axis.set_xticks(range(len(METHODS)), [LABELS[method] for method in METHODS])
    axis.set_ylabel("Achieved information")
    axis.set_title(args.title)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.8)
    axis.spines["bottom"].set_linewidth(1.8)
    axis.tick_params(width=1.8)
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color="0.88", linewidth=0.8)
    axis.xaxis.grid(False)

    stem = args.output_dir / args.output_stem
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    print(f"Saved: {stem.with_suffix('.png')}")
    print(f"Saved: {stem.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
