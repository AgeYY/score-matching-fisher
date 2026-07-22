#!/usr/bin/env python3
"""Plot held-out achieved information for high-dimensional Stringer fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("Flow matching", "GKR", "OLE (cross-fit)")
COLORS = {"Flow matching": "C0", "GKR": "C2", "OLE (cross-fit)": "C1"}
LABELS = {"Flow matching": "FM", "GKR": "GKR", "OLE (cross-fit)": "OLE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-100d", type=Path, required=True)
    parser.add_argument("--input-200d", type=Path, required=True)
    parser.add_argument("--gkr-100d", type=Path, required=True)
    parser.add_argument("--gkr-200d", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    return list(payload["rows"])


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.8)
    axis.spines["bottom"].set_linewidth(1.8)
    axis.tick_params(width=1.8)
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color="0.88", linewidth=0.8)
    axis.xaxis.grid(False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), constrained_layout=True)
    for panel, (dimension, path, gkr_path) in enumerate(
        (
            (100, args.input_100d, args.gkr_100d),
            (200, args.input_200d, args.gkr_200d),
        )
    ):
        rows = _load(path) + _load(gkr_path)
        axis = axes[panel]
        for method in METHODS:
            selected = [row for row in rows if row["method"] == method]
            fractions = sorted({float(row["test_fraction"]) for row in selected})
            means: list[float] = []
            errors: list[float] = []
            for fraction in fractions:
                values = np.asarray(
                    [
                        float(row["mean_achieved_fisher"])
                        for row in selected
                        if float(row["test_fraction"]) == fraction
                    ],
                    dtype=np.float64,
                )
                means.append(float(np.mean(values)))
                errors.append(
                    float(1.96 * np.std(values, ddof=1) / np.sqrt(values.size))
                    if values.size > 1
                    else 0.0
                )
            axis.errorbar(
                100.0 * np.asarray(fractions),
                means,
                yerr=errors,
                color=COLORS[method],
                marker="o",
                markersize=5,
                linewidth=2.2,
                capsize=3,
                label=LABELS[method],
            )
        axis.set_title(f"Stringer, {dimension}D PCA")
        axis.set_xlabel("Test fraction (%)")
        if panel == 0:
            axis.set_ylabel("Achieved information")
        else:
            axis.legend(frameon=False, loc="best")
        _style_axis(axis)

    stem = args.output_dir / "stringer_fm_ole_100d_200d_achieved_information"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    print(f"Saved: {stem.with_suffix('.png')}")
    print(f"Saved: {stem.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
