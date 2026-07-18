#!/usr/bin/env python3
"""Estimate a binned classical correlation RDM for the time-resolved toy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher.time_resolved_rdm_toy import (  # noqa: E402
    estimate_binned_correlation_distance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset-npz",
        type=Path,
        default=ROOT
        / "data/time_resolved_rdm_toy_xdim40_n100_per_class"
        / "two_class_time_resolved_rdm_toy.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "data/time_resolved_rdm_toy_xdim40_n100_per_class"
        / "classical_correlation_bin0p5",
    )
    parser.add_argument("--bin-width", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.dataset_npz, allow_pickle=False) as archive:
        responses = np.asarray(archive["responses"], dtype=np.float64)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        time = np.asarray(archive["time"], dtype=np.float64)
        true_class_means = np.asarray(
            archive["true_class_means"], dtype=np.float64
        )
    result = estimate_binned_correlation_distance(
        responses,
        labels,
        time,
        bin_width=float(args.bin_width),
        true_class_means=true_class_means,
    )
    estimated = result["estimated_correlation_distance"]
    ground_truth = result["true_correlation_distance"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "binned_correlation_distance.npz",
        **result,
        bin_width=np.asarray(float(args.bin_width), dtype=np.float64),
        dataset_npz=np.asarray(str(args.dataset_npz.resolve())),
    )

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 14,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 11,
            "axes.grid": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )
    figure, axis = plt.subplots(figsize=(4.0, 3.5), layout="constrained")
    axis.plot(
        result["bin_centers"],
        estimated,
        color="#4477AA",
        linewidth=2.0,
        marker="o",
        markersize=4.0,
        label="Classical",
    )
    axis.plot(
        result["bin_centers"],
        ground_truth,
        color="0.15",
        linewidth=2.0,
        linestyle="--",
        label="Ground truth",
    )
    upper = max(float(np.max(estimated)), np.finfo(np.float64).eps)
    axis.set_ylim(-0.08 * upper, 1.08 * upper)
    axis.set_xlim(float(time[0]), float(time[-1]))
    axis.set_xlabel("Time")
    axis.set_ylabel("Correlation distance")
    axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        borderaxespad=0.0,
        handlelength=2.2,
    )
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.8)
    axis.spines["bottom"].set_linewidth(1.8)
    axis.tick_params(width=1.8)
    stem = "classical_correlation_distance_vs_time"
    figure.savefig(args.output_dir / f"{stem}.png", dpi=300)
    figure.savefig(args.output_dir / f"{stem}.svg")
    plt.close(figure)

    summary = {
        "dataset_npz": str(args.dataset_npz.resolve()),
        "estimator": (
            "Within each time bin, pool trials and native time samples by class; "
            "compute class mean feature vectors; return one minus their Pearson correlation."
        ),
        "bin_width": float(args.bin_width),
        "n_bins": int(result["bin_centers"].size),
        "estimated_distance_min": float(np.min(estimated)),
        "estimated_distance_max": float(np.max(estimated)),
        "ground_truth_distance_min": float(np.min(ground_truth)),
        "ground_truth_distance_max": float(np.max(ground_truth)),
        "ground_truth_interpretation": (
            "Zero because class 2 is a positive scalar multiple of class 1, "
            "and correlation distance is invariant to positive scaling."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"[correlation-rdm] output={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
