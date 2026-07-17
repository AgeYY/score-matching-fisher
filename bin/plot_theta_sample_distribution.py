#!/usr/bin/env python3
"""Plot the empirical theta distribution of a saved shared dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.shared_dataset_io import load_shared_dataset_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bin-width", type=float, default=0.4)
    return parser.parse_args()


def _style_axis(axis: plt.Axes) -> None:
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(width=1.8)
    axis.grid(False)


def main() -> None:
    args = parse_args()
    dataset_npz = args.dataset_npz.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_width = float(args.bin_width)
    if not np.isfinite(bin_width) or bin_width <= 0.0:
        raise ValueError("--bin-width must be finite and positive.")

    bundle = load_shared_dataset_npz(dataset_npz)
    theta_all = np.asarray(bundle.theta_all, dtype=np.float64).reshape(-1)
    theta_train = np.asarray(bundle.theta_train, dtype=np.float64).reshape(-1)
    theta_validation = np.asarray(bundle.theta_validation, dtype=np.float64).reshape(-1)
    theta_low = float(bundle.meta["theta_low"])
    theta_high = float(bundle.meta["theta_high"])

    half_bins = int(np.ceil(max(abs(theta_low), abs(theta_high)) / bin_width))
    edges = (np.arange(-half_bins, half_bins + 1, dtype=np.float64) - 0.5) * bin_width
    expected_count = theta_all.size * bin_width / (theta_high - theta_low)

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
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), constrained_layout=True)

    axes[0].hist(
        theta_all,
        bins=edges,
        color="C0",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        label=f"All samples ($N={theta_all.size}$)",
    )
    axes[0].axhline(
        expected_count,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="Uniform expectation",
    )
    axes[0].set_xlim(theta_low, theta_high)
    axes[0].set_xlabel(r"$\theta$")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Full range")
    axes[0].legend(frameon=False)
    _style_axis(axes[0])

    zoom = 0.5
    train_local = theta_train[np.abs(theta_train) <= zoom]
    validation_local = theta_validation[np.abs(theta_validation) <= zoom]
    axes[1].scatter(
        train_local,
        np.zeros_like(train_local),
        marker="|",
        s=220,
        linewidths=1.8,
        color="C0",
        label=f"Training ({train_local.size})",
    )
    axes[1].scatter(
        validation_local,
        np.ones_like(validation_local),
        marker="|",
        s=220,
        linewidths=1.8,
        color="C1",
        label=f"Validation ({validation_local.size})",
    )
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1.5)
    axes[1].set_xlim(-zoom, zoom)
    axes[1].set_ylim(-0.6, 1.6)
    axes[1].set_yticks([0.0, 1.0], labels=["Training", "Validation"])
    axes[1].set_xlabel(r"$\theta$")
    axes[1].set_title(r"Samples near $\theta=0$")
    axes[1].legend(frameon=False, loc="upper left")
    _style_axis(axes[1])

    output_stem = output_dir / "theta_sample_distribution"
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".svg"))
    plt.close(fig)

    print(f"n_total={theta_all.size}")
    print(f"n_train={theta_train.size}")
    print(f"n_validation={theta_validation.size}")
    print(f"n_abs_theta_le_0.1={int(np.sum(np.abs(theta_all) <= 0.1))}")
    print(f"n_abs_theta_le_0.2={int(np.sum(np.abs(theta_all) <= 0.2))}")
    print(f"n_abs_theta_le_0.5={int(np.sum(np.abs(theta_all) <= 0.5))}")
    print(f"nearest_theta={theta_all[int(np.argmin(np.abs(theta_all)))]:.8f}")
    print(f"figure_png={output_stem.with_suffix('.png')}")
    print(f"figure_svg={output_stem.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
