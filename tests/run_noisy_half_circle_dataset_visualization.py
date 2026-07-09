#!/usr/bin/env python3
"""Visualize the two-condition noisy half-circle dataset."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import DATA_DIR

from fisher.noisy_half_circle_dataset import (
    NoisyHalfCircleBoundaryDataset,
    plot_noisy_half_circle_boundary_datasets,
)


def main() -> int:
    out_dir = Path(DATA_DIR) / "noisy_half_circle_dataset_example"
    datasets = [
        NoisyHalfCircleBoundaryDataset(radius=1.0, sigma=0.2, center=(-1.0, 0.0), arc="upper", seed=7),
        NoisyHalfCircleBoundaryDataset(radius=1.0, sigma=0.2, center=(1.0, 0.0), arc="lower", seed=11),
    ]
    batches = [dataset.sample(1000) for dataset in datasets]

    png = plot_noisy_half_circle_boundary_datasets(
        batches,
        datasets,
        out_dir / "noisy_half_circle_dataset_example.png",
        title="Noisy half-circle dataset",
    )
    svg = plot_noisy_half_circle_boundary_datasets(
        batches,
        datasets,
        out_dir / "noisy_half_circle_dataset_example.svg",
        title="Noisy half-circle dataset",
    )
    print(f"png: {png}", flush=True)
    print(f"svg: {svg}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
