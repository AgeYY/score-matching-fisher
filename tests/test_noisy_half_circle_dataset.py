from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fisher.noisy_half_circle_dataset import (
    NoisyHalfCircleBoundaryDataset,
    generate_noisy_half_circle_boundary_batch,
    plot_noisy_half_circle_boundary_datasets,
)


def test_noisy_half_circle_batch_uses_base_arc_and_shift() -> None:
    rng = np.random.default_rng(123)
    batch = generate_noisy_half_circle_boundary_batch(
        num=128,
        radius=1.5,
        sigma=0.2,
        center=(-1.0, 0.25),
        arc="lower",
        rng=rng,
    )

    assert batch.x0.shape == (128, 2)
    assert batch.x1.shape == (128, 2)
    np.testing.assert_allclose(np.linalg.norm(batch.x0, axis=1), 1.5)
    assert np.all(batch.x0[:, 1] >= 0.0)

    target_clean = batch.x0.copy()
    target_clean[:, 1] *= -1.0
    expected = target_clean + np.asarray([[-1.0, 0.25]]) + 0.2 * batch.eta
    np.testing.assert_allclose(batch.x1, expected)


def test_noiseless_half_circle_lies_on_shifted_arc() -> None:
    dataset = NoisyHalfCircleBoundaryDataset(radius=2.0, sigma=0.0, center=(1.0, -0.5), arc="lower", seed=7)
    batch = dataset.sample(256)

    centered = batch.x1 - dataset.center_array.reshape(1, 2)
    np.testing.assert_allclose(np.linalg.norm(centered, axis=1), 2.0, atol=1e-12)
    assert np.all(centered[:, 1] <= 1e-12)


def test_noisy_half_circle_validation() -> None:
    with pytest.raises(ValueError, match="radius"):
        NoisyHalfCircleBoundaryDataset(radius=0.0)
    with pytest.raises(ValueError, match="sigma"):
        NoisyHalfCircleBoundaryDataset(sigma=-1.0)
    with pytest.raises(ValueError, match="center"):
        NoisyHalfCircleBoundaryDataset(center=(0.0, 1.0, 2.0))
    with pytest.raises(ValueError, match="arc"):
        NoisyHalfCircleBoundaryDataset(arc="sideways")


def test_visualize_example_noisy_half_circle_dataset(tmp_path: Path) -> None:
    datasets = [
        NoisyHalfCircleBoundaryDataset(radius=1.0, sigma=0.2, center=(-1.0, 0.0), arc="upper", seed=7),
        NoisyHalfCircleBoundaryDataset(radius=1.0, sigma=0.2, center=(1.0, 0.0), arc="lower", seed=11),
    ]
    batches = [dataset.sample(256) for dataset in datasets]

    png = plot_noisy_half_circle_boundary_datasets(
        batches,
        datasets,
        tmp_path / "noisy_half_circle_dataset_example.png",
    )
    svg = plot_noisy_half_circle_boundary_datasets(
        batches,
        datasets,
        tmp_path / "noisy_half_circle_dataset_example.svg",
    )

    assert png.is_file()
    assert png.stat().st_size > 0
    assert svg.is_file()
    assert svg.stat().st_size > 0
