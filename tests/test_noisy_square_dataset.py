from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from fisher.noisy_square_dataset import (
    NoisySquareBoundaryDataset,
    generate_noisy_square_boundary_batch,
    plot_noisy_square_boundary_dataset,
    square_rotation_matrix,
)


def test_square_rotation_matrix_is_orthonormal() -> None:
    rot = square_rotation_matrix(math.pi / 4.0)

    np.testing.assert_allclose(rot.T @ rot, np.eye(2), atol=1e-15)
    np.testing.assert_allclose(float(np.linalg.det(rot)), 1.0, atol=1e-15)


def test_noisy_square_batch_uses_base_boundary_and_rotation() -> None:
    rng = np.random.default_rng(123)
    batch = generate_noisy_square_boundary_batch(
        num=128,
        theta=math.pi / 4.0,
        side_length=1.0,
        sigma=0.03,
        center=(0.25, -0.1),
        rng=rng,
    )

    assert batch.x0.shape == (128, 2)
    assert batch.x1.shape == (128, 2)
    np.testing.assert_allclose(np.max(np.abs(batch.x0), axis=1), 0.5)

    rot = square_rotation_matrix(math.pi / 4.0)
    expected = batch.x0 @ rot.T + np.asarray([[0.25, -0.1]]) + 0.03 * batch.eta
    np.testing.assert_allclose(batch.x1, expected)


def test_noiseless_rotated_square_lies_on_boundary() -> None:
    dataset = NoisySquareBoundaryDataset(theta=math.pi / 6.0, side_length=2.0, sigma=0.0, center=(0.1, -0.2), seed=7)
    batch = dataset.sample(256)

    rot = dataset.rotation
    unrotated = (batch.x1 - dataset.center_array.reshape(1, 2)) @ rot
    np.testing.assert_allclose(np.max(np.abs(unrotated), axis=1), 1.0, atol=1e-12)


def test_noisy_square_validation() -> None:
    with pytest.raises(ValueError, match="side_length"):
        NoisySquareBoundaryDataset(side_length=0.0)
    with pytest.raises(ValueError, match="sigma"):
        NoisySquareBoundaryDataset(sigma=-1.0)
    with pytest.raises(ValueError, match="center"):
        NoisySquareBoundaryDataset(center=(0.0, 1.0, 2.0))


def test_visualize_example_noisy_square_dataset(tmp_path: Path) -> None:
    dataset = NoisySquareBoundaryDataset(theta=math.pi / 4.0, side_length=1.0, sigma=0.03, seed=7)
    batch = dataset.sample(256)

    png = plot_noisy_square_boundary_dataset(batch, dataset, tmp_path / "noisy_square_dataset_example.png")
    svg = plot_noisy_square_boundary_dataset(batch, dataset, tmp_path / "noisy_square_dataset_example.svg")

    assert png.is_file()
    assert png.stat().st_size > 0
    assert svg.is_file()
    assert svg.stat().st_size > 0
