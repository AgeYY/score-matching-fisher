from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from fisher.noisy_line_dataset import (
    NoisyLineDataset,
    generate_noisy_line_batch,
    noisy_line_basis,
    plot_noisy_line_dataset,
)


def test_noisy_line_basis_is_orthonormal() -> None:
    q, n = noisy_line_basis(math.pi / 6.0)

    np.testing.assert_allclose(np.linalg.norm(q), 1.0)
    np.testing.assert_allclose(np.linalg.norm(n), 1.0)
    np.testing.assert_allclose(float(q @ n), 0.0, atol=1e-15)


def test_noisy_line_batch_uses_matched_latent_and_noiseless_base() -> None:
    rng = np.random.default_rng(123)
    batch = generate_noisy_line_batch(
        num=64,
        theta=math.pi / 6.0,
        ell=1.5,
        sigma=0.12,
        shift=(0.25, -0.1),
        rng=rng,
    )

    assert batch.x0.shape == (64, 2)
    assert batch.x1.shape == (64, 2)
    np.testing.assert_allclose(batch.x0[:, 0], batch.u)
    np.testing.assert_allclose(batch.x0[:, 1], 0.0)

    q, n = noisy_line_basis(math.pi / 6.0)
    expected_x1 = np.asarray([[0.25, -0.1]]) + 1.5 * batch.u[:, None] * q + 0.12 * batch.eta[:, None] * n
    np.testing.assert_allclose(batch.x1, expected_x1)


def test_target_coordinates_recover_tangent_and_normal_components() -> None:
    dataset = NoisyLineDataset(theta=math.pi / 6.0, ell=1.5, sigma=0.12, shift=(0.25, -0.1), seed=123)
    batch = dataset.sample(128)

    tangent, normal = dataset.target_coordinates(batch.x1)

    np.testing.assert_allclose(tangent, 1.5 * batch.u, atol=1e-12)
    np.testing.assert_allclose(normal, 0.12 * batch.eta, atol=1e-12)


def test_noisy_line_validation() -> None:
    with pytest.raises(ValueError, match="ell"):
        NoisyLineDataset(ell=0.0)
    with pytest.raises(ValueError, match="sigma"):
        NoisyLineDataset(sigma=-1.0)
    with pytest.raises(ValueError, match="shift"):
        NoisyLineDataset(shift=(0.0, 1.0, 2.0))


def test_visualize_example_noisy_line_dataset(tmp_path: Path) -> None:
    dataset = NoisyLineDataset(theta=math.pi / 6.0, ell=1.5, sigma=0.12, seed=7)
    batch = dataset.sample(256)

    png = plot_noisy_line_dataset(batch, dataset, tmp_path / "noisy_line_dataset_example.png")
    svg = plot_noisy_line_dataset(batch, dataset, tmp_path / "noisy_line_dataset_example.svg")

    assert png.is_file()
    assert png.stat().st_size > 0
    assert svg.is_file()
    assert svg.stat().st_size > 0
