"""Smoke tests for Monte Carlo generative Hellinger GT."""

from __future__ import annotations

import unittest

import numpy as np

from fisher.data import ToyConditionalGaussianDataset, ToyConditionalGaussianRandamp2DSqrtdDataset
from fisher.hellinger_gt import (
    bin_centers_from_edges,
    estimate_hellinger_sq_grid_centers_mc,
    estimate_hellinger_sq_one_sided_mc,
)


class TestHellingerGT(unittest.TestCase):
    def test_bin_centers_from_edges(self) -> None:
        e = np.linspace(0.0, 1.0, 5, dtype=np.float64)
        c = bin_centers_from_edges(e)
        np.testing.assert_allclose(c, 0.5 * (e[:-1] + e[1:]))

    def test_diagonal_zero_small_mc(self) -> None:
        ds = ToyConditionalGaussianDataset(x_dim=3, seed=0, tuning_curve_family="cosine")
        centers = np.linspace(-1.0, 1.0, 4, dtype=np.float64)
        h2 = estimate_hellinger_sq_one_sided_mc(ds, centers, n_mc=200, symmetrize=False)
        np.testing.assert_allclose(np.diag(h2), 0.0, atol=1e-10)

    def test_grid_centers_mc_shape_and_diagonal(self) -> None:
        ds = ToyConditionalGaussianRandamp2DSqrtdDataset(x_dim=2, seed=1)
        centers = np.asarray(
            [
                [-1.0, -1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
            ],
            dtype=np.float64,
        )
        h2 = estimate_hellinger_sq_grid_centers_mc(ds, centers, n_mc=50, symmetrize=True)
        self.assertEqual(h2.shape, (4, 4))
        np.testing.assert_allclose(np.diag(h2), 0.0, atol=1e-10)
        self.assertTrue(np.all(np.isfinite(h2)))
        self.assertTrue(np.all((h2 >= 0.0) & (h2 <= 1.0)))


if __name__ == "__main__":
    unittest.main()
