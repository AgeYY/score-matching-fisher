"""Unit tests for NF posterior-minus-prior ratio helpers."""

from __future__ import annotations

import unittest

import numpy as np

from fisher.nf_hellinger import compute_delta_l, compute_ratio_matrix_posterior_minus_prior


class TestNFPriorRatio(unittest.TestCase):
    def test_ratio_and_delta_from_posterior_minus_prior(self) -> None:
        c_post = np.array(
            [
                [1.2, 0.4, -0.1],
                [0.2, 1.1, 0.0],
                [-0.2, 0.5, 0.9],
            ],
            dtype=np.float64,
        )
        log_prior = np.array([0.3, -0.1, 0.2], dtype=np.float64)
        ratio = compute_ratio_matrix_posterior_minus_prior(c_post, log_prior)
        expected_ratio = c_post - log_prior.reshape(1, -1)
        np.testing.assert_allclose(ratio, expected_ratio, rtol=0.0, atol=1e-12)

        delta = compute_delta_l(ratio)
        expected_delta = expected_ratio - np.diag(expected_ratio).reshape(-1, 1)
        np.testing.assert_allclose(delta, expected_delta, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(np.diag(delta), np.zeros(3, dtype=np.float64), rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
