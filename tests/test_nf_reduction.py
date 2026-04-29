"""Unit tests for ``fisher.nf_reduction``."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from fisher.nf_reduction import NFReductionModel, compute_nf_reduction_c_matrix, train_nf_reduction


class TestNFReduction(unittest.TestCase):
    def test_encode_shapes_and_finite_log_prob(self) -> None:
        torch.manual_seed(0)
        model = NFReductionModel(theta_dim=1, x_dim=4, latent_dim=2, hidden_dim=8, transforms=1, context_dim=4)
        x = torch.randn(5, 4)
        theta = torch.randn(5, 1)
        z, eps, logdet = model.encode_normalized(x)
        self.assertEqual(tuple(z.shape), (5, 2))
        self.assertEqual(tuple(eps.shape), (5, 2))
        self.assertEqual(tuple(logdet.shape), (5,))
        lp = model.log_prob_normalized_x_given_theta(x, theta)
        self.assertEqual(tuple(lp.shape), (5,))
        self.assertTrue(torch.all(torch.isfinite(lp)))

    def test_training_and_c_matrix_smoke(self) -> None:
        torch.manual_seed(1)
        rng = np.random.default_rng(1)
        n = 24
        x_dim = 3
        theta = rng.normal(size=(n, 1)).astype(np.float64)
        x = rng.normal(size=(n, x_dim)).astype(np.float64)
        model = NFReductionModel(theta_dim=1, x_dim=x_dim, latent_dim=1, hidden_dim=8, transforms=1, context_dim=4)
        dev = torch.device("cpu")
        out = train_nf_reduction(
            model=model,
            theta_train=theta[:16],
            x_train=x[:16],
            theta_val=theta[16:],
            x_val=x[16:],
            device=dev,
            epochs=2,
            batch_size=8,
            lr=1e-3,
            patience=3,
            min_delta=0.0,
            ema_alpha=0.5,
            log_every=10,
        )
        self.assertGreaterEqual(np.asarray(out["train_losses"]).size, 1)
        c, z = compute_nf_reduction_c_matrix(
            model=model,
            theta_all=theta,
            x_all=x,
            device=dev,
            x_mean=np.asarray(out["x_mean"], dtype=np.float64),
            x_std=np.asarray(out["x_std"], dtype=np.float64),
            pair_batch_size=256,
        )
        self.assertEqual(c.shape, (n, n))
        self.assertEqual(z.shape, (n, 1))
        self.assertTrue(np.all(np.isfinite(c)))


if __name__ == "__main__":
    unittest.main()
