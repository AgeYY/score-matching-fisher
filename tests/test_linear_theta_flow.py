"""Unit tests for ``fisher.linear_theta_flow``."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from fisher.linear_theta_flow import (
    ConditionalLinearThetaFlowMixtureMLP,
    compute_linear_theta_flow_c_matrix,
    train_linear_theta_flow,
)


class TestLinearThetaFlow(unittest.TestCase):
    def test_model_shapes_and_symmetric_components(self) -> None:
        torch.manual_seed(0)
        m = ConditionalLinearThetaFlowMixtureMLP(theta_dim=2, x_dim=3, num_components=3, hidden_dim=8, depth=1)
        theta = torch.randn(5, 2)
        x = torch.randn(5, 3)
        v, logits = m.component_velocities(theta, x)
        self.assertEqual(tuple(m.A.shape), (3, 2, 2))
        self.assertTrue(torch.allclose(m.A, m.A.transpose(-1, -2)))
        self.assertEqual(tuple(v.shape), (5, 3, 2))
        self.assertEqual(tuple(logits.shape), (5, 3))
        self.assertTrue(torch.all(torch.isfinite(v)))

    def test_log_prob_normalized_finite(self) -> None:
        torch.manual_seed(1)
        m = ConditionalLinearThetaFlowMixtureMLP(theta_dim=1, x_dim=2, num_components=3, hidden_dim=8, depth=1)
        theta = torch.randn(6, 1)
        x = torch.randn(6, 2)
        lp = m.log_prob_normalized(theta, x, solve_jitter=1e-6)
        self.assertEqual(tuple(lp.shape), (6,))
        self.assertTrue(torch.all(torch.isfinite(lp)))

    def test_compute_c_matrix_shape_and_finite(self) -> None:
        torch.manual_seed(2)
        rng = np.random.default_rng(2)
        n = 5
        theta = rng.normal(size=(n, 1)).astype(np.float64)
        x = rng.normal(size=(n, 2)).astype(np.float64)
        m = ConditionalLinearThetaFlowMixtureMLP(theta_dim=1, x_dim=2, num_components=2, hidden_dim=8, depth=1)
        c = compute_linear_theta_flow_c_matrix(
            model=m,
            theta_all=theta,
            x_all=x,
            device=torch.device("cpu"),
            theta_mean=np.zeros(1, dtype=np.float64),
            theta_std=np.ones(1, dtype=np.float64),
            x_mean=np.zeros(2, dtype=np.float64),
            x_std=np.ones(2, dtype=np.float64),
            solve_jitter=1e-6,
            pair_batch_size=256,
        )
        self.assertEqual(c.shape, (n, n))
        self.assertTrue(np.all(np.isfinite(c)))

    def test_train_one_epoch_finite(self) -> None:
        torch.manual_seed(3)
        rng = np.random.default_rng(3)
        x = rng.normal(size=(24, 2)).astype(np.float64)
        theta = (x[:, :1] - 0.5 * x[:, 1:2]) + 0.1 * rng.normal(size=(24, 1))
        m = ConditionalLinearThetaFlowMixtureMLP(theta_dim=1, x_dim=2, num_components=3, hidden_dim=8, depth=1)
        out = train_linear_theta_flow(
            model=m,
            theta_train=theta[:16],
            x_train=x[:16],
            theta_val=theta[16:],
            x_val=x[16:],
            device=torch.device("cpu"),
            epochs=1,
            batch_size=8,
            lr=1e-3,
            t_eps=0.05,
            patience=0,
            log_every=1,
        )
        self.assertEqual(len(out["train_losses"]), 1)
        self.assertEqual(len(out["val_losses"]), 1)
        self.assertTrue(np.isfinite(out["train_losses"][0]))
        self.assertTrue(np.isfinite(out["val_losses"][0]))


if __name__ == "__main__":
    unittest.main()
