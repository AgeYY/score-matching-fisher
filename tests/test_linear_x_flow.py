"""Unit tests for ``fisher.linear_x_flow``."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from fisher.linear_x_flow import (
    ConditionalDiagonalLinearXFlowMLP,
    ConditionalLinearXFlowMLP,
    ConditionalLowRankLinearXFlowMLP,
    ConditionalScalarLinearXFlowMLP,
    compute_linear_x_flow_c_matrix,
    train_linear_x_flow_schedule,
)
from fisher.gaussian_x_flow import path_schedule_from_name


class TestLinearXFlow(unittest.TestCase):
    def test_model_shapes_and_velocity_finite(self) -> None:
        torch.manual_seed(0)
        m = ConditionalLinearXFlowMLP(theta_dim=2, x_dim=3, hidden_dim=16, depth=2)
        th = torch.randn(5, 2)
        x = torch.randn(5, 3)
        b = m.b(th)
        v = m(x, th)
        self.assertEqual(tuple(m.A.shape), (3, 3))
        self.assertTrue(torch.allclose(m.A, m.A.T))
        self.assertEqual(tuple(b.shape), (5, 3))
        self.assertEqual(tuple(v.shape), (5, 3))
        self.assertTrue(torch.all(torch.isfinite(v)))

    def test_drift_matrix_is_symmetric_view_of_b(self) -> None:
        m = ConditionalLinearXFlowMLP(theta_dim=1, x_dim=2, hidden_dim=8, depth=1)
        with torch.no_grad():
            m.B.copy_(torch.tensor([[1.0, 3.0], [-1.0, 2.0]]))
        expected = torch.tensor([[1.0, 1.0], [1.0, 2.0]])
        self.assertTrue(torch.allclose(m.A, expected))
        self.assertTrue(torch.allclose(m.A, m.A.T))

    def test_scalar_drift_matrix(self) -> None:
        m = ConditionalScalarLinearXFlowMLP(theta_dim=1, x_dim=3, hidden_dim=8, depth=1)
        with torch.no_grad():
            m.a.copy_(torch.tensor(0.7))
        self.assertTrue(torch.allclose(m.A, 0.7 * torch.eye(3)))

    def test_diagonal_drift_matrix(self) -> None:
        m = ConditionalDiagonalLinearXFlowMLP(theta_dim=1, x_dim=3, hidden_dim=8, depth=1)
        with torch.no_grad():
            m.a.copy_(torch.tensor([0.1, 0.2, 0.3]))
        self.assertTrue(torch.allclose(m.A, torch.diag(torch.tensor([0.1, 0.2, 0.3]))))

    def test_low_rank_drift_matrix(self) -> None:
        m = ConditionalLowRankLinearXFlowMLP(theta_dim=1, x_dim=3, rank=2, hidden_dim=8, depth=1)
        with torch.no_grad():
            m.a.copy_(torch.tensor([0.1, 0.2, 0.3]))
            m.U.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
            m.s.copy_(torch.tensor([0.5, -0.25]))
        expected = torch.diag(m.a) + (m.U * m.s.unsqueeze(0)) @ m.U.T
        self.assertTrue(torch.allclose(m.A, expected))
        self.assertTrue(torch.allclose(m.A, m.A.T))

    def test_endpoint_mean_covariance_shapes(self) -> None:
        torch.manual_seed(1)
        m = ConditionalLinearXFlowMLP(theta_dim=1, x_dim=2, hidden_dim=8, depth=1)
        with torch.no_grad():
            m.B.copy_(0.2 * torch.eye(2))
        th = torch.randn(4, 1)
        mu, sigma = m.endpoint_mean_covariance(th, solve_jitter=1e-6)
        self.assertEqual(tuple(mu.shape), (4, 2))
        self.assertEqual(tuple(sigma.shape), (2, 2))
        self.assertTrue(torch.all(torch.isfinite(mu)))
        self.assertTrue(torch.all(torch.isfinite(sigma)))

    def test_log_prob_normalized_finite(self) -> None:
        torch.manual_seed(2)
        m = ConditionalLinearXFlowMLP(theta_dim=1, x_dim=3, hidden_dim=16, depth=2)
        with torch.no_grad():
            m.B.copy_(0.1 * torch.eye(3))
        th = torch.randn(6, 1)
        x = torch.randn(6, 3)
        lp = m.log_prob_normalized(x, th, solve_jitter=1e-6)
        self.assertEqual(tuple(lp.shape), (6,))
        self.assertTrue(torch.all(torch.isfinite(lp)))

    def test_compute_c_matrix_shape_and_finite(self) -> None:
        torch.manual_seed(3)
        n = 5
        d = 2
        theta_all = np.random.randn(n, 1).astype(np.float64)
        x_all = np.random.randn(n, d).astype(np.float64)
        dev = torch.device("cpu")
        m = ConditionalLinearXFlowMLP(theta_dim=1, x_dim=d, hidden_dim=16, depth=1)
        with torch.no_grad():
            m.B.copy_(0.1 * torch.eye(d))
        c = compute_linear_x_flow_c_matrix(
            model=m,
            theta_all=theta_all,
            x_all=x_all,
            device=dev,
            x_mean=np.zeros(d, dtype=np.float64),
            x_std=np.ones(d, dtype=np.float64),
            solve_jitter=1e-6,
            pair_batch_size=256,
        )
        self.assertEqual(c.shape, (n, n))
        self.assertTrue(np.all(np.isfinite(c)))

    def test_train_schedule_cosine_one_epoch_finite(self) -> None:
        torch.manual_seed(4)
        rng = np.random.default_rng(4)
        theta = rng.normal(size=(24, 1)).astype(np.float64)
        x = np.concatenate([theta, -theta], axis=1) + 0.1 * rng.normal(size=(24, 2))
        m = ConditionalLinearXFlowMLP(theta_dim=1, x_dim=2, hidden_dim=8, depth=1)
        out = train_linear_x_flow_schedule(
            model=m,
            theta_train=theta[:16],
            x_train=x[:16],
            theta_val=theta[16:],
            x_val=x[16:],
            device=torch.device("cpu"),
            schedule=path_schedule_from_name("cosine"),
            epochs=1,
            batch_size=8,
            lr=1e-3,
            t_eps=1e-3,
            patience=0,
            log_every=1,
        )
        self.assertEqual(len(out["train_losses"]), 1)
        self.assertEqual(len(out["val_losses"]), 1)
        self.assertTrue(np.isfinite(out["train_losses"][0]))
        self.assertTrue(np.isfinite(out["val_losses"][0]))


if __name__ == "__main__":
    unittest.main()
