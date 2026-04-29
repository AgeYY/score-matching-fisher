"""Unit tests for ``fisher.gaussian_x_flow``."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from fisher.gaussian_x_flow import (
    ConditionalDiagonalGaussianCovarianceFMMLP,
    ConditionalGaussianCovarianceFMMLP,
    analytic_gaussian_fm_velocity,
    compute_gaussian_x_flow_c_matrix,
    path_schedule_from_name,
)


class TestGaussianXFlow(unittest.TestCase):
    def test_model_shapes_and_positive_diag(self) -> None:
        torch.manual_seed(0)
        m = ConditionalGaussianCovarianceFMMLP(theta_dim=2, x_dim=3, hidden_dim=16, depth=2, diag_floor=1e-4)
        th = torch.randn(5, 2)
        mu, l = m(th)
        self.assertEqual(tuple(mu.shape), (5, 3))
        self.assertEqual(tuple(l.shape), (5, 3, 3))
        d = torch.diagonal(l, dim1=-2, dim2=-1)
        self.assertTrue(torch.all(d > 0))

    def test_analytic_velocity_finite_linear_cosine(self) -> None:
        torch.manual_seed(1)
        b, d = 4, 3
        xt = torch.randn(b, d)
        t = torch.rand(b, 1)
        mu = torch.randn(b, d)
        l_cov = torch.tril(torch.randn(b, d, d))
        for k in ("linear", "cosine"):
            sch = path_schedule_from_name(k)
            v = analytic_gaussian_fm_velocity(
                xt=xt,
                t=t,
                mu=mu,
                l_cov=l_cov,
                schedule=sch,
                cov_jitter=1e-3,
            )
            self.assertEqual(tuple(v.shape), (b, d))
            self.assertTrue(torch.all(torch.isfinite(v)))

    def test_log_prob_normalized_matches_manual_triangular(self) -> None:
        """``log_prob_normalized`` matches manual quad form from current ``forward`` outputs."""
        torch.manual_seed(2)
        d = 4
        m = ConditionalGaussianCovarianceFMMLP(theta_dim=1, x_dim=d, hidden_dim=16, depth=2, diag_floor=1e-4)
        th = torch.randn(6, 1)
        x = torch.randn(6, d)
        mu, l = m(th)
        diff = x - mu
        z = torch.linalg.solve_triangular(l, diff.unsqueeze(-1), upper=False).squeeze(-1)
        quad = torch.sum(z * z, dim=1)
        diag_l = torch.diagonal(l, dim1=-2, dim2=-1)
        manual = (
            -0.5 * quad
            - torch.sum(torch.log(torch.clamp(diag_l, min=1e-12)), dim=1)
            - 0.5 * float(d) * torch.tensor(2 * np.pi).log()
        )
        lp = m.log_prob_normalized(x, th)
        self.assertTrue(torch.allclose(lp, manual, atol=1e-5, rtol=1e-4))

    def test_compute_c_matrix_identity_setup(self) -> None:
        torch.manual_seed(3)
        n = 5
        d = 2
        theta_all = np.random.randn(n, 1).astype(np.float64)
        x_all = np.random.randn(n, d).astype(np.float64)
        dev = torch.device("cpu")
        m = ConditionalGaussianCovarianceFMMLP(theta_dim=1, x_dim=d, hidden_dim=16, depth=1, diag_floor=1e-4)
        x_mean = np.zeros(d, dtype=np.float64)
        x_std = np.ones(d, dtype=np.float64)
        c = compute_gaussian_x_flow_c_matrix(
            model=m,
            theta_all=theta_all,
            x_all=x_all,
            device=dev,
            x_mean=x_mean,
            x_std=x_std,
            pair_batch_size=256,
        )
        self.assertEqual(c.shape, (n, n))
        self.assertTrue(np.all(np.isfinite(c)))


class TestGaussianXFlowDiagonal(unittest.TestCase):
    def test_diagonal_model_shapes_offdiag_zero(self) -> None:
        torch.manual_seed(10)
        m = ConditionalDiagonalGaussianCovarianceFMMLP(theta_dim=2, x_dim=4, hidden_dim=16, depth=2, diag_floor=1e-4)
        th = torch.randn(6, 2)
        mu, l = m(th)
        self.assertEqual(tuple(mu.shape), (6, 4))
        self.assertEqual(tuple(l.shape), (6, 4, 4))
        eye = torch.eye(4).unsqueeze(0).expand(6, 4, 4)
        off = l * (1.0 - eye.to(l.device))
        self.assertTrue(torch.allclose(off, torch.zeros_like(off)))
        d = torch.diagonal(l, dim1=-2, dim2=-1)
        self.assertTrue(torch.all(d > 0))

    def test_diagonal_analytic_velocity_finite(self) -> None:
        torch.manual_seed(11)
        m = ConditionalDiagonalGaussianCovarianceFMMLP(theta_dim=1, x_dim=5, hidden_dim=8, depth=1, diag_floor=1e-4)
        th = torch.randn(3, 1)
        mu, l_cov = m(th)
        xt = torch.randn(3, 5)
        t = torch.rand(3, 1)
        sch = path_schedule_from_name("linear")
        v = analytic_gaussian_fm_velocity(
            xt=xt, t=t, mu=mu, l_cov=l_cov, schedule=sch, cov_jitter=1e-3
        )
        self.assertEqual(tuple(v.shape), (3, 5))
        self.assertTrue(torch.all(torch.isfinite(v)))

    def test_diagonal_log_prob_matches_manual(self) -> None:
        torch.manual_seed(12)
        d = 5
        m = ConditionalDiagonalGaussianCovarianceFMMLP(theta_dim=1, x_dim=d, hidden_dim=16, depth=2, diag_floor=1e-4)
        th = torch.randn(4, 1)
        x = torch.randn(4, d)
        mu, l = m(th)
        diff = x - mu
        z = torch.linalg.solve_triangular(l, diff.unsqueeze(-1), upper=False).squeeze(-1)
        quad = torch.sum(z * z, dim=1)
        diag_l = torch.diagonal(l, dim1=-2, dim2=-1)
        manual = (
            -0.5 * quad
            - torch.sum(torch.log(torch.clamp(diag_l, min=1e-12)), dim=1)
            - 0.5 * float(d) * torch.tensor(2 * np.pi).log()
        )
        lp = m.log_prob_normalized(x, th)
        self.assertTrue(torch.allclose(lp, manual, atol=1e-5, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
