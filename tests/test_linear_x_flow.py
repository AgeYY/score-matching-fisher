"""Unit tests for ``fisher.linear_x_flow``."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from fisher.linear_x_flow import (
    ConditionalPCANonlinearLinearXFlowMLP,
    ConditionalDiagonalLinearXFlowFiLMLP,
    ConditionalDiagonalLinearXFlowMLP,
    ConditionalLinearXFlowMLP,
    ConditionalLowRankLinearXFlowMLP,
    ConditionalRandomBasisLowRankLinearXFlowMLP,
    ConditionalScalarLinearXFlowMLP,
    ConditionalThetaDiagonalLinearXFlowMLP,
    ConditionalThetaDiagonalSplineLinearXFlowMLP,
    ConditionalTimeDiagonalLinearXFlowMLP,
    _phi_expm1_div_a,
    bspline_basis_phi_batch,
    compute_linear_x_flow_analytic_hellinger_matrix,
    compute_linear_x_flow_c_matrix,
    compute_time_diagonal_linear_x_flow_c_matrix,
    estimate_binned_gaussian_shared_diagonal_covariance,
    fit_residual_pca_basis_from_linear_mean,
    gaussian_hellinger_sq_diag,
    gaussian_hellinger_sq_diag_matrix,
    gaussian_hellinger_sq_full,
    gaussian_hellinger_sq_shared_covariance_matrix,
    open_uniform_clamped_knot_vector,
    train_linear_x_flow,
    train_pca_nonlinear_linear_x_flow,
    train_linear_x_flow_schedule,
    train_time_diagonal_linear_x_flow_schedule,
)
from fisher.gaussian_x_flow import path_schedule_from_name
from fisher.model_weight_ema import (
    evaluate_with_weight_ema,
    init_model_weight_ema,
    load_model_weights_from_ema_state,
    update_model_weight_ema,
)


class TestLinearXFlow(unittest.TestCase):
    def test_gaussian_hellinger_shared_diagonal_matches_shortcut(self) -> None:
        mu = np.asarray([[0.0, 0.0], [1.0, 2.0], [-0.5, 0.25]], dtype=np.float64)
        var = np.asarray([0.5, 2.0], dtype=np.float64)
        cov = np.diag(var)
        h = gaussian_hellinger_sq_shared_covariance_matrix(mu, cov)
        diff = mu[:, None, :] - mu[None, :, :]
        expected = 1.0 - np.exp(-0.125 * np.sum(diff * diff / var.reshape(1, 1, -1), axis=2))
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_allclose(h, expected, rtol=1e-8, atol=1e-10)
        self.assertTrue(np.isfinite(h).all())
        self.assertLessEqual(float(np.max(h)), 1.0)

    def test_gaussian_hellinger_diag_matches_full(self) -> None:
        mu1 = np.asarray([0.2, -0.7, 1.0])
        mu2 = np.asarray([-0.4, 0.1, 0.5])
        v1 = np.asarray([0.6, 1.5, 2.5])
        v2 = np.asarray([1.2, 0.8, 3.0])
        h_diag = gaussian_hellinger_sq_diag(mu1, v1, mu2, v2, jitter=0.0)
        h_full = gaussian_hellinger_sq_full(mu1, np.diag(v1), mu2, np.diag(v2), jitter=0.0)
        self.assertAlmostEqual(h_diag, h_full, places=12)

    def test_gaussian_hellinger_diag_matrix_properties(self) -> None:
        mu = np.asarray([[0.0, 0.0], [1.0, 0.5], [-0.25, 2.0]])
        var = np.asarray([[1.0, 2.0], [1.5, 0.75], [0.8, 3.0]])
        h = gaussian_hellinger_sq_diag_matrix(mu, var)
        self.assertEqual(h.shape, (3, 3))
        self.assertTrue(np.isfinite(h).all())
        np.testing.assert_allclose(h, h.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(h), 0.0, atol=1e-12)
        self.assertGreaterEqual(float(np.min(h)), 0.0)
        self.assertLessEqual(float(np.max(h)), 1.0)

    def test_linear_x_flow_analytic_hellinger_matches_endpoint_formula(self) -> None:
        torch.manual_seed(0)
        m = ConditionalDiagonalLinearXFlowMLP(theta_dim=1, x_dim=2, hidden_dim=8, depth=1)
        theta = np.asarray([[-1.0], [0.0], [1.0]], dtype=np.float64)
        h, mu, cov, is_diag = compute_linear_x_flow_analytic_hellinger_matrix(
            model=m,
            theta_all=theta,
            device=torch.device("cpu"),
        )
        self.assertFalse(is_diag)
        self.assertEqual(mu.shape, (3, 2))
        self.assertEqual(cov.shape, (2, 2))
        expected = gaussian_hellinger_sq_shared_covariance_matrix(mu, cov)
        np.testing.assert_allclose(h, expected, rtol=1e-10, atol=1e-12)

    def test_theta_and_time_diagonal_analytic_hellinger_finite(self) -> None:
        theta = np.asarray([[-1.0], [0.0], [1.0]], dtype=np.float64)
        for model in (
            ConditionalThetaDiagonalLinearXFlowMLP(theta_dim=1, x_dim=2, hidden_dim=8, depth=1),
            ConditionalTimeDiagonalLinearXFlowMLP(theta_dim=1, x_dim=2, hidden_dim=8, depth=1, quadrature_steps=5),
        ):
            h, mu, var, is_diag = compute_linear_x_flow_analytic_hellinger_matrix(
                model=model,
                theta_all=theta,
                device=torch.device("cpu"),
                quadrature_steps=5,
            )
            self.assertTrue(is_diag)
            self.assertEqual(mu.shape, (3, 2))
            self.assertEqual(var.shape, (3, 2))
            self.assertTrue(np.isfinite(h).all())
            np.testing.assert_allclose(h, h.T, atol=1e-12)
            np.testing.assert_allclose(np.diag(h), 0.0, atol=1e-12)

    def test_load_model_weights_from_ema_state_applies_ema(self) -> None:
        m = torch.nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            m.weight.fill_(1.0)
        ema = init_model_weight_ema(m)
        with torch.no_grad():
            m.weight.fill_(3.0)
        update_model_weight_ema(ema, m, decay=0.5)
        load_model_weights_from_ema_state(m, ema)
        self.assertTrue(torch.allclose(m.weight, ema["weight"].to(m.weight.device, dtype=m.weight.dtype)))

    def test_model_weight_ema_updates_float_tensors_and_copies_integer_buffers(self) -> None:
        bn = torch.nn.BatchNorm1d(2)
        with torch.no_grad():
            bn.weight.fill_(1.0)
            bn.running_mean.fill_(2.0)
            bn.num_batches_tracked.fill_(3)
        ema = init_model_weight_ema(bn)
        with torch.no_grad():
            bn.weight.fill_(3.0)
            bn.running_mean.fill_(4.0)
            bn.num_batches_tracked.fill_(7)
        update_model_weight_ema(ema, bn, decay=0.5)
        self.assertTrue(torch.allclose(ema["weight"], torch.full_like(ema["weight"], 2.0)))
        self.assertTrue(torch.allclose(ema["running_mean"], torch.full_like(ema["running_mean"], 3.0)))
        self.assertEqual(int(ema["num_batches_tracked"].item()), 7)

    def test_evaluate_with_weight_ema_restores_raw_weights(self) -> None:
        m = torch.nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()
        ema = init_model_weight_ema(m)
        with torch.no_grad():
            m.weight.fill_(5.0)
        update_model_weight_ema(ema, m, decay=0.5)
        raw_w = m.weight.detach().clone()
        with evaluate_with_weight_ema(m, ema):
            self.assertFalse(torch.allclose(m.weight, raw_w))
        self.assertTrue(torch.allclose(m.weight, raw_w))

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

    def test_diagonal_film_shapes_velocity_and_likelihood_finite(self) -> None:
        torch.manual_seed(0)
        m = ConditionalDiagonalLinearXFlowFiLMLP(theta_dim=2, x_dim=3, hidden_dim=16, depth=2)
        th = torch.randn(5, 2)
        x = torch.randn(5, 3)
        b = m.b(th)
        v = m(x, th)
        self.assertEqual(tuple(m.A.shape), (3, 3))
        off = m.A - torch.diag(torch.diagonal(m.A))
        self.assertTrue(torch.allclose(off, torch.zeros_like(off)))
        self.assertEqual(tuple(b.shape), (5, 3))
        self.assertEqual(tuple(v.shape), (5, 3))
        self.assertTrue(torch.all(torch.isfinite(v)))
        lp = m.log_prob_normalized(x, th, solve_jitter=1e-6)
        self.assertEqual(tuple(lp.shape), (5,))
        self.assertTrue(torch.all(torch.isfinite(lp)))

    def test_diagonal_film_heads_initialized_zero(self) -> None:
        m = ConditionalDiagonalLinearXFlowFiLMLP(theta_dim=2, x_dim=4, hidden_dim=8, depth=3)
        for lin in m.film_layers:
            self.assertTrue(torch.all(lin.weight == 0))
            self.assertTrue(torch.all(lin.bias == 0))

    def test_phi_expm1_div_a_at_zero(self) -> None:
        a = torch.zeros(4)
        phi = _phi_expm1_div_a(a)
        self.assertTrue(torch.allclose(phi, torch.ones_like(phi)))

    def test_theta_diagonal_forward_and_likelihood_finite(self) -> None:
        torch.manual_seed(0)
        m = ConditionalThetaDiagonalLinearXFlowMLP(theta_dim=2, x_dim=3, hidden_dim=16, depth=2)
        th = torch.randn(5, 2)
        x = torch.randn(5, 3)
        v = m(x, th)
        self.assertEqual(tuple(v.shape), (5, 3))
        self.assertTrue(torch.all(torch.isfinite(v)))
        lp = m.log_prob_normalized(x, th, solve_jitter=1e-6)
        self.assertEqual(tuple(lp.shape), (5,))
        self.assertTrue(torch.all(torch.isfinite(lp)))

    def test_theta_diagonal_endpoint_matches_elementwise_formula(self) -> None:
        m = ConditionalThetaDiagonalLinearXFlowMLP(theta_dim=1, x_dim=2, hidden_dim=4, depth=1)
        with torch.no_grad():
            for p in m.trunk.parameters():
                p.zero_()
            m.a_head.weight.zero_()
            m.b_head.weight.zero_()
            m.a_head.bias.copy_(torch.tensor([0.5, -0.2]))
            m.b_head.bias.copy_(torch.tensor([1.0, 2.0]))
        th = torch.zeros(3, 1)
        mu, sig = m.endpoint_mean_covariance_diag(th, solve_jitter=1e-8)
        a = torch.tensor([0.5, -0.2])
        b = torch.tensor([1.0, 2.0])
        phi = _phi_expm1_div_a(a)
        mu_exp = phi * b
        sig_exp = torch.exp(2.0 * a)
        self.assertTrue(torch.allclose(mu[0], mu_exp))
        self.assertTrue(torch.allclose(sig[0], sig_exp + 1e-8))

    def test_theta_diagonal_compute_c_matrix_finite(self) -> None:
        torch.manual_seed(3)
        n = 5
        d = 2
        theta_all = np.random.randn(n, 1).astype(np.float64)
        x_all = np.random.randn(n, d).astype(np.float64)
        dev = torch.device("cpu")
        m = ConditionalThetaDiagonalLinearXFlowMLP(theta_dim=1, x_dim=d, hidden_dim=16, depth=1)
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

    def test_bspline_basis_partition_of_unity_and_nonneg(self) -> None:
        knots = open_uniform_clamped_knot_vector(5, degree=3, dtype=torch.float64)
        u = torch.linspace(0.01, 0.99, steps=50, dtype=torch.float64)
        phi = bspline_basis_phi_batch(u, knots, degree=3)
        self.assertEqual(tuple(phi.shape), (50, 5))
        self.assertTrue(torch.all(phi >= -1e-10))
        sums = torch.sum(phi, dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5))

    def test_bspline_matches_scipy_when_available(self) -> None:
        try:
            from scipy.interpolate import BSpline as _BSpline  # type: ignore[import-not-found]
        except Exception:  # pragma: no cover
            self.skipTest("scipy not available")
        knots = open_uniform_clamped_knot_vector(5, degree=3, dtype=torch.float64)
        t_np = knots.detach().cpu().numpy()
        u = torch.linspace(0.02, 0.98, steps=17, dtype=torch.float64)
        u_np = u.detach().cpu().numpy()
        phi_t = bspline_basis_phi_batch(u, knots, degree=3).detach().cpu().numpy()
        phi_s = np.stack(
            [_BSpline(t_np, np.eye(5)[i], 3)(u_np) for i in range(5)],
            axis=1,
        )
        self.assertTrue(np.allclose(phi_t, phi_s, atol=1e-5, rtol=1e-5))

    def test_spline_theta_diagonal_forward_and_likelihood_finite(self) -> None:
        torch.manual_seed(1)
        m = ConditionalThetaDiagonalSplineLinearXFlowMLP(
            theta_dim=1,
            x_dim=3,
            theta_min=-1.0,
            theta_max=1.0,
            num_basis=5,
        )
        th = torch.linspace(-1.0, 1.0, steps=7).unsqueeze(-1)
        x = torch.randn(7, 3)
        v = m(x, th)
        self.assertEqual(tuple(v.shape), (7, 3))
        self.assertTrue(torch.all(torch.isfinite(v)))
        lp = m.log_prob_normalized(x, th, solve_jitter=1e-6)
        self.assertEqual(tuple(lp.shape), (7,))
        self.assertTrue(torch.all(torch.isfinite(lp)))

    def test_spline_theta_diagonal_endpoint_matches_elementwise_formula(self) -> None:
        m = ConditionalThetaDiagonalSplineLinearXFlowMLP(
            theta_dim=1,
            x_dim=2,
            theta_min=0.0,
            theta_max=1.0,
            num_basis=5,
        )
        with torch.no_grad():
            m.Wa.zero_()
            m.Wb.zero_()
            m.ca.copy_(torch.tensor([0.5, -0.2]))
            m.cb.copy_(torch.tensor([1.0, 2.0]))
        th = torch.tensor([[0.37], [0.71], [0.05]])
        mu, sig = m.endpoint_mean_covariance_diag(th, solve_jitter=1e-8)
        a = torch.tensor([0.5, -0.2])
        b = torch.tensor([1.0, 2.0])
        phi = _phi_expm1_div_a(a)
        mu_exp = phi * b
        sig_exp = torch.exp(2.0 * a)
        for row in range(3):
            self.assertTrue(torch.allclose(mu[row], mu_exp))
            self.assertTrue(torch.allclose(sig[row], sig_exp + 1e-8))

    def test_spline_theta_diagonal_compute_c_matrix_finite(self) -> None:
        torch.manual_seed(9)
        n = 5
        d = 2
        theta_all = np.random.uniform(-0.5, 0.5, size=(n, 1)).astype(np.float64)
        x_all = np.random.randn(n, d).astype(np.float64)
        dev = torch.device("cpu")
        m = ConditionalThetaDiagonalSplineLinearXFlowMLP(
            theta_dim=1,
            x_dim=d,
            theta_min=float(np.min(theta_all)),
            theta_max=float(np.max(theta_all)),
            num_basis=5,
        )
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

    def test_spline_raises_wrong_theta_dim(self) -> None:
        with self.assertRaises(ValueError):
            ConditionalThetaDiagonalSplineLinearXFlowMLP(
                theta_dim=2,
                x_dim=3,
                theta_min=0.0,
                theta_max=1.0,
                num_basis=5,
            )

    def test_spline_raises_small_num_basis(self) -> None:
        with self.assertRaises(ValueError):
            ConditionalThetaDiagonalSplineLinearXFlowMLP(
                theta_dim=1,
                x_dim=3,
                theta_min=0.0,
                theta_max=1.0,
                num_basis=3,
            )

    def test_low_rank_drift_matrix(self) -> None:
        m = ConditionalLowRankLinearXFlowMLP(theta_dim=1, x_dim=3, rank=2, hidden_dim=8, depth=1)
        with torch.no_grad():
            m.a.copy_(torch.tensor([0.1, 0.2, 0.3]))
            m.U.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
            m.s.copy_(torch.tensor([0.5, -0.25]))
        expected = torch.diag(m.a) + (m.U * m.s.unsqueeze(0)) @ m.U.T
        self.assertTrue(torch.allclose(m.A, expected))
        self.assertTrue(torch.allclose(m.A, m.A.T))

    def test_random_basis_low_rank_drift_matrix(self) -> None:
        torch.manual_seed(5)
        m = ConditionalRandomBasisLowRankLinearXFlowMLP(
            theta_dim=1,
            x_dim=4,
            rank=2,
            hidden_dim=8,
            depth=1,
            lambda_a=1e-4,
            lambda_s=1e-4,
        )
        self.assertTrue(torch.allclose(m.Q.T @ m.Q, torch.eye(2), atol=1e-5))
        with torch.no_grad():
            m.a.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))
            m.S_raw.copy_(torch.tensor([[1.0, 3.0], [-1.0, 2.0]]))
        expected_s = torch.tensor([[1.0, 1.0], [1.0, 2.0]])
        expected_a = torch.diag(m.a) + m.Q @ expected_s @ m.Q.T
        self.assertTrue(torch.allclose(m.S, expected_s))
        self.assertTrue(torch.allclose(m.A, expected_a))
        self.assertTrue(torch.allclose(m.A, m.A.T, atol=1e-6))
        self.assertGreaterEqual(float(m.regularization_loss().detach()), 0.0)

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

    def test_pca_nonlinear_zero_init_matches_linear_velocity(self) -> None:
        torch.manual_seed(10)
        base = ConditionalLinearXFlowMLP(theta_dim=1, x_dim=3, hidden_dim=8, depth=1)
        m = ConditionalPCANonlinearLinearXFlowMLP(
            linear_model=base,
            pca_basis=torch.eye(3)[:, :2],
            hidden_dim=8,
            depth=1,
        )
        th = torch.randn(5, 1)
        x = torch.randn(5, 3)
        t = torch.rand(5, 1)
        self.assertTrue(torch.allclose(m(x, th, t), base(x, th), atol=1e-6))

    def test_fit_residual_pca_basis_orthonormal(self) -> None:
        torch.manual_seed(11)
        rng = np.random.default_rng(11)
        base = ConditionalLinearXFlowMLP(theta_dim=1, x_dim=4, hidden_dim=8, depth=1)
        theta = rng.normal(size=(12, 1)).astype(np.float64)
        x = rng.normal(size=(12, 4)).astype(np.float64)
        u = fit_residual_pca_basis_from_linear_mean(
            linear_model=base,
            theta_train=theta,
            x_train_norm=x,
            pca_dim=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(u.shape, (4, 2))
        self.assertTrue(np.allclose(u.T @ u, np.eye(2), atol=1e-5))

    def test_pca_nonlinear_divergence_zero_h_is_trace_a(self) -> None:
        torch.manual_seed(12)
        base = ConditionalLinearXFlowMLP(theta_dim=1, x_dim=3, hidden_dim=8, depth=1)
        with torch.no_grad():
            base.B.copy_(torch.diag(torch.tensor([0.1, 0.2, 0.3])))
        m = ConditionalPCANonlinearLinearXFlowMLP(
            linear_model=base,
            pca_basis=torch.eye(3)[:, :2],
            hidden_dim=8,
            depth=1,
        )
        th = torch.randn(4, 1)
        x = torch.randn(4, 3)
        t = torch.rand(4, 1)
        div = m.divergence(x, th, t)
        self.assertTrue(torch.allclose(div, torch.full((4,), 0.6), atol=1e-6))

    def test_pca_nonlinear_log_prob_finite(self) -> None:
        torch.manual_seed(13)
        base = ConditionalLinearXFlowMLP(theta_dim=1, x_dim=2, hidden_dim=8, depth=1)
        m = ConditionalPCANonlinearLinearXFlowMLP(
            linear_model=base,
            pca_basis=torch.eye(2)[:, :1],
            hidden_dim=8,
            depth=1,
        )
        th = torch.randn(3, 1)
        x = torch.randn(3, 2)
        lp = m.log_prob_normalized(x, th, ode_steps=2)
        self.assertEqual(tuple(lp.shape), (3,))
        self.assertTrue(torch.all(torch.isfinite(lp)))

    def test_train_pca_nonlinear_one_epoch_finite(self) -> None:
        torch.manual_seed(14)
        rng = np.random.default_rng(14)
        theta = rng.normal(size=(20, 1)).astype(np.float64)
        x = np.concatenate([theta, -theta], axis=1) + 0.1 * rng.normal(size=(20, 2))
        base = ConditionalLinearXFlowMLP(theta_dim=1, x_dim=2, hidden_dim=8, depth=1)
        model = ConditionalPCANonlinearLinearXFlowMLP(
            linear_model=base,
            pca_basis=np.eye(2, 1, dtype=np.float32),
            hidden_dim=8,
            depth=1,
        )
        out = train_pca_nonlinear_linear_x_flow(
            model=model,
            theta_train=theta[:12],
            x_train=x[:12],
            theta_val=theta[12:],
            x_val=x[12:],
            device=torch.device("cpu"),
            x_mean=np.zeros(2, dtype=np.float64),
            x_std=np.ones(2, dtype=np.float64),
            epochs=1,
            batch_size=4,
            lr=1e-3,
            patience=0,
            log_every=1,
            weight_ema_decay=0.0,
        )
        self.assertEqual(len(out["train_losses"]), 1)
        self.assertTrue(np.isfinite(out["train_losses"][0]))

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
        self.assertTrue(bool(out["weight_ema_enabled"]))
        self.assertAlmostEqual(float(out["weight_ema_decay"]), 0.9)
        self.assertEqual(out["final_eval_weights"], "ema")
        self.assertTrue(np.isfinite(out["train_losses"][0]))
        self.assertTrue(np.isfinite(out["val_losses"][0]))

    def test_train_schedule_weight_ema_off_final_eval_is_raw(self) -> None:
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
            weight_ema_decay=0.0,
        )
        self.assertFalse(bool(out["weight_ema_enabled"]))
        self.assertEqual(out["final_eval_weights"], "raw")

    def test_time_diagonal_forward_and_likelihood_finite(self) -> None:
        torch.manual_seed(41)
        m = ConditionalTimeDiagonalLinearXFlowMLP(
            theta_dim=2,
            x_dim=3,
            hidden_dim=8,
            depth=1,
            quadrature_steps=8,
        )
        theta = torch.randn(5, 2)
        x = torch.randn(5, 3)
        t = torch.linspace(0.1, 0.9, 5).reshape(-1, 1)
        v = m(x, theta, t)
        self.assertEqual(tuple(v.shape), (5, 3))
        mu, sigma_diag = m.endpoint_mean_covariance_diag(theta, quadrature_steps=8)
        self.assertEqual(tuple(mu.shape), (5, 3))
        self.assertEqual(tuple(sigma_diag.shape), (5, 3))
        lp = m.log_prob_normalized(x, theta, quadrature_steps=8)
        self.assertEqual(tuple(lp.shape), (5,))
        self.assertTrue(torch.all(torch.isfinite(lp)))

    def test_train_time_diagonal_schedule_one_epoch_finite(self) -> None:
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        theta = rng.normal(size=(24, 1)).astype(np.float64)
        x = np.concatenate([theta, -theta], axis=1) + 0.1 * rng.normal(size=(24, 2))
        m = ConditionalTimeDiagonalLinearXFlowMLP(theta_dim=1, x_dim=2, hidden_dim=8, depth=1, quadrature_steps=8)
        out = train_time_diagonal_linear_x_flow_schedule(
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
            weight_ema_decay=0.0,
        )
        self.assertEqual(len(out["train_losses"]), 1)
        self.assertTrue(np.isfinite(out["train_losses"][0]))
        c = compute_time_diagonal_linear_x_flow_c_matrix(
            model=m,
            theta_all=theta[:6],
            x_all=x[:6],
            device=torch.device("cpu"),
            x_mean=out["x_mean"],
            x_std=out["x_std"],
            quadrature_steps=8,
            pair_batch_size=64,
        )
        self.assertEqual(tuple(c.shape), (6, 6))
        self.assertTrue(np.all(np.isfinite(c)))

    def test_binned_gaussian_lxf_covariance_and_bin_means(self) -> None:
        theta = np.repeat(np.asarray([0.0, 1.0, 2.0], dtype=np.float64), 4).reshape(-1, 1)
        bins = np.repeat(np.arange(3, dtype=np.int64), 4)
        x = np.stack(
            [
                np.repeat(np.asarray([-1.0, 0.5, 2.0]), 4),
                np.repeat(np.asarray([0.25, -0.75, 1.25]), 4),
            ],
            axis=1,
        ).astype(np.float64)
        x += np.tile(np.asarray([[-0.1, 0.1], [0.1, -0.1], [-0.05, 0.05], [0.05, -0.05]]), (3, 1))
        meta = estimate_binned_gaussian_shared_diagonal_covariance(
            x_train=x,
            bin_train=bins,
            n_bins=3,
            variance_floor=1e-4,
        )
        shared_var = np.asarray(meta["shared_variance"], dtype=np.float64)
        self.assertTrue(np.allclose(np.exp(2.0 * np.asarray(meta["a"], dtype=np.float64)), shared_var))
        self.assertTrue(np.allclose(np.asarray(meta["bin_counts"]), np.asarray([4, 4, 4])))

    def test_binned_gaussian_lxf_empty_bins_filled_like_binned_gaussian(self) -> None:
        theta = np.asarray([0.0, 0.1, 2.0, 2.1], dtype=np.float64).reshape(-1, 1)
        bins = np.asarray([0, 0, 2, 2], dtype=np.int64)
        x = np.asarray([[0.0], [0.2], [2.0], [2.2]], dtype=np.float64)
        meta = estimate_binned_gaussian_shared_diagonal_covariance(
            x_train=x,
            bin_train=bins,
            n_bins=3,
            variance_floor=1e-4,
        )
        self.assertEqual(int(np.asarray(meta["bin_counts"])[1]), 0)
        means = np.asarray(meta["normalized_bin_means"], dtype=np.float64)
        self.assertTrue(np.allclose(means[1], means[0]))

    def test_binned_gaussian_fixed_a_training_updates_only_b_network(self) -> None:
        torch.manual_seed(21)
        rng = np.random.default_rng(21)
        theta = rng.normal(size=(24, 2)).astype(np.float64)
        x = np.stack([theta[:, 0] + 0.1 * theta[:, 1], -theta[:, 1]], axis=1) + 0.05 * rng.normal(size=(24, 2))
        bins = np.repeat(np.arange(3, dtype=np.int64), 6)
        meta = estimate_binned_gaussian_shared_diagonal_covariance(
            x_train=x[:18],
            bin_train=bins,
            n_bins=3,
            variance_floor=1e-4,
        )
        model = ConditionalDiagonalLinearXFlowMLP(theta_dim=2, x_dim=2, hidden_dim=8, depth=1)
        with torch.no_grad():
            model.a.copy_(torch.as_tensor(meta["a"], dtype=model.a.dtype))
        model.a.requires_grad_(False)
        a0 = model.a.detach().clone()
        b0 = [p.detach().clone() for p in model.b_net.parameters()]
        out = train_linear_x_flow(
            model=model,
            theta_train=theta[:18],
            x_train=x[:18],
            theta_val=theta[18:],
            x_val=x[18:],
            device=torch.device("cpu"),
            epochs=2,
            batch_size=6,
            lr=1e-3,
            patience=0,
            log_every=1,
            weight_ema_decay=0.0,
            restore_best=False,
        )
        self.assertEqual(len(out["train_losses"]), 2)
        self.assertTrue(torch.allclose(model.a.detach(), a0))
        changed = any(not torch.allclose(p.detach(), p0) for p, p0 in zip(model.b_net.parameters(), b0))
        self.assertTrue(changed)
        lp = model.log_prob_observed(
            torch.from_numpy(x[18:].astype(np.float32)),
            torch.from_numpy(theta[18:].astype(np.float32)),
            x_mean=torch.from_numpy(np.asarray(out["x_mean"], dtype=np.float32)),
            x_std=torch.from_numpy(np.asarray(out["x_std"], dtype=np.float32)),
            solve_jitter=1e-6,
        )
        self.assertTrue(torch.all(torch.isfinite(lp)))


if __name__ == "__main__":
    unittest.main()
