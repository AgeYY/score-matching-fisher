"""Unit tests for ``fisher.linear_x_flow``."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from fisher.linear_x_flow import (
    ConditionalDiagonalLinearXFlowMLP,
    ConditionalLinearXFlowMLP,
    ConditionalLowRankLinearXFlowMLP,
    ConditionalRandomBasisLowRankLinearXFlowMLP,
    ConditionalScalarLinearXFlowMLP,
    ConditionalThetaDiagonalLinearXFlowMLP,
    ConditionalThetaDiagonalSplineLinearXFlowMLP,
    _phi_expm1_div_a,
    bspline_basis_phi_batch,
    compute_linear_x_flow_c_matrix,
    open_uniform_clamped_knot_vector,
    train_linear_x_flow_schedule,
)
from fisher.gaussian_x_flow import path_schedule_from_name
from fisher.model_weight_ema import (
    evaluate_with_weight_ema,
    init_model_weight_ema,
    load_model_weights_from_ema_state,
    update_model_weight_ema,
)


class TestLinearXFlow(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
