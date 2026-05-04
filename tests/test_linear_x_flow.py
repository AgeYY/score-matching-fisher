"""Unit tests for ``fisher.linear_x_flow``."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from fisher.linear_x_flow import (
    ConditionalTimeDiagonalLinearXFlowMLP,
    ConditionalTimeLinearXFlowMLP,
    ConditionalTimeLowRankCorrectionLinearXFlowMLP,
    ConditionalTimeThetaOnlyBLowRankCorrectionLinearXFlowMLP,
    ConditionalTimeLowRankLinearXFlowMLP,
    ConditionalTimeRandomBasisLowRankLinearXFlowMLP,
    ConditionalTimeScalarLinearXFlowMLP,
    ConditionalTimeThetaDiagonalLinearXFlowMLP,
    _phi_expm1_div_a,
    compute_linear_x_flow_analytic_hellinger_matrix,
    compute_ode_time_linear_x_flow_c_matrix,
    compute_time_linear_x_flow_c_matrix,
    compute_time_diagonal_linear_x_flow_c_matrix,
    gaussian_hellinger_sq_diag,
    gaussian_hellinger_sq_diag_matrix,
    gaussian_hellinger_sq_full,
    gaussian_hellinger_sq_shared_covariance_matrix,
    train_low_rank_t_warmup_then_full,
    train_low_rank_t_theta_only_b_mean_regression_pretrain_then_freeze_b,
    train_time_linear_x_flow_schedule,
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

    def test_phi_expm1_div_a_at_zero(self) -> None:
        a = torch.zeros(4)
        phi = _phi_expm1_div_a(a)
        self.assertTrue(torch.allclose(phi, torch.ones_like(phi)))

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

    def test_time_lxf_variants_forward_endpoint_and_likelihood_finite(self) -> None:
        torch.manual_seed(43)
        theta = torch.randn(4, 2)
        x = torch.randn(4, 3)
        t = torch.linspace(0.1, 0.9, 4).reshape(-1, 1)
        models = (
            ConditionalTimeLinearXFlowMLP(theta_dim=2, x_dim=3, hidden_dim=8, depth=1, quadrature_steps=5),
            ConditionalTimeScalarLinearXFlowMLP(theta_dim=2, x_dim=3, hidden_dim=8, depth=1, quadrature_steps=5),
            ConditionalTimeThetaDiagonalLinearXFlowMLP(theta_dim=2, x_dim=3, hidden_dim=8, depth=1, quadrature_steps=5),
            ConditionalTimeLowRankLinearXFlowMLP(theta_dim=2, x_dim=3, rank=2, hidden_dim=8, depth=1, quadrature_steps=5),
            ConditionalTimeRandomBasisLowRankLinearXFlowMLP(theta_dim=2, x_dim=3, rank=2, hidden_dim=8, depth=1, quadrature_steps=5),
        )
        for model in models:
            with self.subTest(model=type(model).__name__):
                v = model(x, theta, t)
                self.assertEqual(tuple(v.shape), (4, 3))
                self.assertTrue(torch.all(torch.isfinite(v)))
                h, mu, cov_or_var, is_diag = compute_linear_x_flow_analytic_hellinger_matrix(
                    model=model,
                    theta_all=theta.detach().cpu().numpy(),
                    device=torch.device("cpu"),
                    quadrature_steps=5,
                )
                self.assertEqual(tuple(h.shape), (4, 4))
                self.assertEqual(tuple(mu.shape), (4, 3))
                self.assertTrue(np.isfinite(cov_or_var).all())
                self.assertTrue(np.isfinite(h).all())
                np.testing.assert_allclose(h, h.T, atol=1e-8)
                np.testing.assert_allclose(np.diag(h), 0.0, atol=1e-8)
                if isinstance(model, (ConditionalTimeScalarLinearXFlowMLP, ConditionalTimeThetaDiagonalLinearXFlowMLP)):
                    self.assertTrue(is_diag)
                    self.assertEqual(tuple(cov_or_var.shape), (4, 3))
                else:
                    self.assertFalse(is_diag)
                    self.assertIn(cov_or_var.ndim, (2, 3))

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

    def test_train_time_full_schedule_linear_one_epoch_finite(self) -> None:
        torch.manual_seed(44)
        rng = np.random.default_rng(44)
        theta = rng.normal(size=(24, 1)).astype(np.float64)
        x = np.concatenate([theta, -theta], axis=1) + 0.1 * rng.normal(size=(24, 2))
        m = ConditionalTimeLinearXFlowMLP(theta_dim=1, x_dim=2, hidden_dim=8, depth=1, quadrature_steps=5)
        out = train_time_linear_x_flow_schedule(
            model=m,
            theta_train=theta[:16],
            x_train=x[:16],
            theta_val=theta[16:],
            x_val=x[16:],
            device=torch.device("cpu"),
            schedule=path_schedule_from_name("linear"),
            epochs=1,
            batch_size=8,
            lr=1e-3,
            t_eps=1e-3,
            patience=0,
            log_every=1,
            weight_ema_decay=0.0,
            log_name="linear_x_flow_t",
        )
        self.assertEqual(len(out["train_losses"]), 1)
        self.assertTrue(np.isfinite(out["train_losses"][0]))
        c = compute_time_linear_x_flow_c_matrix(
            model=m,
            theta_all=theta[:6],
            x_all=x[:6],
            device=torch.device("cpu"),
            x_mean=out["x_mean"],
            x_std=out["x_std"],
            quadrature_steps=5,
            pair_batch_size=64,
        )
        self.assertEqual(tuple(c.shape), (6, 6))
        self.assertTrue(np.all(np.isfinite(c)))




    def test_time_low_rank_correction_U_orthonormal_after_steps(self) -> None:
        torch.manual_seed(48)
        m = ConditionalTimeLowRankCorrectionLinearXFlowMLP(
            theta_dim=1, x_dim=3, correction_rank=2, hidden_dim=8, depth=1, quadrature_steps=8
        )
        u0 = m.U.detach().cpu().numpy()
        self.assertTrue(np.allclose(u0.T @ u0, np.eye(2), atol=1e-5))
        opt = torch.optim.AdamW(m.parameters(), lr=0.05, weight_decay=0.0)
        for _ in range(4):
            x = torch.randn(6, 3)
            th = torch.randn(6, 1)
            t = torch.rand(6, 1)
            loss = (m(x, th, t) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        u1 = m.U.detach().cpu().numpy()
        self.assertTrue(np.allclose(u1.T @ u1, np.eye(2), atol=1e-4))

    def test_time_low_rank_correction_zero_h_matches_linear_submodule(self) -> None:
        torch.manual_seed(49)
        m = ConditionalTimeLowRankCorrectionLinearXFlowMLP(
            theta_dim=1, x_dim=3, correction_rank=2, hidden_dim=8, depth=1, quadrature_steps=5
        )
        th = torch.randn(5, 1)
        x = torch.randn(5, 3)
        t = torch.linspace(0.2, 0.8, 5).reshape(-1, 1)
        self.assertTrue(torch.allclose(m(x, th, t), m.linear(x, th, t), atol=1e-6))

    def test_time_low_rank_correction_base_A_initializes_exactly_zero(self) -> None:
        torch.manual_seed(54)
        m = ConditionalTimeLowRankCorrectionLinearXFlowMLP(
            theta_dim=1, x_dim=4, correction_rank=2, hidden_dim=8, depth=1, quadrature_steps=5
        )
        t = torch.linspace(0.0, 1.0, 7).reshape(-1, 1)
        self.assertTrue(torch.allclose(m.linear.A(t), torch.zeros(7, 4, 4), atol=0.0, rtol=0.0))

    def test_time_low_rank_correction_warmup_freezes_non_b_parameters(self) -> None:
        torch.manual_seed(55)
        rng = np.random.default_rng(55)
        theta = rng.normal(size=(24, 1)).astype(np.float64)
        x = np.concatenate([theta, -theta, 0.5 * theta], axis=1) + 0.05 * rng.normal(size=(24, 3))
        m = ConditionalTimeLowRankCorrectionLinearXFlowMLP(
            theta_dim=1, x_dim=3, correction_rank=2, hidden_dim=8, depth=1, quadrature_steps=5
        )
        before = {name: p.detach().clone() for name, p in m.named_parameters()}
        for p in m.parameters():
            p.requires_grad_(False)
        for p in m.linear.b_net.parameters():
            p.requires_grad_(True)
        out = train_time_linear_x_flow_schedule(
            model=m,
            theta_train=theta[:16],
            x_train=x[:16],
            theta_val=theta[16:],
            x_val=x[16:],
            device=torch.device("cpu"),
            schedule=path_schedule_from_name("linear"),
            epochs=1,
            batch_size=8,
            lr=1e-2,
            t_eps=1e-3,
            patience=0,
            log_every=1,
            weight_ema_decay=0.0,
            restore_best=False,
            log_name="linear_x_flow_low_rank_t_warmup_test",
        )
        self.assertEqual(len(out["train_losses"]), 1)
        b_changed = False
        for name, p in m.named_parameters():
            changed = not torch.allclose(p.detach(), before[name])
            if name.startswith("linear.b_net."):
                b_changed = b_changed or changed
            else:
                self.assertFalse(changed, msg=name)
        self.assertTrue(b_changed)
        self.assertTrue(torch.allclose(m.linear.A(torch.full((3, 1), 0.5)), torch.zeros(3, 3, 3), atol=0.0, rtol=0.0))

    def test_train_low_rank_t_warmup_then_full_restores_trainability_and_finite_c_matrix(self) -> None:
        torch.manual_seed(56)
        rng = np.random.default_rng(56)
        theta = rng.normal(size=(24, 1)).astype(np.float64)
        x = np.concatenate([theta, -theta], axis=1) + 0.05 * rng.normal(size=(24, 2))
        m = ConditionalTimeLowRankCorrectionLinearXFlowMLP(
            theta_dim=1, x_dim=2, correction_rank=1, hidden_dim=8, depth=1, quadrature_steps=5
        )
        out = train_low_rank_t_warmup_then_full(
            model=m,
            theta_train=theta[:16],
            x_train=x[:16],
            theta_val=theta[16:],
            x_val=x[16:],
            device=torch.device("cpu"),
            schedule=path_schedule_from_name("linear"),
            warmup_epochs=1,
            epochs=1,
            batch_size=8,
            lr=1e-3,
            t_eps=1e-3,
            patience=0,
            log_every=1,
            weight_ema_decay=0.0,
            restore_best=False,
        )
        self.assertEqual(len(out["warmup_train_losses"]), 1)
        self.assertEqual(len(out["train_losses"]), 1)
        self.assertTrue(np.isfinite(out["warmup_train_losses"][0]))
        self.assertTrue(np.isfinite(out["train_losses"][0]))
        self.assertTrue(all(p.requires_grad for p in m.parameters()))
        c = compute_ode_time_linear_x_flow_c_matrix(
            model=m,
            theta_all=theta[:4],
            x_all=x[:4],
            device=torch.device("cpu"),
            x_mean=out["x_mean"],
            x_std=out["x_std"],
            quadrature_steps=5,
            ode_steps=2,
            pair_batch_size=32,
        )
        self.assertEqual(tuple(c.shape), (4, 4))
        self.assertTrue(np.all(np.isfinite(c)))

    def test_theta_only_b_low_rank_b_independent_of_t(self) -> None:
        torch.manual_seed(77)
        m = ConditionalTimeThetaOnlyBLowRankCorrectionLinearXFlowMLP(
            theta_dim=1, x_dim=2, correction_rank=1, hidden_dim=8, depth=1, quadrature_steps=5
        )
        th = torch.randn(5, 1)
        t1 = torch.zeros(5, 1)
        t2 = torch.ones(5, 1)
        self.assertTrue(torch.allclose(m.linear.b(th, t1), m.linear.b(th, t2)))

    def test_lr_t_ts_mean_regression_pretrain_then_freeze_b_smoke(self) -> None:
        torch.manual_seed(78)
        rng = np.random.default_rng(78)
        theta = rng.normal(size=(32, 1)).astype(np.float64)
        x = np.concatenate([theta, -theta], axis=1) + 0.05 * rng.normal(size=(32, 2))
        m = ConditionalTimeThetaOnlyBLowRankCorrectionLinearXFlowMLP(
            theta_dim=1, x_dim=2, correction_rank=1, hidden_dim=8, depth=1, quadrature_steps=5
        )
        before_b = {n: p.detach().clone() for n, p in m.linear.b_net.named_parameters()}
        before_a = {n: p.detach().clone() for n, p in m.linear.a_net.named_parameters()}
        before_u = m.U.detach().clone()
        out = train_low_rank_t_theta_only_b_mean_regression_pretrain_then_freeze_b(
            model=m,
            theta_train=theta[:24],
            x_train=x[:24],
            theta_val=theta[24:],
            x_val=x[24:],
            device=torch.device("cpu"),
            schedule=path_schedule_from_name("linear"),
            warmup_epochs=1,
            epochs=1,
            batch_size=8,
            lr=1e-2,
            t_eps=1e-3,
            patience=0,
            log_every=1,
            weight_ema_decay=0.0,
            restore_best=False,
            log_name="linear_x_flow_lr_t_ts_test",
        )
        self.assertEqual(out.get("lxf_low_rank_t_warmup_objective"), "mean_regression")
        self.assertTrue(out.get("lxf_low_rank_t_second_stage_freeze_b_enabled"))
        b_changed = any(
            not torch.allclose(p.detach(), before_b[n]) for n, p in m.linear.b_net.named_parameters()
        )
        self.assertTrue(b_changed)
        a_changed = any(
            not torch.allclose(p.detach(), before_a[n]) for n, p in m.linear.a_net.named_parameters()
        )
        u_changed = not torch.allclose(m.U.detach(), before_u)
        self.assertTrue(a_changed or u_changed)



    def test_time_low_rank_correction_divergence_zero_h_is_trace_A(self) -> None:
        torch.manual_seed(50)
        m = ConditionalTimeLowRankCorrectionLinearXFlowMLP(
            theta_dim=1, x_dim=3, correction_rank=2, hidden_dim=8, depth=1, quadrature_steps=6
        )
        th = torch.randn(4, 1)
        x = torch.randn(4, 3)
        t = torch.full((4, 1), 0.5)
        with torch.no_grad():
            a_mat = m.linear.A(t)
            tr = torch.diagonal(a_mat, dim1=-2, dim2=-1).sum(dim=1)
        div = m.divergence(x, th, t)
        self.assertTrue(torch.allclose(div, tr, atol=1e-6))

    def test_time_low_rank_correction_divergence_exact_kwarg_finite(self) -> None:
        torch.manual_seed(52)
        m = ConditionalTimeLowRankCorrectionLinearXFlowMLP(
            theta_dim=1,
            x_dim=3,
            correction_rank=2,
            hidden_dim=8,
            depth=1,
            quadrature_steps=6,
            divergence_estimator="exact",
            hutchinson_probes=3,
        )
        th = torch.randn(3, 1)
        x = torch.randn(3, 3)
        t = torch.full((3, 1), 0.4)
        div = m.divergence(x, th, t)
        self.assertEqual(tuple(div.shape), (3,))
        self.assertTrue(torch.all(torch.isfinite(div)))

    def test_time_low_rank_correction_hutchinson_mean_near_exact(self) -> None:
        torch.manual_seed(53)
        m_exact = ConditionalTimeLowRankCorrectionLinearXFlowMLP(
            theta_dim=1,
            x_dim=3,
            correction_rank=3,
            hidden_dim=12,
            depth=1,
            quadrature_steps=6,
            divergence_estimator="exact",
        )
        m_h = ConditionalTimeLowRankCorrectionLinearXFlowMLP(
            theta_dim=1,
            x_dim=3,
            correction_rank=3,
            hidden_dim=12,
            depth=1,
            quadrature_steps=6,
            divergence_estimator="hutchinson",
            hutchinson_probes=512,
        )
        m_h.load_state_dict(m_exact.state_dict())
        th = torch.randn(4, 1)
        x = torch.randn(4, 3)
        t = torch.full((4, 1), 0.37)
        d_e = m_exact.divergence(x, th, t)
        torch.manual_seed(2026)
        d_h = m_h.divergence(x, th, t)
        self.assertTrue(torch.all(torch.isfinite(d_h)))
        self.assertTrue(torch.allclose(d_e, d_h, rtol=0.25, atol=0.12))

    def test_time_low_rank_correction_log_prob_and_c_matrix_finite(self) -> None:
        torch.manual_seed(51)
        m = ConditionalTimeLowRankCorrectionLinearXFlowMLP(
            theta_dim=1, x_dim=2, correction_rank=1, hidden_dim=8, depth=1, quadrature_steps=8
        )
        th = torch.randn(3, 1)
        x = torch.randn(3, 2)
        lp = m.log_prob_normalized(x, th, quadrature_steps=8, ode_steps=2)
        self.assertEqual(tuple(lp.shape), (3,))
        self.assertTrue(torch.all(torch.isfinite(lp)))
        theta_np = th.detach().cpu().numpy().astype(np.float64)
        x_np = x.detach().cpu().numpy().astype(np.float64)
        xm = np.zeros(2, dtype=np.float64)
        xs = np.ones(2, dtype=np.float64)
        c = compute_ode_time_linear_x_flow_c_matrix(
            model=m,
            theta_all=theta_np,
            x_all=x_np,
            device=torch.device("cpu"),
            x_mean=xm,
            x_std=xs,
            quadrature_steps=8,
            ode_steps=2,
            pair_batch_size=64,
        )
        self.assertEqual(tuple(c.shape), (3, 3))
        self.assertTrue(np.all(np.isfinite(c)))







