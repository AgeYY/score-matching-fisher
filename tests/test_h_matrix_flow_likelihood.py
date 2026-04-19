from __future__ import annotations

import argparse
import unittest

import numpy as np
import torch

from fisher.cli_shared_fisher import add_estimation_arguments
from fisher.h_matrix import HMatrixEstimator
from fisher.models import ConditionalThetaFlowVelocityIIDSoft, PriorThetaFlowVelocityIIDSoft
from fisher.shared_fisher_est import validate_estimation_args


class _DummyPosteriorFlow(torch.nn.Module):
    def forward(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _ = x, t
        return -theta_t


class _DummyPriorFlow(torch.nn.Module):
    def forward(self, theta_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _ = t
        return -theta_t


class _ShiftedPosteriorFlow(torch.nn.Module):
    def forward(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return -theta_t + 0.1 * x[:, :1] + 0.05 * t


class TestHMatrixFlowLikelihood(unittest.TestCase):
    def test_flow_likelihood_identical_post_prior_yields_zero_delta(self) -> None:
        post = _DummyPosteriorFlow()
        prior = _DummyPriorFlow()
        est = HMatrixEstimator(
            model_post=post,
            model_prior=prior,
            sigma_eval=1.0,
            device=torch.device("cpu"),
            pair_batch_size=64,
            field_method="theta_flow",
            flow_scheduler="cosine",
            flow_ode_steps=32,
        )
        theta = np.linspace(-1.0, 1.0, 6, dtype=np.float64).reshape(-1, 1)
        x = np.stack([np.sin(theta.reshape(-1)), np.cos(theta.reshape(-1))], axis=1).astype(np.float64)
        out = est.run(theta=theta, x=x, restore_original_order=False)
        self.assertEqual(out.field_method, "theta_flow")
        self.assertEqual(out.eval_scalar_name, "flow_ode_t_span")
        self.assertLess(float(np.max(np.abs(out.delta_l_matrix))), 1e-9)
        self.assertLess(float(np.max(np.abs(out.h_sym))), 1e-9)

    def test_flow_likelihood_outputs_are_finite_for_shifted_models(self) -> None:
        post = _ShiftedPosteriorFlow()
        prior = _DummyPriorFlow()
        est = HMatrixEstimator(
            model_post=post,
            model_prior=prior,
            sigma_eval=1.0,
            device=torch.device("cpu"),
            pair_batch_size=64,
            field_method="theta_flow",
            flow_scheduler="cosine",
            flow_ode_steps=16,
        )
        theta = np.linspace(-1.2, 1.2, 7, dtype=np.float64).reshape(-1, 1)
        x = np.stack([np.sin(theta.reshape(-1)), np.cos(theta.reshape(-1))], axis=1).astype(np.float64)
        out = est.run(theta=theta, x=x, restore_original_order=False)
        self.assertTrue(np.isfinite(out.c_matrix).all())
        self.assertTrue(np.isfinite(out.delta_l_matrix).all())
        self.assertTrue(np.isfinite(out.h_sym).all())
        self.assertEqual(out.flow_score_mode, "direct_ode_likelihood")

    def test_validate_estimation_args_accepts_theta_flow(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args(["--theta-field-method", "theta_flow", "--flow-arch", "mlp"])
        validate_estimation_args(args)

    def test_validate_estimation_args_accepts_theta_path_integral(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args(["--theta-field-method", "theta_path_integral", "--flow-arch", "mlp"])
        validate_estimation_args(args)

    def test_validate_accepts_theta_flow_iid_soft(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args(["--theta-field-method", "theta_flow", "--flow-arch", "iid_soft"])
        validate_estimation_args(args)

    def test_validate_accepts_theta_path_integral_iid_soft(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args(["--theta-field-method", "theta_path_integral", "--flow-arch", "iid_soft"])
        validate_estimation_args(args)

    def test_theta_iid_soft_forward_shapes(self) -> None:
        torch.manual_seed(0)
        post = ConditionalThetaFlowVelocityIIDSoft(
            x_dim=2, hidden_dim=16, depth=2, interaction_hidden_dim=12, interaction_depth=2
        )
        prior = PriorThetaFlowVelocityIIDSoft(
            hidden_dim=16, depth=2, interaction_hidden_dim=12, interaction_depth=2
        )
        b = 5
        th = torch.randn(b, 1)
        x = torch.randn(b, 2)
        t = torch.rand(b, 1)
        self.assertEqual(post(th, x, t).shape, (b, 1))
        self.assertEqual(prior(th, t).shape, (b, 1))

        d = 4
        post_m = ConditionalThetaFlowVelocityIIDSoft(
            x_dim=2,
            theta_dim=d,
            hidden_dim=16,
            depth=2,
            interaction_hidden_dim=12,
            interaction_depth=2,
        )
        prior_m = PriorThetaFlowVelocityIIDSoft(
            theta_dim=d, hidden_dim=16, depth=2, interaction_hidden_dim=12, interaction_depth=2
        )
        th4 = torch.randn(b, d)
        self.assertEqual(post_m(th4, x, t).shape, (b, d))
        self.assertEqual(prior_m(th4, t).shape, (b, d))

        post_x1 = ConditionalThetaFlowVelocityIIDSoft(
            x_dim=1, hidden_dim=8, depth=1, theta_dim=1, alpha_init=0.0, alpha_learnable=False
        )
        x1col = torch.randn(b, 1)
        self.assertEqual(post_x1(th, x1col, t).shape, (b, 1))

    def test_theta_iid_soft_posterior_pools_over_x_not_theta(self) -> None:
        """With alpha=0, v is mean_i phi(theta~, x_i); perturbing one x_i changes v (theta_dim=1)."""
        torch.manual_seed(1)
        m = ConditionalThetaFlowVelocityIIDSoft(
            x_dim=3,
            hidden_dim=16,
            depth=2,
            theta_dim=1,
            alpha_init=0.0,
            alpha_learnable=False,
        )
        th = torch.randn(4, 1)
        x0 = torch.randn(4, 3)
        t = torch.rand(4, 1)
        o0 = m(th, x0, t)
        x1 = x0.clone()
        x1[:, 0] = x1[:, 0] + 20.0
        o1 = m(th, x1, t)
        self.assertFalse(torch.allclose(o0, o1))
        x2 = x0.clone()
        x2[:, 2] = x2[:, 2] + 20.0
        o2 = m(th, x2, t)
        self.assertFalse(torch.allclose(o0, o2))

    def test_validate_rejects_negative_theta_iid_mult(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args(
            [
                "--theta-field-method",
                "theta_flow",
                "--flow-arch",
                "iid_soft",
                "--flow-theta-iid-interaction-hidden-mult",
                "-1",
            ]
        )
        with self.assertRaises(ValueError):
            validate_estimation_args(args)

    def test_theta_flow_iid_soft_estimator_finite(self) -> None:
        torch.manual_seed(0)
        post = ConditionalThetaFlowVelocityIIDSoft(x_dim=2, hidden_dim=12, depth=1)
        prior = PriorThetaFlowVelocityIIDSoft(hidden_dim=12, depth=1)
        est = HMatrixEstimator(
            model_post=post,
            model_prior=prior,
            sigma_eval=0.9,
            device=torch.device("cpu"),
            pair_batch_size=32,
            field_method="theta_flow",
            flow_scheduler="cosine",
            flow_ode_steps=12,
        )
        theta = np.linspace(-0.4, 0.4, 4, dtype=np.float64).reshape(-1, 1)
        x = np.stack([np.sin(theta.reshape(-1)), np.cos(theta.reshape(-1))], axis=1).astype(np.float64)
        out = est.run(theta=theta, x=x, restore_original_order=False)
        self.assertTrue(np.isfinite(out.h_sym).all())


if __name__ == "__main__":
    unittest.main()
