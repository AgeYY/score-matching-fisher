from __future__ import annotations

import argparse
import unittest

import numpy as np
import torch

from fisher.cli_shared_fisher import add_estimation_arguments
from fisher.h_matrix import HMatrixEstimator
from fisher.models import (
    ConditionalThetaFlowVelocity,
    PriorThetaFlowVelocity,
)
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

    def test_theta_flow_posterior_only_uses_log_post_for_c_matrix(self) -> None:
        post = _ShiftedPosteriorFlow()
        prior = _DummyPriorFlow()
        est_bayes = HMatrixEstimator(
            model_post=post,
            model_prior=prior,
            sigma_eval=1.0,
            device=torch.device("cpu"),
            pair_batch_size=64,
            field_method="theta_flow",
            flow_scheduler="cosine",
            flow_ode_steps=16,
            theta_flow_posterior_only_likelihood=False,
        )
        est_postonly = HMatrixEstimator(
            model_post=post,
            model_prior=None,
            sigma_eval=1.0,
            device=torch.device("cpu"),
            pair_batch_size=64,
            field_method="theta_flow",
            flow_scheduler="cosine",
            flow_ode_steps=16,
            theta_flow_posterior_only_likelihood=True,
        )
        theta = np.linspace(-1.2, 1.2, 7, dtype=np.float64).reshape(-1, 1)
        x = np.stack([np.sin(theta.reshape(-1)), np.cos(theta.reshape(-1))], axis=1).astype(np.float64)
        out_b = est_bayes.run(theta=theta, x=x, restore_original_order=False)
        out_p = est_postonly.run(theta=theta, x=x, restore_original_order=False)
        self.assertEqual(out_p.flow_score_mode, "direct_ode_likelihood_posterior_only")
        assert out_b.theta_flow_log_post_matrix is not None
        assert out_b.theta_flow_log_prior_matrix is not None
        assert out_p.theta_flow_log_prior_matrix is None
        np.testing.assert_allclose(out_p.c_matrix, out_b.theta_flow_log_post_matrix, rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(
            out_b.c_matrix,
            out_b.theta_flow_log_post_matrix - out_b.theta_flow_log_prior_matrix,
            rtol=0.0,
            atol=1e-5,
        )

    def test_theta_flow_posterior_only_rejects_nonnone_prior(self) -> None:
        post = _ShiftedPosteriorFlow()
        prior = _DummyPriorFlow()
        with self.assertRaises(ValueError):
            HMatrixEstimator(
                model_post=post,
                model_prior=prior,
                sigma_eval=1.0,
                device=torch.device("cpu"),
                pair_batch_size=64,
                field_method="theta_flow",
                flow_scheduler="cosine",
                flow_ode_steps=8,
                theta_flow_posterior_only_likelihood=True,
            )

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

    def test_theta_flow_mlp_accepts_vector_theta_state(self) -> None:
        torch.manual_seed(2)
        theta_dim = 4
        post = ConditionalThetaFlowVelocity(x_dim=2, hidden_dim=12, depth=1, theta_dim=theta_dim)
        prior = PriorThetaFlowVelocity(hidden_dim=12, depth=1, theta_dim=theta_dim)
        est = HMatrixEstimator(
            model_post=post,
            model_prior=prior,
            sigma_eval=0.8,
            device=torch.device("cpu"),
            pair_batch_size=32,
            field_method="theta_flow",
            flow_scheduler="cosine",
            flow_ode_steps=8,
        )
        # One-hot theta states; run() should preserve 2D theta rows and sort by active bin.
        theta_idx = np.asarray([2, 0, 3, 1, 0], dtype=np.int64)
        theta = np.eye(theta_dim, dtype=np.float64)[theta_idx]
        x = np.random.default_rng(2).standard_normal((theta.shape[0], 2))
        out = est.run(theta=theta, x=x, restore_original_order=False)
        self.assertEqual(out.theta_used.shape, (theta.shape[0], theta_dim))
        self.assertEqual(out.theta_sorted.shape, (theta.shape[0], theta_dim))
        self.assertTrue(np.isfinite(out.h_sym).all())
        sorted_bins = np.argmax(out.theta_sorted, axis=1)
        self.assertTrue(np.all(np.diff(sorted_bins) >= 0))

    def test_theta_flow_mlp_accepts_fourier_vector_theta_state(self) -> None:
        torch.manual_seed(3)
        theta_scalar = np.linspace(-1.0, 1.0, 7, dtype=np.float64)
        k = 3
        period = 2.0 * (float(theta_scalar.max()) - float(theta_scalar.min()))
        w0 = 2.0 * np.pi / max(period, 1e-12)
        theta_cols = []
        for kk in range(1, k + 1):
            theta_cols.append(np.sin(kk * w0 * theta_scalar).reshape(-1, 1))
            theta_cols.append(np.cos(kk * w0 * theta_scalar).reshape(-1, 1))
        theta = np.concatenate(theta_cols, axis=1)

        theta_dim = int(theta.shape[1])
        post = ConditionalThetaFlowVelocity(x_dim=2, hidden_dim=16, depth=2, theta_dim=theta_dim)
        prior = PriorThetaFlowVelocity(hidden_dim=16, depth=2, theta_dim=theta_dim)
        est = HMatrixEstimator(
            model_post=post,
            model_prior=prior,
            sigma_eval=0.9,
            device=torch.device("cpu"),
            pair_batch_size=32,
            field_method="theta_flow",
            flow_scheduler="cosine",
            flow_ode_steps=8,
        )
        x = np.stack([np.sin(theta_scalar), np.cos(theta_scalar)], axis=1).astype(np.float64)
        out = est.run(theta=theta, x=x, restore_original_order=False)
        self.assertEqual(out.theta_used.shape, (theta.shape[0], theta_dim))
        self.assertEqual(out.theta_sorted.shape, (theta.shape[0], theta_dim))
        self.assertTrue(np.isfinite(out.delta_l_matrix).all())
        self.assertTrue(np.isfinite(out.h_sym).all())


if __name__ == "__main__":
    unittest.main()
