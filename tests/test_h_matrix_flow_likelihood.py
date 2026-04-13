from __future__ import annotations

import argparse
import unittest

import numpy as np
import torch

from fisher.cli_shared_fisher import add_estimation_arguments
from fisher.h_matrix import HMatrixEstimator
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
            field_method="flow_likelihood",
            flow_scheduler="cosine",
            flow_ode_steps=32,
        )
        theta = np.linspace(-1.0, 1.0, 6, dtype=np.float64).reshape(-1, 1)
        x = np.stack([np.sin(theta.reshape(-1)), np.cos(theta.reshape(-1))], axis=1).astype(np.float64)
        out = est.run(theta=theta, x=x, restore_original_order=False)
        self.assertEqual(out.field_method, "flow_likelihood")
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
            field_method="flow_likelihood",
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

    def test_validate_estimation_args_accepts_flow_likelihood(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args([])
        args.theta_field_method = "flow_likelihood"
        validate_estimation_args(args)


if __name__ == "__main__":
    unittest.main()
