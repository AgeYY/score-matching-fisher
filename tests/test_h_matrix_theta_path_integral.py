"""Smoke tests for HMatrixEstimator field_method=theta_path_integral (velocity-to-score + theta-axis integral)."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from fisher.h_matrix import HMatrixEstimator
from fisher.models import ConditionalThetaFlowVelocity, PriorThetaFlowVelocity


class TestHMatrixThetaPathIntegral(unittest.TestCase):
    def test_theta_path_integral_runs_and_finite(self) -> None:
        torch.manual_seed(0)
        post = ConditionalThetaFlowVelocity(x_dim=2, hidden_dim=16, depth=2, use_logit_time=True)
        prior = PriorThetaFlowVelocity(hidden_dim=16, depth=2, use_logit_time=True)
        est = HMatrixEstimator(
            model_post=post,
            model_prior=prior,
            sigma_eval=0.8,
            device=torch.device("cpu"),
            pair_batch_size=64,
            field_method="theta_path_integral",
            flow_scheduler="cosine",
            flow_ode_steps=16,
        )
        theta = np.linspace(-0.5, 0.5, 5, dtype=np.float64).reshape(-1, 1)
        x = np.stack([np.sin(theta.reshape(-1)), np.cos(theta.reshape(-1))], axis=1).astype(np.float64)
        out = est.run(theta=theta, x=x, restore_original_order=False)
        self.assertEqual(out.field_method, "theta_path_integral")
        self.assertEqual(out.eval_scalar_name, "t_eval")
        self.assertEqual(out.flow_score_mode, "velocity_to_epsilon")
        self.assertTrue(np.isfinite(out.g_matrix).all())
        self.assertTrue(np.isfinite(out.c_matrix).all())
        self.assertTrue(np.isfinite(out.delta_l_matrix).all())
        self.assertTrue(np.isfinite(out.h_sym).all())


if __name__ == "__main__":
    unittest.main()
