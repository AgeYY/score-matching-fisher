from __future__ import annotations

import argparse
import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from fisher.cli_shared_fisher import add_estimation_arguments
from fisher.h_matrix import HMatrixEstimator
from fisher.models import ConditionalXFlowVelocity, ConditionalXFlowVelocityThetaFourierMLP
from fisher.shared_fisher_est import effective_flow_x_theta_fourier_omega, validate_estimation_args


class _ThetaInvariantXFlow(ConditionalXFlowVelocity):
    """v(x,θ,t) = -x: divergence constant in x; independent of θ."""

    def __init__(self) -> None:
        super().__init__(x_dim=3, hidden_dim=8, depth=1, use_logit_time=True)

    def forward(self, x_t: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _ = theta, t
        return -x_t


class _ShiftedXFlow(ConditionalXFlowVelocity):
    def __init__(self) -> None:
        super().__init__(x_dim=3, hidden_dim=8, depth=1, use_logit_time=True)

    def forward(self, x_t: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return -x_t + 0.03 * theta + 0.02 * t


class _ThetaInvariantFourierXFlow(ConditionalXFlowVelocityThetaFourierMLP):
    """Same invariant trick as _ThetaInvariantXFlow but using the Fourier-theta MLP class."""

    def __init__(self) -> None:
        super().__init__(
            x_dim=3,
            hidden_dim=8,
            depth=1,
            use_logit_time=True,
            theta_fourier_k=2,
            theta_fourier_omega=1.0,
        )

    def forward(self, x_t: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _ = theta, t
        return -x_t


class TestHMatrixFlowXLikelihood(unittest.TestCase):
    def test_flow_x_theta_invariant_velocity_yields_small_offdiag_delta(self) -> None:
        post = _ThetaInvariantXFlow()
        est = HMatrixEstimator(
            model_post=post,
            model_prior=None,
            sigma_eval=1.0,
            device=torch.device("cpu"),
            pair_batch_size=64,
            field_method="flow_x_likelihood",
            flow_scheduler="cosine",
            flow_ode_steps=24,
        )
        theta = np.linspace(-1.0, 1.0, 5, dtype=np.float64).reshape(-1, 1)
        x = np.stack([np.sin(theta.reshape(-1)), np.cos(theta.reshape(-1)), 0.1 * theta.reshape(-1)], axis=1).astype(
            np.float64
        )
        out = est.run(theta=theta, x=x, restore_original_order=False)
        self.assertEqual(out.field_method, "flow_x_likelihood")
        self.assertEqual(out.eval_scalar_name, "flow_ode_t_span")
        self.assertEqual(out.flow_score_mode, "direct_ode_x_cond_likelihood")
        n = theta.shape[0]
        d = out.delta_l_matrix
        off = d[~np.eye(n, dtype=bool)]
        self.assertLess(float(np.max(np.abs(off))), 1e-5)

    def test_flow_x_shifted_outputs_finite(self) -> None:
        post = _ShiftedXFlow()
        est = HMatrixEstimator(
            model_post=post,
            model_prior=None,
            sigma_eval=1.0,
            device=torch.device("cpu"),
            pair_batch_size=64,
            field_method="flow_x_likelihood",
            flow_scheduler="cosine",
            flow_ode_steps=16,
        )
        theta = np.linspace(-1.2, 1.2, 6, dtype=np.float64).reshape(-1, 1)
        x = np.stack([np.sin(theta.reshape(-1)), np.cos(theta.reshape(-1)), 0.2 * theta.reshape(-1)], axis=1).astype(
            np.float64
        )
        out = est.run(theta=theta, x=x, restore_original_order=False)
        self.assertTrue(np.isfinite(out.c_matrix).all())
        self.assertTrue(np.isfinite(out.delta_l_matrix).all())
        self.assertTrue(np.isfinite(out.h_sym).all())

    def test_flow_x_requires_x_flow_model(self) -> None:
        bad = torch.nn.Linear(1, 1)
        with self.assertRaises(TypeError):
            HMatrixEstimator(
                model_post=bad,  # type: ignore[arg-type]
                model_prior=None,
                sigma_eval=1.0,
                device=torch.device("cpu"),
                pair_batch_size=8,
                field_method="flow_x_likelihood",
                flow_ode_steps=8,
            )

    def test_validate_estimation_args_accepts_flow_x_likelihood(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args([])
        args.theta_field_method = "flow_x_likelihood"
        validate_estimation_args(args)

    def test_effective_theta_fourier_omega_theta_range(self) -> None:
        ns = SimpleNamespace(
            flow_x_theta_fourier_omega_mode="theta_range",
            flow_x_theta_fourier_omega=1.0,
            theta_low=-6.0,
            theta_high=6.0,
        )
        om, _log = effective_flow_x_theta_fourier_omega(ns)
        self.assertAlmostEqual(float(om), 2.0 * math.pi / 12.0, places=10)

    def test_theta_fourier_mlp_forward_shape(self) -> None:
        m = ConditionalXFlowVelocityThetaFourierMLP(
            x_dim=4,
            hidden_dim=16,
            depth=2,
            theta_fourier_k=3,
            theta_fourier_omega=0.7,
        )
        b = 5
        x_t = torch.randn(b, 4)
        theta = torch.randn(b, 1)
        t = torch.rand(b, 1)
        out = m(x_t, theta, t)
        self.assertEqual(tuple(out.shape), (b, 4))
        self.assertTrue(torch.isfinite(out).all())

    def test_flow_x_fourier_theta_invariant_velocity_yields_small_offdiag_delta(self) -> None:
        post = _ThetaInvariantFourierXFlow()
        est = HMatrixEstimator(
            model_post=post,
            model_prior=None,
            sigma_eval=1.0,
            device=torch.device("cpu"),
            pair_batch_size=64,
            field_method="flow_x_likelihood",
            flow_scheduler="cosine",
            flow_ode_steps=24,
        )
        theta = np.linspace(-1.0, 1.0, 5, dtype=np.float64).reshape(-1, 1)
        x = np.stack([np.sin(theta.reshape(-1)), np.cos(theta.reshape(-1)), 0.1 * theta.reshape(-1)], axis=1).astype(
            np.float64
        )
        out = est.run(theta=theta, x=x, restore_original_order=False)
        n = theta.shape[0]
        d = out.delta_l_matrix
        off = d[~np.eye(n, dtype=bool)]
        self.assertLess(float(np.max(np.abs(off))), 1e-5)

    def test_validate_rejects_theta_fourier_arch_for_dsm(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args([])
        args.theta_field_method = "dsm"
        args.flow_score_arch = "theta_fourier_mlp"
        with self.assertRaises(ValueError):
            validate_estimation_args(args)

    def test_validate_rejects_empty_theta_fourier_features(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args(
            [
                "--theta-field-method",
                "flow_x_likelihood",
                "--flow-score-arch",
                "theta_fourier_mlp",
                "--flow-x-theta-fourier-k",
                "0",
                "--flow-x-theta-fourier-no-linear",
                "--flow-x-theta-fourier-no-bias",
            ]
        )
        with self.assertRaises(ValueError):
            validate_estimation_args(args)


if __name__ == "__main__":
    unittest.main()
