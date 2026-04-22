from __future__ import annotations

import argparse
import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from fisher.cli_shared_fisher import add_estimation_arguments
from fisher.h_matrix import HMatrixEstimator
from fisher.models import (
    ConditionalThetaFlowVelocitySoftMoE,
    ConditionalThetaFlowVelocityThetaFourierFiLMPerLayer,
    ConditionalThetaFlowVelocityThetaFourierMLP,
    PriorThetaFlowVelocityThetaFourierFiLMPerLayer,
    PriorThetaFlowVelocityThetaFourierMLP,
)
from fisher.shared_fisher_est import (
    effective_flow_theta_fourier_omega_post,
    effective_flow_theta_fourier_omega_prior,
    validate_estimation_args,
)


class TestThetaFlowThetaFourier(unittest.TestCase):
    def test_conditional_theta_soft_moe_forward_shape(self) -> None:
        m = ConditionalThetaFlowVelocitySoftMoE(
            x_dim=4,
            hidden_dim=16,
            depth=2,
            theta_dim=1,
            num_experts=3,
            router_temperature=0.8,
        )
        b = 5
        theta_t = torch.randn(b, 1)
        x = torch.randn(b, 4)
        t = torch.rand(b, 1)
        out = m(theta_t, x, t)
        self.assertEqual(tuple(out.shape), (b, 1))
        self.assertTrue(torch.isfinite(out).all())

    def test_conditional_forward_shape(self) -> None:
        m = ConditionalThetaFlowVelocityThetaFourierMLP(
            x_dim=4,
            hidden_dim=16,
            depth=2,
            theta_fourier_k=3,
            theta_fourier_omega=0.7,
        )
        b = 5
        theta_t = torch.randn(b, 1)
        x = torch.randn(b, 4)
        t = torch.rand(b, 1)
        out = m(theta_t, x, t)
        self.assertEqual(tuple(out.shape), (b, 1))
        self.assertTrue(torch.isfinite(out).all())

    def test_prior_forward_shape(self) -> None:
        m = PriorThetaFlowVelocityThetaFourierMLP(
            hidden_dim=16,
            depth=2,
            theta_fourier_k=3,
            theta_fourier_omega=0.7,
        )
        b = 5
        theta_t = torch.randn(b, 1)
        t = torch.rand(b, 1)
        out = m(theta_t, t)
        self.assertEqual(tuple(out.shape), (b, 1))
        self.assertTrue(torch.isfinite(out).all())

    def test_effective_omega_theta_range_post_and_prior(self) -> None:
        ns = SimpleNamespace(
            flow_theta_fourier_omega_mode="theta_range",
            flow_theta_fourier_omega=1.0,
            theta_low=-2.0,
            theta_high=4.0,
        )
        om, _ = effective_flow_theta_fourier_omega_post(ns)
        self.assertAlmostEqual(float(om), 2.0 * math.pi / 6.0, places=10)
        ns2 = SimpleNamespace(
            flow_prior_theta_fourier_omega_mode="theta_range",
            flow_prior_theta_fourier_omega=2.0,
            theta_low=0.0,
            theta_high=10.0,
        )
        om2, _ = effective_flow_theta_fourier_omega_prior(ns2)
        self.assertAlmostEqual(float(om2), 2.0 * (2.0 * math.pi / 10.0), places=10)

    def test_validate_accepts_theta_flow_film_fourier(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args(
            [
                "--theta-field-method",
                "theta_flow",
                "--flow-arch",
                "film_fourier",
            ]
        )
        validate_estimation_args(args)

    def test_validate_accepts_theta_flow_soft_moe(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args(
            [
                "--theta-field-method",
                "theta_flow",
                "--flow-arch",
                "soft_moe",
                "--flow-moe-num-experts",
                "3",
                "--flow-moe-router-temperature",
                "0.7",
            ]
        )
        validate_estimation_args(args)

    def test_validate_rejects_empty_theta_fourier_features_posterior(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args(
            [
                "--theta-field-method",
                "theta_flow",
                "--flow-arch",
                "film_fourier",
                "--flow-theta-fourier-k",
                "0",
                "--flow-theta-fourier-no-linear",
                "--flow-theta-fourier-no-bias",
            ]
        )
        with self.assertRaises(ValueError):
            validate_estimation_args(args)

    def test_validate_rejects_legacy_flow_likelihood_method(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args(["--theta-field-method", "flow_likelihood"])
        with self.assertRaises(ValueError) as ctx:
            validate_estimation_args(args)
        self.assertIn("theta_path_integral", str(ctx.exception))

    def test_validate_rejects_legacy_flow_method(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args(["--theta-field-method", "flow"])
        with self.assertRaises(ValueError) as ctx:
            validate_estimation_args(args)
        self.assertIn("theta_flow", str(ctx.exception))

    def test_conditional_theta_film_fourier_forward_shape(self) -> None:
        m = ConditionalThetaFlowVelocityThetaFourierFiLMPerLayer(
            x_dim=4,
            hidden_dim=16,
            depth=2,
            theta_fourier_k=3,
            theta_fourier_omega=0.7,
        )
        b = 5
        theta_t = torch.randn(b, 1)
        x = torch.randn(b, 4)
        t = torch.rand(b, 1)
        out = m(theta_t, x, t)
        self.assertEqual(tuple(out.shape), (b, 1))
        self.assertTrue(torch.isfinite(out).all())

    def test_prior_theta_film_fourier_forward_shape(self) -> None:
        m = PriorThetaFlowVelocityThetaFourierFiLMPerLayer(
            hidden_dim=16,
            depth=2,
            theta_fourier_k=3,
            theta_fourier_omega=0.7,
        )
        b = 5
        theta_t = torch.randn(b, 1)
        t = torch.rand(b, 1)
        out = m(theta_t, t)
        self.assertEqual(tuple(out.shape), (b, 1))
        self.assertTrue(torch.isfinite(out).all())

    def test_flow_likelihood_h_matrix_smoke_fourier_models(self) -> None:
        torch.manual_seed(0)
        post = ConditionalThetaFlowVelocityThetaFourierMLP(
            x_dim=2,
            hidden_dim=16,
            depth=2,
            theta_fourier_k=2,
            theta_fourier_omega=1.0,
        )
        prior = PriorThetaFlowVelocityThetaFourierMLP(
            hidden_dim=16,
            depth=2,
            theta_fourier_k=2,
            theta_fourier_omega=1.0,
        )
        est = HMatrixEstimator(
            model_post=post,
            model_prior=prior,
            sigma_eval=1.0,
            device=torch.device("cpu"),
            pair_batch_size=32,
            field_method="theta_flow",
            flow_scheduler="cosine",
            flow_ode_steps=16,
        )
        theta = np.linspace(-0.5, 0.5, 4, dtype=np.float64).reshape(-1, 1)
        x = np.stack([np.sin(theta.reshape(-1)), np.cos(theta.reshape(-1))], axis=1).astype(np.float64)
        out = est.run(theta=theta, x=x, restore_original_order=False)
        self.assertTrue(np.isfinite(out.c_matrix).all())
        self.assertTrue(np.isfinite(out.delta_l_matrix).all())
        self.assertTrue(np.isfinite(out.h_sym).all())


if __name__ == "__main__":
    unittest.main()
