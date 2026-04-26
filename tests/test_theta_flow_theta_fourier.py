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
    ConditionalThetaFlowVelocityTransformer,
    PriorThetaFlowVelocityThetaFourierFiLMPerLayer,
    PriorThetaFlowVelocityThetaFourierMLP,
    PriorThetaFlowVelocityTransformer,
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

    def test_conditional_theta_soft_moe_shared_backbone_linear_heads(self) -> None:
        """Soft-MoE uses one shared SiLU backbone and one Linear per expert (not deep expert MLPs)."""
        m = ConditionalThetaFlowVelocitySoftMoE(
            x_dim=4,
            hidden_dim=16,
            depth=2,
            theta_dim=1,
            num_experts=4,
            router_temperature=1.0,
        )
        self.assertIsInstance(m.backbone, torch.nn.Sequential)
        self.assertEqual(len(list(m.backbone.children())), 4)  # 2×(Linear, SiLU)
        self.assertEqual(len(m.expert_heads), 4)
        for h in m.expert_heads:
            self.assertIsInstance(h, torch.nn.Linear)
            self.assertEqual(h.in_features, 16)
            self.assertEqual(h.out_features, 1)
        self.assertIsInstance(m.router, torch.nn.Linear)
        self.assertEqual(m.router.out_features, 4)

    def test_conditional_theta_soft_moe_depth_zero_identity_backbone(self) -> None:
        m = ConditionalThetaFlowVelocitySoftMoE(
            x_dim=4,
            hidden_dim=16,
            depth=0,
            theta_dim=1,
            num_experts=2,
        )
        self.assertIsInstance(m.backbone, torch.nn.Identity)
        self.assertEqual(len(m.expert_heads), 2)
        # feats dim = 1 + 4 + 1 = 6
        self.assertEqual(m.expert_heads[0].in_features, 6)
        b = 3
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

    def test_conditional_theta_transformer_forward_shape(self) -> None:
        m = ConditionalThetaFlowVelocityTransformer(
            x_dim=4,
            hidden_dim=16,
            depth=1,
            theta_dim=2,
            num_heads=4,
            ff_mult=2,
            dropout=0.0,
            x_tokens=3,
        )
        b = 5
        theta_t = torch.randn(b, 2)
        x = torch.randn(b, 4)
        t = torch.rand(b, 1)
        out = m(theta_t, x, t)
        self.assertEqual(tuple(out.shape), (b, 2))
        self.assertTrue(torch.isfinite(out).all())

    def test_prior_theta_transformer_forward_shape(self) -> None:
        m = PriorThetaFlowVelocityTransformer(
            hidden_dim=16,
            depth=1,
            theta_dim=2,
            num_heads=4,
            ff_mult=2,
            dropout=0.0,
            latent_tokens=3,
        )
        b = 5
        theta_t = torch.randn(b, 2)
        t = torch.rand(b, 1)
        out = m(theta_t, t)
        self.assertEqual(tuple(out.shape), (b, 2))
        self.assertTrue(torch.isfinite(out).all())

    def test_validate_accepts_theta_flow_transformer_methods(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        for method in ("theta_flow", "theta_flow_reg", "theta_flow_pre_post"):
            args = parser.parse_args(
                [
                    "--theta-field-method",
                    method,
                    "--flow-arch",
                    "transformer",
                    "--flow-hidden-dim",
                    "16",
                    "--prior-hidden-dim",
                    "16",
                    "--flow-transformer-heads",
                    "4",
                    "--flow-transformer-x-tokens",
                    "3",
                    "--flow-prior-transformer-latent-tokens",
                    "2",
                ]
            )
            validate_estimation_args(args)

    def test_validate_rejects_transformer_bad_head_divisibility(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args(
            [
                "--theta-field-method",
                "theta_flow",
                "--flow-arch",
                "transformer",
                "--flow-hidden-dim",
                "10",
                "--prior-hidden-dim",
                "16",
                "--flow-transformer-heads",
                "4",
            ]
        )
        with self.assertRaises(ValueError):
            validate_estimation_args(args)

    def test_validate_rejects_transformer_for_x_flow(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args(["--theta-field-method", "x_flow", "--flow-arch", "transformer"])
        with self.assertRaises(ValueError):
            validate_estimation_args(args)

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
