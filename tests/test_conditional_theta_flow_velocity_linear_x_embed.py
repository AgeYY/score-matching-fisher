"""Tests for ConditionalThetaFlowVelocityLinearXEmbed (theta-flow MLP on [theta, Linear(x), t])."""

from __future__ import annotations

import unittest

import torch

from fisher.models import ConditionalThetaFlowVelocityLinearXEmbed


class TestConditionalThetaFlowVelocityLinearXEmbed(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        b, x_dim, theta_dim = 7, 4, 2
        m = ConditionalThetaFlowVelocityLinearXEmbed(
            x_dim=x_dim,
            hidden_dim=16,
            depth=2,
            use_logit_time=True,
            theta_dim=theta_dim,
        )
        self.assertEqual(m.x_embed.weight.shape, (1, x_dim))
        theta_t = torch.randn(b, theta_dim)
        x = torch.randn(b, x_dim)
        t = torch.rand(b, 1)
        out = m.forward(theta_t, x, t)
        self.assertEqual(tuple(out.shape), (b, theta_dim))

    def test_predict_velocity_shape(self) -> None:
        m = ConditionalThetaFlowVelocityLinearXEmbed(x_dim=3, hidden_dim=8, depth=1, theta_dim=1)
        theta = torch.randn(5, 1)
        x = torch.randn(5, 3)
        v = m.predict_velocity(theta, x, 0.8)
        self.assertEqual(tuple(v.shape), (5, 1))


if __name__ == "__main__":
    unittest.main()
