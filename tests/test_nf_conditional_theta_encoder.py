"""Smoke tests for ConditionalThetaNF x encoders (requires zuko)."""

from __future__ import annotations

import unittest

import torch

from fisher.nf_hellinger import ConditionalThetaNF, require_zuko_for_nf


class TestConditionalThetaNFXEncoder(unittest.TestCase):
    def setUp(self) -> None:
        try:
            require_zuko_for_nf()
        except RuntimeError as e:
            raise unittest.SkipTest(str(e)) from e

    def test_linear_encoder_log_prob_shape(self) -> None:
        m = ConditionalThetaNF(
            x_dim=10,
            context_dim=1,
            hidden_dim=16,
            transforms=2,
            x_encoder="linear",
        )
        b = 11
        x = torch.randn(b, 10)
        theta = torch.randn(b, 1)
        lp = m.log_prob(theta, x)
        self.assertEqual(tuple(lp.shape), (b,))
        self.assertEqual(m.encoder.weight.shape, (1, 10))

    def test_mlp_encoder_matches_context_dim(self) -> None:
        m = ConditionalThetaNF(
            x_dim=5,
            context_dim=8,
            hidden_dim=16,
            transforms=2,
            x_encoder="mlp",
        )
        b = 4
        x = torch.randn(b, 5)
        theta = torch.randn(b, 1)
        lp = m.log_prob(theta, x)
        self.assertEqual(tuple(lp.shape), (b,))


if __name__ == "__main__":
    unittest.main()
