"""Smoke tests for pair-conditioned CTSM-v (CPU, no training loop)."""
from __future__ import annotations

import unittest

import torch

from fisher.ctsm_models import ToyPairConditionedTimeScoreNet
from fisher.ctsm_objectives import ctsm_v_pair_conditioned_loss, estimate_log_ratio_trapz_pair
from fisher.ctsm_paths import TwoSB


class TestPairConditionedCtsm(unittest.TestCase):
    def test_pair_loss_and_trapz_shapes(self) -> None:
        torch.manual_seed(0)
        dim = 2
        batch = 16
        prob_path = TwoSB(dim=dim, var=2.0)
        model = ToyPairConditionedTimeScoreNet(dim=dim, hidden_dim=32)
        x0 = torch.randn(batch, dim)
        x1 = torch.randn(batch, dim)
        a = torch.linspace(-1.0, 1.0, batch).unsqueeze(-1)
        b_theta = torch.linspace(-0.5, 0.5, batch).unsqueeze(-1)
        loss = ctsm_v_pair_conditioned_loss(model, prob_path, x0, x1, a, b_theta)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

        x = torch.randn(batch, dim)
        out = estimate_log_ratio_trapz_pair(model, x, a.squeeze(-1), b_theta.squeeze(-1), n_time=32)
        self.assertEqual(tuple(out.shape), (batch,))
        self.assertTrue(torch.isfinite(out).all())

    def test_pair_loss_with_cosine_schedule_is_finite(self) -> None:
        torch.manual_seed(1)
        dim = 2
        batch = 12
        prob_path = TwoSB(dim=dim, var=2.0, scheduler="cosine")
        model = ToyPairConditionedTimeScoreNet(dim=dim, hidden_dim=32)
        x0 = torch.randn(batch, dim)
        x1 = torch.randn(batch, dim)
        a = torch.linspace(-1.0, 1.0, batch).unsqueeze(-1)
        b_theta = torch.linspace(-0.5, 0.5, batch).unsqueeze(-1)
        loss = ctsm_v_pair_conditioned_loss(model, prob_path, x0, x1, a, b_theta)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
