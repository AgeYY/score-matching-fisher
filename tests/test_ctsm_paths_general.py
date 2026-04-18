from __future__ import annotations

import math
import unittest

import torch

from fisher.ctsm_paths import CosineScheduler, LinearScheduler, TwoEndpointBridge


class TestCtsmPathsGeneral(unittest.TestCase):
    def test_scheduler_value_and_derivative(self) -> None:
        t = torch.tensor([[0.0], [0.25], [0.5], [1.0]], dtype=torch.float64)

        linear = LinearScheduler()
        self.assertTrue(torch.allclose(linear.value(t), t))
        self.assertTrue(torch.allclose(linear.derivative(t), torch.ones_like(t)))

        cosine = CosineScheduler()
        v = cosine.value(t)
        d = cosine.derivative(t)
        self.assertAlmostEqual(float(v[0].item()), 0.0, places=12)
        self.assertAlmostEqual(float(v[-1].item()), 1.0, places=12)
        self.assertAlmostEqual(float(d[0].item()), 0.0, places=12)
        self.assertAlmostEqual(float(d[-1].item()), 0.0, places=12)

    def test_marginal_prob_linear_matches_closed_form(self) -> None:
        torch.manual_seed(0)
        x0 = torch.randn(6, 3)
        x1 = torch.randn(6, 3)
        t = torch.linspace(0.1, 0.9, 6).unsqueeze(-1)
        var = 2.0

        path = TwoEndpointBridge(dim=3, var=var, scheduler="linear")
        mean, std, out_var = path.marginal_prob(x0, x1, t)

        mean_ref = (1.0 - t) * x0 + t * x1
        var_ref = t * (1.0 - t) * var
        std_ref = torch.sqrt(var_ref)
        self.assertTrue(torch.allclose(mean, mean_ref, atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(out_var, var_ref, atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(std, std_ref, atol=1e-7, rtol=1e-6))

    def test_linear_weighted_target_matches_old_formula(self) -> None:
        torch.manual_seed(1)
        batch = 10
        dim = 4
        x0 = torch.randn(batch, dim)
        x1 = torch.randn(batch, dim)
        epsilon = torch.randn(batch, dim)
        t = torch.rand(batch, 1) * 0.8 + 0.1
        factor = 1.0
        var = 2.0

        path = TwoEndpointBridge(dim=dim, var=var, scheduler="linear")
        lambda_new, targets_new = path.full_epsilon_target(epsilon=epsilon, x0=x0, x1=x1, t=t, factor=factor)

        sigma = math.sqrt(var)
        sqrt2 = math.sqrt(2.0)
        temp1 = torch.sqrt(1.0 - 4.0 * t + 4.0 * t**2 + 2.0 * factor * t - 2.0 * factor * t**2)
        lambda_old = sqrt2 * t * (1.0 - t) / temp1
        temp2 = (1.0 - 2.0 * t) / temp1
        mut_d = x1 - x0
        targets_old = (
            -temp2 / sqrt2
            + temp2 / sqrt2 * torch.square(epsilon)
            + sqrt2 * torch.sqrt(t * (1.0 - t)) / temp1 / sigma * epsilon * mut_d
        )

        self.assertTrue(torch.allclose(lambda_new, lambda_old, atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(targets_new, targets_old, atol=1e-7, rtol=1e-6))

    def test_cosine_target_is_finite_near_endpoints(self) -> None:
        torch.manual_seed(2)
        batch = 8
        dim = 3
        x0 = torch.randn(batch, dim)
        x1 = torch.randn(batch, dim)
        epsilon = torch.randn(batch, dim)
        t = torch.tensor([[1e-4], [2e-4], [5e-4], [1e-3], [1.0 - 1e-3], [1.0 - 5e-4], [1.0 - 2e-4], [1.0 - 1e-4]])

        path = TwoEndpointBridge(dim=dim, var=2.0, scheduler="cosine")
        lambda_t, targets = path.full_epsilon_target(epsilon=epsilon, x0=x0, x1=x1, t=t, factor=1.0)
        self.assertTrue(torch.isfinite(lambda_t).all())
        self.assertTrue(torch.isfinite(targets).all())


if __name__ == "__main__":
    unittest.main()
