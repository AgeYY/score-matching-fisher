"""Tests for torchdiffeq-based CTSM log-ratio integration."""

from __future__ import annotations

import unittest

import torch

from fisher.ctsm_objectives import estimate_log_ratio_torch, estimate_log_ratio_trapz


class DeterministicScalarTimeScore(torch.nn.Module):
    def __init__(self, alpha: float = 0.2, beta: float = 0.7, gamma: float = -0.1) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self._dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        x_term = self.alpha * x.sum(dim=-1, keepdim=True)
        return x_term + self.beta * t + self.gamma


class TestEstimateLogRatioTorch(unittest.TestCase):
    def test_shape_dtype_device(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DeterministicScalarTimeScore().to(device)
        x = torch.randn(8, 3, device=device, dtype=torch.float32)

        out = estimate_log_ratio_torch(model, x, eps1=1e-3, eps2=2e-3, n_steps=128)

        self.assertEqual(tuple(out.shape), (x.shape[0],))
        self.assertEqual(out.dtype, x.dtype)
        self.assertEqual(out.device, x.device)
        self.assertTrue(torch.isfinite(out).all())

    def test_matches_known_integral(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DeterministicScalarTimeScore(alpha=0.3, beta=-0.6, gamma=0.2).to(device)
        x = torch.randn(16, 4, device=device, dtype=torch.float64)
        eps1 = 0.01
        eps2 = 0.02
        t0 = eps1
        t1 = 1.0 - eps2

        out = estimate_log_ratio_torch(model, x, eps1=eps1, eps2=eps2, n_steps=512)

        x_term = model.alpha * x.sum(dim=-1)
        expected = x_term * (t1 - t0) + 0.5 * model.beta * (t1**2 - t0**2) + model.gamma * (t1 - t0)
        self.assertTrue(torch.allclose(out, expected, atol=1e-8, rtol=1e-8))

    def test_close_to_trapz(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DeterministicScalarTimeScore(alpha=0.11, beta=0.37, gamma=-0.05).to(device)
        x = torch.randn(32, 5, device=device, dtype=torch.float32)
        eps1 = 1e-4
        eps2 = 1e-4

        out_torch = estimate_log_ratio_torch(model, x, eps1=eps1, eps2=eps2, n_steps=512)
        out_trapz = estimate_log_ratio_trapz(model, x, eps1=eps1, eps2=eps2, n_time=4096)

        self.assertTrue(torch.allclose(out_torch, out_trapz, atol=2e-4, rtol=2e-4))

    def test_invalid_args(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DeterministicScalarTimeScore().to(device)
        x = torch.randn(4, 2, device=device)

        with self.assertRaises(ValueError):
            _ = estimate_log_ratio_torch(model, x, n_steps=0)
        with self.assertRaises(ValueError):
            _ = estimate_log_ratio_torch(model, x, eps1=-1e-3, eps2=1e-3)
        with self.assertRaises(ValueError):
            _ = estimate_log_ratio_torch(model, x, eps1=0.6, eps2=0.4)


if __name__ == "__main__":
    unittest.main()
