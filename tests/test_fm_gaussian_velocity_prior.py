from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from fm_gaussian_velocity_prior import (
    _REPO_ROOT,
    analytical_gaussian_prior_velocity,
    banana_inverse,
    banana_log_prob,
    estimate_diag_gaussian_prior,
    make_scheduler,
    sample_banana,
    sample_gaussian_prior_path,
)


class TestFmGaussianVelocityPrior(unittest.TestCase):
    def test_banana_inverse_and_log_prob_match_latent_gaussian(self) -> None:
        device = torch.device("cpu")
        g = torch.Generator(device=device).manual_seed(0)
        rho = 0.7
        beta = 0.3
        x = sample_banana(256, rho=rho, beta=beta, generator=g, device=device, dtype=torch.float64)
        z = banana_inverse(x, beta=beta)
        x_roundtrip = z.clone()
        x_roundtrip[:, 1] = z[:, 1] + beta * (z[:, 0].square() - 1.0)
        self.assertTrue(torch.allclose(x_roundtrip, x, atol=1e-12, rtol=1e-12))

        cov = torch.tensor([[1.0, rho], [rho, 1.0]], dtype=torch.float64)
        mvn = torch.distributions.MultivariateNormal(torch.zeros(2, dtype=torch.float64), covariance_matrix=cov)
        expected = mvn.log_prob(z)
        actual = banana_log_prob(x, rho=rho, beta=beta)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))

    def test_prior_estimate_uses_training_samples_and_variance_floor(self) -> None:
        x_train = torch.tensor([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]], dtype=torch.float64)
        mu, var = estimate_diag_gaussian_prior(x_train, variance_floor=1e-3)
        self.assertTrue(torch.allclose(mu, torch.tensor([3.0, 2.0], dtype=torch.float64)))
        self.assertTrue(torch.allclose(var, torch.tensor([4.0, 1e-3], dtype=torch.float64)))

    def test_analytical_velocity_matches_conditional_gaussian_regression(self) -> None:
        dtype = torch.float64
        mu = torch.tensor([0.5, -0.25], dtype=dtype)
        var = torch.tensor([1.7, 0.4], dtype=dtype)
        scheduler = make_scheduler("condot")
        t = torch.tensor([[0.35]], dtype=dtype)
        alpha = t
        sigma = 1.0 - t
        gain = (-sigma + alpha * var.reshape(1, -1)) / (sigma.square() + alpha.square() * var.reshape(1, -1))
        intercept = mu.reshape(1, -1) - gain * alpha * mu.reshape(1, -1)
        x_probe = torch.tensor([[0.2, -0.1], [1.3, 0.7], [-0.8, -1.0]], dtype=dtype)
        empirical = intercept + gain * x_probe
        analytical = analytical_gaussian_prior_velocity(x_probe, t.expand(x_probe.shape[0], 1), mu, var, scheduler)
        self.assertTrue(torch.allclose(analytical, empirical, atol=1e-12, rtol=1e-12))

    def test_cosine_prior_path_uses_scheduler_marginal(self) -> None:
        dtype = torch.float64
        scheduler = make_scheduler("cosine")
        mu = torch.tensor([0.4, -0.7], dtype=dtype)
        var = torch.tensor([1.3, 0.6], dtype=dtype)
        t = torch.full((60_000, 1), 0.3, dtype=dtype)
        g = torch.Generator(device="cpu").manual_seed(4)
        x_t = sample_gaussian_prior_path(t.shape[0], t, mu, var, scheduler, generator=g)
        schedule = scheduler(t[:1])
        expected_mean = schedule.alpha_t * mu.reshape(1, -1)
        expected_var = schedule.sigma_t.square() + schedule.alpha_t.square() * var.reshape(1, -1)
        self.assertTrue(torch.allclose(x_t.mean(dim=0), expected_mean.reshape(-1), atol=1.2e-2, rtol=1.2e-2))
        self.assertTrue(torch.allclose(x_t.var(dim=0, unbiased=True), expected_var.reshape(-1), atol=1.5e-2, rtol=1.5e-2))
        v = analytical_gaussian_prior_velocity(x_t[:128], t[:128], mu, var, scheduler)
        self.assertTrue(torch.isfinite(v).all())

    def test_prior_target_is_not_pathwise_endpoint_velocity(self) -> None:
        dtype = torch.float64
        mu = torch.tensor([0.1, -0.2], dtype=dtype)
        var = torch.tensor([1.5, 0.7], dtype=dtype)
        scheduler = make_scheduler("condot")
        t = torch.full((4096, 1), 0.4, dtype=dtype)
        g = torch.Generator(device="cpu").manual_seed(3)
        x_t = sample_gaussian_prior_path(t.shape[0], t, mu, var, scheduler, generator=g)
        analytical = analytical_gaussian_prior_velocity(x_t, t, mu, var, scheduler)

        x0 = torch.randn(x_t.shape, generator=g, dtype=dtype)
        x1 = mu.reshape(1, -1) + torch.sqrt(var).reshape(1, -1) * torch.randn(x_t.shape, generator=g, dtype=dtype)
        pathwise = x1 - x0
        self.assertGreater(float(torch.mean((analytical - pathwise).square()).item()), 0.1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable; smoke run requires --device cuda per AGENTS.md")
    def test_tiny_cuda_smoke_run_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "fm_prior"
            cmd = [
                sys.executable,
                str(_REPO_ROOT / "tests" / "fm_gaussian_velocity_prior.py"),
                "--device",
                "cuda",
                "--train-sizes",
                "32",
                "--seeds",
                "0",
                "--lambda-priors",
                "0 0.1",
                "--train-steps",
                "2",
                "--test-size",
                "16",
                "--n-gen",
                "16",
                "--ode-steps",
                "4",
                "--batch-size",
                "16",
                "--hidden-dim",
                "16",
                "--depth",
                "1",
                "--time-frequencies",
                "2",
                "--output-dir",
                str(out_dir),
                "--n-mmd",
                "16",
                "--n-sliced",
                "8",
            ]
            subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))
            self.assertTrue((out_dir / "summary.json").is_file())
            self.assertTrue((out_dir / "metrics.csv").is_file())
            arr = np.load(out_dir / "n_32" / "seed_0" / "lambda_0" / "samples.npz")
            self.assertEqual(tuple(arr["generated"].shape), (16, 2))


if __name__ == "__main__":
    unittest.main()
