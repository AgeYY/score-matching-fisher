from __future__ import annotations

import math
import json
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

from conditional_fm_knn_gaussian_velocity_prior import (
    _REPO_ROOT,
    ConditionalVelocityMLP,
    KnnDiagGaussianConditionalPrior,
    analytical_gaussian_prior_velocity,
    circular_distance,
    conditional_mean,
    conditional_scales,
    make_scheduler,
    sample_conditional_banana_given_theta,
    sample_gaussian_prior_path,
    true_conditional_cov,
)


class TestConditionalFmKnnGaussianVelocityPrior(unittest.TestCase):
    def test_circular_distance_wraps_at_two_pi(self) -> None:
        theta_a = torch.tensor([[0.05]], dtype=torch.float64)
        theta_b = torch.tensor([[2.0 * math.pi - 0.03, 0.2]], dtype=torch.float64)
        d = circular_distance(theta_a, theta_b)
        self.assertTrue(torch.allclose(d, torch.tensor([[0.08, 0.15]], dtype=torch.float64), atol=1e-12))

    def test_knn_prior_uses_circular_neighbors_and_variance_floor(self) -> None:
        theta_train = torch.tensor([0.02, 2.0 * math.pi - 0.02, math.pi], dtype=torch.float64)
        x_train = torch.tensor([[1.0, 2.0], [3.0, 6.0], [100.0, 200.0]], dtype=torch.float64)
        prior = KnnDiagGaussianConditionalPrior(
            theta_train,
            x_train,
            k=2,
            bandwidth_floor=1.0,
            variance_floor=1e-3,
            weighted_var_correction=False,
        )
        mu, var = prior.query(torch.tensor([0.0], dtype=torch.float64))
        self.assertTrue(torch.allclose(mu[0], torch.tensor([2.0, 4.0], dtype=torch.float64), atol=1e-3))
        self.assertTrue(torch.allclose(var[0], torch.tensor([1.0, 4.0], dtype=torch.float64), atol=1e-3))

        flat_prior = KnnDiagGaussianConditionalPrior(
            torch.tensor([0.0, 0.1], dtype=torch.float64),
            torch.tensor([[2.0, 5.0], [2.0, 5.0]], dtype=torch.float64),
            k=2,
            variance_floor=0.25,
            weighted_var_correction=True,
        )
        _, flat_var = flat_prior.query(torch.tensor([0.05], dtype=torch.float64))
        self.assertTrue(torch.allclose(flat_var[0], torch.tensor([0.25, 0.25], dtype=torch.float64)))

    def test_conditional_generator_matches_true_moments(self) -> None:
        device = torch.device("cpu")
        dtype = torch.float64
        theta_value = 0.7
        theta = torch.full((80_000,), theta_value, device=device, dtype=dtype)
        g = torch.Generator(device=device).manual_seed(0)
        x = sample_conditional_banana_given_theta(theta, rho=0.6, beta=0.4, generator=g)
        expected_mean = conditional_mean(torch.tensor([theta_value], dtype=dtype))[0]
        expected_cov = true_conditional_cov(torch.tensor([theta_value], dtype=dtype), rho=0.6, beta=0.4)[0]
        self.assertTrue(torch.allclose(x.mean(dim=0), expected_mean, atol=6e-3, rtol=6e-3))
        self.assertTrue(torch.allclose(torch.cov(x.T), expected_cov, atol=5e-3, rtol=5e-2))

    def test_analytical_velocity_accepts_batchwise_conditional_prior(self) -> None:
        dtype = torch.float64
        scheduler = make_scheduler("condot")
        t = torch.tensor([[0.35], [0.35]], dtype=dtype)
        mu = torch.tensor([[0.5, -0.25], [0.5, -0.25]], dtype=dtype)
        var = torch.tensor([[1.7, 0.4], [1.7, 0.4]], dtype=dtype)
        alpha = t
        sigma = 1.0 - t
        gain = (-sigma + alpha * var) / (sigma.square() + alpha.square() * var)
        intercept = mu - gain * alpha * mu
        x_probe = torch.tensor([[0.2, -0.1], [1.3, 0.7]], dtype=dtype)
        expected = intercept + gain * x_probe
        actual = analytical_gaussian_prior_velocity(x_probe, t, mu, var, scheduler)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))

    def test_cosine_prior_path_matches_batchwise_scheduler_marginal(self) -> None:
        dtype = torch.float64
        scheduler = make_scheduler("cosine")
        n = 60_000
        t = torch.full((n, 1), 0.3, dtype=dtype)
        mu_one = torch.tensor([[0.4, -0.7]], dtype=dtype)
        var_one = torch.tensor([[1.3, 0.6]], dtype=dtype)
        mu = mu_one.expand(n, -1)
        var = var_one.expand(n, -1)
        g = torch.Generator(device="cpu").manual_seed(4)
        x_t = sample_gaussian_prior_path(t, mu, var, scheduler, generator=g)
        schedule = scheduler(t[:1])
        expected_mean = schedule.alpha_t * mu_one
        expected_var = schedule.sigma_t.square() + schedule.alpha_t.square() * var_one
        self.assertTrue(torch.allclose(x_t.mean(dim=0), expected_mean.reshape(-1), atol=1.2e-2, rtol=1.2e-2))
        self.assertTrue(torch.allclose(x_t.var(dim=0, unbiased=True), expected_var.reshape(-1), atol=1.5e-2, rtol=1.5e-2))

    def test_conditional_model_depends_on_theta_feature(self) -> None:
        torch.manual_seed(2)
        model = ConditionalVelocityMLP(hidden_dim=16, depth=1, time_frequencies=2, theta_frequencies=2)
        x = torch.zeros((4, 2))
        t = torch.full((4, 1), 0.5)
        theta_a = torch.zeros(4)
        theta_b = torch.full((4,), math.pi / 2.0)
        out_a = model(x, t, theta_a)
        out_b = model(x, t, theta_b)
        self.assertFalse(torch.allclose(out_a, out_b))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable; smoke run requires --device cuda per AGENTS.md")
    def test_tiny_cuda_smoke_run_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "conditional_fm_prior"
            cmd = [
                sys.executable,
                str(_REPO_ROOT / "tests" / "conditional_fm_knn_gaussian_velocity_prior.py"),
                "--device",
                "cuda",
                "--train-sizes",
                "32",
                "--seeds",
                "0",
                "--lambda-priors",
                "0 0.1",
                "--eval-thetas",
                "0 1.5707963267948966",
                "--train-steps",
                "10",
                "--val-size",
                "12",
                "--early-stopping-patience",
                "1",
                "--early-stopping-min-delta",
                "1000000",
                "--test-size",
                "16",
                "--n-gen-per-theta",
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
                "--theta-frequencies",
                "2",
                "--knn-k",
                "8",
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
            run_dir = out_dir / "n_32" / "seed_0" / "lambda_0"
            arr = np.load(run_dir / "samples.npz")
            self.assertEqual(tuple(arr["generated_samples"].shape), (2, 16, 2))
            self.assertEqual(tuple(arr["true_samples"].shape), (2, 16, 2))
            self.assertEqual(tuple(arr["val"].shape), (12, 2))
            self.assertEqual(tuple(arr["theta_val"].shape), (12,))
            self.assertEqual(tuple(arr["train"].shape), (20, 2))
            self.assertEqual(tuple(arr["observed"].shape), (32, 2))
            self.assertEqual(tuple(arr["theta_observed"].shape), (32,))

            payload = json.loads((run_dir / "summary.json").read_text())
            self.assertIn("training_summary", payload)
            self.assertEqual(payload["training_summary"]["best_step"], 1)
            self.assertEqual(payload["training_summary"]["stopped_step"], 2)
            self.assertTrue(payload["training_summary"]["early_stopped"])
            self.assertIn("val_fm_loss", payload["train_logs"][0])


if __name__ == "__main__":
    unittest.main()
