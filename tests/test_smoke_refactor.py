from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from fisher.data import ToyConditionalGaussianDataset
from fisher.evaluation import finite_difference_score
from fisher.models import ConditionalScore1D, LocalDecoderLogit
from fisher.trainers import train_local_decoder, train_score_model


class RefactorSmokeTests(unittest.TestCase):
    def test_dataset_shapes(self) -> None:
        ds = ToyConditionalGaussianDataset(seed=123)
        theta, x = ds.sample_joint(64)
        self.assertEqual(theta.shape, (64, 1))
        self.assertEqual(x.shape, (64, 2))
        eigvals = np.linalg.eigvalsh(ds.cov)
        self.assertTrue(np.all(eigvals > 0))

    def test_score_train_smoke(self) -> None:
        ds = ToyConditionalGaussianDataset(seed=123)
        theta, x = ds.sample_joint(256)
        model = ConditionalScore1D(hidden_dim=32, depth=2).to(torch.device("cpu"))
        losses = train_score_model(
            model=model,
            theta_train=theta,
            x_train=x,
            sigma_values=np.array([0.1, 0.06], dtype=np.float64),
            epochs=2,
            batch_size=64,
            lr=1e-3,
            device=torch.device("cpu"),
            log_every=1,
        )
        self.assertEqual(len(losses), 2)
        self.assertTrue(np.isfinite(losses[-1]))
        score_fd = finite_difference_score(x[:32], theta[:32], ds, delta=0.03)
        self.assertEqual(score_fd.shape, (32,))

    def test_decoder_train_smoke(self) -> None:
        ds = ToyConditionalGaussianDataset(seed=123)
        theta_p = np.full((96, 1), 0.6, dtype=np.float64)
        theta_m = np.full((96, 1), 0.4, dtype=np.float64)
        xp = ds.sample_x(theta_p)
        xm = ds.sample_x(theta_m)
        x = np.concatenate([xp, xm], axis=0)
        y = np.concatenate([np.ones(96), np.zeros(96)])
        model = LocalDecoderLogit(hidden_dim=24, depth=2).to(torch.device("cpu"))
        losses = train_local_decoder(
            model=model,
            x_train=x,
            y_train=y,
            epochs=2,
            batch_size=64,
            lr=1e-3,
            device=torch.device("cpu"),
        )
        self.assertEqual(len(losses), 2)
        self.assertTrue(np.isfinite(losses[-1]))

    def test_unified_cli_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out_score = os.path.join(d, "score")
            out_dec = os.path.join(d, "decoder")

            cmd_score = [
                sys.executable,
                "run_fisher.py",
                "score",
                "--epochs",
                "2",
                "--n-train",
                "320",
                "--n-eval",
                "320",
                "--n-bins",
                "8",
                "--min-bin-count",
                "5",
                "--batch-size",
                "64",
                "--output-dir",
                out_score,
                "--log-every",
                "1",
            ]
            subprocess.run(cmd_score, check=True)
            self.assertTrue(os.path.exists(os.path.join(out_score, "metrics_extrapolated.txt")))

            cmd_dec = [
                sys.executable,
                "run_fisher.py",
                "decoder",
                "--epochs",
                "2",
                "--n-bins",
                "5",
                "--n-train-local",
                "80",
                "--n-eval-local",
                "80",
                "--batch-size",
                "64",
                "--output-dir",
                out_dec,
                "--log-every",
                "2",
            ]
            subprocess.run(cmd_dec, check=True)
            self.assertTrue(os.path.exists(os.path.join(out_dec, "metrics.txt")))


if __name__ == "__main__":
    unittest.main()
