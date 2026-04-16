from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fisher.cli_shared_fisher import add_estimation_arguments
from fisher.ctsm_models import ToyPairConditionedTimeScoreNet
from fisher.h_matrix import HMatrixEstimator
from fisher.shared_fisher_est import run_shared_fisher_estimation, validate_estimation_args


class TestHMatrixCtsmV(unittest.TestCase):
    def test_h_matrix_ctsm_v_basic_invariants(self) -> None:
        torch.manual_seed(0)
        model = ToyPairConditionedTimeScoreNet(dim=2, hidden_dim=32)
        est = HMatrixEstimator(
            model_post=model,
            model_prior=None,
            sigma_eval=1e-4,
            device=torch.device("cpu"),
            pair_batch_size=64,
            field_method="ctsm_v",
            ctsm_int_n_time=24,
            ctsm_t_eps=1e-4,
        )
        theta = np.linspace(-1.0, 1.0, 6, dtype=np.float64).reshape(-1, 1)
        x = np.stack([np.sin(theta.reshape(-1)), np.cos(theta.reshape(-1))], axis=1).astype(np.float64)
        out = est.run(theta=theta, x=x, restore_original_order=False)
        self.assertEqual(out.field_method, "ctsm_v")
        self.assertEqual(out.eval_scalar_name, "ctsm_t_eps")
        self.assertIn("pair_conditioned_ctsm_v", str(out.flow_score_mode))
        self.assertTrue(np.isfinite(out.delta_l_matrix).all())
        self.assertTrue(np.isfinite(out.h_sym).all())
        self.assertLess(float(np.max(np.abs(np.diag(out.delta_l_matrix)))), 1e-8)
        self.assertLess(float(np.max(np.abs(np.diag(out.h_sym)))), 1e-8)

    def test_validate_estimation_args_accepts_ctsm_v(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args([])
        args.theta_field_method = "ctsm_v"
        args.compute_h_matrix = True
        args.prior_enable = False
        validate_estimation_args(args)

    def test_shared_estimation_ctsm_v_smoke(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args([])
        args.theta_field_method = "ctsm_v"
        args.compute_h_matrix = True
        args.prior_enable = False
        args.device = "cpu"
        args.x_dim = 2
        args.dataset_family = "cosine_gaussian"
        args.score_data_mode = "full"
        args.score_val_source = "train_split"
        args.score_fisher_eval_data = "full"
        args.h_restore_original_order = True
        args.h_batch_size = 128
        args.h_save_intermediates = True
        args.seed = 5
        args.log_every = 1
        args.ctsm_epochs = 4
        args.ctsm_batch_size = 16
        args.ctsm_hidden_dim = 32
        args.ctsm_two_sb_var = 2.0
        args.ctsm_factor = 1.0
        args.ctsm_t_eps = 1e-4
        args.ctsm_int_n_time = 24

        n = 24
        theta_all = np.linspace(-1.2, 1.2, n, dtype=np.float64).reshape(-1, 1)
        x_all = np.concatenate([theta_all, np.sin(theta_all)], axis=1).astype(np.float64)
        split = n // 2
        theta_train = theta_all[:split]
        x_train = x_all[:split]
        theta_eval = theta_all[split:]
        x_eval = x_all[split:]

        with tempfile.TemporaryDirectory() as td:
            args.output_dir = str(Path(td))
            run_shared_fisher_estimation(
                args,
                dataset=object(),  # ctsm_v branch returns before dataset-dependent Fisher/decoder logic
                theta_all=theta_all,
                x_all=x_all,
                theta_train=theta_train,
                x_train=x_train,
                theta_eval=theta_eval,
                x_eval=x_eval,
                rng=np.random.default_rng(0),
            )
            h_npz = Path(td) / "h_matrix_results_theta_cov.npz"
            losses_npz = Path(td) / "score_prior_training_losses.npz"
            self.assertTrue(h_npz.is_file())
            self.assertTrue(losses_npz.is_file())
            with np.load(h_npz, allow_pickle=True) as z:
                self.assertEqual(str(z["h_field_method"][0]), "ctsm_v")
                h = np.asarray(z["h_sym"], dtype=np.float64)
                self.assertTrue(np.isfinite(h).all())


if __name__ == "__main__":
    unittest.main()
