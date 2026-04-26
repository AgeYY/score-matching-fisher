from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fisher.cli_shared_fisher import add_estimation_arguments
from fisher.h_matrix import HMatrixEstimator
from fisher.models import (
    ConditionalThetaFlowVelocity,
    PriorThetaFlowVelocity,
)
from fisher.shared_fisher_est import run_shared_fisher_estimation, validate_estimation_args
from fisher.trainers import train_conditional_theta_flow_model, train_conditional_theta_flow_pre_post_model


class _DummyPosteriorFlow(torch.nn.Module):
    def forward(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _ = x, t
        return -theta_t


class _DummyPriorFlow(torch.nn.Module):
    def forward(self, theta_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _ = t
        return -theta_t


class _ShiftedPosteriorFlow(torch.nn.Module):
    def forward(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return -theta_t + 0.1 * x[:, :1] + 0.05 * t


class TestHMatrixFlowLikelihood(unittest.TestCase):
    def test_flow_likelihood_identical_post_prior_yields_zero_delta(self) -> None:
        post = _DummyPosteriorFlow()
        prior = _DummyPriorFlow()
        est = HMatrixEstimator(
            model_post=post,
            model_prior=prior,
            sigma_eval=1.0,
            device=torch.device("cpu"),
            pair_batch_size=64,
            field_method="theta_flow",
            flow_scheduler="cosine",
            flow_ode_steps=32,
        )
        theta = np.linspace(-1.0, 1.0, 6, dtype=np.float64).reshape(-1, 1)
        x = np.stack([np.sin(theta.reshape(-1)), np.cos(theta.reshape(-1))], axis=1).astype(np.float64)
        out = est.run(theta=theta, x=x, restore_original_order=False)
        self.assertEqual(out.field_method, "theta_flow")
        self.assertEqual(out.eval_scalar_name, "flow_ode_t_span")
        self.assertLess(float(np.max(np.abs(out.delta_l_matrix))), 1e-9)
        self.assertLess(float(np.max(np.abs(out.h_sym))), 1e-9)
        self.assertIsNotNone(out.log_p_theta_prior)
        self.assertEqual(tuple(np.asarray(out.log_p_theta_prior).shape), (theta.shape[0],))
        self.assertTrue(np.isfinite(np.asarray(out.log_p_theta_prior, dtype=np.float64)).all())

    def test_flow_likelihood_outputs_are_finite_for_shifted_models(self) -> None:
        post = _ShiftedPosteriorFlow()
        prior = _DummyPriorFlow()
        est = HMatrixEstimator(
            model_post=post,
            model_prior=prior,
            sigma_eval=1.0,
            device=torch.device("cpu"),
            pair_batch_size=64,
            field_method="theta_flow",
            flow_scheduler="cosine",
            flow_ode_steps=16,
        )
        theta = np.linspace(-1.2, 1.2, 7, dtype=np.float64).reshape(-1, 1)
        x = np.stack([np.sin(theta.reshape(-1)), np.cos(theta.reshape(-1))], axis=1).astype(np.float64)
        out = est.run(theta=theta, x=x, restore_original_order=False)
        self.assertTrue(np.isfinite(out.c_matrix).all())
        self.assertTrue(np.isfinite(out.delta_l_matrix).all())
        self.assertTrue(np.isfinite(out.h_sym).all())
        self.assertEqual(out.flow_score_mode, "direct_ode_likelihood")
        self.assertIsNotNone(out.log_p_theta_prior)
        self.assertEqual(tuple(np.asarray(out.log_p_theta_prior).shape), (theta.shape[0],))
        self.assertTrue(np.isfinite(np.asarray(out.log_p_theta_prior, dtype=np.float64)).all())

    def test_validate_estimation_args_accepts_theta_flow(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args(["--theta-field-method", "theta_flow", "--flow-arch", "mlp"])
        validate_estimation_args(args)

    def test_validate_estimation_args_accepts_theta_flow_reg(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args(["--theta-field-method", "theta_flow_reg", "--flow-arch", "mlp"])
        validate_estimation_args(args)
        self.assertEqual(args.flow_theta_reg_bin_n_bins, 10)
        self.assertAlmostEqual(float(args.flow_theta_reg_lambda), 0.01)

    def test_validate_estimation_args_accepts_theta_flow_pre_post(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args(["--theta-field-method", "theta_flow_pre_post", "--flow-arch", "mlp"])
        validate_estimation_args(args)
        self.assertEqual(args.flow_theta_pre_post_finetune_epochs, 10000)
        self.assertEqual(args.flow_theta_pre_post_pretrain_synthetic_size, 0)
        self.assertFalse(args.flow_theta_pre_post_pretrain_resample_synthetic_each_epoch)
        self.assertIsNone(args.flow_theta_pre_post_pretrain_early_patience)
        self.assertAlmostEqual(float(args.flow_theta_reg_lambda), 0.01)

        args = parser.parse_args(
            [
                "--theta-field-method",
                "theta_flow_pre_post",
                "--flow-arch",
                "mlp",
                "--flow-theta-pre-post-pretrain-synthetic-size",
                "20",
                "--flow-theta-pre-post-pretrain-resample-synthetic-each-epoch",
                "--flow-theta-pre-post-pretrain-early-patience",
                "10",
            ]
        )
        validate_estimation_args(args)
        self.assertEqual(args.flow_theta_pre_post_pretrain_synthetic_size, 20)
        self.assertTrue(args.flow_theta_pre_post_pretrain_resample_synthetic_each_epoch)
        self.assertEqual(args.flow_theta_pre_post_pretrain_early_patience, 10)

        args = parser.parse_args(
            [
                "--theta-field-method",
                "theta_flow_pre_post",
                "--flow-arch",
                "mlp",
                "--flow-theta-reg-lambda",
                "0",
            ]
        )
        validate_estimation_args(args)
        self.assertEqual(float(args.flow_theta_reg_lambda), 0.0)

        args = parser.parse_args(
            [
                "--theta-field-method",
                "theta_flow_pre_post",
                "--flow-arch",
                "mlp",
                "--flow-theta-pre-post-pretrain-resample-synthetic-each-epoch",
            ]
        )
        with self.assertRaises(ValueError):
            validate_estimation_args(args)

        args = parser.parse_args(
            [
                "--theta-field-method",
                "theta_flow_pre_post",
                "--flow-arch",
                "mlp",
                "--flow-theta-pre-post-pretrain-early-patience",
                "0",
            ]
        )
        with self.assertRaises(ValueError):
            validate_estimation_args(args)

    def test_validate_rejects_bad_theta_flow_reg_parameters(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        for argv in (
            ["--theta-field-method", "theta_flow_reg", "--flow-theta-reg-lambda", "-1"],
            ["--theta-field-method", "theta_flow_reg", "--flow-theta-reg-bin-n-bins", "0"],
            ["--theta-field-method", "theta_flow_reg", "--flow-theta-reg-variance-floor", "0"],
        ):
            args = parser.parse_args(argv)
            with self.assertRaises(ValueError):
                validate_estimation_args(args)

    def test_validate_rejects_x_flow_reg_flags_for_theta_flow_reg(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args(
            ["--theta-field-method", "theta_flow_reg", "--flow-x-reg-bin-n-bins", "11"]
        )
        with self.assertRaises(ValueError):
            validate_estimation_args(args)
        args = parser.parse_args(
            ["--theta-field-method", "theta_flow", "--flow-x-reg-bin-n-bins", "11"]
        )
        with self.assertRaises(ValueError):
            validate_estimation_args(args)

    def test_validate_estimation_args_accepts_theta_path_integral(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args(["--theta-field-method", "theta_path_integral", "--flow-arch", "mlp"])
        validate_estimation_args(args)

    def test_theta_flow_mlp_accepts_vector_theta_state(self) -> None:
        torch.manual_seed(2)
        theta_dim = 4
        post = ConditionalThetaFlowVelocity(x_dim=2, hidden_dim=12, depth=1, theta_dim=theta_dim)
        prior = PriorThetaFlowVelocity(hidden_dim=12, depth=1, theta_dim=theta_dim)
        est = HMatrixEstimator(
            model_post=post,
            model_prior=prior,
            sigma_eval=0.8,
            device=torch.device("cpu"),
            pair_batch_size=32,
            field_method="theta_flow",
            flow_scheduler="cosine",
            flow_ode_steps=8,
        )
        # One-hot theta states; run() should preserve 2D theta rows and sort by active bin.
        theta_idx = np.asarray([2, 0, 3, 1, 0], dtype=np.int64)
        theta = np.eye(theta_dim, dtype=np.float64)[theta_idx]
        x = np.random.default_rng(2).standard_normal((theta.shape[0], 2))
        out = est.run(theta=theta, x=x, restore_original_order=False)
        self.assertEqual(out.theta_used.shape, (theta.shape[0], theta_dim))
        self.assertEqual(out.theta_sorted.shape, (theta.shape[0], theta_dim))
        self.assertTrue(np.isfinite(out.h_sym).all())
        sorted_bins = np.argmax(out.theta_sorted, axis=1)
        self.assertTrue(np.all(np.diff(sorted_bins) >= 0))

    def test_theta_flow_mlp_accepts_fourier_vector_theta_state(self) -> None:
        torch.manual_seed(3)
        theta_scalar = np.linspace(-1.0, 1.0, 7, dtype=np.float64)
        k = 3
        period = 2.0 * (float(theta_scalar.max()) - float(theta_scalar.min()))
        w0 = 2.0 * np.pi / max(period, 1e-12)
        theta_cols = []
        for kk in range(1, k + 1):
            theta_cols.append(np.sin(kk * w0 * theta_scalar).reshape(-1, 1))
            theta_cols.append(np.cos(kk * w0 * theta_scalar).reshape(-1, 1))
        theta = np.concatenate(theta_cols, axis=1)

        theta_dim = int(theta.shape[1])
        post = ConditionalThetaFlowVelocity(x_dim=2, hidden_dim=16, depth=2, theta_dim=theta_dim)
        prior = PriorThetaFlowVelocity(hidden_dim=16, depth=2, theta_dim=theta_dim)
        est = HMatrixEstimator(
            model_post=post,
            model_prior=prior,
            sigma_eval=0.9,
            device=torch.device("cpu"),
            pair_batch_size=32,
            field_method="theta_flow",
            flow_scheduler="cosine",
            flow_ode_steps=8,
        )
        x = np.stack([np.sin(theta_scalar), np.cos(theta_scalar)], axis=1).astype(np.float64)
        out = est.run(theta=theta, x=x, restore_original_order=False)
        self.assertEqual(out.theta_used.shape, (theta.shape[0], theta_dim))
        self.assertEqual(out.theta_sorted.shape, (theta.shape[0], theta_dim))
        self.assertTrue(np.isfinite(out.delta_l_matrix).all())
        self.assertTrue(np.isfinite(out.h_sym).all())

    def test_train_conditional_theta_flow_with_binned_regularization_records_losses(self) -> None:
        torch.manual_seed(0)
        theta = np.linspace(-1.0, 1.0, 24, dtype=np.float64).reshape(-1, 1)
        x = np.concatenate([np.cos(theta), np.sin(theta)], axis=1).astype(np.float64)
        model = ConditionalThetaFlowVelocity(x_dim=2, hidden_dim=16, depth=1).to(torch.device("cpu"))
        out = train_conditional_theta_flow_model(
            model=model,
            theta_train=theta[:16],
            x_train=x[:16],
            epochs=2,
            batch_size=8,
            lr=1e-3,
            device=torch.device("cpu"),
            log_every=99,
            theta_val=theta[16:],
            x_val=x[16:],
            early_stopping_patience=100,
            scheduler_name="vp",
            theta_prior_regularization_lambda=0.01,
            theta_prior_regularization_bin_n_bins=4,
        )
        self.assertEqual(len(out["train_losses"]), 2)
        self.assertEqual(len(out["train_fm_losses"]), 2)
        self.assertEqual(len(out["train_reg_losses"]), 2)
        self.assertTrue(np.isfinite(np.asarray(out["train_fm_losses"], dtype=np.float64)).all())
        self.assertTrue(np.isfinite(np.asarray(out["train_reg_losses"], dtype=np.float64)).all())
        self.assertGreater(float(np.max(out["train_reg_losses"])), 0.0)

    def test_train_conditional_theta_flow_zero_regularization_preserves_zero_reg_losses(self) -> None:
        torch.manual_seed(0)
        theta = np.linspace(-1.0, 1.0, 16, dtype=np.float64).reshape(-1, 1)
        x = np.concatenate([np.cos(theta), np.sin(theta)], axis=1).astype(np.float64)
        model = ConditionalThetaFlowVelocity(x_dim=2, hidden_dim=16, depth=1).to(torch.device("cpu"))
        out = train_conditional_theta_flow_model(
            model=model,
            theta_train=theta,
            x_train=x,
            epochs=2,
            batch_size=8,
            lr=1e-3,
            device=torch.device("cpu"),
            log_every=99,
            theta_prior_regularization_lambda=0.0,
        )
        self.assertEqual(len(out["train_reg_losses"]), 2)
        self.assertTrue(all(abs(float(v)) < 1e-12 for v in out["train_reg_losses"]))

    def test_train_conditional_theta_flow_pre_post_records_phases_and_freezes_backbone(self) -> None:
        torch.manual_seed(0)
        theta = np.linspace(-1.0, 1.0, 24, dtype=np.float64).reshape(-1, 1)
        x = np.concatenate([np.cos(theta), np.sin(theta)], axis=1).astype(np.float64)
        model = ConditionalThetaFlowVelocity(x_dim=2, hidden_dim=16, depth=1).to(torch.device("cpu"))
        out = train_conditional_theta_flow_pre_post_model(
            model=model,
            theta_train=theta,
            x_train=x,
            pretrain_epochs=1,
            finetune_epochs=1,
            batch_size=8,
            pretrain_lr=1e-3,
            finetune_lr=1e-3,
            device=torch.device("cpu"),
            log_every=99,
            scheduler_name="vp",
            theta_prior_regularization_lambda=0.01,
            theta_prior_regularization_bin_n_bins=4,
        )
        self.assertEqual(len(out["theta_pre_post_pretrain_reg_train_losses"]), 1)
        self.assertEqual(len(out["theta_pre_post_finetune_fm_train_losses"]), 1)
        self.assertGreater(float(np.max(out["theta_pre_post_pretrain_reg_train_losses"])), 0.0)
        self.assertGreater(float(np.max(out["theta_pre_post_finetune_fm_train_losses"])), 0.0)
        self.assertTrue(
            np.allclose(
                np.asarray(out["theta_pre_post_pretrain_train_losses"], dtype=np.float64),
                np.asarray(out["theta_pre_post_pretrain_reg_train_losses"], dtype=np.float64),
            )
        )
        start_state = out["theta_pre_post_finetune_start_state"]
        final_state = model.state_dict()
        changed = []
        unchanged = []
        for name, start_tensor in start_state.items():
            same = torch.equal(final_state[name].detach().cpu(), start_tensor)
            if name.startswith("net.2."):
                changed.append(not same)
            else:
                unchanged.append(same)
        self.assertTrue(any(changed))
        self.assertTrue(all(unchanged))

    def test_train_conditional_theta_flow_pre_post_zero_finetune_skips_readout_phase(self) -> None:
        torch.manual_seed(0)
        n = 24
        split = n // 2
        theta = np.linspace(-1.0, 1.0, n, dtype=np.float64).reshape(-1, 1)
        x = np.concatenate([np.cos(theta), np.sin(theta)], axis=1).astype(np.float64)
        model = ConditionalThetaFlowVelocity(x_dim=2, hidden_dim=16, depth=1).to(torch.device("cpu"))
        out = train_conditional_theta_flow_pre_post_model(
            model=model,
            theta_train=theta[:split],
            x_train=x[:split],
            pretrain_epochs=1,
            finetune_epochs=0,
            batch_size=8,
            pretrain_lr=1e-3,
            finetune_lr=1e-3,
            device=torch.device("cpu"),
            log_every=99,
            scheduler_name="vp",
            theta_val=theta[split:],
            x_val=x[split:],
            theta_prior_regularization_lambda=0.01,
            theta_prior_regularization_bin_n_bins=4,
        )
        self.assertEqual(len(out["theta_pre_post_finetune_fm_train_losses"]), 0)
        self.assertEqual(int(out["theta_pre_post_readout_trainable_params"]), 0)
        self.assertEqual(float(out["best_val_loss"]), float(out["theta_pre_post_pretrain_best_val_loss"]))

    def test_train_conditional_theta_flow_pre_post_synthetic_pretrain_records_sizes(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        n = 30
        split = 18
        theta = np.linspace(-1.0, 1.0, n, dtype=np.float64).reshape(-1, 1)
        x = np.concatenate([np.cos(theta), np.sin(theta)], axis=1).astype(np.float64)
        model = ConditionalThetaFlowVelocity(x_dim=2, hidden_dim=16, depth=1).to(torch.device("cpu"))
        out = train_conditional_theta_flow_pre_post_model(
            model=model,
            theta_train=theta[:split],
            x_train=x[:split],
            pretrain_epochs=1,
            finetune_epochs=0,
            batch_size=8,
            pretrain_lr=1e-3,
            finetune_lr=1e-3,
            device=torch.device("cpu"),
            log_every=99,
            scheduler_name="vp",
            theta_val=theta[split:],
            x_val=x[split:],
            theta_prior_regularization_lambda=0.01,
            theta_prior_regularization_bin_n_bins=4,
            pretrain_synthetic_size=20,
            pretrain_early_stopping_patience=7,
        )
        self.assertEqual(int(out["theta_pre_post_pretrain_synthetic_size"]), 20)
        self.assertEqual(int(out["theta_pre_post_pretrain_synthetic_train_size"]), 12)
        self.assertEqual(int(out["theta_pre_post_pretrain_synthetic_val_size"]), 8)
        self.assertFalse(bool(out["theta_pre_post_pretrain_resample_synthetic_each_epoch"]))
        self.assertEqual(int(out["theta_pre_post_pretrain_early_stopping_patience"]), 7)
        self.assertEqual(len(out["theta_pre_post_finetune_fm_train_losses"]), 0)
        self.assertTrue(
            np.allclose(
                np.asarray(out["theta_pre_post_pretrain_train_losses"], dtype=np.float64),
                np.asarray(out["theta_pre_post_pretrain_reg_train_losses"], dtype=np.float64),
            )
        )
        self.assertTrue(
            np.allclose(
                np.asarray(out["theta_pre_post_pretrain_val_losses"], dtype=np.float64),
                np.asarray(out["theta_pre_post_pretrain_reg_val_losses"], dtype=np.float64),
            )
        )

    def test_train_conditional_theta_flow_pre_post_accepts_zero_flow_theta_reg_lambda_metadata(self) -> None:
        """Pretrain loss does not use lambda; value 0 is valid for stored metadata only."""
        torch.manual_seed(0)
        np.random.seed(0)
        n = 24
        split = 16
        theta = np.linspace(-1.0, 1.0, n, dtype=np.float64).reshape(-1, 1)
        x = np.concatenate([np.cos(theta), np.sin(theta)], axis=1).astype(np.float64)
        model = ConditionalThetaFlowVelocity(x_dim=2, hidden_dim=16, depth=1).to(torch.device("cpu"))
        out = train_conditional_theta_flow_pre_post_model(
            model=model,
            theta_train=theta[:split],
            x_train=x[:split],
            pretrain_epochs=1,
            finetune_epochs=0,
            batch_size=8,
            pretrain_lr=1e-3,
            finetune_lr=1e-3,
            device=torch.device("cpu"),
            log_every=99,
            scheduler_name="vp",
            theta_val=theta[split:],
            x_val=x[split:],
            theta_prior_regularization_lambda=0.0,
            theta_prior_regularization_bin_n_bins=4,
            pretrain_synthetic_size=20,
            pretrain_early_stopping_patience=7,
        )
        self.assertEqual(float(out["flow_theta_reg_lambda"]), 0.0)
        self.assertGreater(float(np.max(out["theta_pre_post_pretrain_reg_train_losses"])), 0.0)

    def test_train_conditional_theta_flow_pre_post_online_synthetic_records_mode(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        n = 30
        split = 18
        theta = np.linspace(-1.0, 1.0, n, dtype=np.float64).reshape(-1, 1)
        x = np.concatenate([np.cos(theta), np.sin(theta)], axis=1).astype(np.float64)
        model = ConditionalThetaFlowVelocity(x_dim=2, hidden_dim=16, depth=1).to(torch.device("cpu"))
        out = train_conditional_theta_flow_pre_post_model(
            model=model,
            theta_train=theta[:split],
            x_train=x[:split],
            pretrain_epochs=2,
            finetune_epochs=0,
            batch_size=8,
            pretrain_lr=1e-3,
            finetune_lr=1e-3,
            device=torch.device("cpu"),
            log_every=99,
            scheduler_name="vp",
            theta_val=theta[split:],
            x_val=x[split:],
            theta_prior_regularization_lambda=0.01,
            theta_prior_regularization_bin_n_bins=4,
            pretrain_synthetic_size=20,
            pretrain_resample_synthetic_each_epoch=True,
            pretrain_early_stopping_patience=7,
        )
        self.assertEqual(int(out["theta_pre_post_pretrain_synthetic_size"]), 20)
        self.assertEqual(int(out["theta_pre_post_pretrain_synthetic_train_size"]), 12)
        self.assertEqual(int(out["theta_pre_post_pretrain_synthetic_val_size"]), 8)
        self.assertTrue(bool(out["theta_pre_post_pretrain_resample_synthetic_each_epoch"]))
        self.assertEqual(len(out["theta_pre_post_pretrain_train_losses"]), 2)
        self.assertEqual(len(out["theta_pre_post_pretrain_val_losses"]), 2)
        self.assertEqual(len(out["theta_pre_post_finetune_fm_train_losses"]), 0)
        self.assertTrue(
            np.allclose(
                np.asarray(out["theta_pre_post_pretrain_train_losses"], dtype=np.float64),
                np.asarray(out["theta_pre_post_pretrain_reg_train_losses"], dtype=np.float64),
            )
        )
        self.assertTrue(
            np.allclose(
                np.asarray(out["theta_pre_post_pretrain_val_losses"], dtype=np.float64),
                np.asarray(out["theta_pre_post_pretrain_reg_val_losses"], dtype=np.float64),
            )
        )

    def test_shared_estimation_theta_flow_reg_runs_and_records_metadata(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args([])
        args.theta_field_method = "theta_flow_reg"
        args.compute_h_matrix = True
        args.prior_enable = True
        args.device = "cpu"
        args.x_dim = 2
        args.dataset_family = "cosine_gaussian"
        args.h_restore_original_order = True
        args.h_batch_size = 128
        args.h_save_intermediates = True
        args.seed = 5
        args.log_every = 99
        args.flow_epochs = 2
        args.prior_epochs = 2
        args.flow_batch_size = 8
        args.prior_batch_size = 8
        args.flow_hidden_dim = 16
        args.prior_hidden_dim = 16
        args.flow_depth = 1
        args.prior_depth = 1
        args.flow_early_patience = 100
        args.prior_early_patience = 100
        args.flow_scheduler = "vp"
        args.flow_theta_reg_lambda = 0.01
        args.flow_theta_reg_bin_n_bins = 4

        n = 24
        theta_all = np.linspace(-1.0, 1.0, n, dtype=np.float64).reshape(-1, 1)
        x_all = np.concatenate([np.cos(theta_all), np.sin(theta_all)], axis=1).astype(np.float64)
        split = n // 2
        with tempfile.TemporaryDirectory() as td:
            args.output_dir = str(Path(td))
            run_shared_fisher_estimation(
                args,
                dataset=object(),
                theta_all=theta_all,
                x_all=x_all,
                theta_train=theta_all[:split],
                x_train=x_all[:split],
                theta_validation=theta_all[split:],
                x_validation=x_all[split:],
                rng=np.random.default_rng(0),
            )
            out_dir = Path(td)
            self.assertTrue((out_dir / "h_matrix_results_theta_cov.npz").is_file())
            h_z = np.load(out_dir / "h_matrix_results_theta_cov.npz", allow_pickle=True)
            self.assertIn("log_p_theta_prior", h_z.files)
            self.assertEqual(tuple(h_z["log_p_theta_prior"].shape), (n,))
            self.assertTrue(np.isfinite(h_z["log_p_theta_prior"]).all())
            loss_npz = out_dir / "score_prior_training_losses.npz"
            self.assertTrue(loss_npz.is_file())
            z = np.load(loss_npz, allow_pickle=True)
            self.assertEqual(str(z["theta_field_method"].reshape(-1)[0]), "theta_flow_reg")
            self.assertAlmostEqual(float(z["flow_theta_reg_lambda"]), 0.01)
            self.assertEqual(int(z["flow_theta_reg_bin_n_bins"]), 4)
            self.assertEqual(tuple(z["flow_theta_reg_losses"].shape), (2,))
            self.assertEqual(tuple(z["flow_theta_reg_fm_losses"].shape), (2,))
            self.assertTrue(np.isfinite(z["flow_theta_reg_losses"]).all())
            self.assertGreater(float(np.max(z["flow_theta_reg_losses"])), 0.0)

    def test_shared_estimation_theta_flow_pre_post_runs_and_records_phase_metadata(self) -> None:
        parser = argparse.ArgumentParser()
        add_estimation_arguments(parser)
        args = parser.parse_args([])
        args.theta_field_method = "theta_flow_pre_post"
        args.compute_h_matrix = True
        args.prior_enable = True
        args.device = "cpu"
        args.x_dim = 2
        args.dataset_family = "cosine_gaussian"
        args.h_restore_original_order = True
        args.h_batch_size = 128
        args.h_save_intermediates = True
        args.seed = 5
        args.log_every = 99
        args.flow_epochs = 1
        args.flow_theta_pre_post_pretrain_epochs = 1
        args.flow_theta_pre_post_finetune_epochs = 1
        args.flow_theta_pre_post_pretrain_synthetic_size = 20
        args.flow_theta_pre_post_pretrain_resample_synthetic_each_epoch = True
        args.flow_theta_pre_post_pretrain_early_patience = 9
        args.prior_epochs = 1
        args.flow_batch_size = 8
        args.prior_batch_size = 8
        args.flow_hidden_dim = 16
        args.prior_hidden_dim = 16
        args.flow_depth = 1
        args.prior_depth = 1
        args.flow_early_patience = 100
        args.prior_early_patience = 100
        args.flow_scheduler = "vp"
        args.flow_theta_reg_lambda = 0.01
        args.flow_theta_reg_bin_n_bins = 4

        n = 24
        theta_all = np.linspace(-1.0, 1.0, n, dtype=np.float64).reshape(-1, 1)
        x_all = np.concatenate([np.cos(theta_all), np.sin(theta_all)], axis=1).astype(np.float64)
        split = n // 2
        with tempfile.TemporaryDirectory() as td:
            args.output_dir = str(Path(td))
            run_shared_fisher_estimation(
                args,
                dataset=object(),
                theta_all=theta_all,
                x_all=x_all,
                theta_train=theta_all[:split],
                x_train=x_all[:split],
                theta_validation=theta_all[split:],
                x_validation=x_all[split:],
                rng=np.random.default_rng(0),
            )
            out_dir = Path(td)
            self.assertTrue((out_dir / "h_matrix_results_theta_cov.npz").is_file())
            loss_npz = out_dir / "score_prior_training_losses.npz"
            self.assertTrue(loss_npz.is_file())
            z = np.load(loss_npz, allow_pickle=True)
            self.assertEqual(str(z["theta_field_method"].reshape(-1)[0]), "theta_flow_pre_post")
            self.assertEqual(tuple(z["theta_pre_post_pretrain_reg_train_losses"].shape), (1,))
            self.assertEqual(tuple(z["theta_pre_post_finetune_fm_train_losses"].shape), (1,))
            self.assertEqual(int(z["theta_pre_post_pretrain_synthetic_size"]), 20)
            self.assertEqual(int(z["theta_pre_post_pretrain_synthetic_train_size"]), 10)
            self.assertEqual(int(z["theta_pre_post_pretrain_synthetic_val_size"]), 10)
            self.assertTrue(bool(z["theta_pre_post_pretrain_resample_synthetic_each_epoch"]))
            self.assertEqual(int(z["theta_pre_post_pretrain_early_stopping_patience"]), 9)
            self.assertGreater(int(z["theta_pre_post_readout_trainable_params"]), 0)


if __name__ == "__main__":
    unittest.main()
