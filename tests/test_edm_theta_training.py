from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from fisher.models import ConditionalScore1D, ConditionalThetaEDM, PriorScore1D, PriorThetaEDM
from fisher.shared_fisher_est import build_posterior_score_model, build_prior_score_model, validate_estimation_args
from fisher.trainers import sample_edm_sigmas, train_prior_theta_edm_model, train_theta_edm_model


class TestEDMThetaTraining(unittest.TestCase):
    def test_sample_edm_sigmas_positive(self) -> None:
        sig = sample_edm_sigmas(batch_size=128, p_mean=-1.2, p_std=1.2, device=torch.device("cpu"))
        self.assertEqual(sig.shape, (128, 1))
        self.assertTrue(bool(torch.all(sig > 0.0)))

    def test_conditional_theta_edm_predict_score_matches_formula(self) -> None:
        backbone = ConditionalScore1D(x_dim=2, hidden_dim=16, depth=2, zero_out_init=True)
        model = ConditionalThetaEDM(backbone=backbone, sigma_data=0.5)
        theta = torch.tensor([[0.4], [-0.2], [0.1]], dtype=torch.float32)
        x = torch.zeros((3, 2), dtype=torch.float32)
        sigma_eval = 0.3
        score = model.predict_score(theta, x, sigma_eval=sigma_eval)
        expected = -theta / (sigma_eval**2 + model.sigma_data**2)
        torch.testing.assert_close(score, expected, rtol=1e-5, atol=1e-6)

    def test_edm_trainers_smoke_cpu(self) -> None:
        rng = np.random.default_rng(0)
        n = 64
        theta = rng.normal(size=(n, 1)).astype(np.float32)
        x = rng.normal(size=(n, 2)).astype(np.float32)
        theta_val = theta[:16]
        x_val = x[:16]
        theta_train = theta[16:]
        x_train = x[16:]

        post = ConditionalThetaEDM(
            backbone=ConditionalScore1D(x_dim=2, hidden_dim=32, depth=2, use_log_sigma=False),
            sigma_data=0.5,
        )
        out_post = train_theta_edm_model(
            model=post,
            theta_train=theta_train,
            x_train=x_train,
            epochs=2,
            batch_size=16,
            lr=1e-3,
            device=torch.device("cpu"),
            log_every=1,
            theta_val=theta_val,
            x_val=x_val,
            p_mean=-1.2,
            p_std=1.2,
            sigma_data=0.5,
        )
        self.assertEqual(len(out_post["train_losses"]), 2)

        prior = PriorThetaEDM(
            backbone=PriorScore1D(hidden_dim=32, depth=2, use_log_sigma=False),
            sigma_data=0.5,
        )
        out_prior = train_prior_theta_edm_model(
            model=prior,
            theta_train=theta_train,
            epochs=2,
            batch_size=16,
            lr=1e-3,
            device=torch.device("cpu"),
            log_every=1,
            theta_val=theta_val,
            p_mean=-1.2,
            p_std=1.2,
            sigma_data=0.5,
        )
        self.assertEqual(len(out_prior["train_losses"]), 2)

    def test_shared_builders_return_edm_wrappers(self) -> None:
        args = SimpleNamespace(
            score_train_objective="edm",
            score_sigma_feature_mode="auto",
            score_noise_mode="continuous",
            score_arch="mlp",
            x_dim=2,
            score_hidden_dim=32,
            score_depth=2,
            score_use_layer_norm=False,
            score_zero_out_init=False,
            score_gated_film=False,
            edm_sigma_data=0.7,
            prior_sigma_feature_mode="auto",
            prior_score_arch="mlp",
            prior_hidden_dim=32,
            prior_depth=2,
            prior_use_layer_norm=False,
            prior_zero_out_init=False,
            prior_gated_film=False,
        )
        post = build_posterior_score_model(args, torch.device("cpu"))
        prior = build_prior_score_model(args, torch.device("cpu"))
        self.assertIsInstance(post, ConditionalThetaEDM)
        self.assertIsInstance(prior, PriorThetaEDM)

    def test_validate_rejects_edm_with_sigma_normalization(self) -> None:
        args = SimpleNamespace(
            theta_field_method="dsm",
            score_eval_sigmas=2,
            score_early_patience=1,
            score_early_min_delta=0.0,
            score_early_ema_alpha=0.5,
            score_early_ema_warmup_epochs=0,
            score_sigma_min_alpha=0.05,
            score_sigma_max_alpha=0.25,
            score_arch="mlp",
            prior_score_arch="mlp",
            score_sigma_feature_mode="auto",
            prior_sigma_feature_mode="auto",
            score_optimizer="adamw",
            prior_optimizer="adamw",
            score_weight_decay=0.0,
            prior_weight_decay=0.0,
            score_lr_scheduler="none",
            prior_lr_scheduler="none",
            score_lr_warmup_frac=0.0,
            prior_lr_warmup_frac=0.0,
            score_huber_delta=1.0,
            prior_huber_delta=1.0,
            score_max_grad_norm=0.0,
            prior_max_grad_norm=0.0,
            score_loss_type="mse",
            prior_loss_type="mse",
            score_train_objective="edm",
            score_sigma_sample_mode="uniform_log",
            score_sigma_sample_beta=2.0,
            edm_sigma_data=0.5,
            edm_p_std=1.2,
            score_noise_mode="continuous",
            score_normalize_by_sigma=True,
            prior_normalize_by_sigma=False,
            score_proxy_min_mult=0.1,
            score_proxy_max_mult=2.0,
            score_fixed_sigma=0.02,
            decoder_val_frac=0.2,
            decoder_min_val_class_size=1,
            decoder_early_patience=1,
            decoder_early_min_delta=0.0,
            decoder_early_ema_alpha=0.5,
            decoder_min_class_count=1,
            prior_epochs=1,
            prior_early_patience=1,
            prior_early_min_delta=0.0,
            prior_early_ema_alpha=0.5,
            prior_early_ema_warmup_epochs=0,
            h_batch_size=1,
            h_sigma_eval=-1.0,
            flow_epochs=1,
            flow_batch_size=1,
            flow_lr=1e-3,
            flow_early_patience=1,
            flow_early_min_delta=0.0,
            flow_early_ema_alpha=0.5,
            flow_eval_t=0.5,
            compute_h_matrix=False,
            prior_enable=True,
            skip_shared_fisher_gt_compare=False,
            dsm_stability_preset="legacy",
        )
        with self.assertRaisesRegex(ValueError, "score-normalize-by-sigma"):
            validate_estimation_args(args)


if __name__ == "__main__":
    unittest.main()
