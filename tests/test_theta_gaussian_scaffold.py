from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch

from fisher.cli_shared_fisher import add_estimation_arguments
from fisher.models import ConditionalThetaFlowVelocity, PriorThetaFlowVelocity
from fisher.h_matrix import HMatrixEstimator
from fisher.shared_fisher_est import run_shared_fisher_estimation, validate_estimation_args
from fisher.theta_gaussian_scaffold import ThetaGaussianScaffold
from fisher.trainers import train_conditional_theta_flow_model


def test_scaffold_fit_q0_and_matched_sampling_are_finite() -> None:
    theta = np.linspace(-1.0, 1.0, 40, dtype=np.float64).reshape(-1, 1)
    x = np.concatenate([np.cos(np.pi * theta), np.sin(np.pi * theta)], axis=1)
    scaffold = ThetaGaussianScaffold.fit(
        theta_train=theta,
        x_train=x,
        n_bins=10,
        grid_size=64,
        n_components=3,
        em_steps=5,
        variance_floor=1e-6,
        theta_low=-1.0,
        theta_high=1.0,
    )
    q, logw = scaffold.q0_bins(x[:3])
    assert q.shape == (3, 10)
    assert logw.shape == (3, 10)
    np.testing.assert_allclose(np.sum(q, axis=1), np.ones(3), rtol=1e-8, atol=1e-8)
    pis, means, vars_ = scaffold.mixture_params_np(x[:3])
    assert pis.shape == (3, 3)
    assert means.shape == (3, 3)
    assert vars_.shape == (3, 3)
    np.testing.assert_allclose(np.sum(pis, axis=1), np.ones(3), rtol=1e-8, atol=1e-8)
    assert np.isfinite(means).all()
    assert np.isfinite(vars_).all()
    samples, branch_ids = scaffold.sample_matched_np(theta[:5], x[:5], np.random.default_rng(0))
    assert samples.shape == (5, 1)
    assert branch_ids.shape == (5,)
    assert np.all((branch_ids >= 0) & (branch_ids < 3))
    assert np.isfinite(samples).all()
    assert np.all(samples >= -1.0)
    assert np.all(samples <= 1.0)
    lp = scaffold.log_prob_np(samples, x[:5])
    assert lp.shape == (5,)
    assert np.isfinite(lp).all()


def test_train_conditional_theta_flow_uses_scaffold_source_sampler() -> None:
    class CountingSampler:
        def __init__(self) -> None:
            self.calls = 0

        def sample_matched_torch(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return theta + 0.01 * torch.tanh(x[:, :1])

    theta = np.linspace(-1.0, 1.0, 18, dtype=np.float64).reshape(-1, 1)
    x = np.concatenate([np.cos(theta), np.sin(theta)], axis=1).astype(np.float64)
    model = ConditionalThetaFlowVelocity(x_dim=2, hidden_dim=12, depth=1).to(torch.device("cpu"))
    sampler = CountingSampler()
    out = train_conditional_theta_flow_model(
        model=model,
        theta_train=theta[:12],
        x_train=x[:12],
        theta_val=theta[12:],
        x_val=x[12:],
        epochs=2,
        batch_size=6,
        lr=1e-3,
        device=torch.device("cpu"),
        log_every=99,
        early_stopping_patience=100,
        scheduler_name="vp",
        source_sampler=sampler,
    )
    assert sampler.calls > 0
    assert len(out["train_fm_losses"]) == 2
    assert np.isfinite(np.asarray(out["train_fm_losses"], dtype=np.float64)).all()


def test_h_matrix_gaussian_scaffold_uses_base_log_prob() -> None:
    theta = np.linspace(-1.0, 1.0, 6, dtype=np.float64).reshape(-1, 1)
    x = np.concatenate([np.cos(theta), np.sin(theta)], axis=1).astype(np.float64)
    scaffold = ThetaGaussianScaffold.fit(
        theta_train=theta,
        x_train=x,
        n_bins=3,
        grid_size=32,
        n_components=2,
        em_steps=4,
        theta_low=-1.0,
        theta_high=1.0,
    )
    post = ConditionalThetaFlowVelocity(x_dim=2, hidden_dim=8, depth=1)
    prior = PriorThetaFlowVelocity(hidden_dim=8, depth=1)
    est = HMatrixEstimator(
        model_post=post,
        model_prior=prior,
        sigma_eval=1.0,
        device=torch.device("cpu"),
        pair_batch_size=64,
        field_method="theta_flow_gaussian_scaffold",
        flow_scheduler="vp",
        flow_ode_steps=4,
        theta_gaussian_scaffold=scaffold,
    )
    out = est.run(theta=theta, x=x, restore_original_order=False)
    assert out.field_method == "theta_flow_gaussian_scaffold"
    assert out.theta_flow_log_post_matrix is not None
    assert out.theta_flow_log_prior_matrix is not None
    assert out.theta_flow_log_base_matrix is not None
    assert np.isfinite(out.c_matrix).all()
    assert np.isfinite(out.h_sym).all()


def test_shared_estimation_theta_flow_gaussian_scaffold_smoke() -> None:
    parser = argparse.ArgumentParser()
    add_estimation_arguments(parser)
    args = parser.parse_args([])
    args.theta_field_method = "theta_flow_gaussian_scaffold"
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
    args.flow_hidden_dim = 12
    args.prior_hidden_dim = 12
    args.flow_depth = 1
    args.prior_depth = 1
    args.flow_early_patience = 100
    args.prior_early_patience = 100
    args.flow_scheduler = "vp"
    args.theta_gaussian_scaffold_bin_n_bins = 10
    args.theta_gaussian_scaffold_grid_size = 64
    args.theta_gaussian_scaffold_n_components = 3
    args.theta_gaussian_scaffold_em_steps = 4
    validate_estimation_args(args)

    n = 20
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
        assert (out_dir / "theta_gaussian_scaffold.npz").is_file()
        h_path = out_dir / "h_matrix_results_theta_cov.npz"
        loss_path = out_dir / "score_prior_training_losses.npz"
        assert h_path.is_file()
        assert loss_path.is_file()
        h_z = np.load(h_path, allow_pickle=True)
        assert str(h_z["h_field_method"].reshape(-1)[0]) == "theta_flow_gaussian_scaffold"
        assert "theta_flow_log_base_matrix" in h_z.files
        assert np.isfinite(h_z["theta_flow_log_base_matrix"]).all()
        z = np.load(loss_path, allow_pickle=True)
        assert str(z["theta_field_method"].reshape(-1)[0]) == "theta_flow_gaussian_scaffold"
        assert int(z["theta_gaussian_scaffold_bin_n_bins"]) == 10
        assert int(z["theta_gaussian_scaffold_grid_size"]) == 64
        assert int(z["theta_gaussian_scaffold_n_components"]) == 3
        assert int(z["theta_gaussian_scaffold_em_steps"]) == 4
