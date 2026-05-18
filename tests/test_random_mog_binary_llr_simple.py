"""Unit tests for the simple binary random-MoG LLR diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from fisher.data import ToyCategoricalRandomMoGDataset


def _load_study_module():
    path = Path(__file__).resolve().parent / "study_random_mog_binary_llr_simple.py"
    spec = importlib.util.spec_from_file_location("study_random_mog_binary_llr_simple", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_analytic_binary_llr_matches_dataset_logp_difference() -> None:
    mod = _load_study_module()
    dataset = ToyCategoricalRandomMoGDataset(num_categories=2, x_dim=2, theta_dim=2, seed=11)
    theta, x = dataset.sample_joint(25)
    theta0 = np.tile(np.array([[1.0, 0.0]]), (x.shape[0], 1))
    theta1 = np.tile(np.array([[0.0, 1.0]]), (x.shape[0], 1))

    got = mod.analytic_binary_llr(x, dataset._mog_means, dataset._mog_variances)
    expected = dataset.log_p_x_given_theta(x, theta1) - dataset.log_p_x_given_theta(x, theta0)

    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_parser_defaults_are_binary_native_random_mog() -> None:
    mod = _load_study_module()
    args = mod.build_parser().parse_args([])

    assert args.num_categories == 2
    assert args.theta_dim == 2
    assert args.x_dim == 2
    assert args.pr_project is False
    assert args.pr_dim == 10
    assert args.pr_cache_dir == "data/pr_autoencoder_cache"
    assert "random_mog_binary_llr_simple" in str(args.output_dir)


def test_prepare_work_features_pr_project_uses_high_dimensional_work_features(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = _load_study_module()

    class DummyPRModel(torch.nn.Module):
        def forward(self, x):
            return torch.cat([x, x.sum(dim=1, keepdim=True)], dim=1), x

    def fake_project_x_through_pr_autoencoder(x_low, *, config, seed, device, cache_dir, force_retrain):
        assert config.z_dim == 2
        assert config.h_dim == 3
        x_arr = np.asarray(x_low, dtype=np.float64)
        x_embed = np.column_stack([x_arr, x_arr.sum(axis=1)])
        metrics = {
            "loss": np.array([1.0, 0.5]),
            "recon": np.array([0.8, 0.4]),
            "pr": np.array([1.2, 1.5]),
        }
        return x_embed, tmp_path / "cache_run", False, metrics, DummyPRModel()

    monkeypatch.setattr(mod, "project_x_through_pr_autoencoder", fake_project_x_through_pr_autoencoder)
    args = mod.build_parser().parse_args(["--pr-project", "--pr-dim", "3", "--device", "cpu"])
    x_train = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    x_val = np.array([[2.0, 3.0]], dtype=np.float64)
    x_test = np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float64)
    means = np.array([[0.1, 0.2], [0.8, 0.9]], dtype=np.float64)

    got = mod.prepare_work_features(
        args,
        device=torch.device("cpu"),
        x_train_native=x_train,
        x_val_native=x_val,
        x_test_native=x_test,
        means_native=means,
    )

    assert got["pr_projected"] is True
    assert got["x_dim"] == 3
    assert got["x_train"].shape == (2, 3)
    assert got["x_validation"].shape == (1, 3)
    assert got["x_test"].shape == (2, 3)
    assert got["means"].shape == (2, 3)
    np.testing.assert_allclose(got["x_test"][:, 2], x_test.sum(axis=1))
    np.testing.assert_allclose(got["means"][:, 2], means.sum(axis=1))
    np.testing.assert_allclose(got["pr_train_loss"], np.array([1.0, 0.5]))


def test_save_diagnostic_figure_writes_svg_and_png(tmp_path: Path) -> None:
    mod = _load_study_module()
    rng = np.random.default_rng(3)
    means = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]], dtype=np.float64)
    variances = np.ones((2, 2), dtype=np.float64) * 0.2
    x_train_native = np.vstack([rng.normal(0.0, 0.2, size=(8, 2)), rng.normal(1.0, 0.2, size=(8, 2))])
    x_train = np.column_stack([x_train_native, x_train_native.sum(axis=1)])
    y_train = np.repeat([0, 1], 8)
    x_val = x_train[:6]
    y_val = y_train[:6]
    x_test_native = np.vstack([rng.normal(0.0, 0.2, size=(10, 2)), rng.normal(1.0, 0.2, size=(10, 2))])
    x_test = np.column_stack([x_test_native, x_test_native.sum(axis=1)])
    y_test = np.repeat([0, 1], 10)
    gt = mod.analytic_binary_llr(x_test_native, means[:, :2], variances)

    svg, png = mod.save_diagnostic_figure(
        output_base=tmp_path / "simple_binary_llr_diagnostic",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        means=means,
        ctsm_train_losses=np.array([1.2, 0.9, 0.8]),
        ctsm_val_losses=np.array([1.3, 1.0, 0.85]),
        gt_llr=gt,
        binary_llr=gt + 0.1,
        ctsm_llr=gt - 0.2,
    )

    assert svg == (tmp_path / "simple_binary_llr_diagnostic.svg").resolve()
    assert png == (tmp_path / "simple_binary_llr_diagnostic.png").resolve()
    assert svg.is_file() and svg.stat().st_size > 0
    assert png.is_file() and png.stat().st_size > 0
