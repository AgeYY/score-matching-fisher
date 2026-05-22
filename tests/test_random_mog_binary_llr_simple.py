"""Unit tests for the simple binary random-MoG LLR diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from fisher.data import ToyCategoricalRandomMoGDataset


def _load_study_module():
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "bin" / "bench-binary-llr.py"
    spec = importlib.util.spec_from_file_location("bench_binary_llr", path)
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
    assert "random_mog_binary_llr_bench" in str(args.output_dir)
    assert args.latent_n_mc_train == 32
    assert args.ae_latent_dim is None
    assert args.ae_epochs == 1000
    assert args.vae_kl_weight == 1e-3
    assert args.label_vae_cls_weight == 1.0


def test_parser_normalizes_latent_inner_post_aliases() -> None:
    mod = _load_study_module()
    parser = mod.build_parser()

    assert parser.parse_args(["--method", "latent_belief_ctsm_v_binary_inner_post"]).method == (
        "latent_belief_ctsm_v_binary_inner_post"
    )
    assert parser.parse_args(["--method", "latent-belief-ctsm-v-binary-inner-post"]).method == (
        "latent_belief_ctsm_v_binary_inner_post"
    )
    assert parser.parse_args(["--method", "latent_belief_ctsm_v_binary_innner_post"]).method == (
        "latent_belief_ctsm_v_binary_inner_post"
    )


def test_parser_normalizes_ae_ctsm_v_aliases() -> None:
    mod = _load_study_module()
    parser = mod.build_parser()

    assert parser.parse_args(["--method", "ae-ctsm-v"]).method == "ae_ctsm_v"
    assert parser.parse_args(["--method", "ae_ctsm_v"]).method == "ae_ctsm_v"
    assert parser.parse_args(["--method", "vae-ctsm-v"]).method == "vae_ctsm_v"
    assert parser.parse_args(["--method", "vae_ctsm_v"]).method == "vae_ctsm_v"
    assert parser.parse_args(["--method", "label-vae-ctsm-v"]).method == "label_vae_ctsm_v"
    assert parser.parse_args(["--method", "label_vae_ctsm_v"]).method == "label_vae_ctsm_v"


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


def test_prepare_ae_latent_features_encodes_work_features(monkeypatch) -> None:
    mod = _load_study_module()

    class DummyAE(torch.nn.Module):
        def __init__(self, *, x_dim: int, latent_dim: int, hidden_dim: int, depth: int) -> None:
            super().__init__()
            self.x_dim = int(x_dim)
            self.latent_dim = int(latent_dim)
            self.hidden_dim = int(hidden_dim)
            self.depth = int(depth)

    def fake_train_observation_autoencoder(**kwargs):
        model = kwargs["model"]
        assert model.x_dim == 3
        assert model.latent_dim == 2
        assert kwargs["epochs"] == 5
        return {
            "train_losses": [1.0, 0.5],
            "val_losses": [1.2, 0.6],
            "val_monitor_losses": [1.2, 0.9],
            "best_val_loss": 0.9,
            "best_epoch": 2,
            "stopped_epoch": 5,
            "stopped_early": False,
        }

    def fake_encode_observations(*, model, x, device, batch_size):
        arr = np.asarray(x, dtype=np.float64)
        assert model.latent_dim == 2
        assert batch_size == 4
        return arr[:, :2] + 10.0

    monkeypatch.setattr(mod, "ObservationAutoencoder", DummyAE)
    monkeypatch.setattr(mod, "train_observation_autoencoder", fake_train_observation_autoencoder)
    monkeypatch.setattr(mod, "encode_observations", fake_encode_observations)

    args = mod.build_parser().parse_args(
        ["--method", "ae-ctsm-v", "--ae-latent-dim", "2", "--ae-epochs", "5", "--ae-batch-size", "4"]
    )
    x_train = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], dtype=np.float64)
    x_val = np.array([[2.0, 3.0, 4.0]], dtype=np.float64)
    x_test = np.array([[4.0, 5.0, 6.0], [6.0, 7.0, 8.0]], dtype=np.float64)
    means = np.array([[0.1, 0.2, 0.3], [0.8, 0.9, 1.0]], dtype=np.float64)

    got = mod.prepare_ae_latent_features(
        args,
        device=torch.device("cpu"),
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        means=means,
    )

    assert got["x_dim"] == 2
    assert got["x_train"].shape == (2, 2)
    assert got["x_validation"].shape == (1, 2)
    assert got["x_test"].shape == (2, 2)
    assert got["means"].shape == (2, 2)
    np.testing.assert_allclose(got["x_test"], x_test[:, :2] + 10.0)
    assert got["train_out"]["best_epoch"] == 2


def test_prepare_ae_latent_features_default_latent_dim_is_min_5_work_x_dim(monkeypatch) -> None:
    mod = _load_study_module()
    seen: list[tuple[int, int]] = []

    class DummyAE(torch.nn.Module):
        def __init__(self, *, x_dim: int, latent_dim: int, hidden_dim: int, depth: int) -> None:
            super().__init__()
            self.x_dim = int(x_dim)
            self.latent_dim = int(latent_dim)
            seen.append((self.x_dim, self.latent_dim))

    def fake_train_observation_autoencoder(**kwargs):
        return {
            "train_losses": [],
            "val_losses": [],
            "val_monitor_losses": [],
            "best_val_loss": 0.0,
            "best_epoch": 0,
            "stopped_epoch": 0,
            "stopped_early": False,
        }

    def fake_encode_observations(*, model, x, device, batch_size):
        arr = np.asarray(x, dtype=np.float64)
        return arr[:, : model.latent_dim]

    monkeypatch.setattr(mod, "ObservationAutoencoder", DummyAE)
    monkeypatch.setattr(mod, "train_observation_autoencoder", fake_train_observation_autoencoder)
    monkeypatch.setattr(mod, "encode_observations", fake_encode_observations)

    args = mod.build_parser().parse_args(["--method", "ae-ctsm-v"])
    for work_x_dim, expected_latent_dim in ((3, 3), (8, 5)):
        x_train = np.ones((2, work_x_dim), dtype=np.float64)
        x_val = np.ones((1, work_x_dim), dtype=np.float64)
        x_test = np.ones((2, work_x_dim), dtype=np.float64)
        means = np.ones((2, work_x_dim), dtype=np.float64)
        got = mod.prepare_ae_latent_features(
            args,
            device=torch.device("cpu"),
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            means=means,
        )
        assert got["x_dim"] == expected_latent_dim

    assert seen == [(3, 3), (8, 5)]


def test_prepare_vae_latent_features_repeats_samples_and_uses_min_5_default(monkeypatch) -> None:
    mod = _load_study_module()
    seen: list[tuple[int, int]] = []

    class DummyVAE(torch.nn.Module):
        def __init__(self, *, x_dim: int, latent_dim: int, hidden_dim: int, depth: int) -> None:
            super().__init__()
            self.x_dim = int(x_dim)
            self.latent_dim = int(latent_dim)
            seen.append((self.x_dim, self.latent_dim))

    def fake_train_observation_variational_autoencoder(**kwargs):
        model = kwargs["model"]
        assert model.x_dim == 8
        assert model.latent_dim == 5
        assert kwargs["beta"] == 0.25
        return {
            "train_losses": [1.0],
            "train_recon_losses": [0.8],
            "train_kl_losses": [0.2],
            "val_losses": [1.1],
            "val_recon_losses": [0.9],
            "val_kl_losses": [0.2],
            "val_monitor_losses": [1.1],
            "best_val_loss": 1.1,
            "best_epoch": 1,
            "stopped_epoch": 1,
            "stopped_early": False,
        }

    def fake_sample_observation_vae_latents(*, model, x, device, n_samples, batch_size):
        arr = np.asarray(x, dtype=np.float64)
        base = arr[:, : model.latent_dim]
        return np.repeat(base, int(n_samples), axis=0) + np.tile(
            np.arange(int(n_samples), dtype=np.float64).reshape(-1, 1),
            (arr.shape[0], model.latent_dim),
        )

    def fake_encode_observation_vae_means(*, model, x, device, batch_size):
        arr = np.asarray(x, dtype=np.float64)
        return arr[:, : model.latent_dim] + 100.0

    monkeypatch.setattr(mod, "ObservationVariationalAutoencoder", DummyVAE)
    monkeypatch.setattr(mod, "train_observation_variational_autoencoder", fake_train_observation_variational_autoencoder)
    monkeypatch.setattr(mod, "sample_observation_vae_latents", fake_sample_observation_vae_latents)
    monkeypatch.setattr(mod, "encode_observation_vae_means", fake_encode_observation_vae_means)

    args = mod.build_parser().parse_args(
        ["--method", "vae-ctsm-v", "--vae-n-samples", "4", "--latent-n-mc-eval", "3", "--vae-kl-weight", "0.25"]
    )
    x_train = np.ones((2, 8), dtype=np.float64)
    x_val = np.ones((1, 8), dtype=np.float64) * 2.0
    x_test = np.ones((3, 8), dtype=np.float64) * 3.0
    means = np.ones((2, 8), dtype=np.float64) * 4.0

    got = mod.prepare_vae_latent_features(
        args,
        device=torch.device("cpu"),
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        means=means,
    )

    assert seen == [(8, 5)]
    assert got["x_dim"] == 5
    assert got["x_train"].shape == (8, 5)
    assert got["x_validation"].shape == (4, 5)
    assert got["x_test_samples"].shape == (9, 5)
    assert got["x_train_mean"].shape == (2, 5)
    assert got["x_test_mean"].shape == (3, 5)
    assert got["n_samples"] == 4
    assert got["n_mc_eval"] == 3
    np.testing.assert_allclose(got["means"], np.ones((2, 5), dtype=np.float64) * 104.0)


def test_prepare_label_vae_latent_features_passes_labels_and_repeats_samples(monkeypatch) -> None:
    mod = _load_study_module()
    seen: list[tuple[int, int]] = []

    class DummyVAE(torch.nn.Module):
        def __init__(self, *, x_dim: int, latent_dim: int, hidden_dim: int, depth: int) -> None:
            super().__init__()
            self.x_dim = int(x_dim)
            self.latent_dim = int(latent_dim)
            seen.append((self.x_dim, self.latent_dim))

    def fake_train_label_guided_observation_variational_autoencoder(**kwargs):
        model = kwargs["model"]
        assert model.x_dim == 8
        assert model.latent_dim == 5
        assert kwargs["beta"] == 0.25
        assert kwargs["cls_weight"] == 2.5
        np.testing.assert_array_equal(kwargs["y_train"], np.array([0, 1], dtype=np.int64))
        np.testing.assert_array_equal(kwargs["y_val"], np.array([1], dtype=np.int64))
        return {
            "train_losses": [1.0],
            "train_recon_losses": [0.8],
            "train_kl_losses": [0.2],
            "train_cls_losses": [0.7],
            "val_losses": [1.1],
            "val_recon_losses": [0.9],
            "val_kl_losses": [0.2],
            "val_cls_losses": [0.6],
            "val_monitor_losses": [1.1],
            "best_val_loss": 1.1,
            "best_epoch": 1,
            "stopped_epoch": 1,
            "stopped_early": False,
        }

    def fake_sample_observation_vae_latents(*, model, x, device, n_samples, batch_size):
        arr = np.asarray(x, dtype=np.float64)
        base = arr[:, : model.latent_dim]
        return np.repeat(base, int(n_samples), axis=0) + np.tile(
            np.arange(int(n_samples), dtype=np.float64).reshape(-1, 1),
            (arr.shape[0], model.latent_dim),
        )

    def fake_encode_observation_vae_means(*, model, x, device, batch_size):
        arr = np.asarray(x, dtype=np.float64)
        return arr[:, : model.latent_dim] + 100.0

    monkeypatch.setattr(mod, "ObservationVariationalAutoencoder", DummyVAE)
    monkeypatch.setattr(
        mod,
        "train_label_guided_observation_variational_autoencoder",
        fake_train_label_guided_observation_variational_autoencoder,
    )
    monkeypatch.setattr(mod, "sample_observation_vae_latents", fake_sample_observation_vae_latents)
    monkeypatch.setattr(mod, "encode_observation_vae_means", fake_encode_observation_vae_means)

    args = mod.build_parser().parse_args(
        [
            "--method",
            "label-vae-ctsm-v",
            "--vae-n-samples",
            "4",
            "--latent-n-mc-eval",
            "3",
            "--vae-kl-weight",
            "0.25",
            "--label-vae-cls-weight",
            "2.5",
        ]
    )
    x_train = np.ones((2, 8), dtype=np.float64)
    x_val = np.ones((1, 8), dtype=np.float64) * 2.0
    x_test = np.ones((3, 8), dtype=np.float64) * 3.0
    means = np.ones((2, 8), dtype=np.float64) * 4.0

    got = mod.prepare_label_vae_latent_features(
        args,
        device=torch.device("cpu"),
        x_train=x_train,
        y_train=np.array([0, 1], dtype=np.int64),
        x_val=x_val,
        y_val=np.array([1], dtype=np.int64),
        x_test=x_test,
        means=means,
    )

    assert seen == [(8, 5)]
    assert got["x_dim"] == 5
    assert got["x_train"].shape == (8, 5)
    assert got["x_validation"].shape == (4, 5)
    assert got["x_test_samples"].shape == (9, 5)
    assert got["x_train_mean"].shape == (2, 5)
    assert got["x_test_mean"].shape == (3, 5)
    assert got["n_samples"] == 4
    assert got["n_mc_eval"] == 3
    np.testing.assert_allclose(got["means"], np.ones((2, 5), dtype=np.float64) * 104.0)


def test_logmeanexp_rows_is_stable_for_vae_mc_estimator() -> None:
    mod = _load_study_module()

    got = mod._logmeanexp_rows(np.array([1000.0, 1002.0, -1000.0, -1002.0]), n_rows=2, n_samples=2)
    expected = np.array(
        [
            1002.0 + np.log((np.exp(-2.0) + 1.0) / 2.0),
            -1000.0 + np.log((1.0 + np.exp(-2.0)) / 2.0),
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_projection_panel_title_includes_ae_latent_and_pr_x_dim() -> None:
    mod = _load_study_module()

    assert mod._projection_panel_title("dataset PCA projection (x_dim=30)") == "dataset PCA projection (x_dim=30)"
    assert mod._projection_panel_title(
        "dataset PCA projection (x_dim=8)",
        projection_space_label="AE latent PCA projection",
        projection_latent_dim=5,
        projection_source_x_dim=30,
        projection_source_x_dim_label="PR x_dim",
    ) == "AE latent PCA projection (latent_dim=5; PR x_dim=30)"
    assert mod._projection_panel_title(
        "dataset PCA projection (x_dim=5)",
        projection_space_label="AE latent PCA projection",
        projection_latent_dim=5,
        projection_source_x_dim=100,
    ) == "AE latent PCA projection (latent_dim=5; x_dim=100)"
    assert mod._projection_panel_title(
        "dataset PCA projection (x_dim=5)",
        projection_space_label="VAE latent PCA projection",
        projection_latent_dim=5,
        projection_source_x_dim=30,
        projection_source_x_dim_label="PR x_dim",
    ) == "VAE latent PCA projection (latent_dim=5; PR x_dim=30)"
    assert mod._projection_panel_title(
        "dataset PCA projection (x_dim=5)",
        projection_space_label="Label-VAE latent PCA projection",
        projection_latent_dim=5,
        projection_source_x_dim=30,
        projection_source_x_dim_label="PR x_dim",
    ) == "Label-VAE latent PCA projection (latent_dim=5; PR x_dim=30)"


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
        ae_enabled=True,
        ae_train_losses=np.array([2.0, 1.0, 0.6]),
        ae_val_losses=np.array([2.2, 1.2, 0.7]),
        ae_val_monitor_losses=np.array([2.2, 1.7, 1.2]),
        ae_best_epoch=3,
        projection_space_label="AE latent PCA projection",
        projection_latent_dim=3,
        projection_source_x_dim=30,
        projection_source_x_dim_label="PR x_dim",
    )

    assert svg == (tmp_path / "simple_binary_llr_diagnostic.svg").resolve()
    assert png == (tmp_path / "simple_binary_llr_diagnostic.png").resolve()
    assert svg.is_file() and svg.stat().st_size > 0
    assert png.is_file() and png.stat().st_size > 0
