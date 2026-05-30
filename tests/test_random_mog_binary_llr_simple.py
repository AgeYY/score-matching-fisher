"""Coverage for restored VAE-CTSM-v helpers without the removed bench script."""

from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch

from fisher import vae_ctsm_v as vae_mod
from fisher.h_decoding_categorical_twofig import _train_one_method, build_parser, parse_methods


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        vae_latent_dim=0,
        vae_hidden_dim=16,
        vae_depth=1,
        vae_epochs=3,
        vae_batch_size=4,
        vae_lr=1e-3,
        vae_weight_decay=0.0,
        vae_early_patience=2,
        vae_early_min_delta=1e-4,
        vae_early_ema_alpha=0.05,
        vae_kl_weight=0.25,
        vae_n_samples=4,
        latent_n_mc_eval=3,
        ctsm_hidden_dim=8,
        ctsm_binary_epochs=2,
        ctsm_batch_size=4,
        ctsm_lr=1e-3,
        ctsm_weight_decay=0.0,
        ctsm_two_sb_var=2.0,
        ctsm_path_schedule="linear",
        ctsm_path_eps=1e-12,
        ctsm_factor=1.0,
        ctsm_t_eps=0.01,
        ctsm_int_n_time=8,
        h_batch_size=16,
        clf_min_class_count=1,
        flow_early_patience=2,
        flow_early_min_delta=1e-4,
        flow_early_ema_alpha=0.05,
        flow_restore_best=True,
        log_every=1000,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_parse_methods_accepts_vae_ctsm_v_aliases() -> None:
    assert parse_methods("vae-ctsm-v") == ["vae_ctsm_v"]
    assert parse_methods("vae_ctsm_v, vae-ctsm-v") == ["vae_ctsm_v"]
    assert parse_methods("vae-x-flow, vae_x_flow") == ["vae_x_flow"]
    assert parse_methods("vae-xflow-sir-lrank, vae_xflow_sir_lrank") == ["vae_xflow_sir_lrank"]
    assert parse_methods("vae-bin-gaussian, vae_bin_gaussian") == ["vae_bin_gaussian"]
    parsed = build_parser().parse_args(["--methods", "vae-ctsm-v", "--vae-latent-dim", "2"])
    assert parse_methods(parsed.methods) == ["vae_ctsm_v"]
    assert parsed.vae_latent_dim == 2
    assert parsed.vae_hidden_dim == 128
    assert parsed.vae_depth == 4
    assert parsed.vae_epochs == 5000
    assert parsed.vae_early_patience == 500
    assert parsed.vae_kl_weight == 0.01


def test_logmeanexp_rows_is_stable_for_vae_mc_estimator() -> None:
    got = vae_mod.logmeanexp_rows(
        np.array([1000.0, 1002.0, -1000.0, -1002.0]),
        n_rows=2,
        n_samples=2,
    )
    expected = np.array(
        [
            1002.0 + np.log((np.exp(-2.0) + 1.0) / 2.0),
            -1000.0 + np.log((1.0 + np.exp(-2.0)) / 2.0),
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="n_rows\\*n_samples"):
        vae_mod.logmeanexp_rows(np.array([1.0, 2.0, 3.0]), n_rows=2, n_samples=2)


def test_prepare_vae_latent_features_repeats_samples_and_uses_min_5_default(monkeypatch) -> None:
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
        offsets = np.tile(np.arange(int(n_samples), dtype=np.float64), arr.shape[0])
        return np.repeat(base, int(n_samples), axis=0) + offsets.reshape(-1, 1)

    def fake_encode_observation_vae_means(*, model, x, device, batch_size):
        arr = np.asarray(x, dtype=np.float64)
        return arr[:, : model.latent_dim] + 100.0

    monkeypatch.setattr(vae_mod, "ObservationVariationalAutoencoder", DummyVAE)
    monkeypatch.setattr(vae_mod, "train_observation_variational_autoencoder", fake_train_observation_variational_autoencoder)
    monkeypatch.setattr(vae_mod, "sample_observation_vae_latents", fake_sample_observation_vae_latents)
    monkeypatch.setattr(vae_mod, "encode_observation_vae_means", fake_encode_observation_vae_means)

    got = vae_mod.prepare_vae_latent_features(
        _args(),
        device=torch.device("cpu"),
        x_train=np.ones((2, 8), dtype=np.float64),
        x_val=np.ones((1, 8), dtype=np.float64) * 2.0,
        x_eval=np.ones((3, 8), dtype=np.float64) * 3.0,
    )

    assert seen == [(8, 5)]
    assert got["x_dim"] == 5
    assert got["x_train"].shape == (8, 5)
    assert got["x_validation"].shape == (4, 5)
    assert got["x_eval_samples"].shape == (9, 5)
    assert got["x_train_mean"].shape == (2, 5)
    assert got["x_eval_mean"].shape == (3, 5)
    assert got["n_samples"] == 4
    assert got["n_mc_eval"] == 3


def test_prepare_vae_mean_features_uses_posterior_means(monkeypatch) -> None:
    class DummyVAE(torch.nn.Module):
        def __init__(self, *, x_dim: int, latent_dim: int, hidden_dim: int, depth: int) -> None:
            super().__init__()
            self.x_dim = int(x_dim)
            self.latent_dim = int(latent_dim)

    def fake_train_observation_variational_autoencoder(**kwargs):
        return {"train_losses": [1.0], "val_losses": [1.2], "val_monitor_losses": [1.2]}

    def fake_encode_observation_vae_means(*, model, x, device, batch_size):
        arr = np.asarray(x, dtype=np.float64)
        return arr[:, : model.latent_dim] + 10.0

    monkeypatch.setattr(vae_mod, "ObservationVariationalAutoencoder", DummyVAE)
    monkeypatch.setattr(vae_mod, "train_observation_variational_autoencoder", fake_train_observation_variational_autoencoder)
    monkeypatch.setattr(vae_mod, "encode_observation_vae_means", fake_encode_observation_vae_means)

    got = vae_mod.prepare_vae_mean_features(
        _args(vae_latent_dim=2),
        device=torch.device("cpu"),
        x_train=np.ones((3, 4), dtype=np.float64),
        x_val=np.ones((2, 4), dtype=np.float64) * 2.0,
        x_eval=np.ones((5, 4), dtype=np.float64) * 3.0,
    )

    assert got["x_train"].shape == (3, 2)
    assert got["x_validation"].shape == (2, 2)
    assert got["x_eval"].shape == (5, 2)
    assert got["input_x_dim"] == 4
    np.testing.assert_allclose(got["x_eval"], np.full((5, 2), 13.0))


def test_train_vae_ctsm_v_binary_delta_shapes_and_payload(monkeypatch) -> None:
    args = _args()
    x_train = np.ones((4, 3), dtype=np.float64)
    x_val = np.ones((2, 3), dtype=np.float64)
    x_all = np.ones((4, 3), dtype=np.float64)
    bins_train = np.array([0, 1, 0, 1], dtype=np.int64)
    bins_val = np.array([0, 1], dtype=np.int64)
    bins_all = np.array([0, 1, 0, 1], dtype=np.int64)

    def fake_prepare_vae_latent_features(args_in, *, device, x_train, x_val, x_eval):
        assert args_in is args
        return {
            "x_train": np.repeat(x_train[:, :2], 2, axis=0),
            "x_validation": np.repeat(x_val[:, :2], 2, axis=0),
            "x_eval_samples": np.repeat(x_eval[:, :2], 3, axis=0),
            "x_train_mean": x_train[:, :2],
            "x_validation_mean": x_val[:, :2],
            "x_eval_mean": x_eval[:, :2],
            "x_dim": 2,
            "input_x_dim": 3,
            "train_out": {"train_losses": [0.9], "val_losses": [1.0], "val_monitor_losses": [1.0]},
            "n_samples": 2,
            "n_mc_eval": 3,
        }

    def fake_train_binary_ctsm_v_model(**kwargs):
        assert kwargs["x0_train"].shape == (4, 2)
        assert kwargs["x1_train"].shape == (4, 2)
        return {"train_losses": [0.5], "val_losses": [0.4], "val_monitor_losses": [0.4]}

    def fake_estimate_binary_ctsm_v_log_ratio(model, x, *, device, batch_size, eps1, eps2, n_time):
        assert np.asarray(x).shape == (12, 2)
        return np.log(np.arange(1, 13, dtype=np.float64))

    monkeypatch.setattr(vae_mod, "prepare_vae_latent_features", fake_prepare_vae_latent_features)
    monkeypatch.setattr(vae_mod, "train_binary_ctsm_v_model", fake_train_binary_ctsm_v_model)
    monkeypatch.setattr(vae_mod, "estimate_binary_ctsm_v_log_ratio", fake_estimate_binary_ctsm_v_log_ratio)

    got = vae_mod.train_vae_ctsm_v_binary_delta(
        args,
        dev=torch.device("cpu"),
        x_train=x_train,
        bins_train=bins_train,
        x_val=x_val,
        bins_val=bins_val,
        x_all=x_all,
        bins_all=bins_all,
        k_cat=2,
    )

    assert got["delta_l"].shape == (4, 4)
    assert got["ctsm_binary_llr_1_minus_0"].shape == (4,)
    assert got["vae_payload"]["latent_dim"] == 2
    assert got["vae_payload"]["input_x_dim"] == 3
    np.testing.assert_allclose(got["ctsm_binary_llr_1_minus_0"], np.log([2.0, 5.0, 8.0, 11.0]))
    with pytest.raises(ValueError, match="exactly two categories"):
        vae_mod.train_vae_ctsm_v_binary_delta(
            args,
            dev=torch.device("cpu"),
            x_train=x_train,
            bins_train=bins_train,
            x_val=x_val,
            bins_val=bins_val,
            x_all=x_all,
            bins_all=bins_all,
            k_cat=3,
        )


def test_categorical_train_one_method_dispatches_vae_ctsm_v_multiclass(monkeypatch) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    def fake_vae_mean_result(args, *, dev, x_train, x_val, x_all):
        return {
            "x_train": np.asarray(x_train, dtype=np.float64) + 10.0,
            "x_val": np.asarray(x_val, dtype=np.float64) + 20.0,
            "x_all": np.asarray(x_all, dtype=np.float64) + 30.0,
            "vae_payload": {"enabled": True, "latent_dim": 2},
            "vae_train_out": {"train_losses": [1.0]},
        }

    seen = {}

    def fake_train_ctsm_v_delta(args, **kwargs):
        seen.update(kwargs)
        return {
            "delta_l": np.zeros((3, 3), dtype=np.float64),
            "train_out": {},
            "ctsm_theta_encoding": np.asarray(["one_hot"], dtype=object),
        }

    monkeypatch.setattr(cat, "_vae_mean_result", fake_vae_mean_result)
    monkeypatch.setattr(cat, "_train_ctsm_v_delta", fake_train_ctsm_v_delta)
    out = _train_one_method(
        _args(),
        dev=torch.device("cpu"),
        method_name="vae_ctsm_v",
        theta_train=np.eye(3, dtype=np.float64),
        x_train=np.ones((3, 2), dtype=np.float64),
        theta_val=np.eye(3, dtype=np.float64),
        x_val=np.ones((3, 2), dtype=np.float64),
        theta_all=np.eye(3, dtype=np.float64),
        x_all=np.ones((3, 2), dtype=np.float64),
        bins_train=np.array([0, 1, 2], dtype=np.int64),
        bins_val=np.array([0, 1, 2], dtype=np.int64),
        bins_all=np.array([0, 1, 2], dtype=np.int64),
        k_cat=3,
    )
    assert out["delta_l"].shape == (3, 3)
    assert out["vae_payload"]["enabled"] is True
    assert out["ctsm_theta_encoding"][0] == "one_hot_vae_mean_x"
    assert seen["theta_all"].shape == (3, 3)
    np.testing.assert_allclose(seen["x_train"], np.full((3, 2), 11.0))


@pytest.mark.parametrize(
    ("method_name", "expected_base"),
    [("vae_x_flow", "x_flow"), ("vae_xflow_sir_lrank", "xflow_sir_lrank")],
)
def test_categorical_train_one_method_dispatches_vae_wrapped_methods(monkeypatch, method_name, expected_base) -> None:
    import fisher.h_decoding_categorical_twofig as cat

    def fake_vae_mean_result(args, *, dev, x_train, x_val, x_all):
        return {
            "x_train": np.asarray(x_train, dtype=np.float64) + 10.0,
            "x_val": np.asarray(x_val, dtype=np.float64) + 20.0,
            "x_all": np.asarray(x_all, dtype=np.float64) + 30.0,
            "vae_payload": {"enabled": True, "latent_dim": 2},
            "vae_train_out": {"train_losses": [1.0]},
        }

    seen = {}

    def fake_train_x_flow_delta(args, **kwargs):
        seen["base"] = "x_flow"
        seen.update(kwargs)
        return {"delta_l": np.zeros((2, 2), dtype=np.float64), "train_out": {}}

    def fake_train_linear_x_flow_delta(args, **kwargs):
        seen["base"] = kwargs["method_name"]
        seen.update(kwargs)
        return {"delta_l": np.zeros((2, 2), dtype=np.float64), "train_out": {}}

    monkeypatch.setattr(cat, "_vae_mean_result", fake_vae_mean_result)
    monkeypatch.setattr(cat, "_train_x_flow_delta", fake_train_x_flow_delta)
    monkeypatch.setattr(cat, "_train_linear_x_flow_delta", fake_train_linear_x_flow_delta)

    out = _train_one_method(
        _args(),
        dev=torch.device("cpu"),
        method_name=method_name,
        theta_train=np.eye(2, dtype=np.float64),
        x_train=np.ones((2, 2), dtype=np.float64),
        theta_val=np.eye(2, dtype=np.float64),
        x_val=np.ones((2, 2), dtype=np.float64),
        theta_all=np.eye(2, dtype=np.float64),
        x_all=np.ones((2, 2), dtype=np.float64),
        bins_train=np.array([0, 1], dtype=np.int64),
        bins_val=np.array([0, 1], dtype=np.int64),
        bins_all=np.array([0, 1], dtype=np.int64),
        k_cat=2,
    )

    assert out["vae_payload"]["enabled"] is True
    assert seen["base"] == expected_base
    np.testing.assert_allclose(seen["x_train"], np.full((2, 2), 11.0))
