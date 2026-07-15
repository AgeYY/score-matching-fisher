from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from fisher import distance_comparison as dc
from fisher.flow_matching_skl import FlowSKLResult
from fisher.shared_dataset_io import SharedDatasetBundle


def _load_cli_module():
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "bin" / "compare_mog5_pr_distances.py"
    spec = importlib.util.spec_from_file_location("compare_mog5_pr_distances", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_mahalanobis_cli_module():
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "bin" / "compare_mog5_pr_mahalanobis.py"
    if not path.is_file():
        pytest.skip(f"{path} is not present in this worktree")
    spec = importlib.util.spec_from_file_location("compare_mog5_pr_mahalanobis", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _toy_bundle() -> SharedDatasetBundle:
    theta_all = np.eye(3, dtype=np.float64)[np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)]
    x_all = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 3.0],
            [0.2, 0.1],
            [2.2, 0.1],
            [0.2, 3.1],
        ],
        dtype=np.float64,
    )
    train_idx = np.array([0, 1, 2], dtype=np.int64)
    val_idx = np.array([3, 4, 5], dtype=np.int64)
    return SharedDatasetBundle(
        meta={"dataset_family": "random_mog_categorical", "num_categories": 3, "x_dim": 2},
        theta_all=theta_all,
        x_all=x_all,
        train_idx=train_idx,
        validation_idx=val_idx,
        theta_train=theta_all[train_idx],
        x_train=x_all[train_idx],
        theta_validation=theta_all[val_idx],
        x_validation=x_all[val_idx],
    )


def test_metric_matrices_on_grouped_arrays() -> None:
    x = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    means = dc.class_means(x, labels, num_categories=3)
    expected_means = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
    np.testing.assert_allclose(means, expected_means)

    sq = dc.squared_euclidean_mean_distance_matrix(means)
    np.testing.assert_allclose(sq, np.array([[0.0, 2.0, 2.0], [2.0, 0.0, 4.0], [2.0, 4.0, 0.0]]))

    cosine = dc.cosine_distance_matrix(means)
    manual_cosine = 1.0 - (means @ means.T) / (np.linalg.norm(means, axis=1)[:, None] * np.linalg.norm(means, axis=1)[None, :])
    np.fill_diagonal(manual_cosine, 0.0)
    np.testing.assert_allclose(cosine, manual_cosine)

    corr = dc.correlation_distance_matrix(means)
    centered = means - means.mean(axis=1, keepdims=True)
    manual_corr = 1.0 - (centered @ centered.T) / (
        np.linalg.norm(centered, axis=1)[:, None] * np.linalg.norm(centered, axis=1)[None, :]
    )
    np.fill_diagonal(manual_corr, 0.0)
    np.testing.assert_allclose(corr, manual_corr)

    x2 = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [1.0, 3.0],
            [2.0, 1.0],
            [3.0, 2.0],
        ],
        dtype=np.float64,
    )
    labels2 = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    mah = dc.mahalanobis_sq_matrix(x2, labels2, num_categories=3, ridge=1e-6)
    means2 = dc.class_means(x2, labels2, num_categories=3)
    scatter = np.zeros((2, 2), dtype=np.float64)
    for cls in range(3):
        centered = x2[labels2 == cls] - means2[cls]
        scatter += centered.T @ centered
    pooled = scatter / float(x2.shape[0] - 3)
    cov = pooled + 1e-6 * np.eye(2, dtype=np.float64)
    expected = np.zeros((3, 3), dtype=np.float64)
    for i, j in dc.pair_indices(3):
        delta = means2[int(i)] - means2[int(j)]
        expected[int(i), int(j)] = expected[int(j), int(i)] = float(delta @ np.linalg.solve(cov, delta))
    np.testing.assert_allclose(mah, expected)

    mah_lw = dc.mahalanobis_sq_matrix_ledoit_wolf(x2, labels2, num_categories=3, ridge=1e-6)
    assert mah_lw.shape == (3, 3)
    np.testing.assert_allclose(mah_lw, mah_lw.T)
    np.testing.assert_allclose(np.diag(mah_lw), 0.0)
    assert np.all(np.isfinite(mah_lw))
    assert np.all(mah_lw >= 0.0)


def test_analytic_diagonal_gaussian_skl_matches_manual_two_component_calculation() -> None:
    means = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    variances = np.array([[1.0, 4.0], [2.0, 8.0]], dtype=np.float64)
    got = dc.analytic_diagonal_gaussian_skl_matrix(means, variances)

    diff2 = (means[0] - means[1]) ** 2
    kl_01 = 0.5 * np.sum(np.log(variances[1] / variances[0]) + (variances[0] + diff2) / variances[1] - 1.0)
    kl_10 = 0.5 * np.sum(np.log(variances[0] / variances[1]) + (variances[1] + diff2) / variances[0] - 1.0)
    expected = kl_01 + kl_10
    np.testing.assert_allclose(got, np.array([[0.0, expected], [expected, 0.0]]))


def test_gaussian_fid_matches_closed_form_for_diagonal_covariances() -> None:
    means = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    covariances = np.array([np.diag([1.0, 4.0]), np.diag([9.0, 1.0])], dtype=np.float64)

    got = dc.gaussian_fid_matrix(means, covariances)

    np.testing.assert_allclose(got, np.array([[0.0, 6.0], [6.0, 0.0]]), atol=1e-12)


def test_classical_fid_is_symmetric_and_finite() -> None:
    x = np.array(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [2.0, 1.0], [4.0, 1.0], [3.0, 2.0]],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

    got = dc.fid_matrix(x, labels, num_categories=2)

    assert got.shape == (2, 2)
    np.testing.assert_allclose(got, got.T)
    np.testing.assert_allclose(np.diag(got), 0.0)
    assert np.all(np.isfinite(got))
    assert got[0, 1] > 0.0


def test_ground_truth_symmetric_kl_only_skips_pr_encoding(monkeypatch, tmp_path: Path) -> None:
    calls: list[object] = []

    def fake_encode_with_pr_autoencoder(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("PR encoding should not run for symmetric_kl-only ground truth.")

    monkeypatch.setattr(dc, "encode_with_pr_autoencoder", fake_encode_with_pr_autoencoder)
    means = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    variances = np.ones_like(means)

    got = dc.pr_autoencoder_ground_truth_matrices(
        native_meta={
            "mog_component_means": means,
            "mog_component_variances": variances,
        },
        projected_meta={},
        device=torch.device("cpu"),
        cache_dir=tmp_path,
        samples_per_class=1,
        metrics=(dc.METRIC_SYMMETRIC_KL,),
    )

    assert calls == []
    assert tuple(got.keys()) == (dc.METRIC_SYMMETRIC_KL,)
    np.testing.assert_allclose(got[dc.METRIC_SYMMETRIC_KL], dc.analytic_diagonal_gaussian_skl_matrix(means, variances))


def test_flow_skl_to_metric_readout_scales_fixed_norm_rows() -> None:
    skl = np.array([[9.0, 16.0], [16.0, 9.0]], dtype=np.float64)

    np.testing.assert_allclose(dc.flow_skl_to_metric_readout(dc.METRIC_COSINE, skl, radius=2.0), [[0.0, 2.0], [2.0, 0.0]])
    np.testing.assert_allclose(
        dc.flow_skl_to_metric_readout(dc.METRIC_CORRELATION, skl, radius=2.0),
        [[0.0, 2.0], [2.0, 0.0]],
    )
    np.testing.assert_allclose(
        dc.flow_skl_to_metric_readout(dc.METRIC_SQUARED_EUCLIDEAN, skl, radius=2.0),
        [[0.0, 16.0], [16.0, 0.0]],
    )


def test_flow_fid_readout_restores_shared_x_normalization() -> None:
    zeros = np.zeros((2, 2), dtype=np.float64)
    result = FlowSKLResult(
        symmetric_kl_matrix=zeros,
        canonical_metric_matrix=zeros.copy(),
        canonical_metric_name="model_jeffreys_symmetric_kl",
        endpoint_gaussian_means=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        endpoint_gaussian_covariances=np.array([np.eye(2), np.eye(2)], dtype=np.float64),
    )

    got = dc.flow_result_to_metric_readout(
        dc.METRIC_FID,
        result,
        radius=1.0,
        normalize_meta={
            "flow_normalize_x": True,
            "flow_normalize_x_mean": np.array([10.0, -3.0]),
            "flow_normalize_x_std": np.array([2.0, 5.0]),
        },
    )

    np.testing.assert_allclose(got, [[0.0, 4.0], [4.0, 0.0]], atol=1e-12)


def test_flow_metric_mapping_and_mahalanobis_shared_assembly(monkeypatch, tmp_path: Path) -> None:
    bundle = _toy_bundle()
    calls: list[dict[str, object]] = []
    family_values = {
        "translation": 11.0,
        "translation_fixed_norm": 22.0,
        "translation_centered_fixed_norm": 33.0,
        "shared_affine": 55.0,
        "nonlinear": 44.0,
    }

    def fake_train_and_estimate_flow(**kwargs):
        calls.append(kwargs)
        theta_eval = np.asarray(kwargs["theta_eval"], dtype=np.float64)
        n = int(theta_eval.shape[0])
        family = str(kwargs["velocity_family"])
        mat = np.zeros((n, n), dtype=np.float64)
        value = family_values[family]
        mat += value
        np.fill_diagonal(mat, 0.0)
        canonical = mat + 1000.0
        np.fill_diagonal(canonical, 0.0)
        return FlowSKLResult(
            symmetric_kl_matrix=mat.copy(),
            canonical_metric_matrix=canonical.copy(),
            canonical_metric_name=family,
        )

    monkeypatch.setattr(dc, "train_and_estimate_flow", fake_train_and_estimate_flow)
    metrics = (
        dc.METRIC_SQUARED_EUCLIDEAN,
        dc.METRIC_COSINE,
        dc.METRIC_CORRELATION,
        dc.METRIC_MAHALANOBIS_SQ,
        dc.METRIC_SYMMETRIC_KL,
    )
    matrices, paths = dc.flow_metric_matrices(
        bundle=bundle,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        config=dc.FlowComparisonConfig(epochs=1, ode_method="rk4", radius=2.0),
        metrics=metrics,
    )

    seen_families = [str(c["velocity_family"]) for c in calls]
    assert seen_families[:3] == [
        "translation",
        "translation_fixed_norm",
        "translation_centered_fixed_norm",
    ]
    assert seen_families[3:] == ["shared_affine", "nonlinear"]
    shared_call = calls[3]
    assert all(c["config"].ode_method == "rk4" for c in calls)
    np.testing.assert_allclose(shared_call["theta_train"], bundle.theta_train)
    np.testing.assert_allclose(shared_call["x_train"], bundle.x_train)
    np.testing.assert_allclose(shared_call["theta_val"], bundle.theta_validation)
    np.testing.assert_allclose(shared_call["x_val"], bundle.x_validation)
    np.testing.assert_allclose(shared_call["theta_eval"], np.eye(3, dtype=np.float64))
    assert matrices[dc.METRIC_SQUARED_EUCLIDEAN][0, 1] == 11.0
    assert matrices[dc.METRIC_COSINE][0, 1] == 22.0 / 8.0
    assert matrices[dc.METRIC_CORRELATION][0, 1] == 33.0 / 8.0
    assert matrices[dc.METRIC_SYMMETRIC_KL][0, 1] == 44.0
    np.testing.assert_allclose(
        matrices[dc.METRIC_MAHALANOBIS_SQ],
        55.0 * (np.ones((3, 3), dtype=np.float64) - np.eye(3, dtype=np.float64)),
    )
    assert paths[dc.METRIC_SQUARED_EUCLIDEAN].is_file()
    with np.load(paths[dc.METRIC_CORRELATION], allow_pickle=True) as data:
        assert str(data["velocity_family"][0]) == "translation_centered_fixed_norm"
        assert "corr_soft_eps" not in data.files
        np.testing.assert_allclose(
            data["flow_matching_matrix"],
            matrices[dc.METRIC_CORRELATION],
        )
    assert paths[dc.METRIC_MAHALANOBIS_SQ].is_file()
    assert "mahalanobis_sq:0-1" not in paths


def test_flow_metric_variants_saves_nll_finetuned_estimator(monkeypatch, tmp_path: Path) -> None:
    bundle = _toy_bundle()

    def fake_train_and_estimate_flow_variants(**kwargs):
        del kwargs
        base = np.asarray([[0.0, 4.0, 4.0], [4.0, 0.0, 4.0], [4.0, 4.0, 0.0]], dtype=np.float64)
        fine = np.asarray([[0.0, 2.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 0.0]], dtype=np.float64)
        return (
            FlowSKLResult(base, base.copy(), "model_jeffreys_symmetric_kl"),
            FlowSKLResult(
                fine,
                fine.copy(),
                "model_jeffreys_symmetric_kl",
                train_metadata={
                    "likelihood_finetune_metadata": {
                        "train_nll_losses": np.asarray([3.0, 2.0]),
                        "val_nll_losses": np.asarray([3.2, 2.2]),
                        "val_monitor_nll_losses": np.asarray([3.2, 2.7]),
                        "best_val_nll": 2.7,
                        "best_epoch": 2,
                        "selected_val_nll": 2.7,
                        "selected_epoch": 2,
                    }
                },
            ),
        )

    monkeypatch.setattr(dc, "train_and_estimate_flow_variants", fake_train_and_estimate_flow_variants)
    variants = dc.flow_metric_variants(
        bundle=bundle,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        config=dc.FlowComparisonConfig(epochs=1, likelihood_finetune_epochs=2),
        metrics=(dc.METRIC_SYMMETRIC_KL,),
    )

    np.testing.assert_allclose(variants.flow_matching[dc.METRIC_SYMMETRIC_KL][0, 1], 4.0)
    np.testing.assert_allclose(variants.flow_matching_nll_finetuned[dc.METRIC_SYMMETRIC_KL][0, 1], 2.0)
    fine_path = variants.flow_nll_finetuned_npz_paths[dc.METRIC_SYMMETRIC_KL]
    with np.load(fine_path, allow_pickle=False) as data:
        assert str(data["estimator"][0]) == "flow_matching_nll_finetuned"
        np.testing.assert_allclose(data["nll_train_losses"], [3.0, 2.0])


def test_correlation_flow_routes_to_centered_fixed_norm(monkeypatch, tmp_path: Path) -> None:
    bundle = _toy_bundle()
    calls: list[dict[str, object]] = []

    def fake_train_and_estimate_flow(**kwargs):
        calls.append(kwargs)
        mat = np.asarray([[0.0, 8.0, 8.0], [8.0, 0.0, 8.0], [8.0, 8.0, 0.0]], dtype=np.float64)
        return FlowSKLResult(
            symmetric_kl_matrix=mat,
            canonical_metric_matrix=mat.copy(),
            canonical_metric_name="model_jeffreys_symmetric_kl",
        )

    monkeypatch.setattr(dc, "train_and_estimate_flow", fake_train_and_estimate_flow)
    matrices, paths = dc.flow_metric_matrices(
        bundle=bundle,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        config=dc.FlowComparisonConfig(epochs=1, radius=2.0),
        metrics=(dc.METRIC_CORRELATION,),
    )

    assert [c["velocity_family"] for c in calls] == ["translation_centered_fixed_norm"]
    np.testing.assert_allclose(
        matrices[dc.METRIC_CORRELATION],
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
    )
    with np.load(paths[dc.METRIC_CORRELATION], allow_pickle=True) as data:
        assert str(data["velocity_family"][0]) == "translation_centered_fixed_norm"
        assert "corr_soft_eps" not in data.files


def test_flow_metric_matrices_normalizes_x_train_only_and_persists_metadata(monkeypatch, tmp_path: Path) -> None:
    theta_all = np.eye(2, dtype=np.float64)[np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)]
    x_all = np.array(
        [
            [1.0, 5.0, 9.0],
            [3.0, 5.0, 9.0],
            [5.0, 5.0, 9.0],
            [7.0, 1005.0, 9.0],
            [9.0, 1005.0, 9.0],
            [11.0, 1005.0, 9.0],
        ],
        dtype=np.float64,
    )
    train_idx = np.array([0, 1, 2], dtype=np.int64)
    val_idx = np.array([3, 4, 5], dtype=np.int64)
    bundle = SharedDatasetBundle(
        meta={"dataset_family": "random_mog_categorical", "num_categories": 2, "x_dim": 3},
        theta_all=theta_all,
        x_all=x_all,
        train_idx=train_idx,
        validation_idx=val_idx,
        theta_train=theta_all[train_idx],
        x_train=x_all[train_idx],
        theta_validation=theta_all[val_idx],
        x_validation=x_all[val_idx],
    )
    calls: list[dict[str, object]] = []

    def fake_train_and_estimate_flow(**kwargs):
        calls.append(kwargs)
        mat = np.asarray([[0.0, 1.25], [1.25, 0.0]], dtype=np.float64)
        return FlowSKLResult(
            symmetric_kl_matrix=mat,
            canonical_metric_matrix=mat.copy(),
            canonical_metric_name="model_jeffreys_symmetric_kl",
            train_metadata={"train_losses": np.asarray([1.0], dtype=np.float64)},
        )

    monkeypatch.setattr(dc, "train_and_estimate_flow", fake_train_and_estimate_flow)
    config = dc.FlowComparisonConfig(epochs=1, normalize_x=True, normalize_x_eps=1e-6)

    _, paths = dc.flow_metric_matrices(
        bundle=bundle,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        config=config,
        metrics=(dc.METRIC_SYMMETRIC_KL,),
    )

    assert len(calls) == 1
    call = calls[0]
    expected_mean = np.asarray([3.0, 5.0, 9.0], dtype=np.float64)
    expected_std = np.asarray([np.std([1.0, 3.0, 5.0]), 1.0, 1.0], dtype=np.float64)
    np.testing.assert_allclose(call["x_train"], (bundle.x_train - expected_mean) / expected_std)
    np.testing.assert_allclose(call["x_val"], (bundle.x_validation - expected_mean) / expected_std)
    np.testing.assert_allclose(call["theta_train"], bundle.theta_train)
    np.testing.assert_allclose(call["theta_val"], bundle.theta_validation)
    np.testing.assert_allclose(call["theta_eval"], np.eye(2, dtype=np.float64))

    with np.load(paths[dc.METRIC_SYMMETRIC_KL], allow_pickle=True) as data:
        assert bool(data["flow_normalize_x"][0]) is True
        assert float(data["flow_normalize_x_eps"][0]) == pytest.approx(1e-6)
        np.testing.assert_allclose(data["flow_normalize_x_mean"], expected_mean)
        np.testing.assert_allclose(data["flow_normalize_x_std"], expected_std)


def test_train_and_estimate_flow_uses_model_jeffreys_readout(monkeypatch) -> None:
    bundle = _toy_bundle()
    build_calls: list[dict[str, object]] = []
    train_calls: list[dict[str, object]] = []
    estimate_calls: list[dict[str, object]] = []

    class DummyModel(torch.nn.Module):
        velocity_family = "translation"
        x_dim = 2

        def forward(self, x, theta, t):
            del theta, t
            return torch.zeros_like(x)

    def fake_build_flow_skl_model(**kwargs):
        build_calls.append(kwargs)
        return DummyModel()

    def fake_train_flow_skl_model(**kwargs):
        train_calls.append(kwargs)
        return {
            "train_losses": np.array([1.0], dtype=np.float64),
            "val_losses": np.array([2.0], dtype=np.float64),
            "val_monitor_losses": np.array([1.5], dtype=np.float64),
            "best_val_loss": 2.0,
            "best_epoch": 1,
            "stopped_epoch": 1,
            "stopped_early": False,
            "early_ema_alpha": kwargs["ema_alpha"],
        }

    def fake_estimate_model_symmetric_kl(**kwargs):
        estimate_calls.append(kwargs)
        theta_eval = np.asarray(kwargs["theta_all"], dtype=np.float64)
        mat = np.zeros((int(theta_eval.shape[0]), int(theta_eval.shape[0])), dtype=np.float64)
        return FlowSKLResult(
            symmetric_kl_matrix=mat,
            canonical_metric_matrix=mat.copy(),
            canonical_metric_name="model_jeffreys_symmetric_kl",
        )

    monkeypatch.setattr(dc, "build_flow_skl_model", fake_build_flow_skl_model)
    monkeypatch.setattr(dc, "train_flow_skl_model", fake_train_flow_skl_model)
    monkeypatch.setattr(dc, "estimate_model_symmetric_kl", fake_estimate_model_symmetric_kl)

    result = dc.train_and_estimate_flow(
        theta_train=bundle.theta_train,
        x_train=bundle.x_train,
        theta_val=bundle.theta_validation,
        x_val=bundle.x_validation,
        theta_eval=np.eye(3, dtype=np.float64),
        velocity_family="translation",
        device=torch.device("cpu"),
        seed=123,
        config=dc.FlowComparisonConfig(epochs=1, shared_affine_a_diag_jitter=2e-3, early_ema_alpha=0.2),
    )

    assert build_calls
    assert build_calls[0]["shared_affine_a_diag_jitter"] == 2e-3
    assert "corr_soft_eps" not in build_calls[0]
    assert train_calls
    assert train_calls[0]["ema_alpha"] == pytest.approx(0.2)
    assert train_calls[0]["best_checkpoint_metric"] == "flow_matching"
    assert train_calls[0]["likelihood_validation_every"] == 100
    assert train_calls[0]["fixed_validation_paths"] == 10
    assert estimate_calls
    call = estimate_calls[0]
    assert "theta_data" not in call
    assert "x_data" not in call
    assert "normalization" not in call
    np.testing.assert_allclose(call["theta_all"], np.eye(3, dtype=np.float64))
    assert call["mc_jeffreys_sample"] == 4096
    assert result.canonical_metric_name == "model_jeffreys_symmetric_kl"


def test_flow_comparison_config_normalize_x_defaults_are_opt_in() -> None:
    assert dc.FlowComparisonConfig().normalize_x is False
    assert dc.FlowComparisonConfig().normalize_x_eps == pytest.approx(1e-8)


def test_flow_comparison_config_default_t_eps_is_small_endpoint_clamp() -> None:
    config = dc.FlowComparisonConfig()
    assert config.t_eps == 0.0005
    assert config.early_ema_alpha == 0.05
    assert config.batch_size == 3000
    assert config.lr == pytest.approx(1e-4)
    assert config.hidden_dim == 128
    assert config.depth == 3
    assert config.fixed_validation is True
    assert config.fixed_validation_paths == 10
    assert config.likelihood_finetune_epochs == 500
    assert config.likelihood_finetune_batch_size == 2048
    assert config.likelihood_finetune_lr == pytest.approx(3e-5)
    assert config.likelihood_finetune_ode_steps == 32
    assert config.likelihood_finetune_patience == 150
    assert config.likelihood_finetune_checkpoint_selection == "best"


def test_save_flow_result_npz_persists_monitor_losses_and_ema_alpha(tmp_path: Path) -> None:
    result = FlowSKLResult(
        symmetric_kl_matrix=np.zeros((2, 2), dtype=np.float64),
        canonical_metric_matrix=np.zeros((2, 2), dtype=np.float64),
        canonical_metric_name="model_jeffreys_symmetric_kl",
        train_metadata={
            "train_losses": np.asarray([3.0, 2.0], dtype=np.float64),
            "val_losses": np.asarray([4.0, 1.0], dtype=np.float64),
            "val_monitor_losses": np.asarray([4.0, 3.85], dtype=np.float64),
            "best_val_loss": 3.85,
            "best_epoch": 2,
            "stopped_epoch": 2,
            "stopped_early": False,
            "early_ema_alpha": 0.05,
            "flow_normalize_x": True,
            "flow_normalize_x_mean": np.asarray([1.0, 2.0], dtype=np.float64),
            "flow_normalize_x_std": np.asarray([3.0, 4.0], dtype=np.float64),
            "flow_normalize_x_eps": 1e-6,
        },
    )
    path = dc.save_flow_result_npz(
        tmp_path / "flow.npz",
        result=result,
        metric=dc.METRIC_SYMMETRIC_KL,
        theta_eval=np.eye(2, dtype=np.float64),
        velocity_family="nonlinear",
    )

    with np.load(path, allow_pickle=True) as data:
        np.testing.assert_allclose(data["val_monitor_losses"], [4.0, 3.85])
        assert float(data["best_val_loss"][0]) == pytest.approx(3.85)
        assert float(data["early_ema_alpha"][0]) == pytest.approx(0.05)
        assert bool(data["flow_normalize_x"][0]) is True
        assert float(data["flow_normalize_x_eps"][0]) == pytest.approx(1e-6)
        np.testing.assert_allclose(data["flow_normalize_x_mean"], [1.0, 2.0])
        np.testing.assert_allclose(data["flow_normalize_x_std"], [3.0, 4.0])


def test_save_flow_result_npz_persists_flow_readout(tmp_path: Path) -> None:
    result = FlowSKLResult(
        symmetric_kl_matrix=np.asarray([[0.0, 8.0], [8.0, 0.0]], dtype=np.float64),
        canonical_metric_matrix=np.asarray([[0.0, 8.0], [8.0, 0.0]], dtype=np.float64),
        canonical_metric_name="model_jeffreys_symmetric_kl",
    )
    path = dc.save_flow_result_npz(
        tmp_path / "flow.npz",
        result=result,
        metric=dc.METRIC_CORRELATION,
        theta_eval=np.eye(2, dtype=np.float64),
        velocity_family="translation_centered_fixed_norm",
        flow_metric_matrix=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
    )

    with np.load(path, allow_pickle=True) as data:
        np.testing.assert_allclose(data["symmetric_kl_matrix"], [[0.0, 8.0], [8.0, 0.0]])
        np.testing.assert_allclose(data["flow_matching_matrix"], [[0.0, 1.0], [1.0, 0.0]])
        assert str(data["velocity_family"][0]) == "translation_centered_fixed_norm"
        assert "corr_soft_eps" not in data.files


def test_shared_affine_normalization_preserves_gaussian_symmetric_kl() -> None:
    def gaussian_kl(mean_p, cov_p, mean_q, cov_q):
        d = int(mean_p.shape[0])
        diff = mean_q - mean_p
        sign_p, logdet_p = np.linalg.slogdet(cov_p)
        sign_q, logdet_q = np.linalg.slogdet(cov_q)
        assert sign_p > 0 and sign_q > 0
        solved_cov = np.linalg.solve(cov_q, cov_p)
        solved_diff = np.linalg.solve(cov_q, diff)
        return 0.5 * (np.trace(solved_cov) + float(diff @ solved_diff) - d + logdet_q - logdet_p)

    mean_0 = np.asarray([0.0, 1.0], dtype=np.float64)
    mean_1 = np.asarray([2.0, -1.0], dtype=np.float64)
    cov_0 = np.asarray([[2.0, 0.3], [0.3, 0.8]], dtype=np.float64)
    cov_1 = np.asarray([[1.5, -0.2], [-0.2, 1.2]], dtype=np.float64)
    original = gaussian_kl(mean_0, cov_0, mean_1, cov_1) + gaussian_kl(mean_1, cov_1, mean_0, cov_0)

    shared_mean = np.asarray([10.0, -3.0], dtype=np.float64)
    shared_std = np.asarray([4.0, 0.5], dtype=np.float64)
    a = np.diag(1.0 / shared_std)
    norm_mean_0 = (mean_0 - shared_mean) / shared_std
    norm_mean_1 = (mean_1 - shared_mean) / shared_std
    norm_cov_0 = a @ cov_0 @ a.T
    norm_cov_1 = a @ cov_1 @ a.T
    normalized = gaussian_kl(norm_mean_0, norm_cov_0, norm_mean_1, norm_cov_1) + gaussian_kl(
        norm_mean_1,
        norm_cov_1,
        norm_mean_0,
        norm_cov_0,
    )

    np.testing.assert_allclose(normalized, original, rtol=1e-12, atol=1e-12)


def test_assemble_rows_with_mocked_flow_results() -> None:
    labels = ("category_0", "category_1", "category_2")
    classical = {m: np.ones((3, 3), dtype=np.float64) for m in dc.METRIC_NAMES}
    flow = {m: 2.0 * np.ones((3, 3), dtype=np.float64) for m in dc.METRIC_NAMES}
    flow_finetuned = {m: 2.5 * np.ones((3, 3), dtype=np.float64) for m in dc.METRIC_NAMES}
    gt = {m: 3.0 * np.ones((3, 3), dtype=np.float64) for m in dc.METRIC_NAMES}
    for mats in (classical, flow, flow_finetuned, gt):
        for mat in mats.values():
            np.fill_diagonal(mat, 0.0)

    result = dc.assemble_comparison_result(
        metrics=dc.METRIC_NAMES,
        condition_names=labels,
        classical=classical,
        flow_matching=flow,
        flow_matching_nll_finetuned=flow_finetuned,
        ground_truth=gt,
    )

    assert result.pair_indices.shape == (3, 2)
    assert len(result.rows) == len(dc.METRIC_NAMES) * 3
    first = result.rows[0]
    assert first["metric"] == dc.METRIC_SQUARED_EUCLIDEAN
    assert first["condition_i"] == "category_0"
    assert first["condition_j"] == "category_1"
    assert first["classical"] == 1.0
    assert first["flow_matching"] == 2.0
    assert first["flow_matching_nll_finetuned"] == 2.5
    assert first["ground_truth"] == 3.0
    assert first["abs_error_classical"] == 2.0
    assert first["abs_error_flow"] == 1.0
    assert first["abs_error_flow_nll_finetuned"] == 0.5


def test_cli_default_path_resolution_without_running_training() -> None:
    mod = _load_cli_module()
    repo_root = Path(__file__).resolve().parent.parent
    args = mod.build_parser().parse_args([])

    assert args.n_total == 1_000
    assert args.native_x_dim == 3
    assert args.pr_dim is None
    assert args.seed == 19
    assert args.device == "cuda:0"
    assert args.dataset_train_frac == pytest.approx(0.8)
    assert args.dataset_obs_noise_scale == pytest.approx(1.0)
    assert args.dataset_cov_theta_amp_scale == pytest.approx(1.0)
    assert args.dataset_mog_mean_min_dist is None
    assert args.flow_likelihood_finetune_epochs == 500
    assert args.batch_size == 3000
    assert args.lr == pytest.approx(1e-4)
    assert args.hidden_dim == 128
    assert args.depth == 3
    assert args.fixed_validation is True
    assert args.fixed_validation_paths == 10
    assert args.flow_likelihood_finetune_batch_size == 3000
    assert args.flow_likelihood_finetune_lr == pytest.approx(3e-5)
    assert args.flow_likelihood_finetune_ode_steps == 32
    assert args.flow_likelihood_finetune_patience == 150
    assert args.flow_likelihood_finetune_checkpoint_selection == "best"
    assert args.metric == "all"
    assert mod.resolve_metric_names(args) == dc.METRIC_NAMES
    assert args.gt_samples_per_class == 100_000
    assert args.mc_jeffreys_sample == 4096
    assert args.radius == 1.0
    assert args.ode_method == "midpoint"
    assert args.t_eps == 0.0005
    assert args.early_ema_alpha == 0.05
    assert args.flow_normalize_x is False
    assert args.flow_normalize_x_eps == pytest.approx(1e-8)
    assert mod._flow_config_from_args(args).mc_jeffreys_sample == 4096
    assert mod._flow_config_from_args(args).radius == 1.0
    assert mod._flow_config_from_args(args).t_eps == 0.0005
    assert mod._flow_config_from_args(args).early_ema_alpha == 0.05
    assert mod._flow_config_from_args(args).fixed_validation_paths == 10
    assert mod._flow_config_from_args(args).normalize_x is False
    assert mod._flow_config_from_args(args).normalize_x_eps == pytest.approx(1e-8)
    assert mod.resolve_dataset_dir(args) == repo_root / "data" / "mog_5native_xdim3_n1000"
    assert mod.resolve_output_dir(args) == repo_root / "data" / "mog_5native_xdim3_n1000" / "distance_comparison_flow_skl"

    native_args = mod.build_parser().parse_args(["--native-x-dim", "2", "--pr-dim", "none"])
    assert native_args.pr_dim is None
    assert mod.resolve_dataset_dir(native_args) == repo_root / "data" / "mog_5native_n1000"
    assert mod.resolve_output_dir(native_args) == repo_root / "data" / "mog_5native_n1000" / "distance_comparison_flow_skl"

    native3_args = mod.build_parser().parse_args(["--native-x-dim", "3", "--pr-dim", "none"])
    assert mod.resolve_dataset_dir(native3_args) == repo_root / "data" / "mog_5native_xdim3_n1000"
    assert mod.resolve_output_dir(native3_args) == repo_root / "data" / "mog_5native_xdim3_n1000" / "distance_comparison_flow_skl"

    native3_pr_args = mod.build_parser().parse_args(["--native-x-dim", "3", "--pr-dim", "5"])
    assert mod.resolve_dataset_dir(native3_pr_args) == repo_root / "data" / "mog_5native_xdim3_pr5_n1000"
    assert mod.resolve_output_dir(native3_pr_args) == repo_root / "data" / "mog_5native_xdim3_pr5_n1000" / "distance_comparison_flow_skl"

    invalid_args = mod.build_parser().parse_args(["--native-x-dim", "3", "--pr-dim", "2"])
    with pytest.raises(ValueError, match="--pr-dim must be >= native x_dim=3"):
        mod.validate_args(invalid_args)


def test_cli_early_ema_alpha_override_propagates_to_flow_config() -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(["--early-ema-alpha", "0.2"])
    assert args.early_ema_alpha == pytest.approx(0.2)
    assert mod._flow_config_from_args(args).early_ema_alpha == pytest.approx(0.2)


def test_cli_short_lr_schedule_horizon_propagates_to_flow_config() -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(
        ["--lr-schedule", "cosine", "--min-lr", "1e-6", "--lr-schedule-epochs", "5000"]
    )

    mod.validate_args(args)
    config = mod._flow_config_from_args(args)

    assert config.lr_schedule == "cosine"
    assert config.min_lr == pytest.approx(1e-6)
    assert config.lr_schedule_epochs == 5000


def test_cli_rejects_nonpositive_lr_schedule_horizon() -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(["--lr-schedule-epochs", "0"])

    with pytest.raises(ValueError, match="lr-schedule-epochs"):
        mod.validate_args(args)


def test_cli_accepts_comma_separated_metric_subset() -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(["--metric", "correlation,cosine,correlation"])

    mod.validate_args(args)

    assert mod.resolve_metric_names(args) == ("correlation", "cosine")


def test_cli_rejects_unknown_metric_in_subset() -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(["--metric", "cosine,unknown"])

    with pytest.raises(ValueError, match="Unknown --metric"):
        mod.validate_args(args)


def test_cli_fixed_validation_paths_override_propagates_to_flow_config() -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(["--fixed-validation", "--fixed-validation-paths", "12"])

    mod.validate_args(args)
    config = mod._flow_config_from_args(args)

    assert config.fixed_validation is True
    assert config.fixed_validation_paths == 12


def test_cli_can_disable_fixed_validation_default() -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(["--no-fixed-validation"])

    mod.validate_args(args)

    assert args.fixed_validation is False
    assert mod._flow_config_from_args(args).fixed_validation is False


def test_cli_rejects_nonpositive_fixed_validation_paths() -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(["--fixed-validation-paths", "0"])

    with pytest.raises(ValueError, match="fixed-validation-paths"):
        mod.validate_args(args)


def test_cli_flow_normalize_x_override_propagates_to_flow_config() -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(["--flow-normalize-x", "--flow-normalize-x-eps", "1e-5"])
    assert args.flow_normalize_x is True
    assert args.flow_normalize_x_eps == pytest.approx(1e-5)
    assert mod._flow_config_from_args(args).normalize_x is True
    assert mod._flow_config_from_args(args).normalize_x_eps == pytest.approx(1e-5)


def test_cli_run_passes_selected_metric_only(monkeypatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    metric = dc.METRIC_COSINE
    calls: dict[str, object] = {}
    n_total = 1000
    k = 5

    theta_all = np.eye(k, dtype=np.float64)[np.arange(n_total, dtype=np.int64) % k]
    projected_bundle = SharedDatasetBundle(
        meta={
            "dataset_family": "random_mog_categorical",
            "num_categories": k,
            "x_dim": 5,
            "pr_autoencoder_embedded": True,
            "pr_autoencoder_z_dim": 2,
        },
        theta_all=theta_all,
        x_all=np.zeros((n_total, 5), dtype=np.float64),
        train_idx=np.arange(10, dtype=np.int64),
        validation_idx=np.arange(10, 20, dtype=np.int64),
        theta_train=theta_all[:10],
        x_train=np.zeros((10, 5), dtype=np.float64),
        theta_validation=theta_all[10:20],
        x_validation=np.zeros((10, 5), dtype=np.float64),
    )
    native_bundle = SharedDatasetBundle(
        meta={
            "dataset_family": "random_mog_categorical",
            "num_categories": k,
            "x_dim": 2,
            "mog_component_means": np.zeros((k, 2), dtype=np.float64),
            "mog_component_variances": np.ones((k, 2), dtype=np.float64),
        },
        theta_all=theta_all,
        x_all=np.zeros((n_total, 2), dtype=np.float64),
        train_idx=np.arange(10, dtype=np.int64),
        validation_idx=np.arange(10, 20, dtype=np.int64),
        theta_train=theta_all[:10],
        x_train=np.zeros((10, 2), dtype=np.float64),
        theta_validation=theta_all[10:20],
        x_validation=np.zeros((10, 2), dtype=np.float64),
    )

    def fake_ensure_dataset(args, dataset_dir):
        calls["ensure_dataset"] = (args, dataset_dir)
        return tmp_path / "native.npz", tmp_path / "projected.npz"

    def fake_load_shared_dataset_npz(path):
        return native_bundle if Path(path).name == "native.npz" else projected_bundle

    def fake_classical_metric_matrices(*args, **kwargs):
        calls["classical_metrics"] = tuple(kwargs["metrics"])
        return {metric: np.ones((k, k), dtype=np.float64)}

    def fake_ground_truth(**kwargs):
        calls["ground_truth_metrics"] = tuple(kwargs["metrics"])
        return {metric: 3.0 * np.ones((k, k), dtype=np.float64)}

    def fake_flow_metric_matrices(**kwargs):
        calls["flow_metrics"] = tuple(kwargs["metrics"])
        calls["flow_output_dir"] = Path(kwargs["output_dir"])
        return {metric: 2.0 * np.ones((k, k), dtype=np.float64)}, {metric: tmp_path / "flow.npz"}

    def fake_assemble_comparison_result(**kwargs):
        calls["assemble_metrics"] = tuple(kwargs["metrics"])
        calls["assemble_ground_truth_keys"] = tuple(kwargs["ground_truth"].keys())
        return dc.assemble_comparison_result(**kwargs)

    def fake_write_results_npz(path, result):
        calls["results_metrics"] = tuple(result.metrics)
        return Path(path)

    def fake_write_pairs_csv(path, rows):
        calls["rows"] = rows
        return Path(path)

    def fake_write_summary_json(path, *, result, extra):
        calls["summary_extra"] = dict(extra)
        return Path(path)

    monkeypatch.setattr(mod, "require_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(mod, "ensure_dataset", fake_ensure_dataset)
    monkeypatch.setattr(mod, "load_shared_dataset_npz", fake_load_shared_dataset_npz)
    monkeypatch.setattr(mod, "classical_metric_matrices", fake_classical_metric_matrices)
    monkeypatch.setattr(mod, "pr_autoencoder_ground_truth_matrices", fake_ground_truth)
    monkeypatch.setattr(mod, "flow_metric_matrices", fake_flow_metric_matrices)
    monkeypatch.setattr(mod, "assemble_comparison_result", fake_assemble_comparison_result)
    monkeypatch.setattr(mod, "write_results_npz", fake_write_results_npz)
    monkeypatch.setattr(mod, "write_pairs_csv", fake_write_pairs_csv)
    monkeypatch.setattr(mod, "write_summary_json", fake_write_summary_json)

    args = mod.build_parser().parse_args(
        [
            "--native-x-dim",
            "2",
            "--pr-dim",
            "5",
            "--metric",
            metric,
            "--flow-likelihood-finetune-epochs",
            "0",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    paths = mod.run(args)

    assert mod.resolve_metric_names(args) == (metric,)
    assert calls["classical_metrics"] == (metric,)
    assert calls["ground_truth_metrics"] == (metric,)
    assert calls["flow_metrics"] == (metric,)
    assert calls["assemble_metrics"] == (metric,)
    assert calls["assemble_ground_truth_keys"] == (metric,)
    assert calls["results_metrics"] == (metric,)
    assert calls["flow_output_dir"] == tmp_path / "out" / "flow"
    assert calls["summary_extra"]["metric"] == metric
    assert calls["summary_extra"]["metrics"] == [metric]
    assert calls["summary_extra"]["native_x_dim"] == 2
    assert paths["results_npz"] == tmp_path / "out" / "mog5_pr_distance_comparison_results.npz"


def test_cli_native_mode_uses_native_bundle_and_ground_truth(monkeypatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    metric = dc.METRIC_SQUARED_EUCLIDEAN
    calls: dict[str, object] = {}
    n_total = 1000
    k = 5

    theta_all = np.eye(k, dtype=np.float64)[np.arange(n_total, dtype=np.int64) % k]
    native_x = np.arange(n_total * 2, dtype=np.float64).reshape(n_total, 2)
    native_bundle = SharedDatasetBundle(
        meta={
            "dataset_family": "random_mog_categorical",
            "num_categories": k,
            "x_dim": 2,
            "mog_component_means": np.zeros((k, 2), dtype=np.float64),
            "mog_component_variances": np.ones((k, 2), dtype=np.float64),
        },
        theta_all=theta_all,
        x_all=native_x,
        train_idx=np.arange(10, dtype=np.int64),
        validation_idx=np.arange(10, 20, dtype=np.int64),
        theta_train=theta_all[:10],
        x_train=native_x[:10],
        theta_validation=theta_all[10:20],
        x_validation=native_x[10:20],
    )

    def fake_ensure_dataset(args, dataset_dir):
        calls["ensure_pr_dim"] = args.pr_dim
        calls["dataset_dir"] = Path(dataset_dir)
        return tmp_path / "native.npz", None

    def fake_classical_metric_matrices(x, labels, **kwargs):
        calls["classical_x"] = np.asarray(x)
        calls["classical_metrics"] = tuple(kwargs["metrics"])
        return {metric: np.ones((k, k), dtype=np.float64)}

    def fake_native_ground_truth(**kwargs):
        calls["native_ground_truth_metrics"] = tuple(kwargs["metrics"])
        return {metric: 3.0 * np.ones((k, k), dtype=np.float64)}

    def fake_pr_ground_truth(**kwargs):
        raise AssertionError("native mode must not use PR-autoencoder ground truth")

    def fake_flow_metric_matrices(**kwargs):
        calls["flow_bundle_x"] = np.asarray(kwargs["bundle"].x_all)
        return {metric: 2.0 * np.ones((k, k), dtype=np.float64)}, {metric: tmp_path / "flow.npz"}

    monkeypatch.setattr(mod, "require_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(mod, "ensure_dataset", fake_ensure_dataset)
    monkeypatch.setattr(mod, "load_shared_dataset_npz", lambda path: native_bundle)
    monkeypatch.setattr(mod, "classical_metric_matrices", fake_classical_metric_matrices)
    monkeypatch.setattr(mod, "native_mog_ground_truth_matrices", fake_native_ground_truth)
    monkeypatch.setattr(mod, "pr_autoencoder_ground_truth_matrices", fake_pr_ground_truth)
    monkeypatch.setattr(mod, "flow_metric_matrices", fake_flow_metric_matrices)
    monkeypatch.setattr(mod, "write_results_npz", lambda path, result: Path(path))
    monkeypatch.setattr(mod, "write_pairs_csv", lambda path, rows: Path(path))

    def fake_write_summary_json(path, *, result, extra):
        calls["summary_extra"] = dict(extra)
        return Path(path)

    monkeypatch.setattr(mod, "write_summary_json", fake_write_summary_json)

    args = mod.build_parser().parse_args(
        [
            "--native-x-dim",
            "2",
            "--pr-dim",
            "none",
            "--metric",
            metric,
            "--flow-likelihood-finetune-epochs",
            "0",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    mod.run(args)

    assert calls["ensure_pr_dim"] is None
    np.testing.assert_allclose(calls["classical_x"], native_x)
    np.testing.assert_allclose(calls["flow_bundle_x"], native_x)
    assert calls["classical_metrics"] == (metric,)
    assert calls["native_ground_truth_metrics"] == (metric,)
    assert calls["summary_extra"]["pr_projected"] is False
    assert calls["summary_extra"]["native_x_dim"] == 2
    assert calls["summary_extra"]["pr_dim"] is None
    assert calls["summary_extra"]["projected_npz"] is None
    assert calls["summary_extra"]["work_npz"] == str(tmp_path / "native.npz")


def test_mahalanobis_cli_defaults_match_full_cli_without_running_training() -> None:
    full = _load_cli_module()
    mod = _load_mahalanobis_cli_module()
    repo_root = Path(__file__).resolve().parent.parent

    full_args = full.build_parser().parse_args([])
    args = mod.build_parser().parse_args([])

    assert args.n_total == full_args.n_total == 1_000
    assert args.pr_dim == full_args.pr_dim is None
    assert args.seed == full_args.seed == 19
    assert args.device == full_args.device == "cuda:0"
    assert args.gt_samples_per_class == full_args.gt_samples_per_class == 100_000
    assert args.mc_jeffreys_sample == full_args.mc_jeffreys_sample == 4096
    assert args.radius == full_args.radius == 1.0
    assert args.ode_method == full_args.ode_method == "midpoint"
    assert args.t_eps == full_args.t_eps == 0.0005
    assert mod._flow_config_from_args(args).mc_jeffreys_sample == 4096
    assert mod._flow_config_from_args(args).radius == 1.0
    assert mod._flow_config_from_args(args).t_eps == 0.0005
    assert mod.resolve_dataset_dir(args) == repo_root / "data" / "mog_5native_xdim3_n1000"
    assert mod.resolve_output_dir(args) == repo_root / "data" / "mog_5native_xdim3_n1000" / "mahalanobis_comparison_flow_skl"

    compatible = mod.build_parser().parse_args(
        [
            "--skl-folds",
            "3",
            "--skl-logistic-c",
            "0.5",
            "--radius",
            "2.0",
        ]
    )
    assert compatible.skl_folds == 3
    assert compatible.skl_logistic_c == 0.5
    assert compatible.radius == 2.0


def test_mahalanobis_cli_run_filters_to_mahalanobis_only(monkeypatch, tmp_path: Path) -> None:
    mod = _load_mahalanobis_cli_module()
    metric = dc.METRIC_MAHALANOBIS_SQ
    other_metric = dc.METRIC_SQUARED_EUCLIDEAN
    calls: dict[str, object] = {}
    n_total = 1000
    k = 5

    theta_all = np.eye(k, dtype=np.float64)[np.arange(n_total, dtype=np.int64) % k]
    x_all = np.zeros((n_total, 5), dtype=np.float64)
    projected_bundle = SharedDatasetBundle(
        meta={
            "dataset_family": "random_mog_categorical",
            "num_categories": k,
            "x_dim": 5,
            "pr_autoencoder_embedded": True,
        },
        theta_all=theta_all,
        x_all=x_all,
        train_idx=np.arange(10, dtype=np.int64),
        validation_idx=np.arange(10, 20, dtype=np.int64),
        theta_train=theta_all[:10],
        x_train=x_all[:10],
        theta_validation=theta_all[10:20],
        x_validation=x_all[10:20],
    )
    native_bundle = SharedDatasetBundle(
        meta={
            "dataset_family": "random_mog_categorical",
            "num_categories": k,
            "x_dim": 2,
            "mog_component_means": np.zeros((k, 2), dtype=np.float64),
            "mog_component_variances": np.ones((k, 2), dtype=np.float64),
        },
        theta_all=theta_all,
        x_all=np.zeros((n_total, 2), dtype=np.float64),
        train_idx=np.arange(10, dtype=np.int64),
        validation_idx=np.arange(10, 20, dtype=np.int64),
        theta_train=theta_all[:10],
        x_train=np.zeros((10, 2), dtype=np.float64),
        theta_validation=theta_all[10:20],
        x_validation=np.zeros((10, 2), dtype=np.float64),
    )

    def fake_ensure_dataset(args, dataset_dir):
        calls["ensure_dataset"] = (args, dataset_dir)
        return tmp_path / "native.npz", tmp_path / "projected.npz"

    def fake_load_shared_dataset_npz(path):
        return native_bundle if Path(path).name == "native.npz" else projected_bundle

    def fake_classical_metric_matrices(*args, **kwargs):
        calls["classical_metrics"] = tuple(kwargs["metrics"])
        return {metric: np.ones((k, k), dtype=np.float64)}

    def fake_ground_truth(**kwargs):
        calls["ground_truth_kwargs"] = kwargs
        return {
            metric: 3.0 * np.ones((k, k), dtype=np.float64),
            other_metric: 9.0 * np.ones((k, k), dtype=np.float64),
        }

    def fake_flow_metric_matrices(**kwargs):
        calls["flow_metrics"] = tuple(kwargs["metrics"])
        calls["flow_output_dir"] = Path(kwargs["output_dir"])
        return {metric: 2.0 * np.ones((k, k), dtype=np.float64)}, {metric: tmp_path / "flow.npz"}

    def fake_assemble_comparison_result(**kwargs):
        calls["assemble_metrics"] = tuple(kwargs["metrics"])
        calls["assemble_ground_truth_keys"] = tuple(kwargs["ground_truth"].keys())
        calls["assemble_flow_npz_paths"] = dict(kwargs["flow_npz_paths"])
        return dc.assemble_comparison_result(**kwargs)

    def fake_write_results_npz(path, result):
        calls["results_npz"] = Path(path)
        return Path(path)

    def fake_write_pairs_csv(path, rows):
        calls["pairs_csv"] = Path(path)
        calls["rows"] = rows
        return Path(path)

    def fake_write_summary_json(path, *, result, extra):
        calls["summary_json"] = Path(path)
        calls["summary_extra"] = dict(extra)
        return Path(path)

    monkeypatch.setattr(mod, "require_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(mod, "ensure_dataset", fake_ensure_dataset)
    monkeypatch.setattr(mod, "load_shared_dataset_npz", fake_load_shared_dataset_npz)
    monkeypatch.setattr(mod, "classical_metric_matrices", fake_classical_metric_matrices)
    monkeypatch.setattr(mod, "pr_autoencoder_ground_truth_matrices", fake_ground_truth)
    monkeypatch.setattr(mod, "flow_metric_matrices", fake_flow_metric_matrices)
    monkeypatch.setattr(mod, "assemble_comparison_result", fake_assemble_comparison_result)
    monkeypatch.setattr(mod, "write_results_npz", fake_write_results_npz)
    monkeypatch.setattr(mod, "write_pairs_csv", fake_write_pairs_csv)
    monkeypatch.setattr(mod, "write_summary_json", fake_write_summary_json)

    args = mod.build_parser().parse_args(["--output-dir", str(tmp_path / "out")])
    paths = mod.run(args)

    assert calls["classical_metrics"] == (metric,)
    assert calls["flow_metrics"] == (metric,)
    assert calls["assemble_metrics"] == (metric,)
    assert calls["assemble_ground_truth_keys"] == (metric,)
    assert calls["assemble_flow_npz_paths"] == {metric: tmp_path / "flow.npz"}
    assert calls["flow_output_dir"] == tmp_path / "out" / "flow"
    assert calls["results_npz"].name == "mog5_pr_mahalanobis_comparison_results.npz"
    assert calls["pairs_csv"].name == "mog5_pr_mahalanobis_comparison_pairs.csv"
    assert calls["summary_json"].name == "mog5_pr_mahalanobis_comparison_summary.json"
    assert calls["summary_extra"]["script"] == "bin/compare_mog5_pr_mahalanobis.py"
    assert calls["summary_extra"]["metrics"] == [metric]
    assert paths["output_dir"] == tmp_path / "out"
    assert paths["results_npz"] == tmp_path / "out" / "mog5_pr_mahalanobis_comparison_results.npz"
