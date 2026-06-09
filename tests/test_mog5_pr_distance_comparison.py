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


def test_analytic_diagonal_gaussian_skl_matches_manual_two_component_calculation() -> None:
    means = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    variances = np.array([[1.0, 4.0], [2.0, 8.0]], dtype=np.float64)
    got = dc.analytic_diagonal_gaussian_skl_matrix(means, variances)

    diff2 = (means[0] - means[1]) ** 2
    kl_01 = 0.5 * np.sum(np.log(variances[1] / variances[0]) + (variances[0] + diff2) / variances[1] - 1.0)
    kl_10 = 0.5 * np.sum(np.log(variances[0] / variances[1]) + (variances[1] + diff2) / variances[0] - 1.0)
    expected = kl_01 + kl_10
    np.testing.assert_allclose(got, np.array([[0.0, expected], [expected, 0.0]]))


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
    assert paths[dc.METRIC_MAHALANOBIS_SQ].is_file()
    assert "mahalanobis_sq:0-1" not in paths


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
    assert train_calls
    assert train_calls[0]["ema_alpha"] == pytest.approx(0.2)
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
    assert dc.FlowComparisonConfig().t_eps == 0.0005
    assert dc.FlowComparisonConfig().early_ema_alpha == 0.05


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
    gt = {m: 3.0 * np.ones((3, 3), dtype=np.float64) for m in dc.METRIC_NAMES}
    for mats in (classical, flow, gt):
        for mat in mats.values():
            np.fill_diagonal(mat, 0.0)

    result = dc.assemble_comparison_result(
        metrics=dc.METRIC_NAMES,
        condition_names=labels,
        classical=classical,
        flow_matching=flow,
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
    assert first["ground_truth"] == 3.0
    assert first["abs_error_classical"] == 2.0
    assert first["abs_error_flow"] == 1.0


def test_cli_default_path_resolution_without_running_training() -> None:
    mod = _load_cli_module()
    repo_root = Path(__file__).resolve().parent.parent
    args = mod.build_parser().parse_args([])

    assert args.n_total == 1_000
    assert args.pr_dim == 5
    assert args.seed == 7
    assert args.device == "cuda"
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
    assert mod._flow_config_from_args(args).normalize_x is False
    assert mod._flow_config_from_args(args).normalize_x_eps == pytest.approx(1e-8)
    assert mod.resolve_dataset_dir(args) == repo_root / "data" / "mog_5pr5_n1000"
    assert mod.resolve_output_dir(args) == repo_root / "data" / "mog_5pr5_n1000" / "distance_comparison_flow_skl"


def test_cli_early_ema_alpha_override_propagates_to_flow_config() -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(["--early-ema-alpha", "0.2"])
    assert args.early_ema_alpha == pytest.approx(0.2)
    assert mod._flow_config_from_args(args).early_ema_alpha == pytest.approx(0.2)


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

    args = mod.build_parser().parse_args(["--metric", metric, "--output-dir", str(tmp_path / "out")])
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
    assert paths["results_npz"] == tmp_path / "out" / "mog5_pr_distance_comparison_results.npz"


def test_mahalanobis_cli_defaults_match_full_cli_without_running_training() -> None:
    full = _load_cli_module()
    mod = _load_mahalanobis_cli_module()
    repo_root = Path(__file__).resolve().parent.parent

    full_args = full.build_parser().parse_args([])
    args = mod.build_parser().parse_args([])

    assert args.n_total == full_args.n_total == 1_000
    assert args.pr_dim == full_args.pr_dim == 5
    assert args.seed == full_args.seed == 7
    assert args.device == full_args.device == "cuda"
    assert args.gt_samples_per_class == full_args.gt_samples_per_class == 100_000
    assert args.mc_jeffreys_sample == full_args.mc_jeffreys_sample == 4096
    assert args.radius == full_args.radius == 1.0
    assert args.ode_method == full_args.ode_method == "midpoint"
    assert args.t_eps == full_args.t_eps == 0.0005
    assert mod._flow_config_from_args(args).mc_jeffreys_sample == 4096
    assert mod._flow_config_from_args(args).radius == 1.0
    assert mod._flow_config_from_args(args).t_eps == 0.0005
    assert mod.resolve_dataset_dir(args) == repo_root / "data" / "mog_5pr5_n1000"
    assert mod.resolve_output_dir(args) == repo_root / "data" / "mog_5pr5_n1000" / "mahalanobis_comparison_flow_skl"

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
