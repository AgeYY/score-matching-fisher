from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
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


def test_train_and_estimate_flow_uses_model_jeffreys_readout(monkeypatch) -> None:
    bundle = _toy_bundle()
    estimate_calls: list[dict[str, object]] = []

    class DummyModel(torch.nn.Module):
        velocity_family = "translation"
        x_dim = 2

        def forward(self, x, theta, t):
            del theta, t
            return torch.zeros_like(x)

    def fake_build_flow_skl_model(**kwargs):
        return DummyModel()

    def fake_train_flow_skl_model(**kwargs):
        return {
            "train_losses": np.array([1.0], dtype=np.float64),
            "val_losses": np.array([2.0], dtype=np.float64),
            "best_val_loss": 2.0,
            "best_epoch": 1,
            "stopped_epoch": 1,
            "stopped_early": False,
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
        config=dc.FlowComparisonConfig(epochs=1),
    )

    assert estimate_calls
    call = estimate_calls[0]
    assert "theta_data" not in call
    assert "x_data" not in call
    assert "normalization" not in call
    np.testing.assert_allclose(call["theta_all"], np.eye(3, dtype=np.float64))
    assert call["mc_jeffreys_sample"] == 4096
    assert result.canonical_metric_name == "model_jeffreys_symmetric_kl"


def test_flow_comparison_config_has_no_normalize_x_field() -> None:
    assert "normalize_x" not in dc.FlowComparisonConfig.__dataclass_fields__


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
    assert args.gt_samples_per_class == 100_000
    assert args.mc_jeffreys_sample == 4096
    assert args.radius == 1.0
    assert args.ode_method == "midpoint"
    assert mod._flow_config_from_args(args).mc_jeffreys_sample == 4096
    assert mod._flow_config_from_args(args).radius == 1.0
    assert mod.resolve_dataset_dir(args) == repo_root / "data" / "mog_5pr5_n1000"
    assert mod.resolve_output_dir(args) == repo_root / "data" / "mog_5pr5_n1000" / "distance_comparison_flow_skl"
