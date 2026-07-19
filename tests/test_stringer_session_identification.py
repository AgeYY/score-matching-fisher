from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from matplotlib.axes import Axes

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.continuous_fisher_comparison import ContinuousFlowConfig, METHOD_CLASSICAL_LINEAR, METHOD_FLOW_LINEAR
from fisher.stringer_session_identification import (
    DISTANCE_AREA_L2,
    DISTANCE_PRIMARY,
    DISTANCE_RMSE,
    DIRECTION_A_TO_B,
    FLOW_ORIENTATION_ENCODING_PERIODIC_SINCOS,
    FLOW_ORIENTATION_ENCODING_PERIODIC_RBF,
    FLOW_ORIENTATION_ENCODING_SCALAR,
    STRINGER_FLOW_FIXED_VALIDATION_PATHS,
    STRINGER_FLOW_PERIODIC_RBF_NUM_CENTERS,
    HALF_A,
    HALF_B,
    DISTANCES,
    METHODS,
    HalfCurveResult,
    IdentificationResult,
    SubsampleConvergenceResult,
    bootstrap_indices,
    circular_endpoint_windows,
    compute_identification,
    compute_query_reference_identification,
    curve_distance,
    encode_flow_orientation,
    estimate_affine_mixed_symmetric_kl_fisher_for_conditions,
    fit_half_pca,
    half_curve_signature,
    load_subsample_results_npz,
    plot_all_distance_summary,
    plot_subsample_convergence_summary,
    plot_subsample_logcorr_example,
    plot_subsample_topk_accuracy,
    parse_optional_int,
    parse_positive_int_list,
    select_logcorr_flow_advantage_example,
    stratified_half_split,
    stratified_bootstrap_indices,
    theta_grid_periodic,
    theta_midpoints,
    write_results_npz,
    write_subsample_curves_csv,
    write_subsample_results_npz,
    write_summary_json,
    zscore_log_fisher_curve,
)
from global_setting import (
    DEFAULT_DEVICE,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_TRAINING_MAX_EPOCHS,
)


def _load_cli_module():
    path = _REPO_ROOT / "bin" / "compare_stringer_fisher_session_identification.py"
    spec = importlib.util.spec_from_file_location("compare_stringer_fisher_session_identification", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_optional_int() -> None:
    assert parse_optional_int(None) is None
    assert parse_optional_int("all") is None
    assert parse_optional_int("none") is None
    assert parse_optional_int("3") == 3
    with pytest.raises(ValueError):
        parse_optional_int("0")


def test_parse_positive_int_list_sorts_unique_values() -> None:
    assert parse_positive_int_list("512,64,64,128") == [64, 128, 512]
    with pytest.raises(ValueError):
        parse_positive_int_list("")
    with pytest.raises(ValueError):
        parse_positive_int_list("0,1")


def test_encode_flow_orientation_periodic_wraps_at_period() -> None:
    period = float(np.pi)
    theta = np.asarray([0.0, period, period / 4.0, period / 2.0], dtype=np.float64)

    encoded = encode_flow_orientation(theta, period=period, encoding=FLOW_ORIENTATION_ENCODING_PERIODIC_SINCOS)

    assert encoded.shape == (4, 2)
    np.testing.assert_allclose(encoded[0], encoded[1], atol=1e-12)
    np.testing.assert_allclose(encoded[2], [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(encoded[3], [-1.0, 0.0], atol=1e-12)


def test_encode_flow_orientation_periodic_rbf_is_local_and_wraps_at_period() -> None:
    period = float(np.pi)
    spacing = period / STRINGER_FLOW_PERIODIC_RBF_NUM_CENTERS
    theta = np.asarray([0.0, period, spacing, period / 2.0], dtype=np.float64)

    encoded = encode_flow_orientation(theta, period=period, encoding=FLOW_ORIENTATION_ENCODING_PERIODIC_RBF)

    assert encoded.shape == (4, STRINGER_FLOW_PERIODIC_RBF_NUM_CENTERS)
    np.testing.assert_allclose(encoded[0], encoded[1], atol=1e-12)
    assert int(np.argmax(encoded[0])) == 0
    assert int(np.argmax(encoded[2])) == 1
    assert int(np.argmax(encoded[3])) == STRINGER_FLOW_PERIODIC_RBF_NUM_CENTERS // 2
    np.testing.assert_allclose(encoded.max(axis=1), 1.0, atol=1e-12)


def test_encode_flow_orientation_scalar_keeps_original_theta_column() -> None:
    theta = np.asarray([0.0, 0.25, 0.5], dtype=np.float64)

    encoded = encode_flow_orientation(theta, period=float(np.pi), encoding=FLOW_ORIENTATION_ENCODING_SCALAR)

    assert encoded.shape == (3, 1)
    np.testing.assert_allclose(encoded[:, 0], theta)


def test_flow_orientation_encoding_changes_half_curve_cache_signature() -> None:
    session_info = type("SessionInfo", (), {"session_file": "session_0.npy"})()
    common = dict(
        session_info=session_info,
        session_index=0,
        half_label=HALF_A,
        half_indices=np.asarray([0, 1, 2], dtype=np.int64),
        theta_grid=theta_grid_periodic(float(np.pi), 5),
        period=float(np.pi),
        pca_dim=2,
        pca_random_state=0,
        pca_whiten=True,
        train_frac=0.8,
        seed=1,
        flow_config=ContinuousFlowConfig(epochs=3),
        classical_ridge=1e-6,
        classical_window_radius=None,
        classical_min_endpoint_samples=1,
    )

    periodic = half_curve_signature(flow_orientation_encoding=FLOW_ORIENTATION_ENCODING_PERIODIC_SINCOS, **common)
    periodic_rbf = half_curve_signature(flow_orientation_encoding=FLOW_ORIENTATION_ENCODING_PERIODIC_RBF, **common)
    scalar = half_curve_signature(flow_orientation_encoding=FLOW_ORIENTATION_ENCODING_SCALAR, **common)

    assert len({periodic, periodic_rbf, scalar}) == 3
    assert f'"flow_fixed_validation_paths":{STRINGER_FLOW_FIXED_VALIDATION_PATHS}' in periodic
    assert f'"flow_periodic_rbf_num_centers":{STRINGER_FLOW_PERIODIC_RBF_NUM_CENTERS}' in periodic_rbf


class _DummyAffineModel(torch.nn.Module):
    x_dim = 1

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def endpoint_mean(self, condition: torch.Tensor) -> torch.Tensor:
        return condition[:, :1]

    def A(self, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del condition, t
        return torch.zeros((1, 1, 1), dtype=self.weight.dtype, device=self.weight.device)


def test_condition_fisher_readout_uses_scalar_theta_spacing_with_2d_conditions() -> None:
    theta = np.asarray([0.0, 0.5], dtype=np.float64)
    condition = np.asarray([[0.0, 1.0], [2.0, 0.0]], dtype=np.float64)

    fd = estimate_affine_mixed_symmetric_kl_fisher_for_conditions(
        model=_DummyAffineModel(),
        theta_all=theta,
        condition_all=condition,
        device=torch.device("cpu"),
        ode_steps=1,
        ridge=0.0,
    )

    np.testing.assert_allclose(fd["condition_left"], condition[:1])
    np.testing.assert_allclose(fd["condition_right"], condition[1:])
    np.testing.assert_allclose(fd["adjacent_symmetric_kl"], [4.0])
    np.testing.assert_allclose(fd["fisher"], [16.0])


def test_stratified_half_split_is_complete_disjoint_and_balanced() -> None:
    theta = np.linspace(0.0, np.pi, 101, endpoint=False, dtype=np.float64)

    a, b = stratified_half_split(theta, n_bins=8, period=float(np.pi), seed=11)

    assert np.intersect1d(a, b).size == 0
    np.testing.assert_array_equal(np.sort(np.concatenate([a, b])), np.arange(theta.shape[0]))
    bins = np.floor(theta / np.pi * 8).astype(int)
    for bin_id in range(8):
        na = int(np.sum(bins[a] == bin_id))
        nb = int(np.sum(bins[b] == bin_id))
        assert abs(na - nb) <= 1


def test_stratified_bootstrap_indices_preserves_bin_proportions() -> None:
    theta = np.concatenate(
        [
            np.full(10, 0.05),
            np.full(20, 0.35 * np.pi),
            np.full(30, 0.70 * np.pi),
        ]
    )

    idx = stratified_bootstrap_indices(theta, n_samples=12, n_bins=4, period=float(np.pi), seed=5)
    got_bins = np.floor(theta[idx] / np.pi * 4).astype(int)

    assert idx.shape == (12,)
    assert np.max(np.bincount(got_bins, minlength=4)) > 1
    assert np.bincount(got_bins, minlength=4).tolist() == [2, 4, 6, 0]


def test_bootstrap_indices_uniform_is_deterministic_with_replacement() -> None:
    theta = np.linspace(0.0, np.pi, 5, endpoint=False)

    a = bootstrap_indices(theta, n_samples=20, n_bins=4, period=float(np.pi), seed=17, sampling="uniform")
    b = bootstrap_indices(theta, n_samples=20, n_bins=4, period=float(np.pi), seed=17, sampling="uniform")

    np.testing.assert_array_equal(a, b)
    assert a.shape == (20,)
    assert len(np.unique(a)) < a.shape[0]


def test_stratified_bootstrap_indices_supports_without_replacement() -> None:
    theta = np.concatenate(
        [
            np.full(10, 0.05),
            np.full(20, 0.35 * np.pi),
            np.full(30, 0.70 * np.pi),
        ]
    )

    idx = stratified_bootstrap_indices(theta, n_samples=12, n_bins=4, period=float(np.pi), seed=5, replace=False)
    got_bins = np.floor(theta[idx] / np.pi * 4).astype(int)

    assert idx.shape == (12,)
    assert np.unique(idx).shape[0] == idx.shape[0]
    assert np.bincount(got_bins, minlength=4).tolist() == [2, 4, 6, 0]


def test_bootstrap_indices_uniform_without_replacement_is_unique_and_bounded() -> None:
    theta = np.linspace(0.0, np.pi, 5, endpoint=False)

    idx = bootstrap_indices(theta, n_samples=5, n_bins=4, period=float(np.pi), seed=17, sampling="uniform", replace=False)

    assert idx.shape == (5,)
    assert sorted(idx.tolist()) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="without replacement"):
        bootstrap_indices(theta, n_samples=6, n_bins=4, period=float(np.pi), seed=17, sampling="uniform", replace=False)


def test_circular_endpoint_windows_wrap_zero_and_period() -> None:
    period = float(np.pi)
    theta = np.asarray([0.01, period - 0.01, 0.5 * period, 0.75 * period], dtype=np.float64)
    x = np.arange(8, dtype=np.float64).reshape(4, 2)
    grid = theta_grid_periodic(period, 5)

    windows = circular_endpoint_windows(
        theta_all=theta,
        x_all=x,
        theta_grid=grid,
        period=period,
        radius=0.02,
        min_endpoint_samples=1,
    )

    assert set(windows[0].tolist()) == {0, 1}
    assert set(windows[-1].tolist()) == {0, 1}


def test_fit_half_pca_metadata_is_half_scoped_and_label_blind() -> None:
    rng = np.random.default_rng(13)
    responses = rng.normal(size=(30, 9))

    pca = fit_half_pca(
        responses,
        n_components=4,
        random_state=0,
        whiten=True,
        session_key="session0",
        half_label=HALF_A,
    )

    assert pca.x_all.shape == (30, 4)
    assert pca.metadata["pca_fit_scope"] == "half"
    assert pca.metadata["pca_input_uses_orientation_labels"] is False
    assert pca.metadata["pca_trial_averaging_before_fit"] is False
    assert pca.metadata["half_label"] == HALF_A


def test_primary_curve_distance_is_scale_invariant_after_log_zscore() -> None:
    theta_mid = np.linspace(0.1, 1.0, 5, dtype=np.float64)
    curve = np.asarray([1.0, 2.0, 4.0, 3.0, 1.5], dtype=np.float64)
    scaled = 10.0 * curve

    primary = curve_distance(curve, scaled, theta_mid, distance=DISTANCE_PRIMARY)
    area_l2 = curve_distance(curve, scaled, theta_mid, distance=DISTANCE_AREA_L2)

    assert primary == pytest.approx(0.0, abs=1e-12)
    assert area_l2 > 0.0


def _half_result(session_index: int, session_key: str, half_label: str, curve: np.ndarray) -> HalfCurveResult:
    grid = theta_grid_periodic(float(np.pi), curve.shape[0] + 1)
    return HalfCurveResult(
        session_index=session_index,
        session_key=session_key,
        session_file=f"{session_key}.npy",
        half_label=half_label,
        n_trials=20,
        n_neurons=5,
        theta_grid=grid,
        theta_midpoints=theta_midpoints(grid),
        curves={METHOD_CLASSICAL_LINEAR: curve, METHOD_FLOW_LINEAR: curve},
        pca_metadata={"pca_fit_scope": "half"},
        train_metadata={},
        cache_path=Path(f"{session_key}_{half_label}.npz"),
        flow_npz_path=None,
    )


def test_compute_identification_finds_synthetic_pairs() -> None:
    curves = [
        np.asarray([1.0, 2.0, 3.0, 2.0], dtype=np.float64),
        np.asarray([3.0, 1.5, 1.0, 1.5], dtype=np.float64),
        np.asarray([1.0, 3.0, 1.0, 4.0], dtype=np.float64),
    ]
    halves: list[HalfCurveResult] = []
    for idx, curve in enumerate(curves):
        key = f"session{idx}"
        halves.append(_half_result(idx, key, HALF_A, curve))
        halves.append(_half_result(idx, key, HALF_B, 5.0 * curve))

    _, _, summary = compute_identification(halves, theta_mid=halves[0].theta_midpoints)

    for method in (METHOD_CLASSICAL_LINEAR, METHOD_FLOW_LINEAR):
        got = summary[method][f"{DISTANCE_PRIMARY}_A_to_B"]
        assert got["top1_accuracy"] == pytest.approx(1.0)
        assert got["ranks"] == [1, 1, 1]
        assert f"{DISTANCE_PRIMARY}_B_to_A" not in summary[method]


def test_compute_query_reference_identification_supports_row_metadata() -> None:
    query = [
        _half_result(0, "s0", HALF_A, np.asarray([1.0, 2.0, 1.0], dtype=np.float64)),
        _half_result(1, "s1", HALF_A, np.asarray([3.0, 1.0, 2.0], dtype=np.float64)),
    ]
    reference = [
        _half_result(0, "s0", HALF_B, np.asarray([2.0, 4.0, 2.0], dtype=np.float64)),
        _half_result(1, "s1", HALF_B, np.asarray([6.0, 2.0, 4.0], dtype=np.float64)),
    ]

    _, rows, summary = compute_query_reference_identification(
        query_results=query,
        reference_results=reference,
        theta_mid=query[0].theta_midpoints,
        session_keys=["s0", "s1"],
        query_half=HALF_A,
        candidate_half=HALF_B,
        row_extra={"subset_n": 64, "repeat": 0},
    )

    assert summary[METHOD_CLASSICAL_LINEAR][f"{DISTANCE_PRIMARY}_{DIRECTION_A_TO_B}"]["top1_accuracy"] == pytest.approx(1.0)
    assert all(row["subset_n"] == 64 and row["repeat"] == 0 for row in rows)


def test_plot_all_distance_summary_writes_files(tmp_path: Path) -> None:
    session_keys = ["s0", "s1"]
    grid = theta_grid_periodic(float(np.pi), 3)
    summary = {"identification": {}}
    distances = {}
    for method in METHODS:
        distances[method] = {}
        summary["identification"][method] = {}
        for distance_name in DISTANCES:
            distances[method][distance_name] = {
                "A_to_B": np.asarray([[0.1, 1.0], [0.8, 0.2]], dtype=np.float64),
            }
            summary["identification"][method][f"{distance_name}_{DIRECTION_A_TO_B}"] = {
                "top1_accuracy": 1.0,
                "top2_accuracy": 1.0,
                "top3_accuracy": 1.0,
                "mean_reciprocal_rank": 1.0,
                "ranks": [1, 1],
                "tie_counts": [1, 1],
            }
    result = IdentificationResult(
        session_keys=session_keys,
        theta_grid=grid,
        theta_midpoints=theta_midpoints(grid),
        half_results=[],
        distances=distances,
        pair_rows=[],
        curve_rows=[],
        summary=summary,
    )

    svg, png = plot_all_distance_summary(tmp_path / "all.svg", tmp_path / "all.png", result)

    assert svg.is_file()
    assert png.is_file()


def _subsample_result() -> SubsampleConvergenceResult:
    session_keys = ["s0", "s1"]
    grid = theta_grid_periodic(float(np.pi), 3)
    endpoint_summary = {"identification": {}}
    endpoint_distances = {}
    subset_matrices = {}
    convergence = {}
    for method in METHODS:
        endpoint_distances[method] = {}
        subset_matrices[method] = {}
        endpoint_summary["identification"][method] = {}
        convergence[method] = {}
        for distance_name in DISTANCES:
            mat = np.asarray([[0.1, 1.0], [0.8, 0.2]], dtype=np.float64)
            endpoint_distances[method][distance_name] = {DIRECTION_A_TO_B: mat}
            subset_matrices[method][distance_name] = np.stack([mat + 0.2, mat], axis=0).reshape(2, 1, 2, 2)
            endpoint_summary["identification"][method][f"{distance_name}_{DIRECTION_A_TO_B}"] = {
                "top1_accuracy": 1.0,
                "top2_accuracy": 1.0,
                "top3_accuracy": 1.0,
                "mean_reciprocal_rank": 1.0,
                "ranks": [1, 1],
                "tie_counts": [1, 1],
            }
            convergence[method][distance_name] = {
                "n_values": [64, 128],
                "top1_by_repeat": [[0.5, 1.0], [1.0, 1.0]],
                "top2_by_repeat": [[1.0, 1.0], [1.0, 1.0]],
                "top3_by_repeat": [[1.0, 1.0], [1.0, 1.0]],
                "mrr_by_repeat": [[0.75, 1.0], [1.0, 1.0]],
                "top1_mean": [0.75, 1.0],
                "top2_mean": [1.0, 1.0],
                "top3_mean": [1.0, 1.0],
                "mrr_mean": [0.875, 1.0],
                "full_a": endpoint_summary["identification"][method][f"{distance_name}_{DIRECTION_A_TO_B}"],
            }
    endpoint = IdentificationResult(
        session_keys=session_keys,
        theta_grid=grid,
        theta_midpoints=theta_midpoints(grid),
        half_results=[],
        distances=endpoint_distances,
        pair_rows=[],
        curve_rows=[],
        summary=endpoint_summary,
    )
    return SubsampleConvergenceResult(
        session_keys=session_keys,
        theta_grid=grid,
        theta_midpoints=theta_midpoints(grid),
        n_values=[64, 128],
        repeats=2,
        sampling="stratified",
        endpoint_result=endpoint,
        subset_matrices=subset_matrices,
        pair_rows=[],
        curve_rows=[],
        summary={
            "endpoint_identification": endpoint_summary["identification"],
            "subsample_convergence": convergence,
        },
    )


def test_plot_subsample_convergence_summary_writes_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = _subsample_result()
    calls: list[dict[str, object]] = []
    original_errorbar = Axes.errorbar

    def spy_errorbar(self, x, y, *args, **kwargs):
        calls.append(
            {
                "x": np.asarray(x, dtype=np.float64),
                "y": np.asarray(y, dtype=np.float64),
                "yerr": np.asarray(kwargs.get("yerr"), dtype=np.float64),
                "label": kwargs.get("label"),
            }
        )
        return original_errorbar(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", spy_errorbar)

    svg, png = plot_subsample_convergence_summary(tmp_path / "subsample.svg", tmp_path / "subsample.png", result)

    assert svg.is_file()
    assert png.is_file()
    assert len(calls) == len(DISTANCES) * len(METHODS) * 2
    assert {call["label"] for call in calls} == {"classical", "flow matching"}
    top1_calls = [call for call in calls if np.allclose(call["y"], [0.75, 1.0])]
    top3_calls = [call for call in calls if np.allclose(call["y"], [1.0, 1.0])]
    assert len(top1_calls) == len(DISTANCES) * len(METHODS)
    assert len(top3_calls) == len(DISTANCES) * len(METHODS)
    for call in calls:
        np.testing.assert_array_equal(call["x"], [0.0, 1.0])
    for call in top1_calls:
        np.testing.assert_allclose(call["yerr"], [np.sqrt(0.125), 0.0])
    for call in top3_calls:
        np.testing.assert_allclose(call["yerr"], [0.0, 0.0])


def test_plot_subsample_topk_accuracy_writes_two_row_figure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _subsample_result()
    calls: list[dict[str, object]] = []
    original_errorbar = Axes.errorbar

    def spy_errorbar(self, x, y, *args, **kwargs):
        calls.append(
            {
                "x": np.asarray(x, dtype=np.float64),
                "y": np.asarray(y, dtype=np.float64),
                "yerr": np.asarray(kwargs.get("yerr"), dtype=np.float64),
                "label": kwargs.get("label"),
            }
        )
        return original_errorbar(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", spy_errorbar)

    svg, png = plot_subsample_topk_accuracy(tmp_path / "topk.svg", tmp_path / "topk.png", result)

    assert svg.is_file()
    assert png.is_file()
    assert len(calls) == len(DISTANCES) * len(METHODS) * 2
    assert {call["label"] for call in calls} == {"classical", "flow matching"}
    top1_calls = [call for call in calls if np.allclose(call["y"], [0.75, 1.0])]
    top3_calls = [call for call in calls if np.allclose(call["y"], [1.0, 1.0])]
    assert len(top1_calls) == len(DISTANCES) * len(METHODS)
    assert len(top3_calls) == len(DISTANCES) * len(METHODS)
    for call in calls:
        np.testing.assert_array_equal(call["x"], [0.0, 1.0])


def _logcorr_example_result() -> SubsampleConvergenceResult:
    base = _subsample_result()
    summary = {
        "session_keys": ["s0", "s1"],
        "seed": 0,
        "orientation_period": float(np.pi),
        "subsample_runs": [
            {
                "subset_n": 650,
                "repeat": 0,
                "identification": {
                    METHOD_CLASSICAL_LINEAR: {
                        f"{DISTANCE_PRIMARY}_{DIRECTION_A_TO_B}": {
                            "top1_accuracy": 0.5,
                            "top2_accuracy": 1.0,
                            "top3_accuracy": 1.0,
                            "mean_reciprocal_rank": 0.75,
                            "ranks": [1, 2],
                            "tie_counts": [1, 1],
                        }
                    },
                    METHOD_FLOW_LINEAR: {
                        f"{DISTANCE_PRIMARY}_{DIRECTION_A_TO_B}": {
                            "top1_accuracy": 0.5,
                            "top2_accuracy": 1.0,
                            "top3_accuracy": 1.0,
                            "mean_reciprocal_rank": 0.75,
                            "ranks": [2, 1],
                            "tie_counts": [1, 1],
                        }
                    },
                },
            }
        ],
        "subsample_convergence": {},
    }
    for method in METHODS:
        summary["subsample_convergence"][method] = {}
        for distance_name in DISTANCES:
            summary["subsample_convergence"][method][distance_name] = {
                "top1_by_repeat": [[0.5, 1.0], [1.0, 1.0]],
                "top1_mean": [0.75, 1.0],
            }
    return SubsampleConvergenceResult(
        session_keys=base.session_keys,
        theta_grid=base.theta_grid,
        theta_midpoints=base.theta_midpoints,
        n_values=[650, 2000],
        repeats=2,
        sampling=base.sampling,
        endpoint_result=base.endpoint_result,
        subset_matrices=base.subset_matrices,
        pair_rows=[],
        curve_rows=[],
        summary=summary,
    )


def _write_logcorr_example_curves(path: Path) -> Path:
    rows = []
    theta_values = [0.1, 0.2, 0.3]
    for method in METHODS:
        for subset_n, repeat, half_label, values in (
            (650, 0, HALF_A, [1.0, 2.0, 4.0]),
            ("full", -1, HALF_B, [1.5, 3.0, 6.0]),
        ):
            for theta, fisher in zip(theta_values, values):
                rows.append(
                    {
                        "subset_n": subset_n,
                        "repeat": repeat,
                        "sampling": "stratified",
                        "session_index": 1,
                        "session_key": "s1",
                        "session_file": "s1.npy",
                        "half_label": half_label,
                        "method": method,
                        "theta_midpoint": theta,
                        "theta_left": theta - 0.05,
                        "theta_right": theta + 0.05,
                        "fisher": fisher,
                        "n_trials_half": 650,
                        "n_neurons": 10,
                    }
                )
    return write_subsample_curves_csv(path, rows)


def test_zscore_log_fisher_curve_handles_scale_and_degenerate_values() -> None:
    z = zscore_log_fisher_curve(np.asarray([1.0, np.e, np.e**2], dtype=np.float64))

    assert np.mean(z) == pytest.approx(0.0)
    assert np.std(z) == pytest.approx(1.0)
    np.testing.assert_allclose(zscore_log_fisher_curve(np.asarray([5.0, 5.0])), [0.0, 0.0])
    np.testing.assert_allclose(zscore_log_fisher_curve(np.asarray([0.0, -1.0])), [0.0, 0.0])


def test_select_logcorr_flow_advantage_example_uses_n650_candidate() -> None:
    result = _logcorr_example_result()

    selected = select_logcorr_flow_advantage_example(result.summary, n_subset=650)

    assert selected["subset_n"] == 650
    assert selected["repeat"] == 0
    assert selected["session_index"] == 1
    assert selected["session_key"] == "s1"
    assert selected["classical_rank"] == 2
    assert selected["flow_rank"] == 1


def test_select_logcorr_flow_advantage_example_can_fall_back_without_advantage() -> None:
    result = _logcorr_example_result()
    flow = result.summary["subsample_runs"][0]["identification"][METHOD_FLOW_LINEAR]
    flow[f"{DISTANCE_PRIMARY}_{DIRECTION_A_TO_B}"]["ranks"] = [2, 2]

    selected = select_logcorr_flow_advantage_example(
        result.summary,
        n_subset=650,
        require_advantage=False,
    )

    assert selected["session_index"] == 1
    assert selected["classical_rank"] == selected["flow_rank"] == 2


def test_plot_subsample_logcorr_example_writes_three_panel_figure(tmp_path: Path) -> None:
    result = _logcorr_example_result()
    curves_csv = _write_logcorr_example_curves(tmp_path / "curves.csv")

    svg, png = plot_subsample_logcorr_example(tmp_path / "example.svg", tmp_path / "example.png", result, curves_csv)

    assert svg.is_file()
    assert png.is_file()


def test_subsample_results_npz_round_trip(tmp_path: Path) -> None:
    result = _subsample_result()
    path = write_subsample_results_npz(tmp_path / "subsample.npz", result)

    loaded = load_subsample_results_npz(path, result.summary)

    assert loaded.session_keys == result.session_keys
    assert loaded.n_values == [64, 128]
    assert loaded.repeats == 2
    assert loaded.sampling == "stratified"
    assert loaded.summary["subsample_replace"] is True
    assert loaded.summary["subsample_without_replacement"] is False
    for method in METHODS:
        for distance_name in DISTANCES:
            np.testing.assert_allclose(
                loaded.subset_matrices[method][distance_name],
                result.subset_matrices[method][distance_name],
            )


def test_visualization_only_loader_reads_npz_and_summary(tmp_path: Path) -> None:
    mod = _load_cli_module()
    session_keys = ["s0", "s1"]
    grid = theta_grid_periodic(float(np.pi), 3)
    summary = {"identification": {}, "session_keys": session_keys}
    distances = {}
    for method in METHODS:
        distances[method] = {}
        summary["identification"][method] = {}
        for distance_name in DISTANCES:
            distances[method][distance_name] = {
                "A_to_B": np.eye(2, dtype=np.float64),
            }
            summary["identification"][method][f"{distance_name}_{DIRECTION_A_TO_B}"] = {
                "top1_accuracy": 1.0,
                "top2_accuracy": 1.0,
                "top3_accuracy": 1.0,
                "mean_reciprocal_rank": 1.0,
                "ranks": [1, 1],
                "tie_counts": [1, 1],
            }
    result = IdentificationResult(
        session_keys=session_keys,
        theta_grid=grid,
        theta_midpoints=theta_midpoints(grid),
        half_results=[],
        distances=distances,
        pair_rows=[],
        curve_rows=[],
        summary=summary,
    )
    write_results_npz(tmp_path / "stringer_session_identification_results.npz", result)
    write_summary_json(tmp_path / "stringer_session_identification_summary.json", result)

    loaded = mod._load_visualization_result(tmp_path)

    assert loaded.session_keys == session_keys
    for method in METHODS:
        for distance_name in (DISTANCE_PRIMARY, DISTANCE_AREA_L2, DISTANCE_RMSE):
            np.testing.assert_allclose(loaded.distances[method][distance_name]["A_to_B"], np.eye(2))
            assert "B_to_A" not in loaded.distances[method][distance_name]


def test_cli_defaults_match_session_identification_plan() -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args([])

    assert args.device == DEFAULT_DEVICE
    assert args.session_stimuli_type == "gratings_static"
    assert args.max_sessions is None
    assert args.theta_grid_size == 17
    assert args.pca_dim == 50
    assert args.flow_orientation_encoding == FLOW_ORIENTATION_ENCODING_PERIODIC_SINCOS
    assert args.epochs == DEFAULT_TRAINING_MAX_EPOCHS
    assert args.early_patience == DEFAULT_EARLY_STOPPING_PATIENCE
    assert args.visualization_only is False
    assert args.subsample_a_convergence is False
    assert args.subsample_a_n_list == "200,650,1100,1550,2000"
    assert args.subsample_a_repeats == 5
    assert args.subsample_a_sampling == "stratified"
    assert args.subsample_a_without_replacement is True

    with_replacement_args = mod.build_parser().parse_args(["--no-subsample-a-without-replacement"])
    assert with_replacement_args.subsample_a_without_replacement is False

    scalar_args = mod.build_parser().parse_args(["--flow-orientation-encoding", FLOW_ORIENTATION_ENCODING_SCALAR])
    assert scalar_args.flow_orientation_encoding == FLOW_ORIENTATION_ENCODING_SCALAR

    rbf_args = mod.build_parser().parse_args(["--flow-orientation-encoding", FLOW_ORIENTATION_ENCODING_PERIODIC_RBF])
    assert rbf_args.flow_orientation_encoding == FLOW_ORIENTATION_ENCODING_PERIODIC_RBF
