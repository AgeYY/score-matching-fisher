from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.continuous_fisher_comparison import METHOD_CLASSICAL_LINEAR, METHOD_FLOW_LINEAR
from fisher.stringer_session_identification import (
    DISTANCE_AREA_L2,
    DISTANCE_PRIMARY,
    DISTANCE_RMSE,
    HALF_A,
    HALF_B,
    DISTANCES,
    METHODS,
    HalfCurveResult,
    IdentificationResult,
    circular_endpoint_windows,
    compute_identification,
    curve_distance,
    fit_half_pca,
    plot_all_distance_summary,
    parse_optional_int,
    stratified_half_split,
    theta_grid_periodic,
    theta_midpoints,
    write_results_npz,
    write_summary_json,
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
                "B_to_A": np.asarray([[0.2, 0.9], [0.7, 0.1]], dtype=np.float64),
            }
            for direction in ("A_to_B", "B_to_A"):
                summary["identification"][method][f"{distance_name}_{direction}"] = {
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
                "B_to_A": np.eye(2, dtype=np.float64),
            }
            for direction in ("A_to_B", "B_to_A"):
                summary["identification"][method][f"{distance_name}_{direction}"] = {
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


def test_cli_defaults_match_session_identification_plan() -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args([])

    assert args.device == "cuda:1"
    assert args.session_stimuli_type == "gratings_static"
    assert args.max_sessions is None
    assert args.theta_grid_size == 17
    assert args.pca_dim == 50
    assert args.epochs == 50000
    assert args.early_patience == 1000
    assert args.visualization_only is False
