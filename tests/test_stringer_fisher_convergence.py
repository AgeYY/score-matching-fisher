from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.stringer_fisher_convergence import (
    circular_distance,
    circular_endpoint_windows,
    curve_metrics,
    fit_pca_projection,
    parse_int_list,
    stratified_subset_indices,
    theta_grid_periodic,
)


def _load_cli_module():
    path = _REPO_ROOT / "bin" / "compare_stringer_fisher_convergence.py"
    spec = importlib.util.spec_from_file_location("compare_stringer_fisher_convergence", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_int_list() -> None:
    assert parse_int_list("1000, 2000,3000") == [1000, 2000, 3000]
    assert parse_int_list([4, 5]) == [4, 5]
    with pytest.raises(ValueError):
        parse_int_list("")
    with pytest.raises(ValueError):
        parse_int_list("1,0")


def test_circular_distance_wraps_orientation_period() -> None:
    period = float(np.pi)
    theta = np.asarray([0.01, period - 0.01, 0.5 * period], dtype=np.float64)

    got = circular_distance(theta, 0.0, period)

    np.testing.assert_allclose(got, [0.01, 0.01, 0.5 * period], rtol=1e-12, atol=1e-12)


def test_circular_endpoint_windows_include_both_sides_of_period() -> None:
    period = float(np.pi)
    theta = np.asarray([0.01, period - 0.01, 0.5 * period, 0.7 * period], dtype=np.float64).reshape(-1, 1)
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

    endpoint_zero_x = {tuple(row) for row in windows[0][1]}
    endpoint_period_x = {tuple(row) for row in windows[-1][1]}
    assert endpoint_zero_x == {(0.0, 1.0), (2.0, 3.0)}
    assert endpoint_period_x == endpoint_zero_x


def test_stratified_subset_indices_exact_size_and_reproducible() -> None:
    theta = np.linspace(0.0, np.pi, 101, endpoint=False, dtype=np.float64)

    first = stratified_subset_indices(theta, n_total=40, n_bins=4, period=float(np.pi), seed=123)
    second = stratified_subset_indices(theta, n_total=40, n_bins=4, period=float(np.pi), seed=123)

    assert first.shape == (40,)
    np.testing.assert_array_equal(first, second)
    bin_id = np.floor(theta[first] / np.pi * 4).astype(int)
    assert np.bincount(bin_id, minlength=4).min() > 0


def test_pca_projection_is_trial_level_and_label_blind() -> None:
    rng = np.random.default_rng(7)
    responses = rng.normal(size=(20, 8))

    result = fit_pca_projection(responses, n_components=3, random_state=0, whiten=True)

    assert result.x_all.shape == (20, 3)
    assert result.components.shape == (3, 8)
    assert result.metadata["pca_input_uses_orientation_labels"] is False
    assert result.metadata["pca_input"] == "neural_responses_only"


def test_curve_metrics() -> None:
    curve = np.asarray([1.0, 3.0, 5.0], dtype=np.float64)
    ref = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)

    got = curve_metrics(curve, ref, np.asarray([0.0, 0.5, 1.0], dtype=np.float64))

    assert got["mae"] == pytest.approx(2.0 / 3.0)
    assert got["rmse"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert got["pearson"] == pytest.approx(float(np.corrcoef(curve, ref)[0, 1]))
    assert got["area_normalized_l2"] > 0.0


def test_cli_defaults_are_stringer_convergence_defaults() -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args([])

    assert args.device == "cuda:1"
    assert args.pca_dim == 50
    assert args.theta_grid_size == 17
    assert args.n_list == [1000, 1500, 2000, 3000, 4000]
    assert args.n_repeats == 5
    assert args.epochs == 50000
    assert args.early_patience == 1000
