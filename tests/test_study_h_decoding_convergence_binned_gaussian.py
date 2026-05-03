from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

from fisher.shared_dataset_io import SharedDatasetBundle


def _load_module():
    repo = Path(__file__).resolve().parent.parent
    path = repo / "bin" / "study_h_decoding_convergence.py"
    spec = importlib.util.spec_from_file_location("study_h_decoding_convergence_binned_gaussian", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _subset(mod, x_all: np.ndarray, bin_all: np.ndarray):
    n = int(x_all.shape[0])
    theta = np.arange(n, dtype=np.float64).reshape(-1, 1)
    bundle = SharedDatasetBundle(
        meta={},
        theta_all=theta,
        x_all=np.asarray(x_all, dtype=np.float64),
        train_idx=np.arange(n, dtype=np.int64),
        validation_idx=np.asarray([], dtype=np.int64),
        theta_train=theta,
        x_train=np.asarray(x_all, dtype=np.float64),
        theta_validation=np.empty((0, 1), dtype=np.float64),
        x_validation=np.empty((0, int(x_all.shape[1])), dtype=np.float64),
    )
    return mod.SweepSubset(
        bundle=bundle,
        bin_all=np.asarray(bin_all, dtype=np.int64),
        bin_train=np.asarray(bin_all, dtype=np.int64),
        bin_validation=np.asarray([], dtype=np.int64),
    )


def test_binned_gaussian_hellinger_matches_shared_diagonal_formula() -> None:
    mod = _load_module()
    x_all = np.asarray([[0.0], [0.0], [2.0], [2.0]], dtype=np.float64)
    subset = _subset(mod, x_all=x_all, bin_all=np.asarray([0, 0, 1, 1]))
    h2 = mod._binned_gaussian_hellinger_sq(subset, n_bins=2, variance_floor=1.0)
    expected = 1.0 - math.exp(-0.125 * 4.0)
    assert h2.shape == (2, 2)
    assert np.allclose(np.diag(h2), 0.0)
    assert np.isclose(float(h2[0, 1]), expected)
    assert np.isclose(float(h2[1, 0]), expected)


def test_binned_gaussian_hellinger_fills_empty_bins_and_stays_finite() -> None:
    mod = _load_module()
    x_all = np.asarray([[0.0, 1.0], [3.0, 4.0]], dtype=np.float64)
    subset = _subset(mod, x_all=x_all, bin_all=np.asarray([0, 2]))
    h2 = mod._binned_gaussian_hellinger_sq(subset, n_bins=3, variance_floor=1e-3)
    assert h2.shape == (3, 3)
    assert np.allclose(np.diag(h2), 0.0)
    assert np.isfinite(h2).all()
    assert np.all((h2 >= 0.0) & (h2 <= 1.0))


def test_lxf_bin_likelihood_hellinger_matches_hand_logmeanexp_example() -> None:
    mod = _load_module()
    c = np.asarray(
        [
            [0.0, math.log(0.5), math.log(0.25), math.log(0.25)],
            [math.log(0.5), 0.0, math.log(0.25), math.log(0.125)],
            [math.log(0.25), math.log(0.25), 0.0, math.log(0.5)],
            [math.log(0.125), math.log(0.25), math.log(0.5), 0.0],
        ],
        dtype=np.float64,
    )
    bins = np.asarray([0, 0, 1, 1], dtype=np.int64)
    out = mod._lxf_bin_likelihood_hellinger(c, bins, 2)

    expected_bin_ll = np.column_stack(
        [
            np.log(np.mean(np.exp(c[:, [0, 1]]), axis=1)),
            np.log(np.mean(np.exp(c[:, [2, 3]]), axis=1)),
        ]
    )
    np.testing.assert_allclose(out["bin_log_likelihood"], expected_bin_ll, rtol=1e-12, atol=1e-12)
    expected_delta = expected_bin_ll - expected_bin_ll[np.arange(4), bins][:, None]
    np.testing.assert_allclose(out["bin_delta_l"], expected_delta, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.diag(out["h_binned"]), 0.0, rtol=0.0, atol=0.0)


def test_lxf_bin_likelihood_hellinger_expanded_averages_back_to_binned() -> None:
    mod = _load_module()
    c = np.asarray(
        [
            [1.0, 0.0, -1.0, -2.0],
            [0.5, 1.5, -1.5, -1.0],
            [-2.0, -1.0, 1.0, 0.5],
            [-1.5, -2.5, 0.0, 1.5],
        ],
        dtype=np.float64,
    )
    bins = np.asarray([0, 0, 1, 1], dtype=np.int64)
    out = mod._lxf_bin_likelihood_hellinger(c, bins, 2)
    h_binned_avg, _ = mod.vhb.average_matrix_by_bins(out["h_sym"], bins, 2)
    np.testing.assert_allclose(h_binned_avg, out["h_binned"], rtol=1e-12, atol=1e-12)
    assert np.all(out["h_sym"][bins[:, None] == bins[None, :]] == 0.0)


def test_lxf_bin_likelihood_hellinger_empty_bins_are_nan() -> None:
    mod = _load_module()
    c = np.asarray([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float64)
    out = mod._lxf_bin_likelihood_hellinger(c, np.asarray([0, 2], dtype=np.int64), 3)
    assert out["h_binned"].shape == (3, 3)
    assert np.isnan(out["h_binned"][1, :]).all()
    assert np.isnan(out["h_binned"][:, 1]).all()
    assert np.isfinite(out["h_sym"]).all()


def test_theta2_grid_binning_flattens_row_major() -> None:
    mod = _load_module()
    theta = np.asarray(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    grid = mod.prepare_theta2_grid_binning_for_convergence(
        theta,
        np.arange(theta.shape[0], dtype=np.int64),
        n_ref=theta.shape[0],
        n_bins_x=2,
        n_bins_y=2,
    )
    assert grid.grid_shape == (2, 2)
    assert grid.centers.shape == (4, 2)
    np.testing.assert_array_equal(grid.bin_idx_all, np.asarray([0, 1, 2, 3], dtype=np.int64))
    np.testing.assert_allclose(grid.centers[0], [-0.5, -0.5])
    np.testing.assert_allclose(grid.centers[3], [0.5, 0.5])


def test_removed_linear_x_flow_methods_do_not_normalize() -> None:
    mod = _load_module()
    for token in (
        "bin-gaussian-linear-x-flow-diagonal",
        "bin_lxf_diagonal",
        "linear-x-flow-schedule",
        "linear-x-flow-diagonal-theta-spline",
    ):
        assert mod._normalize_linear_x_flow_method(token) is None
