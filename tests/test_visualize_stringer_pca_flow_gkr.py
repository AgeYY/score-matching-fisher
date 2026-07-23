from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_script():
    path = Path(__file__).resolve().parents[1] / "bin" / "visualize_stringer_pca_flow_gkr.py"
    spec = importlib.util.spec_from_file_location("visualize_stringer_pca_flow_gkr", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stratified_outer_split_is_disjoint_and_complete() -> None:
    module = _load_script()
    theta = np.linspace(0.0, np.pi, 200, endpoint=False)
    train, test = module._stratified_train_test_split(
        theta, train_fraction=0.8, n_strata=16, seed=7
    )
    assert train.size == 160
    assert test.size == 40
    assert np.intersect1d(train, test).size == 0
    assert np.union1d(train, test).size == theta.size


def test_ellipse_parameters_match_diagonal_covariance() -> None:
    module = _load_script()
    width, height, angle = module._ellipse_parameters(np.diag([4.0, 1.0]))
    assert np.isclose(width, 4.0)
    assert np.isclose(height, 2.0)
    assert np.isclose(abs(angle), 180.0) or np.isclose(angle, 0.0)


def test_gaussian_log_likelihood_matches_standard_normal() -> None:
    module = _load_script()
    x = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    mean = np.zeros_like(x)
    covariance = np.repeat(np.eye(2)[None, :, :], 2, axis=0)
    got = module._gaussian_log_likelihood(x, mean, covariance, jitter=0.0)
    expected = np.asarray([-np.log(2.0 * np.pi), -np.log(2.0 * np.pi) - 0.5])
    np.testing.assert_allclose(got, expected)


def test_plot_writes_png_and_svg(tmp_path: Path) -> None:
    module = _load_script()
    rng = np.random.default_rng(7)
    test = rng.normal(size=(80, 2))
    theta_test = np.linspace(0.0, np.pi, test.shape[0], endpoint=False)
    selected = np.linspace(0.0, np.pi, 4, endpoint=False)
    means = np.column_stack([np.cos(2.0 * selected), np.sin(2.0 * selected)])
    covariance = np.repeat(np.eye(2)[None, :, :], selected.size, axis=0) * 0.1
    png, svg = module._plot(
        test_pc12=test,
        theta_test=theta_test,
        selected_theta=selected,
        flow_mean=means,
        flow_covariance=covariance,
        gkr_mean=0.8 * means,
        gkr_covariance=1.2 * covariance,
        flow_test_log_likelihood=rng.normal(-10.0, 1.0, size=test.shape[0]),
        gkr_test_log_likelihood=rng.normal(-9.0, 1.0, size=test.shape[0]),
        session_labels=["GT1", "GT2", "TX38"],
        flow_session_log_likelihood=np.asarray([-10.0, -11.0, -9.5]),
        gkr_session_log_likelihood=np.asarray([-11.0, -10.5, -10.0]),
        output_dir=tmp_path,
    )
    assert png.is_file()
    assert svg.is_file()
