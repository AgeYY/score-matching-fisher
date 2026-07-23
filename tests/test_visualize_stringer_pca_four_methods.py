from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "bin"
        / "visualize_stringer_pca_four_methods.py"
    )
    spec = importlib.util.spec_from_file_location(
        "visualize_stringer_pca_four_methods", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_requested_bin_width_maps_to_equal_periodic_bins() -> None:
    module = _load_script()
    n_bins, effective = module._periodic_bin_spec(0.2)
    assert n_bins == 16
    assert np.isclose(effective, np.pi / 16.0)
    theta = np.asarray([-1e-9, 0.0, effective, np.pi - 1e-9, np.pi])
    got = module._periodic_bin_indices(theta, n_bins=n_bins)
    np.testing.assert_array_equal(got, [15, 0, 1, 15, 0])


def test_binned_ledoit_wolf_returns_finite_moments_and_counts() -> None:
    module = _load_script()
    rng = np.random.default_rng(7)
    theta = np.repeat(np.arange(4) * (np.pi / 4.0), 20)
    x = rng.normal(size=(theta.size, 3)) + np.repeat(
        np.arange(4, dtype=np.float64), 20
    )[:, None]
    means, covariances, counts = module._fit_binned_ledoit_wolf(
        x, theta, n_bins=4
    )
    assert means.shape == (4, 3)
    assert covariances.shape == (4, 3, 3)
    np.testing.assert_array_equal(counts, np.full(4, 20))
    assert np.isfinite(means).all()
    assert np.isfinite(covariances).all()
    assert np.min(np.linalg.eigvalsh(covariances)) > 0.0


def test_grouped_gaussian_likelihood_matches_observationwise_evaluation() -> None:
    module = _load_script()
    rng = np.random.default_rng(11)
    x = rng.normal(size=(12, 3))
    groups = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    means = rng.normal(size=(3, 3))
    matrices = rng.normal(size=(3, 3, 3))
    covariances = np.einsum("nij,nkj->nik", matrices, matrices) + 0.2 * np.eye(3)
    expected = module._gaussian_log_likelihood(
        x,
        means[groups],
        covariances[groups],
        jitter=1e-6,
    )
    observed = module._grouped_gaussian_log_likelihood(
        x,
        groups,
        means,
        covariances,
        jitter=1e-6,
    )
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_relative_likelihood_uses_per_session_reference() -> None:
    module = _load_script()
    values = np.asarray(
        [
            [-5.0, -4.0, -3.0, -2.0],
            [-15.0, -12.0, -10.0, -8.0],
        ]
    )
    relative = module._relative_likelihood_values(
        values,
        ["GKR", "Bin +\nLW", "Affine\nFlow", "Nonlinear\nFlow"],
        reference_label="Bin +\nLW",
    )
    np.testing.assert_allclose(
        relative,
        np.asarray([[-1.0, 0.0, 1.0, 2.0], [-3.0, 0.0, 2.0, 4.0]]),
    )


def test_split_protocol_rejects_test_leakage() -> None:
    module = _load_script()
    module._validate_split_protocol(
        outer_train=np.asarray([0, 1, 2, 3]),
        flow_train=np.asarray([0, 1]),
        flow_validation=np.asarray([2, 3]),
        test=np.asarray([4, 5]),
    )
    with pytest.raises(ValueError, match="overlap"):
        module._validate_split_protocol(
            outer_train=np.asarray([0, 1, 2, 3]),
            flow_train=np.asarray([0, 1]),
            flow_validation=np.asarray([2, 3]),
            test=np.asarray([3, 4]),
        )


def test_plot_writes_four_method_png_and_svg(tmp_path: Path) -> None:
    module = _load_script()
    rng = np.random.default_rng(17)
    test = rng.normal(size=(60, 2))
    generated = rng.normal(size=(70, 2))
    theta_test = rng.uniform(0.0, np.pi, size=test.shape[0])
    generated_theta = rng.uniform(0.0, np.pi, size=generated.shape[0])
    selected = np.linspace(0.0, np.pi, 4, endpoint=False)
    means = np.column_stack([np.cos(2.0 * selected), np.sin(2.0 * selected)])
    covariances = np.repeat(np.eye(2)[None, :, :], selected.size, axis=0) * 0.2
    likelihoods = {
        "GKR": rng.normal(-13.0, 1.0, size=60),
        "Bin +\nLW": rng.normal(-11.0, 1.0, size=60),
        "Affine\nFlow": rng.normal(-12.0, 1.0, size=60),
        "Nonlinear\nFlow": rng.normal(-10.0, 1.0, size=60),
    }
    png, svg = module._plot(
        test_pc12=test,
        theta_test=theta_test,
        selected_theta=selected,
        affine_mean=means,
        affine_covariance=covariances,
        gkr_mean=0.9 * means,
        gkr_covariance=1.1 * covariances,
        binned_mean=1.1 * means,
        binned_covariance=0.9 * covariances,
        generated_pc12=generated,
        generated_theta=generated_theta,
        likelihoods=likelihoods,
        output_dir=tmp_path,
    )
    assert png.is_file() and png.stat().st_size > 0
    assert svg.is_file() and svg.stat().st_size > 0
