from __future__ import annotations

import numpy as np

from fisher.optimal_linear_estimator import (
    cross_fitted_ole_linear_fisher,
    optimal_linear_estimator,
)


def test_optimal_linear_estimator_matches_closed_form() -> None:
    mean_derivative = np.asarray([1.0, 2.0], dtype=np.float64)
    covariance = np.asarray([[2.0, 0.5], [0.5, 1.0]], dtype=np.float64)

    result = optimal_linear_estimator(mean_derivative, covariance)
    precision_derivative = np.linalg.solve(covariance, mean_derivative)
    expected_information = float(mean_derivative @ precision_derivative)
    expected_weights = precision_derivative / expected_information

    np.testing.assert_allclose(result.linear_fisher, expected_information)
    np.testing.assert_allclose(result.variance, 1.0 / expected_information)
    np.testing.assert_allclose(result.weights, expected_weights)
    np.testing.assert_allclose(result.weights @ mean_derivative, 1.0)
    np.testing.assert_allclose(
        result.weights @ covariance @ result.weights, result.variance
    )


def test_optimal_linear_estimator_supports_batched_moments() -> None:
    mean_derivative = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    covariance = np.asarray(
        [np.diag([2.0, 3.0]), np.diag([4.0, 5.0])], dtype=np.float64
    )

    result = optimal_linear_estimator(mean_derivative, covariance)

    np.testing.assert_allclose(result.linear_fisher, [0.5, 0.8])
    np.testing.assert_allclose(result.variance, [2.0, 1.25])
    np.testing.assert_allclose(
        np.einsum("bi,bi->b", result.weights, mean_derivative), [1.0, 1.0]
    )


def test_zero_mean_derivative_has_zero_information() -> None:
    result = optimal_linear_estimator(
        np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)
    )

    np.testing.assert_array_equal(result.weights, np.zeros(3, dtype=np.float64))
    np.testing.assert_allclose(result.linear_fisher, 0.0)
    assert np.isinf(result.variance)


def test_cross_fitted_ole_recovers_heldout_information() -> None:
    rng = np.random.default_rng(123)
    n_per_endpoint = 2_000
    theta = np.repeat([0.0, 1.0], n_per_endpoint)
    x = rng.normal(size=(2 * n_per_endpoint, 3))
    x[n_per_endpoint:, 0] += 1.0

    result = cross_fitted_ole_linear_fisher(
        theta,
        x,
        np.asarray([0.0, 1.0]),
        n_splits=5,
        seed=17,
    )

    assert result.fold_weights.shape == (1, 5, 3)
    assert result.fold_intercepts.shape == (1, 5)
    np.testing.assert_array_equal(result.n_left, [n_per_endpoint])
    np.testing.assert_array_equal(result.n_right, [n_per_endpoint])
    np.testing.assert_allclose(result.linear_fisher_raw, [1.0], atol=0.15)
    np.testing.assert_allclose(result.linear_fisher, result.linear_fisher_raw)


def test_cross_fitted_ole_uses_disjoint_adaptive_endpoint_groups() -> None:
    rng = np.random.default_rng(9)
    theta = np.linspace(-1.0, 1.0, 40, dtype=np.float64)
    x = np.column_stack([theta, theta**2]) + 0.1 * rng.normal(size=(40, 2))

    result = cross_fitted_ole_linear_fisher(
        theta,
        x,
        np.asarray([-1.0, -0.8]),
        n_splits=5,
        seed=4,
        window_radius=0.01,
        min_endpoint_samples=8,
    )

    np.testing.assert_array_equal(result.n_left, [8])
    np.testing.assert_array_equal(result.n_right, [8])
    np.testing.assert_array_equal(np.sum(result.fold_test_counts_left, axis=1), [8])
    np.testing.assert_array_equal(np.sum(result.fold_test_counts_right, axis=1), [8])
    assert np.all(np.isfinite(result.linear_fisher_raw))


def test_cross_fitted_ole_wraps_periodic_endpoint_windows() -> None:
    rng = np.random.default_rng(21)
    period = np.pi
    theta = np.concatenate(
        [
            np.linspace(0.0, 0.15, 20),
            np.linspace(period - 0.15, period - 0.001, 20),
            np.linspace(0.2, 0.4, 20),
        ]
    )
    x = np.column_stack([np.cos(2.0 * theta), np.sin(2.0 * theta)])
    x += 0.05 * rng.normal(size=x.shape)

    result = cross_fitted_ole_linear_fisher(
        theta,
        x,
        np.asarray([0.0, 0.2]),
        n_splits=5,
        seed=8,
        window_radius=0.1,
        min_endpoint_samples=10,
        period=period,
    )

    assert result.n_left[0] > 10
    assert result.n_right[0] >= 10
    assert np.isfinite(result.linear_fisher_raw[0])
