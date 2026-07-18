from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.data import ToyConditionalGaussianRandampSqrtdTwoTrajectoryDataset


def _dataset(**kwargs: float) -> ToyConditionalGaussianRandampSqrtdTwoTrajectoryDataset:
    return ToyConditionalGaussianRandampSqrtdTwoTrajectoryDataset(
        theta_low=-2.0,
        theta_high=2.0,
        x_dim=3,
        seed=17,
        randamp_mu_amp_per_dim=np.asarray([0.6, 1.0, 1.4], dtype=np.float64),
        **kwargs,
    )


def test_component_means_are_base_and_twice_base() -> None:
    dataset = _dataset()
    theta = np.asarray([[-1.0], [0.25], [1.5]], dtype=np.float64)
    means = dataset.component_means(theta)

    np.testing.assert_allclose(means[:, 0], dataset.base_trajectory(theta))
    np.testing.assert_allclose(means[:, 1], 2.0 * dataset.base_trajectory(theta))
    np.testing.assert_allclose(dataset.tuning_curve(theta), 0.5 * (means[:, 0] + means[:, 1]))


def test_sampling_uses_equal_component_probabilities() -> None:
    dataset = _dataset()
    theta = np.zeros((20_000, 1), dtype=np.float64)
    x, component = dataset.sample_x_with_component(theta)

    assert x.shape == (20_000, 3)
    assert component.shape == (20_000,)
    assert abs(float(component.mean()) - 0.5) < 0.015


def test_total_covariance_includes_between_trajectory_variance() -> None:
    dataset = _dataset()
    theta = np.asarray([[-0.8], [0.4]], dtype=np.float64)
    mu = dataset.base_trajectory(theta)
    expected = dataset.component_covariance(theta) + 0.25 * np.einsum("ni,nj->nij", mu, mu)

    np.testing.assert_allclose(dataset.covariance(theta), expected, rtol=1e-12, atol=1e-12)


def test_total_covariance_derivative_matches_finite_difference() -> None:
    dataset = _dataset()
    theta = np.asarray([[-0.8], [0.4]], dtype=np.float64)
    step = 1e-5
    finite_difference = (
        dataset.covariance(theta + step) - dataset.covariance(theta - step)
    ) / (2.0 * step)

    np.testing.assert_allclose(
        dataset.covariance_derivative(theta),
        finite_difference,
        rtol=2e-8,
        atol=2e-9,
    )


def test_log_density_is_equal_weight_gaussian_mixture() -> None:
    dataset = _dataset()
    theta = np.asarray([[-0.5], [0.75]], dtype=np.float64)
    means = dataset.component_means(theta)
    x = means[:, 0] + np.asarray([[0.1, -0.2, 0.05], [-0.1, 0.0, 0.2]], dtype=np.float64)
    variance = dataset._component_variance_diag(theta)
    logdet = np.sum(np.log(variance), axis=1)
    normalizer = dataset.x_dim * np.log(2.0 * np.pi) + logdet
    component_logs = []
    for component in range(2):
        quadratic = np.sum(((x - means[:, component]) ** 2) / variance, axis=1)
        component_logs.append(-0.5 * (normalizer + quadratic))
    expected = np.logaddexp(component_logs[0], component_logs[1]) - np.log(2.0)

    np.testing.assert_allclose(dataset.log_p_x_given_theta(x, theta), expected, rtol=1e-12, atol=1e-12)


def test_theta_score_matches_log_density_finite_difference() -> None:
    dataset = _dataset()
    theta = np.asarray([[-0.5], [0.75]], dtype=np.float64)
    x = dataset.component_means(theta)[:, 0] + np.asarray(
        [[0.1, -0.2, 0.05], [-0.1, 0.0, 0.2]], dtype=np.float64
    )
    step = 1e-5
    finite_difference = (
        dataset.log_p_x_given_theta(x, theta + step)
        - dataset.log_p_x_given_theta(x, theta - step)
    ) / (2.0 * step)

    np.testing.assert_allclose(dataset.theta_score(x, theta), finite_difference, rtol=2e-7, atol=2e-8)


def test_invalid_mixture_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="secondary_trajectory_scale"):
        _dataset(secondary_trajectory_scale=0.0)
    with pytest.raises(ValueError, match="secondary_trajectory_probability"):
        _dataset(secondary_trajectory_probability=1.0)
