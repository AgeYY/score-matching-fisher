from __future__ import annotations

import numpy as np

from fisher.time_resolved_rdm_toy import (
    TwoClassTimeResolvedGaussianToy,
    estimate_binned_correlation_distance,
)


def test_two_class_time_resolved_toy_shapes_and_truth() -> None:
    dataset = TwoClassTimeResolvedGaussianToy(
        x_dim=6,
        n_time_points=17,
        secondary_trajectory_scale=2.0,
        covariance_alpha=0.65,
        seed=7,
    )
    responses, labels = dataset.sample_trials(5)

    assert responses.shape == (10, 17, 6)
    assert labels.shape == (10,)
    np.testing.assert_array_equal(np.bincount(labels), np.asarray([5, 5]))
    np.testing.assert_allclose(dataset.class_means[1], 2.0 * dataset.class_means[0])
    assert dataset.shared_covariances.shape == (17, 6, 6)
    assert np.all(dataset.true_squared_euclidean_distance() >= 0.0)
    assert np.all(dataset.true_squared_mahalanobis_distance() >= 0.0)


def test_two_class_time_resolved_toy_is_reproducible() -> None:
    first = TwoClassTimeResolvedGaussianToy(x_dim=4, n_time_points=9, seed=11)
    second = TwoClassTimeResolvedGaussianToy(x_dim=4, n_time_points=9, seed=11)

    first_x, first_y = first.sample_trials(3)
    second_x, second_y = second.sample_trials(3)
    np.testing.assert_array_equal(first_y, second_y)
    np.testing.assert_allclose(first_x, second_x)
    np.testing.assert_allclose(first.class_means, second.class_means)


def test_binned_correlation_truth_is_zero_for_scaled_trajectory() -> None:
    dataset = TwoClassTimeResolvedGaussianToy(
        x_dim=6, n_time_points=25, secondary_trajectory_scale=2.0, seed=13
    )
    responses, labels = dataset.sample_trials(20)
    result = estimate_binned_correlation_distance(
        responses,
        labels,
        dataset.time,
        bin_width=1.0,
        true_class_means=dataset.class_means,
    )

    assert result["estimated_correlation_distance"].shape == (12,)
    assert np.all(result["estimated_correlation_distance"] >= 0.0)
    np.testing.assert_allclose(
        result["true_correlation_distance"], 0.0, atol=1e-14, rtol=0.0
    )
