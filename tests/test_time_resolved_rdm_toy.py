from __future__ import annotations

import numpy as np

from fisher.time_resolved_rdm_toy import (
    TwoClassTimeResolvedGaussianToy,
    estimate_binned_correlation_distance,
    estimate_binned_metric_distance,
    gaussian_fid_distance,
    mean_vector_distance,
    population_distance_trajectory,
    squared_mahalanobis_distance,
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


def test_controlled_rotation_has_prescribed_correlation_distance() -> None:
    dataset = TwoClassTimeResolvedGaussianToy(
        x_dim=8,
        n_time_points=101,
        trajectory_mode="controlled_rotation",
        secondary_trajectory_scale=2.0,
        seed=17,
    )

    observed = dataset.true_correlation_distance()
    np.testing.assert_allclose(
        observed, dataset.target_correlation_distance, atol=2e-14, rtol=0.0
    )
    assert float(np.min(observed)) >= 0.05 - 1e-12
    assert float(np.max(observed)) > 0.5
    assert float(np.max(observed)) < 0.61
    assert dataset.class_means.shape == (2, 101, 8)


def test_controlled_rotation_is_reproducible() -> None:
    first = TwoClassTimeResolvedGaussianToy(
        x_dim=5, n_time_points=31, trajectory_mode="controlled_rotation", seed=23
    )
    second = TwoClassTimeResolvedGaussianToy(
        x_dim=5, n_time_points=31, trajectory_mode="controlled_rotation", seed=23
    )

    first_x, first_y = first.sample_trials(4)
    second_x, second_y = second.sample_trials(4)
    np.testing.assert_array_equal(first_y, second_y)
    np.testing.assert_allclose(first_x, second_x)
    np.testing.assert_allclose(first.class_means, second.class_means)


def test_sample_seed_changes_noise_but_not_population() -> None:
    dataset = TwoClassTimeResolvedGaussianToy(
        x_dim=5, n_time_points=21, trajectory_mode="controlled_rotation", seed=29
    )
    first_x, first_y = dataset.sample_trials(3, sample_seed=101)
    repeated_x, repeated_y = dataset.sample_trials(3, sample_seed=101)
    second_x, second_y = dataset.sample_trials(3, sample_seed=102)

    np.testing.assert_array_equal(first_y, repeated_y)
    np.testing.assert_allclose(first_x, repeated_x)
    assert not np.array_equal(first_x, second_x)
    assert first_y.shape == second_y.shape
    np.testing.assert_allclose(
        dataset.true_correlation_distance(), dataset.target_correlation_distance
    )


def test_other_metric_primitives_have_known_values() -> None:
    first = np.asarray([1.0, 0.0])
    second = np.asarray([0.0, 2.0])
    identity = np.eye(2)

    assert mean_vector_distance(first, second, "cosine") == 1.0
    assert mean_vector_distance(first, second, "euclidean") == np.sqrt(5.0)
    assert squared_mahalanobis_distance(first, second, identity, ridge=0.0) == 5.0
    assert gaussian_fid_distance(first, identity, second, identity) == 5.0


def test_population_fid_equals_squared_euclidean_for_shared_covariance() -> None:
    dataset = TwoClassTimeResolvedGaussianToy(
        x_dim=7,
        n_time_points=19,
        trajectory_mode="controlled_rotation",
        seed=31,
    )
    fid = population_distance_trajectory(
        dataset.class_means, dataset.shared_covariances, "fid"
    )
    np.testing.assert_allclose(fid, dataset.true_squared_euclidean_distance(), atol=1e-10)
    np.testing.assert_allclose(dataset.true_fid_distance(), fid, atol=1e-10)


def test_binned_other_metrics_are_finite_with_one_trial_per_class() -> None:
    dataset = TwoClassTimeResolvedGaussianToy(
        x_dim=6,
        n_time_points=25,
        trajectory_mode="controlled_rotation",
        seed=37,
    )
    responses, labels = dataset.sample_trials(1, sample_seed=41)
    for metric in ("cosine", "euclidean", "mahalanobis_sq", "fid"):
        result = estimate_binned_metric_distance(
            responses,
            labels,
            dataset.time,
            bin_width=1.0,
            metric=metric,
        )
        assert result["estimated_distance"].shape == (12,)
        assert np.all(np.isfinite(result["estimated_distance"]))
        assert np.all(result["estimated_distance"] >= 0.0)
