from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from fisher.data import ToyConditionalGMMNonGaussianDataset, ToyConditionalGaussianRandampSqrtdDataset
from fisher.fisher_validation import (
    append_paired_probe,
    decoder_directions,
    evaluate_endpoint_decoders,
    evaluate_linear_threshold_decoders,
    finite_endpoint_oracle,
    fisher_predicted_linear_error,
    fit_cross_fitted_ole_direction_estimator,
    finite_grid_probe_increment,
    gaussian_mixture_moments,
    probe_mean,
    stratified_disjoint_subset_indices,
    stratified_train_validation_test_split,
)


def test_paired_probe_shares_noise_and_control_is_identity() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 3))
    theta = np.linspace(-2.0, 2.0, 40)
    noise = rng.normal(size=40)
    control, probe, returned_noise = append_paired_probe(
        x,
        theta,
        peak_fisher=4.0,
        omega=2.0,
        phase=0.3,
        seed=9,
        noise=noise,
    )
    expected_signal = probe_mean(theta, peak_fisher=4.0, omega=2.0, phase=0.3)
    np.testing.assert_allclose(control[:, :-1], x)
    np.testing.assert_allclose(probe[:, :-1], x)
    np.testing.assert_allclose(control[:, -1], returned_noise)
    np.testing.assert_allclose(probe[:, -1] - control[:, -1], expected_signal)


def test_finite_grid_increment_matches_direct_difference() -> None:
    left = np.linspace(-1.0, 0.8, 10)
    right = left + 0.2
    increment = finite_grid_probe_increment(
        left, right, peak_fisher=1.7, omega=0.9, phase=-0.2
    )
    direct = (
        (
            probe_mean(right, peak_fisher=1.7, omega=0.9, phase=-0.2)
            - probe_mean(left, peak_fisher=1.7, omega=0.9, phase=-0.2)
        )
        / (right - left)
    ) ** 2
    np.testing.assert_allclose(increment, direct)


def test_exact_mixture_moments_match_monte_carlo() -> None:
    dataset = ToyConditionalGMMNonGaussianDataset(x_dim=4, seed=8)
    theta = np.asarray([[0.7]])
    mean, _, covariance = gaussian_mixture_moments(dataset, theta)
    dataset.rng = np.random.default_rng(123)
    samples = dataset.sample_x(np.full((120_000, 1), theta.item()))
    np.testing.assert_allclose(np.mean(samples, axis=0), mean[0], atol=1.5e-2)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), covariance[0], atol=2.5e-2)


def test_decoder_evaluation_is_invariant_to_direction_scale() -> None:
    rng = np.random.default_rng(11)
    direction = np.asarray([[1.0, -0.4]])
    left = rng.normal(size=(1, 400, 2))
    right = left + np.asarray([[[0.3, -0.1]]])
    first = evaluate_endpoint_decoders(direction, left, right, np.asarray([0.2]))
    second = evaluate_endpoint_decoders(7.5 * direction, left, right, np.asarray([0.2]))
    np.testing.assert_allclose(first.achieved_fisher_raw, second.achieved_fisher_raw)
    np.testing.assert_allclose(first.roc_auc, second.roc_auc)


def test_matched_linear_threshold_uses_calibration_and_is_sign_invariant() -> None:
    calibration_left = np.asarray([[[-2.0], [-1.0], [-0.5]]])
    calibration_right = np.asarray([[[0.5], [1.0], [2.0]]])
    test_left = np.asarray([[[-3.0], [-0.25], [0.1]]])
    test_right = np.asarray([[[-0.1], [0.25], [3.0]]])
    positive = evaluate_linear_threshold_decoders(
        np.asarray([[1.0]]),
        calibration_left,
        calibration_right,
        test_left,
        test_right,
    )
    negative = evaluate_linear_threshold_decoders(
        np.asarray([[-1.0]]),
        calibration_left,
        calibration_right,
        test_left,
        test_right,
    )
    np.testing.assert_allclose(positive.balanced_error, np.asarray([1.0 / 3.0]))
    np.testing.assert_allclose(positive.balanced_error, negative.balanced_error)
    np.testing.assert_allclose(positive.false_positive_rate, np.asarray([1.0 / 3.0]))
    np.testing.assert_allclose(positive.false_negative_rate, np.asarray([1.0 / 3.0]))


def test_fisher_predicted_linear_error_matches_zero_and_known_separation() -> None:
    result = fisher_predicted_linear_error(
        np.asarray([0.0, 4.0]),
        np.asarray([0.2, 1.0]),
    )
    np.testing.assert_allclose(result[0], 0.5)
    np.testing.assert_allclose(result[1], 0.15865525393145707)


def test_stratified_split_is_disjoint_and_pca_is_fit_on_training_only() -> None:
    theta = np.linspace(0.0, np.pi, 480, endpoint=False)
    split = stratified_train_validation_test_split(
        theta,
        n_strata=16,
        train_fraction=0.64,
        validation_fraction=0.16,
        seed=7,
        period=np.pi,
    )
    assert not np.intersect1d(split.train, split.validation).size
    assert not np.intersect1d(split.train, split.test).size
    assert not np.intersect1d(split.validation, split.test).size
    np.testing.assert_array_equal(
        np.sort(np.concatenate([split.train, split.validation, split.test])), np.arange(theta.size)
    )
    x = np.column_stack([theta, theta**2, np.sin(theta)])
    pca = PCA(n_components=2).fit(x[split.train])
    np.testing.assert_allclose(pca.mean_, np.mean(x[split.train], axis=0))
    assert not np.allclose(pca.mean_, np.mean(x, axis=0))


def test_stratified_split_keeps_test_indices_fixed_across_validation_ratios() -> None:
    theta = np.linspace(0.0, np.pi, 800, endpoint=False)
    splits = [
        stratified_train_validation_test_split(
            theta,
            n_strata=16,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            seed=23,
            period=np.pi,
        )
        for train_fraction, validation_fraction in (
            (0.72, 0.08),
            (0.64, 0.16),
            (0.56, 0.24),
            (0.48, 0.32),
        )
    ]
    for split in splits[1:]:
        np.testing.assert_array_equal(split.test, splits[0].test)
    assert set(splits[0].validation).issubset(set(splits[-1].validation))


def test_stratified_split_keeps_validation_fixed_across_test_ratios() -> None:
    theta = np.linspace(0.0, np.pi, 800, endpoint=False)
    validation_fraction = 0.16
    splits = [
        stratified_train_validation_test_split(
            theta,
            n_strata=16,
            train_fraction=1.0 - validation_fraction - test_fraction,
            validation_fraction=validation_fraction,
            seed=29,
            period=np.pi,
            fixed_partition="validation",
        )
        for test_fraction in (0.1, 0.2, 0.3, 0.4, 0.5)
    ]
    for split in splits[1:]:
        np.testing.assert_array_equal(split.validation, splits[0].validation)
    for smaller, larger in zip(splits[:-1], splits[1:], strict=True):
        assert set(smaller.test).issubset(set(larger.test))
        assert set(larger.train).issubset(set(smaller.train))


def test_stratified_disjoint_subsets_are_exact_disjoint_and_balanced() -> None:
    theta = (np.arange(160, dtype=np.float64) + 0.5) * np.pi / 160.0
    subsets = stratified_disjoint_subset_indices(
        theta,
        32,
        n_subsets=5,
        n_strata=16,
        seed=37,
        period=np.pi,
    )

    assert [subset.size for subset in subsets] == [32] * 5
    assert np.unique(np.concatenate(subsets)).size == 160
    for subset in subsets:
        assert np.unique(subset).size == subset.size
        bins = np.floor(theta[subset] / np.pi * 16).astype(np.int64)
        counts = np.bincount(bins, minlength=16)
        assert int(np.max(counts) - np.min(counts)) <= 1


def test_stratified_disjoint_subsets_reject_invalid_sizes() -> None:
    theta = np.linspace(0.0, 1.0, 20)
    for size, count in ((0, 1), (1, 0), (21, 1), (5, 5)):
        try:
            stratified_disjoint_subset_indices(
                theta,
                size,
                n_subsets=count,
                n_strata=2,
                seed=0,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid size/count {(size, count)} to fail.")


def test_toy_oracle_direction_recovers_finite_pair_information() -> None:
    dataset = ToyConditionalGaussianRandampSqrtdDataset(x_dim=5, seed=5)
    left_theta = np.asarray([[-0.2]])
    right_theta = np.asarray([[0.2]])
    mean_left = dataset.tuning_curve(left_theta)
    mean_right = dataset.tuning_curve(right_theta)
    covariance_left = dataset.covariance(left_theta)
    covariance_right = dataset.covariance(right_theta)
    direction, information = finite_endpoint_oracle(
        mean_left,
        mean_right,
        covariance_left,
        covariance_right,
        np.asarray([0.4]),
    )
    covariance = 0.5 * (covariance_left + covariance_right)
    expected = float(information[0])
    dataset.rng = np.random.default_rng(17)
    left = dataset.sample_x(np.full((30_000, 1), -0.2))[None]
    right = dataset.sample_x(np.full((30_000, 1), 0.2))[None]
    achieved = evaluate_endpoint_decoders(direction, left, right, np.asarray([0.4]))
    np.testing.assert_allclose(achieved.achieved_fisher_raw[0], expected, rtol=0.08)

    rng = np.random.default_rng(29)
    candidate = rng.normal(size=(512, mean_left.shape[1]))
    signal = (mean_right - mean_left)[0]
    candidate_information = (
        (candidate @ signal) ** 2
        / (0.4**2 * np.einsum("nd,df,nf->n", candidate, covariance[0], candidate))
    )
    assert np.max(candidate_information) <= expected * (1.0 + 1e-10)


def test_cross_fitted_ole_direction_can_be_scored_on_external_endpoints() -> None:
    rng = np.random.default_rng(33)
    theta = rng.uniform(-0.5, 0.5, size=800)
    x = np.column_stack([theta, -0.4 * theta, np.zeros_like(theta)])
    x += 0.3 * rng.normal(size=x.shape)
    _, direction = fit_cross_fitted_ole_direction_estimator(
        theta_train=theta,
        x_train=x,
        theta_grid=np.asarray([[-0.2], [0.2]]),
        n_splits=5,
        seed=9,
        min_endpoint_samples=20,
    )
    left = rng.normal(size=(1, 2000, 3)) * 0.3
    right = rng.normal(size=(1, 2000, 3)) * 0.3
    left += np.asarray([[[-0.2, 0.08, 0.0]]])
    right += np.asarray([[[0.2, -0.08, 0.0]]])
    achieved = evaluate_endpoint_decoders(direction, left, right, np.asarray([0.4]))

    assert direction.shape == (1, 3)
    assert achieved.achieved_fisher_raw[0] > 0.0


def test_cross_fitted_ole_ensemble_is_invariant_to_fold_weight_scale() -> None:
    rng = np.random.default_rng(41)
    theta = np.repeat([-0.2, 0.2], 200)
    x = rng.normal(scale=0.5, size=(400, 3))
    x[theta > 0.0] += np.asarray([0.4, -0.1, 0.2])

    result, direction = fit_cross_fitted_ole_direction_estimator(
        theta_train=theta,
        x_train=x,
        theta_grid=np.asarray([-0.2, 0.2]),
        n_splits=5,
        seed=3,
    )
    fold_weights = np.asarray(result.fold_weights[0])
    expected = np.mean(
        fold_weights / np.linalg.norm(fold_weights, axis=1, keepdims=True),
        axis=0,
    )
    expected /= np.linalg.norm(expected)

    np.testing.assert_allclose(direction[0], expected)
