from __future__ import annotations

import numpy as np

from fisher.stringer_nongaussian_surrogate import (
    PeriodicFourierMoments,
    fit_periodic_fourier_moments,
    fit_standardized_residual_bank,
    periodic_fourier_derivative_features,
    periodic_fourier_features,
    sample_moment_matched_surrogate,
)


def _synthetic_periodic_data(seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, np.pi, size=2400)
    phase = 2.0 * theta
    mean = np.stack(
        [
            0.8 * np.cos(phase) + 0.2 * np.sin(2.0 * phase),
            -0.4 * np.sin(phase),
            0.5 * np.cos(2.0 * phase),
        ],
        axis=1,
    )
    scale = 0.35 + 0.05 * np.cos(phase)
    noise = rng.standard_t(df=7.0, size=mean.shape) * scale[:, None]
    return theta, mean + noise


def test_periodic_feature_derivative_matches_finite_difference() -> None:
    theta = np.linspace(0.1, 2.9, 11)
    step = 1e-6
    finite = (
        periodic_fourier_features(theta + step, period=np.pi, n_harmonics=4)
        - periodic_fourier_features(theta - step, period=np.pi, n_harmonics=4)
    ) / (2.0 * step)
    analytic = periodic_fourier_derivative_features(
        theta, period=np.pi, n_harmonics=4
    )
    np.testing.assert_allclose(analytic, finite, rtol=1e-7, atol=1e-7)


def test_linear_fisher_uses_batched_covariance_solves() -> None:
    moments = PeriodicFourierMoments(
        period=np.pi,
        n_harmonics=1,
        mean_coefficients=np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 2.0],
            ]
        ),
        covariance_grid_centers=np.asarray([np.pi / 2.0]),
        covariance_grid=np.asarray([np.diag([2.0, 4.0])]),
    )
    theta = np.asarray([0.0, np.pi / 4.0])
    derivative = moments.mean_derivative(theta)
    expected = np.sum(derivative**2 / np.asarray([2.0, 4.0]), axis=1)
    np.testing.assert_allclose(
        moments.linear_fisher(theta, solve_jitter=0.0), expected
    )


def test_standardized_residual_bank_has_binwise_identity_covariance() -> None:
    theta, responses = _synthetic_periodic_data()
    moments = fit_periodic_fourier_moments(
        theta,
        responses,
        period=np.pi,
        n_harmonics=3,
        covariance_grid_size=16,
    )
    bank = fit_standardized_residual_bank(
        theta, responses, moments, n_bins=8
    )
    assert bank.residuals.shape == responses.shape
    assert np.all(bank.counts > responses.shape[1])
    assert float(np.max(bank.mean_norms)) < 1e-10
    assert float(np.max(bank.covariance_errors)) < 1e-8


def test_conditional_covariance_shrinkage_avoids_near_singular_grids() -> None:
    theta, responses = _synthetic_periodic_data()
    moments = fit_periodic_fourier_moments(
        theta,
        responses,
        period=np.pi,
        n_harmonics=4,
        covariance_grid_size=16,
        covariance_shrinkage=0.25,
    )
    global_covariance = np.cov(responses - moments.mean(theta), rowvar=False)
    lower_bound = 0.25 * float(np.min(np.linalg.eigvalsh(global_covariance)))
    assert float(np.min(np.linalg.eigvalsh(moments.covariance_grid))) >= 0.99 * lower_bound


def test_surrogate_endpoints_preserve_fitted_moments() -> None:
    theta, responses = _synthetic_periodic_data()
    moments = fit_periodic_fourier_moments(
        theta,
        responses,
        period=np.pi,
        n_harmonics=3,
        covariance_grid_size=8,
    )
    bank = fit_standardized_residual_bank(theta, responses, moments, n_bins=8)
    rng = np.random.default_rng(12)
    sample_theta = np.repeat(moments.covariance_grid_centers, 2500)
    rng.shuffle(sample_theta)
    for weight in (0.0, 1.0):
        sample = sample_moment_matched_surrogate(
            sample_theta,
            moments,
            bank,
            non_gaussian_weight=weight,
            seed=22,
        )
        centered = sample - moments.mean(sample_theta)
        grid_ids = moments.covariance_indices(sample_theta)
        for grid_index in range(moments.covariance_grid.shape[0]):
            one = centered[grid_ids == grid_index]
            empirical_covariance = one.T @ one / float(one.shape[0])
            np.testing.assert_allclose(
                one.mean(axis=0), np.zeros(moments.x_dim), atol=0.035
            )
            np.testing.assert_allclose(
                empirical_covariance,
                moments.covariance_grid[grid_index],
                rtol=0.12,
                atol=0.025,
            )
