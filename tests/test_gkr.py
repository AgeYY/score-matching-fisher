from __future__ import annotations

import numpy as np
import torch

from fisher.gkr import (
    GKRConfig,
    PeriodicLogBandwidthKernelCovariance,
    TorchGKR,
    TorchKernelCovariance,
    estimate_gkr_linear_fisher,
    gaussian_residual_log_likelihood,
    restore_gkr_checkpoint,
)
from fisher.fisher_validation import gkr_checkpoint


def test_periodic_log_bandwidth_kernel_matches_equation_and_has_gradient() -> None:
    estimator = PeriodicLogBandwidthKernelCovariance(
        1,
        1,
        circular_period=np.pi,
        initial_lambda=0.25,
        jitter=0.0,
        dtype=torch.float64,
    )
    estimator.set_data(
        np.asarray([[1.0], [3.0]]),
        np.asarray([[0.0], [0.5 * np.pi]]),
    )
    with torch.no_grad():
        estimator.log_lambda.fill_(np.log(0.25))

    covariance = estimator(np.asarray([[0.0]]))
    expected_weight = np.exp(-1.0 / (0.25 + estimator.lambda_epsilon))
    expected = (1.0 + expected_weight * 9.0) / (1.0 + expected_weight)
    np.testing.assert_allclose(
        float(covariance.item()), expected, rtol=1e-12, atol=1e-12
    )

    covariance.sum().backward()
    assert estimator.log_lambda.grad is not None
    assert torch.isfinite(estimator.log_lambda.grad)
    assert float(estimator.log_lambda.grad.abs()) > 0.0


def test_periodic_log_bandwidth_initialization_targets_residual_ess() -> None:
    theta = np.linspace(0.0, np.pi, 200, endpoint=False)[:, None]
    phase = 2.0 * theta[:, 0]
    residuals = np.column_stack([np.sin(phase), np.cos(phase)])
    estimator = PeriodicLogBandwidthKernelCovariance(
        1,
        2,
        circular_period=np.pi,
        neighbors_per_effective_dimension=5.0,
        initialization_grid_size=128,
        jitter=0.0,
        dtype=torch.float64,
    )

    estimator.initialize_parameters(residuals, theta)

    np.testing.assert_allclose(
        estimator.residual_participation_ratio, 2.0, rtol=1e-12
    )
    np.testing.assert_allclose(
        estimator.target_effective_sample_size, 10.0, rtol=1e-12
    )
    np.testing.assert_allclose(
        estimator.initial_effective_sample_size,
        estimator.target_effective_sample_size,
        rtol=0.05,
    )
    assert 0.0 < estimator.initial_lambda < 0.01


def test_variational_gkr_supports_mean_minibatches() -> None:
    theta = np.linspace(-1.0, 1.0, 18, dtype=np.float64)[:, None]
    responses = np.concatenate([theta, theta**2], axis=1)
    model = TorchGKR(
        n_input=1,
        n_output=2,
        config=GKRConfig(
            mean_iterations=2,
            mean_batch_size=6,
            n_inducing=5,
            covariance_epochs=1,
            covariance_batch_size=18,
            prediction_batch_size=7,
            log_every=0,
        ),
        dtype=torch.float64,
        device="cpu",
        seed=3,
    )

    model.fit(responses, theta)
    mean, covariance = model.predict(theta)

    assert mean.shape == responses.shape
    assert covariance.shape == (18, 2, 2)
    assert np.isfinite(mean).all()
    assert np.isfinite(covariance).all()
    assert len(model.mean_loss_history) == 2


def test_restore_log_lambda_gkr_checkpoint_preserves_predictions() -> None:
    theta = np.linspace(0.0, np.pi, 18, endpoint=False)[:, None]
    responses = np.column_stack(
        [np.sin(2.0 * theta[:, 0]), np.cos(2.0 * theta[:, 0])]
    )
    config = GKRConfig(
        mean_iterations=2,
        n_inducing=5,
        covariance_epochs=1,
        covariance_batch_size=18,
        prediction_batch_size=18,
        log_every=0,
    )
    model = TorchGKR(
        n_input=1,
        n_output=2,
        circular_period=np.pi,
        config=config,
        dtype=torch.float64,
        device="cpu",
        seed=3,
    )
    model.covariance_model = PeriodicLogBandwidthKernelCovariance(
        1,
        2,
        circular_period=np.pi,
        initial_lambda=0.2,
        dtype=torch.float64,
    )
    model.fit(responses, theta)
    query = np.asarray([[0.2], [1.1]])
    expected_mean, expected_covariance = model.predict(query)
    checkpoint = gkr_checkpoint(model)
    checkpoint["covariance_kernel_metadata"] = {
        "configuration": {
            "neighbors_per_effective_dimension": 5.0,
            "lambda_epsilon": 1e-8,
            "initialization_grid_size": 256,
        },
        "initial_lambda": model.covariance_model.initial_lambda,
    }

    restored = restore_gkr_checkpoint(checkpoint)
    actual_mean, actual_covariance = restored.predict(query)

    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        actual_covariance, expected_covariance, rtol=1e-12, atol=1e-12
    )


def test_gaussian_residual_log_likelihood_matches_manual_identity_case() -> None:
    residuals = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    covariance = torch.eye(2, dtype=torch.float64).repeat(2, 1, 1)

    got = gaussian_residual_log_likelihood(residuals, covariance, jitter=0.0)

    expected = -0.5 * torch.tensor((1.0 + 4.0) / 2.0, dtype=torch.float64)
    torch.testing.assert_close(got, expected)


def test_kernel_covariance_returns_weighted_residual_outer_products() -> None:
    estimator = TorchKernelCovariance(1, 2, jitter=0.0, dtype=torch.float64)
    estimator.set_data(
        np.asarray([[1.0, 0.0], [0.0, 2.0]]),
        np.asarray([[0.0], [1.0]]),
    )
    with torch.no_grad():
        estimator.kernel_precision_cholesky.zero_()

    covariance = estimator(np.asarray([[0.5]])).detach().numpy()[0]

    np.testing.assert_allclose(covariance, np.diag([0.5, 2.0]), atol=1e-12)


class _KnownGaussianModel:
    n_input = 1
    n_output = 2
    covariance_loss_history = [1.0]
    mean_loss_history = [2.0]

    def predict(self, query):
        theta = np.asarray(query, dtype=np.float64).reshape(-1, 1)
        mean = np.concatenate([2.0 * theta, -3.0 * theta], axis=1)
        covariance = np.repeat(np.diag([4.0, 9.0])[None, :, :], theta.shape[0], axis=0)
        return mean, covariance


def test_estimate_gkr_linear_fisher_recovers_known_linear_gaussian() -> None:
    result = estimate_gkr_linear_fisher(
        _KnownGaussianModel(),
        np.asarray([[-0.5], [0.0], [0.5]]),
        finite_difference_step=1e-3,
        solve_jitter=0.0,
    )

    # 2^2 / 4 + (-3)^2 / 9 = 2.
    np.testing.assert_allclose(result.linear_fisher, 2.0, atol=1e-10)
    assert result.fisher_matrix.shape == (3, 1, 1)
    assert result.mean_jacobian.shape == (3, 2, 1)
    np.testing.assert_allclose(result.covariance_fisher, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.full_fisher, result.linear_fisher, atol=1e-12)


class _VaryingCovarianceModel:
    n_input = 1
    n_output = 1
    covariance_loss_history = []
    mean_loss_history = []

    def predict(self, query):
        theta = np.asarray(query, dtype=np.float64).reshape(-1, 1)
        mean = np.zeros_like(theta)
        covariance = np.exp(2.0 * theta)[:, :, None]
        return mean, covariance


def test_estimate_gkr_full_fisher_includes_covariance_variation() -> None:
    result = estimate_gkr_linear_fisher(
        _VaryingCovarianceModel(),
        np.asarray([[-0.5], [0.0], [0.5]]),
        finite_difference_step=1e-3,
        solve_jitter=0.0,
    )

    # For variance exp(2 theta), 0.5 * (d log variance / d theta)^2 = 2.
    np.testing.assert_allclose(result.linear_fisher, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.covariance_fisher, 2.0, rtol=1e-6)
    np.testing.assert_allclose(result.full_fisher, 2.0, rtol=1e-6)
    assert result.covariance_jacobian.shape == (3, 1, 1, 1)


class _QuadraticMeanModel:
    n_input = 1
    n_output = 1
    covariance_loss_history = []
    mean_loss_history = []

    def predict(self, query):
        theta = np.asarray(query, dtype=np.float64).reshape(-1, 1)
        mean = theta**2
        covariance = np.ones((theta.shape[0], 1, 1), dtype=np.float64)
        return mean, covariance


def test_estimate_gkr_uses_per_query_grid_separations() -> None:
    query = np.asarray([[-0.5], [0.25], [1.0]])
    separation = np.asarray([[0.2], [0.5], [0.8]])

    result = estimate_gkr_linear_fisher(
        _QuadraticMeanModel(),
        query,
        finite_difference_step=separation,
        solve_jitter=0.0,
    )

    # Centered differences recover d(theta^2)/dtheta = 2 theta for any separation.
    np.testing.assert_allclose(result.mean_jacobian[:, 0, 0], 2.0 * query[:, 0])
    np.testing.assert_allclose(result.linear_fisher, (2.0 * query[:, 0]) ** 2)
    np.testing.assert_allclose(result.metadata["finite_difference_step"], separation)
