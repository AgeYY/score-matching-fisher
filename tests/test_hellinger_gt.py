import numpy as np

from fisher.data import ToyConditionalGaussianDataset, ToyConditionalGaussianRandamp2DSqrtdDataset
from fisher.hellinger_gt import (
    estimate_hellinger_sq_grid_centers_analytic,
    hellinger_sq_gaussian_diag,
    hellinger_sq_gaussian_full,
    theta_centers_for_analytic_gt,
)


def test_diag_same_covariance_matches_closed_form() -> None:
    mu1 = np.array([0.0, 1.0, -2.0], dtype=np.float64)
    mu2 = np.array([1.0, -1.0, 2.0], dtype=np.float64)
    var = np.array([2.0, 4.0, 8.0], dtype=np.float64)

    got = hellinger_sq_gaussian_diag(mu1, var, mu2, var)
    expected = 1.0 - np.exp(-0.125 * np.sum(((mu1 - mu2) ** 2) / var))

    assert np.isclose(got, expected)


def test_diag_different_covariances_matches_coordinate_formula() -> None:
    mu1 = np.array([0.0, 2.0], dtype=np.float64)
    mu2 = np.array([1.0, -1.0], dtype=np.float64)
    var1 = np.array([1.0, 3.0], dtype=np.float64)
    var2 = np.array([2.0, 5.0], dtype=np.float64)
    vbar = 0.5 * (var1 + var2)

    db = np.sum(((mu1 - mu2) ** 2) / (8.0 * vbar) + 0.5 * np.log(vbar / np.sqrt(var1 * var2)))
    expected = 1.0 - np.exp(-db)

    assert np.isclose(hellinger_sq_gaussian_diag(mu1, var1, mu2, var2), expected)


def test_full_covariance_matches_diagonal_helper_for_diagonal_covariances() -> None:
    mu1 = np.array([0.0, 1.0], dtype=np.float64)
    mu2 = np.array([2.0, -1.0], dtype=np.float64)
    var1 = np.array([1.5, 2.5], dtype=np.float64)
    var2 = np.array([2.0, 4.0], dtype=np.float64)

    got_full = hellinger_sq_gaussian_full(mu1, np.diag(var1), mu2, np.diag(var2))
    got_diag = hellinger_sq_gaussian_diag(mu1, var1, mu2, var2)

    assert np.isclose(got_full, got_diag)


def test_analytic_grid_centers_matrix_properties() -> None:
    dataset = ToyConditionalGaussianDataset(theta_low=-2.0, theta_high=2.0, x_dim=4, seed=11)
    centers = np.linspace(-1.5, 1.5, 5, dtype=np.float64).reshape(-1, 1)

    h2 = estimate_hellinger_sq_grid_centers_analytic(dataset, centers, symmetrize=True)

    assert h2.shape == (5, 5)
    assert np.all(np.isfinite(h2))
    assert np.all(h2 >= 0.0)
    assert np.all(h2 <= 1.0)
    assert np.allclose(np.diag(h2), 0.0)
    assert np.allclose(h2, h2.T)


def test_theta_centers_for_analytic_gt_uses_midpoint_for_2d_theta1_centers() -> None:
    dataset = ToyConditionalGaussianRandamp2DSqrtdDataset(
        theta_low=-4.0,
        theta_high=6.0,
        x_dim=3,
        seed=5,
    )
    theta1_centers = np.array([-2.0, 0.0, 2.0], dtype=np.float64)

    centers = theta_centers_for_analytic_gt(dataset, theta1_centers)

    assert centers.shape == (3, 2)
    assert np.allclose(centers[:, 0], theta1_centers)
    assert np.allclose(centers[:, 1], 1.0)


def test_theta_centers_for_analytic_gt_preserves_explicit_2d_grid_centers() -> None:
    dataset = ToyConditionalGaussianRandamp2DSqrtdDataset(x_dim=3, seed=5)
    explicit = np.array([[-1.0, -2.0], [0.0, 1.0], [3.0, 4.0]], dtype=np.float64)

    centers = theta_centers_for_analytic_gt(dataset, explicit)

    assert np.array_equal(centers, explicit)
