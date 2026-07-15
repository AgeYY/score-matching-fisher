from __future__ import annotations

import numpy as np
import torch

from fisher.bci_iv_2a_temporal_rdm import (
    classical_temporal_rdms,
    fit_temporal_gaussians,
    gaussian_fid_matrix_batched,
    time_conditioned_affine_endpoint_moments,
)
from fisher.distance_comparison import gaussian_fid_matrix


def test_temporal_rdms_have_expected_shape_and_geometry() -> None:
    rng = np.random.default_rng(11)
    samples = rng.normal(size=(12, 5, 4))
    samples[:, :, 0] += np.arange(5)[None, :] * 0.5
    result = classical_temporal_rdms(samples)

    assert tuple(result.rdms) == ("correlation", "cosine", "euclidean", "fid")
    assert result.means.shape == (5, 4)
    assert result.covariances.shape == (5, 4, 4)
    for matrix in result.rdms.values():
        assert matrix.shape == (5, 5)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-12)
        assert np.all(matrix >= 0.0)


def test_euclidean_temporal_rdm_uses_timewise_trial_means() -> None:
    samples = np.array(
        [
            [[1.0, 0.0], [0.0, 2.0]],
            [[3.0, 0.0], [0.0, 4.0]],
        ]
    )
    result = classical_temporal_rdms(samples)
    expected = np.sqrt((2.0 - 0.0) ** 2 + (0.0 - 3.0) ** 2)
    np.testing.assert_allclose(result.rdms["euclidean"], [[0.0, expected], [expected, 0.0]])


def test_temporal_gaussians_fit_a_distinct_covariance_per_time() -> None:
    samples = np.array(
        [
            [[-1.0, 0.0], [-4.0, 0.0]],
            [[0.0, -1.0], [0.0, -2.0]],
            [[1.0, 0.0], [4.0, 0.0]],
            [[0.0, 1.0], [0.0, 2.0]],
        ]
    )
    _, covariances = fit_temporal_gaussians(samples)
    assert not np.allclose(covariances[0], covariances[1])


def test_gaussian_fid_includes_covariance_difference() -> None:
    means = np.zeros((2, 1), dtype=np.float64)
    covariances = np.array([[[1.0]], [[9.0]]], dtype=np.float64)
    fid = gaussian_fid_matrix(means, covariances)
    np.testing.assert_allclose(fid, [[0.0, 4.0], [4.0, 0.0]], atol=1e-12)


def test_batched_gaussian_fid_matches_reference_full_covariance() -> None:
    rng = np.random.default_rng(17)
    means = rng.normal(size=(7, 3))
    factors = rng.normal(size=(7, 3, 3))
    covariances = factors @ factors.transpose(0, 2, 1) + 0.2 * np.eye(3)[None]
    expected = gaussian_fid_matrix(means, covariances)
    got = gaussian_fid_matrix_batched(
        means,
        covariances,
        device=torch.device("cpu"),
        block_size=3,
    )
    np.testing.assert_allclose(got, expected, atol=1e-9, rtol=1e-9)


def test_time_conditioned_affine_moments_keep_identity_for_zero_matrix() -> None:
    class ZeroAffineModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.x_dim = 2
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def endpoint_mean(self, theta: torch.Tensor) -> torch.Tensor:
            return torch.cat([theta, -theta], dim=1) + self.anchor

        def A(self, theta: torch.Tensor, flow_time: torch.Tensor) -> torch.Tensor:
            del flow_time
            return torch.zeros(theta.shape[0], 2, 2, dtype=theta.dtype, device=theta.device) + self.anchor

    conditions = np.array([[-1.0], [0.0], [1.0]])
    means, covariances = time_conditioned_affine_endpoint_moments(
        ZeroAffineModel(),
        conditions,
        device=torch.device("cpu"),
        covariance_steps=4,
        covariance_ridge=1e-5,
    )
    np.testing.assert_allclose(means, [[-1.0, 1.0], [0.0, 0.0], [1.0, -1.0]])
    np.testing.assert_allclose(covariances, np.broadcast_to(np.eye(2), (3, 2, 2)))
