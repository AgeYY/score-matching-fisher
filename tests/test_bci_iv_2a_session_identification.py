from __future__ import annotations

import numpy as np
import torch

from fisher.bci_iv_2a_dataset import _window_starts
from fisher.bci_iv_2a_session_identification import (
    _time_conditioned_endpoint_covariances,
    classical_mahalanobis_rdms,
    condition_design,
    pearson_similarity,
    subsample_balanced_trials,
    vectorize_rdms,
)
from fisher.flow_matching_skl import build_flow_skl_model


def test_feature_window_grid_has_expected_centers() -> None:
    starts = _window_starts(-2.0, 4.0, 1.0, 0.25)
    assert starts.shape == (21,)
    np.testing.assert_allclose(starts[[0, -1]], [-2.0, 3.0])


def test_condition_design_has_four_class_columns_and_scaled_time() -> None:
    design = condition_design(np.array([0, 3]), np.array([-2.0, 4.0]))
    assert design.shape == (2, 5)
    np.testing.assert_allclose(design[:, :4], [[1, 0, 0, 0], [0, 0, 0, 1]])
    np.testing.assert_allclose(design[:, 4], [-0.5, 1.0])


def test_balanced_subsample_is_reproducible() -> None:
    labels = np.repeat(np.arange(4), 7)
    first = subsample_balanced_trials(labels, 4, seed=9)
    second = subsample_balanced_trials(labels, 4, seed=9)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.bincount(labels[first], minlength=4), [4, 4, 4, 4])


def test_classical_rdms_are_symmetric_and_detect_separation() -> None:
    rng = np.random.default_rng(3)
    labels = np.repeat(np.arange(4), 10)
    x = rng.normal(size=(40, 3, 5))
    x += labels[:, None, None] * 0.7
    rdms = classical_mahalanobis_rdms(x, labels)
    assert rdms.shape == (3, 4, 4)
    np.testing.assert_allclose(rdms, rdms.transpose(0, 2, 1))
    np.testing.assert_allclose(np.diagonal(rdms, axis1=1, axis2=2), 0.0)
    assert np.all(rdms[:, 0, 3] > 0.0)


def test_vectorization_and_correlation() -> None:
    rdms = np.zeros((2, 4, 4), dtype=np.float64)
    upper = np.triu_indices(4, k=1)
    rdms[0][upper] = np.arange(1, 7)
    rdms[1][upper] = np.arange(7, 13)
    rdms += rdms.transpose(0, 2, 1)
    vector = vectorize_rdms(rdms, np.array([0.0, 1.0]), interval=(0.0, 1.0))
    np.testing.assert_array_equal(vector, np.arange(1, 13))
    assert pearson_similarity(vector, vector) == 1.0


def test_covariate_affine_matrix_is_class_shared_at_fixed_eeg_time() -> None:
    model = build_flow_skl_model(
        velocity_family="covariate_affine",
        theta_dim=5,
        x_dim=3,
        hidden_dim=8,
        depth=1,
        affine_condition_indices=(4,),
        divergence_estimator="exact",
    )
    theta = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.25],
            [0.0, 0.0, 0.0, 1.0, 0.25],
        ]
    )
    matrices = model.A(theta, torch.full((2, 1), 0.4)).detach().numpy()
    np.testing.assert_allclose(matrices[0], matrices[1], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(matrices, matrices.transpose(0, 2, 1))


def test_time_conditioned_covariance_integrator_returns_one_spd_matrix_per_time() -> None:
    model = build_flow_skl_model(
        velocity_family="covariate_affine",
        theta_dim=5,
        x_dim=3,
        hidden_dim=8,
        depth=1,
        affine_condition_indices=(4,),
        divergence_estimator="exact",
    )
    times = np.array([-1.0, 0.0, 1.0])
    conditions = condition_design(np.zeros(3, dtype=np.int64), times)
    covariances = _time_conditioned_endpoint_covariances(
        model,
        conditions,
        device=torch.device("cpu"),
        steps=4,
        ridge=1e-5,
    )
    assert covariances.shape == (3, 3, 3)
    np.testing.assert_allclose(covariances, covariances.transpose(0, 2, 1))
    assert np.all(np.linalg.eigvalsh(covariances) > 0.0)
