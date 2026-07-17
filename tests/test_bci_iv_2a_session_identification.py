from __future__ import annotations

import numpy as np
import torch

from global_setting import EARLY_STOPPING_PATIENCE, TRAINING_MAX_EPOCHS
from fisher.bci_iv_2a_dataset import (
    _window_starts,
    native_voltage_sample_times,
    voltage_sample_times,
)
from fisher.bci_iv_2a_session_identification import (
    FlowRDMConfig,
    RDM_MATCHING_INTERVAL,
    _time_conditioned_endpoint_covariances,
    classical_fid_rdms,
    classical_squared_euclidean_rdms,
    classical_mahalanobis_rdms,
    condition_affine_flow_fid_rdms,
    condition_affine_flow_fid_rdms_from_checkpoint,
    condition_design,
    empirical_condition_gaussian_components,
    empirical_condition_means,
    empirical_gaussian_components,
    gaussian_fid_rdms_from_moments,
    gaussian_jeffreys_rdms_from_moments,
    load_condition_affine_flow_checkpoint,
    load_translation_flow_checkpoint,
    pearson_similarity,
    rdms_from_means_and_precisions,
    squared_euclidean_rdms_from_means,
    stratified_mixed_half_split,
    subsample_balanced_trials,
    translation_flow_squared_euclidean_rdms,
    vectorize_rdms,
)
from fisher.flow_matching_skl import build_flow_skl_model


def test_feature_window_grid_has_expected_centers() -> None:
    starts = _window_starts(-2.0, 4.0, 1.0, 0.25)
    assert starts.shape == (21,)
    np.testing.assert_allclose(starts[[0, -1]], [-2.0, 3.0])


def test_voltage_sample_grid_matches_prior_feature_centers() -> None:
    times = voltage_sample_times()
    assert times.shape == (21,)
    np.testing.assert_allclose(times[[0, -1]], [-1.5, 3.5])


def test_native_voltage_sample_grid_keeps_every_250hz_sample() -> None:
    times = native_voltage_sample_times(250.0, -1.5, 3.5)
    assert times.shape == (1251,)
    np.testing.assert_allclose(times[[0, -1]], [-1.5, 3.5])
    np.testing.assert_allclose(np.diff(times), 1.0 / 250.0)


def test_rdm_matching_interval_runs_from_cue_to_trial_end() -> None:
    assert RDM_MATCHING_INTERVAL == (0.0, 3.5)


def test_eeg_flow_keeps_flattened_data_on_device_by_default() -> None:
    assert FlowRDMConfig().device_resident_data is True


def test_eeg_flow_uses_global_training_budget_by_default() -> None:
    config = FlowRDMConfig()
    assert TRAINING_MAX_EPOCHS == 20_000
    assert EARLY_STOPPING_PATIENCE == 1_000
    assert config.epochs == TRAINING_MAX_EPOCHS
    assert config.patience == EARLY_STOPPING_PATIENCE


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


def test_stratified_mixed_half_split_uses_every_trial_once() -> None:
    labels = np.repeat(np.arange(4), [7, 8, 9, 10])
    reference, query = stratified_mixed_half_split(labels, seed=17)
    np.testing.assert_array_equal(
        np.sort(np.concatenate([reference, query])),
        np.arange(labels.size),
    )
    assert np.intersect1d(reference, query).size == 0
    reference_counts = np.bincount(labels[reference], minlength=4)
    query_counts = np.bincount(labels[query], minlength=4)
    assert np.all(reference_counts >= query_counts)
    assert np.all(reference_counts - query_counts <= 1)


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


def test_classical_rdms_support_unstandardized_voltage() -> None:
    rng = np.random.default_rng(31)
    labels = np.repeat(np.arange(4), 8)
    x = rng.normal(scale=0.5, size=(32, 2, 4))
    x += labels[:, None, None] * 0.2
    rdms = classical_mahalanobis_rdms(x, labels, standardize_features=False)
    assert rdms.shape == (2, 4, 4)
    assert np.all(np.isfinite(rdms))
    np.testing.assert_allclose(rdms, rdms.transpose(0, 2, 1))


def test_classical_rdms_equal_reconstructed_empirical_components() -> None:
    rng = np.random.default_rng(37)
    labels = np.repeat(np.arange(4), 9)
    x = rng.normal(size=(36, 4, 3))
    means, _, precisions = empirical_gaussian_components(
        x,
        labels,
        standardize_features=False,
    )
    reconstructed = rdms_from_means_and_precisions(means, precisions)
    direct = classical_mahalanobis_rdms(
        x,
        labels,
        standardize_features=False,
    )
    np.testing.assert_allclose(reconstructed, direct, atol=0.0, rtol=0.0)


def test_classical_squared_euclidean_rdms_use_condition_sample_means() -> None:
    labels = np.repeat(np.arange(4), 3)
    x = np.zeros((12, 2, 3), dtype=np.float64)
    x += labels[:, None, None]
    means = empirical_condition_means(x, labels, standardize_features=False)
    rdms = classical_squared_euclidean_rdms(
        x,
        labels,
        standardize_features=False,
    )
    reconstructed = squared_euclidean_rdms_from_means(means)
    assert means.shape == (2, 4, 3)
    np.testing.assert_allclose(rdms, reconstructed)
    np.testing.assert_allclose(rdms[:, 0, 3], 27.0)
    np.testing.assert_allclose(rdms, rdms.transpose(0, 2, 1))
    np.testing.assert_allclose(np.diagonal(rdms, axis1=1, axis2=2), 0.0)


def test_gaussian_fid_rdms_match_diagonal_closed_form() -> None:
    means = np.zeros((1, 4, 2), dtype=np.float64)
    means[0, 1] = [1.0, 2.0]
    covariances = np.broadcast_to(np.eye(2), (1, 4, 2, 2)).copy()
    covariances[0, 1] = np.diag([4.0, 9.0])
    rdms = gaussian_fid_rdms_from_moments(means, covariances)
    expected = 1.0**2 + 2.0**2 + (2.0 - 1.0) ** 2 + (3.0 - 1.0) ** 2
    np.testing.assert_allclose(rdms[0, 0, 1], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rdms, rdms.transpose(0, 2, 1))
    np.testing.assert_allclose(np.diagonal(rdms, axis1=1, axis2=2), 0.0)


def test_gaussian_jeffreys_rdms_match_one_dimensional_closed_form() -> None:
    means = np.zeros((1, 4, 1), dtype=np.float64)
    means[0, 1, 0] = 2.0
    covariances = np.ones((1, 4, 1, 1), dtype=np.float64)
    covariances[0, 1, 0, 0] = 4.0
    rdms = gaussian_jeffreys_rdms_from_moments(means, covariances)
    expected = 0.5 * (1.0 / 4.0 + 4.0 - 2.0 + 4.0 * (1.0 + 1.0 / 4.0))
    np.testing.assert_allclose(rdms[0, 0, 1], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rdms, rdms.transpose(0, 2, 1))
    np.testing.assert_allclose(np.diagonal(rdms, axis1=1, axis2=2), 0.0)


def test_classical_fid_uses_class_specific_covariances() -> None:
    rng = np.random.default_rng(43)
    labels = np.repeat(np.arange(4), 12)
    x = rng.normal(size=(48, 2, 3))
    x[labels == 3] *= 3.0
    means, covariances = empirical_condition_gaussian_components(
        x,
        labels,
        standardize_features=False,
    )
    rdms = classical_fid_rdms(x, labels, standardize_features=False)
    np.testing.assert_allclose(rdms, gaussian_fid_rdms_from_moments(means, covariances))
    assert not np.allclose(covariances[:, 0], covariances[:, 3])


def test_translation_flow_returns_squared_euclidean_mean_rdms() -> None:
    rng = np.random.default_rng(113)
    labels = np.repeat(np.arange(4), 3)
    x = rng.normal(size=(12, 3, 2)) + labels[:, None, None] * 0.2
    rdms, metadata, means = translation_flow_squared_euclidean_rdms(
        x,
        labels,
        np.array([-1.0, 0.0, 1.0]),
        device=torch.device("cpu"),
        seed=19,
        config=FlowRDMConfig(
            hidden_dim=8,
            depth=1,
            epochs=2,
            batch_size=8,
            patience=0,
            standardize_features=False,
            device_resident_data=True,
        ),
        return_means=True,
    )
    assert rdms.shape == (3, 4, 4)
    assert means.shape == (3, 4, 2)
    np.testing.assert_allclose(rdms, squared_euclidean_rdms_from_means(means))
    np.testing.assert_allclose(rdms, rdms.transpose(0, 2, 1))
    assert metadata["velocity_family"] == "translation"
    assert metadata["distance"] == "squared_euclidean_between_endpoint_means"


def test_translation_flow_checkpoint_is_exact_evaluation_model(tmp_path) -> None:
    rng = np.random.default_rng(127)
    labels = np.repeat(np.arange(4), 3)
    x = rng.normal(size=(12, 2, 2))
    checkpoint = tmp_path / "translation_best.pt"
    _, metadata, means = translation_flow_squared_euclidean_rdms(
        x,
        labels,
        np.array([0.0, 1.0]),
        device=torch.device("cpu"),
        seed=23,
        config=FlowRDMConfig(
            hidden_dim=8,
            depth=1,
            epochs=2,
            batch_size=8,
            patience=0,
            standardize_features=False,
        ),
        return_means=True,
        checkpoint_path=checkpoint,
        checkpoint_context={"recording": "test"},
    )
    model, payload = load_translation_flow_checkpoint(checkpoint, device=torch.device("cpu"))
    conditions = condition_design(
        np.repeat(np.arange(4), 2),
        np.tile(np.array([0.0, 1.0]), 4),
    )
    with torch.no_grad():
        loaded = model.endpoint_mean(torch.tensor(conditions, dtype=torch.float32)).numpy()
    loaded = loaded.reshape(4, 2, 2).transpose(1, 0, 2)
    np.testing.assert_allclose(loaded, means, atol=0.0, rtol=0.0)
    assert metadata["checkpoint_path"] == str(checkpoint.resolve())
    assert payload["context"]["recording"] == "test"


def test_condition_affine_flow_fid_saves_best_checkpoint(tmp_path) -> None:
    rng = np.random.default_rng(131)
    labels = np.repeat(np.arange(4), 3)
    x = rng.normal(size=(12, 2, 2)) + labels[:, None, None] * 0.1
    checkpoint = tmp_path / "condition_affine_best.pt"
    rdms, metadata, components = condition_affine_flow_fid_rdms(
        x,
        labels,
        np.array([0.0, 1.0]),
        device=torch.device("cpu"),
        seed=29,
        config=FlowRDMConfig(
            hidden_dim=8,
            depth=1,
            epochs=2,
            batch_size=8,
            patience=0,
            covariance_ode_steps=3,
            standardize_features=False,
        ),
        return_components=True,
        checkpoint_path=checkpoint,
        checkpoint_context={"role": "reference"},
    )
    model, payload = load_condition_affine_flow_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
    )
    checkpoint_rdms, checkpoint_payload = condition_affine_flow_fid_rdms_from_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
    )
    assert model.training is False
    assert rdms.shape == (2, 4, 4)
    assert checkpoint_rdms.shape == rdms.shape
    np.testing.assert_allclose(checkpoint_rdms, checkpoint_rdms.transpose(0, 2, 1))
    assert components["flow_endpoint_covariances"].shape == (2, 4, 2, 2)
    np.testing.assert_allclose(rdms, rdms.transpose(0, 2, 1))
    assert metadata["covariance_sharing"] == "none_across_class_or_eeg_time"
    assert payload["training"]["selected_epoch"] == metadata["selected_epoch"]
    assert checkpoint_payload["training"] == payload["training"]
    assert payload["context"]["role"] == "reference"


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
