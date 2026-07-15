from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fisher import ctsm_distance as cd
from fisher import distance_comparison as dc


def test_ctsm_v_default_endpoint_cutoff() -> None:
    assert cd.CTSMVJeffreysConfig().t_eps == 1e-4
    assert cd.CTSMVBinaryJeffreysConfig().t_eps == 1e-4


def test_ctsm_v_jeffreys_readout_uses_negative_log_ratio_expectation(monkeypatch) -> None:
    theta = np.eye(3, dtype=np.float32)[np.repeat(np.arange(3), 2)]
    labels = np.repeat(np.arange(3), 2)
    x = np.column_stack([labels, np.zeros_like(labels)]).astype(np.float32)

    def fake_log_ratio(model, x_pair, *, theta_a, theta_b, **kwargs):
        del model, kwargs
        condition_i = int(np.argmax(theta_a))
        condition_j = int(np.argmax(theta_b))
        assert np.all(x_pair[:, 0] == condition_i)
        directed_kl = float(condition_i + condition_j + 1)
        return np.full(x_pair.shape[0], -directed_kl, dtype=np.float64)

    monkeypatch.setattr(cd, "_estimate_log_ratio_batched", fake_log_ratio)
    symmetric, raw_symmetric, directed, endpoints = cd.estimate_ctsm_v_jeffreys_matrix(
        torch.nn.Identity(),
        theta_eval=theta,
        x_eval=x,
        labels=labels,
        num_categories=3,
        device=torch.device("cpu"),
        config=cd.CTSMVJeffreysConfig(integration_steps=2),
    )

    expected_directed = np.array(
        [[0.0, 2.0, 3.0], [2.0, 0.0, 4.0], [3.0, 4.0, 0.0]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(directed, expected_directed)
    np.testing.assert_allclose(raw_symmetric, 2.0 * expected_directed)
    np.testing.assert_allclose(symmetric, raw_symmetric)
    np.testing.assert_allclose(endpoints, np.eye(3, dtype=np.float32))


def test_ctsm_v_tiny_training_and_serialization(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    train_labels = np.repeat(np.arange(2), 12)
    val_labels = np.repeat(np.arange(2), 6)
    eval_labels = np.repeat(np.arange(2), 8)
    theta_train = np.eye(2, dtype=np.float32)[train_labels]
    theta_val = np.eye(2, dtype=np.float32)[val_labels]
    theta_eval = np.eye(2, dtype=np.float32)[eval_labels]
    means = np.array([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32)
    x_train = means[train_labels] + 0.2 * rng.normal(size=(train_labels.size, 2)).astype(np.float32)
    x_val = means[val_labels] + 0.2 * rng.normal(size=(val_labels.size, 2)).astype(np.float32)
    x_eval = means[eval_labels] + 0.2 * rng.normal(size=(eval_labels.size, 2)).astype(np.float32)
    config = cd.CTSMVJeffreysConfig(
        epochs=2,
        batch_size=8,
        hidden_dim=8,
        architecture="film",
        film_depth=1,
        t_eps=0.1,
        integration_steps=5,
        eval_batch_size=8,
        early_patience=2,
        validation_batches_per_epoch=1,
        log_every=10,
    )

    model, result = cd.train_and_estimate_ctsm_v_jeffreys(
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        theta_eval=theta_eval,
        x_eval=x_eval,
        labels_eval=eval_labels,
        num_categories=2,
        device=torch.device("cpu"),
        seed=7,
        config=config,
    )
    np.testing.assert_allclose(result.symmetric_kl_matrix, result.symmetric_kl_matrix.T)
    np.testing.assert_allclose(np.diag(result.symmetric_kl_matrix), 0.0)
    assert np.all(np.isfinite(result.directed_kl_matrix))

    npz_path, checkpoint_path = cd.save_ctsm_v_jeffreys_result(
        tmp_path / "ctsm.npz",
        tmp_path / "ctsm.pt",
        model=model,
        result=result,
    )
    assert npz_path.is_file()
    assert checkpoint_path.is_file()
    with np.load(npz_path, allow_pickle=False) as data:
        assert data["symmetric_kl_matrix"].shape == (2, 2)
        assert data["train_losses"].shape == (2,)


def test_pairwise_binary_ctsm_v_trains_one_model_per_pair(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    train_labels = np.repeat(np.arange(3), 8)
    val_labels = np.repeat(np.arange(3), 4)
    eval_labels = np.repeat(np.arange(3), 5)
    means = np.array([[-0.5, 0.0], [0.5, 0.0], [0.0, 0.7]], dtype=np.float32)
    x_train = means[train_labels] + 0.2 * rng.normal(size=(train_labels.size, 2)).astype(np.float32)
    x_val = means[val_labels] + 0.2 * rng.normal(size=(val_labels.size, 2)).astype(np.float32)
    x_eval = means[eval_labels] + 0.2 * rng.normal(size=(eval_labels.size, 2)).astype(np.float32)
    config = cd.CTSMVBinaryJeffreysConfig(
        epochs=1,
        batch_size=8,
        hidden_dim=8,
        t_eps=0.1,
        integration_steps=5,
        eval_batch_size=8,
        early_patience=1,
        validation_batches_per_epoch=1,
        log_every=10,
    )

    states, result = cd.train_and_estimate_pairwise_binary_ctsm_v_jeffreys(
        x_train=x_train,
        labels_train=train_labels,
        x_val=x_val,
        labels_val=val_labels,
        x_eval=x_eval,
        labels_eval=eval_labels,
        num_categories=3,
        device=torch.device("cpu"),
        seed=5,
        config=config,
    )

    assert set(states) == {"0_1", "0_2", "1_2"}
    assert set(result.pair_metadata) == set(states)
    np.testing.assert_allclose(result.symmetric_kl_matrix, result.symmetric_kl_matrix.T)
    np.testing.assert_allclose(result.raw_symmetric_kl_matrix, result.directed_kl_matrix + result.directed_kl_matrix.T)
    assert result.run_metadata["num_pair_models"] == 3

    npz_path, checkpoint_path = cd.save_pairwise_binary_ctsm_v_jeffreys_result(
        tmp_path / "binary.npz",
        tmp_path / "binary.pt",
        pair_state_dicts=states,
        result=result,
    )
    assert npz_path.is_file() and checkpoint_path.is_file()
    with np.load(npz_path, allow_pickle=False) as data:
        assert tuple(data["pair_keys"].tolist()) == ("0_1", "0_2", "1_2")
        assert "pair_0_1_train_losses" in data.files


def test_distance_comparison_serializes_ctsm_v(tmp_path: Path) -> None:
    metric = dc.METRIC_SYMMETRIC_KL
    labels = ("category_0", "category_1")
    classical = {metric: np.array([[0.0, 1.0], [1.0, 0.0]])}
    flow = {metric: np.array([[0.0, 2.0], [2.0, 0.0]])}
    fine = {metric: np.array([[0.0, 3.0], [3.0, 0.0]])}
    ctsm = {metric: np.array([[0.0, 4.0], [4.0, 0.0]])}
    ctsm_binary = {metric: np.array([[0.0, 4.5], [4.5, 0.0]])}
    truth = {metric: np.array([[0.0, 5.0], [5.0, 0.0]])}
    result = dc.assemble_comparison_result(
        metrics=(metric,),
        condition_names=labels,
        classical=classical,
        flow_matching=flow,
        flow_matching_nll_finetuned=fine,
        ctsm_v=ctsm,
        ctsm_v_binary=ctsm_binary,
        ground_truth=truth,
    )

    assert result.rows[0]["ctsm_v"] == 4.0
    assert result.rows[0]["abs_error_ctsm_v"] == 1.0
    assert result.rows[0]["ctsm_v_binary"] == 4.5
    assert result.rows[0]["abs_error_ctsm_v_binary"] == 0.5
    out = dc.write_results_npz(tmp_path / "comparison.npz", result)
    with np.load(out, allow_pickle=False) as data:
        np.testing.assert_allclose(data["ctsm_v_matrices"][0], ctsm[metric])
        np.testing.assert_allclose(data["abs_error_ctsm_v"][0, 0, 1], 1.0)
        np.testing.assert_allclose(data["ctsm_v_binary_matrices"][0], ctsm_binary[metric])
