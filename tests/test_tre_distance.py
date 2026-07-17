from __future__ import annotations

import numpy as np
import torch

from fisher.tre_distance import (
    TREDensityRatioConfig,
    TelescopingDensityRatio,
    build_tre_waymarks,
    estimate_tre_log_ratio,
    save_pairwise_tre_jeffreys_result,
    train_and_estimate_pairwise_tre_jeffreys,
    train_tre_density_ratio,
    tre_jeffreys_from_log_ratios,
    tre_waymark_coefficients,
)


def test_tre_waymarks_have_exact_endpoints_and_unit_norm_coefficients() -> None:
    x0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    left, right = tre_waymark_coefficients(4, schedule="angle")
    waymarks = build_tre_waymarks(x0, x1, num_bridges=4, schedule="angle")

    torch.testing.assert_close(waymarks[0], x0)
    torch.testing.assert_close(waymarks[-1], x1)
    torch.testing.assert_close(left.square() + right.square(), torch.ones_like(left))


def test_tre_log_ratio_sums_adjacent_classifier_logits() -> None:
    model = TelescopingDensityRatio(input_dim=2, num_bridges=3, architecture="linear")
    with torch.no_grad():
        model.heads.weight.zero_()
        model.heads.bias.copy_(torch.tensor([1.0, 2.0, 3.0]))
    x = torch.randn(5, 2)
    torch.testing.assert_close(model.log_ratio(x), torch.full((5,), 6.0))


def test_tiny_linear_tre_training_returns_finite_oriented_ratio() -> None:
    rng = np.random.default_rng(3)
    x0_train = rng.normal(-1.0, 1.0, size=(128, 1)).astype(np.float32)
    x1_train = rng.normal(1.0, 1.0, size=(128, 1)).astype(np.float32)
    x0_val = rng.normal(-1.0, 1.0, size=(64, 1)).astype(np.float32)
    x1_val = rng.normal(1.0, 1.0, size=(64, 1)).astype(np.float32)
    model, result = train_tre_density_ratio(
        x0_train=x0_train,
        x1_train=x1_train,
        x0_validation=x0_val,
        x1_validation=x1_val,
        device=torch.device("cpu"),
        seed=5,
        config=TREDensityRatioConfig(
            num_bridges=4,
            architecture="linear",
            epochs=30,
            batch_size=64,
            lr=2e-2,
            early_patience=0,
            validation_pairs=128,
            log_every=100,
        ),
    )
    eval_x = np.array([[-1.0], [1.0]], dtype=np.float32)
    log_ratio = estimate_tre_log_ratio(model, eval_x, device=torch.device("cpu"))

    assert np.all(np.isfinite(result.train_losses))
    assert np.all(np.isfinite(result.validation_losses))
    assert log_ratio[0] > log_ratio[1]
    assert tre_jeffreys_from_log_ratios(log_ratio[:1], log_ratio[1:]) > 0.0


def test_pairwise_tre_trains_one_model_per_condition_pair(tmp_path) -> None:
    rng = np.random.default_rng(13)
    train_labels = np.repeat(np.arange(3), 12)
    validation_labels = np.repeat(np.arange(3), 6)
    eval_labels = np.repeat(np.arange(3), 8)
    means = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    x_train = means[train_labels] + 0.3 * rng.normal(size=(train_labels.size, 2)).astype(np.float32)
    x_validation = means[validation_labels] + 0.3 * rng.normal(
        size=(validation_labels.size, 2)
    ).astype(np.float32)
    x_eval = means[eval_labels] + 0.3 * rng.normal(size=(eval_labels.size, 2)).astype(np.float32)
    states, result = train_and_estimate_pairwise_tre_jeffreys(
        x_train=x_train,
        labels_train=train_labels,
        x_validation=x_validation,
        labels_validation=validation_labels,
        x_eval=x_eval,
        labels_eval=eval_labels,
        num_categories=3,
        device=torch.device("cpu"),
        seed=17,
        config=TREDensityRatioConfig(
            num_bridges=2,
            architecture="linear",
            epochs=2,
            batch_size=8,
            lr=1e-2,
            early_patience=0,
            validation_pairs=8,
            log_every=100,
        ),
        eval_batch_size=8,
    )

    assert set(states) == {"0_1", "0_2", "1_2"}
    assert set(result.pair_histories) == set(states)
    assert result.run_metadata["num_pair_models"] == 3
    np.testing.assert_allclose(result.symmetric_kl_matrix, result.symmetric_kl_matrix.T)
    np.testing.assert_allclose(
        result.raw_symmetric_kl_matrix,
        result.directed_kl_matrix + result.directed_kl_matrix.T,
    )

    npz_path, checkpoint_path = save_pairwise_tre_jeffreys_result(
        tmp_path / "tre.npz",
        tmp_path / "tre.pt",
        pair_state_dicts=states,
        result=result,
    )
    assert npz_path.is_file() and checkpoint_path.is_file()
    with np.load(npz_path, allow_pickle=False) as data:
        assert tuple(data["pair_keys"].tolist()) == ("0_1", "0_2", "1_2")
        assert "pair_0_1_train_losses" in data.files
