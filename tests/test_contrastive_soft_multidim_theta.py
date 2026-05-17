"""Multi-dimensional theta support for core contrastive-soft only."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fisher.contrastive_llr import (
    ContrastiveNormalizedDotScorer,
    _theta_pair_distance,
    categorical_soft_targets,
    compute_contrastive_soft_c_matrix,
    compute_contrastive_soft_categorical_c_matrix,
    contrastive_soft_normalization_and_bandwidth_from_train,
    train_contrastive_soft_categorical_llr,
    train_contrastive_soft_llr,
)


def test_theta_pair_distance_scalar_nonperiodic_matches_abs() -> None:
    t = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32)
    d = _theta_pair_distance(t, t, periodic=False, period=0.0)
    expected = torch.abs(t.reshape(-1, 1) - t.reshape(1, -1))
    assert torch.allclose(d, expected)


def test_theta_pair_distance_multidim_euclidean_whitened() -> None:
    th = torch.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=torch.float32)
    d = _theta_pair_distance(th, th, periodic=False, period=0.0)
    assert d.shape == (2, 2)
    assert d[0, 1].item() == pytest.approx(5.0)
    assert d[1, 0].item() == pytest.approx(5.0)


def test_theta_pair_distance_periodic_multidim_raises() -> None:
    th = torch.zeros(2, 3, dtype=torch.float32)
    with pytest.raises(ValueError, match="scalar theta"):
        _theta_pair_distance(th, th, periodic=True, period=6.28)


def test_normalization_rejects_periodic_multidim() -> None:
    th = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float64)
    x = np.zeros((3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="periodic"):
        contrastive_soft_normalization_and_bandwidth_from_train(
            th_tr=th,
            x_tr=x,
            bandwidth_bins=10,
            periodic=True,
            period=float(np.pi),
        )


def test_compute_contrastive_soft_c_matrix_2d_theta_shape() -> None:
    rng = np.random.default_rng(0)
    n, x_dim, d_th = 5, 4, 2
    model = ContrastiveNormalizedDotScorer(
        x_dim=x_dim,
        theta_dim=d_th,
        feature_dim=4,
        hidden_dim=8,
        depth=2,
    )
    th = rng.standard_normal((n, d_th))
    x = rng.standard_normal((n, x_dim))
    theta_mean = np.mean(th, axis=0)
    theta_std = np.maximum(np.std(th, axis=0), 1e-6)
    x_mean = np.zeros(x_dim, dtype=np.float64)
    x_std = np.ones(x_dim, dtype=np.float64)
    c = compute_contrastive_soft_c_matrix(
        model=model,
        theta_all=th,
        x_all=x,
        device=torch.device("cpu"),
        x_mean=x_mean,
        x_std=x_std,
        theta_mean=theta_mean,
        theta_std=theta_std,
        pair_batch_size=1024,
        contrastive_theta_fourier_k=0,
    )
    assert c.shape == (n, n)


def test_compute_contrastive_soft_c_matrix_bundled_fourier_theta_columns() -> None:
    """Pre-built Fourier θ rows (train_fourier_k=0 path) should accept wide θ without internal harmonics."""
    rng = np.random.default_rng(2)
    n, x_dim, feat_d = 6, 4, 14
    model = ContrastiveNormalizedDotScorer(
        x_dim=x_dim,
        theta_dim=feat_d,
        feature_dim=4,
        hidden_dim=8,
        depth=2,
    )
    th = rng.standard_normal((n, feat_d))
    x = rng.standard_normal((n, x_dim))
    theta_mean = np.mean(th, axis=0)
    theta_std = np.maximum(np.std(th, axis=0), 1e-6)
    x_mean = np.zeros(x_dim, dtype=np.float64)
    x_std = np.ones(x_dim, dtype=np.float64)
    c = compute_contrastive_soft_c_matrix(
        model=model,
        theta_all=th,
        x_all=x,
        device=torch.device("cpu"),
        x_mean=x_mean,
        x_std=x_std,
        theta_mean=theta_mean,
        theta_std=theta_std,
        pair_batch_size=1024,
        contrastive_theta_fourier_k=0,
    )
    assert c.shape == (n, n)


def test_train_contrastive_soft_llr_smoke_2d_cpu() -> None:
    rng = np.random.default_rng(1)
    n_tr, n_va, x_dim, d_th = 16, 12, 5, 2
    th_tr = rng.standard_normal((n_tr, d_th))
    th_va = rng.standard_normal((n_va, d_th))
    x_tr = rng.standard_normal((n_tr, x_dim))
    x_va = rng.standard_normal((n_va, x_dim))
    model = ContrastiveNormalizedDotScorer(
        x_dim=x_dim,
        theta_dim=d_th,
        feature_dim=4,
        hidden_dim=16,
        depth=2,
    )
    out = train_contrastive_soft_llr(
        model=model,
        theta_train=th_tr,
        x_train=x_tr,
        theta_val=th_va,
        x_val=x_va,
        device=torch.device("cpu"),
        epochs=2,
        batch_size=8,
        lr=1e-2,
        bandwidth_bins=4,
        contrastive_theta_fourier_k=0,
        log_every=1,
        patience=0,
    )
    assert len(out["train_losses"]) == 2
    assert out["bandwidth_bins"] == 4


def test_categorical_soft_targets_beta() -> None:
    labels = torch.tensor([0, 2], dtype=torch.long)
    hard = categorical_soft_targets(labels, 3, beta=0.0)
    assert torch.allclose(hard, torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    soft = categorical_soft_targets(labels, 3, beta=0.5)
    assert torch.allclose(
        soft,
        torch.tensor([[0.5, 0.25, 0.25], [0.25, 0.25, 0.5]]),
    )


def test_train_contrastive_soft_categorical_llr_smoke_cpu() -> None:
    rng = np.random.default_rng(3)
    x_tr = rng.standard_normal((18, 4))
    x_va = rng.standard_normal((12, 4))
    y_tr = np.tile(np.arange(3), 6)
    y_va = np.tile(np.arange(3), 4)
    model = ContrastiveNormalizedDotScorer(
        x_dim=4,
        theta_dim=3,
        feature_dim=4,
        hidden_dim=12,
        depth=2,
    )
    out = train_contrastive_soft_categorical_llr(
        model=model,
        x_train=x_tr,
        y_train=y_tr,
        x_val=x_va,
        y_val=y_va,
        n_classes=3,
        device=torch.device("cpu"),
        epochs=2,
        batch_size=6,
        lr=1e-2,
        beta=0.0,
        log_every=1,
        patience=0,
    )
    assert len(out["train_losses"]) == 2
    assert out["n_classes"] == 3


def test_compute_contrastive_soft_categorical_c_matrix_duplicates_class_columns() -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_normal((5, 3))
    labels = np.array([0, 1, 0, 2, 1], dtype=np.int64)
    model = ContrastiveNormalizedDotScorer(
        x_dim=3,
        theta_dim=3,
        feature_dim=4,
        hidden_dim=8,
        depth=2,
    )
    c = compute_contrastive_soft_categorical_c_matrix(
        model=model,
        x_all=x,
        y_all=labels,
        n_classes=3,
        device=torch.device("cpu"),
        x_mean=np.zeros(3, dtype=np.float64),
        x_std=np.ones(3, dtype=np.float64),
        pair_batch_size=32,
    )
    assert c.shape == (5, 5)
    assert np.allclose(c[:, 0], c[:, 2])
    assert np.allclose(c[:, 1], c[:, 4])
