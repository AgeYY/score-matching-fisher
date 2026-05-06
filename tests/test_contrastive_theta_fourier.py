"""Unit tests for contrastive-soft scalar θ Fourier augmentation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fisher.contrastive_llr import (
    ContrastiveNormalizedDotScorer,
    augment_scalar_theta_for_dot_scorer,
    dot_scorer_augmented_theta_dim,
    theta_scalar_fourier_columns,
)


def test_dot_scorer_augmented_dim_default_k4() -> None:
    assert dot_scorer_augmented_theta_dim(fourier_k=4, fourier_include_linear=False) == 9
    assert dot_scorer_augmented_theta_dim(fourier_k=0, fourier_include_linear=False) == 1
    assert dot_scorer_augmented_theta_dim(fourier_k=4, fourier_include_linear=True) == 10


def test_theta_scalar_fourier_columns_matches_expected_width() -> None:
    ref = np.linspace(-1.0, 1.0, 20, dtype=np.float64).reshape(-1, 1)
    th = np.array([[0.0], [0.5]], dtype=np.float64)
    fou = theta_scalar_fourier_columns(th.reshape(-1), ref, k=4, period_mult=2.0, include_linear=False)
    assert fou.shape == (2, 8)


def test_augment_scalar_theta_shape_k4() -> None:
    th_tr = np.linspace(0.0, 1.0, 11, dtype=np.float64).reshape(-1, 1)
    th_raw = th_tr.copy()
    th_n = (th_tr - np.mean(th_tr)) / np.maximum(np.std(th_tr), 1e-6)
    aug = augment_scalar_theta_for_dot_scorer(th_n, th_raw, th_tr, 4, 2.0, False)
    assert aug.shape == (11, 9)


def test_normalized_dot_accepts_augmented_theta_dim() -> None:
    m = ContrastiveNormalizedDotScorer(x_dim=5, theta_dim=9, feature_dim=8, hidden_dim=16, depth=2)
    x = torch.randn(4, 5)
    theta = torch.randn(6, 9)
    logits = m.score_matrix(x, theta)
    assert logits.shape == (4, 6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_normalized_dot_cuda_smoke_augmented_theta() -> None:
    dev = torch.device("cuda")
    m = ContrastiveNormalizedDotScorer(x_dim=3, theta_dim=9, feature_dim=8, hidden_dim=16, depth=2).to(dev)
    x = torch.randn(2, 3, device=dev)
    theta = torch.randn(3, 9, device=dev)
    out = m.score_matrix(x, theta)
    assert out.shape == (2, 3)
