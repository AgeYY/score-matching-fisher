"""Tests for bidirectional soft contrastive row-centering of column logits."""

import torch

from fisher.contrastive_llr import ContrastiveLLRMLP
from fisher.contrastive_llr import _bidir_soft_contrastive_loss_parts
from fisher.contrastive_llr import _theta_pair_distance


def test_col_log_probs_invariant_to_per_row_logit_offset():
    """Adding a constant to each row of A before row-centering leaves column-side logits unchanged."""
    torch.manual_seed(0)
    b = 4
    logits = torch.randn(b, b, dtype=torch.float32)
    row_offsets = torch.randn(b, 1, dtype=torch.float32)
    logits_shifted = logits + row_offsets

    centered = logits - logits.mean(dim=1, keepdim=True)
    centered_shift = logits_shifted - logits_shifted.mean(dim=1, keepdim=True)
    col_lp = torch.log_softmax(centered, dim=0)
    col_lp_shift = torch.log_softmax(centered_shift, dim=0)
    assert torch.allclose(col_lp, col_lp_shift, rtol=1e-5, atol=1e-6)


def test_bidir_column_term_matches_manual_centered_softmax():
    """Column loss from _bidir_soft_contrastive_loss_parts matches kernel-weighted centered column CE."""
    torch.manual_seed(1)
    b = 3
    x_dim, theta_dim = 2, 1
    model = ContrastiveLLRMLP(x_dim=x_dim, theta_dim=theta_dim, hidden_dim=8, depth=2)
    model.eval()
    x = torch.randn(b, x_dim)
    theta = torch.randn(b, theta_dim)
    h = 0.5
    periodic = False
    period = 6.283185307179586

    _, _, col_loss = _bidir_soft_contrastive_loss_parts(
        model,
        x,
        theta,
        bandwidth=h,
        periodic=periodic,
        period=period,
    )

    logits = model.score_matrix(x, theta)
    dist = _theta_pair_distance(theta, theta, periodic=periodic, period=period)
    log_w = -0.5 * (dist / h).pow(2)
    col_weights = torch.softmax(log_w, dim=0)
    logits_rc = logits - logits.mean(dim=1, keepdim=True)
    col_log_probs = torch.log_softmax(logits_rc, dim=0)
    expect_col = -(col_weights * col_log_probs).sum(dim=0).mean()
    assert torch.isclose(col_loss, expect_col, rtol=1e-5, atol=1e-5)
