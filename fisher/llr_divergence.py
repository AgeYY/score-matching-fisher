"""Divergence helpers derived from log-likelihood-ratio matrices."""

from __future__ import annotations

import numpy as np


def directed_kl_from_delta_l(delta_l: np.ndarray) -> np.ndarray:
    r"""Estimate directed KL from an LLR matrix.

    The expected input convention is ``delta_l[i, j] = log p_j(x_i) - log p_i(x_i)``.
    Therefore ``E_i[-delta_l[i, j]]`` estimates ``KL(p_i || p_j)``.
    """
    d = np.asarray(delta_l, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("delta_l must be a square 2D array.")
    out = -d.copy()
    np.fill_diagonal(out, 0.0)
    return out


def sym_kl_sample_from_delta_l(delta_l: np.ndarray) -> np.ndarray:
    """Symmetrize directed sample-level KL estimates from ``delta_l``."""
    directed = directed_kl_from_delta_l(delta_l)
    out = 0.5 * (directed + directed.T)
    np.fill_diagonal(out, 0.0)
    return out


def sym_kl_category_from_sample_directed(
    directed_kl: np.ndarray,
    category_labels: np.ndarray,
    *,
    k_cat: int,
) -> np.ndarray:
    """Aggregate directed sample-level KL estimates to categories, then symmetrize."""
    directed = np.asarray(directed_kl, dtype=np.float64)
    if directed.ndim != 2 or directed.shape[0] != directed.shape[1]:
        raise ValueError("directed_kl must be a square 2D array.")
    n = int(directed.shape[0])
    labs = np.asarray(category_labels, dtype=np.int64).reshape(-1)
    if int(labs.shape[0]) != n:
        raise ValueError("category_labels length must match directed_kl.shape[0].")
    nb = int(k_cat)
    if nb < 1:
        raise ValueError("k_cat must be >= 1.")

    col_sum = np.zeros((n, nb), dtype=np.float64)
    col_cnt = np.zeros((n, nb), dtype=np.float64)
    for j in range(n):
        b = int(labs[j])
        if 0 <= b < nb:
            col_sum[:, b] += directed[:, j]
            col_cnt[:, b] += 1.0
    directed_by_target = col_sum / np.maximum(col_cnt, 1.0)

    row_sum = np.zeros((nb, nb), dtype=np.float64)
    row_cnt = np.zeros((nb, nb), dtype=np.float64)
    for i in range(n):
        a = int(labs[i])
        if 0 <= a < nb:
            row_sum[a, :] += directed_by_target[i, :]
            row_cnt[a, :] += 1.0
    directed_cat = row_sum / np.maximum(row_cnt, 1.0)
    out = 0.5 * (directed_cat + directed_cat.T)
    np.fill_diagonal(out, 0.0)
    return out


def symmetric_kl_gaussian_diag_matrix(means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """Pairwise symmetric KL for diagonal Gaussian components."""
    mu = np.asarray(means, dtype=np.float64)
    var = np.asarray(variances, dtype=np.float64)
    if mu.ndim != 2 or var.shape != mu.shape:
        raise ValueError("means and variances must be equal-shape 2D arrays.")
    if np.any(~np.isfinite(var)) or np.any(var <= 0.0):
        raise ValueError("variances must be finite and positive.")
    diff = mu[:, None, :] - mu[None, :, :]
    vi = var[:, None, :]
    vj = var[None, :, :]
    out = 0.25 * np.sum((vi / vj) + (vj / vi) + diff * diff * ((1.0 / vi) + (1.0 / vj)) - 2.0, axis=2)
    out = np.maximum(out, 0.0)
    np.fill_diagonal(out, 0.0)
    return out


def symmetric_kl_gaussian_full_matrix(
    mu: np.ndarray,
    cov_or_diag: np.ndarray,
    *,
    is_diag: bool = False,
    jitter: float = 1e-9,
) -> np.ndarray:
    """Pairwise symmetric KL for full-covariance or diagonal Gaussian components."""
    means = np.asarray(mu, dtype=np.float64)
    cov = np.asarray(cov_or_diag, dtype=np.float64)
    if means.ndim != 2:
        raise ValueError("mu must be a 2D array.")
    k, d = int(means.shape[0]), int(means.shape[1])
    if bool(is_diag):
        return symmetric_kl_gaussian_diag_matrix(means, cov)
    if cov.ndim == 2 and cov.shape == (d, d):
        cov = np.broadcast_to(cov, (k, d, d)).copy()
    elif cov.ndim == 2 and cov.shape == means.shape:
        return symmetric_kl_gaussian_diag_matrix(means, cov)
    if cov.ndim != 3 or cov.shape[0] != means.shape[0] or cov.shape[1] != means.shape[1] or cov.shape[2] != means.shape[1]:
        raise ValueError("cov_or_diag must have shape (k, d, d) for full covariance input.")

    eye = np.eye(d, dtype=np.float64)
    cov_j = cov + float(jitter) * eye.reshape(1, d, d)
    inv = np.linalg.inv(cov_j)
    out = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(i + 1, k):
            diff = means[j] - means[i]
            trace_term = float(np.trace(inv[j] @ cov_j[i]) + np.trace(inv[i] @ cov_j[j]))
            quad_term = float(diff @ (inv[i] + inv[j]) @ diff)
            val = 0.25 * (trace_term + quad_term - 2.0 * float(d))
            out[i, j] = out[j, i] = max(0.0, val)
    return out
