from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fisher.models import (
    ConditionalXScore,
    ConditionalXScoreFiLMPerLayer,
    ConditionalXScoreResidualConcat,
    UnconditionalXScore,
    UnconditionalXScoreFiLMPerLayer,
)


@dataclass
class ClassicalMdsResult:
    embedding: np.ndarray
    eigenvalues_used: np.ndarray
    eigenvalues_all: np.ndarray
    strain_relative: float
    positive_eig_count: int


def compute_cross_score_matrix(
    model: ConditionalXScore | ConditionalXScoreResidualConcat | ConditionalXScoreFiLMPerLayer,
    theta: np.ndarray,
    x: np.ndarray,
    sigma_eval: float,
    device: torch.device,
    row_batch_size: int = 128,
) -> np.ndarray:
    """Compute S_ij = s_phi(x_i, theta_j) with output shape (N, N, x_dim)."""
    theta2 = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
    x2 = np.asarray(x, dtype=np.float64)
    if x2.ndim != 2:
        raise ValueError("x must be a 2D array.")
    n = int(theta2.shape[0])
    if n != int(x2.shape[0]):
        raise ValueError("theta and x must have the same number of rows.")
    if row_batch_size < 1:
        raise ValueError("row_batch_size must be >= 1.")

    model.eval()
    s = np.zeros((n, n, x2.shape[1]), dtype=np.float64)
    theta_grid = np.asarray(theta2, dtype=np.float32)
    with torch.no_grad():
        for i0 in range(0, n, row_batch_size):
            i1 = min(n, i0 + row_batch_size)
            xb = np.asarray(x2[i0:i1], dtype=np.float32)
            b = int(i1 - i0)
            x_rep = np.repeat(xb, repeats=n, axis=0)
            theta_tile = np.tile(theta_grid, (b, 1))
            x_t = torch.from_numpy(x_rep).to(device)
            theta_t = torch.from_numpy(theta_tile).to(device)
            pred = model.predict_score(x=x_t, theta=theta_t, sigma_eval=float(sigma_eval))
            s[i0:i1, :, :] = pred.cpu().numpy().reshape(b, n, x2.shape[1]).astype(np.float64)
    return s


def normalize_scores(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 3:
        raise ValueError("scores must have shape (N, N, x_dim).")
    denom = np.linalg.norm(s, axis=-1, keepdims=True)
    return s / np.maximum(denom, eps)


def compute_unconditional_score_vectors(
    model: UnconditionalXScore | UnconditionalXScoreFiLMPerLayer,
    x: np.ndarray,
    sigma_eval: float,
    device: torch.device,
    row_batch_size: int = 128,
) -> np.ndarray:
    """Compute v_i = s_phi(x_i) with output shape (N, x_dim)."""
    x2 = np.asarray(x, dtype=np.float64)
    if x2.ndim != 2:
        raise ValueError("x must be a 2D array.")
    if row_batch_size < 1:
        raise ValueError("row_batch_size must be >= 1.")
    n = int(x2.shape[0])
    out = np.zeros((n, x2.shape[1]), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for i0 in range(0, n, row_batch_size):
            i1 = min(n, i0 + row_batch_size)
            xb = torch.from_numpy(np.asarray(x2[i0:i1], dtype=np.float32)).to(device)
            pred = model.predict_score(x=xb, sigma_eval=float(sigma_eval))
            out[i0:i1, :] = pred.detach().cpu().numpy().astype(np.float64)
    return out


def normalize_score_rows_l2(score_vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(score_vectors, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError("score_vectors must have shape (N, x_dim).")
    denom = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(denom, eps)


def pairwise_mean_score_distance_matrix(score_vectors: np.ndarray, *, use_sqrt: bool) -> np.ndarray:
    """Pairwise mean-squared distance over x-dim between score vectors."""
    v = np.asarray(score_vectors, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError("score_vectors must have shape (N, x_dim).")
    diff = v[:, None, :] - v[None, :, :]
    d2 = np.mean(diff * diff, axis=-1)
    d2 = 0.5 * (d2 + d2.T)
    np.fill_diagonal(d2, 0.0)
    d2 = np.clip(d2, 0.0, None)
    if use_sqrt:
        d = np.sqrt(d2)
        d = 0.5 * (d + d.T)
        np.fill_diagonal(d, 0.0)
        return d
    return d2


def evaluate_pairwise_score_distance_variants(
    score_vectors: np.ndarray,
    normalize_score: bool,
) -> dict[str, np.ndarray | float]:
    v = normalize_score_rows_l2(score_vectors) if normalize_score else np.asarray(score_vectors, dtype=np.float64)
    d = pairwise_mean_score_distance_matrix(v, use_sqrt=True)
    d2 = pairwise_mean_score_distance_matrix(v, use_sqrt=False)
    mds_d = classical_mds_from_distances(d, n_components=2)
    mds_d2 = classical_mds_from_distances(d2, n_components=2)
    return {
        "distance_d": d,
        "distance_d2": d2,
        "embedding_d": mds_d.embedding,
        "embedding_d2": mds_d2.embedding,
        "strain_d": float(mds_d.strain_relative),
        "strain_d2": float(mds_d2.strain_relative),
        "positive_eig_d": float(mds_d.positive_eig_count),
        "positive_eig_d2": float(mds_d2.positive_eig_count),
        "eigenvalues_d": mds_d.eigenvalues_all,
        "eigenvalues_d2": mds_d2.eigenvalues_all,
    }


def score_distance_matrix_from_cross_scores(
    cross_scores: np.ndarray,
    *,
    use_sqrt: bool,
    average_over_x_dim: bool = True,
) -> np.ndarray:
    """Distance from S_ij using:
    d_ij^2 = 0.5 ||S_ii - S_ij||^2 + 0.5 ||S_ji - S_jj||^2.
    """
    s = np.asarray(cross_scores, dtype=np.float64)
    if s.ndim != 3 or s.shape[0] != s.shape[1]:
        raise ValueError("cross_scores must have shape (N, N, x_dim).")
    n, _, x_dim = s.shape
    s_diag = s[np.arange(n), np.arange(n), :]  # (N, x_dim)
    a = np.sum((s_diag[:, None, :] - s) ** 2, axis=-1)  # (N, N)
    if average_over_x_dim:
        a = a / float(x_dim)
    d2 = 0.5 * (a + a.T)
    d2 = 0.5 * (d2 + d2.T)
    np.fill_diagonal(d2, 0.0)
    d2 = np.clip(d2, 0.0, None)
    if use_sqrt:
        d = np.sqrt(d2)
        d = 0.5 * (d + d.T)
        np.fill_diagonal(d, 0.0)
        return d
    return d2


def _theta_equal_width_bin_ids(
    theta: np.ndarray,
    *,
    theta_low: float,
    theta_high: float,
    n_bins: int,
) -> np.ndarray:
    t = np.asarray(theta, dtype=np.float64).reshape(-1)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")
    if not np.isfinite(theta_low) or not np.isfinite(theta_high) or theta_high <= theta_low:
        raise ValueError("Require finite theta range with theta_high > theta_low.")
    edges = np.linspace(float(theta_low), float(theta_high), int(n_bins) + 1, dtype=np.float64)
    ids = np.digitize(t, edges, right=False) - 1
    # Clamp out-of-range values to edge bins to keep all samples represented.
    return np.clip(ids, 0, int(n_bins) - 1).astype(np.int64)


def theta_bin_averaged_distance_matrix_from_cross_scores(
    cross_scores: np.ndarray,
    theta: np.ndarray,
    *,
    theta_low: float,
    theta_high: float,
    n_bins: int,
    min_bin_count: int = 1,
    use_sqrt: bool,
) -> np.ndarray:
    """Sample-indexed theta-bin-averaged distance from cross scores.

    For i, j:
      Delta_k(i,j) = s(x_k, theta_i) - s(x_k, theta_j)
      d^2(i,j) = mean_{k in K(i,j)} mean_dim(Delta_k(i,j)^2),
    where K(i,j) is the union of samples in bins(theta_i) and bins(theta_j).
    If either bin has too few samples (< min_bin_count), falls back to all rows.
    """
    s = np.asarray(cross_scores, dtype=np.float64)
    if s.ndim != 3 or s.shape[0] != s.shape[1]:
        raise ValueError("cross_scores must have shape (N, N, x_dim).")
    n, _, x_dim = s.shape
    if x_dim < 1:
        raise ValueError("cross_scores x_dim must be >= 1.")
    if min_bin_count < 1:
        raise ValueError("min_bin_count must be >= 1.")

    bin_ids = _theta_equal_width_bin_ids(
        theta,
        theta_low=float(theta_low),
        theta_high=float(theta_high),
        n_bins=int(n_bins),
    )
    if bin_ids.shape[0] != n:
        raise ValueError("theta must have the same number of rows as cross_scores.")

    bin_to_idx: list[np.ndarray] = [np.where(bin_ids == b)[0].astype(np.int64) for b in range(int(n_bins))]
    counts = np.asarray([arr.size for arr in bin_to_idx], dtype=np.int64)
    valid_bin = counts >= int(min_bin_count)
    all_rows = np.arange(n, dtype=np.int64)

    d2 = np.zeros((n, n), dtype=np.float64)
    for b_i in range(int(n_bins)):
        i_idx = bin_to_idx[b_i]
        if i_idx.size == 0:
            continue
        for b_j in range(b_i, int(n_bins)):
            j_idx = bin_to_idx[b_j]
            if j_idx.size == 0:
                continue

            # Sparse bins fallback: use all rows when any participating bin is under-populated.
            if not (bool(valid_bin[b_i]) and bool(valid_bin[b_j])):
                k_idx = all_rows
            elif b_i == b_j:
                k_idx = i_idx
            else:
                k_idx = np.union1d(i_idx, j_idx)
            if k_idx.size == 0:
                k_idx = all_rows

            n_i = i_idx.size
            n_j = j_idx.size
            acc = np.zeros((n_i, n_j), dtype=np.float64)
            for k in k_idx:
                a = s[int(k), i_idx, :]  # (n_i, x_dim)
                b = s[int(k), j_idx, :]  # (n_j, x_dim)
                a2 = np.sum(a * a, axis=1, keepdims=True)
                b2 = np.sum(b * b, axis=1, keepdims=True).T
                acc += (a2 + b2 - 2.0 * (a @ b.T)) / float(x_dim)
            block = acc / float(k_idx.size)

            d2[np.ix_(i_idx, j_idx)] = block
            if b_i != b_j:
                d2[np.ix_(j_idx, i_idx)] = block.T

    d2 = 0.5 * (d2 + d2.T)
    np.fill_diagonal(d2, 0.0)
    d2 = np.clip(d2, 0.0, None)
    if use_sqrt:
        d = np.sqrt(d2)
        d = 0.5 * (d + d.T)
        np.fill_diagonal(d, 0.0)
        return d
    return d2


def classical_mds_from_distances(
    distance_matrix: np.ndarray,
    n_components: int = 2,
    eig_tol: float = 1e-10,
) -> ClassicalMdsResult:
    d = np.asarray(distance_matrix, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be square.")
    n = int(d.shape[0])
    j = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / float(n)
    d2 = d * d
    b = -0.5 * j @ d2 @ j
    b = 0.5 * (b + b.T)
    evals, evecs = np.linalg.eigh(b)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    positive = evals > eig_tol
    n_pos = int(np.sum(positive))

    lam = np.maximum(evals[:n_components], 0.0)
    vec = evecs[:, :n_components]
    emb = vec * np.sqrt(lam.reshape(1, -1))
    b_hat = (vec * lam.reshape(1, -1)) @ vec.T
    denom = float(np.linalg.norm(b, ord="fro"))
    strain = float(np.linalg.norm(b - b_hat, ord="fro") / (denom + 1e-12))
    return ClassicalMdsResult(
        embedding=emb,
        eigenvalues_used=evals[:n_components],
        eigenvalues_all=evals,
        strain_relative=strain,
        positive_eig_count=n_pos,
    )


def evaluate_distance_variants(
    cross_scores: np.ndarray,
    normalize_score: bool,
) -> dict[str, np.ndarray | float]:
    s = normalize_scores(cross_scores) if normalize_score else np.asarray(cross_scores, dtype=np.float64)
    d = score_distance_matrix_from_cross_scores(
        s,
        use_sqrt=True,
        average_over_x_dim=True,
    )
    d2 = score_distance_matrix_from_cross_scores(
        s,
        use_sqrt=False,
        average_over_x_dim=True,
    )
    mds_d = classical_mds_from_distances(d, n_components=2)
    mds_d2 = classical_mds_from_distances(d2, n_components=2)
    return {
        "distance_d": d,
        "distance_d2": d2,
        "embedding_d": mds_d.embedding,
        "embedding_d2": mds_d2.embedding,
        "strain_d": float(mds_d.strain_relative),
        "strain_d2": float(mds_d2.strain_relative),
        "positive_eig_d": float(mds_d.positive_eig_count),
        "positive_eig_d2": float(mds_d2.positive_eig_count),
        "eigenvalues_d": mds_d.eigenvalues_all,
        "eigenvalues_d2": mds_d2.eigenvalues_all,
    }


def evaluate_distance_variants_theta_bin_avg(
    cross_scores: np.ndarray,
    theta: np.ndarray,
    *,
    normalize_score: bool,
    theta_low: float,
    theta_high: float,
    n_bins: int,
    min_bin_count: int,
) -> dict[str, np.ndarray | float]:
    s = normalize_scores(cross_scores) if normalize_score else np.asarray(cross_scores, dtype=np.float64)
    d = theta_bin_averaged_distance_matrix_from_cross_scores(
        s,
        theta,
        theta_low=float(theta_low),
        theta_high=float(theta_high),
        n_bins=int(n_bins),
        min_bin_count=int(min_bin_count),
        use_sqrt=True,
    )
    d2 = theta_bin_averaged_distance_matrix_from_cross_scores(
        s,
        theta,
        theta_low=float(theta_low),
        theta_high=float(theta_high),
        n_bins=int(n_bins),
        min_bin_count=int(min_bin_count),
        use_sqrt=False,
    )
    mds_d = classical_mds_from_distances(d, n_components=2)
    mds_d2 = classical_mds_from_distances(d2, n_components=2)
    return {
        "distance_d": d,
        "distance_d2": d2,
        "embedding_d": mds_d.embedding,
        "embedding_d2": mds_d2.embedding,
        "strain_d": float(mds_d.strain_relative),
        "strain_d2": float(mds_d2.strain_relative),
        "positive_eig_d": float(mds_d.positive_eig_count),
        "positive_eig_d2": float(mds_d2.positive_eig_count),
        "eigenvalues_d": mds_d.eigenvalues_all,
        "eigenvalues_d2": mds_d2.eigenvalues_all,
    }
