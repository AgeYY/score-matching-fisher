from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fisher.models import (
    ConditionalXFlowVelocity,
    ConditionalXFlowVelocityFiLMPerLayer,
    ConditionalXFlowVelocityThetaFourierMLP,
    ConditionalXScore,
    UnconditionalXFlowVelocity,
    UnconditionalXFlowVelocityFiLMPerLayer,
    UnconditionalXScore,
)


@dataclass
class ClassicalMdsResult:
    embedding: np.ndarray
    eigenvalues_used: np.ndarray
    eigenvalues_all: np.ndarray
    strain_relative: float
    positive_eig_count: int


def compute_cross_score_matrix(
    model: ConditionalXScore,
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


def compute_unconditional_score_vectors(
    model: UnconditionalXScore,
    x: np.ndarray,
    sigma_eval: float,
    device: torch.device,
    row_batch_size: int = 128,
) -> np.ndarray:
    """Compute v_i = s(x_i, sigma_eval), shape (N, x_dim)."""
    x2 = np.asarray(x, dtype=np.float64)
    if x2.ndim != 2:
        raise ValueError("x must be a 2D array.")
    n = int(x2.shape[0])
    if row_batch_size < 1:
        raise ValueError("row_batch_size must be >= 1.")

    model.eval()
    out = np.zeros((n, x2.shape[1]), dtype=np.float64)
    with torch.no_grad():
        for i0 in range(0, n, row_batch_size):
            i1 = min(n, i0 + row_batch_size)
            xb = np.asarray(x2[i0:i1], dtype=np.float32)
            x_t = torch.from_numpy(xb).to(device)
            pred = model.predict_score(x=x_t, sigma_eval=float(sigma_eval))
            out[i0:i1, :] = pred.cpu().numpy().astype(np.float64)
    return out


def compute_unconditional_flow_velocity_vectors(
    model: UnconditionalXFlowVelocity | UnconditionalXFlowVelocityFiLMPerLayer,
    x: np.ndarray,
    t_eval: float,
    device: torch.device,
    row_batch_size: int = 128,
) -> np.ndarray:
    """Compute v_i = v(x_i, t_eval), shape (N, x_dim)."""
    x2 = np.asarray(x, dtype=np.float64)
    if x2.ndim != 2:
        raise ValueError("x must be a 2D array.")
    n = int(x2.shape[0])
    if row_batch_size < 1:
        raise ValueError("row_batch_size must be >= 1.")
    if not (0.0 <= float(t_eval) <= 1.0):
        raise ValueError("t_eval must be in [0, 1].")

    model.eval()
    out = np.zeros((n, x2.shape[1]), dtype=np.float64)
    with torch.no_grad():
        for i0 in range(0, n, row_batch_size):
            i1 = min(n, i0 + row_batch_size)
            xb = np.asarray(x2[i0:i1], dtype=np.float32)
            x_t = torch.from_numpy(xb).to(device)
            pred = model.predict_velocity(x_t=x_t, t_eval=float(t_eval))
            out[i0:i1, :] = pred.cpu().numpy().astype(np.float64)
    return out


def compute_cross_flow_velocity_matrix(
    model: ConditionalXFlowVelocity | ConditionalXFlowVelocityFiLMPerLayer | ConditionalXFlowVelocityThetaFourierMLP,
    theta: np.ndarray,
    x: np.ndarray,
    t_eval: float,
    device: torch.device,
    row_batch_size: int = 128,
) -> np.ndarray:
    """Compute V_ij = v_phi(x_i, theta_j, t_eval) with output shape (N, N, x_dim)."""
    theta2 = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
    x2 = np.asarray(x, dtype=np.float64)
    if x2.ndim != 2:
        raise ValueError("x must be a 2D array.")
    n = int(theta2.shape[0])
    if n != int(x2.shape[0]):
        raise ValueError("theta and x must have the same number of rows.")
    if row_batch_size < 1:
        raise ValueError("row_batch_size must be >= 1.")
    if not (0.0 <= float(t_eval) <= 1.0):
        raise ValueError("t_eval must be in [0, 1].")

    model.eval()
    v = np.zeros((n, n, x2.shape[1]), dtype=np.float64)
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
            pred = model.predict_velocity(x_t=x_t, theta=theta_t, t_eval=float(t_eval))
            v[i0:i1, :, :] = pred.cpu().numpy().reshape(b, n, x2.shape[1]).astype(np.float64)
    return v


def normalize_scores(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 3:
        raise ValueError("scores must have shape (N, N, x_dim).")
    denom = np.linalg.norm(s, axis=-1, keepdims=True)
    return s / np.maximum(denom, eps)


def normalize_score_rows_l2(vecs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2 unit rows for (N, x_dim)."""
    v = np.asarray(vecs, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError("vecs must have shape (N, x_dim).")
    denom = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(denom, eps)


def normalize_score_rows_geom(vecs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per row: scale so geometric mean of |component| equals 1."""
    v = np.asarray(vecs, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError("vecs must have shape (N, x_dim).")
    abs_v = np.abs(v)
    gm = np.exp(np.mean(np.log(np.maximum(abs_v, eps)), axis=-1, keepdims=True))
    denom = np.maximum(gm, eps)
    out = v / denom
    near_zero = np.all(abs_v <= eps, axis=-1, keepdims=True)
    return np.where(near_zero, 0.0, out)


def normalize_scores_geom(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per (i,j) vector: scale so geometric mean of |component| equals 1."""
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 3:
        raise ValueError("scores must have shape (N, N, x_dim).")
    abs_s = np.abs(s)
    gm = np.exp(np.mean(np.log(np.maximum(abs_s, eps)), axis=-1, keepdims=True))
    denom = np.maximum(gm, eps)
    out = s / denom
    near_zero = np.all(abs_s <= eps, axis=-1, keepdims=True)
    return np.where(near_zero, 0.0, out)


def score_distance_matrix_from_cross_scores(
    cross_scores: np.ndarray,
    *,
    use_sqrt: bool,
    x_dim_aggregate: str = "geom_mean",
    geom_mean_eps: float = 1e-12,
) -> np.ndarray:
    """Distance from S_ij using:
    d_ij^2 = 0.5 ||S_ii - S_ij||^2 + 0.5 ||S_ji - S_jj||^2.

    Per-pair squared differences along x_dim are aggregated with ``x_dim_aggregate``:
    ``sum`` (full squared norm), ``mean`` (mean squared difference per dim), or
    ``geom_mean`` (geometric mean of per-dimension squared differences).
    """
    s = np.asarray(cross_scores, dtype=np.float64)
    if s.ndim != 3 or s.shape[0] != s.shape[1]:
        raise ValueError("cross_scores must have shape (N, N, x_dim).")
    n, _, x_dim = s.shape
    mode = str(x_dim_aggregate).strip().lower()
    if mode not in ("sum", "mean", "geom_mean"):
        raise ValueError("x_dim_aggregate must be one of: sum, mean, geom_mean.")

    s_diag = s[np.arange(n), np.arange(n), :]  # (N, x_dim)
    diff_sq = (s_diag[:, None, :] - s) ** 2  # (N, N, x_dim)
    if mode == "sum":
        a = np.sum(diff_sq, axis=-1)
    elif mode == "mean":
        a = np.mean(diff_sq, axis=-1)
    else:
        eps = float(geom_mean_eps)
        if eps <= 0.0:
            raise ValueError("geom_mean_eps must be positive.")
        a = np.exp(np.mean(np.log(np.maximum(diff_sq, eps)), axis=-1))
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


def pairwise_mean_score_distance_matrix(vecs: np.ndarray, *, use_sqrt: bool) -> np.ndarray:
    """d_ij^2 = mean_k (v_ik - v_jk)^2 for rows v_i; symmetric, zero diagonal."""
    v = np.asarray(vecs, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError("vecs must have shape (N, x_dim).")
    diff = v[:, None, :] - v[None, :, :]
    d2 = np.mean(diff**2, axis=-1)
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
    score_rows: np.ndarray,
    *,
    score_normalize: str = "none",
) -> dict[str, np.ndarray | float]:
    """Unconditional path: distances from rows v_i in R^{x_dim} (mean squared coordinate gap)."""
    mode = str(score_normalize).strip().lower()
    if mode not in ("none", "l2", "geom"):
        raise ValueError("score_normalize must be one of: none, l2, geom.")
    if mode == "none":
        v = np.asarray(score_rows, dtype=np.float64)
    elif mode == "l2":
        v = normalize_score_rows_l2(score_rows)
    else:
        v = normalize_score_rows_geom(score_rows)
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


def evaluate_distance_variants(
    cross_scores: np.ndarray,
    *,
    score_normalize: str = "none",
    x_dim_aggregate: str = "mean",
) -> dict[str, np.ndarray | float]:
    """score_normalize: none | l2 | geom (L2 unit vectors vs geometric mean |component| scale).

    x_dim_aggregate: how to combine squared gaps along x_dim for the conditional cross-matrix
    distance (sum | mean | geom_mean).
    """
    mode = str(score_normalize).strip().lower()
    if mode not in ("none", "l2", "geom"):
        raise ValueError("score_normalize must be one of: none, l2, geom.")
    if mode == "none":
        s = np.asarray(cross_scores, dtype=np.float64)
    elif mode == "l2":
        s = normalize_scores(cross_scores)
    else:
        s = normalize_scores_geom(cross_scores)
    agg = str(x_dim_aggregate).strip().lower()
    d = score_distance_matrix_from_cross_scores(
        s,
        use_sqrt=True,
        x_dim_aggregate=agg,
    )
    d2 = score_distance_matrix_from_cross_scores(
        s,
        use_sqrt=False,
        x_dim_aggregate=agg,
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
