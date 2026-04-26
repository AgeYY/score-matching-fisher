"""Iterative HeIM-Flow helpers (initialization, MDS, and H updates).

This module is intentionally lightweight and model-agnostic: callers provide a
callback that trains/evaluates a flow model for each iteration and returns the
matrices needed for the HeIM update.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS

from fisher.score_distance import classical_mds_from_distances


@dataclass(frozen=True)
class HeimFlowConfig:
    n_bins: int
    n_iters: int
    mds_dim: int
    init_mode: str = "euclidean"
    distance_transform: str = "hellinger"
    min_class_count: int = 5
    min_bin_count: int = 5
    clf_random_state: int = 7
    convergence_tol: float = 0.0
    clip_low: float = 0.0
    clip_high: float = 1.0


@dataclass(frozen=True)
class HeimIterationInput:
    iteration: int
    theta_state_all: np.ndarray
    theta_state_train: np.ndarray
    theta_state_validation: np.ndarray
    output_dir: str


@dataclass(frozen=True)
class HeimIterationOutput:
    h2_binned: np.ndarray | None
    delta_l_matrix: np.ndarray | None
    h_sym_matrix: np.ndarray | None
    metadata: dict[str, float | int | str]


class HeimEstimatorCallback(Protocol):
    def __call__(self, payload: HeimIterationInput) -> HeimIterationOutput:
        """Train/evaluate one HeIM iteration and return flow-derived matrices."""


@dataclass
class HeimFlowResult:
    init_h2: np.ndarray
    final_h2: np.ndarray
    final_d: np.ndarray
    final_embedding: np.ndarray
    final_theta_state_all: np.ndarray
    history_h2: list[np.ndarray]
    history_d: list[np.ndarray]
    history_embedding: list[np.ndarray]
    rel_change_history: list[float]
    iteration_metadata: list[dict[str, float | int | str]]


def _sanitize_h2_matrix(h2: np.ndarray, *, clip_low: float, clip_high: float) -> np.ndarray:
    out = np.asarray(h2, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[0] != out.shape[1]:
        raise ValueError("H^2 matrix must be square.")
    out = 0.5 * (out + out.T)
    finite = np.isfinite(out)
    out[finite] = np.clip(out[finite], float(clip_low), float(clip_high))
    np.fill_diagonal(out, 0.0)
    return out


def _distance_from_h2(h2: np.ndarray, *, transform: str = "hellinger") -> np.ndarray:
    out = np.asarray(h2, dtype=np.float64)
    finite = np.isfinite(out)
    d = np.full_like(out, np.nan, dtype=np.float64)
    mode = str(transform).strip().lower()
    if mode == "hellinger":
        d[finite] = np.sqrt(np.clip(out[finite], 0.0, 1.0))
    elif mode == "bhattacharyya":
        h2 = np.clip(out[finite], 0.0, np.nextafter(1.0, 0.0))
        d[finite] = -np.log1p(-h2)
    else:
        raise ValueError("transform must be one of {'hellinger','bhattacharyya'}.")
    np.fill_diagonal(d, 0.0)
    return d


def _prepare_distance_for_mds(distance: np.ndarray) -> np.ndarray:
    d = np.asarray(distance, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance matrix must be square.")
    d_work = np.asarray(d, dtype=np.float64).copy()
    finite = np.isfinite(d_work)
    if not np.any(finite):
        raise ValueError("distance matrix has no finite entries for MDS.")
    finite_vals = d_work[finite]
    fill_val = float(np.max(finite_vals)) if finite_vals.size else 0.0
    d_work[~finite] = fill_val
    d_work = 0.5 * (d_work + d_work.T)
    np.fill_diagonal(d_work, 0.0)
    return d_work


def _mds_embedding_from_distance(distance: np.ndarray, *, n_components: int) -> np.ndarray:
    if int(n_components) < 1:
        raise ValueError("n_components must be >= 1.")
    d_work = _prepare_distance_for_mds(distance)
    mds_out = classical_mds_from_distances(d_work, n_components=int(n_components))
    emb = np.asarray(mds_out.embedding, dtype=np.float64)
    if emb.ndim != 2 or emb.shape[0] != d_work.shape[0] or emb.shape[1] != int(n_components):
        raise ValueError("MDS embedding shape mismatch.")
    return emb


def _metric_mds_embedding_from_distance(
    distance: np.ndarray,
    *,
    n_components: int,
    init_embedding: np.ndarray | None = None,
) -> np.ndarray:
    if int(n_components) < 1:
        raise ValueError("n_components must be >= 1.")
    d_work = _prepare_distance_for_mds(distance)
    init = None
    if init_embedding is not None:
        init = np.asarray(init_embedding, dtype=np.float64)
        if init.ndim != 2 or init.shape[0] != d_work.shape[0] or init.shape[1] != int(n_components):
            raise ValueError("init_embedding shape mismatch for metric MDS.")
    mds = MDS(
        n_components=int(n_components),
        metric=True,
        dissimilarity="precomputed",
        n_init=1,
        max_iter=300,
        eps=1e-6,
        random_state=0,
    )
    emb = np.asarray(mds.fit_transform(d_work, init=init), dtype=np.float64)
    if emb.ndim != 2 or emb.shape[0] != d_work.shape[0] or emb.shape[1] != int(n_components):
        raise ValueError("Metric MDS embedding shape mismatch.")
    return emb


def _embed_samples_from_bins(bin_idx: np.ndarray, bin_embedding: np.ndarray) -> np.ndarray:
    bi = np.asarray(bin_idx, dtype=np.int64).reshape(-1)
    emb = np.asarray(bin_embedding, dtype=np.float64)
    if emb.ndim != 2:
        raise ValueError("bin_embedding must be 2D.")
    if bi.size < 1:
        raise ValueError("bin_idx must be non-empty.")
    if int(np.min(bi)) < 0 or int(np.max(bi)) >= emb.shape[0]:
        raise ValueError("bin_idx contains out-of-range entries for bin_embedding.")
    return np.asarray(emb[bi], dtype=np.float64)


def _h2_from_delta_l(delta_l: np.ndarray, bin_idx: np.ndarray, *, n_bins: int) -> np.ndarray:
    d = np.asarray(delta_l, dtype=np.float64)
    bi = np.asarray(bin_idx, dtype=np.int64).reshape(-1)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("delta_l must be square.")
    if d.shape[0] != bi.size:
        raise ValueError("delta_l rows must match bin_idx size.")
    nb = int(n_bins)
    out = np.full((nb, nb), np.nan, dtype=np.float64)
    np.fill_diagonal(out, 0.0)
    bin_rows: list[np.ndarray] = [np.flatnonzero(bi == b) for b in range(nb)]
    for i in range(nb):
        ii = bin_rows[i]
        if ii.size < 1:
            continue
        for j in range(i + 1, nb):
            jj = bin_rows[j]
            if jj.size < 1:
                continue
            # Psi(u) = 1 - sech(u / 2).
            u_i_to_j = np.asarray(d[np.ix_(ii, jj)], dtype=np.float64)
            u_j_to_i = np.asarray(d[np.ix_(jj, ii)], dtype=np.float64)
            psi_i = 1.0 - (1.0 / np.cosh(np.clip(0.5 * u_i_to_j, -60.0, 60.0)))
            psi_j = 1.0 - (1.0 / np.cosh(np.clip(0.5 * u_j_to_i, -60.0, 60.0)))
            m_i = float(np.nanmean(psi_i))
            m_j = float(np.nanmean(psi_j))
            if not np.isfinite(m_i) or not np.isfinite(m_j):
                continue
            h2_ij = 0.5 * (m_i + m_j)
            out[i, j] = h2_ij
            out[j, i] = h2_ij
    finite = np.isfinite(out)
    out[finite] = np.clip(out[finite], 0.0, 1.0)
    np.fill_diagonal(out, 0.0)
    return out


def initialize_h2_classifier(
    *,
    x_train: np.ndarray,
    bin_train: np.ndarray,
    x_eval: np.ndarray,
    bin_eval: np.ndarray,
    n_bins: int,
    min_class_count: int,
    random_state: int,
) -> np.ndarray:
    """Classifier-initialized binned H^2 estimate from pairwise logistic logits."""
    x_tr = np.asarray(x_train, dtype=np.float64)
    x_ev = np.asarray(x_eval, dtype=np.float64)
    bi_tr = np.asarray(bin_train, dtype=np.int64).reshape(-1)
    bi_ev = np.asarray(bin_eval, dtype=np.int64).reshape(-1)
    if x_tr.ndim != 2 or x_ev.ndim != 2:
        raise ValueError("x_train and x_eval must be 2D.")
    if x_tr.shape[1] != x_ev.shape[1]:
        raise ValueError("x_train and x_eval feature dims must match.")
    if x_tr.shape[0] != bi_tr.shape[0] or x_ev.shape[0] != bi_ev.shape[0]:
        raise ValueError("x/bin row mismatch.")
    if int(min_class_count) < 1:
        raise ValueError("min_class_count must be >= 1.")

    nb = int(n_bins)
    out = np.full((nb, nb), np.nan, dtype=np.float64)
    rs = int(random_state)
    for i in range(nb):
        for j in range(i + 1, nb):
            ia_tr = np.flatnonzero(bi_tr == i)
            jb_tr = np.flatnonzero(bi_tr == j)
            ni_tr, nj_tr = int(ia_tr.size), int(jb_tr.size)
            if ni_tr < int(min_class_count) or nj_tr < int(min_class_count):
                continue
            ia_ev = np.flatnonzero(bi_ev == i)
            jb_ev = np.flatnonzero(bi_ev == j)
            ni_ev, nj_ev = int(ia_ev.size), int(jb_ev.size)
            if ni_ev < 1 or nj_ev < 1:
                continue
            x_pair_tr = np.vstack([x_tr[ia_tr], x_tr[jb_tr]])
            y_pair_tr = np.concatenate([np.zeros(ni_tr, dtype=np.int64), np.ones(nj_tr, dtype=np.int64)])
            n_pair_tr = ni_tr + nj_tr
            w_pair_tr = np.concatenate(
                [
                    np.full(ni_tr, 0.5 * float(n_pair_tr) / float(ni_tr), dtype=np.float64),
                    np.full(nj_tr, 0.5 * float(n_pair_tr) / float(nj_tr), dtype=np.float64),
                ]
            )
            x_pair_ev = np.vstack([x_ev[ia_ev], x_ev[jb_ev]])
            try:
                clf = LogisticRegression(solver="lbfgs", random_state=rs)
                clf.fit(x_pair_tr, y_pair_tr, sample_weight=w_pair_tr)
                logits = np.asarray(clf.decision_function(x_pair_ev), dtype=np.float64).reshape(-1)
            except Exception:
                continue
            psi = 1.0 - (1.0 / np.cosh(np.clip(0.5 * logits, -60.0, 60.0)))
            psi_i = psi[:ni_ev]
            psi_j = psi[ni_ev:]
            h2_ij = 0.5 * float(np.mean(psi_i)) + 0.5 * float(np.mean(psi_j))
            out[i, j] = h2_ij
            out[j, i] = h2_ij
    out = _sanitize_h2_matrix(out, clip_low=0.0, clip_high=1.0)
    return out


def initialize_h2_euclidean(
    *,
    x_all: np.ndarray,
    bin_all: np.ndarray,
    n_bins: int,
    min_bin_count: int,
) -> np.ndarray:
    """Shared-Gaussian (Ledoit-Wolf) initialized binned H^2 estimate."""
    x = np.asarray(x_all, dtype=np.float64)
    bi = np.asarray(bin_all, dtype=np.int64).reshape(-1)
    nb = int(n_bins)
    min_count = int(min_bin_count)
    if x.ndim != 2:
        raise ValueError("x_all must be 2D.")
    if x.shape[0] != bi.shape[0]:
        raise ValueError("x_all and bin_all rows must match.")
    if nb < 1:
        raise ValueError("n_bins must be >= 1.")
    if min_count < 1:
        raise ValueError("min_bin_count must be >= 1.")

    means = np.full((nb, x.shape[1]), np.nan, dtype=np.float64)
    valid = np.zeros(nb, dtype=bool)
    residuals: list[np.ndarray] = []
    for b in range(nb):
        idx = np.flatnonzero(bi == b)
        if int(idx.size) < min_count:
            continue
        xb = np.asarray(x[idx], dtype=np.float64)
        mu = np.mean(xb, axis=0)
        means[b, :] = mu
        valid[b] = True
        residuals.append(xb - mu)

    out = np.full((nb, nb), np.nan, dtype=np.float64)
    np.fill_diagonal(out, 0.0)
    if int(np.sum(valid)) < 2 or not residuals:
        return out
    resid = np.vstack(residuals)
    if resid.shape[0] < 2:
        return out
    try:
        lw = LedoitWolf().fit(resid)
        precision = np.asarray(lw.precision_, dtype=np.float64)
    except Exception:
        return out
    for i in range(nb):
        if not valid[i]:
            continue
        for j in range(i + 1, nb):
            if not valid[j]:
                continue
            diff = means[i] - means[j]
            maha2 = float(diff @ precision @ diff)
            if not np.isfinite(maha2):
                continue
            h2_ij = 1.0 - float(np.exp(-0.125 * max(0.0, maha2)))
            out[i, j] = h2_ij
            out[j, i] = h2_ij
    return _sanitize_h2_matrix(out, clip_low=0.0, clip_high=1.0)


def initialize_mean_mahalanobis_distance(
    *,
    x_all: np.ndarray,
    bin_all: np.ndarray,
    n_bins: int,
    min_bin_count: int,
) -> np.ndarray:
    """Unbounded shared-covariance Mahalanobis distance between per-bin observation means.

    For bins i, j with means ``mu_i``, ``mu_j`` and shared Ledoit--Wolf precision on pooled
    within-bin residuals, returns

        D_ij = sqrt((mu_i - mu_j)^T Sigma^{-1} (mu_i - mu_j))

    on the diagonal 0, with NaN where a bin has too few points or a pair is ill-defined.
    This matrix is used directly for the first metric MDS fit (not mapped through bounded H^2).
    """
    x = np.asarray(x_all, dtype=np.float64)
    bi = np.asarray(bin_all, dtype=np.int64).reshape(-1)
    nb = int(n_bins)
    min_count = int(min_bin_count)
    if x.ndim != 2:
        raise ValueError("x_all must be 2D.")
    if x.shape[0] != bi.shape[0]:
        raise ValueError("x_all and bin_all rows must match.")
    if nb < 1:
        raise ValueError("n_bins must be >= 1.")
    if min_count < 1:
        raise ValueError("min_bin_count must be >= 1.")

    means = np.full((nb, x.shape[1]), np.nan, dtype=np.float64)
    valid = np.zeros(nb, dtype=bool)
    residuals: list[np.ndarray] = []
    for b in range(nb):
        idx = np.flatnonzero(bi == b)
        if int(idx.size) < min_count:
            continue
        xb = np.asarray(x[idx], dtype=np.float64)
        mu = np.mean(xb, axis=0)
        means[b, :] = mu
        valid[b] = True
        residuals.append(xb - mu)

    out = np.full((nb, nb), np.nan, dtype=np.float64)
    np.fill_diagonal(out, 0.0)
    if int(np.sum(valid)) < 2 or not residuals:
        return out
    resid = np.vstack(residuals)
    if resid.shape[0] < 2:
        return out
    try:
        lw = LedoitWolf().fit(resid)
        precision = np.asarray(lw.precision_, dtype=np.float64)
    except Exception:
        return out
    for i in range(nb):
        if not valid[i]:
            continue
        for j in range(i + 1, nb):
            if not valid[j]:
                continue
            diff = means[i] - means[j]
            maha2 = float(diff @ precision @ diff)
            if not np.isfinite(maha2) or maha2 < 0.0:
                continue
            dist = float(np.sqrt(max(0.0, maha2)))
            out[i, j] = dist
            out[j, i] = dist
    return out


def run_heim_flow(
    *,
    x_all: np.ndarray,
    x_train: np.ndarray,
    x_validation: np.ndarray,
    bin_all: np.ndarray,
    bin_train: np.ndarray,
    bin_validation: np.ndarray,
    output_root: str,
    cfg: HeimFlowConfig,
    estimate_callback: HeimEstimatorCallback,
) -> HeimFlowResult:
    """Run iterative HeIM-Flow using callback-provided flow estimates."""
    if int(cfg.n_iters) < 1:
        raise ValueError("cfg.n_iters must be >= 1.")
    if int(cfg.mds_dim) < 1:
        raise ValueError("cfg.mds_dim must be >= 1.")
    if int(cfg.n_bins) < 1:
        raise ValueError("cfg.n_bins must be >= 1.")
    mode = str(cfg.init_mode).strip().lower()
    if mode not in ("euclidean", "classifier", "mean_mahalanobis"):
        raise ValueError("cfg.init_mode must be one of {'euclidean','classifier','mean_mahalanobis'}.")
    distance_transform = str(cfg.distance_transform).strip().lower()
    if distance_transform not in ("hellinger", "bhattacharyya"):
        raise ValueError("cfg.distance_transform must be one of {'hellinger','bhattacharyya'}.")

    if mode == "mean_mahalanobis":
        d_k = initialize_mean_mahalanobis_distance(
            x_all=x_all,
            bin_all=bin_all,
            n_bins=int(cfg.n_bins),
            min_bin_count=int(cfg.min_bin_count),
        )
        d_k = np.asarray(d_k, dtype=np.float64)
        h2_k = np.full_like(d_k, np.nan, dtype=np.float64)
        fin = np.isfinite(d_k) & (d_k >= 0.0)
        h2_k[fin] = 1.0 - np.exp(-0.125 * (d_k[fin] ** 2))
        np.fill_diagonal(h2_k, 0.0)
        h2_k = _sanitize_h2_matrix(h2_k, clip_low=float(cfg.clip_low), clip_high=float(cfg.clip_high))
    elif mode == "euclidean":
        h2_k = initialize_h2_euclidean(
            x_all=x_all,
            bin_all=bin_all,
            n_bins=int(cfg.n_bins),
            min_bin_count=int(cfg.min_bin_count),
        )
        h2_k = _sanitize_h2_matrix(h2_k, clip_low=float(cfg.clip_low), clip_high=float(cfg.clip_high))
        d_k = _distance_from_h2(h2_k, transform=distance_transform)
    else:
        h2_k = initialize_h2_classifier(
            x_train=x_train,
            bin_train=bin_train,
            x_eval=x_all,
            bin_eval=bin_all,
            n_bins=int(cfg.n_bins),
            min_class_count=int(cfg.min_class_count),
            random_state=int(cfg.clf_random_state),
        )
        h2_k = _sanitize_h2_matrix(h2_k, clip_low=float(cfg.clip_low), clip_high=float(cfg.clip_high))
        d_k = _distance_from_h2(h2_k, transform=distance_transform)

    history_h2: list[np.ndarray] = [np.asarray(h2_k, dtype=np.float64)]
    history_d: list[np.ndarray] = [np.asarray(d_k, dtype=np.float64)]
    history_embedding: list[np.ndarray] = []
    rel_change_history: list[float] = []
    iteration_metadata: list[dict[str, float | int | str]] = []

    n_train = int(np.asarray(x_train).shape[0])
    n_val = int(np.asarray(x_validation).shape[0])
    if n_train + n_val != int(np.asarray(x_all).shape[0]):
        raise ValueError("x_train + x_validation must match x_all rows.")

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    theta_state_all = np.zeros((int(np.asarray(x_all).shape[0]), int(cfg.mds_dim)), dtype=np.float64)
    prev_embedding: np.ndarray | None = None

    for k in range(int(cfg.n_iters)):
        if prev_embedding is None:
            # Deterministic iteration-0 seed before metric MDS optimization.
            init_seed = _mds_embedding_from_distance(d_k, n_components=int(cfg.mds_dim))
            z_k = _metric_mds_embedding_from_distance(
                d_k,
                n_components=int(cfg.mds_dim),
                init_embedding=init_seed,
            )
        else:
            z_k = _metric_mds_embedding_from_distance(
                d_k,
                n_components=int(cfg.mds_dim),
                init_embedding=prev_embedding,
            )
        history_embedding.append(np.asarray(z_k, dtype=np.float64))
        prev_embedding = np.asarray(z_k, dtype=np.float64)
        theta_state_all = _embed_samples_from_bins(np.asarray(bin_all, dtype=np.int64), z_k)
        theta_state_train = np.asarray(theta_state_all[:n_train], dtype=np.float64)
        theta_state_val = np.asarray(theta_state_all[n_train:], dtype=np.float64)
        iter_dir = str(output_root_path / f"iter_{k:03d}")
        payload = HeimIterationInput(
            iteration=k,
            theta_state_all=theta_state_all,
            theta_state_train=theta_state_train,
            theta_state_validation=theta_state_val,
            output_dir=iter_dir,
        )
        est_out = estimate_callback(payload)
        h2_next: np.ndarray
        if est_out.delta_l_matrix is not None:
            h2_next = _h2_from_delta_l(
                est_out.delta_l_matrix,
                np.asarray(bin_all, dtype=np.int64),
                n_bins=int(cfg.n_bins),
            )
        elif est_out.h2_binned is not None:
            h2_next = np.asarray(est_out.h2_binned, dtype=np.float64)
        elif est_out.h_sym_matrix is not None:
            # Fallback: treat returned symmetric H matrix as sample-wise and average by bin pairs.
            h_sym = np.asarray(est_out.h_sym_matrix, dtype=np.float64)
            h2_next = np.full((int(cfg.n_bins), int(cfg.n_bins)), np.nan, dtype=np.float64)
            bi = np.asarray(bin_all, dtype=np.int64).reshape(-1)
            for i in range(int(cfg.n_bins)):
                ii = np.flatnonzero(bi == i)
                if ii.size < 1:
                    continue
                for j in range(i, int(cfg.n_bins)):
                    jj = np.flatnonzero(bi == j)
                    if jj.size < 1:
                        continue
                    block = np.asarray(h_sym[np.ix_(ii, jj)], dtype=np.float64)
                    v = float(np.nanmean(block))
                    h2_next[i, j] = v
                    h2_next[j, i] = v
        else:
            raise ValueError("estimate_callback must provide at least one of delta_l_matrix/h2_binned/h_sym_matrix.")

        h2_next = _sanitize_h2_matrix(h2_next, clip_low=float(cfg.clip_low), clip_high=float(cfg.clip_high))
        d_next = _distance_from_h2(h2_next, transform=distance_transform)
        denom = float(np.linalg.norm(d_k, ord="fro"))
        numer = float(np.linalg.norm(d_next - d_k, ord="fro"))
        rel = numer / max(denom, 1e-12)
        rel_change_history.append(rel)
        md = dict(est_out.metadata) if est_out.metadata is not None else {}
        md["iteration"] = int(k)
        md["fro_rel_change"] = float(rel)
        md["embedding_method"] = "metric_mds"
        md["embedding_warm_start"] = bool(k > 0)
        md["distance_transform"] = distance_transform
        iteration_metadata.append(md)

        history_h2.append(np.asarray(h2_next, dtype=np.float64))
        history_d.append(np.asarray(d_next, dtype=np.float64))
        h2_k = h2_next
        d_k = d_next
        if float(cfg.convergence_tol) > 0.0 and rel <= float(cfg.convergence_tol):
            break

    return HeimFlowResult(
        init_h2=np.asarray(history_h2[0], dtype=np.float64),
        final_h2=np.asarray(h2_k, dtype=np.float64),
        final_d=np.asarray(d_k, dtype=np.float64),
        final_embedding=np.asarray(history_embedding[-1], dtype=np.float64),
        final_theta_state_all=np.asarray(theta_state_all, dtype=np.float64),
        history_h2=history_h2,
        history_d=history_d,
        history_embedding=history_embedding,
        rel_change_history=rel_change_history,
        iteration_metadata=iteration_metadata,
    )
