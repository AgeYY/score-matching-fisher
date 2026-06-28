"""Bin-marginal conditional likelihood → symmetric H matrix (shared by linear-x-flow and sir_xflow)."""

from __future__ import annotations

import numpy as np


def lxf_bin_likelihood_hellinger(
    c_matrix: np.ndarray,
    bin_all: np.ndarray,
    n_bins: int,
) -> dict[str, np.ndarray]:
    r"""Compute linear-x-flow / likelihood-ratio H from bin-level likelihoods.

    ``c_matrix[i, j]`` is ``log p(x_i | theta_j)``. For each target bin ``b``,
    this estimates ``log p(x_i | B_b)`` as a stable log-mean-exp over all
    ``theta_j`` whose source sample belongs to ``b``. The row's own bin is the
    likelihood-ratio baseline, so same-bin expanded entries are exactly zero.
    """
    c = np.asarray(c_matrix, dtype=np.float64)
    bins = np.asarray(bin_all, dtype=np.int64).reshape(-1)
    nb = int(n_bins)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("c_matrix must be a square 2D array.")
    n = int(c.shape[0])
    if bins.shape[0] != n:
        raise ValueError("bin_all length must match c_matrix rows.")
    if nb < 1:
        raise ValueError("n_bins must be >= 1.")
    if np.any((bins < 0) | (bins >= nb)):
        raise ValueError("bin_all contains labels outside [0, n_bins).")

    counts = np.bincount(bins, minlength=nb).astype(np.int64)
    bin_log_likelihood = np.full((n, nb), np.nan, dtype=np.float64)
    for b in range(nb):
        idx = np.flatnonzero(bins == b)
        if idx.size == 0:
            continue
        vals = c[:, idx]
        vmax = np.max(vals, axis=1)
        finite_max = np.isfinite(vmax)
        out = np.full(n, np.nan, dtype=np.float64)
        if np.any(finite_max):
            shifted = vals[finite_max] - vmax[finite_max, None]
            out[finite_max] = (
                vmax[finite_max]
                + np.log(np.mean(np.exp(shifted), axis=1))
            )
        bin_log_likelihood[:, b] = out

    baseline = bin_log_likelihood[np.arange(n), bins]
    bin_delta_l = bin_log_likelihood - baseline[:, None]
    half_delta = np.clip(0.5 * bin_delta_l, -60.0, 60.0)
    h_directed_bin = 1.0 - (1.0 / np.cosh(half_delta))
    h_directed_bin[np.arange(n), bins] = 0.0

    h_binned_directed = np.full((nb, nb), np.nan, dtype=np.float64)
    for a in range(nb):
        rows = np.flatnonzero(bins == a)
        if rows.size == 0:
            continue
        for b in range(nb):
            if counts[b] > 0:
                h_binned_directed[a, b] = float(np.mean(h_directed_bin[rows, b], dtype=np.float64))
        h_binned_directed[a, a] = 0.0
    h_binned = 0.5 * (h_binned_directed + h_binned_directed.T)

    h_sym = 0.5 * (h_directed_bin[:, bins] + h_directed_bin[:, bins].T)
    same_bin = bins[:, None] == bins[None, :]
    h_sym[same_bin] = 0.0

    skl_directed_bin = -bin_delta_l
    skl_directed_bin[np.arange(n), bins] = 0.0
    skl_binned_directed = np.full((nb, nb), np.nan, dtype=np.float64)
    for a in range(nb):
        rows = np.flatnonzero(bins == a)
        if rows.size == 0:
            continue
        for b in range(nb):
            if counts[b] > 0:
                skl_binned_directed[a, b] = float(np.mean(skl_directed_bin[rows, b], dtype=np.float64))
        skl_binned_directed[a, a] = 0.0
    skl_binned = 0.5 * (skl_binned_directed + skl_binned_directed.T)

    skl_sym = 0.5 * (skl_directed_bin[:, bins] + skl_directed_bin[:, bins].T)
    skl_sym[same_bin] = 0.0
    return {
        "bin_log_likelihood": bin_log_likelihood,
        "bin_delta_l": bin_delta_l,
        "h_directed_bin": h_directed_bin,
        "h_binned": h_binned,
        "h_sym": h_sym,
        "skl_directed_bin": skl_directed_bin,
        "skl_binned_directed": skl_binned_directed,
        "skl_binned": skl_binned,
        "skl_sym": skl_sym,
        "bin_counts": counts,
    }
