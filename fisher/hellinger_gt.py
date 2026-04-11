"""Monte Carlo ground-truth Hellinger distance from generative model log-likelihoods."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from fisher.evaluation import log_p_x_given_theta


def bin_centers_from_edges(edges: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return bin centers from contiguous edges (length n_bins+1 -> n_bins)."""
    e = np.asarray(edges, dtype=np.float64).reshape(-1)
    if e.size < 2:
        raise ValueError("edges must have length >= 2.")
    return 0.5 * (e[:-1] + e[1:])


def estimate_hellinger_sq_one_sided_mc(
    dataset: Any,
    bin_centers: NDArray[np.float64],
    *,
    n_mc: int,
    symmetrize: bool = False,
) -> NDArray[np.float64]:
    """Monte Carlo estimate of squared Hellinger distance between bin-conditional distributions.

    For bin indices i, j with representative stimuli ``theta_i = centers[i]``,
    ``theta_j = centers[j]``, uses the one-sided identity (see ``report/notes/hellinger_idea.tex``):

        H^2_ij = 1 - E_{x ~ p(x|theta_i)}[ exp( (log p(x|theta_j) - log p(x|theta_i)) / 2 ) ].

    Samples ``x`` are drawn from the dataset generative model at fixed ``theta_i`` (same shape
    as ``dataset.sample_x``).

    Parameters
    ----------
    dataset
        Object implementing ``sample_x(theta)`` and compatible with
        ``fisher.evaluation.log_p_x_given_theta`` (e.g. toy conditional datasets).
    bin_centers
        1D array of length ``n_bins`` (theta values for each bin).
    n_mc
        Number of Monte Carlo samples per row ``i``. In ``study_h_decoding_convergence`` this is
        ``n_ref // n_bins`` with ``n_bins * n_mc = n_ref`` (see that script's CLI validation).
    symmetrize
        If True, replace ``H^2`` with ``(H^2 + H^2.T) / 2`` (optional post-processing).

    Returns
    -------
    (n_bins, n_bins) matrix of estimated **squared** Hellinger values in ``[0, 1]`` (clipped).
    """
    centers = np.asarray(bin_centers, dtype=np.float64).reshape(-1)
    n_bins = int(centers.size)
    if n_bins < 1:
        raise ValueError("bin_centers must be non-empty.")
    if n_mc < 1:
        raise ValueError("n_mc must be >= 1.")

    h2 = np.zeros((n_bins, n_bins), dtype=np.float64)

    for i in range(n_bins):
        theta_i = float(centers[i])
        t_col = np.full((n_mc, 1), theta_i, dtype=np.float64)
        x = dataset.sample_x(t_col)
        lp_i = np.asarray(log_p_x_given_theta(x, t_col, dataset), dtype=np.float64).reshape(-1)
        if lp_i.shape[0] != n_mc:
            raise ValueError(f"log_p_x_given_theta length mismatch: got {lp_i.shape[0]}, expected {n_mc}.")

        for j in range(n_bins):
            theta_j = float(centers[j])
            t_j = np.full((n_mc, 1), theta_j, dtype=np.float64)
            lp_j = np.asarray(log_p_x_given_theta(x, t_j, dataset), dtype=np.float64).reshape(-1)
            log_half = 0.5 * (lp_j - lp_i)
            # Stable mean(exp(log_half)) without scipy (numpy may not expose logsumexp)
            m = float(np.max(log_half))
            mean_exp = float(np.mean(np.exp(log_half - m)) * np.exp(m))
            h2[i, j] = 1.0 - mean_exp

    np.clip(h2, 0.0, 1.0, out=h2)
    if symmetrize:
        h2 = 0.5 * (h2 + h2.T)
    return h2
