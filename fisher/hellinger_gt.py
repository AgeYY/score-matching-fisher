"""Ground-truth Hellinger distance from known toy generative models."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from fisher.evaluation import log_p_x_given_theta


def _rng_from_dataset(dataset: Any) -> np.random.Generator:
    r = getattr(dataset, "rng", None)
    if isinstance(r, np.random.Generator):
        return r
    return np.random.default_rng(0)


def bin_centers_from_edges(edges: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return bin centers from contiguous edges (length n_bins+1 -> n_bins)."""
    e = np.asarray(edges, dtype=np.float64).reshape(-1)
    if e.size < 2:
        raise ValueError("edges must have length >= 2.")
    return 0.5 * (e[:-1] + e[1:])


def theta_centers_for_analytic_gt(dataset: Any, centers: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return center theta points for analytical GT Hellinger.

    One-dimensional ``centers`` are interpreted as theta1 bin centers. For
    two-dimensional theta datasets, theta2 is fixed at the midpoint of the
    dataset theta bounds.
    """
    tc = np.asarray(centers, dtype=np.float64)
    if tc.ndim == 2:
        return tc
    tc = tc.reshape(-1)
    theta_dim = int(getattr(dataset, "theta_dim", 1))
    if theta_dim == 1:
        return tc.reshape(-1, 1)
    if theta_dim == 2:
        low = float(getattr(dataset, "theta_low", -6.0))
        high = float(getattr(dataset, "theta_high", 6.0))
        if not (low < high):
            raise ValueError(f"dataset theta bounds invalid: theta_low={low} theta_high={high}")
        theta2_mid = 0.5 * (low + high)
        return np.stack([tc, np.full_like(tc, theta2_mid, dtype=np.float64)], axis=1)
    raise ValueError(f"Analytical GT Hellinger supports theta_dim 1 or 2; got theta_dim={theta_dim}")


def hellinger_sq_gaussian_diag(
    mu1: NDArray[np.float64],
    var1: NDArray[np.float64],
    mu2: NDArray[np.float64],
    var2: NDArray[np.float64],
) -> float:
    """Squared Hellinger distance between diagonal-covariance Gaussians."""
    m1 = np.asarray(mu1, dtype=np.float64).reshape(-1)
    m2 = np.asarray(mu2, dtype=np.float64).reshape(-1)
    v1 = np.asarray(var1, dtype=np.float64).reshape(-1)
    v2 = np.asarray(var2, dtype=np.float64).reshape(-1)
    if m1.shape != m2.shape or m1.shape != v1.shape or m1.shape != v2.shape:
        raise ValueError("mu and variance arrays must have matching 1D shapes.")
    if np.any(v1 <= 0.0) or np.any(v2 <= 0.0):
        raise ValueError("Gaussian variances must be positive.")
    vbar = 0.5 * (v1 + v2)
    db = np.sum(((m1 - m2) ** 2) / (8.0 * vbar) + 0.5 * np.log(vbar / np.sqrt(v1 * v2)))
    h2 = 1.0 - float(np.exp(-float(db)))
    return float(np.clip(h2, 0.0, 1.0))


def hellinger_sq_gaussian_full(
    mu1: NDArray[np.float64],
    cov1: NDArray[np.float64],
    mu2: NDArray[np.float64],
    cov2: NDArray[np.float64],
) -> float:
    """Squared Hellinger distance between full-covariance Gaussians."""
    m1 = np.asarray(mu1, dtype=np.float64).reshape(-1)
    m2 = np.asarray(mu2, dtype=np.float64).reshape(-1)
    c1 = np.asarray(cov1, dtype=np.float64)
    c2 = np.asarray(cov2, dtype=np.float64)
    if m1.shape != m2.shape:
        raise ValueError("Gaussian means must have matching shapes.")
    d = int(m1.size)
    if c1.shape != (d, d) or c2.shape != (d, d):
        raise ValueError("Gaussian covariances must have shape (d, d).")
    cbar = 0.5 * (c1 + c2)
    s1, ld1 = np.linalg.slogdet(c1)
    s2, ld2 = np.linalg.slogdet(c2)
    sb, ldb = np.linalg.slogdet(cbar)
    if s1 <= 0.0 or s2 <= 0.0 or sb <= 0.0:
        raise ValueError("Gaussian covariances must be positive definite.")
    dm = m1 - m2
    quad = float(dm @ np.linalg.solve(cbar, dm))
    db = 0.125 * quad + 0.5 * (float(ldb) - 0.5 * (float(ld1) + float(ld2)))
    h2 = 1.0 - float(np.exp(-float(db)))
    return float(np.clip(h2, 0.0, 1.0))


def estimate_hellinger_sq_grid_centers_analytic(
    dataset: Any,
    theta_centers: NDArray[np.float64],
    *,
    symmetrize: bool = False,
) -> NDArray[np.float64]:
    """Analytical squared Hellinger between Gaussian conditionals at theta centers.

    ``theta_centers`` has shape ``(n_bins, theta_dim)``. The returned matrix is
    center-to-center ``H^2(p(x|theta_i), p(x|theta_j))`` with no sampling.
    """
    centers = np.asarray(theta_centers, dtype=np.float64)
    if centers.ndim == 1:
        centers = centers.reshape(-1, 1)
    if centers.ndim != 2 or centers.shape[0] < 1 or centers.shape[1] < 1:
        raise ValueError("theta_centers must have shape (n_bins, theta_dim).")
    if not hasattr(dataset, "tuning_curve"):
        raise TypeError("Analytical Gaussian Hellinger requires dataset.tuning_curve(theta).")

    mu = np.asarray(dataset.tuning_curve(centers), dtype=np.float64)
    if mu.ndim != 2 or int(mu.shape[0]) != int(centers.shape[0]):
        raise ValueError(
            "dataset.tuning_curve(theta_centers) must return shape (n_bins, x_dim); "
            f"got {mu.shape}."
        )

    use_diag = bool(getattr(dataset, "diagonal_gaussian_observation_noise", False))
    if use_diag:
        if not hasattr(dataset, "_variance_diag_from_mu"):
            raise TypeError("Diagonal analytical Gaussian Hellinger requires dataset._variance_diag_from_mu(mu).")
        var = np.asarray(dataset._variance_diag_from_mu(mu), dtype=np.float64)
        if var.shape != mu.shape:
            raise ValueError(f"variance shape {var.shape} must match mean shape {mu.shape}.")
        n_bins = int(mu.shape[0])
        h2 = np.zeros((n_bins, n_bins), dtype=np.float64)
        for i in range(n_bins):
            for j in range(n_bins):
                h2[i, j] = hellinger_sq_gaussian_diag(mu[i], var[i], mu[j], var[j])
    else:
        if not hasattr(dataset, "covariance"):
            raise TypeError("Full analytical Gaussian Hellinger requires dataset.covariance(theta).")
        cov = np.asarray(dataset.covariance(centers), dtype=np.float64)
        if (
            cov.ndim != 3
            or int(cov.shape[0]) != int(mu.shape[0])
            or int(cov.shape[1]) != int(mu.shape[1])
            or int(cov.shape[2]) != int(mu.shape[1])
        ):
            raise ValueError(
                "dataset.covariance(theta_centers) must return shape (n_bins, x_dim, x_dim); "
                f"got {cov.shape}."
            )
        n_bins = int(mu.shape[0])
        h2 = np.zeros((n_bins, n_bins), dtype=np.float64)
        for i in range(n_bins):
            for j in range(n_bins):
                h2[i, j] = hellinger_sq_gaussian_full(mu[i], cov[i], mu[j], cov[j])

    np.fill_diagonal(h2, 0.0)
    np.clip(h2, 0.0, 1.0, out=h2)
    if symmetrize:
        h2 = 0.5 * (h2 + h2.T)
        np.fill_diagonal(h2, 0.0)
    return h2


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

    Notes
    -----
    When the dataset has ``theta_dim == 2``, ``bin_centers`` are interpreted as θ₁ bin centers.
    For each MC replicate we draw θ₂ independently from Uniform(theta_low, theta_high) (dataset
    bounds), form θ = (θ₁_center, θ₂), sample ``x``, and evaluate likelihoods using the **same**
    θ₂ when contrasting θ₁ centers ``i`` and ``j`` (paired nuisance draw).
    """
    centers = np.asarray(bin_centers, dtype=np.float64).reshape(-1)
    n_bins = int(centers.size)
    if n_bins < 1:
        raise ValueError("bin_centers must be non-empty.")
    if n_mc < 1:
        raise ValueError("n_mc must be >= 1.")

    theta_dim = int(getattr(dataset, "theta_dim", 1))
    h2 = np.zeros((n_bins, n_bins), dtype=np.float64)

    if theta_dim == 1:
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
                m = float(np.max(log_half))
                mean_exp = float(np.mean(np.exp(log_half - m)) * np.exp(m))
                h2[i, j] = 1.0 - mean_exp
    elif theta_dim == 2:
        low = float(getattr(dataset, "theta_low", -6.0))
        high = float(getattr(dataset, "theta_high", 6.0))
        if not (low < high):
            raise ValueError(f"dataset theta bounds invalid for GT MC: theta_low={low} theta_high={high}")
        rng = _rng_from_dataset(dataset)
        for i in range(n_bins):
            theta1_i = float(centers[i])
            theta2_samples = rng.uniform(low, high, size=(n_mc,)).astype(np.float64)
            t_col = np.stack(
                [np.full(n_mc, theta1_i, dtype=np.float64), theta2_samples],
                axis=1,
            )
            x = dataset.sample_x(t_col)
            lp_i = np.asarray(log_p_x_given_theta(x, t_col, dataset), dtype=np.float64).reshape(-1)
            if lp_i.shape[0] != n_mc:
                raise ValueError(f"log_p_x_given_theta length mismatch: got {lp_i.shape[0]}, expected {n_mc}.")
            for j in range(n_bins):
                theta1_j = float(centers[j])
                t_j = np.stack(
                    [np.full(n_mc, theta1_j, dtype=np.float64), theta2_samples],
                    axis=1,
                )
                lp_j = np.asarray(log_p_x_given_theta(x, t_j, dataset), dtype=np.float64).reshape(-1)
                log_half = 0.5 * (lp_j - lp_i)
                m = float(np.max(log_half))
                mean_exp = float(np.mean(np.exp(log_half - m)) * np.exp(m))
                h2[i, j] = 1.0 - mean_exp
    else:
        raise ValueError(f"GT MC Hellinger supports theta_dim 1 or 2; got theta_dim={theta_dim}")

    np.clip(h2, 0.0, 1.0, out=h2)
    if symmetrize:
        h2 = 0.5 * (h2 + h2.T)
    return h2


def estimate_hellinger_sq_grid_centers_mc(
    dataset: Any,
    theta_centers: NDArray[np.float64],
    *,
    n_mc: int,
    symmetrize: bool = False,
) -> NDArray[np.float64]:
    """Monte Carlo squared Hellinger between explicit theta grid centers.

    ``theta_centers`` has shape ``(n_bins, theta_dim)``. For each row center i, samples
    are drawn from ``p(x | theta_centers[i])`` and evaluated against every center j.
    This is the true 2D-grid counterpart of ``estimate_hellinger_sq_one_sided_mc``.
    """
    centers = np.asarray(theta_centers, dtype=np.float64)
    if centers.ndim == 1:
        centers = centers.reshape(-1, 1)
    if centers.ndim != 2 or centers.shape[0] < 1 or centers.shape[1] < 1:
        raise ValueError("theta_centers must have shape (n_bins, theta_dim).")
    if n_mc < 1:
        raise ValueError("n_mc must be >= 1.")

    n_bins = int(centers.shape[0])
    h2 = np.zeros((n_bins, n_bins), dtype=np.float64)
    for i in range(n_bins):
        t_i = np.repeat(centers[i : i + 1], repeats=int(n_mc), axis=0)
        x = dataset.sample_x(t_i)
        lp_i = np.asarray(log_p_x_given_theta(x, t_i, dataset), dtype=np.float64).reshape(-1)
        if lp_i.shape[0] != int(n_mc):
            raise ValueError(f"log_p_x_given_theta length mismatch: got {lp_i.shape[0]}, expected {n_mc}.")
        for j in range(n_bins):
            t_j = np.repeat(centers[j : j + 1], repeats=int(n_mc), axis=0)
            lp_j = np.asarray(log_p_x_given_theta(x, t_j, dataset), dtype=np.float64).reshape(-1)
            log_half = 0.5 * (lp_j - lp_i)
            m = float(np.max(log_half))
            mean_exp = float(np.mean(np.exp(log_half - m)) * np.exp(m))
            h2[i, j] = 1.0 - mean_exp

    np.clip(h2, 0.0, 1.0, out=h2)
    if symmetrize:
        h2 = 0.5 * (h2 + h2.T)
    return h2


def estimate_mean_llr_one_sided_mc(
    dataset: Any,
    bin_centers: NDArray[np.float64],
    *,
    n_mc: int,
) -> NDArray[np.float64]:
    """Monte Carlo estimate of one-sided mean log-likelihood ratio between bin-conditional models.

    For each pair of bin indices (i, j) with ``theta_i = centers[i]``, ``theta_j = centers[j]``,
    with samples ``x ~ p(x | theta_i)``:

        (LLR_gen)_{ij} = E_{x ~ p(x|theta_i)}[ log p(x|theta_j) - log p(x|theta_i) ].

    Uses the same row-wise sampling protocol and ``n_mc`` as
    :func:`estimate_hellinger_sq_one_sided_mc` (one MC batch per row ``i``).
    The result is **directional** (not symmetric in general); diagonals are 0.0.
    """
    centers = np.asarray(bin_centers, dtype=np.float64).reshape(-1)
    n_bins = int(centers.size)
    if n_bins < 1:
        raise ValueError("bin_centers must be non-empty.")
    if n_mc < 1:
        raise ValueError("n_mc must be >= 1.")

    theta_dim = int(getattr(dataset, "theta_dim", 1))
    out = np.zeros((n_bins, n_bins), dtype=np.float64)
    if theta_dim == 1:
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
                out[i, j] = float(np.mean(lp_j - lp_i))
    elif theta_dim == 2:
        low = float(getattr(dataset, "theta_low", -6.0))
        high = float(getattr(dataset, "theta_high", 6.0))
        if not (low < high):
            raise ValueError(f"dataset theta bounds invalid for GT MC: theta_low={low} theta_high={high}")
        rng = _rng_from_dataset(dataset)
        for i in range(n_bins):
            theta1_i = float(centers[i])
            theta2_samples = rng.uniform(low, high, size=(n_mc,)).astype(np.float64)
            t_col = np.stack(
                [np.full(n_mc, theta1_i, dtype=np.float64), theta2_samples],
                axis=1,
            )
            x = dataset.sample_x(t_col)
            lp_i = np.asarray(log_p_x_given_theta(x, t_col, dataset), dtype=np.float64).reshape(-1)
            if lp_i.shape[0] != n_mc:
                raise ValueError(f"log_p_x_given_theta length mismatch: got {lp_i.shape[0]}, expected {n_mc}.")
            for j in range(n_bins):
                theta1_j = float(centers[j])
                t_j = np.stack(
                    [np.full(n_mc, theta1_j, dtype=np.float64), theta2_samples],
                    axis=1,
                )
                lp_j = np.asarray(log_p_x_given_theta(x, t_j, dataset), dtype=np.float64).reshape(-1)
                out[i, j] = float(np.mean(lp_j - lp_i))
    else:
        raise ValueError(f"GT MC LLR supports theta_dim 1 or 2; got theta_dim={theta_dim}")
    np.fill_diagonal(out, 0.0)
    return out
