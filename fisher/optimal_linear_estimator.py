"""Moment-based optimal linear estimator (OLE) utilities.

The OLE is locally unbiased and has minimum variance among linear readouts.
It only uses the conditional mean derivative and response covariance, so the
implementation does not assume that the response distribution is Gaussian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class OptimalLinearEstimatorResult:
    """OLE weights, variance, and inverse-variance linear information."""

    weights: np.ndarray
    variance: np.ndarray
    linear_fisher: np.ndarray


@dataclass(frozen=True)
class CrossFittedOLEResult:
    """Held-out performance of locally fitted OLE decoders."""

    theta_midpoints: np.ndarray
    linear_fisher_raw: np.ndarray
    linear_fisher: np.ndarray
    fold_weights: np.ndarray
    fold_intercepts: np.ndarray
    projected_mean_left: np.ndarray
    projected_mean_right: np.ndarray
    projected_variance_left: np.ndarray
    projected_variance_right: np.ndarray
    n_left: np.ndarray
    n_right: np.ndarray
    fold_test_counts_left: np.ndarray
    fold_test_counts_right: np.ndarray


def optimal_linear_estimator(
    mean_derivative: np.ndarray,
    covariance: np.ndarray,
    *,
    solve_jitter: float = 0.0,
) -> OptimalLinearEstimatorResult:
    r"""Compute the optimal locally unbiased linear readout.

    For each derivative vector :math:`\mu'` and covariance :math:`\Sigma`,
    this returns

    .. math::

        w^* = \frac{\Sigma^{-1}\mu'}{\mu'^T\Sigma^{-1}\mu'},
        \qquad
        \operatorname{Var}(\hat\theta) = \frac{1}{\mu'^T\Sigma^{-1}\mu'}.

    Inputs may describe one estimator with shapes ``(d,)`` and ``(d, d)`` or
    a batch with shapes ``(..., d)`` and ``(..., d, d)``.
    """

    derivative = np.asarray(mean_derivative, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    if derivative.ndim < 1:
        raise ValueError("mean_derivative must have at least one dimension.")
    if cov.ndim < 2 or cov.shape[-2:] != (derivative.shape[-1],) * 2:
        raise ValueError(
            "covariance must end in (response_dim, response_dim) matching "
            "mean_derivative."
        )
    if derivative.shape[:-1] != cov.shape[:-2]:
        raise ValueError("mean_derivative and covariance batch shapes must match.")
    if not np.all(np.isfinite(derivative)) or not np.all(np.isfinite(cov)):
        raise ValueError("mean_derivative and covariance must be finite.")
    if float(solve_jitter) < 0.0:
        raise ValueError("solve_jitter must be nonnegative.")

    response_dim = int(derivative.shape[-1])
    symmetric_covariance = 0.5 * (cov + np.swapaxes(cov, -1, -2))
    if float(solve_jitter) > 0.0:
        symmetric_covariance = symmetric_covariance + float(solve_jitter) * np.eye(
            response_dim, dtype=np.float64
        )
    precision_derivative = np.linalg.solve(
        symmetric_covariance, derivative[..., None]
    )[..., 0]
    linear_fisher = np.einsum(
        "...i,...i->...", derivative, precision_derivative
    )

    if np.any(linear_fisher < -1e-10):
        raise ValueError("covariance must be positive semidefinite.")
    linear_fisher = np.maximum(linear_fisher, 0.0)
    positive = linear_fisher > 0.0
    weights = np.zeros_like(precision_derivative)
    np.divide(
        precision_derivative,
        linear_fisher[..., None],
        out=weights,
        where=positive[..., None],
    )
    variance = np.full_like(linear_fisher, np.inf)
    np.divide(1.0, linear_fisher, out=variance, where=positive)
    return OptimalLinearEstimatorResult(
        weights=weights,
        variance=variance,
        linear_fisher=linear_fisher,
    )


def _disjoint_endpoint_indices(
    theta: np.ndarray,
    *,
    theta_left: float,
    theta_right: float,
    radius: float,
    min_endpoint_samples: int,
    period: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint local endpoint samples, expanding one-sided if needed."""

    midpoint = 0.5 * (float(theta_left) + float(theta_right))
    local_theta = np.asarray(theta, dtype=np.float64)
    if period is not None:
        if float(period) <= 0.0:
            raise ValueError("period must be positive.")
        local_theta = midpoint + np.mod(
            local_theta - midpoint + 0.5 * float(period),
            float(period),
        ) - 0.5 * float(period)
    left = np.flatnonzero(
        (np.abs(local_theta - float(theta_left)) <= float(radius) + 1e-12)
        & (local_theta <= midpoint)
    )
    right = np.flatnonzero(
        (np.abs(local_theta - float(theta_right)) <= float(radius) + 1e-12)
        & (local_theta > midpoint)
    )
    if min(left.size, right.size) < int(min_endpoint_samples):
        required = 2 * int(min_endpoint_samples)
        if theta.size < required:
            raise ValueError("Not enough observations for disjoint endpoint groups.")
        distance_to_pair = np.minimum(
            np.abs(local_theta - float(theta_left)),
            np.abs(local_theta - float(theta_right)),
        )
        nearest = np.argsort(distance_to_pair, kind="mergesort")[:required]
        ordered = nearest[np.argsort(local_theta[nearest], kind="mergesort")]
        left = ordered[: int(min_endpoint_samples)]
        right = ordered[int(min_endpoint_samples) :]
    if np.intersect1d(left, right).size:
        raise RuntimeError("Endpoint groups must be disjoint.")
    return left, right


def _folds(indices: np.ndarray, *, n_splits: int, rng: np.random.Generator) -> list[np.ndarray]:
    shuffled = rng.permutation(np.asarray(indices, dtype=np.int64))
    return [np.asarray(fold, dtype=np.int64) for fold in np.array_split(shuffled, n_splits)]


def cross_fitted_ole_linear_fisher(
    theta_all: np.ndarray,
    x_all: np.ndarray,
    theta_grid: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 0,
    window_radius: float | None = None,
    min_endpoint_samples: int = 8,
    period: float | None = None,
) -> CrossFittedOLEResult:
    r"""Fit local OLE decoders and evaluate every response out of fold.

    Each fold estimates endpoint means and Ledoit--Wolf covariances on its
    training observations.  The resulting decoder is calibrated to predict
    local condition values, frozen, and applied to that fold's held-out
    responses.  Pooled held-out projections give the bias-reduced achieved
    information

    .. math::

        \frac{(\bar y_R-\bar y_L)^2-s_L^2/n_L-s_R^2/n_R}
        {h^2(s_L^2+s_R^2)/2}.
    """

    theta = np.asarray(theta_all, dtype=np.float64).reshape(-1)
    x = np.asarray(x_all, dtype=np.float64)
    grid = np.asarray(theta_grid, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != theta.shape[0]:
        raise ValueError("x_all must have shape (n_samples, response_dim).")
    if grid.size < 2 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("theta_grid must be strictly increasing.")
    if int(n_splits) < 2:
        raise ValueError("n_splits must be at least 2.")
    if int(min_endpoint_samples) < int(n_splits):
        raise ValueError("min_endpoint_samples must be at least n_splits.")
    if window_radius is None:
        radius = 0.5 * float(np.min(np.diff(grid)))
    else:
        radius = float(window_radius)
    if radius <= 0.0:
        raise ValueError("window_radius must be positive.")

    n_pairs = int(grid.size - 1)
    response_dim = int(x.shape[1])
    weights = np.empty((n_pairs, int(n_splits), response_dim), dtype=np.float64)
    intercepts = np.empty((n_pairs, int(n_splits)), dtype=np.float64)
    test_counts_left = np.empty((n_pairs, int(n_splits)), dtype=np.int64)
    test_counts_right = np.empty((n_pairs, int(n_splits)), dtype=np.int64)
    mean_left = np.empty(n_pairs, dtype=np.float64)
    mean_right = np.empty(n_pairs, dtype=np.float64)
    variance_left = np.empty(n_pairs, dtype=np.float64)
    variance_right = np.empty(n_pairs, dtype=np.float64)
    n_left = np.empty(n_pairs, dtype=np.int64)
    n_right = np.empty(n_pairs, dtype=np.int64)
    raw = np.empty(n_pairs, dtype=np.float64)
    rng = np.random.default_rng(int(seed))

    for pair_index, (theta_left, theta_right) in enumerate(
        zip(grid[:-1], grid[1:], strict=True)
    ):
        spacing = float(theta_right - theta_left)
        midpoint = 0.5 * (float(theta_left) + float(theta_right))
        indices_left, indices_right = _disjoint_endpoint_indices(
            theta,
            theta_left=float(theta_left),
            theta_right=float(theta_right),
            radius=radius,
            min_endpoint_samples=int(min_endpoint_samples),
            period=period,
        )
        folds_left = _folds(indices_left, n_splits=int(n_splits), rng=rng)
        folds_right = _folds(indices_right, n_splits=int(n_splits), rng=rng)
        heldout_left: list[np.ndarray] = []
        heldout_right: list[np.ndarray] = []

        for fold_index in range(int(n_splits)):
            test_left = folds_left[fold_index]
            test_right = folds_right[fold_index]
            train_left = np.concatenate(
                [fold for index, fold in enumerate(folds_left) if index != fold_index]
            )
            train_right = np.concatenate(
                [fold for index, fold in enumerate(folds_right) if index != fold_index]
            )
            fitted_left = LedoitWolf().fit(x[train_left])
            fitted_right = LedoitWolf().fit(x[train_right])
            local_mean_left = np.asarray(fitted_left.location_, dtype=np.float64)
            local_mean_right = np.asarray(fitted_right.location_, dtype=np.float64)
            derivative = (local_mean_right - local_mean_left) / spacing
            pooled_covariance = 0.5 * (
                np.asarray(fitted_left.covariance_, dtype=np.float64)
                + np.asarray(fitted_right.covariance_, dtype=np.float64)
            )
            fitted_ole = optimal_linear_estimator(derivative, pooled_covariance)
            weight = np.asarray(fitted_ole.weights, dtype=np.float64)
            if not np.any(weight):
                raise ValueError("The fitted OLE direction has zero information.")
            local_midpoint_mean = 0.5 * (local_mean_left + local_mean_right)
            intercept = midpoint - float(weight @ local_midpoint_mean)
            weights[pair_index, fold_index] = weight
            intercepts[pair_index, fold_index] = intercept
            test_counts_left[pair_index, fold_index] = int(test_left.size)
            test_counts_right[pair_index, fold_index] = int(test_right.size)
            heldout_left.append(x[test_left] @ weight + intercept)
            heldout_right.append(x[test_right] @ weight + intercept)

        projection_left = np.concatenate(heldout_left)
        projection_right = np.concatenate(heldout_right)
        n_left[pair_index] = int(projection_left.size)
        n_right[pair_index] = int(projection_right.size)
        mean_left[pair_index] = float(np.mean(projection_left))
        mean_right[pair_index] = float(np.mean(projection_right))
        variance_left[pair_index] = float(np.var(projection_left, ddof=1))
        variance_right[pair_index] = float(np.var(projection_right, ddof=1))
        numerator = (
            (mean_right[pair_index] - mean_left[pair_index]) ** 2
            - variance_left[pair_index] / n_left[pair_index]
            - variance_right[pair_index] / n_right[pair_index]
        )
        pooled_variance = 0.5 * (
            variance_left[pair_index] + variance_right[pair_index]
        )
        raw[pair_index] = numerator / (spacing**2 * max(pooled_variance, 1e-12))

    return CrossFittedOLEResult(
        theta_midpoints=0.5 * (grid[:-1] + grid[1:]),
        linear_fisher_raw=raw,
        linear_fisher=np.maximum(raw, 0.0),
        fold_weights=weights,
        fold_intercepts=intercepts,
        projected_mean_left=mean_left,
        projected_mean_right=mean_right,
        projected_variance_left=variance_left,
        projected_variance_right=variance_right,
        n_left=n_left,
        n_right=n_right,
        fold_test_counts_left=test_counts_left,
        fold_test_counts_right=test_counts_right,
    )
