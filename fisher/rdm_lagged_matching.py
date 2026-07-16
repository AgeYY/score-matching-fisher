"""Shift-tolerant similarity for time-resolved RDM trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fisher.bci_iv_2a_session_identification import pearson_similarity


@dataclass(frozen=True)
class LaggedCorrelationResult:
    score: float
    lag_samples: int
    zero_lag_same_core_score: float
    n_core_time_points: int


def rdm_upper_triangle_sequence(
    rdms: np.ndarray,
    time_centers: np.ndarray,
    *,
    interval: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the six unique four-condition RDM entries at each selected time."""

    values = np.asarray(rdms, dtype=np.float64)
    times = np.asarray(time_centers, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4):
        raise ValueError("Expected RDMs with shape (time, 4, 4).")
    if times.ndim != 1 or times.size != values.shape[0]:
        raise ValueError("Time centers must match the RDM time dimension.")
    mask = (times >= float(interval[0])) & (times <= float(interval[1]))
    if not np.any(mask):
        raise ValueError(f"No time centers fall in interval {interval}.")
    upper = np.triu_indices(4, k=1)
    return values[mask][:, upper[0], upper[1]], times[mask]


def lagged_pearson_similarity(
    query_sequence: np.ndarray,
    reference_sequence: np.ndarray,
    *,
    max_lag_samples: int,
) -> LaggedCorrelationResult:
    """Maximize flattened Pearson correlation over a bounded constant lag.

    A positive lag compares ``query[t]`` with ``reference[t + lag]``.  Every
    candidate lag uses the same query core and therefore exactly the same
    number of time points.
    """

    query = np.asarray(query_sequence, dtype=np.float64)
    reference = np.asarray(reference_sequence, dtype=np.float64)
    if query.ndim != 2 or reference.ndim != 2 or query.shape != reference.shape:
        raise ValueError("Query and reference sequences must have equal (time, feature) shape.")
    lag_limit = int(max_lag_samples)
    if lag_limit < 0:
        raise ValueError("max_lag_samples must be non-negative.")
    n_times = int(query.shape[0])
    if n_times - 2 * lag_limit < 2:
        raise ValueError("Lag limit leaves fewer than two common-core time points.")

    core_start = lag_limit
    core_stop = n_times - lag_limit
    query_core = query[core_start:core_stop]
    zero_score = pearson_similarity(query_core, reference[core_start:core_stop])

    best_score = -np.inf
    best_lag = 0
    for lag in range(-lag_limit, lag_limit + 1):
        reference_core = reference[core_start + lag : core_stop + lag]
        score = pearson_similarity(query_core, reference_core)
        better_score = score > best_score + 1e-12
        tied_but_simpler = abs(score - best_score) <= 1e-12 and (
            abs(lag), lag
        ) < (abs(best_lag), best_lag)
        if better_score or tied_but_simpler:
            best_score = score
            best_lag = lag

    return LaggedCorrelationResult(
        score=float(best_score),
        lag_samples=int(best_lag),
        zero_lag_same_core_score=float(zero_score),
        n_core_time_points=int(query_core.shape[0]),
    )
