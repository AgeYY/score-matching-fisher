"""Dual-criterion diagnostics for synthetic Fisher-curve estimators."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fisher.stringer_session_identification import (
    DISTANCES,
    curve_distance,
)


METHOD_FLOW = "Flow matching"
METHOD_GKR = "GKR"
TOY_IDENTIFICATION_METHODS = (METHOD_FLOW, METHOD_GKR)
HALF_LABELS = ("A", "B")


def fisher_mae(
    estimates: np.ndarray,
    ground_truth: np.ndarray,
) -> np.ndarray:
    """Return MAE for estimates shaped ``(..., session, theta)``."""
    estimate_array = np.asarray(estimates, dtype=np.float64)
    truth_array = np.asarray(ground_truth, dtype=np.float64)
    if estimate_array.ndim < 2 or truth_array.ndim != 2:
        raise ValueError("estimates must be at least 2D and ground_truth must be 2D.")
    if estimate_array.shape[-2:] != truth_array.shape:
        raise ValueError(
            "The final estimate axes must match ground_truth (session, theta); "
            f"got {estimate_array.shape[-2:]} and {truth_array.shape}."
        )
    return np.mean(np.abs(estimate_array - truth_array), axis=-1)


def identification_matrix(
    half_a: np.ndarray,
    half_b: np.ndarray,
    theta_midpoints: np.ndarray,
    *,
    distance: str,
) -> np.ndarray:
    """Compute A-query to B-reference distances between synthetic sessions."""
    query = np.asarray(half_a, dtype=np.float64)
    reference = np.asarray(half_b, dtype=np.float64)
    if query.ndim != 2 or reference.shape != query.shape:
        raise ValueError("half_a and half_b must have matching [session, theta] shapes.")
    return np.asarray(
        [
            [
                curve_distance(
                    query[query_index],
                    reference[reference_index],
                    theta_midpoints,
                    distance=distance,
                )
                for reference_index in range(reference.shape[0])
            ]
            for query_index in range(query.shape[0])
        ],
        dtype=np.float64,
    )


def identification_summary(matrix: np.ndarray) -> dict[str, object]:
    """Summarize ranks and margins from a square identification matrix."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square.")
    n_sessions = int(values.shape[0])
    ranks = np.empty(n_sessions, dtype=np.int64)
    margins = np.empty(n_sessions, dtype=np.float64)
    for query_index in range(n_sessions):
        row = values[query_index]
        order = np.argsort(row, kind="mergesort")
        ranks[query_index] = int(np.flatnonzero(order == query_index)[0]) + 1
        wrong = np.delete(row, query_index)
        margins[query_index] = float(np.min(wrong) - row[query_index])
    return {
        "top1_accuracy": float(np.mean(ranks == 1)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "ranks": ranks,
        "correct_minus_best_wrong_margin": margins,
    }


def evaluate_identification(
    estimates: Mapping[str, np.ndarray],
    theta_midpoints: np.ndarray,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, dict[str, object]]]]:
    """Evaluate methods whose arrays have shape ``[session, half, theta]``."""
    matrices: dict[str, dict[str, np.ndarray]] = {}
    summaries: dict[str, dict[str, dict[str, object]]] = {}
    for method, values_raw in estimates.items():
        values = np.asarray(values_raw, dtype=np.float64)
        if values.ndim != 3 or values.shape[1] != 2:
            raise ValueError(f"{method} estimates must have shape [session, 2, theta].")
        matrices[method] = {}
        summaries[method] = {}
        for distance in DISTANCES:
            matrix = identification_matrix(
                values[:, 0],
                values[:, 1],
                theta_midpoints,
                distance=distance,
            )
            matrices[method][distance] = matrix
            summaries[method][distance] = identification_summary(matrix)
    return matrices, summaries
