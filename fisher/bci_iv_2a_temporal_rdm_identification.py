"""Identification from BCI IV-2a temporal RDMs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from fisher.bci_iv_2a_dataset import (
    CLASS_NAMES,
    EEG_CHANNEL_COUNT,
    EOG_CHANNEL_INDICES,
    load_trial_table,
)
from fisher.bci_iv_2a_temporal_rdm import TEMPORAL_RDM_METRICS


QUERY_RUNS = (0, 2, 4)
REFERENCE_RUNS = (1, 3, 5)
METHODS = ("classical", "flow")
INTERVALS = {
    "full": (-2.0, 4.0),
    "pre_cue": (-2.0, 0.0),
    "visible_cue": (0.0, 1.25),
    "later_imagery": (1.25, 4.0),
}


@dataclass(frozen=True)
class NativeClassVoltage:
    """Native voltage epochs for one class in one recording."""

    recording_key: str
    voltage_microvolts: np.ndarray
    run_ids: np.ndarray
    trial_indices: np.ndarray
    time_points_seconds: np.ndarray
    sfreq: float


def native_evaluation_indices(
    time_points: np.ndarray,
    *,
    step_seconds: float = 0.1,
) -> np.ndarray:
    """Select one native sample near the center of each non-overlapping step."""

    times = np.asarray(time_points, dtype=np.float64).reshape(-1)
    if times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("time_points must be strictly increasing and contain at least two values.")
    if float(step_seconds) <= 0.0:
        raise ValueError("step_seconds must be positive.")
    native_step = float(np.median(np.diff(times)))
    samples_per_step_float = float(step_seconds) / native_step
    samples_per_step = int(round(samples_per_step_float))
    if samples_per_step < 1 or not np.isclose(samples_per_step_float, samples_per_step, atol=1e-8):
        raise ValueError("step_seconds must be an integer multiple of the native sample interval.")
    offset = samples_per_step // 2
    indices = np.arange(offset, times.size, samples_per_step, dtype=np.int64)
    if indices.size < 2:
        raise ValueError("The evaluation grid must contain at least two time points.")
    return indices


def load_native_class_voltage(
    recording: str | Path,
    *,
    class_name: str,
    tmin: float = -2.0,
    tmax: float = 4.0,
) -> NativeClassVoltage:
    """Load all clean native-voltage trials for one motor-imagery class."""

    import mne

    path = Path(recording)
    if str(class_name) not in CLASS_NAMES:
        raise ValueError(f"Unknown class {class_name!r}; expected one of {CLASS_NAMES}.")
    if float(tmax) <= float(tmin):
        raise ValueError("tmax must be greater than tmin.")
    table = load_trial_table(path)
    label = CLASS_NAMES.index(str(class_name))
    keep = (table.labels == label) & (~table.rejected)
    trial_indices = np.flatnonzero(keep)
    if trial_indices.size < 4:
        raise ValueError(f"{path.name}: fewer than four clean {class_name} trials.")

    sfreq = float(table.sfreq)
    start_offset = int(round(float(tmin) * sfreq))
    stop_offset = int(round(float(tmax) * sfreq))
    n_times = stop_offset - start_offset
    raw = mne.io.read_raw_gdf(
        path,
        eog=list(EOG_CHANNEL_INDICES),
        preload=False,
        verbose="ERROR",
    )
    values = np.empty((trial_indices.size, n_times, EEG_CHANNEL_COUNT), dtype=np.float64)
    for out_index, trial_index in enumerate(trial_indices):
        cue = int(table.cue_samples[int(trial_index)])
        trial = raw.get_data(
            picks=np.arange(EEG_CHANNEL_COUNT),
            start=cue + start_offset,
            stop=cue + stop_offset,
            reject_by_annotation=None,
            verbose="ERROR",
        ).astype(np.float64, copy=False)
        if trial.shape != (EEG_CHANNEL_COUNT, n_times):
            raise ValueError(f"{path.name}: trial {trial_index + 1} has shape {trial.shape}.")
        values[out_index] = (1e6 * trial).T
    times = float(tmin) + np.arange(n_times, dtype=np.float64) / sfreq
    return NativeClassVoltage(
        recording_key=path.stem,
        voltage_microvolts=values,
        run_ids=table.run_ids[trial_indices].copy(),
        trial_indices=trial_indices,
        time_points_seconds=times,
        sfreq=sfreq,
    )


def select_runs(data: NativeClassVoltage, runs: Iterable[int]) -> np.ndarray:
    """Select complete run-disjoint trials from a loaded class."""

    requested = np.asarray(tuple(int(value) for value in runs), dtype=np.int64)
    mask = np.isin(data.run_ids, requested)
    selected = data.voltage_microvolts[mask]
    if selected.shape[0] < 2:
        raise ValueError(f"{data.recording_key}: selected runs contain fewer than two trials.")
    return selected


def shuffled_half_split_indices(n_trials: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle all trials once and return disjoint, exhaustive near-equal halves."""

    count = int(n_trials)
    if count < 4:
        raise ValueError("At least four trials are required for a shuffled half split.")
    permutation = np.random.default_rng(int(seed)).permutation(count)
    split = count // 2
    query = np.sort(permutation[:split]).astype(np.int64, copy=False)
    reference = np.sort(permutation[split:]).astype(np.int64, copy=False)
    if abs(int(query.size) - int(reference.size)) > 1:
        raise RuntimeError("Shuffled halves differ by more than one trial.")
    if np.intersect1d(query, reference).size:
        raise RuntimeError("Shuffled query and reference halves overlap.")
    np.testing.assert_array_equal(
        np.sort(np.concatenate([query, reference])),
        np.arange(count, dtype=np.int64),
    )
    return query, reference


def vectorize_temporal_rdm(
    rdm: np.ndarray,
    time_points: np.ndarray,
    *,
    interval: tuple[float, float] | None = None,
    sqrt_values: bool = False,
) -> np.ndarray:
    """Vectorize a strict upper triangle, optionally inside a time interval."""

    matrix = np.asarray(rdm, dtype=np.float64)
    times = np.asarray(time_points, dtype=np.float64).reshape(-1)
    if matrix.shape != (times.size, times.size):
        raise ValueError("rdm shape must match time_points.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rdm contains non-finite values.")
    if interval is None:
        indices = np.arange(times.size, dtype=np.int64)
    else:
        lower, upper = (float(interval[0]), float(interval[1]))
        if upper <= lower:
            raise ValueError("interval upper bound must exceed lower bound.")
        indices = np.flatnonzero((times >= lower) & (times < upper))
    if indices.size < 2:
        raise ValueError("At least two evaluation times are required for matching.")
    submatrix = matrix[np.ix_(indices, indices)]
    vector = submatrix[np.triu_indices(indices.size, k=1)]
    if bool(sqrt_values):
        if np.min(vector) < -1e-8:
            raise ValueError("Cannot square-root a temporal RDM with negative entries.")
        vector = np.sqrt(np.maximum(vector, 0.0))
    return vector


def pearson_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson similarity with explicit constant-vector rejection."""

    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("Similarity vectors must have equal length of at least two.")
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("Cannot correlate a constant RDM vector.")
    return float(np.clip(np.dot(x, y) / denominator, -1.0, 1.0))


def temporal_rdm_score_matrix(
    query_rdms: list[dict[str, np.ndarray]],
    reference_rdms: list[dict[str, np.ndarray]],
    time_points: np.ndarray,
    *,
    metric: str,
    interval: tuple[float, float] | None = None,
) -> np.ndarray:
    """Match every query RDM to every reference RDM."""

    if str(metric) not in TEMPORAL_RDM_METRICS:
        raise ValueError(f"Unknown temporal RDM metric {metric!r}.")
    if len(query_rdms) != len(reference_rdms) or len(query_rdms) < 2:
        raise ValueError("Query and reference banks must have the same size >= 2.")
    sqrt_values = str(metric) == "fid"
    query_vectors = [
        vectorize_temporal_rdm(item[str(metric)], time_points, interval=interval, sqrt_values=sqrt_values)
        for item in query_rdms
    ]
    reference_vectors = [
        vectorize_temporal_rdm(item[str(metric)], time_points, interval=interval, sqrt_values=sqrt_values)
        for item in reference_rdms
    ]
    return np.asarray(
        [[pearson_similarity(query, reference) for reference in reference_vectors] for query in query_vectors],
        dtype=np.float64,
    )


def correct_match_ranks(scores: np.ndarray) -> np.ndarray:
    """Return one-based correct-reference ranks for a square score matrix."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] < 2:
        raise ValueError("scores must be a square matrix with at least two identities.")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores contain non-finite values.")
    ranks = np.empty(values.shape[0], dtype=np.int64)
    for index, row in enumerate(values):
        order = np.argsort(-row, kind="mergesort")
        ranks[index] = int(np.flatnonzero(order == index)[0]) + 1
    return ranks


def correct_match_margins(scores: np.ndarray) -> np.ndarray:
    """Return true-reference similarity minus the strongest competitor."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] < 2:
        raise ValueError("scores must be a square matrix with at least two identities.")
    return np.asarray(
        [values[index, index] - np.max(np.delete(values[index], index)) for index in range(values.shape[0])],
        dtype=np.float64,
    )


def exact_sign_flip_paired(values: np.ndarray) -> float:
    """Exact two-sided sign-flip p-value for paired recording differences."""

    differences = np.asarray(values, dtype=np.float64).reshape(-1)
    if differences.size < 1 or not np.all(np.isfinite(differences)):
        raise ValueError("values must be a non-empty finite vector.")
    observed = abs(float(np.mean(differences)))
    null = np.empty(2**differences.size, dtype=np.float64)
    for bits in range(null.size):
        signs = np.asarray(
            [1.0 if (bits >> index) & 1 else -1.0 for index in range(differences.size)],
            dtype=np.float64,
        )
        null[bits] = abs(float(np.mean(signs * differences)))
    return float(np.mean(null >= observed - 1e-12))
