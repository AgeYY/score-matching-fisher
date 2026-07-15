"""BCI Competition IV Dataset 2a loading and time-resolved EEG features."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.signal import periodogram


CLASS_EVENT_TO_LABEL = {769: 0, 770: 1, 771: 2, 772: 3}
CLASS_NAMES = ("left_hand", "right_hand", "both_feet", "tongue")
EEG_CHANNEL_COUNT = 22
EOG_CHANNEL_INDICES = (22, 23, 24)
EXPECTED_SFREQ = 250.0
EXPECTED_TRIALS = 288
EXPECTED_TRIALS_PER_CLASS = 72
EXPECTED_RUNS = 6
TRIALS_PER_RUN = 48
DEFAULT_BANDS = (("mu", 8.0, 13.0), ("beta", 13.0, 30.0))


@dataclass(frozen=True)
class BCIIV2aTrialTable:
    session_key: str
    cue_samples: np.ndarray
    labels: np.ndarray
    run_ids: np.ndarray
    rejected: np.ndarray
    sfreq: float
    channel_names: tuple[str, ...]
    annotation_counts: dict[str, int]


@dataclass(frozen=True)
class BCIIV2aFeatures:
    session_key: str
    features: np.ndarray
    labels: np.ndarray
    run_ids: np.ndarray
    cue_samples: np.ndarray
    time_centers: np.ndarray
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]


def list_training_recordings(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    paths = sorted(root.glob("A??T.gdf"))
    if len(paths) != 9:
        raise FileNotFoundError(f"Expected 9 A??T.gdf recordings in {root}, found {len(paths)}.")
    return paths


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _annotation_rows(raw: Any) -> list[tuple[int, float, float, str]]:
    rows: list[tuple[int, float, float, str]] = []
    sfreq = float(raw.info["sfreq"])
    for onset, duration, description in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description,
        strict=True,
    ):
        rows.append((int(round(float(onset) * sfreq)), float(onset), float(duration), str(description)))
    return rows


def load_trial_table(path: str | Path) -> BCIIV2aTrialTable:
    """Read GDF annotations and construct the 288-trial table.

    Motor-task run ids are assigned from cue order in groups of 48. This is
    robust to A04T's shorter calibration block, which changes the number of
    pre-task ``32766`` annotations.
    """

    import mne

    recording = Path(path)
    raw = mne.io.read_raw_gdf(
        recording,
        eog=list(EOG_CHANNEL_INDICES),
        preload=False,
        verbose="ERROR",
    )
    sfreq = float(raw.info["sfreq"])
    if not np.isclose(sfreq, EXPECTED_SFREQ):
        raise ValueError(f"{recording.name}: expected {EXPECTED_SFREQ} Hz, got {sfreq}.")
    if len(raw.ch_names) != EEG_CHANNEL_COUNT + len(EOG_CHANNEL_INDICES):
        raise ValueError(f"{recording.name}: expected 25 channels, got {len(raw.ch_names)}.")

    rows = _annotation_rows(raw)
    annotation_counts: dict[str, int] = {}
    for _, _, _, description in rows:
        annotation_counts[description] = annotation_counts.get(description, 0) + 1

    cues: list[tuple[int, int]] = []
    trial_starts: set[int] = set()
    rejected_starts: set[int] = set()
    for sample, _, _, description in rows:
        try:
            code = int(description)
        except ValueError:
            continue
        if code in CLASS_EVENT_TO_LABEL:
            cues.append((sample, CLASS_EVENT_TO_LABEL[code]))
        elif code == 768:
            trial_starts.add(sample)
        elif code == 1023:
            rejected_starts.add(sample)
    cues.sort(key=lambda item: item[0])
    if len(cues) != EXPECTED_TRIALS:
        raise ValueError(f"{recording.name}: expected {EXPECTED_TRIALS} cues, found {len(cues)}.")

    cue_samples = np.asarray([item[0] for item in cues], dtype=np.int64)
    labels = np.asarray([item[1] for item in cues], dtype=np.int64)
    expected_offset = int(round(2.0 * sfreq))
    start_samples = cue_samples - expected_offset
    if any(int(sample) not in trial_starts for sample in start_samples):
        raise ValueError(f"{recording.name}: at least one cue lacks a trial-start event exactly 2 s earlier.")
    rejected = np.asarray([int(sample) in rejected_starts for sample in start_samples], dtype=bool)
    run_ids = np.arange(EXPECTED_TRIALS, dtype=np.int64) // TRIALS_PER_RUN

    class_counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    if class_counts.tolist() != [EXPECTED_TRIALS_PER_CLASS] * len(CLASS_NAMES):
        raise ValueError(f"{recording.name}: unexpected class counts {class_counts.tolist()}.")
    for run_id in range(EXPECTED_RUNS):
        mask = run_ids == run_id
        counts = np.bincount(labels[mask], minlength=len(CLASS_NAMES))
        if counts.tolist() != [12, 12, 12, 12]:
            raise ValueError(f"{recording.name}: run {run_id + 1} class counts are {counts.tolist()}.")

    return BCIIV2aTrialTable(
        session_key=recording.stem,
        cue_samples=cue_samples,
        labels=labels,
        run_ids=run_ids,
        rejected=rejected,
        sfreq=sfreq,
        channel_names=tuple(raw.ch_names),
        annotation_counts=annotation_counts,
    )


def _window_starts(tmin: float, tmax: float, window_seconds: float, step_seconds: float) -> np.ndarray:
    if window_seconds <= 0.0 or step_seconds <= 0.0:
        raise ValueError("window_seconds and step_seconds must be positive.")
    last = float(tmax) - float(window_seconds)
    if last < float(tmin):
        raise ValueError("Window is longer than the requested epoch.")
    count = int(np.floor((last - float(tmin)) / float(step_seconds) + 1e-9)) + 1
    return float(tmin) + np.arange(count, dtype=np.float64) * float(step_seconds)


def extract_log_bandpower_features(
    path: str | Path,
    *,
    tmin: float = -2.0,
    tmax: float = 4.0,
    window_seconds: float = 1.0,
    step_seconds: float = 0.25,
    bands: Sequence[tuple[str, float, float]] = DEFAULT_BANDS,
    drop_rejected: bool = True,
) -> BCIIV2aFeatures:
    """Extract CAR log-bandpower features for every clean trial and window."""

    import mne

    recording = Path(path)
    table = load_trial_table(recording)
    raw = mne.io.read_raw_gdf(
        recording,
        eog=list(EOG_CHANNEL_INDICES),
        preload=False,
        verbose="ERROR",
    )
    sfreq = float(table.sfreq)
    starts = _window_starts(tmin, tmax, window_seconds, step_seconds)
    centers = starts + 0.5 * float(window_seconds)
    window_n = int(round(float(window_seconds) * sfreq))
    epoch_start_offset = int(round(float(tmin) * sfreq))
    epoch_stop_offset = int(round(float(tmax) * sfreq))
    epoch_n = epoch_stop_offset - epoch_start_offset
    start_offsets = np.asarray(np.round((starts - float(tmin)) * sfreq), dtype=np.int64)
    if np.any(start_offsets + window_n > epoch_n):
        raise ValueError("A feature window extends beyond the requested epoch.")

    keep = ~table.rejected if bool(drop_rejected) else np.ones_like(table.rejected, dtype=bool)
    kept_indices = np.flatnonzero(keep)
    n_features = EEG_CHANNEL_COUNT * len(bands)
    features = np.empty((kept_indices.size, starts.size, n_features), dtype=np.float64)
    feature_names = tuple(
        f"{table.channel_names[channel]}_{band_name}"
        for band_name, _, _ in bands
        for channel in range(EEG_CHANNEL_COUNT)
    )

    freq = np.fft.rfftfreq(window_n, d=1.0 / sfreq)
    band_masks: list[np.ndarray] = []
    for band_index, (_, low, high) in enumerate(bands):
        if not (0.0 <= float(low) < float(high) <= sfreq / 2.0):
            raise ValueError(f"Invalid band {(low, high)} for sampling rate {sfreq}.")
        if band_index < len(bands) - 1:
            mask = (freq >= float(low)) & (freq < float(high))
        else:
            mask = (freq >= float(low)) & (freq <= float(high))
        if np.sum(mask) < 2:
            raise ValueError(f"Band {(low, high)} has fewer than two Fourier bins.")
        band_masks.append(mask)

    for out_index, trial_index in enumerate(kept_indices):
        cue = int(table.cue_samples[trial_index])
        start_sample = cue + epoch_start_offset
        stop_sample = cue + epoch_stop_offset
        if start_sample < 0 or stop_sample > int(raw.n_times):
            raise ValueError(f"{recording.name}: trial {trial_index} epoch is outside the recording.")
        eeg = raw.get_data(
            picks=np.arange(EEG_CHANNEL_COUNT),
            start=start_sample,
            stop=stop_sample,
            reject_by_annotation=None,
            verbose="ERROR",
        ).astype(np.float64, copy=False)
        if eeg.shape != (EEG_CHANNEL_COUNT, epoch_n):
            raise ValueError(f"{recording.name}: unexpected epoch shape {eeg.shape}.")
        if not np.all(np.isfinite(eeg)):
            raise ValueError(f"{recording.name}: non-finite EEG inside trial {trial_index}.")
        eeg = eeg - np.mean(eeg, axis=0, keepdims=True)
        for time_index, offset in enumerate(start_offsets):
            window = eeg[:, int(offset) : int(offset) + window_n]
            got_freq, psd = periodogram(
                window,
                fs=sfreq,
                window="hann",
                detrend="constant",
                scaling="density",
                axis=-1,
            )
            if not np.allclose(got_freq, freq):
                raise RuntimeError("Unexpected periodogram frequency grid.")
            values: list[np.ndarray] = []
            for mask in band_masks:
                power = np.trapezoid(psd[:, mask], freq[mask], axis=1)
                values.append(np.log(np.maximum(power, np.finfo(np.float64).tiny)))
            features[out_index, time_index] = np.concatenate(values, axis=0)

    clean_counts = np.zeros((EXPECTED_RUNS, len(CLASS_NAMES)), dtype=np.int64)
    for run_id in range(EXPECTED_RUNS):
        for label in range(len(CLASS_NAMES)):
            clean_counts[run_id, label] = int(np.sum(keep & (table.run_ids == run_id) & (table.labels == label)))
    metadata = {
        "recording": str(recording.resolve()),
        "session_key": table.session_key,
        "sampling_rate_hz": sfreq,
        "n_total_trials": int(table.labels.size),
        "n_rejected_trials": int(np.sum(table.rejected)),
        "n_clean_trials": int(np.sum(keep)),
        "clean_counts_by_run_class": clean_counts.tolist(),
        "class_names": list(CLASS_NAMES),
        "feature_kind": "car_log_bandpower",
        "bands_hz": [[str(name), float(low), float(high)] for name, low, high in bands],
        "tmin_seconds": float(tmin),
        "tmax_seconds": float(tmax),
        "window_seconds": float(window_seconds),
        "step_seconds": float(step_seconds),
        "drop_rejected": bool(drop_rejected),
    }
    return BCIIV2aFeatures(
        session_key=table.session_key,
        features=features,
        labels=table.labels[keep].copy(),
        run_ids=table.run_ids[keep].copy(),
        cue_samples=table.cue_samples[keep].copy(),
        time_centers=centers,
        feature_names=feature_names,
        metadata=metadata,
    )


def save_features_npz(path: str | Path, features: BCIIV2aFeatures) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        session_key=np.asarray([features.session_key]),
        features=np.asarray(features.features, dtype=np.float64),
        labels=np.asarray(features.labels, dtype=np.int64),
        run_ids=np.asarray(features.run_ids, dtype=np.int64),
        cue_samples=np.asarray(features.cue_samples, dtype=np.int64),
        time_centers=np.asarray(features.time_centers, dtype=np.float64),
        feature_names=np.asarray(features.feature_names),
        metadata_json=np.asarray([json.dumps(features.metadata, sort_keys=True)]),
    )
    return output


def load_features_npz(path: str | Path) -> BCIIV2aFeatures:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        return BCIIV2aFeatures(
            session_key=str(np.asarray(data["session_key"]).reshape(-1)[0]),
            features=np.asarray(data["features"], dtype=np.float64),
            labels=np.asarray(data["labels"], dtype=np.int64),
            run_ids=np.asarray(data["run_ids"], dtype=np.int64),
            cue_samples=np.asarray(data["cue_samples"], dtype=np.int64),
            time_centers=np.asarray(data["time_centers"], dtype=np.float64),
            feature_names=tuple(str(value) for value in np.asarray(data["feature_names"]).reshape(-1)),
            metadata=json.loads(str(np.asarray(data["metadata_json"]).reshape(-1)[0])),
        )
