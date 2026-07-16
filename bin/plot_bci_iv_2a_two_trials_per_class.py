#!/usr/bin/env python3
"""Plot two clean native-voltage trials for each BCI IV-2a class."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher.bci_iv_2a_dataset import (  # noqa: E402
    CANONICAL_EEG_CHANNEL_NAMES,
    CLASS_NAMES,
    EEG_CHANNEL_COUNT,
    EOG_CHANNEL_INDICES,
    load_trial_table,
)


DISPLAY_CLASS_NAMES = {
    "left_hand": "Left hand",
    "right_hand": "Right hand",
    "both_feet": "Both feet",
    "tongue": "Tongue",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recording",
        type=Path,
        default=ROOT / "data/bci_iv_2a/raw/gdf/A01T.gdf",
    )
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--trials-per-class", type=int, default=2)
    parser.add_argument("--tmin", type=float, default=-2.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/data_inspection",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _phase_strip(ax: plt.Axes, *, tmin: float, tmax: float) -> None:
    phases = (
        (-2.0, 0.0, "Fixation + warning", "#d9d9d9"),
        (0.0, 1.25, "Cue + imagery", "#fdb863"),
        (1.25, 4.0, "Imagery only", "#b2abd2"),
    )
    for start, stop, label, color in phases:
        left = max(float(tmin), start)
        right = min(float(tmax), stop)
        if right <= left:
            continue
        ax.axvspan(left, right, color=color, ec="white", lw=1.0)
        ax.text(
            0.5 * (left + right),
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=13,
        )
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _select_trials(
    labels: np.ndarray,
    run_ids: np.ndarray,
    rejected: np.ndarray,
    *,
    run_index: int,
    trials_per_class: int,
) -> np.ndarray:
    selected = np.empty((len(CLASS_NAMES), trials_per_class), dtype=np.int64)
    for label in range(len(CLASS_NAMES)):
        candidates = np.flatnonzero(
            (run_ids == run_index) & (labels == label) & ~rejected
        )
        if candidates.size < trials_per_class:
            raise ValueError(
                f"Run {run_index + 1}, class {CLASS_NAMES[label]} has only "
                f"{candidates.size} clean trials."
            )
        selected[label] = candidates[:trials_per_class]
    return selected


def _load_trials(
    recording: Path,
    *,
    run: int,
    trials_per_class: int,
    tmin: float,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    import mne

    if not 1 <= run <= 6:
        raise ValueError("--run must be between 1 and 6.")
    if trials_per_class < 1:
        raise ValueError("--trials-per-class must be positive.")
    if tmax <= tmin:
        raise ValueError("--tmax must be greater than --tmin.")

    table = load_trial_table(recording)
    selected = _select_trials(
        table.labels,
        table.run_ids,
        table.rejected,
        run_index=run - 1,
        trials_per_class=trials_per_class,
    )
    raw = mne.io.read_raw_gdf(
        recording,
        eog=list(EOG_CHANNEL_INDICES),
        preload=False,
        verbose="ERROR",
    )
    sfreq = float(table.sfreq)
    start_offset = int(round(tmin * sfreq))
    stop_offset = int(round(tmax * sfreq))
    n_samples = stop_offset - start_offset
    voltages = np.empty(
        (len(CLASS_NAMES), trials_per_class, EEG_CHANNEL_COUNT, n_samples),
        dtype=np.float64,
    )
    trial_records: list[dict[str, object]] = []
    for label in range(len(CLASS_NAMES)):
        for example in range(trials_per_class):
            trial_index = int(selected[label, example])
            cue = int(table.cue_samples[trial_index])
            data = raw.get_data(
                picks=np.arange(EEG_CHANNEL_COUNT),
                start=cue + start_offset,
                stop=cue + stop_offset,
                reject_by_annotation=None,
                verbose="ERROR",
            ).astype(np.float64, copy=False)
            if data.shape != (EEG_CHANNEL_COUNT, n_samples):
                raise ValueError(f"Unexpected data shape {data.shape} for trial {trial_index + 1}.")
            voltages[label, example] = data * 1e6
            trial_records.append(
                {
                    "condition": CLASS_NAMES[label],
                    "example_column_one_based": example + 1,
                    "global_trial_one_based": trial_index + 1,
                    "trial_in_run_one_based": trial_index - (run - 1) * 48 + 1,
                    "cue_sample": cue,
                    "artifact_rejected": bool(table.rejected[trial_index]),
                }
            )
    times = np.arange(n_samples, dtype=np.float64) / sfreq + float(tmin)
    metadata: dict[str, object] = {
        "recording": recording.stem,
        "run_one_based": run,
        "selection_rule": "first clean trials within each class in cue order",
        "trials_per_class": trials_per_class,
        "sampling_rate_hz": sfreq,
        "tmin_seconds_cue_relative": float(tmin),
        "tmax_seconds_cue_relative": float(tmax),
        "display_units": "microvolts",
        "canonical_eeg_channel_names": list(CANONICAL_EEG_CHANNEL_NAMES),
        "trial_records": trial_records,
        "operations_before_plot": [
            "left-mastoid acquisition reference",
            "dataset acquisition bandpass from 0.5 to 100 Hz",
            "dataset acquisition 50 Hz notch filter",
            "GDF physical calibration applied by MNE",
            "unit conversion from V to microV for display only",
        ],
        "operations_not_applied": [
            "common-average reference",
            "additional filtering",
            "baseline correction",
            "temporal smoothing or averaging",
            "spectral or band-power transform",
            "cross-trial standardization",
        ],
    }
    return voltages, times, selected, metadata


def _plot(
    voltages: np.ndarray,
    times: np.ndarray,
    selected: np.ndarray,
    metadata: dict[str, object],
    output_dir: Path,
) -> tuple[Path, Path]:
    n_classes, trials_per_class = voltages.shape[:2]
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 8,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig = plt.figure(
        figsize=(4.0 * trials_per_class, 3.5 * n_classes),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        n_classes + 1,
        trials_per_class,
        height_ratios=(0.16, *([1.0] * n_classes)),
    )
    phase_ax = fig.add_subplot(grid[0, :])
    tmin = float(metadata["tmin_seconds_cue_relative"])
    tmax = float(metadata["tmax_seconds_cue_relative"])
    _phase_strip(phase_ax, tmin=tmin, tmax=tmax)

    axes = np.empty((n_classes, trials_per_class), dtype=object)
    limit = float(np.max(np.abs(voltages)))
    image = None
    run = int(metadata["run_one_based"])
    for label in range(n_classes):
        for example in range(trials_per_class):
            ax = fig.add_subplot(grid[label + 1, example])
            axes[label, example] = ax
            image = ax.imshow(
                voltages[label, example],
                aspect="auto",
                origin="upper",
                extent=(
                    times[0],
                    times[-1] + 1.0 / float(metadata["sampling_rate_hz"]),
                    EEG_CHANNEL_COUNT - 0.5,
                    -0.5,
                ),
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
            )
            trial_in_run = int(selected[label, example]) - (run - 1) * 48 + 1
            ax.set_title(f"Example {example + 1} (run trial {trial_in_run})")
            for boundary in (0.0, 1.25):
                ax.axvline(boundary, color="white", linestyle="--", linewidth=1.2, alpha=0.9)
            ax.set_xlim(tmin, tmax)
            ax.set_yticks(np.arange(EEG_CHANNEL_COUNT))
            if example == 0:
                ax.set_yticklabels(CANONICAL_EEG_CHANNEL_NAMES)
                class_name = DISPLAY_CLASS_NAMES[CLASS_NAMES[label]]
                ax.set_ylabel(f"{class_name}\nEEG channel")
            else:
                ax.set_yticklabels([])
            if label == n_classes - 1:
                ax.set_xlabel("Time from cue onset (s)")
            else:
                ax.tick_params(axis="x", labelbottom=False)
            for spine in ax.spines.values():
                spine.set_linewidth(1.8)
            ax.tick_params(width=1.8)

    if image is None:
        raise RuntimeError("No panels were plotted.")
    colorbar = fig.colorbar(
        image,
        ax=axes.reshape(-1).tolist(),
        pad=0.015,
        fraction=0.025,
        shrink=0.75,
    )
    colorbar.set_label(r"Amplitude ($\mu$V)")
    colorbar.ax.tick_params(labelsize=12, width=1.5)

    stem = (
        f"{metadata['recording']}_run{run:02d}_two_clean_trials_per_class_native_voltage"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    args = parse_args()
    if args.device != "cuda:0":
        raise ValueError("Project commands must specify --device cuda:0.")
    voltages, times, selected, metadata = _load_trials(
        args.recording,
        run=args.run,
        trials_per_class=args.trials_per_class,
        tmin=args.tmin,
        tmax=args.tmax,
    )
    png_path, svg_path = _plot(
        voltages,
        times,
        selected,
        metadata,
        args.output_dir,
    )
    metadata_path = png_path.with_suffix(".json")
    metadata["shared_color_limit_microvolts"] = float(np.max(np.abs(voltages)))
    metadata["voltage_array_shape"] = list(voltages.shape)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[two-trials] voltage_shape={voltages.shape}")
    for record in metadata["trial_records"]:
        print(
            f"[two-trials] {record['condition']} example={record['example_column_one_based']} "
            f"run_trial={record['trial_in_run_one_based']} "
            f"global_trial={record['global_trial_one_based']}",
            flush=True,
        )
    print(f"[two-trials] PNG: {png_path}")
    print(f"[two-trials] SVG: {svg_path}")
    print(f"[two-trials] metadata: {metadata_path}")


if __name__ == "__main__":
    main()
