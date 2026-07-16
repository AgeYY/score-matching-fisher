#!/usr/bin/env python3
"""Plot one native BCI IV-2a trial and its unstandardized band-power features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import periodogram
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANONICAL_CHANNEL_NAMES = (
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
)

from fisher.bci_iv_2a_dataset import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_BANDS,
    EEG_CHANNEL_COUNT,
    EOG_CHANNEL_INDICES,
    load_trial_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recording",
        type=Path,
        default=ROOT / "data/bci_iv_2a/raw/gdf/A01T.gdf",
    )
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--trial-in-run", type=int, default=1)
    parser.add_argument("--tmin", type=float, default=-2.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--step-seconds", type=float, default=0.25)
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
            fontsize=12,
        )
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _extract_native_trial(
    recording: Path,
    *,
    run: int,
    trial_in_run: int,
    tmin: float,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, object]]:
    import mne

    table = load_trial_table(recording)
    if not 1 <= run <= 6:
        raise ValueError("--run must be between 1 and 6.")
    if not 1 <= trial_in_run <= 48:
        raise ValueError("--trial-in-run must be between 1 and 48.")
    trial_index = (run - 1) * 48 + trial_in_run - 1
    raw = mne.io.read_raw_gdf(
        recording,
        eog=list(EOG_CHANNEL_INDICES),
        preload=False,
        verbose="ERROR",
    )
    sfreq = float(raw.info["sfreq"])
    cue_sample = int(table.cue_samples[trial_index])
    start_sample = cue_sample + int(round(tmin * sfreq))
    stop_sample = cue_sample + int(round(tmax * sfreq))
    eeg = raw.get_data(
        picks=np.arange(EEG_CHANNEL_COUNT),
        start=start_sample,
        stop=stop_sample,
        reject_by_annotation=None,
        verbose="ERROR",
    ).astype(np.float64, copy=False)
    times = np.arange(eeg.shape[1], dtype=np.float64) / sfreq + tmin
    metadata: dict[str, object] = {
        "recording": recording.stem,
        "run_one_based": run,
        "trial_in_run_one_based": trial_in_run,
        "trial_global_one_based": trial_index + 1,
        "condition": CLASS_NAMES[int(table.labels[trial_index])],
        "artifact_rejected": bool(table.rejected[trial_index]),
        "cue_sample": cue_sample,
        "sampling_rate_hz": sfreq,
        "tmin_seconds_cue_relative": tmin,
        "tmax_seconds_cue_relative": tmax,
        "native_eeg_units": "V",
        "operations_before_plot": [
            "left-mastoid acquisition reference",
            "dataset acquisition bandpass from 0.5 to 100 Hz",
            "dataset acquisition 50 Hz notch filter",
            "GDF physical calibration applied by MNE",
            "unit conversion from V to microV for the raw display only",
        ],
    }
    return eeg, times, tuple(table.channel_names[:EEG_CHANNEL_COUNT]), metadata


def _native_band_power(
    eeg: np.ndarray,
    *,
    sfreq: float,
    tmin: float,
    tmax: float,
    window_seconds: float,
    step_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window_n = int(round(window_seconds * sfreq))
    starts = np.arange(tmin, tmax - window_seconds + 1e-9, step_seconds)
    centers = starts + 0.5 * window_seconds
    offsets = np.asarray(np.round((starts - tmin) * sfreq), dtype=np.int64)
    freq = np.fft.rfftfreq(window_n, d=1.0 / sfreq)
    values = np.empty(
        (len(DEFAULT_BANDS) * EEG_CHANNEL_COUNT, centers.size),
        dtype=np.float64,
    )
    for time_index, offset in enumerate(offsets):
        window = eeg[:, int(offset) : int(offset) + window_n]
        got_freq, psd = periodogram(
            window,
            fs=sfreq,
            window="hann",
            detrend=False,
            scaling="density",
            axis=-1,
        )
        if not np.allclose(got_freq, freq):
            raise RuntimeError("Unexpected periodogram frequency grid.")
        for band_index, (_, low, high) in enumerate(DEFAULT_BANDS):
            if band_index < len(DEFAULT_BANDS) - 1:
                mask = (freq >= low) & (freq < high)
            else:
                mask = (freq >= low) & (freq <= high)
            power = np.trapezoid(psd[:, mask], freq[mask], axis=1)
            row_start = band_index * EEG_CHANNEL_COUNT
            values[row_start : row_start + EEG_CHANNEL_COUNT, time_index] = power
    half_step = 0.5 * step_seconds
    edges = np.concatenate(
        [np.asarray([centers[0] - half_step]), centers + half_step]
    )
    return values, centers, edges


def _plot(
    *,
    eeg: np.ndarray,
    times: np.ndarray,
    channel_names: tuple[str, ...],
    band_power: np.ndarray,
    window_edges: np.ndarray,
    metadata: dict[str, object],
    output_dir: Path,
) -> tuple[Path, Path]:
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
    fig = plt.figure(figsize=(8.0, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=(0.16, 1.0, 1.8))
    phase_ax = fig.add_subplot(grid[0])
    raw_ax = fig.add_subplot(grid[1], sharex=phase_ax)
    feature_ax = fig.add_subplot(grid[2], sharex=phase_ax)

    tmin = float(metadata["tmin_seconds_cue_relative"])
    tmax = float(metadata["tmax_seconds_cue_relative"])
    _phase_strip(phase_ax, tmin=tmin, tmax=tmax)

    eeg_microvolts = eeg * 1e6
    raw_limit = float(np.max(np.abs(eeg_microvolts)))
    raw_image = raw_ax.imshow(
        eeg_microvolts,
        aspect="auto",
        origin="upper",
        extent=(times[0], times[-1] + 1.0 / float(metadata["sampling_rate_hz"]), EEG_CHANNEL_COUNT - 0.5, -0.5),
        cmap="RdBu_r",
        vmin=-raw_limit,
        vmax=raw_limit,
        interpolation="nearest",
    )
    raw_ax.set_yticks(np.arange(EEG_CHANNEL_COUNT))
    raw_ax.set_yticklabels([name.removeprefix("EEG-") for name in channel_names])
    raw_ax.set_ylabel("EEG channel")
    raw_ax.set_title("Native EEG samples")
    raw_ax.tick_params(axis="x", labelbottom=False)
    raw_colorbar = fig.colorbar(raw_image, ax=raw_ax, pad=0.01, fraction=0.025)
    raw_colorbar.set_label(r"Amplitude ($\mu$V)", fontsize=13)
    raw_colorbar.ax.tick_params(labelsize=11, width=1.5)

    positive = band_power[band_power > 0.0]
    if positive.size != band_power.size:
        raise ValueError("Band powers must be strictly positive for logarithmic display.")
    row_edges = np.arange(band_power.shape[0] + 1, dtype=np.float64)
    feature_image = feature_ax.pcolormesh(
        window_edges,
        row_edges,
        band_power,
        cmap="viridis",
        norm=LogNorm(vmin=float(positive.min()), vmax=float(positive.max())),
        shading="flat",
    )
    feature_ax.invert_yaxis()
    feature_labels = [
        f"{channel} {band_name}"
        for band_name, _, _ in DEFAULT_BANDS
        for channel in channel_names
    ]
    feature_ax.set_yticks(np.arange(band_power.shape[0]) + 0.5)
    feature_ax.set_yticklabels(feature_labels, fontsize=6.5)
    feature_ax.axhline(EEG_CHANNEL_COUNT, color="white", linewidth=1.8)
    feature_ax.set_ylabel("Channel and band")
    feature_ax.set_xlabel("Time from cue onset (s)")
    feature_ax.set_title(
        "Native-channel band power: mu 8–13 Hz; beta 13–30 Hz"
    )
    feature_colorbar = fig.colorbar(feature_image, ax=feature_ax, pad=0.01, fraction=0.025)
    feature_colorbar.set_label(r"Integrated PSD (V$^2$; log color scale)", fontsize=13)
    feature_colorbar.ax.tick_params(labelsize=11, width=1.5)

    for ax in (raw_ax, feature_ax):
        ax.set_xlim(tmin, tmax)
        for boundary in (0.0, 1.25):
            ax.axvline(boundary, color="white", linestyle="--", linewidth=1.2, alpha=0.9)
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
        ax.tick_params(width=1.8)

    stem = (
        f"{metadata['recording']}_run{int(metadata['run_one_based']):02d}"
        f"_trial{int(metadata['trial_in_run_one_based']):03d}_native_channel_band_features"
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
    eeg, times, gdf_channel_names, metadata = _extract_native_trial(
        args.recording,
        run=args.run,
        trial_in_run=args.trial_in_run,
        tmin=args.tmin,
        tmax=args.tmax,
    )
    band_power, centers, edges = _native_band_power(
        eeg,
        sfreq=float(metadata["sampling_rate_hz"]),
        tmin=args.tmin,
        tmax=args.tmax,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
    )
    metadata.update(
        {
            "gdf_channel_names": list(gdf_channel_names),
            "canonical_eeg_channel_names": list(CANONICAL_CHANNEL_NAMES),
            "gdf_to_canonical_channel_mapping": dict(
                zip(gdf_channel_names, CANONICAL_CHANNEL_NAMES, strict=True)
            ),
            "feature_operations": [
                "one-second Hann-window periodogram",
                "linear integration over mu [8,13) Hz and beta [13,30] Hz",
            ],
            "feature_operations_not_applied": [
                "common-average reference",
                "band-pass filtering",
                "constant detrending",
                "log transform",
                "artifact-trial removal",
                "cross-trial or split-wise standardization",
            ],
            "bands_hz": [list(band) for band in DEFAULT_BANDS],
            "window_seconds": args.window_seconds,
            "step_seconds": args.step_seconds,
            "window_centers_seconds_cue_relative": centers.tolist(),
            "feature_shape": list(band_power.shape),
        }
    )
    png_path, svg_path = _plot(
        eeg=eeg,
        times=times,
        channel_names=CANONICAL_CHANNEL_NAMES,
        band_power=band_power,
        window_edges=edges,
        metadata=metadata,
        output_dir=args.output_dir,
    )
    metadata_path = png_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[raw-trial] condition={metadata['condition']} rejected={metadata['artifact_rejected']}")
    print(f"[raw-trial] raw_shape={eeg.shape} feature_shape={band_power.shape}")
    print(f"[raw-trial] PNG: {png_path}")
    print(f"[raw-trial] SVG: {svg_path}")
    print(f"[raw-trial] metadata: {metadata_path}")


if __name__ == "__main__":
    main()
