#!/usr/bin/env python3
"""Compute and plot four classical temporal RDMs for one BCI IV-2a run/class."""

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
    CLASS_NAMES,
    EEG_CHANNEL_COUNT,
    EOG_CHANNEL_INDICES,
    load_trial_table,
)
from fisher.bci_iv_2a_temporal_rdm import (  # noqa: E402
    TEMPORAL_RDM_METRICS,
    classical_temporal_rdms,
)


METRIC_TITLES = {
    "correlation": "Correlation distance",
    "cosine": "Cosine distance",
    "euclidean": "Euclidean distance",
    "fid": "Gaussian FID",
}
METRIC_COLORBAR_LABELS = {
    "correlation": r"$1-r$",
    "cosine": r"$1-\cos$",
    "euclidean": r"Distance ($\mu$V)",
    "fid": r"FID ($\mu$V$^2$)",
}
PHASE_BOUNDARIES_SECONDS = (0.0, 1.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recording",
        type=Path,
        default=ROOT / "data/bci_iv_2a/raw/gdf/A01T.gdf",
    )
    parser.add_argument("--run", type=int, default=1, help="One-based motor-task run index.")
    parser.add_argument("--class-name", choices=CLASS_NAMES, default="left_hand")
    parser.add_argument("--tmin", type=float, default=-2.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--bin-width", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/temporal_rdm_examples/A01T_run01_left_hand",
    )
    parser.add_argument("--device", required=True)
    return parser.parse_args()


def _validate_cuda_device(device: str) -> None:
    import torch

    selected = torch.device(str(device))
    if selected.type != "cuda" or selected.index is None:
        raise ValueError("--device must explicitly select a CUDA device, for example cuda:1.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to silently switch to CPU.")
    if selected.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {selected.index} does not exist.")


def _load_binned_voltage(
    recording: Path,
    *,
    run: int,
    class_name: str,
    tmin: float,
    tmax: float,
    bin_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    import mne

    if not 1 <= int(run) <= 6:
        raise ValueError("--run must be between 1 and 6.")
    if float(tmax) <= float(tmin):
        raise ValueError("--tmax must be greater than --tmin.")
    if float(bin_width) <= 0.0:
        raise ValueError("--bin-width must be positive.")

    table = load_trial_table(recording)
    label = CLASS_NAMES.index(str(class_name))
    mask = (table.run_ids == int(run) - 1) & (table.labels == label) & (~table.rejected)
    trial_indices = np.flatnonzero(mask)
    if trial_indices.size != 12:
        raise ValueError(
            f"Expected 12 clean {class_name} trials in run {run}, found {trial_indices.size}."
        )

    sfreq = float(table.sfreq)
    samples_per_bin_float = float(bin_width) * sfreq
    samples_per_bin = int(round(samples_per_bin_float))
    if not np.isclose(samples_per_bin_float, samples_per_bin):
        raise ValueError("--bin-width must correspond to an integer number of native samples.")
    start_offset = int(round(float(tmin) * sfreq))
    stop_offset = int(round(float(tmax) * sfreq))
    n_samples = stop_offset - start_offset
    if n_samples % samples_per_bin != 0:
        raise ValueError("The requested epoch must contain an integer number of time bins.")
    n_bins = n_samples // samples_per_bin

    raw = mne.io.read_raw_gdf(
        recording,
        eog=list(EOG_CHANNEL_INDICES),
        preload=False,
        verbose="ERROR",
    )
    native = np.empty((trial_indices.size, EEG_CHANNEL_COUNT, n_samples), dtype=np.float64)
    for output_index, trial_index in enumerate(trial_indices):
        cue = int(table.cue_samples[int(trial_index)])
        trial = raw.get_data(
            picks=np.arange(EEG_CHANNEL_COUNT),
            start=cue + start_offset,
            stop=cue + stop_offset,
            reject_by_annotation=None,
            verbose="ERROR",
        ).astype(np.float64, copy=False)
        if trial.shape != (EEG_CHANNEL_COUNT, n_samples):
            raise ValueError(f"Trial {trial_index + 1} has unexpected shape {trial.shape}.")
        native[output_index] = trial * 1e6

    binned = native.reshape(
        trial_indices.size,
        EEG_CHANNEL_COUNT,
        n_bins,
        samples_per_bin,
    ).mean(axis=-1).transpose(0, 2, 1)
    edges = float(tmin) + np.arange(n_bins + 1, dtype=np.float64) * float(bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    metadata: dict[str, object] = {
        "recording": recording.stem,
        "recording_path": str(recording.resolve()),
        "run_one_based": int(run),
        "class_name": str(class_name),
        "class_label": int(label),
        "n_trials": int(trial_indices.size),
        "trial_indices_zero_based": trial_indices.tolist(),
        "trial_indices_one_based": (trial_indices + 1).tolist(),
        "sampling_rate_hz": sfreq,
        "n_eeg_channels": EEG_CHANNEL_COUNT,
        "tmin_seconds_cue_relative": float(tmin),
        "tmax_seconds_cue_relative": float(tmax),
        "bin_width_seconds": float(bin_width),
        "samples_per_bin": int(samples_per_bin),
        "n_time_bins": int(n_bins),
        "phase_boundaries_seconds": {
            "cue_onset": 0.0,
            "cue_disappearance_imagery_continues": 1.25,
        },
        "feature_definition": "mean native EEG voltage within each time bin, one 22D vector per trial",
        "voltage_units": "microvolts",
        "covariance_estimator": "separate Ledoit-Wolf covariance in every time bin",
        "distance_definitions": {
            "correlation": "1 - Pearson correlation between trial-mean 22D vectors",
            "cosine": "1 - cosine similarity between trial-mean 22D vectors",
            "euclidean": "Euclidean distance between trial-mean 22D vectors",
            "fid": "Gaussian Frechet distance using separate mean and covariance per time bin",
        },
        "operations_before_rdm": [
            "left-mastoid acquisition reference",
            "dataset acquisition bandpass from 0.5 to 100 Hz",
            "dataset acquisition 50 Hz notch filter",
            "GDF physical calibration applied by MNE",
            "unit conversion from V to microV",
            f"non-overlapping {1000.0 * float(bin_width):g} ms mean-voltage bins",
        ],
        "operations_not_applied": [
            "common-average reference",
            "additional filtering",
            "baseline correction",
            "cross-trial or cross-channel standardization",
        ],
    }
    return binned, centers, edges, trial_indices, metadata


def _plot_rdms(
    rdms: dict[str, np.ndarray],
    *,
    time_edges: np.ndarray,
    bin_width_seconds: float,
    output_dir: Path,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0), constrained_layout=True)
    fig.suptitle(
        f"Classical temporal RDMs ({1000.0 * float(bin_width_seconds):g} ms non-overlapping bins)",
        fontsize=15,
    )
    extent = (
        float(time_edges[0]),
        float(time_edges[-1]),
        float(time_edges[0]),
        float(time_edges[-1]),
    )
    phase_ticks = np.asarray(
        [value for value in PHASE_BOUNDARIES_SECONDS if time_edges[0] <= value <= time_edges[-1]],
        dtype=np.float64,
    )
    ticks = np.unique(np.concatenate([phase_ticks, time_edges[[0, -1]]]))
    tick_labels = [f"{value:g}" for value in ticks]
    for axis, metric in zip(axes.reshape(-1), TEMPORAL_RDM_METRICS, strict=True):
        matrix = rdms[metric]
        image = axis.imshow(
            matrix,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            aspect="equal",
            cmap="viridis",
            vmin=0.0,
            vmax=float(np.max(matrix)),
        )
        axis.set_title(METRIC_TITLES[metric])
        axis.set_xlabel("Time from cue onset (s)")
        axis.set_ylabel("Time from cue onset (s)")
        axis.set_xticks(ticks, labels=tick_labels)
        axis.set_yticks(ticks, labels=tick_labels)
        for boundary in phase_ticks:
            axis.axvline(boundary, color="white", linestyle="--", linewidth=1.3, alpha=0.95)
            axis.axhline(boundary, color="white", linestyle="--", linewidth=1.3, alpha=0.95)
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_linewidth(1.8)
        axis.tick_params(width=1.8)
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(METRIC_COLORBAR_LABELS[metric], fontsize=13)
        colorbar.ax.tick_params(labelsize=11, width=1.8)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "classical_temporal_rdms.png"
    svg_path = output_dir / "classical_temporal_rdms.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    args = parse_args()
    _validate_cuda_device(args.device)
    binned, centers, edges, trial_indices, metadata = _load_binned_voltage(
        args.recording,
        run=args.run,
        class_name=args.class_name,
        tmin=args.tmin,
        tmax=args.tmax,
        bin_width=args.bin_width,
    )
    result = classical_temporal_rdms(binned)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata["device"] = str(args.device)
    metadata_path = args.output_dir / "classical_temporal_rdms.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    npz_path = args.output_dir / "classical_temporal_rdms.npz"
    np.savez_compressed(
        npz_path,
        binned_voltage_microvolts=binned,
        time_centers_seconds=centers,
        time_edges_seconds=edges,
        trial_indices_zero_based=trial_indices,
        mean_voltage_microvolts=result.means,
        covariances_microvolts_sq=result.covariances,
        correlation_rdm=result.rdms["correlation"],
        cosine_rdm=result.rdms["cosine"],
        euclidean_rdm=result.rdms["euclidean"],
        fid_rdm=result.rdms["fid"],
        metadata_json=np.asarray([json.dumps(metadata, sort_keys=True)]),
    )
    png_path, svg_path = _plot_rdms(
        result.rdms,
        time_edges=edges,
        bin_width_seconds=args.bin_width,
        output_dir=args.output_dir,
    )
    print(f"[temporal-rdm] subset={metadata['recording']} run={args.run} class={args.class_name}")
    print(f"[temporal-rdm] binned_voltage_shape={binned.shape}")
    for metric in TEMPORAL_RDM_METRICS:
        matrix = result.rdms[metric]
        print(f"[temporal-rdm] {metric}: shape={matrix.shape} max={np.max(matrix):.8g}")
    print(f"[temporal-rdm] Saved: {npz_path.resolve()}")
    print(f"[temporal-rdm] Saved: {metadata_path.resolve()}")
    print(f"[temporal-rdm] Saved: {png_path.resolve()}")
    print(f"[temporal-rdm] Saved: {svg_path.resolve()}")


if __name__ == "__main__":
    main()
