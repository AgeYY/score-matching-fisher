#!/usr/bin/env python3
"""Fit and plot no-binning flow temporal RDMs for one BCI IV-2a run/class."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from global_setting import (  # noqa: E402
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_TRAINING_MAX_EPOCHS,
)
from fisher.bci_iv_2a_dataset import (  # noqa: E402
    CLASS_NAMES,
    EEG_CHANNEL_COUNT,
    EOG_CHANNEL_INDICES,
    load_trial_table,
)
from fisher.bci_iv_2a_temporal_rdm import (  # noqa: E402
    FlowTemporalRDMConfig,
    TEMPORAL_RDM_METRICS,
    fit_native_time_affine_flow_rdms,
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
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--class-name", choices=CLASS_NAMES, default="left_hand")
    parser.add_argument("--tmin", type=float, default=-2.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--covariance-steps", type=int, default=48)
    parser.add_argument("--covariance-ridge", type=float, default=1e-5)
    parser.add_argument("--fid-block-size", type=int, default=128)
    parser.add_argument("--fid-color-scale", choices=("linear", "sqrt"), default="sqrt")
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Regenerate figures from the existing NPZ without retraining the flow.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/temporal_rdm_examples/A01T_run01_left_hand/flow_no_binning",
    )
    parser.add_argument("--device", required=True)
    return parser.parse_args()


def _validate_cuda_device(device_name: str) -> torch.device:
    device = torch.device(str(device_name))
    if device.type != "cuda" or device.index is None:
        raise ValueError("--device must explicitly select a CUDA device, for example cuda:1.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to silently switch to CPU.")
    if device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {device.index} does not exist.")
    return device


def _load_native_voltage(
    recording: Path,
    *,
    run: int,
    class_name: str,
    tmin: float,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import mne

    if not 1 <= int(run) <= 6:
        raise ValueError("--run must be between 1 and 6.")
    if float(tmax) <= float(tmin):
        raise ValueError("--tmax must be greater than --tmin.")
    table = load_trial_table(recording)
    label = CLASS_NAMES.index(str(class_name))
    mask = (table.run_ids == int(run) - 1) & (table.labels == label) & (~table.rejected)
    trial_indices = np.flatnonzero(mask)
    if trial_indices.size != 12:
        raise ValueError(
            f"Expected 12 clean {class_name} trials in run {run}, found {trial_indices.size}."
        )

    sfreq = float(table.sfreq)
    start_offset = int(round(float(tmin) * sfreq))
    stop_offset = int(round(float(tmax) * sfreq))
    n_samples = stop_offset - start_offset
    raw = mne.io.read_raw_gdf(
        recording,
        eog=list(EOG_CHANNEL_INDICES),
        preload=False,
        verbose="ERROR",
    )
    voltage = np.empty((trial_indices.size, n_samples, EEG_CHANNEL_COUNT), dtype=np.float64)
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
        voltage[output_index] = (trial * 1e6).T
    time_points = float(tmin) + np.arange(n_samples, dtype=np.float64) / sfreq
    metadata: dict[str, Any] = {
        "recording": recording.stem,
        "recording_path": str(recording.resolve()),
        "run_one_based": int(run),
        "class_name": str(class_name),
        "class_label": int(label),
        "n_trials": int(trial_indices.size),
        "trial_indices_zero_based": trial_indices.tolist(),
        "trial_indices_one_based": (trial_indices + 1).tolist(),
        "sampling_rate_hz": sfreq,
        "native_sample_interval_seconds": 1.0 / sfreq,
        "n_native_time_points": int(n_samples),
        "n_eeg_channels": EEG_CHANNEL_COUNT,
        "tmin_seconds_cue_relative": float(tmin),
        "tmax_seconds_cue_relative_exclusive": float(tmax),
        "binning": "none",
        "temporal_averaging": "none",
        "phase_boundaries_seconds": {
            "cue_onset": 0.0,
            "cue_disappearance_imagery_continues": 1.25,
        },
        "feature_definition": "native 22-channel EEG voltage vector at each 4 ms sample",
        "voltage_units": "microvolts",
        "flow_model": "continuous-physical-time centered covariate-affine Gaussian flow",
        "distance_definitions": {
            "correlation": "1 - Pearson correlation between flow endpoint mean vectors",
            "cosine": "1 - cosine similarity between flow endpoint mean vectors",
            "euclidean": "Euclidean distance between flow endpoint mean vectors",
            "fid": "Gaussian Frechet distance between flow endpoint means and full time-specific covariances",
        },
        "operations_before_flow": [
            "left-mastoid acquisition reference",
            "dataset acquisition bandpass from 0.5 to 100 Hz",
            "dataset acquisition 50 Hz notch filter",
            "GDF physical calibration applied by MNE",
            "unit conversion from V to microV",
            "train-trial-only per-channel affine normalization for flow optimization",
        ],
        "operations_not_applied": [
            "temporal binning",
            "temporal averaging",
            "common-average reference",
            "additional filtering",
            "baseline correction",
        ],
    }
    return voltage, time_points, trial_indices, metadata


def _jsonable_training_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    excluded = {"train_losses", "val_losses", "val_monitor_losses", "learning_rates"}
    for key, value in metadata.items():
        if key in excluded:
            continue
        if isinstance(value, np.generic):
            out[key] = value.item()
        elif isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def _plot_rdms(
    rdms: dict[str, np.ndarray],
    *,
    tmin: float,
    tmax: float,
    native_interval_seconds: float,
    fid_color_scale: str,
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
        f"Flow temporal RDMs (native {1000.0 * native_interval_seconds:g} ms samples; no binning)",
        fontsize=15,
    )
    extent = (float(tmin), float(tmax), float(tmin), float(tmax))
    phase_ticks = np.asarray(
        [value for value in PHASE_BOUNDARIES_SECONDS if tmin <= value <= tmax],
        dtype=np.float64,
    )
    ticks = np.unique(np.concatenate([phase_ticks, np.asarray([tmin, tmax])]))
    tick_labels = [f"{value:g}" for value in ticks]
    for axis, metric in zip(axes.reshape(-1), TEMPORAL_RDM_METRICS, strict=True):
        matrix = rdms[metric]
        image_kwargs: dict[str, Any] = {
            "origin": "lower",
            "extent": extent,
            "interpolation": "nearest",
            "aspect": "equal",
            "cmap": "viridis",
        }
        if metric == "fid" and fid_color_scale == "sqrt":
            image_kwargs["norm"] = PowerNorm(gamma=0.5, vmin=0.0, vmax=float(np.max(matrix)))
        else:
            image_kwargs["vmin"] = 0.0
            image_kwargs["vmax"] = float(np.max(matrix))
        image = axis.imshow(matrix, **image_kwargs)
        title = METRIC_TITLES[metric]
        if metric == "fid" and fid_color_scale == "sqrt":
            title += " (sqrt scale)"
        axis.set_title(title)
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
    suffix = "_sqrt_fid_scale" if fid_color_scale == "sqrt" else ""
    png_path = output_dir / f"flow_temporal_rdms_no_binning{suffix}.png"
    svg_path = output_dir / f"flow_temporal_rdms_no_binning{suffix}.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def _plot_fid_only(
    fid_rdm: np.ndarray,
    *,
    tmin: float,
    tmax: float,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Plot raw FID values with a square-root color normalization."""

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
    figure, axis = plt.subplots(figsize=(4.0, 3.5), constrained_layout=True)
    image = axis.imshow(
        fid_rdm,
        origin="lower",
        extent=(float(tmin), float(tmax), float(tmin), float(tmax)),
        interpolation="nearest",
        aspect="equal",
        cmap="viridis",
        norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=float(np.max(fid_rdm))),
    )
    axis.set_title("Gaussian FID (sqrt scale)", fontsize=15)
    axis.set_xlabel("Time from cue onset (s)")
    axis.set_ylabel("Time (s)")
    phase_ticks = np.asarray(
        [value for value in PHASE_BOUNDARIES_SECONDS if tmin <= value <= tmax],
        dtype=np.float64,
    )
    ticks = np.unique(np.concatenate([phase_ticks, np.asarray([tmin, tmax])]))
    tick_labels = [f"{value:g}" for value in ticks]
    axis.set_xticks(ticks, labels=tick_labels)
    axis.set_yticks(ticks, labels=tick_labels)
    for boundary in phase_ticks:
        axis.axvline(boundary, color="white", linestyle="--", linewidth=1.3, alpha=0.95)
        axis.axhline(boundary, color="white", linestyle="--", linewidth=1.3, alpha=0.95)
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(width=1.8)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(r"FID ($\mu$V$^2$)", fontsize=13)
    colorbar.ax.tick_params(labelsize=11, width=1.8)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "flow_gaussian_fid_no_binning_sqrt_scale.png"
    svg_path = output_dir / "flow_gaussian_fid_no_binning_sqrt_scale.svg"
    figure.savefig(png_path, dpi=300)
    figure.savefig(svg_path)
    plt.close(figure)
    return png_path, svg_path


def main() -> None:
    args = parse_args()
    device = _validate_cuda_device(args.device)
    if args.replot_only:
        npz_path = args.output_dir / "flow_temporal_rdms_no_binning.npz"
        metadata_path = args.output_dir / "flow_temporal_rdms_no_binning.json"
        if not npz_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("--replot-only requires the existing flow NPZ and JSON outputs.")
        with np.load(npz_path, allow_pickle=False) as saved:
            rdms = {metric: np.asarray(saved[f"{metric}_rdm"], dtype=np.float64) for metric in TEMPORAL_RDM_METRICS}
        saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        tmin = float(saved_metadata["tmin_seconds_cue_relative"])
        tmax = float(saved_metadata["tmax_seconds_cue_relative_exclusive"])
        native_interval = float(saved_metadata["native_sample_interval_seconds"])
        png_path, svg_path = _plot_rdms(
            rdms,
            tmin=tmin,
            tmax=tmax,
            native_interval_seconds=native_interval,
            fid_color_scale=args.fid_color_scale,
            output_dir=args.output_dir,
        )
        generated = [png_path, svg_path]
        if args.fid_color_scale == "sqrt":
            generated.extend(_plot_fid_only(rdms["fid"], tmin=tmin, tmax=tmax, output_dir=args.output_dir))
        for path in generated:
            print(f"[flow-temporal-rdm replot] Saved: {path.resolve()}", flush=True)
        return
    voltage, time_points, trial_indices, metadata = _load_native_voltage(
        args.recording,
        run=args.run,
        class_name=args.class_name,
        tmin=args.tmin,
        tmax=args.tmax,
    )
    config = FlowTemporalRDMConfig(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        covariance_steps=args.covariance_steps,
        covariance_ridge=args.covariance_ridge,
        fid_block_size=args.fid_block_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result, model = fit_native_time_affine_flow_rdms(
        voltage,
        time_points,
        device=device,
        seed=args.seed,
        config=config,
    )

    metadata.update(
        {
            "device": str(device),
            "seed": int(args.seed),
            "flow_config": asdict(config),
            "train_trial_indices_within_selected_subset": result.train_trial_indices.tolist(),
            "validation_trial_indices_within_selected_subset": result.validation_trial_indices.tolist(),
            "condition_scale_seconds": float(result.condition_scale),
            "training_summary": _jsonable_training_summary(result.train_metadata),
        }
    )
    checkpoint_path = args.output_dir / "flow_temporal_rdm_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "flow_config": asdict(config),
            "seed": int(args.seed),
            "time_points_seconds": time_points,
            "condition_scale_seconds": float(result.condition_scale),
            "x_normalization_mean": result.x_normalization_mean,
            "x_normalization_std": result.x_normalization_std,
            "metadata": metadata,
        },
        checkpoint_path,
    )
    metadata_path = args.output_dir / "flow_temporal_rdms_no_binning.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    npz_path = args.output_dir / "flow_temporal_rdms_no_binning.npz"
    np.savez_compressed(
        npz_path,
        native_voltage_microvolts=voltage.astype(np.float32),
        time_points_seconds=time_points,
        trial_indices_zero_based=trial_indices,
        flow_endpoint_means_microvolts=result.means.astype(np.float32),
        flow_endpoint_covariances_microvolts_sq=result.covariances.astype(np.float32),
        correlation_rdm=result.rdms["correlation"].astype(np.float32),
        cosine_rdm=result.rdms["cosine"].astype(np.float32),
        euclidean_rdm=result.rdms["euclidean"].astype(np.float32),
        fid_rdm=result.rdms["fid"].astype(np.float32),
        train_trial_indices=result.train_trial_indices,
        validation_trial_indices=result.validation_trial_indices,
        x_normalization_mean=result.x_normalization_mean,
        x_normalization_std=result.x_normalization_std,
        train_losses=np.asarray(result.train_metadata["train_losses"], dtype=np.float64),
        validation_losses=np.asarray(result.train_metadata["val_losses"], dtype=np.float64),
        monitored_validation_losses=np.asarray(
            result.train_metadata["val_monitor_losses"], dtype=np.float64
        ),
        metadata_json=np.asarray([json.dumps(metadata, sort_keys=True)]),
    )
    png_path, svg_path = _plot_rdms(
        result.rdms,
        tmin=args.tmin,
        tmax=args.tmax,
        native_interval_seconds=1.0 / float(metadata["sampling_rate_hz"]),
        fid_color_scale=args.fid_color_scale,
        output_dir=args.output_dir,
    )
    fid_figure_paths: tuple[Path, Path] | tuple[()] = ()
    if args.fid_color_scale == "sqrt":
        fid_figure_paths = _plot_fid_only(
            result.rdms["fid"],
            tmin=args.tmin,
            tmax=args.tmax,
            output_dir=args.output_dir,
        )
    print(
        f"[flow-temporal-rdm] subset={metadata['recording']} run={args.run} "
        f"class={args.class_name} no_binning=True",
        flush=True,
    )
    print(f"[flow-temporal-rdm] native_voltage_shape={voltage.shape}", flush=True)
    print(
        f"[flow-temporal-rdm] best_epoch={result.train_metadata['best_epoch']} "
        f"stopped_epoch={result.train_metadata['stopped_epoch']} "
        f"best_val={result.train_metadata['best_val_loss']:.8g}",
        flush=True,
    )
    for metric in TEMPORAL_RDM_METRICS:
        matrix = result.rdms[metric]
        print(
            f"[flow-temporal-rdm] {metric}: shape={matrix.shape} max={np.max(matrix):.8g}",
            flush=True,
        )
    for path in (npz_path, metadata_path, checkpoint_path, png_path, svg_path, *fid_figure_paths):
        print(f"[flow-temporal-rdm] Saved: {path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
