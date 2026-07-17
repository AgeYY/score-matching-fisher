#!/usr/bin/env python3
"""Plot mean RDM distance over EEG time for all metrics and recordings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher.bci_iv_2a_dataset import load_features_npz


METRICS = ("correlation", "cosine", "euclidean", "mahalanobis", "fid", "jeffreys")
METRIC_LABELS = {
    "correlation": "Correlation",
    "cosine": "Cosine",
    "euclidean": "Euclidean",
    "mahalanobis": r"Mahalanobis$^2$",
    "fid": "FID",
    "jeffreys": "Jeffreys (symlog)",
}
ROLES = ("reference", "all_trial")
ROLE_LABELS = {"reference": "reference", "all_trial": "query"}
METHODS = ("classical", "flow", "tre")
METHOD_LABELS = {"classical": "Classical", "flow": "Flow-based", "tre": "TRE"}
METHOD_COLORS = {"classical": "#4477AA", "flow": "#CC6677", "tre": "#228833"}
VISIBLE_CUE_INTERVAL = (0.0, 1.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT
        / "data/bci_iv_2a/multi_distance_session_identification_9recordings_full_query",
    )
    parser.add_argument(
        "--fid-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/fid_session_identification_9recordings_mixed_runs",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/processed/native_voltage_all_timepoints_20uv",
    )
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=[f"A{index:02d}T" for index in range(1, 10)],
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _load_rdms(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        rdms = np.asarray(archive["rdms"], dtype=np.float64)
    if rdms.ndim != 3 or rdms.shape[1:] != (4, 4):
        raise ValueError(f"Unexpected RDM shape {rdms.shape} in {path}.")
    if not np.isfinite(rdms).all():
        raise ValueError(f"Non-finite RDM values in {path}.")
    return rdms


def mean_off_diagonal_distance(rdms: np.ndarray) -> np.ndarray:
    """Average the six unique off-diagonal entries at every EEG sample."""

    upper = np.triu_indices(4, k=1)
    return np.mean(rdms[:, upper[0], upper[1]], axis=1)


def _cache_path(
    run_dir: Path,
    fid_dir: Path,
    metric: str,
    role: str,
    recording: str,
    method: str,
) -> Path:
    if method == "tre":
        if metric != "jeffreys":
            raise ValueError("TRE is only defined for the Jeffreys rows.")
        return (
            run_dir
            / "tre_jeffreys"
            / "rdm_cache"
            / f"{role}_{recording}_tre_jeffreys.npz"
        )
    if metric != "fid":
        return run_dir / "rdm_cache" / f"{role}_{recording}_{metric}_{method}.npz"
    if method == "classical":
        name = f"{role}_{recording}_classical_fid.npz"
    else:
        name = f"{role}_{recording}_condition_affine_flow_fid.npz"
    return fid_dir / "rdm_cache" / name


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.3)
    axis.tick_params(width=1.3, length=4)


def main() -> None:
    args = parse_args()
    if args.device != "cuda:0":
        raise ValueError("Project runs must use --device cuda:0.")
    recordings = list(args.recordings)
    times = np.asarray(
        load_features_npz(args.feature_dir / f"{recordings[0]}.npz").time_centers,
        dtype=np.float64,
    )
    trajectories = np.full(
        (
            len(METRICS),
            len(ROLES),
            len(METHODS),
            len(recordings),
            times.size,
        ),
        np.nan,
        dtype=np.float64,
    )
    for metric_index, metric in enumerate(METRICS):
        for role_index, role in enumerate(ROLES):
            for method_index, method in enumerate(METHODS):
                if method == "tre" and metric != "jeffreys":
                    continue
                for recording_index, recording in enumerate(recordings):
                    path = _cache_path(
                        args.run_dir, args.fid_dir, metric, role, recording, method
                    )
                    rdms = _load_rdms(path)
                    if rdms.shape[0] != times.size:
                        raise ValueError(f"Time dimension mismatch in {path}.")
                    trajectories[
                        metric_index, role_index, method_index, recording_index
                    ] = mean_off_diagonal_distance(rdms)

    output_dir = args.output_dir or args.run_dir / "distance_trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "all_metrics_mean_distance_vs_time_all_recordings.npz",
        mean_distance=trajectories,
        metrics=np.asarray(METRICS),
        roles=np.asarray(ROLES),
        methods=np.asarray(METHODS),
        recordings=np.asarray(recordings),
        time_seconds_cue_relative=times,
        visible_cue_interval_seconds=np.asarray(VISIBLE_CUE_INTERVAL),
        averaging_description=np.asarray(
            "arithmetic mean over the six unique off-diagonal entries of each 4x4 RDM"
        ),
    )

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 13,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    nrows = len(METRICS) * len(ROLES)
    figure, axes = plt.subplots(
        nrows,
        len(recordings),
        figsize=(4.0 * len(recordings), 3.5 * nrows),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    for metric_index, metric in enumerate(METRICS):
        for role_index, role in enumerate(ROLES):
            row = metric_index * len(ROLES) + role_index
            for recording_index, recording in enumerate(recordings):
                axis = axes[row, recording_index]
                pair_values = trajectories[metric_index, :, :, recording_index]
                upper = 1.04 * float(np.nanmax(pair_values))
                if not np.isfinite(upper) or upper <= 0.0:
                    upper = 1.0
                axis.axvspan(
                    *VISIBLE_CUE_INTERVAL,
                    color="0.93",
                    linewidth=0,
                    zorder=0,
                )
                axis.axvline(0.0, color="0.35", linestyle="--", linewidth=1.1, zorder=1)
                for method_index, method in enumerate(METHODS):
                    if method == "tre" and metric != "jeffreys":
                        continue
                    axis.plot(
                        times,
                        trajectories[
                            metric_index,
                            role_index,
                            method_index,
                            recording_index,
                        ],
                        color=METHOD_COLORS[method],
                        linewidth=1.5,
                        label=METHOD_LABELS[method],
                        zorder=2,
                    )
                axis.set_xlim(float(times[0]), float(times[-1]))
                axis.set_ylim(0.0, upper)
                if metric == "jeffreys":
                    axis.set_yscale("symlog", linthresh=0.05, linscale=1.0)
                axis.set_xticks([-1.0, 0.0, 1.0, 2.0, 3.0])
                if row == 0:
                    axis.set_title(recording)
                if recording_index == 0:
                    axis.set_ylabel(
                        f"{METRIC_LABELS[metric]}\n{ROLE_LABELS[role]}",
                        labelpad=8,
                    )
                _style_axis(axis)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linewidth=2.4,
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    handles.append(
        plt.Rectangle((0, 0), 1, 1, facecolor="0.93", edgecolor="none", label="Visible cue")
    )
    figure.legend(handles=handles, frameon=False, loc="outside upper center", ncol=3)
    figure.supxlabel("Time from cue onset (s)")
    figure.supylabel("Mean distance across six condition pairs")
    stem = "all_metrics_mean_distance_vs_time_all_recordings"
    figure.savefig(output_dir / f"{stem}.png", dpi=200, facecolor="white")
    figure.savefig(output_dir / f"{stem}.svg", facecolor="white")
    plt.close(figure)
    print(f"[trajectory-plot] output={output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
