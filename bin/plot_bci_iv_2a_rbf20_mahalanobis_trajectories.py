#!/usr/bin/env python3
"""Plot per-recording classical and RBF-time flow Mahalanobis trajectories."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher.bci_iv_2a_dataset import load_features_npz  # noqa: E402


ROLES = ("reference", "half_query")
ROLE_LABELS = {
    "reference": "Reference split",
    "half_query": "Half-query split",
}
METHODS = ("classical", "flow")
METHOD_COLORS = {"classical": "#4477AA", "flow": "#CC6677"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT
        / "data/bci_iv_2a/rbf8_mahalanobis_session_identification_5recordings_half_query",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/processed/native_voltage_all_timepoints_20uv",
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
    np.testing.assert_allclose(rdms, rdms.transpose(0, 2, 1), atol=1e-7, rtol=0.0)
    return rdms


def _mean_pairwise_distance(rdms: np.ndarray) -> np.ndarray:
    upper = np.triu_indices(4, k=1)
    return np.mean(rdms[:, upper[0], upper[1]], axis=1)


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(width=1.8)


def main() -> None:
    args = parse_args()
    if args.device != "cuda:0":
        raise ValueError("This project requires --device cuda:0.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing CPU fallback.")
    torch.cuda.set_device(0)

    summary_paths = sorted(args.run_dir.glob("mahalanobis_rbf*_half_query_summary.json"))
    if len(summary_paths) != 1:
        raise ValueError(
            f"Expected exactly one RBF Mahalanobis summary in {args.run_dir}, "
            f"found {len(summary_paths)}."
        )
    summary_path = summary_paths[0]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recordings = list(summary["recordings"])
    if not recordings:
        raise ValueError("The run summary contains no recordings.")
    rbf_centers = int(summary["time_rbf_num_centers"])
    matching_interval = tuple(
        float(value) for value in summary["requested_interval_seconds_cue_relative"]
    )
    times = np.asarray(
        load_features_npz(args.feature_dir / f"{recordings[0]}.npz").time_centers,
        dtype=np.float64,
    )

    trajectories = np.empty(
        (len(ROLES), len(METHODS), len(recordings), times.size),
        dtype=np.float64,
    )
    cache_dir = args.run_dir / "rdm_cache"
    for role_index, role in enumerate(ROLES):
        for recording_index, recording in enumerate(recordings):
            paths = {
                "classical": cache_dir
                / f"{role}_{recording}_mahalanobis_classical.npz",
                "flow": cache_dir
                / f"{role}_{recording}_mahalanobis_rbf{rbf_centers}_flow.npz",
            }
            for method_index, method in enumerate(METHODS):
                rdms = _load_rdms(paths[method])
                if rdms.shape[0] != times.size:
                    raise ValueError(f"Time dimension mismatch in {paths[method]}.")
                trajectories[role_index, method_index, recording_index] = (
                    _mean_pairwise_distance(rdms)
                )

    output_dir = args.output_dir or args.run_dir / "mean_rdm_distance"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"per_recording_mahalanobis_rbf{rbf_centers}_trajectories.npz",
        time_seconds_cue_relative=times,
        mean_pairwise_mahalanobis_squared=trajectories,
        roles=np.asarray(ROLES),
        methods=np.asarray(("classical_mahalanobis", f"flow_rbf{rbf_centers}")),
        recordings=np.asarray(recordings),
        matching_interval_seconds=np.asarray(matching_interval),
        averaging_description=np.asarray(
            "arithmetic mean over the six unique off-diagonal entries of each 4x4 RDM"
        ),
    )

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 14,
            "axes.grid": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )
    recordings_per_row = min(2, len(recordings))
    nrows = math.ceil(len(recordings) / recordings_per_row)
    ncols = 2 * recordings_per_row
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.5 * nrows),
        sharex=True,
        sharey=False,
        layout="constrained",
    )
    axes = np.asarray(axes, dtype=object).reshape(nrows, ncols)
    for recording_index, recording in enumerate(recordings):
        row = recording_index // recordings_per_row
        first_column = 2 * (recording_index % recordings_per_row)
        participant_values = trajectories[:, :, recording_index]
        participant_upper = 1.04 * float(np.max(participant_values))
        if not np.isfinite(participant_upper) or participant_upper <= 0.0:
            participant_upper = 1.0
        for role_index, role in enumerate(ROLES):
            axis = axes[row, first_column + role_index]
            axis.axvspan(
                *matching_interval,
                color="0.92",
                linewidth=0.0,
                zorder=0,
            )
            axis.axvline(
                0.0,
                color="0.35",
                linestyle=":",
                linewidth=1.5,
                zorder=1,
            )
            for method_index, method in enumerate(METHODS):
                axis.plot(
                    times,
                    trajectories[role_index, method_index, recording_index],
                    color=METHOD_COLORS[method],
                    linewidth=1.8,
                    zorder=2,
                )
            axis.set_title(f"{recording} — {ROLE_LABELS[role]}")
            axis.set_xlim(float(times[0]), float(times[-1]))
            axis.set_ylim(0.0, participant_upper)
            axis.set_xticks([-1.0, 0.0, 1.0, 2.0, 3.0])
            _style_axis(axis)

    for recording_slot in range(len(recordings), nrows * recordings_per_row):
        row = recording_slot // recordings_per_row
        first_column = 2 * (recording_slot % recordings_per_row)
        axes[row, first_column].set_visible(False)
        axes[row, first_column + 1].set_visible(False)

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS["classical"],
            linewidth=2.4,
            label="Classical Mahalanobis",
        ),
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS["flow"],
            linewidth=2.4,
            label=f"Flow: RBF{rbf_centers}",
        ),
        Line2D(
            [0],
            [0],
            color="0.35",
            linestyle=":",
            linewidth=1.5,
            label="Cue onset",
        ),
        Patch(
            facecolor="0.92",
            edgecolor="none",
            label="RDM matching interval",
        ),
    ]
    figure.legend(handles=handles, frameon=False, loc="outside upper center", ncol=4)
    figure.supxlabel("Time from cue onset (s)")
    figure.supylabel(r"Mean Mahalanobis$^2$ distance")
    stem = f"per_recording_mahalanobis_rbf{rbf_centers}_distance_vs_time"
    figure.savefig(output_dir / f"{stem}.png", dpi=300)
    figure.savefig(output_dir / f"{stem}.svg")
    plt.close(figure)
    print(f"[trajectory-plot] output={output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
