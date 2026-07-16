#!/usr/bin/env python3
"""Plot mean pairwise squared Mahalanobis distance from cached BCI IV-2a RDMs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


METHOD_LABELS = {
    "classical_mahalanobis": "Mahalanobis",
    "time_varying_shared_affine_flow": "Flow",
}
METHOD_COLORS = {
    "classical_mahalanobis": "#4477AA",
    "time_varying_shared_affine_flow": "#CC6677",
}
ROLE_LABELS = {
    "reference": "Reference split",
    "query": "All-trial query split",
}
RDM_MATCHING_INTERVAL = (0.0, 3.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _load_rdms(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        rdms = np.asarray(archive["rdms"], dtype=np.float64)
    if rdms.ndim != 3 or rdms.shape[1:] != (4, 4):
        raise ValueError(f"Unexpected RDM shape {rdms.shape} in {path}.")
    if not np.all(np.isfinite(rdms)):
        raise ValueError(f"Non-finite RDM values in {path}.")
    return rdms


def mean_pairwise_distance(rdms: np.ndarray) -> np.ndarray:
    """Average the six unique off-diagonal distances at every time point."""

    values = np.asarray(rdms, dtype=np.float64)
    upper = np.triu_indices(4, k=1)
    return np.mean(values[:, upper[0], upper[1]], axis=1)


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(width=1.8)


def _plot_per_recording(
    output_dir: Path,
    times: np.ndarray,
    trajectories: np.ndarray,
    roles: list[str],
    methods: list[str],
    recordings: list[str],
) -> None:
    """Plot reference and all-trial trajectories in adjacent panels per recording."""

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    figure, axes = plt.subplots(
        3,
        6,
        figsize=(18.0, 10.5),
        sharex=True,
        sharey=False,
        layout="constrained",
    )
    for recording_index, recording in enumerate(recordings):
        row = recording_index // 3
        first_column = 2 * (recording_index % 3)
        pair_values = trajectories[:, :, recording_index]
        pair_upper = 1.04 * float(np.max(pair_values))
        for role_index, role in enumerate(roles):
            axis = axes[row, first_column + role_index]
            axis.axvspan(*RDM_MATCHING_INTERVAL, color="0.92", linewidth=0, zorder=0)
            axis.axvline(0.0, color="0.35", linestyle=":", linewidth=1.5, zorder=1)
            for method_index, method in enumerate(methods):
                axis.plot(
                    times,
                    trajectories[role_index, method_index, recording_index],
                    color=METHOD_COLORS.get(method),
                    linewidth=1.8,
                    zorder=2,
                )
            axis.set_title(f"{recording} — {ROLE_LABELS[role]}", fontsize=14)
            axis.set_xlim(float(times[0]), float(times[-1]))
            axis.set_ylim(0.0, pair_upper)
            axis.set_xticks([-1.0, 0.0, 1.0, 2.0, 3.0])
            _style_axis(axis)
    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linewidth=2.2,
            label=METHOD_LABELS[method],
        )
        for method in methods
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="0.35",
                linestyle=":",
                linewidth=1.5,
                label="Cue onset",
            ),
            Patch(facecolor="0.92", edgecolor="none", label="RDM matching interval"),
        ]
    )
    figure.legend(handles=handles, frameon=False, loc="outside upper center", ncol=4)
    figure.supxlabel("Time from cue onset (s)")
    figure.supylabel("Mean squared distance")
    figure.savefig(output_dir / "per_recording_mean_rdm_distance_vs_time.png", dpi=300)
    figure.savefig(output_dir / "per_recording_mean_rdm_distance_vs_time.svg")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        device_index = int(args.device.split(":", maxsplit=1)[1]) if ":" in args.device else 0
        torch.cuda.set_device(device_index)
        print(
            f"[mean-rdm] device={args.device} GPU={torch.cuda.get_device_name(device_index)}",
            flush=True,
        )

    source_summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    recordings = list(source_summary["recordings"])
    methods = list(source_summary["methods"])
    times = np.asarray(
        source_summary["input_features"]["time_centers_seconds_cue_relative"],
        dtype=np.float64,
    )
    roles = ["reference", "query"]
    cache_dir = args.run_dir / "rdm_cache"
    trajectories = np.empty(
        (len(roles), len(methods), len(recordings), times.size), dtype=np.float64
    )
    for role_index, role in enumerate(roles):
        for method_index, method in enumerate(methods):
            for recording_index, recording in enumerate(recordings):
                if role == "reference":
                    name = f"reference_{recording}_{method}.npz"
                else:
                    name = f"query_{recording}_nall_rep00_{method}.npz"
                rdms = _load_rdms(cache_dir / name)
                if rdms.shape[0] != times.size:
                    raise ValueError(f"Time dimension mismatch in {name}.")
                trajectories[role_index, method_index, recording_index] = mean_pairwise_distance(rdms)

    mean_trajectories = np.mean(trajectories, axis=2)
    recording_sd = np.std(trajectories, axis=2, ddof=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "mean_rdm_distance_trajectories.npz",
        time_seconds_cue_relative=times,
        per_recording_trajectories=trajectories,
        mean_trajectories=mean_trajectories,
        recording_sd=recording_sd,
        roles=np.asarray(roles),
        methods=np.asarray(methods),
        recordings=np.asarray(recordings),
    )

    rows: list[dict] = []
    for role_index, role in enumerate(roles):
        for method_index, method in enumerate(methods):
            for time_index, time in enumerate(times):
                rows.append(
                    {
                        "role": role,
                        "method": method,
                        "time_seconds_cue_relative": float(time),
                        "mean_pairwise_squared_mahalanobis_distance": float(
                            mean_trajectories[role_index, method_index, time_index]
                        ),
                        "recording_sd": float(recording_sd[role_index, method_index, time_index]),
                        "n_recordings": len(recordings),
                    }
                )
    with (args.output_dir / "mean_rdm_distance_trajectories.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), sharex=True, sharey=True)
    for role_index, (role, axis) in enumerate(zip(roles, axes, strict=True)):
        axis.axvspan(*RDM_MATCHING_INTERVAL, color="0.92", linewidth=0, zorder=0)
        axis.axvline(0.0, color="0.35", linestyle="--", linewidth=1.6, zorder=1)
        for method_index, method in enumerate(methods):
            axis.plot(
                times,
                mean_trajectories[role_index, method_index],
                color=METHOD_COLORS.get(method),
                linewidth=2.2,
                label=METHOD_LABELS.get(method, method),
                zorder=2,
            )
        axis.set_title(ROLE_LABELS[role])
        axis.set_xlabel("Time from cue onset (s)")
        axis.set_xlim(float(times[0]), float(times[-1]))
        _style_axis(axis)
    axes[0].set_ylabel("Mean squared distance")
    axes[0].legend(frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(args.output_dir / "mean_rdm_distance_vs_time.png", dpi=300)
    figure.savefig(args.output_dir / "mean_rdm_distance_vs_time.svg")
    plt.close(figure)

    _plot_per_recording(
        args.output_dir,
        times,
        trajectories,
        roles,
        methods,
        recordings,
    )

    interval_masks = {
        "pre_cue": (times >= -1.5) & (times <= -0.5),
        "early_post_cue": (times >= 0.0) & (times <= 1.25),
        "rdm_matching_interval": (
            (times >= RDM_MATCHING_INTERVAL[0])
            & (times <= RDM_MATCHING_INTERVAL[1])
        ),
    }
    interval_means: dict = {}
    for role_index, role in enumerate(roles):
        interval_means[role] = {}
        for method_index, method in enumerate(methods):
            interval_means[role][method] = {
                interval_name: float(np.mean(mean_trajectories[role_index, method_index, mask]))
                for interval_name, mask in interval_masks.items()
            }
    summary = {
        "experiment": "Mean pairwise squared Mahalanobis distance over EEG time",
        "source_run": str(args.run_dir.resolve()),
        "device": args.device,
        "averaging_order": "six unique condition pairs within RDM, then nine recordings",
        "uncertainty_displayed": False,
        "n_recordings": len(recordings),
        "n_condition_pairs": 6,
        "n_time_points": int(times.size),
        "time_interval_seconds_cue_relative": [float(times[0]), float(times[-1])],
        "rdm_matching_interval_seconds_cue_relative": list(RDM_MATCHING_INTERVAL),
        "rdm_matching_interval_definition": (
            "Time points whose six unique RDM entries are vectorized for "
            "query-to-reference recording matching."
        ),
        "interval_means": interval_means,
        "per_recording_peak_distance": {
            recording: {
                role: {
                    method: float(
                        np.max(trajectories[role_index, method_index, recording_index])
                    )
                    for method_index, method in enumerate(methods)
                }
                for role_index, role in enumerate(roles)
            }
            for recording_index, recording in enumerate(recordings)
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[mean-rdm] output={args.output_dir.resolve()}", flush=True)
    print(json.dumps(interval_means, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
