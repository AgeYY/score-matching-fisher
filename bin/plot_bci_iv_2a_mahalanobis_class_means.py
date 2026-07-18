#!/usr/bin/env python3
"""Compare fitted Mahalanobis-flow endpoint means with binned empirical means."""

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
    load_features_npz,
)


DISPLAY_CLASS_NAMES = {
    "left_hand": "Left hand",
    "right_hand": "Right hand",
    "both_feet": "Both feet",
    "tongue": "Tongue",
}
VISIBLE_CUE_INTERVAL = (0.0, 1.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-file",
        type=Path,
        default=ROOT
        / "data/bci_iv_2a/processed/native_voltage_all_timepoints_20uv/A03T.npz",
    )
    parser.add_argument(
        "--result-file",
        type=Path,
        default=ROOT
        / "data/bci_iv_2a/a03t_reference_mahalanobis_half_trials"
        / "a03t_reference_mahalanobis_rdms.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/a03t_reference_mahalanobis_half_trials",
    )
    parser.add_argument("--bin-width-seconds", type=float, default=0.25)
    return parser.parse_args()


def _time_bins(times: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    if float(width) <= 0.0:
        raise ValueError("--bin-width-seconds must be positive.")
    start = float(times[0])
    stop = float(times[-1])
    n_bins = int(np.ceil((stop - start) / float(width)))
    edges = start + np.arange(n_bins + 1, dtype=np.float64) * float(width)
    edges[-1] = stop
    centers = 0.5 * (edges[:-1] + edges[1:])
    masks: list[np.ndarray] = []
    for index in range(n_bins):
        if index == n_bins - 1:
            mask = (times >= edges[index]) & (times <= edges[index + 1])
        else:
            mask = (times >= edges[index]) & (times < edges[index + 1])
        if not np.any(mask):
            raise RuntimeError(f"Time bin {index} is empty.")
        masks.append(mask)
    return edges, centers, masks


def _binned_means(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    flow_means: np.ndarray,
    masks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    n_bins = len(masks)
    n_classes = len(CLASS_NAMES)
    channels = int(train_values.shape[-1])
    empirical = np.empty((n_bins, n_classes, channels), dtype=np.float64)
    flow = np.empty_like(empirical)
    for bin_index, time_mask in enumerate(masks):
        for class_index in range(n_classes):
            class_values = train_values[train_labels == class_index]
            empirical[bin_index, class_index] = np.mean(
                class_values[:, time_mask, :], axis=(0, 1)
            )
            flow[bin_index, class_index] = np.mean(
                flow_means[time_mask, class_index, :], axis=0
            )
    return empirical, flow


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "axes.grid": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "savefig.bbox": "tight",
        }
    )


def _decorate_time_axis(axis: plt.Axes) -> None:
    axis.axvspan(*VISIBLE_CUE_INTERVAL, color="0.93", linewidth=0, zorder=0)
    axis.axvline(0.0, color="0.35", linestyle="--", linewidth=1.2, zorder=1)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(width=1.8)


def _plot_norms(
    output_dir: Path,
    times: np.ndarray,
    centers: np.ndarray,
    flow_means: np.ndarray,
    empirical_binned: np.ndarray,
    flow_binned: np.ndarray,
) -> None:
    _style()
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 7.0), sharex=True, sharey=True)
    flow_raw_norm = np.linalg.norm(flow_means, axis=-1)
    empirical_norm = np.linalg.norm(empirical_binned, axis=-1)
    flow_bin_norm = np.linalg.norm(flow_binned, axis=-1)
    for class_index, axis in enumerate(axes.flat):
        _decorate_time_axis(axis)
        axis.plot(
            times,
            flow_raw_norm[:, class_index],
            color="#CC6677",
            linewidth=1.8,
            label=r"Flow $\|b_\phi(c,u)\|$",
            zorder=3,
        )
        axis.plot(
            centers,
            flow_bin_norm[:, class_index],
            color="#CC6677",
            linestyle="none",
            marker="o",
            markersize=4.0,
            label="Flow, 250 ms mean",
            zorder=4,
        )
        axis.plot(
            centers,
            empirical_norm[:, class_index],
            color="#4477AA",
            linewidth=1.8,
            marker="o",
            markersize=4.0,
            label="Empirical, 250 ms mean",
            zorder=5,
        )
        axis.set_title(DISPLAY_CLASS_NAMES[CLASS_NAMES[class_index]])
    for axis in axes[-1]:
        axis.set_xlabel("Time from cue onset (s)")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"Mean-vector norm")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
    )
    figure.tight_layout()
    figure.savefig(output_dir / "a03t_mahalanobis_class_mean_norm_vs_time.png", dpi=300)
    figure.savefig(output_dir / "a03t_mahalanobis_class_mean_norm_vs_time.svg")
    plt.close(figure)


def _plot_heatmaps(
    output_dir: Path,
    edges: np.ndarray,
    empirical_binned: np.ndarray,
    flow_binned: np.ndarray,
) -> None:
    _style()
    difference = flow_binned - empirical_binned
    common_limit = float(
        np.quantile(np.abs(np.concatenate([empirical_binned.ravel(), flow_binned.ravel()])), 0.995)
    )
    difference_limit = float(np.quantile(np.abs(difference), 0.995))
    common_limit = max(common_limit, np.finfo(np.float64).eps)
    difference_limit = max(difference_limit, np.finfo(np.float64).eps)
    figure, axes = plt.subplots(
        len(CLASS_NAMES),
        3,
        figsize=(12.0, 14.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    first_image = None
    difference_image = None
    values_by_column = (empirical_binned, flow_binned, difference)
    titles = ("Empirical 250 ms mean", "Flow 250 ms mean", "Flow minus empirical")
    for class_index in range(len(CLASS_NAMES)):
        for column, values in enumerate(values_by_column):
            axis = axes[class_index, column]
            limit = difference_limit if column == 2 else common_limit
            image = axis.imshow(
                values[:, class_index, :].T,
                origin="upper",
                aspect="auto",
                interpolation="nearest",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                extent=(float(edges[0]), float(edges[-1]), len(CANONICAL_EEG_CHANNEL_NAMES) - 0.5, -0.5),
            )
            axis.axvline(0.0, color="0.2", linestyle="--", linewidth=1.0)
            axis.set_xlim(float(edges[0]), float(edges[-1]))
            axis.grid(False)
            for spine in axis.spines.values():
                spine.set_linewidth(1.8)
            axis.tick_params(width=1.8)
            if class_index == 0:
                axis.set_title(titles[column])
            if column == 0:
                axis.set_ylabel(DISPLAY_CLASS_NAMES[CLASS_NAMES[class_index]])
                axis.set_yticks(np.arange(len(CANONICAL_EEG_CHANNEL_NAMES)))
                axis.set_yticklabels(CANONICAL_EEG_CHANNEL_NAMES, fontsize=9)
            if class_index == len(CLASS_NAMES) - 1:
                axis.set_xlabel("Time from cue onset (s)")
            if column == 0:
                first_image = image
            if column == 2:
                difference_image = image
    if first_image is None or difference_image is None:
        raise RuntimeError("Heatmap images were not initialized.")
    figure.colorbar(
        first_image,
        ax=axes[:, :2],
        location="right",
        shrink=0.75,
        label="Mean voltage (20 µV units)",
    )
    figure.colorbar(
        difference_image,
        ax=axes[:, 2],
        location="right",
        shrink=0.75,
        label="Difference (20 µV units)",
    )
    figure.savefig(output_dir / "a03t_mahalanobis_class_mean_channel_heatmaps.png", dpi=300)
    figure.savefig(output_dir / "a03t_mahalanobis_class_mean_channel_heatmaps.svg")
    plt.close(figure)


def _alignment_summary(
    empirical_binned: np.ndarray,
    flow_binned: np.ndarray,
    empirical_native: np.ndarray,
    flow_native: np.ndarray,
    times: np.ndarray,
    bin_width: float,
) -> dict[str, object]:
    empirical_norm = np.linalg.norm(empirical_binned, axis=-1)
    flow_norm = np.linalg.norm(flow_binned, axis=-1)
    cosine = np.sum(empirical_binned * flow_binned, axis=-1) / np.maximum(
        empirical_norm * flow_norm, np.finfo(np.float64).eps
    )
    rmse = np.sqrt(np.mean((flow_binned - empirical_binned) ** 2, axis=-1))
    empirical_native_norm = np.linalg.norm(empirical_native, axis=-1)
    flow_native_norm = np.linalg.norm(flow_native, axis=-1)
    native_cosine = np.sum(empirical_native * flow_native, axis=-1) / np.maximum(
        empirical_native_norm * flow_native_norm, np.finfo(np.float64).eps
    )
    per_class = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        spike_index = int(np.argmax(flow_native_norm[:, class_index]))
        per_class[class_name] = {
            "mean_cosine_alignment": float(np.mean(cosine[:, class_index])),
            "median_cosine_alignment": float(np.median(cosine[:, class_index])),
            "mean_channel_rmse": float(np.mean(rmse[:, class_index])),
            "mean_flow_to_empirical_norm_ratio": float(
                np.mean(flow_norm[:, class_index] / np.maximum(
                    empirical_norm[:, class_index], np.finfo(np.float64).eps
                ))
            ),
            "native_time_mean_cosine_alignment": float(
                np.mean(native_cosine[:, class_index])
            ),
            "native_time_median_cosine_alignment": float(
                np.median(native_cosine[:, class_index])
            ),
            "flow_max_native_norm": float(flow_native_norm[spike_index, class_index]),
            "flow_max_native_norm_time_seconds": float(times[spike_index]),
            "empirical_native_norm_at_flow_max": float(
                empirical_native_norm[spike_index, class_index]
            ),
        }
    return {
        "comparison": "best-checkpoint flow endpoint mean versus train-only empirical temporal-bin mean",
        "bin_width_seconds": float(bin_width),
        "n_bins": int(empirical_binned.shape[0]),
        "overall_mean_cosine_alignment": float(np.mean(cosine)),
        "overall_median_cosine_alignment": float(np.median(cosine)),
        "overall_mean_channel_rmse": float(np.mean(rmse)),
        "per_class": per_class,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_features_npz(args.feature_file)
    with np.load(args.result_file, allow_pickle=False) as result:
        times = np.asarray(result["time_seconds_cue_relative"], dtype=np.float64)
        flow_means = np.asarray(result["flow_means"], dtype=np.float64)
        reference_indices = np.asarray(result["reference_trial_indices"], dtype=np.int64)
        train_trials = np.asarray(result["train_trial_indices"], dtype=np.int64)
    if flow_means.shape != (times.size, len(CLASS_NAMES), len(CANONICAL_EEG_CHANNEL_NAMES)):
        raise ValueError(f"Unexpected flow_means shape: {flow_means.shape}.")
    selected_values = np.asarray(dataset.features[reference_indices], dtype=np.float64)
    selected_labels = np.asarray(dataset.labels[reference_indices], dtype=np.int64)
    train_values = selected_values[train_trials]
    train_labels = selected_labels[train_trials]
    empirical_native = np.stack(
        [
            np.mean(train_values[train_labels == class_index], axis=0)
            for class_index in range(len(CLASS_NAMES))
        ],
        axis=1,
    )
    edges, centers, masks = _time_bins(times, args.bin_width_seconds)
    empirical_binned, flow_binned = _binned_means(
        train_values, train_labels, flow_means, masks
    )
    _plot_norms(
        args.output_dir,
        times,
        centers,
        flow_means,
        empirical_binned,
        flow_binned,
    )
    _plot_heatmaps(args.output_dir, edges, empirical_binned, flow_binned)
    np.savez_compressed(
        args.output_dir / "a03t_mahalanobis_class_mean_comparison.npz",
        bin_edges_seconds=edges,
        bin_centers_seconds=centers,
        empirical_train_binned_means=empirical_binned,
        empirical_train_native_time_means=empirical_native,
        flow_binned_means=flow_binned,
        flow_native_time_means=flow_means,
        train_trial_indices=train_trials,
        reference_trial_indices=reference_indices,
    )
    summary = _alignment_summary(
        empirical_binned,
        flow_binned,
        empirical_native,
        flow_means,
        times,
        args.bin_width_seconds,
    )
    summary["n_train_trials"] = int(train_trials.size)
    summary["result_file"] = str(args.result_file.resolve())
    (args.output_dir / "a03t_mahalanobis_class_mean_comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"[class means] output={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
