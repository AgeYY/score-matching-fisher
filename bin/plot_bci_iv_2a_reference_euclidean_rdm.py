#!/usr/bin/env python3
"""Compare classical and translation-flow Euclidean RDMs on A01T reference trials."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from global_setting import EARLY_STOPPING_PATIENCE, TRAINING_MAX_EPOCHS  # noqa: E402
from fisher.bci_iv_2a_dataset import CLASS_NAMES, load_features_npz  # noqa: E402
from fisher.bci_iv_2a_session_identification import (  # noqa: E402
    FlowRDMConfig,
    REFERENCE_RUNS,
    empirical_condition_means,
    select_half,
    squared_euclidean_rdms_from_means,
    translation_flow_squared_euclidean_rdms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-file",
        type=Path,
        default=ROOT
        / "data/bci_iv_2a/processed/native_voltage_all_timepoints_20uv/A01T.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/reference_euclidean_A01T",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=21260715)
    parser.add_argument("--epochs", type=int, default=TRAINING_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    return parser.parse_args()


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(width=1.8)


def _pair_labels() -> list[str]:
    short = {
        "left_hand": "Left",
        "right_hand": "Right",
        "both_feet": "Feet",
        "tongue": "Tongue",
    }
    labels = []
    for left in range(len(CLASS_NAMES)):
        for right in range(left + 1, len(CLASS_NAMES)):
            labels.append(f"{short[CLASS_NAMES[left]]}–{short[CLASS_NAMES[right]]}")
    return labels


def _pair_trajectories(rdms: np.ndarray) -> np.ndarray:
    upper = np.triu_indices(len(CLASS_NAMES), k=1)
    return np.asarray(rdms, dtype=np.float64)[:, upper[0], upper[1]].T


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if np.std(x) <= np.finfo(np.float64).eps or np.std(y) <= np.finfo(np.float64).eps:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _comparison_metrics(
    times: np.ndarray,
    classical_pairs: np.ndarray,
    flow_pairs: np.ndarray,
) -> dict[str, dict[str, float]]:
    masks = {
        "full": np.ones(times.size, dtype=bool),
        "pre_cue": times < 0.0,
        "cue_to_3p5s": (times >= 0.0) & (times <= 3.5),
    }
    result: dict[str, dict[str, float]] = {}
    for name, mask in masks.items():
        classical = classical_pairs[:, mask]
        flow = flow_pairs[:, mask]
        result[name] = {
            "correlation_all_time_pair_entries": _safe_correlation(classical, flow),
            "correlation_mean_over_pairs_trajectory": _safe_correlation(
                np.mean(classical, axis=0),
                np.mean(flow, axis=0),
            ),
            "rmse": float(np.sqrt(np.mean((flow - classical) ** 2))),
            "classical_mean": float(np.mean(classical)),
            "flow_mean": float(np.mean(flow)),
            "flow_to_classical_mean_ratio": float(
                np.mean(flow) / max(np.mean(classical), np.finfo(np.float64).eps)
            ),
        }
    return result


def _plot_comparison(
    output_dir: Path,
    times: np.ndarray,
    classical_pairs: np.ndarray,
    flow_pairs: np.ndarray,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, 3.5),
        layout="constrained",
    )
    maximum = float(max(np.max(classical_pairs), np.max(flow_pairs)))
    images = []
    for axis, values, title in zip(
        axes[:2],
        (classical_pairs, flow_pairs),
        ("Classical per-time means", "Translation flow"),
        strict=True,
    ):
        image = axis.imshow(
            values,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            extent=(float(times[0]), float(times[-1]), values.shape[0] - 0.5, -0.5),
            cmap="magma",
            vmin=0.0,
            vmax=maximum,
        )
        images.append(image)
        axis.axvline(0.0, color="white", linestyle="--", linewidth=1.5)
        axis.set_title(title)
        axis.set_xlabel("Time from cue (s)")
        axis.set_yticks(np.arange(values.shape[0]), _pair_labels())
        _style_axis(axis)
    axes[1].tick_params(labelleft=False)
    colorbar = figure.colorbar(images[0], ax=axes[:2], location="bottom", shrink=0.82)
    colorbar.set_label("Squared Euclidean distance")

    mean_classical = np.mean(classical_pairs, axis=0)
    mean_flow = np.mean(flow_pairs, axis=0)
    axes[2].plot(
        times,
        mean_classical,
        color="#4477AA",
        linewidth=2.0,
        label="Classical",
    )
    axes[2].plot(
        times,
        mean_flow,
        color="#CC6677",
        linewidth=2.0,
        label="Flow",
    )
    axes[2].axvline(0.0, color="0.35", linestyle="--", linewidth=1.5)
    axes[2].set_title("Mean over six pairs")
    axes[2].set_xlabel("Time from cue (s)")
    axes[2].set_ylabel("Mean squared distance")
    axes[2].set_xlim(float(times[0]), float(times[-1]))
    axes[2].legend(frameon=False, loc="best")
    _style_axis(axes[2])
    figure.savefig(output_dir / "reference_euclidean_rdm_comparison.png", dpi=300)
    figure.savefig(output_dir / "reference_euclidean_rdm_comparison.svg")
    plt.close(figure)


def _plot_losses(output_dir: Path, metadata: dict) -> None:
    train = np.asarray(metadata["train_losses"], dtype=np.float64)
    validation = np.asarray(metadata["validation_losses"], dtype=np.float64)
    monitored = np.asarray(metadata["monitored_validation_losses"], dtype=np.float64)
    epochs = np.arange(1, train.size + 1)
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 11,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    figure, axis = plt.subplots(figsize=(4.0, 3.5))
    axis.plot(epochs, train, color="#4477AA", linewidth=1.4, label="Train")
    axis.plot(epochs, validation, color="#CC6677", linewidth=1.4, alpha=0.6, label="Val")
    axis.plot(epochs, monitored, color="#228833", linewidth=2.0, label="Val EMA")
    axis.axvline(
        int(metadata["best_epoch"]),
        color="0.35",
        linestyle="--",
        linewidth=1.5,
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Flow-matching loss")
    axis.legend(frameon=False, loc="upper right")
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(output_dir / "translation_flow_loss.png", dpi=300)
    figure.savefig(output_dir / "translation_flow_loss.svg")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA; no CPU fallback is permitted.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device.index} is unavailable.")
    torch.cuda.set_device(0 if device.index is None else device.index)
    dataset = load_features_npz(args.feature_file)
    reference_x, reference_y, reference_run_ids = select_half(dataset, REFERENCE_RUNS)
    times = np.asarray(dataset.time_centers, dtype=np.float64)

    classical_means = empirical_condition_means(
        reference_x,
        reference_y,
        standardize_features=False,
    )
    classical_rdms = squared_euclidean_rdms_from_means(classical_means)
    config = FlowRDMConfig(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=1e-5,
        patience=args.patience,
        quadrature_steps=32,
        covariance_ode_steps=48,
        covariance_ridge=1e-5,
        validation_fraction=0.2,
        standardize_features=False,
        device_resident_data=True,
    )
    print(
        f"[euclidean] session={dataset.session_key} reference_trials={reference_x.shape[0]} "
        f"times={times.size} device={device}",
        flush=True,
    )
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    flow_rdms, flow_metadata, flow_means = translation_flow_squared_euclidean_rdms(
        reference_x,
        reference_y,
        times,
        device=device,
        seed=args.seed,
        config=config,
        return_means=True,
    )
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start

    classical_pairs = _pair_trajectories(classical_rdms)
    flow_pairs = _pair_trajectories(flow_rdms)
    metrics = _comparison_metrics(times, classical_pairs, flow_pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "reference_euclidean_components.npz",
        time_seconds_cue_relative=times,
        classical_means=classical_means,
        flow_means=flow_means,
        classical_rdms=classical_rdms,
        flow_rdms=flow_rdms,
        classical_pair_trajectories=classical_pairs,
        flow_pair_trajectories=flow_pairs,
        train_losses=np.asarray(flow_metadata["train_losses"], dtype=np.float64),
        validation_losses=np.asarray(flow_metadata["validation_losses"], dtype=np.float64),
        monitored_validation_losses=np.asarray(
            flow_metadata["monitored_validation_losses"],
            dtype=np.float64,
        ),
    )
    rows = []
    pair_labels = _pair_labels()
    for pair_index, pair_label in enumerate(pair_labels):
        for time_index, time_value in enumerate(times):
            rows.append(
                {
                    "time_seconds_cue_relative": float(time_value),
                    "condition_pair": pair_label,
                    "classical_squared_euclidean": float(classical_pairs[pair_index, time_index]),
                    "flow_squared_euclidean": float(flow_pairs[pair_index, time_index]),
                }
            )
    with (args.output_dir / "reference_euclidean_trajectories.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": "A01T reference-split squared Euclidean RDM comparison",
        "feature_file": str(args.feature_file.resolve()),
        "feature_units": "20_microvolts_per_model_unit",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "session": dataset.session_key,
        "reference_runs_zero_based": sorted(np.unique(reference_run_ids).astype(int).tolist()),
        "n_reference_trials": int(reference_x.shape[0]),
        "n_time_points": int(times.size),
        "time_interval_seconds": [float(times[0]), float(times[-1])],
        "distance": "squared_euclidean_between_condition_means",
        "classical_estimator": "sample_mean_within_each_condition_and_raw_time_sample",
        "flow_estimator": "joint_class_and_time_conditioned_translation_only_flow",
        "flow_fit_elapsed_seconds": float(elapsed_seconds),
        "flow_training": flow_metadata,
        "agreement": metrics,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_comparison(args.output_dir, times, classical_pairs, flow_pairs)
    _plot_losses(args.output_dir, flow_metadata)
    print(json.dumps({"elapsed_seconds": elapsed_seconds, "agreement": metrics}, indent=2))
    print(f"[euclidean] output={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
