#!/usr/bin/env python3
"""Fit constrained flow correlation RDMs for the two-class time-resolved toy."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher.flow_matching_skl import (  # noqa: E402
    build_flow_skl_model,
    train_flow_skl_model,
)
from global_setting import EARLY_STOPPING_PATIENCE, TRAINING_MAX_EPOCHS  # noqa: E402


VELOCITY_FAMILY = "translation_centered_fixed_norm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    dataset_dir = (
        ROOT
        / "data/time_resolved_rdm_toy_controlled_rotation_xdim40_n100_per_class"
    )
    parser.add_argument(
        "--dataset-npz",
        type=Path,
        default=dataset_dir / "two_class_time_resolved_rdm_toy.npz",
    )
    parser.add_argument(
        "--classical-npz",
        type=Path,
        default=dataset_dir
        / "classical_correlation_bin0p5"
        / "binned_correlation_distance.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=dataset_dir / "correlation_classical_flow",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20_260_718)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=TRAINING_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument(
        "--divergence-estimator",
        choices=("exact", "hutchinson"),
        default="hutchinson",
    )
    parser.add_argument("--hutchinson-probes", type=int, default=4)
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Save numerical results and checkpoint without per-case figures.",
    )
    return parser.parse_args()


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _stratified_trial_split(
    labels: np.ndarray, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    class_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    fraction = float(validation_fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("validation_fraction must lie in [0, 1).")
    if fraction == 0.0:
        return (
            np.arange(class_labels.size, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    rng = np.random.default_rng(int(seed))
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    for class_index in range(2):
        candidates = np.flatnonzero(class_labels == class_index)
        shuffled = rng.permutation(candidates)
        n_validation = max(1, int(np.floor(fraction * candidates.size)))
        if n_validation >= candidates.size:
            raise ValueError("Each class must retain at least one training trial.")
        validation_parts.append(shuffled[:n_validation])
        train_parts.append(shuffled[n_validation:])
    return (
        np.sort(np.concatenate(train_parts)).astype(np.int64),
        np.sort(np.concatenate(validation_parts)).astype(np.int64),
    )


def _condition_design(labels: np.ndarray, times: np.ndarray, time_scale: float) -> np.ndarray:
    class_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    time_values = np.asarray(times, dtype=np.float64).reshape(-1)
    if class_labels.shape != time_values.shape:
        raise ValueError("labels and times must have the same shape.")
    if np.any((class_labels < 0) | (class_labels > 1)):
        raise ValueError("labels must be 0 or 1.")
    one_hot = np.eye(2, dtype=np.float64)[class_labels]
    return np.concatenate([one_hot, (time_values / float(time_scale))[:, None]], axis=1)


def _flatten_trials(
    responses: np.ndarray,
    labels: np.ndarray,
    time_values: np.ndarray,
    trial_indices: np.ndarray,
    time_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(trial_indices, dtype=np.int64)
    n_times = int(time_values.size)
    x = np.asarray(responses[selected], dtype=np.float64).reshape(-1, responses.shape[-1])
    sample_labels = np.repeat(labels[selected], n_times)
    sample_times = np.tile(time_values, selected.size)
    return _condition_design(sample_labels, sample_times, time_scale), x


@torch.no_grad()
def _flow_correlation_distance(
    model: torch.nn.Module,
    time_values: np.ndarray,
    time_scale: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    grid_labels = np.repeat(np.arange(2, dtype=np.int64), time_values.size)
    grid_times = np.tile(time_values, 2)
    conditions = _condition_design(grid_labels, grid_times, time_scale)
    theta = torch.from_numpy(conditions.astype(np.float32)).to(
        device=device, dtype=next(model.parameters()).dtype
    )
    endpoint_means = (
        model.endpoint_mean(theta)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
        .reshape(2, time_values.size, -1)
    )
    dot_product = np.einsum(
        "td,td->t", endpoint_means[0], endpoint_means[1], optimize=True
    )
    distance = np.clip(1.0 - dot_product, 0.0, 2.0)
    return distance, endpoint_means


def _save_checkpoint(
    path: Path,
    *,
    model_kwargs: dict[str, Any],
    train_metadata: dict[str, Any],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    time_scale: float,
    args: argparse.Namespace,
) -> None:
    best_state = train_metadata.get("best_state_dict")
    if best_state is None:
        raise RuntimeError("Training did not retain the best state dictionary.")
    payload = {
        "format_version": 1,
        "model_kwargs": model_kwargs,
        "model_state_dict": {
            key: value.detach().cpu().clone() for key, value in best_state.items()
        },
        "training": {
            "best_epoch": int(train_metadata["best_epoch"]),
            "stopped_epoch": int(train_metadata["stopped_epoch"]),
            "best_val_loss": float(train_metadata["best_val_loss"]),
            "checkpoint_selection": "best",
        },
        "train_trial_indices": torch.from_numpy(train_indices),
        "validation_trial_indices": torch.from_numpy(validation_indices),
        "time_scale": float(time_scale),
        "dataset_npz": str(args.dataset_npz.resolve()),
        "seed": int(args.seed),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _plot_distances(
    output_dir: Path,
    *,
    native_time: np.ndarray,
    flow_distance: np.ndarray,
    classical_bin_centers: np.ndarray,
    classical_distance: np.ndarray,
    ground_truth: np.ndarray,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 11,
            "axes.grid": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )
    figure, axis = plt.subplots(figsize=(4.0, 3.5), layout="constrained")
    axis.plot(
        classical_bin_centers,
        classical_distance,
        color="#4477AA",
        linewidth=1.8,
        marker="o",
        markersize=3.8,
        label="Classical",
        zorder=3,
    )
    axis.plot(
        native_time,
        flow_distance,
        color="#CC6677",
        linewidth=2.0,
        label="Flow",
        zorder=2,
    )
    axis.plot(
        native_time,
        ground_truth,
        color="0.15",
        linewidth=2.0,
        linestyle="--",
        label="Ground truth",
        zorder=1,
    )
    upper = max(
        float(np.max(classical_distance)),
        float(np.max(flow_distance)),
        np.finfo(np.float64).eps,
    )
    axis.set_ylim(-0.08 * upper, 1.08 * upper)
    axis.set_xlim(float(native_time[0]), float(native_time[-1]))
    axis.set_xlabel("Time")
    axis.set_ylabel("Correlation distance")
    axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        borderaxespad=0.0,
        handlelength=1.8,
    )
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.8)
    axis.spines["bottom"].set_linewidth(1.8)
    axis.tick_params(width=1.8)
    stem = "correlation_distance_classical_flow_ground_truth"
    figure.savefig(output_dir / f"{stem}.png", dpi=300)
    figure.savefig(output_dir / f"{stem}.svg")
    plt.close(figure)


def _plot_losses(
    output_dir: Path,
    train_metadata: dict[str, Any],
    *,
    validation_mode: str,
) -> None:
    train = np.asarray(train_metadata["train_losses"], dtype=np.float64)
    validation = np.asarray(train_metadata["val_losses"], dtype=np.float64)
    monitored = np.asarray(train_metadata["val_monitor_losses"], dtype=np.float64)
    epochs = np.arange(1, train.size + 1)
    figure, axis = plt.subplots(figsize=(4.0, 3.5), layout="constrained")
    axis.plot(epochs, train, color="#4477AA", linewidth=1.0, alpha=0.6, label="Training")
    held_out = validation_mode == "held-out trials"
    axis.plot(
        epochs,
        validation,
        color="#CC6677",
        linewidth=1.0,
        alpha=0.6,
        label="Validation" if held_out else "Selection loss",
    )
    axis.plot(
        epochs,
        monitored,
        color="#228833",
        linewidth=1.8,
        label="EMA validation" if held_out else "EMA selection",
    )
    axis.axvline(
        int(train_metadata["best_epoch"]),
        color="0.15",
        linestyle=":",
        linewidth=1.6,
        label="Selected epoch",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Flow-matching loss")
    axis.legend(frameon=False, fontsize=10)
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.8)
    axis.spines["bottom"].set_linewidth(1.8)
    axis.tick_params(width=1.8)
    figure.savefig(output_dir / "flow_loss_vs_epoch.png", dpi=300)
    figure.savefig(output_dir / "flow_loss_vs_epoch.svg")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.device != "cuda:0":
        raise ValueError("This project requires --device cuda:0.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing CPU fallback.")
    torch.cuda.set_device(0)
    device = torch.device(args.device)
    _seed_all(int(args.seed))

    with np.load(args.dataset_npz, allow_pickle=False) as archive:
        responses = np.asarray(archive["responses"], dtype=np.float64)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        native_time = np.asarray(archive["time"], dtype=np.float64)
        true_class_means = np.asarray(archive["true_class_means"], dtype=np.float64)
        dataset_ground_truth = (
            np.asarray(archive["true_correlation_distance"], dtype=np.float64)
            if "true_correlation_distance" in archive.files
            else None
        )
    with np.load(args.classical_npz, allow_pickle=False) as archive:
        classical_bin_centers = np.asarray(archive["bin_centers"], dtype=np.float64)
        classical_distance = np.asarray(
            archive["estimated_correlation_distance"], dtype=np.float64
        )
        classical_binned_ground_truth = np.asarray(
            archive["true_correlation_distance"], dtype=np.float64
        )
    train_indices, validation_indices = _stratified_trial_split(
        labels, float(args.validation_fraction), int(args.seed)
    )
    time_scale = max(float(np.max(np.abs(native_time))), np.finfo(np.float64).eps)
    theta_train, x_train = _flatten_trials(
        responses, labels, native_time, train_indices, time_scale
    )
    if validation_indices.size:
        theta_validation, x_validation = _flatten_trials(
            responses, labels, native_time, validation_indices, time_scale
        )
        validation_mode = "held-out trials"
    else:
        theta_validation = theta_train
        x_validation = x_train
        validation_mode = (
            "training endpoints with independent fixed flow-path noise; "
            "not held-out data"
        )
    model_kwargs: dict[str, Any] = {
        "velocity_family": VELOCITY_FAMILY,
        "theta_dim": 3,
        "x_dim": int(responses.shape[-1]),
        "radius": 1.0,
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "network_architecture": "mlp",
        "path_schedule": "cosine",
        "divergence_estimator": str(args.divergence_estimator),
        "hutchinson_probes": int(args.hutchinson_probes),
    }
    model = build_flow_skl_model(**model_kwargs).to(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    train_metadata = train_flow_skl_model(
        model=model,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_validation,
        x_val=x_validation,
        device=device,
        velocity_family=VELOCITY_FAMILY,
        path_schedule="cosine",
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.learning_rate),
        weight_decay=1e-5,
        patience=int(args.patience),
        min_delta=1e-4,
        ema_alpha=0.1,
        max_grad_norm=10.0,
        log_every=max(10, min(500, int(args.epochs) // 20)),
        checkpoint_selection="best",
        fixed_validation=True,
        validation_seed=int(args.seed) + 10_000,
        retain_best_state=True,
        device_resident_data=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "correlation_flow_best.pt"
    _save_checkpoint(
        checkpoint_path,
        model_kwargs=model_kwargs,
        train_metadata=train_metadata,
        train_indices=train_indices,
        validation_indices=validation_indices,
        time_scale=time_scale,
        args=args,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    evaluation_model = build_flow_skl_model(**checkpoint["model_kwargs"]).to(device)
    evaluation_model.load_state_dict(checkpoint["model_state_dict"])
    evaluation_model.eval()
    flow_distance, flow_endpoint_means = _flow_correlation_distance(
        evaluation_model, native_time, time_scale, device
    )
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started

    ground_truth_from_means = np.empty(native_time.size, dtype=np.float64)
    for time_index in range(native_time.size):
        first = true_class_means[0, time_index]
        second = true_class_means[1, time_index]
        first = first - np.mean(first)
        second = second - np.mean(second)
        correlation = np.dot(first, second) / (
            np.linalg.norm(first) * np.linalg.norm(second)
        )
        ground_truth_from_means[time_index] = 1.0 - np.clip(
            correlation, -1.0, 1.0
        )
    if dataset_ground_truth is None:
        ground_truth = ground_truth_from_means
    else:
        np.testing.assert_allclose(
            dataset_ground_truth,
            ground_truth_from_means,
            atol=2e-12,
            rtol=0.0,
        )
        ground_truth = dataset_ground_truth
    flow_at_classical_bins = np.interp(
        classical_bin_centers, native_time, flow_distance
    )
    ground_truth_at_classical_bins = np.interp(
        classical_bin_centers, native_time, ground_truth
    )
    classical_mean_absolute_error = float(
        np.mean(np.abs(classical_distance - ground_truth_at_classical_bins))
    )
    flow_mean_absolute_error = float(np.mean(np.abs(flow_distance - ground_truth)))
    flow_bin_matched_mean_absolute_error = float(
        np.mean(np.abs(flow_at_classical_bins - ground_truth_at_classical_bins))
    )

    np.savez_compressed(
        args.output_dir / "correlation_classical_flow_results.npz",
        native_time=native_time,
        flow_correlation_distance=flow_distance,
        flow_endpoint_means=flow_endpoint_means,
        classical_bin_centers=classical_bin_centers,
        classical_correlation_distance=classical_distance,
        flow_at_classical_bin_centers=flow_at_classical_bins,
        ground_truth_at_classical_bin_centers=ground_truth_at_classical_bins,
        classical_binned_ground_truth=classical_binned_ground_truth,
        true_correlation_distance=ground_truth,
        train_trial_indices=train_indices,
        validation_trial_indices=validation_indices,
        train_losses=np.asarray(train_metadata["train_losses"], dtype=np.float64),
        validation_losses=np.asarray(train_metadata["val_losses"], dtype=np.float64),
        monitored_validation_losses=np.asarray(
            train_metadata["val_monitor_losses"], dtype=np.float64
        ),
    )
    if not bool(args.skip_plots):
        _plot_distances(
            args.output_dir,
            native_time=native_time,
            flow_distance=flow_distance,
            classical_bin_centers=classical_bin_centers,
            classical_distance=classical_distance,
            ground_truth=ground_truth,
        )
        _plot_losses(
            args.output_dir,
            train_metadata,
            validation_mode=validation_mode,
        )
    summary = {
        "dataset_npz": str(args.dataset_npz.resolve()),
        "classical_npz": str(args.classical_npz.resolve()),
        "velocity_family": VELOCITY_FAMILY,
        "condition": "two-class one-hot concatenated with time normalized to [-1, 1]",
        "endpoint_constraint": "centered across features and unit norm",
        "flow_readout": "one minus endpoint-mean dot product",
        "divergence_estimator": str(args.divergence_estimator),
        "hutchinson_probes": int(args.hutchinson_probes),
        "train_trials": int(train_indices.size),
        "validation_trials": int(validation_indices.size),
        "validation_mode": validation_mode,
        "train_samples": int(theta_train.shape[0]),
        "validation_samples": int(theta_validation.shape[0]),
        "best_epoch": int(train_metadata["best_epoch"]),
        "stopped_epoch": int(train_metadata["stopped_epoch"]),
        "best_validation_loss": float(train_metadata["best_val_loss"]),
        "elapsed_seconds": float(elapsed_seconds),
        "classical_distance_min": float(np.min(classical_distance)),
        "classical_distance_max": float(np.max(classical_distance)),
        "flow_distance_min": float(np.min(flow_distance)),
        "flow_distance_max": float(np.max(flow_distance)),
        "classical_mean_absolute_error": classical_mean_absolute_error,
        "flow_mean_absolute_error": flow_mean_absolute_error,
        "flow_bin_matched_mean_absolute_error": flow_bin_matched_mean_absolute_error,
        "bin_matched_error_ratio_classical_over_flow": float(
            classical_mean_absolute_error / flow_bin_matched_mean_absolute_error
        ),
        "ground_truth_distance_min": float(np.min(ground_truth)),
        "ground_truth_distance_max": float(np.max(ground_truth)),
        "seed": int(args.seed),
        "device": args.device,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"[flow-correlation] output={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
