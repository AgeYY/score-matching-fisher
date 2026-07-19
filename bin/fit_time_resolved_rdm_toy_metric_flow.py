#!/usr/bin/env python3
"""Compare one classical and constrained-flow metric on the time-resolved toy."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher.bci_iv_2a_session_identification import (  # noqa: E402
    _time_conditioned_endpoint_covariances,
)
from fisher.flow_matching_skl import (  # noqa: E402
    build_flow_skl_model,
    train_flow_skl_model,
)
from fisher.time_resolved_rdm_toy import (  # noqa: E402
    estimate_binned_metric_distance,
    gaussian_fid_distance,
    mean_vector_distance,
    population_distance_trajectory,
    squared_mahalanobis_distance,
)
from global_setting import EARLY_STOPPING_PATIENCE, TRAINING_MAX_EPOCHS  # noqa: E402


METRICS = ("cosine", "euclidean", "mahalanobis_sq", "fid")
VELOCITY_FAMILIES = {
    "cosine": "translation_fixed_norm",
    "euclidean": "translation",
    "mahalanobis_sq": "covariate_affine_diag",
    "fid": "condition_affine_diag",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric", choices=METRICS, required=True)
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20_260_718)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=TRAINING_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--quadrature-steps", type=int, default=32)
    parser.add_argument("--covariance-ode-steps", type=int, default=48)
    parser.add_argument("--covariance-ridge", type=float, default=1e-5)
    parser.add_argument(
        "--divergence-estimator",
        choices=("exact", "hutchinson"),
        default="hutchinson",
    )
    parser.add_argument("--hutchinson-probes", type=int, default=4)
    return parser.parse_args()


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _stratified_trial_split(
    labels: np.ndarray, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    fraction = float(validation_fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("validation_fraction must lie in [0, 1).")
    if fraction == 0.0:
        return np.arange(y.size, dtype=np.int64), np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    train: list[np.ndarray] = []
    validation: list[np.ndarray] = []
    for class_index in range(2):
        candidates = rng.permutation(np.flatnonzero(y == class_index))
        n_validation = max(1, int(np.floor(fraction * candidates.size)))
        if n_validation >= candidates.size:
            raise ValueError("Each class must retain at least one training trial.")
        validation.append(candidates[:n_validation])
        train.append(candidates[n_validation:])
    return (
        np.sort(np.concatenate(train)).astype(np.int64),
        np.sort(np.concatenate(validation)).astype(np.int64),
    )


def _condition_design(
    labels: np.ndarray, times: np.ndarray, time_scale: float
) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    time_values = np.asarray(times, dtype=np.float64).reshape(-1)
    if y.shape != time_values.shape or np.any((y < 0) | (y > 1)):
        raise ValueError("labels and times must be aligned, with labels in {0,1}.")
    return np.concatenate(
        [np.eye(2, dtype=np.float64)[y], (time_values / float(time_scale))[:, None]],
        axis=1,
    )


def _flatten_trials(
    responses: np.ndarray,
    labels: np.ndarray,
    native_time: np.ndarray,
    trial_indices: np.ndarray,
    time_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(trial_indices, dtype=np.int64)
    n_times = int(native_time.size)
    x = np.asarray(responses[selected], dtype=np.float64).reshape(
        -1, responses.shape[-1]
    )
    theta = _condition_design(
        np.repeat(labels[selected], n_times),
        np.tile(native_time, selected.size),
        time_scale,
    )
    return theta, x


def _model_kwargs(args: argparse.Namespace, x_dim: int) -> dict[str, Any]:
    family = VELOCITY_FAMILIES[str(args.metric)]
    kwargs: dict[str, Any] = {
        "velocity_family": family,
        "theta_dim": 3,
        "x_dim": int(x_dim),
        "radius": 1.0,
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "network_architecture": "mlp",
        "quadrature_steps": int(args.quadrature_steps),
        "path_schedule": "cosine",
        "divergence_estimator": str(args.divergence_estimator),
        "hutchinson_probes": int(args.hutchinson_probes),
    }
    if family == "covariate_affine_diag":
        # The affine matrix sees physical time only, enforcing one covariance
        # shared by the two classes but varying continuously with toy time.
        kwargs["affine_condition_indices"] = (2,)
    return kwargs


@torch.no_grad()
def _flow_endpoint_means(
    model: torch.nn.Module,
    native_time: np.ndarray,
    time_scale: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.repeat(np.arange(2, dtype=np.int64), native_time.size)
    times = np.tile(native_time, 2)
    conditions = _condition_design(labels, times, time_scale)
    theta = torch.from_numpy(conditions.astype(np.float32)).to(
        device=device, dtype=next(model.parameters()).dtype
    )
    means = (
        model.endpoint_mean(theta)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
        .reshape(2, native_time.size, -1)
    )
    return means, conditions


def _flow_distance(
    model: torch.nn.Module,
    metric: str,
    native_time: np.ndarray,
    time_scale: float,
    device: torch.device,
    *,
    covariance_ode_steps: int,
    covariance_ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means, all_conditions = _flow_endpoint_means(
        model, native_time, time_scale, device
    )
    distance = np.empty(native_time.size, dtype=np.float64)
    if metric in {"cosine", "euclidean"}:
        for time_index in range(native_time.size):
            distance[time_index] = mean_vector_distance(
                means[0, time_index], means[1, time_index], metric
            )
        return distance, means, np.empty((0, 0, 0), dtype=np.float64)

    if metric == "mahalanobis_sq":
        time_conditions = _condition_design(
            np.zeros(native_time.size, dtype=np.int64),
            native_time,
            time_scale,
        )
        covariances = _time_conditioned_endpoint_covariances(
            model,
            time_conditions,
            device=device,
            steps=int(covariance_ode_steps),
            ridge=float(covariance_ridge),
        )
        for time_index in range(native_time.size):
            distance[time_index] = squared_mahalanobis_distance(
                means[0, time_index],
                means[1, time_index],
                covariances[time_index],
                ridge=float(covariance_ridge),
            )
        return distance, means, covariances

    if metric != "fid":
        raise ValueError(f"Unsupported metric: {metric!r}.")
    covariance_flat = _time_conditioned_endpoint_covariances(
        model,
        all_conditions,
        device=device,
        steps=int(covariance_ode_steps),
        ridge=float(covariance_ridge),
    )
    covariances = covariance_flat.reshape(
        2, native_time.size, means.shape[-1], means.shape[-1]
    )
    for time_index in range(native_time.size):
        distance[time_index] = gaussian_fid_distance(
            means[0, time_index],
            covariances[0, time_index],
            means[1, time_index],
            covariances[1, time_index],
        )
    return distance, means, covariances


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
        "checkpoint_role": "best_validation_model_used_for_metric_evaluation",
        "metric": str(args.metric),
        "velocity_family": VELOCITY_FAMILIES[str(args.metric)],
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
        "covariance_ode_steps": int(args.covariance_ode_steps),
        "covariance_ridge": float(args.covariance_ridge),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if str(args.device) != "cuda:0":
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
        true_means = np.asarray(archive["true_class_means"], dtype=np.float64)
        true_covariances = np.asarray(
            archive["true_shared_covariances"], dtype=np.float64
        )

    classical = estimate_binned_metric_distance(
        responses,
        labels,
        native_time,
        bin_width=float(args.bin_width),
        metric=str(args.metric),
    )
    ground_truth = population_distance_trajectory(
        true_means, true_covariances, str(args.metric)
    )
    ground_truth_at_bins = np.interp(
        classical["bin_centers"], native_time, ground_truth
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
        theta_validation, x_validation = theta_train, x_train
        validation_mode = (
            "training endpoints with independent fixed flow-path noise; not held-out data"
        )

    model_kwargs = _model_kwargs(args, responses.shape[-1])
    family = VELOCITY_FAMILIES[str(args.metric)]
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
        velocity_family=family,
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
    checkpoint_path = args.output_dir / f"{args.metric}_flow_best.pt"
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
    flow_distance, flow_means, flow_covariances = _flow_distance(
        evaluation_model,
        str(args.metric),
        native_time,
        time_scale,
        device,
        covariance_ode_steps=int(args.covariance_ode_steps),
        covariance_ridge=float(args.covariance_ridge),
    )
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    flow_at_bins = np.interp(classical["bin_centers"], native_time, flow_distance)
    classical_distance = np.asarray(classical["estimated_distance"], dtype=np.float64)
    classical_mae = float(np.mean(np.abs(classical_distance - ground_truth_at_bins)))
    flow_mae = float(np.mean(np.abs(flow_at_bins - ground_truth_at_bins)))

    results_path = args.output_dir / f"{args.metric}_classical_flow_results.npz"
    np.savez_compressed(
        results_path,
        metric=np.asarray(str(args.metric)),
        native_time=native_time,
        ground_truth=ground_truth,
        classical_bin_centers=classical["bin_centers"],
        classical_distance=classical_distance,
        ground_truth_at_classical_bin_centers=ground_truth_at_bins,
        flow_distance=flow_distance,
        flow_at_classical_bin_centers=flow_at_bins,
        flow_endpoint_means=flow_means,
        flow_endpoint_covariances=flow_covariances,
        train_trial_indices=train_indices,
        validation_trial_indices=validation_indices,
        train_losses=np.asarray(train_metadata["train_losses"], dtype=np.float64),
        validation_losses=np.asarray(train_metadata["val_losses"], dtype=np.float64),
        monitored_validation_losses=np.asarray(
            train_metadata["val_monitor_losses"], dtype=np.float64
        ),
    )
    summary = {
        "metric": str(args.metric),
        "dataset_npz": str(args.dataset_npz.resolve()),
        "results_npz": str(results_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "classical_estimator": (
            "500 ms bins; pooled class means; diagonal pooled within-class variance "
            "for squared Mahalanobis or diagonal class-specific variances for FID"
        ),
        "velocity_family": family,
        "covariance_sharing": (
            "diagonal, shared across class, and continuous in time"
            if str(args.metric) == "mahalanobis_sq"
            else "diagonal and class-and-time-specific"
            if str(args.metric) == "fid"
            else "not used"
        ),
        "distance_convention": (
            "squared Mahalanobis"
            if str(args.metric) == "mahalanobis_sq"
            else "squared Gaussian 2-Wasserstein"
            if str(args.metric) == "fid"
            else "literal Euclidean"
            if str(args.metric) == "euclidean"
            else "one minus cosine similarity"
        ),
        "covariance_structure": (
            "diagonal affine, matching the exact diagonal covariance of the toy population"
            if str(args.metric) in {"mahalanobis_sq", "fid"}
            else "not applicable"
        ),
        "classical_mean_absolute_error": classical_mae,
        "flow_bin_matched_mean_absolute_error": flow_mae,
        "bin_matched_error_ratio_classical_over_flow": float(classical_mae / flow_mae),
        "ground_truth_min": float(np.min(ground_truth)),
        "ground_truth_max": float(np.max(ground_truth)),
        "train_trials": int(train_indices.size),
        "validation_trials": int(validation_indices.size),
        "validation_mode": validation_mode,
        "train_samples": int(theta_train.shape[0]),
        "validation_samples": int(theta_validation.shape[0]),
        "best_epoch": int(train_metadata["best_epoch"]),
        "stopped_epoch": int(train_metadata["stopped_epoch"]),
        "best_validation_loss": float(train_metadata["best_val_loss"]),
        "elapsed_seconds": float(elapsed_seconds),
        "seed": int(args.seed),
        "device": str(args.device),
        "divergence_estimator": str(args.divergence_estimator),
        "hutchinson_probes": int(args.hutchinson_probes),
        "covariance_ode_steps": int(args.covariance_ode_steps),
        "covariance_ridge": float(args.covariance_ridge),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"[toy-metric] output={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
