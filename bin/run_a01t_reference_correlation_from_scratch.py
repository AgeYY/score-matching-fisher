#!/usr/bin/env python3
"""Train and plot one recording's reference correlation or Mahalanobis RDMs."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from global_setting import EARLY_STOPPING_PATIENCE, TRAINING_MAX_EPOCHS  # noqa: E402
from fisher.bci_iv_2a_dataset import load_features_npz  # noqa: E402
from fisher.bci_iv_2a_session_identification import (  # noqa: E402
    FlowRDMConfig,
    _stratified_validation_trials,
    _time_conditioned_endpoint_covariances,
    condition_design,
    empirical_condition_means,
    empirical_gaussian_components,
    per_class_counts,
    rdms_from_means_and_precisions,
)
from fisher.flow_matching_skl import build_flow_skl_model, train_flow_skl_model  # noqa: E402


ROLE = "reference"
VELOCITY_FAMILIES = {
    "correlation": "translation_centered_fixed_norm",
    "mahalanobis": "covariate_affine",
}
VISIBLE_CUE_INTERVAL = (0.0, 1.25)
DEFAULT_TIME_RBF_NUM_CENTERS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording",
        default="A03T",
        help="BCI IV-2a training recording (default: A03T).",
    )
    parser.add_argument(
        "--metric",
        choices=tuple(VELOCITY_FAMILIES),
        default="correlation",
        help="RDM metric to estimate (default: correlation).",
    )
    parser.add_argument(
        "--feature-file",
        type=Path,
        default=None,
        help="Override the feature file inferred from --recording.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Override the mixed half-split file inferred from --recording.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to a recording-specific directory under data/.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=40_260_715)
    parser.add_argument("--epochs", type=int, default=TRAINING_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="GPU-resident batch size; use 1024 to reproduce the slower baseline.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument(
        "--eeg-time-embedding",
        choices=("identity", "gaussian-rbf"),
        default="gaussian-rbf",
        help="Embedding of EEG time u for the Mahalanobis affine model.",
    )
    parser.add_argument(
        "--time-rbf-num-centers",
        type=int,
        default=DEFAULT_TIME_RBF_NUM_CENTERS,
        help=f"Number of Gaussian RBF centers for EEG time (default: {DEFAULT_TIME_RBF_NUM_CENTERS}).",
    )
    parser.add_argument(
        "--time-rbf-bandwidth",
        type=float,
        default=None,
        help="Gaussian RBF bandwidth in scaled EEG-time units; default is center spacing.",
    )
    parser.add_argument(
        "--trial-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of reference trials retained independently within each class; "
            "sampling is without replacement and uses --seed (default: 0.5)."
        ),
    )
    return parser.parse_args()


def correlation_rdms_from_means(means: np.ndarray) -> np.ndarray:
    """Return time-resolved correlation-distance RDMs from condition means."""

    values = np.asarray(means, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 4:
        raise ValueError("means must have shape [time, 4, channels].")
    centered = values - np.mean(values, axis=2, keepdims=True)
    norms = np.linalg.norm(centered, axis=2, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Correlation distance is undefined for a zero-norm mean vector.")
    unit = centered / norms
    rdms = np.clip(1.0 - np.einsum("tcf,tdf->tcd", unit, unit), 0.0, 2.0)
    diagonal = np.arange(4)
    rdms[:, diagonal, diagonal] = 0.0
    return rdms


def mean_off_diagonal_distance(rdms: np.ndarray) -> np.ndarray:
    upper = np.triu_indices(4, k=1)
    return np.mean(np.asarray(rdms)[:, upper[0], upper[1]], axis=1)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _stratified_trial_fraction(
    labels: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Select floor(fraction * class count) trials per class without replacement."""

    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("--trial-fraction must be in (0, 1].")
    if float(fraction) == 1.0:
        return np.arange(y.size, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    for label in range(4):
        candidates = np.flatnonzero(y == label)
        n_selected = int(np.floor(float(fraction) * candidates.size))
        if n_selected < 2:
            raise ValueError(
                f"Class {label} retains {n_selected} trials at "
                f"--trial-fraction={fraction}; at least two are required."
            )
        selected.append(rng.choice(candidates, size=n_selected, replace=False))
    return np.sort(np.concatenate(selected))


def _flow_config(args: argparse.Namespace) -> FlowRDMConfig:
    return FlowRDMConfig(
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


def _model_kwargs(
    metric: str,
    config: FlowRDMConfig,
    channels: int,
    times: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "velocity_family": VELOCITY_FAMILIES[metric],
        "theta_dim": 5,
        "x_dim": channels,
        "radius": 1.0,
        "hidden_dim": config.hidden_dim,
        "depth": config.depth,
        "quadrature_steps": config.quadrature_steps,
        "path_schedule": "cosine",
        "divergence_estimator": "exact",
    }
    if metric == "mahalanobis":
        # theta[:, 4] is physical EEG time, so A is shared across classes but
        # is allowed to vary with EEG time.
        kwargs["affine_condition_indices"] = (4,)
        if args.eeg_time_embedding == "gaussian-rbf":
            if int(args.time_rbf_num_centers) < 2:
                raise ValueError("--time-rbf-num-centers must be >= 2.")
            time_scale = max(float(np.max(np.abs(times))), np.finfo(np.float64).eps)
            lower = float(np.min(times) / time_scale)
            upper = float(np.max(times) / time_scale)
            spacing = (upper - lower) / float(int(args.time_rbf_num_centers) - 1)
            bandwidth = (
                spacing
                if args.time_rbf_bandwidth is None
                else float(args.time_rbf_bandwidth)
            )
            kwargs.update(
                {
                    "theta_embedding": "gaussian_rbf",
                    "theta_rbf_indices": (4,),
                    "theta_rbf_num_centers": int(args.time_rbf_num_centers),
                    "theta_rbf_lower": lower,
                    "theta_rbf_upper": upper,
                    "theta_rbf_bandwidth": bandwidth,
                }
            )
    return kwargs


@torch.no_grad()
def _flow_rdms(
    model: torch.nn.Module,
    metric: str,
    times: np.ndarray,
    config: FlowRDMConfig,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    labels = np.repeat(np.arange(4, dtype=np.int64), times.size)
    grid_times = np.tile(times, 4)
    theta = torch.from_numpy(
        condition_design(labels, grid_times).astype(np.float32)
    ).to(device=device, dtype=next(model.parameters()).dtype)
    means = (
        model.endpoint_mean(theta)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
        .reshape(4, times.size, -1)
        .transpose(1, 0, 2)
    )
    components = {"flow_means": means}
    if metric == "correlation":
        return correlation_rdms_from_means(means), components
    if metric != "mahalanobis":
        raise ValueError(f"Unsupported metric: {metric!r}.")
    time_conditions = condition_design(np.zeros(times.size, dtype=np.int64), times)
    covariances = _time_conditioned_endpoint_covariances(
        model,
        time_conditions,
        device=device,
        steps=int(config.covariance_ode_steps),
        ridge=float(config.covariance_ridge),
    )
    distance_covariances = covariances + float(config.covariance_ridge) * np.eye(
        covariances.shape[-1], dtype=np.float64
    )[None, :, :]
    precisions = np.linalg.inv(distance_covariances)
    components.update(
        {
            "flow_endpoint_covariances": covariances,
            "flow_distance_covariances": distance_covariances,
            "flow_precisions": precisions,
        }
    )
    return rdms_from_means_and_precisions(means, precisions), components


def _save_and_reload_best(
    path: Path,
    *,
    metric: str,
    model_kwargs: dict[str, Any],
    train_meta: dict[str, Any],
    config: FlowRDMConfig,
    times: np.ndarray,
    context: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    best_state = train_meta.get("best_state_dict")
    if best_state is None:
        raise RuntimeError("Training did not retain its best validation state.")
    payload = {
        "format_version": 1,
        "checkpoint_role": "best_validation_model_used_for_rdm_evaluation",
        "velocity_family": VELOCITY_FAMILIES[metric],
        "model_kwargs": model_kwargs,
        "model_state_dict": {
            key: value.detach().cpu().clone() for key, value in best_state.items()
        },
        "training": {
            "best_epoch": int(train_meta["best_epoch"]),
            "stopped_epoch": int(train_meta["stopped_epoch"]),
            "best_val_loss": float(train_meta["best_val_loss"]),
            "selected_epoch": int(train_meta["selected_epoch"]),
            "checkpoint_selection": str(train_meta["checkpoint_selection"]),
        },
        "flow_rdm_config": asdict(config),
        "time_centers": torch.from_numpy(times.astype(np.float64)),
        "context": context,
    }
    torch.save(payload, path)
    reloaded = torch.load(path, map_location="cpu", weights_only=True)
    model = build_flow_skl_model(**reloaded["model_kwargs"]).to(device)
    model.load_state_dict(reloaded["model_state_dict"])
    model.eval()
    rdms, components = _flow_rdms(model, metric, times, config, device)
    return rdms, reloaded, components


def _plot(
    output_dir: Path,
    recording: str,
    metric: str,
    times: np.ndarray,
    classical_curve: np.ndarray,
    flow_curve: np.ndarray,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "axes.grid": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "savefig.bbox": "tight",
        }
    )
    figure, axis = plt.subplots(figsize=(4.0, 3.5))
    axis.axvspan(*VISIBLE_CUE_INTERVAL, color="0.93", linewidth=0, zorder=0)
    axis.axvline(0.0, color="0.35", linestyle="--", linewidth=1.3, zorder=1)
    axis.plot(times, classical_curve, color="#4477AA", linewidth=1.5, label="Classical")
    axis.plot(times, flow_curve, color="#CC6677", linewidth=2.0, label="Flow-based")
    axis.set_xlim(float(times[0]), float(times[-1]))
    axis.set_ylim(0.0, 1.04 * float(max(np.max(classical_curve), np.max(flow_curve))))
    axis.set_xlabel("Time from cue onset (s)")
    axis.set_ylabel(
        "Mean correlation distance"
        if metric == "correlation"
        else "Mean Mahalanobis² distance"
    )
    axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        borderaxespad=0.0,
    )
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(width=1.8)
    prefix = recording.lower()
    figure.savefig(
        output_dir / f"{prefix}_reference_{metric}_distance_vs_time.png", dpi=300
    )
    figure.savefig(output_dir / f"{prefix}_reference_{metric}_distance_vs_time.svg")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    metric = str(args.metric)
    velocity_family = VELOCITY_FAMILIES[metric]
    recording = str(args.recording).strip().upper()
    if len(recording) != 4 or recording[0] != "A" or recording[-1] != "T":
        raise ValueError("--recording must look like A01T through A09T.")
    if args.feature_file is None:
        args.feature_file = (
            ROOT
            / "data/bci_iv_2a/processed/native_voltage_all_timepoints_20uv"
            / f"{recording}.npz"
        )
    if args.split_file is None:
        args.split_file = (
            ROOT
            / "data/bci_iv_2a/fid_session_identification_9recordings_mixed_runs"
            / "splits"
            / f"{recording}_mixed_half_split.npz"
        )
    if args.output_dir is None:
        args.output_dir = (
            ROOT
            / "data/bci_iv_2a"
            / f"{recording.lower()}_reference_{metric}_from_scratch"
        )
    if args.device != "cuda:0":
        raise ValueError("This project requires --device cuda:0.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing CPU fallback.")
    torch.cuda.set_device(0)
    device = torch.device(args.device)
    _seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()

    dataset = load_features_npz(args.feature_file)
    with np.load(args.split_file, allow_pickle=False) as split:
        reference_indices = np.asarray(split["reference_indices"], dtype=np.int64)
        saved_labels = np.asarray(split["labels"], dtype=np.int64)
        split_seed = int(split["split_seed"].item())
    if not np.array_equal(saved_labels, dataset.labels):
        raise RuntimeError("Feature and split labels differ.")
    parent_reference_indices = reference_indices.copy()
    parent_labels = np.asarray(dataset.labels[parent_reference_indices], dtype=np.int64)
    retained = _stratified_trial_fraction(
        parent_labels,
        fraction=args.trial_fraction,
        seed=args.seed + 20_000,
    )
    reference_indices = parent_reference_indices[retained]
    values = np.asarray(dataset.features[reference_indices], dtype=np.float64)
    labels = np.asarray(dataset.labels[reference_indices], dtype=np.int64)
    times = np.asarray(dataset.time_centers, dtype=np.float64)

    classical_started = time.perf_counter()
    if metric == "correlation":
        classical_means = empirical_condition_means(
            values, labels, standardize_features=False
        )
        classical_rdms = correlation_rdms_from_means(classical_means)
        classical_components = {"classical_means": classical_means}
    else:
        classical_means, classical_covariances, classical_precisions = (
            empirical_gaussian_components(
                values, labels, standardize_features=False
            )
        )
        classical_rdms = rdms_from_means_and_precisions(
            classical_means, classical_precisions
        )
        classical_components = {
            "classical_means": classical_means,
            "classical_covariances": classical_covariances,
            "classical_precisions": classical_precisions,
        }
    classical_seconds = time.perf_counter() - classical_started

    config = _flow_config(args)
    train_trials, validation_trials = _stratified_validation_trials(
        labels, config.validation_fraction, args.seed
    )

    def flatten(trials: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = values[trials].reshape(-1, values.shape[-1])
        y = np.repeat(labels[trials], times.size)
        t = np.tile(times, trials.size)
        return condition_design(y, t), x

    theta_train, x_train = flatten(train_trials)
    theta_validation, x_validation = flatten(validation_trials)
    model_kwargs = _model_kwargs(metric, config, values.shape[-1], times, args)
    model = build_flow_skl_model(**model_kwargs).to(device)
    torch.cuda.synchronize(device)
    flow_started = time.perf_counter()
    train_meta = train_flow_skl_model(
        model=model,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_validation,
        x_val=x_validation,
        device=device,
        velocity_family=velocity_family,
        path_schedule="cosine",
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        patience=config.patience,
        min_delta=1e-4,
        ema_alpha=0.1,
        max_grad_norm=10.0,
        log_every=max(10, min(500, config.epochs // 20)),
        checkpoint_selection="best",
        fixed_validation=True,
        validation_seed=args.seed + 10_000,
        retain_best_state=True,
        device_resident_data=True,
    )
    context = {
        "recording": recording,
        "role": ROLE,
        "metric": metric,
        "split_seed": split_seed,
        "trial_fraction": float(args.trial_fraction),
        "parent_trial_indices": parent_reference_indices.tolist(),
        "trial_indices": reference_indices.tolist(),
        "per_class_counts": per_class_counts(labels).tolist(),
    }
    prefix = recording.lower()
    checkpoint_path = args.output_dir / f"{prefix}_reference_{metric}_flow_best.pt"
    flow_rdms, checkpoint, flow_components = _save_and_reload_best(
        checkpoint_path,
        metric=metric,
        model_kwargs=model_kwargs,
        train_meta=train_meta,
        config=config,
        times=times,
        context=context,
        device=device,
    )
    torch.cuda.synchronize(device)
    flow_seconds = time.perf_counter() - flow_started

    classical_curve = mean_off_diagonal_distance(classical_rdms)
    flow_curve = mean_off_diagonal_distance(flow_rdms)
    np.savez_compressed(
        args.output_dir / f"{prefix}_reference_{metric}_rdms.npz",
        classical_rdms=classical_rdms,
        flow_rdms=flow_rdms,
        classical_mean_distance=classical_curve,
        flow_mean_distance=flow_curve,
        time_seconds_cue_relative=times,
        reference_trial_indices=reference_indices,
        train_trial_indices=train_trials,
        validation_trial_indices=validation_trials,
        train_losses=np.asarray(train_meta["train_losses"], dtype=np.float64),
        validation_losses=np.asarray(train_meta["val_losses"], dtype=np.float64),
        monitored_validation_losses=np.asarray(
            train_meta["val_monitor_losses"], dtype=np.float64
        ),
        **classical_components,
        **flow_components,
    )
    _plot(args.output_dir, recording, metric, times, classical_curve, flow_curve)
    total_seconds = time.perf_counter() - total_started
    summary = {
        "experiment": f"{recording} reference {metric} RDM from scratch",
        "recording": recording,
        "metric": metric,
        "velocity_family": velocity_family,
        "covariance_sharing": (
            "not_applicable"
            if metric == "correlation"
            else "shared_across_classes_and_varying_with_eeg_time"
        ),
        "eeg_time_embedding": (
            args.eeg_time_embedding if metric == "mahalanobis" else "not_applicable"
        ),
        "time_rbf_num_centers": (
            int(args.time_rbf_num_centers)
            if metric == "mahalanobis" and args.eeg_time_embedding == "gaussian-rbf"
            else None
        ),
        "time_rbf_bandwidth": (
            model_kwargs.get("theta_rbf_bandwidth")
            if metric == "mahalanobis" and args.eeg_time_embedding == "gaussian-rbf"
            else None
        ),
        "cache_usage": "none; classical and flow RDMs recomputed on every invocation",
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0),
        "trial_fraction": float(args.trial_fraction),
        "n_parent_reference_trials": int(parent_reference_indices.size),
        "n_reference_trials": int(reference_indices.size),
        "per_class_counts": per_class_counts(labels).tolist(),
        "n_time_points": int(times.size),
        "time_interval_seconds": [float(times[0]), float(times[-1])],
        "classical_seconds": classical_seconds,
        "flow_training_checkpoint_reload_and_rdm_seconds": flow_seconds,
        "total_figure_pipeline_seconds": total_seconds,
        "best_epoch": int(checkpoint["training"]["best_epoch"]),
        "stopped_epoch": int(checkpoint["training"]["stopped_epoch"]),
        "best_validation_loss": float(checkpoint["training"]["best_val_loss"]),
        "config": asdict(config),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"[{recording} {metric}] output={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
