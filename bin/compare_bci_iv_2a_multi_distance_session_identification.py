#!/usr/bin/env python3
"""Compare native-time RDM distances in nine-recording BCI IV-2a identification."""

from __future__ import annotations

import argparse
import json
import math
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
    classical_mahalanobis_rdms,
    condition_affine_flow_gaussian_components_from_checkpoint,
    condition_design,
    empirical_condition_gaussian_components,
    empirical_condition_means,
    gaussian_jeffreys_rdms_from_moments,
    per_class_counts,
    rdms_from_means_and_precisions,
    squared_euclidean_rdms_from_means,
)
from fisher.flow_matching_skl import build_flow_skl_model, train_flow_skl_model  # noqa: E402


ROLES = ("reference", "all_trial")
FIT_METRICS = ("correlation", "cosine", "euclidean", "mahalanobis")
ALL_METRICS = ("correlation", "cosine", "euclidean", "mahalanobis", "fid", "jeffreys")
VELOCITY_FAMILIES = {
    "correlation": "translation_centered_fixed_norm",
    "cosine": "translation_fixed_norm",
    "euclidean": "translation",
    "mahalanobis": "covariate_affine",
}
METRIC_LABELS = {
    "correlation": "Correlation",
    "cosine": "Cosine",
    "euclidean": "Euclidean",
    "mahalanobis": "Mahalanobis²",
    "fid": "FID",
    "jeffreys": "Jeffreys",
}
VISIBLE_CUE_INTERVAL = (0.0, 1.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/processed/native_voltage_all_timepoints_20uv",
    )
    parser.add_argument(
        "--fid-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/fid_session_identification_9recordings_mixed_runs",
        help="Audited nine-recording FID run supplying splits and FID checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/multi_distance_session_identification_9recordings_full_query",
    )
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=[f"A{index:02d}T" for index in range(1, 10)],
    )
    parser.add_argument("--fit-metric", choices=FIT_METRICS)
    parser.add_argument(
        "--fit-recording",
        choices=[f"A{index:02d}T" for index in range(1, 10)],
        help="Restrict --fit-metric to one recording; useful for resumable jobs and smoke tests.",
    )
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--epochs", type=int, default=TRAINING_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--covariance-ode-steps", type=int, default=48)
    return parser.parse_args()


def _flow_config(args: argparse.Namespace) -> FlowRDMConfig:
    return FlowRDMConfig(
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=1e-5,
        patience=int(args.patience),
        quadrature_steps=32,
        covariance_ode_steps=int(args.covariance_ode_steps),
        covariance_ridge=1e-5,
        validation_fraction=0.2,
        standardize_features=False,
        device_resident_data=True,
    )


def _atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_cache(path: Path, rdms: np.ndarray, metadata: dict[str, Any]) -> None:
    _atomic_npz(
        path,
        rdms=np.asarray(rdms, dtype=np.float64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _load_cache(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        rdms = np.asarray(archive["rdms"], dtype=np.float64)
        metadata = json.loads(str(archive["metadata_json"].item()))
    if rdms.ndim != 3 or rdms.shape[1:] != (4, 4) or not np.isfinite(rdms).all():
        raise ValueError(f"Invalid RDM cache {path}: shape={rdms.shape}.")
    if float(np.min(rdms)) < -1e-8:
        raise ValueError(f"RDM cache {path} contains materially negative values.")
    np.testing.assert_allclose(rdms, rdms.transpose(0, 2, 1), atol=1e-7, rtol=0.0)
    np.testing.assert_allclose(
        np.diagonal(rdms, axis1=1, axis2=2),
        0.0,
        atol=1e-7,
        rtol=0.0,
    )
    return rdms, metadata


def _seed_all(seed: int, device: torch.device) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _metric_from_means(means: np.ndarray, metric: str) -> np.ndarray:
    values = np.asarray(means, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 4:
        raise ValueError("means must have shape [time, class, feature].")
    if metric == "euclidean":
        return np.sqrt(np.maximum(squared_euclidean_rdms_from_means(values), 0.0))
    normalized = values
    if metric == "correlation":
        normalized = normalized - np.mean(normalized, axis=2, keepdims=True)
    elif metric != "cosine":
        raise ValueError(f"Unsupported mean-only metric: {metric!r}.")
    norms = np.linalg.norm(normalized, axis=2, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{metric} distance is undefined for a zero-norm condition mean.")
    unit = normalized / norms
    rdms = 1.0 - np.einsum("tcf,tdf->tcd", unit, unit)
    rdms = np.clip(rdms, 0.0, 2.0)
    diagonal = np.arange(4)
    rdms[:, diagonal, diagonal] = 0.0
    return rdms


def _model_kwargs(metric: str, config: FlowRDMConfig, x_dim: int) -> dict[str, Any]:
    family = VELOCITY_FAMILIES[metric]
    kwargs: dict[str, Any] = {
        "velocity_family": family,
        "theta_dim": 5,
        "x_dim": int(x_dim),
        "radius": 1.0,
        "hidden_dim": int(config.hidden_dim),
        "depth": int(config.depth),
        "quadrature_steps": int(config.quadrature_steps),
        "path_schedule": "cosine",
        "divergence_estimator": "exact",
    }
    if family == "covariate_affine":
        kwargs["affine_condition_indices"] = (4,)
    return kwargs


@torch.no_grad()
def _flow_rdms_from_model(
    model: torch.nn.Module,
    metric: str,
    times: np.ndarray,
    config: FlowRDMConfig,
    device: torch.device,
) -> np.ndarray:
    grid_labels = np.repeat(np.arange(4, dtype=np.int64), times.size)
    grid_times = np.tile(times, 4)
    conditions = condition_design(grid_labels, grid_times)
    dtype = next(model.parameters()).dtype
    condition_tensor = torch.from_numpy(conditions.astype(np.float32)).to(
        device=device,
        dtype=dtype,
    )
    means_flat = model.endpoint_mean(condition_tensor).detach().cpu().numpy()
    means = means_flat.astype(np.float64).reshape(4, times.size, -1).transpose(1, 0, 2)
    if metric in {"correlation", "cosine", "euclidean"}:
        return _metric_from_means(means, metric)
    if metric != "mahalanobis":
        raise ValueError(f"Unsupported flow metric: {metric!r}.")
    time_conditions = condition_design(np.zeros(times.size, dtype=np.int64), times)
    covariances = _time_conditioned_endpoint_covariances(
        model,
        time_conditions,
        device=device,
        steps=int(config.covariance_ode_steps),
        ridge=float(config.covariance_ridge),
    )
    distance_covariances = covariances + float(config.covariance_ridge) * np.eye(
        covariances.shape[-1],
        dtype=np.float64,
    )[None, :, :]
    return rdms_from_means_and_precisions(means, np.linalg.inv(distance_covariances))


def _save_best_checkpoint(
    path: Path,
    *,
    model_kwargs: dict[str, Any],
    best_state: dict[str, torch.Tensor],
    train_meta: dict[str, Any],
    config: FlowRDMConfig,
    times: np.ndarray,
    context: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "checkpoint_role": "best_validation_model_used_for_rdm_evaluation",
            "velocity_family": model_kwargs["velocity_family"],
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
            "time_centers": torch.from_numpy(np.asarray(times, dtype=np.float64)),
            "context": dict(context),
        },
        temporary,
    )
    temporary.replace(path)


def _load_best_checkpoint_rdms(
    checkpoint_path: Path,
    *,
    metric: str,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(payload.get("format_version", -1)) != 1:
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}.")
    if payload.get("velocity_family") != VELOCITY_FAMILIES[metric]:
        raise ValueError(f"Velocity-family mismatch in {checkpoint_path}.")
    model = build_flow_skl_model(**payload["model_kwargs"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    config = FlowRDMConfig(**dict(payload["flow_rdm_config"]))
    times = payload["time_centers"].detach().cpu().numpy().astype(np.float64)
    return _flow_rdms_from_model(model, metric, times, config, device), payload


def _load_split(
    fid_dir: Path,
    recording: str,
    labels: np.ndarray,
) -> tuple[dict[str, np.ndarray], int]:
    path = fid_dir / "splits" / f"{recording}_mixed_half_split.npz"
    with np.load(path, allow_pickle=False) as archive:
        saved_labels = np.asarray(archive["labels"], dtype=np.int64)
        split_seed = int(archive["split_seed"].item())
        roles = {
            "reference": np.asarray(archive["reference_indices"], dtype=np.int64),
            "all_trial": np.asarray(archive["all_trial_indices"], dtype=np.int64),
        }
    if not np.array_equal(saved_labels, labels):
        raise RuntimeError(f"Split labels differ from feature labels for {recording}.")
    if np.intersect1d(roles["reference"], roles["all_trial"]).size:
        raise RuntimeError(f"Reference/query leakage for {recording}.")
    np.testing.assert_array_equal(
        np.union1d(roles["reference"], roles["all_trial"]),
        np.arange(labels.size),
    )
    return roles, split_seed


def _fit_metric(args: argparse.Namespace, metric: str, device: torch.device) -> None:
    config = _flow_config(args)
    metric_index = FIT_METRICS.index(metric)
    cache_dir = args.output_dir / "rdm_cache"
    checkpoint_dir = args.output_dir / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    selected_recordings = (
        list(args.recordings)
        if args.fit_recording is None
        else [str(args.fit_recording)]
    )
    for recording in selected_recordings:
        recording_index = list(args.recordings).index(recording)
        dataset = load_features_npz(args.feature_dir / f"{recording}.npz")
        role_indices, split_seed = _load_split(args.fid_dir, recording, dataset.labels)
        times = np.asarray(dataset.time_centers, dtype=np.float64)
        for role_index, role in enumerate(ROLES):
            indices = role_indices[role]
            cache_path = cache_dir / f"{role}_{recording}_{metric}_flow.npz"
            checkpoint_path = checkpoint_dir / f"{role}_{recording}_{metric}_flow_best.pt"
            seed = int(
                args.seed
                + 20_000_000
                + metric_index * 1_000_000
                + recording_index * 100_000
                + role_index * 10_000
            )
            context = {
                "metric": metric,
                "recording": recording,
                "role": role,
                "split_seed": split_seed,
                "n_trials": int(indices.size),
                "per_class_counts": per_class_counts(dataset.labels[indices]).astype(int).tolist(),
                "trial_indices": indices.astype(int).tolist(),
            }
            if cache_path.is_file() and checkpoint_path.is_file():
                _, metadata = _load_cache(cache_path)
                if metadata.get("config") != asdict(config):
                    raise RuntimeError(f"Cached config mismatch: {cache_path}")
                print(f"[cache] loaded {cache_path.name}", flush=True)
            else:
                _seed_all(seed, device)
                values = np.asarray(dataset.features[indices], dtype=np.float64)
                labels = np.asarray(dataset.labels[indices], dtype=np.int64)
                train_trials, val_trials = _stratified_validation_trials(
                    labels,
                    config.validation_fraction,
                    seed,
                )

                def flatten(trials: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                    x = values[trials].reshape(-1, values.shape[-1])
                    y = np.repeat(labels[trials], times.size)
                    t = np.tile(times, trials.size)
                    return condition_design(y, t), x

                theta_train, x_train = flatten(train_trials)
                theta_val, x_val = flatten(val_trials)
                model_kwargs = _model_kwargs(metric, config, values.shape[-1])
                model = build_flow_skl_model(**model_kwargs).to(device)
                start = time.perf_counter()
                train_meta = train_flow_skl_model(
                    model=model,
                    theta_train=theta_train,
                    x_train=x_train,
                    theta_val=theta_val,
                    x_val=x_val,
                    device=device,
                    velocity_family=VELOCITY_FAMILIES[metric],
                    path_schedule="cosine",
                    epochs=int(config.epochs),
                    batch_size=int(config.batch_size),
                    lr=float(config.learning_rate),
                    weight_decay=float(config.weight_decay),
                    patience=int(config.patience),
                    min_delta=1e-4,
                    ema_alpha=0.1,
                    max_grad_norm=10.0,
                    log_every=max(10, min(500, int(config.epochs) // 20)),
                    checkpoint_selection="best",
                    fixed_validation=True,
                    validation_seed=seed + 10_000,
                    retain_best_state=True,
                    device_resident_data=bool(config.device_resident_data),
                )
                best_state = train_meta.get("best_state_dict")
                if best_state is None:
                    raise RuntimeError("Training did not retain a best checkpoint.")
                _save_best_checkpoint(
                    checkpoint_path,
                    model_kwargs=model_kwargs,
                    best_state=best_state,
                    train_meta=train_meta,
                    config=config,
                    times=times,
                    context=context,
                )
                rdms, payload = _load_best_checkpoint_rdms(
                    checkpoint_path,
                    metric=metric,
                    device=device,
                )
                if payload["context"] != context:
                    raise RuntimeError(f"Checkpoint context mismatch: {checkpoint_path}")
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                metadata = {
                    **context,
                    "method": "flow",
                    "velocity_family": VELOCITY_FAMILIES[metric],
                    "rdm_evaluation_source": "explicitly_reloaded_best_validation_checkpoint",
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "best_epoch": int(train_meta["best_epoch"]),
                    "stopped_epoch": int(train_meta["stopped_epoch"]),
                    "best_val_loss": float(train_meta["best_val_loss"]),
                    "train_losses": np.asarray(train_meta["train_losses"], dtype=float).tolist(),
                    "validation_losses": np.asarray(train_meta["val_losses"], dtype=float).tolist(),
                    "monitored_validation_losses": np.asarray(
                        train_meta["val_monitor_losses"], dtype=float
                    ).tolist(),
                    "elapsed_seconds": float(elapsed),
                    "config": asdict(config),
                }
                _save_cache(cache_path, rdms, metadata)
                print(
                    f"[fit] {cache_path.name} elapsed={elapsed / 60.0:.2f}min "
                    f"best={metadata['best_epoch']} stopped={metadata['stopped_epoch']}",
                    flush=True,
                )
            reloaded, payload = _load_best_checkpoint_rdms(
                checkpoint_path,
                metric=metric,
                device=device,
            )
            cached, metadata = _load_cache(cache_path)
            max_difference = float(np.max(np.abs(reloaded - cached)))
            if max_difference > 1e-7:
                metadata = {
                    **metadata,
                    "pre_reload_cache_max_absolute_difference": max_difference,
                    "rdm_evaluation_source": "explicitly_reloaded_best_validation_checkpoint",
                }
                _save_cache(cache_path, reloaded, metadata)
            if payload["context"] != context:
                raise RuntimeError(f"Checkpoint context mismatch: {checkpoint_path}")
        print(f"[recording] {metric} {recording} complete", flush=True)
    _atomic_json(
        args.output_dir / f"fit_{metric}_summary.json",
        {
            "metric": metric,
            "velocity_family": VELOCITY_FAMILIES[metric],
            "recordings": selected_recordings,
            "config": asdict(config),
            "status": "complete",
        },
    )


def _classical_rdms(
    values: np.ndarray,
    labels: np.ndarray,
    metric: str,
) -> np.ndarray:
    if metric in {"correlation", "cosine", "euclidean"}:
        return _metric_from_means(
            empirical_condition_means(values, labels, standardize_features=False),
            metric,
        )
    if metric == "mahalanobis":
        return classical_mahalanobis_rdms(values, labels, standardize_features=False)
    if metric == "jeffreys":
        means, covariances = empirical_condition_gaussian_components(
            values,
            labels,
            standardize_features=False,
        )
        return gaussian_jeffreys_rdms_from_moments(means, covariances, ridge=1e-8)
    raise ValueError(f"Unsupported classical metric: {metric!r}.")


def _mse_matrix(
    query: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    times: np.ndarray,
    recordings: list[str],
) -> np.ndarray:
    mask = (times >= VISIBLE_CUE_INTERVAL[0]) & (times <= VISIBLE_CUE_INTERVAL[1])
    upper = np.triu_indices(4, k=1)
    output = np.empty((len(recordings), len(recordings)), dtype=np.float64)
    for query_index, query_recording in enumerate(recordings):
        query_vector = query[query_recording][mask][:, upper[0], upper[1]]
        for reference_index, reference_recording in enumerate(recordings):
            reference_vector = reference[reference_recording][mask][:, upper[0], upper[1]]
            output[query_index, reference_index] = float(
                np.mean((query_vector - reference_vector) ** 2, dtype=np.float64)
            )
    return output


def _rank_metrics(mse: np.ndarray, recordings: list[str]) -> dict[str, Any]:
    ranks: list[int] = []
    predictions: list[str] = []
    for query_index in range(len(recordings)):
        order = np.argsort(mse[query_index], kind="mergesort")
        ranks.append(int(np.flatnonzero(order == query_index)[0] + 1))
        predictions.append(recordings[int(order[0])])
    rank_array = np.asarray(ranks, dtype=np.int64)
    return {
        "top1_accuracy": float(np.mean(rank_array == 1)),
        "ranks": ranks,
        "predictions": predictions,
        "mean_reciprocal_rank": float(np.mean(1.0 / rank_array)),
    }


def _plot_grouped_top1(output_dir: Path, top1: np.ndarray) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    positions = np.arange(len(ALL_METRICS), dtype=np.float64)
    width = 0.38
    figure, axis = plt.subplots(figsize=(8.0, 3.5), layout="constrained")
    axis.bar(
        positions - width / 2.0,
        top1[0],
        width=width,
        color="#4477AA",
        linewidth=0.0,
        label="Classical",
    )
    axis.bar(
        positions + width / 2.0,
        top1[1],
        width=width,
        color="#CC6677",
        linewidth=0.0,
        label="Flow-based",
    )
    axis.set_xlim(-0.55, len(ALL_METRICS) - 0.45)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Top-1 accuracy")
    axis.set_xticks(positions, [METRIC_LABELS[metric] for metric in ALL_METRICS], rotation=20)
    axis.legend(frameon=False, ncol=2, loc="upper left")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(width=1.8)
    figure.savefig(output_dir / "multi_distance_full_query_9recordings_top1_bar.png", dpi=300, facecolor="white")
    figure.savefig(output_dir / "multi_distance_full_query_9recordings_top1_bar.svg", facecolor="white")
    plt.close(figure)


def _aggregate(args: argparse.Namespace, device: torch.device) -> None:
    recordings = list(args.recordings)
    datasets = [load_features_npz(args.feature_dir / f"{recording}.npz") for recording in recordings]
    times = np.asarray(datasets[0].time_centers, dtype=np.float64)
    if any(not np.array_equal(dataset.time_centers, times) for dataset in datasets[1:]):
        raise ValueError("Feature files do not share the same native-time grid.")
    cache_dir = args.output_dir / "rdm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    method_mse = np.empty((2, len(ALL_METRICS), len(recordings), len(recordings)), dtype=np.float64)
    metrics_summary: dict[str, Any] = {}

    for metric_index, metric in enumerate(ALL_METRICS):
        classical_reference: dict[str, np.ndarray] = {}
        classical_query: dict[str, np.ndarray] = {}
        flow_reference: dict[str, np.ndarray] = {}
        flow_query: dict[str, np.ndarray] = {}
        for recording, dataset in zip(recordings, datasets, strict=True):
            role_indices, split_seed = _load_split(args.fid_dir, recording, dataset.labels)
            for role in ROLES:
                target_classical = classical_reference if role == "reference" else classical_query
                target_flow = flow_reference if role == "reference" else flow_query
                if metric == "fid":
                    classical_path = args.fid_dir / "rdm_cache" / f"{role}_{recording}_classical_fid.npz"
                    flow_path = (
                        args.fid_dir
                        / "best_checkpoint_full_query_flow_rdm_cache"
                        / f"{role}_{recording}_condition_affine_flow_fid_best_reloaded.npz"
                    )
                    target_classical[recording] = _load_cache(classical_path)[0]
                    target_flow[recording] = _load_cache(flow_path)[0]
                    continue

                classical_path = cache_dir / f"{role}_{recording}_{metric}_classical.npz"
                if classical_path.is_file():
                    classical_rdm = _load_cache(classical_path)[0]
                else:
                    indices = role_indices[role]
                    start = time.perf_counter()
                    classical_rdm = _classical_rdms(
                        dataset.features[indices],
                        dataset.labels[indices],
                        metric,
                    )
                    _save_cache(
                        classical_path,
                        classical_rdm,
                        {
                            "recording": recording,
                            "role": role,
                            "metric": metric,
                            "method": "classical",
                            "split_seed": split_seed,
                            "n_trials": int(indices.size),
                            "per_class_counts": per_class_counts(dataset.labels[indices]).astype(int).tolist(),
                            "elapsed_seconds": float(time.perf_counter() - start),
                            "feature_standardization": "none",
                            "jeffreys_model": (
                                "class_and_time_specific_ledoit_wolf_gaussians"
                                if metric == "jeffreys"
                                else None
                            ),
                        },
                    )
                    classical_rdm = _load_cache(classical_path)[0]
                target_classical[recording] = classical_rdm

                if metric == "jeffreys":
                    checkpoint_path = (
                        args.fid_dir
                        / "checkpoints"
                        / f"{role}_{recording}_condition_affine_flow_fid_best.pt"
                    )
                    means, covariances, payload = (
                        condition_affine_flow_gaussian_components_from_checkpoint(
                            checkpoint_path,
                            device=device,
                        )
                    )
                    context = dict(payload.get("context", {}))
                    if context.get("recording") != recording or context.get("role") != role:
                        raise RuntimeError(f"FID checkpoint context mismatch: {checkpoint_path}")
                    flow_rdm = gaussian_jeffreys_rdms_from_moments(
                        means,
                        covariances,
                        ridge=1e-8,
                    )
                    flow_path = cache_dir / f"{role}_{recording}_jeffreys_flow.npz"
                    _save_cache(
                        flow_path,
                        flow_rdm,
                        {
                            "recording": recording,
                            "role": role,
                            "metric": "jeffreys",
                            "method": "flow",
                            "velocity_family": "condition_affine",
                            "readout": "analytic_gaussian_jeffreys_sum",
                            "source_checkpoint": str(checkpoint_path.resolve()),
                            "best_epoch": int(payload["training"]["best_epoch"]),
                            "rdm_evaluation_source": "explicitly_reloaded_best_validation_checkpoint",
                        },
                    )
                    target_flow[recording] = _load_cache(flow_path)[0]
                else:
                    flow_path = cache_dir / f"{role}_{recording}_{metric}_flow.npz"
                    target_flow[recording] = _load_cache(flow_path)[0]

        classical_mse = _mse_matrix(classical_query, classical_reference, times, recordings)
        flow_mse = _mse_matrix(flow_query, flow_reference, times, recordings)
        method_mse[0, metric_index] = classical_mse
        method_mse[1, metric_index] = flow_mse
        classical_metrics = _rank_metrics(classical_mse, recordings)
        flow_metrics = _rank_metrics(flow_mse, recordings)
        metrics_summary[metric] = {
            "classical": classical_metrics,
            "flow": flow_metrics,
        }
        print(
            f"[result] {metric} classical={classical_metrics['top1_accuracy']:.3f} "
            f"flow={flow_metrics['top1_accuracy']:.3f}",
            flush=True,
        )

    top1 = np.asarray(
        [
            [metrics_summary[metric][method]["top1_accuracy"] for metric in ALL_METRICS]
            for method in ("classical", "flow")
        ],
        dtype=np.float64,
    )
    visible_mask = (times >= VISIBLE_CUE_INTERVAL[0]) & (times <= VISIBLE_CUE_INTERVAL[1])
    _atomic_npz(
        args.output_dir / "multi_distance_full_query_9recordings_results.npz",
        mse=method_mse,
        top1_accuracy=top1,
        methods=np.asarray(["classical", "flow"]),
        metrics=np.asarray(ALL_METRICS),
        recordings=np.asarray(recordings),
        selected_time_seconds_cue_relative=times[visible_mask],
    )
    _atomic_json(
        args.output_dir / "multi_distance_full_query_9recordings_summary.json",
        {
            "experiment": "Nine-recording full-query multi-distance RDM session identification",
            "recordings": recordings,
            "chance_top1": 1.0 / len(recordings),
            "split_source": str((args.fid_dir / "splits").resolve()),
            "query_subsampling": "none; every saved full-query-half trial is used",
            "matching": "raw RDM mean squared error; smallest MSE ranks first",
            "requested_interval_seconds_cue_relative": list(VISIBLE_CUE_INTERVAL),
            "actual_interval_seconds_cue_relative": [
                float(times[visible_mask][0]),
                float(times[visible_mask][-1]),
            ],
            "n_selected_native_times": int(np.sum(visible_mask)),
            "flow_velocity_families": {
                **VELOCITY_FAMILIES,
                "fid": "condition_affine",
                "jeffreys": "condition_affine reused from FID with Gaussian Jeffreys readout",
            },
            "classical_jeffreys": "analytic Jeffreys sum between class-and-time Ledoit-Wolf Gaussians",
            "flow_jeffreys": "analytic Jeffreys sum between best-checkpoint condition-affine Gaussian endpoints",
            "epochs": int(args.epochs),
            "early_stopping_patience": int(args.patience),
            "metrics": metrics_summary,
        },
    )
    _plot_grouped_top1(args.output_dir, top1)
    print(f"[experiment] output={args.output_dir.resolve()}", flush=True)


def main() -> None:
    args = parse_args()
    if len(args.recordings) != 9 or len(set(args.recordings)) != 9:
        raise ValueError("This experiment requires nine unique recordings.")
    if args.aggregate_only and args.fit_metric is not None:
        raise ValueError("--aggregate-only and --fit-metric are mutually exclusive.")
    if args.fit_recording is not None and args.fit_metric is None:
        raise ValueError("--fit-recording requires --fit-metric.")
    if args.fit_recording is not None and args.fit_recording not in args.recordings:
        raise ValueError("--fit-recording must also appear in --recordings.")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA; no CPU fallback is permitted.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device.index} is unavailable.")
    torch.cuda.set_device(0 if device.index is None else device.index)
    torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[experiment] device={device} GPU={torch.cuda.get_device_name(device)} "
        f"epochs={args.epochs} patience={args.patience}",
        flush=True,
    )
    if args.aggregate_only:
        _aggregate(args, device)
        return
    if args.fit_metric is not None:
        _fit_metric(args, args.fit_metric, device)
        return
    for metric in FIT_METRICS:
        _fit_metric(args, metric, device)
    _aggregate(args, device)


if __name__ == "__main__":
    main()
