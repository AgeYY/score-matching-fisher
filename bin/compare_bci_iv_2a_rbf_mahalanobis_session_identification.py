#!/usr/bin/env python3
"""Compare classical and RBF-time flow Mahalanobis session identification."""

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
    classical_mahalanobis_rdms,
    condition_design,
    per_class_counts,
    rdms_from_means_and_precisions,
)
from fisher.flow_matching_skl import build_flow_skl_model, train_flow_skl_model  # noqa: E402


RECORDINGS = tuple(f"A{index:02d}T" for index in range(1, 10))
ROLES = ("reference", "half_query")
VISIBLE_CUE_INTERVAL = (0.0, 1.25)
VELOCITY_FAMILY = "covariate_affine"
DEFAULT_TIME_RBF_NUM_CENTERS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/processed/native_voltage_all_timepoints_20uv",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=ROOT
        / "data/bci_iv_2a/fid_session_identification_9recordings_mixed_runs/splits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to an RBF-count and recording-count specific data directory.",
    )
    parser.add_argument("--recordings", nargs="+", default=list(RECORDINGS))
    parser.add_argument(
        "--fit-recording",
        choices=RECORDINGS,
        help="Fit one recording and exit; useful for resuming a long run.",
    )
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20_260_717)
    parser.add_argument("--query-fraction", type=float, default=0.5)
    parser.add_argument(
        "--time-rbf-num-centers",
        type=int,
        default=DEFAULT_TIME_RBF_NUM_CENTERS,
    )
    parser.add_argument(
        "--time-rbf-bandwidth",
        type=float,
        default=None,
        help="Bandwidth in scaled EEG-time units; default is center spacing.",
    )
    parser.add_argument("--epochs", type=int, default=TRAINING_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--covariance-ode-steps", type=int, default=48)
    return parser.parse_args()


def _atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_rdm_cache(path: Path, rdms: np.ndarray, metadata: dict[str, Any]) -> None:
    _atomic_npz(
        path,
        rdms=np.asarray(rdms, dtype=np.float64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _load_rdm_cache(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        rdms = np.asarray(archive["rdms"], dtype=np.float64)
        metadata = json.loads(str(archive["metadata_json"].item()))
    if rdms.ndim != 3 or rdms.shape[1:] != (4, 4):
        raise ValueError(f"Invalid RDM shape in {path}: {rdms.shape}.")
    if not np.isfinite(rdms).all():
        raise ValueError(f"Non-finite RDM values in {path}.")
    np.testing.assert_allclose(rdms, rdms.transpose(0, 2, 1), atol=1e-7, rtol=0.0)
    np.testing.assert_allclose(
        np.diagonal(rdms, axis1=1, axis2=2),
        0.0,
        atol=1e-7,
        rtol=0.0,
    )
    return rdms, metadata


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _stratified_fraction_subsample(
    indices: np.ndarray,
    labels: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("--query-fraction must lie in (0, 1].")
    rng = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    for class_index in range(4):
        candidates = indices[labels[indices] == class_index]
        count = int(np.floor(float(fraction) * candidates.size))
        if count < 2:
            raise ValueError(
                f"Class {class_index} retains only {count} trials at fraction={fraction}."
            )
        selected.append(rng.choice(candidates, size=count, replace=False))
    return np.sort(np.concatenate(selected))


def _load_role_indices(
    split_dir: Path,
    recording: str,
    labels: np.ndarray,
    *,
    query_fraction: float,
    query_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    split_path = split_dir / f"{recording}_mixed_half_split.npz"
    with np.load(split_path, allow_pickle=False) as archive:
        saved_labels = np.asarray(archive["labels"], dtype=np.int64)
        reference = np.asarray(archive["reference_indices"], dtype=np.int64)
        full_query = np.asarray(archive["all_trial_indices"], dtype=np.int64)
        split_seed = int(archive["split_seed"].item())
    if not np.array_equal(saved_labels, labels):
        raise RuntimeError(f"Feature and split labels differ for {recording}.")
    if np.intersect1d(reference, full_query).size:
        raise RuntimeError(f"Reference/query leakage for {recording}.")
    half_query = _stratified_fraction_subsample(
        full_query,
        labels,
        fraction=float(query_fraction),
        seed=int(query_seed),
    )
    if not np.all(np.isin(half_query, full_query)):
        raise RuntimeError(f"Reduced query escaped the saved query half for {recording}.")
    return (
        {"reference": reference, "half_query": half_query},
        {
            "split_path": str(split_path.resolve()),
            "split_seed": split_seed,
            "query_subsample_seed": int(query_seed),
            "query_fraction": float(query_fraction),
            "sampling": "class-stratified without replacement",
            "full_query_trial_indices": full_query.astype(int).tolist(),
            "full_query_per_class_counts": per_class_counts(labels[full_query])
            .astype(int)
            .tolist(),
        },
    )


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


def _model_kwargs(
    config: FlowRDMConfig,
    x_dim: int,
    times: np.ndarray,
    *,
    rbf_num_centers: int,
    rbf_bandwidth: float | None,
) -> dict[str, Any]:
    if int(rbf_num_centers) < 2:
        raise ValueError("--time-rbf-num-centers must be at least 2.")
    time_scale = max(float(np.max(np.abs(times))), np.finfo(np.float64).eps)
    lower = float(np.min(times) / time_scale)
    upper = float(np.max(times) / time_scale)
    spacing = (upper - lower) / float(int(rbf_num_centers) - 1)
    bandwidth = spacing if rbf_bandwidth is None else float(rbf_bandwidth)
    return {
        "velocity_family": VELOCITY_FAMILY,
        "theta_dim": 5,
        "x_dim": int(x_dim),
        "radius": 1.0,
        "hidden_dim": int(config.hidden_dim),
        "depth": int(config.depth),
        "quadrature_steps": int(config.quadrature_steps),
        "path_schedule": "cosine",
        "divergence_estimator": "exact",
        "affine_condition_indices": (4,),
        "theta_embedding": "gaussian_rbf",
        "theta_rbf_indices": (4,),
        "theta_rbf_num_centers": int(rbf_num_centers),
        "theta_rbf_lower": lower,
        "theta_rbf_upper": upper,
        "theta_rbf_bandwidth": bandwidth,
    }


@torch.no_grad()
def _flow_rdms_from_model(
    model: torch.nn.Module,
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
    means = (
        model.endpoint_mean(condition_tensor)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
        .reshape(4, times.size, -1)
        .transpose(1, 0, 2)
    )
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
            "velocity_family": VELOCITY_FAMILY,
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
            "context": context,
        },
        temporary,
    )
    temporary.replace(path)


def _load_best_checkpoint_rdms(
    path: Path,
    *,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("velocity_family") != VELOCITY_FAMILY:
        raise ValueError(f"Velocity-family mismatch in {path}.")
    model = build_flow_skl_model(**payload["model_kwargs"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    config = FlowRDMConfig(**dict(payload["flow_rdm_config"]))
    times = payload["time_centers"].detach().cpu().numpy().astype(np.float64)
    return _flow_rdms_from_model(model, times, config, device), payload


def _fit_recording(
    args: argparse.Namespace,
    recording: str,
    recording_index: int,
    device: torch.device,
) -> None:
    rbf_tag = f"rbf{int(args.time_rbf_num_centers)}"
    dataset = load_features_npz(args.feature_dir / f"{recording}.npz")
    times = np.asarray(dataset.time_centers, dtype=np.float64)
    query_seed = int(args.seed + 50_000_000 + recording_index * 100_000)
    role_indices, split_metadata = _load_role_indices(
        args.split_dir,
        recording,
        dataset.labels,
        query_fraction=float(args.query_fraction),
        query_seed=query_seed,
    )
    config = _flow_config(args)
    cache_dir = args.output_dir / "rdm_cache"
    checkpoint_dir = args.output_dir / "checkpoints"
    split_output_dir = args.output_dir / "query_splits"
    _atomic_npz(
        split_output_dir / f"{recording}_half_query.npz",
        selected_indices=role_indices["half_query"],
        parent_query_indices=np.asarray(
            split_metadata["full_query_trial_indices"], dtype=np.int64
        ),
        fraction=np.asarray(float(args.query_fraction), dtype=np.float64),
        subsample_seed=np.asarray(query_seed, dtype=np.int64),
    )

    for role_index, role in enumerate(ROLES):
        indices = role_indices[role]
        values = np.asarray(dataset.features[indices], dtype=np.float64)
        labels = np.asarray(dataset.labels[indices], dtype=np.int64)
        common_context = {
            "recording": recording,
            "role": role,
            "metric": "mahalanobis",
            "trial_indices": indices.astype(int).tolist(),
            "n_trials": int(indices.size),
            "per_class_counts": per_class_counts(labels).astype(int).tolist(),
            **split_metadata,
        }

        classical_path = cache_dir / f"{role}_{recording}_mahalanobis_classical.npz"
        if not classical_path.is_file():
            start = time.perf_counter()
            classical_rdms = classical_mahalanobis_rdms(
                values,
                labels,
                standardize_features=False,
            )
            _save_rdm_cache(
                classical_path,
                classical_rdms,
                {
                    **common_context,
                    "method": "classical",
                    "estimator": "native-time pooled Ledoit-Wolf Mahalanobis",
                    "elapsed_seconds": float(time.perf_counter() - start),
                },
            )

        flow_path = cache_dir / f"{role}_{recording}_mahalanobis_{rbf_tag}_flow.npz"
        checkpoint_path = (
            checkpoint_dir / f"{role}_{recording}_mahalanobis_{rbf_tag}_flow_best.pt"
        )
        if flow_path.is_file() and checkpoint_path.is_file():
            print(f"[cache] loaded {flow_path.name}", flush=True)
            continue

        fit_seed = int(args.seed + recording_index * 100_000 + role_index * 10_000)
        _seed_all(fit_seed)
        train_trials, validation_trials = _stratified_validation_trials(
            labels,
            config.validation_fraction,
            fit_seed,
        )

        def flatten(trials: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            x = values[trials].reshape(-1, values.shape[-1])
            y = np.repeat(labels[trials], times.size)
            t = np.tile(times, trials.size)
            return condition_design(y, t), x

        theta_train, x_train = flatten(train_trials)
        theta_validation, x_validation = flatten(validation_trials)
        model_kwargs = _model_kwargs(
            config,
            values.shape[-1],
            times,
            rbf_num_centers=int(args.time_rbf_num_centers),
            rbf_bandwidth=args.time_rbf_bandwidth,
        )
        model = build_flow_skl_model(**model_kwargs).to(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        train_meta = train_flow_skl_model(
            model=model,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_validation,
            x_val=x_validation,
            device=device,
            velocity_family=VELOCITY_FAMILY,
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
            validation_seed=fit_seed + 10_000,
            retain_best_state=True,
            device_resident_data=True,
        )
        best_state = train_meta.get("best_state_dict")
        if best_state is None:
            raise RuntimeError("Training did not retain its best validation state.")
        checkpoint_context = {
            **common_context,
            "fit_seed": fit_seed,
            "time_rbf_num_centers": int(args.time_rbf_num_centers),
            "time_rbf_bandwidth": float(model_kwargs["theta_rbf_bandwidth"]),
        }
        _save_best_checkpoint(
            checkpoint_path,
            model_kwargs=model_kwargs,
            best_state=best_state,
            train_meta=train_meta,
            config=config,
            times=times,
            context=checkpoint_context,
        )
        flow_rdms, payload = _load_best_checkpoint_rdms(
            checkpoint_path,
            device=device,
        )
        if payload["context"] != checkpoint_context:
            raise RuntimeError(f"Checkpoint context mismatch: {checkpoint_path}.")
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        _save_rdm_cache(
            flow_path,
            flow_rdms,
            {
                **checkpoint_context,
                "method": "flow",
                "velocity_family": VELOCITY_FAMILY,
                "theta_embedding": "gaussian_rbf",
                "rdm_evaluation_source": "reloaded best-validation checkpoint",
                "checkpoint_path": str(checkpoint_path.resolve()),
                "best_epoch": int(train_meta["best_epoch"]),
                "stopped_epoch": int(train_meta["stopped_epoch"]),
                "best_val_loss": float(train_meta["best_val_loss"]),
                "train_losses": np.asarray(train_meta["train_losses"], dtype=float).tolist(),
                "validation_losses": np.asarray(
                    train_meta["val_losses"], dtype=float
                ).tolist(),
                "monitored_validation_losses": np.asarray(
                    train_meta["val_monitor_losses"], dtype=float
                ).tolist(),
                "elapsed_seconds": float(elapsed),
                "config": asdict(config),
            },
        )
        print(
            f"[fit] {flow_path.name} elapsed={elapsed / 60.0:.2f}min "
            f"best={train_meta['best_epoch']} stopped={train_meta['stopped_epoch']}",
            flush=True,
        )
    print(f"[recording] {recording} complete", flush=True)


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
            reference_vector = reference[reference_recording][mask][
                :, upper[0], upper[1]
            ]
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
        "correct_recordings": [
            recording
            for recording, rank in zip(recordings, ranks, strict=True)
            if rank == 1
        ],
        "mean_reciprocal_rank": float(np.mean(1.0 / rank_array)),
    }


def _plot_top1(
    output_dir: Path,
    top1: np.ndarray,
    chance: float,
    *,
    rbf_num_centers: int,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    figure, axis = plt.subplots(figsize=(4.0, 3.5))
    axis.bar(
        [0, 1],
        top1,
        width=1.0,
        color=["#4477AA", "#CC6677"],
        linewidth=0.0,
    )
    axis.axhline(chance, color="0.35", linestyle="--", linewidth=1.4)
    axis.set_xlim(-0.5, 1.5)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Top-1 accuracy")
    axis.set_xticks([0, 1], ["Classical", f"Flow: RBF{int(rbf_num_centers)}"])
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.8)
    axis.spines["bottom"].set_linewidth(1.8)
    axis.tick_params(width=1.8)
    stem = f"mahalanobis_rbf{int(rbf_num_centers)}_half_query_top1_bar"
    figure.savefig(output_dir / f"{stem}.png", dpi=300)
    figure.savefig(output_dir / f"{stem}.svg")
    plt.close(figure)


def _aggregate(args: argparse.Namespace) -> dict[str, Any]:
    recordings = list(args.recordings)
    rbf_tag = f"rbf{int(args.time_rbf_num_centers)}"
    datasets = [
        load_features_npz(args.feature_dir / f"{recording}.npz")
        for recording in recordings
    ]
    times = np.asarray(datasets[0].time_centers, dtype=np.float64)
    for dataset in datasets[1:]:
        np.testing.assert_array_equal(dataset.time_centers, times)
    cache_dir = args.output_dir / "rdm_cache"
    classical_reference: dict[str, np.ndarray] = {}
    classical_query: dict[str, np.ndarray] = {}
    flow_reference: dict[str, np.ndarray] = {}
    flow_query: dict[str, np.ndarray] = {}
    fit_metadata: dict[str, Any] = {}
    for recording in recordings:
        paths = {
            "classical_reference": cache_dir
            / f"reference_{recording}_mahalanobis_classical.npz",
            "classical_query": cache_dir
            / f"half_query_{recording}_mahalanobis_classical.npz",
            "flow_reference": cache_dir
            / f"reference_{recording}_mahalanobis_{rbf_tag}_flow.npz",
            "flow_query": cache_dir
            / f"half_query_{recording}_mahalanobis_{rbf_tag}_flow.npz",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing caches for {recording}: {missing}")
        classical_reference[recording] = _load_rdm_cache(paths["classical_reference"])[0]
        classical_query[recording] = _load_rdm_cache(paths["classical_query"])[0]
        flow_reference[recording], reference_metadata = _load_rdm_cache(
            paths["flow_reference"]
        )
        flow_query[recording], query_metadata = _load_rdm_cache(paths["flow_query"])
        fit_metadata[recording] = {
            "reference": {
                key: reference_metadata[key]
                for key in (
                    "n_trials",
                    "per_class_counts",
                    "best_epoch",
                    "stopped_epoch",
                    "best_val_loss",
                    "elapsed_seconds",
                )
            },
            "half_query": {
                key: query_metadata[key]
                for key in (
                    "n_trials",
                    "per_class_counts",
                    "best_epoch",
                    "stopped_epoch",
                    "best_val_loss",
                    "elapsed_seconds",
                )
            },
        }

    classical_mse = _mse_matrix(classical_query, classical_reference, times, recordings)
    flow_mse = _mse_matrix(flow_query, flow_reference, times, recordings)
    classical_metrics = _rank_metrics(classical_mse, recordings)
    flow_metrics = _rank_metrics(flow_mse, recordings)
    chance = 1.0 / len(recordings)
    top1 = np.asarray(
        [classical_metrics["top1_accuracy"], flow_metrics["top1_accuracy"]],
        dtype=np.float64,
    )
    visible_mask = (times >= VISIBLE_CUE_INTERVAL[0]) & (
        times <= VISIBLE_CUE_INTERVAL[1]
    )
    _atomic_npz(
        args.output_dir / f"mahalanobis_{rbf_tag}_half_query_results.npz",
        mse=np.stack([classical_mse, flow_mse]),
        top1_accuracy=top1,
        methods=np.asarray(["classical", f"flow_{rbf_tag}"]),
        recordings=np.asarray(recordings),
        selected_time_seconds_cue_relative=times[visible_mask],
    )
    summary = {
        "experiment": (
            f"{len(recordings)}-recording half-query Mahalanobis RDM session identification"
        ),
        "recordings": recordings,
        "chance_top1": chance,
        "reference_role": "complete saved reference half",
        "query_role": "one deterministic class-stratified 50 percent subsample of the saved query half",
        "sampling": "without replacement",
        "query_fraction": float(args.query_fraction),
        "matching": "raw RDM mean squared error; smallest MSE ranks first",
        "requested_interval_seconds_cue_relative": list(VISIBLE_CUE_INTERVAL),
        "actual_interval_seconds_cue_relative": [
            float(times[visible_mask][0]),
            float(times[visible_mask][-1]),
        ],
        "n_selected_native_times": int(np.sum(visible_mask)),
        "classical_estimator": "native-time pooled Ledoit-Wolf Mahalanobis",
        "flow_estimator": "condition-shared time-varying affine flow with Gaussian RBF EEG-time embedding",
        "time_rbf_num_centers": int(args.time_rbf_num_centers),
        "epochs": int(args.epochs),
        "early_stopping_patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "metrics": {
            "classical": classical_metrics,
            f"flow_{rbf_tag}": flow_metrics,
        },
        "flow_fits": fit_metadata,
    }
    _atomic_json(
        args.output_dir / f"mahalanobis_{rbf_tag}_half_query_summary.json",
        summary,
    )
    _plot_top1(
        args.output_dir,
        top1,
        chance,
        rbf_num_centers=int(args.time_rbf_num_centers),
    )
    print(
        f"[result] classical={top1[0]:.3f} flow_{rbf_tag}={top1[1]:.3f} "
        f"chance={chance:.3f}",
        flush=True,
    )
    print(f"[experiment] output={args.output_dir.resolve()}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    recordings = list(args.recordings)
    if len(recordings) < 2 or len(set(recordings)) != len(recordings):
        raise ValueError("This experiment requires at least two unique recordings.")
    if args.fit_recording is not None and args.fit_recording not in recordings:
        raise ValueError("--fit-recording must also appear in --recordings.")
    if args.aggregate_only and args.fit_recording is not None:
        raise ValueError("--aggregate-only and --fit-recording are mutually exclusive.")
    if args.device != "cuda:0":
        raise ValueError("This project requires --device cuda:0.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing CPU fallback.")
    torch.cuda.set_device(0)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if args.output_dir is None:
        args.output_dir = (
            ROOT
            / "data/bci_iv_2a"
            / (
                f"rbf{int(args.time_rbf_num_centers)}_mahalanobis_"
                f"session_identification_{len(recordings)}recordings_half_query"
            )
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[experiment] device={device} GPU={torch.cuda.get_device_name(device)} "
        f"RBF={args.time_rbf_num_centers} query_fraction={args.query_fraction}",
        flush=True,
    )
    if args.aggregate_only:
        _aggregate(args)
        return
    selected = recordings if args.fit_recording is None else [args.fit_recording]
    for recording in selected:
        _fit_recording(args, recording, recordings.index(recording), device)
    if args.fit_recording is None:
        _aggregate(args)


if __name__ == "__main__":
    main()
