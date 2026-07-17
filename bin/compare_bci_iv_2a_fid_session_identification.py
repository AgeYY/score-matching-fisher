#!/usr/bin/env python3
"""Run five-recording mixed-run BCI IV-2a identification with FID RDMs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

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
    RDM_MATCHING_INTERVAL,
    classical_fid_rdms,
    condition_affine_flow_fid_rdms,
    condition_affine_flow_fid_rdms_from_checkpoint,
    pearson_similarity,
    per_class_counts,
    stratified_mixed_half_split,
)
from fisher.rdm_lagged_matching import rdm_upper_triangle_sequence  # noqa: E402


METHODS = ("classical_fid", "condition_affine_flow_fid")
METHOD_LABELS = {
    "classical_fid": "Classical FID",
    "condition_affine_flow_fid": "Flow FID",
}
METHOD_COLORS = {
    "classical_fid": "#4477AA",
    "condition_affine_flow_fid": "#CC6677",
}
ROLES = ("reference", "all_trial")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/processed/native_voltage_all_timepoints_20uv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/fid_session_identification_5recordings_mixed_runs",
    )
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=["A01T", "A02T", "A03T", "A04T", "A05T"],
    )
    parser.add_argument(
        "--fit-recording",
        choices=["A01T", "A02T", "A03T", "A04T", "A05T", "A06T", "A07T", "A08T", "A09T"],
    )
    parser.add_argument(
        "--fit-quarter-query-flow",
        action="store_true",
        help=(
            "Fit native-time condition-affine flow FID models on the saved quarter-query "
            "subsamples, then run visible-cue MSE identification."
        ),
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


def _atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_cache(path: Path, rdms: np.ndarray, metadata: dict) -> None:
    _atomic_npz(
        path,
        rdms=np.asarray(rdms, dtype=np.float64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _load_cache(path: Path) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as archive:
        rdms = np.asarray(archive["rdms"], dtype=np.float64)
        metadata = json.loads(str(archive["metadata_json"].item()))
    if rdms.ndim != 3 or rdms.shape[1:] != (4, 4) or not np.all(np.isfinite(rdms)):
        raise ValueError(f"Invalid RDM cache {path}: shape={rdms.shape}.")
    if np.min(rdms) < -1e-9:
        raise ValueError(f"FID cache {path} contains negative values.")
    np.testing.assert_allclose(rdms, rdms.transpose(0, 2, 1), atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(np.diagonal(rdms, axis1=1, axis2=2), 0.0, atol=1e-8, rtol=0.0)
    return rdms, metadata


def _run_class_counts(labels: np.ndarray, run_ids: np.ndarray) -> list[list[int]]:
    return [
        [int(np.sum((run_ids == run) & (labels == label))) for label in range(4)]
        for run in range(6)
    ]


def _flow_config(args: argparse.Namespace) -> FlowRDMConfig:
    """Return the canonical dense-time condition-affine flow configuration."""

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


def _fit_one_recording(args: argparse.Namespace, recording: str, device: torch.device) -> None:
    dataset = load_features_npz(args.feature_dir / f"{recording}.npz")
    recording_index = args.recordings.index(recording)
    split_seed = int(args.seed + recording_index * 100_000)
    reference_indices, query_indices = stratified_mixed_half_split(dataset.labels, split_seed)
    if np.intersect1d(reference_indices, query_indices).size:
        raise RuntimeError("Reference/query leakage detected.")
    if np.union1d(reference_indices, query_indices).size != dataset.labels.size:
        raise RuntimeError("Mixed half split did not use every clean trial exactly once.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "rdm_cache"
    checkpoint_dir = args.output_dir / "checkpoints"
    split_dir = args.output_dir / "splits"
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        split_dir / f"{recording}_mixed_half_split.npz",
        reference_indices=reference_indices,
        all_trial_indices=query_indices,
        labels=np.asarray(dataset.labels, dtype=np.int64),
        run_ids=np.asarray(dataset.run_ids, dtype=np.int64),
        split_seed=np.asarray(split_seed, dtype=np.int64),
    )

    config = _flow_config(args)
    role_indices = {
        "reference": reference_indices,
        "all_trial": query_indices,
    }
    fit_rows: list[dict] = []
    for role_index, role in enumerate(ROLES):
        indices = role_indices[role]
        x_role = dataset.features[indices]
        labels_role = dataset.labels[indices]
        runs_role = dataset.run_ids[indices]
        context = {
            "recording": recording,
            "role": role,
            "split_design": "class_stratified_half_split_after_pooling_all_six_runs",
            "split_seed": split_seed,
            "n_trials": int(indices.size),
            "per_class_counts": per_class_counts(labels_role).astype(int).tolist(),
            "run_by_class_counts": _run_class_counts(labels_role, runs_role),
        }
        classical_path = cache_dir / f"{role}_{recording}_classical_fid.npz"
        if classical_path.is_file():
            _, classical_meta = _load_cache(classical_path)
            print(f"[cache] loaded {classical_path.name}", flush=True)
        else:
            start = time.perf_counter()
            classical = classical_fid_rdms(
                x_role,
                labels_role,
                standardize_features=False,
            )
            classical_elapsed = time.perf_counter() - start
            classical_meta = {
                **context,
                "method": "classical_fid",
                "estimator": "class_and_time_specific_ledoit_wolf_gaussian",
                "distance": "squared_gaussian_2_wasserstein_fid",
                "elapsed_seconds": float(classical_elapsed),
            }
            _save_cache(classical_path, classical, classical_meta)
            print(f"[fit] {classical_path.name} elapsed={classical_elapsed:.1f}s", flush=True)
        fit_rows.append(
            {
                "recording": recording,
                "role": role,
                "method": "classical_fid",
                "n_trials": int(indices.size),
                "per_class_counts": per_class_counts(labels_role).astype(int).tolist(),
                "run_by_class_counts": _run_class_counts(labels_role, runs_role),
                "elapsed_seconds": float(classical_meta.get("elapsed_seconds", 0.0)),
                "best_epoch": None,
                "stopped_epoch": None,
                "best_val_loss": None,
                "checkpoint_path": None,
            }
        )

        flow_path = cache_dir / f"{role}_{recording}_condition_affine_flow_fid.npz"
        checkpoint_path = checkpoint_dir / f"{role}_{recording}_condition_affine_flow_fid_best.pt"
        if flow_path.is_file() and checkpoint_path.is_file():
            _, flow_meta = _load_cache(flow_path)
            print(f"[cache] loaded {flow_path.name}", flush=True)
        else:
            flow_seed = int(
                args.seed
                + 10_000_000
                + recording_index * 100_000
                + role_index * 10_000
            )
            start = time.perf_counter()
            flow, flow_meta = condition_affine_flow_fid_rdms(
                x_role,
                labels_role,
                dataset.time_centers,
                device=device,
                seed=flow_seed,
                config=config,
                checkpoint_path=checkpoint_path,
                checkpoint_context=context,
            )
            torch.cuda.synchronize(device)
            flow_elapsed = time.perf_counter() - start
            flow_meta = {
                **flow_meta,
                **context,
                "method": "condition_affine_flow_fid",
                "elapsed_seconds": float(flow_elapsed),
            }
            _save_cache(flow_path, flow, flow_meta)
            print(
                f"[fit] {flow_path.name} elapsed={flow_elapsed / 60.0:.2f}min "
                f"best={flow_meta['best_epoch']} stopped={flow_meta['stopped_epoch']}",
                flush=True,
            )
        if str(Path(flow_meta["checkpoint_path"]).resolve()) != str(checkpoint_path.resolve()):
            raise RuntimeError(f"Cache/checkpoint mismatch for {recording}/{role}.")
        fit_rows.append(
            {
                "recording": recording,
                "role": role,
                "method": "condition_affine_flow_fid",
                "n_trials": int(indices.size),
                "per_class_counts": per_class_counts(labels_role).astype(int).tolist(),
                "run_by_class_counts": _run_class_counts(labels_role, runs_role),
                "elapsed_seconds": float(flow_meta["elapsed_seconds"]),
                "best_epoch": int(flow_meta["best_epoch"]),
                "stopped_epoch": int(flow_meta["stopped_epoch"]),
                "best_val_loss": float(flow_meta["best_val_loss"]),
                "checkpoint_path": str(checkpoint_path.resolve()),
            }
        )

    _atomic_json(
        args.output_dir / f"fit_{recording}.json",
        {
            "recording": recording,
            "split_seed": split_seed,
            "total_clean_trials": int(dataset.labels.size),
            "reference_indices_count": int(reference_indices.size),
            "all_trial_indices_count": int(query_indices.size),
            "flow_config": asdict(config),
            "fits": fit_rows,
        },
    )
    print(f"[recording] {recording} complete", flush=True)


def _fit_quarter_query_flow(args: argparse.Namespace, device: torch.device) -> None:
    """Fit and identify native-time flow FID from saved quarter-query trials."""

    recordings = list(args.recordings)
    config = _flow_config(args)
    cache_dir = args.output_dir / "reduced_query_rdm_cache"
    checkpoint_dir = args.output_dir / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    query_rdms: dict[str, np.ndarray] = {}
    query_metadata: dict[str, dict] = {}
    datasets = [load_features_npz(args.feature_dir / f"{recording}.npz") for recording in recordings]
    times = np.asarray(datasets[0].time_centers, dtype=np.float64)
    if any(not np.array_equal(dataset.time_centers, times) for dataset in datasets[1:]):
        raise ValueError("Feature files do not share the same native-time grid.")

    for recording_index, (recording, dataset) in enumerate(
        zip(recordings, datasets, strict=True)
    ):
        split_path = args.output_dir / "reduced_query_splits" / f"{recording}_quarter_of_query.npz"
        if not split_path.is_file():
            raise FileNotFoundError(
                f"Missing quarter-query split {split_path}; run --aggregate-only first."
            )
        with np.load(split_path, allow_pickle=False) as split_archive:
            selected_indices = np.asarray(split_archive["selected_indices"], dtype=np.int64)
            parent_query_indices = np.asarray(
                split_archive["parent_query_indices"],
                dtype=np.int64,
            )
            split_fraction = float(split_archive["fraction"].item())
            subsample_seed = int(split_archive["subsample_seed"].item())
        if split_fraction != 0.25:
            raise RuntimeError(f"Unexpected query fraction {split_fraction} for {recording}.")
        if selected_indices.size != np.unique(selected_indices).size:
            raise RuntimeError(f"Quarter-query split contains duplicate trials for {recording}.")
        if not np.all(np.isin(selected_indices, parent_query_indices)):
            raise RuntimeError(f"Quarter-query split escaped its saved query half for {recording}.")
        selected_labels = dataset.labels[selected_indices]
        context = {
            "recording": recording,
            "role": "quarter_of_saved_query",
            "split_path": str(split_path.resolve()),
            "fraction_of_saved_query": split_fraction,
            "subsample_seed": subsample_seed,
            "selected_trial_indices": selected_indices.astype(int).tolist(),
            "n_trials": int(selected_indices.size),
            "per_class_counts": per_class_counts(selected_labels).astype(int).tolist(),
            "run_by_class_counts": _run_class_counts(
                selected_labels,
                dataset.run_ids[selected_indices],
            ),
        }
        flow_seed = int(args.seed + 80_000_000 + recording_index * 100_000)
        cache_path = (
            cache_dir / f"quarter_query_{recording}_condition_affine_flow_fid.npz"
        )
        checkpoint_path = (
            checkpoint_dir
            / f"quarter_query_{recording}_condition_affine_flow_fid_best.pt"
        )
        if cache_path.is_file() and checkpoint_path.is_file():
            flow_rdm, flow_meta = _load_cache(cache_path)
            if flow_meta.get("config") != asdict(config):
                raise RuntimeError(
                    f"Cached quarter-query flow config differs from the requested config: "
                    f"{cache_path}"
                )
            if flow_meta.get("selected_trial_indices") != context["selected_trial_indices"]:
                raise RuntimeError(f"Cached quarter-query trials differ for {recording}.")
            print(f"[cache] loaded {cache_path.name}", flush=True)
        else:
            start = time.perf_counter()
            flow_rdm, flow_meta = condition_affine_flow_fid_rdms(
                dataset.features[selected_indices],
                selected_labels,
                times,
                device=device,
                seed=flow_seed,
                config=config,
                checkpoint_path=checkpoint_path,
                checkpoint_context=context,
            )
            torch.cuda.synchronize(device)
            elapsed_seconds = time.perf_counter() - start
            flow_meta = {
                **flow_meta,
                **context,
                "method": "condition_affine_flow_fid",
                "elapsed_seconds": float(elapsed_seconds),
            }
            print(
                f"[fit] {cache_path.name} elapsed={elapsed_seconds / 60.0:.2f}min "
                f"best={flow_meta['best_epoch']} stopped={flow_meta['stopped_epoch']}",
                flush=True,
            )
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing best checkpoint: {checkpoint_path}")
        if str(Path(flow_meta["checkpoint_path"]).resolve()) != str(checkpoint_path.resolve()):
            raise RuntimeError(f"Cache/checkpoint mismatch for quarter-query {recording}.")
        checkpoint_rdm, checkpoint_payload = (
            condition_affine_flow_fid_rdms_from_checkpoint(
                checkpoint_path,
                device=device,
            )
        )
        if checkpoint_payload["context"]["selected_trial_indices"] != context[
            "selected_trial_indices"
        ]:
            raise RuntimeError(f"Checkpoint quarter-query trials differ for {recording}.")
        max_cache_checkpoint_difference = float(np.max(np.abs(flow_rdm - checkpoint_rdm)))
        flow_rdm = checkpoint_rdm
        flow_meta = {
            **flow_meta,
            "rdm_evaluation_source": "explicitly_reloaded_best_validation_checkpoint",
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "pre_reload_cache_max_absolute_difference": max_cache_checkpoint_difference,
        }
        _save_cache(cache_path, flow_rdm, flow_meta)
        print(
            f"[checkpoint] {recording} RDM reloaded from best epoch "
            f"{flow_meta['best_epoch']} max_pre_reload_diff="
            f"{max_cache_checkpoint_difference:.6g}",
            flush=True,
        )
        query_rdms[recording] = flow_rdm
        query_metadata[recording] = flow_meta

    reference_rdms: dict[str, np.ndarray] = {}
    reference_checkpoint_cache_dir = args.output_dir / "best_checkpoint_reference_rdm_cache"
    reference_checkpoint_cache_dir.mkdir(parents=True, exist_ok=True)
    for recording in recordings:
        reference_checkpoint = (
            checkpoint_dir / f"reference_{recording}_condition_affine_flow_fid_best.pt"
        )
        reference_rdm, reference_payload = (
            condition_affine_flow_fid_rdms_from_checkpoint(
                reference_checkpoint,
                device=device,
            )
        )
        reference_cache_path = (
            reference_checkpoint_cache_dir
            / f"reference_{recording}_condition_affine_flow_fid_best_reloaded.npz"
        )
        _save_cache(
            reference_cache_path,
            reference_rdm,
            {
                "recording": recording,
                "role": "reference",
                "method": "condition_affine_flow_fid",
                "rdm_evaluation_source": (
                    "explicitly_reloaded_existing_best_validation_checkpoint"
                ),
                "checkpoint_path": str(reference_checkpoint.resolve()),
                "best_epoch": int(reference_payload["training"]["best_epoch"]),
                "stopped_epoch": int(reference_payload["training"]["stopped_epoch"]),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
            },
        )
        reference_rdms[recording] = _load_cache(reference_cache_path)[0]
    visible_cue_interval = (0.0, 1.25)
    flow_mse = _raw_fid_mse_from_query_reference_rdms(
        query_rdms,
        reference_rdms,
        times,
        recordings,
        interval=visible_cue_interval,
    )
    flow_metrics = _rank_metrics(-flow_mse, recordings)

    classical_summary_path = (
        args.output_dir / "raw_fid_visible_cue_mse_quarter_query_classical_summary.json"
    )
    if not classical_summary_path.is_file():
        raise FileNotFoundError(f"Missing classical quarter-query summary: {classical_summary_path}")
    classical_summary = json.loads(classical_summary_path.read_text(encoding="utf-8"))
    classical_metrics = classical_summary["metrics"]
    comparison_top1 = np.asarray(
        [
            classical_metrics["native_time_quarter_query"]["top1_accuracy"],
            classical_metrics["binned_250ms_quarter_query"]["top1_accuracy"],
            flow_metrics["top1_accuracy"],
        ],
        dtype=np.float64,
    )
    visible_cue_mask = (times >= visible_cue_interval[0]) & (
        times <= visible_cue_interval[1]
    )
    _atomic_npz(
        args.output_dir / "raw_fid_visible_cue_mse_quarter_query_flow_results.npz",
        flow_raw_fid_mse=flow_mse,
        comparison_top1_accuracy=comparison_top1,
        comparison_methods=np.asarray(
            ["classical_native_time", "classical_250ms_bins", "flow_native_time"]
        ),
        recordings=np.asarray(recordings),
        requested_interval=np.asarray(visible_cue_interval),
        selected_time_seconds_cue_relative=times[visible_cue_mask],
    )
    fit_rows = [
        {
            "recording": recording,
            "n_trials": int(query_metadata[recording]["n_trials"]),
            "per_class_counts": query_metadata[recording]["per_class_counts"],
            "seed": int(query_metadata[recording]["seed"]),
            "best_epoch": int(query_metadata[recording]["best_epoch"]),
            "stopped_epoch": int(query_metadata[recording]["stopped_epoch"]),
            "best_val_loss": float(query_metadata[recording]["best_val_loss"]),
            "elapsed_seconds": float(query_metadata[recording]["elapsed_seconds"]),
            "checkpoint_path": str(Path(query_metadata[recording]["checkpoint_path"]).resolve()),
            "rdm_evaluation_source": query_metadata[recording]["rdm_evaluation_source"],
            "pre_reload_cache_max_absolute_difference": float(
                query_metadata[recording]["pre_reload_cache_max_absolute_difference"]
            ),
        }
        for recording in recordings
    ]
    _atomic_json(
        args.output_dir / "raw_fid_visible_cue_mse_quarter_query_flow_summary.json",
        {
            "experiment": "Quarter-query flow FID session identification",
            "distance_representation": "raw squared Gaussian FID",
            "flow_estimator": (
                "joint class-and-native-time condition-affine full-covariance Gaussian flow"
            ),
            "query_reference_distance": (
                "mean squared error over visible-cue native times and six unique class pairs"
            ),
            "ranking_rule": "smallest MSE is the top-1 reference",
            "requested_interval_seconds_cue_relative": list(visible_cue_interval),
            "query_role": (
                "same deterministic class-stratified quarter-query trials used by the "
                "classical estimators"
            ),
            "reference_role": (
                "unchanged full reference-half flow models, with RDMs explicitly recomputed "
                "from their saved best-validation checkpoints"
            ),
            "recordings": recordings,
            "chance_top1": 1.0 / len(recordings),
            "epochs": int(args.epochs),
            "early_stopping_patience": int(args.patience),
            "checkpoint_rule": (
                "exact best-validation model restored for RDM evaluation and saved"
            ),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "flow_metrics": flow_metrics,
            "comparison_top1_accuracy": {
                "classical_native_time": float(comparison_top1[0]),
                "classical_250ms_bins": float(comparison_top1[1]),
                "flow_native_time": float(comparison_top1[2]),
            },
            "fits": fit_rows,
        },
    )
    _plot_top1_bar_values(
        args.output_dir,
        comparison_top1,
        output_stem="raw_fid_visible_cue_mse_quarter_query_three_methods_top1_bar",
        title="Quarter query: raw-FID MSE",
        labels=["Classical\nnative", "Classical\n250 ms", "Flow-based\nnative"],
        colors=[
            METHOD_COLORS["classical_fid"],
            "#EE7733",
            METHOD_COLORS["condition_affine_flow_fid"],
        ],
    )
    _plot_quarter_query_flow_losses(args.output_dir, query_metadata, recordings)
    print(
        f"[result] Quarter-query Flow FID visible-cue raw-FID MSE "
        f"top1={flow_metrics['top1_accuracy']:.3f} ranks={flow_metrics['ranks']}",
        flush=True,
    )
    print(f"[experiment] output={args.output_dir.resolve()}", flush=True)


def _rank_metrics(scores: np.ndarray, recordings: list[str]) -> dict:
    ranks: list[int] = []
    predictions: list[str] = []
    margins: list[float] = []
    for query_index in range(len(recordings)):
        order = np.argsort(-scores[query_index], kind="mergesort")
        rank = int(np.flatnonzero(order == query_index)[0]) + 1
        ranks.append(rank)
        predictions.append(recordings[int(order[0])])
        margins.append(
            float(
                scores[query_index, query_index]
                - np.max(np.delete(scores[query_index], query_index))
            )
        )
    return {
        "top1_accuracy": float(np.mean(np.asarray(ranks) == 1)),
        "mean_reciprocal_rank": float(np.mean(1.0 / np.asarray(ranks))),
        "mean_true_minus_best_competitor_margin": float(np.mean(margins)),
        "ranks": ranks,
        "predicted_recordings": predictions,
        "margins": margins,
    }


def _raw_fid_scores_for_interval(
    cache: dict[tuple[str, str, str], tuple[np.ndarray, dict]],
    times: np.ndarray,
    recordings: list[str],
    *,
    interval: tuple[float, float],
) -> np.ndarray:
    """Return zero-lag raw-FID query-reference correlations in one interval."""

    scores = np.empty(
        (len(METHODS), len(recordings), len(recordings)),
        dtype=np.float64,
    )
    for method_index, method in enumerate(METHODS):
        for query_index, query_recording in enumerate(recordings):
            query_rdm = cache[(method, "all_trial", query_recording)][0]
            query_vector = rdm_upper_triangle_sequence(
                query_rdm,
                times,
                interval=interval,
            )[0].reshape(-1)
            for reference_index, reference_recording in enumerate(recordings):
                reference_rdm = cache[(method, "reference", reference_recording)][0]
                reference_vector = rdm_upper_triangle_sequence(
                    reference_rdm,
                    times,
                    interval=interval,
                )[0].reshape(-1)
                scores[method_index, query_index, reference_index] = pearson_similarity(
                    query_vector,
                    reference_vector,
                )
    return scores


def _raw_fid_mse_for_interval(
    cache: dict[tuple[str, str, str], tuple[np.ndarray, dict]],
    times: np.ndarray,
    recordings: list[str],
    *,
    interval: tuple[float, float],
) -> np.ndarray:
    """Return raw-FID query-reference MSE in one cue-relative interval."""

    distances = np.empty(
        (len(METHODS), len(recordings), len(recordings)),
        dtype=np.float64,
    )
    for method_index, method in enumerate(METHODS):
        for query_index, query_recording in enumerate(recordings):
            query_rdm = cache[(method, "all_trial", query_recording)][0]
            query_vector = rdm_upper_triangle_sequence(
                query_rdm,
                times,
                interval=interval,
            )[0].reshape(-1)
            for reference_index, reference_recording in enumerate(recordings):
                reference_rdm = cache[(method, "reference", reference_recording)][0]
                reference_vector = rdm_upper_triangle_sequence(
                    reference_rdm,
                    times,
                    interval=interval,
                )[0].reshape(-1)
                distances[method_index, query_index, reference_index] = float(
                    np.mean((query_vector - reference_vector) ** 2)
                )
    return distances


def _mean_voltage_in_time_bins(
    features: np.ndarray,
    times: np.ndarray,
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average each trial's native voltage samples within half-open time bins."""

    features = np.asarray(features, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    bin_edges = np.asarray(bin_edges, dtype=np.float64).reshape(-1)
    if features.ndim != 3 or features.shape[1] != times.size:
        raise ValueError(
            "Expected features with shape [trials, native_times, channels] "
            f"matching {times.size} time points; got {features.shape}."
        )
    if bin_edges.size < 2 or np.any(np.diff(bin_edges) <= 0.0):
        raise ValueError("Time-bin edges must be strictly increasing.")

    binned: list[np.ndarray] = []
    counts: list[int] = []
    assignment_count = np.zeros(times.size, dtype=np.int64)
    for left, right in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        mask = (times >= left) & (times < right)
        count = int(np.sum(mask))
        if count == 0:
            raise ValueError(f"Empty time bin [{left}, {right}).")
        assignment_count += mask.astype(np.int64)
        counts.append(count)
        binned.append(np.mean(features[:, mask, :], axis=1))

    selected = (times >= bin_edges[0]) & (times < bin_edges[-1])
    if not np.array_equal(assignment_count[selected], np.ones(np.sum(selected), dtype=np.int64)):
        raise RuntimeError("Native samples were not assigned exactly once within the bin interval.")
    if np.any(assignment_count[~selected] != 0):
        raise RuntimeError("Samples outside the requested interval were assigned to a bin.")
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return np.stack(binned, axis=1), centers, np.asarray(counts, dtype=np.int64)


def _raw_fid_mse_from_binned_classical_rdms(
    cache: dict[tuple[str, str], np.ndarray],
    recordings: list[str],
) -> np.ndarray:
    """Return query-reference MSE over bins and six unique RDM entries."""

    upper = np.triu_indices(4, k=1)
    distances = np.empty((len(recordings), len(recordings)), dtype=np.float64)
    for query_index, query_recording in enumerate(recordings):
        query_vector = cache[("all_trial", query_recording)][:, upper[0], upper[1]].reshape(-1)
        for reference_index, reference_recording in enumerate(recordings):
            reference_vector = cache[("reference", reference_recording)][
                :, upper[0], upper[1]
            ].reshape(-1)
            distances[query_index, reference_index] = float(
                np.mean((query_vector - reference_vector) ** 2)
            )
    return distances


def _stratified_fraction_subsample(
    indices: np.ndarray,
    labels: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Select floor(fraction * n_c) trials without replacement in every class."""

    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}.")
    rng = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    for label in range(4):
        candidates = indices[labels[indices] == label]
        n_selected = int(np.floor(float(fraction) * candidates.size))
        if n_selected < 2:
            raise ValueError(
                f"Condition {label} retains fewer than two trials at fraction={fraction}; "
                f"parent count={candidates.size}."
            )
        selected.append(rng.choice(candidates, size=n_selected, replace=False))
    return np.sort(np.concatenate(selected))


def _raw_fid_mse_from_query_reference_rdms(
    query_rdms: dict[str, np.ndarray],
    reference_rdms: dict[str, np.ndarray],
    times: np.ndarray,
    recordings: list[str],
    *,
    interval: tuple[float, float],
) -> np.ndarray:
    """Return one method's query-reference raw-FID MSE matrix."""

    query_vectors = {
        recording: rdm_upper_triangle_sequence(
            query_rdms[recording],
            times,
            interval=interval,
        )[0].reshape(-1)
        for recording in recordings
    }
    reference_vectors = {
        recording: rdm_upper_triangle_sequence(
            reference_rdms[recording],
            times,
            interval=interval,
        )[0].reshape(-1)
        for recording in recordings
    }
    distances = np.empty((len(recordings), len(recordings)), dtype=np.float64)
    for query_index, query_recording in enumerate(recordings):
        for reference_index, reference_recording in enumerate(recordings):
            distances[query_index, reference_index] = float(
                np.mean(
                    (query_vectors[query_recording] - reference_vectors[reference_recording]) ** 2
                )
            )
    return distances


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(width=1.8)


def _plot_identification(
    output_dir: Path,
    scores: np.ndarray,
    metrics: dict,
    recordings: list[str],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), layout="constrained")
    images = []
    for method_index, method in enumerate(METHODS):
        axis = axes[method_index]
        matrix = scores[method_index]
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        images.append(image)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if abs(matrix[row, column]) > 0.45 else "black",
                )
        axis.set_xticks(np.arange(len(recordings)), recordings, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(recordings)), recordings)
        axis.set_xlabel("Reference")
        if method_index == 0:
            axis.set_ylabel("All-trial query")
        top1 = metrics[method]["sqrt_fid_primary"]["top1_accuracy"]
        axis.set_title(f"{METHOD_LABELS[method]}: top-1 {top1:.0%}")
        _style_axis(axis)
    colorbar = figure.colorbar(images[0], ax=axes[:2], location="bottom", shrink=0.82)
    colorbar.set_label("Zero-lag correlation")

    x = np.arange(len(recordings))
    for method in METHODS:
        margins = metrics[method]["sqrt_fid_primary"]["margins"]
        axes[2].plot(
            x,
            margins,
            marker="o",
            linewidth=2.0,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes[2].axhline(0.0, color="0.35", linestyle="--", linewidth=1.5)
    axes[2].set_xticks(x, recordings, rotation=45, ha="right")
    axes[2].set_xlabel("Query recording")
    axes[2].set_ylabel("True-match margin")
    axes[2].set_title("Identification margin")
    axes[2].legend(frameon=False, loc="best")
    _style_axis(axes[2])
    figure.savefig(output_dir / "fid_session_identification.png", dpi=300, facecolor="white")
    figure.savefig(output_dir / "fid_session_identification.svg", facecolor="white")
    plt.close(figure)


def _plot_top1_bar(
    output_dir: Path,
    metrics: dict,
    *,
    metric_key: str,
    output_stem: str,
) -> None:
    """Plot two top-1 accuracies with exactly touching bars."""

    values = np.asarray(
        [
            metrics["classical_fid"][metric_key]["top1_accuracy"],
            metrics["condition_affine_flow_fid"][metric_key]["top1_accuracy"],
        ],
        dtype=np.float64,
    )
    _plot_top1_bar_values(output_dir, values, output_stem=output_stem)


def _plot_top1_bar_values(
    output_dir: Path,
    values: np.ndarray,
    *,
    output_stem: str,
    title: str | None = None,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
) -> None:
    """Plot supplied top-1 values as exactly touching bars."""

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 16,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if labels is None:
        labels = ["Classical FID", "Flow-based FID"]
    if colors is None:
        colors = [METHOD_COLORS["classical_fid"], METHOD_COLORS["condition_affine_flow_fid"]]
    if values.size == 0 or len(labels) != values.size or len(colors) != values.size:
        raise ValueError("Top-1 values, labels, and colors must have the same nonzero length.")
    positions = np.arange(values.size, dtype=np.float64)
    figure_width = max(4.0, 1.5 * values.size)
    figure, axis = plt.subplots(figsize=(figure_width, 3.5), layout="constrained")
    axis.bar(
        positions,
        values,
        width=1.0,
        color=colors,
        linewidth=0.0,
    )
    axis.set_xlim(-0.5, values.size - 0.5)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Top-1 accuracy")
    if title is not None:
        axis.set_title(title)
    _style_axis(axis)
    figure.savefig(output_dir / f"{output_stem}.png", dpi=300, facecolor="white")
    figure.savefig(output_dir / f"{output_stem}.svg", facecolor="white")
    plt.close(figure)


def _plot_raw_fid_mean_trajectories(
    output_dir: Path,
    cache: dict[tuple[str, str, str], tuple[np.ndarray, dict]],
    times: np.ndarray,
    recordings: list[str],
) -> None:
    """Plot mean raw FID over the six unique class pairs at every EEG time."""

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    upper = np.triu_indices(4, k=1)
    trajectories = np.empty(
        (len(METHODS), len(ROLES), len(recordings), times.size),
        dtype=np.float64,
    )
    for method_index, method in enumerate(METHODS):
        for role_index, role in enumerate(ROLES):
            for recording_index, recording in enumerate(recordings):
                rdms = cache[(method, role, recording)][0]
                trajectories[method_index, role_index, recording_index] = np.mean(
                    rdms[:, upper[0], upper[1]],
                    axis=1,
                )

    figure, axes = plt.subplots(
        2,
        len(recordings),
        figsize=(4.0 * len(recordings), 7.0),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    role_labels = {"reference": "reference", "all_trial": "all-trial query"}
    for role_index, role in enumerate(ROLES):
        for recording_index, recording in enumerate(recordings):
            axis = axes[role_index, recording_index]
            for method_index, method in enumerate(METHODS):
                axis.plot(
                    times,
                    trajectories[method_index, role_index, recording_index],
                    color=METHOD_COLORS[method],
                    linewidth=1.8,
                    label=METHOD_LABELS[method],
                )
            axis.axvline(0.0, color="0.35", linestyle="--", linewidth=1.3)
            axis.set_title(f"{recording}: {role_labels[role]}")
            _style_axis(axis)
    axes[0, -1].legend(frameon=False, loc="upper left")
    figure.supxlabel("Time from cue (s)", fontsize=16)
    figure.supylabel("Mean raw FID across class pairs", fontsize=16)
    output_stem = "raw_fid_mean_distance_vs_time_all_recordings"
    figure.savefig(output_dir / f"{output_stem}.png", dpi=300, facecolor="white")
    figure.savefig(output_dir / f"{output_stem}.svg", facecolor="white")
    plt.close(figure)
    _atomic_npz(
        output_dir / f"{output_stem}.npz",
        mean_raw_fid=trajectories,
        methods=np.asarray(METHODS),
        roles=np.asarray(ROLES),
        recordings=np.asarray(recordings),
        time_seconds_cue_relative=np.asarray(times, dtype=np.float64),
        class_pair_indices=np.stack(upper, axis=1).astype(np.int64),
    )


def _plot_loss_curves(output_dir: Path, cache: dict[tuple[str, str, str], tuple[np.ndarray, dict]]) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), layout="constrained")
    for axis, role in zip(axes, ROLES, strict=True):
        for recording in sorted({key[2] for key in cache}):
            metadata = cache[("condition_affine_flow_fid", role, recording)][1]
            validation = np.asarray(metadata["validation_losses"], dtype=np.float64)
            axis.plot(np.arange(1, validation.size + 1), validation, linewidth=1.1, label=recording)
            best_epoch = int(metadata["best_epoch"])
            if 1 <= best_epoch <= validation.size:
                axis.scatter(best_epoch, validation[best_epoch - 1], s=22, zorder=3)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation flow loss")
        axis.set_title("Reference" if role == "reference" else "All-trial query")
        _style_axis(axis)
    axes[1].legend(frameon=False, ncol=1, loc="best")
    figure.savefig(output_dir / "fid_flow_validation_loss_curves.png", dpi=300, facecolor="white")
    figure.savefig(output_dir / "fid_flow_validation_loss_curves.svg", facecolor="white")
    plt.close(figure)


def _plot_quarter_query_flow_losses(
    output_dir: Path,
    metadata: dict[str, dict],
    recordings: list[str],
) -> None:
    """Plot validation loss and selected best epoch for quarter-query flow fits."""

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    figure, axis = plt.subplots(figsize=(4.0, 3.5), layout="constrained")
    for recording in recordings:
        validation = np.asarray(metadata[recording]["validation_losses"], dtype=np.float64)
        epochs = np.arange(1, validation.size + 1)
        axis.plot(epochs, validation, linewidth=1.2, label=recording)
        best_epoch = int(metadata[recording]["best_epoch"])
        if 1 <= best_epoch <= validation.size:
            axis.scatter(best_epoch, validation[best_epoch - 1], s=22, zorder=3)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation flow loss")
    axis.set_title("Quarter-query flow fits")
    axis.legend(frameon=False, loc="best")
    _style_axis(axis)
    figure.savefig(
        output_dir / "quarter_query_flow_validation_loss_curves.png",
        dpi=300,
        facecolor="white",
    )
    figure.savefig(
        output_dir / "quarter_query_flow_validation_loss_curves.svg",
        facecolor="white",
    )
    plt.close(figure)


def _aggregate(args: argparse.Namespace) -> None:
    recordings = list(args.recordings)
    if len(recordings) != 5 or len(set(recordings)) != 5:
        raise ValueError("This speed-limited experiment requires five unique recordings.")
    datasets = [load_features_npz(args.feature_dir / f"{recording}.npz") for recording in recordings]
    times = np.asarray(datasets[0].time_centers, dtype=np.float64)
    if any(not np.array_equal(dataset.time_centers, times) for dataset in datasets[1:]):
        raise ValueError("Feature files do not share the same native-time grid.")
    cache: dict[tuple[str, str, str], tuple[np.ndarray, dict]] = {}
    fit_rows: list[dict] = []
    for recording in recordings:
        fit_path = args.output_dir / f"fit_{recording}.json"
        if not fit_path.is_file():
            raise FileNotFoundError(f"Missing completed fit summary: {fit_path}")
        fit_payload = json.loads(fit_path.read_text(encoding="utf-8"))
        fit_rows.extend(fit_payload["fits"])
        for role in ROLES:
            for method in METHODS:
                path = args.output_dir / "rdm_cache" / f"{role}_{recording}_{method}.npz"
                rdms, metadata = _load_cache(path)
                if rdms.shape[0] != times.size:
                    raise ValueError(f"Time-grid mismatch in {path}.")
                cache[(method, role, recording)] = (rdms, metadata)
            checkpoint = (
                args.output_dir
                / "checkpoints"
                / f"{role}_{recording}_condition_affine_flow_fid_best.pt"
            )
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                raise FileNotFoundError(f"Missing best checkpoint: {checkpoint}")

    scores = np.empty((len(METHODS), len(recordings), len(recordings)), dtype=np.float64)
    raw_scores = np.empty_like(scores)
    for method_index, method in enumerate(METHODS):
        for query_index, query_recording in enumerate(recordings):
            query_rdm = cache[(method, "all_trial", query_recording)][0]
            query_sqrt = rdm_upper_triangle_sequence(
                np.sqrt(np.maximum(query_rdm, 0.0)),
                times,
                interval=RDM_MATCHING_INTERVAL,
            )[0].reshape(-1)
            query_raw = rdm_upper_triangle_sequence(
                query_rdm,
                times,
                interval=RDM_MATCHING_INTERVAL,
            )[0].reshape(-1)
            for reference_index, reference_recording in enumerate(recordings):
                reference_rdm = cache[(method, "reference", reference_recording)][0]
                reference_sqrt = rdm_upper_triangle_sequence(
                    np.sqrt(np.maximum(reference_rdm, 0.0)),
                    times,
                    interval=RDM_MATCHING_INTERVAL,
                )[0].reshape(-1)
                reference_raw = rdm_upper_triangle_sequence(
                    reference_rdm,
                    times,
                    interval=RDM_MATCHING_INTERVAL,
                )[0].reshape(-1)
                scores[method_index, query_index, reference_index] = pearson_similarity(
                    query_sqrt,
                    reference_sqrt,
                )
                raw_scores[method_index, query_index, reference_index] = pearson_similarity(
                    query_raw,
                    reference_raw,
                )

    metrics: dict[str, dict] = {}
    for method_index, method in enumerate(METHODS):
        metrics[method] = {
            "sqrt_fid_primary": _rank_metrics(scores[method_index], recordings),
            "raw_fid_sensitivity": _rank_metrics(raw_scores[method_index], recordings),
        }
    visible_cue_interval = (0.0, 1.25)
    visible_cue_mask = (
        (times >= visible_cue_interval[0])
        & (times <= visible_cue_interval[1])
    )
    visible_cue_raw_scores = _raw_fid_scores_for_interval(
        cache,
        times,
        recordings,
        interval=visible_cue_interval,
    )
    visible_cue_raw_metrics = {
        method: _rank_metrics(visible_cue_raw_scores[method_index], recordings)
        for method_index, method in enumerate(METHODS)
    }
    visible_cue_raw_mse = _raw_fid_mse_for_interval(
        cache,
        times,
        recordings,
        interval=visible_cue_interval,
    )
    visible_cue_raw_mse_metrics = {
        method: _rank_metrics(-visible_cue_raw_mse[method_index], recordings)
        for method_index, method in enumerate(METHODS)
    }

    bin_width_seconds = 0.25
    visible_cue_bin_edges = np.linspace(
        visible_cue_interval[0],
        visible_cue_interval[1],
        num=6,
        dtype=np.float64,
    )
    binned_cache_dir = args.output_dir / "binned_rdm_cache"
    binned_cache_dir.mkdir(parents=True, exist_ok=True)
    binned_classical_cache: dict[tuple[str, str], np.ndarray] = {}
    visible_cue_bin_centers: np.ndarray | None = None
    visible_cue_bin_sample_counts: np.ndarray | None = None
    for recording, dataset in zip(recordings, datasets, strict=True):
        split_path = args.output_dir / "splits" / f"{recording}_mixed_half_split.npz"
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing exact split definition: {split_path}")
        with np.load(split_path, allow_pickle=False) as split_archive:
            split_labels = np.asarray(split_archive["labels"], dtype=np.int64)
            split_seed = int(split_archive["split_seed"].item())
            role_indices = {
                "reference": np.asarray(split_archive["reference_indices"], dtype=np.int64),
                "all_trial": np.asarray(split_archive["all_trial_indices"], dtype=np.int64),
            }
        if not np.array_equal(split_labels, dataset.labels):
            raise RuntimeError(f"Saved split labels do not match the feature file for {recording}.")
        if np.intersect1d(role_indices["reference"], role_indices["all_trial"]).size:
            raise RuntimeError(f"Saved split has reference/query leakage for {recording}.")
        if np.union1d(role_indices["reference"], role_indices["all_trial"]).size != len(
            dataset.labels
        ):
            raise RuntimeError(f"Saved split does not use every clean trial for {recording}.")

        for role in ROLES:
            indices = role_indices[role]
            binned_features, bin_centers, bin_sample_counts = _mean_voltage_in_time_bins(
                dataset.features[indices],
                times,
                visible_cue_bin_edges,
            )
            if visible_cue_bin_centers is None:
                visible_cue_bin_centers = bin_centers
                visible_cue_bin_sample_counts = bin_sample_counts
            else:
                np.testing.assert_array_equal(bin_centers, visible_cue_bin_centers)
                np.testing.assert_array_equal(
                    bin_sample_counts,
                    visible_cue_bin_sample_counts,
                )
            start = time.perf_counter()
            binned_rdms = classical_fid_rdms(
                binned_features,
                dataset.labels[indices],
                standardize_features=False,
            )
            elapsed_seconds = time.perf_counter() - start
            binned_path = (
                binned_cache_dir / f"{role}_{recording}_classical_fid_250ms.npz"
            )
            _save_cache(
                binned_path,
                binned_rdms,
                {
                    "recording": recording,
                    "role": role,
                    "method": "classical_fid_250ms_mean_voltage_bins",
                    "feature_definition": (
                        "per-trial mean native EEG voltage within each half-open time bin"
                    ),
                    "estimator": "class_and_time_bin_specific_ledoit_wolf_gaussian",
                    "distance": "raw_squared_gaussian_2_wasserstein_fid",
                    "bin_edges_seconds_cue_relative": visible_cue_bin_edges.tolist(),
                    "bin_centers_seconds_cue_relative": bin_centers.tolist(),
                    "native_samples_per_bin": bin_sample_counts.astype(int).tolist(),
                    "n_trials": int(indices.size),
                    "split_seed": split_seed,
                    "split_path": str(split_path.resolve()),
                    "elapsed_seconds": float(elapsed_seconds),
                },
            )
            checked_rdms, _ = _load_cache(binned_path)
            binned_classical_cache[(role, recording)] = checked_rdms
            print(
                f"[fit] {binned_path.name} elapsed={elapsed_seconds:.2f}s",
                flush=True,
            )

    if visible_cue_bin_centers is None or visible_cue_bin_sample_counts is None:
        raise RuntimeError("No 250 ms binned classical RDMs were computed.")
    binned_classical_mse = _raw_fid_mse_from_binned_classical_rdms(
        binned_classical_cache,
        recordings,
    )
    binned_classical_metrics = _rank_metrics(-binned_classical_mse, recordings)
    combined_visible_cue_mse_top1 = np.asarray(
        [
            visible_cue_raw_mse_metrics["classical_fid"]["top1_accuracy"],
            binned_classical_metrics["top1_accuracy"],
            visible_cue_raw_mse_metrics["condition_affine_flow_fid"]["top1_accuracy"],
        ],
        dtype=np.float64,
    )

    reduced_query_split_dir = args.output_dir / "reduced_query_splits"
    reduced_query_cache_dir = args.output_dir / "reduced_query_rdm_cache"
    reduced_query_split_dir.mkdir(parents=True, exist_ok=True)
    reduced_query_cache_dir.mkdir(parents=True, exist_ok=True)
    reduction_specs = {
        "half_query": {
            "fraction": 0.5,
            "split_stem": "half_of_query",
            "role": "half_of_saved_query",
            "seed_offset": 50_000_000,
        },
        "quarter_query": {
            "fraction": 0.25,
            "split_stem": "quarter_of_query",
            "role": "quarter_of_saved_query",
            "seed_offset": 60_000_000,
        },
    }
    reduced_native_mse_by_size: dict[str, np.ndarray] = {}
    reduced_binned_mse_by_size: dict[str, np.ndarray] = {}
    reduced_native_metrics_by_size: dict[str, dict] = {}
    reduced_binned_metrics_by_size: dict[str, dict] = {}
    reduced_query_rows_by_size: dict[str, list[dict]] = {}
    native_references = {
        recording: cache[("classical_fid", "reference", recording)][0]
        for recording in recordings
    }
    for size_label, spec in reduction_specs.items():
        fraction = float(spec["fraction"])
        reduced_native_query_rdms: dict[str, np.ndarray] = {}
        reduced_binned_cache: dict[tuple[str, str], np.ndarray] = {
            ("reference", recording): binned_classical_cache[("reference", recording)]
            for recording in recordings
        }
        reduced_query_rows: list[dict] = []
        for recording_index, (recording, dataset) in enumerate(
            zip(recordings, datasets, strict=True)
        ):
            split_path = args.output_dir / "splits" / f"{recording}_mixed_half_split.npz"
            with np.load(split_path, allow_pickle=False) as split_archive:
                full_query_indices = np.asarray(
                    split_archive["all_trial_indices"],
                    dtype=np.int64,
                )
            subsample_seed = int(
                args.seed + int(spec["seed_offset"]) + recording_index * 100_000
            )
            reduced_indices = _stratified_fraction_subsample(
                full_query_indices,
                dataset.labels,
                fraction=fraction,
                seed=subsample_seed,
            )
            if not np.all(np.isin(reduced_indices, full_query_indices)):
                raise RuntimeError(
                    f"Reduced query escaped the saved query half for {recording}."
                )
            reduced_counts = per_class_counts(dataset.labels[reduced_indices]).astype(int)
            full_counts = per_class_counts(dataset.labels[full_query_indices]).astype(int)
            expected_counts = np.floor(fraction * full_counts).astype(int)
            np.testing.assert_array_equal(reduced_counts, expected_counts)
            _atomic_npz(
                reduced_query_split_dir / f"{recording}_{spec['split_stem']}.npz",
                selected_indices=reduced_indices,
                parent_query_indices=full_query_indices,
                per_class_counts=reduced_counts,
                parent_per_class_counts=full_counts,
                fraction=np.asarray(fraction, dtype=np.float64),
                subsample_seed=np.asarray(subsample_seed, dtype=np.int64),
            )

            start = time.perf_counter()
            reduced_native_rdm = classical_fid_rdms(
                dataset.features[reduced_indices],
                dataset.labels[reduced_indices],
                standardize_features=False,
            )
            native_elapsed_seconds = time.perf_counter() - start
            native_path = (
                reduced_query_cache_dir
                / f"{size_label}_{recording}_classical_fid_native_time.npz"
            )
            common_metadata = {
                "recording": recording,
                "role": str(spec["role"]),
                "parent_query_count": int(full_query_indices.size),
                "selected_query_count": int(reduced_indices.size),
                "parent_per_class_counts": full_counts.tolist(),
                "selected_per_class_counts": reduced_counts.tolist(),
                "fraction_of_saved_query": fraction,
                "subsample_seed": subsample_seed,
                "sampling": "class-stratified without replacement",
            }
            _save_cache(
                native_path,
                reduced_native_rdm,
                {
                    **common_metadata,
                    "method": "classical_fid_native_time",
                    "estimator": "class_and_native_time_specific_ledoit_wolf_gaussian",
                    "distance": "raw_squared_gaussian_2_wasserstein_fid",
                    "elapsed_seconds": float(native_elapsed_seconds),
                },
            )
            reduced_native_query_rdms[recording] = _load_cache(native_path)[0]

            reduced_binned_features, reduced_bin_centers, reduced_bin_counts = (
                _mean_voltage_in_time_bins(
                    dataset.features[reduced_indices],
                    times,
                    visible_cue_bin_edges,
                )
            )
            np.testing.assert_array_equal(reduced_bin_centers, visible_cue_bin_centers)
            np.testing.assert_array_equal(
                reduced_bin_counts,
                visible_cue_bin_sample_counts,
            )
            start = time.perf_counter()
            reduced_binned_rdm = classical_fid_rdms(
                reduced_binned_features,
                dataset.labels[reduced_indices],
                standardize_features=False,
            )
            binned_elapsed_seconds = time.perf_counter() - start
            binned_path = (
                reduced_query_cache_dir
                / f"{size_label}_{recording}_classical_fid_250ms.npz"
            )
            _save_cache(
                binned_path,
                reduced_binned_rdm,
                {
                    **common_metadata,
                    "method": "classical_fid_250ms_mean_voltage_bins",
                    "feature_definition": (
                        "per-trial mean native EEG voltage within each half-open time bin"
                    ),
                    "estimator": "class_and_time_bin_specific_ledoit_wolf_gaussian",
                    "distance": "raw_squared_gaussian_2_wasserstein_fid",
                    "bin_edges_seconds_cue_relative": visible_cue_bin_edges.tolist(),
                    "native_samples_per_bin": reduced_bin_counts.astype(int).tolist(),
                    "elapsed_seconds": float(binned_elapsed_seconds),
                },
            )
            reduced_binned_cache[("all_trial", recording)] = _load_cache(binned_path)[0]
            reduced_query_rows.append(
                {
                    **common_metadata,
                    "selected_indices_path": str(
                        (
                            reduced_query_split_dir
                            / f"{recording}_{spec['split_stem']}.npz"
                        ).resolve()
                    ),
                }
            )
            print(
                f"[fit] {recording} {size_label} classical "
                f"native={native_elapsed_seconds:.2f}s "
                f"250ms={binned_elapsed_seconds:.2f}s n={reduced_indices.size}",
                flush=True,
            )

        native_mse = _raw_fid_mse_from_query_reference_rdms(
            reduced_native_query_rdms,
            native_references,
            times,
            recordings,
            interval=visible_cue_interval,
        )
        binned_mse = _raw_fid_mse_from_binned_classical_rdms(
            reduced_binned_cache,
            recordings,
        )
        reduced_native_mse_by_size[size_label] = native_mse
        reduced_binned_mse_by_size[size_label] = binned_mse
        reduced_native_metrics_by_size[size_label] = _rank_metrics(-native_mse, recordings)
        reduced_binned_metrics_by_size[size_label] = _rank_metrics(-binned_mse, recordings)
        reduced_query_rows_by_size[size_label] = reduced_query_rows

    reduced_native_mse = reduced_native_mse_by_size["half_query"]
    reduced_binned_mse = reduced_binned_mse_by_size["half_query"]
    reduced_native_metrics = reduced_native_metrics_by_size["half_query"]
    reduced_binned_metrics = reduced_binned_metrics_by_size["half_query"]
    reduced_query_rows = reduced_query_rows_by_size["half_query"]
    quarter_native_mse = reduced_native_mse_by_size["quarter_query"]
    quarter_binned_mse = reduced_binned_mse_by_size["quarter_query"]
    quarter_native_metrics = reduced_native_metrics_by_size["quarter_query"]
    quarter_binned_metrics = reduced_binned_metrics_by_size["quarter_query"]
    quarter_query_rows = reduced_query_rows_by_size["quarter_query"]
    classical_query_size_top1 = np.asarray(
        [
            visible_cue_raw_mse_metrics["classical_fid"]["top1_accuracy"],
            reduced_native_metrics["top1_accuracy"],
            binned_classical_metrics["top1_accuracy"],
            reduced_binned_metrics["top1_accuracy"],
        ],
        dtype=np.float64,
    )
    classical_query_size_top1_with_quarter = np.asarray(
        [
            visible_cue_raw_mse_metrics["classical_fid"]["top1_accuracy"],
            reduced_native_metrics["top1_accuracy"],
            quarter_native_metrics["top1_accuracy"],
            binned_classical_metrics["top1_accuracy"],
            reduced_binned_metrics["top1_accuracy"],
            quarter_binned_metrics["top1_accuracy"],
        ],
        dtype=np.float64,
    )
    _atomic_npz(
        args.output_dir / "fid_session_identification_results.npz",
        sqrt_fid_scores=scores,
        raw_fid_scores=raw_scores,
        recordings=np.asarray(recordings),
        methods=np.asarray(METHODS),
        time_seconds_cue_relative=times,
        matching_interval=np.asarray(RDM_MATCHING_INTERVAL),
    )
    _atomic_npz(
        args.output_dir / "raw_fid_visible_cue_session_identification_results.npz",
        raw_fid_scores=visible_cue_raw_scores,
        recordings=np.asarray(recordings),
        methods=np.asarray(METHODS),
        requested_interval=np.asarray(visible_cue_interval),
        selected_time_seconds_cue_relative=times[visible_cue_mask],
    )
    _atomic_json(
        args.output_dir / "raw_fid_visible_cue_session_identification_summary.json",
        {
            "distance_representation": "raw squared Gaussian FID",
            "matching": "zero-lag Pearson correlation",
            "requested_interval_seconds_cue_relative": list(visible_cue_interval),
            "actual_selected_interval_seconds_cue_relative": [
                float(times[visible_cue_mask][0]),
                float(times[visible_cue_mask][-1]),
            ],
            "n_time_points": int(np.sum(visible_cue_mask)),
            "recordings": recordings,
            "chance_top1": 1.0 / len(recordings),
            "metrics": visible_cue_raw_metrics,
        },
    )
    _atomic_npz(
        args.output_dir / "raw_fid_visible_cue_mse_session_identification_results.npz",
        raw_fid_mse=visible_cue_raw_mse,
        recordings=np.asarray(recordings),
        methods=np.asarray(METHODS),
        requested_interval=np.asarray(visible_cue_interval),
        selected_time_seconds_cue_relative=times[visible_cue_mask],
    )
    _atomic_json(
        args.output_dir / "raw_fid_visible_cue_mse_session_identification_summary.json",
        {
            "distance_representation": "raw squared Gaussian FID",
            "query_reference_distance": (
                "mean squared error over selected native times and six unique class pairs"
            ),
            "ranking_rule": "smallest MSE is the top-1 reference",
            "requested_interval_seconds_cue_relative": list(visible_cue_interval),
            "actual_selected_interval_seconds_cue_relative": [
                float(times[visible_cue_mask][0]),
                float(times[visible_cue_mask][-1]),
            ],
            "n_time_points": int(np.sum(visible_cue_mask)),
            "n_class_pairs": 6,
            "n_values_per_query_reference_comparison": int(np.sum(visible_cue_mask) * 6),
            "recordings": recordings,
            "chance_top1": 1.0 / len(recordings),
            "metrics": visible_cue_raw_mse_metrics,
        },
    )
    _atomic_npz(
        args.output_dir / "raw_fid_visible_cue_mse_250ms_binned_classical_results.npz",
        binned_classical_raw_fid_mse=binned_classical_mse,
        combined_top1_accuracy=combined_visible_cue_mse_top1,
        combined_methods=np.asarray(
            ["classical_native_time", "classical_250ms_bins", "flow_native_time"]
        ),
        recordings=np.asarray(recordings),
        bin_edges_seconds_cue_relative=visible_cue_bin_edges,
        bin_centers_seconds_cue_relative=visible_cue_bin_centers,
        native_samples_per_bin=visible_cue_bin_sample_counts,
    )
    _atomic_json(
        args.output_dir / "raw_fid_visible_cue_mse_250ms_binned_classical_summary.json",
        {
            "distance_representation": "raw squared Gaussian FID",
            "feature_definition": (
                "one 22-dimensional vector per trial and bin, formed by averaging all native "
                "EEG voltage samples within each half-open 250 ms bin"
            ),
            "classical_estimator": (
                "separate class-specific Ledoit-Wolf Gaussian in every 250 ms bin"
            ),
            "query_reference_distance": (
                "mean squared error over five time bins and six unique class pairs"
            ),
            "ranking_rule": "smallest MSE is the top-1 reference",
            "requested_interval_seconds_cue_relative": list(visible_cue_interval),
            "bin_width_seconds": bin_width_seconds,
            "bin_edges_seconds_cue_relative": visible_cue_bin_edges.tolist(),
            "bin_centers_seconds_cue_relative": visible_cue_bin_centers.tolist(),
            "native_samples_per_bin": visible_cue_bin_sample_counts.astype(int).tolist(),
            "n_bins": int(visible_cue_bin_centers.size),
            "n_class_pairs": 6,
            "n_values_per_query_reference_comparison": int(
                visible_cue_bin_centers.size * 6
            ),
            "split_reuse": (
                "exact reference/all-trial indices reused from the native-time experiment"
            ),
            "recordings": recordings,
            "chance_top1": 1.0 / len(recordings),
            "binned_classical_metrics": binned_classical_metrics,
            "combined_top1_accuracy": {
                "classical_native_time": float(combined_visible_cue_mse_top1[0]),
                "classical_250ms_bins": float(combined_visible_cue_mse_top1[1]),
                "flow_native_time": float(combined_visible_cue_mse_top1[2]),
            },
        },
    )
    _atomic_npz(
        args.output_dir / "raw_fid_visible_cue_mse_half_query_classical_results.npz",
        native_time_half_query_raw_fid_mse=reduced_native_mse,
        binned_250ms_half_query_raw_fid_mse=reduced_binned_mse,
        top1_accuracy=classical_query_size_top1,
        configurations=np.asarray(
            [
                "native_time_all_query",
                "native_time_half_query",
                "binned_250ms_all_query",
                "binned_250ms_half_query",
            ]
        ),
        recordings=np.asarray(recordings),
    )
    _atomic_json(
        args.output_dir / "raw_fid_visible_cue_mse_half_query_classical_summary.json",
        {
            "experiment": "Classical FID query-size comparison",
            "distance_representation": "raw squared Gaussian FID",
            "query_reference_distance": (
                "mean squared error over the visible cue and six unique class pairs"
            ),
            "ranking_rule": "smallest MSE is the top-1 reference",
            "requested_interval_seconds_cue_relative": list(visible_cue_interval),
            "reference_role": "unchanged saved reference half",
            "full_query_role": "unchanged saved query half",
            "reduced_query_role": (
                "one deterministic class-stratified 50 percent subsample of the saved query "
                "half, sampled without replacement"
            ),
            "n_subsampling_repeats": 1,
            "recordings": recordings,
            "chance_top1": 1.0 / len(recordings),
            "query_counts": reduced_query_rows,
            "metrics": {
                "native_time_all_query": visible_cue_raw_mse_metrics["classical_fid"],
                "native_time_half_query": reduced_native_metrics,
                "binned_250ms_all_query": binned_classical_metrics,
                "binned_250ms_half_query": reduced_binned_metrics,
            },
        },
    )
    _atomic_npz(
        args.output_dir / "raw_fid_visible_cue_mse_quarter_query_classical_results.npz",
        native_time_quarter_query_raw_fid_mse=quarter_native_mse,
        binned_250ms_quarter_query_raw_fid_mse=quarter_binned_mse,
        top1_accuracy=classical_query_size_top1_with_quarter,
        configurations=np.asarray(
            [
                "native_time_all_query",
                "native_time_half_query",
                "native_time_quarter_query",
                "binned_250ms_all_query",
                "binned_250ms_half_query",
                "binned_250ms_quarter_query",
            ]
        ),
        recordings=np.asarray(recordings),
    )
    _atomic_json(
        args.output_dir / "raw_fid_visible_cue_mse_quarter_query_classical_summary.json",
        {
            "experiment": "Classical FID all, half, and quarter query-size comparison",
            "distance_representation": "raw squared Gaussian FID",
            "query_reference_distance": (
                "mean squared error over the visible cue and six unique class pairs"
            ),
            "ranking_rule": "smallest MSE is the top-1 reference",
            "requested_interval_seconds_cue_relative": list(visible_cue_interval),
            "reference_role": "unchanged saved reference half",
            "full_query_role": "unchanged saved query half",
            "quarter_query_role": (
                "one deterministic class-stratified 25 percent subsample drawn directly "
                "from the saved full query half, without replacement"
            ),
            "n_subsampling_repeats_per_reduced_size": 1,
            "recordings": recordings,
            "chance_top1": 1.0 / len(recordings),
            "quarter_query_counts": quarter_query_rows,
            "metrics": {
                "native_time_all_query": visible_cue_raw_mse_metrics["classical_fid"],
                "native_time_half_query": reduced_native_metrics,
                "native_time_quarter_query": quarter_native_metrics,
                "binned_250ms_all_query": binned_classical_metrics,
                "binned_250ms_half_query": reduced_binned_metrics,
                "binned_250ms_quarter_query": quarter_binned_metrics,
            },
        },
    )
    with (args.output_dir / "fit_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "recording",
            "role",
            "method",
            "n_trials",
            "per_class_counts",
            "run_by_class_counts",
            "elapsed_seconds",
            "best_epoch",
            "stopped_epoch",
            "best_val_loss",
            "checkpoint_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fit_rows)
    pair_rows: list[dict] = []
    for method_index, method in enumerate(METHODS):
        ranks = metrics[method]["sqrt_fid_primary"]["ranks"]
        for query_index, query_recording in enumerate(recordings):
            for reference_index, reference_recording in enumerate(recordings):
                pair_rows.append(
                    {
                        "method": method,
                        "query_recording": query_recording,
                        "reference_recording": reference_recording,
                        "is_true_pair": query_index == reference_index,
                        "sqrt_fid_zero_lag_correlation": float(
                            scores[method_index, query_index, reference_index]
                        ),
                        "raw_fid_zero_lag_correlation": float(
                            raw_scores[method_index, query_index, reference_index]
                        ),
                        "true_recording_rank": int(ranks[query_index]),
                    }
                )
    with (args.output_dir / "pair_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    summary = {
        "experiment": "BCI IV-2a five-recording mixed-run time-resolved FID RDM identification",
        "recordings": recordings,
        "chance_top1": 1.0 / len(recordings),
        "device": str(args.device),
        "split_design": (
            "pool all artifact-clean trials across all six runs, then seeded class-stratified "
            "half split; every clean trial used exactly once"
        ),
        "reference_role": "one stratified half",
        "query_role": "the complete other stratified half (all-trial query)",
        "matching": "zero-lag Pearson correlation from cue through 3.5 s",
        "primary_matching_representation": "sqrt(FID), equal to Gaussian 2-Wasserstein distance",
        "sensitivity_matching_representation": "raw squared Gaussian FID",
        "time_interval_seconds_cue_relative": list(RDM_MATCHING_INTERVAL),
        "sample_period_seconds": float(np.median(np.diff(times))),
        "n_time_points_total": int(times.size),
        "n_time_points_matched": int(np.sum((times >= 0.0) & (times <= 3.5))),
        "classical_estimator": "class-and-time-specific Ledoit-Wolf Gaussian plug-in FID",
        "flow_estimator": "joint class-and-time condition-affine full-covariance Gaussian flow",
        "epochs": int(args.epochs),
        "early_stopping_patience": int(args.patience),
        "checkpoint_rule": "exact best-validation model restored for RDM evaluation and saved",
        "metrics": metrics,
        "visible_cue_raw_fid_metrics": visible_cue_raw_metrics,
        "visible_cue_raw_fid_mse_metrics": visible_cue_raw_mse_metrics,
        "visible_cue_250ms_binned_classical_fid_mse_metrics": binned_classical_metrics,
        "visible_cue_half_query_classical_fid_mse_metrics": {
            "native_time": reduced_native_metrics,
            "binned_250ms": reduced_binned_metrics,
        },
        "visible_cue_quarter_query_classical_fid_mse_metrics": {
            "native_time": quarter_native_metrics,
            "binned_250ms": quarter_binned_metrics,
        },
        "fit_summaries": fit_rows,
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    _plot_identification(args.output_dir, scores, metrics, recordings)
    _plot_top1_bar(
        args.output_dir,
        metrics,
        metric_key="sqrt_fid_primary",
        output_stem="fid_top1_accuracy_bar",
    )
    _plot_top1_bar(
        args.output_dir,
        metrics,
        metric_key="raw_fid_sensitivity",
        output_stem="raw_fid_top1_accuracy_bar",
    )
    _plot_top1_bar_values(
        args.output_dir,
        np.asarray(
            [
                visible_cue_raw_metrics["classical_fid"]["top1_accuracy"],
                visible_cue_raw_metrics["condition_affine_flow_fid"]["top1_accuracy"],
            ]
        ),
        output_stem="raw_fid_visible_cue_top1_accuracy_bar",
        title="Visible cue: raw FID",
    )
    _plot_top1_bar_values(
        args.output_dir,
        np.asarray(
            [
                visible_cue_raw_mse_metrics["classical_fid"]["top1_accuracy"],
                visible_cue_raw_mse_metrics["condition_affine_flow_fid"]["top1_accuracy"],
            ]
        ),
        output_stem="raw_fid_visible_cue_mse_top1_accuracy_bar",
        title="Visible cue: raw-FID MSE",
    )
    _plot_top1_bar_values(
        args.output_dir,
        combined_visible_cue_mse_top1,
        output_stem="raw_fid_visible_cue_mse_top1_with_250ms_binned_classical_bar",
        title="Visible cue: raw-FID MSE",
        labels=["Classical\nnative", "Classical\n250 ms", "Flow-based\nnative"],
        colors=[
            METHOD_COLORS["classical_fid"],
            "#EE7733",
            METHOD_COLORS["condition_affine_flow_fid"],
        ],
    )
    _plot_top1_bar_values(
        args.output_dir,
        classical_query_size_top1,
        output_stem="raw_fid_visible_cue_mse_classical_all_vs_half_query_top1_bar",
        title="Classical FID: query-size effect",
        labels=[
            "Native\nall query",
            "Native\nhalf query",
            "250 ms\nall query",
            "250 ms\nhalf query",
        ],
        colors=["#4477AA", "#88ACD0", "#EE7733", "#F6B282"],
    )
    _plot_top1_bar_values(
        args.output_dir,
        classical_query_size_top1_with_quarter,
        output_stem=(
            "raw_fid_visible_cue_mse_classical_all_half_quarter_query_top1_bar"
        ),
        title="Classical FID: query-size effect",
        labels=[
            "Native\nall",
            "Native\n1/2",
            "Native\n1/4",
            "250 ms\nall",
            "250 ms\n1/2",
            "250 ms\n1/4",
        ],
        colors=["#4477AA", "#77A1C8", "#A9C6DF", "#EE7733", "#F39A60", "#F7BD93"],
    )
    _plot_raw_fid_mean_trajectories(
        args.output_dir,
        cache,
        times,
        recordings,
    )
    _plot_loss_curves(args.output_dir, cache)
    for method in METHODS:
        primary = metrics[method]["sqrt_fid_primary"]
        raw = metrics[method]["raw_fid_sensitivity"]
        cue_raw = visible_cue_raw_metrics[method]
        cue_raw_mse = visible_cue_raw_mse_metrics[method]
        print(
            f"[result] {METHOD_LABELS[method]} sqrt-FID top1={primary['top1_accuracy']:.3f} "
            f"MRR={primary['mean_reciprocal_rank']:.3f} "
            f"margin={primary['mean_true_minus_best_competitor_margin']:.3f}; "
            f"raw-FID top1={raw['top1_accuracy']:.3f}; "
            f"visible-cue raw-FID correlation top1={cue_raw['top1_accuracy']:.3f}; "
            f"visible-cue raw-FID MSE top1={cue_raw_mse['top1_accuracy']:.3f}",
            flush=True,
        )
    print(
        f"[result] Classical FID 250 ms bins visible-cue raw-FID MSE "
        f"top1={binned_classical_metrics['top1_accuracy']:.3f} "
        f"ranks={binned_classical_metrics['ranks']}",
        flush=True,
    )
    print(
        f"[result] Half-query Classical FID native-time visible-cue raw-FID MSE "
        f"top1={reduced_native_metrics['top1_accuracy']:.3f} "
        f"ranks={reduced_native_metrics['ranks']}",
        flush=True,
    )
    print(
        f"[result] Half-query Classical FID 250 ms bins visible-cue raw-FID MSE "
        f"top1={reduced_binned_metrics['top1_accuracy']:.3f} "
        f"ranks={reduced_binned_metrics['ranks']}",
        flush=True,
    )
    print(
        f"[result] Quarter-query Classical FID native-time visible-cue raw-FID MSE "
        f"top1={quarter_native_metrics['top1_accuracy']:.3f} "
        f"ranks={quarter_native_metrics['ranks']}",
        flush=True,
    )
    print(
        f"[result] Quarter-query Classical FID 250 ms bins visible-cue raw-FID MSE "
        f"top1={quarter_binned_metrics['top1_accuracy']:.3f} "
        f"ranks={quarter_binned_metrics['ranks']}",
        flush=True,
    )
    print(f"[experiment] output={args.output_dir.resolve()}", flush=True)


def _aggregate_full_query_nine_recordings(args: argparse.Namespace) -> None:
    """Aggregate the full-query raw-FID MSE comparison over all nine recordings."""

    recordings = list(args.recordings)
    if len(recordings) != 9 or len(set(recordings)) != 9:
        raise ValueError("The full-query nine-recording aggregate requires nine recordings.")
    datasets = [load_features_npz(args.feature_dir / f"{recording}.npz") for recording in recordings]
    times = np.asarray(datasets[0].time_centers, dtype=np.float64)
    if any(not np.array_equal(dataset.time_centers, times) for dataset in datasets[1:]):
        raise ValueError("Feature files do not share the same native-time grid.")

    visible_cue_interval = (0.0, 1.25)
    visible_cue_bin_edges = np.linspace(0.0, 1.25, num=6, dtype=np.float64)
    native_classical_query: dict[str, np.ndarray] = {}
    native_classical_reference: dict[str, np.ndarray] = {}
    flow_query: dict[str, np.ndarray] = {}
    flow_reference: dict[str, np.ndarray] = {}
    binned_cache: dict[tuple[str, str], np.ndarray] = {}
    flow_metadata: dict[tuple[str, str], dict] = {}
    binned_cache_dir = args.output_dir / "binned_rdm_cache"
    best_flow_cache_dir = args.output_dir / "best_checkpoint_full_query_flow_rdm_cache"
    binned_cache_dir.mkdir(parents=True, exist_ok=True)
    best_flow_cache_dir.mkdir(parents=True, exist_ok=True)
    bin_centers_common: np.ndarray | None = None
    bin_counts_common: np.ndarray | None = None
    split_rows: list[dict] = []

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The full-query flow checkpoint evaluation requires CUDA.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device.index} is unavailable.")
    torch.cuda.set_device(0 if device.index is None else device.index)
    # Training uses this precision mode, so exact best-checkpoint reevaluation must too.
    torch.set_float32_matmul_precision("high")
    for recording, dataset in zip(recordings, datasets, strict=True):
        split_path = args.output_dir / "splits" / f"{recording}_mixed_half_split.npz"
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing mixed half split: {split_path}")
        with np.load(split_path, allow_pickle=False) as split_archive:
            split_labels = np.asarray(split_archive["labels"], dtype=np.int64)
            split_seed = int(split_archive["split_seed"].item())
            role_indices = {
                "reference": np.asarray(split_archive["reference_indices"], dtype=np.int64),
                "all_trial": np.asarray(split_archive["all_trial_indices"], dtype=np.int64),
            }
        if not np.array_equal(split_labels, dataset.labels):
            raise RuntimeError(f"Saved split labels differ from features for {recording}.")
        if np.intersect1d(role_indices["reference"], role_indices["all_trial"]).size:
            raise RuntimeError(f"Reference/query leakage detected for {recording}.")
        if np.union1d(role_indices["reference"], role_indices["all_trial"]).size != len(
            dataset.labels
        ):
            raise RuntimeError(f"Split does not use every clean trial for {recording}.")
        split_rows.append(
            {
                "recording": recording,
                "split_seed": split_seed,
                "reference_n_trials": int(role_indices["reference"].size),
                "query_n_trials": int(role_indices["all_trial"].size),
                "reference_per_class_counts": per_class_counts(
                    dataset.labels[role_indices["reference"]]
                ).astype(int).tolist(),
                "query_per_class_counts": per_class_counts(
                    dataset.labels[role_indices["all_trial"]]
                ).astype(int).tolist(),
            }
        )

        for role in ROLES:
            indices = role_indices[role]
            classical_path = (
                args.output_dir / "rdm_cache" / f"{role}_{recording}_classical_fid.npz"
            )
            classical_rdm = _load_cache(classical_path)[0]
            if classical_rdm.shape != (times.size, 4, 4):
                raise ValueError(f"Unexpected native classical RDM shape in {classical_path}.")
            if role == "reference":
                native_classical_reference[recording] = classical_rdm
            else:
                native_classical_query[recording] = classical_rdm

            binned_features, bin_centers, bin_counts = _mean_voltage_in_time_bins(
                dataset.features[indices],
                times,
                visible_cue_bin_edges,
            )
            if bin_centers_common is None:
                bin_centers_common = bin_centers
                bin_counts_common = bin_counts
            else:
                np.testing.assert_array_equal(bin_centers, bin_centers_common)
                np.testing.assert_array_equal(bin_counts, bin_counts_common)
            binned_rdm = classical_fid_rdms(
                binned_features,
                dataset.labels[indices],
                standardize_features=False,
            )
            binned_path = (
                binned_cache_dir / f"{role}_{recording}_classical_fid_250ms.npz"
            )
            _save_cache(
                binned_path,
                binned_rdm,
                {
                    "recording": recording,
                    "role": role,
                    "method": "classical_fid_250ms_mean_voltage_bins",
                    "feature_definition": (
                        "per-trial mean native EEG voltage within each half-open time bin"
                    ),
                    "estimator": "class_and_time_bin_specific_ledoit_wolf_gaussian",
                    "distance": "raw_squared_gaussian_2_wasserstein_fid",
                    "bin_edges_seconds_cue_relative": visible_cue_bin_edges.tolist(),
                    "native_samples_per_bin": bin_counts.astype(int).tolist(),
                    "n_trials": int(indices.size),
                    "split_seed": split_seed,
                },
            )
            binned_cache[(role, recording)] = _load_cache(binned_path)[0]

            checkpoint_path = (
                args.output_dir
                / "checkpoints"
                / f"{role}_{recording}_condition_affine_flow_fid_best.pt"
            )
            flow_rdm, checkpoint_payload = (
                condition_affine_flow_fid_rdms_from_checkpoint(
                    checkpoint_path,
                    device=device,
                )
            )
            checkpoint_context = dict(checkpoint_payload.get("context", {}))
            if checkpoint_context.get("recording") != recording:
                raise RuntimeError(f"Checkpoint recording mismatch in {checkpoint_path}.")
            if checkpoint_context.get("role") != role:
                raise RuntimeError(f"Checkpoint role mismatch in {checkpoint_path}.")
            if int(checkpoint_context.get("split_seed", -1)) != split_seed:
                raise RuntimeError(f"Checkpoint split mismatch in {checkpoint_path}.")
            flow_path = (
                best_flow_cache_dir
                / f"{role}_{recording}_condition_affine_flow_fid_best_reloaded.npz"
            )
            flow_meta = {
                "recording": recording,
                "role": role,
                "method": "condition_affine_flow_fid",
                "rdm_evaluation_source": (
                    "explicitly_reloaded_best_validation_checkpoint"
                ),
                "checkpoint_path": str(checkpoint_path.resolve()),
                "best_epoch": int(checkpoint_payload["training"]["best_epoch"]),
                "stopped_epoch": int(checkpoint_payload["training"]["stopped_epoch"]),
                "best_val_loss": float(checkpoint_payload["training"]["best_val_loss"]),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
            }
            original_flow_path = (
                args.output_dir
                / "rdm_cache"
                / f"{role}_{recording}_condition_affine_flow_fid.npz"
            )
            _, original_flow_meta = _load_cache(original_flow_path)
            for loss_key in (
                "train_losses",
                "validation_losses",
                "monitored_validation_losses",
            ):
                flow_meta[loss_key] = original_flow_meta[loss_key]
            _save_cache(flow_path, flow_rdm, flow_meta)
            checked_flow_rdm, checked_flow_meta = _load_cache(flow_path)
            flow_metadata[(role, recording)] = checked_flow_meta
            if role == "reference":
                flow_reference[recording] = checked_flow_rdm
            else:
                flow_query[recording] = checked_flow_rdm

    if bin_centers_common is None or bin_counts_common is None:
        raise RuntimeError("No binned RDMs were computed.")
    native_classical_mse = _raw_fid_mse_from_query_reference_rdms(
        native_classical_query,
        native_classical_reference,
        times,
        recordings,
        interval=visible_cue_interval,
    )
    binned_classical_mse = _raw_fid_mse_from_binned_classical_rdms(
        binned_cache,
        recordings,
    )
    flow_mse = _raw_fid_mse_from_query_reference_rdms(
        flow_query,
        flow_reference,
        times,
        recordings,
        interval=visible_cue_interval,
    )
    metrics = {
        "classical_native_time": _rank_metrics(-native_classical_mse, recordings),
        "classical_250ms_bins": _rank_metrics(-binned_classical_mse, recordings),
        "flow_native_time": _rank_metrics(-flow_mse, recordings),
    }
    top1 = np.asarray(
        [
            metrics["classical_native_time"]["top1_accuracy"],
            metrics["classical_250ms_bins"]["top1_accuracy"],
            metrics["flow_native_time"]["top1_accuracy"],
        ],
        dtype=np.float64,
    )
    visible_cue_mask = (times >= visible_cue_interval[0]) & (
        times <= visible_cue_interval[1]
    )
    _atomic_npz(
        args.output_dir / "raw_fid_visible_cue_mse_full_query_9recordings_results.npz",
        classical_native_time_raw_fid_mse=native_classical_mse,
        classical_250ms_raw_fid_mse=binned_classical_mse,
        flow_native_time_raw_fid_mse=flow_mse,
        top1_accuracy=top1,
        methods=np.asarray(
            ["classical_native_time", "classical_250ms_bins", "flow_native_time"]
        ),
        recordings=np.asarray(recordings),
        requested_interval=np.asarray(visible_cue_interval),
        selected_time_seconds_cue_relative=times[visible_cue_mask],
        bin_edges_seconds_cue_relative=visible_cue_bin_edges,
        bin_centers_seconds_cue_relative=bin_centers_common,
        native_samples_per_bin=bin_counts_common,
    )
    fit_rows = []
    for recording in recordings:
        fit_path = args.output_dir / f"fit_{recording}.json"
        if fit_path.is_file():
            fit_rows.extend(json.loads(fit_path.read_text(encoding="utf-8"))["fits"])
    _atomic_json(
        args.output_dir / "raw_fid_visible_cue_mse_full_query_9recordings_summary.json",
        {
            "experiment": "Nine-recording full-query FID RDM session identification",
            "recordings": recordings,
            "chance_top1": 1.0 / len(recordings),
            "split_design": (
                "pool all clean trials across six runs, then use one seeded class-stratified "
                "mixed half as reference and the complete complementary half as query"
            ),
            "query_subsampling": "none; every saved full-query-half trial is used",
            "distance_representation": "raw squared Gaussian FID",
            "query_reference_distance": (
                "mean squared error over visible-cue times and six unique class pairs"
            ),
            "ranking_rule": "smallest MSE is the top-1 reference",
            "requested_interval_seconds_cue_relative": list(visible_cue_interval),
            "actual_selected_interval_seconds_cue_relative": [
                float(times[visible_cue_mask][0]),
                float(times[visible_cue_mask][-1]),
            ],
            "n_native_time_points": int(np.sum(visible_cue_mask)),
            "n_250ms_bins": int(bin_centers_common.size),
            "native_samples_per_bin": bin_counts_common.astype(int).tolist(),
            "flow_checkpoint_rule": (
                "query and reference RDMs explicitly recomputed from each saved exact "
                "best-validation checkpoint"
            ),
            "epochs": int(args.epochs),
            "early_stopping_patience": int(args.patience),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "metrics": metrics,
            "splits": split_rows,
            "fit_summaries": fit_rows,
        },
    )
    _plot_top1_bar_values(
        args.output_dir,
        top1,
        output_stem="raw_fid_visible_cue_mse_full_query_9recordings_top1_bar",
        title="Full query: nine recordings",
        labels=["Classical\nnative", "Classical\n250 ms", "Flow-based\nnative"],
        colors=[
            METHOD_COLORS["classical_fid"],
            "#EE7733",
            METHOD_COLORS["condition_affine_flow_fid"],
        ],
    )
    print(
        "[result] full-query nine-recording top1 "
        f"classical_native={top1[0]:.3f} classical_250ms={top1[1]:.3f} "
        f"flow_native={top1[2]:.3f}",
        flush=True,
    )
    for method, method_metrics in metrics.items():
        print(
            f"[ranks] {method} {method_metrics['ranks']}",
            flush=True,
        )
    print(f"[experiment] output={args.output_dir.resolve()}", flush=True)


def main() -> None:
    args = parse_args()
    action_count = int(args.aggregate_only) + int(args.fit_recording is not None) + int(
        args.fit_quarter_query_flow
    )
    if action_count > 1:
        raise ValueError(
            "--aggregate-only, --fit-recording, and --fit-quarter-query-flow are mutually "
            "exclusive."
        )
    if len(args.recordings) not in {5, 9} or len(set(args.recordings)) != len(args.recordings):
        raise ValueError("This experiment requires either five or nine unique recordings.")
    missing = [
        str(args.feature_dir / f"{recording}.npz")
        for recording in args.recordings
        if not (args.feature_dir / f"{recording}.npz").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing feature files: {missing}")
    if args.aggregate_only:
        if len(args.recordings) == 9:
            _aggregate_full_query_nine_recordings(args)
        else:
            _aggregate(args)
        return

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA; no CPU fallback is permitted.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device.index} is unavailable.")
    torch.cuda.set_device(0 if device.index is None else device.index)
    torch.set_float32_matmul_precision("high")
    print(
        f"[experiment] device={device} GPU={torch.cuda.get_device_name(device)} "
        f"epochs={args.epochs} patience={args.patience}",
        flush=True,
    )
    if args.fit_recording is not None:
        if args.fit_recording not in args.recordings:
            raise ValueError(f"--fit-recording {args.fit_recording} is not in --recordings.")
        _fit_one_recording(args, args.fit_recording, device)
        return
    if args.fit_quarter_query_flow:
        _fit_quarter_query_flow(args, device)
        return
    for recording in args.recordings:
        _fit_one_recording(args, recording, device)
    if len(args.recordings) == 9:
        _aggregate_full_query_nine_recordings(args)
    else:
        _aggregate(args)


if __name__ == "__main__":
    main()
