#!/usr/bin/env python3
"""Compare classical and no-binning flow temporal-RDM identification."""

from __future__ import annotations

import argparse
import csv
import json
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

from global_setting import (  # noqa: E402
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_TRAINING_MAX_EPOCHS,
)
from fisher.bci_iv_2a_dataset import CLASS_NAMES  # noqa: E402
from fisher.bci_iv_2a_temporal_rdm import (  # noqa: E402
    FlowTemporalRDMConfig,
    TEMPORAL_RDM_METRICS,
    classical_temporal_rdms,
    fit_native_time_affine_flow_rdms,
)
from fisher.bci_iv_2a_temporal_rdm_identification import (  # noqa: E402
    INTERVALS,
    METHODS,
    QUERY_RUNS,
    REFERENCE_RUNS,
    correct_match_margins,
    correct_match_ranks,
    exact_sign_flip_paired,
    load_native_class_voltage,
    native_evaluation_indices,
    select_runs,
    shuffled_half_split_indices,
    temporal_rdm_score_matrix,
)


METHOD_LABELS = {"classical": "Classical", "flow": "Flow"}
METRIC_LABELS = {
    "correlation": "Correlation",
    "cosine": "Cosine",
    "euclidean": "Euclidean",
    "fid": r"Gaussian $W_2$",
}
COLORS = {"classical": "#4C78A8", "flow": "#E45756"}


def _json_safe(value: Any) -> Any:
    """Convert NumPy-heavy training metadata into JSON-native values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _default_data_dir() -> Path:
    local = ROOT / "data/bci_iv_2a/raw/gdf"
    sibling = ROOT.parent / "eeg-session-identification/data/bci_iv_2a/raw/gdf"
    return local if local.exists() else sibling


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=_default_data_dir())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/temporal_rdm_session_identification_5recordings_left_hand",
    )
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=["A01T", "A02T", "A03T", "A04T", "A05T"],
    )
    parser.add_argument("--class-name", choices=CLASS_NAMES, default="left_hand")
    parser.add_argument("--tmin", type=float, default=-2.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--evaluation-step", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--split-seed", type=int, default=20260716)
    parser.add_argument(
        "--split-mode",
        choices=("run_disjoint", "shuffled_half"),
        default="run_disjoint",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=TEMPORAL_RDM_METRICS,
        default=list(TEMPORAL_RDM_METRICS),
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--covariance-steps", type=int, default=48)
    parser.add_argument("--covariance-ridge", type=float, default=1e-5)
    parser.add_argument("--fid-block-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", required=True)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> torch.device:
    if len(args.recordings) != 5 or len(set(args.recordings)) != 5:
        raise ValueError("This speed pilot requires exactly five distinct recordings.")
    device = torch.device(str(args.device))
    if device.type != "cuda" or device.index is None:
        raise ValueError("--device must explicitly select a CUDA device, for example cuda:1.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to silently switch to CPU.")
    if int(device.index) >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {device.index} does not exist.")
    for recording in args.recordings:
        path = args.data_dir / f"{recording}.gdf"
        if not path.is_file():
            raise FileNotFoundError(path)
    return device


def _save_rdm_cache(
    path: Path,
    *,
    rdms: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"{metric}_rdm": rdms[metric] for metric in TEMPORAL_RDM_METRICS})
    path.with_suffix(".json").write_text(
        json.dumps(_json_safe(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_rdm_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        rdms = {metric: np.asarray(payload[f"{metric}_rdm"], dtype=np.float64) for metric in TEMPORAL_RDM_METRICS}
    return rdms


def _fit_or_load_half(
    *,
    recording_key: str,
    half_name: str,
    samples: np.ndarray,
    native_times: np.ndarray,
    evaluation_indices: np.ndarray,
    output_dir: Path,
    device: torch.device,
    seed: int,
    config: FlowTemporalRDMConfig,
    overwrite: bool,
    source_trial_indices: np.ndarray,
    source_run_ids: np.ndarray,
    split_mode: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    cache_dir = output_dir / "rdm_cache"
    results: dict[str, dict[str, np.ndarray]] = {}
    summaries: dict[str, Any] = {}
    evaluation_times = native_times[evaluation_indices]
    for method in METHODS:
        cache_path = cache_dir / f"{recording_key}_{half_name}_{method}.npz"
        if cache_path.exists() and not bool(overwrite):
            cached_metadata = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
            expected_trials = np.asarray(source_trial_indices, dtype=np.int64).tolist()
            expected_runs = np.asarray(source_run_ids, dtype=np.int64).tolist()
            if (
                cached_metadata.get("source_trial_indices_zero_based") != expected_trials
                or cached_metadata.get("source_run_ids_zero_based") != expected_runs
                or cached_metadata.get("split_mode") != str(split_mode)
                or int(cached_metadata.get("seed", -1)) != int(seed)
            ):
                raise ValueError(
                    f"Incompatible cache metadata in {cache_path}; use --overwrite or a new output directory."
                )
            results[method] = _load_rdm_cache(cache_path)
            summaries[method] = cached_metadata
            print(f"[temporal-id] cache {recording_key} {half_name} {method}", flush=True)
            continue
        started = time.monotonic()
        if method == "classical":
            fitted = classical_temporal_rdms(samples[:, evaluation_indices, :])
            rdms = fitted.rdms
            method_metadata: dict[str, Any] = {
                "estimator": "independent-time Ledoit-Wolf Gaussians",
                "temporal_averaging": "none",
                "flow_training": None,
            }
        else:
            fitted, model = fit_native_time_affine_flow_rdms(
                samples,
                native_times,
                evaluation_time_points=evaluation_times,
                device=device,
                seed=int(seed),
                config=config,
            )
            rdms = fitted.rdms
            checkpoint_path = cache_dir / f"{recording_key}_{half_name}_flow_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "flow_config": asdict(config),
                    "seed": int(seed),
                    "recording": recording_key,
                    "half": half_name,
                    "evaluation_times_seconds": evaluation_times,
                },
                checkpoint_path,
            )
            method_metadata = {
                "estimator": "continuous-time centered covariate-affine Gaussian flow",
                "temporal_averaging": "none",
                "flow_training": fitted.train_metadata,
                "train_trial_indices_within_half": fitted.train_trial_indices.tolist(),
                "validation_trial_indices_within_half": fitted.validation_trial_indices.tolist(),
                "checkpoint": str(checkpoint_path.resolve()),
            }
        duration = float(time.monotonic() - started)
        metadata = {
            "recording": recording_key,
            "half": half_name,
            "method": method,
            "n_trials": int(samples.shape[0]),
            "n_native_times": int(samples.shape[1]),
            "n_eeg_channels": int(samples.shape[2]),
            "n_evaluation_times": int(evaluation_indices.size),
            "evaluation_times_seconds": evaluation_times.tolist(),
            "duration_seconds": duration,
            "seed": int(seed),
            "split_mode": str(split_mode),
            "source_trial_indices_zero_based": np.asarray(source_trial_indices, dtype=np.int64).tolist(),
            "source_trial_indices_one_based": (
                np.asarray(source_trial_indices, dtype=np.int64) + 1
            ).tolist(),
            "source_run_ids_zero_based": np.asarray(source_run_ids, dtype=np.int64).tolist(),
            "source_run_ids_one_based": (
                np.asarray(source_run_ids, dtype=np.int64) + 1
            ).tolist(),
            **method_metadata,
        }
        _save_rdm_cache(cache_path, rdms=rdms, metadata=metadata)
        results[method] = rdms
        summaries[method] = metadata
        print(
            f"[temporal-id] fitted {recording_key} {half_name} {method} "
            f"trials={samples.shape[0]} duration={duration:.1f}s",
            flush=True,
        )
    return results, summaries


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )


def _style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
    ax.tick_params(width=1.8)


def _plot_score_heatmaps(
    output_dir: Path,
    scores: dict[str, dict[str, dict[str, np.ndarray]]],
    recording_keys: list[str],
    metrics: tuple[str, ...],
) -> tuple[Path, Path]:
    _set_plot_style()
    full = scores["full"]
    all_values = np.concatenate(
        [full[method][metric].reshape(-1) for method in METHODS for metric in metrics]
    )
    vmin = float(np.min(all_values))
    vmax = float(np.max(all_values))
    if np.isclose(vmin, vmax):
        vmin, vmax = -1.0, 1.0
    fig, axes = plt.subplots(
        2,
        len(metrics),
        figsize=(4.0 * len(metrics), 7.0),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for row, method in enumerate(METHODS):
        for column, metric in enumerate(metrics):
            ax = axes[row, column]
            matrix = full[method][metric]
            image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="viridis", aspect="equal")
            ax.set_title(METRIC_LABELS[metric])
            ax.set_xticks(np.arange(len(recording_keys)), recording_keys, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(recording_keys)), recording_keys)
            if row == 1:
                ax.set_xlabel("Reference")
            if column == 0:
                ax.set_ylabel(f"{METHOD_LABELS[method]} query")
            for query in range(len(recording_keys)):
                for reference in range(len(recording_keys)):
                    color = "white" if matrix[query, reference] < (vmin + vmax) / 2.0 else "black"
                    ax.text(reference, query, f"{matrix[query, reference]:.2f}", ha="center", va="center", fontsize=8, color=color)
            _style_axes(ax)
    if image is None:
        raise RuntimeError("No score heat map was created.")
    fig.colorbar(image, ax=axes, label="RDM pattern correlation", shrink=0.82)
    png = output_dir / "temporal_rdm_identification_score_heatmaps.png"
    svg = output_dir / "temporal_rdm_identification_score_heatmaps.svg"
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def _plot_performance_summary(
    output_dir: Path,
    summary: dict[str, Any],
    metrics: tuple[str, ...],
) -> tuple[Path, Path]:
    _set_plot_style()
    x = np.arange(len(metrics), dtype=np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), constrained_layout=True)
    for method in METHODS:
        top1 = [summary["intervals"]["full"][method][metric]["top1_accuracy"] for metric in metrics]
        top2 = [summary["intervals"]["full"][method][metric]["top2_accuracy"] for metric in metrics]
        mean_rank = [summary["intervals"]["full"][method][metric]["mean_correct_rank"] for metric in metrics]
        axes[0].plot(x, top1, marker="o", linewidth=1.8, color=COLORS[method], label=METHOD_LABELS[method])
        axes[1].plot(x, top2, marker="o", linewidth=1.8, color=COLORS[method], label=METHOD_LABELS[method])
        axes[2].plot(x, mean_rank, marker="o", linewidth=1.8, color=COLORS[method], label=METHOD_LABELS[method])
    axes[0].axhline(summary["chance_top1"], linestyle="--", linewidth=1.5, color="#555555")
    axes[1].axhline(summary["chance_top2"], linestyle="--", linewidth=1.5, color="#555555")
    axes[2].axhline(summary["chance_mean_rank"], linestyle="--", linewidth=1.5, color="#555555")
    axes[0].set_ylabel("Top-1 accuracy")
    axes[1].set_ylabel("Top-2 accuracy")
    axes[2].set_ylabel("Mean correct rank")
    for ax in axes:
        ax.set_xticks(x, [METRIC_LABELS[metric] for metric in metrics], rotation=30, ha="right")
        ax.set_xlabel("Temporal-RDM distance")
        _style_axes(ax)
    axes[0].legend(frameon=False)
    axes[0].set_ylim(0.0, 1.02)
    axes[1].set_ylim(0.0, 1.02)
    axes[2].set_ylim(1.0, float(len(summary["recordings"])))
    png = output_dir / "temporal_rdm_identification_performance.png"
    svg = output_dir / "temporal_rdm_identification_performance.svg"
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def _build_summary(
    *,
    args: argparse.Namespace,
    evaluation_times: np.ndarray,
    trial_counts: dict[str, dict[str, int]],
    split_metadata: dict[str, dict[str, Any]],
    fit_summaries: dict[str, dict[str, dict[str, Any]]],
    scores: dict[str, dict[str, dict[str, np.ndarray]]],
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "experiment": "five-recording one-class temporal-RDM identification",
        "recordings": list(args.recordings),
        "class_name": str(args.class_name),
        "chance_top1": 1.0 / len(args.recordings),
        "chance_top2": min(2.0 / len(args.recordings), 1.0),
        "chance_mean_rank": (len(args.recordings) + 1.0) / 2.0,
        "split_mode": str(args.split_mode),
        "split_seed": int(args.split_seed),
        "query_runs_one_based": (
            [value + 1 for value in QUERY_RUNS] if args.split_mode == "run_disjoint" else None
        ),
        "reference_runs_one_based": (
            [value + 1 for value in REFERENCE_RUNS]
            if args.split_mode == "run_disjoint"
            else None
        ),
        "metrics": list(metrics),
        "time_range_seconds_cue_relative": [float(args.tmin), float(args.tmax)],
        "evaluation_step_seconds": float(args.evaluation_step),
        "evaluation_times_seconds": evaluation_times.tolist(),
        "n_evaluation_times": int(evaluation_times.size),
        "fid_matching_transform": "square root (Gaussian 2-Wasserstein distance)",
        "matching_similarity": "Pearson correlation of strict upper-triangle RDM vectors",
        "trial_counts": trial_counts,
        "split_metadata": split_metadata,
        "flow_config": {
            "hidden_dim": int(args.hidden_dim),
            "depth": int(args.depth),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "patience": int(args.patience),
            "validation_fraction": float(args.validation_fraction),
            "covariance_steps": int(args.covariance_steps),
            "covariance_ridge": float(args.covariance_ridge),
        },
        "fit_summaries": fit_summaries,
        "intervals": {},
        "paired_flow_minus_classical": {},
    }
    for interval_name, method_scores in scores.items():
        summary["intervals"][interval_name] = {}
        for method in METHODS:
            summary["intervals"][interval_name][method] = {}
            for metric in metrics:
                matrix = method_scores[method][metric]
                ranks = correct_match_ranks(matrix)
                margins = correct_match_margins(matrix)
                summary["intervals"][interval_name][method][metric] = {
                    "top1_accuracy": float(np.mean(ranks == 1)),
                    "n_correct": int(np.sum(ranks == 1)),
                    "top2_accuracy": float(np.mean(ranks <= 2)),
                    "n_top2_correct": int(np.sum(ranks <= 2)),
                    "correct_ranks": ranks.tolist(),
                    "mean_correct_rank": float(np.mean(ranks)),
                    "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
                    "correct_match_margins": margins.tolist(),
                    "mean_correct_match_margin": float(np.mean(margins)),
                }
        summary["paired_flow_minus_classical"][interval_name] = {}
        for metric in metrics:
            classical_matrix = method_scores["classical"][metric]
            flow_matrix = method_scores["flow"][metric]
            classical_top1 = (correct_match_ranks(classical_matrix) == 1).astype(np.float64)
            flow_top1 = (correct_match_ranks(flow_matrix) == 1).astype(np.float64)
            classical_top2 = (correct_match_ranks(classical_matrix) <= 2).astype(np.float64)
            flow_top2 = (correct_match_ranks(flow_matrix) <= 2).astype(np.float64)
            margin_difference = correct_match_margins(flow_matrix) - correct_match_margins(classical_matrix)
            summary["paired_flow_minus_classical"][interval_name][metric] = {
                "top1_accuracy_difference": float(np.mean(flow_top1 - classical_top1)),
                "top1_exact_sign_flip_p": exact_sign_flip_paired(flow_top1 - classical_top1),
                "top2_accuracy_difference": float(np.mean(flow_top2 - classical_top2)),
                "top2_exact_sign_flip_p": exact_sign_flip_paired(flow_top2 - classical_top2),
                "mean_margin_difference": float(np.mean(margin_difference)),
                "margin_exact_sign_flip_p": exact_sign_flip_paired(margin_difference),
            }
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = _validate_args(args)
    metrics = tuple(dict.fromkeys(str(metric) for metric in args.metrics))
    if not metrics:
        raise ValueError("At least one temporal-RDM metric is required.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = FlowTemporalRDMConfig(
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        patience=int(args.patience),
        validation_fraction=float(args.validation_fraction),
        covariance_steps=int(args.covariance_steps),
        covariance_ridge=float(args.covariance_ridge),
        fid_block_size=int(args.fid_block_size),
    )

    bank: dict[str, dict[str, list[dict[str, np.ndarray]]]] = {
        method: {"query": [], "reference": []} for method in METHODS
    }
    trial_counts: dict[str, dict[str, int]] = {}
    split_metadata: dict[str, dict[str, Any]] = {}
    fit_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    evaluation_times: np.ndarray | None = None
    for recording_index, recording_key in enumerate(args.recordings):
        recording = args.data_dir / f"{recording_key}.gdf"
        loaded = load_native_class_voltage(
            recording,
            class_name=str(args.class_name),
            tmin=float(args.tmin),
            tmax=float(args.tmax),
        )
        evaluation_indices = native_evaluation_indices(
            loaded.time_points_seconds,
            step_seconds=float(args.evaluation_step),
        )
        this_evaluation_times = loaded.time_points_seconds[evaluation_indices]
        if evaluation_times is None:
            evaluation_times = this_evaluation_times
        else:
            np.testing.assert_allclose(evaluation_times, this_evaluation_times, atol=1e-12, rtol=0.0)
        if args.split_mode == "run_disjoint":
            query_indices = np.flatnonzero(np.isin(loaded.run_ids, QUERY_RUNS))
            reference_indices = np.flatnonzero(np.isin(loaded.run_ids, REFERENCE_RUNS))
        else:
            recording_split_seed = int(args.split_seed) + recording_index
            query_indices, reference_indices = shuffled_half_split_indices(
                loaded.voltage_microvolts.shape[0],
                seed=recording_split_seed,
            )
        query = loaded.voltage_microvolts[query_indices]
        reference = loaded.voltage_microvolts[reference_indices]
        if query.shape[0] + reference.shape[0] != loaded.voltage_microvolts.shape[0]:
            raise RuntimeError(f"{recording_key}: split does not use every clean trial exactly once.")
        trial_counts[recording_key] = {"query": int(query.shape[0]), "reference": int(reference.shape[0])}
        split_metadata[recording_key] = {
            "recording_split_seed": (
                int(args.split_seed) + recording_index
                if args.split_mode == "shuffled_half"
                else None
            ),
            "n_all_clean_class_trials": int(loaded.voltage_microvolts.shape[0]),
            "all_source_trial_indices_zero_based": loaded.trial_indices.tolist(),
            "query_source_trial_indices_zero_based": loaded.trial_indices[query_indices].tolist(),
            "reference_source_trial_indices_zero_based": loaded.trial_indices[reference_indices].tolist(),
            "query_run_ids_one_based": (loaded.run_ids[query_indices] + 1).tolist(),
            "reference_run_ids_one_based": (loaded.run_ids[reference_indices] + 1).tolist(),
            "query_counts_by_run": np.bincount(
                loaded.run_ids[query_indices], minlength=6
            ).astype(int).tolist(),
            "reference_counts_by_run": np.bincount(
                loaded.run_ids[reference_indices], minlength=6
            ).astype(int).tolist(),
        }
        fit_summaries[recording_key] = {}
        for half_index, (half_name, samples, selected_indices) in enumerate(
            (
                ("query", query, query_indices),
                ("reference", reference, reference_indices),
            )
        ):
            half_seed = int(args.seed) + 100 * recording_index + half_index
            fitted, metadata = _fit_or_load_half(
                recording_key=recording_key,
                half_name=half_name,
                samples=samples,
                native_times=loaded.time_points_seconds,
                evaluation_indices=evaluation_indices,
                output_dir=args.output_dir,
                device=device,
                seed=half_seed,
                config=config,
                overwrite=bool(args.overwrite),
                source_trial_indices=loaded.trial_indices[selected_indices],
                source_run_ids=loaded.run_ids[selected_indices],
                split_mode=str(args.split_mode),
            )
            fit_summaries[recording_key][half_name] = metadata
            for method in METHODS:
                bank[method][half_name].append(fitted[method])

    if evaluation_times is None:
        raise RuntimeError("No recordings were loaded.")
    scores: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for interval_name, interval in INTERVALS.items():
        scores[interval_name] = {method: {} for method in METHODS}
        for method in METHODS:
            for metric in metrics:
                scores[interval_name][method][metric] = temporal_rdm_score_matrix(
                    bank[method]["query"],
                    bank[method]["reference"],
                    evaluation_times,
                    metric=metric,
                    interval=interval,
                )

    summary = _build_summary(
        args=args,
        evaluation_times=evaluation_times,
        trial_counts=trial_counts,
        split_metadata=split_metadata,
        fit_summaries=fit_summaries,
        scores=scores,
        metrics=metrics,
    )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    npz_payload: dict[str, np.ndarray] = {"evaluation_times_seconds": evaluation_times}
    for interval_name, method_scores in scores.items():
        for method in METHODS:
            for metric in metrics:
                npz_payload[f"{interval_name}_{method}_{metric}_scores"] = method_scores[method][metric]
    np.savez_compressed(args.output_dir / "results.npz", **npz_payload)

    rows: list[dict[str, Any]] = []
    for interval_name, method_scores in scores.items():
        for method in METHODS:
            for metric in metrics:
                matrix = method_scores[method][metric]
                ranks = correct_match_ranks(matrix)
                margins = correct_match_margins(matrix)
                for query_index, query_key in enumerate(args.recordings):
                    for reference_index, reference_key in enumerate(args.recordings):
                        rows.append(
                            {
                                "interval": interval_name,
                                "method": method,
                                "metric": metric,
                                "query_recording": query_key,
                                "reference_recording": reference_key,
                                "similarity": float(matrix[query_index, reference_index]),
                                "correct_rank": int(ranks[query_index]),
                                "correct_margin": float(margins[query_index]),
                            }
                        )
    with (args.output_dir / "pair_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    generated = [
        *_plot_score_heatmaps(args.output_dir, scores, list(args.recordings), metrics),
        *_plot_performance_summary(args.output_dir, summary, metrics),
    ]
    print(
        f"[temporal-id] chance_top1={summary['chance_top1']:.3f} "
        f"chance_top2={summary['chance_top2']:.3f}",
        flush=True,
    )
    for method in METHODS:
        for metric in metrics:
            values = summary["intervals"]["full"][method][metric]
            print(
                f"[temporal-id] {method} {metric}: "
                f"top1={values['n_correct']}/5 top2={values['n_top2_correct']}/5 "
                f"mean_rank={values['mean_correct_rank']:.3f} "
                f"margin={values['mean_correct_match_margin']:.6f}",
                flush=True,
            )
    for path in (summary_path, args.output_dir / "results.npz", args.output_dir / "pair_scores.csv", *generated):
        print(f"[temporal-id] Saved: {path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
