#!/usr/bin/env python3
"""Run five-recording BCI IV-2a identification with Euclidean RDMs."""

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
    QUERY_RUNS,
    RDM_MATCHING_INTERVAL,
    REFERENCE_RUNS,
    classical_squared_euclidean_rdms,
    per_class_counts,
    select_half,
    subsample_balanced_trials,
    translation_flow_squared_euclidean_rdms,
)
from fisher.rdm_lagged_matching import (  # noqa: E402
    lagged_pearson_similarity,
    rdm_upper_triangle_sequence,
)


METHODS = ("classical_squared_euclidean", "translation_flow")
METHOD_LABELS = {
    "classical_squared_euclidean": "Classical",
    "translation_flow": "Translation flow",
}
METHOD_COLORS = {
    "classical_squared_euclidean": "#4477AA",
    "translation_flow": "#CC6677",
}
PRE_CUE_INTERVAL = (-1.5, -0.5)


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
        default=ROOT / "data/bci_iv_2a/euclidean_session_identification_5recordings",
    )
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=["A01T", "A02T", "A03T", "A04T", "A05T"],
    )
    parser.add_argument(
        "--reuse-a01-reference-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/reference_euclidean_A01T",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--epochs", type=int, default=TRAINING_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument(
        "--max-lag-ms",
        type=float,
        nargs="+",
        default=[0.0, 25.0, 50.0, 100.0, 200.0],
    )
    return parser.parse_args()


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(width=1.8)


def _save_cache(path: Path, rdms: np.ndarray, metadata: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            rdms=np.asarray(rdms, dtype=np.float64),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(path)


def _load_cache(path: Path) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as archive:
        rdms = np.asarray(archive["rdms"], dtype=np.float64)
        metadata = json.loads(str(archive["metadata_json"].item()))
    if rdms.ndim != 3 or rdms.shape[1:] != (4, 4) or not np.all(np.isfinite(rdms)):
        raise ValueError(f"Invalid RDM cache {path}: shape={rdms.shape}.")
    return rdms, metadata


def _reuse_a01_reference(
    source_dir: Path,
    cache_dir: Path,
    times: np.ndarray,
) -> dict[str, tuple[np.ndarray, dict]]:
    components_path = source_dir / "reference_euclidean_components.npz"
    summary_path = source_dir / "summary.json"
    if not components_path.is_file() or not summary_path.is_file():
        return {}
    with np.load(components_path, allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["time_seconds_cue_relative"], times)
        classical = np.asarray(archive["classical_rdms"], dtype=np.float64)
        flow = np.asarray(archive["flow_rdms"], dtype=np.float64)
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    reused = {
        "classical_squared_euclidean": (
            classical,
            {
                "reused_from": str(source_dir.resolve()),
                "estimator": "sample_mean_within_each_condition_and_raw_time_sample",
            },
        ),
        "translation_flow": (
            flow,
            {
                **source_summary["flow_training"],
                "elapsed_seconds": float(source_summary["flow_fit_elapsed_seconds"]),
                "reused_from": str(source_dir.resolve()),
            },
        ),
    }
    for method, (rdms, metadata) in reused.items():
        _save_cache(cache_dir / f"reference_A01T_{method}.npz", rdms, metadata)
    return reused


def _fit_or_load(
    cache_path: Path,
    *,
    method: str,
    x: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    device: torch.device,
    seed: int,
    config: FlowRDMConfig,
    context: dict,
) -> tuple[np.ndarray, dict]:
    if cache_path.is_file():
        rdms, metadata = _load_cache(cache_path)
        if rdms.shape[0] != times.size:
            raise ValueError(f"Time dimension mismatch in {cache_path}.")
        print(f"[cache] loaded {cache_path.name}", flush=True)
        return rdms, metadata

    start = time.perf_counter()
    if method == "classical_squared_euclidean":
        rdms = classical_squared_euclidean_rdms(
            x,
            labels,
            standardize_features=False,
        )
        metadata = {
            "estimator": "sample_mean_within_each_condition_and_raw_time_sample",
        }
    elif method == "translation_flow":
        rdms, metadata = translation_flow_squared_euclidean_rdms(
            x,
            labels,
            times,
            device=device,
            seed=seed,
            config=config,
        )
    else:
        raise ValueError(f"Unknown method {method!r}.")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    metadata = {
        **metadata,
        **context,
        "n_trials": int(x.shape[0]),
        "per_class_counts": per_class_counts(labels).astype(int).tolist(),
        "elapsed_seconds": float(elapsed),
    }
    _save_cache(cache_path, rdms, metadata)
    print(
        f"[fit] {cache_path.name} elapsed={elapsed / 60.0:.2f} min",
        flush=True,
    )
    return rdms, metadata


def _rank_metrics(scores: np.ndarray, recordings: list[str]) -> dict:
    n_recordings = len(recordings)
    ranks = np.empty(n_recordings, dtype=np.int64)
    predictions = np.empty(n_recordings, dtype=np.int64)
    margins = np.empty(n_recordings, dtype=np.float64)
    for query_index in range(n_recordings):
        order = np.argsort(-scores[query_index], kind="mergesort")
        ranks[query_index] = int(np.flatnonzero(order == query_index)[0]) + 1
        predictions[query_index] = int(order[0])
        margins[query_index] = float(
            scores[query_index, query_index]
            - np.max(np.delete(scores[query_index], query_index))
        )
    return {
        "top1_accuracy": float(np.mean(ranks == 1)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "mean_true_minus_best_competitor_margin": float(np.mean(margins)),
        "ranks": ranks.astype(int).tolist(),
        "predicted_recordings": [recordings[index] for index in predictions],
        "correct_recordings": [
            recordings[index] for index in range(n_recordings) if ranks[index] == 1
        ],
    }


def _plot_results(
    output_dir: Path,
    scores: np.ndarray,
    actual_lag_ms: np.ndarray,
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
    figure.patch.set_facecolor("white")
    for axis in axes:
        axis.set_facecolor("white")
    images = []
    for method_index, (method, axis) in enumerate(zip(METHODS, axes[:2], strict=True)):
        matrix = scores[method_index, 0]
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
        axis.set_xlabel("Reference recording")
        if method_index == 0:
            axis.set_ylabel("Query recording")
        axis.set_title(f"{METHOD_LABELS[method]}: zero lag")
        _style_axis(axis)
    colorbar = figure.colorbar(images[0], ax=axes[:2], location="bottom", shrink=0.82)
    colorbar.set_label("Pearson correlation")

    for method in METHODS:
        top1 = [
            metrics[method][str(index)]["top1_accuracy"]
            for index in range(actual_lag_ms.size)
        ]
        axes[2].plot(
            actual_lag_ms,
            top1,
            marker="o",
            linewidth=2.0,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes[2].axhline(
        1.0 / len(recordings),
        color="0.35",
        linestyle="--",
        linewidth=1.5,
        label="Chance",
    )
    axes[2].set_xlabel("Maximum lag (ms)")
    axes[2].set_ylabel("Top-1 accuracy")
    axes[2].set_ylim(0.0, 1.04)
    axes[2].set_title("Lag sensitivity")
    axes[2].legend(frameon=False, loc="best")
    _style_axis(axes[2])
    figure.savefig(
        output_dir / "euclidean_session_identification.png",
        dpi=300,
        facecolor="white",
        transparent=False,
    )
    figure.savefig(
        output_dir / "euclidean_session_identification.svg",
        facecolor="white",
        transparent=False,
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA; no CPU fallback is permitted.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device.index} is unavailable.")
    torch.cuda.set_device(0 if device.index is None else device.index)
    if len(args.recordings) != 5 or len(set(args.recordings)) != 5:
        raise ValueError("This speed-limited experiment requires five unique recordings.")
    feature_paths = [args.feature_dir / f"{recording}.npz" for recording in args.recordings]
    missing = [str(path) for path in feature_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing feature files: {missing}")
    datasets = [load_features_npz(path) for path in feature_paths]
    recordings = [dataset.session_key for dataset in datasets]
    times = np.asarray(datasets[0].time_centers, dtype=np.float64)
    if any(not np.array_equal(dataset.time_centers, times) for dataset in datasets[1:]):
        raise ValueError("Feature files do not share the same time grid.")
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "rdm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[experiment] recordings={recordings} device={device} "
        f"GPU={torch.cuda.get_device_name(device)}",
        flush=True,
    )
    print(
        f"[experiment] interval={RDM_MATCHING_INTERVAL} all-query only "
        f"epochs={config.epochs} patience={config.patience}",
        flush=True,
    )

    cached: dict[tuple[str, str, str], tuple[np.ndarray, dict]] = {}
    if args.reuse_a01_reference_dir is not None and "A01T" in recordings:
        for method, result in _reuse_a01_reference(
            args.reuse_a01_reference_dir,
            cache_dir,
            times,
        ).items():
            cached[(method, "reference", "A01T")] = result
        if cached:
            print("[reuse] imported A01T reference Euclidean fit", flush=True)

    total_start = time.perf_counter()
    query_n_per_class: dict[str, int] = {}
    fit_summaries: list[dict] = []
    for subject_index, dataset in enumerate(datasets):
        x_reference, y_reference, _ = select_half(dataset, REFERENCE_RUNS)
        x_query, y_query, _ = select_half(dataset, QUERY_RUNS)
        n_per_class = int(np.min(per_class_counts(y_query)))
        selected = subsample_balanced_trials(
            y_query,
            n_per_class,
            args.seed + subject_index * 100_000,
        )
        x_query = x_query[selected]
        y_query = y_query[selected]
        query_n_per_class[dataset.session_key] = n_per_class
        role_data = {
            "reference": (x_reference, y_reference),
            "query": (x_query, y_query),
        }
        for role, (x_role, y_role) in role_data.items():
            for method in METHODS:
                key = (method, role, dataset.session_key)
                if key in cached:
                    rdms, metadata = cached[key]
                else:
                    role_seed = (
                        args.seed
                        + (1_000_002 if role == "reference" else 20_000_000)
                        + subject_index * (10 if role == "reference" else 100_000)
                    )
                    rdms, metadata = _fit_or_load(
                        cache_dir / f"{role}_{dataset.session_key}_{method}.npz",
                        method=method,
                        x=x_role,
                        labels=y_role,
                        times=times,
                        device=device,
                        seed=role_seed,
                        config=config,
                        context={
                            "recording": dataset.session_key,
                            "role": role,
                            "seed": int(role_seed),
                        },
                    )
                    cached[key] = (rdms, metadata)
                fit_summaries.append(
                    {
                        "recording": dataset.session_key,
                        "role": role,
                        "method": method,
                        "n_trials": int(x_role.shape[0]),
                        "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0)),
                        "best_epoch": metadata.get("best_epoch"),
                        "stopped_epoch": metadata.get("stopped_epoch"),
                        "best_val_loss": metadata.get("best_val_loss"),
                        "reused": "reused_from" in metadata,
                    }
                )
        print(f"[recording] {dataset.session_key} fits complete", flush=True)

    sample_period_seconds = float(np.median(np.diff(times)))
    requested_lag_ms = sorted({0.0, *[float(value) for value in args.max_lag_ms]})
    lag_samples = np.asarray(
        [int(round(value / (sample_period_seconds * 1_000.0))) for value in requested_lag_ms],
        dtype=np.int64,
    )
    keep = np.concatenate(([True], np.diff(lag_samples) != 0))
    lag_samples = lag_samples[keep]
    requested_lag_ms = [
        value for value, selected_value in zip(requested_lag_ms, keep, strict=True) if selected_value
    ]
    actual_lag_ms = lag_samples.astype(np.float64) * sample_period_seconds * 1_000.0
    sequences: dict[tuple[str, str, str], np.ndarray] = {}
    pre_sequences: dict[tuple[str, str, str], np.ndarray] = {}
    for method in METHODS:
        for role in ("reference", "query"):
            for recording in recordings:
                # The fitting helpers cache squared distances because those are
                # their natural moment outputs.  Session identification uses
                # the literal Euclidean norm requested by the experiment.
                squared_rdms = cached[(method, role, recording)][0]
                if np.any(squared_rdms < -1e-12):
                    raise ValueError(
                        f"Squared RDM for {method}/{role}/{recording} contains "
                        "materially negative entries."
                    )
                rdms = np.sqrt(np.maximum(squared_rdms, 0.0))
                sequences[(method, role, recording)] = rdm_upper_triangle_sequence(
                    rdms,
                    times,
                    interval=RDM_MATCHING_INTERVAL,
                )[0]
                pre_sequences[(method, role, recording)] = rdm_upper_triangle_sequence(
                    rdms,
                    times,
                    interval=PRE_CUE_INTERVAL,
                )[0]

    scores = np.empty(
        (len(METHODS), lag_samples.size, len(recordings), len(recordings)),
        dtype=np.float64,
    )
    best_lags = np.empty_like(scores, dtype=np.int64)
    pre_scores = np.empty((len(METHODS), len(recordings), len(recordings)), dtype=np.float64)
    for method_index, method in enumerate(METHODS):
        for query_index, query_recording in enumerate(recordings):
            query = sequences[(method, "query", query_recording)]
            pre_query = pre_sequences[(method, "query", query_recording)]
            for reference_index, reference_recording in enumerate(recordings):
                reference = sequences[(method, "reference", reference_recording)]
                pre_reference = pre_sequences[(method, "reference", reference_recording)]
                pre_scores[method_index, query_index, reference_index] = (
                    lagged_pearson_similarity(
                        pre_query,
                        pre_reference,
                        max_lag_samples=0,
                    ).score
                )
                for lag_index, lag_limit in enumerate(lag_samples):
                    result = lagged_pearson_similarity(
                        query,
                        reference,
                        max_lag_samples=int(lag_limit),
                    )
                    scores[method_index, lag_index, query_index, reference_index] = result.score
                    best_lags[method_index, lag_index, query_index, reference_index] = result.lag_samples

    metrics: dict = {}
    for method_index, method in enumerate(METHODS):
        metrics[method] = {}
        for lag_index, lag_ms in enumerate(actual_lag_ms):
            values = _rank_metrics(scores[method_index, lag_index], recordings)
            values["actual_max_lag_ms"] = float(lag_ms)
            values["true_pair_reference_minus_query_lag_ms"] = (
                best_lags[
                    method_index,
                    lag_index,
                    np.arange(len(recordings)),
                    np.arange(len(recordings)),
                ].astype(np.float64)
                * sample_period_seconds
                * 1_000.0
            ).tolist()
            metrics[method][str(lag_index)] = values
        metrics[method]["pre_cue_zero_lag"] = _rank_metrics(
            pre_scores[method_index],
            recordings,
        )

    aggregation_wall_seconds = time.perf_counter() - total_start
    fit_elapsed_seconds = float(
        sum(
            row["elapsed_seconds"]
            for row in fit_summaries
            if not row["reused"]
        )
    )
    np.savez_compressed(
        args.output_dir / "euclidean_session_identification_results.npz",
        scores=scores,
        pre_cue_scores=pre_scores,
        best_lag_samples=best_lags,
        lag_limits_samples=lag_samples,
        actual_lag_limits_ms=actual_lag_ms,
        recordings=np.asarray(recordings),
        methods=np.asarray(METHODS),
        interval=np.asarray(RDM_MATCHING_INTERVAL),
    )
    with (args.output_dir / "fit_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fit_summaries[0]))
        writer.writeheader()
        writer.writerows(fit_summaries)
    score_rows = []
    for method_index, method in enumerate(METHODS):
        for lag_index, lag_ms in enumerate(actual_lag_ms):
            ranks = metrics[method][str(lag_index)]["ranks"]
            for query_index, query_recording in enumerate(recordings):
                for reference_index, reference_recording in enumerate(recordings):
                    score_rows.append(
                        {
                            "method": method,
                            "actual_max_lag_ms": float(lag_ms),
                            "query_recording": query_recording,
                            "reference_recording": reference_recording,
                            "is_true_pair": query_index == reference_index,
                            "correlation": float(scores[method_index, lag_index, query_index, reference_index]),
                            "best_lag_ms": float(
                                best_lags[method_index, lag_index, query_index, reference_index]
                                * sample_period_seconds
                                * 1_000.0
                            ),
                            "true_recording_rank": int(ranks[query_index]),
                        }
                    )
    with (args.output_dir / "pair_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)

    summary = {
        "experiment": "BCI IV-2a five-recording Euclidean RDM identification",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "recordings": recordings,
        "methods": list(METHODS),
        "chance_top1": 1.0 / len(recordings),
        "query_runs_one_based": [1, 3, 5],
        "reference_runs_one_based": [2, 4, 6],
        "query_design": "largest balanced all-trial subset; one query per recording",
        "query_trials_per_class": query_n_per_class,
        "time_interval_seconds_cue_relative": list(RDM_MATCHING_INTERVAL),
        "pre_cue_control_interval_seconds": list(PRE_CUE_INTERVAL),
        "sample_period_seconds": sample_period_seconds,
        "requested_lag_limits_ms": requested_lag_ms,
        "actual_lag_limits_ms": actual_lag_ms.tolist(),
        "distance": "euclidean_norm_between_condition_means",
        "rdm_cache_representation": (
            "squared Euclidean distances; square root applied before matching"
        ),
        "classical_estimator": "sample mean within each class and raw time sample",
        "flow_estimator": "joint class-and-time-conditioned translation-only flow",
        "flow_config": asdict(config),
        "metrics": metrics,
        "fit_summaries": fit_summaries,
        "fit_elapsed_seconds_excluding_reused_A01T_fit": fit_elapsed_seconds,
        "aggregation_wall_seconds_this_invocation": float(aggregation_wall_seconds),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_results(args.output_dir, scores, actual_lag_ms, metrics, recordings)
    for method in METHODS:
        primary = metrics[method]["0"]
        pre = metrics[method]["pre_cue_zero_lag"]
        print(
            f"[result] {METHOD_LABELS[method]} zero-lag top1={primary['top1_accuracy']:.3f} "
            f"MRR={primary['mean_reciprocal_rank']:.3f} "
            f"margin={primary['mean_true_minus_best_competitor_margin']:.3f} "
            f"pre-cue-top1={pre['top1_accuracy']:.3f}",
            flush=True,
        )
    print(
        f"[experiment] fit_elapsed_excluding_reused_A01T="
        f"{fit_elapsed_seconds / 60.0:.2f} min",
        flush=True,
    )
    print(f"[experiment] output={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
