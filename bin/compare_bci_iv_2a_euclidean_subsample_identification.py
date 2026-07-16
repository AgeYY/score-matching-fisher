#!/usr/bin/env python3
"""Five-recording zero-lag Euclidean RDM identification versus query size."""

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
    classical_squared_euclidean_rdms,
    pearson_similarity,
    per_class_counts,
    select_half,
    subsample_balanced_trials,
    translation_flow_squared_euclidean_rdms,
)
from fisher.rdm_lagged_matching import rdm_upper_triangle_sequence  # noqa: E402


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
        default=ROOT / "data/bci_iv_2a/euclidean_subsample_5recordings",
    )
    parser.add_argument(
        "--reuse-full-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/euclidean_session_identification_5recordings",
    )
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=["A01T", "A02T", "A03T", "A04T", "A05T"],
    )
    parser.add_argument(
        "--query-recordings",
        nargs="+",
        help="Optional worker mode: fit only these query recordings and exit.",
    )
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--n-values",
        nargs="+",
        default=["4", "8", "12", "18", "24", "all"],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--epochs", type=int, default=TRAINING_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    return parser.parse_args()


def _validate_n_labels(values: list[str]) -> tuple[str, ...]:
    labels = tuple(str(value).lower() for value in values)
    if not labels or labels[-1] != "all" or "all" in labels[:-1]:
        raise ValueError("--n-values must end with exactly one 'all'.")
    if len(set(labels)) != len(labels):
        raise ValueError("--n-values must not contain duplicates.")
    for value in labels[:-1]:
        if not value.isdigit() or int(value) < 1:
            raise ValueError("Finite --n-values entries must be positive integers.")
    return labels


def _save_cache(path: Path, rdms: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    if rdms.ndim != 3 or rdms.shape[1:] != (4, 4):
        raise ValueError(f"Invalid RDM shape in {path}: {rdms.shape}.")
    if not np.all(np.isfinite(rdms)) or np.any(rdms < -1e-12):
        raise ValueError(f"Invalid squared distances in {path}.")
    return rdms, metadata


def _literal_euclidean_sequence(
    squared_rdms: np.ndarray,
    times: np.ndarray,
    *,
    interval: tuple[float, float],
) -> np.ndarray:
    rdms = np.sqrt(np.maximum(np.asarray(squared_rdms, dtype=np.float64), 0.0))
    sequence, _ = rdm_upper_triangle_sequence(rdms, times, interval=interval)
    return sequence.reshape(-1)


def _fit_or_load_query(
    path: Path,
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
    if path.is_file():
        print(f"[cache] {path.name}", flush=True)
        return _load_cache(path)
    start = time.perf_counter()
    if method == "classical_squared_euclidean":
        rdms = classical_squared_euclidean_rdms(
            x,
            labels,
            standardize_features=False,
        )
        metadata = {"estimator": "sample_condition_means_at_each_raw_time"}
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
        "seed": int(seed),
        "n_trials": int(x.shape[0]),
        "per_class_counts": per_class_counts(labels).astype(int).tolist(),
        "elapsed_seconds": float(elapsed),
    }
    _save_cache(path, rdms, metadata)
    print(f"[fit] {path.name} elapsed={elapsed / 60.0:.2f} min", flush=True)
    return rdms, metadata


def _full_cache_path(full_dir: Path, role: str, recording: str, method: str) -> Path:
    return full_dir / "rdm_cache" / f"{role}_{recording}_{method}.npz"


def _finite_cache_path(
    cache_dir: Path,
    recording: str,
    n_label: str,
    repeat: int,
    method: str,
) -> Path:
    return cache_dir / f"query_{recording}_n{n_label}_rep{repeat:02d}_{method}.npz"


def _rank_score_matrix(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    n_recordings = values.shape[0]
    ranks = np.empty(n_recordings, dtype=np.int64)
    predictions = np.empty(n_recordings, dtype=np.int64)
    margins = np.empty(n_recordings, dtype=np.float64)
    for query_index in range(n_recordings):
        order = np.argsort(-values[query_index], kind="mergesort")
        ranks[query_index] = int(np.flatnonzero(order == query_index)[0]) + 1
        predictions[query_index] = int(order[0])
        margins[query_index] = float(
            values[query_index, query_index]
            - np.max(np.delete(values[query_index], query_index))
        )
    return ranks, predictions, margins


def _aggregate(
    *,
    args: argparse.Namespace,
    datasets: list,
    times: np.ndarray,
    n_labels: tuple[str, ...],
    effective_n: np.ndarray,
    config: FlowRDMConfig,
) -> dict:
    recordings = [dataset.session_key for dataset in datasets]
    n_recordings = len(recordings)
    cache_dir = args.output_dir / "rdm_cache"
    reference_vectors: dict[tuple[str, str, str], np.ndarray] = {}
    for method in METHODS:
        for recording in recordings:
            path = _full_cache_path(args.reuse_full_dir, "reference", recording, method)
            rdms, _ = _load_cache(path)
            reference_vectors[(method, recording, "primary")] = _literal_euclidean_sequence(
                rdms, times, interval=RDM_MATCHING_INTERVAL
            )
            reference_vectors[(method, recording, "pre_cue")] = _literal_euclidean_sequence(
                rdms, times, interval=PRE_CUE_INTERVAL
            )

    scores = np.full(
        (len(METHODS), len(n_labels), args.repeats, n_recordings, n_recordings),
        np.nan,
        dtype=np.float64,
    )
    pre_scores = np.full_like(scores, np.nan)
    fit_rows: list[dict] = []
    for method_index, method in enumerate(METHODS):
        for n_index, n_label in enumerate(n_labels):
            n_repeats = 1 if n_label == "all" else args.repeats
            for repeat in range(n_repeats):
                for query_index, recording in enumerate(recordings):
                    path = (
                        _full_cache_path(args.reuse_full_dir, "query", recording, method)
                        if n_label == "all"
                        else _finite_cache_path(cache_dir, recording, n_label, repeat, method)
                    )
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing required cache: {path}")
                    rdms, metadata = _load_cache(path)
                    query = _literal_euclidean_sequence(
                        rdms, times, interval=RDM_MATCHING_INTERVAL
                    )
                    pre_query = _literal_euclidean_sequence(
                        rdms, times, interval=PRE_CUE_INTERVAL
                    )
                    for reference_index, candidate in enumerate(recordings):
                        scores[method_index, n_index, repeat, query_index, reference_index] = (
                            pearson_similarity(
                                query,
                                reference_vectors[(method, candidate, "primary")],
                            )
                        )
                        pre_scores[
                            method_index, n_index, repeat, query_index, reference_index
                        ] = pearson_similarity(
                            pre_query,
                            reference_vectors[(method, candidate, "pre_cue")],
                        )
                    fit_rows.append(
                        {
                            "recording": recording,
                            "n_label": n_label,
                            "effective_n_per_class": int(effective_n[n_index, query_index]),
                            "repeat": int(repeat),
                            "method": method,
                            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0)),
                            "best_epoch": metadata.get("best_epoch"),
                            "stopped_epoch": metadata.get("stopped_epoch"),
                            "best_val_loss": metadata.get("best_val_loss"),
                            "reused_full_fit": n_label == "all",
                        }
                    )

    ranks = np.full(scores.shape[:-1], np.nan, dtype=np.float64)
    predictions = np.full(scores.shape[:-1], -1, dtype=np.int64)
    margins = np.full(scores.shape[:-1], np.nan, dtype=np.float64)
    pre_ranks = np.full(pre_scores.shape[:-1], np.nan, dtype=np.float64)
    metrics: dict = {}
    pair_rows: list[dict] = []
    for method_index, method in enumerate(METHODS):
        metrics[method] = {}
        for n_index, n_label in enumerate(n_labels):
            n_repeats = 1 if n_label == "all" else args.repeats
            top1_by_repeat = []
            mrr_by_repeat = []
            pre_top1_by_repeat = []
            for repeat in range(n_repeats):
                repeat_ranks, repeat_predictions, repeat_margins = _rank_score_matrix(
                    scores[method_index, n_index, repeat]
                )
                repeat_pre_ranks, _, _ = _rank_score_matrix(
                    pre_scores[method_index, n_index, repeat]
                )
                ranks[method_index, n_index, repeat] = repeat_ranks
                predictions[method_index, n_index, repeat] = repeat_predictions
                margins[method_index, n_index, repeat] = repeat_margins
                pre_ranks[method_index, n_index, repeat] = repeat_pre_ranks
                top1_by_repeat.append(float(np.mean(repeat_ranks == 1)))
                mrr_by_repeat.append(float(np.mean(1.0 / repeat_ranks)))
                pre_top1_by_repeat.append(float(np.mean(repeat_pre_ranks == 1)))
                for query_index, query_recording in enumerate(recordings):
                    for reference_index, reference_recording in enumerate(recordings):
                        pair_rows.append(
                            {
                                "method": method,
                                "n_label": n_label,
                                "effective_n_per_class": int(
                                    effective_n[n_index, query_index]
                                ),
                                "repeat": int(repeat),
                                "query_recording": query_recording,
                                "reference_recording": reference_recording,
                                "is_true_pair": query_index == reference_index,
                                "correlation": float(
                                    scores[
                                        method_index,
                                        n_index,
                                        repeat,
                                        query_index,
                                        reference_index,
                                    ]
                                ),
                                "true_recording_rank": int(repeat_ranks[query_index]),
                            }
                        )
            top1 = np.asarray(top1_by_repeat, dtype=np.float64)
            per_recording = np.mean(
                ranks[method_index, n_index, :n_repeats] == 1,
                axis=0,
            )
            metrics[method][n_label] = {
                "top1_accuracy": float(np.mean(top1)),
                "top1_repeat_sd": (
                    float(np.std(top1, ddof=1)) if top1.size > 1 else None
                ),
                "top1_accuracy_by_repeat": top1.tolist(),
                "mean_reciprocal_rank": float(np.mean(mrr_by_repeat)),
                "mean_true_minus_best_competitor_margin": float(
                    np.mean(margins[method_index, n_index, :n_repeats])
                ),
                "pre_cue_top1_accuracy": float(np.mean(pre_top1_by_repeat)),
                "top1_accuracy_by_recording": per_recording.tolist(),
                "effective_n_per_class": effective_n[n_index].astype(int).tolist(),
                "n_identification_repeats": int(n_repeats),
            }

    np.savez_compressed(
        args.output_dir / "euclidean_subsample_results.npz",
        scores=scores,
        pre_cue_scores=pre_scores,
        ranks=ranks,
        pre_cue_ranks=pre_ranks,
        predictions=predictions,
        margins=margins,
        effective_n=effective_n,
        recordings=np.asarray(recordings),
        methods=np.asarray(METHODS),
        n_labels=np.asarray(n_labels),
        interval=np.asarray(RDM_MATCHING_INTERVAL),
    )
    with (args.output_dir / "fit_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fit_rows[0]))
        writer.writeheader()
        writer.writerows(fit_rows)
    with (args.output_dir / "pair_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    summary = {
        "experiment": "BCI IV-2a five-recording Euclidean RDM subsample identification",
        "recordings": recordings,
        "methods": list(METHODS),
        "method_labels": METHOD_LABELS,
        "n_labels": list(n_labels),
        "repeats": int(args.repeats),
        "all_data_repeats": 1,
        "seed": int(args.seed),
        "chance_top1": 1.0 / n_recordings,
        "query_runs_one_based": [1, 3, 5],
        "reference_runs_one_based": [2, 4, 6],
        "sampling": "balanced within class; random without replacement per repeat",
        "matching": "zero-lag flattened Pearson correlation only",
        "time_interval_seconds_cue_relative": list(RDM_MATCHING_INTERVAL),
        "pre_cue_control_interval_seconds": list(PRE_CUE_INTERVAL),
        "distance": "euclidean_norm_between_condition_means",
        "rdm_cache_representation": (
            "squared Euclidean distances; square root applied before matching"
        ),
        "uncertainty": (
            "sample standard deviation across complete five-recording subsampling repeats"
        ),
        "flow_config": asdict(config),
        "effective_n_per_class": {
            label: effective_n[index].astype(int).tolist()
            for index, label in enumerate(n_labels)
        },
        "metrics": metrics,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot(args.output_dir, summary, n_labels)
    return summary


def _plot(output_dir: Path, summary: dict, n_labels: tuple[str, ...]) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 12,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    x = np.arange(len(n_labels))
    figure, axis = plt.subplots(figsize=(4.0, 3.5))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    for method in METHODS:
        values = np.asarray(
            [summary["metrics"][method][label]["top1_accuracy"] for label in n_labels]
        )
        errors = np.asarray(
            [
                np.nan
                if summary["metrics"][method][label]["top1_repeat_sd"] is None
                else summary["metrics"][method][label]["top1_repeat_sd"]
                for label in n_labels
            ]
        )
        axis.plot(
            x,
            values,
            marker="o",
            linewidth=2.0,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        finite = np.isfinite(errors)
        axis.errorbar(
            x[finite],
            values[finite],
            yerr=errors[finite],
            fmt="none",
            capsize=3,
            linewidth=1.5,
            color=METHOD_COLORS[method],
        )
    axis.axhline(
        float(summary["chance_top1"]),
        color="0.35",
        linestyle="--",
        linewidth=1.5,
        label="Chance",
    )
    axis.set_xticks(x, n_labels)
    axis.set_xlabel("Query trials per class")
    axis.set_ylabel("Top-1 accuracy")
    axis.set_ylim(0.0, 1.04)
    axis.legend(frameon=False, loc="best")
    axis.tick_params(width=1.8)
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    figure.savefig(
        output_dir / "top1_accuracy_vs_n.png",
        dpi=300,
        facecolor="white",
        transparent=False,
    )
    figure.savefig(
        output_dir / "top1_accuracy_vs_n.svg",
        facecolor="white",
        transparent=False,
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    n_labels = _validate_n_labels(args.n_values)
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2 for finite-size uncertainty.")
    if len(args.recordings) != 5 or len(set(args.recordings)) != 5:
        raise ValueError("This speed-limited experiment requires five unique recordings.")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA; no CPU fallback is permitted.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device.index} is unavailable.")
    torch.cuda.set_device(0 if device.index is None else device.index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "rdm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.feature_dir / f"{recording}.npz" for recording in args.recordings]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing feature files: {missing}")
    datasets = [load_features_npz(path) for path in paths]
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
    effective_n = np.zeros((len(n_labels), len(recordings)), dtype=np.int64)
    for subject, dataset in enumerate(datasets):
        _, y_query, _ = select_half(dataset, QUERY_RUNS)
        max_balanced = int(np.min(per_class_counts(y_query)))
        requested = [int(value) for value in n_labels[:-1]] + [max_balanced]
        if any(value > max_balanced for value in requested[:-1]):
            raise ValueError(
                f"{dataset.session_key}: requested n exceeds balanced maximum {max_balanced}."
            )
        effective_n[:, subject] = requested
    required_full = [
        _full_cache_path(args.reuse_full_dir, role, recording, method)
        for role in ("reference", "query")
        for recording in recordings
        for method in METHODS
    ]
    missing_full = [str(path) for path in required_full if not path.is_file()]
    if missing_full:
        raise FileNotFoundError(f"Missing converged full-data caches: {missing_full}")

    selected_recordings = set(recordings)
    if args.query_recordings is not None:
        selected_recordings = set(args.query_recordings)
        unknown = sorted(selected_recordings - set(recordings))
        if unknown:
            raise ValueError(f"Unknown query recordings: {unknown}")
    print(
        f"[experiment] zero-lag only recordings={recordings} device={device} "
        f"GPU={torch.cuda.get_device_name(device)}",
        flush=True,
    )
    print(
        f"[experiment] n_values={list(n_labels)} repeats={args.repeats} "
        f"epochs={config.epochs} patience={config.patience}",
        flush=True,
    )
    if not args.aggregate_only:
        for subject, dataset in enumerate(datasets):
            if dataset.session_key not in selected_recordings:
                continue
            x_query, y_query, _ = select_half(dataset, QUERY_RUNS)
            for n_index, n_label in enumerate(n_labels[:-1]):
                n_per_class = int(n_label)
                for repeat in range(args.repeats):
                    subset_seed = args.seed + subject * 100_000 + n_index * 1_000 + repeat
                    selected = subsample_balanced_trials(
                        y_query, n_per_class, subset_seed
                    )
                    x_sub = x_query[selected]
                    y_sub = y_query[selected]
                    for method_index, method in enumerate(METHODS):
                        fit_seed = subset_seed + method_index * 20_000_000
                        _fit_or_load_query(
                            _finite_cache_path(
                                cache_dir,
                                dataset.session_key,
                                n_label,
                                repeat,
                                method,
                            ),
                            method=method,
                            x=x_sub,
                            labels=y_sub,
                            times=times,
                            device=device,
                            seed=fit_seed,
                            config=config,
                            context={
                                "recording": dataset.session_key,
                                "n_label": n_label,
                                "n_per_class": n_per_class,
                                "repeat": int(repeat),
                                "selected_trial_indices": selected.astype(int).tolist(),
                                "sampling": "balanced_without_replacement",
                            },
                        )
                    print(
                        f"[query] {dataset.session_key} n={n_label} "
                        f"repeat={repeat + 1}/{args.repeats} complete",
                        flush=True,
                    )
        if args.query_recordings is not None:
            print(
                f"[worker] complete recordings={sorted(selected_recordings)}",
                flush=True,
            )
            return

    summary = _aggregate(
        args=args,
        datasets=datasets,
        times=times,
        n_labels=n_labels,
        effective_n=effective_n,
        config=config,
    )
    for method in METHODS:
        values = [
            summary["metrics"][method][label]["top1_accuracy"]
            for label in n_labels
        ]
        print(f"[result] {METHOD_LABELS[method]} top1={values}", flush=True)
    print(f"[experiment] output={args.output_dir.resolve()}", flush=True)
    print("=== zero-lag Euclidean subsample experiment complete ===", flush=True)


if __name__ == "__main__":
    main()
