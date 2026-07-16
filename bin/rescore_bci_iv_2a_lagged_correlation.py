#!/usr/bin/env python3
"""Rescore cached BCI IV-2a RDM trajectories with bounded lagged correlation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher.rdm_lagged_matching import (  # noqa: E402
    lagged_pearson_similarity,
    rdm_upper_triangle_sequence,
)
from fisher.bci_iv_2a_session_identification import RDM_MATCHING_INTERVAL  # noqa: E402


METHOD_LABELS = {
    "classical_mahalanobis": "Mahalanobis",
    "time_varying_shared_affine_flow": "Flow",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-lag-ms",
        type=float,
        nargs="+",
        default=[0.0, 25.0, 50.0, 100.0, 200.0],
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _load_rdms(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        rdms = np.asarray(archive["rdms"], dtype=np.float64)
    if not np.all(np.isfinite(rdms)):
        raise ValueError(f"Non-finite RDM values in {path}.")
    return rdms


def _rank_metrics(scores: np.ndarray, recording_keys: list[str]) -> dict:
    n_recordings = len(recording_keys)
    ranks = np.empty(n_recordings, dtype=np.int64)
    predictions = np.empty(n_recordings, dtype=np.int64)
    margins = np.empty(n_recordings, dtype=np.float64)
    for query in range(n_recordings):
        order = np.argsort(-scores[query], kind="mergesort")
        ranks[query] = int(np.flatnonzero(order == query)[0]) + 1
        predictions[query] = int(order[0])
        margins[query] = float(
            scores[query, query] - np.max(np.delete(scores[query], query))
        )
    return {
        "top1_accuracy": float(np.mean(ranks == 1)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "mean_true_minus_best_competitor_margin": float(np.mean(margins)),
        "ranks": ranks.astype(int).tolist(),
        "predicted_recordings": [recording_keys[index] for index in predictions],
        "correct_recordings": [
            recording_keys[index] for index in range(n_recordings) if ranks[index] == 1
        ],
    }


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(width=1.8)


def _plot_metric_sweep(
    output_dir: Path,
    actual_lag_ms: np.ndarray,
    methods: list[str],
    metrics: dict,
    *,
    n_recordings: int,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    colors = ["#4477AA", "#CC6677"]
    for method_index, method in enumerate(methods):
        top1 = [metrics[method][str(index)]["lagged"]["top1_accuracy"] for index in range(actual_lag_ms.size)]
        mrr = [metrics[method][str(index)]["lagged"]["mean_reciprocal_rank"] for index in range(actual_lag_ms.size)]
        label = METHOD_LABELS.get(method, method)
        axes[0].plot(actual_lag_ms, top1, marker="o", linewidth=2.0, color=colors[method_index], label=label)
        axes[1].plot(actual_lag_ms, mrr, marker="o", linewidth=2.0, color=colors[method_index], label=label)
    axes[0].axhline(1.0 / n_recordings, color="0.35", linestyle="--", linewidth=1.6, label="Chance")
    expected_random_mrr = float(np.sum(1.0 / np.arange(1, n_recordings + 1)) / n_recordings)
    axes[1].axhline(expected_random_mrr, color="0.35", linestyle="--", linewidth=1.6)
    axes[0].set_xlabel("Maximum lag (ms)")
    axes[0].set_ylabel("Top-1 accuracy")
    axes[1].set_xlabel("Maximum lag (ms)")
    axes[1].set_ylabel("Mean reciprocal rank")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_ylim(0.0, 1.02)
    axes[0].set_xticks(actual_lag_ms)
    axes[1].set_xticks(actual_lag_ms)
    axes[0].legend(frameon=False, loc="best")
    for axis in axes:
        _style_axis(axis)
    figure.tight_layout()
    figure.savefig(output_dir / "lagged_correlation_metric_sweep.png", dpi=300)
    figure.savefig(output_dir / "lagged_correlation_metric_sweep.svg")
    plt.close(figure)


def _plot_heatmaps(
    output_dir: Path,
    scores: np.ndarray,
    zero_core_scores: np.ndarray,
    methods: list[str],
    recording_keys: list[str],
    actual_max_lag_ms: float,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    figure, axes = plt.subplots(
        len(methods),
        2,
        figsize=(8.0, 3.5 * len(methods)),
        squeeze=False,
        layout="constrained",
    )
    images = []
    labels = [key.removeprefix("A").removesuffix("T") for key in recording_keys]
    for method_index, method in enumerate(methods):
        panels = [zero_core_scores[method_index, -1], scores[method_index, -1]]
        titles = ["Zero lag", f"Best lag within ±{actual_max_lag_ms:g} ms"]
        for column, (panel, title) in enumerate(zip(panels, titles, strict=True)):
            axis = axes[method_index, column]
            image = axis.imshow(panel, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="equal")
            images.append(image)
            if method_index == 0:
                axis.set_title(title)
            axis.set_xticks(range(len(labels)), labels)
            axis.set_yticks(range(len(labels)), labels)
            if method_index == len(methods) - 1:
                axis.set_xlabel("Reference recording")
            if column == 0:
                axis.set_ylabel(f"{METHOD_LABELS.get(method, method)}\nQuery recording")
            _style_axis(axis)
    figure.colorbar(images[-1], ax=axes, label="RDM correlation", shrink=0.82)
    figure.savefig(output_dir / "lagged_correlation_identification_heatmaps.png", dpi=300)
    figure.savefig(output_dir / "lagged_correlation_identification_heatmaps.svg")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        device_index = int(args.device.split(":", maxsplit=1)[1]) if ":" in args.device else 0
        torch.cuda.set_device(device_index)
        print(f"[lagged-score] device={args.device} GPU={torch.cuda.get_device_name(device_index)}", flush=True)

    source_summary_path = args.run_dir / "summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    recording_keys = list(source_summary["recordings"])
    methods = list(source_summary["methods"])
    times = np.asarray(source_summary["input_features"]["time_centers_seconds_cue_relative"], dtype=np.float64)
    interval = RDM_MATCHING_INTERVAL
    sample_period_seconds = float(np.median(np.diff(times)))

    requested_lag_ms = sorted({0.0, *[float(value) for value in args.max_lag_ms]})
    if requested_lag_ms[0] < 0.0:
        raise ValueError("Maximum lags must be non-negative.")
    lag_samples = np.asarray(
        [int(round(value / (sample_period_seconds * 1000.0))) for value in requested_lag_ms],
        dtype=np.int64,
    )
    unique_indices = np.concatenate(([True], np.diff(lag_samples) != 0))
    lag_samples = lag_samples[unique_indices]
    requested_lag_ms = [value for value, keep in zip(requested_lag_ms, unique_indices, strict=True) if keep]
    actual_lag_ms = lag_samples.astype(np.float64) * sample_period_seconds * 1000.0

    sequences: dict[tuple[str, str, str], np.ndarray] = {}
    selected_times = None
    cache_dir = args.run_dir / "rdm_cache"
    for method in methods:
        for recording in recording_keys:
            paths = {
                "reference": cache_dir / f"reference_{recording}_{method}.npz",
                "query": cache_dir / f"query_{recording}_nall_rep00_{method}.npz",
            }
            for role, path in paths.items():
                sequence, interval_times = rdm_upper_triangle_sequence(
                    _load_rdms(path), times, interval=interval
                )
                sequences[(method, role, recording)] = sequence
                if selected_times is None:
                    selected_times = interval_times
                else:
                    np.testing.assert_allclose(interval_times, selected_times)

    n_methods = len(methods)
    n_limits = len(lag_samples)
    n_recordings = len(recording_keys)
    scores = np.empty((n_methods, n_limits, n_recordings, n_recordings), dtype=np.float64)
    zero_core_scores = np.empty_like(scores)
    best_lags = np.empty_like(scores, dtype=np.int64)
    core_time_points = np.empty(n_limits, dtype=np.int64)
    for method_index, method in enumerate(methods):
        for limit_index, lag_limit in enumerate(lag_samples):
            for query_index, query_recording in enumerate(recording_keys):
                query = sequences[(method, "query", query_recording)]
                for candidate_index, candidate_recording in enumerate(recording_keys):
                    reference = sequences[(method, "reference", candidate_recording)]
                    result = lagged_pearson_similarity(
                        query, reference, max_lag_samples=int(lag_limit)
                    )
                    scores[method_index, limit_index, query_index, candidate_index] = result.score
                    zero_core_scores[method_index, limit_index, query_index, candidate_index] = result.zero_lag_same_core_score
                    best_lags[method_index, limit_index, query_index, candidate_index] = result.lag_samples
                    core_time_points[limit_index] = result.n_core_time_points

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict = {}
    for method_index, method in enumerate(methods):
        metrics[method] = {}
        for limit_index in range(n_limits):
            lagged_metrics = _rank_metrics(scores[method_index, limit_index], recording_keys)
            zero_metrics = _rank_metrics(zero_core_scores[method_index, limit_index], recording_keys)
            true_lags = best_lags[method_index, limit_index, np.arange(n_recordings), np.arange(n_recordings)]
            lagged_metrics["true_pair_reference_minus_query_lag_ms"] = (
                true_lags.astype(np.float64) * sample_period_seconds * 1000.0
            ).tolist()
            lagged_metrics["median_absolute_true_pair_lag_ms"] = float(
                np.median(np.abs(true_lags)) * sample_period_seconds * 1000.0
            )
            metrics[method][str(limit_index)] = {
                "lagged": lagged_metrics,
                "zero_lag_same_core": zero_metrics,
            }

    np.savez_compressed(
        args.output_dir / "lagged_correlation_results.npz",
        scores=scores,
        zero_lag_same_core_scores=zero_core_scores,
        best_lag_samples=best_lags,
        lag_limits_samples=lag_samples,
        actual_lag_limits_ms=actual_lag_ms,
        requested_lag_limits_ms=np.asarray(requested_lag_ms),
        core_time_points=core_time_points,
        methods=np.asarray(methods),
        recordings=np.asarray(recording_keys),
        interval=np.asarray(interval),
    )

    rows: list[dict] = []
    for method_index, method in enumerate(methods):
        for limit_index in range(n_limits):
            rank_values = metrics[method][str(limit_index)]["lagged"]["ranks"]
            for query_index, query_recording in enumerate(recording_keys):
                for candidate_index, candidate_recording in enumerate(recording_keys):
                    rows.append(
                        {
                            "method": method,
                            "requested_max_lag_ms": requested_lag_ms[limit_index],
                            "actual_max_lag_ms": actual_lag_ms[limit_index],
                            "core_time_points": int(core_time_points[limit_index]),
                            "query_recording": query_recording,
                            "candidate_recording": candidate_recording,
                            "is_true_pair": query_index == candidate_index,
                            "lagged_correlation": float(scores[method_index, limit_index, query_index, candidate_index]),
                            "reference_minus_query_lag_ms": float(best_lags[method_index, limit_index, query_index, candidate_index] * sample_period_seconds * 1000.0),
                            "zero_lag_same_core_correlation": float(zero_core_scores[method_index, limit_index, query_index, candidate_index]),
                            "true_recording_rank": int(rank_values[query_index]),
                        }
                    )
    with (args.output_dir / "lagged_correlation_pair_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": "BCI IV-2a cached-RDM identification with bounded global-lag Pearson correlation",
        "source_run": str(args.run_dir.resolve()),
        "device": args.device,
        "recordings": recording_keys,
        "methods": methods,
        "chance_top1": 1.0 / n_recordings,
        "interval_seconds_cue_relative": list(interval),
        "sample_period_seconds": sample_period_seconds,
        "requested_lag_limits_ms": requested_lag_ms,
        "actual_lag_limits_ms": actual_lag_ms.tolist(),
        "lag_limits_samples": lag_samples.astype(int).tolist(),
        "core_time_points": core_time_points.astype(int).tolist(),
        "lag_sign_convention": "positive means reference time is later than query time",
        "fixed_overlap_rule": "for limit L, query indices L:-L are fixed for every candidate lag in [-L,L]",
        "metrics": metrics,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    _plot_metric_sweep(
        args.output_dir,
        actual_lag_ms,
        methods,
        metrics,
        n_recordings=n_recordings,
    )
    _plot_heatmaps(
        args.output_dir,
        scores,
        zero_core_scores,
        methods,
        recording_keys,
        float(actual_lag_ms[-1]),
    )
    for method in methods:
        print(f"[lagged-score] {METHOD_LABELS.get(method, method)}", flush=True)
        for limit_index, lag_ms in enumerate(actual_lag_ms):
            result = metrics[method][str(limit_index)]["lagged"]
            zero = metrics[method][str(limit_index)]["zero_lag_same_core"]
            print(
                f"  max_lag={lag_ms:g} ms core={core_time_points[limit_index]} "
                f"top1={result['top1_accuracy']:.3f} MRR={result['mean_reciprocal_rank']:.3f} "
                f"margin={result['mean_true_minus_best_competitor_margin']:.3f} "
                f"zero_core_top1={zero['top1_accuracy']:.3f}",
                flush=True,
            )
    print(f"[lagged-score] output={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
