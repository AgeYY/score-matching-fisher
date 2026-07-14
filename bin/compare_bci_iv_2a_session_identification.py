#!/usr/bin/env python3
"""Run within-recording BCI IV-2a identification from time-resolved RDMs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher.bci_iv_2a_dataset import load_features_npz  # noqa: E402
from fisher.bci_iv_2a_session_identification import (  # noqa: E402
    FlowRDMConfig,
    QUERY_RUNS,
    REFERENCE_RUNS,
    classical_mahalanobis_rdms,
    load_rdm_cache,
    pearson_similarity,
    per_class_counts,
    save_rdm_cache,
    select_half,
    shared_affine_flow_rdms,
    subsample_balanced_trials,
    time_varying_shared_affine_flow_rdms,
    vectorize_rdms,
)


METHODS = (
    "classical_mahalanobis",
    "shared_affine_flow",
    "time_varying_shared_affine_flow",
)
METHOD_LABELS = {
    "classical_mahalanobis": "Mahalanobis",
    "shared_affine_flow": r"Global-$\Sigma$ flow",
    "time_varying_shared_affine_flow": r"$\Sigma(u)$ flow",
}
METHOD_SEED_OFFSETS = {
    "classical_mahalanobis": 0,
    "shared_affine_flow": 10_000_000,
    "time_varying_shared_affine_flow": 20_000_000,
}
N_LABELS = ("4", "8", "12", "18", "24", "all")
PRIMARY_INTERVAL = (2.0, 3.5)
PRE_CUE_INTERVAL = (-1.5, -0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-dir", type=Path, default=ROOT / "data/bci_iv_2a/processed/log_bandpower"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/bci_iv_2a/session_identification"
    )
    parser.add_argument(
        "--reuse-cache-dir",
        type=Path,
        default=None,
        help="Optional prior rdm_cache directory from which matching cache files are copied.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=None,
        help="Recording stems to include, for example A01T A02T A03T.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHOD_LABELS),
        default=list(METHODS),
    )
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    return parser.parse_args()


def fit_or_load(
    cache_path: Path,
    *,
    method: str,
    x: np.ndarray,
    labels: np.ndarray,
    time_centers: np.ndarray,
    device: torch.device,
    seed: int,
    config: FlowRDMConfig,
    context: dict,
    reuse_cache_path: Path | None = None,
) -> np.ndarray:
    if not cache_path.exists() and reuse_cache_path is not None and reuse_cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reuse_cache_path, cache_path)
    if cache_path.exists():
        rdms, _ = load_rdm_cache(cache_path)
        return rdms
    if method == "classical_mahalanobis":
        rdms = classical_mahalanobis_rdms(x, labels)
        metadata = {**context, "method": method, "seed": int(seed)}
    elif method == "shared_affine_flow":
        rdms, fit_metadata = shared_affine_flow_rdms(
            x,
            labels,
            time_centers,
            device=device,
            seed=int(seed),
            config=config,
        )
        metadata = {**context, "method": method, "fit": fit_metadata}
    elif method == "time_varying_shared_affine_flow":
        rdms, fit_metadata = time_varying_shared_affine_flow_rdms(
            x,
            labels,
            time_centers,
            device=device,
            seed=int(seed),
            config=config,
        )
        metadata = {**context, "method": method, "fit": fit_metadata}
    else:
        raise ValueError(method)
    save_rdm_cache(cache_path, rdms=rdms, metadata=metadata)
    return rdms


def save_outputs(
    output_dir: Path,
    *,
    scores: np.ndarray,
    pre_scores: np.ndarray,
    effective_n: np.ndarray,
    repeats: int,
    config: FlowRDMConfig,
    seed: int,
    recording_keys: list[str],
) -> dict:
    n_recordings = len(recording_keys)
    if n_recordings < 2:
        raise ValueError("Identification requires at least two recordings.")
    ranks = np.full(scores.shape[:-1], np.nan, dtype=np.float64)
    pre_ranks = np.full(pre_scores.shape[:-1], np.nan, dtype=np.float64)
    for method_index in range(len(METHODS)):
        for n_index in range(len(N_LABELS)):
            for repeat in range(repeats):
                for subject in range(n_recordings):
                    values = scores[method_index, n_index, repeat, subject]
                    pre_values = pre_scores[method_index, n_index, repeat, subject]
                    if np.all(np.isfinite(values)):
                        order = np.argsort(-values, kind="mergesort")
                        ranks[method_index, n_index, repeat, subject] = int(np.flatnonzero(order == subject)[0]) + 1
                    if np.all(np.isfinite(pre_values)):
                        order = np.argsort(-pre_values, kind="mergesort")
                        pre_ranks[method_index, n_index, repeat, subject] = (
                            int(np.flatnonzero(order == subject)[0]) + 1
                        )
    np.savez_compressed(
        output_dir / "results.npz",
        scores=scores,
        pre_cue_scores=pre_scores,
        ranks=ranks,
        pre_cue_ranks=pre_ranks,
        effective_n=effective_n,
        methods=np.asarray(METHODS),
        n_labels=np.asarray(N_LABELS),
        primary_interval=np.asarray(PRIMARY_INTERVAL),
        pre_cue_interval=np.asarray(PRE_CUE_INTERVAL),
    )
    rows: list[dict] = []
    for method_index, method in enumerate(METHODS):
        for n_index, n_label in enumerate(N_LABELS):
            for repeat in range(repeats):
                for subject in range(n_recordings):
                    for candidate in range(n_recordings):
                        rows.append(
                            {
                                "method": method,
                                "n_label": n_label,
                                "effective_n_per_class": int(effective_n[n_index, subject]),
                                "repeat": repeat,
                                "query_recording": recording_keys[subject],
                                "candidate_recording": recording_keys[candidate],
                                "correlation": float(scores[method_index, n_index, repeat, subject, candidate]),
                                "pre_cue_correlation": float(
                                    pre_scores[method_index, n_index, repeat, subject, candidate]
                                ),
                                "true_rank": int(ranks[method_index, n_index, repeat, subject]),
                                "pre_cue_true_rank": int(
                                    pre_ranks[method_index, n_index, repeat, subject]
                                ),
                            }
                        )
    with (output_dir / "pair_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict = {
        "experiment": "BCI Competition IV 2a within-recording RDM identification",
        "recordings": recording_keys,
        "query_runs_one_based": [1, 3, 5],
        "reference_runs_one_based": [2, 4, 6],
        "methods": list(METHODS),
        "n_labels": list(N_LABELS),
        "repeats": int(repeats),
        "seed": int(seed),
        "chance_top1": 1.0 / float(n_recordings),
        "primary_interval_seconds_cue_relative": list(PRIMARY_INTERVAL),
        "pre_cue_interval_seconds_cue_relative": list(PRE_CUE_INTERVAL),
        "flow_config": asdict(config),
        "metrics": {},
        "paired_method_comparison": {},
    }
    for method_index, method in enumerate(METHODS):
        summary["metrics"][method] = {}
        for n_index, n_label in enumerate(N_LABELS):
            rank_values = ranks[method_index, n_index]
            pre_rank_values = pre_ranks[method_index, n_index]
            per_subject_top1 = np.mean(rank_values == 1, axis=0)
            margins = []
            for repeat in range(repeats):
                for subject in range(n_recordings):
                    candidate_scores = scores[method_index, n_index, repeat, subject]
                    margins.append(
                        float(candidate_scores[subject] - np.max(np.delete(candidate_scores, subject)))
                    )
            summary["metrics"][method][n_label] = {
                "top1_accuracy": float(np.mean(rank_values == 1)),
                "top1_subject_sem": float(
                    np.std(per_subject_top1, ddof=1) / np.sqrt(n_recordings)
                ),
                "mean_reciprocal_rank": float(np.mean(1.0 / rank_values)),
                "pre_cue_top1_accuracy": float(np.mean(pre_rank_values == 1)),
                "mean_true_minus_best_competitor_margin": float(np.mean(margins)),
                "top1_accuracy_by_recording": per_subject_top1.tolist(),
                "effective_n_per_class": effective_n[n_index].astype(int).tolist(),
            }
    for method_index, method in enumerate(METHODS[1:], start=1):
        summary["paired_method_comparison"][method] = {}
        for n_index, n_label in enumerate(N_LABELS):
            classical_by_subject = np.mean(ranks[0, n_index] == 1, axis=0)
            flow_by_subject = np.mean(ranks[method_index, n_index] == 1, axis=0)
            differences = flow_by_subject - classical_by_subject
            observed = float(np.mean(differences))
            null = []
            for bits in range(2**n_recordings):
                signs = np.asarray(
                    [
                        1.0 if (bits >> subject) & 1 else -1.0
                        for subject in range(n_recordings)
                    ]
                )
                null.append(float(np.mean(signs * differences)))
            exact_p = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
            summary["paired_method_comparison"][method][n_label] = {
                "flow_minus_classical_top1": observed,
                "exact_two_sided_subject_sign_flip_p": exact_p,
                "unit": "recording-level top1 accuracy averaged over subsampling repeats",
            }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def plot_outputs(output_dir: Path, summary: dict, scores: np.ndarray) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 11,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    x = np.arange(len(N_LABELS))
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    colors = ("#4C78A8", "#E45756", "#54A24B")
    for method, color in zip(METHODS, colors[: len(METHODS)], strict=True):
        values = [summary["metrics"][method][label]["top1_accuracy"] for label in N_LABELS]
        errors = [summary["metrics"][method][label]["top1_subject_sem"] for label in N_LABELS]
        ax.errorbar(
            x,
            values,
            yerr=errors,
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=METHOD_LABELS[method],
            color=color,
        )
    ax.axhline(
        float(summary["chance_top1"]),
        color="0.35",
        linestyle="--",
        linewidth=1.5,
        label="chance",
    )
    ax.set_xticks(x, N_LABELS)
    ax.set_xlabel("Query trials per class")
    ax.set_ylabel("Top-1 recording accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.tick_params(width=1.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "top1_accuracy_vs_n.svg")
    fig.savefig(output_dir / "top1_accuracy_vs_n.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(
        1,
        len(METHODS),
        figsize=(4.0 * len(METHODS), 3.5),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for method_index, (method, ax) in enumerate(zip(METHODS, axes, strict=True)):
        matrix = np.mean(scores[method_index, -1], axis=0)
        image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax.set_title(METHOD_LABELS[method])
        ax.set_xlabel("Reference recording")
        ax.set_ylabel("Query recording")
        ticks = np.arange(scores.shape[-1])
        tick_labels = [str(key)[1:-1] for key in summary["recordings"]]
        ax.set_xticks(ticks, tick_labels)
        ax.set_yticks(ticks, tick_labels)
        ax.tick_params(width=1.8)
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
    fig.colorbar(image, ax=axes, label="RDM correlation", shrink=0.85)
    fig.savefig(output_dir / "all_trials_identification_heatmaps.svg")
    fig.savefig(output_dir / "all_trials_identification_heatmaps.png", dpi=300)
    plt.close(fig)


def main() -> None:
    global METHODS
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.patience < 0:
        raise ValueError("--patience must be >= 0")
    selected_methods = tuple(args.methods)
    if len(set(selected_methods)) != len(selected_methods):
        raise ValueError("--methods must not contain duplicates.")
    if not selected_methods or selected_methods[0] != "classical_mahalanobis":
        raise ValueError("--methods must list classical_mahalanobis first.")
    METHODS = selected_methods
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA; no CPU fallback is permitted.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device.index} is unavailable.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "rdm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = FlowRDMConfig(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )
    if args.recordings is None:
        feature_paths = sorted(args.feature_dir.glob("A??T.npz"))
        if len(feature_paths) != 9:
            raise FileNotFoundError(
                f"Expected nine feature files in {args.feature_dir}, found {len(feature_paths)}"
            )
    else:
        if len(set(args.recordings)) != len(args.recordings):
            raise ValueError("--recordings must not contain duplicates.")
        feature_paths = [args.feature_dir / f"{recording}.npz" for recording in args.recordings]
        missing = [str(path) for path in feature_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing requested feature files: {missing}")
    if len(feature_paths) < 2:
        raise ValueError("At least two recordings are required for identification.")
    datasets = [load_features_npz(path) for path in feature_paths]
    recording_keys = [dataset.session_key for dataset in datasets]
    n_recordings = len(datasets)
    times = datasets[0].time_centers
    if any(not np.array_equal(dataset.time_centers, times) for dataset in datasets[1:]):
        raise ValueError("Feature files do not share the same time grid.")

    print(f"[experiment] device={device} GPU={torch.cuda.get_device_name(device)}", flush=True)
    print(f"[experiment] output={args.output_dir.resolve()}", flush=True)
    print(f"[experiment] recordings={recording_keys} methods={list(METHODS)}", flush=True)
    print(
        f"[experiment] epochs={config.epochs} patience={config.patience} "
        "checkpoint=best",
        flush=True,
    )
    reference_vectors: dict[tuple[str, str], np.ndarray] = {}
    pre_reference_vectors: dict[tuple[str, str], np.ndarray] = {}
    for subject, dataset in enumerate(datasets):
        x_ref, y_ref, _ = select_half(dataset, REFERENCE_RUNS)
        for method in METHODS:
            cache_path = cache_dir / f"reference_{dataset.session_key}_{method}.npz"
            seed = (
                args.seed
                + 1_000_000
                + subject * 10
                + METHOD_SEED_OFFSETS[method] // 10_000_000
            )
            rdms = fit_or_load(
                cache_path,
                method=method,
                x=x_ref,
                labels=y_ref,
                time_centers=times,
                device=device,
                seed=seed,
                config=config,
                context={"role": "reference", "recording": dataset.session_key},
                reuse_cache_path=(
                    None
                    if args.reuse_cache_dir is None or method != "classical_mahalanobis"
                    else args.reuse_cache_dir / cache_path.name
                ),
            )
            reference_vectors[(method, dataset.session_key)] = vectorize_rdms(
                rdms, times, interval=PRIMARY_INTERVAL
            )
            pre_reference_vectors[(method, dataset.session_key)] = vectorize_rdms(
                rdms, times, interval=PRE_CUE_INTERVAL
            )
        print(f"[reference] {dataset.session_key} complete", flush=True)

    scores = np.full(
        (len(METHODS), len(N_LABELS), args.repeats, n_recordings, n_recordings),
        np.nan,
        dtype=np.float64,
    )
    pre_scores = np.full_like(scores, np.nan)
    effective_n = np.zeros((len(N_LABELS), n_recordings), dtype=np.int64)
    for subject, dataset in enumerate(datasets):
        x_query, y_query, _ = select_half(dataset, QUERY_RUNS)
        max_balanced = int(np.min(per_class_counts(y_query)))
        requested = [4, 8, 12, 18, 24, max_balanced]
        effective_n[:, subject] = np.asarray(requested, dtype=np.int64)
        for n_index, n_per_class in enumerate(requested):
            for repeat in range(args.repeats):
                subset_seed = args.seed + subject * 100_000 + n_index * 1_000 + repeat
                selected = subsample_balanced_trials(y_query, n_per_class, subset_seed)
                x_sub, y_sub = x_query[selected], y_query[selected]
                for method_index, method in enumerate(METHODS):
                    cache_path = cache_dir / (
                        f"query_{dataset.session_key}_n{N_LABELS[n_index]}_rep{repeat:02d}_{method}.npz"
                    )
                    flow_seed = subset_seed + METHOD_SEED_OFFSETS[method]
                    rdms = fit_or_load(
                        cache_path,
                        method=method,
                        x=x_sub,
                        labels=y_sub,
                        time_centers=times,
                        device=device,
                        seed=flow_seed,
                        config=config,
                        context={
                            "role": "query",
                            "recording": dataset.session_key,
                            "n_label": N_LABELS[n_index],
                            "n_per_class": int(n_per_class),
                            "repeat": int(repeat),
                            "selected_trial_indices": selected.tolist(),
                        },
                        reuse_cache_path=(
                            None
                            if args.reuse_cache_dir is None or method != "classical_mahalanobis"
                            else args.reuse_cache_dir / cache_path.name
                        ),
                    )
                    query_vector = vectorize_rdms(rdms, times, interval=PRIMARY_INTERVAL)
                    pre_query_vector = vectorize_rdms(rdms, times, interval=PRE_CUE_INTERVAL)
                    for candidate, candidate_dataset in enumerate(datasets):
                        scores[method_index, n_index, repeat, subject, candidate] = pearson_similarity(
                            query_vector, reference_vectors[(method, candidate_dataset.session_key)]
                        )
                        pre_scores[method_index, n_index, repeat, subject, candidate] = pearson_similarity(
                            pre_query_vector, pre_reference_vectors[(method, candidate_dataset.session_key)]
                        )
            print(
                f"[query] {dataset.session_key} n={N_LABELS[n_index]} "
                f"effective={n_per_class} repeats={args.repeats} complete",
                flush=True,
            )
    summary = save_outputs(
        args.output_dir,
        scores=scores,
        pre_scores=pre_scores,
        effective_n=effective_n,
        repeats=args.repeats,
        config=config,
        seed=args.seed,
        recording_keys=recording_keys,
    )
    plot_outputs(args.output_dir, summary, scores)
    print(f"[experiment] Saved: {args.output_dir / 'summary.json'}", flush=True)
    print("=== BCI IV-2a within-session experiment complete ===", flush=True)


if __name__ == "__main__":
    main()
