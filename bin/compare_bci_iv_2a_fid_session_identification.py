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

    config = FlowRDMConfig(
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
) -> None:
    """Plot supplied classical/flow top-1 values as exactly touching bars."""

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
    if values.shape != (2,):
        raise ValueError("Top-1 bar plot requires classical and flow values.")
    positions = np.arange(2, dtype=np.float64)
    figure, axis = plt.subplots(figsize=(4.0, 3.5), layout="constrained")
    axis.bar(
        positions,
        values,
        width=1.0,
        color=[METHOD_COLORS["classical_fid"], METHOD_COLORS["condition_affine_flow_fid"]],
        linewidth=0.0,
    )
    axis.set_xlim(-0.5, 1.5)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(positions, ["Classical FID", "Flow-based FID"])
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
    print(f"[experiment] output={args.output_dir.resolve()}", flush=True)


def main() -> None:
    args = parse_args()
    if args.aggregate_only and args.fit_recording is not None:
        raise ValueError("--aggregate-only and --fit-recording are mutually exclusive.")
    if len(args.recordings) != 5 or len(set(args.recordings)) != 5:
        raise ValueError("This speed-limited experiment requires five unique recordings.")
    missing = [
        str(args.feature_dir / f"{recording}.npz")
        for recording in args.recordings
        if not (args.feature_dir / f"{recording}.npz").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing feature files: {missing}")
    if args.aggregate_only:
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
    for recording in args.recordings:
        _fit_one_recording(args, recording, device)
    _aggregate(args)


if __name__ == "__main__":
    main()
