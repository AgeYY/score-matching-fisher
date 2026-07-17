#!/usr/bin/env python3
"""Fit cross-fitted time-conditioned TRE Jeffreys RDMs for BCI IV-2a."""

from __future__ import annotations

import argparse
import json
import random
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
from fisher.tre_distance import TREDensityRatioConfig  # noqa: E402
from fisher.tre_time_conditioned import (  # noqa: E402
    evaluate_time_conditioned_log_ratio,
    train_time_conditioned_tre_density_ratio,
)


ROLES = ("reference", "all_trial")
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
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "data/bci_iv_2a/multi_distance_session_identification_9recordings_full_query"
        / "tre_jeffreys",
    )
    parser.add_argument(
        "--recordings", nargs="+", default=[f"A{index:02d}T" for index in range(1, 10)]
    )
    parser.add_argument("--fit-recording", choices=[f"A{i:02d}T" for i in range(1, 10)])
    parser.add_argument("--fit-role", choices=ROLES)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--crossfit-folds", type=int, default=2)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--num-bridges", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--early-patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-pairs", type=int, default=2048)
    return parser.parse_args()


def _load_role_indices(fid_dir: Path, recording: str) -> dict[str, np.ndarray]:
    path = fid_dir / "splits" / f"{recording}_mixed_half_split.npz"
    with np.load(path, allow_pickle=False) as archive:
        roles = {
            "reference": np.asarray(archive["reference_indices"], dtype=np.int64),
            "all_trial": np.asarray(archive["all_trial_indices"], dtype=np.int64),
        }
    if np.intersect1d(roles["reference"], roles["all_trial"]).size:
        raise RuntimeError(f"Reference/query leakage in {path}.")
    return roles


def _class_folds(
    role_indices: np.ndarray,
    labels: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> dict[int, list[np.ndarray]]:
    if folds < 2:
        raise ValueError("crossfit_folds must be >= 2.")
    output: dict[int, list[np.ndarray]] = {}
    for condition in range(4):
        indices = role_indices[labels[role_indices] == condition].copy()
        rng = np.random.default_rng(seed + 1009 * condition)
        rng.shuffle(indices)
        split = [np.asarray(part, dtype=np.int64) for part in np.array_split(indices, folds)]
        if any(part.size < 1 for part in split):
            raise ValueError(f"Too few condition-{condition} trials for {folds} folds.")
        output[condition] = split
    return output


def _train_validation_indices(
    folds: list[np.ndarray],
    held_out_fold: int,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    outer_train = np.concatenate(
        [part for fold_index, part in enumerate(folds) if fold_index != held_out_fold]
    )
    rng = np.random.default_rng(seed)
    shuffled = outer_train.copy()
    rng.shuffle(shuffled)
    n_validation = max(2, int(round(validation_fraction * shuffled.size)))
    n_validation = min(n_validation, shuffled.size - 2)
    return shuffled[n_validation:], shuffled[:n_validation]


def _config(args: argparse.Namespace) -> TREDensityRatioConfig:
    return TREDensityRatioConfig(
        num_bridges=args.num_bridges,
        waymark_schedule="angle",
        architecture="mlp",
        hidden_dim=128,
        depth=3,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=1e-3,
        weight_decay=0.0,
        early_patience=args.early_patience,
        early_min_delta=1e-5,
        max_grad_norm=10.0,
        validation_pairs=args.validation_pairs,
        standardize=True,
        log_every=max(1, min(50, args.epochs // 10)),
    )


def _fit_role(
    args: argparse.Namespace,
    recording: str,
    role: str,
    device: torch.device,
) -> None:
    cache_dir = args.output_dir / "rdm_cache"
    checkpoint_dir = args.output_dir / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{role}_{recording}_tre_jeffreys.npz"
    expected_checkpoints = args.crossfit_folds * 6
    checkpoint_glob = list(checkpoint_dir.glob(f"{role}_{recording}_pair*_fold*_tre_best.pt"))
    if cache_path.is_file() and len(checkpoint_glob) == expected_checkpoints:
        print(f"[TRE cache] loaded {cache_path.name}", flush=True)
        return

    dataset = load_features_npz(args.feature_dir / f"{recording}.npz")
    labels = np.asarray(dataset.labels, dtype=np.int64)
    features = np.asarray(dataset.features, dtype=np.float32)
    times = np.asarray(dataset.time_centers, dtype=np.float32)
    role_indices = _load_role_indices(args.fid_dir, recording)[role]
    recording_index = list(args.recordings).index(recording)
    role_index = ROLES.index(role)
    base_seed = args.seed + recording_index * 1_000_000 + role_index * 100_000
    folds = _class_folds(
        role_indices, labels, folds=args.crossfit_folds, seed=base_seed
    )
    config = _config(args)
    rdms = np.zeros((times.size, 4, 4), dtype=np.float64)
    raw_rdms = np.zeros_like(rdms)
    pair_summaries: dict[str, dict] = {}

    for condition_i in range(4):
        for condition_j in range(condition_i + 1, 4):
            sum_i = np.zeros(times.size, dtype=np.float64)
            sum_j = np.zeros(times.size, dtype=np.float64)
            count_i = 0
            count_j = 0
            fold_summaries: list[dict] = []
            pair_number = condition_i * 4 + condition_j
            for fold_index in range(args.crossfit_folds):
                pair_seed = base_seed + pair_number * 10_000 + fold_index * 100
                train_i, validation_i = _train_validation_indices(
                    folds[condition_i],
                    fold_index,
                    validation_fraction=args.validation_fraction,
                    seed=pair_seed + 1,
                )
                train_j, validation_j = _train_validation_indices(
                    folds[condition_j],
                    fold_index,
                    validation_fraction=args.validation_fraction,
                    seed=pair_seed + 2,
                )
                eval_i = folds[condition_i][fold_index]
                eval_j = folds[condition_j][fold_index]
                print(
                    f"[TRE fit] {recording} {role} pair={condition_i}_{condition_j} "
                    f"fold={fold_index + 1}/{args.crossfit_folds}",
                    flush=True,
                )
                model, training = train_time_conditioned_tre_density_ratio(
                    x0_train=features[train_i],
                    x1_train=features[train_j],
                    x0_validation=features[validation_i],
                    x1_validation=features[validation_j],
                    times=times,
                    device=device,
                    seed=pair_seed,
                    config=config,
                )
                ratio_i = evaluate_time_conditioned_log_ratio(
                    model, features[eval_i], times, device=device
                )
                ratio_j = evaluate_time_conditioned_log_ratio(
                    model, features[eval_j], times, device=device
                )
                sum_i += ratio_i.sum(axis=0)
                sum_j += -ratio_j.sum(axis=0)
                count_i += ratio_i.shape[0]
                count_j += ratio_j.shape[0]
                checkpoint_path = checkpoint_dir / (
                    f"{role}_{recording}_pair{condition_i}_{condition_j}_"
                    f"fold{fold_index:02d}_tre_best.pt"
                )
                torch.save(
                    {
                        "state_dict": {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        },
                        "model": {
                            "input_dim": 23,
                            "num_bridges": config.num_bridges,
                            "architecture": config.architecture,
                            "hidden_dim": config.hidden_dim,
                            "depth": config.depth,
                        },
                        "training": {
                            "best_epoch": training.best_epoch,
                            "best_validation_loss": training.best_validation_loss,
                            "stopped_epoch": training.stopped_epoch,
                            "train_losses": training.train_losses,
                            "validation_losses": training.validation_losses,
                        },
                        "context": {
                            "recording": recording,
                            "role": role,
                            "condition_i": condition_i,
                            "condition_j": condition_j,
                            "fold": fold_index,
                            "train_i": train_i,
                            "train_j": train_j,
                            "validation_i": validation_i,
                            "validation_j": validation_j,
                            "evaluation_i": eval_i,
                            "evaluation_j": eval_j,
                            "seed": pair_seed,
                        },
                        "config": asdict(config),
                    },
                    checkpoint_path,
                )
                fold_summaries.append(
                    {
                        "fold": fold_index,
                        "best_epoch": training.best_epoch,
                        "stopped_epoch": training.stopped_epoch,
                        "best_validation_loss": training.best_validation_loss,
                        "training_seconds": training.training_seconds,
                    }
                )
            directed_i_j = sum_i / count_i
            directed_j_i = sum_j / count_j
            raw = directed_i_j + directed_j_i
            symmetric = np.maximum(0.0, raw)
            raw_rdms[:, condition_i, condition_j] = raw
            raw_rdms[:, condition_j, condition_i] = raw
            rdms[:, condition_i, condition_j] = symmetric
            rdms[:, condition_j, condition_i] = symmetric
            pair_summaries[f"{condition_i}_{condition_j}"] = {
                "n_evaluation_i": count_i,
                "n_evaluation_j": count_j,
                "folds": fold_summaries,
            }

    metadata = {
        "recording": recording,
        "role": role,
        "method": "two_fold_cross_fitted_time_conditioned_tre",
        "distance": "jeffreys_symmetric_kl_clipped_at_zero",
        "time_conditioning": "same_eeg_time_for_both_waymark_endpoints",
        "n_role_trials": int(role_indices.size),
        "per_class_counts": np.bincount(labels[role_indices], minlength=4).tolist(),
        "crossfit_folds": args.crossfit_folds,
        "config": asdict(config),
        "pair_summaries": pair_summaries,
    }
    np.savez_compressed(
        cache_path,
        rdms=rdms,
        raw_rdms=raw_rdms,
        times=times,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"[TRE cache] saved {cache_path.resolve()}", flush=True)


def _aggregate(args: argparse.Namespace) -> None:
    upper = np.triu_indices(4, 1)
    recordings = list(args.recordings)
    reference: dict[str, np.ndarray] = {}
    query: dict[str, np.ndarray] = {}
    times = None
    for recording in recordings:
        for role, target in (("reference", reference), ("all_trial", query)):
            path = args.output_dir / "rdm_cache" / f"{role}_{recording}_tre_jeffreys.npz"
            with np.load(path, allow_pickle=False) as archive:
                target[recording] = np.asarray(archive["rdms"], dtype=np.float64)
                current_times = np.asarray(archive["times"], dtype=np.float64)
            if times is None:
                times = current_times
            elif not np.array_equal(times, current_times):
                raise RuntimeError(f"Time mismatch in {path}.")
    assert times is not None
    mask = (times >= VISIBLE_CUE_INTERVAL[0]) & (times < VISIBLE_CUE_INTERVAL[1])
    mse = np.empty((len(recordings), len(recordings)), dtype=np.float64)
    for query_index, query_recording in enumerate(recordings):
        query_vector = query[query_recording][mask][:, upper[0], upper[1]]
        for reference_index, reference_recording in enumerate(recordings):
            reference_vector = reference[reference_recording][mask][:, upper[0], upper[1]]
            mse[query_index, reference_index] = np.mean(
                np.square(query_vector - reference_vector)
            )
    prediction_indices = np.argmin(mse, axis=1)
    accuracy = float(np.mean(prediction_indices == np.arange(len(recordings))))
    np.savez_compressed(
        args.output_dir / "tre_jeffreys_session_identification.npz",
        mse=mse,
        top1_accuracy=np.asarray(accuracy),
        predictions=np.asarray([recordings[index] for index in prediction_indices]),
        recordings=np.asarray(recordings),
        selected_times=times[mask],
    )
    summary = {
        "top1_accuracy": accuracy,
        "top1_count": int(round(accuracy * len(recordings))),
        "recordings": recordings,
        "predictions": [recordings[index] for index in prediction_indices],
        "matching": "mean squared error over six RDM entries and visible-cue times",
        "visible_cue_interval_seconds": list(VISIBLE_CUE_INTERVAL),
        "n_selected_time_points": int(mask.sum()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    baseline_path = (
        args.output_dir.parent / "multi_distance_full_query_9recordings_results.npz"
    )
    with np.load(baseline_path, allow_pickle=False) as archive:
        baseline_top1 = np.asarray(archive["top1_accuracy"], dtype=np.float64)
        metrics = list(archive["metrics"].astype(str))
    jeffreys_index = metrics.index("jeffreys")
    accuracies = np.asarray(
        [baseline_top1[0, jeffreys_index], baseline_top1[1, jeffreys_index], accuracy]
    )
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
    figure, axis = plt.subplots(figsize=(4.0, 3.5))
    positions = np.arange(3)
    axis.bar(
        positions,
        accuracies,
        width=0.82,
        color=["#4477AA", "#CC6677", "#228833"],
    )
    axis.set_xticks(positions, ["Classical", "Flow", "TRE"])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Top-1 accuracy")
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(width=1.8)
    figure.savefig(args.output_dir / "jeffreys_three_method_top1_bar.png", dpi=300)
    figure.savefig(args.output_dir / "jeffreys_three_method_top1_bar.svg")
    plt.close(figure)
    print(f"[TRE result] top1={accuracy:.3f} ({summary['top1_count']}/9)", flush=True)


def main() -> None:
    args = parse_args()
    if args.device != "cuda:0":
        raise ValueError("Project experiments must use --device cuda:0.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing CPU fallback.")
    torch.cuda.set_device(0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[TRE experiment] device={args.device} GPU={torch.cuda.get_device_name(0)} "
        f"epochs={args.epochs} patience={args.early_patience}",
        flush=True,
    )
    if not args.aggregate_only:
        recordings = list(args.recordings) if args.fit_recording is None else [args.fit_recording]
        roles = list(ROLES) if args.fit_role is None else [args.fit_role]
        for recording in recordings:
            for role in roles:
                _fit_role(args, recording, role, torch.device(args.device))
    if args.fit_recording is None and args.fit_role is None:
        _aggregate(args)


if __name__ == "__main__":
    main()
