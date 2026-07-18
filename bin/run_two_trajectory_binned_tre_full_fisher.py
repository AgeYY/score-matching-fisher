#!/usr/bin/env python3
"""Estimate two-trajectory full Fisher with adjacent-bin TRE-8."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.shared_fisher_est import require_device
from fisher.tre_distance import TREDensityRatioConfig, train_and_estimate_binned_tre_fisher
from global_setting import DEFAULT_EARLY_STOPPING_PATIENCE, DEFAULT_TRAINING_MAX_EPOCHS


RESULTS_NAME = "two_trajectory_binned_tre_full_fisher_results.npz"
SUMMARY_NAME = "two_trajectory_binned_tre_full_fisher_summary.json"
CHECKPOINT_NAME = "two_trajectory_binned_tre_full_fisher_models.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-bridges", type=int, default=8)
    parser.add_argument("--waymark-schedule", choices=("angle", "linear_alpha"), default="angle")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--validation-pairs", type=int, default=2_048)
    parser.add_argument("--min-window-samples", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = require_device(str(args.device))
    case_dir = args.case_dir.expanduser().resolve()
    dataset_path = case_dir / "two_trajectory_dataset.npz"
    flow_results_path = case_dir / "two_trajectory_full_fisher_results.npz"
    if not dataset_path.is_file() or not flow_results_path.is_file():
        raise FileNotFoundError(f"Missing flow case inputs under {case_dir}.")
    started = time.perf_counter()
    with np.load(dataset_path, allow_pickle=False) as data:
        theta_all = np.asarray(data["theta_all"], dtype=np.float64)
        x_all = np.asarray(data["x_all"], dtype=np.float32)
        train_index = np.asarray(data["train_index"], dtype=np.int64)
        validation_index = np.asarray(data["validation_index"], dtype=np.int64)
    with np.load(flow_results_path, allow_pickle=False) as data:
        theta_grid = np.asarray(data["theta_grid"], dtype=np.float64)
        theta_midpoints = np.asarray(data["theta_midpoints"], dtype=np.float64)
        truth = np.asarray(data["ground_truth_full_fisher"], dtype=np.float64)

    config = TREDensityRatioConfig(
        num_bridges=int(args.num_bridges),
        waymark_schedule=str(args.waymark_schedule),
        architecture="mlp",
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.learning_rate),
        weight_decay=0.0,
        early_patience=int(args.early_patience),
        early_min_delta=1e-5,
        max_grad_norm=10.0,
        validation_pairs=int(args.validation_pairs),
        standardize=True,
        log_every=1_000,
    )
    states, result = train_and_estimate_binned_tre_fisher(
        theta_train=theta_all[train_index],
        x_train=x_all[train_index],
        theta_validation=theta_all[validation_index],
        x_validation=x_all[validation_index],
        theta_eval=theta_all,
        x_eval=x_all,
        theta_grid=theta_grid,
        device=device,
        seed=int(args.seed),
        config=config,
        min_train_samples=int(args.min_window_samples),
        min_validation_samples=int(args.min_window_samples),
        min_eval_samples=int(args.min_window_samples),
    )
    results_path = case_dir / RESULTS_NAME
    np.savez_compressed(
        results_path,
        theta_grid=theta_grid,
        theta_midpoints=theta_midpoints,
        ground_truth_full_fisher=truth,
        tre_full_fisher=result.fisher,
        tre_jeffreys=result.jeffreys,
        tre_raw_jeffreys=result.raw_jeffreys,
    )
    checkpoint_path = case_dir / CHECKPOINT_NAME
    torch.save(
        {
            "pair_state_dicts": states,
            "pair_metadata": result.pair_metadata,
            "run_metadata": result.run_metadata,
        },
        checkpoint_path,
    )
    summary = {
        "case_dir": str(case_dir),
        "seed": int(args.seed),
        "estimator": "adjacent-bin TRE",
        "num_bridges": int(args.num_bridges),
        "fisher_conversion": "max(0, Jeffreys) / theta_spacing^2",
        "evaluation_data": "all observations in each local theta window",
        "flow_case_train_validation_split_reused": True,
        "tre_mae": float(np.mean(np.abs(result.fisher - truth))),
        "runtime_seconds": float(time.perf_counter() - started),
        "run_metadata": result.run_metadata,
        "results_npz": str(results_path),
        "checkpoint": str(checkpoint_path),
    }
    summary_path = case_dir / SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
