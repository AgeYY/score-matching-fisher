#!/usr/bin/env python3
"""Fit GKR and append its linear-Fisher estimate to a cached comparison result."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.gkr import GKRConfig, TorchGKR, estimate_gkr_linear_fisher
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import require_device
from global_setting import DEFAULT_DEVICE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--result-npz", type=Path, required=True)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mean-iterations", type=int, default=300)
    parser.add_argument("--mean-lr", type=float, default=0.05)
    parser.add_argument("--n-inducing", type=int, default=200)
    parser.add_argument("--covariance-epochs", type=int, default=30)
    parser.add_argument("--covariance-lr", type=float, default=0.1)
    parser.add_argument("--covariance-batch-size", type=int, default=3000)
    parser.add_argument("--validation-fraction", type=float, default=0.33)
    parser.add_argument("--covariance-jitter", type=float, default=1e-6)
    parser.add_argument("--likelihood-jitter", type=float, default=1e-5)
    parser.add_argument("--prediction-batch-size", type=int, default=3000)
    parser.add_argument("--solve-jitter", type=float, default=1e-6)
    parser.add_argument("--log-every", type=int, default=25)
    return parser


def _config(args: argparse.Namespace) -> GKRConfig:
    return GKRConfig(
        mean_iterations=args.mean_iterations,
        mean_learning_rate=args.mean_lr,
        n_inducing=args.n_inducing,
        covariance_epochs=args.covariance_epochs,
        covariance_learning_rate=args.covariance_lr,
        covariance_batch_size=args.covariance_batch_size,
        validation_fraction=args.validation_fraction,
        covariance_jitter=args.covariance_jitter,
        likelihood_jitter=args.likelihood_jitter,
        prediction_batch_size=args.prediction_batch_size,
        log_every=args.log_every,
    )


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, object]:
    dataset_npz = Path(args.dataset_npz).expanduser()
    result_npz = Path(args.result_npz).expanduser()
    if not dataset_npz.is_file():
        raise FileNotFoundError(dataset_npz)
    if not result_npz.is_file():
        raise FileNotFoundError(result_npz)
    with np.load(result_npz, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    for key in (
        "theta_grid",
        "theta_midpoints",
        "ground_truth_native_linear_fisher",
        "ground_truth_native_full_fisher",
    ):
        if key not in arrays:
            raise KeyError(f"Cached result does not contain {key!r}: {result_npz}")
    theta_grid = np.asarray(arrays["theta_grid"], dtype=np.float64).reshape(-1, 1)
    theta_midpoints = np.asarray(arrays["theta_midpoints"], dtype=np.float64).reshape(-1, 1)
    expected_midpoints = 0.5 * (theta_grid[:-1] + theta_grid[1:])
    if theta_grid.shape[1] != 1 or theta_grid.shape[0] < 2:
        raise ValueError("Continuous Fisher GKR requires a scalar theta grid with at least two points.")
    if theta_midpoints.shape != expected_midpoints.shape or not np.allclose(
        theta_midpoints, expected_midpoints, rtol=1e-12, atol=1e-12
    ):
        raise ValueError("theta_midpoints must be the adjacent theta-grid midpoints.")
    grid_separation = np.diff(theta_grid, axis=0)
    cached_separation = np.asarray(
        arrays.get("gkr_finite_difference_step", np.asarray([])), dtype=np.float64
    ).reshape(-1, 1)
    cache_matches_grid = cached_separation.shape == grid_separation.shape and np.allclose(
        cached_separation, grid_separation, rtol=1e-12, atol=1e-12
    )
    if (
        "gkr_linear_fisher" in arrays
        and "gkr_full_fisher" in arrays
        and cache_matches_grid
        and not args.force
    ):
        print(f"[gkr-cache] existing estimate retained: {result_npz}", flush=True)
        return {
            "linear": {
                "mae": float(np.mean(np.asarray(arrays["gkr_linear_abs_error"]))),
                "mean_relative_absolute_error": float(
                    np.mean(np.asarray(arrays["gkr_linear_rel_error"]))
                ),
            },
            "full": {
                "mae": float(np.mean(np.asarray(arrays["gkr_full_abs_error"]))),
                "mean_relative_absolute_error": float(
                    np.mean(np.asarray(arrays["gkr_full_rel_error"]))
                ),
            },
        }
    device = require_device(str(args.device))
    bundle = load_shared_dataset_npz(dataset_npz)
    config = _config(args)
    model = TorchGKR(
        n_input=bundle.theta_train.shape[1],
        n_output=bundle.x_train.shape[1],
        config=config,
        dtype=torch.float64,
        device=device,
        seed=args.seed,
    )
    model.fit(bundle.x_train, bundle.theta_train)
    estimate = estimate_gkr_linear_fisher(
        model,
        theta_midpoints,
        finite_difference_step=grid_separation,
        solve_jitter=args.solve_jitter,
    )
    linear_truth = np.asarray(arrays["ground_truth_native_linear_fisher"], dtype=np.float64)
    full_truth = np.asarray(arrays["ground_truth_native_full_fisher"], dtype=np.float64)
    linear_absolute_error = np.abs(estimate.linear_fisher - linear_truth)
    linear_relative_error = linear_absolute_error / np.maximum(np.abs(linear_truth), 1e-12)
    full_absolute_error = np.abs(estimate.full_fisher - full_truth)
    full_relative_error = full_absolute_error / np.maximum(np.abs(full_truth), 1e-12)
    arrays.update(
        {
            "gkr_linear_fisher": estimate.linear_fisher,
            "gkr_linear_abs_error": linear_absolute_error,
            "gkr_linear_rel_error": linear_relative_error,
            "gkr_covariance_fisher": estimate.covariance_fisher,
            "gkr_full_fisher": estimate.full_fisher,
            "gkr_full_abs_error": full_absolute_error,
            "gkr_full_rel_error": full_relative_error,
            "gkr_mean": estimate.mean,
            "gkr_covariance": estimate.covariance,
            "gkr_mean_jacobian": estimate.mean_jacobian,
            "gkr_covariance_jacobian": estimate.covariance_jacobian,
            "gkr_fisher_matrix": estimate.fisher_matrix,
            "gkr_covariance_fisher_matrix": estimate.covariance_fisher_matrix,
            "gkr_full_fisher_matrix": estimate.full_fisher_matrix,
            "gkr_mean_loss": estimate.mean_loss,
            "gkr_covariance_loss": estimate.covariance_loss,
            "gkr_finite_difference_step": grid_separation,
        }
    )
    _atomic_save_npz(result_npz, arrays)
    metrics = {
        "linear": {
            "mae": float(np.mean(linear_absolute_error)),
            "mean_relative_absolute_error": float(np.mean(linear_relative_error)),
        },
        "full": {
            "mae": float(np.mean(full_absolute_error)),
            "mean_relative_absolute_error": float(np.mean(full_relative_error)),
        },
    }
    if result_npz.name == "continuous_pr_fisher_results.npz":
        metadata_path = result_npz.parent / "gkr_fisher_summary.json"
    else:
        metadata_path = result_npz.with_name(f"{result_npz.stem}_gkr_summary.json")
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_npz": str(dataset_npz),
                "result_npz": str(result_npz),
                "device": str(device),
                "seed": args.seed,
                "n_train": int(bundle.x_train.shape[0]),
                "config": asdict(config),
                "finite_difference_step": grid_separation.reshape(-1).tolist(),
                "finite_difference_source": "adjacent_theta_grid_spacing",
                "solve_jitter": args.solve_jitter,
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"updated_result_npz: {result_npz}", flush=True)
    print(f"gkr_summary_json: {metadata_path}", flush=True)
    return metrics


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
