#!/usr/bin/env python3
"""Temporary isolated GKR run for the 100D, N=5000 continuous toy dataset."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.continuous_fisher_comparison import (
    METHOD_GT_NATIVE_FULL,
    METHOD_GT_NATIVE_LINEAR,
    make_native_dataset_npz,
    native_ground_truth_curves,
    theta_grid_from_meta,
    theta_midpoints,
)
from fisher.gkr import GKRConfig, TorchGKR, estimate_gkr_linear_fisher
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import require_device


X_DIM = 100
N_TOTAL = 5_000
TRAIN_FRAC = 0.8
SEED = 7
THETA_GRID_SIZE = 31
DATASET_FAMILY = "randamp_gaussian_sqrtd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "gkr_fixed_xdim100_n5000",
    )
    parser.add_argument("--force-dataset", action="store_true")
    return parser.parse_args()


def _plot(
    *,
    theta: np.ndarray,
    ground_truth_linear: np.ndarray,
    ground_truth_full: np.ndarray,
    gkr_linear: np.ndarray,
    gkr_full: np.ndarray,
    output_stem: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), sharex=True)
    panels = (
        (ground_truth_linear, gkr_linear, "Linear Fisher"),
        (ground_truth_full, gkr_full, "Full Fisher"),
    )
    for axis, (truth, estimate, title) in zip(axes, panels, strict=True):
        mae = float(np.mean(np.abs(estimate - truth)))
        axis.plot(theta, truth, color="black", linestyle="--", linewidth=2.4, label="Ground truth")
        axis.plot(theta, estimate, color="C2", linewidth=2.2, label="GKR")
        axis.set_xlabel(r"$\theta$")
        axis.set_ylabel("Fisher information")
        axis.set_title(f"{title} (MAE={mae:.3f})")
        for spine in axis.spines.values():
            spine.set_linewidth(1.8)
        axis.tick_params(width=1.8)
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = require_device(str(args.device))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_npz = output_dir / "randamp_gaussian_sqrtd_xdim100_n5000.npz"

    make_native_dataset_npz(
        output_npz=dataset_npz,
        dataset_family=DATASET_FAMILY,
        x_dim=X_DIM,
        n_total=N_TOTAL,
        train_frac=TRAIN_FRAC,
        seed=SEED,
        force=bool(args.force_dataset),
    )
    bundle = load_shared_dataset_npz(dataset_npz)
    theta_grid = theta_grid_from_meta(bundle.meta, theta_grid_size=THETA_GRID_SIZE)
    query = theta_midpoints(theta_grid)
    separation = np.diff(theta_grid, axis=0)
    truth = native_ground_truth_curves(query, dict(bundle.meta))

    config = GKRConfig(
        mean_iterations=300,
        mean_learning_rate=0.05,
        n_inducing=200,
        covariance_epochs=30,
        covariance_learning_rate=0.1,
        covariance_batch_size=3000,
        validation_fraction=0.33,
        covariance_jitter=1e-6,
        likelihood_jitter=1e-5,
        prediction_batch_size=3000,
        standardize_responses=True,
        log_every=25,
    )
    model = TorchGKR(
        n_input=bundle.theta_train.shape[1],
        n_output=bundle.x_train.shape[1],
        config=config,
        dtype=torch.float64,
        device=device,
        seed=SEED,
    )
    model.fit(bundle.x_train, bundle.theta_train)
    estimate = estimate_gkr_linear_fisher(
        model,
        query,
        finite_difference_step=separation,
        solve_jitter=1e-6,
    )

    linear_truth = np.asarray(truth[METHOD_GT_NATIVE_LINEAR], dtype=np.float64)
    full_truth = np.asarray(truth[METHOD_GT_NATIVE_FULL], dtype=np.float64)
    results_npz = output_dir / "gkr_fixed_xdim100_n5000_results.npz"
    np.savez_compressed(
        results_npz,
        theta_grid=theta_grid,
        theta_midpoints=query,
        gkr_finite_difference_step=separation,
        ground_truth_linear_fisher=linear_truth,
        ground_truth_full_fisher=full_truth,
        gkr_linear_fisher=estimate.linear_fisher,
        gkr_covariance_fisher=estimate.covariance_fisher,
        gkr_full_fisher=estimate.full_fisher,
        gkr_mean=estimate.mean,
        gkr_covariance=estimate.covariance,
        gkr_mean_jacobian=estimate.mean_jacobian,
        gkr_covariance_jacobian=estimate.covariance_jacobian,
        gkr_mean_loss=estimate.mean_loss,
        gkr_covariance_loss=estimate.covariance_loss,
    )

    figure_stem = output_dir / "gkr_fixed_xdim100_n5000_fisher_curves"
    _plot(
        theta=query[:, 0],
        ground_truth_linear=linear_truth,
        ground_truth_full=full_truth,
        gkr_linear=estimate.linear_fisher,
        gkr_full=estimate.full_fisher,
        output_stem=figure_stem,
    )
    summary = {
        "dataset_family": DATASET_FAMILY,
        "x_dim": X_DIM,
        "n_total": N_TOTAL,
        "n_train": int(bundle.x_train.shape[0]),
        "n_validation": int(bundle.x_validation.shape[0]),
        "seed": SEED,
        "device": str(device),
        "theta_grid_size": THETA_GRID_SIZE,
        "theta_grid_separation": separation[:, 0].tolist(),
        "gkr_config": asdict(config),
        "linear_fisher_mae": float(np.mean(np.abs(estimate.linear_fisher - linear_truth))),
        "full_fisher_mae": float(np.mean(np.abs(estimate.full_fisher - full_truth))),
        "dataset_npz": str(dataset_npz),
        "results_npz": str(results_npz),
        "figure_png": str(figure_stem.with_suffix(".png")),
        "figure_svg": str(figure_stem.with_suffix(".svg")),
    }
    summary_path = output_dir / "gkr_fixed_xdim100_n5000_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
