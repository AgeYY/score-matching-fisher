#!/usr/bin/env python3
"""Temporary isolated GKR Linear Fisher run for a fixed-size toy dataset."""

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
N_TOTAL = 10_000
LARGE_RUN_MEAN_BATCH_SIZE = 3000
TRAIN_FRAC = 0.8
SEED = 7
DEFAULT_THETA_SPACING = 0.4
DATASET_FAMILY = "randamp_gaussian_sqrtd"
DEFAULT_COVARIANCE_ALPHA = 0.65


def _spacing_suffix(spacing: float) -> str:
    if np.isclose(float(spacing), DEFAULT_THETA_SPACING):
        return ""
    return "_h" + f"{float(spacing):g}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--x-dim", type=int, default=X_DIM)
    parser.add_argument("--n-total", type=int, default=N_TOTAL)
    parser.add_argument("--dataset-seed", type=int, default=SEED)
    parser.add_argument("--training-seed", type=int, default=None)
    parser.add_argument("--theta-spacing", type=float, default=DEFAULT_THETA_SPACING)
    parser.add_argument(
        "--covariance-alpha",
        type=float,
        default=DEFAULT_COVARIANCE_ALPHA,
        help="Effective mean covariance-variation amplitude for the dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--force-dataset", action="store_true")
    return parser.parse_args()


def _plot(
    *,
    theta: np.ndarray,
    ground_truth: np.ndarray,
    estimate: np.ndarray,
    x_dim: int,
    n_total: int,
    output_stem: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 16,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    mae = float(np.mean(np.abs(estimate - ground_truth)))
    fig, axis = plt.subplots(figsize=(5.2, 3.8))
    axis.plot(
        theta,
        ground_truth,
        color="black",
        linestyle="--",
        linewidth=2.5,
        label="Ground truth",
    )
    axis.plot(theta, estimate, color="C2", linewidth=2.3, label="GKR")
    axis.set_xlabel(r"$\theta$")
    axis.set_ylabel("Linear Fisher information")
    axis.set_title(f"{int(x_dim)}D, {int(n_total):,} samples (MAE={mae:.3f})")
    axis.legend(frameon=False, loc="best")
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(width=1.8)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    x_dim = int(args.x_dim)
    n_total = int(args.n_total)
    dataset_seed = int(args.dataset_seed)
    training_seed = dataset_seed if args.training_seed is None else int(args.training_seed)
    theta_spacing = float(args.theta_spacing)
    covariance_alpha = float(args.covariance_alpha)
    if x_dim < 1:
        raise ValueError("--x-dim must be at least 1.")
    if n_total < 2:
        raise ValueError("--n-total must be at least 2.")
    if not np.isfinite(theta_spacing) or theta_spacing <= 0.0:
        raise ValueError("--theta-spacing must be finite and positive.")
    if not np.isfinite(covariance_alpha) or covariance_alpha <= 0.0:
        raise ValueError("--covariance-alpha must be finite and positive.")
    cov_theta_amp_scale = covariance_alpha / DEFAULT_COVARIANCE_ALPHA
    device = require_device(str(args.device))
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else REPO_ROOT
        / "data"
        / (
            f"gkr_fixed_xdim{x_dim}_n{n_total}_linear"
            + ("" if dataset_seed == SEED else f"_datasetseed{dataset_seed}")
        )
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = (
        f"gkr_fixed_xdim{x_dim}_n{n_total}_linear"
        f"{_spacing_suffix(theta_spacing)}"
    )
    dataset_npz = output_dir / f"randamp_gaussian_sqrtd_xdim{x_dim}_n{n_total}.npz"

    make_native_dataset_npz(
        output_npz=dataset_npz,
        dataset_family=DATASET_FAMILY,
        x_dim=x_dim,
        n_total=n_total,
        train_frac=TRAIN_FRAC,
        seed=dataset_seed,
        cov_theta_amp_scale=cov_theta_amp_scale,
        force=bool(args.force_dataset),
    )
    bundle = load_shared_dataset_npz(dataset_npz)
    theta_span = float(bundle.meta["theta_high"]) - float(bundle.meta["theta_low"])
    n_intervals = int(round(theta_span / theta_spacing))
    if n_intervals < 1 or not np.isclose(
        theta_span / float(n_intervals), theta_spacing, rtol=1e-8, atol=1e-10
    ):
        raise ValueError("--theta-spacing must evenly divide the dataset theta range.")
    theta_grid_size = n_intervals + 1
    theta_grid = theta_grid_from_meta(bundle.meta, theta_grid_size=theta_grid_size)
    query = theta_midpoints(theta_grid)
    separation = np.diff(theta_grid, axis=0)
    truth = native_ground_truth_curves(query, dict(bundle.meta))

    config = GKRConfig(
        mean_iterations=300,
        mean_learning_rate=0.05,
        mean_batch_size=(LARGE_RUN_MEAN_BATCH_SIZE if n_total > 10_000 else None),
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
        seed=training_seed,
    )
    model.fit(bundle.x_train, bundle.theta_train)
    gkr = estimate_gkr_linear_fisher(
        model,
        query,
        finite_difference_step=separation,
        solve_jitter=1e-6,
    )
    linear_truth = np.asarray(truth[METHOD_GT_NATIVE_LINEAR], dtype=np.float64)
    linear_mae = float(np.mean(np.abs(gkr.linear_fisher - linear_truth)))

    results_npz = output_dir / f"{artifact_stem}_results.npz"
    np.savez_compressed(
        results_npz,
        theta_grid=theta_grid,
        theta_midpoints=query,
        gkr_finite_difference_step=separation,
        ground_truth_linear_fisher=linear_truth,
        gkr_linear_fisher=gkr.linear_fisher,
        gkr_mean=gkr.mean,
        gkr_covariance=gkr.covariance,
        gkr_mean_jacobian=gkr.mean_jacobian,
        gkr_mean_loss=gkr.mean_loss,
        gkr_covariance_loss=gkr.covariance_loss,
    )

    figure_stem = output_dir / f"{artifact_stem}_fisher"
    _plot(
        theta=query[:, 0],
        ground_truth=linear_truth,
        estimate=gkr.linear_fisher,
        x_dim=x_dim,
        n_total=n_total,
        output_stem=figure_stem,
    )
    summary = {
        "dataset_family": DATASET_FAMILY,
        "x_dim": x_dim,
        "n_total": n_total,
        "n_train": int(bundle.x_train.shape[0]),
        "n_validation": int(bundle.x_validation.shape[0]),
        "dataset_seed": dataset_seed,
        "gkr_training_seed": training_seed,
        "device": str(device),
        "theta_grid_size": theta_grid_size,
        "theta_spacing": theta_spacing,
        "theta_grid_separation": separation[:, 0].tolist(),
        "covariance_alpha": covariance_alpha,
        "cov_theta_amp_scale": cov_theta_amp_scale,
        "gkr_config": asdict(config),
        "linear_fisher_mae": linear_mae,
        "dataset_npz": str(dataset_npz),
        "results_npz": str(results_npz),
        "figure_png": str(figure_stem.with_suffix(".png")),
        "figure_svg": str(figure_stem.with_suffix(".svg")),
    }
    summary_path = output_dir / f"{artifact_stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
