#!/usr/bin/env python3
"""Fit full Fisher information for the two-trajectory Gaussian mixture."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.data import ToyConditionalGaussianRandampSqrtdTwoTrajectoryDataset
from fisher.dataset_family_recipes import family_recipe_dict
from fisher.flow_matching_skl import (
    build_flow_skl_model,
    estimate_adjacent_model_jeffreys_fisher,
    train_flow_skl_model,
)
from fisher.shared_fisher_est import require_device
from global_setting import DEFAULT_EARLY_STOPPING_PATIENCE, DEFAULT_TRAINING_MAX_EPOCHS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--x-dim", type=int, default=50)
    parser.add_argument("--n-total", type=int, default=5_000)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=None,
        help="Fixed population seed; defaults to --seed for backward compatibility.",
    )
    parser.add_argument("--covariance-alpha", type=float, default=0.65)
    parser.add_argument("--theta-spacing", type=float, default=0.4)
    parser.add_argument("--gt-samples-per-theta", type=int, default=100_000)
    parser.add_argument("--mc-jeffreys-samples", type=int, default=4_096)
    parser.add_argument("--ode-steps", type=int, default=32)
    parser.add_argument("--hutchinson-probes", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_MAX_EPOCHS)
    parser.add_argument(
        "--early-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "two_trajectory_full_fisher_xdim50_n5000",
    )
    return parser.parse_args()


def build_dataset(*, x_dim: int, seed: int, covariance_alpha: float):
    recipe = family_recipe_dict("randamp_gaussian_sqrtd")
    return ToyConditionalGaussianRandampSqrtdTwoTrajectoryDataset(
        theta_low=-6.0,
        theta_high=6.0,
        x_dim=int(x_dim),
        seed=int(seed),
        sigma_x1=float(recipe["sigma_x1"]),
        sigma_x2=float(recipe["sigma_x2"]),
        cov_theta_amp1=float(covariance_alpha),
        cov_theta_amp2=float(covariance_alpha),
        randamp_mu_low=float(recipe["randamp_mu_low"]),
        randamp_mu_high=float(recipe["randamp_mu_high"]),
        randamp_kappa=float(recipe["randamp_kappa"]),
        randamp_omega=float(recipe["randamp_omega"]),
    )


def ground_truth_full_fisher(
    dataset: ToyConditionalGaussianRandampSqrtdTwoTrajectoryDataset,
    theta: np.ndarray,
    *,
    samples_per_theta: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset.rng = np.random.default_rng(int(seed))
    fisher = np.empty(theta.shape[0], dtype=np.float64)
    standard_error = np.empty_like(fisher)
    for index, value in enumerate(theta[:, 0]):
        condition = np.full((int(samples_per_theta), 1), float(value), dtype=np.float64)
        x = dataset.sample_x(condition)
        squared_score = np.square(dataset.theta_score(x, condition))
        fisher[index] = float(np.mean(squared_score))
        standard_error[index] = float(
            np.std(squared_score, ddof=1) / np.sqrt(float(samples_per_theta))
        )
    return fisher, standard_error


def plot_result(
    *, theta: np.ndarray, truth: np.ndarray, flow: np.ndarray, output_dir: Path
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    mae = float(np.mean(np.abs(flow - truth)))
    fig, axis = plt.subplots(figsize=(4.0, 3.5))
    axis.plot(theta, truth, color="black", linestyle="--", linewidth=2.4, label="Ground truth")
    axis.plot(theta, flow, color="C0", linewidth=2.3, label=f"Flow matching ({mae:.3f})")
    axis.set_xlabel(r"$\theta$")
    axis.set_ylabel("Full Fisher information")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.8)
    axis.spines["bottom"].set_linewidth(1.8)
    axis.tick_params(width=1.8)
    axis.legend(frameon=False, loc="best")
    fig.tight_layout()
    stem = output_dir / "two_trajectory_full_fisher_curve"
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def main() -> int:
    args = parse_args()
    if int(args.x_dim) < 1 or int(args.n_total) < 2:
        raise ValueError("--x-dim must be positive and --n-total must be at least 2.")
    if not 0.0 < float(args.train_frac) < 1.0:
        raise ValueError("--train-frac must be in (0, 1).")
    if float(args.covariance_alpha) < 0.0:
        raise ValueError("--covariance-alpha must be non-negative.")
    device = require_device(str(args.device))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    dataset_seed = int(args.seed) if args.dataset_seed is None else int(args.dataset_seed)

    dataset = build_dataset(
        x_dim=int(args.x_dim), seed=dataset_seed, covariance_alpha=float(args.covariance_alpha)
    )
    dataset.rng = np.random.default_rng(int(args.seed))
    theta_all, x_all = dataset.sample_joint(int(args.n_total))
    split_rng = np.random.default_rng(int(args.seed))
    permutation = split_rng.permutation(int(args.n_total))
    n_train = int(float(args.train_frac) * int(args.n_total))
    train_index, validation_index = permutation[:n_train], permutation[n_train:]
    np.savez_compressed(
        output_dir / "two_trajectory_dataset.npz",
        theta_all=theta_all,
        x_all=x_all,
        train_index=train_index,
        validation_index=validation_index,
    )

    span = float(dataset.theta_high - dataset.theta_low)
    intervals = int(round(span / float(args.theta_spacing)))
    if not np.isclose(span / intervals, float(args.theta_spacing)):
        raise ValueError("--theta-spacing must evenly divide the theta range.")
    theta_grid = np.linspace(dataset.theta_low, dataset.theta_high, intervals + 1)[:, None]
    theta_midpoints = 0.5 * (theta_grid[:-1] + theta_grid[1:])
    truth, truth_se = ground_truth_full_fisher(
        build_dataset(
            x_dim=int(args.x_dim),
            seed=dataset_seed,
            covariance_alpha=float(args.covariance_alpha),
        ),
        theta_midpoints,
        samples_per_theta=int(args.gt_samples_per_theta),
        seed=dataset_seed + 100_000,
    )

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    model = build_flow_skl_model(
        velocity_family="nonlinear",
        theta_dim=1,
        x_dim=int(args.x_dim),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        path_schedule="cosine",
        divergence_estimator="hutchinson",
        hutchinson_probes=int(args.hutchinson_probes),
        theta_embedding="gaussian_rbf",
        theta_rbf_num_centers=8,
        theta_rbf_lower=dataset.theta_low,
        theta_rbf_upper=dataset.theta_high,
        theta_rbf_bandwidth=None,
    ).to(device)
    metadata = train_flow_skl_model(
        model=model,
        theta_train=theta_all[train_index],
        x_train=x_all[train_index],
        theta_val=theta_all[validation_index],
        x_val=x_all[validation_index],
        device=device,
        velocity_family="nonlinear",
        path_schedule="cosine",
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.learning_rate),
        lr_schedule="constant",
        weight_decay=0.0,
        t_eps=5e-4,
        patience=int(args.early_patience),
        min_delta=1e-4,
        ema_alpha=0.05,
        max_grad_norm=10.0,
        log_every=50,
        checkpoint_selection="last",
        best_checkpoint_metric="flow_matching",
        fixed_validation=True,
        fixed_validation_paths=10,
        validation_seed=int(args.seed) + 10_000,
        retain_best_state=True,
    )
    torch.save(model.state_dict(), output_dir / "flow_model_last.pt")
    best_state = metadata.pop("best_state_dict")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / "flow_model_best.pt")

    estimate = estimate_adjacent_model_jeffreys_fisher(
        model=model,
        theta_all=theta_grid,
        device=device,
        mc_jeffreys_sample=int(args.mc_jeffreys_samples),
        ode_steps=int(args.ode_steps),
        ode_method="midpoint",
        batch_size=1_024,
        solve_jitter=1e-6,
        quadrature_steps=64,
    )
    flow = np.asarray(estimate["fisher"], dtype=np.float64)
    figure_png, figure_svg = plot_result(
        theta=theta_midpoints[:, 0], truth=truth, flow=flow, output_dir=output_dir
    )
    results_path = output_dir / "two_trajectory_full_fisher_results.npz"
    np.savez_compressed(
        results_path,
        theta_grid=theta_grid,
        theta_midpoints=theta_midpoints,
        ground_truth_full_fisher=truth,
        ground_truth_standard_error=truth_se,
        flow_full_fisher=flow,
        adjacent_jeffreys=np.asarray(estimate["adjacent_jeffreys"], dtype=np.float64),
        train_losses=np.asarray(metadata["train_losses"], dtype=np.float64),
        validation_losses=np.asarray(metadata["val_losses"], dtype=np.float64),
        validation_monitor_losses=np.asarray(metadata["val_monitor_losses"], dtype=np.float64),
    )
    summary = {
        "dataset": "0.5 N(mu(theta), Sigma(theta)) + 0.5 N(2 mu(theta), Sigma(theta))",
        "x_dim": int(args.x_dim),
        "n_total": int(args.n_total),
        "n_train": int(n_train),
        "n_validation": int(args.n_total) - int(n_train),
        "seed": int(args.seed),
        "dataset_seed": dataset_seed,
        "covariance_alpha": float(args.covariance_alpha),
        "theta_spacing": float(args.theta_spacing),
        "ground_truth": {
            "method": "Monte Carlo expectation of the squared exact latent-mixture condition score",
            "samples_per_theta": int(args.gt_samples_per_theta),
            "maximum_standard_error": float(np.max(truth_se)),
        },
        "flow": {
            "velocity_family": "nonlinear FiLM",
            "theta_embedding": "eight-center Gaussian RBF",
            "hidden_dim": int(args.hidden_dim),
            "depth": int(args.depth),
            "learning_rate": float(args.learning_rate),
            "max_epochs": int(args.epochs),
            "early_stopping_patience": int(args.early_patience),
            "best_epoch": int(metadata["best_epoch"]),
            "stopped_epoch": int(metadata["stopped_epoch"]),
            "mc_jeffreys_samples": int(args.mc_jeffreys_samples),
            "ode_steps": int(args.ode_steps),
            "divergence_estimator": "Hutchinson",
            "hutchinson_probes": int(args.hutchinson_probes),
        },
        "flow_mae": float(np.mean(np.abs(flow - truth))),
        "runtime_seconds": float(time.perf_counter() - started),
        "dataset_npz": str(output_dir / "two_trajectory_dataset.npz"),
        "results_npz": str(results_path),
        "figure_png": str(figure_png),
        "figure_svg": str(figure_svg),
        "best_model": str(output_dir / "flow_model_best.pt"),
        "last_model": str(output_dir / "flow_model_last.pt"),
    }
    summary_path = output_dir / "two_trajectory_full_fisher_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
