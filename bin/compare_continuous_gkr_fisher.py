#!/usr/bin/env python3
"""Estimate linear Fisher information on the continuous toy dataset with GKR."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.continuous_fisher_comparison import (
    classical_linear_fisher,
    make_native_dataset_npz,
    native_linear_fisher_curve,
    theta_grid_from_meta,
    theta_midpoints,
)
from fisher.gkr import GKRConfig, TorchGKR, estimate_gkr_linear_fisher
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import analytic_fisher_curve, build_dataset_from_meta, require_device
from global_setting import DEFAULT_DEVICE

RESULTS_NPZ = "continuous_gkr_fisher_results.npz"
SUMMARY_JSON = "continuous_gkr_fisher_summary.json"
FIGURE_PNG = "continuous_gkr_fisher.png"
FIGURE_SVG = "continuous_gkr_fisher.svg"
UPSTREAM_COMMIT = "4237462f89f44e9a9f0a56a9485c82bf6d37f466"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset-family", default="randamp_gaussian_sqrtd")
    parser.add_argument("--native-x-dim", type=int, default=4)
    parser.add_argument("--n-total", type=int, default=1000)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--theta-grid-size", type=int, default=31)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force-dataset", action="store_true")

    parser.add_argument("--gkr-mean-iterations", type=int, default=300)
    parser.add_argument("--gkr-mean-lr", type=float, default=0.05)
    parser.add_argument("--gkr-n-inducing", type=int, default=200)
    parser.add_argument("--gkr-cov-epochs", type=int, default=30)
    parser.add_argument("--gkr-cov-lr", type=float, default=0.1)
    parser.add_argument("--gkr-cov-batch-size", type=int, default=3000)
    parser.add_argument("--gkr-validation-fraction", type=float, default=0.33)
    parser.add_argument("--gkr-cov-jitter", type=float, default=1e-6)
    parser.add_argument("--gkr-likelihood-jitter", type=float, default=1e-5)
    parser.add_argument("--gkr-prediction-batch-size", type=int, default=3000)
    parser.add_argument("--gkr-solve-jitter", type=float, default=1e-6)
    parser.add_argument("--gkr-log-every", type=int, default=25)
    parser.add_argument("--classical-window-radius", type=float, default=None)
    parser.add_argument("--classical-min-endpoint-samples", type=int, default=8)
    parser.add_argument("--classical-ridge", type=float, default=1e-6)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.dataset_family != "randamp_gaussian_sqrtd":
        raise ValueError("This experiment expects randamp_gaussian_sqrtd.")
    if args.native_x_dim < 1 or args.n_total < 2:
        raise ValueError("native-x-dim must be positive and n-total must be at least two.")
    if not 0.0 < args.train_frac < 1.0:
        raise ValueError("train-frac must be in (0, 1).")
    if args.theta_grid_size < 2:
        raise ValueError("theta-grid-size must be at least two.")
    if args.gkr_n_inducing < 1:
        raise ValueError("gkr-n-inducing must be positive.")
    if not 0.0 < args.gkr_validation_fraction < 1.0:
        raise ValueError("gkr-validation-fraction must be in (0, 1).")


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    dataset_dir = (
        Path(args.dataset_dir).expanduser()
        if args.dataset_dir is not None
        else _REPO_ROOT
        / "data"
        / f"{args.dataset_family}_xdim{args.native_x_dim}_native_n{args.n_total}"
    )
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir is not None
        else dataset_dir / "gkr_fisher"
    )
    dataset_npz = dataset_dir / f"{args.dataset_family}_xdim{args.native_x_dim}_native.npz"
    return dataset_dir, output_dir, dataset_npz


def _config(args: argparse.Namespace) -> GKRConfig:
    return GKRConfig(
        mean_iterations=args.gkr_mean_iterations,
        mean_learning_rate=args.gkr_mean_lr,
        n_inducing=args.gkr_n_inducing,
        covariance_epochs=args.gkr_cov_epochs,
        covariance_learning_rate=args.gkr_cov_lr,
        covariance_batch_size=args.gkr_cov_batch_size,
        validation_fraction=args.gkr_validation_fraction,
        covariance_jitter=args.gkr_cov_jitter,
        likelihood_jitter=args.gkr_likelihood_jitter,
        prediction_batch_size=args.gkr_prediction_batch_size,
        log_every=args.gkr_log_every,
    )


def _error_summary(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = np.asarray(estimate) - np.asarray(truth)
    return {
        "mae": float(np.mean(np.abs(error))),
        "mean_relative_absolute_error": float(
            np.mean(np.abs(error) / np.maximum(np.abs(truth), 1e-12))
        ),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def _plot(
    *,
    theta: np.ndarray,
    truth: np.ndarray,
    truth_full: np.ndarray,
    classical: np.ndarray,
    gkr: np.ndarray,
    gkr_full: np.ndarray,
    mean_loss: np.ndarray,
    covariance_loss: np.ndarray,
    png: Path,
    svg: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(theta, truth, color="black", linewidth=2.2, label="Ground truth")
    ax.plot(theta, classical, color="C1", linewidth=1.8, label="Classical local Gaussian")
    ax.plot(theta, gkr, color="C0", linewidth=1.8, label="GKR")
    ax.set_xlabel(r"Condition $\theta$")
    ax.set_ylabel("Linear Fisher information")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="0.9", linewidth=0.8)

    ax = axes[1]
    ax.plot(theta, truth_full, color="black", linewidth=2.2, label="Ground truth")
    ax.plot(theta, gkr_full, color="C2", linewidth=1.8, label="GKR full")
    ax.set_xlabel(r"Condition $\theta$")
    ax.set_ylabel("Full Fisher information")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="0.9", linewidth=0.8)

    ax = axes[2]
    if mean_loss.size:
        ax.plot(np.arange(1, mean_loss.size + 1), mean_loss, color="C0", label="GP mean")
    if covariance_loss.size:
        x = np.linspace(1, max(mean_loss.size, 1), covariance_loss.size)
        ax.plot(x, covariance_loss, color="C1", label="Covariance kernel")
    ax.set_xlabel("Optimization progress")
    ax.set_ylabel("Training objective")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    fig.savefig(png, dpi=220)
    fig.savefig(svg)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Path]:
    validate_args(args)
    device = require_device(str(args.device))
    dataset_dir, output_dir, dataset_npz = resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    make_native_dataset_npz(
        output_npz=dataset_npz,
        dataset_family=args.dataset_family,
        x_dim=args.native_x_dim,
        n_total=args.n_total,
        train_frac=args.train_frac,
        seed=args.seed,
        force=args.force_dataset,
    )
    bundle = load_shared_dataset_npz(dataset_npz)
    grid = theta_grid_from_meta(bundle.meta, theta_grid_size=args.theta_grid_size)
    query = theta_midpoints(grid)
    dataset = build_dataset_from_meta(dict(bundle.meta))
    truth = native_linear_fisher_curve(query, dataset)
    truth_full = analytic_fisher_curve(query, dataset)
    classical = classical_linear_fisher(
        theta_all=bundle.theta_train,
        x_all=bundle.x_train,
        theta_grid=grid,
        ridge=args.classical_ridge,
        window_radius=args.classical_window_radius,
        min_endpoint_samples=args.classical_min_endpoint_samples,
    )

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
    result = estimate_gkr_linear_fisher(
        model,
        query,
        finite_difference_step=np.diff(grid, axis=0),
        solve_jitter=args.gkr_solve_jitter,
    )

    results_npz = output_dir / RESULTS_NPZ
    np.savez_compressed(
        results_npz,
        theta_grid=grid,
        theta_midpoints=query,
        ground_truth_linear_fisher=truth,
        ground_truth_full_fisher=truth_full,
        classical_linear_fisher=classical,
        gkr_linear_fisher=result.linear_fisher,
        gkr_covariance_fisher=result.covariance_fisher,
        gkr_full_fisher=result.full_fisher,
        gkr_mean=result.mean,
        gkr_covariance=result.covariance,
        gkr_mean_jacobian=result.mean_jacobian,
        gkr_covariance_jacobian=result.covariance_jacobian,
        gkr_fisher_matrix=result.fisher_matrix,
        gkr_covariance_fisher_matrix=result.covariance_fisher_matrix,
        gkr_full_fisher_matrix=result.full_fisher_matrix,
        gkr_mean_loss=result.mean_loss,
        gkr_covariance_loss=result.covariance_loss,
        gkr_finite_difference_step=np.diff(grid, axis=0),
    )
    figure_png = output_dir / FIGURE_PNG
    figure_svg = output_dir / FIGURE_SVG
    _plot(
        theta=query[:, 0],
        truth=truth,
        truth_full=truth_full,
        classical=classical,
        gkr=result.linear_fisher,
        gkr_full=result.full_fisher,
        mean_loss=result.mean_loss,
        covariance_loss=result.covariance_loss,
        png=figure_png,
        svg=figure_svg,
    )
    summary = {
        "script": "bin/compare_continuous_gkr_fisher.py",
        "method": "GKR Gaussian Fisher",
        "linear_fisher_only": False,
        "device": str(device),
        "seed": args.seed,
        "n_total": args.n_total,
        "n_train": int(bundle.x_train.shape[0]),
        "native_x_dim": args.native_x_dim,
        "theta_grid_size": args.theta_grid_size,
        "gkr_config": asdict(config),
        "finite_difference_step": np.diff(grid[:, 0]).tolist(),
        "finite_difference_source": "adjacent_theta_grid_spacing",
        "solve_jitter": args.gkr_solve_jitter,
        "errors": {
            "gkr": _error_summary(result.linear_fisher, truth),
            "gkr_full": _error_summary(result.full_fisher, truth_full),
            "classical": _error_summary(classical, truth),
        },
        "upstream": {
            "paper": "https://www.nature.com/articles/s41467-025-62856-x",
            "repository": "https://github.com/AgeYY/speed_grid_cell_information",
            "commit": UPSTREAM_COMMIT,
            "source_notebook": "GKR_demo_torch_colab.ipynb",
            "license": "MIT",
        },
        "artifacts": {
            "dataset_npz": str(dataset_npz),
            "results_npz": str(results_npz),
            "figure_png": str(figure_png),
            "figure_svg": str(figure_svg),
        },
    }
    summary_json = output_dir / SUMMARY_JSON
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["errors"], indent=2), flush=True)
    print(f"results_npz: {results_npz}", flush=True)
    print(f"summary_json: {summary_json}", flush=True)
    print(f"figure_png: {figure_png}", flush=True)
    return {
        "results_npz": results_npz,
        "summary_json": summary_json,
        "figure_png": figure_png,
        "figure_svg": figure_svg,
    }


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
