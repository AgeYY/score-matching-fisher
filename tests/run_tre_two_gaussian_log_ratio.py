#!/usr/bin/env python3
"""Compare Torch TRE and direct logistic log-ratio estimation on two Gaussians."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import DATA_DIR, DEFAULT_DEVICE
from fisher.tre_distance import (
    TRE_ARCHITECTURES,
    TRE_WAYMARK_SCHEDULES,
    TREDensityRatioConfig,
    estimate_tre_log_ratio,
    train_tre_density_ratio,
    tre_jeffreys_from_log_ratios,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a Torch TRE model and the existing StandardScaler + LogisticRegression "
            "baseline to two equal-covariance Gaussians, then compare both estimators "
            "with the analytic log-density ratio."
        )
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-train", type=int, default=4_000, help="Training samples per Gaussian.")
    parser.add_argument("--n-validation", type=int, default=1_000, help="Validation samples per Gaussian.")
    parser.add_argument("--n-test", type=int, default=5_000, help="Held-out test samples per Gaussian.")
    parser.add_argument("--num-bridges", type=int, default=8)
    parser.add_argument("--waymark-schedule", choices=TRE_WAYMARK_SCHEDULES, default="angle")
    parser.add_argument("--architecture", choices=TRE_ARCHITECTURES, default="linear")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--early-patience", type=int, default=100)
    parser.add_argument("--validation-pairs", type=int, default=2_048)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--mean-scale",
        type=float,
        default=1.0,
        help="Multiply both Gaussian means by this factor while keeping covariance fixed.",
    )
    regularization = parser.add_mutually_exclusive_group()
    regularization.add_argument(
        "--classical-l2",
        type=float,
        default=None,
        help="Classical logistic-regression L2 penalty lambda; converted to C=1/lambda.",
    )
    regularization.add_argument(
        "--logistic-c",
        type=float,
        default=None,
        help="Classical logistic-regression inverse L2 strength C (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "tre_two_gaussian_log_ratio",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("n_train", "n_validation", "n_test"):
        if int(getattr(args, name)) < 2:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 2.")
    if float(args.mean_scale) <= 0.0:
        raise ValueError("--mean-scale must be positive.")
    if args.classical_l2 is not None and float(args.classical_l2) <= 0.0:
        raise ValueError("--classical-l2 must be positive.")
    if args.logistic_c is not None and float(args.logistic_c) <= 0.0:
        raise ValueError("--logistic-c must be positive.")


def _problem_parameters(*, mean_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean0 = float(mean_scale) * np.array([-3.0, -0.75], dtype=np.float64)
    mean1 = float(mean_scale) * np.array([3.0, 0.75], dtype=np.float64)
    covariance = np.array([[1.0, 0.25], [0.25, 0.8]], dtype=np.float64)
    return mean0, mean1, covariance


def _sample_split(
    rng: np.random.Generator,
    *,
    n: int,
    mean0: np.ndarray,
    mean1: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x0 = rng.multivariate_normal(mean0, covariance, size=int(n)).astype(np.float32)
    x1 = rng.multivariate_normal(mean1, covariance, size=int(n)).astype(np.float32)
    return x0, x1


def analytic_equal_covariance_log_ratio(
    x: np.ndarray,
    *,
    mean0: np.ndarray,
    mean1: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Return ``log N(mean0, covariance) - log N(mean1, covariance)``."""

    x_arr = np.asarray(x, dtype=np.float64)
    precision = np.linalg.inv(np.asarray(covariance, dtype=np.float64))
    centered0 = x_arr - np.asarray(mean0, dtype=np.float64)
    centered1 = x_arr - np.asarray(mean1, dtype=np.float64)
    quadratic0 = np.einsum("ni,ij,nj->n", centered0, precision, centered0)
    quadratic1 = np.einsum("ni,ij,nj->n", centered1, precision, centered1)
    return 0.5 * (quadratic1 - quadratic0)


def _pointwise_metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    residual = estimate - truth
    return {
        "mae": float(np.mean(np.abs(residual), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(residual * residual, dtype=np.float64))),
        "correlation": float(np.corrcoef(estimate, truth)[0, 1]),
        "bias": float(np.mean(residual, dtype=np.float64)),
    }


def _fit_classical_logistic(
    x0_train: np.ndarray,
    x1_train: np.ndarray,
    *,
    logistic_c: float,
    seed: int,
):
    x_train = np.vstack((x0_train, x1_train))
    # This matches fisher.distance_comparison: endpoint 0 receives label one,
    # so decision_function estimates log p0(x) - log p1(x).
    labels = np.concatenate(
        (np.ones(len(x0_train), dtype=np.int64), np.zeros(len(x1_train), dtype=np.int64))
    )
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(logistic_c),
            class_weight="balanced",
            max_iter=2_000,
            random_state=int(seed),
            solver="lbfgs",
        ),
    )
    classifier.fit(x_train, labels)
    return classifier


def _plot_result(
    *,
    x0_test: np.ndarray,
    x1_test: np.ndarray,
    true_log_ratio: np.ndarray,
    tre_log_ratio: np.ndarray,
    classical_log_ratio: np.ndarray,
    train_losses: np.ndarray,
    validation_losses: np.ndarray,
    metrics: dict[str, object],
    output_dir: Path,
    seed: int,
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 13,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    rng = np.random.default_rng(int(seed) + 19)
    dataset_count = min(500, len(x0_test), len(x1_test))
    idx0 = rng.choice(len(x0_test), size=dataset_count, replace=False)
    idx1 = rng.choice(len(x1_test), size=dataset_count, replace=False)
    calibration_count = min(2_500, true_log_ratio.size)
    calibration_idx = rng.choice(true_log_ratio.size, size=calibration_count, replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), constrained_layout=True)
    axes[0].scatter(x0_test[idx0, 0], x0_test[idx0, 1], s=11, alpha=0.55, label="$p_0$")
    axes[0].scatter(x1_test[idx1, 0], x1_test[idx1, 1], s=11, alpha=0.55, label="$p_1$")
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    axes[0].set_title("Held-out data")
    axes[0].legend(frameon=False)

    truth_subset = true_log_ratio[calibration_idx]
    axes[1].scatter(
        truth_subset,
        tre_log_ratio[calibration_idx],
        s=10,
        alpha=0.35,
        label=f"TRE (RMSE {metrics['tre']['rmse']:.2f})",
    )
    axes[1].scatter(
        truth_subset,
        classical_log_ratio[calibration_idx],
        s=10,
        alpha=0.35,
        label=f"Classical (RMSE {metrics['classical']['rmse']:.2f})",
    )
    lo = float(min(np.min(truth_subset), np.min(tre_log_ratio), np.min(classical_log_ratio)))
    hi = float(max(np.max(truth_subset), np.max(tre_log_ratio), np.max(classical_log_ratio)))
    axes[1].plot([lo, hi], [lo, hi], color="black", linewidth=1.8, linestyle="--", label="Exact")
    axes[1].set_xlim(lo, hi)
    axes[1].set_ylim(lo, hi)
    axes[1].set_xlabel("Analytic log ratio")
    axes[1].set_ylabel("Estimated log ratio")
    axes[1].set_title("Log-ratio calibration")
    axes[1].legend(frameon=False, loc="upper left")

    epochs = np.arange(1, len(train_losses) + 1)
    axes[2].plot(epochs, train_losses, linewidth=2.0, label="Training")
    axes[2].plot(epochs, validation_losses, linewidth=2.0, label="Validation")
    axes[2].axvline(int(metrics["tre_training"]["best_epoch"]), color="black", linestyle="--", linewidth=1.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Adjacent BCE")
    axes[2].set_title("TRE optimization")
    axes[2].legend(frameon=False)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
        ax.tick_params(width=1.8)

    png_path = output_dir / "tre_two_gaussian_log_ratio.png"
    svg_path = output_dir / "tre_two_gaussian_log_ratio.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    device = torch.device(str(args.device))
    if device.type != "cuda":
        raise RuntimeError("This project experiment must run on CUDA; pass --device cuda:0.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to silently fall back to CPU.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    mean0, mean1, covariance = _problem_parameters(mean_scale=float(args.mean_scale))
    x0_train, x1_train = _sample_split(
        rng, n=int(args.n_train), mean0=mean0, mean1=mean1, covariance=covariance
    )
    x0_validation, x1_validation = _sample_split(
        rng, n=int(args.n_validation), mean0=mean0, mean1=mean1, covariance=covariance
    )
    x0_test, x1_test = _sample_split(
        rng, n=int(args.n_test), mean0=mean0, mean1=mean1, covariance=covariance
    )

    config = TREDensityRatioConfig(
        num_bridges=int(args.num_bridges),
        waymark_schedule=str(args.waymark_schedule),
        architecture=str(args.architecture),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        early_patience=int(args.early_patience),
        validation_pairs=int(args.validation_pairs),
        log_every=int(args.log_every),
    )
    tre_model, tre_training = train_tre_density_ratio(
        x0_train=x0_train,
        x1_train=x1_train,
        x0_validation=x0_validation,
        x1_validation=x1_validation,
        device=device,
        seed=int(args.seed),
        config=config,
    )
    classical_l2 = None if args.classical_l2 is None else float(args.classical_l2)
    logistic_c = (
        float(args.logistic_c)
        if args.logistic_c is not None
        else (1.0 if classical_l2 is None else 1.0 / classical_l2)
    )
    classical_model = _fit_classical_logistic(
        x0_train,
        x1_train,
        logistic_c=logistic_c,
        seed=int(args.seed),
    )

    x_test = np.vstack((x0_test, x1_test))
    true_log_ratio = analytic_equal_covariance_log_ratio(
        x_test,
        mean0=mean0,
        mean1=mean1,
        covariance=covariance,
    )
    tre_log_ratio = estimate_tre_log_ratio(tre_model, x_test, device=device)
    classical_log_ratio = np.asarray(classical_model.decision_function(x_test), dtype=np.float64)
    precision = np.linalg.inv(covariance)
    mean_delta = mean0 - mean1
    true_jeffreys = float(mean_delta @ precision @ mean_delta)
    tre_jeffreys = tre_jeffreys_from_log_ratios(
        tre_log_ratio[: len(x0_test)], tre_log_ratio[len(x0_test) :]
    )
    classical_jeffreys = tre_jeffreys_from_log_ratios(
        classical_log_ratio[: len(x0_test)], classical_log_ratio[len(x0_test) :]
    )

    metrics: dict[str, object] = {
        "problem": {
            "mean0": mean0.tolist(),
            "mean1": mean1.tolist(),
            "covariance": covariance.tolist(),
            "mean_scale": float(args.mean_scale),
            "true_jeffreys": true_jeffreys,
        },
        "sample_sizes_per_gaussian": {
            "train": int(args.n_train),
            "validation": int(args.n_validation),
            "test": int(args.n_test),
        },
        "tre": {
            **_pointwise_metrics(tre_log_ratio, true_log_ratio),
            "jeffreys": tre_jeffreys,
            "jeffreys_abs_error": abs(tre_jeffreys - true_jeffreys),
        },
        "classical": {
            **_pointwise_metrics(classical_log_ratio, true_log_ratio),
            "jeffreys": classical_jeffreys,
            "jeffreys_abs_error": abs(classical_jeffreys - true_jeffreys),
            "l2_strength": classical_l2,
            "logistic_c": logistic_c,
        },
        "tre_training": {
            "best_epoch": int(tre_training.best_epoch),
            "best_validation_loss": float(tre_training.best_validation_loss),
            "stopped_epoch": int(tre_training.stopped_epoch),
            "stopped_early": bool(tre_training.stopped_early),
            "training_seconds": float(tre_training.training_seconds),
            "config": asdict(config),
        },
        "seed": int(args.seed),
        "device": str(device),
    }

    np.savez_compressed(
        output_dir / "tre_two_gaussian_log_ratio.npz",
        x0_test=x0_test,
        x1_test=x1_test,
        true_log_ratio=true_log_ratio,
        tre_log_ratio=tre_log_ratio,
        classical_log_ratio=classical_log_ratio,
        train_losses=tre_training.train_losses,
        validation_losses=tre_training.validation_losses,
    )
    torch.save(
        {
            "model_state_dict": tre_model.state_dict(),
            "input_dim": tre_model.input_dim,
            "num_bridges": tre_model.num_bridges,
            "architecture": tre_model.architecture,
            "hidden_dim": tre_model.hidden_dim,
            "depth": tre_model.depth,
            "training": metrics["tre_training"],
        },
        output_dir / "tre_best_model.pt",
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    png_path, svg_path = _plot_result(
        x0_test=x0_test,
        x1_test=x1_test,
        true_log_ratio=true_log_ratio,
        tre_log_ratio=tre_log_ratio,
        classical_log_ratio=classical_log_ratio,
        train_losses=tre_training.train_losses,
        validation_losses=tre_training.validation_losses,
        metrics=metrics,
        output_dir=output_dir,
        seed=int(args.seed),
    )

    print(f"Ground-truth Jeffreys: {true_jeffreys:.6f}")
    print(
        f"TRE: Jeffreys={tre_jeffreys:.6f}, RMSE={metrics['tre']['rmse']:.6f}, "
        f"correlation={metrics['tre']['correlation']:.6f}"
    )
    print(
        f"Classical: Jeffreys={classical_jeffreys:.6f}, RMSE={metrics['classical']['rmse']:.6f}, "
        f"correlation={metrics['classical']['correlation']:.6f}"
    )
    print(f"TRE training seconds: {tre_training.training_seconds:.3f}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved figure: {png_path}")
    print(f"Saved vector figure: {svg_path}")


if __name__ == "__main__":
    main()
