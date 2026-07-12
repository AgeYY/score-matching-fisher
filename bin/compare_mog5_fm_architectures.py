#!/usr/bin/env python3
"""Compare constrained FM parameter networks under fixed long training."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.distance_comparison import (
    FLOW_VELOCITY_FAMILY_BY_METRIC,
    METRIC_CORRELATION,
    METRIC_COSINE,
    METRIC_MAHALANOBIS_SQ,
    METRIC_SQUARED_EUCLIDEAN,
    FlowComparisonConfig,
    build_flow_skl_model,
    classical_metric_matrices,
    flow_metric_matrices,
    labels_from_theta,
    native_mog_ground_truth_matrices,
)
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import require_device
from global_setting import DATA_DIR


METRICS = (
    METRIC_SQUARED_EUCLIDEAN,
    METRIC_COSINE,
    METRIC_CORRELATION,
    METRIC_MAHALANOBIS_SQ,
)
METRIC_LABELS = {
    METRIC_SQUARED_EUCLIDEAN: "Euclidean$^2$",
    METRIC_COSINE: "Cosine",
    METRIC_CORRELATION: "Correlation",
    METRIC_MAHALANOBIS_SQ: "Mahalanobis$^2$",
}


@dataclass(frozen=True)
class ArchitectureSpec:
    key: str
    label: str
    network_architecture: str
    hidden_dim: int
    depth: int


ARCHITECTURES = (
    ArchitectureSpec("mlp_256x5", "MLP 256x5", "mlp", 256, 5),
    ArchitectureSpec("mlp_128x3", "MLP 128x3", "mlp", 128, 3),
    ArchitectureSpec("residual_mlp_128x3", "Residual MLP 128x3", "residual_mlp", 128, 3),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--gt-samples-per-class", type=int, default=100_000)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(DATA_DIR) / "mog_5native_xdim3_n3000",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "mog5_fm_architecture_comparison_n3000_r2",
    )
    return parser


def _mean_relative_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    rows, cols = np.triu_indices(int(reference.shape[0]), k=1)
    denominator = np.maximum(np.abs(reference[rows, cols]), 1e-12)
    return float(np.mean(np.abs(estimate[rows, cols] - reference[rows, cols]) / denominator))


def _parameter_count(spec: ArchitectureSpec, metric: str, *, theta_dim: int, x_dim: int) -> int:
    model = build_flow_skl_model(
        velocity_family=FLOW_VELOCITY_FAMILY_BY_METRIC[metric],
        theta_dim=int(theta_dim),
        x_dim=int(x_dim),
        hidden_dim=int(spec.hidden_dim),
        depth=int(spec.depth),
        network_architecture=str(spec.network_architecture),
        path_schedule="cosine",
        divergence_estimator="exact",
    )
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def _flow_config(args: argparse.Namespace, spec: ArchitectureSpec) -> FlowComparisonConfig:
    return FlowComparisonConfig(
        epochs=int(args.epochs),
        early_patience=0,
        early_min_delta=0.0,
        early_ema_alpha=0.05,
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        lr_schedule="cosine",
        min_lr=float(args.min_lr),
        weight_decay=0.0,
        hidden_dim=int(spec.hidden_dim),
        depth=int(spec.depth),
        network_architecture=str(spec.network_architecture),
        path_schedule="cosine",
        t_eps=0.0005,
        quadrature_steps=64,
        mc_jeffreys_sample=4096,
        ode_steps=64,
        ode_method="midpoint",
        divergence_estimator="exact",
        max_grad_norm=10.0,
        log_every=int(args.log_every),
        checkpoint_selection="last",
        fixed_validation=True,
        likelihood_finetune_epochs=0,
    )


def _dataset_path(dataset_root: Path, repeat_idx: int) -> Path:
    return Path(dataset_root) / f"repeat_{int(repeat_idx):02d}" / "random_mog_categorical.npz"


def run(args: argparse.Namespace) -> dict[str, Path]:
    if int(args.n_repeats) < 1:
        raise ValueError("--n-repeats must be at least 1.")
    device = require_device(str(args.device))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundles = []
    for repeat_idx in range(int(args.n_repeats)):
        path = _dataset_path(Path(args.dataset_root), repeat_idx)
        if not path.is_file():
            raise FileNotFoundError(f"Missing repeat dataset: {path}")
        bundles.append(load_shared_dataset_npz(path))

    ground_truth = native_mog_ground_truth_matrices(
        native_meta=dict(bundles[0].meta),
        samples_per_class=int(args.gt_samples_per_class),
        seed=int(args.seed) + 12_345,
        mahalanobis_ridge=1e-6,
        metrics=METRICS,
    )

    rows: list[dict[str, object]] = []
    loss_records: list[dict[str, object]] = []
    run_start = time.perf_counter()
    for repeat_idx, bundle in enumerate(bundles):
        repeat_seed = int(args.seed) + int(repeat_idx)
        labels_all = labels_from_theta(bundle.theta_all, num_categories=5)
        labels_train = labels_from_theta(bundle.theta_train, num_categories=5)
        classical_all = classical_metric_matrices(
            bundle.x_all,
            labels_all,
            num_categories=5,
            metrics=METRICS,
            mahalanobis_ridge=1e-6,
        )
        closed_form_train = classical_metric_matrices(
            bundle.x_train,
            labels_train,
            num_categories=5,
            metrics=METRICS,
            mahalanobis_ridge=1e-6,
        )
        theta_dim = int(bundle.theta_train.shape[1])
        x_dim = int(bundle.x_train.shape[1])

        for spec in ARCHITECTURES:
            print(
                f"[architecture-comparison] repeat={repeat_idx} seed={repeat_seed} architecture={spec.key}",
                flush=True,
            )
            architecture_start = time.perf_counter()
            config = _flow_config(args, spec)
            matrices, paths = flow_metric_matrices(
                bundle=bundle,
                device=device,
                output_dir=output_dir / f"repeat_{repeat_idx:02d}" / spec.key / "flow",
                config=config,
                seed=repeat_seed,
                metrics=METRICS,
            )
            architecture_seconds = time.perf_counter() - architecture_start

            for metric in METRICS:
                result_path = Path(paths[metric])
                with np.load(result_path, allow_pickle=True) as saved:
                    train_losses = np.asarray(saved["train_losses"], dtype=np.float64)
                    val_losses = np.asarray(saved["val_losses"], dtype=np.float64)
                    learning_rates = np.asarray(saved["learning_rates"], dtype=np.float64)
                rows.append(
                    {
                        "repeat_idx": int(repeat_idx),
                        "repeat_seed": int(repeat_seed),
                        "architecture": spec.key,
                        "architecture_label": spec.label,
                        "network_architecture": spec.network_architecture,
                        "hidden_dim": int(spec.hidden_dim),
                        "depth": int(spec.depth),
                        "metric": metric,
                        "parameter_count": _parameter_count(spec, metric, theta_dim=theta_dim, x_dim=x_dim),
                        "relative_error_to_ground_truth": _mean_relative_error(matrices[metric], ground_truth[metric]),
                        "relative_gap_to_train_optimum": _mean_relative_error(
                            matrices[metric], closed_form_train[metric]
                        ),
                        "classical_all_error_to_ground_truth": _mean_relative_error(
                            classical_all[metric], ground_truth[metric]
                        ),
                        "train_optimum_error_to_ground_truth": _mean_relative_error(
                            closed_form_train[metric], ground_truth[metric]
                        ),
                        "final_train_loss": float(train_losses[-1]),
                        "final_fixed_val_loss": float(val_losses[-1]),
                        "mean_last_100_train_loss": float(np.mean(train_losses[-100:])),
                        "mean_last_100_fixed_val_loss": float(np.mean(val_losses[-100:])),
                        "final_recorded_lr": float(learning_rates[-1]),
                        "architecture_runtime_seconds": float(architecture_seconds),
                        "result_npz": str(result_path.resolve()),
                    }
                )
                loss_records.append(
                    {
                        "repeat_idx": int(repeat_idx),
                        "architecture": spec.key,
                        "metric": metric,
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "learning_rates": learning_rates,
                    }
                )

    rows_frame = pd.DataFrame(rows)
    summary = (
        rows_frame.groupby(["architecture", "architecture_label", "metric"], as_index=False)
        .agg(
            mean_relative_error_to_ground_truth=("relative_error_to_ground_truth", "mean"),
            std_relative_error_to_ground_truth=("relative_error_to_ground_truth", "std"),
            mean_relative_gap_to_train_optimum=("relative_gap_to_train_optimum", "mean"),
            std_relative_gap_to_train_optimum=("relative_gap_to_train_optimum", "std"),
            mean_final_train_loss=("final_train_loss", "mean"),
            mean_final_fixed_val_loss=("final_fixed_val_loss", "mean"),
            mean_parameter_count=("parameter_count", "mean"),
            mean_runtime_seconds=("architecture_runtime_seconds", "mean"),
        )
    )
    rows_path = output_dir / "architecture_comparison_rows.csv"
    summary_path = output_dir / "architecture_comparison_summary.csv"
    losses_path = output_dir / "architecture_comparison_losses.npz"
    config_path = output_dir / "architecture_comparison_config.json"
    rows_frame.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    np.savez_compressed(losses_path, records=np.asarray(loss_records, dtype=object))
    config_path.write_text(
        json.dumps(
            {
                "architectures": [asdict(spec) for spec in ARCHITECTURES],
                "metrics": list(METRICS),
                "n_total": 3000,
                "n_train": 2400,
                "n_validation": 600,
                "n_repeats": int(args.n_repeats),
                "repeat_seeds": [int(args.seed) + idx for idx in range(int(args.n_repeats))],
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "min_lr": float(args.min_lr),
                "lr_schedule": "cosine",
                "checkpoint_selection": "last",
                "fixed_validation": True,
                "device": str(device),
                "total_runtime_seconds": float(time.perf_counter() - run_start),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    performance_png, performance_svg = plot_performance(summary, output_dir)
    losses_png, losses_svg = plot_validation_losses(loss_records, output_dir)
    return {
        "rows_csv": rows_path,
        "summary_csv": summary_path,
        "losses_npz": losses_path,
        "config_json": config_path,
        "performance_png": performance_png,
        "performance_svg": performance_svg,
        "validation_losses_png": losses_png,
        "validation_losses_svg": losses_svg,
    }


def _style_axes(axes) -> None:
    for axis in np.asarray(axes).reshape(-1):
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_linewidth(1.8)
        axis.tick_params(width=1.8)


def plot_performance(summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    colors = ("#4C72B0", "#55A868", "#C44E52")
    x = np.arange(len(METRICS), dtype=np.float64)
    width = 0.25
    for index, spec in enumerate(ARCHITECTURES):
        selected = summary[summary["architecture"] == spec.key].set_index("metric")
        offset = (index - 1) * width
        gt_values = [float(selected.loc[metric, "mean_relative_error_to_ground_truth"]) for metric in METRICS]
        gt_std = [float(selected.loc[metric, "std_relative_error_to_ground_truth"]) for metric in METRICS]
        gap_values = [float(selected.loc[metric, "mean_relative_gap_to_train_optimum"]) for metric in METRICS]
        gap_std = [float(selected.loc[metric, "std_relative_gap_to_train_optimum"]) for metric in METRICS]
        axes[0].bar(x + offset, gt_values, width, yerr=gt_std, color=colors[index], label=spec.label, capsize=3)
        axes[1].bar(x + offset, gap_values, width, yerr=gap_std, color=colors[index], label=spec.label, capsize=3)
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels([METRIC_LABELS[metric] for metric in METRICS], rotation=25, ha="right")
    axes[0].set_ylabel("Relative error to GT")
    axes[1].set_ylabel("Gap to train optimum")
    axes[0].set_title("Distance estimation")
    axes[1].set_title("Optimization gap")
    axes[0].legend(frameon=False)
    _style_axes(axes)
    fig.tight_layout(w_pad=1.4)
    png_path = output_dir / "architecture_comparison_performance.png"
    svg_path = output_dir / "architecture_comparison_performance.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def plot_validation_losses(loss_records: list[dict[str, object]], output_dir: Path) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0), sharex=True)
    colors = ("#4C72B0", "#55A868", "#C44E52")
    for axis, metric in zip(axes.reshape(-1), METRICS, strict=True):
        for color, spec in zip(colors, ARCHITECTURES, strict=True):
            curves = np.stack(
                [
                    np.asarray(record["val_losses"], dtype=np.float64)
                    for record in loss_records
                    if record["metric"] == metric and record["architecture"] == spec.key
                ],
                axis=0,
            )
            mean = curves.mean(axis=0)
            low = curves.min(axis=0)
            high = curves.max(axis=0)
            stride = max(1, int(mean.size // 500))
            epochs = np.arange(1, int(mean.size) + 1)[::stride]
            axis.plot(epochs, mean[::stride], color=color, linewidth=2.2, label=spec.label)
            axis.fill_between(epochs, low[::stride], high[::stride], color=color, alpha=0.16, linewidth=0)
        axis.set_title(METRIC_LABELS[metric])
        axis.set_ylabel("Fixed validation FM loss")
    for axis in axes[-1, :]:
        axis.set_xlabel("Epoch")
    axes[0, 0].legend(frameon=False, fontsize=12)
    _style_axes(axes)
    fig.tight_layout(w_pad=1.1, h_pad=1.0)
    png_path = output_dir / "architecture_comparison_validation_losses.png"
    svg_path = output_dir / "architecture_comparison_validation_losses.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    args = build_parser().parse_args()
    outputs = run(args)
    for name, path in outputs.items():
        print(f"{name}: {Path(path).resolve()}", flush=True)


if __name__ == "__main__":
    main()
