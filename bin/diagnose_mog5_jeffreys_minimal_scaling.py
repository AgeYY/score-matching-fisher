#!/usr/bin/env python3
"""Minimal MoG5 Jeffreys scaling diagnostic: classical versus flow matching."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.distance_comparison import (
    FlowComparisonConfig,
    METRIC_SYMMETRIC_KL,
    _build_and_train_flow_model,
    _estimate_trained_flow,
    classical_metric_matrices,
    labels_from_theta,
    native_mog_ground_truth_matrices,
)
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import require_device
from global_setting import DATA_DIR


def _parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("Expected positive comma-separated integers.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-list", type=_parse_int_list, default=[3000, 10_000])
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--early-patience", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--path-schedule", choices=("linear", "cosine"), default="cosine")
    parser.add_argument("--mc-jeffreys-sample", type=int, default=4096)
    parser.add_argument("--ode-steps", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--best-checkpoint-metric",
        choices=("flow_matching", "validation_nll"),
        default="flow_matching",
    )
    parser.add_argument("--likelihood-validation-every", type=int, default=100)
    parser.add_argument("--likelihood-validation-ode-steps", type=int, default=32)
    parser.add_argument("--fixed-validation-paths", type=int, default=10)
    parser.add_argument("--force-datasets", action="store_true")
    parser.add_argument(
        "--template-npz",
        type=Path,
        default=Path(DATA_DIR) / "mog5_seed7_jeffreys_template_n100000" / "random_mog_categorical.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "mog5_jeffreys_minimal_n3000_10000_r5_constant_lr_e20000_pat1000_fmval10paths",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.n_repeats) < 1:
        raise ValueError("--n-repeats must be positive.")
    if int(args.epochs) < 1:
        raise ValueError("--epochs must be positive.")
    if int(args.early_patience) < 0:
        raise ValueError("--early-patience must be nonnegative.")
    if int(args.batch_size) < 1:
        raise ValueError("--batch-size must be positive.")
    if float(args.lr) <= 0.0:
        raise ValueError("--lr must be positive.")
    if int(args.likelihood_validation_every) < 1:
        raise ValueError("--likelihood-validation-every must be positive.")
    if int(args.likelihood_validation_ode_steps) < 1:
        raise ValueError("--likelihood-validation-ode-steps must be positive.")
    if int(args.fixed_validation_paths) < 1:
        raise ValueError("--fixed-validation-paths must be positive.")


def _flow_config(args: argparse.Namespace) -> FlowComparisonConfig:
    return FlowComparisonConfig(
        epochs=int(args.epochs),
        early_patience=int(args.early_patience),
        early_min_delta=1e-4,
        early_ema_alpha=0.05,
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        lr_schedule="constant",
        min_lr=0.0,
        weight_decay=0.0,
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        network_architecture="mlp",
        path_schedule=str(args.path_schedule),
        t_eps=5e-4,
        quadrature_steps=64,
        mc_jeffreys_sample=int(args.mc_jeffreys_sample),
        ode_steps=int(args.ode_steps),
        ode_method="midpoint",
        divergence_estimator="exact",
        max_grad_norm=10.0,
        log_every=int(args.log_every),
        checkpoint_selection="best",
        best_checkpoint_metric=str(args.best_checkpoint_metric),
        likelihood_validation_every=int(args.likelihood_validation_every),
        likelihood_validation_ode_steps=int(args.likelihood_validation_ode_steps),
        likelihood_validation_ode_method="midpoint",
        fixed_validation=True,
        fixed_validation_paths=int(args.fixed_validation_paths),
        likelihood_finetune_epochs=0,
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _dataset_path(output_dir: Path, n_total: int, repeat_idx: int) -> Path:
    return output_dir / "datasets" / f"n{int(n_total)}" / f"repeat_{int(repeat_idx):02d}" / "random_mog_categorical.npz"


def _ensure_dataset(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    n_total: int,
    repeat_idx: int,
) -> tuple[Path, float]:
    path = _dataset_path(output_dir, n_total, repeat_idx)
    if path.is_file() and not bool(args.force_datasets):
        return path, 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(_REPO_ROOT / "bin" / "make_mog5_pr_dataset.py"),
        "--n-total",
        str(int(n_total)),
        "--native-x-dim",
        "3",
        "--pr-dim",
        "none",
        "--seed",
        str(int(args.seed) + int(repeat_idx)),
        "--train-frac",
        "0.8",
        "--native-template-npz",
        str(Path(args.template_npz)),
        "--output-dir",
        str(path.parent),
        "--device",
        str(args.device),
        "--skip-viz",
    ]
    if bool(args.force_datasets):
        command.append("--force")
    start = time.perf_counter()
    subprocess.run(command, cwd=_REPO_ROOT, check=True)
    runtime = time.perf_counter() - start
    if not path.is_file():
        raise RuntimeError(f"Dataset generation did not create {path}")
    return path, runtime


def _pair_errors(estimate: np.ndarray, ground_truth: np.ndarray) -> tuple[float, float]:
    rows, cols = np.triu_indices(int(ground_truth.shape[0]), k=1)
    absolute = np.abs(np.asarray(estimate)[rows, cols] - np.asarray(ground_truth)[rows, cols])
    relative = absolute / np.maximum(np.abs(np.asarray(ground_truth)[rows, cols]), 1e-12)
    return float(absolute.mean()), float(relative.mean())


def _plain_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key != "best_state_dict" and isinstance(value, (str, int, float, bool, np.ndarray))
    }


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["n_total", "estimator"], as_index=False)
        .agg(
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_mrae=("mrae", "mean"),
            std_mrae=("mrae", "std"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            total_runtime_seconds=("runtime_seconds", "sum"),
            n_repeats=("repeat_idx", "nunique"),
        )
        .sort_values(["n_total", "estimator"])
    )


def _plot_errors(rows: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    colors = {"classical": "#4C72B0", "flow_matching": "#DD8452"}
    labels = {"classical": "Classical", "flow_matching": "Flow matching"}
    n_values = sorted(int(value) for value in rows["n_total"].unique())
    x_positions = np.arange(len(n_values), dtype=np.float64)
    x_by_n = {value: float(position) for position, value in enumerate(n_values)}
    with plt.rc_context(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "axes.grid": False,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))
        for estimator in ("classical", "flow_matching"):
            curve = summary[summary["estimator"].eq(estimator)].sort_values("n_total")
            raw = rows[rows["estimator"].eq(estimator)]
            for axis, mean_key, std_key, raw_key, ylabel in (
                (axes[0], "mean_mrae", "std_mrae", "mrae", "MRAE"),
                (axes[1], "mean_mae", "std_mae", "mae", "MAE"),
            ):
                axis.errorbar(
                    [x_by_n[int(value)] for value in curve["n_total"]],
                    curve[mean_key],
                    yerr=curve[std_key],
                    marker="o",
                    linewidth=1.8,
                    capsize=4,
                    color=colors[estimator],
                    label=labels[estimator],
                )
                for repeat_idx, repeat_rows in raw.groupby("repeat_idx"):
                    del repeat_idx
                    axis.plot(
                        [x_by_n[int(value)] for value in repeat_rows["n_total"]],
                        repeat_rows[raw_key],
                        marker=".",
                        linewidth=0.8,
                        alpha=0.35,
                        color=colors[estimator],
                    )
                axis.set_xticks(x_positions)
                axis.set_xticklabels([f"{value:,}" for value in n_values])
                axis.set_xlabel("Total sample size $N$")
                axis.set_ylabel(ylabel)
                for spine in axis.spines.values():
                    spine.set_linewidth(1.8)
                axis.tick_params(width=1.8)
        axes[0].legend(frameon=False)
        fig.tight_layout()
    png = output_dir / "jeffreys_minimal_scaling_errors.png"
    svg = output_dir / "jeffreys_minimal_scaling_errors.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


def _plot_losses(loss_histories: dict[int, list[tuple[np.ndarray, np.ndarray]]], output_dir: Path) -> tuple[Path, Path]:
    n_values = sorted(loss_histories)
    with plt.rc_context(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 13,
            "axes.grid": False,
        }
    ):
        fig, axes = plt.subplots(1, len(n_values), figsize=(4.0 * len(n_values), 3.8), squeeze=False)
        for axis, n_total in zip(axes[0], n_values):
            train = _pad_loss_histories([item[0] for item in loss_histories[n_total]])
            val = _pad_loss_histories([item[1] for item in loss_histories[n_total]])
            epochs = np.arange(1, train.shape[1] + 1)
            for values in train:
                axis.plot(epochs, values, color="#4C72B0", alpha=0.25, linewidth=0.8)
            for values in val:
                axis.plot(epochs, values, color="#DD8452", alpha=0.25, linewidth=0.8)
            axis.plot(epochs, np.nanmean(train, axis=0), color="#4C72B0", linewidth=1.8, label="Train")
            axis.plot(epochs, np.nanmean(val, axis=0), color="#DD8452", linewidth=1.8, label="Validation")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("FM loss")
            axis.set_title(f"$N={n_total:,}$")
            for spine in axis.spines.values():
                spine.set_linewidth(1.8)
            axis.tick_params(width=1.8)
        axes[0, 0].legend(frameon=False)
        fig.tight_layout()
    png = output_dir / "jeffreys_minimal_scaling_fm_losses.png"
    svg = output_dir / "jeffreys_minimal_scaling_fm_losses.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


def _pad_loss_histories(histories: list[np.ndarray]) -> np.ndarray:
    if not histories:
        raise ValueError("Loss histories must be non-empty.")
    max_epochs = max(int(np.asarray(values).size) for values in histories)
    padded = np.full((len(histories), max_epochs), np.nan, dtype=np.float64)
    for row, values in enumerate(histories):
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        padded[row, : flat.size] = flat
    return padded


def _plot_combined_diagnostics(
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    loss_histories: dict[int, list[tuple[np.ndarray, np.ndarray]]],
    output_dir: Path,
) -> tuple[Path, Path]:
    colors = {"classical": "#4C72B0", "flow_matching": "#DD8452"}
    labels = {"classical": "Classical", "flow_matching": "Flow matching"}
    n_values = sorted(int(value) for value in rows["n_total"].unique())
    x_positions = np.arange(len(n_values), dtype=np.float64)
    x_by_n = {value: float(position) for position, value in enumerate(n_values)}
    with plt.rc_context(
        {
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "axes.grid": False,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.2))
        for estimator in ("classical", "flow_matching"):
            curve = summary[summary["estimator"].eq(estimator)].sort_values("n_total")
            raw = rows[rows["estimator"].eq(estimator)]
            for axis, mean_key, std_key, raw_key, ylabel in (
                (axes[0, 0], "mean_mrae", "std_mrae", "mrae", "MRAE"),
                (axes[0, 1], "mean_mae", "std_mae", "mae", "MAE"),
            ):
                axis.errorbar(
                    [x_by_n[int(value)] for value in curve["n_total"]],
                    curve[mean_key],
                    yerr=curve[std_key],
                    marker="o",
                    linewidth=1.8,
                    capsize=4,
                    color=colors[estimator],
                    label=labels[estimator],
                )
                for _, repeat_rows in raw.groupby("repeat_idx"):
                    axis.plot(
                        [x_by_n[int(value)] for value in repeat_rows["n_total"]],
                        repeat_rows[raw_key],
                        marker=".",
                        linewidth=0.8,
                        alpha=0.35,
                        color=colors[estimator],
                    )
                axis.set_xticks(x_positions)
                axis.set_xticklabels([f"{value:,}" for value in n_values])
                axis.set_xlabel("Total sample size $N$")
                axis.set_ylabel(ylabel)

        axes[0, 0].legend(frameon=False)
        for axis, n_total in zip(axes[1], n_values):
            train = _pad_loss_histories([item[0] for item in loss_histories[n_total]])
            val = _pad_loss_histories([item[1] for item in loss_histories[n_total]])
            epochs = np.arange(1, train.shape[1] + 1)
            for values in train:
                axis.plot(epochs, values, color="#4C72B0", alpha=0.2, linewidth=0.7)
            for values in val:
                axis.plot(epochs, values, color="#DD8452", alpha=0.2, linewidth=0.7)
            axis.plot(epochs, np.nanmean(train, axis=0), color="#4C72B0", linewidth=1.8, label="Train")
            axis.plot(epochs, np.nanmean(val, axis=0), color="#DD8452", linewidth=1.8, label="Validation")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("FM loss")
            axis.set_title(f"$N={n_total:,}$")
        axes[1, 0].legend(frameon=False)
        for axis in axes.flat:
            axis.grid(False)
            for spine in axis.spines.values():
                spine.set_linewidth(1.8)
            axis.tick_params(width=1.8)
        fig.tight_layout()
    png = output_dir / "jeffreys_minimal_scaling_combined.png"
    svg = output_dir / "jeffreys_minimal_scaling_combined.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


def run(args: argparse.Namespace) -> dict[str, Path | float]:
    _validate_args(args)
    device = require_device(str(args.device))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _flow_config(args)
    template = load_shared_dataset_npz(Path(args.template_npz))
    ground_truth = native_mog_ground_truth_matrices(
        native_meta=dict(template.meta),
        metrics=(METRIC_SYMMETRIC_KL,),
    )[METRIC_SYMMETRIC_KL]
    np.save(output_dir / "ground_truth_jeffreys.npy", ground_truth)

    experiment_start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    loss_histories: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
        int(value): [] for value in args.n_list
    }
    case_runtimes: list[dict[str, Any]] = []

    for n_total in args.n_list:
        for repeat_idx in range(int(args.n_repeats)):
            case_start = time.perf_counter()
            repeat_seed = int(args.seed) + int(repeat_idx)
            print(f"[minimal-jeffreys] N={n_total} repeat={repeat_idx} seed={repeat_seed}", flush=True)
            dataset_path, dataset_seconds = _ensure_dataset(
                args,
                output_dir=output_dir,
                n_total=int(n_total),
                repeat_idx=int(repeat_idx),
            )
            bundle = load_shared_dataset_npz(dataset_path)
            labels = labels_from_theta(bundle.theta_all, num_categories=5)

            classical_start = time.perf_counter()
            classical = classical_metric_matrices(
                bundle.x_all,
                labels,
                num_categories=5,
                metrics=(METRIC_SYMMETRIC_KL,),
                mahalanobis_ridge=1e-6,
            )[METRIC_SYMMETRIC_KL]
            classical_seconds = time.perf_counter() - classical_start
            classical_mae, classical_mrae = _pair_errors(classical, ground_truth)

            _sync(device)
            fm_start = time.perf_counter()
            model, metadata = _build_and_train_flow_model(
                theta_train=bundle.theta_train,
                x_train=bundle.x_train,
                theta_val=bundle.theta_validation,
                x_val=bundle.x_validation,
                velocity_family="nonlinear",
                device=device,
                seed=repeat_seed,
                config=config,
            )
            _sync(device)
            fm_train_seconds = time.perf_counter() - fm_start

            _sync(device)
            estimate_start = time.perf_counter()
            result = _estimate_trained_flow(
                model=model,
                theta_eval=np.eye(5, dtype=np.float64),
                velocity_family="nonlinear",
                device=device,
                seed=repeat_seed + 100_000,
                config=config,
                train_metadata=metadata,
            )
            _sync(device)
            fm_estimate_seconds = time.perf_counter() - estimate_start
            fm = np.asarray(result.symmetric_kl_matrix, dtype=np.float64)
            fm_mae, fm_mrae = _pair_errors(fm, ground_truth)

            case_dir = output_dir / "cases" / f"n{int(n_total)}_repeat{int(repeat_idx):02d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            np.save(case_dir / "classical_jeffreys.npy", classical)
            np.save(case_dir / "flow_matching_jeffreys.npy", fm)
            plain_metadata = _plain_metadata(metadata)
            torch.save(
                {
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "metadata": plain_metadata,
                    "config": asdict(config),
                },
                case_dir / "flow_matching_best_checkpoint.pt",
            )
            np.savez_compressed(
                case_dir / "flow_matching_losses.npz",
                train_losses=np.asarray(metadata["train_losses"], dtype=np.float64),
                val_losses=np.asarray(metadata["val_losses"], dtype=np.float64),
                val_monitor_losses=np.asarray(metadata["val_monitor_losses"], dtype=np.float64),
                learning_rates=np.asarray(metadata["learning_rates"], dtype=np.float64),
                likelihood_validation_epochs=np.asarray(
                    metadata["likelihood_validation_epochs"], dtype=np.int64
                ),
                likelihood_validation_nlls=np.asarray(
                    metadata["likelihood_validation_nlls"], dtype=np.float64
                ),
                best_epoch=np.asarray([metadata["best_epoch"]], dtype=np.int64),
                selected_epoch=np.asarray([metadata["selected_epoch"]], dtype=np.int64),
            )
            loss_histories[int(n_total)].append(
                (
                    np.asarray(metadata["train_losses"], dtype=np.float64),
                    np.asarray(metadata["val_losses"], dtype=np.float64),
                )
            )

            rows.extend(
                [
                    {
                        "n_total": int(n_total),
                        "repeat_idx": int(repeat_idx),
                        "repeat_seed": repeat_seed,
                        "estimator": "classical",
                        "mae": classical_mae,
                        "mrae": classical_mrae,
                        "runtime_seconds": classical_seconds,
                        "selected_epoch": 0,
                        "stopped_epoch": 0,
                        "stopped_early": False,
                    },
                    {
                        "n_total": int(n_total),
                        "repeat_idx": int(repeat_idx),
                        "repeat_seed": repeat_seed,
                        "estimator": "flow_matching",
                        "mae": fm_mae,
                        "mrae": fm_mrae,
                        "runtime_seconds": fm_train_seconds + fm_estimate_seconds,
                        "selected_epoch": int(metadata["selected_epoch"]),
                        "stopped_epoch": int(metadata["stopped_epoch"]),
                        "stopped_early": bool(metadata["stopped_early"]),
                    },
                ]
            )
            case_seconds = time.perf_counter() - case_start
            case_runtimes.append(
                {
                    "n_total": int(n_total),
                    "repeat_idx": int(repeat_idx),
                    "repeat_seed": repeat_seed,
                    "n_train": int(bundle.x_train.shape[0]),
                    "batches_per_epoch": int(math.ceil(bundle.x_train.shape[0] / int(args.batch_size))),
                    "optimizer_steps": int(metadata["n_total_steps"]),
                    "stopped_epoch": int(metadata["stopped_epoch"]),
                    "stopped_early": bool(metadata["stopped_early"]),
                    "dataset_seconds": dataset_seconds,
                    "classical_seconds": classical_seconds,
                    "fm_train_seconds": fm_train_seconds,
                    "fm_estimate_seconds": fm_estimate_seconds,
                    "case_seconds": case_seconds,
                }
            )
            print(
                f"[minimal-jeffreys result] N={n_total} repeat={repeat_idx} "
                f"classical_mrae={classical_mrae:.6f} fm_mrae={fm_mrae:.6f} "
                f"fm_seconds={fm_train_seconds + fm_estimate_seconds:.2f}",
                flush=True,
            )

    elapsed_seconds = time.perf_counter() - experiment_start
    rows_frame = pd.DataFrame(rows).sort_values(["n_total", "repeat_idx", "estimator"])
    summary = _summarize(rows_frame)
    runtime_frame = pd.DataFrame(case_runtimes).sort_values(["n_total", "repeat_idx"])
    rows_path = output_dir / "jeffreys_minimal_scaling_rows.csv"
    summary_path = output_dir / "jeffreys_minimal_scaling_summary.csv"
    runtime_path = output_dir / "jeffreys_minimal_scaling_runtime.csv"
    rows_frame.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    runtime_frame.to_csv(runtime_path, index=False)
    error_png, error_svg = _plot_errors(rows_frame, summary, output_dir)
    loss_png, loss_svg = _plot_losses(loss_histories, output_dir)
    combined_png, combined_svg = _plot_combined_diagnostics(
        rows_frame,
        summary,
        loss_histories,
        output_dir,
    )
    config_path = output_dir / "jeffreys_minimal_scaling_config.json"
    config_path.write_text(
        json.dumps(
            {
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "flow_config": asdict(config),
                "elapsed_seconds": elapsed_seconds,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "rows_csv": rows_path,
        "summary_csv": summary_path,
        "runtime_csv": runtime_path,
        "config_json": config_path,
        "error_figure_png": error_png,
        "error_figure_svg": error_svg,
        "loss_figure_png": loss_png,
        "loss_figure_svg": loss_svg,
        "combined_figure_png": combined_png,
        "combined_figure_svg": combined_svg,
        "elapsed_seconds": elapsed_seconds,
    }


def main() -> None:
    outputs = run(build_parser().parse_args())
    for key, value in outputs.items():
        print(f"{key}: {value}", flush=True)


if __name__ == "__main__":
    main()
