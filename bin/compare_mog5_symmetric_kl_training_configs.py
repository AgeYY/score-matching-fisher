#!/usr/bin/env python3
"""Compare previous and long-cosine FM training for MoG5 symmetric KL."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
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
    METRIC_SYMMETRIC_KL,
    _seed_flow_rng,
    classical_metric_matrices,
    labels_from_theta,
    native_mog_ground_truth_matrices,
)
from fisher.flow_matching_skl import build_flow_skl_model, estimate_model_symmetric_kl, train_flow_skl_model
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import require_device
from global_setting import DATA_DIR


@dataclass(frozen=True)
class TrainingConfig:
    key: str
    label: str
    hidden_dim: int
    depth: int
    epochs: int
    lr_schedule: str
    min_lr: float
    patience: int
    fixed_validation: bool
    checkpoint_selection: str
    retain_best_state: bool = False


CONFIGS = (
    TrainingConfig(
        key="previous",
        label="Previous best",
        hidden_dim=256,
        depth=5,
        epochs=20_000,
        lr_schedule="constant",
        min_lr=0.0,
        patience=1_000,
        fixed_validation=False,
        checkpoint_selection="best",
    ),
    TrainingConfig(
        key="long_cosine",
        label="Long cosine",
        hidden_dim=128,
        depth=3,
        epochs=20_000,
        lr_schedule="cosine",
        min_lr=1e-6,
        patience=0,
        fixed_validation=True,
        checkpoint_selection="last",
        retain_best_state=True,
    ),
)

VARIANT_LABELS = {
    "previous_best": "Previous best",
    "long_cosine_best": "Long cosine best",
    "long_cosine_last": "Long cosine last",
}
VARIANT_ORDER = tuple(VARIANT_LABELS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--gt-samples-per-class", type=int, default=100_000)
    parser.add_argument("--mc-jeffreys-sample", type=int, default=4096)
    parser.add_argument("--ode-steps", type=int, default=64)
    parser.add_argument("--smoke-epochs", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(DATA_DIR) / "mog_5native_xdim3_n3000",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "mog5_symmetric_kl_training_config_comparison_n3000_r2",
    )
    return parser


def _dataset_path(dataset_root: Path, repeat_idx: int) -> Path:
    return Path(dataset_root) / f"repeat_{int(repeat_idx):02d}" / "random_mog_categorical.npz"


def _mean_relative_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    rows, cols = np.triu_indices(int(reference.shape[0]), k=1)
    denominator = np.maximum(np.abs(reference[rows, cols]), 1e-12)
    return float(np.mean(np.abs(estimate[rows, cols] - reference[rows, cols]) / denominator))


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _metadata_without_state(metadata: dict[str, Any], *, selected_epoch: int) -> dict[str, Any]:
    out = {key: value for key, value in metadata.items() if key != "best_state_dict"}
    out["selected_epoch"] = int(selected_epoch)
    return out


def _train_config(
    *,
    spec: TrainingConfig,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    theta_eval: np.ndarray,
    device: torch.device,
    seed: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _seed_flow_rng(int(seed), device)
    model = build_flow_skl_model(
        velocity_family="nonlinear",
        theta_dim=int(theta_train.shape[1]),
        x_dim=int(x_train.shape[1]),
        hidden_dim=int(spec.hidden_dim),
        depth=int(spec.depth),
        path_schedule="cosine",
        divergence_estimator="exact",
    ).to(device)
    start = time.perf_counter()
    metadata = train_flow_skl_model(
        model=model,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        device=device,
        velocity_family="nonlinear",
        path_schedule="cosine",
        epochs=int(spec.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        lr_schedule=str(spec.lr_schedule),
        min_lr=float(spec.min_lr),
        weight_decay=0.0,
        t_eps=0.0005,
        patience=int(spec.patience),
        min_delta=1e-4,
        ema_alpha=0.05,
        max_grad_norm=10.0,
        log_every=int(args.log_every),
        checkpoint_selection=str(spec.checkpoint_selection),
        fixed_validation=bool(spec.fixed_validation),
        validation_seed=int(seed) + 500_000 if spec.fixed_validation else None,
        retain_best_state=bool(spec.retain_best_state),
    )
    runtime_seconds = time.perf_counter() - start
    if spec.key == "previous":
        states = [("previous_best", _clone_state_dict(model), int(metadata["best_epoch"]))]
    else:
        best_state = metadata.get("best_state_dict")
        if not isinstance(best_state, dict):
            raise RuntimeError("Long-cosine training did not retain its best checkpoint.")
        states = [
            ("long_cosine_best", best_state, int(metadata["best_epoch"])),
            ("long_cosine_last", _clone_state_dict(model), int(metadata["stopped_epoch"])),
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for variant, state, selected_epoch in states:
        model.load_state_dict(state)
        plain_metadata = _metadata_without_state(metadata, selected_epoch=selected_epoch)
        _seed_flow_rng(int(seed) + 100_000, device)
        result = estimate_model_symmetric_kl(
            model=model,
            theta_all=theta_eval,
            device=device,
            velocity_family="nonlinear",
            mc_jeffreys_sample=int(args.mc_jeffreys_sample),
            ode_steps=int(args.ode_steps),
            ode_method="midpoint",
            batch_size=int(args.batch_size),
            solve_jitter=1e-6,
            quadrature_steps=64,
            fisher_kind="none",
            train_metadata=plain_metadata,
        )
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        model_path = variant_dir / "symmetric_kl_model.pt"
        result_path = variant_dir / "symmetric_kl_result.npz"
        torch.save(state, model_path)
        np.savez_compressed(
            result_path,
            symmetric_kl_matrix=np.asarray(result.symmetric_kl_matrix, dtype=np.float64),
            train_losses=np.asarray(metadata["train_losses"], dtype=np.float64),
            val_losses=np.asarray(metadata["val_losses"], dtype=np.float64),
            val_monitor_losses=np.asarray(metadata["val_monitor_losses"], dtype=np.float64),
            learning_rates=np.asarray(metadata["learning_rates"], dtype=np.float64),
            selected_epoch=np.asarray([selected_epoch], dtype=np.int64),
            best_epoch=np.asarray([int(metadata["best_epoch"])], dtype=np.int64),
            stopped_epoch=np.asarray([int(metadata["stopped_epoch"])], dtype=np.int64),
        )
        records.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "matrix": np.asarray(result.symmetric_kl_matrix, dtype=np.float64),
                "selected_epoch": int(selected_epoch),
                "best_epoch": int(metadata["best_epoch"]),
                "stopped_epoch": int(metadata["stopped_epoch"]),
                "stopped_early": bool(metadata["stopped_early"]),
                "runtime_seconds": float(runtime_seconds),
                "model_path": str(model_path.resolve()),
                "result_path": str(result_path.resolve()),
            }
        )
    return records, metadata


def _plot_results(rows: pd.DataFrame, histories: dict[tuple[int, str], dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    colors = {"previous": "#7A7A7A", "long_cosine": "#4C72B0"}

    summary = rows.groupby(["variant", "variant_label"], as_index=False).agg(
        mean_error=("relative_error_to_ground_truth", "mean"),
        std_error=("relative_error_to_ground_truth", "std"),
    )
    indexed = summary.set_index("variant")
    y = np.arange(len(VARIANT_ORDER))
    means = [float(indexed.loc[key, "mean_error"]) for key in VARIANT_ORDER]
    stds = [float(indexed.loc[key, "std_error"]) for key in VARIANT_ORDER]
    variant_colors = ("#7A7A7A", "#4C72B0", "#C44E52")
    axes[0].errorbar(means, y, xerr=stds, fmt="none", ecolor="black", capsize=3)
    for index, (mean, color) in enumerate(zip(means, variant_colors, strict=True)):
        axes[0].scatter(mean, y[index], color=color, s=42, zorder=3)
    classical = float(rows["classical_error_to_ground_truth"].mean())
    axes[0].axvline(classical, color="black", linestyle="--", linewidth=1.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([VARIANT_LABELS[key] for key in VARIANT_ORDER])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Relative error to GT")
    axes[0].set_title("Jeffreys error")

    validation_tail: list[np.ndarray] = []
    for (repeat_idx, key), metadata in histories.items():
        epochs = np.arange(1, len(metadata["val_monitor_losses"]) + 1)
        monitor = np.asarray(metadata["val_monitor_losses"], dtype=np.float64)
        axes[1].plot(
            epochs,
            monitor,
            color=colors[key],
            alpha=0.75,
            linewidth=1.2,
            label=("Previous" if key == "previous" else "Long cosine") if repeat_idx == 0 else None,
        )
        best_epoch = int(metadata["best_epoch"])
        axes[1].scatter(
            best_epoch,
            monitor[best_epoch - 1],
            color=colors[key],
            edgecolor="white",
            linewidth=0.6,
            s=30,
            zorder=3,
        )
        validation_tail.append(monitor[min(99, len(monitor) - 1) :])
        axes[2].plot(
            epochs,
            metadata["learning_rates"],
            color=colors[key],
            alpha=0.75,
            linewidth=1.2,
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation FM loss (EMA)")
    axes[1].set_title("Validation trajectory")
    axes[1].set_xlim(100, 20_000)
    tail = np.concatenate(validation_tail)
    padding = max(0.01, 0.08 * float(tail.max() - tail.min()))
    axes[1].set_ylim(float(tail.min()) - padding, float(tail.max()) + padding)
    axes[1].legend(frameon=False)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning rate")
    axes[2].set_title("Learning-rate trajectory")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for axis in axes:
        for spine in axis.spines.values():
            spine.set_linewidth(1.4)
        axis.tick_params(width=1.4)
    fig.tight_layout(w_pad=1.2)
    png_path = output_dir / "symmetric_kl_training_config_comparison.png"
    svg_path = output_dir / "symmetric_kl_training_config_comparison.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


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
        metrics=(METRIC_SYMMETRIC_KL,),
    )[METRIC_SYMMETRIC_KL]
    configs = CONFIGS
    if int(args.smoke_epochs) > 0:
        configs = tuple(
            replace(spec, epochs=int(args.smoke_epochs), patience=0, retain_best_state=spec.key == "long_cosine")
            for spec in CONFIGS
        )

    rows: list[dict[str, Any]] = []
    histories: dict[tuple[int, str], dict[str, Any]] = {}
    run_start = time.perf_counter()
    for repeat_idx, bundle in enumerate(bundles):
        repeat_seed = int(args.seed) + repeat_idx
        labels_all = labels_from_theta(bundle.theta_all, num_categories=5)
        classical = classical_metric_matrices(
            bundle.x_all,
            labels_all,
            num_categories=5,
            metrics=(METRIC_SYMMETRIC_KL,),
            mahalanobis_ridge=1e-6,
        )[METRIC_SYMMETRIC_KL]
        theta_eval = np.eye(5, dtype=np.float64)
        for spec in configs:
            print(
                f"[symmetric-kl-config] repeat={repeat_idx} seed={repeat_seed} config={spec.key}",
                flush=True,
            )
            outputs, metadata = _train_config(
                spec=spec,
                theta_train=bundle.theta_train,
                x_train=bundle.x_train,
                theta_val=bundle.theta_validation,
                x_val=bundle.x_validation,
                theta_eval=theta_eval,
                device=device,
                seed=repeat_seed,
                args=args,
                output_dir=output_dir / f"repeat_{repeat_idx:02d}",
            )
            histories[(repeat_idx, spec.key)] = metadata
            for output in outputs:
                matrix = np.asarray(output.pop("matrix"), dtype=np.float64)
                rows.append(
                    {
                        "repeat_idx": int(repeat_idx),
                        "repeat_seed": int(repeat_seed),
                        **output,
                        "relative_error_to_ground_truth": _mean_relative_error(matrix, ground_truth),
                        "classical_error_to_ground_truth": _mean_relative_error(classical, ground_truth),
                    }
                )

    rows_frame = pd.DataFrame(rows)
    summary = (
        rows_frame.groupby(["variant", "variant_label"], as_index=False)
        .agg(
            mean_relative_error_to_ground_truth=("relative_error_to_ground_truth", "mean"),
            std_relative_error_to_ground_truth=("relative_error_to_ground_truth", "std"),
            mean_selected_epoch=("selected_epoch", "mean"),
            mean_stopped_epoch=("stopped_epoch", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
    )
    rows_path = output_dir / "symmetric_kl_training_config_rows.csv"
    summary_path = output_dir / "symmetric_kl_training_config_summary.csv"
    config_path = output_dir / "symmetric_kl_training_config.json"
    rows_frame.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "configs": [asdict(spec) for spec in configs],
                "n_total": 3000,
                "n_train": 2400,
                "n_validation": 600,
                "n_repeats": int(args.n_repeats),
                "repeat_seeds": [int(args.seed) + idx for idx in range(int(args.n_repeats))],
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "mc_jeffreys_sample": int(args.mc_jeffreys_sample),
                "ode_steps": int(args.ode_steps),
                "device": str(device),
                "total_runtime_seconds": float(time.perf_counter() - run_start),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    png_path, svg_path = _plot_results(rows_frame, histories, output_dir)
    return {
        "rows_csv": rows_path,
        "summary_csv": summary_path,
        "config_json": config_path,
        "figure_png": png_path,
        "figure_svg": svg_path,
    }


def main() -> None:
    args = build_parser().parse_args()
    outputs = run(args)
    for name, path in outputs.items():
        print(f"{name}: {Path(path).resolve()}", flush=True)


if __name__ == "__main__":
    main()
