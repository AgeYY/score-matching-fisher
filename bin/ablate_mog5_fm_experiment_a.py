#!/usr/bin/env python3
"""Ablate the training changes in constrained-MoG5 FM Experiment A."""

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
    FLOW_VELOCITY_FAMILY_BY_METRIC,
    METRIC_CORRELATION,
    METRIC_COSINE,
    METRIC_MAHALANOBIS_SQ,
    METRIC_SQUARED_EUCLIDEAN,
    _seed_flow_rng,
    classical_metric_matrices,
    flow_skl_to_metric_readout,
    labels_from_theta,
    native_mog_ground_truth_matrices,
)
from fisher.flow_matching_skl import build_flow_skl_model, estimate_model_symmetric_kl, train_flow_skl_model
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
class TrajectorySpec:
    key: str
    label: str
    epochs: int
    lr_schedule: str
    min_lr: float
    lr_schedule_epochs: int
    patience: int
    checkpoint_selection: str
    fixed_validation: bool
    retain_best_state: bool = False


TRAJECTORIES = (
    TrajectorySpec("experiment_a", "Experiment A", 20_000, "cosine", 1e-6, 20_000, 0, "last", True, True),
    TrajectorySpec("short_2k", "Short: 2k epochs", 2_000, "cosine", 1e-6, 20_000, 0, "last", True),
    TrajectorySpec("constant_lr", "No LR decay", 20_000, "constant", 0.0, 20_000, 0, "last", True),
    TrajectorySpec("resampled_best", "Best, resampled val", 20_000, "cosine", 1e-6, 20_000, 0, "best", False),
    TrajectorySpec("early_stop_fixed", "Early stop, fixed val", 20_000, "cosine", 1e-6, 20_000, 1_000, "best", True),
    TrajectorySpec("previous_like", "Previous-like", 20_000, "constant", 0.0, 20_000, 1_000, "best", False),
)

VARIANT_LABELS = {
    "experiment_a_last": "Experiment A",
    "experiment_a_best": "Best checkpoint",
    "short_2k": "Short: 2k epochs",
    "constant_lr": "No LR decay",
    "resampled_best": "Best, resampled val",
    "early_stop_fixed": "Early stop, fixed val",
    "previous_like": "Previous-like",
}
VARIANT_ORDER = tuple(VARIANT_LABELS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--gt-samples-per-class", type=int, default=100_000)
    parser.add_argument("--smoke-epochs", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(DATA_DIR) / "mog_5native_xdim3_n3000",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "mog5_fm_experiment_a_ablation_n3000_r2",
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


def _plain_metadata(metadata: dict[str, Any], *, variant: str, selected_epoch: int) -> dict[str, Any]:
    out = {key: value for key, value in metadata.items() if key != "best_state_dict"}
    out["ablation_variant"] = str(variant)
    out["selected_epoch"] = int(selected_epoch)
    return out


def _estimate_metric(
    *,
    model: torch.nn.Module,
    theta_eval: np.ndarray,
    metric: str,
    device: torch.device,
    seed: int,
    metadata: dict[str, Any],
) -> np.ndarray:
    family = FLOW_VELOCITY_FAMILY_BY_METRIC[metric]
    _seed_flow_rng(int(seed), device)
    result = estimate_model_symmetric_kl(
        model=model,
        theta_all=theta_eval,
        device=device,
        velocity_family=family,
        radius=1.0,
        mc_jeffreys_sample=4096,
        ode_steps=64,
        ode_method="midpoint",
        batch_size=3000,
        solve_jitter=1e-6,
        quadrature_steps=64,
        fisher_kind="none",
        train_metadata=metadata,
    )
    return flow_skl_to_metric_readout(metric, result.symmetric_kl_matrix, radius=1.0)


def _train_trajectory(
    *,
    spec: TrajectorySpec,
    metric: str,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    theta_eval: np.ndarray,
    device: torch.device,
    seed: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, Any]]:
    family = FLOW_VELOCITY_FAMILY_BY_METRIC[metric]
    _seed_flow_rng(int(seed), device)
    model = build_flow_skl_model(
        velocity_family=family,
        theta_dim=int(theta_train.shape[1]),
        x_dim=int(x_train.shape[1]),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        network_architecture="mlp",
        path_schedule="cosine",
        divergence_estimator="exact",
        shared_affine_a_diag_jitter=1e-3,
    ).to(device)
    start = time.perf_counter()
    metadata = train_flow_skl_model(
        model=model,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        device=device,
        velocity_family=family,
        path_schedule="cosine",
        epochs=int(spec.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        lr_schedule=str(spec.lr_schedule),
        min_lr=float(spec.min_lr),
        lr_schedule_epochs=int(spec.lr_schedule_epochs),
        weight_decay=0.0,
        t_eps=0.0005,
        patience=int(spec.patience),
        min_delta=1e-4,
        ema_alpha=0.05,
        max_grad_norm=10.0,
        log_every=int(args.log_every),
        checkpoint_selection=str(spec.checkpoint_selection),
        fixed_validation=bool(spec.fixed_validation),
        validation_seed=int(seed) + 500_000,
        retain_best_state=bool(spec.retain_best_state),
    )
    runtime_seconds = time.perf_counter() - start
    selected: list[tuple[str, dict[str, torch.Tensor], int]] = []
    if spec.key == "experiment_a":
        selected.append(("experiment_a_last", _clone_state_dict(model), int(metadata["stopped_epoch"])))
        best_state = metadata.get("best_state_dict")
        if not isinstance(best_state, dict):
            raise RuntimeError("Experiment A did not retain its best checkpoint.")
        selected.append(("experiment_a_best", best_state, int(metadata["best_epoch"])))
    else:
        selected.append((spec.key, _clone_state_dict(model), int(metadata["selected_epoch"])))

    outputs: list[dict[str, Any]] = []
    for variant, state, selected_epoch in selected:
        model.load_state_dict(state)
        variant_metadata = _plain_metadata(metadata, variant=variant, selected_epoch=selected_epoch)
        matrix = _estimate_metric(
            model=model,
            theta_eval=theta_eval,
            metric=metric,
            device=device,
            seed=int(seed) + 100_000,
            metadata=variant_metadata,
        )
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        model_path = variant_dir / f"{metric}_model.pt"
        result_path = variant_dir / f"{metric}_result.npz"
        torch.save(state, model_path)
        np.savez_compressed(
            result_path,
            metric_matrix=np.asarray(matrix, dtype=np.float64),
            train_losses=np.asarray(metadata["train_losses"], dtype=np.float64),
            val_losses=np.asarray(metadata["val_losses"], dtype=np.float64),
            val_monitor_losses=np.asarray(metadata["val_monitor_losses"], dtype=np.float64),
            learning_rates=np.asarray(metadata["learning_rates"], dtype=np.float64),
            best_epoch=np.asarray([int(metadata["best_epoch"])], dtype=np.int64),
            selected_epoch=np.asarray([int(selected_epoch)], dtype=np.int64),
            stopped_epoch=np.asarray([int(metadata["stopped_epoch"])], dtype=np.int64),
            stopped_early=np.asarray([bool(metadata["stopped_early"])], dtype=np.bool_),
            checkpoint_selection=np.asarray([variant], dtype=object),
        )
        outputs.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "matrix": matrix,
                "selected_epoch": int(selected_epoch),
                "best_epoch": int(metadata["best_epoch"]),
                "stopped_epoch": int(metadata["stopped_epoch"]),
                "stopped_early": bool(metadata["stopped_early"]),
                "final_train_loss": float(np.asarray(metadata["train_losses"])[-1]),
                "final_val_loss": float(np.asarray(metadata["val_losses"])[-1]),
                "runtime_seconds": float(runtime_seconds),
                "result_npz": str(result_path.resolve()),
                "model_path": str(model_path.resolve()),
            }
        )
    return outputs


def _effect_rows(rows: pd.DataFrame) -> pd.DataFrame:
    comparisons = {
        "shorter_training": ("short_2k", "experiment_a_last"),
        "no_lr_decay": ("constant_lr", "experiment_a_last"),
        "best_instead_of_last": ("experiment_a_best", "experiment_a_last"),
        "resampled_instead_of_fixed_validation": ("resampled_best", "experiment_a_best"),
        "early_stopping": ("early_stop_fixed", "experiment_a_best"),
        "combined_previous_like": ("previous_like", "experiment_a_last"),
    }
    records: list[dict[str, Any]] = []
    indexed = rows.set_index(["repeat_idx", "metric", "variant"])
    for effect, (ablation, reference) in comparisons.items():
        for repeat_idx in sorted(rows["repeat_idx"].unique()):
            for metric in METRICS:
                ablation_row = indexed.loc[(repeat_idx, metric, ablation)]
                reference_row = indexed.loc[(repeat_idx, metric, reference)]
                records.append(
                    {
                        "effect": effect,
                        "ablation_variant": ablation,
                        "reference_variant": reference,
                        "repeat_idx": int(repeat_idx),
                        "metric": metric,
                        "delta_error_to_ground_truth": float(
                            ablation_row["relative_error_to_ground_truth"]
                            - reference_row["relative_error_to_ground_truth"]
                        ),
                        "delta_gap_to_train_optimum": float(
                            ablation_row["relative_gap_to_train_optimum"]
                            - reference_row["relative_gap_to_train_optimum"]
                        ),
                    }
                )
    return pd.DataFrame(records)


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
    rows: list[dict[str, Any]] = []
    trajectories = TRAJECTORIES
    if int(args.smoke_epochs) > 0:
        cap = int(args.smoke_epochs)
        trajectories = tuple(
            replace(
                spec,
                epochs=min(int(spec.epochs), cap),
                lr_schedule_epochs=max(1, min(int(spec.lr_schedule_epochs), cap)),
                patience=0,
            )
            for spec in TRAJECTORIES
        )
    run_start = time.perf_counter()
    for repeat_idx, bundle in enumerate(bundles):
        repeat_seed = int(args.seed) + repeat_idx
        labels_all = labels_from_theta(bundle.theta_all, num_categories=5)
        labels_train = labels_from_theta(bundle.theta_train, num_categories=5)
        classical_all = classical_metric_matrices(
            bundle.x_all,
            labels_all,
            num_categories=5,
            metrics=METRICS,
            mahalanobis_ridge=1e-6,
        )
        train_optimum = classical_metric_matrices(
            bundle.x_train,
            labels_train,
            num_categories=5,
            metrics=METRICS,
            mahalanobis_ridge=1e-6,
        )
        theta_eval = np.eye(5, dtype=np.float64)
        for spec in trajectories:
            for metric in METRICS:
                print(
                    f"[experiment-a-ablation] repeat={repeat_idx} seed={repeat_seed} "
                    f"trajectory={spec.key} metric={metric}",
                    flush=True,
                )
                outputs = _train_trajectory(
                    spec=spec,
                    metric=metric,
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
                for output in outputs:
                    matrix = np.asarray(output.pop("matrix"), dtype=np.float64)
                    rows.append(
                        {
                            "repeat_idx": int(repeat_idx),
                            "repeat_seed": int(repeat_seed),
                            "metric": metric,
                            **output,
                            "relative_error_to_ground_truth": _mean_relative_error(matrix, ground_truth[metric]),
                            "relative_gap_to_train_optimum": _mean_relative_error(matrix, train_optimum[metric]),
                            "classical_all_error_to_ground_truth": _mean_relative_error(
                                classical_all[metric], ground_truth[metric]
                            ),
                            "train_optimum_error_to_ground_truth": _mean_relative_error(
                                train_optimum[metric], ground_truth[metric]
                            ),
                        }
                    )

    rows_frame = pd.DataFrame(rows)
    summary = (
        rows_frame.groupby(["variant", "variant_label", "metric"], as_index=False)
        .agg(
            mean_relative_error_to_ground_truth=("relative_error_to_ground_truth", "mean"),
            std_relative_error_to_ground_truth=("relative_error_to_ground_truth", "std"),
            mean_relative_gap_to_train_optimum=("relative_gap_to_train_optimum", "mean"),
            std_relative_gap_to_train_optimum=("relative_gap_to_train_optimum", "std"),
            mean_selected_epoch=("selected_epoch", "mean"),
            mean_stopped_epoch=("stopped_epoch", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
    )
    effects = _effect_rows(rows_frame)
    effect_summary = (
        effects.groupby(["effect", "ablation_variant", "reference_variant", "metric"], as_index=False)
        .agg(
            mean_delta_error_to_ground_truth=("delta_error_to_ground_truth", "mean"),
            std_delta_error_to_ground_truth=("delta_error_to_ground_truth", "std"),
            mean_delta_gap_to_train_optimum=("delta_gap_to_train_optimum", "mean"),
            std_delta_gap_to_train_optimum=("delta_gap_to_train_optimum", "std"),
        )
    )
    rows_path = output_dir / "experiment_a_ablation_rows.csv"
    summary_path = output_dir / "experiment_a_ablation_summary.csv"
    effects_path = output_dir / "experiment_a_ablation_effect_rows.csv"
    effect_summary_path = output_dir / "experiment_a_ablation_effect_summary.csv"
    config_path = output_dir / "experiment_a_ablation_config.json"
    rows_frame.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    effects.to_csv(effects_path, index=False)
    effect_summary.to_csv(effect_summary_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "trajectories": [asdict(spec) for spec in trajectories],
                "metrics": list(METRICS),
                "hidden_dim": int(args.hidden_dim),
                "depth": int(args.depth),
                "n_total": 3000,
                "n_train": 2400,
                "n_validation": 600,
                "n_repeats": int(args.n_repeats),
                "repeat_seeds": [int(args.seed) + idx for idx in range(int(args.n_repeats))],
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "smoke_epochs": int(args.smoke_epochs),
                "device": str(device),
                "total_runtime_seconds": float(time.perf_counter() - run_start),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    performance_png, performance_svg = plot_performance(summary, rows_frame, output_dir)
    effects_png, effects_svg = plot_effects(effect_summary, output_dir)
    return {
        "rows_csv": rows_path,
        "summary_csv": summary_path,
        "effect_rows_csv": effects_path,
        "effect_summary_csv": effect_summary_path,
        "config_json": config_path,
        "performance_png": performance_png,
        "performance_svg": performance_svg,
        "effects_png": effects_png,
        "effects_svg": effects_svg,
    }


def _style_axes(axes) -> None:
    for axis in np.asarray(axes).reshape(-1):
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_linewidth(1.8)
        axis.tick_params(width=1.8)


def plot_performance(summary: pd.DataFrame, rows: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 13,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0), sharex=True)
    colors = ("#4C72B0", "#7A7A7A", "#C44E52", "#DD8452", "#8172B3", "#55A868", "#937860")
    for axis, metric in zip(axes.reshape(-1), METRICS, strict=True):
        selected = summary[summary["metric"] == metric].set_index("variant")
        y = np.arange(len(VARIANT_ORDER), dtype=np.float64)
        values = [float(selected.loc[variant, "mean_relative_error_to_ground_truth"]) for variant in VARIANT_ORDER]
        errors = [float(selected.loc[variant, "std_relative_error_to_ground_truth"]) for variant in VARIANT_ORDER]
        axis.errorbar(values, y, xerr=errors, fmt="none", ecolor="black", elinewidth=1.5, capsize=3)
        for index, (value, color) in enumerate(zip(values, colors, strict=True)):
            axis.scatter(value, y[index], s=62, color=color, zorder=3)
        classical = float(rows[rows["metric"] == metric]["classical_all_error_to_ground_truth"].mean())
        axis.axvline(classical, color="black", linestyle="--", linewidth=1.8)
        axis.set_title(METRIC_LABELS[metric])
        axis.set_yticks(y)
        axis.set_yticklabels([VARIANT_LABELS[variant] for variant in VARIANT_ORDER])
        axis.invert_yaxis()
        axis.set_xlabel("Relative error to GT")
    _style_axes(axes)
    fig.tight_layout(w_pad=1.0, h_pad=1.0)
    png_path = output_dir / "experiment_a_ablation_performance.png"
    svg_path = output_dir / "experiment_a_ablation_performance.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def plot_effects(effect_summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    effect_order = (
        "shorter_training",
        "no_lr_decay",
        "best_instead_of_last",
        "resampled_instead_of_fixed_validation",
        "early_stopping",
        "combined_previous_like",
    )
    effect_labels = (
        "2k vs 20k epochs",
        "Constant vs decayed LR",
        "Best vs last checkpoint",
        "Resampled vs fixed val",
        "Early stop vs full run",
        "Previous-like vs A",
    )
    matrix = np.empty((len(effect_order), len(METRICS)), dtype=np.float64)
    indexed = effect_summary.set_index(["effect", "metric"])
    for row, effect in enumerate(effect_order):
        for col, metric in enumerate(METRICS):
            matrix[row, col] = float(indexed.loc[(effect, metric), "mean_delta_error_to_ground_truth"])
    limit = max(1e-6, float(np.max(np.abs(matrix))))
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(len(METRICS)))
    axis.set_xticklabels((r"Euc.$^2$", "Cos.", "Corr.", r"Mah.$^2$"), fontsize=13)
    axis.set_yticks(np.arange(len(effect_order)))
    axis.set_yticklabels(effect_labels)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if abs(matrix[row, col]) > 0.55 * limit else "black"
            axis.text(col, row, f"{matrix[row, col]:+.3f}", ha="center", va="center", fontsize=10, color=color)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.04, pad=0.03)
    colorbar.set_label("Change in relative error", fontsize=14)
    colorbar.ax.tick_params(labelsize=12)
    _style_axes([axis])
    fig.tight_layout()
    png_path = output_dir / "experiment_a_ablation_effects.png"
    svg_path = output_dir / "experiment_a_ablation_effects.svg"
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
