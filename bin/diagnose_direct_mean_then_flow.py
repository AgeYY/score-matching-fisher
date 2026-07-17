#!/usr/bin/env python3
"""Fine-tune a directly regressed mean model with the joint flow-matching objective."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.flow_matching_skl import (
    build_flow_skl_model,
    estimate_affine_mixed_symmetric_kl_fisher,
    train_flow_skl_model,
)
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_meta, require_device
from global_setting import DEFAULT_EARLY_STOPPING_PATIENCE, DEFAULT_TRAINING_MAX_EPOCHS


SEED = 7
BATCH_SIZE = 2048
LEARNING_RATE = 1e-4
HIDDEN_DIM = 256
DEPTH = 5
PATH_SCHEDULE = "cosine"
T_EPS = 5e-4
FIXED_VALIDATION_PATHS = 10
ODE_STEPS = 64
AFFINE_RIDGE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--gkr-result-npz", type=Path, required=True)
    parser.add_argument("--direct-model", type=Path, required=True)
    parser.add_argument("--baseline-flow-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    return parser.parse_args()


def _style_axis(axis: plt.Axes) -> None:
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(width=1.8)


def _save_figure(
    *,
    theta: np.ndarray,
    ground_truth_mean: np.ndarray,
    pretrained_mean: np.ndarray,
    finetuned_mean: np.ndarray,
    ground_truth_fisher: np.ndarray,
    gkr_fisher: np.ndarray,
    baseline_flow_fisher: np.ndarray,
    finetuned_fisher: np.ndarray,
    train_losses: np.ndarray,
    validation_losses: np.ndarray,
    output_stem: Path,
) -> None:
    mean_min = float(min(np.min(ground_truth_mean), np.min(finetuned_mean)))
    mean_max = float(max(np.max(ground_truth_mean), np.max(finetuned_mean)))
    mean_extent = [float(theta[0]), float(theta[-1]), 0.5, ground_truth_mean.shape[1] + 0.5]
    pretrained_rmse = float(np.sqrt(np.mean((pretrained_mean - ground_truth_mean) ** 2)))
    finetuned_rmse = float(np.sqrt(np.mean((finetuned_mean - ground_truth_mean) ** 2)))
    baseline_fisher_mae = float(np.mean(np.abs(baseline_flow_fisher - ground_truth_fisher)))
    finetuned_fisher_mae = float(np.mean(np.abs(finetuned_fisher - ground_truth_fisher)))
    gkr_fisher_mae = float(np.mean(np.abs(gkr_fisher - ground_truth_fisher)))

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 11,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0), constrained_layout=True)
    mean_axes = axes[0]
    images = []
    for axis, values, title in (
        (mean_axes[0], ground_truth_mean, "Ground-truth mean"),
        (
            mean_axes[1],
            finetuned_mean,
            f"Direct + FM mean ({finetuned_rmse:.3f})",
        ),
    ):
        images.append(
            axis.imshow(
                values.T,
                aspect="auto",
                origin="lower",
                extent=mean_extent,
                interpolation="nearest",
                cmap="viridis",
                vmin=mean_min,
                vmax=mean_max,
            )
        )
        axis.set_title(title)
        axis.set_xlabel(r"$\theta$")
        _style_axis(axis)
    mean_axes[0].set_ylabel("Response dimension")
    colorbar = fig.colorbar(images[-1], ax=mean_axes, orientation="horizontal", pad=0.14, fraction=0.08)
    colorbar.set_label("Conditional mean")
    colorbar.ax.tick_params(width=1.8)

    fisher_axis = axes[1, 0]
    fisher_axis.plot(theta, ground_truth_fisher, color="black", linestyle="--", linewidth=2.4, label="Ground truth")
    fisher_axis.plot(theta, gkr_fisher, color="C2", linewidth=2.0, label=f"GKR ({gkr_fisher_mae:.3f})")
    fisher_axis.plot(
        theta,
        baseline_flow_fisher,
        color="C0",
        linewidth=2.0,
        label=f"Joint FM ({baseline_fisher_mae:.3f})",
    )
    fisher_axis.plot(
        theta,
        finetuned_fisher,
        color="C3",
        linewidth=2.0,
        label=f"Direct + FM ({finetuned_fisher_mae:.3f})",
    )
    fisher_axis.set_xlabel(r"$\theta$")
    fisher_axis.set_ylabel("Linear Fisher information")
    fisher_axis.legend(frameon=False)
    _style_axis(fisher_axis)

    loss_axis = axes[1, 1]
    epochs = np.arange(1, train_losses.size + 1)
    loss_axis.plot(epochs, train_losses, color="C0", linewidth=2.0, label="Training")
    loss_axis.plot(epochs, validation_losses, color="C1", linewidth=2.0, label="Validation")
    loss_axis.set_xlabel("FM epoch")
    loss_axis.set_ylabel("FM loss")
    loss_axis.set_title(f"Pretrained mean RMSE={pretrained_rmse:.3f}")
    loss_axis.legend(frameon=False)
    _style_axis(loss_axis)

    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_npz = args.dataset_npz.resolve()
    gkr_result_npz = args.gkr_result_npz.resolve()
    direct_model_path = args.direct_model.resolve()
    baseline_flow_result_path = args.baseline_flow_result.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = require_device(str(args.device))

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    bundle = load_shared_dataset_npz(dataset_npz)
    dataset = build_dataset_from_meta(dict(bundle.meta))
    with np.load(gkr_result_npz, allow_pickle=False) as archive:
        theta_grid = np.asarray(archive["theta_grid"], dtype=np.float64)
        theta_midpoints = np.asarray(archive["theta_midpoints"], dtype=np.float64)
        ground_truth_fisher = np.asarray(archive["ground_truth_linear_fisher"], dtype=np.float64)
        gkr_fisher = np.asarray(archive["gkr_linear_fisher"], dtype=np.float64)
    with np.load(baseline_flow_result_path, allow_pickle=False) as archive:
        baseline_flow_fisher = np.asarray(archive["flow_linear_fisher"], dtype=np.float64)

    ground_truth_mean = np.asarray(dataset.tuning_curve(theta_midpoints), dtype=np.float64)
    model = build_flow_skl_model(
        velocity_family="condition_affine",
        theta_dim=1,
        x_dim=ground_truth_mean.shape[1],
        hidden_dim=HIDDEN_DIM,
        depth=DEPTH,
        quadrature_steps=64,
        path_schedule=PATH_SCHEDULE,
        divergence_estimator="exact",
    ).to(device)
    model.load_state_dict(
        torch.load(direct_model_path, map_location=device, weights_only=True)
    )
    model.eval()
    theta_tensor = torch.from_numpy(theta_midpoints.astype(np.float32)).to(device)
    with torch.no_grad():
        pretrained_mean = model.endpoint_mean(theta_tensor).detach().cpu().numpy().astype(np.float64)

    training = train_flow_skl_model(
        model=model,
        theta_train=np.asarray(bundle.theta_train, dtype=np.float64).reshape(-1, 1),
        x_train=np.asarray(bundle.x_train, dtype=np.float64),
        theta_val=np.asarray(bundle.theta_validation, dtype=np.float64).reshape(-1, 1),
        x_val=np.asarray(bundle.x_validation, dtype=np.float64),
        device=device,
        velocity_family="condition_affine",
        path_schedule=PATH_SCHEDULE,
        epochs=DEFAULT_TRAINING_MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        lr_schedule="constant",
        weight_decay=0.0,
        t_eps=T_EPS,
        patience=DEFAULT_EARLY_STOPPING_PATIENCE,
        min_delta=1e-4,
        ema_alpha=0.05,
        max_grad_norm=10.0,
        log_every=50,
        checkpoint_selection="best",
        best_checkpoint_metric="flow_matching",
        fixed_validation=True,
        fixed_validation_paths=FIXED_VALIDATION_PATHS,
        validation_seed=SEED + 10_000,
    )
    model.eval()
    with torch.no_grad():
        finetuned_mean = model.endpoint_mean(theta_tensor).detach().cpu().numpy().astype(np.float64)
    fisher_result = estimate_affine_mixed_symmetric_kl_fisher(
        model=model,
        theta_all=theta_grid,
        device=device,
        ridge=AFFINE_RIDGE,
        ode_steps=ODE_STEPS,
    )
    finetuned_fisher = np.asarray(fisher_result["fisher"], dtype=np.float64)

    model_path = output_dir / "direct_mean_then_flow_selected_model.pt"
    torch.save(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        model_path,
    )
    result_path = output_dir / "direct_mean_then_flow_results.npz"
    np.savez_compressed(
        result_path,
        theta_grid=theta_grid,
        theta_midpoints=theta_midpoints,
        ground_truth_mean=ground_truth_mean,
        pretrained_mean=pretrained_mean,
        finetuned_mean=finetuned_mean,
        ground_truth_linear_fisher=ground_truth_fisher,
        gkr_linear_fisher=gkr_fisher,
        baseline_flow_linear_fisher=baseline_flow_fisher,
        finetuned_linear_fisher=finetuned_fisher,
        train_losses=np.asarray(training["train_losses"], dtype=np.float64),
        validation_losses=np.asarray(training["val_losses"], dtype=np.float64),
        validation_monitor_losses=np.asarray(training["val_monitor_losses"], dtype=np.float64),
    )
    figure_stem = output_dir / "direct_mean_then_flow_mean_fisher"
    _save_figure(
        theta=theta_midpoints[:, 0],
        ground_truth_mean=ground_truth_mean,
        pretrained_mean=pretrained_mean,
        finetuned_mean=finetuned_mean,
        ground_truth_fisher=ground_truth_fisher,
        gkr_fisher=gkr_fisher,
        baseline_flow_fisher=baseline_flow_fisher,
        finetuned_fisher=finetuned_fisher,
        train_losses=np.asarray(training["train_losses"], dtype=np.float64),
        validation_losses=np.asarray(training["val_monitor_losses"], dtype=np.float64),
        output_stem=figure_stem,
    )
    summary = {
        "dataset_npz": str(dataset_npz),
        "device": str(device),
        "n_train": int(bundle.x_train.shape[0]),
        "n_validation": int(bundle.x_validation.shape[0]),
        "direct_pretrained_mean_rmse": float(
            np.sqrt(np.mean((pretrained_mean - ground_truth_mean) ** 2))
        ),
        "finetuned_mean_rmse": float(
            np.sqrt(np.mean((finetuned_mean - ground_truth_mean) ** 2))
        ),
        "gkr_linear_fisher_mae": float(np.mean(np.abs(gkr_fisher - ground_truth_fisher))),
        "baseline_flow_linear_fisher_mae": float(
            np.mean(np.abs(baseline_flow_fisher - ground_truth_fisher))
        ),
        "finetuned_linear_fisher_mae": float(
            np.mean(np.abs(finetuned_fisher - ground_truth_fisher))
        ),
        "fm_best_epoch": int(training["best_epoch"]),
        "fm_stopped_epoch": int(training["stopped_epoch"]),
        "fm_best_validation_loss": float(training["best_val_loss"]),
        "model_path": str(model_path),
        "result_npz": str(result_path),
        "figure_png": str(figure_stem.with_suffix(".png")),
        "figure_svg": str(figure_stem.with_suffix(".svg")),
    }
    summary_path = output_dir / "direct_mean_then_flow_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
