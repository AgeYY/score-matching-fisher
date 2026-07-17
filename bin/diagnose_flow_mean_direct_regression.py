#!/usr/bin/env python3
"""Test whether direct mean regression fixes a flow model's conditional-mean artifact."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.flow_matching_skl import build_flow_skl_model
from fisher.model_weight_ema import scalar_val_ema_update
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_meta, require_device
from global_setting import DEFAULT_EARLY_STOPPING_PATIENCE, DEFAULT_TRAINING_MAX_EPOCHS


SEED = 7
BATCH_SIZE = 2048
LEARNING_RATE = 1e-4
HIDDEN_DIM = 256
DEPTH = 5
MIN_DELTA = 1e-4
EMA_ALPHA = 0.05
MAX_GRAD_NORM = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--gkr-result-npz", type=Path, required=True)
    parser.add_argument("--flow-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    return parser.parse_args()


def _make_loader(theta: np.ndarray, x: np.ndarray, *, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(theta, dtype=np.float32).reshape(-1, 1)),
        torch.from_numpy(np.asarray(x, dtype=np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(SEED + (1 if shuffle else 2))
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
    )


def _train_direct_mean(
    *,
    model: torch.nn.Module,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_validation: np.ndarray,
    x_validation: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    train_loader = _make_loader(theta_train, x_train, shuffle=True)
    validation_loader = _make_loader(theta_validation, x_validation, shuffle=False)
    parameters = list(model.b_net.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=0.0)

    train_losses: list[float] = []
    validation_losses: list[float] = []
    validation_monitor_losses: list[float] = []
    validation_ema: float | None = None
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    stopped_epoch = DEFAULT_TRAINING_MAX_EPOCHS

    for epoch in range(1, DEFAULT_TRAINING_MAX_EPOCHS + 1):
        model.train()
        epoch_train: list[float] = []
        for theta_batch, x_batch in train_loader:
            theta_batch = theta_batch.to(device)
            x_batch = x_batch.to(device)
            loss = torch.mean((model.endpoint_mean(theta_batch) - x_batch) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM)
            optimizer.step()
            epoch_train.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(epoch_train))
        train_losses.append(train_loss)

        model.eval()
        epoch_validation: list[float] = []
        with torch.no_grad():
            for theta_batch, x_batch in validation_loader:
                theta_batch = theta_batch.to(device)
                x_batch = x_batch.to(device)
                loss = torch.mean((model.endpoint_mean(theta_batch) - x_batch) ** 2)
                epoch_validation.append(float(loss.detach().cpu()))
        validation_loss = float(np.mean(epoch_validation))
        validation_losses.append(validation_loss)
        validation_ema = scalar_val_ema_update(validation_ema, validation_loss, EMA_ALPHA)
        monitor_loss = float(validation_ema)
        validation_monitor_losses.append(monitor_loss)

        if monitor_loss < best_loss - MIN_DELTA:
            best_loss = monitor_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 100 == 0:
            print(
                f"[direct-mean {epoch:5d}/{DEFAULT_TRAINING_MAX_EPOCHS}] "
                f"train={train_loss:.6f} val={validation_loss:.6f} "
                f"val_smooth={monitor_loss:.6f} best={best_loss:.6f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
        if patience_counter >= DEFAULT_EARLY_STOPPING_PATIENCE:
            stopped_epoch = epoch
            print(
                f"[direct-mean early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best={best_loss:.6f}",
                flush=True,
            )
            break

    model.load_state_dict(best_state)
    return {
        "train_losses": np.asarray(train_losses, dtype=np.float64),
        "validation_losses": np.asarray(validation_losses, dtype=np.float64),
        "validation_monitor_losses": np.asarray(validation_monitor_losses, dtype=np.float64),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "best_validation_loss": float(best_loss),
    }


def _save_mean_figure(
    *,
    theta: np.ndarray,
    ground_truth: np.ndarray,
    gkr: np.ndarray,
    flow: np.ndarray,
    direct: np.ndarray,
    output_stem: Path,
) -> None:
    arrays = (ground_truth, gkr, flow, direct)
    value_min = float(min(np.min(values) for values in arrays))
    value_max = float(max(np.max(values) for values in arrays))
    extent = [float(theta[0]), float(theta[-1]), 0.5, ground_truth.shape[1] + 0.5]
    panels = (
        (ground_truth, "Ground truth"),
        (gkr, f"GKR ({np.sqrt(np.mean((gkr - ground_truth) ** 2)):.3f})"),
        (flow, f"Flow matching ({np.sqrt(np.mean((flow - ground_truth) ** 2)):.3f})"),
        (direct, f"Direct regression ({np.sqrt(np.mean((direct - ground_truth) ** 2)):.3f})"),
    )

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8.0, 7.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for index, (axis, (values, title)) in enumerate(zip(axes.flat, panels, strict=True)):
        image = axis.imshow(
            values.T,
            aspect="auto",
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap="viridis",
            vmin=value_min,
            vmax=value_max,
        )
        axis.set_title(title)
        row, column = divmod(index, 2)
        if row == 1:
            axis.set_xlabel(r"$\theta$")
        if column == 0:
            axis.set_ylabel("Response dimension")
        axis.tick_params(labelbottom=row == 1, labelleft=column == 0)
        for spine in axis.spines.values():
            spine.set_linewidth(1.8)
        axis.tick_params(width=1.8)
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, orientation="horizontal", pad=0.05, fraction=0.05)
    colorbar.set_label("Conditional mean")
    colorbar.ax.tick_params(width=1.8)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".svg"))
    plt.close(fig)


def _save_loss_figure(metadata: dict[str, object], output_stem: Path) -> None:
    train = np.asarray(metadata["train_losses"], dtype=np.float64)
    validation = np.asarray(metadata["validation_monitor_losses"], dtype=np.float64)
    epochs = np.arange(1, train.size + 1)
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axis = plt.subplots(figsize=(4.0, 3.5))
    axis.plot(epochs, train, linewidth=2.0, label="Training")
    axis.plot(epochs, validation, linewidth=2.0, label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean regression loss")
    axis.legend(frameon=False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(width=1.8)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_npz = args.dataset_npz.resolve()
    gkr_result_npz = args.gkr_result_npz.resolve()
    flow_model_path = args.flow_model.resolve()
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
        theta = np.asarray(archive["theta_midpoints"], dtype=np.float64).reshape(-1, 1)
        gkr_mean = np.asarray(archive["gkr_mean"], dtype=np.float64)
    ground_truth_mean = np.asarray(dataset.tuning_curve(theta), dtype=np.float64)

    flow_model = build_flow_skl_model(
        velocity_family="condition_affine",
        theta_dim=1,
        x_dim=ground_truth_mean.shape[1],
        hidden_dim=HIDDEN_DIM,
        depth=DEPTH,
        quadrature_steps=64,
        path_schedule="cosine",
        divergence_estimator="exact",
    ).to(device)
    flow_model.load_state_dict(
        torch.load(flow_model_path, map_location=device, weights_only=True)
    )
    flow_model.eval()

    direct_model = build_flow_skl_model(
        velocity_family="condition_affine",
        theta_dim=1,
        x_dim=ground_truth_mean.shape[1],
        hidden_dim=HIDDEN_DIM,
        depth=DEPTH,
        quadrature_steps=64,
        path_schedule="cosine",
        divergence_estimator="exact",
    ).to(device)
    training = _train_direct_mean(
        model=direct_model,
        theta_train=bundle.theta_train,
        x_train=bundle.x_train,
        theta_validation=bundle.theta_validation,
        x_validation=bundle.x_validation,
        device=device,
    )

    theta_tensor = torch.from_numpy(theta.astype(np.float32)).to(device)
    with torch.no_grad():
        flow_mean = flow_model.endpoint_mean(theta_tensor).detach().cpu().numpy().astype(np.float64)
        direct_mean = direct_model.endpoint_mean(theta_tensor).detach().cpu().numpy().astype(np.float64)

    model_path = output_dir / "direct_mean_selected_model.pt"
    torch.save(
        {key: value.detach().cpu() for key, value in direct_model.state_dict().items()},
        model_path,
    )
    result_path = output_dir / "direct_mean_diagnostic_results.npz"
    np.savez_compressed(
        result_path,
        theta=theta,
        ground_truth_mean=ground_truth_mean,
        gkr_mean=gkr_mean,
        flow_mean=flow_mean,
        direct_mean=direct_mean,
        train_losses=training["train_losses"],
        validation_losses=training["validation_losses"],
        validation_monitor_losses=training["validation_monitor_losses"],
    )
    mean_figure_stem = output_dir / "ground_truth_gkr_flow_direct_mean_heatmaps"
    loss_figure_stem = output_dir / "direct_mean_training_loss"
    _save_mean_figure(
        theta=theta[:, 0],
        ground_truth=ground_truth_mean,
        gkr=gkr_mean,
        flow=flow_mean,
        direct=direct_mean,
        output_stem=mean_figure_stem,
    )
    _save_loss_figure(training, loss_figure_stem)

    summary = {
        "dataset_npz": str(dataset_npz),
        "device": str(device),
        "n_train": int(bundle.x_train.shape[0]),
        "n_validation": int(bundle.x_validation.shape[0]),
        "hidden_dim": HIDDEN_DIM,
        "depth": DEPTH,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_epochs": DEFAULT_TRAINING_MAX_EPOCHS,
        "early_stopping_patience": DEFAULT_EARLY_STOPPING_PATIENCE,
        "best_epoch": int(training["best_epoch"]),
        "stopped_epoch": int(training["stopped_epoch"]),
        "best_validation_loss": float(training["best_validation_loss"]),
        "gkr_mean_rmse": float(np.sqrt(np.mean((gkr_mean - ground_truth_mean) ** 2))),
        "flow_mean_rmse": float(np.sqrt(np.mean((flow_mean - ground_truth_mean) ** 2))),
        "direct_mean_rmse": float(np.sqrt(np.mean((direct_mean - ground_truth_mean) ** 2))),
        "gkr_mean_mae": float(np.mean(np.abs(gkr_mean - ground_truth_mean))),
        "flow_mean_mae": float(np.mean(np.abs(flow_mean - ground_truth_mean))),
        "direct_mean_mae": float(np.mean(np.abs(direct_mean - ground_truth_mean))),
        "model_path": str(model_path),
        "result_npz": str(result_path),
        "mean_figure_png": str(mean_figure_stem.with_suffix(".png")),
        "loss_figure_png": str(loss_figure_stem.with_suffix(".png")),
    }
    summary_path = output_dir / "direct_mean_diagnostic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
