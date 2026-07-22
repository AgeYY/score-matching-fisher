#!/usr/bin/env python3
"""NLL-fine-tune a fixed Linear Fisher flow model."""

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
    finetune_flow_skl_cnf_likelihood,
)
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import require_device


SEED = 7
HIDDEN_DIM = 256
DEPTH = 5
PATH_SCHEDULE = "cosine"
NLL_EPOCHS = 500
NLL_PATIENCE = 100
NLL_BATCH_SIZE = 2048
NLL_LEARNING_RATE = 1e-3
NLL_ODE_STEPS = 32
FISHER_ODE_STEPS = 64
AFFINE_RIDGE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--x-dim", type=int, default=100)
    parser.add_argument("--n-total", type=int, default=10_000)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def _plot(
    *,
    theta: np.ndarray,
    ground_truth: np.ndarray,
    gkr: np.ndarray,
    flow: np.ndarray,
    flow_nll: np.ndarray,
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
            "legend.fontsize": 12.5,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    gkr_mae = float(np.mean(np.abs(gkr - ground_truth)))
    flow_mae = float(np.mean(np.abs(flow - ground_truth)))
    nll_mae = float(np.mean(np.abs(flow_nll - ground_truth)))
    fig, axis = plt.subplots(figsize=(6.8, 4.2))
    axis.plot(
        theta,
        ground_truth,
        color="black",
        linestyle="--",
        linewidth=2.5,
        label="Ground truth",
    )
    axis.plot(theta, gkr, color="C2", linewidth=2.2, label=f"GKR (MAE={gkr_mae:.3f})")
    axis.plot(
        theta,
        flow,
        color="C0",
        linewidth=2.2,
        label=f"Flow matching (MAE={flow_mae:.3f})",
    )
    axis.plot(
        theta,
        flow_nll,
        color="C1",
        linewidth=2.3,
        label=f"Flow matching + NLL (MAE={nll_mae:.3f})",
    )
    axis.set_xlabel(r"$\theta$")
    axis.set_ylabel("Linear Fisher information")
    axis.set_title(f"{int(x_dim)}D, {int(n_total):,} samples")
    axis.legend(frameon=False, loc="lower left")
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
    if x_dim < 1 or n_total < 2:
        raise ValueError("--x-dim must be positive and --n-total must be at least 2.")
    device = require_device(str(args.device))
    input_dir = (
        args.input_dir
        if args.input_dir is not None
        else REPO_ROOT / "data" / f"gkr_fixed_xdim{x_dim}_n{n_total}_linear"
    ).resolve()
    comparison_stem = f"gkr_flow_fixed_xdim{x_dim}_n{n_total}_linear"
    nll_stem = f"gkr_flow_nll_fixed_xdim{x_dim}_n{n_total}_linear"
    dataset_npz = input_dir / f"randamp_gaussian_sqrtd_xdim{x_dim}_n{n_total}.npz"
    comparison_npz = input_dir / f"{comparison_stem}_results.npz"
    fm_model_path = input_dir / "flow_linear_selected_model.pt"
    for path in (dataset_npz, comparison_npz, fm_model_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    bundle = load_shared_dataset_npz(dataset_npz)
    if int(bundle.x_train.shape[1]) != x_dim:
        raise ValueError("Dataset dimensionality does not match --x-dim.")
    if int(bundle.x_train.shape[0] + bundle.x_validation.shape[0]) != n_total:
        raise ValueError("Dataset size does not match --n-total.")
    with np.load(comparison_npz, allow_pickle=False) as cached:
        theta_grid = np.asarray(cached["theta_grid"], dtype=np.float64)
        theta_midpoints = np.asarray(cached["theta_midpoints"], dtype=np.float64)
        ground_truth = np.asarray(cached["ground_truth_linear_fisher"], dtype=np.float64)
        gkr = np.asarray(cached["gkr_linear_fisher"], dtype=np.float64)
        flow = np.asarray(cached["flow_linear_fisher"], dtype=np.float64)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model = build_flow_skl_model(
        velocity_family="condition_affine",
        theta_dim=1,
        x_dim=int(bundle.x_train.shape[1]),
        hidden_dim=HIDDEN_DIM,
        depth=DEPTH,
        quadrature_steps=64,
        path_schedule=PATH_SCHEDULE,
        divergence_estimator="exact",
        hutchinson_probes=1,
    ).to(device)
    state = torch.load(fm_model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    metadata = finetune_flow_skl_cnf_likelihood(
        model=model,
        theta_train=np.asarray(bundle.theta_train, dtype=np.float64).reshape(-1, 1),
        x_train=np.asarray(bundle.x_train, dtype=np.float64),
        theta_val=np.asarray(bundle.theta_validation, dtype=np.float64).reshape(-1, 1),
        x_val=np.asarray(bundle.x_validation, dtype=np.float64),
        device=device,
        epochs=NLL_EPOCHS,
        batch_size=NLL_BATCH_SIZE,
        lr=NLL_LEARNING_RATE,
        weight_decay=0.0,
        ode_steps=NLL_ODE_STEPS,
        ode_method="midpoint",
        patience=NLL_PATIENCE,
        min_delta=1e-4,
        ema_alpha=0.05,
        max_grad_norm=10.0,
        checkpoint_selection="best",
        log_every=10,
    )
    fine_result = estimate_affine_mixed_symmetric_kl_fisher(
        model=model,
        theta_all=theta_grid,
        device=device,
        ridge=AFFINE_RIDGE,
        ode_steps=FISHER_ODE_STEPS,
    )
    flow_nll = np.asarray(fine_result["fisher"], dtype=np.float64)

    fine_model_path = input_dir / "flow_linear_nll_selected_model.pt"
    torch.save(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        fine_model_path,
    )
    results_npz = input_dir / f"{nll_stem}_results.npz"
    np.savez_compressed(
        results_npz,
        theta_grid=theta_grid,
        theta_midpoints=theta_midpoints,
        ground_truth_linear_fisher=ground_truth,
        gkr_linear_fisher=gkr,
        flow_linear_fisher=flow,
        flow_nll_linear_fisher=flow_nll,
        nll_train_losses=np.asarray(metadata["train_nll_losses"], dtype=np.float64),
        nll_validation_losses=np.asarray(metadata["val_nll_losses"], dtype=np.float64),
        nll_validation_monitor_losses=np.asarray(
            metadata["val_monitor_nll_losses"], dtype=np.float64
        ),
    )

    figure_stem = input_dir / f"{nll_stem}_fisher"
    _plot(
        theta=theta_midpoints[:, 0],
        ground_truth=ground_truth,
        gkr=gkr,
        flow=flow,
        flow_nll=flow_nll,
        x_dim=x_dim,
        n_total=n_total,
        output_stem=figure_stem,
    )
    summary = {
        "dataset_npz": str(dataset_npz),
        "x_dim": int(bundle.x_train.shape[1]),
        "n_total": int(bundle.x_train.shape[0] + bundle.x_validation.shape[0]),
        "n_train": int(bundle.x_train.shape[0]),
        "n_validation": int(bundle.x_validation.shape[0]),
        "device": str(device),
        "seed": SEED,
        "gkr_linear_fisher_mae": float(np.mean(np.abs(gkr - ground_truth))),
        "flow_linear_fisher_mae": float(np.mean(np.abs(flow - ground_truth))),
        "flow_nll_linear_fisher_mae": float(np.mean(np.abs(flow_nll - ground_truth))),
        "nll_training": {
            "max_epochs": NLL_EPOCHS,
            "early_stopping_patience": NLL_PATIENCE,
            "selected_epoch": int(metadata["selected_epoch"]),
            "stopped_epoch": int(metadata["stopped_epoch"]),
            "stopped_early": bool(metadata["stopped_early"]),
            "initial_validation_nll": float(metadata["initial_val_nll"]),
            "selected_validation_nll": float(metadata["selected_val_nll"]),
            "batch_size": NLL_BATCH_SIZE,
            "learning_rate": NLL_LEARNING_RATE,
            "ode_steps": NLL_ODE_STEPS,
            "ode_method": "midpoint",
        },
        "model_path": str(fine_model_path),
        "results_npz": str(results_npz),
        "figure_png": str(figure_stem.with_suffix(".png")),
        "figure_svg": str(figure_stem.with_suffix(".svg")),
    }
    summary_path = input_dir / f"{nll_stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
