#!/usr/bin/env python3
"""Fit Linear Fisher with flow matching and compare with a fixed GKR run."""

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
from fisher.shared_fisher_est import require_device
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
DEFAULT_THETA_SPACING = 0.4


def _spacing_suffix(spacing: float) -> str:
    if np.isclose(float(spacing), DEFAULT_THETA_SPACING):
        return ""
    return "_h" + f"{float(spacing):g}".replace(".", "p")


def _signed_value_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--x-dim", type=int, default=100)
    parser.add_argument("--n-total", type=int, default=10_000)
    parser.add_argument("--theta-spacing", type=float, default=DEFAULT_THETA_SPACING)
    parser.add_argument("--training-seed", type=int, default=SEED)
    parser.add_argument(
        "--theta-embedding",
        choices=("identity", "gaussian-rbf"),
        default="gaussian-rbf",
    )
    parser.add_argument("--theta-rbf-num-centers", type=int, default=8)
    parser.add_argument("--theta-rbf-bandwidth", type=float, default=None)
    parser.add_argument(
        "--theta-input-shift",
        type=float,
        default=0.0,
        help="Add this constant to theta before passing it to the flow model.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Evaluate an existing flow checkpoint instead of retraining.",
    )
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
            "legend.fontsize": 13,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    gkr_mae = float(np.mean(np.abs(gkr - ground_truth)))
    flow_mae = float(np.mean(np.abs(flow - ground_truth)))
    fig, axis = plt.subplots(figsize=(4.0, 3.5))
    axis.plot(
        theta,
        ground_truth,
        color="black",
        linestyle="--",
        linewidth=2.5,
        label="Ground truth",
    )
    axis.plot(theta, gkr, color="C2", linewidth=2.3, label=f"GKR ({gkr_mae:.3f})")
    axis.plot(
        theta,
        flow,
        color="C0",
        linewidth=2.3,
        label=f"Flow ({flow_mae:.3f})",
    )
    axis.set_xlabel(r"$\theta$")
    axis.set_ylabel("Linear Fisher information")
    axis.legend(frameon=False, fontsize=11)
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
    theta_spacing = float(args.theta_spacing)
    training_seed = int(args.training_seed)
    theta_embedding = str(args.theta_embedding).replace("-", "_")
    theta_rbf_num_centers = int(args.theta_rbf_num_centers)
    theta_rbf_bandwidth = args.theta_rbf_bandwidth
    theta_input_shift = float(args.theta_input_shift)
    if x_dim < 1 or n_total < 2:
        raise ValueError("--x-dim must be positive and --n-total must be at least 2.")
    if not np.isfinite(theta_spacing) or theta_spacing <= 0.0:
        raise ValueError("--theta-spacing must be finite and positive.")
    if not np.isfinite(theta_input_shift):
        raise ValueError("--theta-input-shift must be finite.")
    if theta_rbf_num_centers < 2:
        raise ValueError("--theta-rbf-num-centers must be >= 2.")
    if theta_rbf_bandwidth is not None and (
        not np.isfinite(float(theta_rbf_bandwidth)) or float(theta_rbf_bandwidth) <= 0.0
    ):
        raise ValueError("--theta-rbf-bandwidth must be finite and positive.")
    device = require_device(str(args.device))
    input_dir = (
        args.input_dir
        if args.input_dir is not None
        else REPO_ROOT / "data" / f"gkr_fixed_xdim{x_dim}_n{n_total}_linear"
    ).resolve()
    spacing_suffix = _spacing_suffix(theta_spacing)
    artifact_stem = f"gkr_fixed_xdim{x_dim}_n{n_total}_linear{spacing_suffix}"
    comparison_stem = f"gkr_flow_fixed_xdim{x_dim}_n{n_total}_linear{spacing_suffix}"
    seed_suffix = "" if training_seed == SEED else f"_trainseed{training_seed}"
    shift_suffix = (
        ""
        if np.isclose(theta_input_shift, 0.0)
        else f"_thetashift{_signed_value_token(theta_input_shift)}"
    )
    embedding_suffix = ""
    if theta_embedding == "gaussian_rbf":
        bandwidth_token = (
            "auto"
            if theta_rbf_bandwidth is None
            else f"{float(theta_rbf_bandwidth):g}".replace(".", "p")
        )
        embedding_suffix = f"_rbf{theta_rbf_num_centers}_h{bandwidth_token}"
    comparison_stem += embedding_suffix + seed_suffix + shift_suffix
    dataset_npz = input_dir / f"randamp_gaussian_sqrtd_xdim{x_dim}_n{n_total}.npz"
    gkr_npz = input_dir / f"{artifact_stem}_results.npz"
    if not dataset_npz.is_file() or not gkr_npz.is_file():
        raise FileNotFoundError(
            "Run bin/tmp_fit_gkr_fixed_xdim100_n10000_linear.py before this script."
        )

    bundle = load_shared_dataset_npz(dataset_npz)
    dataset_seed = int(bundle.meta.get("seed", SEED))
    if int(bundle.x_train.shape[1]) != x_dim:
        raise ValueError("Dataset dimensionality does not match --x-dim.")
    if int(bundle.x_train.shape[0] + bundle.x_validation.shape[0]) != n_total:
        raise ValueError("Dataset size does not match --n-total.")
    with np.load(gkr_npz, allow_pickle=False) as cached:
        theta_grid = np.asarray(cached["theta_grid"], dtype=np.float64)
        theta_midpoints = np.asarray(cached["theta_midpoints"], dtype=np.float64)
        ground_truth = np.asarray(cached["ground_truth_linear_fisher"], dtype=np.float64)
        gkr = np.asarray(cached["gkr_linear_fisher"], dtype=np.float64)

    torch.manual_seed(training_seed)
    np.random.seed(training_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training_seed)

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
        theta_embedding=theta_embedding,
        theta_rbf_num_centers=theta_rbf_num_centers,
        theta_rbf_lower=float(bundle.meta["theta_low"]),
        theta_rbf_upper=float(bundle.meta["theta_high"]),
        theta_rbf_bandwidth=theta_rbf_bandwidth,
    ).to(device)
    training_kwargs = dict(
        model=model,
        theta_train=(
            np.asarray(bundle.theta_train, dtype=np.float64).reshape(-1, 1)
            + theta_input_shift
        ),
        x_train=np.asarray(bundle.x_train, dtype=np.float64),
        theta_val=(
            np.asarray(bundle.theta_validation, dtype=np.float64).reshape(-1, 1)
            + theta_input_shift
        ),
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
        validation_seed=training_seed + 10_000,
    )
    metadata = None
    if args.model_path is None:
        metadata = train_flow_skl_model(**training_kwargs)
    else:
        model.load_state_dict(
            torch.load(args.model_path.resolve(), map_location=device, weights_only=True)
        )
    flow_result = estimate_affine_mixed_symmetric_kl_fisher(
        model=model,
        theta_all=theta_grid + theta_input_shift,
        device=device,
        ridge=AFFINE_RIDGE,
        ode_steps=ODE_STEPS,
    )
    flow = np.asarray(flow_result["fisher"], dtype=np.float64)
    flow_mae = float(np.mean(np.abs(flow - ground_truth)))
    gkr_mae = float(np.mean(np.abs(gkr - ground_truth)))

    model_path = (
        args.model_path.resolve()
        if args.model_path is not None
        else input_dir / f"flow_linear{embedding_suffix}{seed_suffix}{shift_suffix}_selected_model.pt"
    )
    if args.model_path is None:
        torch.save(
            {key: value.detach().cpu() for key, value in model.state_dict().items()},
            model_path,
        )
    results_npz = input_dir / f"{comparison_stem}_results.npz"
    np.savez_compressed(
        results_npz,
        theta_grid=theta_grid,
        theta_midpoints=theta_midpoints,
        ground_truth_linear_fisher=ground_truth,
        gkr_linear_fisher=gkr,
        flow_linear_fisher=flow,
        flow_adjacent_symmetric_kl=np.asarray(
            flow_result["adjacent_symmetric_kl"], dtype=np.float64
        ),
        flow_train_losses=(
            np.asarray(metadata["train_losses"], dtype=np.float64)
            if metadata is not None
            else np.empty(0, dtype=np.float64)
        ),
        flow_validation_losses=(
            np.asarray(metadata["val_losses"], dtype=np.float64)
            if metadata is not None
            else np.empty(0, dtype=np.float64)
        ),
        flow_validation_monitor_losses=np.asarray(
            metadata["val_monitor_losses"] if metadata is not None else [], dtype=np.float64
        ),
    )

    figure_stem = input_dir / f"{comparison_stem}_fisher"
    _plot(
        theta=theta_midpoints[:, 0],
        ground_truth=ground_truth,
        gkr=gkr,
        flow=flow,
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
        "dataset_seed": dataset_seed,
        "training_seed": training_seed,
        "gkr_linear_fisher_mae": gkr_mae,
        "flow_linear_fisher_mae": flow_mae,
        "theta_spacing": theta_spacing,
        "theta_input_shift": theta_input_shift,
        "flow_training": {
            "velocity_family": "condition_affine",
            "max_epochs": DEFAULT_TRAINING_MAX_EPOCHS,
            "early_stopping_patience": DEFAULT_EARLY_STOPPING_PATIENCE,
            "selected_epoch": int(metadata["selected_epoch"]) if metadata is not None else None,
            "stopped_epoch": int(metadata["stopped_epoch"]) if metadata is not None else None,
            "stopped_early": bool(metadata["stopped_early"]) if metadata is not None else None,
            "best_validation_loss": float(metadata["best_val_loss"]) if metadata is not None else None,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "learning_rate_schedule": "constant",
            "reused_checkpoint": args.model_path is not None,
            "hidden_dim": HIDDEN_DIM,
            "depth": DEPTH,
            "theta_embedding": theta_embedding,
            "theta_rbf_num_centers": theta_rbf_num_centers,
            "theta_rbf_bandwidth": (
                float(theta_rbf_bandwidth)
                if theta_rbf_bandwidth is not None
                else float(bundle.meta["theta_high"] - bundle.meta["theta_low"])
                / float(theta_rbf_num_centers - 1)
            ),
            "path_schedule": PATH_SCHEDULE,
            "fixed_validation_paths": FIXED_VALIDATION_PATHS,
            "ode_steps": ODE_STEPS,
        },
        "model_path": str(model_path),
        "results_npz": str(results_npz),
        "figure_png": str(figure_stem.with_suffix(".png")),
        "figure_svg": str(figure_stem.with_suffix(".svg")),
    }
    summary_path = input_dir / f"{comparison_stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
