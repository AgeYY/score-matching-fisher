#!/usr/bin/env python3
"""Compare RBF-conditioned affine flow and GKR linear Fisher estimates."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.continuous_fisher_comparison import (
    make_native_dataset_npz,
    native_linear_fisher_curve,
    theta_grid_from_meta,
    theta_midpoints,
)
from fisher.flow_matching_skl import (
    build_flow_skl_model,
    estimate_affine_mixed_symmetric_kl_fisher,
    train_flow_skl_model,
)
from fisher.gkr import GKRConfig, TorchGKR, estimate_gkr_linear_fisher
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_meta, require_device
from global_setting import DEFAULT_EARLY_STOPPING_PATIENCE, DEFAULT_TRAINING_MAX_EPOCHS


def _n_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not values or any(value < 2 for value in values):
        raise argparse.ArgumentTypeError("--n-list requires comma-separated integers >= 2.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--native-x-dim", type=int, default=5)
    parser.add_argument("--n-list", type=_n_list, default=[100, 500, 1000])
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--theta-grid-size", type=int, default=61)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_MAX_EPOCHS)
    parser.add_argument("--early-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--theta-rbf-num-centers", type=int, default=8)
    parser.add_argument("--theta-rbf-bandwidth", type=float, default=None)
    parser.add_argument("--ode-steps", type=int, default=64)
    return parser.parse_args()


def _mae(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(estimate) - np.asarray(truth))))


def _plot(
    *,
    cases: list[dict[str, object]],
    output_dir: Path,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(1, len(cases) + 1, figsize=(14.5, 3.4), constrained_layout=True)
    curve_axes = axes[:-1]
    y_values: list[np.ndarray] = []
    for axis, case in zip(curve_axes, cases, strict=True):
        theta = np.asarray(case["theta_midpoints"], dtype=np.float64)
        truth = np.asarray(case["ground_truth"], dtype=np.float64)
        flow = np.asarray(case["flow"], dtype=np.float64)
        gkr = np.asarray(case["gkr"], dtype=np.float64)
        y_values.extend((truth, flow, gkr))
        axis.plot(theta, truth, color="black", linestyle="--", linewidth=2.1, label="Ground truth")
        axis.plot(theta, flow, color="C0", linewidth=2.0, label="RBF flow")
        axis.plot(theta, gkr, color="C2", linewidth=2.0, label="GKR")
        axis.set_title(rf"$N={int(case['n_total'])}$")
        axis.set_xlabel(r"$\theta$")
        axis.grid(axis="y", color="0.9", linewidth=0.8)
    curve_axes[0].set_ylabel("Linear Fisher information")
    curve_axes[0].legend(frameon=False)
    low = min(float(np.min(values)) for values in y_values)
    high = max(float(np.max(values)) for values in y_values)
    margin = 0.05 * max(high - low, 1e-8)
    for axis in curve_axes:
        axis.set_ylim(low - margin, high + margin)

    error_axis = axes[-1]
    n_values = np.asarray([int(case["n_total"]) for case in cases], dtype=np.int64)
    flow_mae = np.asarray([float(case["flow_mae"]) for case in cases])
    gkr_mae = np.asarray([float(case["gkr_mae"]) for case in cases])
    error_axis.plot(n_values, flow_mae, color="C0", marker="o", linewidth=2.0, label="RBF flow")
    error_axis.plot(n_values, gkr_mae, color="C2", marker="^", linewidth=2.0, label="GKR")
    error_axis.set_xlabel("Total observations")
    error_axis.set_ylabel("Mean absolute error")
    error_axis.set_xticks(n_values)
    error_axis.grid(axis="y", color="0.9", linewidth=0.8)
    error_axis.legend(frameon=False)

    png = output_dir / "rbf_flow_vs_gkr_linear_fisher.png"
    svg = output_dir / "rbf_flow_vs_gkr_linear_fisher.svg"
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def main() -> None:
    args = parse_args()
    if int(args.native_x_dim) < 1:
        raise ValueError("--native-x-dim must be >= 1.")
    if not 0.0 < float(args.train_frac) < 1.0:
        raise ValueError("--train-frac must be in (0, 1).")
    if int(args.theta_grid_size) < 2:
        raise ValueError("--theta-grid-size must be >= 2.")
    if int(args.theta_rbf_num_centers) < 2:
        raise ValueError("--theta-rbf-num-centers must be >= 2.")
    if args.theta_rbf_bandwidth is not None and float(args.theta_rbf_bandwidth) <= 0.0:
        raise ValueError("--theta-rbf-bandwidth must be positive.")

    device = require_device(str(args.device))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gkr_config = GKRConfig()
    cases: list[dict[str, object]] = []

    for n_total in args.n_list:
        case_dir = output_dir / f"n{int(n_total)}"
        case_dir.mkdir(parents=True, exist_ok=True)
        dataset_npz = case_dir / f"randamp_gaussian_sqrtd_xdim{int(args.native_x_dim)}_native.npz"
        make_native_dataset_npz(
            output_npz=dataset_npz,
            dataset_family="randamp_gaussian_sqrtd",
            x_dim=int(args.native_x_dim),
            n_total=int(n_total),
            train_frac=float(args.train_frac),
            seed=int(args.seed),
            force=bool(args.force_dataset),
        )
        bundle = load_shared_dataset_npz(dataset_npz)
        theta_grid = theta_grid_from_meta(bundle.meta, theta_grid_size=int(args.theta_grid_size))
        theta_query = theta_midpoints(theta_grid)
        population = build_dataset_from_meta(dict(bundle.meta))
        truth = native_linear_fisher_curve(theta_query, population)

        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(args.seed))
        model = build_flow_skl_model(
            velocity_family="condition_affine",
            theta_dim=1,
            x_dim=int(args.native_x_dim),
            hidden_dim=int(args.hidden_dim),
            depth=int(args.depth),
            quadrature_steps=64,
            path_schedule="cosine",
            divergence_estimator="exact",
            theta_embedding="gaussian_rbf",
            theta_rbf_num_centers=int(args.theta_rbf_num_centers),
            theta_rbf_lower=float(bundle.meta["theta_low"]),
            theta_rbf_upper=float(bundle.meta["theta_high"]),
            theta_rbf_bandwidth=args.theta_rbf_bandwidth,
        ).to(device)
        flow_meta = train_flow_skl_model(
            model=model,
            theta_train=np.asarray(bundle.theta_train, dtype=np.float64).reshape(-1, 1),
            x_train=np.asarray(bundle.x_train, dtype=np.float64),
            theta_val=np.asarray(bundle.theta_validation, dtype=np.float64).reshape(-1, 1),
            x_val=np.asarray(bundle.x_validation, dtype=np.float64),
            device=device,
            velocity_family="condition_affine",
            path_schedule="cosine",
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            lr_schedule="constant",
            weight_decay=0.0,
            t_eps=5e-4,
            patience=int(args.early_patience),
            min_delta=1e-4,
            ema_alpha=0.05,
            max_grad_norm=10.0,
            log_every=50,
            checkpoint_selection="best",
            best_checkpoint_metric="flow_matching",
            fixed_validation=True,
            fixed_validation_paths=10,
            validation_seed=int(args.seed) + 10_000,
        )
        flow_result = estimate_affine_mixed_symmetric_kl_fisher(
            model=model,
            theta_all=theta_grid,
            device=device,
            ridge=1e-6,
            ode_steps=int(args.ode_steps),
        )
        flow = np.asarray(flow_result["fisher"], dtype=np.float64)
        torch.save(
            {key: value.detach().cpu() for key, value in model.state_dict().items()},
            case_dir / "rbf_flow_selected_model.pt",
        )

        gkr_model = TorchGKR(
            n_input=1,
            n_output=int(args.native_x_dim),
            config=gkr_config,
            dtype=torch.float64,
            device=device,
            seed=int(args.seed),
        )
        gkr_model.fit(bundle.x_train, bundle.theta_train)
        gkr_result = estimate_gkr_linear_fisher(
            gkr_model,
            theta_query,
            finite_difference_step=np.diff(theta_grid, axis=0),
            solve_jitter=1e-6,
        )
        gkr = np.asarray(gkr_result.linear_fisher, dtype=np.float64)
        case = {
            "n_total": int(n_total),
            "n_train": int(bundle.x_train.shape[0]),
            "n_validation": int(bundle.x_validation.shape[0]),
            "theta_midpoints": theta_query[:, 0],
            "ground_truth": truth,
            "flow": flow,
            "gkr": gkr,
            "flow_mae": _mae(flow, truth),
            "gkr_mae": _mae(gkr, truth),
            "flow_selected_epoch": int(flow_meta["selected_epoch"]),
            "flow_stopped_epoch": int(flow_meta["stopped_epoch"]),
        }
        cases.append(case)
        np.savez_compressed(
            case_dir / "rbf_flow_vs_gkr_linear_fisher_results.npz",
            theta_grid=theta_grid,
            theta_midpoints=theta_query,
            ground_truth_linear_fisher=truth,
            rbf_flow_linear_fisher=flow,
            gkr_linear_fisher=gkr,
            flow_train_losses=np.asarray(flow_meta["train_losses"], dtype=np.float64),
            flow_validation_losses=np.asarray(flow_meta["val_losses"], dtype=np.float64),
            gkr_mean_loss=np.asarray(gkr_result.mean_loss, dtype=np.float64),
            gkr_covariance_loss=np.asarray(gkr_result.covariance_loss, dtype=np.float64),
        )
        print(
            f"[comparison] N={n_total} flow_mae={case['flow_mae']:.6f} "
            f"gkr_mae={case['gkr_mae']:.6f}",
            flush=True,
        )

    figure_png, figure_svg = _plot(cases=cases, output_dir=output_dir)
    spacing = 12.0 / float(int(args.theta_rbf_num_centers) - 1)
    summary = {
        "native_x_dim": int(args.native_x_dim),
        "n_list": [int(value) for value in args.n_list],
        "seed": int(args.seed),
        "device": str(device),
        "theta_grid_size": int(args.theta_grid_size),
        "theta_grid_spacing": 12.0 / float(int(args.theta_grid_size) - 1),
        "theta_embedding": "gaussian_rbf",
        "theta_rbf_num_centers": int(args.theta_rbf_num_centers),
        "theta_rbf_bandwidth": (
            float(args.theta_rbf_bandwidth) if args.theta_rbf_bandwidth is not None else spacing
        ),
        "flow_config": {
            "epochs": int(args.epochs),
            "early_stopping_patience": int(args.early_patience),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "learning_rate_schedule": "constant",
            "hidden_dim": int(args.hidden_dim),
            "depth": int(args.depth),
            "path_schedule": "cosine",
            "fixed_validation_paths": 10,
            "ode_steps": int(args.ode_steps),
        },
        "gkr_config": asdict(gkr_config),
        "cases": [
            {
                key: value
                for key, value in case.items()
                if key not in {"theta_midpoints", "ground_truth", "flow", "gkr"}
            }
            for case in cases
        ],
        "figure_png": str(figure_png),
        "figure_svg": str(figure_svg),
    }
    summary_path = output_dir / "rbf_flow_vs_gkr_linear_fisher_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["cases"], indent=2), flush=True)
    print(f"figure_svg: {figure_svg}", flush=True)
    print(f"summary_json: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
