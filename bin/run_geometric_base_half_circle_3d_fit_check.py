#!/usr/bin/env python3
"""Visual check for 3D similarity geometric-base flow on noisy half-circle data."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import DATA_DIR, DEFAULT_DEVICE

from fisher.geometric_base_flow_skl import (
    HalfCircle3DBase,
    NoisyGeometricBase,
    build_geometric_base_velocity_model,
    estimate_smoothed_curve_symmetric_kl,
    finetune_geometric_base_cnf_likelihood,
    push_base_curve,
    push_initial_points,
    train_geometric_base_affine_flow,
)
from fisher.shared_fisher_est import require_device


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--condition-values", type=str, default="0.0,1.0")
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--base-radius", type=float, default=1.0)
    p.add_argument("--base-noise-sigma", type=float, default=0.1)
    p.add_argument("--target-sigma", type=float, default=0.2)
    p.add_argument("--left-center-x", type=float, default=-1.0)
    p.add_argument("--left-center-y", type=float, default=0.0)
    p.add_argument("--left-center-z", type=float, default=0.0)
    p.add_argument("--right-center-x", type=float, default=1.0)
    p.add_argument("--right-center-y", type=float, default=0.0)
    p.add_argument("--right-center-z", type=float, default=0.0)
    p.add_argument("--n-per-condition", type=int, default=3000)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--max-test-plot-per-condition", type=int, default=600)

    p.add_argument("--path-schedule", choices=("linear", "straight", "cosine"), default="cosine")
    p.add_argument("--smooth-sigma", type=float, default=0.12)
    p.add_argument("--mc-skl-samples", type=int, default=1024)
    p.add_argument("--density-mc-samples", type=int, default=512)
    p.add_argument("--ode-steps", type=int, default=64)
    p.add_argument("--ode-method", type=str, default="midpoint")
    p.add_argument("--curve-points", type=int, default=300)
    p.add_argument("--generated-samples-per-condition", type=int, default=600)

    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument(
        "--velocity-family",
        choices=("lie-similarity-3d", "centered-affine"),
        default="lie-similarity-3d",
    )
    p.add_argument("--t-eps", type=float, default=0.0005)
    p.add_argument("--early-patience", type=int, default=1000)
    p.add_argument("--early-min-delta", type=float, default=1e-4)
    p.add_argument("--early-ema-alpha", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=100)

    p.add_argument("--nf-likelihood-finetune", action="store_true")
    p.add_argument("--nf-epochs", type=int, default=500)
    p.add_argument("--nf-batch-size", type=int, default=0, help="0 reuses --batch-size.")
    p.add_argument("--nf-lr", type=float, default=1e-4)
    p.add_argument("--nf-weight-decay", type=float, default=0.0)
    p.add_argument("--nf-density-points", type=int, default=512)
    p.add_argument("--nf-checkpoint-selection", choices=("last", "best"), default="last")
    return p


def resolve_output_paths(output_dir: Path | None) -> dict[str, Path]:
    out_dir = Path(DATA_DIR) / "geometric_base_half_circle_3d_fit_check" if output_dir is None else Path(output_dir)
    out_dir = out_dir.expanduser().resolve()
    return {
        "output_dir": out_dir,
        "png": out_dir / "geometric_base_half_circle_3d_fit_check.png",
        "svg": out_dir / "geometric_base_half_circle_3d_fit_check.svg",
        "summary": out_dir / "geometric_base_half_circle_3d_fit_check_summary.json",
    }


def _parse_condition_values(text: str) -> np.ndarray:
    vals = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if len(vals) != 2:
        raise ValueError("--condition-values must contain exactly two comma-separated values.")
    if not np.all(np.isfinite(vals)):
        raise ValueError("--condition-values must be finite.")
    return np.asarray(vals, dtype=np.float64).reshape(-1, 1)


def _condition_one_hot(n_conditions: int) -> np.ndarray:
    count = int(n_conditions)
    if count != 2:
        raise ValueError("Exactly two half-circle conditions are required.")
    return np.eye(count, dtype=np.float64)


def _half_circle_3d_from_u(
    u: np.ndarray,
    *,
    radius: float,
    center: tuple[float, float, float],
    arc: str,
) -> np.ndarray:
    u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
    r = float(radius)
    if not math.isfinite(r) or r <= 0.0:
        raise ValueError("radius must be finite and positive.")
    arc_norm = str(arc).strip().lower()
    if arc_norm not in ("upper", "lower"):
        raise ValueError("arc must be upper or lower.")
    theta = math.pi * np.clip(u_arr, 0.0, 1.0)
    y_sign = 1.0 if arc_norm == "upper" else -1.0
    base = np.column_stack((r * np.cos(theta), y_sign * r * np.sin(theta), np.zeros_like(theta)))
    return base + np.asarray(center, dtype=np.float64).reshape(1, 3)


def _split_indices(
    n_total: int,
    *,
    train_frac: float,
    val_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(n_total)
    if count < 3:
        raise ValueError("n_per_condition must be >= 3.")
    trf = float(train_frac)
    vf = float(val_frac)
    if not (0.0 < trf < 1.0):
        raise ValueError("--train-frac must be in (0, 1).")
    if not (0.0 < vf < 1.0):
        raise ValueError("--val-frac must be in (0, 1).")
    if trf + vf >= 1.0:
        raise ValueError("--train-frac + --val-frac must be < 1.")

    idx = rng.permutation(count)
    n_train = int(round(trf * count))
    n_val = int(round(vf * count))
    n_train = min(max(n_train, 1), count - 2)
    n_val = min(max(n_val, 1), count - n_train - 1)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    if train_idx.size < 1 or val_idx.size < 1 or test_idx.size < 1:
        raise ValueError("Split configuration produced an empty train, validation, or test split.")
    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


def _make_noisy_half_circle_3d_data(args: argparse.Namespace, condition_values: np.ndarray) -> dict[str, Any]:
    n_total = int(args.n_per_condition)
    max_plot = int(args.max_test_plot_per_condition)
    if max_plot < 1:
        raise ValueError("--max-test-plot-per-condition must be >= 1.")
    condition_eval = _condition_one_hot(int(condition_values.shape[0]))
    split_rng = np.random.default_rng(int(args.seed) + 101)
    plot_rng = np.random.default_rng(int(args.seed) + 202)
    centers = [
        (float(args.left_center_x), float(args.left_center_y), float(args.left_center_z)),
        (float(args.right_center_x), float(args.right_center_y), float(args.right_center_z)),
    ]
    if not np.all(np.isfinite(np.asarray(centers, dtype=np.float64))):
        raise ValueError("centers must be finite.")
    sigma = float(args.target_sigma)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("--target-sigma must be finite and nonnegative.")
    arcs = ["upper", "lower"]
    theta_train_parts: list[np.ndarray] = []
    x_train_parts: list[np.ndarray] = []
    theta_val_parts: list[np.ndarray] = []
    x_val_parts: list[np.ndarray] = []
    theta_test_parts: list[np.ndarray] = []
    x_test_parts: list[np.ndarray] = []
    theta_test_plot_parts: list[np.ndarray] = []
    theta_test_plot_scalar_parts: list[np.ndarray] = []
    x_test_plot_parts: list[np.ndarray] = []
    target_curves: list[np.ndarray] = []
    split_counts: dict[str, dict[str, int]] = {}

    for idx, value in enumerate(condition_values[:, 0]):
        rng = np.random.default_rng(int(args.seed) + int(idx))
        u = rng.uniform(0.0, 1.0, size=(n_total, 1)).astype(np.float64, copy=False)
        clean = _half_circle_3d_from_u(u, radius=float(args.radius), center=centers[idx], arc=arcs[idx])
        x_all = clean + sigma * rng.standard_normal(size=(n_total, 3)).astype(np.float64, copy=False)
        train_idx, val_idx, test_idx = _split_indices(
            n_total,
            train_frac=float(args.train_frac),
            val_frac=float(args.val_frac),
            rng=split_rng,
        )
        theta_col = np.repeat(condition_eval[idx : idx + 1], n_total, axis=0)
        theta_scalar_col = np.full((n_total, 1), float(value), dtype=np.float64)
        theta_train_parts.append(theta_col[train_idx])
        x_train_parts.append(x_all[train_idx])
        theta_val_parts.append(theta_col[val_idx])
        x_val_parts.append(x_all[val_idx])
        theta_test_parts.append(theta_col[test_idx])
        x_test_parts.append(x_all[test_idx])

        plot_idx = test_idx
        if int(plot_idx.size) > max_plot:
            plot_idx = np.sort(plot_rng.choice(plot_idx, size=max_plot, replace=False))
        theta_test_plot_parts.append(theta_col[plot_idx])
        theta_test_plot_scalar_parts.append(theta_scalar_col[plot_idx])
        x_test_plot_parts.append(x_all[plot_idx])
        target_u = np.linspace(0.0, 1.0, int(args.curve_points), dtype=np.float64)
        target_curves.append(_half_circle_3d_from_u(target_u, radius=float(args.radius), center=centers[idx], arc=arcs[idx]))
        split_counts[f"condition_{idx}"] = {
            "train": int(train_idx.size),
            "validation": int(val_idx.size),
            "test": int(test_idx.size),
            "test_plotted": int(plot_idx.size),
        }

    theta_train = np.concatenate(theta_train_parts, axis=0)
    x_train = np.concatenate(x_train_parts, axis=0)
    theta_val = np.concatenate(theta_val_parts, axis=0)
    x_val = np.concatenate(x_val_parts, axis=0)
    theta_test = np.concatenate(theta_test_parts, axis=0)
    x_test = np.concatenate(x_test_parts, axis=0)
    theta_test_plot = np.concatenate(theta_test_plot_parts, axis=0)
    theta_test_plot_scalar = np.concatenate(theta_test_plot_scalar_parts, axis=0)
    x_test_plot = np.concatenate(x_test_plot_parts, axis=0)
    shuffle_rng = np.random.default_rng(int(args.seed) + 303)
    perm = shuffle_rng.permutation(int(theta_train.shape[0]))
    theta_train = theta_train[perm]
    x_train = x_train[perm]
    val_perm = shuffle_rng.permutation(int(theta_val.shape[0]))
    theta_val = theta_val[val_perm]
    x_val = x_val[val_perm]
    return {
        "theta_train": theta_train,
        "x_train": x_train,
        "theta_val": theta_val,
        "x_val": x_val,
        "theta_test": theta_test,
        "x_test": x_test,
        "theta_test_plot": theta_test_plot,
        "theta_test_plot_scalar": theta_test_plot_scalar,
        "x_test_plot": x_test_plot,
        "condition_eval": condition_eval,
        "theta_encoding": "one_hot",
        "split_counts": split_counts,
        "target_curves": target_curves,
        "centers": centers,
        "arcs": arcs,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _set_3d_equal_axes(ax: Any, arrays: list[np.ndarray]) -> None:
    xyz = np.concatenate([np.asarray(arr, dtype=np.float64).reshape(-1, 3) for arr in arrays if np.asarray(arr).size > 0], axis=0)
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _plot_overlay(
    *,
    png_path: Path,
    svg_path: Path,
    condition_values: np.ndarray,
    x_plot: np.ndarray,
    theta_plot_scalar: np.ndarray,
    base_curve: np.ndarray,
    base_samples: np.ndarray,
    target_curves: list[np.ndarray],
    fitted_curves: list[np.ndarray],
    generated_samples: list[np.ndarray],
    skl_value: float,
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    val_monitor_losses: np.ndarray,
    nf_likelihood_metadata: dict[str, Any] | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#4c78a8", "#f58518"]
    fig = plt.figure(figsize=(12.4, 5.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax_loss = fig.add_subplot(1, 2, 2)

    ax.scatter(base_samples[:, 0], base_samples[:, 1], base_samples[:, 2], s=10, alpha=0.22, color="#2f2f2f", label="base samples")
    all_arrays = [base_samples, base_curve, x_plot]
    for idx, condition_value in enumerate(condition_values[:, 0]):
        mask = np.isclose(theta_plot_scalar[:, 0], float(condition_value))
        color = colors[idx % len(colors)]
        ax.scatter(
            x_plot[mask, 0],
            x_plot[mask, 1],
            x_plot[mask, 2],
            s=14,
            alpha=0.48,
            color=color,
            linewidths=0,
            label=f"Test dataset {idx + 1}",
        )
        target = np.asarray(target_curves[idx], dtype=np.float64)
        curve = np.asarray(fitted_curves[idx], dtype=np.float64)
        gen = np.asarray(generated_samples[idx], dtype=np.float64)
        ax.plot(target[:, 0], target[:, 1], target[:, 2], color=color, linewidth=1.2, linestyle=":", label=f"target half-circle {idx + 1}")
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=color, linewidth=2.4, label=f"fitted half-circle {idx + 1}")
        ax.scatter(gen[:, 0], gen[:, 1], gen[:, 2], s=12, alpha=0.35, marker="x", color=color, linewidths=0.8, label=f"generated samples {idx + 1}")
        all_arrays.extend([target, curve, gen])
    ax.plot(base_curve[:, 0], base_curve[:, 1], base_curve[:, 2], color="#2f2f2f", linewidth=2.0, linestyle="--", label="base half-circle")
    ax.text2D(0.02, 0.98, f"curve SKL = {skl_value:.4g}", transform=ax.transAxes, va="top", ha="left", fontsize=13)
    _set_3d_equal_axes(ax, all_arrays)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.view_init(elev=22.0, azim=-58.0)
    ax.set_title("3D similarity flow half-circle fit")
    ax.legend(frameon=False, loc="best", fontsize=7)

    epochs = np.arange(1, int(train_losses.size) + 1, dtype=np.int64)
    ax_loss.plot(epochs, train_losses, color="#4c78a8", linewidth=1.8, label="train loss")
    ax_loss.plot(epochs, val_losses, color="#f58518", linewidth=1.8, label="validation loss")
    if int(val_monitor_losses.size) == int(train_losses.size):
        ax_loss.plot(epochs, val_monitor_losses, color="#444444", linewidth=1.4, linestyle="--", label="validation EMA")
    if nf_likelihood_metadata is not None:
        nf_train = np.asarray(nf_likelihood_metadata["train_nll_losses"], dtype=np.float64)
        nf_val = np.asarray(nf_likelihood_metadata["val_nll_losses"], dtype=np.float64)
        nf_epochs = np.arange(1, int(nf_train.size) + 1, dtype=np.int64)
        ax_nf = ax_loss.twinx()
        ax_nf.plot(nf_epochs, nf_train, color="#b279a2", linewidth=1.2, alpha=0.75, label="NF train NLL")
        ax_nf.plot(nf_epochs, nf_val, color="#b279a2", linewidth=1.5, linestyle=":", label="NF validation NLL")
        ax_nf.set_ylabel("NF NLL")
        ax_nf.legend(frameon=False, loc="lower right", fontsize=8)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("FM loss")
    ax_loss.set_title("Training history")
    ax_loss.set_yscale("log")
    ax_loss.grid(alpha=0.25, linewidth=0.8)
    ax_loss.legend(frameon=False, loc="best", fontsize=8)

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dev = require_device(str(args.device))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    paths = resolve_output_paths(args.output_dir)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    condition_values = _parse_condition_values(str(args.condition_values))
    data = _make_noisy_half_circle_3d_data(args, condition_values)
    condition_eval = np.asarray(data["condition_eval"], dtype=np.float64)
    theta_train = np.asarray(data["theta_train"], dtype=np.float64)
    x_train = np.asarray(data["x_train"], dtype=np.float64)
    theta_val = np.asarray(data["theta_val"], dtype=np.float64)
    x_val = np.asarray(data["x_val"], dtype=np.float64)

    base_noise_sigma = float(args.base_noise_sigma)
    if not np.isfinite(base_noise_sigma) or base_noise_sigma <= 0.0:
        raise ValueError("--base-noise-sigma must be finite and > 0 for NF likelihood.")
    generated_sample_count = int(args.generated_samples_per_condition)
    if generated_sample_count < 1:
        raise ValueError("--generated-samples-per-condition must be >= 1.")

    clean_base = HalfCircle3DBase(center=(0.0, 0.0, 0.0), radius=float(args.base_radius))
    base = NoisyGeometricBase(clean_base, sigma=base_noise_sigma)
    model = build_geometric_base_velocity_model(
        velocity_family=str(args.velocity_family),
        theta_dim=int(condition_eval.shape[1]),
        x_dim=3,
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        path_schedule=str(args.path_schedule),
    ).to(dev)
    train_meta = train_geometric_base_affine_flow(
        model=model,
        base=base,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        device=dev,
        path_schedule=str(args.path_schedule),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        t_eps=float(args.t_eps),
        patience=int(args.early_patience),
        min_delta=float(args.early_min_delta),
        ema_alpha=float(args.early_ema_alpha),
        max_grad_norm=float(args.max_grad_norm),
        log_every=max(1, int(args.log_every)),
    )
    nf_likelihood_meta = None
    if bool(args.nf_likelihood_finetune):
        nf_batch_size = int(args.nf_batch_size) if int(args.nf_batch_size) > 0 else int(args.batch_size)
        nf_likelihood_meta = finetune_geometric_base_cnf_likelihood(
            model=model,
            base=base,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            condition_eval=condition_eval,
            device=dev,
            epochs=int(args.nf_epochs),
            batch_size=nf_batch_size,
            lr=float(args.nf_lr),
            weight_decay=float(args.nf_weight_decay),
            density_points=int(args.nf_density_points),
            ode_steps=int(args.ode_steps),
            ode_method=str(args.ode_method),
            checkpoint_selection=str(args.nf_checkpoint_selection),
            log_every=max(1, int(args.log_every)),
        )

    result = estimate_smoothed_curve_symmetric_kl(
        model=model,
        base=base,
        theta_all=condition_eval,
        device=dev,
        smooth_sigma=float(args.smooth_sigma),
        mc_skl_samples=int(args.mc_skl_samples),
        density_mc_samples=int(args.density_mc_samples),
        ode_steps=int(args.ode_steps),
        ode_method=str(args.ode_method),
        batch_size=int(args.batch_size),
        train_metadata=train_meta,
    )

    curve_u = torch.linspace(base.u_low, base.u_high, int(args.curve_points), dtype=torch.float32).reshape(-1, 1)
    base_curve = base.points_from_u(curve_u).detach().cpu().numpy().astype(np.float64)
    base_samples_t = base.sample(generated_sample_count, device=dev, dtype=torch.float32)
    base_samples = base_samples_t.detach().cpu().numpy().astype(np.float64)
    fitted_curves: list[np.ndarray] = []
    generated_samples: list[np.ndarray] = []
    for theta_row in condition_eval:
        curve, _ = push_base_curve(
            model=model,
            base=base,
            theta=theta_row.reshape(1, -1),
            device=dev,
            u=curve_u,
            ode_steps=int(args.ode_steps),
            ode_method=str(args.ode_method),
        )
        fitted_curves.append(curve.detach().cpu().numpy().astype(np.float64))
        pushed = push_initial_points(
            model=model,
            x0=base_samples_t,
            theta=theta_row.reshape(1, -1),
            device=dev,
            ode_steps=int(args.ode_steps),
            ode_method=str(args.ode_method),
        )
        generated_samples.append(pushed.detach().cpu().numpy().astype(np.float64))
    skl_value = float(result.symmetric_kl_matrix[0, 1])
    _plot_overlay(
        png_path=paths["png"],
        svg_path=paths["svg"],
        condition_values=condition_values,
        x_plot=np.asarray(data["x_test_plot"], dtype=np.float64),
        theta_plot_scalar=np.asarray(data["theta_test_plot_scalar"], dtype=np.float64),
        base_curve=base_curve,
        base_samples=base_samples,
        target_curves=[np.asarray(arr, dtype=np.float64) for arr in data["target_curves"]],
        fitted_curves=fitted_curves,
        generated_samples=generated_samples,
        skl_value=skl_value,
        train_losses=np.asarray(train_meta["train_losses"], dtype=np.float64),
        val_losses=np.asarray(train_meta["val_losses"], dtype=np.float64),
        val_monitor_losses=np.asarray(train_meta["val_monitor_losses"], dtype=np.float64),
        nf_likelihood_metadata=nf_likelihood_meta,
    )

    training_parameters = {
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "velocity_family": str(args.velocity_family),
        "path_schedule": str(args.path_schedule),
        "t_eps": float(args.t_eps),
        "early_patience": int(args.early_patience),
        "early_min_delta": float(args.early_min_delta),
        "early_ema_alpha": float(args.early_ema_alpha),
        "max_grad_norm": float(args.max_grad_norm),
        "n_per_condition": int(args.n_per_condition),
        "train_frac": float(args.train_frac),
        "val_frac": float(args.val_frac),
        "condition_values": condition_values.reshape(-1),
        "theta_encoding": "one_hot",
        "condition_eval": condition_eval,
        "radius": float(args.radius),
        "base_radius": float(args.base_radius),
        "base_distribution": "noisy_half_circle_3d",
        "base_noise_sigma": base_noise_sigma,
        "target_sigma": float(args.target_sigma),
        "centers": data["centers"],
        "arcs": data["arcs"],
        "smooth_sigma": float(args.smooth_sigma),
        "mc_skl_samples": int(args.mc_skl_samples),
        "density_mc_samples": int(args.density_mc_samples),
        "ode_steps": int(args.ode_steps),
        "ode_method": str(args.ode_method),
        "curve_points": int(args.curve_points),
        "generated_samples_per_condition": generated_sample_count,
        "nf_likelihood_finetune": bool(args.nf_likelihood_finetune),
        "nf_epochs": int(args.nf_epochs),
        "nf_batch_size": int(args.nf_batch_size) if int(args.nf_batch_size) > 0 else int(args.batch_size),
        "nf_lr": float(args.nf_lr),
        "nf_weight_decay": float(args.nf_weight_decay),
        "nf_density_points": int(args.nf_density_points),
        "nf_checkpoint_selection": str(args.nf_checkpoint_selection),
    }
    summary = {
        "script": "bin/run_geometric_base_half_circle_3d_fit_check.py",
        "device": str(dev),
        "condition_values": condition_values.reshape(-1),
        "theta_encoding": "one_hot",
        "condition_eval": condition_eval,
        "training_parameters": training_parameters,
        "split_counts": data["split_counts"],
        "train_shape": list(theta_train.shape),
        "validation_shape": list(theta_val.shape),
        "test_shape": list(np.asarray(data["theta_test"], dtype=np.float64).shape),
        "smooth_sigma": float(args.smooth_sigma),
        "target_sigma": float(args.target_sigma),
        "symmetric_kl_matrix": result.symmetric_kl_matrix,
        "skl_value": skl_value,
        "best_epoch": int(train_meta["best_epoch"]),
        "best_val_loss": float(train_meta["best_val_loss"]),
        "stopped_epoch": int(train_meta["stopped_epoch"]),
        "stopped_early": bool(train_meta["stopped_early"]),
        "nf_likelihood_finetune_metadata": nf_likelihood_meta,
        "base_samples_shape": list(base_samples.shape),
        "generated_sample_shapes": [list(arr.shape) for arr in generated_samples],
        "png": paths["png"],
        "svg": paths["svg"],
        "summary": paths["summary"],
    }
    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"png: {paths['png']}", flush=True)
    print(f"svg: {paths['svg']}", flush=True)
    print(f"summary_json: {paths['summary']}", flush=True)
    print(f"skl: {skl_value:.12g}", flush=True)
    print(f"best_epoch: {int(train_meta['best_epoch'])}", flush=True)
    print(f"best_val_loss: {float(train_meta['best_val_loss']):.12g}", flush=True)
    if nf_likelihood_meta is not None:
        print(
            f"nf_selected_epoch: {int(nf_likelihood_meta['selected_epoch'])} "
            f"nf_selected_val_nll: {float(nf_likelihood_meta['selected_val_nll']):.12g}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
