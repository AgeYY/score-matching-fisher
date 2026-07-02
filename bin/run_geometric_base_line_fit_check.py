#!/usr/bin/env python3
"""Quick visual check for affine geometric-base flow on noisy-line data."""

from __future__ import annotations

import argparse
import json
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
    LineSegmentBase,
    estimate_smoothed_curve_symmetric_kl,
    finetune_geometric_base_nll,
    push_base_curve,
    train_geometric_base_affine_flow,
)
from fisher.noisy_line_dataset import NoisyLineDataset
from fisher.flow_matching_skl import CenteredConditionAffineFlowSKLModel
from fisher.shared_fisher_est import require_device


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--theta-values", type=str, default="0.7853981633974483,2.356194490192345")
    p.add_argument("--ell", type=float, default=1.5)
    p.add_argument("--target-sigma", type=float, default=0.12)
    p.add_argument("--shift-x", type=float, default=0.0)
    p.add_argument("--shift-y", type=float, default=0.0)
    p.add_argument("--n-per-theta", type=int, default=3000)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--max-test-plot-per-theta", type=int, default=500)

    p.add_argument("--path-schedule", choices=("linear", "straight", "cosine"), default="cosine")
    p.add_argument("--smooth-sigma", type=float, default=0.12)
    p.add_argument("--mc-skl-samples", type=int, default=1024)
    p.add_argument("--density-mc-samples", type=int, default=512)
    p.add_argument("--ode-steps", type=int, default=64)
    p.add_argument("--ode-method", type=str, default="midpoint")
    p.add_argument("--curve-points", type=int, default=300)

    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--t-eps", type=float, default=0.0005)
    p.add_argument("--early-patience", type=int, default=1000)
    p.add_argument("--early-min-delta", type=float, default=1e-4)
    p.add_argument("--early-ema-alpha", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=100)

    p.add_argument("--nll-finetune", action="store_true")
    p.add_argument("--nll-epochs", type=int, default=2000)
    p.add_argument("--nll-batch-size", type=int, default=0, help="0 reuses --batch-size.")
    p.add_argument("--nll-lr", type=float, default=1e-4)
    p.add_argument("--nll-weight-decay", type=float, default=0.0)
    p.add_argument("--nll-particles", type=int, default=128)
    p.add_argument("--nll-sigma-min", type=float, default=1e-4)
    p.add_argument("--nll-sigma-init", type=float, default=0.1, help="0 reuses --target-sigma.")
    p.add_argument("--nll-endpoint-solver", choices=("particle-ode", "affine-map"), default="particle-ode")
    p.add_argument("--nll-checkpoint-selection", choices=("last", "best"), default="last")
    return p


def resolve_output_paths(output_dir: Path | None) -> dict[str, Path]:
    out_dir = Path(DATA_DIR) / "geometric_base_line_fit_check" if output_dir is None else Path(output_dir)
    out_dir = out_dir.expanduser().resolve()
    return {
        "output_dir": out_dir,
        "png": out_dir / "geometric_base_line_fit_check.png",
        "svg": out_dir / "geometric_base_line_fit_check.svg",
        "summary": out_dir / "geometric_base_line_fit_check_summary.json",
    }


def _parse_theta_values(text: str) -> np.ndarray:
    vals = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if len(vals) != 2:
        raise ValueError("--theta-values must contain exactly two comma-separated values for this diagnostic.")
    if not np.all(np.isfinite(vals)):
        raise ValueError("--theta-values must be finite.")
    return np.asarray(vals, dtype=np.float64).reshape(-1, 1)


def _condition_one_hot(n_conditions: int) -> np.ndarray:
    count = int(n_conditions)
    if count < 2:
        raise ValueError("At least two conditions are required.")
    return np.eye(count, dtype=np.float64)


def _split_indices(
    n_total: int,
    *,
    train_frac: float,
    val_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(n_total)
    if count < 3:
        raise ValueError("--n-per-theta must be >= 3 so train/validation/test splits are non-empty.")
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


def _make_noisy_line_data(args: argparse.Namespace, theta_eval: np.ndarray) -> dict[str, Any]:
    n_total = int(args.n_per_theta)
    max_plot = int(args.max_test_plot_per_theta)
    if max_plot < 1:
        raise ValueError("--max-test-plot-per-theta must be >= 1.")
    theta_train_parts: list[np.ndarray] = []
    x_train_parts: list[np.ndarray] = []
    theta_val_parts: list[np.ndarray] = []
    x_val_parts: list[np.ndarray] = []
    theta_test_parts: list[np.ndarray] = []
    x_test_parts: list[np.ndarray] = []
    theta_test_plot_parts: list[np.ndarray] = []
    x_test_plot_parts: list[np.ndarray] = []
    theta_test_plot_scalar_parts: list[np.ndarray] = []
    split_counts: dict[str, dict[str, int]] = {}
    datasets: list[NoisyLineDataset] = []
    condition_eval = _condition_one_hot(int(theta_eval.shape[0]))
    split_rng = np.random.default_rng(int(args.seed) + 101)
    plot_rng = np.random.default_rng(int(args.seed) + 202)
    for idx, theta_value in enumerate(theta_eval[:, 0]):
        ds = NoisyLineDataset(
            theta=float(theta_value),
            ell=float(args.ell),
            sigma=float(args.target_sigma),
            shift=(float(args.shift_x), float(args.shift_y)),
            seed=int(args.seed) + int(idx),
        )
        batch = ds.sample(n_total)
        train_idx, val_idx, test_idx = _split_indices(
            n_total,
            train_frac=float(args.train_frac),
            val_frac=float(args.val_frac),
            rng=split_rng,
        )
        theta_col = np.repeat(condition_eval[idx : idx + 1], n_total, axis=0)
        theta_scalar_col = np.full((n_total, 1), float(theta_value), dtype=np.float64)
        x_all = batch.x1.astype(np.float64, copy=False)
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
        split_counts[f"theta_{idx}"] = {
            "train": int(train_idx.size),
            "validation": int(val_idx.size),
            "test": int(test_idx.size),
            "test_plotted": int(plot_idx.size),
        }
        datasets.append(ds)
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
        "datasets": datasets,
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


def _plot_overlay(
    *,
    png_path: Path,
    svg_path: Path,
    theta_eval: np.ndarray,
    x_plot: np.ndarray,
    theta_plot_scalar: np.ndarray,
    base_curve: np.ndarray,
    fitted_curves: list[np.ndarray],
    skl_value: float,
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    val_monitor_losses: np.ndarray,
    nll_metadata: dict[str, Any] | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#4c78a8", "#f58518"]
    fig, (ax, ax_loss) = plt.subplots(
        1,
        2,
        figsize=(11.6, 5.4),
        gridspec_kw={"width_ratios": [1.12, 1.0]},
    )
    for idx, theta_value in enumerate(theta_eval[:, 0]):
        mask = np.isclose(theta_plot_scalar[:, 0], float(theta_value))
        color = colors[idx % len(colors)]
        ax.scatter(
            x_plot[mask, 0],
            x_plot[mask, 1],
            s=16,
            alpha=0.48,
            color=color,
            linewidths=0,
            label=f"Test dataset {idx + 1}",
        )
        curve = fitted_curves[idx]
        ax.plot(
            curve[:, 0],
            curve[:, 1],
            color=color,
            linewidth=2.5,
            label=f"fitted line {idx + 1}",
        )
    ax.plot(
        base_curve[:, 0],
        base_curve[:, 1],
        color="#2f2f2f",
        linewidth=2.0,
        linestyle="--",
        label="base line",
    )
    ax.text(
        0.02,
        0.98,
        f"SKL = {skl_value:.4g}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=13,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Geometric-base affine flow line fit")
    ax.legend(frameon=False, loc="best", fontsize=9)

    epochs = np.arange(1, int(train_losses.size) + 1, dtype=np.int64)
    ax_loss.plot(epochs, train_losses, color="#4c78a8", linewidth=1.8, label="train loss")
    ax_loss.plot(epochs, val_losses, color="#f58518", linewidth=1.8, label="validation loss")
    if int(val_monitor_losses.size) == int(train_losses.size):
        ax_loss.plot(epochs, val_monitor_losses, color="#444444", linewidth=1.4, linestyle="--", label="validation EMA")
    if nll_metadata is not None:
        nll_train = np.asarray(nll_metadata["train_nll_losses"], dtype=np.float64)
        nll_val = np.asarray(nll_metadata["val_nll_losses"], dtype=np.float64)
        nll_epochs = np.arange(1, int(nll_train.size) + 1, dtype=np.int64)
        ax_nll = ax_loss.twinx()
        ax_nll.plot(nll_epochs, nll_train, color="#54a24b", linewidth=1.2, alpha=0.75, label="NLL train")
        ax_nll.plot(nll_epochs, nll_val, color="#54a24b", linewidth=1.5, linestyle=":", label="NLL validation")
        ax_nll.set_ylabel("NLL")
        ax_nll.legend(frameon=False, loc="lower right", fontsize=9)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("FM loss")
    ax_loss.set_title("Training history")
    ax_loss.set_yscale("log")
    ax_loss.grid(alpha=0.25, linewidth=0.8)
    ax_loss.legend(frameon=False, loc="best", fontsize=9)

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
    theta_eval = _parse_theta_values(str(args.theta_values))
    data = _make_noisy_line_data(args, theta_eval)
    condition_eval = np.asarray(data["condition_eval"], dtype=np.float64)
    theta_train = np.asarray(data["theta_train"], dtype=np.float64)
    x_train = np.asarray(data["x_train"], dtype=np.float64)
    theta_val = np.asarray(data["theta_val"], dtype=np.float64)
    x_val = np.asarray(data["x_val"], dtype=np.float64)

    base = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0))
    model = CenteredConditionAffineFlowSKLModel(
        theta_dim=int(condition_eval.shape[1]),
        x_dim=2,
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
    nll_meta = None
    if bool(args.nll_finetune):
        nll_batch_size = int(args.nll_batch_size) if int(args.nll_batch_size) > 0 else int(args.batch_size)
        nll_sigma_init = float(args.nll_sigma_init) if float(args.nll_sigma_init) > 0.0 else float(args.target_sigma)
        nll_meta = finetune_geometric_base_nll(
            model=model,
            base=base,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            condition_eval=condition_eval,
            device=dev,
            epochs=int(args.nll_epochs),
            batch_size=nll_batch_size,
            lr=float(args.nll_lr),
            weight_decay=float(args.nll_weight_decay),
            sigma_min=float(args.nll_sigma_min),
            sigma_init=nll_sigma_init,
            n_particles=int(args.nll_particles),
            ode_steps=int(args.ode_steps),
            ode_method=str(args.ode_method),
            nll_endpoint_solver=str(args.nll_endpoint_solver),
            checkpoint_selection=str(args.nll_checkpoint_selection),
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
    fitted_curves: list[np.ndarray] = []
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
    skl_value = float(result.symmetric_kl_matrix[0, 1])
    _plot_overlay(
        png_path=paths["png"],
        svg_path=paths["svg"],
        theta_eval=theta_eval,
        x_plot=np.asarray(data["x_test_plot"], dtype=np.float64),
        theta_plot_scalar=np.asarray(data["theta_test_plot_scalar"], dtype=np.float64),
        base_curve=base_curve,
        fitted_curves=fitted_curves,
        skl_value=skl_value,
        train_losses=np.asarray(train_meta["train_losses"], dtype=np.float64),
        val_losses=np.asarray(train_meta["val_losses"], dtype=np.float64),
        val_monitor_losses=np.asarray(train_meta["val_monitor_losses"], dtype=np.float64),
        nll_metadata=nll_meta,
    )

    training_parameters = {
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "path_schedule": str(args.path_schedule),
        "t_eps": float(args.t_eps),
        "early_patience": int(args.early_patience),
        "early_min_delta": float(args.early_min_delta),
        "early_ema_alpha": float(args.early_ema_alpha),
        "max_grad_norm": float(args.max_grad_norm),
        "n_per_theta": int(args.n_per_theta),
        "train_frac": float(args.train_frac),
        "val_frac": float(args.val_frac),
        "max_test_plot_per_theta": int(args.max_test_plot_per_theta),
        "theta_values": theta_eval.reshape(-1),
        "theta_encoding": "one_hot",
        "condition_eval": condition_eval,
        "ell": float(args.ell),
        "target_sigma": float(args.target_sigma),
        "shift": [float(args.shift_x), float(args.shift_y)],
        "smooth_sigma": float(args.smooth_sigma),
        "mc_skl_samples": int(args.mc_skl_samples),
        "density_mc_samples": int(args.density_mc_samples),
        "ode_steps": int(args.ode_steps),
        "ode_method": str(args.ode_method),
        "nll_finetune": bool(args.nll_finetune),
        "nll_epochs": int(args.nll_epochs),
        "nll_batch_size": int(args.nll_batch_size) if int(args.nll_batch_size) > 0 else int(args.batch_size),
        "nll_lr": float(args.nll_lr),
        "nll_weight_decay": float(args.nll_weight_decay),
        "nll_particles": int(args.nll_particles),
        "nll_sigma_min": float(args.nll_sigma_min),
        "nll_sigma_init": float(args.nll_sigma_init) if float(args.nll_sigma_init) > 0.0 else float(args.target_sigma),
        "nll_endpoint_solver": str(args.nll_endpoint_solver),
        "nll_checkpoint_selection": str(args.nll_checkpoint_selection),
    }
    summary = {
        "script": "bin/run_geometric_base_line_fit_check.py",
        "device": str(dev),
        "theta_values": theta_eval.reshape(-1),
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
        "nll_finetune_metadata": nll_meta,
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
    if nll_meta is not None:
        print(
            f"nll_selected_epoch: {int(nll_meta['selected_epoch'])} "
            f"nll_selected_val_nll: {float(nll_meta['selected_val_nll']):.12g} "
            f"learned_sigmas: {np.array2string(np.asarray(nll_meta['learned_sigmas'], dtype=np.float64), precision=6, separator=',')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
