#!/usr/bin/env python3
"""Compare regular CFM and minibatch OT-CFM on geometric-base noisy-line fitting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import DATA_DIR, DEFAULT_DEVICE

from fisher.flow_matching_skl import CenteredConditionAffineFlowSKLModel
from fisher.geometric_base_flow_skl import (
    LineSegmentBase,
    estimate_smoothed_curve_symmetric_kl,
    push_base_curve,
    train_geometric_base_affine_flow,
)
from fisher.shared_fisher_est import require_device
from run_geometric_base_line_fit_check import (  # type: ignore[import-not-found]
    _jsonable,
    _make_noisy_line_data,
    _parse_theta_values,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--run-method", choices=("regular", "ot", "both"), default="both")

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
    p.add_argument("--ot-method", choices=("exact", "sinkhorn", "unbalanced", "partial"), default="sinkhorn")
    p.add_argument("--ot-reg", type=float, default=0.05)
    p.add_argument("--ot-reg-m", type=float, default=1.0)
    p.add_argument("--ot-normalize-cost", action="store_true")
    p.add_argument("--ot-num-threads", type=str, default="1")
    p.add_argument("--smooth-sigma", type=float, default=0.12)
    p.add_argument("--mc-skl-samples", type=int, default=1024)
    p.add_argument("--density-mc-samples", type=int, default=512)
    p.add_argument("--ode-steps", type=int, default=64)
    p.add_argument("--ode-method", type=str, default="midpoint")
    p.add_argument("--curve-points", type=int, default=300)

    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--t-eps", type=float, default=0.0005)
    p.add_argument("--early-patience", type=int, default=1000)
    p.add_argument("--early-min-delta", type=float, default=1e-4)
    p.add_argument("--early-ema-alpha", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=100)
    return p


def resolve_output_paths(output_dir: Path | None) -> dict[str, Path]:
    out_dir = Path(DATA_DIR) / "geometric_base_line_fit_ot_compare" if output_dir is None else Path(output_dir)
    out_dir = out_dir.expanduser().resolve()
    return {
        "output_dir": out_dir,
        "png": out_dir / "geometric_base_line_fit_ot_compare.png",
        "svg": out_dir / "geometric_base_line_fit_ot_compare.svg",
        "summary": out_dir / "geometric_base_line_fit_ot_compare_summary.json",
    }


def _line_angle_degrees(curve: np.ndarray) -> float:
    arr = np.asarray(curve, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 2:
        raise ValueError("curve must have shape [N, 2] with N >= 2.")
    delta = arr[-1] - arr[0]
    angle = float(np.degrees(np.arctan2(delta[1], delta[0])) % 180.0)
    return angle


def _angle_error_degrees(got: float, target: float) -> float:
    diff = abs((float(got) - float(target) + 90.0) % 180.0 - 90.0)
    return float(diff)


def _fit_one(
    *,
    method_name: str,
    source_pairing: str,
    args: argparse.Namespace,
    data: dict[str, Any],
    condition_eval: np.ndarray,
    theta_eval: np.ndarray,
    base: LineSegmentBase,
    curve_u: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(int(args.seed) + 1000)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed) + 1000)
    model = CenteredConditionAffineFlowSKLModel(
        theta_dim=int(condition_eval.shape[1]),
        x_dim=2,
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        path_schedule=str(args.path_schedule),
    ).to(device)
    train_meta = train_geometric_base_affine_flow(
        model=model,
        base=base,
        theta_train=np.asarray(data["theta_train"], dtype=np.float64),
        x_train=np.asarray(data["x_train"], dtype=np.float64),
        theta_val=np.asarray(data["theta_val"], dtype=np.float64),
        x_val=np.asarray(data["x_val"], dtype=np.float64),
        device=device,
        path_schedule=str(args.path_schedule),
        source_pairing=str(source_pairing),
        ot_method=str(args.ot_method),
        ot_reg=float(args.ot_reg),
        ot_reg_m=float(args.ot_reg_m),
        ot_normalize_cost=bool(args.ot_normalize_cost),
        ot_num_threads=str(args.ot_num_threads),
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
    result = estimate_smoothed_curve_symmetric_kl(
        model=model,
        base=base,
        theta_all=condition_eval,
        device=device,
        smooth_sigma=float(args.smooth_sigma),
        mc_skl_samples=int(args.mc_skl_samples),
        density_mc_samples=int(args.density_mc_samples),
        ode_steps=int(args.ode_steps),
        ode_method=str(args.ode_method),
        batch_size=int(args.batch_size),
        train_metadata=train_meta,
    )
    fitted_curves: list[np.ndarray] = []
    for theta_row in condition_eval:
        curve, _ = push_base_curve(
            model=model,
            base=base,
            theta=theta_row.reshape(1, -1),
            device=device,
            u=curve_u,
            ode_steps=int(args.ode_steps),
            ode_method=str(args.ode_method),
        )
        fitted_curves.append(curve.detach().cpu().numpy().astype(np.float64))
    fitted_angles = np.asarray([_line_angle_degrees(c) for c in fitted_curves], dtype=np.float64)
    target_angles = np.asarray(np.degrees(theta_eval[:, 0]) % 180.0, dtype=np.float64)
    angle_errors = np.asarray(
        [_angle_error_degrees(g, t) for g, t in zip(fitted_angles, target_angles)],
        dtype=np.float64,
    )
    return {
        "method_name": str(method_name),
        "source_pairing": str(source_pairing),
        "ot_method": str(train_meta.get("ot_method", args.ot_method)),
        "model": model,
        "train_metadata": train_meta,
        "symmetric_kl_matrix": result.symmetric_kl_matrix,
        "skl_value": float(result.symmetric_kl_matrix[0, 1]),
        "fitted_curves": fitted_curves,
        "target_angles_degrees": target_angles,
        "fitted_angles_degrees": fitted_angles,
        "angle_errors_degrees": angle_errors,
        "mean_angle_error_degrees": float(np.mean(angle_errors)),
        "best_epoch": int(train_meta["best_epoch"]),
        "best_val_loss": float(train_meta["best_val_loss"]),
        "stopped_epoch": int(train_meta["stopped_epoch"]),
        "stopped_early": bool(train_meta["stopped_early"]),
    }


def _plot_comparison(
    *,
    png_path: Path,
    svg_path: Path,
    theta_eval: np.ndarray,
    x_plot: np.ndarray,
    theta_plot_scalar: np.ndarray,
    base_curve: np.ndarray,
    method_results: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#4c78a8", "#f58518"]
    n_method_panels = len(method_results)
    fig, axes = plt.subplots(
        1,
        n_method_panels + 1,
        figsize=(5.3 * (n_method_panels + 1), 5.3),
        gridspec_kw={"width_ratios": [1.0] * n_method_panels + [1.12]},
    )
    axes = np.atleast_1d(axes)
    for ax, method in zip(axes[:n_method_panels], method_results):
        for idx, theta_value in enumerate(theta_eval[:, 0]):
            mask = np.isclose(theta_plot_scalar[:, 0], float(theta_value))
            color = colors[idx % len(colors)]
            ax.scatter(
                x_plot[mask, 0],
                x_plot[mask, 1],
                s=15,
                alpha=0.48,
                color=color,
                linewidths=0,
                label=f"Test dataset {idx + 1}",
            )
            curve = np.asarray(method["fitted_curves"][idx], dtype=np.float64)
            angle = float(method["fitted_angles_degrees"][idx])
            err = float(method["angle_errors_degrees"][idx])
            ax.plot(
                curve[:, 0],
                curve[:, 1],
                color=color,
                linewidth=2.5,
                label=f"fitted line {idx + 1} ({angle:.1f} deg, err {err:.1f})",
            )
        ax.plot(base_curve[:, 0], base_curve[:, 1], color="#2f2f2f", linewidth=2.0, linestyle="--", label="base line")
        ax.text(
            0.02,
            0.98,
            f"SKL = {float(method['skl_value']):.4g}\nmean angle err = {float(method['mean_angle_error_degrees']):.2f} deg",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=11,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(str(method["method_name"]))
        ax.legend(frameon=False, loc="best", fontsize=8)

    ax_loss = axes[n_method_panels]
    for method in method_results:
        meta = method["train_metadata"]
        train_losses = np.asarray(meta["train_losses"], dtype=np.float64)
        val_monitor = np.asarray(meta["val_monitor_losses"], dtype=np.float64)
        epochs = np.arange(1, int(train_losses.size) + 1, dtype=np.int64)
        label = str(method["method_name"])
        ax_loss.plot(epochs, train_losses, linewidth=1.4, alpha=0.7, label=f"{label} train")
        ax_loss.plot(epochs, val_monitor, linewidth=1.8, linestyle="--", label=f"{label} val EMA")
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


def _selected_methods(run_method: str) -> list[tuple[str, str]]:
    key = str(run_method).strip().lower().replace("-", "_")
    if key == "regular":
        return [("Regular CFM", "random")]
    if key == "ot":
        return [("OT-CFM", "ot")]
    if key == "both":
        return [("Regular CFM", "random"), ("OT-CFM", "ot")]
    raise ValueError("run_method must be one of: regular, ot, both.")


def _summary_for_method(method: dict[str, Any]) -> dict[str, Any]:
    meta = dict(method["train_metadata"])
    return {
        "method_name": method["method_name"],
        "source_pairing": method["source_pairing"],
        "ot_method": method["ot_method"],
        "symmetric_kl_matrix": method["symmetric_kl_matrix"],
        "skl_value": method["skl_value"],
        "target_angles_degrees": method["target_angles_degrees"],
        "fitted_angles_degrees": method["fitted_angles_degrees"],
        "angle_errors_degrees": method["angle_errors_degrees"],
        "mean_angle_error_degrees": method["mean_angle_error_degrees"],
        "best_epoch": method["best_epoch"],
        "best_val_loss": method["best_val_loss"],
        "stopped_epoch": method["stopped_epoch"],
        "stopped_early": method["stopped_early"],
        "train_metadata": meta,
    }


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

    base = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0))
    curve_u = torch.linspace(base.u_low, base.u_high, int(args.curve_points), dtype=torch.float32).reshape(-1, 1)
    base_curve = base.points_from_u(curve_u).detach().cpu().numpy().astype(np.float64)

    methods = _selected_methods(str(args.run_method))
    method_results = [
        _fit_one(
            method_name=method_name,
            source_pairing=source_pairing,
            args=args,
            data=data,
            condition_eval=condition_eval,
            theta_eval=theta_eval,
            base=base,
            curve_u=curve_u,
            device=dev,
        )
        for method_name, source_pairing in methods
    ]
    _plot_comparison(
        png_path=paths["png"],
        svg_path=paths["svg"],
        theta_eval=theta_eval,
        x_plot=np.asarray(data["x_test_plot"], dtype=np.float64),
        theta_plot_scalar=np.asarray(data["theta_test_plot_scalar"], dtype=np.float64),
        base_curve=base_curve,
        method_results=method_results,
    )

    training_parameters = {
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "path_schedule": str(args.path_schedule),
        "run_method": str(args.run_method),
        "ot_method": str(args.ot_method),
        "ot_reg": float(args.ot_reg),
        "ot_reg_m": float(args.ot_reg_m),
        "ot_normalize_cost": bool(args.ot_normalize_cost),
        "ot_num_threads": str(args.ot_num_threads),
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
    }
    summary = {
        "script": "tests/run_geometric_base_line_fit_ot_compare.py",
        "device": str(dev),
        "training_parameters": training_parameters,
        "split_counts": data["split_counts"],
        "train_shape": list(np.asarray(data["theta_train"], dtype=np.float64).shape),
        "validation_shape": list(np.asarray(data["theta_val"], dtype=np.float64).shape),
        "test_shape": list(np.asarray(data["theta_test"], dtype=np.float64).shape),
        "methods": [_summary_for_method(m) for m in method_results],
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
    for method in method_results:
        print(
            f"{method['method_name']}: pairing={method['source_pairing']} "
            f"ot_method={method['ot_method']} "
            f"skl={float(method['skl_value']):.12g} "
            f"mean_angle_error={float(method['mean_angle_error_degrees']):.6g} "
            f"best_epoch={int(method['best_epoch'])} "
            f"best_val_loss={float(method['best_val_loss']):.12g}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
