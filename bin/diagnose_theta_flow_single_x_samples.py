#!/usr/bin/env python3
"""Sample theta-flow posterior for one fixed x and visualize theta distribution."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import torch

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from global_setting import DATA_DIR

from fisher.models import (
    ConditionalThetaFlowVelocity,
    ConditionalThetaFlowVelocityFiLMPerLayer,
    ConditionalThetaFlowVelocityIIDSoft,
    ConditionalThetaFlowVelocityThetaFourierFiLMPerLayer,
)
from fisher.shared_fisher_est import build_dataset_from_meta
from fisher.shared_dataset_io import load_shared_dataset_npz


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Load a fitted theta-flow posterior checkpoint, fix one x from a shared dataset NPZ, "
            "sample many theta draws from the conditional flow posterior, and plot the sampled theta distribution."
        )
    )
    p.add_argument(
        "--run-dir",
        type=str,
        default=str(Path(DATA_DIR) / "repro_theta_flow_mlp_n200" / "sweep_runs" / "n_000200"),
        help="Run directory containing theta_flow_posterior_checkpoint.pt.",
    )
    p.add_argument(
        "--dataset-npz",
        type=str,
        default=None,
        help="Shared dataset NPZ. If omitted, auto-resolve from run-dir parents.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Posterior checkpoint path. Defaults to <run-dir>/theta_flow_posterior_checkpoint.pt.",
    )
    p.add_argument("--x-index", type=int, default=0, help="Row index in x_all to hold fixed.")
    p.add_argument("--n-samples", type=int, default=20000, help="Number of theta samples to generate.")
    p.add_argument("--seed", type=int, default=7, help="Torch/NumPy RNG seed.")
    p.add_argument(
        "--ode-method",
        type=str,
        default="midpoint",
        choices=["euler", "midpoint", "rk4", "dopri5", "heun3", "bosh3", "fehlberg2", "adaptive_heun"],
        help="ODE integration method for ODESolver.sample.",
    )
    p.add_argument("--ode-atol", type=float, default=1e-5, help="Absolute tolerance for adaptive solvers.")
    p.add_argument("--ode-rtol", type=float, default=1e-5, help="Relative tolerance for adaptive solvers.")
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda"],
        help="Execution device. Per repo policy this diagnostic requires CUDA.",
    )
    return p


def _resolve_dataset_npz(run_dir: Path, user_path: str | None) -> Path:
    if user_path:
        p = Path(user_path).expanduser()
        return p if p.is_absolute() else (Path.cwd() / p)

    candidates = [
        run_dir / "shared_dataset.npz",
        run_dir.parent / "shared_dataset.npz",
        run_dir.parent.parent / "shared_dataset.npz",
        run_dir.parent.parent.parent / "shared_dataset.npz",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    tried = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not auto-resolve --dataset-npz from --run-dir. Tried:\n"
        f"  - {tried}\n"
        "Pass --dataset-npz explicitly."
    )


def _build_posterior_model_from_checkpoint(ckpt: dict[str, Any], device: torch.device) -> torch.nn.Module:
    arch = str(ckpt.get("flow_arch", "")).strip().lower()
    hparams_raw = ckpt.get("model_hparams", {})
    if not isinstance(hparams_raw, dict):
        raise ValueError("Checkpoint model_hparams is not a dict.")
    hparams = dict(hparams_raw)
    if arch == "mlp":
        model = ConditionalThetaFlowVelocity(**hparams).to(device)
    elif arch == "film":
        model = ConditionalThetaFlowVelocityFiLMPerLayer(**hparams).to(device)
    elif arch == "film_fourier":
        model = ConditionalThetaFlowVelocityThetaFourierFiLMPerLayer(**hparams).to(device)
    elif arch == "iid_soft":
        model = ConditionalThetaFlowVelocityIIDSoft(**hparams).to(device)
    else:
        raise ValueError(f"Unsupported flow_arch in checkpoint: {arch!r}")
    state_dict = ckpt.get("state_dict", None)
    if state_dict is None:
        raise ValueError("Checkpoint missing state_dict.")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _save_png_svg(fig: plt.Figure, png_path: Path, *, dpi: int = 170) -> Path:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi)
    svg_path = png_path.with_suffix(".svg")
    fig.savefig(svg_path)
    return svg_path


def _time_to_batch_column(t: torch.Tensor | float, ref: torch.Tensor) -> torch.Tensor:
    t_tensor = t if torch.is_tensor(t) else torch.tensor(float(t), device=ref.device, dtype=ref.dtype)
    t_tensor = t_tensor.to(device=ref.device, dtype=ref.dtype)
    batch = int(ref.shape[0])
    if t_tensor.ndim == 0:
        return t_tensor.expand(batch).unsqueeze(-1)
    if t_tensor.ndim == 1:
        if t_tensor.shape[0] == 1:
            return t_tensor.expand(batch).unsqueeze(-1)
        if t_tensor.shape[0] != batch:
            raise ValueError("ODE solver provided 1D time tensor with mismatched batch size.")
        return t_tensor.unsqueeze(-1)
    if t_tensor.ndim == 2:
        if t_tensor.shape[0] == 1:
            return t_tensor.expand(batch, t_tensor.shape[1])
        if t_tensor.shape[0] != batch:
            raise ValueError("ODE solver provided 2D time tensor with mismatched batch size.")
        return t_tensor
    raise ValueError("Unsupported time tensor rank from ODE solver.")


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = (Path.cwd() / run_dir)
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Per repo policy, do not fallback silently.")
    device = torch.device(args.device)

    ckpt_path = Path(args.checkpoint).expanduser() if args.checkpoint else (run_dir / "theta_flow_posterior_checkpoint.pt")
    if not ckpt_path.is_absolute():
        ckpt_path = (Path.cwd() / ckpt_path)
    ckpt_path = ckpt_path.resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Posterior checkpoint not found: {ckpt_path}\n"
            "Re-run the training with the current code so theta-flow checkpoints are emitted."
        )

    dataset_npz = _resolve_dataset_npz(run_dir, args.dataset_npz)
    bundle = load_shared_dataset_npz(dataset_npz)
    x_all = np.asarray(bundle.x_all, dtype=np.float64)
    if x_all.ndim != 2 or x_all.shape[0] < 1:
        raise ValueError(f"Dataset x_all must be shape (N,d) with N>=1, got {x_all.shape}.")

    x_index = int(args.x_index)
    if x_index < 0 or x_index >= int(x_all.shape[0]):
        raise ValueError(f"--x-index out of range: {x_index} not in [0, {x_all.shape[0] - 1}]")
    n_samples = int(args.n_samples)
    if n_samples < 1:
        raise ValueError("--n-samples must be >= 1.")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    model = _build_posterior_model_from_checkpoint(ckpt, device)

    theta_dim = int(ckpt.get("theta_dim_flow", 1))
    if theta_dim < 1:
        raise ValueError(f"Invalid theta_dim_flow in checkpoint: {theta_dim}")
    x_fixed = x_all[x_index : x_index + 1]
    x_cond = np.repeat(x_fixed.astype(np.float32, copy=False), n_samples, axis=0)
    x_cond_t = torch.from_numpy(x_cond).to(device)

    try:
        from flow_matching.solver.ode_solver import ODESolver
    except ImportError as e:
        raise ImportError(
            "This diagnostic requires flow_matching ODESolver. Install flow_matching in the geo_diffusion env."
        ) from e

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    theta_init = torch.randn((n_samples, theta_dim), device=device, dtype=torch.float32)
    t_grid = torch.tensor([0.0, 1.0], device=device, dtype=theta_init.dtype)
    def _vel(x: torch.Tensor, t: torch.Tensor, **model_extras: Any) -> torch.Tensor:
        x_cond_ = model_extras.get("x_cond", None)
        if x_cond_ is None:
            raise ValueError("theta-flow sampling requires model_extras['x_cond'].")
        return model(x, x_cond_, _time_to_batch_column(t, x))

    solver = ODESolver(velocity_model=_vel)
    with torch.no_grad():
        theta_samples_t = solver.sample(
            x_init=theta_init,
            step_size=None,
            method=str(args.ode_method),
            atol=float(args.ode_atol),
            rtol=float(args.ode_rtol),
            time_grid=t_grid,
            return_intermediates=False,
            enable_grad=False,
            x_cond=x_cond_t,
        )
    theta_samples = theta_samples_t.detach().cpu().numpy().astype(np.float64)

    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    out_npz = diag_dir / "theta_flow_single_x_posterior_samples.npz"
    np.savez_compressed(
        out_npz,
        theta_samples=theta_samples,
        theta_samples_dim0=theta_samples[:, 0],
        x_fixed=x_fixed,
        x_index=np.int64(x_index),
        n_samples=np.int64(n_samples),
        checkpoint_path=np.asarray([str(ckpt_path)], dtype=object),
        dataset_npz=np.asarray([str(dataset_npz)], dtype=object),
        flow_arch=np.asarray([str(ckpt.get("flow_arch", ""))], dtype=object),
    )

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.2), gridspec_kw={"width_ratios": [1.05, 1.25]})
    ax = axes[0]
    dim0 = theta_samples[:, 0]
    ax.hist(dim0, bins=100, density=True, alpha=0.8, color="#1f77b4", edgecolor="none")
    ax.set_xlabel("theta sample value (dim 0)")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"Posterior theta samples (x_index={x_index}, n={n_samples}, theta_dim={theta_dim})")
    if theta_dim > 1:
        ax.text(
            0.99,
            0.97,
            "Showing only theta dim 0",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
        )

    ax_tc = axes[1]
    ds = build_dataset_from_meta(bundle.meta)
    n_curve = 500
    t_grid = np.linspace(float(ds.theta_low), float(ds.theta_high), n_curve, dtype=np.float64).reshape(-1, 1)
    mu_grid = np.asarray(ds.tuning_curve(t_grid), dtype=np.float64)
    x_fixed_row = np.asarray(x_fixed, dtype=np.float64).reshape(-1)
    n_dim_plot = int(min(mu_grid.shape[1], x_fixed_row.shape[0]))
    for j in range(n_dim_plot):
        c = f"C{j % 10}"
        ax_tc.plot(
            t_grid[:, 0],
            mu_grid[:, j],
            color=c,
            linewidth=1.6,
            alpha=0.95,
            label=(f"mu_{j+1}(theta)" if j < 10 else None),
        )
        ax_tc.axhline(
            float(x_fixed_row[j]),
            color=c,
            linestyle="--",
            linewidth=0.9,
            alpha=0.55,
        )
    ax_tc.set_xlabel("theta")
    ax_tc.set_ylabel("x / tuning value")
    ax_tc.set_title("Dataset tuning curves with fixed x_j overlays (dashed)")
    ax_tc.grid(True, alpha=0.25)
    if n_dim_plot <= 10:
        ax_tc.legend(loc="upper right", fontsize=8, ncol=2)

    x_text = np.array2string(x_fixed_row, precision=3, suppress_small=False, max_line_width=1000)
    title_raw = f"Fixed x value (index={x_index}): {x_text}"
    title_wrapped = "\n".join(textwrap.wrap(title_raw, width=95))
    fig.suptitle(title_wrapped, fontsize=11, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    out_png = diag_dir / "theta_flow_single_x_posterior_hist.png"
    out_svg = _save_png_svg(fig, out_png, dpi=180)
    plt.close(fig)

    out_txt = diag_dir / "theta_flow_single_x_posterior_summary.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        f.write(f"run_dir: {run_dir}\n")
        f.write(f"checkpoint: {ckpt_path}\n")
        f.write(f"dataset_npz: {dataset_npz}\n")
        f.write(f"x_index: {x_index}\n")
        f.write(f"n_samples: {n_samples}\n")
        f.write(f"theta_dim: {theta_dim}\n")
        f.write(f"flow_arch: {ckpt.get('flow_arch', '')}\n")
        f.write(f"theta_dim0_mean: {float(np.mean(dim0)):.8f}\n")
        f.write(f"theta_dim0_std: {float(np.std(dim0)):.8f}\n")
        f.write(f"theta_dim0_q05: {float(np.quantile(dim0, 0.05)):.8f}\n")
        f.write(f"theta_dim0_q50: {float(np.quantile(dim0, 0.50)):.8f}\n")
        f.write(f"theta_dim0_q95: {float(np.quantile(dim0, 0.95)):.8f}\n")

    print(f"[diag] run_dir={run_dir}")
    print(f"[diag] checkpoint={ckpt_path}")
    print(f"[diag] dataset_npz={dataset_npz}")
    print(f"[diag] saved samples={out_npz}")
    print(f"[diag] saved figure_png={out_png}")
    print(f"[diag] saved figure_svg={out_svg}")
    print(f"[diag] saved summary={out_txt}")


if __name__ == "__main__":
    main()
