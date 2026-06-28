#!/usr/bin/env python3
"""Fit an affine flow-matching velocity on a noisy-line target."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import DATA_DIR, DEFAULT_DEVICE

from fisher.gaussian_x_flow import GaussianAffinePathSchedule, path_schedule_from_name


RESULTS_NPZ_NAME = "noisy_line_affine_fm_results.npz"
SUMMARY_JSON_NAME = "noisy_line_affine_fm_summary.json"
MODEL_STATE_NAME = "model_state.pt"
FIGURE_STEM = "noisy_line_affine_fm_overlay"


@dataclass(frozen=True)
class NoisyLineBatch:
    x0: np.ndarray
    x1: np.ndarray
    u: np.ndarray
    eta: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument("--output-dir", type=Path, default=Path(DATA_DIR) / "noisy_line_affine_fm_standard")
    p.add_argument("--force", action="store_true", help="Overwrite existing output artifacts.")

    p.add_argument("--theta", type=float, default=math.pi / 6.0)
    p.add_argument("--ell", type=float, default=1.5)
    p.add_argument("--sigma", type=float, default=0.12)
    p.add_argument("--shift-x", type=float, default=0.0)
    p.add_argument("--shift-y", type=float, default=0.0)
    p.add_argument("--path-schedule", choices=("linear", "straight", "cosine", "cos"), default="linear")

    p.add_argument("--train-n", type=int, default=16_384)
    p.add_argument("--val-n", type=int, default=4_096)
    p.add_argument("--plot-n", type=int, default=4_000)
    p.add_argument("--steps", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=1_024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--t-eps", type=float, default=0.0)
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--ode-steps", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    return p


def require_requested_device(name: str) -> torch.device:
    device = torch.device(str(name))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable. Per repo policy, do not fallback silently.")
        if device.index is not None and int(device.index) >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {device.index} requested but only {torch.cuda.device_count()} device(s) are visible."
            )
    return device


def noisy_line_basis(theta: float) -> tuple[np.ndarray, np.ndarray]:
    th = float(theta)
    q = np.asarray([math.cos(th), math.sin(th)], dtype=np.float64)
    n = np.asarray([-math.sin(th), math.cos(th)], dtype=np.float64)
    return q, n


def generate_noisy_line_batch(
    *,
    num: int,
    theta: float,
    ell: float,
    sigma: float,
    shift: tuple[float, float],
    rng: np.random.Generator,
) -> NoisyLineBatch:
    count = int(num)
    if count < 1:
        raise ValueError("num must be >= 1.")
    if float(ell) <= 0.0:
        raise ValueError("ell must be > 0.")
    if float(sigma) < 0.0:
        raise ValueError("sigma must be >= 0.")
    u = rng.uniform(-0.5, 0.5, size=(count, 1)).astype(np.float64, copy=False)
    eta = rng.standard_normal(size=(count, 1)).astype(np.float64, copy=False)
    q, n = noisy_line_basis(float(theta))
    a = np.asarray(shift, dtype=np.float64).reshape(1, 2)
    x0 = np.concatenate([u, np.zeros_like(u)], axis=1)
    x1 = a + float(ell) * u * q.reshape(1, 2) + float(sigma) * eta * n.reshape(1, 2)
    return NoisyLineBatch(
        x0=x0.astype(np.float64, copy=False),
        x1=x1.astype(np.float64, copy=False),
        u=u.reshape(-1).astype(np.float64, copy=False),
        eta=eta.reshape(-1).astype(np.float64, copy=False),
    )


class TimeAffineVelocity(nn.Module):
    """Time-dependent affine velocity ``v(x,t)=A(t)x+b(t)`` in 2D."""

    def __init__(self, *, hidden_dim: int = 64, depth: int = 2) -> None:
        super().__init__()
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be >= 1.")
        if int(depth) < 1:
            raise ValueError("depth must be >= 1.")
        layers: list[nn.Module] = []
        in_dim = 3
        for _ in range(int(depth)):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.SiLU())
            in_dim = int(hidden_dim)
        out = nn.Linear(in_dim, 6)
        nn.init.xavier_uniform_(out.weight, gain=0.01)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    @staticmethod
    def time_features(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.reshape(1, 1)
        elif t.ndim == 1:
            t = t.unsqueeze(-1)
        if t.ndim != 2 or int(t.shape[1]) != 1:
            raise ValueError("t must have shape [B] or [B, 1].")
        return torch.cat([t, torch.sin(math.pi * t), torch.cos(math.pi * t)], dim=1)

    def coefficients(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.net(self.time_features(t))
        a = params[:, :4].reshape(-1, 2, 2)
        b = params[:, 4:].reshape(-1, 2)
        return a, b

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or int(x.shape[1]) != 2:
            raise ValueError("x must have shape [B, 2].")
        if t.ndim == 0 or (t.ndim == 1 and int(t.shape[0]) == 1) or (t.ndim == 2 and int(t.shape[0]) == 1):
            t = t.reshape(1, 1).expand(int(x.shape[0]), 1)
        elif t.ndim == 1:
            t = t.unsqueeze(-1)
        if int(t.shape[0]) != int(x.shape[0]):
            raise ValueError("x and t batch sizes must match.")
        a, b = self.coefficients(t)
        return torch.bmm(a, x.unsqueeze(-1)).squeeze(-1) + b


def _to_tensor(a: np.ndarray, *, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)


def _sample_t(batch: int, *, t_eps: float, device: torch.device) -> torch.Tensor:
    eps = float(t_eps)
    if eps < 0.0 or eps >= 0.5:
        raise ValueError("t_eps must be in [0, 0.5).")
    return eps + (1.0 - 2.0 * eps) * torch.rand(int(batch), 1, device=device)


def flow_matching_loss(
    model: TimeAffineVelocity,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    schedule: GaussianAffinePathSchedule,
) -> torch.Tensor:
    alpha, beta, alpha_dot, beta_dot = schedule.ab_ad_bd(t)
    xt = alpha * x0 + beta * x1
    target = alpha_dot * x0 + beta_dot * x1
    pred = model(xt, t)
    return torch.mean(torch.sum((pred - target) ** 2, dim=1))


def train_affine_velocity(
    *,
    model: TimeAffineVelocity,
    train: NoisyLineBatch,
    val: NoisyLineBatch,
    schedule: GaussianAffinePathSchedule,
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    t_eps: float,
    val_every: int,
    log_every: int,
    max_grad_norm: float,
    seed: int,
) -> dict[str, Any]:
    if int(steps) < 1:
        raise ValueError("steps must be >= 1.")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1.")
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    x0_train = _to_tensor(train.x0, device=device)
    x1_train = _to_tensor(train.x1, device=device)
    x0_val = _to_tensor(val.x0, device=device)
    x1_val = _to_tensor(val.x1, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    train_steps: list[int] = []
    train_losses: list[float] = []
    val_steps: list[int] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    n_train = int(x0_train.shape[0])

    for step in range(1, int(steps) + 1):
        idx = torch.randint(0, n_train, (int(batch_size),), device=device)
        t = _sample_t(int(batch_size), t_eps=float(t_eps), device=device)
        loss = flow_matching_loss(model, x0_train[idx], x1_train[idx], t, schedule)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if float(max_grad_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        opt.step()

        if step == 1 or step % int(log_every) == 0 or step == int(steps):
            loss_value = float(loss.detach().cpu())
            train_steps.append(step)
            train_losses.append(loss_value)
            print(f"[noisy-line-affine-fm] step={step} train_loss={loss_value:.6g}", flush=True)

        if step == 1 or step % int(val_every) == 0 or step == int(steps):
            model.eval()
            with torch.no_grad():
                t_val = _sample_t(int(x0_val.shape[0]), t_eps=float(t_eps), device=device)
                val_loss = float(flow_matching_loss(model, x0_val, x1_val, t_val, schedule).detach().cpu())
            model.train()
            val_steps.append(step)
            val_losses.append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[noisy-line-affine-fm] step={step} val_loss={val_loss:.6g}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "train_steps": train_steps,
        "train_losses": train_losses,
        "val_steps": val_steps,
        "val_losses": val_losses,
        "best_step": int(best_step),
        "best_val_loss": float(best_val),
    }


@torch.no_grad()
def sample_flow_endpoint(
    *,
    model: TimeAffineVelocity,
    x0: np.ndarray,
    device: torch.device,
    ode_steps: int,
) -> np.ndarray:
    steps = int(ode_steps)
    if steps < 1:
        raise ValueError("ode_steps must be >= 1.")
    model.eval()
    x = _to_tensor(x0, device=device)
    dt = 1.0 / float(steps)
    for i in range(steps):
        t0 = torch.full((int(x.shape[0]), 1), float(i) * dt, dtype=x.dtype, device=device)
        half = torch.full_like(t0, float(i) * dt + 0.5 * dt)
        t1 = torch.full_like(t0, float(i + 1) * dt)
        k1 = model(x, t0)
        k2 = model(x + 0.5 * dt * k1, half)
        k3 = model(x + 0.5 * dt * k2, half)
        k4 = model(x + dt * k3, t1)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x.detach().cpu().numpy().astype(np.float64)


def plot_overlay(path_base: Path, *, target_x1: np.ndarray, generated_x1: np.ndarray) -> tuple[Path, Path]:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {
            "font.size": 13.0,
            "axes.titlesize": 14.0,
            "axes.labelsize": 13.0,
            "legend.fontsize": 12.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        fig, ax = plt.subplots(figsize=(6.0, 5.6), layout="constrained")
        ax.scatter(
            target_x1[:, 0],
            target_x1[:, 1],
            s=10,
            alpha=0.28,
            color="#4C78A8",
            edgecolors="none",
            label="target noisy line",
            rasterized=True,
        )
        ax.scatter(
            generated_x1[:, 0],
            generated_x1[:, 1],
            s=12,
            alpha=0.58,
            color="#F58518",
            marker="x",
            linewidths=0.7,
            label="flow-generated samples",
            rasterized=True,
        )
        pts = np.vstack([target_x1, generated_x1])
        lo = np.nanmin(pts, axis=0)
        hi = np.nanmax(pts, axis=0)
        center = 0.5 * (lo + hi)
        radius = 0.56 * float(np.nanmax(hi - lo))
        radius = max(radius, 0.5)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("Noisy-line target vs affine-flow endpoints")
        ax.grid(False)
        ax.legend(frameon=False, loc="best")
    svg = path_base.with_suffix(".svg")
    png = path_base.with_suffix(".png")
    fig.savefig(svg)
    fig.savefig(png, dpi=400)
    plt.close(fig)
    return svg, png


def _save_summary(path: Path, summary: dict[str, Any]) -> Path:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def _fail_if_outputs_exist(output_dir: Path, *, force: bool) -> None:
    if bool(force):
        return
    existing = [
        output_dir / RESULTS_NPZ_NAME,
        output_dir / SUMMARY_JSON_NAME,
        output_dir / MODEL_STATE_NAME,
        output_dir / f"{FIGURE_STEM}.png",
        output_dir / f"{FIGURE_STEM}.svg",
    ]
    present = [p for p in existing if p.exists()]
    if present:
        names = ", ".join(str(p) for p in present)
        raise FileExistsError(f"Output artifacts already exist; pass --force to overwrite: {names}")


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    _fail_if_outputs_exist(output_dir, force=bool(args.force))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = require_requested_device(str(args.device))
    schedule = path_schedule_from_name(str(args.path_schedule))
    rng = np.random.default_rng(int(args.seed))
    shift = (float(args.shift_x), float(args.shift_y))
    train = generate_noisy_line_batch(
        num=int(args.train_n),
        theta=float(args.theta),
        ell=float(args.ell),
        sigma=float(args.sigma),
        shift=shift,
        rng=rng,
    )
    val = generate_noisy_line_batch(
        num=int(args.val_n),
        theta=float(args.theta),
        ell=float(args.ell),
        sigma=float(args.sigma),
        shift=shift,
        rng=rng,
    )
    plot_target = generate_noisy_line_batch(
        num=int(args.plot_n),
        theta=float(args.theta),
        ell=float(args.ell),
        sigma=float(args.sigma),
        shift=shift,
        rng=rng,
    )
    plot_base = generate_noisy_line_batch(
        num=int(args.plot_n),
        theta=float(args.theta),
        ell=float(args.ell),
        sigma=0.0,
        shift=(0.0, 0.0),
        rng=rng,
    )

    model = TimeAffineVelocity(hidden_dim=int(args.hidden_dim), depth=int(args.depth)).to(device)
    train_meta = train_affine_velocity(
        model=model,
        train=train,
        val=val,
        schedule=schedule,
        device=device,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        t_eps=float(args.t_eps),
        val_every=max(1, int(args.val_every)),
        log_every=max(1, int(args.log_every)),
        max_grad_norm=float(args.max_grad_norm),
        seed=int(args.seed),
    )
    generated_x1 = sample_flow_endpoint(
        model=model,
        x0=plot_base.x0,
        device=device,
        ode_steps=int(args.ode_steps),
    )
    svg, png = plot_overlay(output_dir / FIGURE_STEM, target_x1=plot_target.x1, generated_x1=generated_x1)

    q, normal = noisy_line_basis(float(args.theta))
    results_npz = output_dir / RESULTS_NPZ_NAME
    np.savez_compressed(
        results_npz,
        train_x0=train.x0,
        train_x1=train.x1,
        val_x0=val.x0,
        val_x1=val.x1,
        plot_target_x1=plot_target.x1,
        plot_base_x0=plot_base.x0,
        generated_x1=generated_x1,
        q=q,
        n=normal,
        shift=np.asarray(shift, dtype=np.float64),
        theta=np.asarray([float(args.theta)], dtype=np.float64),
        ell=np.asarray([float(args.ell)], dtype=np.float64),
        sigma=np.asarray([float(args.sigma)], dtype=np.float64),
        train_steps=np.asarray(train_meta["train_steps"], dtype=np.int64),
        train_losses=np.asarray(train_meta["train_losses"], dtype=np.float64),
        val_steps=np.asarray(train_meta["val_steps"], dtype=np.int64),
        val_losses=np.asarray(train_meta["val_losses"], dtype=np.float64),
    )
    model_state = output_dir / MODEL_STATE_NAME
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "train_metadata": train_meta,
        },
        model_state,
    )
    summary = {
        "script": "bin/run_noisy_line_affine_fm.py",
        "output_dir": str(output_dir),
        "device": str(args.device),
        "seed": int(args.seed),
        "theta": float(args.theta),
        "ell": float(args.ell),
        "sigma": float(args.sigma),
        "shift": [float(args.shift_x), float(args.shift_y)],
        "path_schedule": str(args.path_schedule),
        "train_n": int(args.train_n),
        "val_n": int(args.val_n),
        "plot_n": int(args.plot_n),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "t_eps": float(args.t_eps),
        "ode_steps": int(args.ode_steps),
        "best_step": int(train_meta["best_step"]),
        "best_val_loss": float(train_meta["best_val_loss"]),
        "results_npz": str(results_npz),
        "model_state": str(model_state),
        "figure": [str(svg), str(png)],
    }
    summary_json = _save_summary(output_dir / SUMMARY_JSON_NAME, summary)
    print(f"results_npz: {results_npz}", flush=True)
    print(f"summary_json: {summary_json}", flush=True)
    print(f"model_state: {model_state}", flush=True)
    print(f"figure_png: {png}", flush=True)


if __name__ == "__main__":
    main()
