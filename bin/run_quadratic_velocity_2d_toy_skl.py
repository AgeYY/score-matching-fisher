#!/usr/bin/env python3
"""Run the 2D quadratic-velocity toy symmetric-KL experiment."""

from __future__ import annotations

import argparse
import csv
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
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import DATA_DIR

from fisher.flow_matching_skl import (
    build_flow_skl_model,
    estimate_model_symmetric_kl,
    flow_skl_result_to_npz_dict,
    train_flow_skl_model,
)
from fisher.model_weight_ema import scalar_val_ema_update
from fisher.shared_fisher_est import require_device


RESULTS_CSV_NAME = "quadratic_velocity_2d_toy_skl_errors.csv"
RESULTS_NPZ_NAME = "quadratic_velocity_2d_toy_skl_results.npz"
SUMMARY_JSON_NAME = "quadratic_velocity_2d_toy_skl_summary.json"


@dataclass(frozen=True)
class ToyDataset:
    z_all: np.ndarray
    theta_all: np.ndarray
    x_all: np.ndarray
    labels: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    z_train: np.ndarray
    theta_train: np.ndarray
    x_train: np.ndarray
    z_val: np.ndarray
    theta_val: np.ndarray
    x_val: np.ndarray
    true_skl: float


@dataclass(frozen=True)
class ModelSpec:
    name: str
    velocity_family: str


_ALL_MODEL_SPECS = {
    "affine": ModelSpec(name="affine", velocity_family="condition_affine"),
    "quadratic": ModelSpec(name="quadratic", velocity_family="condition_quadratic"),
    "neural": ModelSpec(name="neural", velocity_family="nonlinear"),
}

_MODEL_DISPLAY_NAMES = {
    "affine": "Affine",
    "quadratic": "Quadratic",
    "neural": "Neural",
}

_MODEL_COLORS = {
    "affine": "#4C78A8",
    "quadratic": "#F58518",
    "neural": "#54A24B",
}


def _parse_int_list(value: str) -> list[int]:
    vals = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer.")
    return vals


def _parse_model_list(value: str) -> list[str]:
    vals = [part.strip().lower() for part in str(value).split(",") if part.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated model name.")
    invalid = [v for v in vals if v not in _ALL_MODEL_SPECS]
    if invalid:
        allowed = ",".join(_ALL_MODEL_SPECS)
        raise argparse.ArgumentTypeError(f"Unknown model(s) {invalid}; allowed values are {allowed}.")
    deduped: list[str] = []
    for val in vals:
        if val not in deduped:
            deduped.append(val)
    return deduped


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "quadratic_velocity_2d_toy_skl_low_n_transition",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--plots-only", action="store_true", help="Regenerate plots from existing CSV/NPZ outputs.")
    p.add_argument(
        "--models",
        type=_parse_model_list,
        default=["affine", "quadratic", "neural"],
        help="Comma-separated subset/order of velocity classes to run: affine,quadratic,neural.",
    )
    p.add_argument("--n-list", type=_parse_int_list, default=[4, 5, 8, 10, 16, 30, 50, 100])
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--x-dim", type=int, default=4, help="Even observed dimension; each adjacent pair is a quadratic shear.")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--amplitude", type=float, default=0.5)
    p.add_argument(
        "--training-mode",
        choices=("exact", "cfm"),
        default="exact",
        help=(
            "exact trains on the analytic ODE path from the note; cfm uses the repository's "
            "standard independent-endpoint conditional flow-matching objective."
        ),
    )

    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--early-patience", type=int, default=500)
    p.add_argument("--early-min-delta", type=float, default=1e-4)
    p.add_argument("--early-ema-alpha", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--path-schedule", choices=("cosine", "linear", "straight"), default="linear")
    p.add_argument("--t-eps", type=float, default=0.0005)
    p.add_argument("--quadrature-steps", type=int, default=64)
    p.add_argument("--divergence-estimator", choices=("hutchinson", "exact"), default="exact")
    p.add_argument("--hutchinson-probes", type=int, default=1)
    p.add_argument("--mc-jeffreys-sample", type=int, default=8192)
    p.add_argument("--ode-steps", type=int, default=64)
    p.add_argument("--ode-method", type=str, default="midpoint")
    p.add_argument("--solve-jitter", type=float, default=1e-6)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--plot-stat", choices=("median_iqr", "mean_sd"), default="median_iqr")
    p.add_argument("--plot-ymin", type=float, default=-0.35)
    p.add_argument("--plot-ymax", type=float, default=6.0)
    p.add_argument("--dataset-plot-n", type=int, default=1000)
    return p


def _validate_x_dim(x_dim: int) -> int:
    xd = int(x_dim)
    if xd < 2 or xd % 2 != 0:
        raise ValueError("x_dim must be an even integer >= 2.")
    return xd


def true_quadratic_toy_skl(amplitude: float, *, x_dim: int = 2) -> float:
    return float((int(_validate_x_dim(x_dim)) // 2) * 8.0 * float(amplitude) ** 2)


def generate_quadratic_toy_dataset(
    *,
    n_per_condition: int,
    amplitude: float,
    x_dim: int,
    seed: int,
    train_frac: float,
) -> ToyDataset:
    n = int(n_per_condition)
    if n < 2:
        raise ValueError("n_per_condition must be >= 2.")
    if not (0.0 < float(train_frac) < 1.0):
        raise ValueError("train_frac must be in (0, 1).")
    xd = _validate_x_dim(int(x_dim))
    amp = float(amplitude)
    s = math.sqrt(1.0 + 2.0 * amp * amp)
    rng = np.random.default_rng(int(seed))

    xs: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    thetas: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for condition, a_c in enumerate((-amp, amp)):
        z = rng.standard_normal(size=(n, xd)).astype(np.float64, copy=False)
        y = z.copy()
        for first in range(0, xd, 2):
            second = first + 1
            y[:, first] = (z[:, first] + float(a_c) * (z[:, second] ** 2 - 1.0)) / s
        zs.append(z)
        xs.append(y)
        thetas.append(np.eye(2, dtype=np.float64)[np.full(n, condition, dtype=np.int64)])
        labels.append(np.full(n, condition, dtype=np.int64))

    z_all = np.vstack(zs).astype(np.float64, copy=False)
    x_all = np.vstack(xs).astype(np.float64, copy=False)
    theta_all = np.vstack(thetas).astype(np.float64, copy=False)
    label_all = np.concatenate(labels).astype(np.int64, copy=False)

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for condition in range(2):
        idx = np.flatnonzero(label_all == condition)
        rng.shuffle(idx)
        n_train = int(math.floor(float(train_frac) * float(idx.size)))
        n_train = min(max(n_train, 1), int(idx.size) - 1)
        train_parts.append(idx[:n_train])
        val_parts.append(idx[n_train:])
    train_idx = np.concatenate(train_parts).astype(np.int64, copy=False)
    val_idx = np.concatenate(val_parts).astype(np.int64, copy=False)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return ToyDataset(
        z_all=z_all,
        theta_all=theta_all,
        x_all=x_all,
        labels=label_all,
        train_idx=train_idx,
        val_idx=val_idx,
        z_train=z_all[train_idx],
        theta_train=theta_all[train_idx],
        x_train=x_all[train_idx],
        z_val=z_all[val_idx],
        theta_val=theta_all[val_idx],
        x_val=x_all[val_idx],
        true_skl=true_quadratic_toy_skl(amp, x_dim=xd),
    )


def _seed_for_repeat(base_seed: int, repeat_idx: int) -> int:
    return int(base_seed) + int(repeat_idx)


def _as_2d_float32(value: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got shape {arr.shape}.")
    return np.ascontiguousarray(arr)


def _exact_path_state_and_velocity(
    z: torch.Tensor,
    theta: torch.Tensor,
    t: torch.Tensor,
    *,
    amplitude: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytic quadratic-toy flow path and velocity from the journal note."""

    if z.ndim != 2 or int(z.shape[1]) < 2 or int(z.shape[1]) % 2 != 0:
        raise ValueError(f"z must have shape (batch, even_x_dim>=2); got {tuple(z.shape)}.")
    if theta.ndim != 2 or theta.shape[1] != 2:
        raise ValueError(f"theta must have shape (batch, 2); got {tuple(theta.shape)}.")
    if t.ndim == 1:
        t_col = t[:, None]
    elif t.ndim == 2 and t.shape[1] == 1:
        t_col = t
    else:
        raise ValueError(f"t must have shape (batch,) or (batch, 1); got {tuple(t.shape)}.")

    amp = float(amplitude)
    lam = math.log(math.sqrt(1.0 + 2.0 * amp * amp))
    a_c = amp * (theta[:, 1:2] - theta[:, 0:1])
    decay = torch.exp(-float(lam) * t_col)

    x_t = z.clone()
    v_t = torch.zeros_like(z)
    for first in range(0, int(z.shape[1]), 2):
        second = first + 1
        z_first = z[:, first : first + 1]
        z_second = z[:, second : second + 1]
        feature = z_second.square() - 1.0
        x_t[:, first : first + 1] = decay * (z_first + a_c * t_col * feature)
        x_t[:, second : second + 1] = z_second
        v_t[:, first : first + 1] = -float(lam) * x_t[:, first : first + 1] + a_c * decay * feature
    return x_t, t_col, v_t


def train_flow_skl_model_exact_quadratic_toy(
    *,
    model: torch.nn.Module,
    theta_train: np.ndarray,
    z_train: np.ndarray,
    theta_val: np.ndarray | None,
    z_val: np.ndarray | None,
    device: torch.device,
    amplitude: float,
    velocity_family: str,
    epochs: int = 1000,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    t_eps: float = 0.0005,
    patience: int = 0,
    min_delta: float = 1e-4,
    ema_alpha: float = 0.05,
    max_grad_norm: float = 10.0,
    log_every: int = 50,
) -> dict[str, Any]:
    """Train on the analytic ODE path from the 2D quadratic toy note."""

    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1.")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1.")
    if float(lr) <= 0.0:
        raise ValueError("lr must be > 0.")
    te = float(t_eps)
    if not (0.0 < te < 0.5):
        raise ValueError("t_eps must be in (0, 0.5).")
    alpha = float(ema_alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("ema_alpha must be in (0, 1].")

    th_tr = _as_2d_float32(theta_train, name="theta_train")
    z_tr = _as_2d_float32(z_train, name="z_train")
    if theta_val is None or z_val is None:
        th_va = th_tr
        z_va = z_tr
    else:
        th_va = _as_2d_float32(theta_val, name="theta_val")
        z_va = _as_2d_float32(z_val, name="z_val")
    if th_tr.shape[0] < 1 or z_tr.shape[0] < 1 or th_va.shape[0] < 1 or z_va.shape[0] < 1:
        raise ValueError("train and validation splits must be non-empty.")
    if th_tr.shape[0] != z_tr.shape[0] or th_va.shape[0] != z_va.shape[0]:
        raise ValueError("theta and z splits must have matching row counts.")
    if th_tr.shape[1] != 2 or th_va.shape[1] != 2:
        raise ValueError("theta splits must have shape (n, 2).")
    _validate_x_dim(int(z_tr.shape[1]))
    if z_tr.shape[1] != z_va.shape[1]:
        raise ValueError("train and validation z splits must have the same x_dim.")

    train_ds = TensorDataset(torch.from_numpy(th_tr), torch.from_numpy(z_tr))
    val_ds = TensorDataset(torch.from_numpy(th_va), torch.from_numpy(z_va))
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)

    model.to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_monitor_losses: list[float] = []
    val_ema: float | None = None
    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    stopped_early = False
    stopped_epoch = int(epochs)
    n_clipped_steps = 0
    n_total_steps = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        ep_losses: list[float] = []
        for tb, zb in train_loader:
            tb = tb.to(device)
            zb = zb.to(device)
            bs = int(zb.shape[0])
            t_raw = torch.rand(bs, device=device, dtype=zb.dtype)
            t = te + (1.0 - 2.0 * te) * t_raw
            x_t, t_col, v_t = _exact_path_state_and_velocity(zb, tb, t, amplitude=float(amplitude))
            loss = torch.mean((model(x_t, tb, t_col) - v_t) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            n_total_steps += 1
            if float(max_grad_norm) > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    float(max_grad_norm),
                )
                if float(grad_norm) > float(max_grad_norm):
                    n_clipped_steps += 1
            opt.step()
            ep_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(ep_losses))
        train_losses.append(train_loss)

        model.eval()
        val_ep: list[float] = []
        with torch.no_grad():
            for tb, zb in val_loader:
                tb = tb.to(device)
                zb = zb.to(device)
                bs = int(zb.shape[0])
                t_raw = torch.rand(bs, device=device, dtype=zb.dtype)
                t = te + (1.0 - 2.0 * te) * t_raw
                x_t, t_col, v_t = _exact_path_state_and_velocity(zb, tb, t, amplitude=float(amplitude))
                val_ep.append(float(torch.mean((model(x_t, tb, t_col) - v_t) ** 2).detach().cpu()))
        val_loss = float(np.mean(val_ep))
        val_losses.append(val_loss)
        val_ema = scalar_val_ema_update(val_ema, val_loss, alpha)
        val_smooth = float(val_ema)
        val_monitor_losses.append(val_smooth)

        if val_smooth < best_val - float(min_delta):
            best_val = val_smooth
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % max(1, int(log_every)) == 0 or epoch == int(epochs):
            print(
                f"[quadratic-toy exact {velocity_family} {epoch:4d}/{int(epochs)}] "
                f"train={train_loss:.6f} val={val_loss:.6f} val_smooth={val_smooth:.6f} "
                f"best_smooth={best_val:.6f} best_epoch={best_epoch}",
                flush=True,
            )
        if int(patience) > 0 and patience_counter >= int(patience):
            stopped_early = True
            stopped_epoch = int(epoch)
            print(
                f"[quadratic-toy exact {velocity_family} early-stop] epoch={epoch} "
                f"best_epoch={best_epoch} best_smooth={best_val:.6f} patience={int(patience)}",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "velocity_family": str(velocity_family),
        "network_architecture": str(getattr(model, "network_architecture", "film")),
        "training_mode": "exact",
        "train_losses": np.asarray(train_losses, dtype=np.float64),
        "val_losses": np.asarray(val_losses, dtype=np.float64),
        "val_monitor_losses": np.asarray(val_monitor_losses, dtype=np.float64),
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
        "n_clipped_steps": int(n_clipped_steps),
        "n_total_steps": int(n_total_steps),
        "path_schedule": "analytic_quadratic_toy",
        "early_ema_alpha": float(alpha),
    }


def _model_specs(model_names: list[str]) -> list[ModelSpec]:
    return [_ALL_MODEL_SPECS[str(name)] for name in model_names]


def _case_dir(output_dir: Path, *, n_per_condition: int, seed: int, model_name: str | None = None) -> Path:
    base = Path(output_dir) / f"N_{int(n_per_condition)}" / f"seed_{int(seed)}"
    return base if model_name is None else base / str(model_name)


def _load_case_result(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "estimate": float(np.asarray(data["estimate_skl"]).reshape(-1)[0]),
            "true_skl": float(np.asarray(data["true_skl"]).reshape(-1)[0]),
            "relative_error": float(np.asarray(data["relative_error"]).reshape(-1)[0]),
            "best_epoch": int(np.asarray(data["best_epoch"]).reshape(-1)[0]) if "best_epoch" in data.files else -1,
            "best_val_loss": float(np.asarray(data["best_val_loss"]).reshape(-1)[0])
            if "best_val_loss" in data.files
            else float("nan"),
        }


def train_one_model(
    *,
    dataset: ToyDataset,
    spec: ModelSpec,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    result_npz = output_dir / "flow_matching_skl_results.npz"
    if result_npz.is_file() and not bool(args.force):
        return result_npz, _load_case_result(result_npz)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    model = build_flow_skl_model(
        velocity_family=spec.velocity_family,
        theta_dim=2,
        x_dim=int(dataset.z_train.shape[1]),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        quadrature_steps=int(args.quadrature_steps),
        path_schedule=str(args.path_schedule),
        divergence_estimator=str(args.divergence_estimator),
        hutchinson_probes=int(args.hutchinson_probes),
    ).to(device)
    if str(args.training_mode) == "exact":
        train_meta = train_flow_skl_model_exact_quadratic_toy(
            model=model,
            theta_train=dataset.theta_train,
            z_train=dataset.z_train,
            theta_val=dataset.theta_val,
            z_val=dataset.z_val,
            device=device,
            amplitude=float(args.amplitude),
            velocity_family=spec.velocity_family,
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
    else:
        train_meta = train_flow_skl_model(
            model=model,
            theta_train=dataset.theta_train,
            x_train=dataset.x_train,
            theta_val=dataset.theta_val,
            x_val=dataset.x_val,
            device=device,
            velocity_family=spec.velocity_family,
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
        train_meta["training_mode"] = "cfm"
    theta_eval = np.eye(2, dtype=np.float64)
    result = estimate_model_symmetric_kl(
        model=model,
        theta_all=theta_eval,
        device=device,
        velocity_family=spec.velocity_family,
        mc_jeffreys_sample=int(args.mc_jeffreys_sample),
        ode_steps=int(args.ode_steps),
        ode_method=str(args.ode_method),
        batch_size=int(args.batch_size),
        solve_jitter=float(args.solve_jitter),
        quadrature_steps=int(args.quadrature_steps),
        fisher_kind="none",
        train_metadata=train_meta,
    )
    estimate = float(result.symmetric_kl_matrix[0, 1])
    truth = float(dataset.true_skl)
    rel_error = abs(estimate - truth) / truth if truth > 0.0 else float("nan")

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = flow_skl_result_to_npz_dict(result)
    for text_key in ("canonical_metric_name", "network_architecture"):
        if text_key in fields:
            fields[text_key] = np.asarray([str(np.asarray(fields[text_key]).reshape(-1)[0])])
    fields.update(
        {
            "model_name": np.asarray([spec.name]),
            "velocity_family": np.asarray([spec.velocity_family]),
            "training_mode": np.asarray([str(args.training_mode)]),
            "theta_eval": theta_eval,
            "estimate_skl": np.asarray([estimate], dtype=np.float64),
            "true_skl": np.asarray([truth], dtype=np.float64),
            "relative_error": np.asarray([rel_error], dtype=np.float64),
        }
    )
    for key in ("train_losses", "val_losses", "val_monitor_losses"):
        if key in result.train_metadata:
            fields[key] = np.asarray(result.train_metadata[key], dtype=np.float64)
    for key in ("best_val_loss", "best_epoch", "stopped_epoch", "stopped_early", "early_ema_alpha"):
        if key in result.train_metadata:
            fields[key] = np.asarray([result.train_metadata[key]])
    np.savez_compressed(result_npz, **fields)
    return result_npz, {
        "estimate": estimate,
        "true_skl": truth,
        "relative_error": rel_error,
        "best_epoch": int(result.train_metadata.get("best_epoch", -1)),
        "best_val_loss": float(result.train_metadata.get("best_val_loss", float("nan"))),
    }


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "n_per_condition",
        "repeat_idx",
        "seed",
        "model",
        "velocity_family",
        "training_mode",
        "estimate_skl",
        "true_skl",
        "abs_error",
        "relative_error",
        "best_epoch",
        "best_val_loss",
        "result_npz",
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_rows_csv(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _aggregate(rows: list[dict[str, Any]], *, n_list: list[int], specs: list[ModelSpec], n_seeds: int) -> dict[str, Any]:
    model_names = [s.name for s in specs]
    estimates = np.full((len(n_list), int(n_seeds), len(specs)), np.nan, dtype=np.float64)
    true_skl = np.full((len(n_list), int(n_seeds)), np.nan, dtype=np.float64)
    n_idx = {int(n): i for i, n in enumerate(n_list)}
    model_idx = {m: i for i, m in enumerate(model_names)}
    for row in rows:
        ni = n_idx[int(row["n_per_condition"])]
        si = int(row["repeat_idx"])
        mi = model_idx[str(row["model"])]
        estimates[ni, si, mi] = float(row["estimate_skl"])
        true_skl[ni, si] = float(row["true_skl"])
    return {
        "n_list": np.asarray(n_list, dtype=np.int64),
        "model_names": tuple(model_names),
        "estimates": estimates,
        "true_skl": true_skl,
    }


def _write_aggregate_npz(path: Path, aggregate: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        n_list=np.asarray(aggregate["n_list"], dtype=np.int64),
        model_names=np.asarray(aggregate["model_names"]),
        estimates=np.asarray(aggregate["estimates"], dtype=np.float64),
        true_skl=np.asarray(aggregate["true_skl"], dtype=np.float64),
    )
    return path


def _read_aggregate_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {
            "n_list": np.asarray(data["n_list"], dtype=np.int64),
            "model_names": tuple(str(v) for v in np.asarray(data["model_names"]).tolist()),
            "estimates": np.asarray(data["estimates"], dtype=np.float64),
            "true_skl": np.asarray(data["true_skl"], dtype=np.float64),
        }


def _filter_aggregate_n_list(aggregate: dict[str, Any], n_list: list[int]) -> dict[str, Any]:
    requested = [int(v) for v in n_list]
    current = [int(v) for v in np.asarray(aggregate["n_list"], dtype=np.int64).tolist()]
    idx = [current.index(n) for n in requested if n in current]
    if not idx:
        raise ValueError(f"None of requested n_list={requested} appears in aggregate n_list={current}.")
    kept = [current[i] for i in idx]
    return {
        "n_list": np.asarray(kept, dtype=np.int64),
        "model_names": tuple(str(v) for v in aggregate["model_names"]),
        "estimates": np.asarray(aggregate["estimates"], dtype=np.float64)[idx, :, :],
        "true_skl": np.asarray(aggregate["true_skl"], dtype=np.float64)[idx, :],
    }


def _filter_aggregate_models(aggregate: dict[str, Any], model_names: list[str]) -> dict[str, Any]:
    requested = [str(v) for v in model_names]
    current = [str(v) for v in aggregate["model_names"]]
    idx = [current.index(model) for model in requested if model in current]
    if not idx:
        raise ValueError(f"None of requested models={requested} appears in aggregate models={current}.")
    kept = [current[i] for i in idx]
    return {
        "n_list": np.asarray(aggregate["n_list"], dtype=np.int64),
        "model_names": tuple(kept),
        "estimates": np.asarray(aggregate["estimates"], dtype=np.float64)[:, :, idx],
        "true_skl": np.asarray(aggregate["true_skl"], dtype=np.float64),
    }


def _filter_rows_n_list(rows: list[dict[str, Any]], n_list: list[int]) -> list[dict[str, Any]]:
    keep = {int(v) for v in n_list}
    return [r for r in rows if int(r["n_per_condition"]) in keep]


def _filter_rows_models(rows: list[dict[str, Any]], model_names: list[str]) -> list[dict[str, Any]]:
    keep = {str(v) for v in model_names}
    return [r for r in rows if str(r["model"]) in keep]


def _sd(values: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    count = np.sum(np.isfinite(arr), axis=axis)
    mean = np.nanmean(arr, axis=axis, keepdims=True)
    sq = np.where(np.isfinite(arr), (arr - mean) ** 2, 0.0)
    var = np.sum(sq, axis=axis) / np.maximum(count - 1, 1)
    return np.where(count > 1, np.sqrt(var), 0.0)


def _median_iqr(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    centers = np.full(arr.shape[0], np.nan, dtype=np.float64)
    yerr = np.zeros((2, arr.shape[0]), dtype=np.float64)
    for i in range(arr.shape[0]):
        finite = arr[i, np.isfinite(arr[i])]
        if finite.size == 0:
            continue
        q25, q50, q75 = np.quantile(finite, [0.25, 0.5, 0.75])
        centers[i] = float(q50)
        yerr[0, i] = float(q50 - q25)
        yerr[1, i] = float(q75 - q50)
    return centers, yerr


def _plot_dataset_panel(
    ax: plt.Axes,
    *,
    amplitude: float,
    x_dim: int,
    seed: int,
    n_per_condition: int,
) -> None:
    dataset = generate_quadratic_toy_dataset(
        n_per_condition=int(n_per_condition),
        amplitude=float(amplitude),
        x_dim=int(x_dim),
        seed=int(seed),
        train_frac=0.8,
    )
    colors = {0: _MODEL_COLORS["affine"], 1: _MODEL_COLORS["quadratic"]}
    labels = {0: "condition 0", 1: "condition 1"}
    for condition in (0, 1):
        mask = dataset.labels == condition
        pts = dataset.x_all[mask][:100]
        ax.scatter(
            pts[:, 1],
            pts[:, 0],
            s=12,
            alpha=0.34,
            color=colors[condition],
            edgecolors="none",
            label=f"{labels[condition]} samples",
            rasterized=True,
        )

    amp = float(amplitude)
    s = math.sqrt(1.0 + 2.0 * amp * amp)
    y2 = np.linspace(-2.8, 2.8, 240)
    feature = y2**2 - 1.0
    ax.plot(
        y2,
        (-amp * feature) / s,
        color=colors[0],
        linewidth=1.5,
        linestyle=(0, (4, 2)),
        label=r"$\mathbb{E}[y_1\mid y_2,c=0]$",
    )
    ax.plot(
        y2,
        (amp * feature) / s,
        color=colors[1],
        linewidth=1.5,
        linestyle=(0, (4, 2)),
        label=r"$\mathbb{E}[y_1\mid y_2,c=1]$",
    )
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-4.2, 4.2)
    ax.set_box_aspect(1.0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Toy data geometry", pad=5)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, loc="upper right", handletextpad=0.3, borderaxespad=0.2)


def plot_distance(
    path_base: Path,
    aggregate: dict[str, Any],
    *,
    amplitude: float,
    x_dim: int,
    dataset_plot_seed: int,
    dataset_plot_n: int,
    plot_stat: str = "median_iqr",
    plot_ymin: float | None = -0.35,
    plot_ymax: float | None = 6.0,
) -> tuple[Path, Path]:
    n_list = np.asarray(aggregate["n_list"], dtype=np.int64)
    n_total = 2 * n_list
    estimates = np.asarray(aggregate["estimates"], dtype=np.float64)
    truth = float(np.nanmean(np.asarray(aggregate["true_skl"], dtype=np.float64)))
    model_names = tuple(str(v) for v in aggregate["model_names"])

    if str(plot_stat) not in {"median_iqr", "mean_sd"}:
        raise ValueError("plot_stat must be 'median_iqr' or 'mean_sd'.")

    with plt.rc_context(
        {
            "font.size": 15.0,
            "axes.titlesize": 15.0,
            "axes.labelsize": 15.0,
            "xtick.labelsize": 15.0,
            "ytick.labelsize": 15.0,
            "legend.fontsize": 15.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(8.4, 3.85),
            gridspec_kw={"width_ratios": [1.0, 1.55], "wspace": 0.26},
        )
        data_ax, ax = axes
        _plot_dataset_panel(
            data_ax,
            amplitude=float(amplitude),
            x_dim=int(x_dim),
            seed=int(dataset_plot_seed),
            n_per_condition=int(dataset_plot_n),
        )
        for model_i, model in enumerate(model_names):
            vals = estimates[:, :, model_i]
            if str(plot_stat) == "median_iqr":
                y, yerr = _median_iqr(vals)
                suffix = "median +/- IQR"
            else:
                y = np.nanmean(vals, axis=1)
                yerr = _sd(vals, axis=1)
                suffix = "mean +/- SD"
            color = _MODEL_COLORS.get(model, f"C{model_i}")
            display = _MODEL_DISPLAY_NAMES.get(model, model)
            label = f"{display} {suffix}"
            if vals.shape[1] > 1:
                for seed_i in range(vals.shape[1]):
                    finite = np.isfinite(vals[:, seed_i])
                    ax.scatter(
                        n_total[finite],
                        vals[finite, seed_i],
                        s=10,
                        alpha=0.18,
                        color=color,
                        edgecolors="none",
                        rasterized=True,
                    )
            finite_y = np.isfinite(y)
            ax.errorbar(
                n_total[finite_y],
                y[finite_y],
                yerr=yerr[:, finite_y] if np.ndim(yerr) == 2 else yerr[finite_y],
                marker="o",
                markersize=3.4,
                linewidth=1.35,
                elinewidth=0.9,
                capsize=2.3,
                capthick=0.8,
                color=color,
                label=label,
            )
        ax.axhline(truth, color="black", linestyle=(0, (4, 2)), linewidth=1.0, label=f"truth = {truth:.2f}")
        ax.set_xscale("log")
        ax.set_xticks(n_total)
        ax.set_xticklabels([str(int(v)) for v in n_total], rotation=30, ha="right")
        if plot_ymin is not None or plot_ymax is not None:
            ax.set_ylim(plot_ymin, plot_ymax)
        ax.set_xlabel("Total number of points")
        ax.set_ylabel("Estimated symmetric KL")
        ax.set_title("SKL estimate vs total sample size", pad=5)
        ax.grid(False)
        ax.legend(
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.88,
            loc="upper left",
            handlelength=1.6,
            borderaxespad=0.2,
            labelspacing=0.35,
        )
        ax.spines["left"].set_linewidth(1.8)
        ax.spines["bottom"].set_linewidth(1.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", width=1.8, length=4.5)
        for label, panel_ax in zip(("A", "B"), (data_ax, ax), strict=True):
            panel_ax.text(
                -0.13,
                1.04,
                label,
                transform=panel_ax.transAxes,
                fontsize=15,
                fontweight="bold",
                va="bottom",
                ha="left",
            )
        fig.subplots_adjust(left=0.065, right=0.995, bottom=0.26, top=0.88, wspace=0.30)
    svg = path_base.with_suffix(".svg")
    png = path_base.with_suffix(".png")
    fig.savefig(svg)
    fig.savefig(png, dpi=400)
    plt.close(fig)
    return svg, png


def _write_summary(path: Path, summary: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def _summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(int(r["n_per_condition"]), str(r["model"])) for r in rows})
    for n, model in keys:
        vals = [r for r in rows if int(r["n_per_condition"]) == n and str(r["model"]) == model]
        estimates = np.asarray([float(r["estimate_skl"]) for r in vals], dtype=np.float64)
        finite = estimates[np.isfinite(estimates)]
        truth = float(vals[0]["true_skl"])
        if finite.size > 0:
            q25, q50, q75 = np.quantile(finite, [0.25, 0.5, 0.75])
        else:
            q25 = q50 = q75 = float("nan")
        out.append(
            {
                "n_per_condition": int(n),
                "model": model,
                "true_skl": truth,
                "mean_estimate_skl": float(np.nanmean(estimates)),
                "sd_estimate_skl": float(np.nanstd(estimates, ddof=1)) if finite.size > 1 else 0.0,
                "median_estimate_skl": float(q50),
                "iqr25_estimate_skl": float(q25),
                "iqr75_estimate_skl": float(q75),
                "mean_abs_error": float(np.nanmean(np.abs(estimates - truth))),
                "median_abs_error": float(np.nanmedian(np.abs(estimates - truth))),
                "mean_relative_error": float(np.nanmean(np.abs(estimates - truth) / truth))
                if truth > 0.0
                else float("nan"),
                "median_relative_error": float(np.nanmedian(np.abs(estimates - truth) / truth))
                if truth > 0.0
                else float("nan"),
            }
        )
    return out


def main() -> None:
    args = build_parser().parse_args()
    if int(args.n_seeds) < 1:
        raise ValueError("--n-seeds must be >= 1.")
    if float(args.amplitude) <= 0.0:
        raise ValueError("--amplitude must be > 0.")
    x_dim = _validate_x_dim(int(args.x_dim))
    device = require_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_models = [str(v) for v in args.models]
    specs = _model_specs(selected_models)
    n_list = [int(v) for v in args.n_list]
    results_csv = output_dir / RESULTS_CSV_NAME
    results_npz = output_dir / RESULTS_NPZ_NAME
    summary_json = output_dir / SUMMARY_JSON_NAME

    if bool(args.plots_only):
        if not results_csv.is_file() or not results_npz.is_file():
            raise FileNotFoundError("--plots-only requires existing CSV and NPZ outputs.")
        rows = _filter_rows_models(_filter_rows_n_list(_read_rows_csv(results_csv), n_list), selected_models)
        aggregate = _filter_aggregate_models(_filter_aggregate_n_list(_read_aggregate_npz(results_npz), n_list), selected_models)
        existing_summary: dict[str, Any] = {}
        if summary_json.is_file():
            try:
                existing_summary = json.loads(summary_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing_summary = {}
        plot_x_dim = _validate_x_dim(int(existing_summary.get("x_dim", x_dim)))
        truth = float(np.nanmean(np.asarray(aggregate["true_skl"], dtype=np.float64)))
        plot_amplitude = math.sqrt(truth / (8.0 * (plot_x_dim // 2))) if truth >= 0.0 else float(args.amplitude)
        svg, png = plot_distance(
            output_dir / "quadratic_velocity_2d_toy_skl_vs_n",
            aggregate,
            amplitude=plot_amplitude,
            x_dim=plot_x_dim,
            dataset_plot_seed=int(args.seed),
            dataset_plot_n=int(args.dataset_plot_n),
            plot_stat=str(args.plot_stat),
            plot_ymin=float(args.plot_ymin) if args.plot_ymin is not None else None,
            plot_ymax=float(args.plot_ymax) if args.plot_ymax is not None else None,
        )
        summary = {
            **existing_summary,
            "script": "bin/run_quadratic_velocity_2d_toy_skl.py",
            "output_dir": str(output_dir),
            "device": str(args.device),
            "training_mode": rows[0].get("training_mode", "unknown") if rows else "unknown",
            "amplitude": plot_amplitude,
            "x_dim": int(plot_x_dim),
            "n_pairs": int(plot_x_dim // 2),
            "true_skl": truth,
            "n_list": [int(v) for v in np.asarray(aggregate["n_list"], dtype=np.int64).tolist()],
            "n_total_list": [int(2 * v) for v in np.asarray(aggregate["n_list"], dtype=np.int64).tolist()],
            "n_seeds": int(np.asarray(aggregate["estimates"]).shape[1]),
            "models": [str(v) for v in aggregate["model_names"]],
            "plot_stat": str(args.plot_stat),
            "plot_ymin": float(args.plot_ymin) if args.plot_ymin is not None else None,
            "plot_ymax": float(args.plot_ymax) if args.plot_ymax is not None else None,
            "dataset_plot_n": int(args.dataset_plot_n),
            "results_csv": str(results_csv),
            "results_npz": str(results_npz),
            "figure": [str(svg), str(png)],
            "rows_summary": _summarize_rows(rows),
        }
        print(f"summary_json: {_write_summary(summary_json, summary)}", flush=True)
        print(f"figure_png: {png}", flush=True)
        return

    rows: list[dict[str, Any]] = []
    for n_per_condition in n_list:
        for repeat_idx in range(int(args.n_seeds)):
            seed = _seed_for_repeat(int(args.seed), int(repeat_idx))
            dataset = generate_quadratic_toy_dataset(
                n_per_condition=int(n_per_condition),
                amplitude=float(args.amplitude),
                x_dim=x_dim,
                seed=seed,
                train_frac=float(args.train_frac),
            )
            case_dir = _case_dir(output_dir, n_per_condition=int(n_per_condition), seed=seed)
            case_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                case_dir / "dataset.npz",
                z_all=dataset.z_all,
                theta_all=dataset.theta_all,
                x_all=dataset.x_all,
                labels=dataset.labels,
                train_idx=dataset.train_idx,
                val_idx=dataset.val_idx,
                true_skl=np.asarray([dataset.true_skl], dtype=np.float64),
                amplitude=np.asarray([float(args.amplitude)], dtype=np.float64),
                x_dim=np.asarray([x_dim], dtype=np.int64),
            )
            for spec in specs:
                model_dir = _case_dir(
                    output_dir,
                    n_per_condition=int(n_per_condition),
                    seed=seed,
                    model_name=spec.name,
                )
                print(
                    f"[quadratic-toy] N={int(n_per_condition)} seed={seed} "
                    f"model={spec.name} family={spec.velocity_family}",
                    flush=True,
                )
                result_npz, vals = train_one_model(
                    dataset=dataset,
                    spec=spec,
                    args=args,
                    device=device,
                    seed=seed,
                    output_dir=model_dir,
                )
                estimate = float(vals["estimate"])
                truth = float(vals["true_skl"])
                rows.append(
                    {
                        "n_per_condition": int(n_per_condition),
                        "repeat_idx": int(repeat_idx),
                        "seed": int(seed),
                        "model": spec.name,
                        "velocity_family": spec.velocity_family,
                        "training_mode": str(args.training_mode),
                        "estimate_skl": estimate,
                        "true_skl": truth,
                        "abs_error": abs(estimate - truth),
                        "relative_error": float(vals["relative_error"]),
                        "best_epoch": int(vals["best_epoch"]),
                        "best_val_loss": float(vals["best_val_loss"]),
                        "result_npz": str(result_npz),
                    }
                )

    rows.sort(key=lambda r: (int(r["n_per_condition"]), int(r["repeat_idx"]), str(r["model"])))
    _write_rows_csv(results_csv, rows)
    aggregate = _aggregate(rows, n_list=n_list, specs=specs, n_seeds=int(args.n_seeds))
    _write_aggregate_npz(results_npz, aggregate)
    svg, png = plot_distance(
        output_dir / "quadratic_velocity_2d_toy_skl_vs_n",
        aggregate,
        amplitude=float(args.amplitude),
        x_dim=x_dim,
        dataset_plot_seed=int(args.seed),
        dataset_plot_n=int(args.dataset_plot_n),
        plot_stat=str(args.plot_stat),
        plot_ymin=float(args.plot_ymin) if args.plot_ymin is not None else None,
        plot_ymax=float(args.plot_ymax) if args.plot_ymax is not None else None,
    )
    summary = {
        "script": "bin/run_quadratic_velocity_2d_toy_skl.py",
        "output_dir": str(output_dir),
        "device": str(args.device),
        "amplitude": float(args.amplitude),
        "x_dim": int(x_dim),
        "n_pairs": int(x_dim // 2),
        "true_skl": true_quadratic_toy_skl(float(args.amplitude), x_dim=x_dim),
        "n_list": n_list,
        "n_total_list": [int(2 * v) for v in n_list],
        "n_seeds": int(args.n_seeds),
        "seed": int(args.seed),
        "train_frac": float(args.train_frac),
        "training_mode": str(args.training_mode),
        "models": [s.name for s in specs],
        "plot_stat": str(args.plot_stat),
        "plot_ymin": float(args.plot_ymin) if args.plot_ymin is not None else None,
        "plot_ymax": float(args.plot_ymax) if args.plot_ymax is not None else None,
        "dataset_plot_n": int(args.dataset_plot_n),
        "results_csv": str(results_csv),
        "results_npz": str(results_npz),
        "figure": [str(svg), str(png)],
        "rows_summary": _summarize_rows(rows),
    }
    print(f"results_npz: {results_npz}", flush=True)
    print(f"results_csv: {results_csv}", flush=True)
    print(f"summary_json: {_write_summary(summary_json, summary)}", flush=True)
    print(f"figure_png: {png}", flush=True)


if __name__ == "__main__":
    main()
