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
import matplotlib.ticker as mticker
import numpy as np
import torch

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
    theta_train: np.ndarray
    x_train: np.ndarray
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
    "tanh": ModelSpec(name="tanh", velocity_family="condition_tanh"),
    "neural": ModelSpec(name="neural", velocity_family="nonlinear"),
}

_MODEL_DISPLAY_NAMES = {
    "affine": "Affine",
    "quadratic": "Quadratic",
    "tanh": "Tanh",
    "neural": "Neural",
}

_MODEL_COLORS = {
    "affine": "#4C78A8",
    "quadratic": "#F58518",
    "tanh": "#B279A2",
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
        help="Comma-separated subset/order of velocity classes to run: affine,quadratic,tanh,neural.",
    )
    p.add_argument("--n-list", type=_parse_int_list, default=[4, 5, 8, 10, 16, 30, 50, 100])
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--x-dim", type=int, default=4, help="Even observed dimension; each adjacent pair is a quadratic shear.")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--amplitude", type=float, default=0.5)

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
    p.add_argument(
        "--plot-break-total-after",
        type=int,
        default=0,
        help=(
            "If positive and larger total sample sizes are present, break panel B's x-axis "
            "after this total number of points."
        ),
    )
    p.add_argument(
        "--plot-loss-panels",
        action="store_true",
        help=(
            "Also write a merged diagnostic figure with the SKL figure on top and "
            "train/EMA-validation loss curves below."
        ),
    )
    p.add_argument(
        "--loss-curve-max-points",
        type=int,
        default=450,
        help="Maximum plotted epoch points per loss curve after deterministic downsampling.",
    )
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
        theta_train=theta_all[train_idx],
        x_train=x_all[train_idx],
        theta_val=theta_all[val_idx],
        x_val=x_all[val_idx],
        true_skl=true_quadratic_toy_skl(amp, x_dim=xd),
    )


def _seed_for_repeat(base_seed: int, repeat_idx: int) -> int:
    return int(base_seed) + int(repeat_idx)


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
        x_dim=int(dataset.x_train.shape[1]),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        quadrature_steps=int(args.quadrature_steps),
        path_schedule=str(args.path_schedule),
        divergence_estimator=str(args.divergence_estimator),
        hutchinson_probes=int(args.hutchinson_probes),
    ).to(device)
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
            "training_mode": np.asarray(["cfm"]),
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
    plot_break_total_after: int = 0,
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
        break_after = int(plot_break_total_after)
        use_x_break = bool(
            break_after > 0 and np.any(n_total <= break_after) and np.any(n_total > break_after)
        )
        if use_x_break:
            fig = plt.figure(figsize=(10.8, 4.0))
            outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.18], wspace=0.27)
            data_ax = fig.add_subplot(outer[0, 0])
            inner = outer[0, 1].subgridspec(1, 2, width_ratios=[1.68, 0.58], wspace=0.055)
            ax_left = fig.add_subplot(inner[0, 0])
            ax_right = fig.add_subplot(inner[0, 1], sharey=ax_left)
            plot_axes = (ax_left, ax_right)
            plot_masks = (n_total <= break_after, n_total > break_after)
        else:
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(8.4, 3.85),
                gridspec_kw={"width_ratios": [1.0, 1.55], "wspace": 0.26},
            )
            data_ax, ax_left = axes
            ax_right = None
            plot_axes = (ax_left,)
            plot_masks = (np.ones_like(n_total, dtype=bool),)

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
            label = display
            for axis_i, (axis, mask) in enumerate(zip(plot_axes, plot_masks, strict=True)):
                if vals.shape[1] > 1:
                    for seed_i in range(vals.shape[1]):
                        finite = mask & np.isfinite(vals[:, seed_i])
                        if not np.any(finite):
                            continue
                        axis.scatter(
                            n_total[finite],
                            vals[finite, seed_i],
                            s=10,
                            alpha=0.18,
                            color=color,
                            edgecolors="none",
                            rasterized=True,
                        )
                finite_y = mask & np.isfinite(y)
                if not np.any(finite_y):
                    continue
                axis.errorbar(
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
                    label=label if axis_i == 0 else "_nolegend_",
                )

        for axis_i, (axis, mask) in enumerate(zip(plot_axes, plot_masks, strict=True)):
            axis.axhline(
                truth,
                color="black",
                linestyle=(0, (4, 2)),
                linewidth=1.0,
                label=f"truth = {truth:.2f}" if axis_i == 0 else "_nolegend_",
            )
            axis.set_xscale("log")
            axis.xaxis.set_minor_locator(mticker.NullLocator())
            axis.xaxis.set_minor_formatter(mticker.NullFormatter())
            ticks = n_total[mask]
            axis.set_xticks(ticks)
            tick_rotation = 0 if use_x_break and axis_i == 1 else 30
            tick_ha = "center" if tick_rotation == 0 else "right"
            axis.set_xticklabels([str(int(v)) for v in ticks], rotation=tick_rotation, ha=tick_ha)
            if ticks.size:
                lo = float(np.nanmin(ticks))
                hi = float(np.nanmax(ticks))
                if hi > lo:
                    axis.set_xlim(lo / 1.12, hi * 1.12)
                else:
                    axis.set_xlim(lo / 1.35, hi * 1.35)
            if plot_ymin is not None or plot_ymax is not None:
                axis.set_ylim(plot_ymin, plot_ymax)
            axis.grid(False)
            axis.spines["bottom"].set_linewidth(1.8)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(axis="both", width=1.8, length=4.5)

        ax_left.set_xlabel("Total number of points")
        ax_left.set_ylabel("Estimated symmetric KL")
        ax_left.set_title("SKL estimate vs total sample size", pad=5)
        ax_left.spines["left"].set_linewidth(1.8)
        if use_x_break and ax_right is not None:
            ax_right.spines["left"].set_visible(False)
            ax_right.tick_params(axis="y", left=False, labelleft=False)
            d = 0.014
            kwargs = dict(color="black", clip_on=False, linewidth=1.2)
            ax_left.plot((1 - d, 1 + d), (-d, +d), transform=ax_left.transAxes, **kwargs)
            ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_left.transAxes, **kwargs)
            ax_right.plot((-d, +d), (-d, +d), transform=ax_right.transAxes, **kwargs)
            ax_right.plot((-d, +d), (1 - d, 1 + d), transform=ax_right.transAxes, **kwargs)

        ax_left.legend(
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.88,
            loc="upper left",
            handlelength=1.6,
            borderaxespad=0.2,
            labelspacing=0.35,
        )
        for label, panel_ax in zip(("A", "B"), (data_ax, ax_left), strict=True):
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
        fig.subplots_adjust(left=0.065, right=0.995, bottom=0.26, top=0.88)
    svg = path_base.with_suffix(".svg")
    png = path_base.with_suffix(".png")
    fig.savefig(svg)
    fig.savefig(png, dpi=400)
    plt.close(fig)
    return svg, png


def _median_loss_curve(curves: list[np.ndarray]) -> np.ndarray:
    cleaned = [
        np.asarray(curve, dtype=np.float64).reshape(-1)
        for curve in curves
        if np.asarray(curve, dtype=np.float64).size > 0
    ]
    if not cleaned:
        return np.empty(0, dtype=np.float64)
    max_len = max(int(curve.size) for curve in cleaned)
    padded = np.full((len(cleaned), max_len), np.nan, dtype=np.float64)
    for i, curve in enumerate(cleaned):
        finite_curve = np.where(np.isfinite(curve) & (curve > 0.0), curve, np.nan)
        padded[i, : finite_curve.size] = finite_curve
    med = np.full(max_len, np.nan, dtype=np.float64)
    for j in range(max_len):
        finite = padded[:, j][np.isfinite(padded[:, j])]
        if finite.size:
            med[j] = float(np.median(finite))
    return med


def _downsample_curve(y: np.ndarray, *, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(arr) & (arr > 0.0)
    if not np.any(finite):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    x = np.arange(1, arr.size + 1, dtype=np.int64)[finite]
    vals = arr[finite]
    maxp = max(2, int(max_points))
    if vals.size > maxp:
        idx = np.unique(np.linspace(0, vals.size - 1, maxp).astype(np.int64))
        x = x[idx]
        vals = vals[idx]
    return x, vals


def _loss_curves_for_rows(rows: list[dict[str, Any]], *, n_per_condition: int, model: str) -> tuple[np.ndarray, np.ndarray]:
    train_curves: list[np.ndarray] = []
    val_ema_curves: list[np.ndarray] = []
    matching = [
        row
        for row in rows
        if int(row["n_per_condition"]) == int(n_per_condition) and str(row["model"]) == str(model)
    ]
    matching.sort(key=lambda row: int(row["repeat_idx"]))
    for row in matching:
        result_path = Path(str(row.get("result_npz", ""))).expanduser()
        if not result_path.is_file():
            continue
        with np.load(result_path, allow_pickle=False) as data:
            if "train_losses" not in data.files or "val_monitor_losses" not in data.files:
                continue
            train_curves.append(np.asarray(data["train_losses"], dtype=np.float64))
            val_ema_curves.append(np.asarray(data["val_monitor_losses"], dtype=np.float64))
    return _median_loss_curve(train_curves), _median_loss_curve(val_ema_curves)


def plot_distance_with_loss_panels(
    path_base: Path,
    *,
    distance_png: Path,
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    loss_curve_max_points: int = 450,
) -> tuple[Path, Path]:
    n_list = np.asarray(aggregate["n_list"], dtype=np.int64)
    n_total = 2 * n_list
    model_names = tuple(str(v) for v in aggregate["model_names"])
    if not Path(distance_png).is_file():
        raise FileNotFoundError(f"distance_png does not exist: {distance_png}")

    loss_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    row_limits: dict[str, tuple[float, float]] = {}
    for model in model_names:
        row_values: list[np.ndarray] = []
        for n in n_list:
            train_med, val_ema_med = _loss_curves_for_rows(rows, n_per_condition=int(n), model=model)
            loss_cache[(model, int(n))] = (train_med, val_ema_med)
            for curve in (train_med, val_ema_med):
                finite = curve[np.isfinite(curve) & (curve > 0.0)]
                if finite.size:
                    row_values.append(finite)
        if row_values:
            vals = np.concatenate(row_values)
            ymin = max(float(np.nanmin(vals)) / 1.45, 1e-8)
            ymax = float(np.nanmax(vals)) * 1.45
            if not math.isfinite(ymin) or not math.isfinite(ymax) or ymin >= ymax:
                ymin, ymax = 1e-5, 1.0
        else:
            ymin, ymax = 1e-5, 1.0
        row_limits[model] = (ymin, ymax)

    with plt.rc_context(
        {
            "font.size": 9.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 10.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        fig = plt.figure(figsize=(22.0, 15.0))
        outer = fig.add_gridspec(2, 1, height_ratios=[1.02, 1.0], hspace=0.18)
        top_ax = fig.add_subplot(outer[0, 0])
        top_img = plt.imread(str(distance_png))
        top_ax.imshow(top_img)
        top_ax.set_axis_off()

        loss_grid = outer[1, 0].subgridspec(
            len(model_names),
            len(n_list),
            wspace=0.13,
            hspace=0.24,
        )
        axes: list[plt.Axes] = []
        for model_i, model in enumerate(model_names):
            color = _MODEL_COLORS.get(model, f"C{model_i}")
            display = _MODEL_DISPLAY_NAMES.get(model, model)
            ymin, ymax = row_limits[model]
            for n_i, n in enumerate(n_list):
                ax = fig.add_subplot(loss_grid[model_i, n_i])
                axes.append(ax)
                train_med, val_ema_med = loss_cache[(model, int(n))]
                x_train, y_train = _downsample_curve(train_med, max_points=int(loss_curve_max_points))
                x_val, y_val = _downsample_curve(val_ema_med, max_points=int(loss_curve_max_points))
                if y_train.size:
                    ax.plot(x_train, y_train, color="0.25", linewidth=0.85, label="train")
                if y_val.size:
                    ax.plot(
                        x_val,
                        y_val,
                        color=color,
                        linestyle=(0, (4, 2)),
                        linewidth=0.95,
                        label="val EMA",
                    )
                if not y_train.size and not y_val.size:
                    ax.text(
                        0.5,
                        0.5,
                        "not run",
                        transform=ax.transAxes,
                        color="0.45",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
                ax.set_yscale("log")
                ax.set_ylim(ymin, ymax)
                ax.grid(False)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
                ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
                ax.yaxis.set_minor_locator(mticker.NullLocator())
                if model_i == 0:
                    ax.set_title(f"T={int(n_total[n_i])}\nN={int(n)}", pad=3)
                if n_i == 0:
                    ax.set_ylabel(f"{display}\nloss")
                else:
                    ax.tick_params(axis="y", labelleft=False)
                if model_i == len(model_names) - 1:
                    ax.set_xlabel("epoch")
                else:
                    ax.tick_params(axis="x", labelbottom=False)

        legend_handles = [
            plt.Line2D([0], [0], color="0.25", linewidth=1.2, label="training loss"),
            plt.Line2D([0], [0], color="black", linestyle=(0, (4, 2)), linewidth=1.2, label="EMA validation loss"),
        ]
        fig.legend(
            handles=legend_handles,
            frameon=False,
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.012),
        )
        fig.text(
            0.022,
            0.512,
            "C",
            fontsize=22,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        fig.text(
            0.5,
            0.531,
            "Loss curves by velocity class and sample size (median over repeats)",
            fontsize=13,
            ha="center",
            va="bottom",
        )
        fig.subplots_adjust(left=0.035, right=0.995, bottom=0.065, top=0.995)

    svg = path_base.with_suffix(".svg")
    png = path_base.with_suffix(".png")
    fig.savefig(svg)
    fig.savefig(png, dpi=300)
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
            plot_break_total_after=int(args.plot_break_total_after),
        )
        loss_figure: list[str] | None = None
        if bool(args.plot_loss_panels):
            loss_svg, loss_png = plot_distance_with_loss_panels(
                output_dir / "quadratic_velocity_2d_toy_skl_vs_n_with_losses",
                distance_png=png,
                rows=rows,
                aggregate=aggregate,
                loss_curve_max_points=int(args.loss_curve_max_points),
            )
            loss_figure = [str(loss_svg), str(loss_png)]
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
            "plot_break_total_after": int(args.plot_break_total_after),
            "plot_loss_panels": bool(args.plot_loss_panels),
            "loss_curve_max_points": int(args.loss_curve_max_points),
            "dataset_plot_n": int(args.dataset_plot_n),
            "results_csv": str(results_csv),
            "results_npz": str(results_npz),
            "figure": [str(svg), str(png)],
            "loss_figure": loss_figure,
            "rows_summary": _summarize_rows(rows),
        }
        print(f"summary_json: {_write_summary(summary_json, summary)}", flush=True)
        print(f"figure_png: {png}", flush=True)
        if loss_figure is not None:
            print(f"loss_figure_png: {loss_figure[1]}", flush=True)
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
                        "training_mode": "cfm",
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
        plot_break_total_after=int(args.plot_break_total_after),
    )
    loss_figure = None
    if bool(args.plot_loss_panels):
        loss_svg, loss_png = plot_distance_with_loss_panels(
            output_dir / "quadratic_velocity_2d_toy_skl_vs_n_with_losses",
            distance_png=png,
            rows=rows,
            aggregate=aggregate,
            loss_curve_max_points=int(args.loss_curve_max_points),
        )
        loss_figure = [str(loss_svg), str(loss_png)]
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
        "training_mode": "cfm",
        "models": [s.name for s in specs],
        "plot_stat": str(args.plot_stat),
        "plot_ymin": float(args.plot_ymin) if args.plot_ymin is not None else None,
        "plot_ymax": float(args.plot_ymax) if args.plot_ymax is not None else None,
        "plot_break_total_after": int(args.plot_break_total_after),
        "plot_loss_panels": bool(args.plot_loss_panels),
        "loss_curve_max_points": int(args.loss_curve_max_points),
        "dataset_plot_n": int(args.dataset_plot_n),
        "results_csv": str(results_csv),
        "results_npz": str(results_npz),
        "figure": [str(svg), str(png)],
        "loss_figure": loss_figure,
        "rows_summary": _summarize_rows(rows),
    }
    print(f"results_npz: {results_npz}", flush=True)
    print(f"results_csv: {results_csv}", flush=True)
    print(f"summary_json: {_write_summary(summary_json, summary)}", flush=True)
    print(f"figure_png: {png}", flush=True)
    if loss_figure is not None:
        print(f"loss_figure_png: {loss_figure[1]}", flush=True)


if __name__ == "__main__":
    main()
