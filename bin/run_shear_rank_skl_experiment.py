#!/usr/bin/env python3
"""Run the two-condition hidden-shear rank SKL experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from fisher.shear_rank_dataset import (
    centered_cosine_feature,
    generate_shear_rank_dataset,
    save_shear_rank_dataset_npz,
)


RESULTS_NPZ_NAME = "shear_rank_skl_results.npz"
RESULTS_CSV_NAME = "shear_rank_skl_errors.csv"
SUMMARY_JSON_NAME = "shear_rank_skl_summary.json"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    velocity_family: str
    rank: int


def _parse_int_list(value: str) -> list[int]:
    vals = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer.")
    return vals


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=Path, default=Path(DATA_DIR) / "two_condition_shear_rank_skl")
    p.add_argument("--force", action="store_true", help="Rerun model cases even when cached NPZs exist.")
    p.add_argument(
        "--skip-null",
        action="store_true",
        help="Skip the null a0=a1 dataset branch; by default both sign_flip and null modes run.",
    )
    p.add_argument(
        "--parallel-devices",
        type=str,
        default="",
        help=(
            "Comma-separated devices for process-level case parallelism, e.g. '0,1' or 'cuda:0,cuda:1'. "
            "Empty disables parallelism. Each listed device gets one worker."
        ),
    )

    p.add_argument("--n-list", type=_parse_int_list, default=[50, 100, 500, 1000, 2000])
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--x-dim", type=int, default=4)
    p.add_argument("--r-star", type=int, default=2)
    p.add_argument("--amplitude", type=float, default=0.7)
    p.add_argument("--omega", type=float, default=2.5)
    p.add_argument("--q-seed", type=int, default=12345)

    p.add_argument("--ranks", type=_parse_int_list, default=[0, 2, 4])
    p.add_argument("--no-full", action="store_true", help="Do not train the full nonlinear MLP model.")

    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--early-patience", type=int, default=1000)
    p.add_argument("--early-min-delta", type=float, default=1e-4)
    p.add_argument("--early-ema-alpha", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--path-schedule", choices=("cosine", "linear", "straight"), default="cosine")
    p.add_argument("--t-eps", type=float, default=0.0005)
    p.add_argument("--quadrature-steps", type=int, default=64)
    p.add_argument("--divergence-estimator", choices=("hutchinson", "exact"), default="hutchinson")
    p.add_argument("--hutchinson-probes", type=int, default=1)
    p.add_argument("--shared-affine-a-diag-jitter", type=float, default=1e-3)
    p.add_argument("--mc-jeffreys-sample", type=int, default=4096)
    p.add_argument("--ode-steps", type=int, default=64)
    p.add_argument("--ode-method", type=str, default="midpoint")
    p.add_argument("--solve-jitter", type=float, default=1e-6)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--fixed-n", type=int, default=1000, help="N value for tradeoff and bias/variance figures.")
    return p


def model_specs(ranks: list[int], *, include_full: bool) -> list[ModelSpec]:
    out: list[ModelSpec] = []
    seen: set[str] = set()
    for rank in ranks:
        r = int(rank)
        if r == 0:
            spec = ModelSpec(name="affine", velocity_family="condition_affine", rank=0)
        elif r > 0:
            spec = ModelSpec(name=f"rank_{r}", velocity_family="shared_affine_low_rank", rank=r)
        else:
            raise ValueError("Ranks must be nonnegative.")
        if spec.name not in seen:
            out.append(spec)
            seen.add(spec.name)
    if include_full:
        out.append(ModelSpec(name="full", velocity_family="nonlinear", rank=-1))
    return out


def _seed_for_repeat(base_seed: int, repeat_idx: int) -> int:
    return int(base_seed) + int(repeat_idx)


def _resolve_parallel_devices(value: str) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    devices: list[str] = []
    for raw in text.split(","):
        token = raw.strip()
        if not token:
            continue
        devices.append(f"cuda:{token}" if token.isdigit() else token)
    if not devices:
        raise ValueError("--parallel-devices must contain at least one non-empty device token.")
    if len(set(devices)) != len(devices):
        raise ValueError("--parallel-devices must not contain duplicate devices.")
    return devices


def _validate_device_name(device_name: str) -> torch.device:
    dev = torch.device(str(device_name))
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable. Per repo policy, do not fallback silently.")
        if dev.index is not None and int(dev.index) >= torch.cuda.device_count():
            raise ValueError(
                f"Requested {device_name!r}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
    return dev


def _case_dir(output_dir: Path, *, mode: str, n_per_condition: int, seed: int, model_name: str | None = None) -> Path:
    base = Path(output_dir) / str(mode) / f"N_{int(n_per_condition)}" / f"seed_{int(seed)}"
    if model_name is None:
        return base
    return base / str(model_name)


def _load_case_result(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "estimate": float(np.asarray(data["estimate_skl"]).reshape(-1)[0]),
            "true_skl": float(np.asarray(data["true_skl"]).reshape(-1)[0]),
            "rel_error": float(np.asarray(data["relative_error"]).reshape(-1)[0]),
            "best_epoch": int(np.asarray(data["best_epoch"]).reshape(-1)[0]) if "best_epoch" in data.files else -1,
            "best_val_loss": float(np.asarray(data["best_val_loss"]).reshape(-1)[0])
            if "best_val_loss" in data.files
            else float("nan"),
        }


def train_one_model(
    *,
    dataset: Any,
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

    bundle = dataset.bundle
    low_rank_basis = None
    low_rank_dim = max(1, int(spec.rank))
    if int(spec.rank) > 0:
        low_rank_basis = np.asarray(dataset.q_matrix[:, : int(spec.rank)], dtype=np.float64)
        low_rank_dim = int(spec.rank)

    model = build_flow_skl_model(
        velocity_family=spec.velocity_family,
        theta_dim=2,
        x_dim=int(args.x_dim),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        low_rank_dim=low_rank_dim,
        quadrature_steps=int(args.quadrature_steps),
        path_schedule=str(args.path_schedule),
        divergence_estimator=str(args.divergence_estimator),
        hutchinson_probes=int(args.hutchinson_probes),
        shared_affine_a_diag_jitter=float(args.shared_affine_a_diag_jitter),
        low_rank_basis=low_rank_basis,
    ).to(device)
    train_meta = train_flow_skl_model(
        model=model,
        theta_train=bundle.theta_train,
        x_train=bundle.x_train,
        theta_val=bundle.theta_validation,
        x_val=bundle.x_validation,
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
    true_skl = float(dataset.true_skl_matrix[0, 1])
    rel_error = abs(estimate - true_skl) / true_skl if true_skl > 0.0 else float("nan")

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = flow_skl_result_to_npz_dict(result)
    for text_key in ("canonical_metric_name", "network_architecture"):
        if text_key in fields:
            text_val = str(np.asarray(fields[text_key]).reshape(-1)[0])
            fields[text_key] = np.asarray([text_val])
    fields.update(
        {
            "model_name": np.asarray([spec.name]),
            "velocity_family": np.asarray([spec.velocity_family]),
            "rank": np.asarray([int(spec.rank)], dtype=np.int64),
            "theta_eval": theta_eval,
            "estimate_skl": np.asarray([estimate], dtype=np.float64),
            "true_skl": np.asarray([true_skl], dtype=np.float64),
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
        "true_skl": true_skl,
        "rel_error": rel_error,
        "best_epoch": int(result.train_metadata.get("best_epoch", -1)),
        "best_val_loss": float(result.train_metadata.get("best_val_loss", float("nan"))),
    }


def _run_dataset_case(
    *,
    args: argparse.Namespace,
    specs: list[ModelSpec],
    mode: str,
    n_per_condition: int,
    repeat_idx: int,
    device_name: str,
) -> list[dict[str, Any]]:
    dev = _validate_device_name(str(device_name))
    output_dir = Path(args.output_dir).expanduser().resolve()
    seed = _seed_for_repeat(int(args.seed), int(repeat_idx))
    dataset = generate_shear_rank_dataset(
        n_per_condition=int(n_per_condition),
        x_dim=int(args.x_dim),
        r_star=int(args.r_star),
        amplitude=float(args.amplitude),
        omega=float(args.omega),
        seed=int(seed),
        q_seed=int(args.q_seed),
        train_frac=float(args.train_frac),
        mode=str(mode),
    )
    case_dir = _case_dir(output_dir, mode=mode, n_per_condition=int(n_per_condition), seed=seed)
    dataset_npz = case_dir / "dataset.npz"
    if bool(args.force) or not dataset_npz.is_file():
        save_shear_rank_dataset_npz(dataset_npz, dataset)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        model_dir = _case_dir(
            output_dir,
            mode=mode,
            n_per_condition=int(n_per_condition),
            seed=seed,
            model_name=spec.name,
        )
        print(
            f"[shear-rank] device={dev} mode={mode} N={int(n_per_condition)} seed={seed} "
            f"model={spec.name}",
            flush=True,
        )
        result_npz, values = train_one_model(
            dataset=dataset,
            spec=spec,
            args=args,
            device=dev,
            seed=seed,
            output_dir=model_dir,
        )
        estimate = float(values["estimate"])
        truth = float(values["true_skl"])
        rows.append(
            {
                "mode": str(mode),
                "n_per_condition": int(n_per_condition),
                "repeat_idx": int(repeat_idx),
                "seed": int(seed),
                "model": spec.name,
                "velocity_family": spec.velocity_family,
                "rank": int(spec.rank),
                "estimate_skl": estimate,
                "true_skl": truth,
                "abs_error": abs(estimate - truth),
                "relative_error": float(values["rel_error"]),
                "best_epoch": int(values["best_epoch"]),
                "best_val_loss": float(values["best_val_loss"]),
                "result_npz": str(result_npz),
            }
        )
    return rows


def _parallel_worker(
    task_queue: Any,
    result_queue: Any,
    args: argparse.Namespace,
    specs: list[ModelSpec],
    device_name: str,
) -> None:
    while True:
        task = task_queue.get()
        if task is None:
            return
        mode, n_per_condition, repeat_idx = task
        try:
            rows = _run_dataset_case(
                args=args,
                specs=specs,
                mode=str(mode),
                n_per_condition=int(n_per_condition),
                repeat_idx=int(repeat_idx),
                device_name=str(device_name),
            )
            result_queue.put(("rows", rows))
        except BaseException:
            result_queue.put(("error", str(device_name), traceback.format_exc()))
            return


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["mode"]),
        int(row["n_per_condition"]),
        int(row["repeat_idx"]),
        int(row["rank"]),
        str(row["model"]),
    )


def _run_cases_parallel(
    *,
    args: argparse.Namespace,
    specs: list[ModelSpec],
    modes: list[str],
    devices: list[str],
) -> list[dict[str, Any]]:
    tasks = [
        (mode, int(n_per_condition), int(repeat_idx))
        for mode in modes
        for n_per_condition in [int(v) for v in args.n_list]
        for repeat_idx in range(int(args.n_seeds))
    ]
    if not tasks:
        return []
    for device_name in devices:
        _validate_device_name(device_name)

    ctx = mp.get_context("spawn")
    task_queues = [ctx.Queue() for _ in devices]
    result_queue = ctx.Queue()
    workers = [
        ctx.Process(
            target=_parallel_worker,
            args=(task_queues[i], result_queue, args, specs, devices[i]),
            daemon=False,
        )
        for i in range(len(devices))
    ]
    for worker in workers:
        worker.start()
    for task_idx, task in enumerate(tasks):
        task_queues[task_idx % len(devices)].put(task)
    for queue in task_queues:
        queue.put(None)

    rows: list[dict[str, Any]] = []
    n_done = 0
    try:
        while n_done < len(tasks):
            message = result_queue.get()
            if message[0] == "rows":
                rows.extend(message[1])
                n_done += 1
                continue
            if message[0] == "error":
                _, device_name, tb = message
                raise RuntimeError(f"Parallel worker on {device_name} failed:\n{tb}")
            raise RuntimeError(f"Unexpected parallel worker message: {message!r}")
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join()
    return rows


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "mode",
        "n_per_condition",
        "repeat_idx",
        "seed",
        "model",
        "velocity_family",
        "rank",
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


def _aggregate(rows: list[dict[str, Any]], *, modes: list[str], n_list: list[int], specs: list[ModelSpec], n_seeds: int) -> dict[str, Any]:
    model_names = [s.name for s in specs]
    estimates = np.full((len(modes), len(n_list), int(n_seeds), len(specs)), np.nan, dtype=np.float64)
    true_skl = np.full((len(modes), len(n_list), int(n_seeds)), np.nan, dtype=np.float64)
    rel_errors = np.full_like(estimates, np.nan)
    mode_idx = {m: i for i, m in enumerate(modes)}
    n_idx = {int(n): i for i, n in enumerate(n_list)}
    model_idx = {m: i for i, m in enumerate(model_names)}
    for row in rows:
        mi = mode_idx[str(row["mode"])]
        ni = n_idx[int(row["n_per_condition"])]
        si = int(row["repeat_idx"])
        ji = model_idx[str(row["model"])]
        estimates[mi, ni, si, ji] = float(row["estimate_skl"])
        true_skl[mi, ni, si] = float(row["true_skl"])
        rel_errors[mi, ni, si, ji] = float(row["relative_error"])
    return {
        "modes": tuple(modes),
        "n_list": np.asarray(n_list, dtype=np.int64),
        "model_names": tuple(model_names),
        "model_ranks": np.asarray([s.rank for s in specs], dtype=np.int64),
        "estimates": estimates,
        "true_skl": true_skl,
        "relative_errors": rel_errors,
    }


def _write_aggregate_npz(path: Path, aggregate: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        modes=np.asarray(aggregate["modes"]),
        n_list=np.asarray(aggregate["n_list"], dtype=np.int64),
        model_names=np.asarray(aggregate["model_names"]),
        model_ranks=np.asarray(aggregate["model_ranks"], dtype=np.int64),
        estimates=np.asarray(aggregate["estimates"], dtype=np.float64),
        true_skl=np.asarray(aggregate["true_skl"], dtype=np.float64),
        relative_errors=np.asarray(aggregate["relative_errors"], dtype=np.float64),
    )
    return path


def _sem(arr: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    count = np.sum(np.isfinite(values), axis=axis)
    moved = np.moveaxis(values, axis, 0)
    out_shape = moved.shape[1:]
    flat = moved.reshape(moved.shape[0], -1)
    out = np.zeros(flat.shape[1], dtype=np.float64)
    flat_count = np.asarray(count).reshape(-1)
    valid = flat_count > 1
    if np.any(valid):
        out[valid] = np.nanstd(flat[:, valid], axis=0, ddof=1) / np.sqrt(flat_count[valid])
    return out.reshape(out_shape)


def _mode_position(aggregate: dict[str, Any], mode: str) -> int | None:
    modes = tuple(str(v) for v in aggregate["modes"])
    return modes.index(mode) if mode in modes else None


def _fixed_n_position(n_list: np.ndarray, fixed_n: int) -> int:
    n_arr = np.asarray(n_list, dtype=np.int64)
    matches = np.flatnonzero(n_arr == int(fixed_n))
    if matches.size:
        return int(matches[0])
    return int(np.argmin(np.abs(n_arr - int(fixed_n))))


def plot_dataset_geometry(path_base: Path, *, args: argparse.Namespace) -> tuple[Path, Path]:
    n_plot = min(max(max(int(n) for n in args.n_list), 200), 1000)
    dataset = generate_shear_rank_dataset(
        n_per_condition=n_plot,
        x_dim=int(args.x_dim),
        r_star=int(args.r_star),
        amplitude=float(args.amplitude),
        omega=float(args.omega),
        seed=int(args.seed),
        q_seed=int(args.q_seed),
        train_frac=float(args.train_frac),
        mode="sign_flip",
    )
    hidden = np.asarray(dataset.bundle.x_all, dtype=np.float64) @ np.asarray(dataset.q_matrix, dtype=np.float64)
    labels = np.argmax(np.asarray(dataset.bundle.theta_all, dtype=np.float64), axis=1)
    x_grid = np.linspace(-3.0, 3.0, 400)
    nu = float(dataset.nu)
    amp = float(args.amplitude)
    scale = math.sqrt(1.0 + amp * amp * nu)
    curve = amp * centered_cosine_feature(x_grid, omega=float(args.omega)) / scale

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for condition, color, label in ((0, "C0", "condition 0"), (1, "C1", "condition 1")):
        idx = np.flatnonzero(labels == condition)
        if idx.size > 500:
            idx = idx[:500]
        ax.scatter(hidden[idx, 1], hidden[idx, 0], s=8, alpha=0.35, color=color, label=label)
    ax.plot(x_grid, -curve, color="C0", linewidth=2.0)
    ax.plot(x_grid, curve, color="C1", linewidth=2.0)
    ax.set_xlabel("hidden y2")
    ax.set_ylabel("hidden y1")
    ax.set_title("Native nonlinear shear pair")
    ax.legend(frameon=False)
    fig.tight_layout()
    svg = path_base.with_suffix(".svg")
    png = path_base.with_suffix(".png")
    fig.savefig(svg)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    return svg, png


def plot_distance_vs_n(path_base: Path, aggregate: dict[str, Any]) -> tuple[Path, Path] | None:
    mode_i = _mode_position(aggregate, "sign_flip")
    if mode_i is None:
        return None
    n_list = np.asarray(aggregate["n_list"], dtype=np.int64)
    estimates = np.asarray(aggregate["estimates"], dtype=np.float64)[mode_i]
    truth = float(np.nanmean(np.asarray(aggregate["true_skl"], dtype=np.float64)[mode_i]))
    model_names = tuple(str(v) for v in aggregate["model_names"])

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for model_i, model in enumerate(model_names):
        vals = estimates[:, :, model_i]
        ax.errorbar(n_list, np.nanmean(vals, axis=1), yerr=_sem(vals, axis=1), marker="o", linewidth=1.5, label=model)
    ax.axhline(truth, color="black", linestyle="--", linewidth=1.5, label="truth")
    ax.set_xscale("log")
    ax.set_xlabel("N per condition")
    ax.set_ylabel("estimated Jeffreys SKL")
    ax.set_title("Estimated distance vs sample size")
    ax.legend(frameon=False, fontsize=8, ncols=2)
    fig.tight_layout()
    svg = path_base.with_suffix(".svg")
    png = path_base.with_suffix(".png")
    fig.savefig(svg)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    return svg, png


def plot_tradeoff(path_base: Path, aggregate: dict[str, Any], *, fixed_n: int) -> tuple[Path, Path] | None:
    mode_i = _mode_position(aggregate, "sign_flip")
    if mode_i is None:
        return None
    n_pos = _fixed_n_position(np.asarray(aggregate["n_list"], dtype=np.int64), int(fixed_n))
    vals = np.asarray(aggregate["relative_errors"], dtype=np.float64)[mode_i, n_pos]
    labels = tuple(str(v) for v in aggregate["model_names"])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.errorbar(x, np.nanmean(vals, axis=0), yerr=_sem(vals, axis=0), marker="o", linewidth=1.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("relative error")
    ax.set_title(f"Complexity tradeoff at N={int(np.asarray(aggregate['n_list'])[n_pos])}")
    fig.tight_layout()
    svg = path_base.with_suffix(".svg")
    png = path_base.with_suffix(".png")
    fig.savefig(svg)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    return svg, png


def plot_bias_variance(path_base: Path, aggregate: dict[str, Any], *, fixed_n: int) -> tuple[Path, Path] | None:
    mode_i = _mode_position(aggregate, "sign_flip")
    if mode_i is None:
        return None
    n_pos = _fixed_n_position(np.asarray(aggregate["n_list"], dtype=np.int64), int(fixed_n))
    estimates = np.asarray(aggregate["estimates"], dtype=np.float64)[mode_i, n_pos]
    truth = float(np.nanmean(np.asarray(aggregate["true_skl"], dtype=np.float64)[mode_i, n_pos]))
    bias2 = (np.nanmean(estimates, axis=0) - truth) ** 2
    var = np.zeros(estimates.shape[1], dtype=np.float64)
    counts = np.sum(np.isfinite(estimates), axis=0)
    valid = counts > 1
    if np.any(valid):
        var[valid] = np.nanvar(estimates[:, valid], axis=0, ddof=1)
    labels = tuple(str(v) for v in aggregate["model_names"])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.bar(x, bias2, label="bias^2")
    ax.bar(x, var, bottom=bias2, label="variance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("MSE components")
    ax.set_title(f"Bias-variance at N={int(np.asarray(aggregate['n_list'])[n_pos])}")
    ax.legend(frameon=False)
    fig.tight_layout()
    svg = path_base.with_suffix(".svg")
    png = path_base.with_suffix(".png")
    fig.savefig(svg)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    return svg, png


def plot_null_false_positive(path_base: Path, aggregate: dict[str, Any]) -> tuple[Path, Path] | None:
    mode_i = _mode_position(aggregate, "null")
    if mode_i is None:
        return None
    n_list = np.asarray(aggregate["n_list"], dtype=np.int64)
    estimates = np.asarray(aggregate["estimates"], dtype=np.float64)[mode_i]
    model_names = tuple(str(v) for v in aggregate["model_names"])
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for model_i, model in enumerate(model_names):
        vals = estimates[:, :, model_i]
        ax.errorbar(n_list, np.nanmean(vals, axis=1), yerr=_sem(vals, axis=1), marker="o", linewidth=1.5, label=model)
    ax.set_xscale("log")
    ax.set_xlabel("N per condition")
    ax.set_ylabel("estimated Jeffreys SKL")
    ax.set_title("Null false-positive distance")
    ax.legend(frameon=False, fontsize=8, ncols=2)
    fig.tight_layout()
    svg = path_base.with_suffix(".svg")
    png = path_base.with_suffix(".png")
    fig.savefig(svg)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    return svg, png


def _write_summary(path: Path, summary: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def run(args: argparse.Namespace) -> dict[str, Path]:
    if int(args.n_seeds) < 1:
        raise ValueError("--n-seeds must be >= 1.")
    if int(args.x_dim) < int(args.r_star):
        raise ValueError("--x-dim must be >= --r-star.")
    dev = require_device(str(args.device))
    output_dir = Path(args.output_dir).expanduser().resolve()
    specs = model_specs([int(v) for v in args.ranks], include_full=not bool(args.no_full))
    modes = ["sign_flip"] + ([] if bool(args.skip_null) else ["null"])
    parallel_devices = _resolve_parallel_devices(str(getattr(args, "parallel_devices", "")))
    if len(parallel_devices) > 1:
        print(f"[shear-rank] parallel devices: {', '.join(parallel_devices)}", flush=True)
        rows = _run_cases_parallel(args=args, specs=specs, modes=modes, devices=parallel_devices)
    else:
        device_name = parallel_devices[0] if parallel_devices else str(dev)
        rows = []
        for mode in modes:
            for n_per_condition in [int(v) for v in args.n_list]:
                for repeat_idx in range(int(args.n_seeds)):
                    rows.extend(
                        _run_dataset_case(
                            args=args,
                            specs=specs,
                            mode=str(mode),
                            n_per_condition=int(n_per_condition),
                            repeat_idx=int(repeat_idx),
                            device_name=device_name,
                        )
                    )
    rows = sorted(rows, key=_row_sort_key)

    aggregate = _aggregate(rows, modes=modes, n_list=[int(v) for v in args.n_list], specs=specs, n_seeds=int(args.n_seeds))
    results_npz = _write_aggregate_npz(output_dir / RESULTS_NPZ_NAME, aggregate)
    results_csv = _write_rows_csv(output_dir / RESULTS_CSV_NAME, rows)

    figure_paths: dict[str, tuple[Path, Path] | None] = {
        "dataset_geometry": plot_dataset_geometry(output_dir / "shear_rank_dataset_geometry", args=args),
        "distance_vs_n": plot_distance_vs_n(output_dir / "shear_rank_distance_vs_n", aggregate),
        "tradeoff": plot_tradeoff(output_dir / "shear_rank_tradeoff", aggregate, fixed_n=int(args.fixed_n)),
        "bias_variance": plot_bias_variance(output_dir / "shear_rank_bias_variance", aggregate, fixed_n=int(args.fixed_n)),
        "null_false_positive": plot_null_false_positive(output_dir / "shear_rank_null_false_positive", aggregate),
    }
    summary_json = _write_summary(
        output_dir / SUMMARY_JSON_NAME,
        {
            "script": "bin/run_shear_rank_skl_experiment.py",
            "device": str(dev),
            "output_dir": str(output_dir),
            "results_npz": str(results_npz),
            "results_csv": str(results_csv),
            "n_list": [int(v) for v in args.n_list],
            "n_seeds": int(args.n_seeds),
            "seed": int(args.seed),
            "modes": modes,
            "models": [spec.name for spec in specs],
            "x_dim": int(args.x_dim),
            "r_star": int(args.r_star),
            "amplitude": float(args.amplitude),
            "omega": float(args.omega),
            "train_frac": float(args.train_frac),
            "figures": {
                key: None if value is None else [str(value[0]), str(value[1])]
                for key, value in figure_paths.items()
            },
        },
    )
    print(f"results_npz: {results_npz}", flush=True)
    print(f"results_csv: {results_csv}", flush=True)
    print(f"summary_json: {summary_json}", flush=True)
    return {"results_npz": results_npz, "results_csv": results_csv, "summary_json": summary_json}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
