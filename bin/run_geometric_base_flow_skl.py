#!/usr/bin/env python3
"""Train affine geometric-base flow matching and estimate smoothed-line SKL."""

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
    ConditionTimeAffineVelocity,
    LineSegmentBase,
    estimate_smoothed_curve_symmetric_kl,
    geometric_flow_result_to_npz_dict,
    train_geometric_base_affine_flow,
)
from fisher.noisy_line_dataset import NoisyLineDataset
from fisher.shared_dataset_io import SharedDatasetBundle, load_shared_dataset_npz
from fisher.shared_fisher_est import require_device


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--dataset-npz", type=Path, default=None)
    src.add_argument("--toy", choices=("noisy_line",), default=None)
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--theta-values", type=str, default="0.0,0.5235987755982988")
    p.add_argument("--ell", type=float, default=1.5)
    p.add_argument("--target-sigma", type=float, default=0.12)
    p.add_argument("--shift-x", type=float, default=0.0)
    p.add_argument("--shift-y", type=float, default=0.0)
    p.add_argument("--n-total-per-theta", type=int, default=3000)
    p.add_argument("--train-frac", type=float, default=0.7)

    p.add_argument("--path-schedule", choices=("linear", "straight", "cosine"), default="cosine")
    p.add_argument("--source-pairing", choices=("random", "ot"), default="random")
    p.add_argument("--ot-method", choices=("exact", "sinkhorn", "unbalanced", "partial"), default="sinkhorn")
    p.add_argument("--ot-reg", type=float, default=0.05)
    p.add_argument("--ot-reg-m", type=float, default=1.0)
    p.add_argument("--ot-normalize-cost", action="store_true")
    p.add_argument("--ot-num-threads", type=str, default="1")
    p.add_argument("--fisher-kind", choices=("none", "full", "linear", "both"), default="none")
    p.add_argument("--smooth-sigma", type=float, default=0.12)
    p.add_argument("--mc-skl-samples", type=int, default=4096)
    p.add_argument("--density-mc-samples", type=int, default=1024)
    p.add_argument("--ode-steps", type=int, default=64)
    p.add_argument("--ode-method", type=str, default="midpoint")

    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--t-eps", type=float, default=0.0005)
    p.add_argument("--early-patience", type=int, default=0)
    p.add_argument("--early-min-delta", type=float, default=1e-4)
    p.add_argument("--early-ema-alpha", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=50)
    return p


def _parse_theta_values(text: str) -> np.ndarray:
    vals = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if len(vals) < 2:
        raise ValueError("--theta-values must contain at least two comma-separated values.")
    if not np.all(np.isfinite(vals)):
        raise ValueError("--theta-values must be finite.")
    return np.asarray(vals, dtype=np.float64).reshape(-1, 1)


def _stratified_split(labels: np.ndarray, *, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < float(train_frac) < 1.0):
        raise ValueError("--train-frac must be in (0, 1).")
    rng = np.random.default_rng(int(seed))
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        if int(idx.size) < 2:
            raise ValueError("Each condition needs at least two samples for train/validation.")
        rng.shuffle(idx)
        n_train = int(round(float(train_frac) * float(idx.size)))
        n_train = min(max(n_train, 1), int(idx.size) - 1)
        train_parts.append(idx[:n_train])
        val_parts.append(idx[n_train:])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def _toy_noisy_line_bundle(args: argparse.Namespace) -> tuple[SharedDatasetBundle, np.ndarray, str]:
    theta_eval = _parse_theta_values(str(args.theta_values))
    n_per = int(args.n_total_per_theta)
    if n_per < 2:
        raise ValueError("--n-total-per-theta must be >= 2.")
    theta_parts: list[np.ndarray] = []
    x_parts: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for idx, theta_value in enumerate(theta_eval[:, 0]):
        ds = NoisyLineDataset(
            theta=float(theta_value),
            ell=float(args.ell),
            sigma=float(args.target_sigma),
            shift=(float(args.shift_x), float(args.shift_y)),
            seed=int(args.seed) + int(idx),
        )
        batch = ds.sample(n_per)
        theta_parts.append(np.full((n_per, 1), float(theta_value), dtype=np.float64))
        x_parts.append(batch.x1.astype(np.float64, copy=False))
        labels.append(np.full(n_per, int(idx), dtype=np.int64))
    theta_all = np.concatenate(theta_parts, axis=0)
    x_all = np.concatenate(x_parts, axis=0)
    label_all = np.concatenate(labels, axis=0)
    train_idx, val_idx = _stratified_split(label_all, train_frac=float(args.train_frac), seed=int(args.seed) + 101)
    meta: dict[str, Any] = {
        "dataset_family": "noisy_line",
        "theta_type": "continuous",
        "theta_encoding": "native",
        "theta_dim": 1,
        "x_dim": 2,
        "theta_values": theta_eval.reshape(-1).tolist(),
        "ell": float(args.ell),
        "target_sigma": float(args.target_sigma),
        "shift": [float(args.shift_x), float(args.shift_y)],
        "n_total_per_theta": int(n_per),
        "train_frac": float(args.train_frac),
        "seed": int(args.seed),
    }
    bundle = SharedDatasetBundle(
        meta=meta,
        theta_all=theta_all,
        x_all=x_all,
        train_idx=train_idx,
        validation_idx=val_idx,
        theta_train=theta_all[train_idx],
        x_train=x_all[train_idx],
        theta_validation=theta_all[val_idx],
        x_validation=x_all[val_idx],
    )
    return bundle, theta_eval, "toy:noisy_line"


def _theta_eval_from_bundle(bundle: SharedDatasetBundle) -> np.ndarray:
    theta_all = np.asarray(bundle.theta_all, dtype=np.float64)
    if theta_all.ndim == 1:
        theta_all = theta_all.reshape(-1, 1)
    unique = np.unique(theta_all, axis=0)
    if int(unique.shape[1]) == 1:
        unique = unique[np.argsort(unique[:, 0])]
    return unique.astype(np.float64, copy=False)


def _load_bundle(args: argparse.Namespace) -> tuple[SharedDatasetBundle, np.ndarray, str]:
    if args.dataset_npz is not None:
        path = Path(args.dataset_npz).expanduser()
        bundle = load_shared_dataset_npz(path)
        return bundle, _theta_eval_from_bundle(bundle), str(path.resolve())
    if args.toy is None:
        args.toy = "noisy_line"
    return _toy_noisy_line_bundle(args)


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


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2, sort_keys=True)
        f.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dev = require_device(str(args.device))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    bundle, theta_eval, dataset_label = _load_bundle(args)
    theta_train = np.asarray(bundle.theta_train, dtype=np.float64)
    x_train = np.asarray(bundle.x_train, dtype=np.float64)
    theta_val = np.asarray(bundle.theta_validation, dtype=np.float64)
    x_val = np.asarray(bundle.x_validation, dtype=np.float64)
    theta_dim = int(theta_train.shape[1] if theta_train.ndim == 2 else 1)
    x_dim = int(x_train.shape[1] if x_train.ndim == 2 else 1)

    base = LineSegmentBase(anchor=np.zeros(x_dim, dtype=np.float64), direction=np.eye(x_dim, dtype=np.float64)[0])
    model = ConditionTimeAffineVelocity(
        theta_dim=theta_dim,
        x_dim=x_dim,
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
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
        source_pairing=str(args.source_pairing),
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
        theta_all=theta_eval,
        device=dev,
        smooth_sigma=float(args.smooth_sigma),
        mc_skl_samples=int(args.mc_skl_samples),
        density_mc_samples=int(args.density_mc_samples),
        ode_steps=int(args.ode_steps),
        ode_method=str(args.ode_method),
        batch_size=int(args.batch_size),
        fisher_kind=str(args.fisher_kind),
        train_metadata=train_meta,
    )

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = Path(DATA_DIR) / "geometric_base_flow_skl" / "line_affine"
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_npz = out_dir / "geometric_base_flow_skl_results.npz"
    summary_json = out_dir / "geometric_base_flow_skl_summary.json"
    model_state = out_dir / "model_state.pt"

    npz_fields = geometric_flow_result_to_npz_dict(result)
    npz_fields.update(
        {
            "theta_eval": theta_eval.astype(np.float64),
            "theta_train_shape": np.asarray(theta_train.shape, dtype=np.int64),
            "x_train_shape": np.asarray(x_train.shape, dtype=np.int64),
            "train_idx": np.asarray(bundle.train_idx, dtype=np.int64),
            "validation_idx": np.asarray(bundle.validation_idx, dtype=np.int64),
        }
    )
    np.savez_compressed(results_npz, **npz_fields)
    torch.save({"model_state_dict": model.state_dict(), "train_metadata": train_meta}, model_state)

    summary = {
        "script": "bin/run_geometric_base_flow_skl.py",
        "dataset": dataset_label,
        "device": str(dev),
        "path_schedule": str(args.path_schedule),
        "source_pairing": str(args.source_pairing),
        "ot_method": str(args.ot_method),
        "ot_reg": float(args.ot_reg),
        "ot_reg_m": float(args.ot_reg_m),
        "ot_normalize_cost": bool(args.ot_normalize_cost),
        "ot_num_threads": str(args.ot_num_threads),
        "ode_method": str(args.ode_method),
        "theta_eval_shape": list(theta_eval.shape),
        "theta_train_shape": list(theta_train.shape),
        "x_train_shape": list(x_train.shape),
        "canonical_metric_name": result.canonical_metric_name,
        "symmetric_kl_shape": list(result.symmetric_kl_matrix.shape),
        "smooth_sigma": float(args.smooth_sigma),
        "mc_skl_samples": int(args.mc_skl_samples),
        "density_mc_samples": int(args.density_mc_samples),
        "best_epoch": int(train_meta["best_epoch"]),
        "best_val_loss": float(train_meta["best_val_loss"]),
        "stopped_epoch": int(train_meta["stopped_epoch"]),
        "stopped_early": bool(train_meta["stopped_early"]),
        "output_dir": str(out_dir),
        "results_npz": str(results_npz),
        "summary_json": str(summary_json),
        "model_state": str(model_state),
    }
    _write_summary(summary_json, summary)

    print(f"results_npz: {results_npz}", flush=True)
    print(f"summary_json: {summary_json}", flush=True)
    print(f"model_state: {model_state}", flush=True)
    print(f"canonical_metric_name: {result.canonical_metric_name}", flush=True)
    print(f"best_epoch: {int(train_meta['best_epoch'])}", flush=True)
    print(f"best_val_loss: {float(train_meta['best_val_loss']):.12g}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
