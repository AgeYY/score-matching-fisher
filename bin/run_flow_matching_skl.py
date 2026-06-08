#!/usr/bin/env python3
"""Train flow-matching endpoint models and estimate model-sampled symmetric KL."""

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

from global_setting import DATA_DIR

from fisher.data import ToyCategoricalRandomMoGDataset
from fisher.flow_matching_skl import (
    VELOCITY_FAMILIES,
    build_flow_skl_model,
    estimate_model_symmetric_kl,
    flow_skl_result_to_npz_dict,
    train_flow_skl_model,
)
from fisher.shared_dataset_io import SharedDatasetBundle, load_shared_dataset_npz
from fisher.shared_fisher_est import require_device


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--velocity-family", choices=VELOCITY_FAMILIES, default="condition_affine")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--dataset-npz", type=Path, default=None)
    src.add_argument("--toy", choices=("random_mog",), default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--path-schedule", choices=("cosine", "linear", "straight"), default="cosine")
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--fisher-kind", choices=("none", "full", "linear", "both"), default="none")
    p.add_argument("--mc-jeffreys-sample", dest="mc_jeffreys_sample", type=int, default=4096)
    p.add_argument("--mc-samples", dest="mc_jeffreys_sample", type=int, help=argparse.SUPPRESS)
    p.add_argument("--ode-steps", type=int, default=64)
    p.add_argument("--ode-method", type=str, default="midpoint")
    p.add_argument("--output-dir", type=Path, default=None)

    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-total", type=int, default=300)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--x-dim", type=int, default=2)
    p.add_argument("--num-categories", type=int, default=2)
    p.add_argument("--mog-mean-min-dist", type=float, default=None)

    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--low-rank-dim", type=int, default=4)
    p.add_argument("--quadrature-steps", type=int, default=64)
    p.add_argument("--divergence-estimator", choices=("hutchinson", "exact"), default="hutchinson")
    p.add_argument("--hutchinson-probes", type=int, default=1)
    p.add_argument("--t-eps", type=float, default=0.05)
    p.add_argument("--early-patience", type=int, default=0)
    p.add_argument("--early-min-delta", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--solve-jitter", type=float, default=1e-6)
    p.add_argument("--eval-max-points", type=int, default=128)
    return p


def _labels_from_one_hot(theta: np.ndarray) -> np.ndarray:
    arr = np.asarray(theta, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("theta must be a two-dimensional one-hot matrix.")
    return np.argmax(arr, axis=1).astype(np.int64)


def _stratified_split(
    theta: np.ndarray,
    *,
    train_frac: float,
    num_categories: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < float(train_frac) < 1.0):
        raise ValueError("--train-frac must be in (0, 1).")
    labels = _labels_from_one_hot(theta)
    rng = np.random.default_rng(int(seed))
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for k in range(int(num_categories)):
        idx = np.flatnonzero(labels == k)
        if idx.size < 2:
            raise ValueError(f"Category {k} has {idx.size} rows; need at least two for train/validation.")
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


def _bundle_from_toy_random_mog(args: argparse.Namespace) -> tuple[SharedDatasetBundle, dict[str, Any]]:
    ds = ToyCategoricalRandomMoGDataset(
        x_dim=int(args.x_dim),
        num_categories=int(args.num_categories),
        seed=int(args.seed),
        mog_mean_min_dist=args.mog_mean_min_dist,
    )
    theta_all, x_all = ds.sample_joint(int(args.n_total))
    theta_all = np.asarray(theta_all, dtype=np.float64)
    x_all = np.asarray(x_all, dtype=np.float64)
    train_idx, val_idx = _stratified_split(
        theta_all,
        train_frac=float(args.train_frac),
        num_categories=int(args.num_categories),
        seed=int(args.seed) + 1,
    )
    meta: dict[str, Any] = {
        "dataset_family": "random_mog_categorical",
        "theta_type": "categorical",
        "theta_encoding": "one_hot",
        "theta_dim": int(args.num_categories),
        "num_categories": int(args.num_categories),
        "x_dim": int(args.x_dim),
        "n_total": int(args.n_total),
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
    gt = {
        "toy_component_means": np.asarray(ds._mog_means, dtype=np.float64),
        "toy_component_variances": np.asarray(ds._mog_variances, dtype=np.float64),
    }
    return bundle, gt


def _load_bundle(args: argparse.Namespace) -> tuple[SharedDatasetBundle, dict[str, Any], str]:
    if args.dataset_npz is not None:
        path = Path(args.dataset_npz).expanduser()
        bundle = load_shared_dataset_npz(path)
        return bundle, {}, str(path.resolve())
    if args.toy is None:
        args.toy = "random_mog"
    bundle, extra = _bundle_from_toy_random_mog(args)
    return bundle, extra, "toy:random_mog"


def _theta_eval_from_bundle(bundle: SharedDatasetBundle, *, eval_max_points: int) -> np.ndarray:
    meta = dict(bundle.meta)
    theta_all = np.asarray(bundle.theta_all, dtype=np.float64)
    theta_type = str(meta.get("theta_type", "")).strip().lower()
    theta_encoding = str(meta.get("theta_encoding", "")).strip().lower()
    if theta_type == "categorical":
        if theta_encoding == "one_hot" or (theta_all.ndim == 2 and theta_all.shape[1] > 1):
            k = int(meta.get("num_categories", theta_all.shape[1]))
            return np.eye(k, dtype=np.float64)
        labels = np.unique(np.rint(theta_all.reshape(-1)).astype(np.int64))
        return labels.reshape(-1, 1).astype(np.float64)

    theta = theta_all if theta_all.ndim == 2 else theta_all.reshape(-1, 1)
    unique = np.unique(theta, axis=0)
    if int(unique.shape[0]) <= int(eval_max_points):
        if int(unique.shape[1]) == 1:
            return unique[np.argsort(unique[:, 0])]
        return unique
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(int(unique.shape[0]), size=int(eval_max_points), replace=False))
    sub = unique[idx]
    if int(sub.shape[1]) == 1:
        sub = sub[np.argsort(sub[:, 0])]
    return sub.astype(np.float64)


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

    bundle, extra_npz, dataset_label = _load_bundle(args)
    theta_train = np.asarray(bundle.theta_train, dtype=np.float64)
    x_train = np.asarray(bundle.x_train, dtype=np.float64)
    theta_val = np.asarray(bundle.theta_validation, dtype=np.float64)
    x_val = np.asarray(bundle.x_validation, dtype=np.float64)
    theta_eval = _theta_eval_from_bundle(bundle, eval_max_points=int(args.eval_max_points))

    if int(theta_train.shape[0]) < 1 or int(theta_val.shape[0]) < 1:
        raise ValueError("Dataset must provide non-empty train and validation splits.")
    theta_dim = int(theta_train.shape[1] if theta_train.ndim == 2 else 1)
    x_dim = int(x_train.shape[1] if x_train.ndim == 2 else 1)

    model = build_flow_skl_model(
        velocity_family=str(args.velocity_family),
        theta_dim=theta_dim,
        x_dim=x_dim,
        radius=float(args.radius),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        low_rank_dim=int(args.low_rank_dim),
        quadrature_steps=int(args.quadrature_steps),
        path_schedule=str(args.path_schedule),
        divergence_estimator=str(args.divergence_estimator),
        hutchinson_probes=int(args.hutchinson_probes),
    ).to(dev)

    train_meta = train_flow_skl_model(
        model=model,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        device=dev,
        velocity_family=str(args.velocity_family),
        path_schedule=str(args.path_schedule),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        t_eps=float(args.t_eps),
        patience=int(args.early_patience),
        min_delta=float(args.early_min_delta),
        max_grad_norm=float(args.max_grad_norm),
        log_every=max(1, int(args.log_every)),
    )

    result = estimate_model_symmetric_kl(
        model=model,
        theta_all=theta_eval,
        device=dev,
        velocity_family=str(args.velocity_family),
        radius=float(args.radius),
        mc_jeffreys_sample=int(args.mc_jeffreys_sample),
        ode_steps=int(args.ode_steps),
        ode_method=str(args.ode_method),
        batch_size=int(args.batch_size),
        solve_jitter=float(args.solve_jitter),
        quadrature_steps=int(args.quadrature_steps),
        fisher_kind=str(args.fisher_kind),
        train_metadata=train_meta,
    )

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = Path(DATA_DIR) / "flow_matching_skl" / str(args.velocity_family)
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_npz = out_dir / "flow_matching_skl_results.npz"
    summary_json = out_dir / "flow_matching_skl_summary.json"

    npz_fields = flow_skl_result_to_npz_dict(result)
    npz_fields.update(
        {
            "theta_eval": theta_eval.astype(np.float64),
            "train_losses": np.asarray(train_meta["train_losses"], dtype=np.float64),
            "val_losses": np.asarray(train_meta["val_losses"], dtype=np.float64),
            "train_idx": np.asarray(bundle.train_idx, dtype=np.int64),
            "validation_idx": np.asarray(bundle.validation_idx, dtype=np.int64),
        }
    )
    npz_fields.update(extra_npz)
    np.savez_compressed(results_npz, **npz_fields)

    summary = {
        "script": "bin/run_flow_matching_skl.py",
        "dataset": dataset_label,
        "velocity_family": str(args.velocity_family),
        "device": str(dev),
        "path_schedule": str(args.path_schedule),
        "ode_method": str(args.ode_method),
        "theta_train_shape": list(theta_train.shape),
        "x_train_shape": list(x_train.shape),
        "theta_eval_shape": list(theta_eval.shape),
        "canonical_metric_name": result.canonical_metric_name,
        "symmetric_kl_shape": list(result.symmetric_kl_matrix.shape),
        "best_epoch": int(train_meta["best_epoch"]),
        "best_val_loss": float(train_meta["best_val_loss"]),
        "stopped_epoch": int(train_meta["stopped_epoch"]),
        "stopped_early": bool(train_meta["stopped_early"]),
        "output_dir": str(out_dir),
        "results_npz": str(results_npz),
        "summary_json": str(summary_json),
    }
    _write_summary(summary_json, summary)

    print(f"results_npz: {results_npz}", flush=True)
    print(f"summary_json: {summary_json}", flush=True)
    print(f"canonical_metric_name: {result.canonical_metric_name}", flush=True)
    print(f"best_epoch: {int(train_meta['best_epoch'])}", flush=True)
    print(f"best_val_loss: {float(train_meta['best_val_loss']):.12g}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
