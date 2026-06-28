#!/usr/bin/env python3
"""Train scheduled linear x-flow on a tiny random categorical MoG and print endpoint metrics."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.data import ToyCategoricalRandomMoGDataset
from fisher.gaussian_x_flow import path_schedule_from_name
from fisher.h_decoding_categorical_twofig import hellinger_gt_sq_category_matrix
from fisher.llr_divergence import (
    symmetric_kl_gaussian_diag_matrix,
    symmetric_kl_gaussian_full_matrix,
)
from fisher.linear_x_flow import (
    ConditionalTimeLinearXFlowMLP,
    compute_linear_x_flow_analytic_hellinger_matrix,
    train_time_linear_x_flow_schedule,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "In-memory random MoG check for the time-dependent linear x-flow "
            "analytic endpoint Hellinger estimate."
        )
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-total", type=int, default=300)
    p.add_argument("--x-dim", type=int, default=2)
    p.add_argument("--num-categories", type=int, default=2)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--early-patience", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--path-schedule", type=str, default="cosine")
    p.add_argument("--metric", choices=("hellinger", "symmetric_kl", "both"), default="both")
    p.add_argument("--quadrature-steps", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--t-eps", type=float, default=0.05)
    p.add_argument("--early-min-delta", type=float, default=1e-4)
    p.add_argument("--early-ema-alpha", type=float, default=0.05)
    p.add_argument("--weight-ema-decay", type=float, default=0.9)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--solve-jitter", type=float, default=1e-6)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--plot-dataset", type=Path, default=None)
    p.add_argument("--plot-only", action="store_true", default=False)
    p.add_argument("--no-restore-best", dest="restore_best", action="store_false")
    p.set_defaults(restore_best=True)
    return p


def _labels_from_one_hot(theta: np.ndarray) -> np.ndarray:
    if theta.ndim != 2:
        raise ValueError("theta must be a two-dimensional one-hot matrix.")
    return np.argmax(theta, axis=1).astype(np.int64)


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
            raise ValueError(
                f"Category {k} has {idx.size} sampled rows; need at least two "
                "to place one row in train and validation."
            )
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


def _check_unit_interval(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise RuntimeError(f"{name} is not finite: {value!r}")
    if float(value) < -1e-9 or float(value) > 1.0 + 1e-9:
        raise RuntimeError(f"{name}={value:.12g} is outside [0, 1].")


def _check_finite_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise RuntimeError(f"{name} is not finite: {value!r}")
    if float(value) < -1e-9:
        raise RuntimeError(f"{name}={value:.12g} is negative.")


def _plot_dataset(
    *,
    ds: ToyCategoricalRandomMoGDataset,
    theta: np.ndarray,
    x: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    if int(x.shape[1]) != 2:
        raise ValueError("Dataset plotting requires x_dim=2.")
    labels = _labels_from_one_hot(theta)
    means = np.asarray(ds._mog_means, dtype=np.float64)
    variances = np.asarray(ds._mog_variances, dtype=np.float64)
    counts = np.bincount(labels, minlength=int(ds.num_categories))

    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for k in range(int(ds.num_categories)):
        mask = labels == k
        color = cmap(k % 10)
        ax.scatter(
            x[mask, 0],
            x[mask, 1],
            s=22,
            alpha=0.72,
            linewidths=0.0,
            color=color,
            label=f"category {k} (n={int(counts[k])})",
        )
        ax.scatter(
            means[k, 0],
            means[k, 1],
            marker="x",
            s=90,
            linewidths=2.0,
            color=color,
        )
        for n_std, alpha in ((1.0, 0.28), (2.0, 0.14)):
            ellipse = Ellipse(
                xy=(float(means[k, 0]), float(means[k, 1])),
                width=2.0 * float(n_std) * math.sqrt(float(variances[k, 0])),
                height=2.0 * float(n_std) * math.sqrt(float(variances[k, 1])),
                angle=0.0,
                facecolor="none",
                edgecolor=color,
                linewidth=1.4,
                alpha=alpha,
            )
            ax.add_patch(ellipse)

    ax.set_title("ToyCategoricalRandomMoGDataset")
    ax.set_xlabel("x[0]")
    ax.set_ylabel("x[1]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d0d0d0", linewidth=0.8, alpha=0.7)
    ax.legend(loc="best", frameon=True)
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.num_categories) != 2:
        raise ValueError("This check prints the single off-diagonal pair, so --num-categories must be 2.")
    if int(args.x_dim) != 2:
        raise ValueError("This check is specified for --x-dim 2.")
    if int(args.n_total) < 2 * int(args.num_categories):
        raise ValueError("--n-total must be large enough to split each category into train and validation.")

    requested_device = str(args.device).strip().lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available on this machine.")
    device = torch.device(requested_device)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    ds = ToyCategoricalRandomMoGDataset(
        x_dim=int(args.x_dim),
        num_categories=int(args.num_categories),
        seed=int(args.seed),
    )
    theta_all, x_all = ds.sample_joint(int(args.n_total))
    theta_all = np.asarray(theta_all, dtype=np.float64)
    x_all = np.asarray(x_all, dtype=np.float64)

    if args.plot_dataset is not None:
        _plot_dataset(ds=ds, theta=theta_all, x=x_all, output_path=Path(args.plot_dataset))
        print(f"dataset_plot: {Path(args.plot_dataset).expanduser().resolve()}", flush=True)
        if bool(args.plot_only):
            return 0

    train_idx, val_idx = _stratified_split(
        theta_all,
        train_frac=float(args.train_frac),
        num_categories=int(args.num_categories),
        seed=int(args.seed) + 1,
    )
    theta_train = theta_all[train_idx]
    x_train = x_all[train_idx]
    theta_val = theta_all[val_idx]
    x_val = x_all[val_idx]

    model = ConditionalTimeLinearXFlowMLP(
        theta_dim=int(theta_all.shape[1]),
        x_dim=int(x_all.shape[1]),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        quadrature_steps=int(args.quadrature_steps),
    ).to(device)

    train_out = train_time_linear_x_flow_schedule(
        model=model,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        device=device,
        schedule=path_schedule_from_name(str(args.path_schedule)),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        t_eps=float(args.t_eps),
        patience=int(args.early_patience),
        min_delta=float(args.early_min_delta),
        ema_alpha=float(args.early_ema_alpha),
        weight_ema_decay=float(args.weight_ema_decay),
        max_grad_norm=float(args.max_grad_norm),
        log_every=max(1, int(args.log_every)),
        restore_best=bool(args.restore_best),
        log_name="linear_x_flow_t",
    )

    theta_eval = np.eye(int(args.num_categories), dtype=np.float64)
    estimated_h2_matrix, endpoint_mu, endpoint_cov_or_diag, endpoint_is_diag = compute_linear_x_flow_analytic_hellinger_matrix(
        model=model,
        theta_all=theta_eval,
        device=device,
        solve_jitter=float(args.solve_jitter),
        quadrature_steps=int(args.quadrature_steps),
    )
    ground_truth_h2_matrix = hellinger_gt_sq_category_matrix(ds)

    estimated_h2 = float(estimated_h2_matrix[0, 1])
    ground_truth_h2 = float(ground_truth_h2_matrix[0, 1])
    estimated_h = math.sqrt(max(0.0, min(1.0, estimated_h2)))
    ground_truth_h = math.sqrt(max(0.0, min(1.0, ground_truth_h2)))
    abs_error_h = abs(estimated_h - ground_truth_h)

    estimated_skl_matrix = symmetric_kl_gaussian_full_matrix(
        endpoint_mu,
        endpoint_cov_or_diag,
        is_diag=bool(endpoint_is_diag),
    )
    ground_truth_skl_matrix = symmetric_kl_gaussian_diag_matrix(
        np.asarray(ds._mog_means, dtype=np.float64),
        np.asarray(ds._mog_variances, dtype=np.float64),
    )
    estimated_skl = float(estimated_skl_matrix[0, 1])
    ground_truth_skl = float(ground_truth_skl_matrix[0, 1])
    abs_error_skl = abs(estimated_skl - ground_truth_skl)

    if args.metric in {"hellinger", "both"}:
        for name, value in (
            ("estimated_H", estimated_h),
            ("ground_truth_H", ground_truth_h),
            ("estimated_H2", estimated_h2),
            ("ground_truth_H2", ground_truth_h2),
        ):
            _check_unit_interval(name, value)
        print(f"estimated_H: {estimated_h:.12g}", flush=True)
        print(f"ground_truth_H: {ground_truth_h:.12g}", flush=True)
        print(f"abs_error_H: {abs_error_h:.12g}", flush=True)
        print(f"estimated_H2: {estimated_h2:.12g}", flush=True)
        print(f"ground_truth_H2: {ground_truth_h2:.12g}", flush=True)

    if args.metric in {"symmetric_kl", "both"}:
        for name, value in (
            ("estimated_symmetric_KL", estimated_skl),
            ("ground_truth_symmetric_KL", ground_truth_skl),
            ("abs_error_symmetric_KL", abs_error_skl),
        ):
            _check_finite_nonnegative(name, value)
        print(f"estimated_symmetric_KL: {estimated_skl:.12g}", flush=True)
        print(f"ground_truth_symmetric_KL: {ground_truth_skl:.12g}", flush=True)
        print(f"abs_error_symmetric_KL: {abs_error_skl:.12g}", flush=True)

    print(f"best_epoch: {int(train_out['best_epoch'])}", flush=True)
    print(f"best_val_loss: {float(train_out['best_val_loss']):.12g}", flush=True)
    print(f"stopped_epoch: {int(train_out['stopped_epoch'])}", flush=True)
    print(f"stopped_early: {bool(train_out['stopped_early'])}", flush=True)
    print(f"final_eval_weights: {train_out['final_eval_weights']}", flush=True)
    print(f"endpoint_is_diag: {bool(endpoint_is_diag)}", flush=True)
    print(f"n_train: {int(theta_train.shape[0])}", flush=True)
    print(f"n_val: {int(theta_val.shape[0])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
