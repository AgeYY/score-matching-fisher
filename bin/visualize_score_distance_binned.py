#!/usr/bin/env python3
"""Train x-score/velocity models, build score-distance matrix, bin by theta, and visualize.

This script mirrors the vector-field distance pipeline used by visualize_score_distance_mds.py,
but replaces MDS with theta-binned matrix averaging.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt
import numpy as np
import torch

from global_setting import DATAROOT
from fisher.models import (
    ConditionalXFlowVelocity,
    ConditionalXFlowVelocityFiLMPerLayer,
    ConditionalXScore,
    UnconditionalXFlowVelocity,
    UnconditionalXFlowVelocityFiLMPerLayer,
    UnconditionalXScore,
)
from fisher.score_distance import (
    compute_cross_flow_velocity_matrix,
    compute_cross_score_matrix,
    compute_unconditional_flow_velocity_vectors,
    compute_unconditional_score_vectors,
    evaluate_distance_variants,
    evaluate_pairwise_score_distance_variants,
)
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import require_device
from fisher.trainers import (
    geometric_sigma_schedule,
    train_conditional_x_flow_model,
    train_conditional_x_score_model_ncsm_continuous,
    train_unconditional_x_flow_model,
    train_unconditional_x_score_model_ncsm_continuous,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Load shared dataset, train x-score/velocity model, build vector-field distance matrix, "
            "bin theta, and save binned distance heatmaps."
        )
    )
    p.add_argument("--dataset-npz", type=str, required=True, help="Path to shared dataset .npz.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default=str(Path(DATAROOT) / "outputs_score_distance_binned"))
    p.add_argument(
        "--method",
        type=str,
        default="flow",
        choices=["dsm", "flow"],
        help="Train x model with denoising score matching (dsm) or flow matching (flow).",
    )
    p.add_argument(
        "--dsm-conditioning",
        type=str,
        default="unconditional",
        choices=["conditional", "unconditional"],
        help=(
            "DSM only: conditional uses s(x,theta,sigma) and cross-matrix distances; "
            "unconditional uses s(x,sigma) and pairwise distances between rows."
        ),
    )
    p.add_argument(
        "--distance-x-aggregate",
        type=str,
        default="mean",
        choices=["mean", "geom_mean", "sum"],
        help="DSM conditional only: aggregate squared gaps along x_dim in cross-score distance.",
    )
    p.add_argument(
        "--flow-conditioning",
        type=str,
        default="conditional",
        choices=["conditional", "unconditional"],
        help=(
            "Flow only: conditional uses v(x,theta,t) cross-matrix distances; "
            "unconditional uses v(x,t) pairwise row distances."
        ),
    )
    p.add_argument(
        "--data-split",
        type=str,
        default="full",
        choices=["full", "eval"],
        help="Rows used for the distance matrix: full uses (theta_all, x_all), eval uses held-out split.",
    )
    p.add_argument(
        "--distance-variant",
        type=str,
        default="raw_d",
        choices=["raw_d", "raw_d2", "norm_d", "norm_d2"],
        help=(
            "Distance matrix variant to bin: raw_d/raw_d2 (no vector normalization) "
            "or norm_d/norm_d2 (using --score-norm)."
        ),
    )
    p.add_argument("--num-theta-bins", type=int, default=15)
    p.add_argument(
        "--theta-bin-mode",
        type=str,
        default="range",
        choices=["range", "meta_range"],
        help="'range' uses min/max of theta rows; 'meta_range' uses NPZ meta theta_low/theta_high.",
    )

    p.add_argument("--score-epochs", type=int, default=10000)
    p.add_argument("--score-batch-size", type=int, default=256)
    p.add_argument("--score-lr", type=float, default=1e-3)
    p.add_argument("--score-hidden-dim", type=int, default=128)
    p.add_argument("--score-depth", type=int, default=3)
    p.add_argument("--score-sigma-min-alpha", type=float, default=0.01)
    p.add_argument("--score-sigma-max-alpha", type=float, default=0.25)
    p.add_argument("--score-eval-sigmas", type=int, default=12)
    p.add_argument("--score-val-frac", type=float, default=0.1)
    p.add_argument("--score-early-patience", type=int, default=1000)
    p.add_argument("--score-early-min-delta", type=float, default=1e-4)
    p.add_argument("--score-early-ema-alpha", type=float, default=0.05)
    p.add_argument("--score-restore-best", action="store_true", default=True)
    p.add_argument("--no-score-restore-best", action="store_false", dest="score_restore_best")
    p.add_argument(
        "--score-norm",
        type=str,
        default="geom",
        choices=["l2", "geom"],
        help="Per-vector normalization for norm_* distance variants.",
    )

    p.add_argument("--flow-epochs", type=int, default=10000)
    p.add_argument("--flow-batch-size", type=int, default=256)
    p.add_argument("--flow-lr", type=float, default=1e-3)
    p.add_argument("--flow-hidden-dim", type=int, default=128)
    p.add_argument("--flow-depth", type=int, default=3)
    p.add_argument(
        "--flow-arch",
        type=str,
        default="plain",
        choices=["plain", "film_per_layer"],
        help="Flow architecture.",
    )
    p.add_argument("--flow-scheduler", type=str, default="cosine", choices=["cosine", "vp", "linear_vp"])
    p.add_argument("--flow-eval-t", type=float, default=0.5)
    p.add_argument("--flow-val-frac", type=float, default=0.1)
    p.add_argument("--flow-early-patience", type=int, default=1000)
    p.add_argument("--flow-early-min-delta", type=float, default=1e-4)
    p.add_argument("--flow-early-ema-alpha", type=float, default=0.05)
    p.add_argument("--flow-restore-best", action="store_true", default=True)
    p.add_argument("--no-flow-restore-best", action="store_false", dest="flow_restore_best")
    p.add_argument("--cross-row-batch-size", type=int, default=128)
    p.add_argument("--save-cross-scores", action="store_true", default=False)
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def _split_train_val(
    theta: np.ndarray, x: np.ndarray, val_frac: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(theta.shape[0])
    if n < 2:
        raise ValueError("Need at least 2 samples for train/validation split.")
    n_val = int(round(float(val_frac) * n))
    n_val = max(1, min(n_val, n - 1))
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    fit_idx = perm[n_val:]
    return theta[fit_idx], x[fit_idx], theta[val_idx], x[val_idx]


def theta_bin_edges(theta: np.ndarray, meta: dict, n_bins: int, mode: str) -> tuple[np.ndarray, float, float]:
    th = np.asarray(theta, dtype=np.float64).reshape(-1)
    if n_bins < 1:
        raise ValueError("--num-theta-bins must be >= 1.")
    if mode == "range":
        lo = float(np.min(th))
        hi = float(np.max(th))
    elif mode == "meta_range":
        lo = float(meta["theta_low"])
        hi = float(meta["theta_high"])
    else:
        raise ValueError(f"Unknown theta-bin-mode: {mode}")
    if hi <= lo:
        raise ValueError(f"Invalid theta range for binning: [{lo}, {hi}]")
    edges = np.linspace(lo, hi, n_bins + 1, dtype=np.float64)
    return edges, lo, hi


def theta_to_bin_index(theta: np.ndarray, edges: np.ndarray, n_bins: int) -> np.ndarray:
    th = np.asarray(theta, dtype=np.float64).reshape(-1)
    idx = np.searchsorted(edges, th, side="right") - 1
    return np.clip(idx, 0, n_bins - 1).astype(np.int64)


def average_matrix_by_bins(
    mat: np.ndarray,
    bin_idx: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(mat, dtype=np.float64)
    n = int(m.shape[0])
    if m.shape != (n, n) or bin_idx.shape[0] != n:
        raise ValueError("Matrix and bin_idx shape mismatch.")
    binned = np.full((n_bins, n_bins), np.nan, dtype=np.float64)
    count = np.zeros((n_bins, n_bins), dtype=np.int64)

    for i in range(n_bins):
        rows = np.flatnonzero(bin_idx == i)
        n_i = int(rows.size)
        for j in range(n_bins):
            cols = np.flatnonzero(bin_idx == j)
            n_j = int(cols.size)
            if n_i == 0 or n_j == 0:
                continue
            block = m[np.ix_(rows, cols)]
            binned[i, j] = float(np.mean(block))
            count[i, j] = n_i * n_j
    return binned, count


def _distance_matrix_from_variant(
    raw: dict[str, np.ndarray | float], norm: dict[str, np.ndarray | float], variant: str
) -> np.ndarray:
    if variant == "raw_d":
        return np.asarray(raw["distance_d"], dtype=np.float64)
    if variant == "raw_d2":
        return np.asarray(raw["distance_d2"], dtype=np.float64)
    if variant == "norm_d":
        return np.asarray(norm["distance_d"], dtype=np.float64)
    if variant == "norm_d2":
        return np.asarray(norm["distance_d2"], dtype=np.float64)
    raise ValueError(f"Unknown distance variant: {variant}")


def main() -> None:
    args = parse_args()
    device = require_device(str(args.device))
    os.makedirs(args.output_dir, exist_ok=True)

    bundle = load_shared_dataset_npz(args.dataset_npz)
    seed = int(bundle.meta["seed"])
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.data_split == "full":
        theta_use = np.asarray(bundle.theta_all, dtype=np.float64)
        x_use = np.asarray(bundle.x_all, dtype=np.float64)
    else:
        theta_use = np.asarray(bundle.theta_eval, dtype=np.float64)
        x_use = np.asarray(bundle.x_eval, dtype=np.float64)
        if theta_use.shape[0] == 0:
            raise ValueError("data_split=eval requires non-empty eval split in dataset npz.")

    if args.method == "dsm":
        theta_fit, x_fit, theta_val, x_val = _split_train_val(theta_use, x_use, args.score_val_frac, rng)
        theta_std = float(np.std(theta_fit))
        sigma_min = float(args.score_sigma_min_alpha * theta_std)
        sigma_max = float(args.score_sigma_max_alpha * theta_std)
        if sigma_min <= 0.0 or sigma_max <= 0.0:
            raise ValueError("sigma bounds must be positive.")
        sigma_eval_grid = geometric_sigma_schedule(
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            n_levels=int(args.score_eval_sigmas),
            descending=True,
        )
        sigma_eval = float(np.min(sigma_eval_grid))
        flow_eval_t = float(args.flow_eval_t)

        if str(args.dsm_conditioning) == "unconditional":
            model = UnconditionalXScore(
                x_dim=x_use.shape[1],
                hidden_dim=int(args.score_hidden_dim),
                depth=int(args.score_depth),
                use_log_sigma=True,
            ).to(device)
            train_out = train_unconditional_x_score_model_ncsm_continuous(
                model=model,
                x_train=x_fit,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                epochs=int(args.score_epochs),
                batch_size=int(args.score_batch_size),
                lr=float(args.score_lr),
                device=device,
                log_every=max(1, int(args.log_every)),
                x_val=x_val,
                early_stopping_patience=int(args.score_early_patience),
                early_stopping_min_delta=float(args.score_early_min_delta),
                early_stopping_ema_alpha=float(args.score_early_ema_alpha),
                restore_best=bool(args.score_restore_best),
            )
            cross_scores = compute_unconditional_score_vectors(
                model=model,
                x=x_use,
                sigma_eval=sigma_eval,
                device=device,
                row_batch_size=int(args.cross_row_batch_size),
            )
            raw = evaluate_pairwise_score_distance_variants(cross_scores, score_normalize="none")
            norm = evaluate_pairwise_score_distance_variants(cross_scores, score_normalize=str(args.score_norm))
            method_detail = (
                "dsm(unconditional): "
                f"sigma_eval={sigma_eval:.6f}, sigma_levels={int(args.score_eval_sigmas)}, "
                "pairwise_mean_over_x_dim"
            )
        else:
            model = ConditionalXScore(
                x_dim=x_use.shape[1],
                hidden_dim=int(args.score_hidden_dim),
                depth=int(args.score_depth),
                use_log_sigma=True,
            ).to(device)
            train_out = train_conditional_x_score_model_ncsm_continuous(
                model=model,
                theta_train=theta_fit,
                x_train=x_fit,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                epochs=int(args.score_epochs),
                batch_size=int(args.score_batch_size),
                lr=float(args.score_lr),
                device=device,
                log_every=max(1, int(args.log_every)),
                theta_val=theta_val,
                x_val=x_val,
                early_stopping_patience=int(args.score_early_patience),
                early_stopping_min_delta=float(args.score_early_min_delta),
                early_stopping_ema_alpha=float(args.score_early_ema_alpha),
                restore_best=bool(args.score_restore_best),
            )
            cross_scores = compute_cross_score_matrix(
                model=model,
                theta=theta_use,
                x=x_use,
                sigma_eval=sigma_eval,
                device=device,
                row_batch_size=int(args.cross_row_batch_size),
            )
            raw = evaluate_distance_variants(
                cross_scores,
                score_normalize="none",
                x_dim_aggregate=str(args.distance_x_aggregate),
            )
            norm = evaluate_distance_variants(
                cross_scores,
                score_normalize=str(args.score_norm),
                x_dim_aggregate=str(args.distance_x_aggregate),
            )
            method_detail = (
                "dsm(conditional): "
                f"sigma_eval={sigma_eval:.6f}, sigma_levels={int(args.score_eval_sigmas)}, "
                f"x_dim_aggregate={args.distance_x_aggregate}"
            )
    else:
        theta_fit, x_fit, theta_val, x_val = _split_train_val(theta_use, x_use, args.flow_val_frac, rng)
        theta_std = float(np.std(theta_fit))
        sigma_min = float("nan")
        sigma_max = float("nan")
        sigma_eval = float("nan")
        sigma_eval_grid = np.asarray([], dtype=np.float64)
        flow_eval_t = float(args.flow_eval_t)
        if not (0.0 <= flow_eval_t <= 1.0):
            raise ValueError("--flow-eval-t must be in [0, 1].")

        flow_kw = dict(
            x_dim=x_use.shape[1],
            hidden_dim=int(args.flow_hidden_dim),
            depth=int(args.flow_depth),
            use_logit_time=True,
        )
        if str(args.flow_conditioning) == "unconditional":
            if str(args.flow_arch) == "film_per_layer":
                model = UnconditionalXFlowVelocityFiLMPerLayer(**flow_kw).to(device)
            else:
                model = UnconditionalXFlowVelocity(**flow_kw).to(device)
            train_out = train_unconditional_x_flow_model(
                model=model,
                x_train=x_fit,
                epochs=int(args.flow_epochs),
                batch_size=int(args.flow_batch_size),
                lr=float(args.flow_lr),
                device=device,
                log_every=max(1, int(args.log_every)),
                x_val=x_val,
                early_stopping_patience=int(args.flow_early_patience),
                early_stopping_min_delta=float(args.flow_early_min_delta),
                early_stopping_ema_alpha=float(args.flow_early_ema_alpha),
                restore_best=bool(args.flow_restore_best),
                scheduler_name=str(args.flow_scheduler),
            )
            cross_scores = compute_unconditional_flow_velocity_vectors(
                model=model,
                x=x_use,
                t_eval=flow_eval_t,
                device=device,
                row_batch_size=int(args.cross_row_batch_size),
            )
            raw = evaluate_pairwise_score_distance_variants(cross_scores, score_normalize="none")
            norm = evaluate_pairwise_score_distance_variants(cross_scores, score_normalize=str(args.score_norm))
            method_detail = (
                "flow(unconditional): "
                f"arch={args.flow_arch}, scheduler={args.flow_scheduler}, t_eval={flow_eval_t:.6f}, "
                "pairwise_mean_over_x_dim"
            )
        else:
            if str(args.flow_arch) == "film_per_layer":
                model = ConditionalXFlowVelocityFiLMPerLayer(**flow_kw).to(device)
            else:
                model = ConditionalXFlowVelocity(**flow_kw).to(device)
            train_out = train_conditional_x_flow_model(
                model=model,
                theta_train=theta_fit,
                x_train=x_fit,
                epochs=int(args.flow_epochs),
                batch_size=int(args.flow_batch_size),
                lr=float(args.flow_lr),
                device=device,
                log_every=max(1, int(args.log_every)),
                theta_val=theta_val,
                x_val=x_val,
                early_stopping_patience=int(args.flow_early_patience),
                early_stopping_min_delta=float(args.flow_early_min_delta),
                early_stopping_ema_alpha=float(args.flow_early_ema_alpha),
                restore_best=bool(args.flow_restore_best),
                scheduler_name=str(args.flow_scheduler),
            )
            cross_scores = compute_cross_flow_velocity_matrix(
                model=model,
                theta=theta_use,
                x=x_use,
                t_eval=flow_eval_t,
                device=device,
                row_batch_size=int(args.cross_row_batch_size),
            )
            raw = evaluate_distance_variants(cross_scores, score_normalize="none", x_dim_aggregate="geom_mean")
            norm = evaluate_distance_variants(cross_scores, score_normalize=str(args.score_norm), x_dim_aggregate="geom_mean")
            method_detail = (
                "flow(conditional): "
                f"arch={args.flow_arch}, scheduler={args.flow_scheduler}, t_eval={flow_eval_t:.6f}, "
                "x_dim_aggregate=geom_mean"
            )

    train_losses = np.asarray(train_out["train_losses"], dtype=np.float64)
    val_losses = np.asarray(train_out["val_losses"], dtype=np.float64)
    val_smooth = np.asarray(train_out["val_monitor_losses"], dtype=np.float64)
    best_epoch = int(train_out["best_epoch"])
    stopped_epoch = int(train_out["stopped_epoch"])
    stopped_early = bool(train_out["stopped_early"])

    distance = _distance_matrix_from_variant(raw, norm, str(args.distance_variant))
    edges, edge_lo, edge_hi = theta_bin_edges(theta_use, bundle.meta, int(args.num_theta_bins), str(args.theta_bin_mode))
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = theta_to_bin_index(theta_use, edges, int(args.num_theta_bins))
    dist_binned, count_matrix = average_matrix_by_bins(distance, bin_idx, int(args.num_theta_bins))

    loss_fig_path = os.path.join(args.output_dir, "score_x_loss_vs_epoch.png")
    epochs = np.arange(1, train_losses.size + 1)
    plt.figure(figsize=(8.6, 4.8))
    plt.plot(epochs, train_losses, label="train", color="#1f77b4")
    if val_losses.size == train_losses.size and np.any(np.isfinite(val_losses)):
        plt.plot(epochs, val_losses, label="val", color="#d62728")
    if val_smooth.size == train_losses.size and np.any(np.isfinite(val_smooth)):
        plt.plot(epochs, val_smooth, label="val_ema", color="#ff7f0e", linestyle="--")
    if 1 <= best_epoch <= train_losses.size:
        plt.axvline(best_epoch, color="#2ca02c", linestyle="--", linewidth=1.2, label=f"best={best_epoch}")
    if 1 <= stopped_epoch <= train_losses.size:
        plt.axvline(stopped_epoch, color="#9467bd", linestyle=":", linewidth=1.2, label=f"stop={stopped_epoch}")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    if args.method == "dsm":
        title = "x-score NCSM training"
    else:
        title = "x-velocity flow-matching training"
    plt.title(title)
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_fig_path, dpi=180)
    plt.close()

    heatmap_path = os.path.join(args.output_dir, "score_distance_binned_heatmap.png")
    plt.figure(figsize=(7.0, 6.0))
    im = plt.imshow(dist_binned, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04, label=f"mean distance ({args.distance_variant})")
    plt.xlabel(r"bin $j$")
    plt.ylabel(r"bin $i$")
    plt.title(f"Theta-binned vector-field distance ({int(args.num_theta_bins)} bins)")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=180)
    plt.close()

    count_heatmap_path = os.path.join(args.output_dir, "score_distance_binned_count_heatmap.png")
    plt.figure(figsize=(7.0, 6.0))
    log_counts = np.log1p(count_matrix.astype(np.float64))
    im2 = plt.imshow(log_counts, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(im2, fraction=0.046, pad=0.04, label=r"$\log(1+N_{ij})$")
    plt.xlabel(r"bin $j$")
    plt.ylabel(r"bin $i$")
    plt.title("Log pair counts per bin pair")
    plt.tight_layout()
    plt.savefig(count_heatmap_path, dpi=180)
    plt.close()

    npz_path = os.path.join(args.output_dir, "score_distance_binned_results.npz")
    payload: dict[str, object] = {
        "method": np.asarray([str(args.method)], dtype=object),
        "dsm_conditioning": np.asarray([str(args.dsm_conditioning)], dtype=object),
        "flow_conditioning": np.asarray([str(args.flow_conditioning)], dtype=object),
        "distance_variant": np.asarray([str(args.distance_variant)], dtype=object),
        "distance_x_aggregate": np.asarray([str(args.distance_x_aggregate)], dtype=object),
        "score_vector_norm": np.asarray([str(args.score_norm)], dtype=object),
        "theta": theta_use.reshape(-1),
        "x": x_use,
        "distance_selected": distance,
        "distance_raw_d": np.asarray(raw["distance_d"], dtype=np.float64),
        "distance_raw_d2": np.asarray(raw["distance_d2"], dtype=np.float64),
        "distance_norm_d": np.asarray(norm["distance_d"], dtype=np.float64),
        "distance_norm_d2": np.asarray(norm["distance_d2"], dtype=np.float64),
        "distance_binned": dist_binned,
        "count_matrix": count_matrix,
        "theta_bin_edges": edges,
        "theta_bin_centers": centers,
        "bin_index_per_sample": bin_idx,
        "num_theta_bins": np.asarray([int(args.num_theta_bins)], dtype=np.int64),
        "theta_bin_mode": np.asarray([str(args.theta_bin_mode)], dtype=object),
        "theta_bin_edge_lo": np.asarray([edge_lo], dtype=np.float64),
        "theta_bin_edge_hi": np.asarray([edge_hi], dtype=np.float64),
        "sigma_eval": np.asarray([sigma_eval], dtype=np.float64),
        "sigma_eval_grid": sigma_eval_grid.astype(np.float64),
        "flow_eval_t": np.asarray([flow_eval_t], dtype=np.float64),
        "flow_arch": np.asarray([str(args.flow_arch) if args.method == "flow" else "n/a"], dtype=object),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_smooth_losses": val_smooth,
        "best_epoch": np.asarray([best_epoch], dtype=np.int64),
        "stopped_epoch": np.asarray([stopped_epoch], dtype=np.int64),
        "stopped_early": np.asarray([int(stopped_early)], dtype=np.int64),
        "dataset_npz": np.asarray([os.path.abspath(args.dataset_npz)], dtype=object),
    }
    if bool(args.save_cross_scores):
        if (args.method == "dsm" and str(args.dsm_conditioning) == "unconditional") or (
            args.method == "flow" and str(args.flow_conditioning) == "unconditional"
        ):
            payload["score_rows"] = cross_scores.astype(np.float32)
        else:
            payload["cross_scores"] = cross_scores.astype(np.float32)
    np.savez_compressed(npz_path, **payload)

    summary_path = os.path.join(args.output_dir, "score_distance_binned_summary.txt")
    n_finite = int(np.sum(np.isfinite(dist_binned)))
    n_nan = int(np.sum(~np.isfinite(dist_binned)))
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Vector-field distance binned summary\n")
        f.write(f"dataset_npz: {args.dataset_npz}\n")
        f.write(f"output_dir: {args.output_dir}\n")
        f.write(f"method: {args.method}\n")
        f.write(f"method_detail: {method_detail}\n")
        f.write(f"data_split: {args.data_split}\n")
        f.write(f"n_samples: {theta_use.shape[0]}\n")
        f.write(f"x_dim: {x_use.shape[1]}\n")
        f.write(f"distance_variant: {args.distance_variant}\n")
        f.write(f"score_norm: {args.score_norm}\n")
        if args.method == "dsm":
            f.write(
                f"sigma_schedule: theta_std={theta_std:.6f} sigma_min={sigma_min:.6f} "
                f"sigma_max={sigma_max:.6f} sigma_eval={sigma_eval:.6f}\n"
            )
        else:
            f.write(f"theta_std={theta_std:.6f}\n")
            f.write(f"flow_eval_t={flow_eval_t:.6f}\n")
        f.write(f"num_theta_bins: {int(args.num_theta_bins)}\n")
        f.write(f"theta_bin_mode: {args.theta_bin_mode}\n")
        f.write(f"theta_bin_edges: [{edge_lo}, {edge_hi}] (mode-dependent)\n")
        f.write(f"distance_binned finite cells: {n_finite} nan cells: {n_nan}\n")
        if n_finite > 0:
            f.write(
                f"distance_binned min (finite): {float(np.nanmin(dist_binned))} "
                f"max (finite): {float(np.nanmax(dist_binned))}\n"
            )
        f.write(
            "training: "
            f"best_epoch={best_epoch}, stopped_epoch={stopped_epoch}, stopped_early={stopped_early}\n"
        )
        f.write("artifacts:\n")
        f.write(f"  {loss_fig_path}\n")
        f.write(f"  {heatmap_path}\n")
        f.write(f"  {count_heatmap_path}\n")
        f.write(f"  {npz_path}\n")
        f.write(f"  {summary_path}\n")

    print(f"[score_distance_binned:{args.method}] Saved:")
    print(f"  - {loss_fig_path}")
    print(f"  - {heatmap_path}")
    print(f"  - {count_heatmap_path}")
    print(f"  - {npz_path}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()
