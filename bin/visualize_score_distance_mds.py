#!/usr/bin/env python3
"""Train conditional x-score, build score-distance matrix, and run MDS."""

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
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import Isomap

try:
    import umap
except ImportError as e:
    raise ImportError(
        "visualize_score_distance_mds.py requires umap-learn for UMAP panels. "
        "Install with: pip install umap-learn"
    ) from e

from global_setting import DATA_DIR
from fisher.models import (
    ConditionalXFlowVelocity,
    ConditionalXFlowVelocityFiLMPerLayer,
    ConditionalXScore,
    UnconditionalXFlowVelocity,
    UnconditionalXFlowVelocityFiLMPerLayer,
    UnconditionalXScore,
)
from fisher.score_distance import (
    classical_mds_from_distances,
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
            "Load shared dataset, train x-score (conditional on theta or unconditional), "
            "build score-based distances, and run MDS."
        )
    )
    p.add_argument("--dataset-npz", type=str, required=True, help="Path to shared dataset .npz.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default=str(Path(DATA_DIR) / "outputs_score_distance_mds"))
    p.add_argument(
        "--method",
        type=str,
        default="dsm",
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
            "unconditional uses s(x,sigma) only and pairwise mean score distances between samples."
        ),
    )
    p.add_argument(
        "--distance-x-aggregate",
        type=str,
        default="mean",
        choices=["mean", "geom_mean", "sum"],
        help="DSM conditional only: how to aggregate squared gaps along x_dim in the cross-score distance.",
    )
    p.add_argument(
        "--flow-conditioning",
        type=str,
        default="conditional",
        choices=["conditional", "unconditional"],
        help=(
            "Flow only: conditional uses v(x,theta,t) and cross-matrix distances (geom_mean); "
            "unconditional uses v(x,t) only and pairwise mean velocity distances between samples."
        ),
    )
    p.add_argument(
        "--data-split",
        type=str,
        default="full",
        choices=["full", "eval"],
        help="Rows used for score matrix: full uses (theta_all,x_all), eval uses held-out split.",
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
        help="Per-vector normalization for the 'norm' distance/MDS variant: L2 unit norm or geometric (|.|) scale.",
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
        help="Flow only: plain stacked MLP on [x,theta,logit_t] or FiLM-modulate hidden layers from (theta, logit_t).",
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
    p.add_argument(
        "--isomap-n-neighbors",
        type=int,
        default=100,
        help="Isomap n_neighbors on x (capped at n_samples - 1).",
    )
    p.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=100,
        help="UMAP n_neighbors on x (capped at n_samples - 1).",
    )
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument(
        "--umap-random-state",
        type=int,
        default=-1,
        help="UMAP random seed; -1 uses dataset seed from NPZ meta.",
    )
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


def euclidean_distance_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be 2D for Euclidean distances.")
    n = x.shape[0]
    if n < 2:
        return np.zeros((n, n), dtype=np.float64)
    d = squareform(pdist(x, metric="euclidean"))
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d


def fit_isomap_2d(x: np.ndarray, n_neighbors: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    k = max(2, min(int(n_neighbors), n - 1))
    iso = Isomap(n_components=2, n_neighbors=k)
    return iso.fit_transform(x).astype(np.float64)


def fit_umap_2d(x: np.ndarray, n_neighbors: int, min_dist: float, random_state: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 2:
        return np.zeros((n, 2), dtype=np.float64)
    k = max(2, min(int(n_neighbors), n - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=k,
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=int(random_state),
    )
    return reducer.fit_transform(x).astype(np.float64)


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

        dsm_cond = str(args.dsm_conditioning)
        print(
            "[score_x:dsm] "
            f"conditioning={dsm_cond} "
            f"data_split={args.data_split} n={theta_use.shape[0]} fit={theta_fit.shape[0]} val={theta_val.shape[0]} "
            f"theta_std={theta_std:.6f} sigma_min={sigma_min:.6f} sigma_max={sigma_max:.6f} sigma_eval={sigma_eval:.6f}"
        )

        if dsm_cond == "unconditional":
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
            summary_method_detail = (
                f"dsm(unconditional): sigma_min={sigma_min:.6f}, sigma_max={sigma_max:.6f}, "
                f"sigma_eval={sigma_eval:.6f}, sigma_eval_levels={int(args.score_eval_sigmas)}, "
                "distance=pairwise_mean_over_x_dim"
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
            summary_method_detail = (
                f"dsm(conditional): sigma_min={sigma_min:.6f}, sigma_max={sigma_max:.6f}, sigma_eval={sigma_eval:.6f}, "
                f"sigma_eval_levels={int(args.score_eval_sigmas)}, "
                f"cross_distance_x_agg={args.distance_x_aggregate}"
            )
    else:
        theta_fit, x_fit, theta_val, x_val = _split_train_val(theta_use, x_use, args.flow_val_frac, rng)
        theta_std = float(np.std(theta_fit))
        sigma_min = float("nan")
        sigma_max = float("nan")
        sigma_eval = float("nan")
        sigma_eval_grid = np.asarray([], dtype=np.float64)
        if not (0.0 <= float(args.flow_eval_t) <= 1.0):
            raise ValueError("--flow-eval-t must be in [0, 1].")

        flow_cond = str(args.flow_conditioning)
        print(
            "[score_x:flow] "
            f"arch={args.flow_arch} "
            f"conditioning={flow_cond} "
            f"data_split={args.data_split} n={theta_use.shape[0]} fit={theta_fit.shape[0]} val={theta_val.shape[0]} "
            f"theta_std={theta_std:.6f} scheduler={args.flow_scheduler} t_eval={float(args.flow_eval_t):.4f}"
        )

        if flow_cond == "unconditional":
            flow_kw = dict(
                x_dim=x_use.shape[1],
                hidden_dim=int(args.flow_hidden_dim),
                depth=int(args.flow_depth),
                use_logit_time=True,
            )
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
                t_eval=float(args.flow_eval_t),
                device=device,
                row_batch_size=int(args.cross_row_batch_size),
            )
            summary_method_detail = (
                f"flow(unconditional): arch={args.flow_arch}, scheduler={args.flow_scheduler}, "
                f"t_eval={float(args.flow_eval_t):.6f}, "
                f"epochs={int(args.flow_epochs)}, batch_size={int(args.flow_batch_size)}, lr={float(args.flow_lr):.6g}, "
                "distance=pairwise_mean_over_x_dim"
            )
        else:
            flow_kw = dict(
                x_dim=x_use.shape[1],
                hidden_dim=int(args.flow_hidden_dim),
                depth=int(args.flow_depth),
                use_logit_time=True,
            )
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
                t_eval=float(args.flow_eval_t),
                device=device,
                row_batch_size=int(args.cross_row_batch_size),
            )
            summary_method_detail = (
                f"flow(conditional): arch={args.flow_arch}, scheduler={args.flow_scheduler}, "
                f"t_eval={float(args.flow_eval_t):.6f}, "
                f"epochs={int(args.flow_epochs)}, batch_size={int(args.flow_batch_size)}, lr={float(args.flow_lr):.6g}, "
                "cross_distance_x_agg=geom_mean"
            )

    train_losses = np.asarray(train_out["train_losses"], dtype=np.float64)
    val_losses = np.asarray(train_out["val_losses"], dtype=np.float64)
    val_smooth = np.asarray(train_out["val_monitor_losses"], dtype=np.float64)
    best_epoch = int(train_out["best_epoch"])
    stopped_epoch = int(train_out["stopped_epoch"])
    stopped_early = bool(train_out["stopped_early"])

    if args.method == "dsm" and str(args.dsm_conditioning) == "unconditional":
        print(f"[cross_matrix:{args.method}] shape={cross_scores.shape} (N,x_dim) unconditional score rows")
        raw = evaluate_pairwise_score_distance_variants(cross_scores, score_normalize="none")
        norm = evaluate_pairwise_score_distance_variants(cross_scores, score_normalize=str(args.score_norm))
    elif args.method == "dsm":
        print(f"[cross_matrix:{args.method}] shape={cross_scores.shape} (N,N,x_dim)")
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
    elif args.method == "flow" and str(args.flow_conditioning) == "unconditional":
        print(f"[cross_matrix:{args.method}] shape={cross_scores.shape} (N,x_dim) unconditional flow velocity rows")
        raw = evaluate_pairwise_score_distance_variants(cross_scores, score_normalize="none")
        norm = evaluate_pairwise_score_distance_variants(cross_scores, score_normalize=str(args.score_norm))
    else:
        print(f"[cross_matrix:{args.method}] shape={cross_scores.shape} (N,N,x_dim)")
        raw = evaluate_distance_variants(cross_scores, score_normalize="none", x_dim_aggregate="geom_mean")
        norm = evaluate_distance_variants(
            cross_scores, score_normalize=str(args.score_norm), x_dim_aggregate="geom_mean"
        )

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
        train_title = (
            "Unconditional x-score NCSM training"
            if str(args.dsm_conditioning) == "unconditional"
            else "Conditional x-score NCSM training"
        )
    else:
        train_title = (
            "Unconditional x-velocity flow-matching training"
            if str(args.flow_conditioning) == "unconditional"
            else "Conditional x-velocity flow-matching training"
        )
    plt.title(train_title)
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_fig_path, dpi=180)
    plt.close()

    theta_color = theta_use.reshape(-1)
    n_s = int(theta_use.shape[0])

    # Baseline embeddings on x: Euclidean MDS, Isomap, UMAP.
    d_euclid = euclidean_distance_matrix(x_use)
    mds_euclid = classical_mds_from_distances(d_euclid, n_components=2)
    emb_euclid = np.asarray(mds_euclid.embedding, dtype=np.float64)

    isomap_k = max(2, min(int(args.isomap_n_neighbors), n_s - 1))
    emb_isomap = fit_isomap_2d(x_use, n_neighbors=isomap_k)

    umap_k = max(2, min(int(args.umap_n_neighbors), n_s - 1))
    umap_rs = seed if int(args.umap_random_state) < 0 else int(args.umap_random_state)
    emb_umap = fit_umap_2d(x_use, n_neighbors=umap_k, min_dist=float(args.umap_min_dist), random_state=umap_rs)

    # Primary: raw (unnormalized) vectors, then distance d (sqrt form).
    primary_emb = np.asarray(raw["embedding_d"], dtype=np.float64)

    emb_fig_path = os.path.join(args.output_dir, "score_distance_mds_theta_color.png")
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.0), layout="constrained")
    ax_flat = np.asarray(axes).ravel()
    if args.method == "dsm":
        if str(args.dsm_conditioning) == "unconditional":
            primary_title = r"Pairwise score distance (mean) $\rightarrow$ Classical MDS (raw, $d$)"
        else:
            primary_title = r"Score distance $\rightarrow$ Classical MDS (raw score, $d$)"
    elif str(args.flow_conditioning) == "unconditional":
        primary_title = r"Pairwise flow velocity distance (mean) $\rightarrow$ Classical MDS (raw, $d$)"
    else:
        primary_title = r"Flow-velocity distance $\rightarrow$ Classical MDS (raw velocity, $d$)"
    panels: list[tuple[np.ndarray, str, str, str]] = [
        (primary_emb, primary_title, "MDS 1", "MDS 2"),
        (emb_euclid, r"Euclidean $x$ → Classical MDS", "MDS 1", "MDS 2"),
        (emb_isomap, f"Isomap on $x$ ($k$={isomap_k})", "Isomap 1", "Isomap 2"),
        (
            emb_umap,
            rf"UMAP on $x$ ($k$={umap_k}, min\_dist={float(args.umap_min_dist):.3g}, seed={umap_rs})",
            "UMAP 1",
            "UMAP 2",
        ),
    ]
    for ax, (emb, title, xl, yl) in zip(ax_flat, panels):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=theta_color, s=10, alpha=0.65, cmap="viridis")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    fig.colorbar(sc, ax=ax_flat.tolist(), label=r"$\theta$", shrink=0.55, aspect=28)
    fig.suptitle("2D embeddings colored by $\\theta$ (comparison)", fontsize=13)
    plt.savefig(emb_fig_path, dpi=180, bbox_inches="tight")
    plt.close()

    npz_payload: dict[str, object] = {
        "method": np.asarray([str(args.method)]),
        "dsm_conditioning": np.asarray([str(args.dsm_conditioning)]),
        "flow_conditioning": np.asarray([str(args.flow_conditioning)]),
        "distance_x_aggregate": np.asarray([str(args.distance_x_aggregate)]),
        "score_vector_norm": np.asarray([str(args.score_norm)]),
        "theta": theta_use.reshape(-1),
        "x": x_use,
        "sigma_eval": np.asarray([sigma_eval], dtype=np.float64),
        "sigma_eval_grid": sigma_eval_grid.astype(np.float64),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_smooth_losses": val_smooth,
        "best_epoch": np.asarray([best_epoch], dtype=np.int64),
        "stopped_epoch": np.asarray([stopped_epoch], dtype=np.int64),
        "stopped_early": np.asarray([int(stopped_early)], dtype=np.int64),
        "embedding_raw_d": np.asarray(raw["embedding_d"], dtype=np.float64),
        "embedding_raw_d2": np.asarray(raw["embedding_d2"], dtype=np.float64),
        "embedding_norm_d": np.asarray(norm["embedding_d"], dtype=np.float64),
        "embedding_norm_d2": np.asarray(norm["embedding_d2"], dtype=np.float64),
        "distance_raw_d": np.asarray(raw["distance_d"], dtype=np.float64),
        "distance_raw_d2": np.asarray(raw["distance_d2"], dtype=np.float64),
        "distance_norm_d": np.asarray(norm["distance_d"], dtype=np.float64),
        "distance_norm_d2": np.asarray(norm["distance_d2"], dtype=np.float64),
        "strain_raw_d": np.asarray([float(raw["strain_d"])], dtype=np.float64),
        "strain_raw_d2": np.asarray([float(raw["strain_d2"])], dtype=np.float64),
        "strain_norm_d": np.asarray([float(norm["strain_d"])], dtype=np.float64),
        "strain_norm_d2": np.asarray([float(norm["strain_d2"])], dtype=np.float64),
        "positive_eig_raw_d": np.asarray([float(raw["positive_eig_d"])], dtype=np.float64),
        "positive_eig_raw_d2": np.asarray([float(raw["positive_eig_d2"])], dtype=np.float64),
        "positive_eig_norm_d": np.asarray([float(norm["positive_eig_d"])], dtype=np.float64),
        "positive_eig_norm_d2": np.asarray([float(norm["positive_eig_d2"])], dtype=np.float64),
        "embedding_mds_euclidean_x": emb_euclid,
        "embedding_isomap_x": emb_isomap,
        "embedding_umap_x": emb_umap,
        "strain_mds_euclidean_x": np.asarray([float(mds_euclid.strain_relative)], dtype=np.float64),
        "isomap_n_neighbors_used": np.asarray([isomap_k], dtype=np.int64),
        "umap_n_neighbors_used": np.asarray([umap_k], dtype=np.int64),
        "umap_min_dist": np.asarray([float(args.umap_min_dist)], dtype=np.float64),
        "umap_random_state_used": np.asarray([umap_rs], dtype=np.int64),
        "flow_eval_t": np.asarray([float(args.flow_eval_t)], dtype=np.float64),
        "flow_arch": np.asarray([str(args.flow_arch) if args.method == "flow" else "n/a"]),
    }
    if bool(args.save_cross_scores):
        if (args.method == "dsm" and str(args.dsm_conditioning) == "unconditional") or (
            args.method == "flow" and str(args.flow_conditioning) == "unconditional"
        ):
            npz_payload["score_rows"] = cross_scores.astype(np.float32)
        else:
            npz_payload["cross_scores"] = cross_scores.astype(np.float32)
    npz_path = os.path.join(args.output_dir, "score_distance_mds_results.npz")
    np.savez_compressed(npz_path, **npz_payload)

    summary_path = os.path.join(args.output_dir, "score_distance_mds_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Vector-field distance + MDS summary\n")
        f.write(f"dataset_npz: {args.dataset_npz}\n")
        f.write(f"output_dir: {args.output_dir}\n")
        f.write(f"method: {args.method}\n")
        if args.method == "dsm":
            f.write(
                f"dsm_conditioning: {args.dsm_conditioning} "
                f"(unconditional: pairwise mean score distance; conditional: cross-matrix + distance_x_aggregate)\n"
            )
            f.write(f"distance_x_aggregate (dsm conditional only): {args.distance_x_aggregate}\n")
        if args.method == "flow":
            f.write(f"flow_arch: {args.flow_arch}\n")
            f.write(
                f"flow_conditioning: {args.flow_conditioning} "
                f"(unconditional: pairwise mean velocity distance; conditional: cross-matrix + geom_mean)\n"
            )
        f.write(f"score_norm_variant: {args.score_norm} (per-vector norm for norm_* strains; raw_* uses none)\n")
        f.write(f"method_detail: {summary_method_detail}\n")
        f.write(f"data_split: {args.data_split}\n")
        f.write(f"n_samples: {theta_use.shape[0]}\n")
        f.write(f"x_dim: {x_use.shape[1]}\n")
        if args.method == "dsm":
            f.write(
                "training: "
                f"epochs={args.score_epochs}, batch_size={args.score_batch_size}, lr={args.score_lr}, "
                f"best_epoch={best_epoch}, stopped_epoch={stopped_epoch}, stopped_early={stopped_early}\n"
            )
            f.write(
                "sigma_schedule: "
                f"theta_std={theta_std:.6f}, sigma_min={sigma_min:.6f}, sigma_max={sigma_max:.6f}, "
                f"sigma_eval(min)={sigma_eval:.6f}\n"
            )
        else:
            f.write(
                "training: "
                f"epochs={args.flow_epochs}, batch_size={args.flow_batch_size}, lr={args.flow_lr}, "
                f"scheduler={args.flow_scheduler}, t_eval={float(args.flow_eval_t):.6f}, "
                f"best_epoch={best_epoch}, stopped_epoch={stopped_epoch}, stopped_early={stopped_early}\n"
            )
            f.write(f"theta_std={theta_std:.6f}\n")
        if (args.method == "dsm" and str(args.dsm_conditioning) == "unconditional") or (
            args.method == "flow" and str(args.flow_conditioning) == "unconditional"
        ):
            f.write(
                "per-sample vectors (score or velocity rows): "
                f"v shape={cross_scores.shape}, row_batch_size={args.cross_row_batch_size}, "
                f"save_cross_scores saves score_rows in npz={bool(args.save_cross_scores)}\n"
            )
        else:
            f.write(
                "cross_vectors: "
                f"S shape={cross_scores.shape}, row_batch_size={args.cross_row_batch_size}, "
                f"save_cross_scores={bool(args.save_cross_scores)}\n"
            )
        f.write(
            "figure score_distance_mds_theta_color.png (top-left score panel): "
            "raw vector + d (no per-vector normalization before distances).\n"
        )
        f.write("MDS quality (strain, lower is better):\n")
        f.write(f"  raw score + d:    strain={float(raw['strain_d']):.6f}, positive_eigs={int(raw['positive_eig_d'])}\n")
        f.write(f"  raw score + d^2:  strain={float(raw['strain_d2']):.6f}, positive_eigs={int(raw['positive_eig_d2'])}\n")
        f.write(
            f"  norm ({args.score_norm}) + d:   strain={float(norm['strain_d']):.6f}, "
            f"positive_eigs={int(norm['positive_eig_d'])}\n"
        )
        f.write(
            f"  norm ({args.score_norm}) + d^2: strain={float(norm['strain_d2']):.6f}, "
            f"positive_eigs={int(norm['positive_eig_d2'])}\n"
        )
        f.write(
            "Baselines on x: "
            f"Euclidean MDS strain={float(mds_euclid.strain_relative):.6f}, "
            f"Isomap k={isomap_k}, UMAP k={umap_k} min_dist={float(args.umap_min_dist):.6g} seed={umap_rs}\n"
        )
        f.write("artifacts:\n")
        f.write(f"  {loss_fig_path}\n")
        f.write(f"  {emb_fig_path}\n")
        f.write(f"  {npz_path}\n")
        f.write(f"  {summary_path}\n")

    print(f"[score_distance_mds:{args.method}] Saved:")
    print(f"  - {loss_fig_path}")
    print(f"  - {emb_fig_path}")
    print(f"  - {npz_path}")
    print(f"  - {summary_path}")
    print(
        "[score_distance_mds] strains: "
        f"raw_d={float(raw['strain_d']):.6f}, "
        f"raw_d2={float(raw['strain_d2']):.6f}, "
        f"norm({args.score_norm})_d={float(norm['strain_d']):.6f}, "
        f"norm({args.score_norm})_d2={float(norm['strain_d2']):.6f}"
    )


if __name__ == "__main__":
    main()
