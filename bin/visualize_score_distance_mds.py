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

from global_setting import DATAROOT
from fisher.models import (
    ConditionalXScore,
    ConditionalXScoreFiLMPerLayer,
    ConditionalXScoreResidualConcat,
    UnconditionalXScore,
    UnconditionalXScoreFiLMPerLayer,
)
from fisher.score_distance import (
    classical_mds_from_distances,
    compute_cross_score_matrix,
    compute_unconditional_score_vectors,
    evaluate_distance_variants,
    evaluate_distance_variants_theta_bin_avg,
    evaluate_pairwise_score_distance_variants,
)
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import require_device
from fisher.trainers import (
    geometric_sigma_schedule,
    train_conditional_x_score_model_ncsm_continuous,
    train_unconditional_x_score_model_ncsm_continuous,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Load shared dataset, train conditional x-score with continuous NCSM, "
            "build S_ij=s(x_i|theta_j), compute score distance matrix, and run MDS."
        )
    )
    p.add_argument("--dataset-npz", type=str, required=True, help="Path to shared dataset .npz.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default=str(Path(DATAROOT) / "outputs_score_distance_mds"))
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
    p.add_argument(
        "--dsm-conditioning",
        type=str,
        default="conditional",
        choices=["conditional", "unconditional"],
        help=(
            "conditional: train s(x_tilde, theta, sigma) and build cross-score matrix S_ij. "
            "unconditional: train s(x_tilde, sigma) without theta and build pairwise distances over x samples."
        ),
    )
    p.add_argument(
        "--score-arch",
        type=str,
        default="plain",
        choices=["plain", "residual_concat", "film_per_layer"],
        help=(
            "plain: feed [x̃,θ,logσ] once at input (stacked MLP). "
            "residual_concat: concat θ and log σ into every block with residual hidden updates. "
            "film_per_layer: FiLM-modulate every hidden layer from (θ, log σ) without hidden concatenation. "
            "For unconditional DSM, FiLM uses log σ only."
        ),
    )
    p.add_argument("--score-sigma-min-alpha", type=float, default=0.01)
    p.add_argument("--score-sigma-max-alpha", type=float, default=0.25)
    p.add_argument("--score-eval-sigmas", type=int, default=12)
    p.add_argument("--score-val-frac", type=float, default=0.1)
    p.add_argument("--score-early-patience", type=int, default=1000)
    p.add_argument("--score-early-min-delta", type=float, default=1e-4)
    p.add_argument("--score-early-ema-alpha", type=float, default=0.05)
    p.add_argument("--score-restore-best", action="store_true", default=True)
    p.add_argument("--no-score-restore-best", action="store_false", dest="score_restore_best")
    p.add_argument("--cross-row-batch-size", type=int, default=128)
    p.add_argument("--save-cross-scores", action="store_true", default=False)
    p.add_argument(
        "--distance-definition",
        type=str,
        default="original",
        choices=["original", "theta_bin_avg"],
        help=(
            "Conditional DSM distance backend. "
            "original: d_ij^2 = 0.5||S_ii-S_ij||^2 + 0.5||S_ji-S_jj||^2. "
            "theta_bin_avg: average ||s(x_k,theta_i)-s(x_k,theta_j)||^2 over k in bins(theta_i) U bins(theta_j)."
        ),
    )
    p.add_argument(
        "--distance-theta-bins",
        type=int,
        default=50,
        help="Number of equal-width theta bins used by --distance-definition theta_bin_avg.",
    )
    p.add_argument(
        "--distance-theta-low",
        type=float,
        default=None,
        help="Lower bound for equal-width theta bins (default: min(theta_use)).",
    )
    p.add_argument(
        "--distance-theta-high",
        type=float,
        default=None,
        help="Upper bound for equal-width theta bins (default: max(theta_use)).",
    )
    p.add_argument(
        "--distance-min-bin-count",
        type=int,
        default=1,
        help="Minimum sample count for a valid theta bin in theta_bin_avg mode; sparse bins fall back to all rows.",
    )
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

    theta_dist_low = float(np.min(theta_use)) if args.distance_theta_low is None else float(args.distance_theta_low)
    theta_dist_high = float(np.max(theta_use)) if args.distance_theta_high is None else float(args.distance_theta_high)
    if theta_dist_high <= theta_dist_low:
        raise ValueError("distance-theta-high must be > distance-theta-low.")

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

    print(
        "[score_x] "
        f"data_split={args.data_split} n={theta_use.shape[0]} fit={theta_fit.shape[0]} val={theta_val.shape[0]} "
        f"dsm_conditioning={args.dsm_conditioning} "
        f"arch={args.score_arch} "
        f"theta_std={theta_std:.6f} sigma_min={sigma_min:.6f} sigma_max={sigma_max:.6f} sigma_eval={sigma_eval:.6f}"
    )

    score_kw = dict(
        x_dim=x_use.shape[1],
        hidden_dim=int(args.score_hidden_dim),
        depth=int(args.score_depth),
        use_log_sigma=True,
    )
    if str(args.dsm_conditioning) == "unconditional":
        if str(args.score_arch) == "film_per_layer":
            model = UnconditionalXScoreFiLMPerLayer(**score_kw).to(device)
        else:
            model = UnconditionalXScore(**score_kw).to(device)
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
    else:
        if str(args.score_arch) == "residual_concat":
            model = ConditionalXScoreResidualConcat(**score_kw).to(device)
        elif str(args.score_arch) == "film_per_layer":
            model = ConditionalXScoreFiLMPerLayer(**score_kw).to(device)
        else:
            model = ConditionalXScore(**score_kw).to(device)
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

    train_losses = np.asarray(train_out["train_losses"], dtype=np.float64)
    val_losses = np.asarray(train_out["val_losses"], dtype=np.float64)
    val_smooth = np.asarray(train_out["val_monitor_losses"], dtype=np.float64)
    best_epoch = int(train_out["best_epoch"])
    stopped_epoch = int(train_out["stopped_epoch"])
    stopped_early = bool(train_out["stopped_early"])

    cross_scores: np.ndarray | None = None
    score_vectors: np.ndarray | None = None
    distance_label = "unconditional_pairwise_mean"
    if str(args.dsm_conditioning) == "unconditional":
        score_vectors = compute_unconditional_score_vectors(
            model=model,
            x=x_use,
            sigma_eval=sigma_eval,
            device=device,
            row_batch_size=int(args.cross_row_batch_size),
        )
        print(f"[score_vector] shape={score_vectors.shape} (N,x_dim)")
        raw = evaluate_pairwise_score_distance_variants(score_vectors, normalize_score=False)
        norm = evaluate_pairwise_score_distance_variants(score_vectors, normalize_score=True)
    else:
        cross_scores = compute_cross_score_matrix(
            model=model,
            theta=theta_use,
            x=x_use,
            sigma_eval=sigma_eval,
            device=device,
            row_batch_size=int(args.cross_row_batch_size),
        )
        print(f"[score_matrix] shape={cross_scores.shape} (N,N,x_dim)")
        if str(args.distance_definition) == "theta_bin_avg":
            distance_label = "theta_bin_avg"
            raw = evaluate_distance_variants_theta_bin_avg(
                cross_scores,
                theta=theta_use,
                normalize_score=False,
                theta_low=theta_dist_low,
                theta_high=theta_dist_high,
                n_bins=int(args.distance_theta_bins),
                min_bin_count=int(args.distance_min_bin_count),
            )
            norm = evaluate_distance_variants_theta_bin_avg(
                cross_scores,
                theta=theta_use,
                normalize_score=True,
                theta_low=theta_dist_low,
                theta_high=theta_dist_high,
                n_bins=int(args.distance_theta_bins),
                min_bin_count=int(args.distance_min_bin_count),
            )
        else:
            distance_label = "original"
            raw = evaluate_distance_variants(cross_scores, normalize_score=False)
            norm = evaluate_distance_variants(cross_scores, normalize_score=True)

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
    if str(args.dsm_conditioning) == "unconditional":
        plt.title("Unconditional x-score NCSM training")
    else:
        plt.title("Conditional x-score NCSM training")
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

    # Primary: raw (unnormalized) per-(i,j) score vectors, then distance d (sqrt form).
    primary_emb = np.asarray(raw["embedding_d"], dtype=np.float64)

    emb_fig_path = os.path.join(args.output_dir, "score_distance_mds_theta_color.png")
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.0), layout="constrained")
    ax_flat = np.asarray(axes).ravel()
    panels: list[tuple[np.ndarray, str, str, str]] = [
        (
            primary_emb,
            rf"Score distance ({distance_label}) $\rightarrow$ Classical MDS (raw score, $d$)",
            "MDS 1",
            "MDS 2",
        ),
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
        "theta": theta_use.reshape(-1),
        "x": x_use,
        "dsm_conditioning": np.asarray([str(args.dsm_conditioning)]),
        "distance_definition": np.asarray([str(distance_label)]),
        "distance_theta_bins": np.asarray([int(args.distance_theta_bins)], dtype=np.int64),
        "distance_theta_low": np.asarray([theta_dist_low], dtype=np.float64),
        "distance_theta_high": np.asarray([theta_dist_high], dtype=np.float64),
        "distance_min_bin_count": np.asarray([int(args.distance_min_bin_count)], dtype=np.int64),
        "score_arch": np.asarray([str(args.score_arch)]),
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
    }
    if bool(args.save_cross_scores):
        if cross_scores is not None:
            npz_payload["cross_scores"] = cross_scores.astype(np.float32)
        if score_vectors is not None:
            npz_payload["score_vectors"] = score_vectors.astype(np.float32)
    npz_path = os.path.join(args.output_dir, "score_distance_mds_results.npz")
    np.savez_compressed(npz_path, **npz_payload)

    summary_path = os.path.join(args.output_dir, "score_distance_mds_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Score distance + MDS summary\n")
        f.write(f"dataset_npz: {args.dataset_npz}\n")
        f.write(f"output_dir: {args.output_dir}\n")
        f.write(f"data_split: {args.data_split}\n")
        f.write(f"dsm_conditioning: {args.dsm_conditioning}\n")
        f.write(f"distance_definition: {distance_label}\n")
        f.write(
            "theta_binning: "
            f"bins={int(args.distance_theta_bins)}, low={theta_dist_low:.6f}, high={theta_dist_high:.6f}, "
            f"min_bin_count={int(args.distance_min_bin_count)}\n"
        )
        f.write(f"n_samples: {theta_use.shape[0]}\n")
        f.write(f"x_dim: {x_use.shape[1]}\n")
        f.write(f"score_arch: {args.score_arch}\n")
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
        if cross_scores is not None:
            f.write(
                "cross_score: "
                f"S shape={cross_scores.shape}, row_batch_size={args.cross_row_batch_size}, "
                f"save_cross_scores={bool(args.save_cross_scores)}\n"
            )
        if score_vectors is not None:
            f.write(
                "score_vectors: "
                f"V shape={score_vectors.shape}, row_batch_size={args.cross_row_batch_size}, "
                f"save_cross_scores={bool(args.save_cross_scores)}\n"
            )
        f.write(
            "figure score_distance_mds_theta_color.png (top-left score panel): "
            f"raw score + d with distance_definition={distance_label} "
            "(unnormalized score vectors in x_dim before distances).\n"
        )
        f.write("MDS quality (strain, lower is better):\n")
        f.write(f"  raw score + d:    strain={float(raw['strain_d']):.6f}, positive_eigs={int(raw['positive_eig_d'])}\n")
        f.write(f"  raw score + d^2:  strain={float(raw['strain_d2']):.6f}, positive_eigs={int(raw['positive_eig_d2'])}\n")
        f.write(f"  norm score + d:   strain={float(norm['strain_d']):.6f}, positive_eigs={int(norm['positive_eig_d'])}\n")
        f.write(f"  norm score + d^2: strain={float(norm['strain_d2']):.6f}, positive_eigs={int(norm['positive_eig_d2'])}\n")
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

    print("[score_distance_mds] Saved:")
    print(f"  - {loss_fig_path}")
    print(f"  - {emb_fig_path}")
    print(f"  - {npz_path}")
    print(f"  - {summary_path}")
    print(
        "[score_distance_mds] strains: "
        f"raw_d={float(raw['strain_d']):.6f}, "
        f"raw_d2={float(raw['strain_d2']):.6f}, "
        f"norm_d={float(norm['strain_d']):.6f}, "
        f"norm_d2={float(norm['strain_d2']):.6f}"
    )


if __name__ == "__main__":
    main()
