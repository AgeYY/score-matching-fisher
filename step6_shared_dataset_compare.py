#!/usr/bin/env python3
"""Step 6: shared-dataset comparison against analytic Fisher ground truth.

Workflow:
1) Sample one joint dataset (theta, x) from p(theta)p(x|theta), then split train/eval.
2) Fit score-matching Fisher estimator on the shared train split.
3) Fit decoder local-classification Fisher estimator using shared train/eval subsets.
4) Compute analytic Fisher curve and compare all curves in one figure.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from fisher.data import ToyConditionalGaussianDataset
from fisher.evaluation import evaluate_score_fisher, parse_sigma_alpha_list
from fisher.models import ConditionalScore1D, LocalDecoderLogit
from fisher.trainers import train_local_decoder, train_score_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shared-dataset score-vs-decoder comparison with analytic GT.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--theta-low", type=float, default=-3.0)
    p.add_argument("--theta-high", type=float, default=3.0)
    p.add_argument("--sigma-x1", type=float, default=0.30)
    p.add_argument("--sigma-x2", type=float, default=0.22)
    p.add_argument("--rho", type=float, default=0.15)
    p.add_argument("--n-total", type=int, default=60000)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--device", type=str, default="cuda")

    # Score method args.
    p.add_argument("--score-epochs", type=int, default=120)
    p.add_argument("--score-batch-size", type=int, default=256)
    p.add_argument("--score-lr", type=float, default=1e-3)
    p.add_argument("--score-hidden-dim", type=int, default=128)
    p.add_argument("--score-depth", type=int, default=3)
    p.add_argument("--score-sigma-alpha-list", type=float, nargs="+", default=[0.08, 0.06, 0.045, 0.03, 0.02])

    # Shared eval curve settings.
    p.add_argument("--n-bins", type=int, default=35)
    p.add_argument("--eval-margin", type=float, default=0.30)
    p.add_argument("--score-min-bin-count", type=int, default=80)
    p.add_argument("--fd-delta", type=float, default=0.03)

    # Decoder local-classification settings (from shared dataset neighborhoods).
    p.add_argument("--decoder-epsilon", type=float, default=0.12)
    p.add_argument("--decoder-bandwidth", type=float, default=0.10)
    p.add_argument("--decoder-epochs", type=int, default=80)
    p.add_argument("--decoder-batch-size", type=int, default=256)
    p.add_argument("--decoder-lr", type=float, default=1e-3)
    p.add_argument("--decoder-hidden-dim", type=int, default=64)
    p.add_argument("--decoder-depth", type=int, default=2)
    p.add_argument("--decoder-min-class-count", type=int, default=60)
    p.add_argument("--decoder-train-cap", type=int, default=1200)
    p.add_argument("--decoder-eval-cap", type=int, default=1200)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--output-dir", type=str, default="outputs_step6_shared_dataset")
    return p.parse_args()


def require_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Per repo policy, do not fallback silently.")
    return torch.device(name)


def analytic_fisher_curve(centers: np.ndarray, dataset: ToyConditionalGaussianDataset) -> np.ndarray:
    t = centers.reshape(-1, 1)
    dmu1 = 1.10 * 1.25 * np.cos(1.25 * t) + 0.28
    dmu2 = -0.85 * 1.05 * np.sin(1.05 * t + 0.30) - 0.24 * t + 0.05
    dmu = np.concatenate([dmu1, dmu2], axis=1)  # (B,2)
    inv_cov = np.linalg.inv(dataset.cov)
    fisher = np.einsum("bi,ij,bj->b", dmu, inv_cov, dmu)
    return fisher.astype(np.float64)


def compute_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    a = pred[valid]
    b = gt[valid]
    if a.size == 0:
        return {"n_valid": 0.0, "rmse": float("nan"), "mae": float("nan"), "corr": float("nan")}
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    mae = float(np.mean(np.abs(a - b)))
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size >= 2 else float("nan")
    return {"n_valid": float(a.size), "rmse": rmse, "mae": mae, "corr": corr}


def _subset_x_by_theta(
    theta: np.ndarray,
    x: np.ndarray,
    target: float,
    bandwidth: float,
    cap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    idx = np.where(np.abs(theta.reshape(-1) - target) <= bandwidth)[0]
    if idx.size == 0:
        return np.zeros((0, x.shape[1]), dtype=np.float64)
    if cap > 0 and idx.size > cap:
        idx = rng.choice(idx, size=cap, replace=False)
    return x[idx]


def fit_decoder_from_shared_data(
    centers: np.ndarray,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_eval: np.ndarray,
    x_eval: np.ndarray,
    epsilon: float,
    bandwidth: float,
    min_class_count: int,
    train_cap: int,
    eval_cap: int,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    depth: int,
    device: torch.device,
    log_every: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fisher = np.full(centers.size, np.nan, dtype=np.float64)
    se = np.full(centers.size, np.nan, dtype=np.float64)
    valid = np.zeros(centers.size, dtype=bool)

    for i, theta0 in enumerate(centers):
        theta_plus = float(theta0 + 0.5 * epsilon)
        theta_minus = float(theta0 - 0.5 * epsilon)

        xtr_pos = _subset_x_by_theta(theta_train, x_train, theta_plus, bandwidth, train_cap, rng)
        xtr_neg = _subset_x_by_theta(theta_train, x_train, theta_minus, bandwidth, train_cap, rng)
        xev_pos = _subset_x_by_theta(theta_eval, x_eval, theta_plus, bandwidth, eval_cap, rng)
        xev_neg = _subset_x_by_theta(theta_eval, x_eval, theta_minus, bandwidth, eval_cap, rng)

        ntr = min(xtr_pos.shape[0], xtr_neg.shape[0])
        nev = min(xev_pos.shape[0], xev_neg.shape[0])
        if ntr < min_class_count or nev < min_class_count:
            continue

        if xtr_pos.shape[0] != ntr:
            xtr_pos = xtr_pos[rng.choice(xtr_pos.shape[0], size=ntr, replace=False)]
        if xtr_neg.shape[0] != ntr:
            xtr_neg = xtr_neg[rng.choice(xtr_neg.shape[0], size=ntr, replace=False)]
        if xev_pos.shape[0] != nev:
            xev_pos = xev_pos[rng.choice(xev_pos.shape[0], size=nev, replace=False)]
        if xev_neg.shape[0] != nev:
            xev_neg = xev_neg[rng.choice(xev_neg.shape[0], size=nev, replace=False)]

        xtr = np.concatenate([xtr_pos, xtr_neg], axis=0)
        ytr = np.concatenate([np.ones(ntr, dtype=np.float64), np.zeros(ntr, dtype=np.float64)], axis=0)

        model = LocalDecoderLogit(hidden_dim=hidden_dim, depth=depth).to(device)
        _ = train_local_decoder(
            model=model,
            x_train=xtr,
            y_train=ytr,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )
        model.eval()
        with torch.no_grad():
            mix = np.concatenate([xev_pos, xev_neg], axis=0)
            mix_t = torch.from_numpy(mix.astype(np.float32)).to(device)
            logits_mix = model(mix_t).cpu().numpy().reshape(-1)
        fisher_samples = (logits_mix**2) / (epsilon**2)
        fisher[i] = float(np.mean(fisher_samples))
        se[i] = float(np.std(fisher_samples, ddof=1) / np.sqrt(fisher_samples.size))
        valid[i] = True

        if i == 0 or (i + 1) % log_every == 0 or (i + 1) == centers.size:
            print(
                f"[decoder theta {i+1:3d}/{centers.size}] theta0={theta0:+.3f} "
                f"ntr={ntr} nev={nev} fisher={fisher[i]:.4f}"
            )

    return fisher, se, valid


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = require_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dataset = ToyConditionalGaussianDataset(
        theta_low=args.theta_low,
        theta_high=args.theta_high,
        sigma_x1=args.sigma_x1,
        sigma_x2=args.sigma_x2,
        rho=args.rho,
        seed=args.seed,
    )

    theta_all, x_all = dataset.sample_joint(args.n_total)
    perm = rng.permutation(args.n_total)
    n_train = int(args.train_frac * args.n_total)
    n_train = min(max(n_train, 1), args.n_total - 1)
    tr_idx = perm[:n_train]
    ev_idx = perm[n_train:]
    theta_train, x_train = theta_all[tr_idx], x_all[tr_idx]
    theta_eval, x_eval = theta_all[ev_idx], x_all[ev_idx]

    print(f"[data] total={args.n_total} train={theta_train.shape[0]} eval={theta_eval.shape[0]}")

    # Score method on shared split.
    sigma_alpha = parse_sigma_alpha_list(args.score_sigma_alpha_list)
    theta_std = float(np.std(theta_train))
    sigma_values = sigma_alpha * theta_std
    print(f"[score] theta_std={theta_std:.6f}, sigma_values={sigma_values.tolist()}")
    score_model = ConditionalScore1D(hidden_dim=args.score_hidden_dim, depth=args.score_depth).to(device)
    score_losses = train_score_model(
        model=score_model,
        theta_train=theta_train,
        x_train=x_train,
        sigma_values=sigma_values,
        epochs=args.score_epochs,
        batch_size=args.score_batch_size,
        lr=args.score_lr,
        device=device,
        log_every=max(1, args.log_every),
    )
    eval_low = args.theta_low + args.eval_margin
    eval_high = args.theta_high - args.eval_margin
    score_eval = evaluate_score_fisher(
        model=score_model,
        theta_eval=theta_eval,
        x_eval=x_eval,
        dataset=dataset,
        sigma_values=sigma_values,
        fd_delta=args.fd_delta,
        n_bins=args.n_bins,
        min_bin_count=args.score_min_bin_count,
        eval_low=eval_low,
        eval_high=eval_high,
        device=device,
    )
    centers = score_eval.curves.centers

    # Decoder method from same shared split.
    decoder_fisher, decoder_se, decoder_valid = fit_decoder_from_shared_data(
        centers=centers,
        theta_train=theta_train,
        x_train=x_train,
        theta_eval=theta_eval,
        x_eval=x_eval,
        epsilon=args.decoder_epsilon,
        bandwidth=args.decoder_bandwidth,
        min_class_count=args.decoder_min_class_count,
        train_cap=args.decoder_train_cap,
        eval_cap=args.decoder_eval_cap,
        epochs=args.decoder_epochs,
        batch_size=args.decoder_batch_size,
        lr=args.decoder_lr,
        hidden_dim=args.decoder_hidden_dim,
        depth=args.decoder_depth,
        device=device,
        log_every=max(1, args.log_every),
        rng=rng,
    )

    gt = analytic_fisher_curve(centers, dataset)
    score_valid = np.isfinite(score_eval.curves.fisher_model) & score_eval.curves.valid

    score_metrics = compute_metrics(score_eval.curves.fisher_model, gt, score_valid)
    decoder_metrics = compute_metrics(decoder_fisher, gt, decoder_valid)

    fig_path = os.path.join(args.output_dir, "fisher_curve_shared_dataset_vs_analytic.png")
    plt.figure(figsize=(9.0, 5.6))
    plt.plot(centers, gt, color="black", linewidth=2.6, label="Analytic Fisher (GT)")
    plt.plot(
        centers[score_valid],
        score_eval.curves.fisher_model[score_valid],
        color="#1f77b4",
        linewidth=2.2,
        label=r"Score matching ($\sigma\to0$ extrapolated)",
    )
    if np.any(np.isfinite(score_eval.curves.se_model[score_valid])):
        plt.fill_between(
            centers[score_valid],
            score_eval.curves.fisher_model[score_valid] - 1.96 * score_eval.curves.se_model[score_valid],
            score_eval.curves.fisher_model[score_valid] + 1.96 * score_eval.curves.se_model[score_valid],
            color="#1f77b4",
            alpha=0.12,
            linewidth=0.0,
        )
    plt.plot(
        centers[decoder_valid],
        decoder_fisher[decoder_valid],
        color="#2ca02c",
        linewidth=2.2,
        label="Decoder local classification (shared data)",
    )
    if np.any(np.isfinite(decoder_se[decoder_valid])):
        plt.fill_between(
            centers[decoder_valid],
            decoder_fisher[decoder_valid] - 1.96 * decoder_se[decoder_valid],
            decoder_fisher[decoder_valid] + 1.96 * decoder_se[decoder_valid],
            color="#2ca02c",
            alpha=0.12,
            linewidth=0.0,
        )
    plt.xlabel(r"$\theta$")
    plt.ylabel("Fisher information")
    plt.title("Shared Dataset Comparison: Score vs Decoder vs Analytic GT")
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

    npz_path = os.path.join(args.output_dir, "shared_dataset_compare_curves.npz")
    np.savez(
        npz_path,
        centers=centers,
        fisher_gt=gt,
        fisher_score=score_eval.curves.fisher_model,
        fisher_score_se=score_eval.curves.se_model,
        fisher_score_valid=score_valid.astype(np.int32),
        fisher_decoder=decoder_fisher,
        fisher_decoder_se=decoder_se,
        fisher_decoder_valid=decoder_valid.astype(np.int32),
        score_losses=np.asarray(score_losses, dtype=np.float64),
    )

    metrics_path = os.path.join(args.output_dir, "metrics_vs_analytic.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Shared dataset Fisher comparison against analytic GT\n")
        f.write(f"n_total: {args.n_total}\n")
        f.write(f"train_frac: {args.train_frac}\n")
        f.write(f"decoder_epsilon: {args.decoder_epsilon}\n")
        f.write(f"decoder_bandwidth: {args.decoder_bandwidth}\n")
        f.write(
            "score_vs_gt: "
            f"valid={int(score_metrics['n_valid'])}/{args.n_bins}, "
            f"rmse={score_metrics['rmse']:.6f}, "
            f"mae={score_metrics['mae']:.6f}, "
            f"corr={score_metrics['corr']:.6f}\n"
        )
        f.write(
            "decoder_vs_gt: "
            f"valid={int(decoder_metrics['n_valid'])}/{args.n_bins}, "
            f"rmse={decoder_metrics['rmse']:.6f}, "
            f"mae={decoder_metrics['mae']:.6f}, "
            f"corr={decoder_metrics['corr']:.6f}\n"
        )

    print("[summary]")
    print(
        "  score vs GT: "
        f"valid={int(score_metrics['n_valid'])}/{args.n_bins}, "
        f"rmse={score_metrics['rmse']:.4f}, mae={score_metrics['mae']:.4f}, corr={score_metrics['corr']:.4f}"
    )
    print(
        "  decoder vs GT: "
        f"valid={int(decoder_metrics['n_valid'])}/{args.n_bins}, "
        f"rmse={decoder_metrics['rmse']:.4f}, mae={decoder_metrics['mae']:.4f}, corr={decoder_metrics['corr']:.4f}"
    )
    print("Saved artifacts:")
    print(f"  - {fig_path}")
    print(f"  - {npz_path}")
    print(f"  - {metrics_path}")


if __name__ == "__main__":
    main()
