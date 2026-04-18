#!/usr/bin/env python3
"""Generate and visualize samples from two 1D Gaussians (no unittest)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.ctsm_models import ToyPairConditionedTimeScoreNetFiLM
from fisher.ctsm_objectives import ctsm_v_pair_conditioned_loss, estimate_log_ratio_trapz_pair
from fisher.ctsm_paths import TwoSB
from fisher.h_matrix import HMatrixEstimator
from fisher.models import ConditionalThetaFlowVelocity, PriorThetaFlowVelocity
from fisher.trainers import train_conditional_theta_flow_model, train_prior_theta_flow_model
from global_setting import DATAROOT


DEFAULT_OUTPUT = Path(DATAROOT) / "tests" / "gaussian_two_dist_viz.png"
DEFAULT_RATIO_OUTPUT = Path(DATAROOT) / "tests" / "gaussian_two_dist_log_ratio_100.csv"
DEFAULT_CTSM_RATIO_OUTPUT = Path(DATAROOT) / "tests" / "gaussian_two_dist_ctsm_log_ratio_100.csv"
DEFAULT_CTSM_LOSS_OUTPUT = Path(DATAROOT) / "tests" / "gaussian_two_dist_ctsm_loss.png"
DEFAULT_CTSM_SCATTER_OUTPUT = Path(DATAROOT) / "tests" / "gaussian_two_dist_ctsm_scatter.png"
DEFAULT_CTSM_SUMMARY_OUTPUT = Path(DATAROOT) / "tests" / "gaussian_two_dist_ctsm_training_summary.json"
DEFAULT_FLOW_RATIO_OUTPUT = Path(DATAROOT) / "tests" / "gaussian_two_dist_theta_flow_log_ratio_100.csv"
DEFAULT_FLOW_POST_LOSS_OUTPUT = Path(DATAROOT) / "tests" / "gaussian_two_dist_theta_flow_post_loss.png"
DEFAULT_FLOW_PRIOR_LOSS_OUTPUT = Path(DATAROOT) / "tests" / "gaussian_two_dist_theta_flow_prior_loss.png"
DEFAULT_FLOW_SCATTER_OUTPUT = Path(DATAROOT) / "tests" / "gaussian_two_dist_theta_flow_scatter.png"
DEFAULT_FLOW_SUMMARY_OUTPUT = Path(DATAROOT) / "tests" / "gaussian_two_dist_theta_flow_training_summary.json"
DEFAULT_COMPARE_CSV_OUTPUT = Path(DATAROOT) / "tests" / "gaussian_two_dist_compare_log_ratio.csv"
DEFAULT_COMPARE_JSON_OUTPUT = Path(DATAROOT) / "tests" / "gaussian_two_dist_compare_summary.json"
DEFAULT_COMPARE_SCATTER_OUTPUT = Path(DATAROOT) / "tests" / "gaussian_two_dist_compare_scatter.png"

# ConditionalThetaFlowVelocity requires x_dim >= 2; pad 1D x with a zero second coordinate.
FLOW_X_DIM = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample from N(-1,1) and N(1,1), then save a visualization.")
    p.add_argument("--n-per-dist", type=int, default=2000, help="Number of samples per Gaussian.")
    p.add_argument("--n-ratio", type=int, default=100, help="Number of points for log-likelihood-ratio evaluation.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device for sampling ('cuda' or 'cpu'). Default is cuda.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output PNG path.",
    )
    p.add_argument(
        "--ratio-output",
        type=str,
        default=str(DEFAULT_RATIO_OUTPUT),
        help="Output CSV path for log-likelihood-ratio samples.",
    )
    p.add_argument("--enable-ctsm", action="store_true", help="Enable pair-conditioned CTSM-v estimation.")
    p.add_argument("--ctsm-steps", type=int, default=None, help="Deprecated alias for --ctsm-max-epochs.")
    p.add_argument("--ctsm-max-epochs", type=int, default=5000, help="CTSM-v maximum training epochs.")
    p.add_argument("--ctsm-batch-size", type=int, default=512, help="CTSM-v batch size.")
    p.add_argument("--ctsm-hidden-dim", type=int, default=128, help="CTSM-v hidden dimension.")
    p.add_argument("--ctsm-lr", type=float, default=2e-3, help="CTSM-v learning rate.")
    p.add_argument("--ctsm-two-sb-var", type=float, default=2.0, help="CTSM-v TwoSB bridge variance.")
    p.add_argument("--ctsm-factor", type=float, default=1.0, help="CTSM-v target factor.")
    p.add_argument("--ctsm-t-eps", type=float, default=1e-5, help="CTSM-v sampled time clamp.")
    p.add_argument("--ctsm-n-time", type=int, default=200, help="Trapezoid points for CTSM-v ratio integration.")
    p.add_argument("--ctsm-val-pool-size", type=int, default=4096, help="Fixed held-out validation pool size.")
    p.add_argument("--ctsm-val-batches-per-epoch", type=int, default=8, help="Validation mini-batches per epoch.")
    p.add_argument("--ctsm-early-patience", type=int, default=500, help="Early stopping patience on EMA monitor.")
    p.add_argument("--ctsm-early-min-delta", type=float, default=1e-4, help="Minimum EMA improvement to reset patience.")
    p.add_argument("--ctsm-early-ema-alpha", type=float, default=0.05, help="EMA alpha for validation monitor.")
    p.add_argument(
        "--ctsm-early-ema-warmup-epochs",
        type=int,
        default=0,
        help="Warmup epochs that use raw val loss as monitor before EMA updates.",
    )
    p.add_argument(
        "--ctsm-no-restore-best",
        action="store_true",
        help="If set, do not restore the best EMA checkpoint at the end.",
    )
    p.add_argument("--ctsm-film-depth", type=int, default=3, help="FiLM network depth.")
    p.add_argument("--ctsm-film-gated", action="store_true", help="Use tanh-gated FiLM modulation.")
    p.add_argument("--ctsm-film-use-raw-time", action="store_true", help="Use raw t (not logit(t)) in FiLM conditioning.")
    p.add_argument("--ctsm-m-scale", type=float, default=1.0, help="Scale for m=(a+b)/2 conditioning.")
    p.add_argument("--ctsm-delta-scale", type=float, default=0.5, help="Scale for delta=(b-a) conditioning.")
    p.add_argument("--ctsm-log-every", type=int, default=100, help="Progress logging interval.")
    p.add_argument(
        "--ctsm-ratio-output",
        type=str,
        default=str(DEFAULT_CTSM_RATIO_OUTPUT),
        help="Output CSV path for CTSM-v ratio estimates.",
    )
    p.add_argument(
        "--ctsm-loss-output",
        type=str,
        default=str(DEFAULT_CTSM_LOSS_OUTPUT),
        help="Output PNG path for CTSM-v training loss curve.",
    )
    p.add_argument(
        "--ctsm-scatter-output",
        type=str,
        default=str(DEFAULT_CTSM_SCATTER_OUTPUT),
        help="Output PNG path for analytic-vs-CTSM ratio scatter.",
    )
    p.add_argument(
        "--ctsm-summary-output",
        type=str,
        default=str(DEFAULT_CTSM_SUMMARY_OUTPUT),
        help="Output JSON path for CTSM-v early-stopping/training summary.",
    )
    p.add_argument(
        "--enable-theta-flow",
        action="store_true",
        help="Enable theta-space flow (ODE likelihood Bayes ratio) vs analytic log-ratio.",
    )
    p.add_argument("--flow-train-n", type=int, default=12000, help="Number of (theta,x) pairs for theta-flow training.")
    p.add_argument("--flow-val-n", type=int, default=2048, help="Held-out pairs for theta-flow validation / early stopping.")
    p.add_argument("--flow-epochs", type=int, default=5000, help="Max epochs for posterior theta-flow training.")
    p.add_argument("--flow-prior-epochs", type=int, default=5000, help="Max epochs for prior theta-flow training.")
    p.add_argument("--flow-batch-size", type=int, default=512, help="Theta-flow training batch size.")
    p.add_argument("--flow-lr", type=float, default=1e-3, help="Theta-flow Adam LR (posterior).")
    p.add_argument("--flow-prior-lr", type=float, default=1e-3, help="Theta-flow Adam LR (prior).")
    p.add_argument("--flow-hidden-dim", type=int, default=128, help="Hidden width for theta-flow MLPs.")
    p.add_argument("--flow-depth", type=int, default=3, help="Depth for theta-flow MLPs.")
    p.add_argument(
        "--flow-scheduler",
        type=str,
        default="cosine",
        help="flow_matching path scheduler for theta-flow training (cosine, vp, linear_vp).",
    )
    p.add_argument(
        "--flow-eval-t",
        type=float,
        default=0.8,
        help="t_eval passed to HMatrixEstimator for theta_flow (must be in [0,1]; ODE uses flow-ode-steps).",
    )
    p.add_argument("--flow-ode-steps", type=int, default=64, help="ODE steps for theta_flow likelihood (>=2).")
    p.add_argument("--flow-early-patience", type=int, default=500, help="Early stopping patience (posterior / prior).")
    p.add_argument("--flow-early-min-delta", type=float, default=1e-4, help="Min improvement for early stopping.")
    p.add_argument("--flow-early-ema-alpha", type=float, default=0.05, help="EMA alpha for validation monitor.")
    p.add_argument(
        "--flow-no-restore-best",
        action="store_true",
        help="If set, do not restore best theta-flow checkpoint after training.",
    )
    p.add_argument("--flow-log-every", type=int, default=100, help="Progress logging interval for theta-flow training.")
    p.add_argument(
        "--flow-ratio-output",
        type=str,
        default=str(DEFAULT_FLOW_RATIO_OUTPUT),
        help="Output CSV for theta-flow log-ratio estimates.",
    )
    p.add_argument(
        "--flow-post-loss-output",
        type=str,
        default=str(DEFAULT_FLOW_POST_LOSS_OUTPUT),
        help="Output PNG for posterior theta-flow training loss.",
    )
    p.add_argument(
        "--flow-prior-loss-output",
        type=str,
        default=str(DEFAULT_FLOW_PRIOR_LOSS_OUTPUT),
        help="Output PNG for prior theta-flow training loss.",
    )
    p.add_argument(
        "--flow-scatter-output",
        type=str,
        default=str(DEFAULT_FLOW_SCATTER_OUTPUT),
        help="Output PNG for analytic vs theta-flow scatter.",
    )
    p.add_argument(
        "--flow-summary-output",
        type=str,
        default=str(DEFAULT_FLOW_SUMMARY_OUTPUT),
        help="Output JSON for theta-flow training summary.",
    )
    p.add_argument(
        "--compare-csv-output",
        type=str,
        default=str(DEFAULT_COMPARE_CSV_OUTPUT),
        help="Unified comparison CSV (analytic vs enabled methods).",
    )
    p.add_argument(
        "--compare-json-output",
        type=str,
        default=str(DEFAULT_COMPARE_JSON_OUTPUT),
        help="Unified comparison metrics JSON.",
    )
    p.add_argument(
        "--compare-scatter-output",
        type=str,
        default=str(DEFAULT_COMPARE_SCATTER_OUTPUT),
        help="Combined scatter figure for enabled methods vs analytic.",
    )
    return p.parse_args()


def gaussian_log_prob_unit_var(x: torch.Tensor, mean: float) -> torch.Tensor:
    return -0.5 * math.log(2.0 * math.pi) - 0.5 * (x - mean) ** 2


def format_via_data_symlink(abs_path: Path) -> Path:
    display_path = abs_path
    data_symlink = _REPO_ROOT / "data"
    if data_symlink.exists():
        try:
            data_real = data_symlink.resolve()
            out_real = abs_path.resolve()
            if str(out_real).startswith(str(data_real) + "/"):
                display_path = data_symlink / out_real.relative_to(data_real)
        except OSError:
            pass
    return display_path


def compute_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return float("nan")
    if torch.std(x, unbiased=False) == 0 or torch.std(y, unbiased=False) == 0:
        return float("nan")
    corr_tensor = torch.corrcoef(torch.stack([x, y]))
    return float(corr_tensor[0, 1].item())


def pad_x_to_flow_dim(x1: np.ndarray) -> np.ndarray:
    """Stack [x, 0] so x_dim == FLOW_X_DIM for ConditionalThetaFlowVelocity."""
    x1 = np.asarray(x1, dtype=np.float64).reshape(-1, 1)
    z = np.zeros_like(x1, dtype=np.float64)
    return np.concatenate([x1, z], axis=1)


def sample_theta_x_flow_dataset(
    n: int,
    *,
    rng: np.random.Generator,
    mean_a: float,
    mean_b: float,
    std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Balanced mixture: theta in {mean_a, mean_b}, x ~ N(theta, std^2). Returns theta (N,1), x_pad (N,2)."""
    if n < 2:
        raise ValueError("flow dataset size must be >= 2.")
    half = n // 2
    rest = n - 2 * half
    theta_list: list[float] = [mean_a] * half + [mean_b] * half
    if rest > 0:
        extra = rng.choice([mean_a, mean_b], size=rest)
        theta_list.extend(float(t) for t in extra)
    rng.shuffle(theta_list)
    theta = np.asarray(theta_list, dtype=np.float64).reshape(-1, 1)
    noise = rng.standard_normal(size=(n, 1)) * float(std)
    x1 = theta + noise
    x_pad = pad_x_to_flow_dim(x1)
    return theta, x_pad


def method_metrics(analytic: torch.Tensor, estimate: torch.Tensor) -> dict[str, float]:
    diff = estimate - analytic
    return {
        "mse": float(torch.mean(diff**2).item()),
        "mae": float(torch.mean(torch.abs(diff)).item()),
        "corr": float(compute_corr(analytic, estimate)),
        "mean_bias": float(torch.mean(diff).item()),
        "std_estimate": float(estimate.std(unbiased=False).item()),
    }


def theta_flow_bayes_log_ratio_for_x(
    *,
    estimator: HMatrixEstimator,
    x_scalar: float,
) -> float:
    """Match HMatrixEstimator.compute_log_ratio_matrix: r[i,j]=log p(theta_j|x_i)-log p(theta_j).

    For theta in {-1,+1} and fixed x, returns r[0,1]-r[0,0] == log p(x|+1)-log p(x|-1) when the flow is exact.
    """
    theta_sorted = np.asarray([[-1.0], [1.0]], dtype=np.float64)
    x2 = np.asarray([[float(x_scalar), 0.0], [float(x_scalar), 0.0]], dtype=np.float64)
    r = estimator.compute_log_ratio_matrix(theta_sorted, x2)
    return float(r[0, 1] - r[0, 0])


def main() -> None:
    args = parse_args()
    ctsm_max_epochs = int(args.ctsm_steps) if args.ctsm_steps is not None else int(args.ctsm_max_epochs)
    ctsm_restore_best = not bool(args.ctsm_no_restore_best)
    flow_restore_best = not bool(args.flow_no_restore_best)
    if args.n_per_dist <= 0:
        raise ValueError("--n-per-dist must be > 0.")
    if args.n_ratio <= 0:
        raise ValueError("--n-ratio must be > 0.")
    if args.n_ratio % 2 != 0:
        raise ValueError("--n-ratio must be even for balanced 50/50 sampling.")
    if args.enable_ctsm:
        if ctsm_max_epochs <= 0:
            raise ValueError("--ctsm-max-epochs must be > 0 (or --ctsm-steps when provided).")
        if args.ctsm_batch_size <= 0:
            raise ValueError("--ctsm-batch-size must be > 0.")
        if args.ctsm_hidden_dim <= 0:
            raise ValueError("--ctsm-hidden-dim must be > 0.")
        if args.ctsm_n_time < 2:
            raise ValueError("--ctsm-n-time must be >= 2.")
        if args.ctsm_log_every <= 0:
            raise ValueError("--ctsm-log-every must be > 0.")
        if args.ctsm_val_pool_size < 2:
            raise ValueError("--ctsm-val-pool-size must be >= 2.")
        if args.ctsm_val_pool_size % 2 != 0:
            raise ValueError("--ctsm-val-pool-size must be even.")
        if args.ctsm_val_batches_per_epoch <= 0:
            raise ValueError("--ctsm-val-batches-per-epoch must be > 0.")
        if args.ctsm_early_patience <= 0:
            raise ValueError("--ctsm-early-patience must be > 0.")
        if args.ctsm_early_ema_warmup_epochs < 0:
            raise ValueError("--ctsm-early-ema-warmup-epochs must be >= 0.")
        if not (0.0 < float(args.ctsm_early_ema_alpha) <= 1.0):
            raise ValueError("--ctsm-early-ema-alpha must be in (0, 1].")
        if not (0.0 <= args.ctsm_t_eps < 0.5):
            raise ValueError("--ctsm-t-eps must be in [0, 0.5).")
    if args.enable_theta_flow:
        if int(args.flow_train_n) < 2:
            raise ValueError("--flow-train-n must be >= 2.")
        if int(args.flow_val_n) < 1:
            raise ValueError("--flow-val-n must be >= 1.")
        if int(args.flow_train_n) + int(args.flow_val_n) < 4:
            raise ValueError("Need flow-train-n + flow-val-n >= 4 for train/val split.")
        if int(args.flow_epochs) <= 0 or int(args.flow_prior_epochs) <= 0:
            raise ValueError("--flow-epochs and --flow-prior-epochs must be > 0.")
        if args.flow_batch_size <= 0:
            raise ValueError("--flow-batch-size must be > 0.")
        if args.flow_hidden_dim <= 0 or args.flow_depth <= 0:
            raise ValueError("--flow-hidden-dim and --flow-depth must be > 0.")
        if not (0.0 <= float(args.flow_eval_t) <= 1.0):
            raise ValueError("--flow-eval-t must be in [0, 1].")
        if int(args.flow_ode_steps) < 2:
            raise ValueError("--flow-ode-steps must be >= 2.")
        if args.flow_early_patience <= 0:
            raise ValueError("--flow-early-patience must be > 0.")
        if not (0.0 < float(args.flow_early_ema_alpha) <= 1.0):
            raise ValueError("--flow-early-ema-alpha must be in (0, 1].")
        if args.flow_log_every <= 0:
            raise ValueError("--flow-log-every must be > 0.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested (`--device cuda`) but CUDA is unavailable on this machine.")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    std = 1.0
    mean_a = -1.0
    mean_b = 1.0

    x_a = mean_a + std * torch.randn(args.n_per_dist, device=device)
    x_b = mean_b + std * torch.randn(args.n_per_dist, device=device)

    x_a_cpu = x_a.detach().cpu()
    x_b_cpu = x_b.detach().cpu()

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ratio_out_path = Path(args.ratio_output).resolve()
    ratio_out_path.parent.mkdir(parents=True, exist_ok=True)
    ctsm_ratio_out_path = Path(args.ctsm_ratio_output).resolve()
    ctsm_ratio_out_path.parent.mkdir(parents=True, exist_ok=True)
    ctsm_loss_out_path = Path(args.ctsm_loss_output).resolve()
    ctsm_loss_out_path.parent.mkdir(parents=True, exist_ok=True)
    ctsm_scatter_out_path = Path(args.ctsm_scatter_output).resolve()
    ctsm_scatter_out_path.parent.mkdir(parents=True, exist_ok=True)
    ctsm_summary_out_path = Path(args.ctsm_summary_output).resolve()
    ctsm_summary_out_path.parent.mkdir(parents=True, exist_ok=True)
    flow_ratio_out_path = Path(args.flow_ratio_output).resolve()
    flow_post_loss_out_path = Path(args.flow_post_loss_output).resolve()
    flow_prior_loss_out_path = Path(args.flow_prior_loss_output).resolve()
    flow_scatter_out_path = Path(args.flow_scatter_output).resolve()
    flow_summary_out_path = Path(args.flow_summary_output).resolve()
    compare_csv_out_path = Path(args.compare_csv_output).resolve()
    compare_json_out_path = Path(args.compare_json_output).resolve()
    compare_scatter_out_path = Path(args.compare_scatter_output).resolve()
    for p in (
        flow_ratio_out_path,
        flow_post_loss_out_path,
        flow_prior_loss_out_path,
        flow_scatter_out_path,
        flow_summary_out_path,
        compare_csv_out_path,
        compare_json_out_path,
        compare_scatter_out_path,
    ):
        p.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    bins = 60
    plt.hist(x_a_cpu.numpy(), bins=bins, density=True, alpha=0.55, label="N(-1, 1)")
    plt.hist(x_b_cpu.numpy(), bins=bins, density=True, alpha=0.55, label="N(1, 1)")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title("Two Gaussian Distributions (variance = 1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    n_half = args.n_ratio // 2
    x_ratio_a = mean_a + std * torch.randn(n_half, device=device)
    x_ratio_b = mean_b + std * torch.randn(n_half, device=device)
    x_ratio = torch.cat([x_ratio_a, x_ratio_b], dim=0)
    labels = torch.cat(
        [
            torch.full((n_half,), -1.0, device=device),
            torch.full((n_half,), 1.0, device=device),
        ],
        dim=0,
    )
    perm = torch.randperm(args.n_ratio, device=device)
    x_ratio = x_ratio[perm]
    labels = labels[perm]
    logp_mu1 = gaussian_log_prob_unit_var(x_ratio, mean=1.0)
    logp_mu_minus1 = gaussian_log_prob_unit_var(x_ratio, mean=-1.0)
    log_ratio = logp_mu1 - logp_mu_minus1

    x_ratio_cpu = x_ratio.detach().cpu()
    labels_cpu = labels.detach().cpu()
    log_ratio_cpu = log_ratio.detach().cpu()

    with ratio_out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "source_mean", "logp_mu1_minus_logp_mu_minus1"])
        for x_i, src_i, lr_i in zip(x_ratio_cpu.tolist(), labels_cpu.tolist(), log_ratio_cpu.tolist()):
            writer.writerow([f"{x_i:.10f}", f"{src_i:.1f}", f"{lr_i:.10f}"])

    ctsm_loss_history: list[float] = []
    ctsm_val_losses: list[float] = []
    ctsm_val_monitor_losses: list[float] = []
    ctsm_ratio_cpu = torch.full_like(log_ratio_cpu, float("nan"))
    ctsm_metrics: dict[str, float] = {}
    ctsm_training_summary: dict[str, float | int | bool] = {}
    tf_ratio_cpu = torch.full_like(log_ratio_cpu, float("nan"))
    tf_metrics: dict[str, float] = {}
    tf_training_summary: dict[str, float | int | bool | dict[str, float | int | bool]] = {}
    if args.enable_ctsm:
        prob_path = TwoSB(dim=1, var=float(args.ctsm_two_sb_var))
        ctsm_model = ToyPairConditionedTimeScoreNetFiLM(
            dim=1,
            hidden_dim=int(args.ctsm_hidden_dim),
            depth=int(args.ctsm_film_depth),
            m_scale=float(args.ctsm_m_scale),
            delta_scale=float(args.ctsm_delta_scale),
            use_logit_time=not bool(args.ctsm_film_use_raw_time),
            gated_film=bool(args.ctsm_film_gated),
        ).to(device)
        optimizer = torch.optim.Adam(ctsm_model.parameters(), lr=float(args.ctsm_lr))
        val_half = int(args.ctsm_val_pool_size) // 2
        x_val_a = mean_a + std * torch.randn(val_half, 1, device=device)
        x_val_b = mean_b + std * torch.randn(val_half, 1, device=device)
        a_const_train = torch.full((int(args.ctsm_batch_size), 1), -1.0, device=device)
        b_const_train = torch.full((int(args.ctsm_batch_size), 1), 1.0, device=device)
        a_const_val = torch.full((int(args.ctsm_batch_size), 1), -1.0, device=device)
        b_const_val = torch.full((int(args.ctsm_batch_size), 1), 1.0, device=device)

        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0
        best_monitor = float("inf")
        ema_monitor: float | None = None
        patience_bad = 0
        stopped_early = False

        progress = tqdm(range(ctsm_max_epochs), desc="pair-conditioned CTSM-v (FiLM)")
        for step in progress:
            x0 = mean_a + std * torch.randn(int(args.ctsm_batch_size), 1, device=device)
            x1 = mean_b + std * torch.randn(int(args.ctsm_batch_size), 1, device=device)
            loss = ctsm_v_pair_conditioned_loss(
                model=ctsm_model,
                prob_path=prob_path,
                x0=x0,
                x1=x1,
                a=a_const_train,
                b=b_const_train,
                factor=float(args.ctsm_factor),
                t_eps=float(args.ctsm_t_eps),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            ctsm_loss_history.append(loss_value)
            ctsm_model.eval()
            val_epoch_losses: list[float] = []
            with torch.no_grad():
                for _ in range(int(args.ctsm_val_batches_per_epoch)):
                    ia_v = torch.randint(0, val_half, (int(args.ctsm_batch_size),), device=device)
                    ib_v = torch.randint(0, val_half, (int(args.ctsm_batch_size),), device=device)
                    x0_v = x_val_a[ia_v]
                    x1_v = x_val_b[ib_v]
                    v_loss = ctsm_v_pair_conditioned_loss(
                        model=ctsm_model,
                        prob_path=prob_path,
                        x0=x0_v,
                        x1=x1_v,
                        a=a_const_val,
                        b=b_const_val,
                        factor=float(args.ctsm_factor),
                        t_eps=float(args.ctsm_t_eps),
                    )
                    val_epoch_losses.append(float(v_loss.item()))
            val_loss_value = float(sum(val_epoch_losses) / max(1, len(val_epoch_losses)))
            ctsm_val_losses.append(val_loss_value)

            epoch = step + 1
            if ema_monitor is None:
                ema_monitor = val_loss_value
            elif epoch <= int(args.ctsm_early_ema_warmup_epochs):
                ema_monitor = val_loss_value
            else:
                alpha = float(args.ctsm_early_ema_alpha)
                ema_monitor = alpha * val_loss_value + (1.0 - alpha) * ema_monitor
            ctsm_val_monitor_losses.append(float(ema_monitor))

            improved = float(ema_monitor) < (best_monitor - float(args.ctsm_early_min_delta))
            if improved:
                best_monitor = float(ema_monitor)
                best_epoch = int(epoch)
                patience_bad = 0
                if ctsm_restore_best:
                    best_state = deepcopy(ctsm_model.state_dict())
            else:
                patience_bad += 1

            if step % int(args.ctsm_log_every) == 0:
                progress.set_postfix(train=f"{loss_value:.4f}", val=f"{val_loss_value:.4f}", ema=f"{float(ema_monitor):.4f}")
            if patience_bad >= int(args.ctsm_early_patience):
                stopped_early = True
                break

        stopped_epoch = int(len(ctsm_loss_history))
        if ctsm_restore_best and best_state is not None:
            ctsm_model.load_state_dict(best_state)
        if best_epoch <= 0:
            best_epoch = stopped_epoch
            best_monitor = float(ctsm_val_monitor_losses[-1]) if ctsm_val_monitor_losses else float("nan")

        x_ratio_ctsm = x_ratio.unsqueeze(-1)
        a_eval = torch.full((args.n_ratio, 1), -1.0, device=device)
        b_eval = torch.full((args.n_ratio, 1), 1.0, device=device)
        ctsm_ratio = estimate_log_ratio_trapz_pair(
            ctsm_model,
            x_ratio_ctsm,
            a_eval,
            b_eval,
            n_time=int(args.ctsm_n_time),
            eps1=float(args.ctsm_t_eps),
            eps2=float(args.ctsm_t_eps),
        )
        ctsm_ratio_cpu = ctsm_ratio.detach().cpu()

        abs_error = torch.abs(ctsm_ratio_cpu - log_ratio_cpu)
        ctsm_metrics = {
            "mse": float(torch.mean((ctsm_ratio_cpu - log_ratio_cpu) ** 2).item()),
            "mae": float(abs_error.mean().item()),
            "corr": float(compute_corr(log_ratio_cpu, ctsm_ratio_cpu)),
            "final_loss": float(ctsm_loss_history[-1]),
        }
        ctsm_training_summary = {
            "max_epochs": int(ctsm_max_epochs),
            "stopped_epoch": int(stopped_epoch),
            "best_epoch": int(best_epoch),
            "stopped_early": bool(stopped_early),
            "restore_best": bool(ctsm_restore_best),
            "best_val_ema": float(best_monitor),
            "final_train_loss": float(ctsm_loss_history[-1]),
            "final_val_loss": float(ctsm_val_losses[-1]),
            "final_val_ema": float(ctsm_val_monitor_losses[-1]),
            "early_patience": int(args.ctsm_early_patience),
            "early_min_delta": float(args.ctsm_early_min_delta),
            "early_ema_alpha": float(args.ctsm_early_ema_alpha),
            "early_ema_warmup_epochs": int(args.ctsm_early_ema_warmup_epochs),
        }
        with ctsm_summary_out_path.open("w", encoding="utf-8") as f:
            json.dump(ctsm_training_summary, f, indent=2, sort_keys=True)

        with ctsm_ratio_out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "source_mean", "analytic_log_ratio", "ctsm_log_ratio", "abs_error"])
            for x_i, src_i, a_i, c_i, e_i in zip(
                x_ratio_cpu.tolist(),
                labels_cpu.tolist(),
                log_ratio_cpu.tolist(),
                ctsm_ratio_cpu.tolist(),
                abs_error.tolist(),
            ):
                writer.writerow([f"{x_i:.10f}", f"{src_i:.1f}", f"{a_i:.10f}", f"{c_i:.10f}", f"{e_i:.10f}"])

        plt.figure(figsize=(6, 3.5))
        plt.plot(ctsm_loss_history, label="train")
        plt.plot(ctsm_val_losses, label="val")
        plt.plot(ctsm_val_monitor_losses, linestyle="--", label=f"val EMA (alpha={args.ctsm_early_ema_alpha:g})")
        plt.xlabel("step")
        plt.ylabel("CTSM-v loss")
        plt.title("Pair-conditioned FiLM CTSM-v training")
        plt.legend()
        plt.tight_layout()
        plt.savefig(ctsm_loss_out_path, dpi=150)
        plt.close()

        plt.figure(figsize=(4.5, 4.5))
        plt.scatter(
            log_ratio_cpu.numpy(),
            ctsm_ratio_cpu.numpy(),
            s=14,
            alpha=0.65,
        )
        lo = min(float(log_ratio_cpu.min().item()), float(ctsm_ratio_cpu.min().item()))
        hi = max(float(log_ratio_cpu.max().item()), float(ctsm_ratio_cpu.max().item()))
        plt.plot([lo, hi], [lo, hi], "--", color="gray")
        plt.xlabel("analytic log-ratio")
        plt.ylabel("CTSM-v log-ratio")
        plt.title("CTSM-v estimate vs analytic")
        plt.tight_layout()
        plt.savefig(ctsm_scatter_out_path, dpi=150)
        plt.close()

    if args.enable_theta_flow:
        rng = np.random.default_rng(int(args.seed) + 17)
        n_total = int(args.flow_train_n) + int(args.flow_val_n)
        theta_all_np, x_all_np = sample_theta_x_flow_dataset(
            n_total,
            rng=rng,
            mean_a=mean_a,
            mean_b=mean_b,
            std=std,
        )
        theta_train_np = theta_all_np[: int(args.flow_train_n)]
        x_train_np = x_all_np[: int(args.flow_train_n)]
        theta_val_np = theta_all_np[int(args.flow_train_n) :]
        x_val_np = x_all_np[int(args.flow_train_n) :]

        post_model = ConditionalThetaFlowVelocity(
            x_dim=int(FLOW_X_DIM),
            hidden_dim=int(args.flow_hidden_dim),
            depth=int(args.flow_depth),
            use_logit_time=True,
        ).to(device)
        prior_model = PriorThetaFlowVelocity(
            hidden_dim=int(args.flow_hidden_dim),
            depth=int(args.flow_depth),
            use_logit_time=True,
        ).to(device)

        post_out = train_conditional_theta_flow_model(
            model=post_model,
            theta_train=theta_train_np,
            x_train=x_train_np,
            epochs=int(args.flow_epochs),
            batch_size=int(args.flow_batch_size),
            lr=float(args.flow_lr),
            device=device,
            log_every=int(args.flow_log_every),
            theta_val=theta_val_np,
            x_val=x_val_np,
            early_stopping_patience=int(args.flow_early_patience),
            early_stopping_min_delta=float(args.flow_early_min_delta),
            early_stopping_ema_alpha=float(args.flow_early_ema_alpha),
            restore_best=bool(flow_restore_best),
            scheduler_name=str(args.flow_scheduler),
        )
        prior_out = train_prior_theta_flow_model(
            model=prior_model,
            theta_train=theta_train_np,
            epochs=int(args.flow_prior_epochs),
            batch_size=int(args.flow_batch_size),
            lr=float(args.flow_prior_lr),
            device=device,
            log_every=int(args.flow_log_every),
            theta_val=theta_val_np,
            early_stopping_patience=int(args.flow_early_patience),
            early_stopping_min_delta=float(args.flow_early_min_delta),
            early_stopping_ema_alpha=float(args.flow_early_ema_alpha),
            restore_best=bool(flow_restore_best),
            scheduler_name=str(args.flow_scheduler),
        )

        post_train_losses = post_out["train_losses"]
        post_val_losses = post_out["val_losses"]
        post_val_monitor = post_out.get("val_monitor_losses", [])
        prior_train_losses = prior_out["train_losses"]
        prior_val_losses = prior_out["val_losses"]
        prior_val_monitor = prior_out.get("val_monitor_losses", [])

        epochs_post = np.arange(1, len(post_train_losses) + 1)
        plt.figure(figsize=(6, 3.5))
        plt.plot(epochs_post, post_train_losses, label="train")
        if len(post_val_losses) == len(post_train_losses) and np.any(np.isfinite(post_val_losses)):
            plt.plot(epochs_post, post_val_losses, label="val")
        if len(post_val_monitor) == len(post_train_losses) and np.any(np.isfinite(post_val_monitor)):
            plt.plot(epochs_post, post_val_monitor, linestyle="--", label=f"val EMA (α={args.flow_early_ema_alpha:g})")
        plt.xlabel("epoch")
        plt.ylabel("flow-matching loss")
        plt.title("Posterior theta-flow training (conditional)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(flow_post_loss_out_path, dpi=150)
        plt.close()

        epochs_prior = np.arange(1, len(prior_train_losses) + 1)
        plt.figure(figsize=(6, 3.5))
        plt.plot(epochs_prior, prior_train_losses, label="train")
        if len(prior_val_losses) == len(prior_train_losses) and np.any(np.isfinite(prior_val_losses)):
            plt.plot(epochs_prior, prior_val_losses, label="val")
        if len(prior_val_monitor) == len(prior_train_losses) and np.any(np.isfinite(prior_val_monitor)):
            plt.plot(epochs_prior, prior_val_monitor, linestyle="--", label=f"val EMA (α={args.flow_early_ema_alpha:g})")
        plt.xlabel("epoch")
        plt.ylabel("flow-matching loss")
        plt.title("Prior theta-flow training")
        plt.legend()
        plt.tight_layout()
        plt.savefig(flow_prior_loss_out_path, dpi=150)
        plt.close()

        h_estimator = HMatrixEstimator(
            model_post=post_model,
            model_prior=prior_model,
            sigma_eval=float(args.flow_eval_t),
            device=device,
            pair_batch_size=max(64, int(args.n_ratio) * 4),
            field_method="theta_flow",
            flow_scheduler=str(args.flow_scheduler),
            flow_ode_steps=int(args.flow_ode_steps),
        )
        tf_list: list[float] = []
        for i in range(int(args.n_ratio)):
            tf_list.append(
                theta_flow_bayes_log_ratio_for_x(
                    estimator=h_estimator,
                    x_scalar=float(x_ratio_cpu[i].item()),
                )
            )
        tf_ratio_cpu = torch.tensor(tf_list, dtype=torch.float32)

        abs_tf = torch.abs(tf_ratio_cpu - log_ratio_cpu)
        tf_metrics = {
            **method_metrics(log_ratio_cpu, tf_ratio_cpu),
            "final_post_train_loss": float(post_train_losses[-1]) if len(post_train_losses) else float("nan"),
            "final_prior_train_loss": float(prior_train_losses[-1]) if len(prior_train_losses) else float("nan"),
        }
        tf_training_summary = {
            "flow_train_n": int(args.flow_train_n),
            "flow_val_n": int(args.flow_val_n),
            "flow_epochs_post": int(args.flow_epochs),
            "flow_epochs_prior": int(args.flow_prior_epochs),
            "flow_batch_size": int(args.flow_batch_size),
            "flow_lr_post": float(args.flow_lr),
            "flow_lr_prior": float(args.flow_prior_lr),
            "flow_hidden_dim": int(args.flow_hidden_dim),
            "flow_depth": int(args.flow_depth),
            "flow_scheduler": str(args.flow_scheduler),
            "flow_eval_t": float(args.flow_eval_t),
            "flow_ode_steps": int(args.flow_ode_steps),
            "restore_best": bool(flow_restore_best),
            "posterior_best_epoch": int(post_out.get("best_epoch", 0)),
            "posterior_stopped_epoch": int(post_out.get("stopped_epoch", 0)),
            "posterior_stopped_early": bool(post_out.get("stopped_early", False)),
            "posterior_best_val_loss": float(post_out.get("best_val_loss", float("nan"))),
            "prior_best_epoch": int(prior_out.get("best_epoch", 0)),
            "prior_stopped_epoch": int(prior_out.get("stopped_epoch", 0)),
            "prior_stopped_early": bool(prior_out.get("stopped_early", False)),
            "prior_best_val_loss": float(prior_out.get("best_val_loss", float("nan"))),
        }

        with flow_summary_out_path.open("w", encoding="utf-8") as f:
            json.dump({"training": tf_training_summary, "metrics_vs_analytic": tf_metrics}, f, indent=2, sort_keys=True)

        with flow_ratio_out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "source_mean", "analytic_log_ratio", "theta_flow_log_ratio", "abs_error"])
            for x_i, src_i, a_i, t_i, e_i in zip(
                x_ratio_cpu.tolist(),
                labels_cpu.tolist(),
                log_ratio_cpu.tolist(),
                tf_ratio_cpu.tolist(),
                abs_tf.tolist(),
            ):
                writer.writerow([f"{x_i:.10f}", f"{src_i:.1f}", f"{a_i:.10f}", f"{t_i:.10f}", f"{e_i:.10f}"])

        plt.figure(figsize=(4.5, 4.5))
        plt.scatter(log_ratio_cpu.numpy(), tf_ratio_cpu.numpy(), s=14, alpha=0.65)
        lo = min(float(log_ratio_cpu.min().item()), float(tf_ratio_cpu.min().item()))
        hi = max(float(log_ratio_cpu.max().item()), float(tf_ratio_cpu.max().item()))
        plt.plot([lo, hi], [lo, hi], "--", color="gray")
        plt.xlabel("analytic log-ratio")
        plt.ylabel("theta-flow log-ratio")
        plt.title("Theta-flow (Bayes) vs analytic")
        plt.tight_layout()
        plt.savefig(flow_scatter_out_path, dpi=150)
        plt.close()

    if args.enable_ctsm or args.enable_theta_flow:
        compare_payload: dict[str, object] = {
            "ratio_convention": "log p(x|mu=1,var=1) - log p(x|mu=-1,var=1)",
            "theta_flow_note": (
                "theta_flow column uses (log p(+1|x)-log p(-1|x)) - (log p(+1)-log p(-1)) "
                "from ODE likelihoods; equals log p(x|+1)-log p(x|-1) when flows match the model."
            ),
            "n_ratio": int(args.n_ratio),
            "seed": int(args.seed),
            "analytic": {
                "mean": float(log_ratio_cpu.mean().item()),
                "std": float(log_ratio_cpu.std(unbiased=False).item()),
            },
        }
        if args.enable_ctsm:
            compare_payload["ctsm_v"] = {"enabled": True, **{k: float(v) for k, v in ctsm_metrics.items()}}
        else:
            compare_payload["ctsm_v"] = {"enabled": False}
        if args.enable_theta_flow:
            compare_payload["theta_flow"] = {"enabled": True, **{k: float(v) for k, v in tf_metrics.items()}}
        else:
            compare_payload["theta_flow"] = {"enabled": False}

        if args.enable_ctsm and args.enable_theta_flow:
            compare_payload["head_to_head"] = {
                "mse_ctsm": float(ctsm_metrics["mse"]),
                "mse_theta_flow": float(tf_metrics["mse"]),
                "mse_delta_ctsm_minus_theta_flow": float(ctsm_metrics["mse"] - tf_metrics["mse"]),
                "mae_ctsm": float(ctsm_metrics["mae"]),
                "mae_theta_flow": float(tf_metrics["mae"]),
                "corr_ctsm": float(ctsm_metrics["corr"]),
                "corr_theta_flow": float(tf_metrics["corr"]),
            }

        with compare_json_out_path.open("w", encoding="utf-8") as f:
            json.dump(compare_payload, f, indent=2, sort_keys=True)

        with compare_csv_out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "x",
                    "source_mean",
                    "analytic_log_ratio",
                    "ctsm_log_ratio",
                    "theta_flow_log_ratio",
                    "abs_err_ctsm",
                    "abs_err_theta_flow",
                ]
            )
            for i in range(int(args.n_ratio)):
                x_i = float(x_ratio_cpu[i].item())
                src_i = float(labels_cpu[i].item())
                a_i = float(log_ratio_cpu[i].item())
                c_i = float(ctsm_ratio_cpu[i].item()) if args.enable_ctsm else float("nan")
                t_i = float(tf_ratio_cpu[i].item()) if args.enable_theta_flow else float("nan")
                e_c = float(abs(c_i - a_i)) if args.enable_ctsm else float("nan")
                e_t = float(abs(t_i - a_i)) if args.enable_theta_flow else float("nan")
                writer.writerow(
                    [
                        f"{x_i:.10f}",
                        f"{src_i:.1f}",
                        f"{a_i:.10f}",
                        f"{c_i:.10f}" if args.enable_ctsm else "",
                        f"{t_i:.10f}" if args.enable_theta_flow else "",
                        f"{e_c:.10f}" if args.enable_ctsm else "",
                        f"{e_t:.10f}" if args.enable_theta_flow else "",
                    ]
                )

        if args.enable_ctsm and args.enable_theta_flow:
            fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
            lo = float(log_ratio_cpu.min().item())
            hi = float(log_ratio_cpu.max().item())
            ax0, ax1 = axes[0], axes[1]
            ax0.scatter(log_ratio_cpu.numpy(), ctsm_ratio_cpu.numpy(), s=14, alpha=0.65)
            ax0.plot([lo, hi], [lo, hi], "--", color="gray")
            ax0.set_xlabel("analytic log-ratio")
            ax0.set_ylabel("CTSM-v log-ratio")
            ax0.set_title("CTSM-v vs analytic")
            ax1.scatter(log_ratio_cpu.numpy(), tf_ratio_cpu.numpy(), s=14, alpha=0.65, color="#d62728")
            ax1.plot([lo, hi], [lo, hi], "--", color="gray")
            ax1.set_xlabel("analytic log-ratio")
            ax1.set_ylabel("theta-flow log-ratio")
            ax1.set_title("Theta-flow vs analytic")
            plt.suptitle("Gaussian two-dist: method comparison")
            plt.tight_layout()
            plt.savefig(compare_scatter_out_path, dpi=150)
            plt.close()
        elif args.enable_ctsm:
            plt.figure(figsize=(4.5, 4.5))
            plt.scatter(log_ratio_cpu.numpy(), ctsm_ratio_cpu.numpy(), s=14, alpha=0.65)
            lo = min(float(log_ratio_cpu.min().item()), float(ctsm_ratio_cpu.min().item()))
            hi = max(float(log_ratio_cpu.max().item()), float(ctsm_ratio_cpu.max().item()))
            plt.plot([lo, hi], [lo, hi], "--", color="gray")
            plt.xlabel("analytic log-ratio")
            plt.ylabel("CTSM-v log-ratio")
            plt.title("Comparison: CTSM-v vs analytic")
            plt.tight_layout()
            plt.savefig(compare_scatter_out_path, dpi=150)
            plt.close()
        elif args.enable_theta_flow:
            plt.figure(figsize=(4.5, 4.5))
            plt.scatter(log_ratio_cpu.numpy(), tf_ratio_cpu.numpy(), s=14, alpha=0.65, color="#d62728")
            lo = min(float(log_ratio_cpu.min().item()), float(tf_ratio_cpu.min().item()))
            hi = max(float(log_ratio_cpu.max().item()), float(tf_ratio_cpu.max().item()))
            plt.plot([lo, hi], [lo, hi], "--", color="gray")
            plt.xlabel("analytic log-ratio")
            plt.ylabel("theta-flow log-ratio")
            plt.title("Comparison: theta-flow vs analytic")
            plt.tight_layout()
            plt.savefig(compare_scatter_out_path, dpi=150)
            plt.close()

    display_output = format_via_data_symlink(out_path)
    display_ratio_output = format_via_data_symlink(ratio_out_path)
    display_ctsm_ratio_output = format_via_data_symlink(ctsm_ratio_out_path)
    display_ctsm_loss_output = format_via_data_symlink(ctsm_loss_out_path)
    display_ctsm_scatter_output = format_via_data_symlink(ctsm_scatter_out_path)
    display_ctsm_summary_output = format_via_data_symlink(ctsm_summary_out_path)
    display_flow_ratio_output = format_via_data_symlink(flow_ratio_out_path)
    display_flow_post_loss_output = format_via_data_symlink(flow_post_loss_out_path)
    display_flow_prior_loss_output = format_via_data_symlink(flow_prior_loss_out_path)
    display_flow_scatter_output = format_via_data_symlink(flow_scatter_out_path)
    display_flow_summary_output = format_via_data_symlink(flow_summary_out_path)
    display_compare_csv_output = format_via_data_symlink(compare_csv_out_path)
    display_compare_json_output = format_via_data_symlink(compare_json_out_path)
    display_compare_scatter_output = format_via_data_symlink(compare_scatter_out_path)

    print(f"device={device}")
    print(f"n_per_dist={args.n_per_dist}")
    print(f"n_ratio={args.n_ratio}")
    print("ratio_convention=log p(x|mu=1,var=1) - log p(x|mu=-1,var=1)")
    print(f"output={display_output}")
    print(f"output_abs={out_path}")
    print(f"ratio_output={display_ratio_output}")
    print(f"ratio_output_abs={ratio_out_path}")
    print(
        "dist_a: mean={:.6f}, var={:.6f}".format(
            float(x_a_cpu.mean().item()),
            float(x_a_cpu.var(unbiased=False).item()),
        )
    )
    print(
        "dist_b: mean={:.6f}, var={:.6f}".format(
            float(x_b_cpu.mean().item()),
            float(x_b_cpu.var(unbiased=False).item()),
        )
    )
    print(
        "log_ratio: mean={:.6f}, std={:.6f}, min={:.6f}, max={:.6f}".format(
            float(log_ratio_cpu.mean().item()),
            float(log_ratio_cpu.std(unbiased=False).item()),
            float(log_ratio_cpu.min().item()),
            float(log_ratio_cpu.max().item()),
        )
    )
    print("labels: dist_a -> -1, dist_b -> +1")
    print("first_10_x_log_ratio:")
    for idx, (x_i, lr_i) in enumerate(zip(x_ratio_cpu[:10].tolist(), log_ratio_cpu[:10].tolist())):
        print(f"  {idx:02d}: x={x_i:.6f}, log_ratio={lr_i:.6f}")

    if args.enable_ctsm:
        print("ctsm_enabled=True")
        print("ctsm_model=ToyPairConditionedTimeScoreNetFiLM")
        print(
            "ctsm_config: max_epochs={} batch_size={} hidden_dim={} lr={} two_sb_var={} n_time={} t_eps={}".format(
                ctsm_max_epochs,
                args.ctsm_batch_size,
                args.ctsm_hidden_dim,
                args.ctsm_lr,
                args.ctsm_two_sb_var,
                args.ctsm_n_time,
                args.ctsm_t_eps,
            )
        )
        print(
            "ctsm_early_stop: stopped_early={} stopped_epoch={} best_epoch={} best_val_ema={:.6f} restore_best={}".format(
                bool(ctsm_training_summary.get("stopped_early", False)),
                int(ctsm_training_summary.get("stopped_epoch", 0)),
                int(ctsm_training_summary.get("best_epoch", 0)),
                float(ctsm_training_summary.get("best_val_ema", float("nan"))),
                bool(ctsm_training_summary.get("restore_best", ctsm_restore_best)),
            )
        )
        print(
            "ctsm_metrics: final_loss={:.6f}, mse={:.6f}, mae={:.6f}, corr={:.6f}".format(
                ctsm_metrics["final_loss"],
                ctsm_metrics["mse"],
                ctsm_metrics["mae"],
                ctsm_metrics["corr"],
            )
        )
        print(f"ctsm_ratio_output={display_ctsm_ratio_output}")
        print(f"ctsm_ratio_output_abs={ctsm_ratio_out_path}")
        print(f"ctsm_loss_output={display_ctsm_loss_output}")
        print(f"ctsm_loss_output_abs={ctsm_loss_out_path}")
        print(f"ctsm_scatter_output={display_ctsm_scatter_output}")
        print(f"ctsm_scatter_output_abs={ctsm_scatter_out_path}")
        print(f"ctsm_summary_output={display_ctsm_summary_output}")
        print(f"ctsm_summary_output_abs={ctsm_summary_out_path}")
        print("first_10_x_ctsm_log_ratio:")
        for idx, (x_i, lr_i) in enumerate(zip(x_ratio_cpu[:10].tolist(), ctsm_ratio_cpu[:10].tolist())):
            print(f"  {idx:02d}: x={x_i:.6f}, ctsm_log_ratio={lr_i:.6f}")
    else:
        print("ctsm_enabled=False")

    if args.enable_theta_flow:
        print("theta_flow_enabled=True")
        print(
            "theta_flow_config: train_n={} val_n={} post_epochs={} prior_epochs={} batch={} "
            "lr_post={} lr_prior={} hidden={} depth={} scheduler={} flow_eval_t={} ode_steps={}".format(
                int(args.flow_train_n),
                int(args.flow_val_n),
                int(args.flow_epochs),
                int(args.flow_prior_epochs),
                int(args.flow_batch_size),
                float(args.flow_lr),
                float(args.flow_prior_lr),
                int(args.flow_hidden_dim),
                int(args.flow_depth),
                str(args.flow_scheduler),
                float(args.flow_eval_t),
                int(args.flow_ode_steps),
            )
        )
        print(
            "theta_flow_metrics: mse={:.6f}, mae={:.6f}, corr={:.6f}, mean_bias={:.6f}, std_est={:.6f}".format(
                float(tf_metrics["mse"]),
                float(tf_metrics["mae"]),
                float(tf_metrics["corr"]),
                float(tf_metrics["mean_bias"]),
                float(tf_metrics["std_estimate"]),
            )
        )
        print(f"theta_flow_ratio_output={display_flow_ratio_output}")
        print(f"theta_flow_ratio_output_abs={flow_ratio_out_path}")
        print(f"theta_flow_post_loss_output={display_flow_post_loss_output}")
        print(f"theta_flow_post_loss_output_abs={flow_post_loss_out_path}")
        print(f"theta_flow_prior_loss_output={display_flow_prior_loss_output}")
        print(f"theta_flow_prior_loss_output_abs={flow_prior_loss_out_path}")
        print(f"theta_flow_scatter_output={display_flow_scatter_output}")
        print(f"theta_flow_scatter_output_abs={flow_scatter_out_path}")
        print(f"theta_flow_summary_output={display_flow_summary_output}")
        print(f"theta_flow_summary_output_abs={flow_summary_out_path}")
        print("first_10_x_theta_flow_log_ratio:")
        for idx, (x_i, lr_i) in enumerate(zip(x_ratio_cpu[:10].tolist(), tf_ratio_cpu[:10].tolist())):
            print(f"  {idx:02d}: x={x_i:.6f}, theta_flow_log_ratio={lr_i:.6f}")
    else:
        print("theta_flow_enabled=False")

    if args.enable_ctsm or args.enable_theta_flow:
        print(f"compare_csv_output={display_compare_csv_output}")
        print(f"compare_csv_output_abs={compare_csv_out_path}")
        print(f"compare_json_output={display_compare_json_output}")
        print(f"compare_json_output_abs={compare_json_out_path}")
        print(f"compare_scatter_output={display_compare_scatter_output}")
        print(f"compare_scatter_output_abs={compare_scatter_out_path}")
        if args.enable_ctsm and args.enable_theta_flow:
            print(
                "head_to_head: mse_ctsm={:.6f} mse_theta_flow={:.6f} delta_mse(ctsm-flow)={:.6f}".format(
                    float(ctsm_metrics["mse"]),
                    float(tf_metrics["mse"]),
                    float(ctsm_metrics["mse"] - tf_metrics["mse"]),
                )
            )


if __name__ == "__main__":
    main()
