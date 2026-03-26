#!/usr/bin/env python3
"""Step 3: multi-noise DSM Fisher estimation with sigma->0 extrapolation.

Pipeline:
1) Train one conditional denoising score model with multiple sigma levels.
2) Compute per-sigma Fisher curves from paired held-out samples (no kernel averaging).
3) Extrapolate each theta-bin curve linearly in sigma^2 to estimate Fisher at sigma=0.
4) Compare extrapolated curve against finite-difference baseline.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from step2_toy_dataset_uniform_theta import ToyConditionalGaussianDataset


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_sigma_alpha_list(items: list[float]) -> np.ndarray:
    arr = np.asarray(items, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("sigma alpha list must have at least 2 values.")
    if np.any(arr <= 0):
        raise ValueError("all sigma alpha values must be positive.")
    # Keep deterministic order (large to small is easier to read in logs/plots).
    arr = np.unique(arr)[::-1]
    return arr


class ConditionalScore1D(nn.Module):
    """Score model for s(theta_tilde, x, sigma), with scalar theta."""

    def __init__(self, hidden_dim: int = 128, depth: int = 3) -> None:
        super().__init__()
        in_dim = 1 + 2 + 1  # theta_tilde, x(2), sigma
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, theta_tilde: torch.Tensor, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        feats = torch.cat([theta_tilde, x, sigma], dim=-1)
        return self.net(feats)

    def train_step(
        self,
        theta: torch.Tensor,
        x: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        sigma_values: torch.Tensor,
    ) -> float:
        self.train()
        # Sample sigma independently per sample from discrete sigma grid.
        sigma_idx = torch.randint(
            low=0, high=sigma_values.numel(), size=(theta.shape[0],), device=theta.device
        )
        sigma = sigma_values[sigma_idx].unsqueeze(-1)  # (B,1)
        eps = torch.randn_like(theta)
        theta_tilde = theta + sigma * eps
        target = -(theta_tilde - theta) / (sigma**2)
        pred = self.forward(theta_tilde, x, sigma)
        loss = torch.mean((pred - target) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def predict_score(self, theta: torch.Tensor, x: torch.Tensor, sigma_eval: float) -> torch.Tensor:
        self.eval()
        sigma = torch.full((theta.shape[0], 1), sigma_eval, device=theta.device)
        return self.forward(theta, x, sigma)


def log_p_x_given_theta(x: np.ndarray, theta: np.ndarray, dataset: ToyConditionalGaussianDataset) -> np.ndarray:
    """Analytic log density log p(x|theta) for Gaussian x|theta."""
    mu = dataset.tuning_curve(theta)
    delta = x - mu
    inv_cov = np.linalg.inv(dataset.cov)
    quad = np.einsum("ni,ij,nj->n", delta, inv_cov, delta)
    _, logdet = np.linalg.slogdet(dataset.cov)
    d = x.shape[1]
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)


def finite_difference_score(
    x: np.ndarray,
    theta: np.ndarray,
    dataset: ToyConditionalGaussianDataset,
    delta: float,
) -> np.ndarray:
    theta_plus = theta + delta
    theta_minus = theta - delta
    lp = log_p_x_given_theta(x, theta_plus, dataset)
    lm = log_p_x_given_theta(x, theta_minus, dataset)
    return ((lp - lm) / (2.0 * delta)).reshape(-1)


@dataclass
class BinnedStats:
    centers: np.ndarray
    mean: np.ndarray
    se: np.ndarray
    counts: np.ndarray
    valid: np.ndarray


@dataclass
class BinnedFisher:
    centers: np.ndarray
    fisher_model: np.ndarray
    fisher_fd: np.ndarray
    se_model: np.ndarray
    se_fd: np.ndarray
    counts: np.ndarray
    valid: np.ndarray


def bin_mean_and_se(
    theta: np.ndarray,
    values: np.ndarray,
    theta_low: float,
    theta_high: float,
    n_bins: int,
    min_count: int,
) -> BinnedStats:
    bins = np.linspace(theta_low, theta_high, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    idx = np.digitize(theta.reshape(-1), bins) - 1
    in_range = (idx >= 0) & (idx < n_bins)
    idx = idx[in_range]
    vals = values.reshape(-1)[in_range]

    mean = np.full(n_bins, np.nan, dtype=np.float64)
    se = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        mask = idx == b
        c = int(mask.sum())
        counts[b] = c
        if c < min_count:
            continue
        vv = vals[mask]
        mean[b] = float(np.mean(vv))
        se[b] = float(np.std(vv, ddof=1) / np.sqrt(c)) if c > 1 else np.nan

    valid = np.isfinite(mean)
    return BinnedStats(centers=centers, mean=mean, se=se, counts=counts, valid=valid)


def extrapolate_sigma2_to_zero(
    sigma_values: np.ndarray, fisher_per_sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-theta-bin linear fit y = a + b * sigma^2; returns a, b, r2."""
    x = np.asarray(sigma_values, dtype=np.float64) ** 2
    y = np.asarray(fisher_per_sigma, dtype=np.float64)  # (K, B)
    if y.ndim != 2 or y.shape[0] != x.size:
        raise ValueError("fisher_per_sigma must have shape (n_sigma, n_bins).")

    n_bins = y.shape[1]
    intercept = np.full(n_bins, np.nan, dtype=np.float64)
    slope = np.full(n_bins, np.nan, dtype=np.float64)
    r2 = np.full(n_bins, np.nan, dtype=np.float64)

    for b in range(n_bins):
        yy = y[:, b]
        mask = np.isfinite(yy)
        if int(mask.sum()) < 2:
            continue
        xx = x[mask]
        yb = yy[mask]
        p = np.polyfit(xx, yb, deg=1)
        slope[b] = float(p[0])
        intercept[b] = float(p[1])
        pred = slope[b] * xx + intercept[b]
        ss_res = float(np.sum((yb - pred) ** 2))
        ss_tot = float(np.sum((yb - np.mean(yb)) ** 2))
        r2[b] = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    return intercept, slope, r2


def compute_curve_metrics(curves: BinnedFisher) -> dict[str, float]:
    a = curves.fisher_model[curves.valid]
    b = curves.fisher_fd[curves.valid]
    if a.size == 0:
        return {
            "n_valid_bins": 0.0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "relative_rmse": float("nan"),
            "corr": float("nan"),
        }
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    mae = float(np.mean(np.abs(a - b)))
    rel = float(rmse / (np.mean(np.abs(b)) + 1e-12))
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size >= 2 else float("nan")
    return {
        "n_valid_bins": float(a.size),
        "rmse": rmse,
        "mae": mae,
        "relative_rmse": rel,
        "corr": corr,
    }


def plot_training_loss(losses: list[float], out_path: str) -> None:
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(losses, linewidth=2.0)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Score Model Training Loss (Multi-Sigma DSM)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_fisher_curve(curves: BinnedFisher, out_path: str) -> None:
    x = curves.centers
    m = curves.fisher_model
    f = curves.fisher_fd
    sm = curves.se_model
    sf = curves.se_fd
    valid = curves.valid

    plt.figure(figsize=(8.0, 5.0))
    plt.plot(
        x[valid], m[valid], color="#1f77b4", linewidth=2.4, label=r"Extrapolated $\hat I_{0}(\theta)$"
    )
    plt.plot(x[valid], f[valid], color="#d62728", linewidth=2.1, label="Finite-difference baseline")
    if np.any(np.isfinite(sm[valid])):
        plt.fill_between(
            x[valid],
            m[valid] - 1.96 * sm[valid],
            m[valid] + 1.96 * sm[valid],
            color="#1f77b4",
            alpha=0.15,
            linewidth=0.0,
        )
    if np.any(np.isfinite(sf[valid])):
        plt.fill_between(
            x[valid],
            f[valid] - 1.96 * sf[valid],
            f[valid] + 1.96 * sf[valid],
            color="#d62728",
            alpha=0.12,
            linewidth=0.0,
        )
    plt.xlabel(r"$\theta$ (bin centers)")
    plt.ylabel("Fisher information")
    plt.title(r"Fisher Curve: Extrapolated $\sigma \to 0$ vs Finite-Difference")
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_extrapolation_diagnostics(
    centers: np.ndarray,
    sigma_values: np.ndarray,
    fisher_per_sigma: np.ndarray,
    intercept: np.ndarray,
    slope: np.ndarray,
    valid_bins: np.ndarray,
    out_path: str,
) -> None:
    sigma2 = sigma_values**2
    valid_idx = np.where(valid_bins)[0]
    if valid_idx.size == 0:
        return

    # Pick up to 6 representative bins across theta range.
    pick_n = min(6, valid_idx.size)
    pick = np.linspace(0, valid_idx.size - 1, pick_n).round().astype(int)
    chosen = valid_idx[pick]

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True, sharey=False)
    axes_flat = axes.flatten()
    for ax in axes_flat[pick_n:]:
        ax.axis("off")

    for k, b in enumerate(chosen):
        ax = axes_flat[k]
        yy = fisher_per_sigma[:, b]
        ax.scatter(sigma2, yy, color="#1f77b4", s=30, alpha=0.85, label="bin means")
        xline = np.linspace(0.0, sigma2.max() * 1.05, 100)
        yline = intercept[b] + slope[b] * xline
        ax.plot(xline, yline, color="#d62728", linewidth=1.8, label="linear fit")
        ax.set_title(rf"$\theta\approx{centers[b]:.2f}$")
        ax.grid(alpha=0.25, linestyle=":")
    for ax in axes[1, :]:
        ax.set_xlabel(r"$\sigma^2$")
    axes[0, 0].set_ylabel(r"$I_\sigma(\theta)$")
    axes[1, 0].set_ylabel(r"$I_\sigma(\theta)$")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(r"Extrapolation Diagnostics: $I_\sigma(\theta)$ vs $\sigma^2$", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def to_loader(theta: np.ndarray, x: np.ndarray, batch_size: int) -> DataLoader:
    t = torch.from_numpy(theta.astype(np.float32))
    xx = torch.from_numpy(x.astype(np.float32))
    ds = TensorDataset(t, xx)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 3 multi-sigma Fisher extrapolation (no kernel averaging).")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--sigma-alpha-list", type=float, nargs="+", default=[0.08, 0.06, 0.045, 0.03, 0.02])
    p.add_argument("--n-train", type=int, default=28000)
    p.add_argument("--n-eval", type=int, default=18000)
    p.add_argument("--theta-low", type=float, default=-3.0)
    p.add_argument("--theta-high", type=float, default=3.0)
    p.add_argument("--sigma-x1", type=float, default=0.30)
    p.add_argument("--sigma-x2", type=float, default=0.22)
    p.add_argument("--rho", type=float, default=0.15)
    p.add_argument("--fd-delta", type=float, default=0.03)
    p.add_argument("--n-bins", type=int, default=35)
    p.add_argument("--min-bin-count", type=int, default=80)
    p.add_argument("--eval-margin", type=float, default=0.30)
    p.add_argument("--output-dir", type=str, default="outputs_step3_multi_sigma")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    device = torch.device(args.device)

    dataset = ToyConditionalGaussianDataset(
        theta_low=args.theta_low,
        theta_high=args.theta_high,
        sigma_x1=args.sigma_x1,
        sigma_x2=args.sigma_x2,
        rho=args.rho,
        seed=args.seed,
    )

    theta_train, x_train = dataset.sample_joint(args.n_train)
    theta_eval, x_eval = dataset.sample_joint(args.n_eval)

    theta_std = float(np.std(theta_train))
    sigma_alpha = parse_sigma_alpha_list(args.sigma_alpha_list)
    sigma_values = sigma_alpha * theta_std
    print(f"[sigma] theta_std={theta_std:.6f}")
    print(f"[sigma] alpha grid={sigma_alpha.tolist()}")
    print(f"[sigma] absolute grid={sigma_values.tolist()}")

    loader = to_loader(theta_train, x_train, args.batch_size)
    model = ConditionalScore1D(hidden_dim=args.hidden_dim, depth=args.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sigma_values_t = torch.from_numpy(sigma_values.astype(np.float32)).to(device)

    losses: list[float] = []
    for epoch in range(1, args.epochs + 1):
        epoch_losses: list[float] = []
        for tb, xb in loader:
            tb = tb.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            loss = model.train_step(tb, xb, optimizer, sigma_values=sigma_values_t)
            epoch_losses.append(loss)
        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"[epoch {epoch:4d}/{args.epochs}] train_loss={mean_loss:.6f}")

    # Evaluate model scores on paired held-out samples for each sigma.
    with torch.no_grad():
        t_eval_t = torch.from_numpy(theta_eval.astype(np.float32)).to(device)
        x_eval_t = torch.from_numpy(x_eval.astype(np.float32)).to(device)
        score_model_by_sigma = []
        for s in sigma_values:
            pred = model.predict_score(t_eval_t, x_eval_t, sigma_eval=float(s)).cpu().numpy().reshape(-1)
            score_model_by_sigma.append(pred)
    score_model_by_sigma_arr = np.stack(score_model_by_sigma, axis=0)  # (K, N)
    score_fd = finite_difference_score(x_eval, theta_eval, dataset, delta=args.fd_delta)

    eval_low = args.theta_low + args.eval_margin
    eval_high = args.theta_high - args.eval_margin

    # Baseline Fisher curve from finite-difference scores.
    fd_stats = bin_mean_and_se(
        theta=theta_eval,
        values=score_fd**2,
        theta_low=eval_low,
        theta_high=eval_high,
        n_bins=args.n_bins,
        min_count=args.min_bin_count,
    )

    # Model Fisher curves for each sigma.
    fisher_per_sigma = []
    se_per_sigma = []
    counts_ref = None
    centers_ref = None
    for k in range(score_model_by_sigma_arr.shape[0]):
        st = bin_mean_and_se(
            theta=theta_eval,
            values=score_model_by_sigma_arr[k] ** 2,
            theta_low=eval_low,
            theta_high=eval_high,
            n_bins=args.n_bins,
            min_count=args.min_bin_count,
        )
        fisher_per_sigma.append(st.mean)
        se_per_sigma.append(st.se)
        if counts_ref is None:
            counts_ref = st.counts
            centers_ref = st.centers
    fisher_per_sigma_arr = np.stack(fisher_per_sigma, axis=0)  # (K,B)
    se_per_sigma_arr = np.stack(se_per_sigma, axis=0)

    fisher0, slope, r2 = extrapolate_sigma2_to_zero(sigma_values=sigma_values, fisher_per_sigma=fisher_per_sigma_arr)
    # Model SE proxy at sigma->0: use smallest-sigma SE as an uncertainty proxy.
    se_model0 = se_per_sigma_arr[-1]

    valid = np.isfinite(fisher0) & fd_stats.valid
    curves = BinnedFisher(
        centers=centers_ref,
        fisher_model=fisher0,
        fisher_fd=fd_stats.mean,
        se_model=se_model0,
        se_fd=fd_stats.se,
        counts=counts_ref,
        valid=valid,
    )
    metrics = compute_curve_metrics(curves)

    loss_path = os.path.join(args.output_dir, "training_loss.png")
    fisher_path = os.path.join(args.output_dir, "fisher_curve_extrapolated_vs_fd.png")
    diag_path = os.path.join(args.output_dir, "extrapolation_diagnostics.png")
    metrics_path = os.path.join(args.output_dir, "metrics_extrapolated.txt")
    npz_path = os.path.join(args.output_dir, "binned_fisher_multi_sigma.npz")

    plot_training_loss(losses, loss_path)
    plot_fisher_curve(curves, fisher_path)
    plot_extrapolation_diagnostics(
        centers=centers_ref,
        sigma_values=sigma_values,
        fisher_per_sigma=fisher_per_sigma_arr,
        intercept=fisher0,
        slope=slope,
        valid_bins=valid,
        out_path=diag_path,
    )

    np.savez(
        npz_path,
        centers=centers_ref,
        sigma_values=sigma_values,
        fisher_per_sigma=fisher_per_sigma_arr,
        se_per_sigma=se_per_sigma_arr,
        fisher_extrapolated=fisher0,
        fisher_fd=fd_stats.mean,
        se_fd=fd_stats.se,
        slope=slope,
        r2=r2,
        counts=counts_ref,
        valid=valid.astype(np.int32),
    )

    mean_r2 = float(np.nanmean(r2[valid])) if np.any(valid) else float("nan")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Multi-sigma score-to-Fisher evaluation (no kernel averaging)\n")
        f.write("Extrapolation model: I_sigma(theta)=a(theta)+b(theta)*sigma^2\n")
        f.write(f"sigma_alpha: {sigma_alpha.tolist()}\n")
        f.write(f"sigma_values: {sigma_values.tolist()}\n")
        f.write(f"n_valid_bins: {metrics['n_valid_bins']:.0f}\n")
        f.write(f"rmse: {metrics['rmse']:.6f}\n")
        f.write(f"mae: {metrics['mae']:.6f}\n")
        f.write(f"relative_rmse: {metrics['relative_rmse']:.6f}\n")
        f.write(f"corr: {metrics['corr']:.6f}\n")
        f.write(f"mean_extrapolation_r2: {mean_r2:.6f}\n")

    print("[evaluation]")
    print(f"  valid bins: {int(metrics['n_valid_bins'])}/{args.n_bins}")
    print(f"  rmse: {metrics['rmse']:.6f}")
    print(f"  mae: {metrics['mae']:.6f}")
    print(f"  relative_rmse: {metrics['relative_rmse']:.6f}")
    print(f"  corr: {metrics['corr']:.6f}")
    print(f"  mean extrapolation r2: {mean_r2:.6f}")
    print("Saved artifacts:")
    print(f"  - {loss_path}")
    print(f"  - {fisher_path}")
    print(f"  - {diag_path}")
    print(f"  - {metrics_path}")
    print(f"  - {npz_path}")


if __name__ == "__main__":
    main()
