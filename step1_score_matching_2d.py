#!/usr/bin/env python3
"""Step 1: conditional denoising score matching on a 2D toy problem.

This script learns the posterior score in theta-space using paired (x, theta)
samples and validates the learned score against an analytic ground truth.
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


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_joint(n: int, sigma_x: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample from theta ~ N(0, I), x | theta ~ N(theta, sigma_x^2 I)."""
    theta = torch.randn(n, 2, device=device)
    x = theta + sigma_x * torch.randn(n, 2, device=device)
    return x, theta


def analytic_smoothed_posterior_score(
    theta_tilde: torch.Tensor,
    x: torch.Tensor,
    sigma_x: float,
    sigma_dsm: float,
) -> torch.Tensor:
    """Analytic score of p_sigma(theta_tilde | x) for the Gaussian toy model."""
    # Posterior theta|x is Gaussian with mean alpha*x and variance v*I.
    alpha = 1.0 / (1.0 + sigma_x**2)
    v = sigma_x**2 / (1.0 + sigma_x**2)
    smooth_var = v + sigma_dsm**2
    mean = alpha * x
    return -(theta_tilde - mean) / smooth_var


class ConditionalScoreModel(nn.Module):
    """MLP score network for s_phi(theta_tilde, x, sigma)."""

    def __init__(self, hidden_dim: int = 128, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 2 + 2 + 1  # theta_tilde (2), x (2), sigma (1)
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, theta_tilde: torch.Tensor, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        feats = torch.cat([theta_tilde, x, sigma], dim=-1)
        return self.net(feats)

    def train_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        sigma_dsm: float,
    ) -> float:
        self.train()
        x, theta = batch
        eps = torch.randn_like(theta)
        theta_tilde = theta + sigma_dsm * eps
        target = -(theta_tilde - theta) / (sigma_dsm**2)
        sigma = torch.full((theta.shape[0], 1), sigma_dsm, device=theta.device)
        pred = self.forward(theta_tilde, x, sigma)
        loss = torch.mean((pred - target) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def predict_score(self, theta_tilde: torch.Tensor, x: torch.Tensor, sigma_dsm: float) -> torch.Tensor:
        self.eval()
        sigma = torch.full((theta_tilde.shape[0], 1), sigma_dsm, device=theta_tilde.device)
        return self.forward(theta_tilde, x, sigma)


@dataclass
class EvalMetrics:
    mse: float
    cosine_mean: float


@torch.no_grad()
def evaluate_model(
    model: ConditionalScoreModel,
    x: torch.Tensor,
    theta: torch.Tensor,
    sigma_x: float,
    sigma_dsm: float,
) -> EvalMetrics:
    eps = torch.randn_like(theta)
    theta_tilde = theta + sigma_dsm * eps
    pred = model.predict_score(theta_tilde, x, sigma_dsm)
    true = analytic_smoothed_posterior_score(theta_tilde, x, sigma_x=sigma_x, sigma_dsm=sigma_dsm)
    mse = float(torch.mean((pred - true) ** 2).item())
    pred_n = torch.linalg.norm(pred, dim=-1).clamp_min(1e-8)
    true_n = torch.linalg.norm(true, dim=-1).clamp_min(1e-8)
    cosine = torch.sum(pred * true, dim=-1) / (pred_n * true_n)
    return EvalMetrics(mse=mse, cosine_mean=float(torch.mean(cosine).item()))


def build_loader(x: torch.Tensor, theta: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(x.cpu(), theta.cpu())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def plot_loss(train_losses: list[float], out_path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Denoising Score Matching Loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


@torch.no_grad()
def plot_score_fields(
    model: ConditionalScoreModel,
    sigma_x: float,
    sigma_dsm: float,
    out_quiver_path: str,
    out_err_path: str,
    device: torch.device,
) -> None:
    # Fix one observation x0 and inspect score field over theta grid.
    x0 = torch.tensor([[1.5, -1.0]], device=device)
    grid_min, grid_max, grid_n = -3.0, 3.0, 25
    g = torch.linspace(grid_min, grid_max, grid_n, device=device)
    gx, gy = torch.meshgrid(g, g, indexing="ij")
    theta_grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    x_rep = x0.repeat(theta_grid.shape[0], 1)

    pred = model.predict_score(theta_grid, x_rep, sigma_dsm)
    true = analytic_smoothed_posterior_score(theta_grid, x_rep, sigma_x=sigma_x, sigma_dsm=sigma_dsm)
    err = torch.linalg.norm(pred - true, dim=-1).reshape(grid_n, grid_n).cpu().numpy()

    gx_np = gx.cpu().numpy()
    gy_np = gy.cpu().numpy()
    pred_np = pred.reshape(grid_n, grid_n, 2).cpu().numpy()
    true_np = true.reshape(grid_n, grid_n, 2).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    axes[0].quiver(gx_np, gy_np, true_np[..., 0], true_np[..., 1], color="tab:blue", angles="xy")
    axes[0].set_title("True Smoothed Score Field")
    axes[1].quiver(gx_np, gy_np, pred_np[..., 0], pred_np[..., 1], color="tab:orange", angles="xy")
    axes[1].set_title("Learned Score Field")
    for ax in axes:
        ax.set_xlabel(r"$\tilde{\theta}_1$")
        ax.set_ylabel(r"$\tilde{\theta}_2$")
        ax.set_aspect("equal")
    fig.suptitle(rf"Score Field at fixed $x=[1.5,-1.0]$, $\sigma={sigma_dsm:.3f}$")
    fig.tight_layout()
    fig.savefig(out_quiver_path, dpi=180)
    plt.close(fig)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        err.T,
        origin="lower",
        extent=[grid_min, grid_max, grid_min, grid_max],
        cmap="magma",
        aspect="equal",
    )
    plt.colorbar(im, label=r"$\|s_\phi - s_{\mathrm{true}}\|_2$")
    plt.xlabel(r"$\tilde{\theta}_1$")
    plt.ylabel(r"$\tilde{\theta}_2$")
    plt.title("Score Error Heatmap")
    plt.tight_layout()
    plt.savefig(out_err_path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D conditional score matching demo.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--n-val", type=int, default=5000)
    parser.add_argument("--sigma-x", type=float, default=0.7)
    parser.add_argument("--sigma-dsm", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="outputs_step1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    device = torch.device(args.device)

    x_train, theta_train = sample_joint(args.n_train, args.sigma_x, device=device)
    x_val, theta_val = sample_joint(args.n_val, args.sigma_x, device=device)
    train_loader = build_loader(x_train, theta_train, args.batch_size, shuffle=True)

    model = ConditionalScoreModel(hidden_dim=args.hidden_dim, depth=args.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_losses: list[float] = []

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        for xb, thetab in train_loader:
            xb = xb.to(device, non_blocking=True)
            thetab = thetab.to(device, non_blocking=True)
            loss = model.train_step((xb, thetab), optimizer, sigma_dsm=args.sigma_dsm)
            epoch_losses.append(loss)
        mean_loss = float(np.mean(epoch_losses))
        train_losses.append(mean_loss)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"[epoch {epoch:4d}/{args.epochs}] train_loss={mean_loss:.6f}")

    metrics = evaluate_model(model, x_val, theta_val, sigma_x=args.sigma_x, sigma_dsm=args.sigma_dsm)
    print(
        f"[eval] score_mse={metrics.mse:.6f}, mean_cosine_similarity={metrics.cosine_mean:.6f}"
    )

    loss_path = os.path.join(args.output_dir, "loss_curve.png")
    quiver_path = os.path.join(args.output_dir, "score_field_quiver.png")
    err_path = os.path.join(args.output_dir, "score_error_heatmap.png")
    plot_loss(train_losses, loss_path)
    plot_score_fields(
        model=model,
        sigma_x=args.sigma_x,
        sigma_dsm=args.sigma_dsm,
        out_quiver_path=quiver_path,
        out_err_path=err_path,
        device=device,
    )

    print("Saved artifacts:")
    print(f"  - {loss_path}")
    print(f"  - {quiver_path}")
    print(f"  - {err_path}")


if __name__ == "__main__":
    main()
