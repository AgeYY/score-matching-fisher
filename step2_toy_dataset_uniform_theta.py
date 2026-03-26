#!/usr/bin/env python3
"""Step 2: toy dataset with uniform theta and Gaussian p(x|theta).

This script defines a reusable toy generator:
  theta ~ Uniform[theta_low, theta_high]
  x | theta ~ N(mu(theta), Sigma)
where mu(theta) is a nonlinear 2D tuning curve.
It also creates visualizations to inspect the dataset geometry.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@dataclass
class ToyConditionalGaussianDataset:
    theta_low: float = -3.0
    theta_high: float = 3.0
    sigma_x1: float = 0.30
    sigma_x2: float = 0.22
    rho: float = 0.15
    seed: int = 42

    def __post_init__(self) -> None:
        if not (self.theta_low < self.theta_high):
            raise ValueError("theta_low must be smaller than theta_high.")
        if not (-0.99 < self.rho < 0.99):
            raise ValueError("rho must be in (-0.99, 0.99).")
        self.rng = set_seed(self.seed)
        self.cov = np.array(
            [
                [self.sigma_x1**2, self.rho * self.sigma_x1 * self.sigma_x2],
                [self.rho * self.sigma_x1 * self.sigma_x2, self.sigma_x2**2],
            ],
            dtype=np.float64,
        )
        # Numerical safety for covariance decomposition.
        self.cov = self.cov + 1e-8 * np.eye(2, dtype=np.float64)
        self.cov_chol = np.linalg.cholesky(self.cov)

    def sample_theta(self, n: int) -> np.ndarray:
        theta = self.rng.uniform(self.theta_low, self.theta_high, size=(n, 1))
        return theta.astype(np.float64)

    def tuning_curve(self, theta: np.ndarray) -> np.ndarray:
        """Nonlinear 2D mean map mu(theta). theta shape: (N,1)."""
        t = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
        mu1 = 1.10 * np.sin(1.25 * t) + 0.28 * t
        mu2 = 0.85 * np.cos(1.05 * t + 0.30) - 0.12 * (t**2) + 0.05 * t
        return np.concatenate([mu1, mu2], axis=1)

    def sample_x(self, theta: np.ndarray) -> np.ndarray:
        mu = self.tuning_curve(theta)
        eps = self.rng.standard_normal(size=mu.shape)
        x = mu + eps @ self.cov_chol.T
        return x.astype(np.float64)

    def sample_joint(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        theta = self.sample_theta(n)
        x = self.sample_x(theta)
        return theta, x


def summarize_dataset(theta: np.ndarray, x: np.ndarray, dataset: ToyConditionalGaussianDataset) -> None:
    print("Dataset summary")
    print(f"  theta shape: {theta.shape}")
    print(f"  x shape: {x.shape}")
    print(f"  theta min/max: {theta.min():.4f} / {theta.max():.4f}")
    print(f"  theta mean/std: {theta.mean():.4f} / {theta.std():.4f}")
    print(f"  x mean: [{x[:,0].mean():.4f}, {x[:,1].mean():.4f}]")
    print(f"  x std:  [{x[:,0].std():.4f}, {x[:,1].std():.4f}]")
    print("  configured covariance Sigma:")
    print(dataset.cov)

    # Quick sanity check: bin by theta and compare empirical mean to tuning curve mean.
    n_bins = 18
    bins = np.linspace(dataset.theta_low, dataset.theta_high, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    idx = np.digitize(theta.ravel(), bins) - 1
    valid = (idx >= 0) & (idx < n_bins)
    idx = idx[valid]
    x_valid = x[valid]
    empirical = np.zeros((n_bins, 2), dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = idx == b
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            empirical[b] = x_valid[mask].mean(axis=0)
    model_mean = dataset.tuning_curve(centers[:, None])
    used = counts > 0
    mae = np.mean(np.abs(empirical[used] - model_mean[used])) if used.any() else np.nan
    print(f"  binned E[x|theta] vs tuning curve MAE: {mae:.4f}")


def plot_joint_scatter(theta: np.ndarray, x: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(x[:, 0], x[:, 1], c=theta.ravel(), s=8, alpha=0.55, cmap="viridis")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(r"Joint Samples of $x$ Colored by $\theta$")
    cb = plt.colorbar(sc)
    cb.set_label(r"$\theta$")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_tuning_curve(dataset: ToyConditionalGaussianDataset, out_path: str) -> None:
    t = np.linspace(dataset.theta_low, dataset.theta_high, 500, dtype=np.float64)[:, None]
    mu = dataset.tuning_curve(t)
    plt.figure(figsize=(8, 4.5))
    plt.plot(t[:, 0], mu[:, 0], label=r"$\mu_1(\theta)$", linewidth=2.2)
    plt.plot(t[:, 0], mu[:, 1], label=r"$\mu_2(\theta)$", linewidth=2.2)
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"Mean of $x|\theta$")
    plt.title("Nonlinear Tuning Curve")
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_conditional_slices(
    dataset: ToyConditionalGaussianDataset,
    slice_thetas: np.ndarray,
    n_slice: int,
    out_path: str,
) -> None:
    cols = len(slice_thetas)
    fig, axes = plt.subplots(1, cols, figsize=(4.0 * cols, 4.1), sharex=True, sharey=True)
    if cols == 1:
        axes = [axes]

    for ax, th in zip(axes, slice_thetas):
        theta_block = np.full((n_slice, 1), fill_value=th, dtype=np.float64)
        x_slice = dataset.sample_x(theta_block)
        mu = dataset.tuning_curve(np.array([[th]], dtype=np.float64))[0]
        ax.scatter(x_slice[:, 0], x_slice[:, 1], s=12, alpha=0.45, color="#1f77b4")
        ax.scatter([mu[0]], [mu[1]], marker="x", s=120, color="#d62728", linewidths=2.0, label="mu(theta)")
        ax.set_title(rf"$\theta={th:.2f}$")
        ax.set_xlabel("x1")
        ax.grid(alpha=0.2, linestyle=":")
    axes[0].set_ylabel("x2")
    axes[0].legend(loc="upper right")
    fig.suptitle(r"Conditional Slices: Samples from $p(x|\theta)$", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and visualize uniform-theta toy dataset.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-joint", type=int, default=12000)
    parser.add_argument("--n-slice", type=int, default=1400)
    parser.add_argument("--theta-low", type=float, default=-3.0)
    parser.add_argument("--theta-high", type=float, default=3.0)
    parser.add_argument("--sigma-x1", type=float, default=0.30)
    parser.add_argument("--sigma-x2", type=float, default=0.22)
    parser.add_argument("--rho", type=float, default=0.15)
    parser.add_argument("--slice-thetas", type=float, nargs="+", default=[-2.4, -0.8, 0.8, 2.4])
    parser.add_argument("--output-dir", type=str, default="outputs_step2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = ToyConditionalGaussianDataset(
        theta_low=args.theta_low,
        theta_high=args.theta_high,
        sigma_x1=args.sigma_x1,
        sigma_x2=args.sigma_x2,
        rho=args.rho,
        seed=args.seed,
    )

    theta, x = dataset.sample_joint(args.n_joint)
    summarize_dataset(theta, x, dataset)

    joint_path = os.path.join(args.output_dir, "joint_scatter_theta_color.png")
    tuning_path = os.path.join(args.output_dir, "tuning_curve.png")
    slices_path = os.path.join(args.output_dir, "conditional_slices.png")

    plot_joint_scatter(theta, x, joint_path)
    plot_tuning_curve(dataset, tuning_path)
    plot_conditional_slices(dataset, np.asarray(args.slice_thetas, dtype=np.float64), args.n_slice, slices_path)

    print("Saved artifacts:")
    print(f"  - {joint_path}")
    print(f"  - {tuning_path}")
    print(f"  - {slices_path}")


if __name__ == "__main__":
    main()
