#!/usr/bin/env python3
"""Step 2: toy dataset with uniform theta and Gaussian p(x|theta)."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from fisher.data import ToyConditionalGMMNonGaussianDataset, ToyConditionalGaussianDataset


def summarize_dataset(theta: np.ndarray, x: np.ndarray, dataset: ToyConditionalGaussianDataset) -> None:
    print("Dataset summary")
    print(f"  theta shape: {theta.shape}")
    print(f"  x shape: {x.shape}")
    print(f"  theta min/max: {theta.min():.4f} / {theta.max():.4f}")
    print(f"  theta mean/std: {theta.mean():.4f} / {theta.std():.4f}")
    print(f"  x mean: [{x[:,0].mean():.4f}, {x[:,1].mean():.4f}]")
    print(f"  x std:  [{x[:,0].std():.4f}, {x[:,1].std():.4f}]")
    print("  covariance summary:")
    if hasattr(dataset, "covariance_components"):
        if hasattr(dataset, "cov"):
            print(dataset.cov)
        s1, s2, rho_t = dataset.covariance_components(theta)
        print(
            "  theta-dependent covariance ranges: "
            f"sigma1 in [{s1.min():.4f}, {s1.max():.4f}], "
            f"sigma2 in [{s2.min():.4f}, {s2.max():.4f}], "
            f"rho in [{rho_t.min():.4f}, {rho_t.max():.4f}]"
        )
    elif hasattr(dataset, "component_covariances"):
        print("  non-Gaussian mixture with theta-dependent component covariances.")
    if hasattr(dataset, "_mix_weight"):
        pi, _ = dataset._mix_weight(theta)
        print(f"  mixture weight pi(theta) range: [{pi.min():.4f}, {pi.max():.4f}]")

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


def plot_tuning_curve(dataset: ToyConditionalGaussianDataset | ToyConditionalGMMNonGaussianDataset, out_path: str) -> None:
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
    dataset: ToyConditionalGaussianDataset | ToyConditionalGMMNonGaussianDataset,
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
        if hasattr(dataset, "component_means"):
            m1, m2 = dataset.component_means(np.array([[th]], dtype=np.float64))
            ax.scatter([m1[0, 0]], [m1[0, 1]], marker="^", s=80, color="#2ca02c", alpha=0.9, label="comp mean 1")
            ax.scatter([m2[0, 0]], [m2[0, 1]], marker="v", s=80, color="#9467bd", alpha=0.9, label="comp mean 2")
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
    parser.add_argument("--dataset-family", type=str, default="gaussian", choices=["gaussian", "gmm_non_gauss"])
    parser.add_argument("--n-joint", type=int, default=12000)
    parser.add_argument("--n-slice", type=int, default=1400)
    parser.add_argument("--theta-low", type=float, default=-3.0)
    parser.add_argument("--theta-high", type=float, default=3.0)
    parser.add_argument("--sigma-x1", type=float, default=0.30)
    parser.add_argument("--sigma-x2", type=float, default=0.22)
    parser.add_argument("--rho", type=float, default=0.15)
    parser.add_argument("--cov-theta-amp1", type=float, default=0.35)
    parser.add_argument("--cov-theta-amp2", type=float, default=0.30)
    parser.add_argument("--cov-theta-amp-rho", type=float, default=0.30)
    parser.add_argument("--cov-theta-freq1", type=float, default=0.90)
    parser.add_argument("--cov-theta-freq2", type=float, default=0.75)
    parser.add_argument("--cov-theta-freq-rho", type=float, default=1.10)
    parser.add_argument("--cov-theta-phase1", type=float, default=0.20)
    parser.add_argument("--cov-theta-phase2", type=float, default=-0.35)
    parser.add_argument("--cov-theta-phase-rho", type=float, default=0.40)
    parser.add_argument("--rho-clip", type=float, default=0.85)
    parser.add_argument("--gmm-sep-scale", type=float, default=1.10)
    parser.add_argument("--gmm-sep-freq", type=float, default=0.85)
    parser.add_argument("--gmm-sep-phase", type=float, default=0.35)
    parser.add_argument("--gmm-mix-logit-scale", type=float, default=1.40)
    parser.add_argument("--gmm-mix-bias", type=float, default=0.00)
    parser.add_argument("--gmm-mix-freq", type=float, default=0.95)
    parser.add_argument("--gmm-mix-phase", type=float, default=-0.20)
    parser.add_argument("--slice-thetas", type=float, nargs="+", default=[-2.4, -0.8, 0.8, 2.4])
    parser.add_argument("--output-dir", type=str, default="data/outputs_step2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset_family == "gaussian":
        dataset: ToyConditionalGaussianDataset | ToyConditionalGMMNonGaussianDataset = ToyConditionalGaussianDataset(
            theta_low=args.theta_low,
            theta_high=args.theta_high,
            sigma_x1=args.sigma_x1,
            sigma_x2=args.sigma_x2,
            rho=args.rho,
            cov_theta_amp1=args.cov_theta_amp1,
            cov_theta_amp2=args.cov_theta_amp2,
            cov_theta_amp_rho=args.cov_theta_amp_rho,
            cov_theta_freq1=args.cov_theta_freq1,
            cov_theta_freq2=args.cov_theta_freq2,
            cov_theta_freq_rho=args.cov_theta_freq_rho,
            cov_theta_phase1=args.cov_theta_phase1,
            cov_theta_phase2=args.cov_theta_phase2,
            cov_theta_phase_rho=args.cov_theta_phase_rho,
            rho_clip=args.rho_clip,
            seed=args.seed,
        )
    else:
        dataset = ToyConditionalGMMNonGaussianDataset(
            theta_low=args.theta_low,
            theta_high=args.theta_high,
            sigma_x1=args.sigma_x1,
            sigma_x2=args.sigma_x2,
            rho=args.rho,
            sep_scale=args.gmm_sep_scale,
            sep_freq=args.gmm_sep_freq,
            sep_phase=args.gmm_sep_phase,
            mix_logit_scale=args.gmm_mix_logit_scale,
            mix_bias=args.gmm_mix_bias,
            mix_freq=args.gmm_mix_freq,
            mix_phase=args.gmm_mix_phase,
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
