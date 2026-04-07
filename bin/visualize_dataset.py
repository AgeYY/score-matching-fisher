#!/usr/bin/env python3
"""Visualize the uniform-theta toy dataset (Gaussian or GMM) and save diagnostic figures."""

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

from global_setting import DATA_DIR
from fisher.data import ToyConditionalGMMNonGaussianDataset, ToyConditionalGaussianDataset
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_meta


def pca_project(x: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return PCA projection and PCA basis for centered x."""
    x0 = x - x.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(x0, full_matrices=False)
    basis = vh[:n_components].T
    proj = x0 @ basis
    return proj, x.mean(axis=0), basis


def pca_transform(x: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (x - mean.reshape(1, -1)) @ basis


def summarize_dataset(
    theta: np.ndarray,
    x: np.ndarray,
    dataset: ToyConditionalGaussianDataset | ToyConditionalGMMNonGaussianDataset,
) -> None:
    print("Dataset summary")
    print(f"  theta shape: {theta.shape}")
    print(f"  x shape: {x.shape}")
    print(f"  x_dim: {x.shape[1]}")
    print(f"  theta min/max: {theta.min():.4f} / {theta.max():.4f}")
    print(f"  theta mean/std: {theta.mean():.4f} / {theta.std():.4f}")

    n_show = min(6, x.shape[1])
    x_mean = ", ".join(f"{v:.4f}" for v in x.mean(axis=0)[:n_show])
    x_std = ", ".join(f"{v:.4f}" for v in x.std(axis=0)[:n_show])
    suffix = " ..." if x.shape[1] > n_show else ""
    print(f"  x mean (first dims): [{x_mean}]{suffix}")
    print(f"  x std  (first dims): [{x_std}]{suffix}")

    print("  covariance summary:")
    if hasattr(dataset, "covariance_scales"):
        scales = dataset.covariance_scales(theta)
        smin = scales.min(axis=0)[:n_show]
        smax = scales.max(axis=0)[:n_show]
        print(f"  theta-dependent sigma min (first dims): {np.round(smin, 4).tolist()}{suffix}")
        print(f"  theta-dependent sigma max (first dims): {np.round(smax, 4).tolist()}{suffix}")
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
    empirical = np.zeros((n_bins, x.shape[1]), dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = idx == b
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            empirical[b] = x_valid[mask].mean(axis=0)
    model_mean = dataset.tuning_curve(centers[:, None])
    used = counts > 0
    mae = np.mean(np.abs(empirical[used] - model_mean[used])) if used.any() else np.nan
    print(f"  binned E[x|theta] vs tuning curve MAE (all dims): {mae:.4f}")


def plot_joint_and_tuning(
    theta: np.ndarray,
    x: np.ndarray,
    dataset: ToyConditionalGaussianDataset | ToyConditionalGMMNonGaussianDataset,
    out_path: str,
) -> None:
    """Single figure: left = joint scatter (PCA if x_dim>2), right = tuning curves."""
    fig, (ax_scatter, ax_tune) = plt.subplots(1, 2, figsize=(14.5, 5.2))

    if x.shape[1] == 2:
        proj = x
        xlabel, ylabel = "x1", "x2"
        title_s = r"Joint Samples of $x$ Colored by $\theta$"
    else:
        proj, _, _ = pca_project(x, n_components=2)
        xlabel, ylabel = "PC1", "PC2"
        title_s = rf"Joint Samples (PCA Projection, x_dim={x.shape[1]}) Colored by $\theta$"

    sc = ax_scatter.scatter(proj[:, 0], proj[:, 1], c=theta.ravel(), s=8, alpha=0.55, cmap="viridis")
    ax_scatter.set_xlabel(xlabel)
    ax_scatter.set_ylabel(ylabel)
    ax_scatter.set_title(title_s)
    fig.colorbar(sc, ax=ax_scatter, label=r"$\theta$")

    t = np.linspace(dataset.theta_low, dataset.theta_high, 500, dtype=np.float64)[:, None]
    mu = dataset.tuning_curve(t)
    n_plot = int(mu.shape[1])
    for j in range(n_plot):
        ax_tune.plot(t[:, 0], mu[:, j], label=rf"$\mu_{{{j+1}}}(\theta)$", linewidth=2.0)
    ax_tune.set_xlabel(r"$\theta$")
    ax_tune.set_ylabel(r"Mean of $x|\theta$")
    ax_tune.set_title(f"Tuning curves (all {n_plot} dimensions)")
    ax_tune.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    leg_kw: dict = {}
    if n_plot > 6:
        leg_kw = {"ncol": 2, "fontsize": 8}
    if n_plot > 14:
        leg_kw = {"ncol": 3, "fontsize": 7}
    ax_tune.legend(**leg_kw)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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

    # Build a shared PCA frame for x_dim > 2 for comparable slice panels.
    pca_mean = None
    pca_basis = None
    if dataset.x_dim > 2:
        fit_theta = np.repeat(slice_thetas.reshape(-1, 1), repeats=max(150, n_slice // 6), axis=0)
        fit_x = dataset.sample_x(fit_theta)
        _, pca_mean, pca_basis = pca_project(fit_x, n_components=2)

    for ax, th in zip(axes, slice_thetas):
        theta_block = np.full((n_slice, 1), fill_value=th, dtype=np.float64)
        x_slice = dataset.sample_x(theta_block)
        mu = dataset.tuning_curve(np.array([[th]], dtype=np.float64))[0]

        if dataset.x_dim == 2:
            x_plot = x_slice
            mu_plot = mu
            xlabel, ylabel = "x1", "x2"
        else:
            x_plot = pca_transform(x_slice, pca_mean, pca_basis)
            mu_plot = pca_transform(mu.reshape(1, -1), pca_mean, pca_basis)[0]
            xlabel, ylabel = "PC1", "PC2"

        ax.scatter(x_plot[:, 0], x_plot[:, 1], s=12, alpha=0.45, color="#1f77b4")
        ax.scatter([mu_plot[0]], [mu_plot[1]], marker="x", s=120, color="#d62728", linewidths=2.0, label="mu(theta)")

        if hasattr(dataset, "component_means"):
            m1, m2 = dataset.component_means(np.array([[th]], dtype=np.float64))
            if dataset.x_dim > 2:
                m1p = pca_transform(m1, pca_mean, pca_basis)[0]
                m2p = pca_transform(m2, pca_mean, pca_basis)[0]
            else:
                m1p, m2p = m1[0], m2[0]
            ax.scatter([m1p[0]], [m1p[1]], marker="^", s=80, color="#2ca02c", alpha=0.9, label="comp mean 1")
            ax.scatter([m2p[0]], [m2p[1]], marker="v", s=80, color="#9467bd", alpha=0.9, label="comp mean 2")

        ax.set_title(rf"$\theta={th:.2f}$")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.2, linestyle=":")

    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc="upper right")
    fig.suptitle(r"Conditional Slices: Samples from $p(x|\theta)$", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and visualize uniform-theta toy dataset, or visualize a shared .npz from fisher_make_dataset.py."
    )
    parser.add_argument(
        "--dataset-npz",
        type=str,
        default=None,
        help="Path to a shared dataset .npz from fisher_make_dataset.py. When set, data and model params come from file metadata; other dataset flags are ignored.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dataset-family", type=str, default="gaussian", choices=["gaussian", "gmm_non_gauss"])
    parser.add_argument(
        "--tuning-curve-family",
        type=str,
        default="cosine",
        choices=["cosine", "von_mises_raw"],
        help="Mean tuning curve (must match dataset .npz when using --dataset-npz).",
    )
    parser.add_argument("--vm-mu-amp", type=float, default=1.0)
    parser.add_argument("--vm-kappa", type=float, default=1.0)
    parser.add_argument("--vm-omega", type=float, default=1.0)
    parser.add_argument("--n-joint", type=int, default=12000)
    parser.add_argument("--n-slice", type=int, default=1400)
    parser.add_argument("--theta-low", type=float, default=-3.0)
    parser.add_argument("--theta-high", type=float, default=3.0)
    parser.add_argument("--x-dim", type=int, default=2)
    parser.add_argument("--sigma-x1", type=float, default=0.30)
    parser.add_argument("--sigma-x2", type=float, default=0.30)
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
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(DATA_DIR) / "outputs_step2"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset_npz is not None:
        bundle = load_shared_dataset_npz(args.dataset_npz)
        meta = bundle.meta
        dataset = build_dataset_from_meta(meta)
        theta, x = bundle.theta_all, bundle.x_all
        n_total = int(meta.get("n_total", theta.shape[0]))
        train_frac = float(meta.get("train_frac", float("nan")))
        print(
            "[data] source=shared_npz "
            f"path={args.dataset_npz} "
            f"family={meta.get('dataset_family')} "
            f"seed={meta.get('seed')}"
        )
        print(
            f"[data] total={n_total} train={bundle.theta_train.shape[0]} eval={bundle.theta_eval.shape[0]} "
            f"train_frac={train_frac}"
        )
        if x.shape[1] < 2:
            raise ValueError("Loaded x must have x_dim >= 2.")
    else:
        if args.x_dim < 2:
            raise ValueError("--x-dim must be >= 2.")
        if args.dataset_family == "gaussian":
            dataset: ToyConditionalGaussianDataset | ToyConditionalGMMNonGaussianDataset = ToyConditionalGaussianDataset(
                theta_low=args.theta_low,
                theta_high=args.theta_high,
                x_dim=args.x_dim,
                tuning_curve_family=args.tuning_curve_family,
                vm_mu_amp=args.vm_mu_amp,
                vm_kappa=args.vm_kappa,
                vm_omega=args.vm_omega,
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
                x_dim=args.x_dim,
                tuning_curve_family=args.tuning_curve_family,
                vm_mu_amp=args.vm_mu_amp,
                vm_kappa=args.vm_kappa,
                vm_omega=args.vm_omega,
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
        print(f"[data] source=fresh_sample n_joint={args.n_joint} seed={args.seed}")

    summarize_dataset(theta, x, dataset)

    joint_tuning_path = os.path.join(args.output_dir, "joint_scatter_and_tuning_curve.png")
    slices_path = os.path.join(args.output_dir, "conditional_slices.png")

    plot_joint_and_tuning(theta, x, dataset, joint_tuning_path)
    plot_conditional_slices(dataset, np.asarray(args.slice_thetas, dtype=np.float64), args.n_slice, slices_path)

    print("Saved artifacts:")
    print(f"  - {joint_tuning_path}")
    print(f"  - {slices_path}")


if __name__ == "__main__":
    main()
