"""Shared helpers for toy-dataset diagnostics (joint scatter + tuning curves)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fisher.data import (
    ToyConditionalGMMNonGaussianDataset,
    ToyConditionalGaussianDataset,
    ToyCosSinPiecewiseNoiseDataset,
    ToyLinearPiecewiseNoiseDataset,
)


def pca_project(x: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return PCA projection and PCA basis for centered x."""
    x0 = x - x.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(x0, full_matrices=False)
    basis = vh[:n_components].T
    proj = x0 @ basis
    return proj, x.mean(axis=0), basis


def summarize_dataset(
    theta: np.ndarray,
    x: np.ndarray,
    dataset: ToyConditionalGaussianDataset
    | ToyConditionalGMMNonGaussianDataset
    | ToyCosSinPiecewiseNoiseDataset
    | ToyLinearPiecewiseNoiseDataset,
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
    dataset: ToyConditionalGaussianDataset
    | ToyConditionalGMMNonGaussianDataset
    | ToyCosSinPiecewiseNoiseDataset
    | ToyLinearPiecewiseNoiseDataset,
    out_path: str,
    *,
    scatter_max_points: int | None = 1000,
    scatter_subsample_seed: int = 0,
) -> None:
    """Single figure: left = joint scatter (PCA if x_dim>2), right = tuning curves.

    The scatter uses at most ``scatter_max_points`` rows (uniform random subset without
    replacement when there are more rows). Pass ``None`` to plot every point. The tuning-curve
    panel is model-based and unchanged by subsampling.
    """
    theta_plot = np.asarray(theta, dtype=np.float64)
    x_plot = np.asarray(x, dtype=np.float64)
    n = int(theta_plot.shape[0])
    if scatter_max_points is not None and n > int(scatter_max_points):
        k = int(scatter_max_points)
        rng = np.random.default_rng(int(scatter_subsample_seed))
        pick = rng.choice(n, size=k, replace=False)
        theta_plot = theta_plot[pick]
        x_plot = x_plot[pick]

    fig, (ax_scatter, ax_tune) = plt.subplots(1, 2, figsize=(14.5, 5.2))

    if x_plot.shape[1] == 2:
        proj = x_plot
        xlabel, ylabel = "x1", "x2"
        title_s = r"Joint Samples of $x$ Colored by $\theta$"
    else:
        proj, _, _ = pca_project(x_plot, n_components=2)
        xlabel, ylabel = "PC1", "PC2"
        title_s = rf"Joint Samples (PCA Projection, x_dim={x_plot.shape[1]}) Colored by $\theta$"

    sc = ax_scatter.scatter(
        proj[:, 0], proj[:, 1], c=theta_plot.ravel(), s=8, alpha=0.55, cmap="viridis"
    )
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
    svg_path = str(Path(out_path).with_suffix(".svg"))
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
