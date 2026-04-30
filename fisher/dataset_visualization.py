"""Shared helpers for toy-dataset diagnostics (joint scatter + tuning curves)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

if TYPE_CHECKING:
    from matplotlib.axes import Axes

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


def _moving_average_rows(y: np.ndarray, window: int) -> np.ndarray:
    """Per-column centered moving average along rows (same length as ``y``)."""
    arr = np.asarray(y, dtype=np.float64)
    w = int(window)
    if w < 3 or arr.ndim != 2 or int(arr.shape[0]) < w:
        return arr
    if w % 2 == 0:
        w += 1
    k = np.ones(w, dtype=np.float64) / float(w)
    out = np.empty_like(arr)
    for j in range(int(arr.shape[1])):
        out[:, j] = np.convolve(arr[:, j], k, mode="same")
    return out


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


def plot_joint_and_tuning_on_axes(
    fig: plt.Figure,
    ax_tune: "Axes",
    ax_manifold: "Axes",
    theta: np.ndarray,
    x: np.ndarray,
    dataset: ToyConditionalGaussianDataset
    | ToyConditionalGMMNonGaussianDataset
    | ToyCosSinPiecewiseNoiseDataset
    | ToyLinearPiecewiseNoiseDataset,
    *,
    mu_override: np.ndarray | None = None,
    theta_grid_override: np.ndarray | None = None,
    n_theta_grid: int = 500,
    smooth_mean_window: int = 0,
    manifold_trajectory_linewidth: float = 4.0,
) -> None:
    """Draw tuning curves on ``ax_tune`` and manifold / samples on ``ax_manifold``.

    For ``x_dim >= 2``, ``ax_manifold`` shows the mean curve in its PCA plane as a ``theta``-colored
    polyline (``LineCollection``) plus light vertex markers, with sample projections overlaid in the
    same basis. For ``x_dim == 1``, it overlays
    ``mu(theta)`` as a line and scatter ``(theta, x)``.

    ``theta`` and ``x`` should already reflect any scatter subsampling desired by the caller.

    If ``mu_override`` is set, it must be an array of shape ``(T, d)`` giving the mean curve at
    ``T`` theta values. When ``theta_grid_override`` is provided, it must have shape ``(T, 1)``
    (same ``T`` as ``mu_override`` rows); otherwise theta values are ``linspace(theta_low,
    theta_high, n_theta_grid)``. If ``mu_override`` is ``None``, ``mu`` is ``dataset.tuning_curve(t)``
    on that dense grid and ``theta_grid_override`` must be ``None``.

    When ``theta_grid_override`` is set and ``smooth_mean_window >= 3``, ``mu`` is lightly smoothed
    along the theta-ordered rows (moving average) before PCA for visualization.

    The scatter coordinates ``x`` must have the same trailing dimension ``d`` as ``mu`` for the
    PCA overlay when ``d >= 2``.
    """
    theta_plot = np.asarray(theta, dtype=np.float64)
    x_plot = np.asarray(x, dtype=np.float64)

    if mu_override is None:
        if theta_grid_override is not None:
            raise ValueError("theta_grid_override is only valid when mu_override is set.")
        t = np.linspace(dataset.theta_low, dataset.theta_high, int(n_theta_grid), dtype=np.float64)[
            :, None
        ]
        mu = np.asarray(dataset.tuning_curve(t), dtype=np.float64)
    else:
        mu = np.asarray(mu_override, dtype=np.float64)
        if theta_grid_override is not None:
            t = np.asarray(theta_grid_override, dtype=np.float64).reshape(-1, 1)
            if int(mu.shape[0]) != int(t.shape[0]):
                raise ValueError(
                    f"mu_override rows ({mu.shape[0]}) must match theta_grid_override rows ({t.shape[0]})."
                )
        else:
            t = np.linspace(dataset.theta_low, dataset.theta_high, int(n_theta_grid), dtype=np.float64)[
                :, None
            ]
            if mu.ndim != 2 or int(mu.shape[0]) != int(t.shape[0]):
                raise ValueError(
                    f"mu_override must have shape ({t.shape[0]}, d); got {mu.shape} for n_theta_grid={n_theta_grid}."
                )

    if theta_grid_override is not None and int(smooth_mean_window) >= 3 and mu.shape[0] >= int(smooth_mean_window):
        mu = _moving_average_rows(np.asarray(mu, dtype=np.float64, order="C"), int(smooth_mean_window))
    n_plot = int(mu.shape[1])
    if x_plot.ndim != 2 or int(x_plot.shape[1]) != n_plot:
        raise ValueError(
            f"x must have shape (N, {n_plot}) to match tuning curve dimension; got {x_plot.shape}."
        )

    theta_norm = Normalize(
        vmin=float(dataset.theta_low),
        vmax=float(dataset.theta_high),
    )
    sm_theta = ScalarMappable(norm=theta_norm, cmap="viridis")
    sm_theta.set_array([])

    for j in range(n_plot):
        ax_tune.plot(t[:, 0], mu[:, j], color="black", linewidth=2.0)
    ax_tune.set_xlabel(r"$\theta$")
    ax_tune.set_ylabel(r"Mean of $x|\theta$")

    if n_plot >= 2:
        mu_pc, mean_mu, basis_mu = pca_project(mu, n_components=2)
        x0 = x_plot - mean_mu
        proj_x = x0 @ basis_mu
        th_line = np.asarray(t, dtype=np.float64).ravel()
        t_mu = int(mu_pc.shape[0])
        if t_mu >= 2:
            pts = mu_pc[:, :2]
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            seg_theta = 0.5 * (th_line[:-1] + th_line[1:])
            lc = LineCollection(
                segs,
                cmap="viridis",
                norm=theta_norm,
                array=seg_theta,
                linewidth=float(manifold_trajectory_linewidth),
                capstyle="round",
                zorder=2,
            )
            ax_manifold.add_collection(lc)
        ax_manifold.scatter(
            mu_pc[:, 0],
            mu_pc[:, 1],
            c=th_line,
            s=8,
            alpha=0.45,
            cmap="viridis",
            norm=theta_norm,
            zorder=3,
        )
        ax_manifold.scatter(
            proj_x[:, 0],
            proj_x[:, 1],
            c=theta_plot.ravel(),
            s=8,
            alpha=0.35,
            cmap="viridis",
            norm=theta_norm,
            zorder=4,
        )
        ax_manifold.set_aspect("equal", adjustable="datalim")
        ax_manifold.set_axis_off()
        ax_manifold.set_box_aspect(1)
        cbar = fig.colorbar(sm_theta, ax=ax_manifold, fraction=0.035, pad=0.02)
        cbar.set_label(r"$\theta$")
    else:
        th = theta_plot.ravel()
        ax_manifold.plot(t[:, 0], mu[:, 0], color="#4c78a8", linewidth=2.0, zorder=1)
        ax_manifold.scatter(
            th,
            x_plot[:, 0],
            c=th,
            s=10,
            alpha=0.35,
            cmap="viridis",
            norm=theta_norm,
            zorder=2,
        )
        ax_manifold.set_xlabel(r"$\theta$")
        ax_manifold.set_ylabel(r"$x_1$ / $\mu(\theta)$")
        ax_manifold.set_box_aspect(1)
        fig.colorbar(sm_theta, ax=ax_manifold, fraction=0.035, pad=0.02).set_label(r"$\theta$")

    ax_tune.set_box_aspect(1)


def plot_joint_and_tuning(
    theta: np.ndarray,
    x: np.ndarray,
    dataset: ToyConditionalGaussianDataset
    | ToyConditionalGMMNonGaussianDataset
    | ToyCosSinPiecewiseNoiseDataset
    | ToyLinearPiecewiseNoiseDataset,
    out_path: str,
    *,
    scatter_max_points: int | None = 400,
    scatter_subsample_seed: int = 0,
) -> None:
    """Two panels: per-dim tuning curves, then PCA manifold with samples overlaid (or 1D overlay).

    When ``x_dim >= 2``, the right panel uses one ``pca_project`` on the dense mean curve ``mu``;
    samples are projected with the same ``mean_mu`` and ``basis_mu``. A single colorbar maps
    color to ``theta``.

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

    fig, (ax_tune, ax_manifold) = plt.subplots(1, 2, figsize=(9.0, 3.2), layout="constrained")
    plot_joint_and_tuning_on_axes(
        fig,
        ax_tune,
        ax_manifold,
        theta_plot,
        x_plot,
        dataset,
        mu_override=None,
    )

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    svg_path = str(Path(out_path).with_suffix(".svg"))
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
