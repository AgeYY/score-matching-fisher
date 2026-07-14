"""Shared helpers for toy-dataset diagnostics (joint scatter + tuning curves)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
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


def _categorical_labels(theta: np.ndarray, k: int) -> np.ndarray:
    arr = np.asarray(theta)
    if arr.ndim == 2 and int(arr.shape[1]) == int(k):
        return np.argmax(np.asarray(arr, dtype=np.float64), axis=1).astype(np.int64)
    return np.asarray(arr).reshape(-1).astype(np.int64)


def plot_mog5_native_scatter_covariance(
    npz_path: str | Path,
    *,
    svg_path: str | Path,
    png_path: str | Path,
    max_points: int = 500,
) -> tuple[Path, Path]:
    """Plot the (x1, x2) view of a native MoG5 shared-dataset NPZ."""
    from fisher.shared_dataset_io import load_shared_dataset_npz

    bundle = load_shared_dataset_npz(npz_path)
    meta = dict(bundle.meta)
    if str(meta.get("dataset_family", "")) != "random_mog_categorical":
        raise ValueError(f"Expected random_mog_categorical NPZ, got {meta.get('dataset_family')!r}.")
    k = int(meta.get("num_categories", -1))
    if k != 5:
        raise ValueError(f"Expected num_categories=5, got {meta.get('num_categories')!r}.")
    x_dim = int(meta.get("x_dim", -1))
    if x_dim < 2:
        raise ValueError(f"Expected native x_dim >= 2, got {meta.get('x_dim')!r}.")

    x = np.asarray(bundle.x_all, dtype=np.float64)
    theta = np.asarray(bundle.theta_all)
    if x.ndim != 2 or int(x.shape[1]) != x_dim or int(x.shape[1]) < 2:
        raise ValueError(f"Expected x_all shape (N, {x_dim}) with x_dim >= 2, got {x.shape}.")
    if int(theta.shape[0]) != int(x.shape[0]):
        raise ValueError(f"theta_all and x_all row counts differ: {theta.shape[0]} vs {x.shape[0]}.")

    means = np.asarray(meta.get("mog_component_means"), dtype=np.float64)
    variances = np.asarray(meta.get("mog_component_variances"), dtype=np.float64)
    if means.shape != (5, x_dim):
        raise ValueError(f"Expected mog_component_means shape (5, {x_dim}), got {means.shape}.")
    if variances.shape != (5, x_dim):
        raise ValueError(f"Expected mog_component_variances shape (5, {x_dim}), got {variances.shape}.")
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
        raise ValueError("MoG component means/variances must be finite with strictly positive variances.")

    labels = _categorical_labels(theta, 5)
    if labels.shape != (int(x.shape[0]),):
        raise ValueError(f"Expected one label per row, got labels shape {labels.shape}.")
    if np.any((labels < 0) | (labels >= 5)):
        raise ValueError("MoG labels must be in [0, 4].")

    n = int(x.shape[0])
    cap = min(max(0, int(max_points)), n)
    if cap < n:
        rng = np.random.default_rng(0)
        plot_idx = np.sort(rng.choice(n, size=cap, replace=False))
    else:
        plot_idx = np.arange(n, dtype=np.int64)

    colors = [plt.get_cmap("tab10")(i) for i in range(5)]
    fig, ax = plt.subplots(figsize=(4.0, 3.5), constrained_layout=True)
    for cls in range(5):
        mask = labels[plot_idx] == cls
        if np.any(mask):
            pts = x[plot_idx][mask]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=12,
                alpha=0.55,
                color=colors[cls],
                edgecolors="none",
                label=f"category {cls}",
            )

    for cls in range(5):
        mu = means[cls, :2]
        sigma = np.sqrt(variances[cls, :2])
        ax.add_patch(
            Ellipse(
                xy=(float(mu[0]), float(mu[1])),
                width=float(2.0 * sigma[0]),
                height=float(2.0 * sigma[1]),
                angle=0.0,
                facecolor=colors[cls],
                edgecolor=colors[cls],
                linewidth=2.0,
                alpha=0.38,
                zorder=2,
            )
        )
        ax.scatter(
            [float(mu[0])],
            [float(mu[1])],
            marker="X",
            s=88,
            color=colors[cls],
            edgecolors="black",
            linewidths=1.0,
            zorder=4,
        )

    ax.set_title("MoG5 native dataset", fontsize=16)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()
    ax.legend(
        loc="best",
        frameon=False,
        fontsize=11,
        handletextpad=0.4,
        labelspacing=0.3,
    )

    svg_path = Path(svg_path)
    png_path = Path(png_path)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return svg_path, png_path


def pca_project(x: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return PCA projection and PCA basis for centered x."""
    x0 = x - x.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(x0, full_matrices=False)
    basis = vh[:n_components].T
    proj = x0 @ basis
    return proj, x.mean(axis=0), basis


def _project_covariances_to_basis(covariances: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project covariance matrices into the column space of ``basis``."""
    cov = np.asarray(covariances, dtype=np.float64)
    b = np.asarray(basis, dtype=np.float64)
    if cov.ndim != 3 or int(cov.shape[1]) != int(cov.shape[2]):
        raise ValueError(f"covariances must have shape (N, d, d); got {cov.shape}.")
    if b.ndim != 2 or int(b.shape[0]) != int(cov.shape[1]):
        raise ValueError(f"basis must have shape ({cov.shape[1]}, k); got {b.shape}.")
    return np.einsum("ia,nij,jb->nab", b, cov, b)


def _covariance_ellipse_parameters(cov_2d: np.ndarray) -> tuple[float, float, float]:
    """Return Matplotlib ellipse width, height, and angle for a 2D 1-sigma covariance."""
    cov = np.asarray(cov_2d, dtype=np.float64)
    if cov.shape != (2, 2):
        raise ValueError(f"cov_2d must have shape (2, 2); got {cov.shape}.")
    vals, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    width, height = 2.0 * np.sqrt(vals)
    return float(width), float(height), angle


def _draw_tuning_curves(ax_tune: "Axes", t: np.ndarray, mu: np.ndarray) -> None:
    for j in range(int(mu.shape[1])):
        ax_tune.plot(t[:, 0], mu[:, j], color="black", linewidth=2.0)
    ax_tune.set_xlabel(r"$\theta$")
    ax_tune.set_ylabel(r"Mean of $x|\theta$")
    ax_tune.set_box_aspect(1)


def _draw_pca_covariance_ellipses(
    fig: plt.Figure,
    ax_covariance: "Axes",
    t: np.ndarray,
    mu: np.ndarray,
    dataset: ToyConditionalGaussianDataset
    | ToyConditionalGMMNonGaussianDataset
    | ToyCosSinPiecewiseNoiseDataset
    | ToyLinearPiecewiseNoiseDataset,
    *,
    theta_norm: Normalize,
    sm_theta: ScalarMappable,
    n_covariance_ellipses: int = 10,
    manifold_trajectory_linewidth: float = 4.0,
) -> None:
    mu_pc, mean_mu, basis_mu = pca_project(mu, n_components=2)
    if int(mu_pc.shape[1]) < 2:
        pad = 2 - int(mu_pc.shape[1])
        mu_pc = np.concatenate([mu_pc, np.zeros((int(mu_pc.shape[0]), pad), dtype=np.float64)], axis=1)
    th_line = np.asarray(t, dtype=np.float64).ravel()
    if int(mu_pc.shape[0]) >= 2:
        pts = mu_pc[:, :2]
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        seg_theta = 0.5 * (th_line[:-1] + th_line[1:])
        lc_cov = LineCollection(
            segs,
            cmap="viridis",
            norm=theta_norm,
            array=seg_theta,
            linewidth=float(manifold_trajectory_linewidth),
            capstyle="round",
            zorder=2,
        )
        ax_covariance.add_collection(lc_cov)
    n_ell = max(0, int(n_covariance_ellipses))
    if n_ell > 0 and hasattr(dataset, "covariance"):
        theta_ell = np.linspace(float(dataset.theta_low), float(dataset.theta_high), n_ell, dtype=np.float64)[:, None]
        mu_ell = np.asarray(dataset.tuning_curve(theta_ell), dtype=np.float64)
        centers = (mu_ell - mean_mu) @ basis_mu
        cov_ell = np.asarray(dataset.covariance(theta_ell), dtype=np.float64)
        cov_pc = _project_covariances_to_basis(cov_ell, basis_mu[:, :2])
        for center, cov2, th_val in zip(centers[:, :2], cov_pc, theta_ell[:, 0], strict=True):
            width, height, angle = _covariance_ellipse_parameters(cov2)
            color = sm_theta.to_rgba(float(th_val))
            fill_color = (color[0], color[1], color[2], 0.2)
            edge_color = (color[0], color[1], color[2], 0.9)
            ax_covariance.add_patch(
                Ellipse(
                    xy=(float(center[0]), float(center[1])),
                    width=width,
                    height=height,
                    angle=angle,
                    facecolor=fill_color,
                    edgecolor=edge_color,
                    linestyle="--",
                    linewidth=1.15,
                    zorder=1,
                )
            )
    ax_covariance.scatter(
        mu_pc[:, 0],
        mu_pc[:, 1],
        c=th_line,
        s=8,
        alpha=0.45,
        cmap="viridis",
        norm=theta_norm,
        zorder=3,
    )
    ax_covariance.set_aspect("equal", adjustable="datalim")
    ax_covariance.set_axis_off()
    ax_covariance.set_box_aspect(1)
    cbar_cov = fig.colorbar(sm_theta, ax=ax_covariance, fraction=0.035, pad=0.02)
    cbar_cov.set_label(r"$\theta$")


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

    if str(getattr(dataset, "theta_type", "")) == "categorical":
        k = int(getattr(dataset, "num_categories", theta.shape[1] if np.asarray(theta).ndim == 2 else 1))
        labels = _categorical_labels(theta, k)
        counts = np.bincount(np.clip(labels, 0, k - 1), minlength=k)
        empirical = np.zeros((k, x.shape[1]), dtype=np.float64)
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                empirical[c] = x[mask].mean(axis=0)
        model_mean = dataset.tuning_curve(np.eye(k, dtype=np.float64))
        used = counts > 0
        mae = np.mean(np.abs(empirical[used] - model_mean[used])) if used.any() else np.nan
        print(f"  category counts: {counts.tolist()}")
        print(f"  empirical E[x|category] vs component mean MAE (all dims): {mae:.4f}")
        return

    theta_arr = np.asarray(theta, dtype=np.float64)
    theta_dim = int(theta_arr.shape[1]) if theta_arr.ndim == 2 else 1
    if theta_dim != 1:
        return
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


def _plot_joint_and_tuning_2d(
    theta: np.ndarray,
    x: np.ndarray,
    dataset: ToyConditionalGaussianDataset
    | ToyConditionalGMMNonGaussianDataset
    | ToyCosSinPiecewiseNoiseDataset
    | ToyLinearPiecewiseNoiseDataset,
    out_path: str,
) -> None:
    th = np.asarray(theta, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)
    if th.ndim != 2 or int(th.shape[1]) != 2:
        raise ValueError(f"2D visualization requires theta shape (N, 2); got {th.shape}.")
    n_heat = min(4, int(xx.shape[1]))
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.6), layout="constrained")
    t1 = np.linspace(float(dataset.theta_low), float(dataset.theta_high), 80)
    t2 = np.linspace(float(dataset.theta_low), float(dataset.theta_high), 80)
    g1, g2 = np.meshgrid(t1, t2, indexing="xy")
    grid = np.column_stack([g1.ravel(), g2.ravel()])
    mu = np.asarray(dataset.tuning_curve(grid), dtype=np.float64)
    for j in range(n_heat):
        ax = axes.ravel()[j]
        im = ax.imshow(
            mu[:, j].reshape(g1.shape),
            origin="lower",
            extent=[t1[0], t1[-1], t2[0], t2[-1]],
            aspect="auto",
            cmap="viridis",
        )
        ax.set_title(rf"$\mu_{{{j + 1}}}(\theta_1,\theta_2)$")
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for j in range(n_heat, 4):
        axes.ravel()[j].set_axis_off()

    proj, _, _ = pca_project(xx, n_components=2)
    for k, color_idx in enumerate((0, 1)):
        ax = axes.ravel()[4 + k]
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=th[:, color_idx], s=8, alpha=0.45, cmap="viridis")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
        ax.set_title(rf"PCA $x$ colored by $\theta_{color_idx + 1}$")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    svg_path = str(Path(out_path).with_suffix(".svg"))
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


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
    ax_covariance: "Axes | None" = None,
    n_covariance_ellipses: int = 10,
) -> None:
    """Draw tuning curves on ``ax_tune`` and manifold / samples on ``ax_manifold``.

    For ``x_dim >= 2``, ``ax_manifold`` shows the mean curve in its PCA plane as a ``theta``-colored
    polyline (``LineCollection``) plus light vertex markers, with sample projections overlaid in the
    same basis. When ``ax_covariance`` is provided, it shows the same PCA mean curve with projected
    1-sigma covariance ellipses and no sample scatter. For ``x_dim == 1``, it overlays
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

    _draw_tuning_curves(ax_tune, t, mu)

    if n_plot >= 2:
        mu_pc, mean_mu, basis_mu = pca_project(mu, n_components=2)
        if int(mu_pc.shape[1]) < 2:
            pad = 2 - int(mu_pc.shape[1])
            mu_pc = np.concatenate(
                [mu_pc, np.zeros((int(mu_pc.shape[0]), pad), dtype=np.float64)], axis=1
            )
        x0 = x_plot - mean_mu
        proj_x = x0 @ basis_mu
        if int(proj_x.shape[1]) < 2:
            pad = 2 - int(proj_x.shape[1])
            proj_x = np.concatenate(
                [proj_x, np.zeros((int(proj_x.shape[0]), pad), dtype=np.float64)], axis=1
            )
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
        if ax_covariance is not None:
            _draw_pca_covariance_ellipses(
                fig,
                ax_covariance,
                t,
                mu,
                dataset,
                theta_norm=theta_norm,
                sm_theta=sm_theta,
                n_covariance_ellipses=int(n_covariance_ellipses),
                manifold_trajectory_linewidth=float(manifold_trajectory_linewidth),
            )
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
        if ax_covariance is not None:
            ax_covariance.set_axis_off()

    ax_tune.set_box_aspect(1)


def plot_tuning_and_covariance_on_axes(
    fig: plt.Figure,
    ax_tune: "Axes",
    ax_covariance: "Axes",
    dataset: ToyConditionalGaussianDataset
    | ToyConditionalGMMNonGaussianDataset
    | ToyCosSinPiecewiseNoiseDataset
    | ToyLinearPiecewiseNoiseDataset,
    *,
    n_theta_grid: int = 500,
    n_covariance_ellipses: int = 10,
    manifold_trajectory_linewidth: float = 4.0,
) -> None:
    """Draw scalar continuous tuning curves and the PCA covariance-ellipse panel."""
    if str(getattr(dataset, "theta_type", "")) == "categorical":
        raise ValueError("plot_tuning_and_covariance_on_axes requires a scalar continuous dataset.")
    theta_dim = int(getattr(dataset, "theta_dim", 1))
    if theta_dim != 1:
        raise ValueError("plot_tuning_and_covariance_on_axes requires theta_dim == 1.")
    x_dim = int(getattr(dataset, "x_dim", 0))
    if x_dim < 2:
        raise ValueError("plot_tuning_and_covariance_on_axes requires x_dim >= 2.")
    t = np.linspace(float(dataset.theta_low), float(dataset.theta_high), int(n_theta_grid), dtype=np.float64)[:, None]
    mu = np.asarray(dataset.tuning_curve(t), dtype=np.float64)
    if mu.ndim != 2 or int(mu.shape[1]) < 2:
        raise ValueError(f"Expected tuning curve shape (T, d>=2), got {mu.shape}.")

    theta_norm = Normalize(vmin=float(dataset.theta_low), vmax=float(dataset.theta_high))
    sm_theta = ScalarMappable(norm=theta_norm, cmap="viridis")
    sm_theta.set_array([])
    _draw_tuning_curves(ax_tune, t, mu)
    _draw_pca_covariance_ellipses(
        fig,
        ax_covariance,
        t,
        mu,
        dataset,
        theta_norm=theta_norm,
        sm_theta=sm_theta,
        n_covariance_ellipses=int(n_covariance_ellipses),
        manifold_trajectory_linewidth=float(manifold_trajectory_linewidth),
    )


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
    """Plot dataset diagnostics: tuning curves plus PCA/sample and covariance views.

    When ``x_dim >= 2`` for scalar continuous datasets, this uses one ``pca_project`` on the dense
    mean curve ``mu``; samples and covariance ellipses are projected with the same ``mean_mu`` and
    ``basis_mu``. A colorbar maps color to ``theta``. One-dimensional, categorical, and native
    two-dimensional-theta visualizations keep their previous layouts.

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

    # Native (theta_1, theta_2) grids only: K-way one-hot rows also have width 2 when K=2.
    if (
        theta_plot.ndim == 2
        and int(theta_plot.shape[1]) == 2
        and str(getattr(dataset, "theta_type", "")) != "categorical"
    ):
        _plot_joint_and_tuning_2d(theta_plot, x_plot, dataset, out_path)
        return

    if str(getattr(dataset, "theta_type", "")) == "categorical":
        fig, (ax_tune, ax_manifold) = plt.subplots(1, 2, figsize=(9.0, 3.2), layout="constrained")
        k = int(getattr(dataset, "num_categories", theta_plot.shape[1] if np.asarray(theta_plot).ndim == 2 else 1))
        labels = _categorical_labels(theta_plot, k)
        cats = np.eye(k, dtype=np.float64)
        mu = np.asarray(dataset.tuning_curve(cats), dtype=np.float64)
        for j in range(mu.shape[1]):
            ax_tune.plot(np.arange(k), mu[:, j], color="black", linewidth=1.5, alpha=0.8)
        ax_tune.set_xlabel("category")
        ax_tune.set_ylabel(r"Mean of $x|category$")
        cat_norm = Normalize(
            vmin=float(dataset.theta_low),
            vmax=float(dataset.theta_high),
        )
        if x_plot.shape[1] >= 2:
            proj_x, _, _ = pca_project(x_plot, n_components=2)
            sc = ax_manifold.scatter(
                proj_x[:, 0],
                proj_x[:, 1],
                c=labels.astype(np.float64),
                s=8,
                alpha=0.45,
                cmap="viridis",
                norm=cat_norm,
            )
            ax_manifold.set_aspect("equal", adjustable="datalim")
            ax_manifold.set_axis_off()
            fig.colorbar(sc, ax=ax_manifold, fraction=0.035, pad=0.02).set_label("category")
        else:
            ax_manifold.scatter(
                labels,
                x_plot[:, 0],
                c=labels.astype(np.float64),
                s=10,
                alpha=0.45,
                cmap="viridis",
                norm=cat_norm,
            )
            ax_manifold.set_xlabel("category")
            ax_manifold.set_ylabel(r"$x_1$")
        ax_tune.set_box_aspect(1)
        ax_manifold.set_box_aspect(1)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        svg_path = str(Path(out_path).with_suffix(".svg"))
        fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)
        return

    if x_plot.ndim == 2 and int(x_plot.shape[1]) >= 2:
        fig, (ax_tune, ax_manifold, ax_covariance) = plt.subplots(
            1, 3, figsize=(12.6, 3.2), layout="constrained"
        )
    else:
        fig, (ax_tune, ax_manifold) = plt.subplots(1, 2, figsize=(9.0, 3.2), layout="constrained")
        ax_covariance = None

    plot_joint_and_tuning_on_axes(
        fig,
        ax_tune,
        ax_manifold,
        theta_plot,
        x_plot,
        dataset,
        mu_override=None,
        ax_covariance=ax_covariance,
    )

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    svg_path = str(Path(out_path).with_suffix(".svg"))
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
