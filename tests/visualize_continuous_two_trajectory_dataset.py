#!/usr/bin/env python3
"""Visualize the current continuous Gaussian model and a two-trajectory mixture.

Run from the repository root:
    mamba run -n geo_diffusion python tests/visualize_continuous_two_trajectory_dataset.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, Normalize, to_rgba
from matplotlib.patches import Ellipse

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import DATA_DIR
from fisher.data import (
    ToyConditionalGaussianRandampSqrtdDataset,
    ToyConditionalGaussianRandampSqrtdTwoTrajectoryDataset,
)
from fisher.dataset_family_recipes import family_recipe_dict
from fisher.dataset_visualization import _covariance_ellipse_parameters, _project_covariances_to_basis


def _dataset_kwargs(
    *,
    x_dim: int,
    seed: int,
    covariance_alpha: float | None,
) -> dict[str, float | int | str]:
    recipe = family_recipe_dict("randamp_gaussian_sqrtd")
    keys = (
        "sigma_x1",
        "sigma_x2",
        "cov_theta_amp1",
        "cov_theta_amp2",
        "randamp_mu_low",
        "randamp_mu_high",
        "randamp_kappa",
        "randamp_omega",
    )
    kwargs: dict[str, float | int | str] = {
        "theta_low": -6.0,
        "theta_high": 6.0,
        "x_dim": int(x_dim),
        "seed": int(seed),
        **{key: recipe[key] for key in keys},
    }
    if covariance_alpha is not None:
        if not np.isfinite(covariance_alpha) or covariance_alpha < 0.0:
            raise ValueError("--covariance-alpha must be finite and non-negative.")
        kwargs["cov_theta_amp1"] = float(covariance_alpha)
        kwargs["cov_theta_amp2"] = float(covariance_alpha)
    return kwargs


def _fit_pca(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a shared two-dimensional PCA basis to high-dimensional samples."""
    center = x.mean(axis=0)
    _, singular_values, vh = np.linalg.svd(x - center, full_matrices=False)
    explained_ratio = singular_values**2 / np.sum(singular_values**2)
    return center, vh[:2].T, explained_ratio[:2]


def _draw_covariance_ellipses(
    ax: plt.Axes,
    *,
    centers: np.ndarray,
    covariances: np.ndarray,
    basis: np.ndarray,
    theta: np.ndarray,
    cmap: Colormap,
    norm: Normalize,
) -> None:
    """Draw projected one-standard-deviation covariance ellipses."""
    projected = _project_covariances_to_basis(covariances, basis)
    for center, covariance, theta_value in zip(
        centers, projected, np.asarray(theta).reshape(-1), strict=True
    ):
        color = cmap(norm(float(theta_value)))
        width, height, angle = _covariance_ellipse_parameters(covariance)
        ax.add_patch(
            Ellipse(
                xy=(float(center[0]), float(center[1])),
                width=width,
                height=height,
                angle=angle,
                facecolor=to_rgba(color, alpha=0.10),
                edgecolor=to_rgba(color, alpha=0.68),
                linestyle="--",
                linewidth=1.2,
                zorder=2,
            )
        )


def _draw_colored_trajectory(
    ax: plt.Axes,
    *,
    trajectory: np.ndarray,
    theta: np.ndarray,
    cmap: Colormap,
    norm: Normalize,
    linestyle: str,
) -> None:
    points = np.asarray(trajectory, dtype=np.float64)
    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    segments = np.stack([points[:-1], points[1:]], axis=1)
    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=3.3,
        linestyle=linestyle,
        zorder=3,
    )
    collection.set_array(0.5 * (values[:-1] + values[1:]))
    ax.add_collection(collection)
    ax.update_datalim(points)
    ax.autoscale_view()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = Path(DATA_DIR) / "continuous_two_trajectory_dataset_visualization"
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    parser.add_argument("--x-dim", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--n-covariance-ellipses", type=int, default=8)
    parser.add_argument(
        "--covariance-alpha",
        type=float,
        default=None,
        help="Override alpha in Var_j(theta) = d*sigma_b^2 + alpha*abs(mu_j(theta)).",
    )
    parser.add_argument(
        "--pca-fit",
        choices=("clean-trajectory", "pooled-samples"),
        default="clean-trajectory",
        help="Fit the shared PCA basis to mu(theta) or to pooled noisy samples.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    kwargs = _dataset_kwargs(
        x_dim=int(args.x_dim),
        seed=int(args.seed),
        covariance_alpha=args.covariance_alpha,
    )
    current = ToyConditionalGaussianRandampSqrtdDataset(**kwargs)
    mixture = ToyConditionalGaussianRandampSqrtdTwoTrajectoryDataset(
        **kwargs,
        randamp_mu_amp_per_dim=current._randamp_amp.copy(),
    )

    theta_current, x_current = current.sample_joint(int(args.n_samples))
    theta_mixture = mixture.sample_theta(int(args.n_samples))
    x_mixture, component = mixture.sample_x_with_component(theta_mixture)

    theta_grid = np.linspace(current.theta_low, current.theta_high, 500, dtype=np.float64)[:, None]
    base_curve = current.tuning_curve(theta_grid)
    second_curve = 2.0 * base_curve
    if args.pca_fit == "clean-trajectory":
        pca_input = base_curve
    else:
        pca_input = np.concatenate([x_current, x_mixture], axis=0)
    center, basis, _ = _fit_pca(pca_input)
    base_pc = (base_curve - center) @ basis
    second_pc = (second_curve - center) @ basis
    ellipse_theta = np.linspace(
        current.theta_low,
        current.theta_high,
        max(1, int(args.n_covariance_ellipses)),
        dtype=np.float64,
    )[:, None]
    ellipse_base = current.tuning_curve(ellipse_theta)
    ellipse_base_pc = (ellipse_base - center) @ basis
    ellipse_second_pc = (2.0 * ellipse_base - center) @ basis
    component_covariance = current.covariance(ellipse_theta)

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "legend.fontsize": 14,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    cmap = plt.get_cmap("viridis")
    theta_norm = Normalize(vmin=current.theta_low, vmax=current.theta_high)
    fig, axes = plt.subplots(
        1, 2, figsize=(8.0, 3.5), sharex=True, sharey=True, constrained_layout=True
    )

    _draw_covariance_ellipses(
        axes[0],
        centers=ellipse_base_pc,
        covariances=component_covariance,
        basis=basis,
        theta=ellipse_theta,
        cmap=cmap,
        norm=theta_norm,
    )
    _draw_colored_trajectory(
        axes[0],
        trajectory=base_pc,
        theta=theta_grid,
        cmap=cmap,
        norm=theta_norm,
        linestyle="solid",
    )
    axes[0].set_title("Conditional Gaussian")
    for label, linestyle in ((0, "solid"), (1, "dashed")):
        trajectory = base_pc if label == 0 else second_pc
        ellipse_centers = ellipse_base_pc if label == 0 else ellipse_second_pc
        _draw_covariance_ellipses(
            axes[1],
            centers=ellipse_centers,
            covariances=component_covariance,
            basis=basis,
            theta=ellipse_theta,
            cmap=cmap,
            norm=theta_norm,
        )
        _draw_colored_trajectory(
            axes[1],
            trajectory=trajectory,
            theta=theta_grid,
            cmap=cmap,
            norm=theta_norm,
            linestyle=linestyle,
        )
    axes[1].set_title("Two-trajectory Gaussian mixture")
    for ax in axes:
        ax.set_axis_off()
        ax.set_aspect("equal", adjustable="box")
    colorbar = fig.colorbar(
        ScalarMappable(norm=theta_norm, cmap=cmap),
        ax=axes,
        orientation="vertical",
        fraction=0.035,
        pad=0.03,
        aspect=28,
    )
    colorbar.set_label(r"$\theta$", fontsize=16, rotation=0, labelpad=12)
    colorbar.set_ticks(np.linspace(current.theta_low, current.theta_high, 5))
    colorbar.ax.tick_params(labelsize=14, width=1.4)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "continuous_gaussian_vs_two_trajectory_mixture.png"
    svg_path = args.output_dir / "continuous_gaussian_vs_two_trajectory_mixture.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    print(f"current_dataset: X|theta ~ N(mu(theta), Sigma(theta)); samples={x_current.shape[0]}")
    print("two_trajectory_dataset: 0.5 N(mu(theta), Sigma(theta)) + 0.5 N(2 mu(theta), Sigma(theta))")
    print(f"covariance_alpha: {current._sigma_activity_alpha[0]:.6g}")
    print(f"pca_fit: {args.pca_fit}")
    print(f"observed_secondary_fraction: {float(component.mean()):.4f}")
    print(f"png: {png_path.resolve()}")
    print(f"svg: {svg_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
