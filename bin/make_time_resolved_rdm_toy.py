#!/usr/bin/env python3
"""Generate and visualize a labeled two-trajectory time-resolved RDM toy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher.dataset_visualization import (  # noqa: E402
    _covariance_ellipse_parameters,
    _project_covariances_to_basis,
)
from fisher.time_resolved_rdm_toy import (  # noqa: E402
    TwoClassTimeResolvedGaussianToy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/time_resolved_rdm_toy_xdim40_n100_per_class",
    )
    parser.add_argument("--x-dim", type=int, default=40)
    parser.add_argument("--n-trials-per-class", type=int, default=100)
    parser.add_argument("--n-time-points", type=int, default=301)
    parser.add_argument("--time-low", type=float, default=-6.0)
    parser.add_argument("--time-high", type=float, default=6.0)
    parser.add_argument("--secondary-scale", type=float, default=2.0)
    parser.add_argument("--covariance-alpha", type=float, default=0.65)
    parser.add_argument("--n-covariance-ellipses", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _fit_clean_trajectory_pca(base_mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(base_mean, axis=0)
    _, _, right_vectors = np.linalg.svd(base_mean - center, full_matrices=False)
    return center, right_vectors[:2].T


def _draw_colored_trajectory(
    axis: plt.Axes,
    trajectory: np.ndarray,
    time: np.ndarray,
    *,
    cmap: Colormap,
    norm: Normalize,
    linestyle: str,
) -> None:
    points = np.asarray(trajectory, dtype=np.float64)
    segments = np.stack([points[:-1], points[1:]], axis=1)
    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=3.0,
        linestyle=linestyle,
        zorder=3,
    )
    collection.set_array(0.5 * (time[:-1] + time[1:]))
    axis.add_collection(collection)
    axis.update_datalim(points)
    axis.autoscale_view()


def _draw_covariance_ellipses(
    axis: plt.Axes,
    centers: np.ndarray,
    covariances: np.ndarray,
    basis: np.ndarray,
    times: np.ndarray,
    *,
    cmap: Colormap,
    norm: Normalize,
) -> None:
    projected = _project_covariances_to_basis(covariances, basis)
    for center, covariance, time_value in zip(
        centers, projected, times, strict=True
    ):
        color = cmap(norm(float(time_value)))
        width, height, angle = _covariance_ellipse_parameters(covariance)
        axis.add_patch(
            Ellipse(
                xy=(float(center[0]), float(center[1])),
                width=width,
                height=height,
                angle=angle,
                facecolor=to_rgba(color, alpha=0.08),
                edgecolor=to_rgba(color, alpha=0.55),
                linestyle="--",
                linewidth=1.0,
                zorder=2,
            )
        )


def _plot_dataset(
    dataset: TwoClassTimeResolvedGaussianToy,
    output_dir: Path,
    *,
    n_covariance_ellipses: int,
) -> None:
    center, basis = _fit_clean_trajectory_pca(dataset.base_mean)
    projected_means = np.einsum(
        "ctd,dk->ctk", dataset.class_means - center[None, None, :], basis
    )
    ellipse_indices = np.linspace(
        0,
        dataset.time.size - 1,
        max(1, int(n_covariance_ellipses)),
        dtype=np.int64,
    )
    ellipse_times = dataset.time[ellipse_indices]
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=float(dataset.time[0]), vmax=float(dataset.time[-1]))

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 16,
            "legend.fontsize": 14,
            "axes.grid": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )
    figure, axis = plt.subplots(figsize=(4.0, 3.5), layout="constrained")
    linestyles = ("solid", "dashed")
    for class_index in range(2):
        _draw_covariance_ellipses(
            axis,
            projected_means[class_index, ellipse_indices],
            dataset.shared_covariances[ellipse_indices],
            basis,
            ellipse_times,
            cmap=cmap,
            norm=norm,
        )
        _draw_colored_trajectory(
            axis,
            projected_means[class_index],
            dataset.time,
            cmap=cmap,
            norm=norm,
            linestyle=linestyles[class_index],
        )
    axis.set_title("Two labeled time trajectories")
    axis.set_axis_off()
    axis.set_aspect("equal", adjustable="box")
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="0.15",
            linewidth=2.8,
            linestyle="solid",
            label=r"Class 1: $\mu(t)$",
        ),
        Line2D(
            [0],
            [0],
            color="0.15",
            linewidth=2.8,
            linestyle="dashed",
            label=rf"Class 2: ${float(dataset.secondary_trajectory_scale):g}\mu(t)$",
        ),
    ]
    axis.legend(handles=legend_handles, frameon=False, loc="best")
    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axis,
        orientation="vertical",
        fraction=0.055,
        pad=0.03,
        aspect=24,
    )
    colorbar.set_label("Time", fontsize=16)
    colorbar.set_ticks(np.linspace(dataset.time[0], dataset.time[-1], 5))
    colorbar.ax.tick_params(labelsize=14, width=1.4)

    stem = "two_class_time_resolved_trajectories"
    figure.savefig(output_dir / f"{stem}.png", dpi=300, pad_inches=0.12)
    figure.savefig(output_dir / f"{stem}.svg", pad_inches=0.12)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    dataset = TwoClassTimeResolvedGaussianToy(
        x_dim=int(args.x_dim),
        n_time_points=int(args.n_time_points),
        time_low=float(args.time_low),
        time_high=float(args.time_high),
        secondary_trajectory_scale=float(args.secondary_scale),
        covariance_alpha=float(args.covariance_alpha),
        seed=int(args.seed),
    )
    responses, labels = dataset.sample_trials(int(args.n_trials_per_class))
    metadata = dataset.metadata(n_trials_per_class=int(args.n_trials_per_class))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "two_class_time_resolved_rdm_toy.npz",
        time=dataset.time,
        responses=responses,
        labels=labels,
        class_names=np.asarray(metadata["class_names"]),
        class_scales=dataset.class_scales,
        true_class_means=dataset.class_means,
        true_shared_covariances=dataset.shared_covariances,
        true_squared_euclidean=dataset.true_squared_euclidean_distance(),
        true_squared_mahalanobis=dataset.true_squared_mahalanobis_distance(),
        tuning_amplitudes=dataset.base_dataset._randamp_amp,
        tuning_centers=dataset.base_dataset._tuning_centers_theta,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_dataset(
        dataset,
        args.output_dir,
        n_covariance_ellipses=int(args.n_covariance_ellipses),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    print(f"[dataset] output={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
