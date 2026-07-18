#!/usr/bin/env python3
"""Plot the toy trajectories beside their correlation-distance estimates."""

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
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher.dataset_visualization import (  # noqa: E402
    _covariance_ellipse_parameters,
    _project_covariances_to_basis,
)


def parse_args() -> argparse.Namespace:
    dataset_dir = ROOT / "data/time_resolved_rdm_toy_xdim40_n100_per_class"
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset-npz",
        type=Path,
        default=dataset_dir / "two_class_time_resolved_rdm_toy.npz",
    )
    parser.add_argument(
        "--results-npz",
        type=Path,
        default=dataset_dir
        / "correlation_classical_flow"
        / "correlation_classical_flow_results.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=dataset_dir / "correlation_classical_flow",
    )
    parser.add_argument("--n-covariance-ellipses", type=int, default=16)
    return parser.parse_args()


def _fit_trajectory_pca(class_means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base_mean = np.asarray(class_means[0], dtype=np.float64)
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
        linewidth=2.8,
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
    projected_covariances = _project_covariances_to_basis(covariances, basis)
    for center, covariance, time_value in zip(
        centers, projected_covariances, times, strict=True
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
                edgecolor=to_rgba(color, alpha=0.50),
                linestyle="--",
                linewidth=0.9,
                zorder=2,
            )
        )


def main() -> None:
    args = parse_args()
    with np.load(args.dataset_npz, allow_pickle=False) as archive:
        time = np.asarray(archive["time"], dtype=np.float64)
        class_means = np.asarray(archive["true_class_means"], dtype=np.float64)
        shared_covariances = np.asarray(
            archive["true_shared_covariances"], dtype=np.float64
        )
        class_scales = np.asarray(archive["class_scales"], dtype=np.float64)
    with np.load(args.results_npz, allow_pickle=False) as archive:
        native_time = np.asarray(archive["native_time"], dtype=np.float64)
        flow_distance = np.asarray(
            archive["flow_correlation_distance"], dtype=np.float64
        )
        classical_bin_centers = np.asarray(
            archive["classical_bin_centers"], dtype=np.float64
        )
        classical_distance = np.asarray(
            archive["classical_correlation_distance"], dtype=np.float64
        )
        ground_truth = np.asarray(
            archive["true_correlation_distance"], dtype=np.float64
        )

    center, basis = _fit_trajectory_pca(class_means)
    projected_means = np.einsum(
        "ctd,dk->ctk", class_means - center[None, None, :], basis
    )
    ellipse_indices = np.linspace(
        0,
        time.size - 1,
        max(1, int(args.n_covariance_ellipses)),
        dtype=np.int64,
    )
    ellipse_times = time[ellipse_indices]
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=float(time[0]), vmax=float(time[-1]))

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
            "axes.grid": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), layout="constrained")

    trajectory_axis = axes[0]
    for class_index, linestyle in enumerate(("solid", "dashed")):
        _draw_covariance_ellipses(
            trajectory_axis,
            projected_means[class_index, ellipse_indices],
            shared_covariances[ellipse_indices],
            basis,
            ellipse_times,
            cmap=cmap,
            norm=norm,
        )
        _draw_colored_trajectory(
            trajectory_axis,
            projected_means[class_index],
            time,
            cmap=cmap,
            norm=norm,
            linestyle=linestyle,
        )
    trajectory_axis.set_title("Two-class trajectories")
    trajectory_axis.set_axis_off()
    trajectory_axis.set_aspect("equal", adjustable="box")
    trajectory_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="0.15",
                linewidth=2.6,
                label=rf"Class 1: ${class_scales[0]:g}\mu(t)$",
            ),
            Line2D(
                [0],
                [0],
                color="0.15",
                linewidth=2.6,
                linestyle="dashed",
                label=rf"Class 2: ${class_scales[1]:g}\mu(t)$",
            ),
        ],
        frameon=False,
        loc="best",
    )
    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=trajectory_axis,
        orientation="vertical",
        fraction=0.055,
        pad=0.02,
        aspect=24,
    )
    colorbar.set_label("Time")
    colorbar.set_ticks(np.linspace(time[0], time[-1], 5))
    colorbar.ax.tick_params(labelsize=11, width=1.2)

    distance_axis = axes[1]
    distance_axis.plot(
        classical_bin_centers,
        classical_distance,
        color="#4477AA",
        linewidth=1.7,
        marker="o",
        markersize=3.5,
        label="Classical",
        zorder=3,
    )
    distance_axis.plot(
        native_time,
        flow_distance,
        color="#CC6677",
        linewidth=2.0,
        label="Flow",
        zorder=2,
    )
    distance_axis.plot(
        native_time,
        ground_truth,
        color="0.15",
        linewidth=1.5,
        linestyle="--",
        label="Ground truth",
        zorder=1,
    )
    distance_axis.set_title("Correlation-distance estimates")
    distance_axis.set_xlabel("Time")
    distance_axis.set_ylabel("Correlation distance")
    distance_axis.set_xlim(float(native_time[0]), float(native_time[-1]))
    upper = max(float(np.max(classical_distance)), float(np.max(flow_distance)))
    distance_axis.set_ylim(-0.08 * upper, 1.08 * upper)
    distance_axis.spines["top"].set_visible(False)
    distance_axis.spines["right"].set_visible(False)
    distance_axis.tick_params(width=1.2, length=4)
    distance_axis.legend(frameon=False, loc="upper right")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "time_resolved_rdm_toy_trajectory_and_correlation_estimation"
    png_path = args.output_dir / f"{stem}.png"
    svg_path = args.output_dir / f"{stem}.svg"
    figure.savefig(png_path, dpi=300, pad_inches=0.08)
    figure.savefig(svg_path, pad_inches=0.08)
    plt.close(figure)
    print(f"Saved: {png_path.resolve()}")
    print(f"Saved: {svg_path.resolve()}")


if __name__ == "__main__":
    main()
