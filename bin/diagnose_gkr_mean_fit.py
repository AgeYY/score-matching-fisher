#!/usr/bin/env python3
"""Compare a cached GKR conditional mean with the analytic toy mean."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.shared_fisher_est import build_dataset_from_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize analytic and cached GKR means for every response dimension."
    )
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--result-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-columns", type=int, default=10)
    return parser.parse_args()


def _load_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=True) as archive:
        raw = np.asarray(archive["meta_json_utf8"], dtype=np.uint8).tobytes()
    return json.loads(raw.decode("utf-8"))


def _style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(width=1.5)
    ax.grid(False)


def _save_heatmaps(
    *,
    theta: np.ndarray,
    ground_truth: np.ndarray,
    predicted: np.ndarray,
    output_stem: Path,
) -> None:
    residual = predicted - ground_truth
    mean_limit = float(np.max(np.abs(np.concatenate([ground_truth, predicted], axis=0))))
    residual_limit = float(np.max(np.abs(residual)))
    extent = [float(theta[0]), float(theta[-1]), 0.5, ground_truth.shape[1] + 0.5]

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.grid": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    panels = (
        (ground_truth.T, "Ground-truth mean", -mean_limit, mean_limit),
        (predicted.T, "GKR mean", -mean_limit, mean_limit),
        (residual.T, "GKR minus ground truth", -residual_limit, residual_limit),
    )
    for ax, (values, title, vmin, vmax) in zip(axes, panels, strict=True):
        image = ax.imshow(
            values,
            aspect="auto",
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel("Response dimension")
        _style_axis(ax)
        colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.20, fraction=0.08)
        colorbar.ax.tick_params(labelsize=12, width=1.2)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _save_curve_grid(
    *,
    theta: np.ndarray,
    ground_truth: np.ndarray,
    predicted: np.ndarray,
    per_dimension_rmse: np.ndarray,
    output_stem: Path,
    columns: int,
) -> None:
    dimension = int(ground_truth.shape[1])
    ncols = min(max(1, int(columns)), dimension)
    nrows = int(math.ceil(dimension / ncols))
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 13,
            "axes.grid": False,
        }
    )
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.0 * ncols, 1.6 * nrows),
        sharex=True,
        squeeze=False,
    )
    for index, ax in enumerate(axes.flat):
        if index >= dimension:
            ax.set_visible(False)
            continue
        ax.plot(theta, ground_truth[:, index], color="black", linewidth=1.8, label="Ground truth")
        ax.plot(theta, predicted[:, index], color="C1", linewidth=1.5, label="GKR")
        ax.set_title(f"x{index + 1}  RMSE={per_dimension_rmse[index]:.2f}")
        _style_axis(ax)
        if index // ncols == nrows - 1:
            ax.set_xlabel(r"$\theta$")
        if index % ncols == 0:
            ax.set_ylabel("Mean")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(left=0.05, right=0.995, bottom=0.05, top=0.95, hspace=0.55, wspace=0.30)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_npz = args.dataset_npz.resolve()
    result_npz = args.result_npz.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(dataset_npz)
    dataset = build_dataset_from_meta(metadata)
    with np.load(result_npz, allow_pickle=True) as archive:
        theta = np.asarray(archive["theta_midpoints"], dtype=np.float64).reshape(-1, 1)
        predicted_mean = np.asarray(archive["gkr_mean"], dtype=np.float64)
        predicted_jacobian = np.asarray(archive["gkr_mean_jacobian"], dtype=np.float64)

    ground_truth_mean = np.asarray(dataset.tuning_curve(theta), dtype=np.float64)
    ground_truth_jacobian = np.asarray(dataset.tuning_curve_derivative(theta), dtype=np.float64)
    if ground_truth_mean.shape != predicted_mean.shape:
        raise ValueError(
            f"Mean shape mismatch: ground truth {ground_truth_mean.shape}, GKR {predicted_mean.shape}."
        )
    predicted_jacobian = predicted_jacobian.reshape(theta.shape[0], ground_truth_mean.shape[1], -1)
    ground_truth_jacobian = ground_truth_jacobian.reshape(theta.shape[0], ground_truth_mean.shape[1], -1)
    if predicted_jacobian.shape != ground_truth_jacobian.shape:
        raise ValueError(
            "Mean-Jacobian shape mismatch: "
            f"ground truth {ground_truth_jacobian.shape}, GKR {predicted_jacobian.shape}."
        )

    mean_residual = predicted_mean - ground_truth_mean
    jacobian_residual = predicted_jacobian - ground_truth_jacobian
    mean_rmse_by_dimension = np.sqrt(np.mean(mean_residual**2, axis=0))
    mean_mae_by_dimension = np.mean(np.abs(mean_residual), axis=0)
    jacobian_rmse_by_dimension = np.sqrt(np.mean(jacobian_residual**2, axis=(0, 2)))
    mean_scale = np.sqrt(np.mean(ground_truth_mean**2, axis=0))
    jacobian_scale = np.sqrt(np.mean(ground_truth_jacobian**2, axis=(0, 2)))
    mean_nrmse = mean_rmse_by_dimension / np.maximum(mean_scale, 1e-12)
    jacobian_nrmse = jacobian_rmse_by_dimension / np.maximum(jacobian_scale, 1e-12)

    _save_heatmaps(
        theta=theta[:, 0],
        ground_truth=ground_truth_mean,
        predicted=predicted_mean,
        output_stem=output_dir / "gkr_mean_fit_heatmaps",
    )
    _save_curve_grid(
        theta=theta[:, 0],
        ground_truth=ground_truth_mean,
        predicted=predicted_mean,
        per_dimension_rmse=mean_rmse_by_dimension,
        output_stem=output_dir / "gkr_mean_fit_all_dimensions",
        columns=args.grid_columns,
    )

    csv_path = output_dir / "gkr_mean_fit_per_dimension.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "response_dimension",
                "mean_mae",
                "mean_rmse",
                "mean_nrmse",
                "mean_derivative_rmse",
                "mean_derivative_nrmse",
            ]
        )
        for index in range(ground_truth_mean.shape[1]):
            writer.writerow(
                [
                    index + 1,
                    float(mean_mae_by_dimension[index]),
                    float(mean_rmse_by_dimension[index]),
                    float(mean_nrmse[index]),
                    float(jacobian_rmse_by_dimension[index]),
                    float(jacobian_nrmse[index]),
                ]
            )

    worst_mean = np.argsort(mean_nrmse)[::-1][:10]
    worst_jacobian = np.argsort(jacobian_nrmse)[::-1][:10]
    summary = {
        "dataset_npz": str(dataset_npz),
        "result_npz": str(result_npz),
        "n_total": int(metadata["n_total"]),
        "n_train": int(round(float(metadata["train_frac"]) * int(metadata["n_total"]))),
        "x_dim": int(ground_truth_mean.shape[1]),
        "theta_points": int(theta.shape[0]),
        "mean_mae": float(np.mean(np.abs(mean_residual))),
        "mean_rmse": float(np.sqrt(np.mean(mean_residual**2))),
        "mean_relative_rmse": float(
            np.sqrt(np.mean(mean_residual**2))
            / max(float(np.sqrt(np.mean(ground_truth_mean**2))), 1e-12)
        ),
        "mean_derivative_rmse": float(np.sqrt(np.mean(jacobian_residual**2))),
        "mean_derivative_relative_rmse": float(
            np.sqrt(np.mean(jacobian_residual**2))
            / max(float(np.sqrt(np.mean(ground_truth_jacobian**2))), 1e-12)
        ),
        "worst_mean_dimensions_by_nrmse": [int(index + 1) for index in worst_mean],
        "worst_derivative_dimensions_by_nrmse": [int(index + 1) for index in worst_jacobian],
    }
    summary_path = output_dir / "gkr_mean_fit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"heatmap_png: {(output_dir / 'gkr_mean_fit_heatmaps.png')}")
    print(f"curve_grid_png: {(output_dir / 'gkr_mean_fit_all_dimensions.png')}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
