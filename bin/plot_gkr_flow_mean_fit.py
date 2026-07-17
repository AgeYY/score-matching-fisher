#!/usr/bin/env python3
"""Compare analytic, GKR, and flow conditional means on a shared theta grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.flow_matching_skl import build_flow_skl_model
from fisher.shared_fisher_est import build_dataset_from_meta, require_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--gkr-result-npz", type=Path, required=True)
    parser.add_argument("--flow-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--path-schedule", default="cosine")
    parser.add_argument(
        "--theta-embedding",
        choices=("identity", "gaussian-rbf"),
        default="gaussian-rbf",
    )
    parser.add_argument("--theta-rbf-num-centers", type=int, default=8)
    parser.add_argument("--theta-rbf-bandwidth", type=float, default=None)
    parser.add_argument("--theta-input-shift", type=float, default=0.0)
    return parser.parse_args()


def _load_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=True) as archive:
        raw = np.asarray(archive["meta_json_utf8"], dtype=np.uint8).tobytes()
    return json.loads(raw.decode("utf-8"))


def _save_heatmaps(
    *,
    theta: np.ndarray,
    ground_truth: np.ndarray,
    gkr: np.ndarray,
    flow: np.ndarray,
    output_stem: Path,
) -> None:
    arrays = (ground_truth, gkr, flow)
    value_min = float(min(np.min(values) for values in arrays))
    value_max = float(max(np.max(values) for values in arrays))
    gkr_rmse = float(np.sqrt(np.mean((gkr - ground_truth) ** 2)))
    flow_rmse = float(np.sqrt(np.mean((flow - ground_truth) ** 2)))
    extent = [float(theta[0]), float(theta[-1]), 0.5, ground_truth.shape[1] + 0.5]

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), sharex=True, sharey=True)
    panels = (
        (ground_truth, "Ground truth"),
        (gkr, f"GKR (RMSE={gkr_rmse:.3f})"),
        (flow, f"Flow matching (RMSE={flow_rmse:.3f})"),
    )
    image = None
    for axis, (values, title) in zip(axes, panels, strict=True):
        image = axis.imshow(
            values.T,
            aspect="auto",
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap="viridis",
            vmin=value_min,
            vmax=value_max,
        )
        axis.set_title(title)
        axis.set_xlabel(r"$\theta$")
        for spine in axis.spines.values():
            spine.set_linewidth(1.8)
        axis.tick_params(width=1.8)
    axes[0].set_ylabel("Response dimension")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, orientation="horizontal", pad=0.23, fraction=0.08)
    colorbar.set_label("Conditional mean")
    colorbar.ax.tick_params(width=1.8)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_npz = args.dataset_npz.resolve()
    gkr_result_npz = args.gkr_result_npz.resolve()
    flow_model_path = args.flow_model.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = require_device(str(args.device))
    theta_input_shift = float(args.theta_input_shift)
    if not np.isfinite(theta_input_shift):
        raise ValueError("--theta-input-shift must be finite.")

    metadata = _load_metadata(dataset_npz)
    dataset = build_dataset_from_meta(metadata)
    with np.load(gkr_result_npz, allow_pickle=False) as archive:
        theta = np.asarray(archive["theta_midpoints"], dtype=np.float64).reshape(-1, 1)
        gkr_mean = np.asarray(archive["gkr_mean"], dtype=np.float64)

    ground_truth_mean = np.asarray(dataset.tuning_curve(theta), dtype=np.float64)
    model = build_flow_skl_model(
        velocity_family="condition_affine",
        theta_dim=1,
        x_dim=ground_truth_mean.shape[1],
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        quadrature_steps=64,
        path_schedule=str(args.path_schedule),
        divergence_estimator="exact",
        theta_embedding=str(args.theta_embedding).replace("-", "_"),
        theta_rbf_num_centers=int(args.theta_rbf_num_centers),
        theta_rbf_lower=float(metadata["theta_low"]),
        theta_rbf_upper=float(metadata["theta_high"]),
        theta_rbf_bandwidth=args.theta_rbf_bandwidth,
    ).to(device)
    model.load_state_dict(
        torch.load(flow_model_path, map_location=device, weights_only=True)
    )
    model.eval()
    with torch.no_grad():
        theta_tensor = torch.from_numpy(
            (theta + theta_input_shift).astype(np.float32)
        ).to(device)
        flow_mean = model.endpoint_mean(theta_tensor).detach().cpu().numpy().astype(np.float64)

    if gkr_mean.shape != ground_truth_mean.shape or flow_mean.shape != ground_truth_mean.shape:
        raise ValueError(
            "Mean shape mismatch: "
            f"ground truth={ground_truth_mean.shape}, GKR={gkr_mean.shape}, flow={flow_mean.shape}."
        )

    output_stem = output_dir / "ground_truth_gkr_flow_mean_heatmaps"
    _save_heatmaps(
        theta=theta[:, 0],
        ground_truth=ground_truth_mean,
        gkr=gkr_mean,
        flow=flow_mean,
        output_stem=output_stem,
    )
    summary = {
        "dataset_npz": str(dataset_npz),
        "gkr_result_npz": str(gkr_result_npz),
        "flow_model": str(flow_model_path),
        "device": str(device),
        "theta_input_shift": theta_input_shift,
        "theta_embedding": str(args.theta_embedding),
        "theta_rbf_num_centers": int(args.theta_rbf_num_centers),
        "theta_rbf_bandwidth": args.theta_rbf_bandwidth,
        "theta_points": int(theta.shape[0]),
        "x_dim": int(ground_truth_mean.shape[1]),
        "gkr_mean_mae": float(np.mean(np.abs(gkr_mean - ground_truth_mean))),
        "gkr_mean_rmse": float(np.sqrt(np.mean((gkr_mean - ground_truth_mean) ** 2))),
        "flow_mean_mae": float(np.mean(np.abs(flow_mean - ground_truth_mean))),
        "flow_mean_rmse": float(np.sqrt(np.mean((flow_mean - ground_truth_mean) ** 2))),
        "flow_mean_rmse_at_theta_nearest_zero": float(
            np.sqrt(
                np.mean(
                    (
                        flow_mean[int(np.argmin(np.abs(theta[:, 0])))]
                        - ground_truth_mean[int(np.argmin(np.abs(theta[:, 0])))]
                    )
                    ** 2
                )
            )
        ),
        "figure_png": str(output_stem.with_suffix(".png")),
        "figure_svg": str(output_stem.with_suffix(".svg")),
    }
    summary_path = output_dir / "ground_truth_gkr_flow_mean_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
