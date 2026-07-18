#!/usr/bin/env python3
"""Evaluate a trained two-trajectory full-Fisher flow checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.flow_matching_skl import (
    build_flow_skl_model,
    estimate_adjacent_model_jeffreys_fisher,
)
from fisher.shared_fisher_est import require_device
from run_two_trajectory_full_fisher import (
    build_dataset,
    ground_truth_full_fisher,
    plot_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--x-dim", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--covariance-alpha", type=float, default=0.65)
    parser.add_argument("--theta-spacing", type=float, default=0.2)
    parser.add_argument("--gt-samples-per-theta", type=int, default=100_000)
    parser.add_argument("--mc-jeffreys-samples", type=int, default=4_096)
    parser.add_argument("--ode-steps", type=int, default=32)
    parser.add_argument(
        "--divergence-estimator", choices=("exact", "hutchinson"), default="hutchinson"
    )
    parser.add_argument("--hutchinson-probes", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = require_device(str(args.device))
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if float(args.theta_spacing) <= 0.0:
        raise ValueError("--theta-spacing must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    dataset = build_dataset(
        x_dim=int(args.x_dim),
        seed=int(args.seed),
        covariance_alpha=float(args.covariance_alpha),
    )
    span = float(dataset.theta_high - dataset.theta_low)
    intervals = int(round(span / float(args.theta_spacing)))
    if intervals < 1 or not np.isclose(span / intervals, float(args.theta_spacing)):
        raise ValueError("--theta-spacing must evenly divide the theta range.")
    theta_grid = np.linspace(dataset.theta_low, dataset.theta_high, intervals + 1)[:, None]
    theta_midpoints = 0.5 * (theta_grid[:-1] + theta_grid[1:])
    truth, truth_se = ground_truth_full_fisher(
        dataset,
        theta_midpoints,
        samples_per_theta=int(args.gt_samples_per_theta),
        seed=int(args.seed) + 100_000,
    )

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    model = build_flow_skl_model(
        velocity_family="nonlinear",
        theta_dim=1,
        x_dim=int(args.x_dim),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        path_schedule="cosine",
        divergence_estimator=str(args.divergence_estimator),
        hutchinson_probes=int(args.hutchinson_probes),
        theta_embedding="gaussian_rbf",
        theta_rbf_num_centers=8,
        theta_rbf_lower=dataset.theta_low,
        theta_rbf_upper=dataset.theta_high,
        theta_rbf_bandwidth=None,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    estimate = estimate_adjacent_model_jeffreys_fisher(
        model=model,
        theta_all=theta_grid,
        device=device,
        mc_jeffreys_sample=int(args.mc_jeffreys_samples),
        ode_steps=int(args.ode_steps),
        ode_method="midpoint",
        batch_size=1_024,
        solve_jitter=1e-6,
        quadrature_steps=64,
    )
    flow = np.asarray(estimate["fisher"], dtype=np.float64)
    figure_png, figure_svg = plot_result(
        theta=theta_midpoints[:, 0], truth=truth, flow=flow, output_dir=output_dir
    )
    results_path = output_dir / "two_trajectory_full_fisher_results.npz"
    np.savez_compressed(
        results_path,
        theta_grid=theta_grid,
        theta_midpoints=theta_midpoints,
        ground_truth_full_fisher=truth,
        ground_truth_standard_error=truth_se,
        flow_full_fisher=flow,
        adjacent_jeffreys=np.asarray(estimate["adjacent_jeffreys"], dtype=np.float64),
    )
    summary = {
        "checkpoint": str(checkpoint),
        "device": str(device),
        "x_dim": int(args.x_dim),
        "seed": int(args.seed),
        "covariance_alpha": float(args.covariance_alpha),
        "theta_spacing": float(args.theta_spacing),
        "ground_truth_samples_per_theta": int(args.gt_samples_per_theta),
        "ground_truth_maximum_standard_error": float(np.max(truth_se)),
        "mc_jeffreys_samples": int(args.mc_jeffreys_samples),
        "ode_steps": int(args.ode_steps),
        "divergence_estimator": str(args.divergence_estimator),
        "hutchinson_probes": (
            int(args.hutchinson_probes)
            if str(args.divergence_estimator) == "hutchinson"
            else None
        ),
        "flow_mae": float(np.mean(np.abs(flow - truth))),
        "runtime_seconds": float(time.perf_counter() - started),
        "results_npz": str(results_path),
        "figure_png": str(figure_png),
        "figure_svg": str(figure_svg),
    }
    summary_path = output_dir / "two_trajectory_full_fisher_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
