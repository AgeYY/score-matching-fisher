#!/usr/bin/env python3
"""Diagnose mean and covariance contributions to pre-cue flow RDM distance."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from global_setting import EARLY_STOPPING_PATIENCE, TRAINING_MAX_EPOCHS  # noqa: E402
from fisher.bci_iv_2a_dataset import load_features_npz  # noqa: E402
from fisher.bci_iv_2a_session_identification import (  # noqa: E402
    FlowRDMConfig,
    QUERY_RUNS,
    REFERENCE_RUNS,
    empirical_gaussian_components,
    per_class_counts,
    rdms_from_means_and_precisions,
    select_half,
    subsample_balanced_trials,
    time_varying_shared_affine_flow_rdm_components,
)


DECOMPOSITION_LABELS = {
    "empirical_mean_empirical_covariance": r"Empirical $\mu$ / empirical $\Sigma$",
    "flow_mean_empirical_covariance": r"Flow $\mu$ / empirical $\Sigma$",
    "empirical_mean_flow_covariance": r"Empirical $\mu$ / flow $\Sigma$",
    "flow_mean_flow_covariance": r"Flow $\mu$ / flow $\Sigma$",
}
DECOMPOSITION_COLORS = {
    "empirical_mean_empirical_covariance": "#4477AA",
    "flow_mean_empirical_covariance": "#EE7733",
    "empirical_mean_flow_covariance": "#228833",
    "flow_mean_flow_covariance": "#CC6677",
}
ROLE_LABELS = {"reference": "Reference split", "query": "All-trial query split"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-file",
        type=Path,
        default=ROOT / "data/bci_iv_2a/processed/native_voltage_all_timepoints_20uv/A01T.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/flow_pre_cue_component_diagnostic_A01T",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--epochs", type=int, default=TRAINING_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--standardization",
        choices=("none", "per_half"),
        default="none",
    )
    return parser.parse_args()


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(width=1.8)


def _mean_pairwise_rdm(rdms: np.ndarray) -> np.ndarray:
    upper = np.triu_indices(4, k=1)
    return np.mean(rdms[:, upper[0], upper[1]], axis=1)


def _mean_pairwise_squared_euclidean(means: np.ndarray) -> np.ndarray:
    values = np.asarray(means, dtype=np.float64)
    distances = []
    for left in range(4):
        for right in range(left + 1, 4):
            delta = values[:, left] - values[:, right]
            distances.append(np.sum(delta * delta, axis=1))
    return np.mean(np.stack(distances, axis=1), axis=1)


def _plot_distance_decomposition(
    output_dir: Path,
    times: np.ndarray,
    role_results: dict[str, dict],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 11,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), sharex=True, sharey=False)
    for role, axis in zip(("reference", "query"), axes, strict=True):
        for name in DECOMPOSITION_LABELS:
            axis.plot(
                times,
                role_results[role]["mean_rdm_trajectories"][name],
                color=DECOMPOSITION_COLORS[name],
                linewidth=1.8,
                label=DECOMPOSITION_LABELS[name],
            )
        axis.axvline(0.0, color="0.35", linestyle=":", linewidth=1.5)
        axis.set_title(ROLE_LABELS[role])
        axis.set_xlabel("Time from cue onset (s)")
        axis.set_xlim(float(times[0]), float(times[-1]))
        axis.set_ylim(bottom=0.0)
        _style_axis(axis)
    axes[0].set_ylabel("Mean squared distance")
    axes[0].legend(frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(output_dir / "rdm_distance_decomposition.png", dpi=300)
    figure.savefig(output_dir / "rdm_distance_decomposition.svg")
    plt.close(figure)


def _plot_component_diagnostics(
    output_dir: Path,
    times: np.ndarray,
    role_results: dict[str, dict],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 12,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    figure, axes = plt.subplots(3, 2, figsize=(8.0, 10.5), sharex=True)
    for column, role in enumerate(("reference", "query")):
        result = role_results[role]
        axes[0, column].plot(
            times,
            result["empirical_mean_separation"],
            color="#4477AA",
            linewidth=1.6,
            label=r"Empirical $\mu$",
        )
        axes[0, column].plot(
            times,
            result["flow_mean_separation"],
            color="#CC6677",
            linewidth=1.8,
            label=r"Flow $\mu$",
        )
        axes[0, column].set_title(ROLE_LABELS[role])
        axes[0, column].set_ylabel(r"Mean $\|\Delta\mu\|_2^2$")

        axes[1, column].plot(
            times,
            result["empirical_covariance_eigenvalues"][:, 0],
            color="#4477AA",
            linewidth=1.6,
            label=r"Empirical $\Sigma$",
        )
        axes[1, column].plot(
            times,
            result["flow_covariance_eigenvalues"][:, 0],
            color="#CC6677",
            linewidth=1.8,
            label=r"Flow $\Sigma$",
        )
        axes[1, column].set_yscale("log")
        axes[1, column].set_ylabel("Minimum eigenvalue")

        axes[2, column].plot(
            times,
            result["empirical_covariance_condition_number"],
            color="#4477AA",
            linewidth=1.6,
            label=r"Empirical $\Sigma$",
        )
        axes[2, column].plot(
            times,
            result["flow_covariance_condition_number"],
            color="#CC6677",
            linewidth=1.8,
            label=r"Flow $\Sigma$",
        )
        axes[2, column].set_yscale("log")
        axes[2, column].set_ylabel("Condition number")
        axes[2, column].set_xlabel("Time from cue onset (s)")

        for row in range(3):
            axes[row, column].axvline(
                0.0, color="0.35", linestyle=":", linewidth=1.5
            )
            axes[row, column].set_xlim(float(times[0]), float(times[-1]))
            _style_axis(axes[row, column])
        if column == 0:
            for row in range(3):
                axes[row, column].legend(frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(output_dir / "flow_component_diagnostics.png", dpi=300)
    figure.savefig(output_dir / "flow_component_diagnostics.svg")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires CUDA; no CPU fallback is permitted.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device.index} is unavailable.")
    torch.cuda.set_device(0 if device.index is None else device.index)

    dataset = load_features_npz(args.feature_file)
    if dataset.session_key != "A01T":
        raise ValueError(f"This diagnostic expects A01T; got {dataset.session_key}.")
    times = np.asarray(dataset.time_centers, dtype=np.float64)
    config = FlowRDMConfig(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        standardize_features=args.standardization == "per_half",
    )
    reference_x, reference_y, _ = select_half(dataset, REFERENCE_RUNS)
    query_x, query_y, _ = select_half(dataset, QUERY_RUNS)
    query_n_per_class = int(np.min(per_class_counts(query_y)))
    query_selected = subsample_balanced_trials(
        query_y,
        query_n_per_class,
        seed=args.seed,
    )
    split_data = {
        "reference": (reference_x, reference_y, args.seed + 1_000_002),
        "query": (
            query_x[query_selected],
            query_y[query_selected],
            args.seed + 20_000_000,
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    role_results: dict[str, dict] = {}
    summary: dict = {
        "experiment": "A01T pre-cue flow RDM component diagnostic",
        "feature_file": str(args.feature_file.resolve()),
        "device": args.device,
        "config": asdict(config),
        "time_interval_seconds_cue_relative": [float(times[0]), float(times[-1])],
        "roles": {},
    }
    csv_rows: list[dict] = []
    upper = np.triu_indices(4, k=1)
    interval_masks = {
        "pre_cue": times < 0.0,
        "post_cue": (times >= 0.0) & (times <= 3.5),
    }

    for role, (x, labels, flow_seed) in split_data.items():
        print(
            f"[diagnostic] role={role} trials={x.shape[0]} seed={flow_seed} start",
            flush=True,
        )
        flow_rdms, fit_metadata, flow_components = (
            time_varying_shared_affine_flow_rdm_components(
                x,
                labels,
                times,
                device=device,
                seed=flow_seed,
                config=config,
            )
        )
        empirical_means, empirical_covariances, empirical_precisions = (
            empirical_gaussian_components(
                x,
                labels,
                standardize_features=config.standardize_features,
            )
        )
        flow_means = flow_components["flow_means"]
        flow_covariances = flow_components["flow_distance_covariances"]
        flow_precisions = flow_components["flow_precisions"]
        decomposed_rdms = {
            "empirical_mean_empirical_covariance": rdms_from_means_and_precisions(
                empirical_means, empirical_precisions
            ),
            "flow_mean_empirical_covariance": rdms_from_means_and_precisions(
                flow_means, empirical_precisions
            ),
            "empirical_mean_flow_covariance": rdms_from_means_and_precisions(
                empirical_means, flow_precisions
            ),
            "flow_mean_flow_covariance": flow_rdms,
        }
        empirical_eigenvalues = np.linalg.eigvalsh(empirical_covariances)
        flow_eigenvalues = np.linalg.eigvalsh(flow_covariances)
        empirical_condition = empirical_eigenvalues[:, -1] / empirical_eigenvalues[:, 0]
        flow_condition = flow_eigenvalues[:, -1] / flow_eigenvalues[:, 0]
        mean_rdm_trajectories = {
            name: _mean_pairwise_rdm(rdms) for name, rdms in decomposed_rdms.items()
        }
        result = {
            "mean_rdm_trajectories": mean_rdm_trajectories,
            "empirical_mean_separation": _mean_pairwise_squared_euclidean(
                empirical_means
            ),
            "flow_mean_separation": _mean_pairwise_squared_euclidean(flow_means),
            "empirical_covariance_eigenvalues": empirical_eigenvalues,
            "flow_covariance_eigenvalues": flow_eigenvalues,
            "empirical_covariance_condition_number": empirical_condition,
            "flow_covariance_condition_number": flow_condition,
        }
        role_results[role] = result

        np.savez_compressed(
            args.output_dir / f"{role}_components.npz",
            time_seconds_cue_relative=times,
            labels=labels,
            empirical_means=empirical_means,
            empirical_covariances=empirical_covariances,
            empirical_precisions=empirical_precisions,
            flow_means=flow_means,
            flow_endpoint_covariances=flow_components["flow_endpoint_covariances"],
            flow_distance_covariances=flow_covariances,
            flow_precisions=flow_precisions,
            empirical_covariance_eigenvalues=empirical_eigenvalues,
            flow_covariance_eigenvalues=flow_eigenvalues,
            empirical_covariance_condition_number=empirical_condition,
            flow_covariance_condition_number=flow_condition,
            **{f"rdms_{name}": rdms for name, rdms in decomposed_rdms.items()},
            fit_metadata_json=np.asarray([json.dumps(fit_metadata, sort_keys=True)]),
        )

        interval_means = {
            name: {
                interval_name: float(np.mean(trajectory[mask]))
                for interval_name, mask in interval_masks.items()
            }
            for name, trajectory in mean_rdm_trajectories.items()
        }
        summary["roles"][role] = {
            "n_trials": int(x.shape[0]),
            "per_class_counts": per_class_counts(labels).astype(int).tolist(),
            "flow_seed": int(flow_seed),
            "best_epoch": int(fit_metadata["best_epoch"]),
            "stopped_epoch": int(fit_metadata["stopped_epoch"]),
            "best_val_loss": float(fit_metadata["best_val_loss"]),
            "interval_mean_distances": interval_means,
            "pre_cue_mean_separation": {
                "empirical": float(np.mean(result["empirical_mean_separation"][times < 0])),
                "flow": float(np.mean(result["flow_mean_separation"][times < 0])),
            },
            "pre_cue_covariance": {
                "empirical_median_minimum_eigenvalue": float(
                    np.median(empirical_eigenvalues[times < 0, 0])
                ),
                "flow_median_minimum_eigenvalue": float(
                    np.median(flow_eigenvalues[times < 0, 0])
                ),
                "empirical_median_condition_number": float(
                    np.median(empirical_condition[times < 0])
                ),
                "flow_median_condition_number": float(
                    np.median(flow_condition[times < 0])
                ),
            },
        }
        for name, trajectory in mean_rdm_trajectories.items():
            for time, distance in zip(times, trajectory, strict=True):
                csv_rows.append(
                    {
                        "role": role,
                        "decomposition": name,
                        "time_seconds_cue_relative": float(time),
                        "mean_pairwise_squared_distance": float(distance),
                    }
                )
        print(
            f"[diagnostic] role={role} best_epoch={fit_metadata['best_epoch']} "
            f"stopped_epoch={fit_metadata['stopped_epoch']} complete",
            flush=True,
        )

    with (args.output_dir / "distance_trajectories.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_distance_decomposition(args.output_dir, times, role_results)
    _plot_component_diagnostics(args.output_dir, times, role_results)
    print(f"[diagnostic] output={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
