#!/usr/bin/env python3
"""Compare identity-time and RBF-time A03T Mahalanobis flow estimates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher.bci_iv_2a_dataset import CLASS_NAMES  # noqa: E402
from fisher.bci_iv_2a_session_identification import (  # noqa: E402
    rdms_from_means_and_precisions,
)


DISPLAY_CLASS_NAMES = {
    "left_hand": "Left hand",
    "right_hand": "Right hand",
    "both_feet": "Both feet",
    "tongue": "Tongue",
}
VISIBLE_CUE_INTERVAL = (0.0, 1.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identity-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/a03t_reference_mahalanobis_half_trials",
    )
    parser.add_argument(
        "--rbf-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/a03t_reference_mahalanobis_rbf_half_trials",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/a03t_reference_mahalanobis_rbf_half_trials",
    )
    return parser.parse_args()


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "axes.grid": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "savefig.bbox": "tight",
        }
    )


def _decorate(axis: plt.Axes) -> None:
    axis.axvspan(*VISIBLE_CUE_INTERVAL, color="0.93", linewidth=0, zorder=0)
    axis.axvline(0.0, color="0.35", linestyle="--", linewidth=1.2, zorder=1)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    axis.tick_params(width=1.8)


def _mean_curve(rdms: np.ndarray) -> np.ndarray:
    upper = np.triu_indices(4, k=1)
    return np.mean(rdms[:, upper[0], upper[1]], axis=1)


def _curve_stats(curve: np.ndarray, classical: np.ndarray, times: np.ndarray) -> dict[str, float | list[float]]:
    cue = (times >= VISIBLE_CUE_INTERVAL[0]) & (times <= VISIBLE_CUE_INTERVAL[1])
    return {
        "minimum": float(np.min(curve)),
        "maximum": float(np.max(curve)),
        "maximum_time_seconds": float(times[int(np.argmax(curve))]),
        "rmse_vs_classical_all_time": float(np.sqrt(np.mean((curve - classical) ** 2))),
        "rmse_vs_classical_visible_cue": float(
            np.sqrt(np.mean((curve[cue] - classical[cue]) ** 2))
        ),
        "correlation_with_classical_all_time": float(np.corrcoef(curve, classical)[0, 1]),
    }


def _plot_rdm_curves(
    output_dir: Path,
    times: np.ndarray,
    classical: np.ndarray,
    identity: np.ndarray,
    rbf: np.ndarray,
) -> None:
    _style()
    figure, axis = plt.subplots(figsize=(4.0, 3.5))
    _decorate(axis)
    axis.plot(times, classical, color="#4477AA", linewidth=1.3, label="Classical")
    axis.plot(times, identity, color="#CC6677", linewidth=1.8, label="Flow: scalar time")
    axis.plot(times, rbf, color="#228833", linewidth=1.8, label="Flow: RBF time")
    axis.set_xlim(float(times[0]), float(times[-1]))
    axis.set_ylim(0.0, 1.04 * float(max(classical.max(), identity.max(), rbf.max())))
    axis.set_xlabel("Time from cue onset (s)")
    axis.set_ylabel("Mean Mahalanobis² distance")
    axis.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)
    figure.savefig(output_dir / "a03t_mahalanobis_time_embedding_rdm_comparison.png", dpi=300)
    figure.savefig(output_dir / "a03t_mahalanobis_time_embedding_rdm_comparison.svg")
    plt.close(figure)


def _plot_mean_norms(
    output_dir: Path,
    times: np.ndarray,
    centers: np.ndarray,
    empirical_binned: np.ndarray,
    identity_native: np.ndarray,
    identity_binned: np.ndarray,
    rbf_native: np.ndarray,
    rbf_binned: np.ndarray,
) -> None:
    _style()
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 7.0), sharex=True, sharey=True)
    empirical_norm = np.linalg.norm(empirical_binned, axis=-1)
    identity_native_norm = np.linalg.norm(identity_native, axis=-1)
    identity_binned_norm = np.linalg.norm(identity_binned, axis=-1)
    rbf_native_norm = np.linalg.norm(rbf_native, axis=-1)
    rbf_binned_norm = np.linalg.norm(rbf_binned, axis=-1)
    for class_index, axis in enumerate(axes.flat):
        _decorate(axis)
        axis.plot(
            times,
            identity_native_norm[:, class_index],
            color="#CC6677",
            linewidth=1.4,
            alpha=0.85,
            label="Flow: scalar time",
        )
        axis.plot(
            times,
            rbf_native_norm[:, class_index],
            color="#228833",
            linewidth=1.6,
            label="Flow: RBF time",
        )
        axis.plot(
            centers,
            empirical_norm[:, class_index],
            color="#4477AA",
            linewidth=1.6,
            marker="o",
            markersize=3.5,
            label="Empirical, 250 ms",
        )
        axis.plot(
            centers,
            identity_binned_norm[:, class_index],
            color="#CC6677",
            linestyle="none",
            marker="o",
            markersize=3.0,
        )
        axis.plot(
            centers,
            rbf_binned_norm[:, class_index],
            color="#228833",
            linestyle="none",
            marker="o",
            markersize=3.0,
        )
        axis.set_title(DISPLAY_CLASS_NAMES[CLASS_NAMES[class_index]])
    for axis in axes[-1]:
        axis.set_xlabel("Time from cue onset (s)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Mean-vector norm")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    figure.tight_layout()
    figure.savefig(output_dir / "a03t_mahalanobis_time_embedding_mean_comparison.png", dpi=300)
    figure.savefig(output_dir / "a03t_mahalanobis_time_embedding_mean_comparison.svg")
    plt.close(figure)


def _plot_loss_curves(
    output_dir: Path,
    identity: np.lib.npyio.NpzFile,
    rbf: np.lib.npyio.NpzFile,
    identity_summary: dict[str, object],
    rbf_summary: dict[str, object],
) -> None:
    """Show the complete and late-stage optimization histories."""
    _style()
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 7.0), sharex="col")
    experiments = (
        ("Scalar EEG time", identity, identity_summary),
        (
            f"{int(rbf_summary['time_rbf_num_centers'])}-center RBF EEG time",
            rbf,
            rbf_summary,
        ),
    )
    colors = {
        "Training": "#4477AA",
        "Fixed validation": "#CC6677",
        "Validation EMA": "#228833",
    }
    for column, (title, result, run_summary) in enumerate(experiments):
        train = np.asarray(result["train_losses"], dtype=np.float64)
        validation = np.asarray(result["validation_losses"], dtype=np.float64)
        validation_ema = np.asarray(
            result["monitored_validation_losses"], dtype=np.float64
        )
        epochs = np.arange(1, train.size + 1)
        best_epoch = int(run_summary["best_epoch"])
        stopped_epoch = int(run_summary["stopped_epoch"])

        for row, axis in enumerate(axes[:, column]):
            axis.plot(
                epochs,
                train,
                color=colors["Training"],
                linewidth=0.9,
                alpha=0.55,
                label="Training",
            )
            axis.plot(
                epochs,
                validation,
                color=colors["Fixed validation"],
                linewidth=0.9,
                alpha=0.48,
                label="Fixed validation",
            )
            axis.plot(
                epochs,
                validation_ema,
                color=colors["Validation EMA"],
                linewidth=1.8,
                label="Validation EMA",
            )
            axis.axvline(best_epoch, color="0.15", linestyle="--", linewidth=1.3)
            axis.axvline(stopped_epoch, color="0.45", linestyle=":", linewidth=1.3)
            axis.text(
                0.97,
                0.94,
                f"best: {best_epoch:,}\nstop: {stopped_epoch:,}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=12,
            )
            axis.set_xlim(1, stopped_epoch)
            axis.grid(False)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_linewidth(1.8)
            axis.spines["bottom"].set_linewidth(1.8)
            axis.tick_params(width=1.8)
            if row == 1:
                plateau_start = min(500, max(1, train.size // 10))
                plateau_values = np.concatenate(
                    (
                        train[plateau_start - 1 :],
                        validation[plateau_start - 1 :],
                        validation_ema[plateau_start - 1 :],
                    )
                )
                lower, upper = np.quantile(plateau_values, [0.002, 0.998])
                padding = 0.08 * (upper - lower)
                axis.set_ylim(lower - padding, upper + padding)
                axis.set_xlim(plateau_start, stopped_epoch)
        axes[0, column].set_title(title)
        axes[1, column].set_xlabel("Epoch")

    axes[0, 0].set_ylabel("Flow-matching loss")
    axes[1, 0].set_ylabel("Loss (plateau zoom)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "a03t_mahalanobis_time_embedding_loss_curves.png", dpi=300
    )
    figure.savefig(output_dir / "a03t_mahalanobis_time_embedding_loss_curves.svg")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_name = "a03t_reference_mahalanobis_rdms.npz"
    identity = np.load(args.identity_dir / result_name)
    rbf = np.load(args.rbf_dir / result_name)
    identity_means = np.load(args.identity_dir / "a03t_mahalanobis_class_mean_comparison.npz")
    rbf_means = np.load(args.rbf_dir / "a03t_mahalanobis_class_mean_comparison.npz")
    identity_run_summary = json.loads((args.identity_dir / "summary.json").read_text())
    rbf_run_summary = json.loads((args.rbf_dir / "summary.json").read_text())
    times = np.asarray(rbf["time_seconds_cue_relative"], dtype=np.float64)
    np.testing.assert_allclose(identity["time_seconds_cue_relative"], times)
    np.testing.assert_allclose(identity["classical_rdms"], rbf["classical_rdms"])
    np.testing.assert_array_equal(identity["reference_trial_indices"], rbf["reference_trial_indices"])
    np.testing.assert_array_equal(identity["train_trial_indices"], rbf["train_trial_indices"])

    classical_curve = _mean_curve(rbf["classical_rdms"])
    identity_curve = _mean_curve(identity["flow_rdms"])
    rbf_curve = _mean_curve(rbf["flow_rdms"])
    _plot_rdm_curves(args.output_dir, times, classical_curve, identity_curve, rbf_curve)
    _plot_mean_norms(
        args.output_dir,
        times,
        rbf_means["bin_centers_seconds"],
        rbf_means["empirical_train_binned_means"],
        identity_means["flow_native_time_means"],
        identity_means["flow_binned_means"],
        rbf_means["flow_native_time_means"],
        rbf_means["flow_binned_means"],
    )
    _plot_loss_curves(
        args.output_dir,
        identity,
        rbf,
        identity_run_summary,
        rbf_run_summary,
    )

    summary: dict[str, object] = {
        "identity_time": _curve_stats(identity_curve, classical_curve, times),
        "rbf_time": _curve_stats(rbf_curve, classical_curve, times),
    }
    for name, result in (("identity_time", identity), ("rbf_time", rbf)):
        eigenvalues = np.linalg.eigvalsh(result["flow_distance_covariances"])
        summary[name]["minimum_covariance_eigenvalue"] = float(eigenvalues.min())  # type: ignore[index]
        summary[name]["maximum_covariance_eigenvalue"] = float(eigenvalues.max())  # type: ignore[index]
        summary[name]["maximum_covariance_condition_number"] = float(  # type: ignore[index]
            np.max(eigenvalues[:, -1] / eigenvalues[:, 0])
        )
        flow_mean_classical_precision = _mean_curve(
            rdms_from_means_and_precisions(
                result["flow_means"], result["classical_precisions"]
            )
        )
        classical_mean_flow_precision = _mean_curve(
            rdms_from_means_and_precisions(
                result["classical_means"], result["flow_precisions"]
            )
        )
        summary[name]["hybrid_flow_mean_classical_precision"] = _curve_stats(  # type: ignore[index]
            flow_mean_classical_precision, classical_curve, times
        )
        summary[name]["hybrid_classical_mean_flow_precision"] = _curve_stats(  # type: ignore[index]
            classical_mean_flow_precision, classical_curve, times
        )
    identity_mean_summary = json.loads(
        (args.identity_dir / "a03t_mahalanobis_class_mean_comparison.json").read_text()
    )
    rbf_mean_summary = json.loads(
        (args.rbf_dir / "a03t_mahalanobis_class_mean_comparison.json").read_text()
    )
    summary["binned_mean_fit"] = {
        "identity_time": {
            key: identity_mean_summary[key]
            for key in (
                "overall_mean_cosine_alignment",
                "overall_median_cosine_alignment",
                "overall_mean_channel_rmse",
            )
        },
        "rbf_time": {
            key: rbf_mean_summary[key]
            for key in (
                "overall_mean_cosine_alignment",
                "overall_median_cosine_alignment",
                "overall_mean_channel_rmse",
            )
        },
    }
    (args.output_dir / "a03t_mahalanobis_time_embedding_comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "a03t_mahalanobis_time_embedding_comparison.npz",
        time_seconds_cue_relative=times,
        classical_mean_distance=classical_curve,
        identity_time_flow_mean_distance=identity_curve,
        rbf_time_flow_mean_distance=rbf_curve,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
