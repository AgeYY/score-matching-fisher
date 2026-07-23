#!/usr/bin/env python3
"""Summarize matched Stringer GKR kernel-parameterization fits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from global_setting import DATA_DIR
from summarize_stringer_gkr_training_ablations import (
    Ablation,
    _json_ready,
    _load,
    _relative_to_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--original-root",
        type=Path,
        default=Path(DATA_DIR) / "stringer_pca82_four_methods_all_sessions",
    )
    parser.add_argument(
        "--log-lambda-root",
        type=Path,
        default=Path(DATA_DIR)
        / "stringer_gkr_log_lambda_lr001_cov100_all_sessions",
    )
    parser.add_argument(
        "--precision-root",
        type=Path,
        default=Path(DATA_DIR)
        / "stringer_gkr_precision_lr001_cov100_all_sessions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR)
        / "stringer_gkr_log_lambda_vs_precision_lr001_cov100",
    )
    parser.add_argument("--likelihood-jitter", type=float, default=1e-5)
    return parser.parse_args()


def _draw_paired(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    labels: tuple[str, str],
    ylabel: str,
    reference: float,
) -> None:
    positions = np.arange(2, dtype=np.float64)
    colors = ("C0", "C1")
    means = np.mean(values, axis=0)
    sem = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
    axis.bar(
        positions,
        means,
        yerr=sem,
        width=0.62,
        color=colors,
        edgecolor=colors,
        alpha=0.48,
        linewidth=1.8,
        capsize=4,
        error_kw={
            "ecolor": "black",
            "elinewidth": 1.6,
            "capthick": 1.6,
        },
        zorder=2,
    )
    for row in values:
        axis.plot(
            positions,
            row,
            color="0.55",
            linewidth=1.2,
            alpha=0.55,
            zorder=3,
        )
        axis.scatter(
            positions,
            row,
            color="black",
            edgecolor="white",
            linewidth=0.45,
            s=38,
            zorder=4,
        )
    axis.axhline(
        reference,
        color="0.35",
        linewidth=1.3,
        linestyle="--",
        zorder=1,
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel(ylabel)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.8)
    axis.spines["bottom"].set_linewidth(1.8)
    axis.tick_params(width=1.8)
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color="0.88", linewidth=0.8)
    axis.xaxis.grid(False)


def _plot(
    relative_likelihood: np.ndarray,
    mahalanobis: np.ndarray,
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
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
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.5, 3.5),
        constrained_layout=True,
    )
    labels = (r"Log-$\lambda$", "Precision")
    _draw_paired(
        axes[0],
        relative_likelihood,
        labels=labels,
        ylabel="Test log likelihood\nrelative to Bin+LW",
        reference=0.0,
    )
    axes[0].set_title("Held-out likelihood")
    _draw_paired(
        axes[1],
        mahalanobis,
        labels=labels,
        ylabel=r"Normalized Mahalanobis $q/d$",
        reference=1.0,
    )
    axes[1].set_title("Covariance calibration")
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "stringer_gkr_kernel_parameterizations.png"
    svg = output_dir / "stringer_gkr_kernel_parameterizations.svg"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    return png, svg


def main() -> int:
    args = parse_args()
    methods = (
        Ablation(
            "log_lambda",
            r"Log-$\lambda$",
            args.log_lambda_root.expanduser().resolve(),
        ),
        Ablation(
            "precision",
            "Precision",
            args.precision_root.expanduser().resolve(),
        ),
    )
    labels, bin_likelihood, likelihood, mahalanobis, sessions = _load(
        args.original_root.expanduser().resolve(),
        methods,
        jitter=float(args.likelihood_jitter),
    )
    relative = _relative_to_baseline(likelihood, bin_likelihood)
    output_dir = args.output_dir.expanduser().resolve()
    png, svg = _plot(
        relative,
        mahalanobis,
        output_dir=output_dir / "figures",
    )
    delta = likelihood[:, 0] - likelihood[:, 1]
    summary = {
        "protocol": {
            "sessions": len(labels),
            "session_labels": labels,
            "pca_dim": 82,
            "fit_fraction": 0.8,
            "test_fraction": 0.2,
            "standardize_responses": True,
            "mean_learning_rate": 0.01,
            "covariance_learning_rate": 0.01,
            "covariance_epochs": 100,
        },
        "sessions": sessions,
        "across_session": {
            method.key: {
                "mean_test_log_likelihood_relative_to_bin_lw": float(
                    np.mean(relative[:, index])
                ),
                "sem_test_log_likelihood_relative_to_bin_lw": float(
                    np.std(relative[:, index], ddof=1)
                    / np.sqrt(relative.shape[0])
                ),
                "mean_normalized_mahalanobis": float(
                    np.mean(mahalanobis[:, index])
                ),
                "sem_normalized_mahalanobis": float(
                    np.std(mahalanobis[:, index], ddof=1)
                    / np.sqrt(mahalanobis.shape[0])
                ),
            }
            for index, method in enumerate(methods)
        },
        "log_lambda_minus_precision_likelihood": {
            "mean": float(np.mean(delta)),
            "sem": float(np.std(delta, ddof=1) / np.sqrt(delta.size)),
            "sessions_improved": int(np.count_nonzero(delta > 0.0)),
        },
        "artifacts": {"png": png, "svg": svg},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_json_ready(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_ready(summary["across_session"]), indent=2))
    print(f"Saved: {summary_path}")
    print(f"Saved: {png}")
    print(f"Saved: {svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
