#!/usr/bin/env python3
"""Summarize Stringer GKR standardization and learning-rate ablations."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from global_setting import DATA_DIR


@dataclass(frozen=True)
class Ablation:
    key: str
    label: str
    root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--original-root",
        type=Path,
        default=Path(DATA_DIR) / "stringer_pca82_four_methods_all_sessions",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path(DATA_DIR)
        / "stringer_gkr_conventional_kernel_cov300_all_sessions",
    )
    parser.add_argument(
        "--no-standardize-root",
        type=Path,
        default=Path(DATA_DIR)
        / "stringer_gkr_conventional_kernel_no_standardize_cov300_all_sessions",
    )
    parser.add_argument(
        "--lower-lr-root",
        type=Path,
        default=Path(DATA_DIR)
        / "stringer_gkr_conventional_kernel_lr001_cov300_all_sessions",
    )
    parser.add_argument(
        "--lowest-lr-root",
        type=Path,
        default=Path(DATA_DIR)
        / "stringer_gkr_conventional_kernel_lr0001_cov300_all_sessions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "stringer_gkr_training_ablations",
    )
    parser.add_argument("--likelihood-jitter", type=float, default=1e-5)
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as saved:
        return {key: np.asarray(saved[key]) for key in saved.files}


def _session_dirs(root: Path) -> list[Path]:
    result = sorted(path for path in root.glob("session_*") if path.is_dir())
    if len(result) != 6:
        raise ValueError(f"Expected six sessions under {root}, found {len(result)}.")
    return result


def _normalized_mahalanobis(
    observations: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    *,
    jitter: float,
) -> float:
    x = np.asarray(observations, dtype=np.float64)
    mean = np.asarray(means, dtype=np.float64)
    covariance = np.asarray(covariances, dtype=np.float64)
    if x.ndim != 2 or mean.shape != x.shape:
        raise ValueError("observations and means must have shape [n, d].")
    if covariance.shape != (x.shape[0], x.shape[1], x.shape[1]):
        raise ValueError("covariances must have shape [n, d, d].")
    eye = np.eye(x.shape[1], dtype=np.float64)
    symmetric = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
    cholesky = np.linalg.cholesky(symmetric + float(jitter) * eye[None])
    whitened = np.linalg.solve(
        cholesky, (x - mean)[..., None]
    ).squeeze(-1)
    return float(np.mean(np.sum(whitened**2, axis=1)) / x.shape[1])


def _relative_to_baseline(values: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    reference = np.asarray(baseline, dtype=np.float64).reshape(-1)
    if data.ndim != 2 or data.shape[0] != reference.size:
        raise ValueError("values and baseline must contain the same sessions.")
    return data - reference[:, None]


def _load(
    original_root: Path,
    ablations: tuple[Ablation, ...],
    *,
    jitter: float,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    original_dirs = _session_dirs(original_root)
    ablation_dirs = [_session_dirs(ablation.root) for ablation in ablations]
    labels: list[str] = []
    bin_likelihood: list[float] = []
    likelihood_rows: list[list[float]] = []
    mahalanobis_rows: list[list[float]] = []
    sessions: list[dict[str, Any]] = []
    for index, original_dir in enumerate(original_dirs):
        original_summary = json.loads(
            (original_dir / "summary.json").read_text(encoding="utf-8")
        )
        original = _load_npz(original_dir / "four_method_results.npz")
        base_dir = Path(original_summary["signature"]["base_result_dir"])
        pca = _load_npz(base_dir / "pca82_dataset.npz")
        test = np.asarray(original["test_indices"], dtype=np.int64)
        if not np.array_equal(test, np.asarray(pca["test_indices"], dtype=np.int64)):
            raise ValueError("Original and PCA test indices differ.")
        x_test = np.asarray(pca["x"], dtype=np.float64)[test]
        label = str(original_summary["session_label"])
        labels.append(label)
        bin_value = float(np.mean(original["binned_test_log_likelihood"]))
        bin_likelihood.append(bin_value)
        likelihood_row: list[float] = []
        mahalanobis_row: list[float] = []
        session_methods: dict[str, Any] = {}
        for ablation, directories in zip(ablations, ablation_dirs, strict=True):
            directory = directories[index]
            summary = json.loads(
                (directory / "summary.json").read_text(encoding="utf-8")
            )
            if str(summary["protocol"]["session"]) != label:
                raise ValueError(f"Session mismatch under {directory}.")
            result = _load_npz(directory / "conventional_gkr_results.npz")
            likelihood = float(
                np.mean(result["conventional_test_log_likelihood"])
            )
            mahalanobis = _normalized_mahalanobis(
                x_test,
                result["test_mean"],
                result["test_covariance"],
                jitter=float(jitter),
            )
            likelihood_row.append(likelihood)
            mahalanobis_row.append(mahalanobis)
            session_methods[ablation.key] = {
                "mean_test_log_likelihood": likelihood,
                "test_log_likelihood_relative_to_bin_lw": likelihood - bin_value,
                "normalized_mahalanobis": mahalanobis,
                "learned_precision": float(summary["learned_precision"]),
                "minimum_covariance_loss_epoch": int(
                    summary["covariance_loss"]["minimum_epoch"]
                ),
            }
        likelihood_rows.append(likelihood_row)
        mahalanobis_rows.append(mahalanobis_row)
        sessions.append(
            {
                "session_index": index,
                "session_label": label,
                "bin_lw_mean_test_log_likelihood": bin_value,
                "methods": session_methods,
            }
        )
    return (
        labels,
        np.asarray(bin_likelihood, dtype=np.float64),
        np.asarray(likelihood_rows, dtype=np.float64),
        np.asarray(mahalanobis_rows, dtype=np.float64),
        sessions,
    )


def _draw_paired_bars(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    labels: tuple[str, ...],
    colors: tuple[str, ...],
    ylabel: str,
) -> None:
    positions = np.arange(len(labels), dtype=np.float64)
    means = np.mean(values, axis=0)
    sem = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
    axis.bar(
        positions,
        means,
        yerr=sem,
        width=0.68,
        color=colors,
        edgecolor=colors,
        alpha=0.50,
        linewidth=1.8,
        capsize=3.5,
        error_kw={"ecolor": "black", "elinewidth": 1.6, "capthick": 1.6},
        zorder=2,
    )
    for row in values:
        axis.plot(
            positions, row, color="0.45", linewidth=1.2, alpha=0.45, zorder=3
        )
        axis.scatter(
            positions,
            row,
            color="black",
            edgecolor="white",
            linewidth=0.45,
            s=34,
            zorder=4,
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
    labels: tuple[str, ...],
    output_dir: Path,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 15,
            "axes.titlesize": 15,
            "xtick.labelsize": 11,
            "ytick.labelsize": 14,
            "axes.grid": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), constrained_layout=True)
    colors = ("C0", "C2", "C3", "C4")
    _draw_paired_bars(
        axes[0],
        relative_likelihood,
        labels=labels,
        colors=colors,
        ylabel="Test log likelihood\nrelative to Bin+LW",
    )
    axes[0].axhline(0.0, color="0.35", linewidth=1.2, linestyle="--", zorder=1)
    axes[0].set_title("Held-out likelihood")
    _draw_paired_bars(
        axes[1],
        mahalanobis,
        labels=labels,
        colors=colors,
        ylabel=r"Normalized Mahalanobis $q/d$",
    )
    axes[1].axhline(1.0, color="0.35", linewidth=1.2, linestyle="--", zorder=1)
    axes[1].set_title("Covariance calibration")
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "stringer_gkr_training_ablations.png"
    svg = output_dir / "stringer_gkr_training_ablations.svg"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    return png, svg


def main() -> int:
    args = parse_args()
    ablations = (
        Ablation(
            "current",
            "Current",
            args.baseline_root.expanduser().resolve(),
        ),
        Ablation(
            "no_standardization",
            "No\nstandardization",
            args.no_standardize_root.expanduser().resolve(),
        ),
        Ablation(
            "lower_learning_rate",
            "LR 0.01",
            args.lower_lr_root.expanduser().resolve(),
        ),
        Ablation(
            "lowest_learning_rate",
            "LR 0.001",
            args.lowest_lr_root.expanduser().resolve(),
        ),
    )
    output_dir = args.output_dir.expanduser().resolve()
    session_labels, bin_likelihood, likelihood, mahalanobis, sessions = _load(
        args.original_root.expanduser().resolve(),
        ablations,
        jitter=float(args.likelihood_jitter),
    )
    relative = _relative_to_baseline(likelihood, bin_likelihood)
    png, svg = _plot(
        relative,
        mahalanobis,
        labels=tuple(ablation.label for ablation in ablations),
        output_dir=output_dir / "figures",
    )
    summary = {
        "protocol": {
            "sessions": 6,
            "pca_dim": 82,
            "fit_fraction": 0.8,
            "test_fraction": 0.2,
            "covariance_epochs": 300,
            "ablations": {
                "current": {
                    "standardize_responses": True,
                    "mean_learning_rate": 0.05,
                    "covariance_learning_rate": 0.1,
                },
                "no_standardization": {
                    "standardize_responses": False,
                    "mean_learning_rate": 0.05,
                    "covariance_learning_rate": 0.1,
                },
                "lower_learning_rate": {
                    "standardize_responses": True,
                    "mean_learning_rate": 0.01,
                    "covariance_learning_rate": 0.01,
                },
                "lowest_learning_rate": {
                    "standardize_responses": True,
                    "mean_learning_rate": 0.001,
                    "covariance_learning_rate": 0.001,
                },
            },
        },
        "session_labels": session_labels,
        "sessions": sessions,
        "across_session": {
            ablation.key: {
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
            for index, ablation in enumerate(ablations)
        },
        "paired_contrasts": {
            "no_standardization_minus_current_likelihood": {
                "mean": float(np.mean(likelihood[:, 1] - likelihood[:, 0])),
                "sem": float(
                    np.std(likelihood[:, 1] - likelihood[:, 0], ddof=1)
                    / np.sqrt(likelihood.shape[0])
                ),
                "sessions_improved": int(
                    np.count_nonzero(likelihood[:, 1] > likelihood[:, 0])
                ),
            },
            "lower_lr_minus_current_likelihood": {
                "mean": float(np.mean(likelihood[:, 2] - likelihood[:, 0])),
                "sem": float(
                    np.std(likelihood[:, 2] - likelihood[:, 0], ddof=1)
                    / np.sqrt(likelihood.shape[0])
                ),
                "sessions_improved": int(
                    np.count_nonzero(likelihood[:, 2] > likelihood[:, 0])
                ),
            },
            "lowest_lr_minus_current_likelihood": {
                "mean": float(np.mean(likelihood[:, 3] - likelihood[:, 0])),
                "sem": float(
                    np.std(likelihood[:, 3] - likelihood[:, 0], ddof=1)
                    / np.sqrt(likelihood.shape[0])
                ),
                "sessions_improved": int(
                    np.count_nonzero(likelihood[:, 3] > likelihood[:, 0])
                ),
            },
        },
        "artifacts": {"png": png, "svg": svg},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(_json_ready(summary["across_session"]), indent=2), flush=True)
    print(json.dumps(_json_ready(summary["paired_contrasts"]), indent=2), flush=True)
    print(f"Saved: {summary_path}", flush=True)
    print(f"Saved: {png}", flush=True)
    print(f"Saved: {svg}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
