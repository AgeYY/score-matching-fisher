#!/usr/bin/env python3
"""Diagnose Stringer GKR likelihood through mean/covariance hybrid models."""

from __future__ import annotations

import argparse
import json
import sys
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

PERIOD = float(np.pi)
KERNEL_LABELS = (
    "Original\n30 ep.",
    "Periodic\n100 ep.",
    "Periodic\n300 ep.",
    "Bin+LW",
)
HYBRID_LABELS = (
    "Bin\nLW",
    "GKR\nLW",
    "Bin\nGKR",
    "GKR\nGKR",
)


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
        "--gkr-100-root",
        type=Path,
        default=Path(DATA_DIR) / "stringer_gkr_conventional_kernel_all_sessions",
    )
    parser.add_argument(
        "--gkr-300-root",
        type=Path,
        default=Path(DATA_DIR)
        / "stringer_gkr_conventional_kernel_cov300_all_sessions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "stringer_gkr_covariance_diagnosis",
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


def _periodic_bin_indices(theta: np.ndarray, *, n_bins: int) -> np.ndarray:
    wrapped = np.mod(np.asarray(theta, dtype=np.float64).reshape(-1), PERIOD)
    indices = np.floor(wrapped / (PERIOD / int(n_bins))).astype(np.int64)
    return np.clip(indices, 0, int(n_bins) - 1)


def _gaussian_log_likelihood(
    observations: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    *,
    jitter: float,
) -> np.ndarray:
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
    residual = x - mean
    whitened = np.linalg.solve(cholesky, residual[..., None]).squeeze(-1)
    log_determinant = 2.0 * np.log(
        np.diagonal(cholesky, axis1=-2, axis2=-1)
    ).sum(axis=1)
    return -0.5 * (
        x.shape[1] * np.log(2.0 * np.pi)
        + log_determinant
        + np.sum(whitened**2, axis=1)
    )


def _shared_covariance_log_likelihood(
    observations: np.ndarray,
    means: np.ndarray,
    group_indices: np.ndarray,
    group_covariances: np.ndarray,
    *,
    jitter: float,
) -> np.ndarray:
    """Evaluate arbitrary means with one covariance factorization per group."""

    x = np.asarray(observations, dtype=np.float64)
    mean = np.asarray(means, dtype=np.float64)
    groups = np.asarray(group_indices, dtype=np.int64).reshape(-1)
    covariance = np.asarray(group_covariances, dtype=np.float64)
    if x.ndim != 2 or mean.shape != x.shape or groups.shape[0] != x.shape[0]:
        raise ValueError("observations, means, and groups must contain the same rows.")
    if covariance.ndim != 3 or covariance.shape[1:] != (
        x.shape[1],
        x.shape[1],
    ):
        raise ValueError("group_covariances must have shape [groups, d, d].")
    if np.any(groups < 0) or np.any(groups >= covariance.shape[0]):
        raise ValueError("group_indices contains an invalid group.")
    result = np.empty(x.shape[0], dtype=np.float64)
    eye = np.eye(x.shape[1], dtype=np.float64)
    log_norm = x.shape[1] * np.log(2.0 * np.pi)
    for group in np.unique(groups):
        index = np.flatnonzero(groups == group)
        symmetric = 0.5 * (covariance[group] + covariance[group].T)
        cholesky = np.linalg.cholesky(symmetric + float(jitter) * eye)
        residual = x[index] - mean[index]
        whitened = np.linalg.solve(cholesky, residual.T).T
        log_determinant = 2.0 * np.log(np.diag(cholesky)).sum()
        result[index] = -0.5 * (
            log_norm + log_determinant + np.sum(whitened**2, axis=1)
        )
    return result


def _relative_to_baseline(values: np.ndarray, baseline_index: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("values must have shape [sessions, methods].")
    return array - array[:, int(baseline_index) : int(baseline_index) + 1]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {key: np.asarray(saved[key]) for key in saved.files}


def _case_dirs(root: Path) -> list[Path]:
    result = sorted(path for path in root.glob("session_*") if path.is_dir())
    if len(result) != 6:
        raise ValueError(f"Expected six sessions under {root}, found {len(result)}.")
    return result


def _load_session(
    original_dir: Path,
    gkr100_dir: Path,
    gkr300_dir: Path,
    *,
    likelihood_jitter: float,
) -> dict[str, Any]:
    original_summary = json.loads(
        (original_dir / "summary.json").read_text(encoding="utf-8")
    )
    original = _load_npz(original_dir / "four_method_results.npz")
    gkr100 = _load_npz(gkr100_dir / "conventional_gkr_results.npz")
    gkr300 = _load_npz(gkr300_dir / "conventional_gkr_results.npz")
    if int(gkr300["covariance_epochs"]) != 300:
        raise ValueError(f"{gkr300_dir} is not a 300-epoch covariance fit.")
    base_dir = Path(original_summary["signature"]["base_result_dir"])
    pca = _load_npz(base_dir / "pca82_dataset.npz")
    test = np.asarray(original["test_indices"], dtype=np.int64)
    if not np.array_equal(test, np.asarray(pca["test_indices"], dtype=np.int64)):
        raise ValueError("Original four-method and PCA test indices differ.")
    x_test = np.asarray(pca["x"], dtype=np.float64)[test]
    theta_test = np.asarray(pca["theta"], dtype=np.float64)[test]
    if gkr300["test_mean"].shape != x_test.shape:
        raise ValueError("GKR test means do not match the held-out observations.")
    n_bins = int(original["binned_all_mean"].shape[0])
    bins = _periodic_bin_indices(theta_test, n_bins=n_bins)
    bin_mean = np.asarray(original["binned_all_mean"], dtype=np.float64)[bins]
    bin_covariance_all = np.asarray(
        original["binned_all_covariance"], dtype=np.float64
    )
    gkr_mean = np.asarray(gkr300["test_mean"], dtype=np.float64)
    gkr_covariance = np.asarray(gkr300["test_covariance"], dtype=np.float64)
    bin_lw = np.asarray(original["binned_test_log_likelihood"], dtype=np.float64)
    gkr_mean_lw_cov = _shared_covariance_log_likelihood(
        x_test,
        gkr_mean,
        bins,
        bin_covariance_all,
        jitter=float(likelihood_jitter),
    )
    bin_mean_gkr_cov = _gaussian_log_likelihood(
        x_test,
        bin_mean,
        gkr_covariance,
        jitter=float(likelihood_jitter),
    )
    gkr_mean_gkr_cov = _gaussian_log_likelihood(
        x_test,
        gkr_mean,
        gkr_covariance,
        jitter=float(likelihood_jitter),
    )
    np.testing.assert_allclose(
        gkr_mean_gkr_cov,
        np.asarray(gkr300["conventional_test_log_likelihood"]),
        rtol=1e-10,
        atol=1e-10,
    )
    kernel_values = np.asarray(
        [
            np.mean(original["gkr_test_log_likelihood"]),
            np.mean(gkr100["conventional_test_log_likelihood"]),
            np.mean(gkr300["conventional_test_log_likelihood"]),
            np.mean(bin_lw),
        ],
        dtype=np.float64,
    )
    hybrid_values = np.asarray(
        [
            np.mean(bin_lw),
            np.mean(gkr_mean_lw_cov),
            np.mean(bin_mean_gkr_cov),
            np.mean(gkr_mean_gkr_cov),
        ],
        dtype=np.float64,
    )
    return {
        "session_index": int(original_summary["session_index"]),
        "session_label": str(original_summary["session_label"]),
        "kernel_values": kernel_values,
        "hybrid_values": hybrid_values,
        "gkr300_covariance_losses": np.asarray(
            gkr300["covariance_losses"], dtype=np.float64
        ),
        "gkr300_precision": float(gkr300["learned_precision"]),
    }


def _axis_limits(values: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    mean = np.mean(data, axis=0)
    low = float(min(np.min(data), np.min(mean - errors), 0.0))
    high = float(max(np.max(data), np.max(mean + errors), 0.0))
    span = max(high - low, 1.0)
    return low - 0.1 * span, high + 0.1 * span


def _draw_panel(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    labels: tuple[str, ...],
    title: str,
    colors: tuple[str, ...],
) -> None:
    mean = np.mean(values, axis=0)
    sem = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
    positions = np.arange(values.shape[1], dtype=np.float64)
    axis.bar(
        positions,
        mean,
        yerr=sem,
        width=0.68,
        color=colors,
        edgecolor=colors,
        alpha=0.50,
        linewidth=1.8,
        capsize=3.5,
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
            color="0.45",
            linewidth=1.2,
            alpha=0.45,
            zorder=3,
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
    axis.axhline(0.0, color="0.35", linewidth=1.2, linestyle="--", zorder=1)
    axis.set_xticks(positions, labels)
    axis.set_ylim(*_axis_limits(values, sem))
    axis.set_title(title)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.8)
    axis.spines["bottom"].set_linewidth(1.8)
    axis.tick_params(width=1.8)
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color="0.88", linewidth=0.8)
    axis.xaxis.grid(False)


def _plot(
    kernel_relative: np.ndarray,
    hybrid_relative: np.ndarray,
    *,
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
    _draw_panel(
        axes[0],
        kernel_relative,
        labels=KERNEL_LABELS,
        title="Periodic-kernel training",
        colors=("C4", "C2", "C0", "C1"),
    )
    _draw_panel(
        axes[1],
        hybrid_relative,
        labels=HYBRID_LABELS,
        title="Mean / covariance swap",
        colors=("C1", "C0", "C2", "C4"),
    )
    axes[1].set_xlabel("Mean / covariance")
    axes[0].set_ylabel("Test log likelihood\nrelative to Bin+LW")
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "stringer_gkr_covariance_diagnosis.png"
    svg = output_dir / "stringer_gkr_covariance_diagnosis.svg"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    return png, svg


def main() -> int:
    args = parse_args()
    original_root = args.original_root.expanduser().resolve()
    gkr100_root = args.gkr_100_root.expanduser().resolve()
    gkr300_root = args.gkr_300_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    original_dirs = _case_dirs(original_root)
    gkr100_dirs = _case_dirs(gkr100_root)
    gkr300_dirs = _case_dirs(gkr300_root)
    sessions = [
        _load_session(
            original_dir,
            gkr100_dir,
            gkr300_dir,
            likelihood_jitter=float(args.likelihood_jitter),
        )
        for original_dir, gkr100_dir, gkr300_dir in zip(
            original_dirs, gkr100_dirs, gkr300_dirs, strict=True
        )
    ]
    indices = [session["session_index"] for session in sessions]
    if indices != list(range(6)):
        raise ValueError(f"Expected ordered session indices 0 through 5, got {indices}.")
    kernel = np.vstack([session["kernel_values"] for session in sessions])
    hybrid = np.vstack([session["hybrid_values"] for session in sessions])
    kernel_relative = _relative_to_baseline(kernel, baseline_index=3)
    hybrid_relative = _relative_to_baseline(hybrid, baseline_index=0)
    png, svg = _plot(
        kernel_relative,
        hybrid_relative,
        output_dir=output_dir / "figures",
    )
    covariance_replacement_gain = (
        hybrid[:, 1] - hybrid[:, 3]
    )
    mean_replacement_gain = hybrid[:, 2] - hybrid[:, 3]
    gkr300_minus_gkr100 = kernel[:, 2] - kernel[:, 1]
    summary = {
        "protocol": {
            "sessions": 6,
            "pca_dim": 82,
            "fit_fraction": 0.8,
            "test_fraction": 0.2,
            "kernel": "conventional periodic exp(-precision * sin(pi delta / period)^2)",
            "gkr_mean_iterations": 300,
            "gkr_covariance_epochs": 300,
            "baseline": "16-bin sample mean plus Ledoit-Wolf covariance",
            "hybrid_interpretation": (
                "Positive GKR-mean/LW-cov minus GKR-mean/GKR-cov isolates "
                "improvement from replacing GKR covariance. Positive "
                "Bin-mean/GKR-cov minus GKR-mean/GKR-cov isolates improvement "
                "from replacing GKR mean."
            ),
        },
        "sessions": [
            {
                "session_index": session["session_index"],
                "session_label": session["session_label"],
                "kernel_log_likelihood": dict(
                    zip(KERNEL_LABELS, session["kernel_values"], strict=True)
                ),
                "hybrid_log_likelihood": dict(
                    zip(HYBRID_LABELS, session["hybrid_values"], strict=True)
                ),
                "replace_gkr_covariance_with_lw_gain": (
                    session["hybrid_values"][1] - session["hybrid_values"][3]
                ),
                "replace_gkr_mean_with_binned_gain": (
                    session["hybrid_values"][2] - session["hybrid_values"][3]
                ),
                "gkr300_covariance_loss_first": session[
                    "gkr300_covariance_losses"
                ][0],
                "gkr300_covariance_loss_last": session[
                    "gkr300_covariance_losses"
                ][-1],
                "gkr300_covariance_loss_minimum": np.min(
                    session["gkr300_covariance_losses"]
                ),
                "gkr300_covariance_loss_minimum_epoch": int(
                    np.argmin(session["gkr300_covariance_losses"]) + 1
                ),
                "gkr300_precision": session["gkr300_precision"],
            }
            for session in sessions
        ],
        "across_session_mean_relative_to_bin_lw": {
            label: np.mean(kernel_relative, axis=0)[index]
            for index, label in enumerate(KERNEL_LABELS)
        },
        "hybrid_across_session_mean_relative_to_bin_lw": {
            label: np.mean(hybrid_relative, axis=0)[index]
            for index, label in enumerate(HYBRID_LABELS)
        },
        "diagnostic_contrasts": {
            "gkr300_minus_gkr100_mean": np.mean(gkr300_minus_gkr100),
            "gkr300_minus_gkr100_sem": np.std(
                gkr300_minus_gkr100, ddof=1
            )
            / np.sqrt(gkr300_minus_gkr100.size),
            "replace_gkr_covariance_with_lw_gain_mean": np.mean(
                covariance_replacement_gain
            ),
            "replace_gkr_covariance_with_lw_gain_sem": np.std(
                covariance_replacement_gain, ddof=1
            )
            / np.sqrt(covariance_replacement_gain.size),
            "replace_gkr_mean_with_binned_gain_mean": np.mean(
                mean_replacement_gain
            ),
            "replace_gkr_mean_with_binned_gain_sem": np.std(
                mean_replacement_gain, ddof=1
            )
            / np.sqrt(mean_replacement_gain.size),
            "sessions_improved_by_replacing_gkr_covariance": int(
                np.count_nonzero(covariance_replacement_gain > 0.0)
            ),
            "sessions_improved_by_replacing_gkr_mean": int(
                np.count_nonzero(mean_replacement_gain > 0.0)
            ),
        },
        "artifacts": {"png": png, "svg": svg},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(_json_ready(summary), indent=2), flush=True)
    print(f"Saved: {summary_path}", flush=True)
    print(f"Saved: {png}", flush=True)
    print(f"Saved: {svg}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
