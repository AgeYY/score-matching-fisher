#!/usr/bin/env python3
"""Compare four conditional density estimators on one held-out Stringer session."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from sklearn.covariance import LedoitWolf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.flow_matching_skl import (
    build_flow_skl_model,
    flow_endpoint_log_prob,
    sample_flow_endpoint_conditions,
    train_flow_skl_model,
)
from fisher.shared_fisher_est import require_device
from fisher.stringer_session_identification import encode_flow_orientation
from global_setting import (
    DATA_DIR,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_TRAINING_MAX_EPOCHS,
)

PERIOD = float(np.pi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--base-result-dir",
        type=Path,
        default=Path(DATA_DIR)
        / "stringer_pca82_flow_gkr_all_sessions"
        / "session_00_GT1",
        help="Existing single-session PCA82 affine-Flow/GKR result directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "stringer_pca82_four_methods_session0",
    )
    parser.add_argument("--bin-width", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_MAX_EPOCHS)
    parser.add_argument(
        "--early-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--fixed-validation-paths", type=int, default=10)
    parser.add_argument("--hutchinson-probes", type=int, default=4)
    parser.add_argument("--ode-steps", type=int, default=32)
    parser.add_argument("--likelihood-batch-size", type=int, default=128)
    parser.add_argument("--generated-samples", type=int, default=1000)
    parser.add_argument(
        "--display-generated-samples",
        type=int,
        default=600,
        help="Maximum generated points shown; does not affect fitting or likelihood.",
    )
    parser.add_argument("--likelihood-jitter", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true")
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


def _periodic_bin_spec(requested_width: float, *, period: float = PERIOD) -> tuple[int, float]:
    if not math.isfinite(float(requested_width)) or float(requested_width) <= 0.0:
        raise ValueError("bin width must be finite and positive.")
    n_bins = max(1, int(round(float(period) / float(requested_width))))
    return n_bins, float(period) / float(n_bins)


def _periodic_bin_indices(
    theta: np.ndarray,
    *,
    n_bins: int,
    period: float = PERIOD,
) -> np.ndarray:
    if int(n_bins) < 1:
        raise ValueError("n_bins must be positive.")
    wrapped = np.mod(np.asarray(theta, dtype=np.float64).reshape(-1), float(period))
    indices = np.floor(wrapped / (float(period) / float(n_bins))).astype(np.int64)
    return np.clip(indices, 0, int(n_bins) - 1)


def _fit_binned_ledoit_wolf(
    x: np.ndarray,
    theta: np.ndarray,
    *,
    n_bins: int,
    period: float = PERIOD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations = np.asarray(x, dtype=np.float64)
    if observations.ndim != 2:
        raise ValueError("x must have shape [n, d].")
    bins = _periodic_bin_indices(theta, n_bins=int(n_bins), period=float(period))
    if bins.size != observations.shape[0]:
        raise ValueError("x and theta must contain the same number of observations.")
    means = np.empty((int(n_bins), observations.shape[1]), dtype=np.float64)
    covariances = np.empty(
        (int(n_bins), observations.shape[1], observations.shape[1]), dtype=np.float64
    )
    counts = np.bincount(bins, minlength=int(n_bins)).astype(np.int64)
    for bin_index in range(int(n_bins)):
        xb = observations[bins == bin_index]
        if xb.shape[0] < 2:
            raise ValueError(f"bin {bin_index} has fewer than two training observations.")
        estimator = LedoitWolf(assume_centered=False).fit(xb)
        means[bin_index] = estimator.location_
        covariances[bin_index] = estimator.covariance_
    return means, covariances, counts


def _gaussian_log_likelihood(
    x: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    jitter: float,
) -> np.ndarray:
    observations = np.asarray(x, dtype=np.float64)
    means = np.asarray(mean, dtype=np.float64)
    covariances = np.asarray(covariance, dtype=np.float64)
    if observations.ndim != 2 or means.shape != observations.shape:
        raise ValueError("x and mean must have the same shape [n, d].")
    expected_covariance_shape = (
        observations.shape[0],
        observations.shape[1],
        observations.shape[1],
    )
    if covariances.shape != expected_covariance_shape:
        raise ValueError(f"covariance must have shape {expected_covariance_shape}.")
    eye = np.eye(observations.shape[1], dtype=np.float64)
    symmetric = 0.5 * (covariances + np.swapaxes(covariances, -1, -2))
    cholesky = np.linalg.cholesky(symmetric + float(jitter) * eye[None, :, :])
    residual = observations - means
    whitened = np.linalg.solve(cholesky, residual[..., None]).squeeze(-1)
    log_determinant = 2.0 * np.log(
        np.diagonal(cholesky, axis1=-2, axis2=-1)
    ).sum(axis=1)
    return -0.5 * (
        observations.shape[1] * np.log(2.0 * np.pi)
        + log_determinant
        + np.sum(whitened**2, axis=1)
    )


def _grouped_gaussian_log_likelihood(
    x: np.ndarray,
    group_indices: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    *,
    jitter: float,
) -> np.ndarray:
    """Evaluate shared Gaussian moments with one factorization per group."""

    observations = np.asarray(x, dtype=np.float64)
    groups = np.asarray(group_indices, dtype=np.int64).reshape(-1)
    group_means = np.asarray(means, dtype=np.float64)
    group_covariances = np.asarray(covariances, dtype=np.float64)
    if observations.ndim != 2 or groups.shape[0] != observations.shape[0]:
        raise ValueError("x and group_indices must contain the same rows.")
    if group_means.ndim != 2 or group_means.shape[1] != observations.shape[1]:
        raise ValueError("means must have shape [n_groups, response_dim].")
    expected = (
        group_means.shape[0],
        observations.shape[1],
        observations.shape[1],
    )
    if group_covariances.shape != expected:
        raise ValueError(f"covariances must have shape {expected}.")
    if np.any(groups < 0) or np.any(groups >= group_means.shape[0]):
        raise ValueError("group_indices contains an invalid group.")
    result = np.empty(observations.shape[0], dtype=np.float64)
    eye = np.eye(observations.shape[1], dtype=np.float64)
    log_norm = observations.shape[1] * np.log(2.0 * np.pi)
    for group in np.unique(groups):
        index = np.flatnonzero(groups == group)
        covariance = 0.5 * (
            group_covariances[group] + group_covariances[group].T
        )
        cholesky = np.linalg.cholesky(covariance + float(jitter) * eye)
        residual = observations[index] - group_means[group]
        whitened = np.linalg.solve(cholesky, residual.T).T
        log_determinant = 2.0 * np.log(np.diag(cholesky)).sum()
        result[index] = -0.5 * (
            log_norm + log_determinant + np.sum(whitened**2, axis=1)
        )
    return result


def _flow_log_likelihood_batched(
    *,
    model: torch.nn.Module,
    x: np.ndarray,
    condition: np.ndarray,
    device: torch.device,
    batch_size: int,
    ode_steps: int,
) -> np.ndarray:
    observations = np.asarray(x, dtype=np.float64)
    conditions = np.asarray(condition, dtype=np.float64)
    if observations.shape[0] != conditions.shape[0]:
        raise ValueError("x and condition must contain the same number of rows.")
    dtype = next(model.parameters()).dtype
    values: list[np.ndarray] = []
    model.eval()
    for start in range(0, observations.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), observations.shape[0])
        xb = torch.as_tensor(observations[start:stop], dtype=dtype, device=device)
        cb = torch.as_tensor(conditions[start:stop], dtype=dtype, device=device)
        log_prob = flow_endpoint_log_prob(
            model,
            xb,
            cb,
            ode_steps=int(ode_steps),
            ode_method="midpoint",
            enable_grad=False,
        )
        values.append(log_prob.detach().cpu().numpy().astype(np.float64, copy=False))
        print(f"[nonlinear likelihood] {stop}/{observations.shape[0]}", flush=True)
    return np.concatenate(values)


def _sample_flow_batched(
    *,
    model: torch.nn.Module,
    condition: np.ndarray,
    device: torch.device,
    batch_size: int,
    ode_steps: int,
) -> np.ndarray:
    samples: list[np.ndarray] = []
    model.eval()
    for start in range(0, condition.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), condition.shape[0])
        endpoint = sample_flow_endpoint_conditions(
            model=model,
            theta_all=condition[start:stop],
            device=device,
            ode_steps=int(ode_steps),
            ode_method="midpoint",
        )
        samples.append(endpoint.detach().cpu().numpy().astype(np.float64, copy=False))
    return np.vstack(samples)


def _ellipse_parameters(covariance: np.ndarray) -> tuple[float, float, float]:
    cov = np.asarray(covariance, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (cov + cov.T))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    principal = eigenvectors[:, order[0]]
    width, height = 2.0 * np.sqrt(eigenvalues)
    angle = float(np.degrees(np.arctan2(principal[1], principal[0])))
    return float(width), float(height), angle


def _axis_limits(
    test_pc12: np.ndarray,
    generated_pc12: np.ndarray,
    moment_sets: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    points = [np.asarray(test_pc12), np.asarray(generated_pc12)]
    for means, covariances in moment_sets:
        means_2d = np.asarray(means)[:, :2]
        covariance_2d = np.asarray(covariances)[:, :2, :2]
        radius = np.sqrt(
            np.maximum(np.diagonal(covariance_2d, axis1=1, axis2=2), 0.0)
        )
        points.extend([means_2d - radius, means_2d + radius])
    joined = np.vstack(points)
    low = np.quantile(joined, 0.002, axis=0)
    high = np.quantile(joined, 0.998, axis=0)
    padding = 0.05 * np.maximum(high - low, 1e-9)
    return (float(low[0] - padding[0]), float(high[0] + padding[0])), (
        float(low[1] - padding[1]),
        float(high[1] + padding[1]),
    )


def _plot(
    *,
    test_pc12: np.ndarray,
    theta_test: np.ndarray,
    selected_theta: np.ndarray,
    affine_mean: np.ndarray,
    affine_covariance: np.ndarray,
    gkr_mean: np.ndarray,
    gkr_covariance: np.ndarray,
    binned_mean: np.ndarray,
    binned_covariance: np.ndarray,
    generated_pc12: np.ndarray,
    generated_theta: np.ndarray,
    likelihoods: dict[str, np.ndarray],
    display_generated_samples: int = 600,
    likelihood_session_values: np.ndarray | None = None,
    session_labels: list[str] | None = None,
    gkr_title: str = "GKR",
    binned_title: str = r"Binning + LW ($h\approx0.2$)",
    moment_order: tuple[str, str, str] = ("gkr", "binned", "affine"),
    likelihood_short_labels: list[str] | None = None,
    likelihood_reference_label: str | None = None,
    output_dir: Path,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "savefig.bbox": "tight",
        }
    )
    cmap = matplotlib.colormaps["twilight_shifted"]
    norm = Normalize(vmin=0.0, vmax=PERIOD)
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(17.0, 3.5),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.0, 0.83]},
        constrained_layout=True,
    )
    moment_axes = [axes[0], axes[1], axes[2]]
    nonlinear_axis = axes[3]
    likelihood_axis = axes[4]
    for axis in moment_axes:
        axis.scatter(
            test_pc12[:, 0],
            test_pc12[:, 1],
            c=np.mod(theta_test, PERIOD),
            cmap=cmap,
            norm=norm,
            marker="+",
            s=24,
            alpha=0.50,
            linewidths=0.7,
            rasterized=True,
            zorder=1,
        )
    n_generated = int(np.asarray(generated_pc12).shape[0])
    n_display = min(int(display_generated_samples), n_generated)
    if n_display < 1:
        raise ValueError("display_generated_samples must be positive.")
    display_indices = np.linspace(0, n_generated - 1, n_display, dtype=np.int64)
    generated_display = np.asarray(generated_pc12)[display_indices]
    theta_display = np.asarray(generated_theta)[display_indices]
    nonlinear_axis.scatter(
        generated_display[:, 0],
        generated_display[:, 1],
        c=np.mod(theta_display, PERIOD),
        cmap=cmap,
        norm=norm,
        s=18,
        alpha=0.55,
        linewidths=0.0,
        rasterized=True,
        zorder=2,
    )
    moment_data = {
        "gkr": (gkr_mean, gkr_covariance, str(gkr_title)),
        "binned": (binned_mean, binned_covariance, str(binned_title)),
        "affine": (affine_mean, affine_covariance, "Affine Flow"),
    }
    if len(set(moment_order)) != 3 or set(moment_order) != set(moment_data):
        raise ValueError(
            "moment_order must contain gkr, binned, and affine exactly once."
        )
    for axis, method in zip(moment_axes, moment_order, strict=True):
        axis.set_title(moment_data[method][2])
    axes[3].set_title("Unconstrained Flow")

    for axis, method in zip(moment_axes, moment_order, strict=True):
        means, covariances, _ = moment_data[method]
        for theta_value, mean, covariance in zip(
            selected_theta, means, covariances, strict=True
        ):
            color = cmap(norm(float(theta_value)))
            width, height, angle = _ellipse_parameters(covariance[:2, :2])
            axis.add_patch(
                Ellipse(
                    xy=mean[:2],
                    width=width,
                    height=height,
                    angle=angle,
                    facecolor="none",
                    edgecolor=color,
                    linewidth=2.2,
                    alpha=0.95,
                    zorder=3,
                )
            )
            axis.scatter(
                mean[0],
                mean[1],
                s=42,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                zorder=4,
            )

    xlim, ylim = _axis_limits(
        test_pc12,
        generated_display,
        [
            (affine_mean, affine_covariance),
            (gkr_mean, gkr_covariance),
            (binned_mean, binned_covariance),
        ],
    )
    method_axes = [*moment_axes, nonlinear_axis]
    for axis in method_axes:
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_aspect("equal", adjustable="box")
        axis.set_axis_off()

    colorbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=method_axes,
        location="right",
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label(r"Orientation $\theta$")
    colorbar.set_ticks(
        [0.0, 0.5 * PERIOD, PERIOD],
        labels=[r"$0$", r"$\pi/2$", r"$\pi$"],
    )
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="+",
                color="black",
                linestyle="none",
                markersize=9,
                markeredgewidth=1.5,
                label="Held-out data",
            ),
            Line2D(
                [],
                [],
                marker="o",
                color="black",
                linestyle="none",
                markersize=6,
                label="Generated data",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.39, -0.03),
        ncol=2,
        frameon=False,
        fontsize=14,
        handletextpad=0.5,
        columnspacing=1.2,
    )

    labels = list(likelihoods)
    arrays = [np.asarray(likelihoods[label], dtype=np.float64) for label in labels]
    if likelihood_session_values is None:
        summary_values = np.asarray([[np.mean(values) for values in arrays]])
        means = summary_values[0]
        errors = np.asarray(
            [np.std(values, ddof=1) / np.sqrt(values.size) for values in arrays]
        )
    else:
        summary_values = np.asarray(likelihood_session_values, dtype=np.float64)
        if summary_values.ndim != 2 or summary_values.shape[1] != len(labels):
            raise ValueError("likelihood_session_values must have shape [sessions, methods].")
        if summary_values.shape[0] < 2:
            raise ValueError("Multi-session likelihood summaries require at least two sessions.")
        if likelihood_reference_label is not None:
            summary_values = _relative_likelihood_values(
                summary_values,
                labels,
                reference_label=likelihood_reference_label,
            )
        means = np.mean(summary_values, axis=0)
        errors = np.std(summary_values, axis=0, ddof=1) / np.sqrt(summary_values.shape[0])
    y_min = float(np.min(np.minimum(means - errors, np.min(summary_values, axis=0))))
    y_max = float(np.max(np.maximum(means + errors, np.max(summary_values, axis=0))))
    span = max(y_max - y_min, 1.0)
    y_low, y_high = y_min - 0.15 * span, y_max + 0.15 * span
    positions = np.arange(len(labels))
    colors = ["C2", "C1", "C0", "C3"]
    bar_values = means if likelihood_reference_label is not None else means - y_low
    bar_bottom = 0.0 if likelihood_reference_label is not None else y_low
    likelihood_axis.bar(
        positions,
        bar_values,
        bottom=bar_bottom,
        width=0.68,
        color=colors,
        alpha=0.50,
        edgecolor=colors,
        linewidth=1.8,
    )
    if summary_values.shape[0] > 1:
        for row_index, row in enumerate(summary_values):
            label = None if session_labels is None else str(session_labels[row_index])
            likelihood_axis.plot(
                positions,
                row,
                color="0.45",
                linewidth=1.2,
                alpha=0.45,
                zorder=2,
            )
            likelihood_axis.scatter(
                positions,
                row,
                color="black",
                edgecolor="white",
                linewidth=0.45,
                s=34,
                label=label,
                zorder=3,
            )
    short_labels = (
        ["GKR", "Bin+LW", "Affine", "Nonlinear"]
        if likelihood_short_labels is None
        else list(likelihood_short_labels)
    )
    if len(labels) != len(short_labels):
        raise ValueError("The likelihood panel expects exactly four methods.")
    likelihood_axis.set_xticks(positions, short_labels, rotation=0, ha="center")
    likelihood_axis.tick_params(axis="x", labelsize=13)
    if likelihood_reference_label is None:
        likelihood_axis.set_ylabel(
            "Mean test log likelihood\n" r"(higher is better $\uparrow$)"
        )
    else:
        reference_short = str(likelihood_reference_label).replace("\n", " ")
        likelihood_axis.set_ylabel(
            f"Test log likelihood relative\nto {reference_short}"
        )
        likelihood_axis.axhline(
            0.0,
            color="0.35",
            linewidth=1.2,
            linestyle="--",
            zorder=1,
        )
    likelihood_axis.set_title("")
    likelihood_axis.set_ylim(y_low, y_high)
    likelihood_axis.set_axisbelow(True)
    likelihood_axis.yaxis.grid(True, color="0.90", linewidth=0.8)
    likelihood_axis.xaxis.grid(False)
    likelihood_axis.spines[["top", "right"]].set_visible(False)
    likelihood_axis.spines["left"].set_linewidth(1.8)
    likelihood_axis.spines["bottom"].set_linewidth(1.8)
    likelihood_axis.tick_params(width=1.8)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "stringer_pca82_test_four_methods"
    png, svg = stem.with_suffix(".png"), stem.with_suffix(".svg")
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _standard_error(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.std(array, ddof=1) / np.sqrt(array.size))


def _relative_likelihood_values(
    values: np.ndarray,
    labels: list[str],
    *,
    reference_label: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(labels):
        raise ValueError("values must have shape [sessions, methods].")
    try:
        reference_index = labels.index(str(reference_label))
    except ValueError as exc:
        raise ValueError(
            f"Likelihood reference {reference_label!r} is not among {labels}."
        ) from exc
    return array - array[:, reference_index : reference_index + 1]


def _validate_split_protocol(
    *,
    outer_train: np.ndarray,
    flow_train: np.ndarray,
    flow_validation: np.ndarray,
    test: np.ndarray,
) -> None:
    outer = np.asarray(outer_train, dtype=np.int64)
    flow_fit = np.asarray(flow_train, dtype=np.int64)
    flow_val = np.asarray(flow_validation, dtype=np.int64)
    held_out = np.asarray(test, dtype=np.int64)
    if np.intersect1d(outer, held_out).size != 0:
        raise ValueError("Outer training and held-out test indices overlap.")
    if np.intersect1d(flow_fit, held_out).size != 0:
        raise ValueError("Flow training and held-out test indices overlap.")
    if np.intersect1d(flow_val, held_out).size != 0:
        raise ValueError("Flow validation and held-out test indices overlap.")
    if not np.array_equal(
        np.sort(np.concatenate([flow_fit, flow_val])), np.sort(outer)
    ):
        raise ValueError("Flow train/validation indices do not partition outer training.")


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.epochs) < 1 or int(args.early_patience) < 0:
        raise ValueError("epochs must be positive and early patience nonnegative.")
    if int(args.batch_size) < 1 or int(args.likelihood_batch_size) < 1:
        raise ValueError("batch sizes must be positive.")
    if int(args.generated_samples) < 1 or int(args.ode_steps) < 1:
        raise ValueError("generated samples and ODE steps must be positive.")
    if int(args.display_generated_samples) < 1:
        raise ValueError("display generated samples must be positive.")


def main() -> int:
    args = parse_args()
    _validate_args(args)
    device = require_device(str(args.device))
    base_dir = args.base_result_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    result_path = output_dir / "four_method_results.npz"
    summary_path = output_dir / "summary.json"
    best_checkpoint = output_dir / "nonlinear_flow_best.pt"
    last_checkpoint = output_dir / "nonlinear_flow_last.pt"
    started = time.perf_counter()

    pca_path = base_dir / "pca82_dataset.npz"
    moments_path = base_dir / "selected_theta_moments.npz"
    base_summary_path = base_dir / "summary.json"
    if not pca_path.is_file() or not moments_path.is_file() or not base_summary_path.is_file():
        raise FileNotFoundError(
            f"Expected existing PCA, moment, and summary artifacts under {base_dir}."
        )
    base_summary = json.loads(base_summary_path.read_text(encoding="utf-8"))
    session_index = int(base_summary["session_index"])
    session_label = str(base_summary["session_label"])
    with np.load(pca_path, allow_pickle=False) as saved:
        theta = np.asarray(saved["theta"], dtype=np.float64)
        x_all = np.asarray(saved["x"], dtype=np.float64)
        outer_train = np.asarray(saved["train_indices"], dtype=np.int64)
        test = np.asarray(saved["test_indices"], dtype=np.int64)
        explained_variance_ratio = np.asarray(
            saved["explained_variance_ratio"], dtype=np.float64
        )
    with np.load(moments_path, allow_pickle=False) as saved:
        selected_theta = np.asarray(saved["selected_theta"], dtype=np.float64)
        affine_mean = np.asarray(saved["flow_mean"], dtype=np.float64)
        affine_covariance = np.asarray(saved["flow_covariance"], dtype=np.float64)
        gkr_mean = np.asarray(saved["gkr_mean"], dtype=np.float64)
        gkr_covariance = np.asarray(saved["gkr_covariance"], dtype=np.float64)
        affine_ll = np.asarray(saved["flow_test_log_likelihood"], dtype=np.float64)
        gkr_ll = np.asarray(saved["gkr_test_log_likelihood"], dtype=np.float64)
        flow_train = np.asarray(saved["flow_train_indices"], dtype=np.int64)
        flow_validation = np.asarray(saved["flow_validation_indices"], dtype=np.int64)
        saved_test = np.asarray(saved["test_indices"], dtype=np.int64)
        saved_outer_train = np.asarray(saved["outer_train_indices"], dtype=np.int64)
    if not np.array_equal(test, saved_test) or not np.array_equal(
        outer_train, saved_outer_train
    ):
        raise ValueError("Base PCA and moment artifacts use different outer splits.")
    _validate_split_protocol(
        outer_train=outer_train,
        flow_train=flow_train,
        flow_validation=flow_validation,
        test=test,
    )

    n_bins, effective_bin_width = _periodic_bin_spec(float(args.bin_width))
    bin_mean, bin_covariance, bin_counts = _fit_binned_ledoit_wolf(
        x_all[outer_train], theta[outer_train], n_bins=n_bins
    )
    selected_bins = _periodic_bin_indices(selected_theta, n_bins=n_bins)
    test_bins = _periodic_bin_indices(theta[test], n_bins=n_bins)
    binned_selected_mean = bin_mean[selected_bins]
    binned_selected_covariance = bin_covariance[selected_bins]
    binned_ll = _grouped_gaussian_log_likelihood(
        x_all[test],
        test_bins,
        bin_mean,
        bin_covariance,
        jitter=float(args.likelihood_jitter),
    )

    condition_all = encode_flow_orientation(
        theta, period=PERIOD, encoding="periodic-rbf"
    )
    signature = {
        "base_result_dir": str(base_dir),
        "bin_width_requested": float(args.bin_width),
        "bin_count": int(n_bins),
        "bin_width_effective": float(effective_bin_width),
        "epochs": int(args.epochs),
        "early_patience": int(args.early_patience),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "fixed_validation_paths": int(args.fixed_validation_paths),
        "hutchinson_probes": int(args.hutchinson_probes),
        "ode_steps": int(args.ode_steps),
        "likelihood_batch_size": int(args.likelihood_batch_size),
        "generated_samples": int(args.generated_samples),
        "likelihood_jitter": float(args.likelihood_jitter),
        "seed": int(args.seed),
    }

    if result_path.is_file() and summary_path.is_file() and not args.force:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("signature") == signature:
            with np.load(result_path, allow_pickle=False) as saved:
                png, svg = _plot(
                    test_pc12=np.asarray(saved["test_pc12"]),
                    theta_test=np.asarray(saved["theta_test"]),
                    selected_theta=np.asarray(saved["selected_theta"]),
                    affine_mean=np.asarray(saved["affine_mean"]),
                    affine_covariance=np.asarray(saved["affine_covariance"]),
                    gkr_mean=np.asarray(saved["gkr_mean"]),
                    gkr_covariance=np.asarray(saved["gkr_covariance"]),
                    binned_mean=np.asarray(saved["binned_selected_mean"]),
                    binned_covariance=np.asarray(saved["binned_selected_covariance"]),
                    generated_pc12=np.asarray(saved["nonlinear_generated_x"])[:, :2],
                    generated_theta=np.asarray(saved["nonlinear_generated_theta"]),
                    likelihoods={
                        "GKR": np.asarray(saved["gkr_test_log_likelihood"]),
                        "Bin +\nLW": np.asarray(saved["binned_test_log_likelihood"]),
                        "Affine\nFlow": np.asarray(saved["affine_test_log_likelihood"]),
                        "Nonlinear\nFlow": np.asarray(saved["nonlinear_test_log_likelihood"]),
                    },
                    display_generated_samples=int(args.display_generated_samples),
                    output_dir=figures_dir,
                )
            summary["session_index"] = session_index
            summary["session_label"] = session_label
            summary.pop("session", None)
            summary.setdefault("binning", {})["fit_scope"] = "outer_training_split_only"
            summary["binning"]["held_out_test_overlap"] = 0
            summary["display_generated_samples"] = int(args.display_generated_samples)
            summary_path.write_text(
                json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8"
            )
            print(f"Saved: {png}", flush=True)
            print(f"Saved: {svg}", flush=True)
            return 0

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    model = build_flow_skl_model(
        velocity_family="nonlinear",
        theta_dim=int(condition_all.shape[1]),
        x_dim=int(x_all.shape[1]),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        path_schedule="cosine",
        divergence_estimator="hutchinson",
        hutchinson_probes=int(args.hutchinson_probes),
        theta_embedding="identity",
    ).to(device)
    training = train_flow_skl_model(
        model=model,
        theta_train=condition_all[flow_train],
        x_train=x_all[flow_train],
        theta_val=condition_all[flow_validation],
        x_val=x_all[flow_validation],
        device=device,
        velocity_family="nonlinear",
        path_schedule="cosine",
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        lr_schedule="constant",
        weight_decay=0.0,
        t_eps=5e-4,
        patience=int(args.early_patience),
        min_delta=1e-4,
        ema_alpha=0.05,
        max_grad_norm=10.0,
        log_every=50,
        checkpoint_selection="last",
        best_checkpoint_metric="flow_matching",
        fixed_validation=True,
        fixed_validation_paths=int(args.fixed_validation_paths),
        validation_seed=int(args.seed) + 10_000,
        retain_best_state=True,
    )
    torch.save(_cpu_state_dict(model), last_checkpoint)
    best_state = training.get("best_state_dict")
    if not isinstance(best_state, dict):
        raise RuntimeError("Flow training did not retain a best checkpoint.")
    model.load_state_dict(best_state)
    torch.save(_cpu_state_dict(model), best_checkpoint)

    rng = np.random.default_rng(int(args.seed) + 20_000)
    generated_theta = rng.uniform(0.0, PERIOD, size=int(args.generated_samples))
    generated_condition = encode_flow_orientation(
        generated_theta, period=PERIOD, encoding="periodic-rbf"
    )
    torch.manual_seed(int(args.seed) + 20_001)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed) + 20_001)
    generated_x = _sample_flow_batched(
        model=model,
        condition=generated_condition,
        device=device,
        batch_size=int(args.likelihood_batch_size),
        ode_steps=int(args.ode_steps),
    )
    torch.manual_seed(int(args.seed) + 30_000)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed) + 30_000)
    nonlinear_ll = _flow_log_likelihood_batched(
        model=model,
        x=x_all[test],
        condition=condition_all[test],
        device=device,
        batch_size=int(args.likelihood_batch_size),
        ode_steps=int(args.ode_steps),
    )

    training_arrays = {
        key: np.asarray(training[key])
        for key in (
            "train_losses",
            "val_losses",
            "val_monitor_losses",
            "learning_rates",
        )
    }
    np.savez_compressed(
        result_path,
        test_pc12=x_all[test, :2],
        theta_test=theta[test],
        selected_theta=selected_theta,
        affine_mean=affine_mean,
        affine_covariance=affine_covariance,
        gkr_mean=gkr_mean,
        gkr_covariance=gkr_covariance,
        binned_selected_mean=binned_selected_mean,
        binned_selected_covariance=binned_selected_covariance,
        binned_all_mean=bin_mean,
        binned_all_covariance=bin_covariance,
        binned_counts=bin_counts,
        nonlinear_generated_x=generated_x,
        nonlinear_generated_theta=generated_theta,
        affine_test_log_likelihood=affine_ll,
        gkr_test_log_likelihood=gkr_ll,
        binned_test_log_likelihood=binned_ll,
        nonlinear_test_log_likelihood=nonlinear_ll,
        outer_train_indices=outer_train,
        flow_train_indices=flow_train,
        flow_validation_indices=flow_validation,
        test_indices=test,
        **training_arrays,
    )
    png, svg = _plot(
        test_pc12=x_all[test, :2],
        theta_test=theta[test],
        selected_theta=selected_theta,
        affine_mean=affine_mean,
        affine_covariance=affine_covariance,
        gkr_mean=gkr_mean,
        gkr_covariance=gkr_covariance,
        binned_mean=binned_selected_mean,
        binned_covariance=binned_selected_covariance,
        generated_pc12=generated_x[:, :2],
        generated_theta=generated_theta,
        likelihoods={
            "GKR": gkr_ll,
            "Bin +\nLW": binned_ll,
            "Affine\nFlow": affine_ll,
            "Nonlinear\nFlow": nonlinear_ll,
        },
        display_generated_samples=int(args.display_generated_samples),
        output_dir=figures_dir,
    )
    likelihood_summary = {
        "affine_flow": {
            "mean": float(np.mean(affine_ll)),
            "standard_error": _standard_error(affine_ll),
            "definition": "conditional 82D Gaussian likelihood from affine endpoint moments",
        },
        "gkr": {
            "mean": float(np.mean(gkr_ll)),
            "standard_error": _standard_error(gkr_ll),
            "definition": "conditional 82D Gaussian likelihood from GKR moments",
        },
        "binning_ledoit_wolf": {
            "mean": float(np.mean(binned_ll)),
            "standard_error": _standard_error(binned_ll),
            "definition": "conditional 82D Gaussian likelihood from periodic-bin Ledoit-Wolf moments",
        },
        "nonlinear_flow": {
            "mean": float(np.mean(nonlinear_ll)),
            "standard_error": _standard_error(nonlinear_ll),
            "definition": "conditional CNF likelihood from reverse ODE and Hutchinson divergence",
        },
    }
    summary = {
        "signature": signature,
        "session_index": session_index,
        "session_label": session_label,
        "n_trials": int(theta.size),
        "pca_dim": int(x_all.shape[1]),
        "pca_whiten": False,
        "pca_fit_scope": "outer_training_split_only",
        "pca_explained_variance_ratio_sum": float(np.sum(explained_variance_ratio)),
        "n_outer_train": int(outer_train.size),
        "n_flow_train": int(flow_train.size),
        "n_flow_validation": int(flow_validation.size),
        "n_test": int(test.size),
        "display_generated_samples": int(args.display_generated_samples),
        "binning": {
            "requested_width": float(args.bin_width),
            "period": PERIOD,
            "n_bins": int(n_bins),
            "effective_width": float(effective_bin_width),
            "training_counts": bin_counts,
            "fit_scope": "outer_training_split_only",
            "held_out_test_overlap": 0,
            "covariance": "Ledoit-Wolf shrinkage",
        },
        "nonlinear_flow": {
            "velocity": "unconstrained conditional FiLM",
            "condition_encoding": "periodic-rbf8 supplied as model condition",
            "hidden_dim": int(args.hidden_dim),
            "depth": int(args.depth),
            "path_schedule": "cosine",
            "divergence_estimator": "Hutchinson",
            "hutchinson_probes": int(args.hutchinson_probes),
            "ode_steps": int(args.ode_steps),
            "selected_checkpoint": "best fixed-validation FM checkpoint",
            "selected_epoch": int(training["best_epoch"]),
            "stopped_epoch": int(training["stopped_epoch"]),
            "stopped_early": bool(training["stopped_early"]),
        },
        "test_log_likelihood": likelihood_summary,
        "likelihood_comparison_note": (
            "Affine Flow, GKR, and binning use Gaussian moment likelihoods; "
            "nonlinear Flow uses its full CNF likelihood."
        ),
        "runtime_seconds": float(time.perf_counter() - started),
        "artifacts": {
            "results": str(result_path),
            "nonlinear_best_checkpoint": str(best_checkpoint),
            "nonlinear_last_checkpoint": str(last_checkpoint),
            "png": str(png),
            "svg": str(svg),
        },
    }
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
