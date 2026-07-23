#!/usr/bin/env python3
"""Visualize held-out Stringer PCA data with Flow and GKR Gaussian moments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
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
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.fisher_validation import (
    fit_flow_direction_estimator,
    gkr_checkpoint,
    stratified_disjoint_subset_indices,
)
from fisher.flow_matching_skl import (
    DEFAULT_AFFINE_COVARIANCE_ODE_STEPS,
    estimate_affine_endpoint_gaussians,
)
from fisher.gkr import GKRConfig, TorchGKR
from fisher.shared_fisher_est import require_device
from fisher.stringer_dataset import list_stringer_sessions, load_stringer_session
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
    parser.add_argument("--session-index", type=int, default=0)
    parser.add_argument("--pca-dim", type=int, default=82)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--flow-validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-strata", type=int, default=16)
    parser.add_argument("--selected-theta-count", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_MAX_EPOCHS)
    parser.add_argument(
        "--early-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument(
        "--ode-steps", type=int, default=DEFAULT_AFFINE_COVARIANCE_ODE_STEPS
    )
    parser.add_argument("--gkr-mean-iterations", type=int, default=300)
    parser.add_argument("--gkr-mean-lr", type=float, default=0.05)
    parser.add_argument("--gkr-n-inducing", type=int, default=200)
    parser.add_argument("--gkr-covariance-epochs", type=int, default=30)
    parser.add_argument("--gkr-covariance-lr", type=float, default=0.1)
    parser.add_argument("--gkr-covariance-batch-size", type=int, default=3000)
    parser.add_argument("--likelihood-jitter", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "stringer_pca82_flow_gkr_visualization_session0",
    )
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


def _stratified_train_test_split(
    theta: np.ndarray,
    *,
    train_fraction: float,
    n_strata: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    n_train = min(max(int(round(float(train_fraction) * values.size)), 1), values.size - 1)
    train = stratified_disjoint_subset_indices(
        values,
        n_train,
        n_subsets=1,
        n_strata=int(n_strata),
        seed=int(seed),
        period=PERIOD,
    )[0]
    test = np.setdiff1d(np.arange(values.size, dtype=np.int64), train)
    return train, test


def _ellipse_parameters(covariance: np.ndarray, *, n_std: float = 1.0) -> tuple[float, float, float]:
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.shape != (2, 2):
        raise ValueError("covariance must be 2 by 2.")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (cov + cov.T))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    principal = eigenvectors[:, order[0]]
    width, height = 2.0 * float(n_std) * np.sqrt(eigenvalues)
    angle = float(np.degrees(np.arctan2(principal[1], principal[0])))
    return float(width), float(height), angle


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
    if covariances.shape != (observations.shape[0], observations.shape[1], observations.shape[1]):
        raise ValueError("covariance must have shape [n, d, d].")
    if float(jitter) < 0.0:
        raise ValueError("jitter must be nonnegative.")
    eye = np.eye(observations.shape[1], dtype=np.float64)
    symmetric = 0.5 * (covariances + np.swapaxes(covariances, -1, -2))
    cholesky = np.linalg.cholesky(symmetric + float(jitter) * eye[None, :, :])
    residual = observations - means
    whitened = np.linalg.solve(cholesky, residual[..., None]).squeeze(-1)
    log_determinant = 2.0 * np.log(
        np.diagonal(cholesky, axis1=-2, axis2=-1)
    ).sum(axis=1)
    normalizer = observations.shape[1] * np.log(2.0 * np.pi)
    return -0.5 * (normalizer + log_determinant + np.sum(whitened**2, axis=1))


def _axis_limits(
    test_pc12: np.ndarray,
    moment_sets: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    points = [np.asarray(test_pc12, dtype=np.float64)]
    for means, covariances in moment_sets:
        means_2d = np.asarray(means, dtype=np.float64)[:, :2]
        covariance_2d = np.asarray(covariances, dtype=np.float64)[:, :2, :2]
        radius = np.sqrt(np.maximum(np.diagonal(covariance_2d, axis1=1, axis2=2), 0.0))
        points.extend([means_2d - radius, means_2d + radius])
    joined = np.vstack(points)
    low = np.min(joined, axis=0)
    high = np.max(joined, axis=0)
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
    flow_mean: np.ndarray,
    flow_covariance: np.ndarray,
    gkr_mean: np.ndarray,
    gkr_covariance: np.ndarray,
    flow_test_log_likelihood: np.ndarray,
    gkr_test_log_likelihood: np.ndarray,
    session_labels: list[str] | None = None,
    flow_session_log_likelihood: np.ndarray | None = None,
    gkr_session_log_likelihood: np.ndarray | None = None,
    output_dir: Path,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 13,
            "savefig.bbox": "tight",
        }
    )
    cmap = matplotlib.colormaps["twilight_shifted"]
    norm = Normalize(vmin=0.0, vmax=PERIOD)
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.5), constrained_layout=True)
    for axis in axes[:3]:
        axis.scatter(
            test_pc12[:, 0],
            test_pc12[:, 1],
            c=np.mod(theta_test, PERIOD),
            cmap=cmap,
            norm=norm,
            s=8,
            alpha=0.45,
            linewidths=0.0,
            rasterized=True,
            zorder=1,
        )
        axis.set_xlabel("PC1")
        axis.set_aspect("equal", adjustable="box")
        axis.set_axisbelow(True)
        axis.yaxis.grid(True, color="0.90", linewidth=0.8)
        axis.xaxis.grid(False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines["left"].set_linewidth(1.8)
        axis.spines["bottom"].set_linewidth(1.8)
        axis.tick_params(width=1.8)
    axes[0].set_ylabel("PC2")
    axes[0].set_title("Held-out PCA data")
    axes[1].set_title("Flow matching")
    axes[2].set_title("GKR")

    for axis, means, covariances in (
        (axes[1], flow_mean, flow_covariance),
        (axes[2], gkr_mean, gkr_covariance),
    ):
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
        [(flow_mean, flow_covariance), (gkr_mean, gkr_covariance)],
    )
    axes[0].set_xlim(*xlim)
    axes[0].set_ylim(*ylim)
    for axis in axes[1:3]:
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
    axes[1].legend(
        handles=[
            Line2D([], [], marker="o", color="black", linestyle="none", label="Mean"),
            Line2D([], [], color="black", linewidth=2.2, label="1-SD covariance"),
        ],
        frameon=False,
        loc="lower left",
    )
    scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(
        scalar_mappable, ax=axes[:3], location="right", fraction=0.025, pad=0.02
    )
    colorbar.set_label(r"Orientation $\theta$")
    colorbar.set_ticks(
        [0.0, 0.5 * PERIOD, PERIOD],
        labels=[r"$0$", r"$\pi/2$", r"$\pi$"],
    )

    flow_sessions = (
        np.asarray([np.mean(flow_test_log_likelihood)], dtype=np.float64)
        if flow_session_log_likelihood is None
        else np.asarray(flow_session_log_likelihood, dtype=np.float64).reshape(-1)
    )
    gkr_sessions = (
        np.asarray([np.mean(gkr_test_log_likelihood)], dtype=np.float64)
        if gkr_session_log_likelihood is None
        else np.asarray(gkr_session_log_likelihood, dtype=np.float64).reshape(-1)
    )
    if flow_sessions.shape != gkr_sessions.shape or flow_sessions.size < 1:
        raise ValueError("Flow and GKR session likelihood arrays must have equal nonzero size.")
    labels = (
        ["Session"] * flow_sessions.size
        if session_labels is None
        else [str(label) for label in session_labels]
    )
    if len(labels) != flow_sessions.size:
        raise ValueError("session_labels must match the number of session likelihoods.")
    session_values = np.column_stack([flow_sessions, gkr_sessions])
    means = np.mean(session_values, axis=0)
    standard_errors = (
        np.std(session_values, axis=0, ddof=1) / np.sqrt(session_values.shape[0])
        if session_values.shape[0] > 1
        else np.asarray(
            [
                np.std(flow_test_log_likelihood, ddof=1)
                / np.sqrt(np.asarray(flow_test_log_likelihood).size),
                np.std(gkr_test_log_likelihood, ddof=1)
                / np.sqrt(np.asarray(gkr_test_log_likelihood).size),
            ]
        )
    )
    value_low = float(np.min(session_values))
    value_high = float(np.max(session_values))
    value_span = max(value_high - value_low, 1.0)
    y_low = value_low - 0.12 * value_span
    y_high = value_high + 0.12 * value_span
    axes[3].bar(
        [0, 1],
        means - y_low,
        bottom=y_low,
        color=["C0", "C2"],
        width=0.68,
        alpha=0.28,
        edgecolor=["C0", "C2"],
        linewidth=1.8,
    )
    session_colors = matplotlib.colormaps["tab10"](
        np.linspace(0.0, 1.0, max(session_values.shape[0], 2), endpoint=False)
    )
    for index, (label, values) in enumerate(zip(labels, session_values, strict=True)):
        color = session_colors[index]
        axes[3].plot([0, 1], values, color=color, linewidth=1.4, alpha=0.85, zorder=3)
        axes[3].scatter(
            [0, 1],
            values,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            s=34,
            label=label,
            zorder=4,
        )
    axes[3].errorbar(
        [0, 1],
        means,
        yerr=standard_errors,
        color="black",
        linestyle="none",
        marker="D",
        markersize=5,
        linewidth=1.8,
        capsize=4,
        capthick=1.8,
        zorder=5,
    )
    axes[3].set_xticks([0, 1], ["Flow", "GKR"])
    axes[3].set_ylabel("Mean test log likelihood\n" r"(higher is better $\uparrow$)")
    axes[3].set_title("Held-out likelihood")
    axes[3].set_ylim(y_low, y_high)
    axes[3].set_axisbelow(True)
    axes[3].yaxis.grid(True, color="0.90", linewidth=0.8)
    axes[3].xaxis.grid(False)
    axes[3].spines[["top", "right"]].set_visible(False)
    axes[3].spines["left"].set_linewidth(1.8)
    axes[3].spines["bottom"].set_linewidth(1.8)
    axes[3].tick_params(width=1.8)
    if session_values.shape[0] > 1:
        axes[3].legend(
            frameon=False,
            fontsize=10,
            ncol=2,
            loc="lower left",
            handletextpad=0.3,
            columnspacing=0.7,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "stringer_pca82_test_flow_gkr_moments"
    png, svg = stem.with_suffix(".png"), stem.with_suffix(".svg")
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def _validate(args: argparse.Namespace) -> None:
    if int(args.pca_dim) < 2:
        raise ValueError("--pca-dim must be at least 2.")
    if not 0.0 < float(args.train_fraction) < 1.0:
        raise ValueError("--train-fraction must be in (0, 1).")
    if not 0.0 < float(args.flow_validation_fraction) < 1.0:
        raise ValueError("--flow-validation-fraction must be in (0, 1).")
    if int(args.selected_theta_count) < 2:
        raise ValueError("--selected-theta-count must be at least 2.")


def main() -> int:
    args = parse_args()
    _validate(args)
    device = require_device(str(args.device))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    moments_path = output_dir / "selected_theta_moments.npz"
    pca_path = output_dir / "pca82_dataset.npz"

    session = load_stringer_session(
        None,
        session_stimuli_type="gratings_static",
        session_index=int(args.session_index),
        orientation_period=PERIOD,
    )
    session_label = str(
        list_stringer_sessions("gratings_static")[int(args.session_index)].mouse_name
    )
    theta = np.asarray(session.grating_orientation, dtype=np.float64).reshape(-1)
    responses = np.asarray(session.neural_responses)
    if int(args.pca_dim) >= min(responses.shape):
        raise ValueError("--pca-dim must be below both dataset dimensions.")
    signature = {
        "session_file": str(session.session_file),
        "session_shape": list(responses.shape),
        "pca_dim": int(args.pca_dim),
        "train_fraction": float(args.train_fraction),
        "flow_validation_fraction": float(args.flow_validation_fraction),
        "split_strata": int(args.split_strata),
        "selected_theta_count": int(args.selected_theta_count),
        "epochs": int(args.epochs),
        "early_patience": int(args.early_patience),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "ode_steps": int(args.ode_steps),
        "gkr_mean_iterations": int(args.gkr_mean_iterations),
        "gkr_mean_lr": float(args.gkr_mean_lr),
        "gkr_n_inducing": int(args.gkr_n_inducing),
        "gkr_covariance_epochs": int(args.gkr_covariance_epochs),
        "gkr_covariance_lr": float(args.gkr_covariance_lr),
        "gkr_covariance_batch_size": int(args.gkr_covariance_batch_size),
        "likelihood_jitter": float(args.likelihood_jitter),
        "seed": int(args.seed),
    }
    started = time.perf_counter()
    if moments_path.is_file() and summary_path.is_file() and not args.force:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("signature") == signature:
            with np.load(moments_path, allow_pickle=False) as saved:
                png, svg = _plot(
                    test_pc12=np.asarray(saved["test_pc12"], dtype=np.float64),
                    theta_test=np.asarray(saved["theta_test"], dtype=np.float64),
                    selected_theta=np.asarray(saved["selected_theta"], dtype=np.float64),
                    flow_mean=np.asarray(saved["flow_mean"], dtype=np.float64),
                    flow_covariance=np.asarray(saved["flow_covariance"], dtype=np.float64),
                    gkr_mean=np.asarray(saved["gkr_mean"], dtype=np.float64),
                    gkr_covariance=np.asarray(saved["gkr_covariance"], dtype=np.float64),
                    flow_test_log_likelihood=np.asarray(
                        saved["flow_test_log_likelihood"], dtype=np.float64
                    ),
                    gkr_test_log_likelihood=np.asarray(
                        saved["gkr_test_log_likelihood"], dtype=np.float64
                    ),
                    output_dir=output_dir / "figures",
                )
            print(f"Saved: {png}", flush=True)
            print(f"Saved: {svg}", flush=True)
            return 0

    outer_train, test = _stratified_train_test_split(
        theta,
        train_fraction=float(args.train_fraction),
        n_strata=int(args.split_strata),
        seed=int(args.seed),
    )
    pca = PCA(
        n_components=int(args.pca_dim),
        whiten=False,
        svd_solver="randomized",
        random_state=int(args.seed),
    )
    pca.fit(responses[outer_train])
    x_all = pca.transform(responses).astype(np.float64)
    np.savez_compressed(
        pca_path,
        theta=theta,
        x=x_all,
        train_indices=outer_train,
        test_indices=test,
        components=pca.components_,
        mean=pca.mean_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        singular_values=pca.singular_values_,
    )

    flow_train_local, flow_validation_local = _stratified_train_test_split(
        theta[outer_train],
        train_fraction=1.0 - float(args.flow_validation_fraction),
        n_strata=int(args.split_strata),
        seed=int(args.seed) + 100_000,
    )
    flow_train = outer_train[flow_train_local]
    flow_validation = outer_train[flow_validation_local]
    selected_theta = np.linspace(
        0.0,
        PERIOD,
        int(args.selected_theta_count),
        endpoint=False,
        dtype=np.float64,
    )
    condition_all = encode_flow_orientation(theta, period=PERIOD, encoding="periodic-rbf")
    condition_selected = encode_flow_orientation(
        selected_theta, period=PERIOD, encoding="periodic-rbf"
    )
    flow_model, flow_training, _, _ = fit_flow_direction_estimator(
        theta_train=theta[flow_train],
        x_train=x_all[flow_train],
        theta_validation=theta[flow_validation],
        x_validation=x_all[flow_validation],
        theta_grid=selected_theta.reshape(-1, 1),
        condition_train=condition_all[flow_train],
        condition_validation=condition_all[flow_validation],
        condition_grid=condition_selected,
        device=device,
        seed=int(args.seed),
        epochs=int(args.epochs),
        patience=int(args.early_patience),
        batch_size=int(args.batch_size),
        learning_rate=float(args.lr),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        ode_steps=int(args.ode_steps),
    )
    flow_mean, flow_covariance = estimate_affine_endpoint_gaussians(
        model=flow_model,
        theta_all=condition_selected,
        device=device,
        ode_steps=int(args.ode_steps),
    )
    torch.save(flow_model.state_dict(), output_dir / "flow_selected_model.pt")

    gkr_config = GKRConfig(
        mean_iterations=int(args.gkr_mean_iterations),
        mean_learning_rate=float(args.gkr_mean_lr),
        n_inducing=int(args.gkr_n_inducing),
        covariance_epochs=int(args.gkr_covariance_epochs),
        covariance_learning_rate=float(args.gkr_covariance_lr),
        covariance_batch_size=int(args.gkr_covariance_batch_size),
    )
    gkr_model = TorchGKR(
        n_input=1,
        n_output=int(args.pca_dim),
        circular_period=PERIOD,
        config=gkr_config,
        dtype=torch.float64,
        device=device,
        seed=int(args.seed),
    )
    gkr_model.fit(x_all[outer_train], theta[outer_train].reshape(-1, 1))
    gkr_mean, gkr_covariance = gkr_model.predict(selected_theta.reshape(-1, 1))
    torch.save(gkr_checkpoint(gkr_model), output_dir / "gkr_model.pt")

    condition_test = encode_flow_orientation(
        theta[test], period=PERIOD, encoding="periodic-rbf"
    )
    flow_test_mean, flow_test_covariance = estimate_affine_endpoint_gaussians(
        model=flow_model,
        theta_all=condition_test,
        device=device,
        ode_steps=int(args.ode_steps),
    )
    gkr_test_mean, gkr_test_covariance = gkr_model.predict(theta[test].reshape(-1, 1))
    flow_test_log_likelihood = _gaussian_log_likelihood(
        x_all[test],
        flow_test_mean,
        flow_test_covariance,
        jitter=float(args.likelihood_jitter),
    )
    gkr_test_log_likelihood = _gaussian_log_likelihood(
        x_all[test],
        gkr_test_mean,
        gkr_test_covariance,
        jitter=float(args.likelihood_jitter),
    )

    np.savez_compressed(
        moments_path,
        test_pc12=x_all[test, :2],
        theta_test=theta[test],
        selected_theta=selected_theta,
        flow_mean=flow_mean,
        flow_covariance=flow_covariance,
        gkr_mean=gkr_mean,
        gkr_covariance=gkr_covariance,
        flow_test_log_likelihood=flow_test_log_likelihood,
        gkr_test_log_likelihood=gkr_test_log_likelihood,
        flow_train_indices=flow_train,
        flow_validation_indices=flow_validation,
        outer_train_indices=outer_train,
        test_indices=test,
        flow_train_losses=np.asarray(flow_training["train_losses"], dtype=np.float64),
        flow_validation_losses=np.asarray(flow_training["val_losses"], dtype=np.float64),
        gkr_mean_losses=np.asarray(gkr_model.mean_loss_history, dtype=np.float64),
        gkr_covariance_losses=np.asarray(gkr_model.covariance_loss_history, dtype=np.float64),
    )
    png, svg = _plot(
        test_pc12=x_all[test, :2],
        theta_test=theta[test],
        selected_theta=selected_theta,
        flow_mean=flow_mean,
        flow_covariance=flow_covariance,
        gkr_mean=gkr_mean,
        gkr_covariance=gkr_covariance,
        flow_test_log_likelihood=flow_test_log_likelihood,
        gkr_test_log_likelihood=gkr_test_log_likelihood,
        output_dir=output_dir / "figures",
    )
    summary = {
        "signature": signature,
        "session_index": int(args.session_index),
        "session_label": session_label,
        "session_file": str(session.session_file),
        "n_trials": int(theta.size),
        "n_neurons": int(responses.shape[1]),
        "n_outer_train": int(outer_train.size),
        "n_test": int(test.size),
        "n_flow_train": int(flow_train.size),
        "n_flow_validation": int(flow_validation.size),
        "pca_fit_scope": "outer_training_split_only",
        "pca_explained_variance_ratio_sum": float(
            np.sum(pca.explained_variance_ratio_)
        ),
        "flow_condition_encoding": "periodic-rbf8",
        "flow_selected_epoch": int(flow_training["selected_epoch"]),
        "flow_stopped_epoch": int(flow_training["stopped_epoch"]),
        "gkr_config": asdict(gkr_config),
        "test_log_likelihood": {
            "definition": "mean conditional 82D Gaussian joint log likelihood",
            "jitter": float(args.likelihood_jitter),
            "flow_mean": float(np.mean(flow_test_log_likelihood)),
            "flow_standard_error": float(
                np.std(flow_test_log_likelihood, ddof=1)
                / np.sqrt(flow_test_log_likelihood.size)
            ),
            "gkr_mean": float(np.mean(gkr_test_log_likelihood)),
            "gkr_standard_error": float(
                np.std(gkr_test_log_likelihood, ddof=1)
                / np.sqrt(gkr_test_log_likelihood.size)
            ),
        },
        "ellipse_scale": "one_standard_deviation",
        "runtime_seconds": float(time.perf_counter() - started),
        "artifacts": {
            "pca_dataset": str(pca_path),
            "moments": str(moments_path),
            "flow_checkpoint": str(output_dir / "flow_selected_model.pt"),
            "gkr_checkpoint": str(output_dir / "gkr_model.pt"),
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
