#!/usr/bin/env python3
"""Refit Stringer GKR with a conventional periodic covariance kernel."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.gkr import GKRConfig, TorchGKR, TorchKernelCovariance
from fisher.shared_fisher_est import require_device
from global_setting import DATA_DIR

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
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "stringer_gkr_conventional_kernel_gt1",
    )
    parser.add_argument("--covariance-epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _as_2d(
    value: np.ndarray | torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 2:
        raise ValueError("Expected a two-dimensional tensor.")
    return tensor


class ConventionalPeriodicKernelCovariance(TorchKernelCovariance):
    """Kernel covariance using exp(-d) with the standard periodic RBF distance."""

    def forward(
        self,
        query: np.ndarray | torch.Tensor,
        *,
        batch_size: int = 3000,
    ) -> torch.Tensor:
        if self.train_inputs.numel() == 0:
            raise RuntimeError("Call set_data before predicting covariance.")
        query_t = _as_2d(query, dtype=self.dtype, device=self.device)
        precision = self.precision()
        numerator = torch.zeros(
            query_t.shape[0],
            self.n_output,
            self.n_output,
            dtype=self.dtype,
            device=self.device,
        )
        denominator = torch.zeros(
            query_t.shape[0], dtype=self.dtype, device=self.device
        )
        periods = (
            (None,) * self.n_input
            if self.circular_period is None
            else (float(self.circular_period),) * self.n_input
        )
        for start in range(0, self.train_inputs.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), self.train_inputs.shape[0])
            inputs = self.train_inputs[start:stop]
            residuals = self.train_residuals[start:stop]
            difference = inputs[:, None, :] - query_t[None, :, :]
            for dimension, period in enumerate(periods):
                if period is not None:
                    difference[..., dimension] = torch.sin(
                        torch.pi * difference[..., dimension] / period
                    )
            distance = torch.einsum(
                "bqi,ij,bqj->bq", difference, precision, difference
            )
            weights = torch.exp(-distance)
            grams = torch.einsum("bi,bj->bij", residuals, residuals)
            numerator += torch.einsum("bq,bij->qij", weights, grams)
            denominator += weights.sum(dim=0)
        covariance = numerator / denominator.clamp_min(1e-12)[:, None, None]
        eye = torch.eye(
            self.n_output, dtype=self.dtype, device=self.device
        )
        return covariance + self.jitter * eye.unsqueeze(0)


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
    eye = np.eye(x.shape[1], dtype=np.float64)
    stabilized = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
    cholesky = np.linalg.cholesky(stabilized + float(jitter) * eye[None])
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


def _ellipse_parameters(covariance: np.ndarray) -> tuple[float, float, float]:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    principal = eigenvectors[:, order[0]]
    return (
        float(2.0 * np.sqrt(eigenvalues[0])),
        float(2.0 * np.sqrt(eigenvalues[1])),
        float(np.degrees(np.arctan2(principal[1], principal[0]))),
    )


def _plot_panel(
    axis: plt.Axes,
    *,
    test_pc12: np.ndarray,
    theta_test: np.ndarray,
    selected_theta: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    title: str,
    norm: Normalize,
) -> None:
    cmap = matplotlib.colormaps["twilight_shifted"]
    axis.scatter(
        test_pc12[:, 0],
        test_pc12[:, 1],
        c=theta_test,
        cmap=cmap,
        norm=norm,
        marker="+",
        s=15,
        linewidths=0.65,
        alpha=0.32,
        zorder=1,
    )
    for theta, mean, covariance in zip(
        selected_theta, means, covariances, strict=True
    ):
        color = cmap(norm(float(theta)))
        width, height, angle = _ellipse_parameters(covariance[:2, :2])
        axis.add_patch(
            Ellipse(
                mean[:2],
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
            color=color,
            edgecolor="black",
            linewidth=0.7,
            s=34,
            zorder=4,
        )
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines[:].set_visible(False)


def _plot(
    *,
    test_pc12: np.ndarray,
    theta_test: np.ndarray,
    selected_theta: np.ndarray,
    original_mean: np.ndarray,
    original_covariance: np.ndarray,
    conventional_mean: np.ndarray,
    conventional_covariance: np.ndarray,
    original_likelihood: float,
    conventional_likelihood: float,
    output_dir: Path,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 16,
            "legend.fontsize": 14,
            "axes.grid": False,
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(7.5, 3.5), constrained_layout=True
    )
    norm = Normalize(vmin=0.0, vmax=PERIOD)
    _plot_panel(
        axes[0],
        test_pc12=test_pc12,
        theta_test=theta_test,
        selected_theta=selected_theta,
        means=original_mean,
        covariances=original_covariance,
        title=f"Original GKR\nmean LL = {original_likelihood:.1f}",
        norm=norm,
    )
    _plot_panel(
        axes[1],
        test_pc12=test_pc12,
        theta_test=theta_test,
        selected_theta=selected_theta,
        means=conventional_mean,
        covariances=conventional_covariance,
        title=f"Conventional periodic GKR\nmean LL = {conventional_likelihood:.1f}",
        norm=norm,
    )
    all_points = np.vstack(
        [test_pc12, original_mean[:, :2], conventional_mean[:, :2]]
    )
    low = np.min(all_points, axis=0)
    high = np.max(all_points, axis=0)
    span = np.maximum(high - low, 1.0)
    for axis in axes:
        axis.set_xlim(low[0] - 0.06 * span[0], high[0] + 0.06 * span[0])
        axis.set_ylim(low[1] - 0.06 * span[1], high[1] + 0.06 * span[1])
    colorbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap="twilight_shifted"),
        ax=axes,
        orientation="vertical",
        fraction=0.035,
        pad=0.025,
    )
    colorbar.set_label(r"Orientation $\theta$")
    colorbar.set_ticks(
        [0.0, 0.5 * PERIOD, PERIOD],
        labels=[r"$0$", r"$\pi/2$", r"$\pi$"],
    )
    handles = [
        Line2D(
            [], [], marker="+", linestyle="none", color="black",
            markersize=8, label="Held-out data"
        ),
        Line2D(
            [], [], marker="o", linestyle="none", markerfacecolor="black",
            markeredgecolor="black", markersize=6, label="Fitted mean"
        ),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.47, -0.02),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "gkr_conventional_periodic_kernel_comparison.png"
    svg = output_dir / "gkr_conventional_periodic_kernel_comparison.svg"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    return png, svg


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


def main() -> int:
    args = parse_args()
    if int(args.covariance_epochs) < 1:
        raise ValueError("--covariance-epochs must be positive.")
    device = require_device(str(args.device))
    base_dir = args.base_result_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pca_path = base_dir / "pca82_dataset.npz"
    moments_path = base_dir / "selected_theta_moments.npz"
    base_summary_path = base_dir / "summary.json"
    if (
        not pca_path.is_file()
        or not moments_path.is_file()
        or not base_summary_path.is_file()
    ):
        raise FileNotFoundError(f"Missing base artifacts under {base_dir}.")
    base_summary = json.loads(base_summary_path.read_text(encoding="utf-8"))
    session_index = int(base_summary["session_index"])
    session_label = str(base_summary["session_label"])
    with np.load(pca_path, allow_pickle=False) as saved:
        theta = np.asarray(saved["theta"], dtype=np.float64)
        x = np.asarray(saved["x"], dtype=np.float64)
        fit = np.asarray(saved["train_indices"], dtype=np.int64)
        test = np.asarray(saved["test_indices"], dtype=np.int64)
    with np.load(moments_path, allow_pickle=False) as saved:
        selected_theta = np.asarray(saved["selected_theta"], dtype=np.float64)
        original_mean = np.asarray(saved["gkr_mean"], dtype=np.float64)
        original_covariance = np.asarray(saved["gkr_covariance"], dtype=np.float64)
        original_test_likelihood = np.asarray(
            saved["gkr_test_log_likelihood"], dtype=np.float64
        )

    result_path = output_dir / "conventional_gkr_results.npz"
    checkpoint_path = output_dir / "conventional_gkr_model.pt"
    started = time.perf_counter()
    if result_path.is_file() and not args.force:
        with np.load(result_path, allow_pickle=False) as saved:
            required = {
                "test_mean",
                "test_covariance",
                "covariance_epochs",
            }
            missing = required - set(saved.files)
            if missing:
                raise ValueError(
                    f"Cached result is missing {sorted(missing)}; pass --force."
                )
            saved_epochs = int(saved["covariance_epochs"])
            if saved_epochs != int(args.covariance_epochs):
                raise ValueError(
                    f"Cached result uses {saved_epochs} covariance epochs, "
                    f"not {int(args.covariance_epochs)}; pass --force."
                )
            conventional_mean = np.asarray(saved["conventional_mean"])
            conventional_covariance = np.asarray(saved["conventional_covariance"])
            test_mean = np.asarray(saved["test_mean"])
            test_covariance = np.asarray(saved["test_covariance"])
            conventional_test_likelihood = np.asarray(
                saved["conventional_test_log_likelihood"]
            )
            covariance_losses = np.asarray(saved["covariance_losses"])
            learned_precision = float(saved["learned_precision"])
    else:
        config = GKRConfig(
            mean_iterations=300,
            mean_learning_rate=0.05,
            n_inducing=200,
            covariance_epochs=int(args.covariance_epochs),
            covariance_learning_rate=0.1,
            covariance_batch_size=3000,
            validation_fraction=0.33,
            covariance_jitter=1e-6,
            likelihood_jitter=1e-5,
            prediction_batch_size=3000,
            standardize_responses=True,
            log_every=25,
        )
        model = TorchGKR(
            n_input=1,
            n_output=x.shape[1],
            circular_period=PERIOD,
            config=config,
            dtype=torch.float64,
            device=device,
            seed=int(args.seed),
        )
        model.covariance_model = ConventionalPeriodicKernelCovariance(
            n_input=1,
            n_output=x.shape[1],
            circular_period=PERIOD,
            jitter=config.covariance_jitter,
            dtype=torch.float64,
            device=device,
        )
        model.fit(x[fit], theta[fit, None])
        conventional_mean, conventional_covariance = model.predict(
            selected_theta[:, None]
        )
        test_mean, test_covariance = model.predict(theta[test, None])
        conventional_test_likelihood = _gaussian_log_likelihood(
            x[test],
            test_mean,
            test_covariance,
            jitter=config.likelihood_jitter,
        )
        covariance_losses = np.asarray(
            model.covariance_loss_history, dtype=np.float64
        )
        learned_precision = float(
            model.covariance_model.precision().detach().cpu().item()
        )
        np.savez_compressed(
            result_path,
            selected_theta=selected_theta,
            test_pc12=x[test, :2],
            theta_test=theta[test],
            conventional_mean=conventional_mean,
            conventional_covariance=conventional_covariance,
            test_mean=test_mean,
            test_covariance=test_covariance,
            conventional_test_log_likelihood=conventional_test_likelihood,
            original_test_log_likelihood=original_test_likelihood,
            covariance_losses=covariance_losses,
            learned_precision=np.asarray(learned_precision),
            covariance_epochs=np.asarray(int(args.covariance_epochs)),
        )
        torch.save(
            {
                "mean_model": model.mean_model.state_dict(),
                "mean_likelihood": model.mean_likelihood.state_dict(),
                "covariance_model": model.covariance_model.state_dict(),
                "config": vars(config),
            },
            checkpoint_path,
        )

    effective_radius = math.asin(
        min(1.0, 1.0 / math.sqrt(max(learned_precision, 1e-12)))
    )
    original_mean_likelihood = float(np.mean(original_test_likelihood))
    conventional_mean_likelihood = float(np.mean(conventional_test_likelihood))
    png, svg = _plot(
        test_pc12=x[test, :2],
        theta_test=theta[test],
        selected_theta=selected_theta,
        original_mean=original_mean,
        original_covariance=original_covariance,
        conventional_mean=conventional_mean,
        conventional_covariance=conventional_covariance,
        original_likelihood=original_mean_likelihood,
        conventional_likelihood=conventional_mean_likelihood,
        output_dir=output_dir / "figures",
    )
    summary = {
        "protocol": {
            "session_index": session_index,
            "session": session_label,
            "pca_dim": int(x.shape[1]),
            "fit_count": int(fit.size),
            "test_count": int(test.size),
            "fit_fraction": float(fit.size / x.shape[0]),
            "test_fraction": float(test.size / x.shape[0]),
            "kernel": (
                "exp(-d), d = sin(pi * delta / period)^T "
                "precision sin(pi * delta / period)"
            ),
            "covariance_epochs": int(args.covariance_epochs),
        },
        "original_mean_test_log_likelihood": original_mean_likelihood,
        "conventional_mean_test_log_likelihood": conventional_mean_likelihood,
        "likelihood_improvement": (
            conventional_mean_likelihood - original_mean_likelihood
        ),
        "learned_precision": learned_precision,
        "effective_exp_minus_one_radius_radians": effective_radius,
        "effective_exp_minus_one_radius_degrees": math.degrees(effective_radius),
        "covariance_loss": {
            "first": float(covariance_losses[0]),
            "last": float(covariance_losses[-1]),
            "minimum": float(np.min(covariance_losses)),
            "minimum_epoch": int(np.argmin(covariance_losses) + 1),
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "artifacts": {
            "results": result_path,
            "checkpoint": checkpoint_path,
            "png": png,
            "svg": svg,
        },
    }
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
