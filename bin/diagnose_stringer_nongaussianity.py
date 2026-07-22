#!/usr/bin/env python3
"""Test whether residual non-Gaussianity explains Stringer Fisher rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.continuous_fisher_comparison import ContinuousFlowConfig
from fisher.gkr import GKRConfig, TorchGKR, estimate_gkr_linear_fisher
from fisher.shared_fisher_est import require_device
from fisher.stringer_dataset import (
    StringerSessionInfo,
    list_stringer_sessions,
    load_stringer_session,
)
from fisher.stringer_nongaussian_surrogate import (
    PeriodicFourierMoments,
    StandardizedResidualBank,
    fit_periodic_fourier_moments,
    fit_standardized_residual_bank,
    sample_moment_matched_surrogate,
)
from fisher.stringer_session_identification import (
    DISTANCE_AREA_L2,
    DISTANCE_PRIMARY,
    DISTANCE_RMSE,
    FLOW_ORIENTATION_ENCODING_PERIODIC_RBF,
    make_shared_bundle,
    split_train_validation,
    stratified_half_split,
    theta_grid_periodic,
    theta_midpoints,
    train_flow_linear_curve,
)
from fisher.toy_fisher_identification import (
    METHOD_FLOW,
    METHOD_GKR,
    TOY_IDENTIFICATION_METHODS,
    evaluate_identification,
    fisher_mae,
)
from global_setting import (
    DATA_DIR,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_TRAINING_MAX_EPOCHS,
)


DISTANCES = (DISTANCE_PRIMARY, DISTANCE_AREA_L2, DISTANCE_RMSE)
DISTANCE_LABELS = {
    DISTANCE_PRIMARY: "Log correlation",
    DISTANCE_AREA_L2: "Area L2",
    DISTANCE_RMSE: "Raw RMSE",
}
METHOD_COLORS = {METHOD_FLOW: "C0", METHOD_GKR: "C2"}
METHOD_LABELS = {METHOD_FLOW: "Flow matching", METHOD_GKR: "GKR"}


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not values or any(not 0.0 <= item <= 1.0 for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated values in [0, 1].")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "stringer_nongaussianity_diagnosis",
    )
    parser.add_argument("--max-sessions", type=int, default=6)
    parser.add_argument("--lambda-list", type=parse_float_list, default=[0.0, 1.0])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--orientation-period", type=float, default=float(np.pi))
    parser.add_argument("--theta-grid-size", type=int, default=17)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--pca-random-state", type=int, default=0)
    parser.add_argument("--no-pca-whiten", action="store_true")
    parser.add_argument("--moment-harmonics", type=int, default=4)
    parser.add_argument("--moment-ridge", type=float, default=1e-3)
    parser.add_argument("--covariance-grid-size", type=int, default=32)
    parser.add_argument("--covariance-shrinkage", type=float, default=0.25)
    parser.add_argument("--residual-bins", type=int, default=16)
    parser.add_argument("--eigenvalue-floor-relative", type=float, default=1e-4)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_MAX_EPOCHS)
    parser.add_argument("--early-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--ode-steps", type=int, default=64)
    parser.add_argument("--force-moments", action="store_true")
    parser.add_argument("--force-fits", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if int(args.max_sessions) < 2:
        raise ValueError("--max-sessions must be at least 2.")
    if int(args.repeats) < 1:
        raise ValueError("--repeats must be positive.")
    if int(args.pca_dim) < 1:
        raise ValueError("--pca-dim must be positive.")
    if not 0.0 < float(args.train_frac) < 1.0:
        raise ValueError("--train-frac must be in (0, 1).")
    if int(args.residual_bins) < 2 or int(args.covariance_grid_size) < 2:
        raise ValueError("Residual and covariance grid sizes must be at least 2.")


def flow_config(args: argparse.Namespace) -> ContinuousFlowConfig:
    return ContinuousFlowConfig(
        epochs=int(args.epochs),
        early_patience=int(args.early_patience),
        early_min_delta=1e-4,
        early_ema_alpha=0.05,
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=0.0,
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        path_schedule="cosine",
        t_eps=5e-4,
        quadrature_steps=64,
        mc_jeffreys_sample=0,
        ode_steps=int(args.ode_steps),
        ode_method="midpoint",
        divergence_estimator="exact",
        hutchinson_probes=1,
        shared_affine_a_diag_jitter=1e-3,
        solve_jitter=1e-6,
        max_grad_norm=10.0,
        log_every=50,
        affine_ridge=1e-6,
    )


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _moment_signature(
    args: argparse.Namespace, session_info: StringerSessionInfo
) -> str:
    return _hash(
        {
            "version": 2,
            "session_file": str(session_info.session_file),
            "session_mtime_ns": session_info.session_file.stat().st_mtime_ns,
            "period": float(args.orientation_period),
            "pca_dim": int(args.pca_dim),
            "pca_random_state": int(args.pca_random_state),
            "pca_whiten": not bool(args.no_pca_whiten),
            "moment_harmonics": int(args.moment_harmonics),
            "moment_ridge": float(args.moment_ridge),
            "covariance_grid_size": int(args.covariance_grid_size),
            "covariance_shrinkage": float(args.covariance_shrinkage),
            "residual_bins": int(args.residual_bins),
            "eigenvalue_floor_relative": float(args.eigenvalue_floor_relative),
            "half_split_seed": int(args.seed),
        }
    )


def _moment_cache_path(output_dir: Path, session_info: StringerSessionInfo) -> Path:
    return output_dir / "moments" / f"{session_info.session_file.stem}_moments.npz"


def _load_moment_cache(
    path: Path, signature: str
) -> tuple[PeriodicFourierMoments, StandardizedResidualBank, dict[str, np.ndarray]] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        if str(data["signature"].item()) != signature:
            return None
        moments = PeriodicFourierMoments(
            period=float(data["period"].item()),
            n_harmonics=int(data["n_harmonics"].item()),
            mean_coefficients=np.asarray(data["mean_coefficients"], dtype=np.float64),
            covariance_grid_centers=np.asarray(
                data["covariance_grid_centers"], dtype=np.float64
            ),
            covariance_grid=np.asarray(data["covariance_grid"], dtype=np.float64),
        )
        bank = StandardizedResidualBank(
            residuals=np.asarray(data["standardized_residuals"], dtype=np.float64),
            bin_ids=np.asarray(data["residual_bin_ids"], dtype=np.int64),
            n_bins=int(data["residual_n_bins"].item()),
            period=float(data["period"].item()),
            counts=np.asarray(data["residual_counts"], dtype=np.int64),
            mean_norms=np.asarray(data["residual_mean_norms"], dtype=np.float64),
            covariance_errors=np.asarray(
                data["residual_covariance_errors"], dtype=np.float64
            ),
        )
        auxiliary = {
            "theta_all": np.asarray(data["theta_all"], dtype=np.float64),
            "half_a_indices": np.asarray(data["half_a_indices"], dtype=np.int64),
            "half_b_indices": np.asarray(data["half_b_indices"], dtype=np.int64),
            "pca_explained_variance_ratio": np.asarray(
                data["pca_explained_variance_ratio"], dtype=np.float64
            ),
        }
    return moments, bank, auxiliary


def fit_or_load_moments(
    *,
    args: argparse.Namespace,
    session_info: StringerSessionInfo,
    session_index: int,
    output_dir: Path,
) -> tuple[PeriodicFourierMoments, StandardizedResidualBank, dict[str, np.ndarray]]:
    signature = _moment_signature(args, session_info)
    path = _moment_cache_path(output_dir, session_info)
    cached = None if bool(args.force_moments) else _load_moment_cache(path, signature)
    if cached is not None:
        print(f"[moments] cache hit {session_info.session_file.stem}", flush=True)
        return cached

    print(f"[moments] fitting shared PCA and moments {session_info.session_file.stem}", flush=True)
    session = load_stringer_session(
        session_info, orientation_period=float(args.orientation_period)
    )
    theta = np.asarray(session.grating_orientation, dtype=np.float64).reshape(-1)
    responses = np.asarray(session.neural_responses, dtype=np.float32)
    pca = PCA(
        n_components=int(args.pca_dim),
        whiten=not bool(args.no_pca_whiten),
        svd_solver="randomized",
        random_state=int(args.pca_random_state) + int(session_index),
    )
    projected = pca.fit_transform(responses).astype(np.float64, copy=False)
    moments = fit_periodic_fourier_moments(
        theta,
        projected,
        period=float(args.orientation_period),
        n_harmonics=int(args.moment_harmonics),
        relative_ridge=float(args.moment_ridge),
        covariance_grid_size=int(args.covariance_grid_size),
        covariance_shrinkage=float(args.covariance_shrinkage),
        eigenvalue_floor_relative=float(args.eigenvalue_floor_relative),
    )
    bank = fit_standardized_residual_bank(
        theta,
        projected,
        moments,
        n_bins=int(args.residual_bins),
        eigenvalue_floor_relative=float(args.eigenvalue_floor_relative),
    )
    half_a, half_b = stratified_half_split(
        theta,
        n_bins=int(args.residual_bins),
        period=float(args.orientation_period),
        seed=int(args.seed) + int(session_index),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        signature=np.asarray(signature),
        period=np.asarray(moments.period),
        n_harmonics=np.asarray(moments.n_harmonics, dtype=np.int64),
        mean_coefficients=moments.mean_coefficients,
        covariance_grid_centers=moments.covariance_grid_centers,
        covariance_grid=moments.covariance_grid,
        standardized_residuals=bank.residuals,
        residual_bin_ids=bank.bin_ids,
        residual_n_bins=np.asarray(bank.n_bins, dtype=np.int64),
        residual_counts=bank.counts,
        residual_mean_norms=bank.mean_norms,
        residual_covariance_errors=bank.covariance_errors,
        theta_all=theta,
        half_a_indices=half_a,
        half_b_indices=half_b,
        pca_explained_variance_ratio=np.asarray(
            pca.explained_variance_ratio_, dtype=np.float64
        ),
    )
    del projected, responses
    loaded = _load_moment_cache(path, signature)
    if loaded is None:
        raise RuntimeError(f"Failed to reload moment cache {path}.")
    return loaded


def _lambda_token(value: float) -> str:
    return f"{float(value):.3f}".replace(".", "p")


def _fit_signature(
    args: argparse.Namespace,
    *,
    session_key: str,
    half_label: str,
    lambda_value: float,
    repeat: int,
    moment_signature: str,
) -> str:
    return _hash(
        {
            "version": 1,
            "session_key": session_key,
            "half_label": half_label,
            "lambda": float(lambda_value),
            "repeat": int(repeat),
            "moment_signature": moment_signature,
            "train_frac": float(args.train_frac),
            "seed": int(args.seed),
            "theta_grid_size": int(args.theta_grid_size),
            "flow_config": asdict(flow_config(args)),
            "flow_orientation_encoding": FLOW_ORIENTATION_ENCODING_PERIODIC_RBF,
            "gkr_config": asdict(GKRConfig()),
        }
    )


def _fit_cache_path(
    output_dir: Path,
    *,
    session_key: str,
    half_label: str,
    lambda_value: float,
    repeat: int,
) -> Path:
    return (
        output_dir
        / "fits"
        / f"lambda_{_lambda_token(lambda_value)}"
        / f"repeat_{int(repeat):03d}"
        / f"{session_key}_half_{half_label}.npz"
    )


def fit_or_load_half(
    *,
    args: argparse.Namespace,
    device: torch.device,
    session_info: StringerSessionInfo,
    session_index: int,
    half_label: str,
    theta_half: np.ndarray,
    moments: PeriodicFourierMoments,
    bank: StandardizedResidualBank,
    moment_signature: str,
    theta_grid: np.ndarray,
    lambda_value: float,
    repeat: int,
    output_dir: Path,
) -> dict[str, np.ndarray]:
    session_key = session_info.session_file.stem
    signature = _fit_signature(
        args,
        session_key=session_key,
        half_label=half_label,
        lambda_value=lambda_value,
        repeat=repeat,
        moment_signature=moment_signature,
    )
    path = _fit_cache_path(
        output_dir,
        session_key=session_key,
        half_label=half_label,
        lambda_value=lambda_value,
        repeat=repeat,
    )
    if path.is_file() and not bool(args.force_fits):
        with np.load(path, allow_pickle=False) as data:
            if str(data["signature"].item()) == signature:
                print(
                    f"[fit] cache hit lambda={lambda_value:g} repeat={repeat} "
                    f"session={session_key} half={half_label}",
                    flush=True,
                )
                return {key: np.asarray(data[key]) for key in data.files}

    common_seed = (
        int(args.seed)
        + 1_000_003 * int(repeat)
        + 10_007 * int(session_index)
        + (0 if half_label == "A" else 1_009)
    )
    responses = sample_moment_matched_surrogate(
        theta_half,
        moments,
        bank,
        non_gaussian_weight=float(lambda_value),
        seed=common_seed,
    )
    train_indices, validation_indices = split_train_validation(
        theta_half.shape[0],
        train_frac=float(args.train_frac),
        seed=common_seed + 20_000,
    )
    bundle = make_shared_bundle(
        theta_all=theta_half,
        x_all=responses,
        train_idx=train_indices,
        validation_idx=validation_indices,
        meta={
            "dataset_family": "stringer_moment_matched_surrogate",
            "session_key": session_key,
            "half_label": half_label,
            "lambda": float(lambda_value),
            "repeat": int(repeat),
        },
    )
    flow_path = path.with_name(path.stem + "_flow.npz")
    print(
        f"[fit] flow lambda={lambda_value:g} repeat={repeat} "
        f"session={session_key} half={half_label} n={theta_half.shape[0]}",
        flush=True,
    )
    flow_curve, flow_metadata, _ = train_flow_linear_curve(
        bundle=bundle,
        theta_grid=theta_grid,
        period=float(args.orientation_period),
        flow_orientation_encoding=FLOW_ORIENTATION_ENCODING_PERIODIC_RBF,
        device=device,
        config=flow_config(args),
        seed=common_seed + 30_000,
        output_npz=flow_path,
    )

    print(
        f"[fit] GKR lambda={lambda_value:g} repeat={repeat} "
        f"session={session_key} half={half_label}",
        flush=True,
    )
    gkr_model = TorchGKR(
        n_input=1,
        n_output=int(args.pca_dim),
        circular_period=float(args.orientation_period),
        config=GKRConfig(),
        dtype=torch.float64,
        device=device,
        seed=common_seed + 40_000,
    )
    gkr_model.fit(responses[train_indices], theta_half[train_indices, None])
    mids = theta_midpoints(theta_grid)
    gkr_result = estimate_gkr_linear_fisher(
        gkr_model,
        mids,
        finite_difference_step=np.diff(theta_grid, axis=0),
        solve_jitter=1e-6,
    )
    del gkr_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        signature=np.asarray(signature),
        lambda_value=np.asarray(lambda_value),
        repeat=np.asarray(repeat, dtype=np.int64),
        session_index=np.asarray(session_index, dtype=np.int64),
        session_key=np.asarray(session_key),
        half_label=np.asarray(half_label),
        n_trials=np.asarray(theta_half.shape[0], dtype=np.int64),
        n_train=np.asarray(train_indices.size, dtype=np.int64),
        flow_linear_fisher=np.asarray(flow_curve, dtype=np.float64),
        gkr_linear_fisher=np.asarray(gkr_result.linear_fisher, dtype=np.float64),
        flow_train_losses=np.asarray(flow_metadata["train_losses"], dtype=np.float64),
        flow_validation_losses=np.asarray(flow_metadata["val_losses"], dtype=np.float64),
        flow_validation_monitor_losses=np.asarray(
            flow_metadata["val_monitor_losses"], dtype=np.float64
        ),
        flow_selected_epoch=np.asarray(flow_metadata["selected_epoch"], dtype=np.int64),
        flow_stopped_epoch=np.asarray(flow_metadata["stopped_epoch"], dtype=np.int64),
        gkr_mean_losses=np.asarray(gkr_result.mean_loss, dtype=np.float64),
        gkr_covariance_losses=np.asarray(gkr_result.covariance_loss, dtype=np.float64),
        surrogate_response_mean=np.asarray(responses.mean(axis=0), dtype=np.float64),
        surrogate_response_covariance=np.asarray(
            np.cov(responses, rowvar=False), dtype=np.float64
        ),
    )
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def summarize_results(
    *,
    lambda_values: list[float],
    estimates: dict[str, np.ndarray],
    ground_truth: np.ndarray,
    theta_mid: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, dict[str, np.ndarray]]]]:
    n_lambda, n_repeats = next(iter(estimates.values())).shape[:2]
    mae = {
        method: np.empty(
            (n_lambda, n_repeats, ground_truth.shape[0], 2), dtype=np.float64
        )
        for method in TOY_IDENTIFICATION_METHODS
    }
    identification = {
        method: {
            distance: {
                "top1": np.empty((n_lambda, n_repeats), dtype=np.float64),
                "mrr": np.empty((n_lambda, n_repeats), dtype=np.float64),
                "margins": np.empty(
                    (n_lambda, n_repeats, ground_truth.shape[0]), dtype=np.float64
                ),
            }
            for distance in DISTANCES
        }
        for method in TOY_IDENTIFICATION_METHODS
    }
    for lambda_index, _lambda in enumerate(lambda_values):
        for repeat in range(n_repeats):
            one = {
                method: estimates[method][lambda_index, repeat]
                for method in TOY_IDENTIFICATION_METHODS
            }
            _, summaries = evaluate_identification(one, theta_mid)
            for method in TOY_IDENTIFICATION_METHODS:
                mae[method][lambda_index, repeat] = np.moveaxis(
                    fisher_mae(
                        np.moveaxis(one[method], 1, 0),
                        ground_truth,
                    ),
                    0,
                    1,
                )
                for distance in DISTANCES:
                    summary = summaries[method][distance]
                    identification[method][distance]["top1"][lambda_index, repeat] = float(
                        summary["top1_accuracy"]
                    )
                    identification[method][distance]["mrr"][lambda_index, repeat] = float(
                        summary["mean_reciprocal_rank"]
                    )
                    identification[method][distance]["margins"][lambda_index, repeat] = np.asarray(
                        summary["correct_minus_best_wrong_margin"], dtype=np.float64
                    )
    return mae, identification


def _errorbar(
    axis: Any,
    x: np.ndarray,
    values: np.ndarray,
    *,
    method: str,
) -> None:
    flattened = values.reshape(values.shape[0], -1)
    mean = np.mean(flattened, axis=1)
    sem = np.std(flattened, axis=1, ddof=1) / np.sqrt(flattened.shape[1])
    axis.errorbar(
        x,
        mean,
        yerr=sem,
        color=METHOD_COLORS[method],
        marker="o" if method == METHOD_FLOW else "^",
        linewidth=2.0,
        capsize=3,
        label=METHOD_LABELS[method],
    )


def plot_sweep(
    *,
    lambda_values: list[float],
    mae: dict[str, np.ndarray],
    identification: dict[str, dict[str, dict[str, np.ndarray]]],
    n_sessions: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "axes.grid": False,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.5), constrained_layout=True)
    x = np.asarray(lambda_values, dtype=np.float64)
    for method in TOY_IDENTIFICATION_METHODS:
        _errorbar(axes[0], x, mae[method], method=method)
    axes[0].set_title("Known-truth accuracy")
    axes[0].set_xlabel(r"Non-Gaussian weight $\lambda$")
    axes[0].set_ylabel("Mean absolute error")
    axes[0].legend(frameon=False)

    chance = 1.0 / float(n_sessions)
    for axis, distance in zip(axes[1:], DISTANCES, strict=True):
        axis.axhline(chance, color="0.5", linestyle="--", linewidth=1.3)
        for method in TOY_IDENTIFICATION_METHODS:
            values = identification[method][distance]["top1"]
            mean = np.mean(values, axis=1)
            std = np.std(values, axis=1, ddof=1) if values.shape[1] > 1 else np.zeros_like(mean)
            axis.errorbar(
                x,
                mean,
                yerr=std,
                color=METHOD_COLORS[method],
                marker="o" if method == METHOD_FLOW else "^",
                linewidth=2.0,
                capsize=3,
                label=METHOD_LABELS[method],
            )
        axis.set_title(DISTANCE_LABELS[distance])
        axis.set_xlabel(r"Non-Gaussian weight $\lambda$")
        axis.set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Top-1 identification")

    for axis in axes:
        axis.set_xticks(x)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(1.8)
        axis.spines["bottom"].set_linewidth(1.8)
        axis.tick_params(width=1.8)
    png = output_dir / "stringer_nongaussianity_sweep.png"
    svg = output_dir / "stringer_nongaussianity_sweep.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


def plot_example_curves(
    *,
    lambda_values: list[float],
    theta_mid: np.ndarray,
    ground_truth: np.ndarray,
    estimates: dict[str, np.ndarray],
    output_dir: Path,
) -> tuple[Path, Path]:
    selected = [0] if len(lambda_values) == 1 else [0, len(lambda_values) - 1]
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 11,
            "axes.grid": False,
        }
    )
    fig, axes = plt.subplots(1, len(selected), figsize=(4.0 * len(selected), 3.5), squeeze=False)
    theta = np.asarray(theta_mid, dtype=np.float64).reshape(-1)
    for panel_index, lambda_index in enumerate(selected):
        axis = axes[0, panel_index]
        axis.plot(theta, ground_truth[0], color="black", linestyle="--", linewidth=2.2, label="Surrogate truth")
        for method in TOY_IDENTIFICATION_METHODS:
            color = METHOD_COLORS[method]
            short = "Flow" if method == METHOD_FLOW else "GKR"
            axis.plot(theta, estimates[method][lambda_index, 0, 0, 0], color=color, linewidth=2.0, label=f"{short}, half A")
            axis.plot(theta, estimates[method][lambda_index, 0, 0, 1], color=color, linewidth=1.8, linestyle=":", label=f"{short}, half B")
        axis.set_title(rf"$\lambda={lambda_values[lambda_index]:g}$")
        axis.set_xlabel(r"Orientation $\theta$")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(1.8)
        axis.spines["bottom"].set_linewidth(1.8)
        axis.tick_params(width=1.8)
    axes[0, 0].set_ylabel("Linear Fisher information")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    png = output_dir / "stringer_nongaussianity_example_curves.png"
    svg = output_dir / "stringer_nongaussianity_example_curves.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = require_device(str(args.device))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions = list_stringer_sessions("gratings_static", data_dir=args.data_dir)[
        : int(args.max_sessions)
    ]
    if len(sessions) < 2:
        raise ValueError("Need at least two available gratings_static sessions.")
    theta_grid = theta_grid_periodic(
        float(args.orientation_period), int(args.theta_grid_size)
    )
    mids = theta_midpoints(theta_grid)
    n_curves = mids.shape[0]
    lambda_values = [float(value) for value in args.lambda_list]

    moment_bundles = []
    ground_truth = np.empty((len(sessions), n_curves), dtype=np.float64)
    for session_index, session_info in enumerate(sessions):
        moments, bank, auxiliary = fit_or_load_moments(
            args=args,
            session_info=session_info,
            session_index=session_index,
            output_dir=output_dir,
        )
        ground_truth[session_index] = moments.linear_fisher(mids, solve_jitter=1e-6)
        moment_bundles.append((moments, bank, auxiliary))

    estimates = {
        method: np.empty(
            (len(lambda_values), int(args.repeats), len(sessions), 2, n_curves),
            dtype=np.float64,
        )
        for method in TOY_IDENTIFICATION_METHODS
    }
    selected_epochs = np.empty(
        (len(lambda_values), int(args.repeats), len(sessions), 2), dtype=np.int64
    )
    for lambda_index, lambda_value in enumerate(lambda_values):
        for repeat in range(int(args.repeats)):
            for session_index, session_info in enumerate(sessions):
                moments, bank, auxiliary = moment_bundles[session_index]
                moment_signature = _moment_signature(args, session_info)
                theta_all = auxiliary["theta_all"]
                for half_index, (half_label, index_key) in enumerate(
                    (("A", "half_a_indices"), ("B", "half_b_indices"))
                ):
                    theta_half = np.asarray(
                        theta_all[auxiliary[index_key]], dtype=np.float64
                    )
                    result = fit_or_load_half(
                        args=args,
                        device=device,
                        session_info=session_info,
                        session_index=session_index,
                        half_label=half_label,
                        theta_half=theta_half,
                        moments=moments,
                        bank=bank,
                        moment_signature=moment_signature,
                        theta_grid=theta_grid,
                        lambda_value=lambda_value,
                        repeat=repeat,
                        output_dir=output_dir,
                    )
                    estimates[METHOD_FLOW][lambda_index, repeat, session_index, half_index] = result[
                        "flow_linear_fisher"
                    ]
                    estimates[METHOD_GKR][lambda_index, repeat, session_index, half_index] = result[
                        "gkr_linear_fisher"
                    ]
                    selected_epochs[lambda_index, repeat, session_index, half_index] = int(
                        result["flow_selected_epoch"].item()
                    )

    mae, identification = summarize_results(
        lambda_values=lambda_values,
        estimates=estimates,
        ground_truth=ground_truth,
        theta_mid=mids,
    )
    sweep_png, sweep_svg = plot_sweep(
        lambda_values=lambda_values,
        mae=mae,
        identification=identification,
        n_sessions=len(sessions),
        output_dir=output_dir,
    )
    curves_png, curves_svg = plot_example_curves(
        lambda_values=lambda_values,
        theta_mid=mids,
        ground_truth=ground_truth,
        estimates=estimates,
        output_dir=output_dir,
    )

    result_path = output_dir / "stringer_nongaussianity_results.npz"
    np.savez_compressed(
        result_path,
        lambda_values=np.asarray(lambda_values, dtype=np.float64),
        theta_grid=theta_grid,
        theta_midpoints=mids,
        session_keys=np.asarray([session.session_file.stem for session in sessions]),
        ground_truth_linear_fisher=ground_truth,
        flow_linear_fisher=estimates[METHOD_FLOW],
        gkr_linear_fisher=estimates[METHOD_GKR],
        flow_mae=mae[METHOD_FLOW],
        gkr_mae=mae[METHOD_GKR],
        flow_selected_epochs=selected_epochs,
        **{
            f"{method.lower().replace(' ', '_')}_{distance}_{metric}": values
            for method in TOY_IDENTIFICATION_METHODS
            for distance in DISTANCES
            for metric, values in identification[method][distance].items()
        },
    )
    summary = {
        "hypothesis": "Higher-order Stringer non-Gaussianity changes the relative performance of flow matching and GKR.",
        "intervention": "moment-matched Gaussian-to-whitened-empirical-residual interpolation",
        "lambda_values": lambda_values,
        "n_repeats": int(args.repeats),
        "session_keys": [session.session_file.stem for session in sessions],
        "representation": {
            "scope": "one shared label-blind PCA fit per session before surrogate generation",
            "pca_dim": int(args.pca_dim),
            "whiten": not bool(args.no_pca_whiten),
        },
        "moment_model": {
            "estimator": "periodic Fourier ridge regression of mean and residual outer products",
            "harmonics": int(args.moment_harmonics),
            "ridge": float(args.moment_ridge),
            "covariance_grid_size": int(args.covariance_grid_size),
            "covariance_shrinkage_to_global": float(args.covariance_shrinkage),
            "residual_bins": int(args.residual_bins),
        },
        "moment_diagnostics": [
            {
                "session_key": session.session_file.stem,
                "pca_explained_variance_ratio_sum": float(
                    np.sum(auxiliary["pca_explained_variance_ratio"])
                ),
                "conditional_covariance_min_eigenvalue": float(
                    np.min(np.linalg.eigvalsh(moments.covariance_grid))
                ),
                "conditional_covariance_max_eigenvalue": float(
                    np.max(np.linalg.eigvalsh(moments.covariance_grid))
                ),
                "standardized_residual_min_bin_count": int(np.min(bank.counts)),
                "standardized_residual_max_mean_norm": float(
                    np.max(bank.mean_norms)
                ),
                "standardized_residual_max_covariance_error": float(
                    np.max(bank.covariance_errors)
                ),
                "ground_truth_fisher_min": float(np.min(ground_truth[index])),
                "ground_truth_fisher_max": float(np.max(ground_truth[index])),
            }
            for index, (session, (moments, bank, auxiliary)) in enumerate(
                zip(sessions, moment_bundles, strict=True)
            )
        ],
        "flow_config": {
            **asdict(flow_config(args)),
            "orientation_encoding": FLOW_ORIENTATION_ENCODING_PERIODIC_RBF,
            "fixed_validation_paths": 10,
            "selected_epochs": selected_epochs.tolist(),
        },
        "gkr_config": asdict(GKRConfig()),
        "mae": {
            method: [
                {
                    "lambda": lambda_values[index],
                    "mean": float(np.mean(values[index])),
                    "std": float(np.std(values[index], ddof=1)),
                }
                for index in range(len(lambda_values))
            ]
            for method, values in mae.items()
        },
        "identification": {
            method: {
                distance: {
                    metric: np.asarray(values).tolist()
                    for metric, values in metrics.items()
                }
                for distance, metrics in distances.items()
            }
            for method, distances in identification.items()
        },
        "artifacts": {
            "results_npz": str(result_path),
            "sweep_png": str(sweep_png),
            "sweep_svg": str(sweep_svg),
            "example_curves_png": str(curves_png),
            "example_curves_svg": str(curves_svg),
        },
    }
    summary_path = output_dir / "stringer_nongaussianity_summary.json"
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["mae"], indent=2), flush=True)
    for method in TOY_IDENTIFICATION_METHODS:
        print(
            method,
            {
                distance: identification[method][distance]["top1"].tolist()
                for distance in DISTANCES
            },
            flush=True,
        )
    print(f"sweep_png: {sweep_png}", flush=True)
    print(f"summary_json: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
