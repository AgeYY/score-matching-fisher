#!/usr/bin/env python3
"""Compare ground-truth error and split-half identification on toy data."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.continuous_fisher_comparison import native_linear_fisher_curve
from fisher.data import RANDAMP_SQRTD_VAR_MU_LAW_ADDITIVE
from fisher.dataset_family_recipes import family_recipe_dict
from fisher.flow_matching_skl import (
    build_flow_skl_model,
    estimate_affine_mixed_symmetric_kl_fisher,
    train_flow_skl_model,
)
from fisher.gkr import GKRConfig, TorchGKR, estimate_gkr_linear_fisher
from fisher.shared_fisher_est import build_dataset_from_meta, require_device
from fisher.stringer_session_identification import (
    DISTANCE_AREA_L2,
    DISTANCE_PRIMARY,
    DISTANCE_RMSE,
    split_train_validation,
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


DISTANCE_LABELS = {
    DISTANCE_PRIMARY: "Log correlation",
    DISTANCE_AREA_L2: "Area L2",
    DISTANCE_RMSE: "Raw RMSE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(DATA_DIR) / "toy_fisher_evaluation_criterion")
    parser.add_argument(
        "--dataset-family",
        choices=("randamp_gaussian_sqrtd", "cosine_gmm"),
        default="randamp_gaussian_sqrtd",
    )
    parser.add_argument("--n-sessions", type=int, default=6)
    parser.add_argument("--n-per-half", type=int, default=1000)
    parser.add_argument("--x-dim", type=int, default=50)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--theta-low", type=float, default=-6.0)
    parser.add_argument("--theta-high", type=float, default=6.0)
    parser.add_argument("--theta-grid-size", type=int, default=61)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_MAX_EPOCHS)
    parser.add_argument("--early-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--theta-rbf-num-centers", type=int, default=8)
    parser.add_argument("--theta-rbf-bandwidth", type=float, default=None)
    parser.add_argument("--ode-steps", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if int(args.n_sessions) < 2:
        raise ValueError("--n-sessions must be at least 2.")
    if int(args.n_per_half) < 2:
        raise ValueError("--n-per-half must be at least 2.")
    if int(args.x_dim) < 1:
        raise ValueError("--x-dim must be positive.")
    if str(args.dataset_family) == "cosine_gmm" and int(args.x_dim) < 2:
        raise ValueError("--x-dim must be at least 2 for cosine_gmm.")
    if not 0.0 < float(args.train_frac) < 1.0:
        raise ValueError("--train-frac must be in (0, 1).")
    if not float(args.theta_low) < float(args.theta_high):
        raise ValueError("--theta-low must be smaller than --theta-high.")
    if int(args.theta_grid_size) < 3:
        raise ValueError("--theta-grid-size must be at least 3.")


def population_meta(args: argparse.Namespace, session_index: int) -> dict[str, Any]:
    family = str(args.dataset_family)
    population_seed = int(args.seed) + 100_003 * int(session_index)
    recipe = family_recipe_dict(family)
    if family == "cosine_gmm":
        # cosine_gmm's seed controls sampling only. Draw fixed mixture
        # parameters so each synthetic session represents a distinct population.
        rng = np.random.default_rng(population_seed)
        recipe.update(
            {
                "gmm_sep_scale": float(recipe["gmm_sep_scale"]) * float(rng.uniform(0.75, 1.25)),
                "gmm_sep_phase": float(recipe["gmm_sep_phase"]) + float(rng.uniform(-np.pi, np.pi)),
                "gmm_mix_logit_scale": float(recipe["gmm_mix_logit_scale"])
                * float(rng.uniform(0.75, 1.25)),
                "gmm_mix_bias": float(recipe["gmm_mix_bias"]) + float(rng.uniform(-0.35, 0.35)),
                "gmm_mix_phase": float(recipe["gmm_mix_phase"]) + float(rng.uniform(-np.pi, np.pi)),
            }
        )
    meta = {
        "dataset_family": family,
        "seed": population_seed,
        "theta_low": float(args.theta_low),
        "theta_high": float(args.theta_high),
        "x_dim": int(args.x_dim),
        **recipe,
    }
    if family == "randamp_gaussian_sqrtd":
        meta["randamp_sqrtd_obs_var_mu_law"] = RANDAMP_SQRTD_VAR_MU_LAW_ADDITIVE
    return meta


def population_linear_fisher_curve(theta: np.ndarray, dataset: Any) -> np.ndarray:
    """Return exact marginal linear Fisher information for either toy family."""
    if not (hasattr(dataset, "_mix_weight") and hasattr(dataset, "component_means")):
        return native_linear_fisher_curve(theta, dataset)
    values = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
    probability, probability_derivative = dataset._mix_weight(values)
    mean1, mean2 = dataset.component_means(values)
    covariance1, covariance2, _, _ = dataset.component_covariances(values)
    base_derivative = dataset.tuning_curve_derivative(values)
    _, separation_derivative = dataset._separation(values)
    derivative1 = base_derivative + separation_derivative
    derivative2 = base_derivative - separation_derivative
    p = probability[:, None]
    mean = p * mean1 + (1.0 - p) * mean2
    mean_derivative = (
        probability_derivative[:, None] * (mean1 - mean2)
        + p * derivative1
        + (1.0 - p) * derivative2
    )
    delta1 = mean1 - mean
    delta2 = mean2 - mean
    covariance = (
        probability[:, None, None]
        * (covariance1 + np.einsum("ni,nj->nij", delta1, delta1))
        + (1.0 - probability)[:, None, None]
        * (covariance2 + np.einsum("ni,nj->nij", delta2, delta2))
    )
    inverse = np.linalg.inv(covariance)
    return np.einsum("bi,bij,bj->b", mean_derivative, inverse, mean_derivative).astype(np.float64)


def _signature(args: argparse.Namespace, session_index: int, half_index: int) -> str:
    payload = {
        "version": 1,
        "session_index": int(session_index),
        "half_index": int(half_index),
        **{
            key: value
            for key, value in vars(args).items()
            if key not in {"output_dir", "force", "device"}
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _set_seed(seed: int, device: torch.device) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _fit_half(
    *,
    args: argparse.Namespace,
    device: torch.device,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    theta_grid: np.ndarray,
    session_index: int,
    half_index: int,
    cache_path: Path,
) -> dict[str, Any]:
    signature = _signature(args, session_index, half_index)
    if cache_path.is_file() and not bool(args.force):
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_signature = str(cached["signature"].item())
            if cached_signature == signature:
                return {key: np.asarray(cached[key]) for key in cached.files}

    fit_seed = int(args.seed) + 10_000 * int(session_index) + 1_000 * int(half_index)
    train_idx, validation_idx = split_train_validation(
        int(theta_all.shape[0]), train_frac=float(args.train_frac), seed=fit_seed
    )
    theta_train = np.asarray(theta_all[train_idx], dtype=np.float64).reshape(-1, 1)
    x_train = np.asarray(x_all[train_idx], dtype=np.float64)
    theta_validation = np.asarray(theta_all[validation_idx], dtype=np.float64).reshape(-1, 1)
    x_validation = np.asarray(x_all[validation_idx], dtype=np.float64)

    _set_seed(fit_seed, device)
    flow_model = build_flow_skl_model(
        velocity_family="condition_affine",
        theta_dim=1,
        x_dim=int(args.x_dim),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        quadrature_steps=64,
        path_schedule="cosine",
        divergence_estimator="exact",
        theta_embedding="gaussian_rbf",
        theta_rbf_num_centers=int(args.theta_rbf_num_centers),
        theta_rbf_lower=float(args.theta_low),
        theta_rbf_upper=float(args.theta_high),
        theta_rbf_bandwidth=args.theta_rbf_bandwidth,
    ).to(device)
    flow_meta = train_flow_skl_model(
        model=flow_model,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_validation,
        x_val=x_validation,
        device=device,
        velocity_family="condition_affine",
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
        checkpoint_selection="best",
        best_checkpoint_metric="flow_matching",
        fixed_validation=True,
        fixed_validation_paths=10,
        validation_seed=fit_seed + 50_000,
    )
    flow_result = estimate_affine_mixed_symmetric_kl_fisher(
        model=flow_model,
        theta_all=theta_grid,
        device=device,
        ridge=1e-6,
        ode_steps=int(args.ode_steps),
    )
    flow_curve = np.asarray(flow_result["fisher"], dtype=np.float64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = cache_path.with_suffix(".pt")
    torch.save(
        {key: value.detach().cpu() for key, value in flow_model.state_dict().items()},
        model_path,
    )
    del flow_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    gkr_config = GKRConfig()
    gkr_model = TorchGKR(
        n_input=1,
        n_output=int(args.x_dim),
        config=gkr_config,
        dtype=torch.float64,
        device=device,
        seed=fit_seed,
    )
    gkr_model.fit(x_train, theta_train)
    theta_midpoints = 0.5 * (theta_grid[:-1] + theta_grid[1:])
    gkr_result = estimate_gkr_linear_fisher(
        gkr_model,
        theta_midpoints,
        finite_difference_step=np.diff(theta_grid, axis=0),
        solve_jitter=1e-6,
    )
    gkr_curve = np.asarray(gkr_result.linear_fisher, dtype=np.float64)
    del gkr_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    np.savez_compressed(
        cache_path,
        signature=np.asarray(signature),
        session_index=np.asarray(session_index, dtype=np.int64),
        half_index=np.asarray(half_index, dtype=np.int64),
        train_indices=train_idx,
        validation_indices=validation_idx,
        flow_linear_fisher=flow_curve,
        gkr_linear_fisher=gkr_curve,
        flow_train_losses=np.asarray(flow_meta["train_losses"], dtype=np.float64),
        flow_validation_losses=np.asarray(flow_meta["val_losses"], dtype=np.float64),
        flow_validation_monitor_losses=np.asarray(flow_meta["val_monitor_losses"], dtype=np.float64),
        flow_selected_epoch=np.asarray(flow_meta["selected_epoch"], dtype=np.int64),
        flow_stopped_epoch=np.asarray(flow_meta["stopped_epoch"], dtype=np.int64),
        gkr_mean_losses=np.asarray(gkr_result.mean_loss, dtype=np.float64),
        gkr_covariance_losses=np.asarray(gkr_result.covariance_loss, dtype=np.float64),
        flow_model_path=np.asarray(str(model_path)),
        gkr_config_json=np.asarray(json.dumps(asdict(gkr_config), sort_keys=True)),
    )
    with np.load(cache_path, allow_pickle=False) as cached:
        return {key: np.asarray(cached[key]) for key in cached.files}


def _plot(
    *,
    theta_midpoints: np.ndarray,
    ground_truth: np.ndarray,
    estimates: dict[str, np.ndarray],
    mae: dict[str, np.ndarray],
    summaries: dict[str, dict[str, dict[str, object]]],
    output_dir: Path,
    artifact_stem: str,
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
    theta = np.asarray(theta_midpoints, dtype=np.float64).reshape(-1)

    curve_axis = axes[0]
    curve_axis.plot(theta, ground_truth[0], color="black", linestyle="--", linewidth=2.2, label="Ground truth")
    styles = {
        METHOD_FLOW: ("C0", "Flow"),
        METHOD_GKR: ("C2", "GKR"),
    }
    for method in TOY_IDENTIFICATION_METHODS:
        color, label = styles[method]
        curve_axis.plot(theta, estimates[method][0, 0], color=color, linewidth=2.0, label=f"{label}, half A")
        curve_axis.plot(theta, estimates[method][0, 1], color=color, linewidth=1.8, linestyle=":", label=f"{label}, half B")
    curve_axis.set_title("Example session")
    curve_axis.set_xlabel(r"$\theta$")
    curve_axis.set_ylabel("Linear Fisher information")
    curve_axis.legend(frameon=False, fontsize=10)

    mae_axis = axes[1]
    rng = np.random.default_rng(91)
    for method_index, method in enumerate(TOY_IDENTIFICATION_METHODS):
        values = np.asarray(mae[method], dtype=np.float64).reshape(-1)
        jitter = rng.uniform(-0.08, 0.08, size=values.size)
        color = styles[method][0]
        mae_axis.scatter(
            np.full(values.size, method_index) + jitter,
            values,
            color=color,
            alpha=0.75,
            s=30,
        )
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        mae_axis.errorbar(method_index, mean, yerr=sem, color="black", marker="_", markersize=20, capsize=4, linewidth=2.0)
    mae_axis.set_xticks(range(len(TOY_IDENTIFICATION_METHODS)), ["Flow", "GKR"])
    mae_axis.set_title("Known-truth accuracy")
    mae_axis.set_ylabel("Mean absolute error")

    top1_axis = axes[2]
    distances = (DISTANCE_PRIMARY, DISTANCE_AREA_L2, DISTANCE_RMSE)
    x_positions = np.arange(len(distances), dtype=np.float64)
    width = 0.34
    for method_index, method in enumerate(TOY_IDENTIFICATION_METHODS):
        values = [float(summaries[method][distance]["top1_accuracy"]) for distance in distances]
        top1_axis.bar(
            x_positions + (method_index - 0.5) * width,
            values,
            width=width,
            color=styles[method][0],
            label=styles[method][1],
        )
    top1_axis.set_xticks(
        x_positions,
        [DISTANCE_LABELS[distance] for distance in distances],
        rotation=22,
        ha="right",
    )
    top1_axis.set_ylim(0.0, 1.05)
    top1_axis.set_title("Session identification")
    top1_axis.set_ylabel("Top-1 accuracy")
    top1_axis.legend(frameon=False)

    margin_axis = axes[3]
    session_positions = np.arange(1, ground_truth.shape[0] + 1)
    margin_axis.axhline(0.0, color="0.45", linestyle="--", linewidth=1.4)
    for method in TOY_IDENTIFICATION_METHODS:
        margins = np.asarray(
            summaries[method][DISTANCE_PRIMARY]["correct_minus_best_wrong_margin"],
            dtype=np.float64,
        )
        margin_axis.plot(
            session_positions,
            margins,
            color=styles[method][0],
            marker="o" if method == METHOD_FLOW else "^",
            linewidth=2.0,
            label=styles[method][1],
        )
    margin_axis.set_xticks(session_positions)
    margin_axis.set_title("Log-correlation margin")
    margin_axis.set_xlabel("Synthetic session")
    margin_axis.set_ylabel("Best wrong distance minus correct")
    margin_axis.legend(frameon=False)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(1.8)
        axis.spines["bottom"].set_linewidth(1.8)
        axis.tick_params(width=1.8)
    png = output_dir / f"{artifact_stem}.png"
    svg = output_dir / f"{artifact_stem}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


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


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = require_device(str(args.device))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    theta_grid = np.linspace(
        float(args.theta_low),
        float(args.theta_high),
        int(args.theta_grid_size),
        dtype=np.float64,
    ).reshape(-1, 1)
    theta_midpoints = 0.5 * (theta_grid[:-1] + theta_grid[1:])

    ground_truth = np.empty((int(args.n_sessions), theta_midpoints.shape[0]), dtype=np.float64)
    estimates = {
        method: np.empty(
            (int(args.n_sessions), 2, theta_midpoints.shape[0]), dtype=np.float64
        )
        for method in TOY_IDENTIFICATION_METHODS
    }
    selected_epochs = np.empty((int(args.n_sessions), 2), dtype=np.int64)
    population_metadata: list[dict[str, Any]] = []

    for session_index in range(int(args.n_sessions)):
        meta = population_meta(args, session_index)
        population_metadata.append(meta)
        population = build_dataset_from_meta(meta)
        ground_truth[session_index] = population_linear_fisher_curve(theta_midpoints, population)
        halves = [population.sample_joint(int(args.n_per_half)) for _ in range(2)]
        for half_index, (theta_all, x_all) in enumerate(halves):
            print(
                f"[fit] session={session_index + 1}/{args.n_sessions} "
                f"half={'AB'[half_index]} device={device}",
                flush=True,
            )
            cache_path = output_dir / "fits" / f"session_{session_index:02d}_half_{'ab'[half_index]}.npz"
            result = _fit_half(
                args=args,
                device=device,
                theta_all=theta_all,
                x_all=x_all,
                theta_grid=theta_grid,
                session_index=session_index,
                half_index=half_index,
                cache_path=cache_path,
            )
            estimates[METHOD_FLOW][session_index, half_index] = result["flow_linear_fisher"]
            estimates[METHOD_GKR][session_index, half_index] = result["gkr_linear_fisher"]
            selected_epochs[session_index, half_index] = int(result["flow_selected_epoch"].item())

    mae = {
        method: fisher_mae(np.moveaxis(values, 1, 0), ground_truth)
        for method, values in estimates.items()
    }
    matrices, summaries = evaluate_identification(estimates, theta_midpoints)
    artifact_stem = (
        "toy_gaussian_fisher_dual_criterion"
        if str(args.dataset_family) == "randamp_gaussian_sqrtd"
        else "toy_cosine_gmm_fisher_dual_criterion"
    )
    figure_png, figure_svg = _plot(
        theta_midpoints=theta_midpoints,
        ground_truth=ground_truth,
        estimates=estimates,
        mae=mae,
        summaries=summaries,
        output_dir=output_dir,
        artifact_stem=artifact_stem,
    )

    results_path = output_dir / f"{artifact_stem}_results.npz"
    np.savez_compressed(
        results_path,
        theta_grid=theta_grid,
        theta_midpoints=theta_midpoints,
        ground_truth_linear_fisher=ground_truth,
        flow_linear_fisher=estimates[METHOD_FLOW],
        gkr_linear_fisher=estimates[METHOD_GKR],
        flow_mae=mae[METHOD_FLOW],
        gkr_mae=mae[METHOD_GKR],
        flow_selected_epochs=selected_epochs,
        **{
            f"{method.lower().replace(' ', '_')}_{distance}_matrix": matrix
            for method, method_matrices in matrices.items()
            for distance, matrix in method_matrices.items()
        },
    )
    summary = {
        "hypothesis": (
            "Ground-truth Fisher error and split-half session identification may rank "
            "the same estimators differently."
        ),
        "dataset_family": str(args.dataset_family),
        "x_dim": int(args.x_dim),
        "n_sessions": int(args.n_sessions),
        "n_per_half": int(args.n_per_half),
        "train_fraction_per_half": float(args.train_frac),
        "population_seeds": [int(population_meta(args, index)["seed"]) for index in range(int(args.n_sessions))],
        "population_metadata": population_metadata,
        "flow": {
            "velocity_family": "condition_affine",
            "theta_embedding": "gaussian_rbf",
            "theta_rbf_num_centers": int(args.theta_rbf_num_centers),
            "epochs": int(args.epochs),
            "early_stopping_patience": int(args.early_patience),
            "learning_rate": float(args.lr),
            "fixed_validation_paths": 10,
            "selected_epochs": selected_epochs.tolist(),
        },
        "gkr": asdict(GKRConfig()),
        "ground_truth_mae": {
            method: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "by_half_and_session": np.asarray(values).tolist(),
            }
            for method, values in mae.items()
        },
        "identification": _json_ready(summaries),
        "figure_png": str(figure_png),
        "figure_svg": str(figure_svg),
        "results_npz": str(results_path),
    }
    summary_path = output_dir / f"{artifact_stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["ground_truth_mae"], indent=2), flush=True)
    print(json.dumps(summary["identification"], indent=2), flush=True)
    print(f"figure_png: {figure_png}", flush=True)
    print(f"summary_json: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
