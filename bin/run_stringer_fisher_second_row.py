#!/usr/bin/env python3
"""Fit Stringer Fisher estimators and create standalone comparison figures."""

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
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.fisher_validation import (
    decoder_directions,
    evaluate_windowed_decoders,
    fit_cross_fitted_ole_direction_estimator,
    fit_flow_direction_estimator,
    fit_gkr_direction_estimator,
    gkr_checkpoint,
    stratified_disjoint_subset_indices,
)
from fisher.flow_matching_skl import (
    DEFAULT_AFFINE_COVARIANCE_ODE_STEPS,
    build_flow_skl_model,
    estimate_adjacent_model_jeffreys_fisher,
    train_flow_skl_model,
)
from fisher.optimal_linear_estimator import optimal_linear_estimator
from fisher.shared_fisher_est import require_device
from fisher.stringer_dataset import list_stringer_sessions, load_stringer_session
from fisher.stringer_session_identification import encode_flow_orientation
from fisher.tre_distance import TREDensityRatioConfig, train_and_estimate_binned_tre_fisher
from global_setting import (
    DATA_DIR,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_TRAINING_MAX_EPOCHS,
)

PERIOD = float(np.pi)
LINEAR_METHOD_KEYS = ("gkr", "bin_lw", "ole_crossfit", "affine_flow")
LINEAR_METHOD_TITLES = ("GKR", "Binning + LW", "OLE (crossfit)", "Affine Flow")
FULL_METHOD_KEYS = ("unconstrained_flow", "tre8")
FULL_METHOD_TITLES = ("Unconstrained Flow", "TRE-8")
BAR_LABELS = ("GKR", "Bin+LW", "OLE", "Affine")
BAR_COLORS = ("C2", "C1", "C4", "C0")


def _csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device", help="Required unless --aggregate-only is used.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "stringer_fisher_test50_pca82_all_sessions",
    )
    parser.add_argument("--session-indices", type=_csv_ints, default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument(
        "--gkr-curve-root",
        type=Path,
        default=None,
        help="Optional six-session log-lambda GKR curve tree used during aggregation.",
    )

    parser.add_argument("--pca-dim", type=int, default=82)
    parser.add_argument("--theta-grid-size", type=int, default=17)
    parser.add_argument("--train-fraction", type=float, default=0.4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_MAX_EPOCHS)
    parser.add_argument(
        "--early-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument(
        "--affine-ode-steps",
        type=int,
        default=DEFAULT_AFFINE_COVARIANCE_ODE_STEPS,
    )
    parser.add_argument("--full-ode-steps", type=int, default=32)
    parser.add_argument("--full-mc-samples", type=int, default=4096)
    parser.add_argument("--full-hutchinson-probes", type=int, default=4)
    parser.add_argument("--full-likelihood-batch-size", type=int, default=1024)

    parser.add_argument("--ole-crossfit-folds", type=int, default=5)
    parser.add_argument("--ole-min-endpoint-samples", type=int, default=8)
    parser.add_argument("--tre-num-bridges", type=int, default=8)
    parser.add_argument("--tre-hidden-dim", type=int, default=128)
    parser.add_argument("--tre-depth", type=int, default=3)
    parser.add_argument("--tre-batch-size", type=int, default=512)
    parser.add_argument("--tre-lr", type=float, default=1e-3)
    parser.add_argument("--tre-validation-pairs", type=int, default=2048)
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


def _periodic_distance(theta: np.ndarray, center: float) -> np.ndarray:
    delta = np.abs(np.asarray(theta, dtype=np.float64).reshape(-1) - float(center))
    wrapped = np.mod(delta, PERIOD)
    return np.minimum(wrapped, PERIOD - wrapped)


def _endpoint_indices(
    theta: np.ndarray,
    center: float,
    *,
    radius: float,
    min_samples: int,
) -> np.ndarray:
    distance = _periodic_distance(theta, float(center))
    index = np.flatnonzero(distance <= float(radius) + 1e-12)
    if index.size < int(min_samples):
        index = np.argsort(distance, kind="mergesort")[: int(min_samples)]
    return np.asarray(index, dtype=np.int64)


def exact_stratified_fit_validation_test_indices(
    theta: np.ndarray,
    *,
    train_fraction: float,
    validation_fraction: float,
    n_strata: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create an exact-size stratified split with half the observations held out."""

    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    test_fraction = 1.0 - float(train_fraction) - float(validation_fraction)
    if not np.isclose(test_fraction, 0.5):
        raise ValueError("This split helper requires a 50% test fraction.")
    n_test = int(np.floor(test_fraction * values.size))
    test = stratified_disjoint_subset_indices(
        values,
        n_test,
        n_subsets=1,
        n_strata=int(n_strata),
        seed=int(seed),
        period=PERIOD,
    )[0]
    fit = np.setdiff1d(np.arange(values.size, dtype=np.int64), test)
    conditional_validation_fraction = float(validation_fraction) / (
        float(train_fraction) + float(validation_fraction)
    )
    n_validation = int(round(conditional_validation_fraction * fit.size))
    validation_local = stratified_disjoint_subset_indices(
        values[fit],
        n_validation,
        n_subsets=1,
        n_strata=int(n_strata),
        seed=int(seed) + 10_000,
        period=PERIOD,
    )[0]
    validation = fit[validation_local]
    train = np.setdiff1d(fit, validation)
    if (
        np.intersect1d(train, validation).size
        or np.intersect1d(train, test).size
        or np.intersect1d(validation, test).size
        or np.unique(np.concatenate([train, validation, test])).size != values.size
    ):
        raise RuntimeError("The exact stratified split must be exhaustive and disjoint.")
    return train, validation, test


def fit_binned_lw_direction_estimator(
    *,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_grid: np.ndarray,
    min_endpoint_samples: int = 8,
) -> dict[str, np.ndarray]:
    """Fit local endpoint Gaussian moments with Ledoit-Wolf covariance."""

    theta = np.asarray(theta_train, dtype=np.float64).reshape(-1)
    x = np.asarray(x_train, dtype=np.float64)
    grid = np.asarray(theta_grid, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != theta.size:
        raise ValueError("theta_train and x_train must contain the same rows.")
    spacing = np.diff(grid)
    if grid.size < 2 or np.any(spacing <= 0.0):
        raise ValueError("theta_grid must be strictly increasing.")
    radius = 0.5 * float(np.min(spacing))
    n_pairs = grid.size - 1
    response_dim = x.shape[1]
    mean_left = np.empty((n_pairs, response_dim), dtype=np.float64)
    mean_right = np.empty_like(mean_left)
    covariance_left = np.empty((n_pairs, response_dim, response_dim), dtype=np.float64)
    covariance_right = np.empty_like(covariance_left)
    counts = np.empty((n_pairs, 2), dtype=np.int64)
    for pair_index, (left, right) in enumerate(zip(grid[:-1], grid[1:], strict=True)):
        left_index = _endpoint_indices(
            theta, float(left), radius=radius, min_samples=int(min_endpoint_samples)
        )
        right_index = _endpoint_indices(
            theta, float(right), radius=radius, min_samples=int(min_endpoint_samples)
        )
        left_fit = LedoitWolf().fit(x[left_index])
        right_fit = LedoitWolf().fit(x[right_index])
        mean_left[pair_index] = left_fit.location_
        mean_right[pair_index] = right_fit.location_
        covariance_left[pair_index] = left_fit.covariance_
        covariance_right[pair_index] = right_fit.covariance_
        counts[pair_index] = (left_index.size, right_index.size)
    derivative = (mean_right - mean_left) / spacing[:, None]
    mixed_covariance = 0.5 * (covariance_left + covariance_right)
    estimate = optimal_linear_estimator(
        derivative, mixed_covariance, solve_jitter=1e-6
    )
    direction = decoder_directions(mean_right - mean_left, mixed_covariance)
    return {
        "theta_midpoints": 0.5 * (grid[:-1] + grid[1:]),
        "linear_fisher": np.asarray(estimate.linear_fisher, dtype=np.float64),
        "direction": direction,
        "mean_left": mean_left,
        "mean_right": mean_right,
        "covariance_left": covariance_left,
        "covariance_right": covariance_right,
        "endpoint_counts": counts,
    }


def _session_signature(
    args: argparse.Namespace, *, session_file: Path, session_index: int
) -> dict[str, Any]:
    return {
        "session_file": str(session_file),
        "session_index": int(session_index),
        "pca_dim": int(args.pca_dim),
        "pca": "centered_unwhitened_fit_on_outer_fit_half",
        "theta_grid_size": int(args.theta_grid_size),
        "train_fraction": float(args.train_fraction),
        "validation_fraction": float(args.validation_fraction),
        "test_fraction": float(
            1.0 - float(args.train_fraction) - float(args.validation_fraction)
        ),
        "seed": int(args.seed) + int(session_index),
        "epochs": int(args.epochs),
        "early_patience": int(args.early_patience),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "affine_ode_steps": int(args.affine_ode_steps),
        "full_ode_steps": int(args.full_ode_steps),
        "full_mc_samples": int(args.full_mc_samples),
        "full_hutchinson_probes": int(args.full_hutchinson_probes),
        "ole_crossfit_folds": int(args.ole_crossfit_folds),
        "tre_num_bridges": int(args.tre_num_bridges),
        "tre_hidden_dim": int(args.tre_hidden_dim),
        "tre_depth": int(args.tre_depth),
        "tre_batch_size": int(args.tre_batch_size),
        "tre_lr": float(args.tre_lr),
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {key: np.asarray(saved[key]) for key in saved.files}


def _prepare_session(
    args: argparse.Namespace,
    *,
    session_index: int,
    case_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    info = list_stringer_sessions("gratings_static")[int(session_index)]
    signature = _session_signature(
        args, session_file=Path(info.session_file), session_index=int(session_index)
    )
    path = case_dir / "pca82_split.npz"
    metadata_path = case_dir / "split_metadata.json"
    if path.is_file() and metadata_path.is_file() and not args.force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") != signature:
            raise ValueError(
                f"Cached split signature differs in {case_dir}; pass --force or use a new output."
            )
        return _load_npz(path), metadata

    session = load_stringer_session(
        None,
        session_stimuli_type="gratings_static",
        session_index=int(session_index),
        orientation_period=PERIOD,
    )
    theta = np.asarray(session.grating_orientation, dtype=np.float64)
    responses = np.asarray(session.neural_responses)
    train_index, validation_index, test_index = (
        exact_stratified_fit_validation_test_indices(
            theta,
            n_strata=int(args.theta_grid_size) - 1,
            train_fraction=float(args.train_fraction),
            validation_fraction=float(args.validation_fraction),
            seed=int(args.seed) + int(session_index),
        )
    )
    fit_index = np.sort(np.concatenate([train_index, validation_index]))
    pca = PCA(
        n_components=int(args.pca_dim),
        whiten=False,
        svd_solver="randomized",
        random_state=int(args.seed) + int(session_index),
    )
    pca.fit(responses[fit_index])
    x = pca.transform(responses).astype(np.float32)
    arrays = {
        "theta": theta,
        "x": x,
        "train_index": train_index,
        "validation_index": validation_index,
        "fit_index": fit_index,
        "test_index": test_index,
        "pca_components": pca.components_,
        "pca_mean": pca.mean_,
        "pca_explained_variance": pca.explained_variance_,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    metadata = {
        "signature": signature,
        "session_label": str(info.mouse_name),
        "n_observations": int(theta.size),
        "n_neurons": int(responses.shape[1]),
        "n_train": int(train_index.size),
        "n_validation": int(validation_index.size),
        "n_fit": int(fit_index.size),
        "n_test": int(test_index.size),
        "actual_test_fraction": float(test_index.size / theta.size),
        "split_disjoint": bool(
            np.intersect1d(fit_index, test_index).size == 0
            and np.unique(np.concatenate([fit_index, test_index])).size == theta.size
        ),
        "pca_fit_scope": "outer_fit_half_only",
        "pca_explained_variance_ratio_sum": float(
            np.sum(pca.explained_variance_ratio_)
        ),
    }
    metadata_path.write_text(
        json.dumps(_json_ready(metadata), indent=2) + "\n", encoding="utf-8"
    )
    return arrays, metadata


def _fit_linear_methods(
    args: argparse.Namespace,
    *,
    case_dir: Path,
    arrays: dict[str, np.ndarray],
    theta_grid: np.ndarray,
    device: torch.device,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    theta = arrays["theta"]
    x = arrays["x"]
    train = arrays["train_index"].astype(np.int64)
    validation = arrays["validation_index"].astype(np.int64)
    fit = arrays["fit_index"].astype(np.int64)
    condition = encode_flow_orientation(theta, period=PERIOD, encoding="periodic-rbf")
    condition_grid = encode_flow_orientation(
        theta_grid, period=PERIOD, encoding="periodic-rbf"
    )
    results: dict[str, dict[str, np.ndarray]] = {}

    gkr_path = case_dir / "gkr_linear.npz"
    if gkr_path.is_file() and not args.force:
        results["gkr"] = _load_npz(gkr_path)
    else:
        model, estimate, direction = fit_gkr_direction_estimator(
            theta_train=theta[fit],
            x_train=x[fit],
            theta_grid=theta_grid,
            device=device,
            seed=int(seed),
            circular_period=PERIOD,
        )
        result = {
            "theta_midpoints": 0.5
            * (theta_grid[:-1, 0] + theta_grid[1:, 0]),
            "linear_fisher": np.asarray(
                estimate.linear_fisher, dtype=np.float64
            ).reshape(-1),
            "direction": np.asarray(direction, dtype=np.float64),
            "mean_loss": np.asarray(estimate.mean_loss, dtype=np.float64),
            "covariance_loss": np.asarray(
                estimate.covariance_loss, dtype=np.float64
            ),
        }
        np.savez_compressed(gkr_path, **result)
        torch.save(gkr_checkpoint(model), case_dir / "gkr_model.pt")
        del model
        results["gkr"] = result
        if device.type == "cuda":
            torch.cuda.empty_cache()

    bin_path = case_dir / "bin_lw_linear.npz"
    if bin_path.is_file() and not args.force:
        results["bin_lw"] = _load_npz(bin_path)
    else:
        result = fit_binned_lw_direction_estimator(
            theta_train=theta[fit],
            x_train=x[fit],
            theta_grid=theta_grid,
            min_endpoint_samples=int(args.ole_min_endpoint_samples),
        )
        np.savez_compressed(bin_path, **result)
        results["bin_lw"] = result

    ole_path = case_dir / "ole_crossfit_linear.npz"
    if ole_path.is_file() and not args.force:
        results["ole_crossfit"] = _load_npz(ole_path)
    else:
        estimate, direction = fit_cross_fitted_ole_direction_estimator(
            theta_train=theta[fit],
            x_train=x[fit],
            theta_grid=theta_grid,
            n_splits=int(args.ole_crossfit_folds),
            seed=int(seed) + 20_000,
            min_endpoint_samples=int(args.ole_min_endpoint_samples),
            period=PERIOD,
        )
        result = {
            "theta_midpoints": np.asarray(
                estimate.theta_midpoints, dtype=np.float64
            ).reshape(-1),
            "linear_fisher": np.asarray(
                estimate.linear_fisher, dtype=np.float64
            ).reshape(-1),
            "linear_fisher_raw": np.asarray(
                estimate.linear_fisher_raw, dtype=np.float64
            ).reshape(-1),
            "direction": np.asarray(direction, dtype=np.float64),
            "fold_weights": np.asarray(estimate.fold_weights, dtype=np.float64),
        }
        np.savez_compressed(ole_path, **result)
        results["ole_crossfit"] = result

    affine_path = case_dir / "affine_flow_linear.npz"
    if affine_path.is_file() and not args.force:
        results["affine_flow"] = _load_npz(affine_path)
    else:
        model, training, estimate, direction = fit_flow_direction_estimator(
            theta_train=theta[train],
            x_train=x[train],
            theta_validation=theta[validation],
            x_validation=x[validation],
            theta_grid=theta_grid,
            condition_train=condition[train],
            condition_validation=condition[validation],
            condition_grid=condition_grid,
            device=device,
            seed=int(seed),
            epochs=int(args.epochs),
            patience=int(args.early_patience),
            batch_size=min(int(args.batch_size), int(train.size)),
            learning_rate=float(args.lr),
            hidden_dim=int(args.hidden_dim),
            depth=int(args.depth),
            ode_steps=int(args.affine_ode_steps),
        )
        result = {
            "theta_midpoints": np.asarray(
                estimate["theta_midpoints"], dtype=np.float64
            ).reshape(-1),
            "linear_fisher": np.asarray(
                estimate["fisher"], dtype=np.float64
            ).reshape(-1),
            "direction": np.asarray(direction, dtype=np.float64),
            "delta_mu": np.asarray(estimate["delta_mu"], dtype=np.float64),
            "mixed_covariance": np.asarray(
                estimate["mixed_covariance"], dtype=np.float64
            ),
            "train_losses": np.asarray(training["train_losses"], dtype=np.float64),
            "validation_losses": np.asarray(
                training["val_losses"], dtype=np.float64
            ),
            "selected_epoch": np.asarray(int(training["selected_epoch"])),
            "best_epoch": np.asarray(int(training["best_epoch"])),
            "stopped_epoch": np.asarray(int(training["stopped_epoch"])),
        }
        np.savez_compressed(affine_path, **result)
        torch.save(
            {key: value.detach().cpu() for key, value in model.state_dict().items()},
            case_dir / "affine_flow_selected.pt",
        )
        del model
        results["affine_flow"] = result
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def _fit_unconstrained_full_fisher(
    args: argparse.Namespace,
    *,
    case_dir: Path,
    arrays: dict[str, np.ndarray],
    theta_grid: np.ndarray,
    device: torch.device,
    seed: int,
) -> dict[str, np.ndarray]:
    path = case_dir / "unconstrained_flow_full.npz"
    if path.is_file() and not args.force:
        return _load_npz(path)
    theta = arrays["theta"]
    x = arrays["x"]
    train = arrays["train_index"].astype(np.int64)
    validation = arrays["validation_index"].astype(np.int64)
    condition = encode_flow_orientation(theta, period=PERIOD, encoding="periodic-rbf")
    condition_grid = encode_flow_orientation(
        theta_grid, period=PERIOD, encoding="periodic-rbf"
    )
    torch.manual_seed(int(seed) + 100_000)
    np.random.seed(int(seed) + 100_000)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed) + 100_000)
    model = build_flow_skl_model(
        velocity_family="nonlinear",
        theta_dim=int(condition.shape[1]),
        x_dim=int(x.shape[1]),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        path_schedule="cosine",
        divergence_estimator="hutchinson",
        hutchinson_probes=int(args.full_hutchinson_probes),
        theta_embedding="identity",
    ).to(device)
    training = train_flow_skl_model(
        model=model,
        theta_train=condition[train],
        x_train=x[train],
        theta_val=condition[validation],
        x_val=x[validation],
        device=device,
        velocity_family="nonlinear",
        path_schedule="cosine",
        epochs=int(args.epochs),
        batch_size=min(int(args.batch_size), int(train.size)),
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
        fixed_validation_paths=10,
        validation_seed=int(seed) + 110_000,
        retain_best_state=True,
    )
    best_state = training.pop("best_state_dict")
    torch.save(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        case_dir / "unconstrained_flow_last.pt",
    )
    model.load_state_dict(best_state)
    torch.save(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        case_dir / "unconstrained_flow_best.pt",
    )
    estimate = estimate_adjacent_model_jeffreys_fisher(
        model=model,
        theta_all=theta_grid,
        condition_all=condition_grid,
        device=device,
        mc_jeffreys_sample=int(args.full_mc_samples),
        ode_steps=int(args.full_ode_steps),
        ode_method="midpoint",
        batch_size=int(args.full_likelihood_batch_size),
        solve_jitter=1e-6,
        quadrature_steps=64,
    )
    result = {
        "theta_midpoints": np.asarray(
            estimate["theta_midpoints"], dtype=np.float64
        ).reshape(-1),
        "full_fisher": np.asarray(estimate["fisher"], dtype=np.float64).reshape(-1),
        "adjacent_jeffreys": np.asarray(
            estimate["adjacent_jeffreys"], dtype=np.float64
        ).reshape(-1),
        "dtheta": np.asarray(estimate["dtheta"], dtype=np.float64).reshape(-1),
        "train_losses": np.asarray(training["train_losses"], dtype=np.float64),
        "validation_losses": np.asarray(training["val_losses"], dtype=np.float64),
        "selected_epoch": np.asarray(int(training["best_epoch"])),
        "best_epoch": np.asarray(int(training["best_epoch"])),
        "stopped_epoch": np.asarray(int(training["stopped_epoch"])),
    }
    np.savez_compressed(path, **result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _fit_tre8_full_fisher(
    args: argparse.Namespace,
    *,
    case_dir: Path,
    arrays: dict[str, np.ndarray],
    theta_grid: np.ndarray,
    device: torch.device,
    seed: int,
) -> dict[str, np.ndarray]:
    path = case_dir / "tre8_full.npz"
    if path.is_file() and not args.force:
        return _load_npz(path)
    theta = arrays["theta"]
    x = arrays["x"]
    train = arrays["train_index"].astype(np.int64)
    validation = arrays["validation_index"].astype(np.int64)
    fit = arrays["fit_index"].astype(np.int64)
    config = TREDensityRatioConfig(
        num_bridges=int(args.tre_num_bridges),
        waymark_schedule="angle",
        architecture="mlp",
        hidden_dim=int(args.tre_hidden_dim),
        depth=int(args.tre_depth),
        epochs=int(args.epochs),
        batch_size=int(args.tre_batch_size),
        lr=float(args.tre_lr),
        weight_decay=0.0,
        early_patience=int(args.early_patience),
        early_min_delta=1e-5,
        max_grad_norm=10.0,
        validation_pairs=int(args.tre_validation_pairs),
        standardize=True,
        log_every=1000,
    )
    states, estimate = train_and_estimate_binned_tre_fisher(
        theta_train=theta[train],
        x_train=x[train],
        theta_validation=theta[validation],
        x_validation=x[validation],
        theta_eval=theta[fit],
        x_eval=x[fit],
        theta_grid=theta_grid,
        theta_period=PERIOD,
        device=device,
        seed=int(seed) + 200_000,
        config=config,
        min_train_samples=2,
        min_validation_samples=2,
        min_eval_samples=2,
    )
    result = {
        "theta_midpoints": 0.5 * (theta_grid[:-1, 0] + theta_grid[1:, 0]),
        "full_fisher": np.asarray(estimate.fisher, dtype=np.float64),
        "jeffreys": np.asarray(estimate.jeffreys, dtype=np.float64),
        "raw_jeffreys": np.asarray(estimate.raw_jeffreys, dtype=np.float64),
    }
    np.savez_compressed(path, **result)
    torch.save(
        {
            "pair_state_dicts": states,
            "pair_metadata": estimate.pair_metadata,
            "run_metadata": estimate.run_metadata,
        },
        case_dir / "tre8_models.pt",
    )
    (case_dir / "tre8_metadata.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "run_metadata": estimate.run_metadata,
                "pair_metadata": estimate.pair_metadata,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _evaluate_linear_methods(
    *,
    arrays: dict[str, np.ndarray],
    theta_grid: np.ndarray,
    linear: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    theta = arrays["theta"]
    x = arrays["x"]
    test = arrays["test_index"].astype(np.int64)
    spacing = np.diff(theta_grid[:, 0])
    half_width = 0.5 * float(np.min(spacing))
    result: dict[str, np.ndarray] = {}
    for key in LINEAR_METHOD_KEYS:
        evaluation = evaluate_windowed_decoders(
            np.asarray(linear[key]["direction"], dtype=np.float64),
            x[test],
            theta[test],
            theta_grid[:-1, 0],
            theta_grid[1:, 0],
            half_width=half_width,
            period=PERIOD,
        )
        result[f"{key}_achieved_raw"] = np.asarray(
            evaluation.achieved_fisher_raw, dtype=np.float64
        )
        result[f"{key}_achieved_display"] = np.asarray(
            evaluation.achieved_fisher_display, dtype=np.float64
        )
        result[f"{key}_test_counts_left"] = np.asarray(
            evaluation.n_left, dtype=np.int64
        )
        result[f"{key}_test_counts_right"] = np.asarray(
            evaluation.n_right, dtype=np.int64
        )
    return result


def _fit_session(
    args: argparse.Namespace,
    *,
    session_index: int,
    device: torch.device,
) -> dict[str, Any]:
    sessions = list_stringer_sessions("gratings_static")
    info = sessions[int(session_index)]
    label = str(info.mouse_name)
    case_dir = args.output_dir / f"session_{int(session_index):02d}_{label}"
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    arrays, split_metadata = _prepare_session(
        args, session_index=int(session_index), case_dir=case_dir
    )
    theta_grid = np.linspace(
        0.0, PERIOD, int(args.theta_grid_size), dtype=np.float64
    ).reshape(-1, 1)
    seed = int(args.seed) + int(session_index)
    linear = _fit_linear_methods(
        args,
        case_dir=case_dir,
        arrays=arrays,
        theta_grid=theta_grid,
        device=device,
        seed=seed,
    )
    unconstrained = _fit_unconstrained_full_fisher(
        args,
        case_dir=case_dir,
        arrays=arrays,
        theta_grid=theta_grid,
        device=device,
        seed=seed,
    )
    tre8 = _fit_tre8_full_fisher(
        args,
        case_dir=case_dir,
        arrays=arrays,
        theta_grid=theta_grid,
        device=device,
        seed=seed,
    )
    evaluation = _evaluate_linear_methods(
        arrays=arrays, theta_grid=theta_grid, linear=linear
    )
    combined: dict[str, np.ndarray] = {
        "theta_midpoints": 0.5 * (theta_grid[:-1, 0] + theta_grid[1:, 0]),
        "unconstrained_flow_full_fisher": unconstrained["full_fisher"],
        "tre8_full_fisher": tre8["full_fisher"],
        **evaluation,
    }
    for key in LINEAR_METHOD_KEYS:
        combined[f"{key}_linear_fisher"] = np.asarray(
            linear[key]["linear_fisher"], dtype=np.float64
        )
    np.savez_compressed(case_dir / "fisher_results.npz", **combined)
    summary = {
        "session_index": int(session_index),
        "session_label": label,
        "device": str(device),
        "split": split_metadata,
        "mean_achieved_linear_fisher": {
            title: float(np.mean(evaluation[f"{key}_achieved_raw"]))
            for key, title in zip(
                LINEAR_METHOD_KEYS, LINEAR_METHOD_TITLES, strict=True
            )
        },
        "mean_estimated_linear_fisher": {
            title: float(np.mean(linear[key]["linear_fisher"]))
            for key, title in zip(
                LINEAR_METHOD_KEYS, LINEAR_METHOD_TITLES, strict=True
            )
        },
        "mean_estimated_full_fisher": {
            FULL_METHOD_TITLES[0]: float(np.mean(unconstrained["full_fisher"])),
            FULL_METHOD_TITLES[1]: float(np.mean(tre8["full_fisher"])),
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (case_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[session complete] {label} runtime={summary['runtime_seconds']:.1f}s "
        f"result={case_dir / 'fisher_results.npz'}",
        flush=True,
    )
    return {"summary": summary, "arrays": combined, "case_dir": case_dir}


def _load_completed_cases(output_dir: Path) -> list[dict[str, Any]]:
    cases = []
    for case_dir in sorted(path for path in output_dir.glob("session_*_*") if path.is_dir()):
        summary_path = case_dir / "summary.json"
        result_path = case_dir / "fisher_results.npz"
        if summary_path.is_file() and result_path.is_file():
            cases.append(
                {
                    "case_dir": case_dir,
                    "summary": json.loads(summary_path.read_text(encoding="utf-8")),
                    "arrays": _load_npz(result_path),
                }
            )
    cases.sort(key=lambda case: int(case["summary"]["session_index"]))
    if len(cases) != 6 or [
        int(case["summary"]["session_index"]) for case in cases
    ] != list(range(6)):
        raise ValueError(
            f"Expected completed sessions 0 through 5 under {output_dir}; found {len(cases)}."
        )
    return cases


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.8)
    axis.spines["bottom"].set_linewidth(1.8)
    axis.tick_params(width=1.8)
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color="0.88", linewidth=0.8)
    axis.xaxis.grid(False)


def _achieved_axis_limits(
    achieved: np.ndarray,
    sem: np.ndarray,
    *,
    padding_fraction: float = 0.1,
) -> tuple[float, float]:
    """Return padded limits that expose method differences without forcing zero."""

    values = np.asarray(achieved, dtype=np.float64)
    errors = np.asarray(sem, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[1] != errors.size:
        raise ValueError("achieved must be [session, method] and match sem.")
    lower = float(min(np.min(values), np.min(np.mean(values, axis=0) - errors)))
    upper = float(max(np.max(values), np.max(np.mean(values, axis=0) + errors)))
    span = max(upper - lower, 1.0)
    padding = float(padding_fraction) * span
    return lower - padding, upper + padding


def _plot_achieved_information(
    cases: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "axes.grid": False,
        }
    )
    fig, axis = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    achieved = np.asarray(
        [
            [
                np.mean(case["arrays"][f"{key}_achieved_raw"])
                for key in LINEAR_METHOD_KEYS
            ]
            for case in cases
        ],
        dtype=np.float64,
    )
    positions = np.arange(len(LINEAR_METHOD_KEYS), dtype=np.float64)
    means = np.mean(achieved, axis=0)
    sem = np.std(achieved, axis=0, ddof=1) / np.sqrt(achieved.shape[0])
    axis.bar(
        positions,
        means,
        yerr=sem,
        color=BAR_COLORS,
        alpha=0.55,
        edgecolor=BAR_COLORS,
        linewidth=1.8,
        capsize=3.0,
        zorder=2,
    )
    for session_values in achieved:
        axis.plot(
            positions,
            session_values,
            color="0.45",
            linewidth=1.1,
            alpha=0.7,
            zorder=3,
        )
        axis.scatter(
            positions,
            session_values,
            color="black",
            s=28,
            alpha=0.9,
            zorder=4,
        )
    axis.set_xticks(positions, BAR_LABELS, rotation=20, ha="right")
    axis.set_ylabel("Held-out achieved\nlinear Fisher")
    axis.set_title("50% held-out")
    axis.set_ylim(*_achieved_axis_limits(achieved, sem))
    _style_axis(axis)

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "stringer_heldout_achieved_linear_fisher.png"
    svg = output_dir / "stringer_heldout_achieved_linear_fisher.svg"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    return png, svg


def _plot_fisher_curves(
    cases: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "axes.grid": False,
        }
    )
    fig, axes = plt.subplots(
        1,
        6,
        figsize=(21.5, 3.5),
        constrained_layout=True,
    )
    theta_ticks = [0.0, 0.5 * np.pi, np.pi]
    theta_ticklabels = [r"$0$", r"$\pi/2$", r"$\pi$"]
    for panel_index, (key, title) in enumerate(
        zip(LINEAR_METHOD_KEYS, LINEAR_METHOD_TITLES, strict=True)
    ):
        axis = axes[panel_index]
        curves = np.stack(
            [
                np.asarray(case["arrays"][f"{key}_linear_fisher"], dtype=np.float64)
                for case in cases
            ]
        )
        theta_midpoints = np.asarray(cases[0]["arrays"]["theta_midpoints"])
        for curve in curves:
            axis.plot(
                theta_midpoints,
                curve,
                color="0.35",
                linewidth=1.0,
                alpha=0.25,
                zorder=1,
            )
        mean = np.mean(curves, axis=0)
        sem = np.std(curves, axis=0, ddof=1) / np.sqrt(curves.shape[0])
        axis.errorbar(
            theta_midpoints,
            mean,
            yerr=sem,
            color="black",
            linewidth=2.2,
            marker="o",
            markersize=3.5,
            capsize=2.5,
            elinewidth=1.2,
            label="Mean ± SEM",
            zorder=3,
        )
        axis.set_title(title)
        axis.set_xticks(theta_ticks, theta_ticklabels)
        axis.set_xlim(0.0, PERIOD)
        axis.set_xlabel(r"Orientation $\theta$")
        if panel_index == 0:
            axis.set_ylabel("Linear Fisher")

    for offset, (key, title) in enumerate(
        zip(FULL_METHOD_KEYS, FULL_METHOD_TITLES, strict=True), start=4
    ):
        axis = axes[offset]
        curves = np.stack(
            [
                np.asarray(case["arrays"][f"{key}_full_fisher"], dtype=np.float64)
                for case in cases
            ]
        )
        theta_midpoints = np.asarray(cases[0]["arrays"]["theta_midpoints"])
        for curve in curves:
            axis.plot(
                theta_midpoints,
                curve,
                color="0.35",
                linewidth=1.0,
                alpha=0.25,
                zorder=1,
            )
        mean = np.mean(curves, axis=0)
        sem = np.std(curves, axis=0, ddof=1) / np.sqrt(curves.shape[0])
        axis.errorbar(
            theta_midpoints,
            mean,
            yerr=sem,
            color="black",
            linewidth=2.2,
            marker="o",
            markersize=3.5,
            capsize=2.5,
            elinewidth=1.2,
            label="Mean ± SEM",
            zorder=3,
        )
        axis.set_title(title)
        axis.set_xticks(theta_ticks, theta_ticklabels)
        axis.set_xlim(0.0, PERIOD)
        axis.set_xlabel(r"Orientation $\theta$")
        if offset == 4:
            axis.set_ylabel("Full Fisher")
    for axis in axes:
        _style_axis(axis)

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "stringer_linear_and_full_fisher_curves.png"
    svg = output_dir / "stringer_linear_and_full_fisher_curves.svg"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    return png, svg


def _aggregate(args: argparse.Namespace) -> dict[str, Any]:
    cases = _load_completed_cases(args.output_dir)
    if args.gkr_curve_root is not None:
        gkr_root = args.gkr_curve_root.expanduser().resolve()
        for case in cases:
            session_index = int(case["summary"]["session_index"])
            matches = sorted(gkr_root.glob(f"session_{session_index:02d}_*"))
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one upgraded GKR session {session_index} under {gkr_root}."
                )
            curve_path = matches[0] / "log_lambda_linear_fisher.npz"
            if not curve_path.is_file():
                raise FileNotFoundError(f"Missing upgraded GKR curve: {curve_path}")
            upgraded = _load_npz(curve_path)
            np.testing.assert_allclose(
                case["arrays"]["theta_midpoints"],
                upgraded["theta_midpoints"],
                rtol=0.0,
                atol=1e-12,
            )
            case["arrays"]["gkr_linear_fisher"] = np.asarray(
                upgraded["linear_fisher"], dtype=np.float64
            )
    figure_dir = args.output_dir / "figures"
    achieved_png, achieved_svg = _plot_achieved_information(
        cases, output_dir=figure_dir
    )
    curves_png, curves_svg = _plot_fisher_curves(
        cases, output_dir=figure_dir
    )
    achieved_rows = []
    for case in cases:
        for key, title in zip(
            LINEAR_METHOD_KEYS, LINEAR_METHOD_TITLES, strict=True
        ):
            values = np.asarray(
                case["arrays"][f"{key}_achieved_raw"], dtype=np.float64
            )
            achieved_rows.append(
                {
                    "session_index": int(case["summary"]["session_index"]),
                    "session_label": str(case["summary"]["session_label"]),
                    "method": title,
                    "mean_achieved_linear_fisher": float(np.mean(values)),
                    "median_achieved_linear_fisher": float(np.median(values)),
                }
            )
    summary = {
        "protocol": {
            "outer_split": "40% train, 10% validation, 50% untouched test",
            "pca": "82D centered, unwhitened, fit on the outer 50% fitting pool",
            "condition": "periodic eight-center RBF over orientation [0, pi)",
            "gkr_linear_curve_source": (
                "session-local GKR fits"
                if args.gkr_curve_root is None
                else str(args.gkr_curve_root.expanduser().resolve())
            ),
            "gkr_linear_curve_note": (
                None
                if args.gkr_curve_root is None
                else (
                    "Linear Fisher curves use the saved upgraded periodic "
                    "log-lambda GKR models and their original 80/20 fit/test PCA."
                )
            ),
            "achieved_information": (
                "bias-reduced endpoint information evaluated only on the outer test half"
            ),
            "linear_methods": list(LINEAR_METHOD_TITLES),
            "full_methods": list(FULL_METHOD_TITLES),
        },
        "achieved_information": achieved_rows,
        "artifacts": {
            "achieved_information_png": str(achieved_png),
            "achieved_information_svg": str(achieved_svg),
            "fisher_curves_png": str(curves_png),
            "fisher_curves_svg": str(curves_svg),
        },
    }
    summary_path = args.output_dir / "all_sessions_summary.json"
    summary_path.write_text(
        json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved: {summary_path}", flush=True)
    print(f"Saved: {achieved_png}", flush=True)
    print(f"Saved: {achieved_svg}", flush=True)
    print(f"Saved: {curves_png}", flush=True)
    print(f"Saved: {curves_svg}", flush=True)
    return summary


def _validate_args(args: argparse.Namespace) -> None:
    test_fraction = (
        1.0 - float(args.train_fraction) - float(args.validation_fraction)
    )
    if not np.isclose(test_fraction, 0.5):
        raise ValueError("This figure protocol requires exactly 50% outer test data.")
    if min(float(args.train_fraction), float(args.validation_fraction)) <= 0.0:
        raise ValueError("train and validation fractions must be positive.")
    if int(args.theta_grid_size) < 3:
        raise ValueError("theta-grid-size must be at least three.")
    if int(args.pca_dim) < 1:
        raise ValueError("pca-dim must be positive.")
    if int(args.tre_num_bridges) != 8:
        raise ValueError("This requested comparison requires TRE-8.")


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    _validate_args(args)
    if args.aggregate_only:
        _aggregate(args)
        return 0
    if not args.device:
        raise ValueError("--device is required for fitting.")
    device = require_device(str(args.device))
    sessions = list_stringer_sessions("gratings_static")
    indices = (
        list(range(len(sessions)))
        if args.session_indices is None
        else [int(index) for index in args.session_indices]
    )
    if len(set(indices)) != len(indices):
        raise ValueError("session-indices must be unique.")
    if any(index < 0 or index >= len(sessions) for index in indices):
        raise ValueError("session-indices contains an out-of-range value.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for position, session_index in enumerate(indices):
        print(
            f"[session] {position + 1}/{len(indices)} "
            f"index={session_index} label={sessions[session_index].mouse_name}",
            flush=True,
        )
        _fit_session(args, session_index=session_index, device=device)
    if not args.skip_aggregate:
        _aggregate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
