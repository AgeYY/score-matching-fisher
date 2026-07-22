#!/usr/bin/env python3
"""Run independent detailed evaluations of linear Fisher estimators."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.continuous_fisher_comparison import make_native_dataset_npz, theta_grid_from_meta
from fisher.fisher_validation import (
    decoder_directions,
    evaluate_endpoint_decoders,
    evaluate_linear_threshold_decoders,
    evaluate_windowed_decoders,
    evaluate_windowed_linear_threshold_decoders,
    finite_endpoint_oracle,
    fisher_predicted_linear_error,
    fit_cross_fitted_ole_direction_estimator,
    fit_flow_direction_estimator,
    fit_gkr_direction_estimator,
    gkr_checkpoint,
    population_linear_moments,
    stratified_train_validation_test_split,
)
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_meta, require_device
from fisher.stringer_dataset import load_stringer_session
from fisher.stringer_session_identification import encode_flow_orientation
from global_setting import DATA_DIR, DEFAULT_EARLY_STOPPING_PATIENCE, DEFAULT_TRAINING_MAX_EPOCHS

DATASET_LABELS = {
    "randamp_gaussian_sqrtd": "Gaussian toy",
    "cosine_gmm": "Gaussian-mixture toy",
    "stringer": "Stringer",
}


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _allocations(value: str) -> list[tuple[float, float]]:
    result = []
    for item in value.split(","):
        train, validation = item.strip().split(":", maxsplit=1)
        result.append((float(train), float(validation)))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "direction",
        choices=("reference", "validation-allocation", "train-test-allocation"),
    )
    parser.add_argument("--dataset", choices=tuple(DATASET_LABELS), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "fisher_validation_directions",
    )
    parser.add_argument("--seeds", type=_csv_ints, default=[7, 19, 23, 31, 47])
    parser.add_argument("--train-fraction", type=float, default=0.64)
    parser.add_argument("--validation-fraction", type=float, default=0.16)
    parser.add_argument(
        "--allocations",
        type=_allocations,
        default=[(0.72, 0.08), (0.64, 0.16), (0.56, 0.24), (0.48, 0.32)],
    )
    parser.add_argument(
        "--test-fractions",
        type=_csv_floats,
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        help="Test fractions for train-test-allocation; validation remains fixed.",
    )
    parser.add_argument("--x-dim", type=int, default=50)
    parser.add_argument("--n-total", type=int, default=3000)
    parser.add_argument("--theta-grid-size", type=int, default=61)
    parser.add_argument("--endpoint-calibration-samples", type=int, default=1000)
    parser.add_argument("--endpoint-test-samples", type=int, default=2000)
    parser.add_argument("--session-index", type=int, default=0)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument(
        "--pca-whiten",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whiten Stringer PCA scores after fitting PCA on the training split.",
    )
    parser.add_argument("--stringer-grid-size", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_MAX_EPOCHS)
    parser.add_argument("--early-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--ode-steps", type=int, default=64)
    parser.add_argument("--ole-crossfit-folds", type=int, default=5)
    parser.add_argument("--ole-crossfit-seed", type=int, default=20_260_721)
    parser.add_argument("--ole-min-endpoint-samples", type=int, default=8)
    parser.add_argument("--ole-window-radius", type=float, default=None)
    estimator_group = parser.add_mutually_exclusive_group()
    estimator_group.add_argument(
        "--skip-gkr",
        action="store_true",
        help="Fit and evaluate only Flow Matching and cross-fitted OLE.",
    )
    estimator_group.add_argument(
        "--gkr-only",
        action="store_true",
        help="Fit and evaluate only GKR, preserving existing Flow Matching caches.",
    )
    estimator_group.add_argument(
        "--fm-only",
        action="store_true",
        help="Fit and evaluate only Flow Matching, without GKR or OLE.",
    )
    parser.add_argument("--force-source", action="store_true")
    parser.add_argument("--force-fit", action="store_true")
    return parser.parse_args()


def _ratio_token(train_fraction: float, validation_fraction: float) -> str:
    test_fraction = 1.0 - float(train_fraction) - float(validation_fraction)
    return (
        f"train{train_fraction:.2f}_val{validation_fraction:.2f}_test{test_fraction:.2f}"
        .replace(".", "p")
    )


def _training_signature(args: argparse.Namespace, *, n_train: int, n_validation: int) -> dict[str, Any]:
    return {
        "n_train": int(n_train),
        "n_validation": int(n_validation),
        "epochs": int(args.epochs),
        "early_patience": int(args.early_patience),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "ode_steps": int(args.ode_steps),
        "pca_whiten": bool(getattr(args, "pca_whiten", True)),
        "fit_gkr": not bool(args.skip_gkr or args.fm_only),
    }


def _jsonable_training(training: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "selected_epoch",
        "stopped_epoch",
        "best_epoch",
        "best_val_loss",
        "checkpoint_selection",
        "fixed_validation",
        "fixed_validation_paths",
    )
    return {key: training[key] for key in keys if key in training}


def _fit_estimators(
    *,
    case_dir: Path,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_validation: np.ndarray,
    x_validation: np.ndarray,
    theta_grid: np.ndarray,
    period: float | None,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    fit_dir = case_dir / "fit"
    fit_dir.mkdir(parents=True, exist_ok=True)
    estimates_path = fit_dir / "estimates.npz"
    metadata_path = fit_dir / "metadata.json"
    flow_path = fit_dir / "flow_selected_model.pt"
    gkr_path = fit_dir / "gkr_model.pt"
    signature = _training_signature(
        args,
        n_train=x_train.shape[0],
        n_validation=x_validation.shape[0],
    )
    required_paths = [estimates_path, metadata_path, flow_path]
    if not (args.skip_gkr or args.fm_only):
        required_paths.append(gkr_path)
    if all(path.is_file() for path in required_paths) and not args.force_fit:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("training_signature") == signature:
            with np.load(estimates_path) as saved:
                return {key: np.asarray(saved[key]) for key in saved.files} | {"metadata": metadata}

    condition_train = condition_validation = condition_grid = None
    if period is not None:
        condition_train = encode_flow_orientation(theta_train, period=period, encoding="periodic-rbf")
        condition_validation = encode_flow_orientation(
            theta_validation, period=period, encoding="periodic-rbf"
        )
        condition_grid = encode_flow_orientation(theta_grid, period=period, encoding="periodic-rbf")

    flow_model, flow_training, flow_estimate, flow_direction = fit_flow_direction_estimator(
        theta_train=theta_train,
        x_train=x_train,
        theta_validation=theta_validation,
        x_validation=x_validation,
        theta_grid=theta_grid,
        condition_train=condition_train,
        condition_validation=condition_validation,
        condition_grid=condition_grid,
        device=device,
        seed=seed,
        epochs=args.epochs,
        patience=args.early_patience,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        ode_steps=args.ode_steps,
    )
    arrays = {
        "theta_midpoints": np.asarray(flow_estimate["theta_midpoints"], dtype=np.float64).reshape(-1),
        "theta_left": np.asarray(flow_estimate["theta_left"], dtype=np.float64).reshape(-1),
        "theta_right": np.asarray(flow_estimate["theta_right"], dtype=np.float64).reshape(-1),
        "dtheta": np.asarray(flow_estimate["dtheta"], dtype=np.float64).reshape(-1),
        "flow_fisher": np.asarray(flow_estimate["fisher"], dtype=np.float64).reshape(-1),
        "flow_direction": np.asarray(flow_direction, dtype=np.float64),
        "flow_train_loss": np.asarray(flow_training["train_losses"], dtype=np.float64),
        "flow_validation_loss": np.asarray(flow_training["val_losses"], dtype=np.float64),
    }
    gkr_model = None
    if not (args.skip_gkr or args.fm_only):
        gkr_model, gkr_estimate, gkr_direction = fit_gkr_direction_estimator(
            theta_train=theta_train,
            x_train=x_train,
            theta_grid=theta_grid,
            device=device,
            seed=seed,
            circular_period=period,
        )
        arrays.update(
            {
                "gkr_fisher": np.asarray(
                    gkr_estimate.linear_fisher, dtype=np.float64
                ).reshape(-1),
                "gkr_direction": np.asarray(gkr_direction, dtype=np.float64),
                "gkr_mean_loss": np.asarray(gkr_estimate.mean_loss, dtype=np.float64),
                "gkr_covariance_loss": np.asarray(
                    gkr_estimate.covariance_loss, dtype=np.float64
                ),
            }
        )
    np.savez_compressed(estimates_path, **arrays)
    torch.save({key: value.detach().cpu() for key, value in flow_model.state_dict().items()}, flow_path)
    if gkr_model is not None:
        torch.save(gkr_checkpoint(gkr_model), gkr_path)
    metadata = {
        "seed": int(seed),
        "training_signature": signature,
        "flow_training": _jsonable_training(flow_training),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    del flow_model
    if gkr_model is not None:
        del gkr_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays | {"metadata": metadata}


def _fit_ole_estimator(
    *,
    case_dir: Path,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_grid: np.ndarray,
    period: float | None,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, np.ndarray]:
    fit_dir = case_dir / "fit"
    fit_dir.mkdir(parents=True, exist_ok=True)
    estimates_path = fit_dir / "ole_crossfit.npz"
    metadata_path = fit_dir / "ole_crossfit_metadata.json"
    signature = {
        "ensemble": "mean_unit_fold_directions_v1",
        "n_train": int(np.asarray(x_train).shape[0]),
        "response_dim": int(np.asarray(x_train).shape[1]),
        "theta_grid": np.asarray(theta_grid, dtype=np.float64).reshape(-1).tolist(),
        "n_splits": int(args.ole_crossfit_folds),
        "seed": int(args.ole_crossfit_seed) + int(seed),
        "window_radius": args.ole_window_radius,
        "min_endpoint_samples": int(args.ole_min_endpoint_samples),
        "period": period,
    }
    if estimates_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            with np.load(estimates_path) as saved:
                return {key: np.asarray(saved[key]) for key in saved.files}

    result, direction = fit_cross_fitted_ole_direction_estimator(
        theta_train=theta_train,
        x_train=x_train,
        theta_grid=theta_grid,
        n_splits=int(args.ole_crossfit_folds),
        seed=int(args.ole_crossfit_seed) + int(seed),
        window_radius=args.ole_window_radius,
        min_endpoint_samples=int(args.ole_min_endpoint_samples),
        period=period,
    )
    arrays = {
        "ole_direction": np.asarray(direction, dtype=np.float64),
        "ole_fisher": np.asarray(result.linear_fisher, dtype=np.float64),
        "ole_fisher_raw": np.asarray(result.linear_fisher_raw, dtype=np.float64),
        "ole_fold_weights": np.asarray(result.fold_weights, dtype=np.float64),
        "ole_fold_intercepts": np.asarray(result.fold_intercepts, dtype=np.float64),
        "ole_fold_test_counts_left": np.asarray(
            result.fold_test_counts_left, dtype=np.int64
        ),
        "ole_fold_test_counts_right": np.asarray(
            result.fold_test_counts_right, dtype=np.int64
        ),
    }
    np.savez_compressed(estimates_path, **arrays)
    metadata_path.write_text(
        json.dumps({"signature": signature}, indent=2) + "\n",
        encoding="utf-8",
    )
    return arrays


def _fit_gkr_only_estimator(
    *,
    case_dir: Path,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_grid: np.ndarray,
    period: float | None,
    seed: int,
    device: torch.device,
    force_fit: bool,
) -> dict[str, Any]:
    fit_dir = case_dir / "fit"
    fit_dir.mkdir(parents=True, exist_ok=True)
    estimates_path = fit_dir / "gkr_only_estimates.npz"
    metadata_path = fit_dir / "gkr_only_metadata.json"
    model_path = fit_dir / "gkr_only_model.pt"
    signature = {
        "n_train": int(np.asarray(x_train).shape[0]),
        "response_dim": int(np.asarray(x_train).shape[1]),
        "theta_grid": np.asarray(theta_grid, dtype=np.float64).reshape(-1).tolist(),
        "period": period,
        "seed": int(seed),
    }
    if (
        all(path.is_file() for path in (estimates_path, metadata_path, model_path))
        and not force_fit
    ):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            with np.load(estimates_path) as saved:
                return {key: np.asarray(saved[key]) for key in saved.files} | {
                    "metadata": metadata
                }

    model, estimate, direction = fit_gkr_direction_estimator(
        theta_train=theta_train,
        x_train=x_train,
        theta_grid=theta_grid,
        device=device,
        seed=seed,
        circular_period=period,
    )
    arrays = {
        "gkr_fisher": np.asarray(estimate.linear_fisher, dtype=np.float64).reshape(-1),
        "gkr_direction": np.asarray(direction, dtype=np.float64),
        "gkr_mean_loss": np.asarray(estimate.mean_loss, dtype=np.float64),
        "gkr_covariance_loss": np.asarray(estimate.covariance_loss, dtype=np.float64),
    }
    np.savez_compressed(estimates_path, **arrays)
    torch.save(gkr_checkpoint(model), model_path)
    metadata = {"signature": signature}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays | {"metadata": metadata}


def _sample_toy_endpoints(population: Any, grid: np.ndarray, n: int, seed: int) -> np.ndarray:
    population.rng = np.random.default_rng(int(seed))
    return np.stack(
        [population.sample_x(np.full((int(n), 1), value, dtype=np.float64)) for value in grid[:, 0]],
        axis=0,
    )


def _prepare_toy(
    args: argparse.Namespace,
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, Any]:
    source_dir = args.output_dir / "source" / args.dataset / f"seed{seed}"
    source_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = source_dir / "dataset.npz"
    make_native_dataset_npz(
        output_npz=dataset_path,
        dataset_family=args.dataset,
        x_dim=args.x_dim,
        n_total=args.n_total,
        train_frac=0.8,
        seed=seed,
        force=args.force_source,
    )
    bundle = load_shared_dataset_npz(dataset_path)
    split = stratified_train_validation_test_split(
        bundle.theta_all,
        n_strata=20,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
        fixed_partition=(
            "validation" if args.direction == "train-test-allocation" else "test"
        ),
    )
    grid = theta_grid_from_meta(bundle.meta, theta_grid_size=args.theta_grid_size)
    population = build_dataset_from_meta(dict(bundle.meta))
    calibration = _sample_toy_endpoints(
        population, grid, args.endpoint_calibration_samples, seed + 900_000
    )
    if args.direction == "train-test-allocation":
        test_fraction = 1.0 - float(train_fraction) - float(validation_fraction)
        max_test_fraction = max(float(value) for value in args.test_fractions)
        max_test_per_endpoint = max(
            2,
            int(np.ceil(args.n_total * max_test_fraction / grid.shape[0])),
        )
        test_per_endpoint = max(
            2,
            int(np.floor(args.n_total * test_fraction / grid.shape[0] + 1e-9)),
        )
        test_pool = _sample_toy_endpoints(
            population,
            grid,
            max_test_per_endpoint,
            seed + 1_000_000,
        )
        test = test_pool[:, :test_per_endpoint]
    else:
        test = _sample_toy_endpoints(
            population,
            grid,
            args.endpoint_test_samples,
            seed + 1_000_000,
        )
    mean_left, _, covariance_left = population_linear_moments(population, grid[:-1])
    mean_right, _, covariance_right = population_linear_moments(population, grid[1:])
    oracle_direction, oracle_fisher = finite_endpoint_oracle(
        mean_left,
        mean_right,
        covariance_left,
        covariance_right,
        np.diff(grid[:, 0]),
    )
    return {
        "theta_train": bundle.theta_all[split.train],
        "x_train": bundle.x_all[split.train],
        "theta_validation": bundle.theta_all[split.validation],
        "x_validation": bundle.x_all[split.validation],
        "theta_test": bundle.theta_all[split.test],
        "x_test": bundle.x_all[split.test],
        "grid": grid,
        "period": None,
        "calibration_endpoint": calibration,
        "test_endpoint": test,
        "oracle_direction": oracle_direction,
        "oracle_fisher": oracle_fisher,
        "split": split,
        "source": str(dataset_path),
    }


def _prepare_stringer(
    args: argparse.Namespace,
    *,
    session: Any,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    case_dir: Path,
) -> dict[str, Any]:
    period = float(np.pi)
    theta_all = np.asarray(session.grating_orientation, dtype=np.float64)
    response_all = np.asarray(session.neural_responses, dtype=np.float64)
    split = stratified_train_validation_test_split(
        theta_all,
        n_strata=args.stringer_grid_size - 1,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
        period=period,
        fixed_partition=(
            "validation" if args.direction == "train-test-allocation" else "test"
        ),
    )
    pca = PCA(
        n_components=args.pca_dim,
        whiten=bool(args.pca_whiten),
        svd_solver="randomized",
        random_state=seed,
    )
    pca.fit(response_all[split.train])
    x_train = pca.transform(response_all[split.train]).astype(np.float64)
    x_validation = pca.transform(response_all[split.validation]).astype(np.float64)
    x_test = pca.transform(response_all[split.test]).astype(np.float64)
    case_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        case_dir / "split_and_pca.npz",
        train_index=split.train,
        validation_index=split.validation,
        test_index=split.test,
        stratum=split.stratum,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        pca_explained_variance=pca.explained_variance_,
        pca_whiten=np.asarray(bool(args.pca_whiten)),
    )
    return {
        "theta_train": theta_all[split.train],
        "x_train": x_train,
        "theta_validation": theta_all[split.validation],
        "x_validation": x_validation,
        "theta_test": theta_all[split.test],
        "x_test": x_test,
        "grid": np.linspace(0.0, period, args.stringer_grid_size, dtype=np.float64).reshape(-1, 1),
        "period": period,
        "split": split,
        "source": str(session.session_file),
    }


def _method_row(
    *,
    args: argparse.Namespace,
    prepared: dict[str, Any],
    fit: dict[str, Any],
    method: str,
    direction: np.ndarray,
    predicted_fisher: np.ndarray,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, Any]:
    grid = np.asarray(prepared["grid"], dtype=np.float64)
    dtheta = np.diff(grid[:, 0])
    if prepared["period"] is None:
        decoder = evaluate_endpoint_decoders(
            direction,
            prepared["test_endpoint"][:-1],
            prepared["test_endpoint"][1:],
            dtheta,
        )
        threshold = evaluate_linear_threshold_decoders(
            direction,
            prepared["calibration_endpoint"][:-1],
            prepared["calibration_endpoint"][1:],
            prepared["test_endpoint"][:-1],
            prepared["test_endpoint"][1:],
        )
    else:
        half_width = 0.5 * float(dtheta[0])
        decoder = evaluate_windowed_decoders(
            direction,
            prepared["x_test"],
            prepared["theta_test"],
            grid[:-1, 0],
            grid[1:, 0],
            half_width=half_width,
            period=float(prepared["period"]),
        )
        threshold = evaluate_windowed_linear_threshold_decoders(
            direction,
            prepared["x_train"],
            prepared["theta_train"],
            prepared["x_test"],
            prepared["theta_test"],
            grid[:-1, 0],
            grid[1:, 0],
            half_width=half_width,
            period=float(prepared["period"]),
        )
    predicted = np.asarray(predicted_fisher, dtype=np.float64).reshape(-1)
    predicted_error = fisher_predicted_linear_error(predicted, dtheta)
    achieved_error = fisher_predicted_linear_error(decoder.achieved_fisher_display, dtheta)
    training = fit.get("metadata", {}).get("flow_training", {}) if method == "Flow matching" else {}
    return {
        "dataset": DATASET_LABELS[args.dataset],
        "dataset_token": args.dataset,
        "seed": int(seed),
        "method": method,
        "train_fraction": float(train_fraction),
        "validation_fraction": float(validation_fraction),
        "test_fraction": float(1.0 - train_fraction - validation_fraction),
        "n_train": int(prepared["x_train"].shape[0]),
        "n_validation": int(prepared["x_validation"].shape[0]),
        "n_test": int(prepared["x_test"].shape[0]),
        "mean_test_samples_per_endpoint": float(
            np.mean(np.concatenate([decoder.n_left, decoder.n_right]))
        ),
        "min_test_samples_per_endpoint": int(
            np.min(np.concatenate([decoder.n_left, decoder.n_right]))
        ),
        "selected_epoch": training.get("selected_epoch"),
        "best_validation_loss": training.get("best_val_loss"),
        "theta_midpoints": (0.5 * (grid[:-1, 0] + grid[1:, 0])).tolist(),
        "dtheta": dtheta.tolist(),
        "predicted_fisher": predicted.tolist(),
        "achieved_fisher": decoder.achieved_fisher_raw.tolist(),
        "achieved_fisher_display": decoder.achieved_fisher_display.tolist(),
        "auc": decoder.roc_auc.tolist(),
        "balanced_error": threshold.balanced_error.tolist(),
        "false_positive_rate": threshold.false_positive_rate.tolist(),
        "false_negative_rate": threshold.false_negative_rate.tolist(),
        "predicted_error": predicted_error.tolist(),
        "achieved_predicted_error": achieved_error.tolist(),
        "mean_achieved_fisher": float(np.mean(decoder.achieved_fisher_raw)),
        "median_achieved_fisher": float(np.median(decoder.achieved_fisher_raw)),
        "worst_decile_achieved_fisher": float(np.quantile(decoder.achieved_fisher_raw, 0.1)),
        "mean_auc": float(np.mean(decoder.roc_auc)),
        "mean_balanced_error": float(np.mean(threshold.balanced_error)),
        "mean_absolute_error_calibration": float(np.mean(np.abs(predicted_error - threshold.balanced_error))),
    }


def _run_case(
    args: argparse.Namespace,
    *,
    session: Any | None,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    ratio = _ratio_token(train_fraction, validation_fraction)
    cache_name = (
        "cache_train_test_allocation"
        if args.direction == "train-test-allocation"
        else "cache"
    )
    case_dir = args.output_dir / cache_name / args.dataset / f"seed{seed}" / ratio
    if args.dataset == "stringer":
        prepared = _prepare_stringer(
            args,
            session=session,
            seed=seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            case_dir=case_dir,
        )
    else:
        prepared = _prepare_toy(
            args,
            seed=seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
    if args.gkr_only:
        fit = _fit_gkr_only_estimator(
            case_dir=case_dir,
            theta_train=prepared["theta_train"],
            x_train=prepared["x_train"],
            theta_grid=prepared["grid"],
            period=prepared["period"],
            seed=seed,
            device=device,
            force_fit=bool(args.force_fit),
        )
        return [
            _method_row(
                args=args,
                prepared=prepared,
                fit=fit,
                method="GKR",
                direction=np.asarray(fit["gkr_direction"]),
                predicted_fisher=np.asarray(fit["gkr_fisher"]),
                seed=seed,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
            )
        ]

    fit = _fit_estimators(
        case_dir=case_dir,
        theta_train=prepared["theta_train"],
        x_train=prepared["x_train"],
        theta_validation=prepared["theta_validation"],
        x_validation=prepared["x_validation"],
        theta_grid=prepared["grid"],
        period=prepared["period"],
        args=args,
        seed=seed,
        device=device,
    )
    if args.fm_only:
        return [
            _method_row(
                args=args,
                prepared=prepared,
                fit=fit,
                method="Flow matching",
                direction=np.asarray(fit["flow_direction"]),
                predicted_fisher=np.asarray(fit["flow_fisher"]),
                seed=seed,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
            )
        ]

    ole_theta = np.concatenate(
        [prepared["theta_train"], prepared["theta_validation"]], axis=0
    )
    ole_x = np.concatenate(
        [prepared["x_train"], prepared["x_validation"]], axis=0
    )
    ole = _fit_ole_estimator(
        case_dir=case_dir,
        theta_train=ole_theta,
        x_train=ole_x,
        theta_grid=prepared["grid"],
        period=prepared["period"],
        args=args,
        seed=seed,
    )
    rows = []
    fitted_methods = [("Flow matching", "flow_direction", "flow_fisher")]
    if not args.skip_gkr:
        fitted_methods.append(("GKR", "gkr_direction", "gkr_fisher"))
    for method, direction_key, fisher_key in fitted_methods:
        rows.append(
            _method_row(
                args=args,
                prepared=prepared,
                fit=fit,
                method=method,
                direction=np.asarray(fit[direction_key]),
                predicted_fisher=np.asarray(fit[fisher_key]),
                seed=seed,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
            )
        )
    ole_row = _method_row(
        args=args,
        prepared=prepared,
        fit=fit,
        method="OLE (cross-fit)",
        direction=np.asarray(ole["ole_direction"]),
        predicted_fisher=np.asarray(ole["ole_fisher"]),
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    ole_row["ole_fit_pool_size"] = int(ole_x.shape[0])
    ole_row["ole_fit_pool"] = "train+validation"
    rows.append(ole_row)
    if args.dataset != "stringer":
        rows.append(
            _method_row(
                args=args,
                prepared=prepared,
                fit=fit,
                method="Oracle",
                direction=np.asarray(prepared["oracle_direction"]),
                predicted_fisher=np.asarray(prepared["oracle_fisher"]),
                seed=seed,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
            )
        )
    case_summary = {
        "source": prepared["source"],
        "seed": int(seed),
        "train_fraction": float(train_fraction),
        "validation_fraction": float(validation_fraction),
        "test_fraction": float(1.0 - train_fraction - validation_fraction),
        "rows": rows,
    }
    (case_dir / "evaluation.json").write_text(
        json.dumps(case_summary, indent=2) + "\n", encoding="utf-8"
    )
    return rows


def main() -> None:
    args = parse_args()
    device = require_device(args.device)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = None
    if args.dataset == "stringer":
        session = load_stringer_session(
            None,
            session_stimuli_type="gratings_static",
            session_index=args.session_index,
            orientation_period=np.pi,
        )
    if args.direction == "reference":
        allocations = [(args.train_fraction, args.validation_fraction)]
    elif args.direction == "validation-allocation":
        allocations = list(args.allocations)
    else:
        allocations = []
        for test_fraction in args.test_fractions:
            train_fraction = 1.0 - float(args.validation_fraction) - float(test_fraction)
            if min(train_fraction, float(args.validation_fraction), float(test_fraction)) <= 0.0:
                raise ValueError(
                    "Each train-test allocation must leave positive train, validation, and test fractions."
                )
            allocations.append((train_fraction, float(args.validation_fraction)))
    rows: list[dict[str, Any]] = []
    for train_fraction, validation_fraction in allocations:
        for seed in args.seeds:
            print(
                f"[{args.direction}] dataset={args.dataset} seed={seed} "
                f"split={train_fraction:.2f}/{validation_fraction:.2f}/"
                f"{1.0 - train_fraction - validation_fraction:.2f}",
                flush=True,
            )
            rows.extend(
                _run_case(
                    args,
                    session=session,
                    seed=seed,
                    train_fraction=train_fraction,
                    validation_fraction=validation_fraction,
                    device=device,
                )
            )
    summary = {
        "direction": args.direction,
        "dataset": DATASET_LABELS[args.dataset],
        "dataset_token": args.dataset,
        "device": str(device),
        "rows": rows,
        "config": {**vars(args), "output_dir": str(args.output_dir)},
    }
    suffix = "_gkr" if args.gkr_only else "_fm" if args.fm_only else ""
    summary_path = args.output_dir / f"{args.direction}_{args.dataset}{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
