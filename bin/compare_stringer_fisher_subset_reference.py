#!/usr/bin/env python3
"""Compare full-Fisher estimators on five disjoint Stringer subsets per K."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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

from fisher.fisher_validation import stratified_disjoint_subset_indices
from fisher.flow_matching_skl import (
    build_flow_skl_model,
    estimate_adjacent_model_jeffreys_fisher,
    train_flow_skl_model,
)
from fisher.shared_fisher_est import require_device
from fisher.stringer_dataset import load_stringer_session
from fisher.stringer_session_identification import encode_flow_orientation
from fisher.tre_distance import TREDensityRatioConfig, train_and_estimate_binned_tre_fisher
from global_setting import DATA_DIR, DEFAULT_EARLY_STOPPING_PATIENCE, DEFAULT_TRAINING_MAX_EPOCHS

ORIENTATION_PERIOD = float(np.pi)
N_SUBSETS = 5
METHOD_FLOW = "Flow matching"
METHOD_TRE = "Binned TRE-8"
METHODS = (METHOD_FLOW, METHOD_TRE)
COLORS = {METHOD_FLOW: "C0", METHOD_TRE: "C1"}


def _csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "stringer_full_fisher_subset_reference_pca82",
    )
    parser.add_argument("--session-index", type=int, default=0)
    parser.add_argument("--pca-dim", type=int, default=82)
    parser.add_argument("--pca-whiten", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--theta-grid-size", type=int, default=17)
    parser.add_argument("--subset-sizes", type=_csv_ints, default=[200, 400, 600, 800])
    parser.add_argument("--subset-seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.8)

    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_MAX_EPOCHS)
    parser.add_argument(
        "--early-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE
    )
    parser.add_argument("--flow-batch-size", type=int, default=2_048)
    parser.add_argument("--flow-learning-rate", type=float, default=1e-4)
    parser.add_argument("--flow-hidden-dim", type=int, default=256)
    parser.add_argument("--flow-depth", type=int, default=5)
    parser.add_argument("--flow-ode-steps", type=int, default=32)
    parser.add_argument("--flow-hutchinson-probes", type=int, default=4)
    parser.add_argument("--flow-mc-jeffreys-samples", type=int, default=4_096)

    parser.add_argument("--tre-num-bridges", type=int, default=8)
    parser.add_argument("--tre-hidden-dim", type=int, default=128)
    parser.add_argument("--tre-depth", type=int, default=3)
    parser.add_argument("--tre-batch-size", type=int, default=512)
    parser.add_argument("--tre-learning-rate", type=float, default=1e-3)
    parser.add_argument("--tre-validation-pairs", type=int, default=2_048)
    parser.add_argument("--min-window-samples", type=int, default=2)

    parser.add_argument("--force-pca", action="store_true")
    parser.add_argument("--force-reference", action="store_true")
    parser.add_argument("--force-fit", action="store_true")
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _validate_args(args: argparse.Namespace, *, n_observations: int, n_features: int) -> None:
    if int(args.pca_dim) < 1 or int(args.pca_dim) > min(n_observations, n_features):
        raise ValueError("--pca-dim is outside the valid PCA range.")
    if int(args.theta_grid_size) < 3:
        raise ValueError("--theta-grid-size must be at least 3.")
    if not 0.0 < float(args.train_fraction) < 1.0:
        raise ValueError("--train-fraction must be in (0, 1).")
    if len(set(args.subset_sizes)) != len(args.subset_sizes):
        raise ValueError("--subset-sizes must not contain duplicates.")
    largest_allowed = n_observations // N_SUBSETS
    if any(int(value) < 2 * (int(args.theta_grid_size) - 1) for value in args.subset_sizes):
        raise ValueError("Each K must be at least twice the number of orientation intervals.")
    if any(int(value) > largest_allowed for value in args.subset_sizes):
        raise ValueError(
            f"Each K must satisfy K <= full dataset size // {N_SUBSETS} = {largest_allowed}."
        )


def _preprocess_full_session(
    args: argparse.Namespace,
    *,
    theta: np.ndarray,
    responses: np.ndarray,
    session_file: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    output = args.output_dir / f"full_pca{int(args.pca_dim)}.npz"
    metadata_path = args.output_dir / f"full_pca{int(args.pca_dim)}_metadata.json"
    signature = {
        "session_file": str(session_file),
        "n_observations": int(responses.shape[0]),
        "n_neurons": int(responses.shape[1]),
        "pca_dim": int(args.pca_dim),
        "pca_whiten": bool(args.pca_whiten),
        "random_state": 0,
    }
    if output.is_file() and metadata_path.is_file() and not args.force_pca:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            with np.load(output, allow_pickle=False) as saved:
                np.testing.assert_allclose(saved["theta"], theta)
                return np.asarray(saved["x"], dtype=np.float32), metadata

    pca = PCA(
        n_components=int(args.pca_dim),
        whiten=bool(args.pca_whiten),
        svd_solver="randomized",
        random_state=0,
    )
    x = pca.fit_transform(np.asarray(responses)).astype(np.float32)
    np.savez_compressed(
        output,
        theta=np.asarray(theta, dtype=np.float64),
        x=x,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        pca_explained_variance=pca.explained_variance_,
        pca_explained_variance_ratio=pca.explained_variance_ratio_,
        pca_whiten=np.asarray(bool(args.pca_whiten)),
    )
    metadata = {
        "signature": signature,
        "artifact": str(output),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return x, metadata


def _train_validation_indices(
    theta: np.ndarray, *, train_fraction: float, n_strata: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    n_train = min(max(int(round(float(train_fraction) * theta.size)), 1), theta.size - 1)
    train = stratified_disjoint_subset_indices(
        theta,
        n_train,
        n_subsets=1,
        n_strata=int(n_strata),
        seed=int(seed),
        period=ORIENTATION_PERIOD,
    )[0]
    validation = np.setdiff1d(np.arange(theta.size, dtype=np.int64), train)
    return train, validation


def _tre_config(args: argparse.Namespace) -> TREDensityRatioConfig:
    return TREDensityRatioConfig(
        num_bridges=int(args.tre_num_bridges),
        waymark_schedule="angle",
        architecture="mlp",
        hidden_dim=int(args.tre_hidden_dim),
        depth=int(args.tre_depth),
        epochs=int(args.epochs),
        batch_size=int(args.tre_batch_size),
        lr=float(args.tre_learning_rate),
        weight_decay=0.0,
        early_patience=int(args.early_patience),
        early_min_delta=1e-5,
        max_grad_norm=10.0,
        validation_pairs=int(args.tre_validation_pairs),
        standardize=True,
        log_every=1_000,
    )


def _fit_tre(
    args: argparse.Namespace,
    *,
    theta: np.ndarray,
    x: np.ndarray,
    theta_grid: np.ndarray,
    case_dir: Path,
    seed: int,
    signature: dict[str, Any],
    device: torch.device,
    force: bool,
) -> dict[str, Any]:
    estimates_path = case_dir / "tre_estimates.npz"
    metadata_path = case_dir / "tre_metadata.json"
    if estimates_path.is_file() and metadata_path.is_file() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            with np.load(estimates_path, allow_pickle=False) as saved:
                return {key: np.asarray(saved[key]) for key in saved.files} | {"metadata": metadata}

    train, validation = _train_validation_indices(
        theta,
        train_fraction=float(args.train_fraction),
        n_strata=int(args.theta_grid_size) - 1,
        seed=int(seed) + 30_000,
    )
    states, result = train_and_estimate_binned_tre_fisher(
        theta_train=theta[train],
        x_train=x[train],
        theta_validation=theta[validation],
        x_validation=x[validation],
        theta_eval=theta,
        x_eval=x,
        theta_grid=theta_grid,
        theta_period=ORIENTATION_PERIOD,
        device=device,
        seed=int(seed),
        config=_tre_config(args),
        min_train_samples=int(args.min_window_samples),
        min_validation_samples=int(args.min_window_samples),
        min_eval_samples=int(args.min_window_samples),
    )
    arrays = {
        "theta_grid": theta_grid,
        "theta_midpoints": 0.5 * (theta_grid[:-1] + theta_grid[1:]),
        "tre_full_fisher": np.asarray(result.fisher, dtype=np.float64),
        "tre_jeffreys": np.asarray(result.jeffreys, dtype=np.float64),
        "tre_raw_jeffreys": np.asarray(result.raw_jeffreys, dtype=np.float64),
        "train_index": train,
        "validation_index": validation,
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(estimates_path, **arrays)
    torch.save(
        {
            "pair_state_dicts": states,
            "pair_metadata": result.pair_metadata,
            "run_metadata": result.run_metadata,
        },
        case_dir / "tre_models.pt",
    )
    metadata = {
        "signature": signature,
        "n_train": int(train.size),
        "n_validation": int(validation.size),
        "run_metadata": result.run_metadata,
        "pair_metadata": result.pair_metadata,
    }
    metadata_path.write_text(json.dumps(_json_ready(metadata), indent=2) + "\n", encoding="utf-8")
    return arrays | {"metadata": metadata}


def _fit_flow(
    args: argparse.Namespace,
    *,
    theta: np.ndarray,
    x: np.ndarray,
    theta_grid: np.ndarray,
    case_dir: Path,
    seed: int,
    signature: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    estimates_path = case_dir / "flow_estimates.npz"
    metadata_path = case_dir / "flow_metadata.json"
    if estimates_path.is_file() and metadata_path.is_file() and not args.force_fit:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            with np.load(estimates_path, allow_pickle=False) as saved:
                return {key: np.asarray(saved[key]) for key in saved.files} | {"metadata": metadata}

    train, validation = _train_validation_indices(
        theta,
        train_fraction=float(args.train_fraction),
        n_strata=int(args.theta_grid_size) - 1,
        seed=int(seed) + 30_000,
    )
    condition = encode_flow_orientation(
        theta, period=ORIENTATION_PERIOD, encoding="periodic-rbf"
    )
    condition_grid = encode_flow_orientation(
        theta_grid, period=ORIENTATION_PERIOD, encoding="periodic-rbf"
    )
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = build_flow_skl_model(
        velocity_family="nonlinear",
        theta_dim=int(condition.shape[1]),
        x_dim=int(x.shape[1]),
        hidden_dim=int(args.flow_hidden_dim),
        depth=int(args.flow_depth),
        path_schedule="cosine",
        divergence_estimator="hutchinson",
        hutchinson_probes=int(args.flow_hutchinson_probes),
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
        batch_size=int(args.flow_batch_size),
        lr=float(args.flow_learning_rate),
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
        validation_seed=int(seed) + 10_000,
        retain_best_state=True,
    )
    best_state = training.pop("best_state_dict")
    torch.save(model.state_dict(), case_dir / "flow_model_last.pt")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), case_dir / "flow_model_best.pt")
    estimate = estimate_adjacent_model_jeffreys_fisher(
        model=model,
        theta_all=theta_grid,
        condition_all=condition_grid,
        device=device,
        mc_jeffreys_sample=int(args.flow_mc_jeffreys_samples),
        ode_steps=int(args.flow_ode_steps),
        ode_method="midpoint",
        batch_size=1_024,
        solve_jitter=1e-6,
        quadrature_steps=64,
    )
    arrays = {
        "theta_grid": theta_grid,
        "theta_midpoints": np.asarray(estimate["theta_midpoints"], dtype=np.float64),
        "flow_full_fisher": np.asarray(estimate["fisher"], dtype=np.float64),
        "flow_adjacent_jeffreys": np.asarray(estimate["adjacent_jeffreys"], dtype=np.float64),
        "train_losses": np.asarray(training["train_losses"], dtype=np.float64),
        "validation_losses": np.asarray(training["val_losses"], dtype=np.float64),
        "train_index": train,
        "validation_index": validation,
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(estimates_path, **arrays)
    metadata = {
        "signature": signature,
        "n_train": int(train.size),
        "n_validation": int(validation.size),
        "best_epoch": int(training["best_epoch"]),
        "stopped_epoch": int(training["stopped_epoch"]),
        "best_validation_loss": float(training["best_val_loss"]),
        "condition_encoding": "periodic-rbf8",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays | {"metadata": metadata}


def _common_signature(args: argparse.Namespace, *, n_total: int, seed: int) -> dict[str, Any]:
    return {
        "n_total": int(n_total),
        "seed": int(seed),
        "session_index": int(args.session_index),
        "pca_dim": int(args.pca_dim),
        "pca_whiten": bool(args.pca_whiten),
        "theta_grid_size": int(args.theta_grid_size),
        "theta_period": ORIENTATION_PERIOD,
        "train_fraction": float(args.train_fraction),
        "max_epochs": int(args.epochs),
        "early_stopping_patience": int(args.early_patience),
        "tre_num_bridges": int(args.tre_num_bridges),
        "tre_hidden_dim": int(args.tre_hidden_dim),
        "tre_depth": int(args.tre_depth),
        "tre_batch_size": int(args.tre_batch_size),
        "tre_learning_rate": float(args.tre_learning_rate),
        "tre_validation_pairs": int(args.tre_validation_pairs),
        "min_window_samples": int(args.min_window_samples),
    }


def _flow_signature(args: argparse.Namespace, *, n_total: int, seed: int) -> dict[str, Any]:
    return _common_signature(args, n_total=n_total, seed=seed) | {
        "flow_hidden_dim": int(args.flow_hidden_dim),
        "flow_depth": int(args.flow_depth),
        "flow_batch_size": int(args.flow_batch_size),
        "flow_learning_rate": float(args.flow_learning_rate),
        "flow_ode_steps": int(args.flow_ode_steps),
        "flow_hutchinson_probes": int(args.flow_hutchinson_probes),
        "flow_mc_jeffreys_samples": int(args.flow_mc_jeffreys_samples),
        "flow_condition": "periodic-rbf8",
    }


def _rows_for_case(
    *,
    k: int,
    repeat: int,
    subset_seed: int,
    subset_index: np.ndarray,
    reference: np.ndarray,
    flow: dict[str, Any],
    tre: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, curve in (
        (METHOD_FLOW, np.asarray(flow["flow_full_fisher"], dtype=np.float64)),
        (METHOD_TRE, np.asarray(tre["tre_full_fisher"], dtype=np.float64)),
    ):
        rows.append(
            {
                "method": method,
                "K": int(k),
                "repeat": int(repeat),
                "subset_seed": int(subset_seed),
                "n_unique": int(np.unique(subset_index).size),
                "mean_full_fisher": float(np.mean(curve)),
                "absolute_error_to_reference_mean": float(abs(np.mean(curve) - np.mean(reference))),
                "curve_mae_to_reference": float(np.mean(np.abs(curve - reference))),
                "curve": curve.tolist(),
            }
        )
    return rows


def _write_rows_csv(rows: list[dict[str, Any]], output: Path) -> Path:
    fields = (
        "method",
        "K",
        "repeat",
        "subset_seed",
        "n_unique",
        "mean_full_fisher",
        "absolute_error_to_reference_mean",
        "curve_mae_to_reference",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    return output


def _plot(
    rows: list[dict[str, Any]], reference: np.ndarray, output_dir: Path, *, pca_dim: int
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 13,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axis = plt.subplots(figsize=(4.0, 3.5), constrained_layout=True)
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        sizes = sorted({int(row["K"]) for row in selected})
        means, standard_deviations = [], []
        for size in sizes:
            values = np.asarray(
                [row["mean_full_fisher"] for row in selected if int(row["K"]) == size],
                dtype=np.float64,
            )
            means.append(float(np.mean(values)))
            standard_deviations.append(float(np.std(values, ddof=1)))
        axis.errorbar(
            sizes,
            means,
            yerr=standard_deviations,
            color=COLORS[method],
            marker="o",
            markersize=5,
            linewidth=2.0,
            capsize=3,
            label=method,
        )
    axis.axhline(
        float(np.mean(reference)),
        color="0.15",
        linestyle="--",
        linewidth=1.8,
        label="Full-data TRE-8",
    )
    axis.set_xlabel("Subset size $K$")
    axis.set_ylabel("Mean full Fisher information")
    axis.set_title(f"Stringer, PCA {int(pca_dim)}D")
    axis.legend(frameon=False)
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color="0.88", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines["left"].set_linewidth(1.8)
    axis.spines["bottom"].set_linewidth(1.8)
    axis.tick_params(width=1.8)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "stringer_full_fisher_vs_subset_size"
    png, svg = stem.with_suffix(".png"), stem.with_suffix(".svg")
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def main() -> int:
    args = parse_args()
    device = require_device(str(args.device))
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    session = load_stringer_session(
        None,
        session_stimuli_type="gratings_static",
        session_index=int(args.session_index),
        orientation_period=ORIENTATION_PERIOD,
    )
    theta = np.asarray(session.grating_orientation, dtype=np.float64)
    responses = np.asarray(session.neural_responses)
    _validate_args(
        args, n_observations=int(theta.size), n_features=int(responses.shape[1])
    )
    x, pca_metadata = _preprocess_full_session(
        args,
        theta=theta,
        responses=responses,
        session_file=Path(session.session_file),
    )
    theta_grid = np.linspace(
        0.0, ORIENTATION_PERIOD, int(args.theta_grid_size), dtype=np.float64
    ).reshape(-1, 1)

    reference_seed = int(args.subset_seed) + 900_000
    reference_signature = _common_signature(
        args, n_total=int(theta.size), seed=reference_seed
    ) | {"role": "full_data_binned_tre_reference"}
    print(f"[reference] binned TRE-8 n={theta.size}", flush=True)
    reference_fit = _fit_tre(
        args,
        theta=theta,
        x=x,
        theta_grid=theta_grid,
        case_dir=args.output_dir / "full_data_tre_reference",
        seed=reference_seed,
        signature=reference_signature,
        device=device,
        force=bool(args.force_reference),
    )
    reference = np.asarray(reference_fit["tre_full_fisher"], dtype=np.float64)

    rows: list[dict[str, Any]] = []
    subset_indices: dict[str, list[int]] = {}
    for k in sorted(int(value) for value in args.subset_sizes):
        allocation_seed = int(args.subset_seed) + 100_003 * k
        subsets = stratified_disjoint_subset_indices(
            theta,
            k,
            n_subsets=N_SUBSETS,
            n_strata=int(args.theta_grid_size) - 1,
            seed=allocation_seed,
            period=ORIENTATION_PERIOD,
        )
        if np.unique(np.concatenate(subsets)).size != N_SUBSETS * k:
            raise AssertionError(f"Subsets for K={k} are not pairwise disjoint.")
        for repeat, index in enumerate(subsets):
            fit_seed = int(args.subset_seed) + 100_000 * k + repeat
            case_dir = args.output_dir / "subsets" / f"K{k}" / f"repeat{repeat}"
            case_dir.mkdir(parents=True, exist_ok=True)
            subset_indices[f"K{k}_repeat{repeat}"] = index.tolist()
            np.save(case_dir / "full_dataset_indices.npy", index)
            print(f"[subset] K={k} repeat={repeat + 1}/{N_SUBSETS}", flush=True)
            flow_fit = _fit_flow(
                args,
                theta=theta[index],
                x=x[index],
                theta_grid=theta_grid,
                case_dir=case_dir,
                seed=fit_seed,
                signature=_flow_signature(args, n_total=k, seed=fit_seed),
                device=device,
            )
            tre_fit = _fit_tre(
                args,
                theta=theta[index],
                x=x[index],
                theta_grid=theta_grid,
                case_dir=case_dir,
                seed=fit_seed,
                signature=_common_signature(args, n_total=k, seed=fit_seed)
                | {"role": "subset_binned_tre"},
                device=device,
                force=bool(args.force_fit),
            )
            rows.extend(
                _rows_for_case(
                    k=k,
                    repeat=repeat,
                    subset_seed=allocation_seed,
                    subset_index=index,
                    reference=reference,
                    flow=flow_fit,
                    tre=tre_fit,
                )
            )

    csv_path = _write_rows_csv(rows, args.output_dir / "full_fisher_by_subset.csv")
    png, svg = _plot(rows, reference, args.output_dir / "figures", pca_dim=int(args.pca_dim))
    summary = {
        "session_file": str(session.session_file),
        "n_full": int(theta.size),
        "response_dim_original": int(responses.shape[1]),
        "response_dim_pca": int(x.shape[1]),
        "pca": pca_metadata,
        "orientation_period": ORIENTATION_PERIOD,
        "theta_grid": theta_grid[:, 0].tolist(),
        "theta_midpoints": (0.5 * (theta_grid[:-1, 0] + theta_grid[1:, 0])).tolist(),
        "n_subsets_per_K": N_SUBSETS,
        "sampling": "without replacement and pairwise disjoint within each K",
        "full_data_tre_reference_curve": reference.tolist(),
        "full_data_tre_reference_mean": float(np.mean(reference)),
        "rows": rows,
        "subset_indices": subset_indices,
        "config": _json_ready(vars(args)),
        "runtime_seconds": float(time.perf_counter() - started),
        "artifacts": {
            "summary_json": str(args.output_dir / "stringer_full_fisher_subset_reference_summary.json"),
            "rows_csv": str(csv_path),
            "figure_png": str(png),
            "figure_svg": str(svg),
            "reference_npz": str(args.output_dir / "full_data_tre_reference" / "tre_estimates.npz"),
        },
    }
    summary_path = args.output_dir / "stringer_full_fisher_subset_reference_summary.json"
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("n_full", "response_dim_pca", "full_data_tre_reference_mean", "artifacts")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
