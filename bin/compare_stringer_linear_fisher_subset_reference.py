#!/usr/bin/env python3
"""Compare subset linear-Fisher estimates with a full-data cross-fitted OLE reference."""

from __future__ import annotations

import argparse
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

from fisher.fisher_validation import (
    fit_cross_fitted_ole_direction_estimator,
    fit_flow_direction_estimator,
    fit_gkr_direction_estimator,
    gkr_checkpoint,
    stratified_disjoint_subset_indices,
)
from fisher.flow_matching_skl import DEFAULT_AFFINE_COVARIANCE_ODE_STEPS, build_flow_skl_model
from fisher.shared_fisher_est import require_device
from fisher.stringer_dataset import load_stringer_session
from fisher.stringer_session_identification import (
    encode_flow_orientation,
    estimate_affine_mixed_symmetric_kl_fisher_for_conditions,
)
from global_setting import DATA_DIR, DEFAULT_EARLY_STOPPING_PATIENCE, DEFAULT_TRAINING_MAX_EPOCHS

PERIOD = float(np.pi)
METHODS = ("Flow matching", "GKR", "OLE (cross-fit)")
COLORS = {"Flow matching": "C0", "GKR": "C2", "OLE (cross-fit)": "C1"}


def _csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "stringer_linear_fisher_subset_reference_pca10_nowhiten",
    )
    parser.add_argument("--session-index", type=int, default=0)
    parser.add_argument("--pca-dim", type=int, default=10)
    parser.add_argument("--pca-whiten", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--theta-grid-size", type=int, default=17)
    parser.add_argument("--subset-sizes", type=_csv_ints, default=[500, 1000, 2000, 3000, 4000])
    parser.add_argument("--subset-seed", type=int, default=7)
    parser.add_argument("--flow-validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_MAX_EPOCHS)
    parser.add_argument("--early-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--ode-steps", type=int, default=DEFAULT_AFFINE_COVARIANCE_ODE_STEPS)
    parser.add_argument("--ole-crossfit-folds", type=int, default=5)
    parser.add_argument("--ole-crossfit-seed", type=int, default=20_260_721)
    parser.add_argument("--ole-min-endpoint-samples", type=int, default=8)
    parser.add_argument("--force-pca", action="store_true")
    parser.add_argument("--force-reference", action="store_true")
    parser.add_argument("--force-fit", action="store_true")
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


def _nested_subsets(theta: np.ndarray, sizes: list[int], *, n_strata: int, seed: int) -> list[np.ndarray]:
    return [
        stratified_disjoint_subset_indices(
            theta,
            int(size),
            n_subsets=1,
            n_strata=int(n_strata),
            seed=int(seed),
            period=PERIOD,
        )[0]
        for size in sizes
    ]


def _validate(args: argparse.Namespace, *, n_samples: int, n_features: int) -> None:
    if int(args.pca_dim) < 1 or int(args.pca_dim) > min(n_samples, n_features):
        raise ValueError("--pca-dim is outside the valid range.")
    if not 0.0 < float(args.flow_validation_fraction) < 1.0:
        raise ValueError("--flow-validation-fraction must be in (0, 1).")
    if len(set(args.subset_sizes)) != len(args.subset_sizes):
        raise ValueError("--subset-sizes must be unique.")
    if any(size < 2 * args.theta_grid_size or size > n_samples for size in args.subset_sizes):
        raise ValueError("Subset sizes must be feasible for the dataset and orientation grid.")


def _preprocess(
    args: argparse.Namespace,
    *,
    theta: np.ndarray,
    responses: np.ndarray,
    session_file: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    suffix = "whiten" if args.pca_whiten else "centered"
    path = args.output_dir / f"full_pca{args.pca_dim}_{suffix}.npz"
    metadata_path = path.with_name(f"{path.stem}_metadata.json")
    signature = {
        "session_file": str(session_file),
        "shape": list(responses.shape),
        "pca_dim": int(args.pca_dim),
        "pca_whiten": bool(args.pca_whiten),
        "random_state": 0,
    }
    if path.is_file() and metadata_path.is_file() and not args.force_pca:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_allclose(saved["theta"], theta)
                return np.asarray(saved["x"], dtype=np.float64), metadata

    pca = PCA(
        n_components=int(args.pca_dim),
        whiten=bool(args.pca_whiten),
        svd_solver="randomized",
        random_state=0,
    )
    x = pca.fit_transform(responses).astype(np.float64)
    np.savez_compressed(
        path,
        theta=theta,
        x=x,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        pca_explained_variance=pca.explained_variance_,
        pca_explained_variance_ratio=pca.explained_variance_ratio_,
    )
    metadata = {
        "signature": signature,
        "artifact": str(path),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return x, metadata


def _fit_ole(
    args: argparse.Namespace,
    *,
    theta: np.ndarray,
    x: np.ndarray,
    grid: np.ndarray,
    seed: int,
) -> np.ndarray:
    result, _ = fit_cross_fitted_ole_direction_estimator(
        theta_train=theta,
        x_train=x,
        theta_grid=grid,
        n_splits=int(args.ole_crossfit_folds),
        seed=int(seed),
        min_endpoint_samples=int(args.ole_min_endpoint_samples),
        period=PERIOD,
    )
    return np.asarray(result.linear_fisher, dtype=np.float64).reshape(-1)


def _reference(
    args: argparse.Namespace,
    *,
    theta: np.ndarray,
    x: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    path = args.output_dir / "full_ole_reference.npz"
    metadata_path = args.output_dir / "full_ole_reference_metadata.json"
    signature = {
        "n": int(theta.size),
        "pca_dim": int(args.pca_dim),
        "pca_whiten": bool(args.pca_whiten),
        "grid": grid[:, 0].tolist(),
        "folds": int(args.ole_crossfit_folds),
        "seed": int(args.ole_crossfit_seed),
    }
    if path.is_file() and metadata_path.is_file() and not args.force_reference:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            with np.load(path, allow_pickle=False) as saved:
                return np.asarray(saved["fisher"], dtype=np.float64)
    fisher = _fit_ole(args, theta=theta, x=x, grid=grid, seed=int(args.ole_crossfit_seed))
    np.savez_compressed(path, fisher=fisher)
    metadata_path.write_text(json.dumps({"signature": signature}, indent=2) + "\n", encoding="utf-8")
    return fisher


def _train_validation(
    theta: np.ndarray, *, validation_fraction: float, n_strata: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    n_train = min(
        max(int(round((1.0 - validation_fraction) * theta.size)), 1), theta.size - 1
    )
    train = _nested_subsets(theta, [n_train], n_strata=n_strata, seed=seed)[0]
    validation = np.setdiff1d(np.arange(theta.size, dtype=np.int64), train)
    return train, validation


def _signature(args: argparse.Namespace, *, n: int, seed: int) -> dict[str, Any]:
    return {
        "n": int(n),
        "seed": int(seed),
        "pca_dim": int(args.pca_dim),
        "pca_whiten": bool(args.pca_whiten),
        "grid_size": int(args.theta_grid_size),
        "validation_fraction": float(args.flow_validation_fraction),
        "epochs": int(args.epochs),
        "patience": int(args.early_patience),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "ode_steps": int(args.ode_steps),
        "covariance_integrator": "midpoint_matrix_exponential",
        "ole_folds": int(args.ole_crossfit_folds),
    }


def _same_training_signature(cached: dict[str, Any], current: dict[str, Any]) -> bool:
    readout_keys = {"ode_steps", "covariance_integrator"}
    return (
        {key: value for key, value in cached.items() if key not in readout_keys}
        == {key: value for key, value in current.items() if key not in readout_keys}
    )


def _recompute_saved_flow_readout(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    grid: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    condition_grid = encode_flow_orientation(grid, period=PERIOD, encoding="periodic-rbf")
    model = build_flow_skl_model(
        velocity_family="condition_affine",
        theta_dim=int(condition_grid.shape[1]),
        x_dim=int(args.pca_dim),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        quadrature_steps=64,
        path_schedule="cosine",
        divergence_estimator="exact",
        theta_embedding="identity",
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    estimate = estimate_affine_mixed_symmetric_kl_fisher_for_conditions(
        model=model,
        theta_all=grid,
        condition_all=condition_grid,
        device=device,
        ridge=1e-6,
        ode_steps=int(args.ode_steps),
    )
    del model
    return np.asarray(estimate["fisher"], dtype=np.float64).reshape(-1)


def _fit_subset(
    args: argparse.Namespace,
    *,
    theta: np.ndarray,
    x: np.ndarray,
    grid: np.ndarray,
    global_index: np.ndarray,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    case_dir = args.output_dir / "subsets" / f"n{theta.size}"
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / "estimates.npz"
    metadata_path = case_dir / "metadata.json"
    flow_checkpoint = case_dir / "flow_selected_model.pt"
    signature = _signature(args, n=int(theta.size), seed=int(seed))
    if path.is_file() and metadata_path.is_file() and not args.force_fit:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            with np.load(path, allow_pickle=False) as saved:
                return {key: np.asarray(saved[key]) for key in saved.files} | {"metadata": metadata}
        if flow_checkpoint.is_file() and _same_training_signature(metadata.get("signature", {}), signature):
            print(
                f"[subset] n={theta.size} recomputing flow readout with "
                f"midpoint matrix exponential ({int(args.ode_steps)} steps)",
                flush=True,
            )
            with np.load(path, allow_pickle=False) as saved:
                arrays = {key: np.asarray(saved[key]) for key in saved.files}
            arrays["flow_fisher"] = _recompute_saved_flow_readout(
                args,
                checkpoint=flow_checkpoint,
                grid=grid,
                device=device,
            )
            np.savez_compressed(path, **arrays)
            metadata["signature"] = signature
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return arrays | {"metadata": metadata}

    train, validation = _train_validation(
        theta,
        validation_fraction=float(args.flow_validation_fraction),
        n_strata=int(args.theta_grid_size) - 1,
        seed=int(seed) + 100_000,
    )
    condition = encode_flow_orientation(theta, period=PERIOD, encoding="periodic-rbf")
    condition_grid = encode_flow_orientation(grid, period=PERIOD, encoding="periodic-rbf")
    flow_model, training, flow_estimate, _ = fit_flow_direction_estimator(
        theta_train=theta[train],
        x_train=x[train],
        theta_validation=theta[validation],
        x_validation=x[validation],
        theta_grid=grid,
        condition_train=condition[train],
        condition_validation=condition[validation],
        condition_grid=condition_grid,
        device=device,
        seed=int(seed),
        epochs=int(args.epochs),
        patience=int(args.early_patience),
        batch_size=int(args.batch_size),
        learning_rate=float(args.lr),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        ode_steps=int(args.ode_steps),
    )
    gkr_model, gkr_estimate, _ = fit_gkr_direction_estimator(
        theta_train=theta,
        x_train=x,
        theta_grid=grid,
        device=device,
        seed=int(seed),
        circular_period=PERIOD,
    )
    ole = _fit_ole(
        args,
        theta=theta,
        x=x,
        grid=grid,
        seed=int(args.ole_crossfit_seed) + int(seed),
    )
    arrays = {
        "flow_fisher": np.asarray(flow_estimate["fisher"], dtype=np.float64).reshape(-1),
        "gkr_fisher": np.asarray(gkr_estimate.linear_fisher, dtype=np.float64).reshape(-1),
        "ole_fisher": ole,
        "flow_train_loss": np.asarray(training["train_losses"], dtype=np.float64),
        "flow_validation_loss": np.asarray(training["val_losses"], dtype=np.float64),
        "global_index": np.asarray(global_index, dtype=np.int64),
        "flow_train_index": train,
        "flow_validation_index": validation,
    }
    np.savez_compressed(path, **arrays)
    torch.save(
        {key: value.detach().cpu() for key, value in flow_model.state_dict().items()},
        flow_checkpoint,
    )
    torch.save(gkr_checkpoint(gkr_model), case_dir / "gkr_model.pt")
    metadata = {
        "signature": signature,
        "flow_selected_epoch": int(training["selected_epoch"]),
        "flow_stopped_epoch": int(training["stopped_epoch"]),
        "flow_best_validation_loss": float(training["best_val_loss"]),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    del flow_model, gkr_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays | {"metadata": metadata}


def _rows(fit: dict[str, Any], *, reference: np.ndarray, n: int) -> list[dict[str, Any]]:
    scale = max(float(np.sqrt(np.mean(reference**2))), 1e-12)
    result = []
    for method, key in (
        ("Flow matching", "flow_fisher"),
        ("GKR", "gkr_fisher"),
        ("OLE (cross-fit)", "ole_fisher"),
    ):
        fisher = np.asarray(fit[key], dtype=np.float64)
        rmse = float(np.sqrt(np.mean((fisher - reference) ** 2)))
        result.append(
            {
                "method": method,
                "subset_size": int(n),
                "mean_fisher": float(np.mean(fisher)),
                "curve_normalized_rmse_to_full_ole": rmse / scale,
                "fisher": fisher.tolist(),
                "flow_selected_epoch": (
                    fit["metadata"].get("flow_selected_epoch") if method == "Flow matching" else None
                ),
            }
        )
    return result


def _plot(
    rows: list[dict[str, Any]], reference: np.ndarray, output_dir: Path, *, pca_dim: int, whiten: bool
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 12,
            "savefig.bbox": "tight",
        }
    )
    fig, axis = plt.subplots(figsize=(4.0, 3.5), constrained_layout=True)
    for method in METHODS:
        selected = sorted(
            (row for row in rows if row["method"] == method), key=lambda row: row["subset_size"]
        )
        axis.plot(
            [row["subset_size"] for row in selected],
            [row["mean_fisher"] for row in selected],
            color=COLORS[method],
            marker="o",
            markersize=5,
            linewidth=2.2,
            label=method,
        )
    axis.axhline(
        float(np.mean(reference)), color="0.15", linestyle="--", linewidth=2.0, label="Full-data OLE"
    )
    axis.set_xlabel("Subset size")
    axis.set_ylabel("Mean linear Fisher")
    label = "whitened" if whiten else "centered"
    axis.set_title(f"Stringer, {pca_dim}D PCA ({label})")
    axis.legend(frameon=False)
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color="0.88", linewidth=0.8)
    axis.xaxis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines["left"].set_linewidth(1.8)
    axis.spines["bottom"].set_linewidth(1.8)
    axis.tick_params(width=1.8)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "stringer_linear_fisher_subset_to_full_ole_reference"
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
        orientation_period=PERIOD,
    )
    theta = np.asarray(session.grating_orientation, dtype=np.float64)
    responses = np.asarray(session.neural_responses)
    _validate(args, n_samples=int(theta.size), n_features=int(responses.shape[1]))
    x, pca_metadata = _preprocess(
        args,
        theta=theta,
        responses=responses,
        session_file=Path(session.session_file),
    )
    grid = np.linspace(0.0, PERIOD, int(args.theta_grid_size), dtype=np.float64).reshape(-1, 1)
    reference = _reference(args, theta=theta, x=x, grid=grid)
    subset_sizes = sorted(int(size) for size in args.subset_sizes)
    subsets = _nested_subsets(
        theta,
        subset_sizes,
        n_strata=int(args.theta_grid_size) - 1,
        seed=int(args.subset_seed),
    )
    rows: list[dict[str, Any]] = []
    for size, index in zip(subset_sizes, subsets, strict=True):
        print(f"[subset] n={size}", flush=True)
        fit = _fit_subset(
            args,
            theta=theta[index],
            x=x[index],
            grid=grid,
            global_index=index,
            seed=int(args.subset_seed),
            device=device,
        )
        rows.extend(_rows(fit, reference=reference, n=size))
    png, svg = _plot(
        rows,
        reference,
        args.output_dir / "figures",
        pca_dim=int(args.pca_dim),
        whiten=bool(args.pca_whiten),
    )
    summary = {
        "session_file": str(session.session_file),
        "n_full": int(theta.size),
        "pca": pca_metadata,
        "full_ole_fisher": reference.tolist(),
        "full_ole_mean_fisher": float(np.mean(reference)),
        "rows": rows,
        "config": _json_ready(vars(args)),
        "runtime_seconds": float(time.perf_counter() - started),
        "artifacts": {"png": str(png), "svg": str(svg)},
    }
    summary_path = args.output_dir / "stringer_linear_fisher_subset_reference_summary.json"
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {summary_path}", flush=True)
    print(f"Saved: {png}", flush=True)
    print(f"Saved: {svg}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
