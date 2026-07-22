#!/usr/bin/env python3
"""Compare Flow Matching, GKR, and cross-fitted OLE across sample density."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.continuous_fisher_comparison import make_native_dataset_npz, theta_grid_from_meta
from fisher.fisher_validation import (
    fit_cross_fitted_ole_direction_estimator,
    fit_flow_direction_estimator,
    fit_gkr_direction_estimator,
    gkr_checkpoint,
    population_linear_moments,
)
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_meta, require_device
from global_setting import (
    DATA_DIR,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_TRAINING_MAX_EPOCHS,
)

DATASETS = {
    "randamp_gaussian_sqrtd": "Gaussian",
    "cosine_gmm": "Gaussian mixture",
}
METHODS = ("Flow Matching", "GKR", "OLE (cross-fit)")
COLORS = {"Flow Matching": "C0", "GKR": "C2", "OLE (cross-fit)": "C1"}
MARKERS = {"Flow Matching": "o", "GKR": "^", "OLE (cross-fit)": "s"}


def _csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("Expected a comma-separated integer list.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--dataset",
        choices=("all", *DATASETS),
        default="all",
    )
    parser.add_argument("--x-dim", type=int, default=50)
    parser.add_argument(
        "--n-list", type=_csv_ints, default=[125, 250, 500, 1000, 3000, 10000]
    )
    parser.add_argument("--seeds", type=_csv_ints, default=[7, 8, 9, 10, 11])
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--theta-grid-size", type=int, default=31)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_MAX_EPOCHS)
    parser.add_argument(
        "--early-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--ode-steps", type=int, default=64)
    parser.add_argument("--ole-crossfit-folds", type=int, default=5)
    parser.add_argument("--ole-min-endpoint-samples", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "toy_linear_fisher_density_xdim50_r5",
    )
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Fit requested cases without writing shared aggregate artifacts.",
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


def _signature(args: argparse.Namespace, *, dataset: str, seed: int, n_total: int) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "seed": int(seed),
        "n_total": int(n_total),
        "x_dim": int(args.x_dim),
        "train_fraction": float(args.train_fraction),
        "theta_grid_size": int(args.theta_grid_size),
        "epochs": int(args.epochs),
        "early_patience": int(args.early_patience),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "ode_steps": int(args.ode_steps),
        "ole_crossfit_folds": int(args.ole_crossfit_folds),
        "ole_min_endpoint_samples": int(args.ole_min_endpoint_samples),
        "estimator_sample_pool": "training_split_only",
        "flow_validation_pool": "validation_split_only",
        "response_standardization": "flow_train_split_featurewise",
    }


def _truth(population: Any, theta_midpoints: np.ndarray) -> np.ndarray:
    _, derivative, covariance = population_linear_moments(population, theta_midpoints)
    precision_derivative = np.linalg.solve(covariance, derivative[..., None])[..., 0]
    return np.einsum("ki,ki->k", derivative, precision_derivative)


def _fit_case(
    args: argparse.Namespace,
    *,
    dataset: str,
    seed: int,
    n_total: int,
    device: torch.device,
) -> dict[str, Any]:
    case_dir = args.output_dir / dataset / f"seed{seed}" / f"n{n_total}"
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "linear_fisher_density_result.npz"
    metadata_path = case_dir / "metadata.json"
    signature = _signature(args, dataset=dataset, seed=seed, n_total=n_total)
    if result_path.is_file() and metadata_path.is_file() and not args.force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            with np.load(result_path, allow_pickle=False) as saved:
                return {key: np.asarray(saved[key]) for key in saved.files} | {
                    "metadata": metadata
                }

    dataset_path = case_dir / "dataset.npz"
    make_native_dataset_npz(
        output_npz=dataset_path,
        dataset_family=dataset,
        x_dim=int(args.x_dim),
        n_total=int(n_total),
        train_frac=float(args.train_fraction),
        seed=int(seed),
        force=bool(args.force),
    )
    bundle = load_shared_dataset_npz(dataset_path)
    theta_grid = theta_grid_from_meta(bundle.meta, theta_grid_size=int(args.theta_grid_size))
    theta_midpoints = 0.5 * (theta_grid[:-1] + theta_grid[1:])
    population = build_dataset_from_meta(dict(bundle.meta))
    truth = _truth(population, theta_midpoints)

    flow_model, flow_training, flow_estimate, _ = fit_flow_direction_estimator(
        theta_train=bundle.theta_train,
        x_train=bundle.x_train,
        theta_validation=bundle.theta_validation,
        x_validation=bundle.x_validation,
        theta_grid=theta_grid,
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
    flow = np.asarray(flow_estimate["fisher"], dtype=np.float64).reshape(-1)
    torch.save(
        {key: value.detach().cpu() for key, value in flow_model.state_dict().items()},
        case_dir / "flow_selected_model.pt",
    )

    gkr_model, gkr_estimate, _ = fit_gkr_direction_estimator(
        theta_train=bundle.theta_train,
        x_train=bundle.x_train,
        theta_grid=theta_grid,
        device=device,
        seed=int(seed),
    )
    gkr = np.asarray(gkr_estimate.linear_fisher, dtype=np.float64).reshape(-1)
    torch.save(gkr_checkpoint(gkr_model), case_dir / "gkr_model.pt")

    ole_result, _ = fit_cross_fitted_ole_direction_estimator(
        theta_train=bundle.theta_train,
        x_train=bundle.x_train,
        theta_grid=theta_grid,
        n_splits=int(args.ole_crossfit_folds),
        seed=20_260_721 + int(seed),
        min_endpoint_samples=int(args.ole_min_endpoint_samples),
    )
    ole = np.asarray(ole_result.linear_fisher, dtype=np.float64).reshape(-1)
    ole_raw = np.asarray(ole_result.linear_fisher_raw, dtype=np.float64).reshape(-1)
    endpoint_counts = np.concatenate(
        [np.asarray(ole_result.n_left), np.asarray(ole_result.n_right)]
    )
    arrays = {
        "theta_grid": theta_grid,
        "theta_midpoints": theta_midpoints,
        "ground_truth": truth,
        "flow_fisher": flow,
        "gkr_fisher": gkr,
        "ole_fisher": ole,
        "ole_fisher_raw": ole_raw,
        "ole_endpoint_counts": endpoint_counts,
        "flow_train_losses": np.asarray(flow_training["train_losses"], dtype=np.float64),
        "flow_validation_losses": np.asarray(flow_training["val_losses"], dtype=np.float64),
        "gkr_mean_losses": np.asarray(gkr_estimate.mean_loss, dtype=np.float64),
        "gkr_covariance_losses": np.asarray(gkr_estimate.covariance_loss, dtype=np.float64),
    }
    np.savez_compressed(result_path, **arrays)
    method_curves = {"Flow Matching": flow, "GKR": gkr, "OLE (cross-fit)": ole}
    metadata = {
        "signature": signature,
        "n_train": int(bundle.x_train.shape[0]),
        "n_validation": int(bundle.x_validation.shape[0]),
        "train_density": float(bundle.x_train.shape[0] / int(args.x_dim)),
        "median_ole_endpoint_count": float(np.median(endpoint_counts)),
        "median_ole_endpoint_density": float(np.median(endpoint_counts) / int(args.x_dim)),
        "flow_selected_epoch": int(flow_training["selected_epoch"]),
        "flow_stopped_epoch": int(flow_training["stopped_epoch"]),
        "mae": {
            method: float(np.mean(np.abs(curve - truth)))
            for method, curve in method_curves.items()
        },
    }
    metadata_path.write_text(
        json.dumps(_json_ready(metadata), indent=2) + "\n", encoding="utf-8"
    )
    del flow_model, gkr_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays | {"metadata": metadata}


def _read_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in args.seeds:
            for n_total in args.n_list:
                metadata_path = (
                    args.output_dir / dataset / f"seed{seed}" / f"n{n_total}" / "metadata.json"
                )
                if not metadata_path.is_file():
                    continue
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("signature") != _signature(
                    args, dataset=dataset, seed=seed, n_total=n_total
                ):
                    continue
                for method in METHODS:
                    rows.append(
                        {
                            "dataset": dataset,
                            "dataset_label": DATASETS[dataset],
                            "seed": int(seed),
                            "n_total": int(n_total),
                            "n_train": int(metadata["n_train"]),
                            "train_density": float(metadata["train_density"]),
                            "median_ole_endpoint_count": float(
                                metadata["median_ole_endpoint_count"]
                            ),
                            "median_ole_endpoint_density": float(
                                metadata["median_ole_endpoint_density"]
                            ),
                            "method": method,
                            "mae": float(metadata["mae"][method]),
                        }
                    )
    return rows


def _group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for n_total in sorted({int(row["n_total"]) for row in rows if row["dataset"] == dataset}):
            for method in METHODS:
                selected = [
                    row
                    for row in rows
                    if row["dataset"] == dataset
                    and int(row["n_total"]) == n_total
                    and row["method"] == method
                ]
                if not selected:
                    continue
                values = np.asarray([row["mae"] for row in selected], dtype=np.float64)
                grouped.append(
                    {
                        "dataset": dataset,
                        "dataset_label": DATASETS[dataset],
                        "n_total": n_total,
                        "n_train": int(selected[0]["n_train"]),
                        "train_density": float(selected[0]["train_density"]),
                        "median_ole_endpoint_density": float(
                            np.mean([row["median_ole_endpoint_density"] for row in selected])
                        ),
                        "method": method,
                        "n_repeats": int(values.size),
                        "mae_mean": float(np.mean(values)),
                        "mae_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    }
                )
    return grouped


def _plot(grouped: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        for method in METHODS:
            selected = [
                row for row in grouped if row["dataset"] == dataset and row["method"] == method
            ]
            selected.sort(key=lambda row: float(row["train_density"]))
            x = np.asarray([row["train_density"] for row in selected], dtype=np.float64)
            y = np.asarray([row["mae_mean"] for row in selected], dtype=np.float64)
            error = np.asarray([row["mae_std"] for row in selected], dtype=np.float64)
            axis.errorbar(
                x,
                y,
                yerr=error,
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2.0,
                markersize=5.5,
                capsize=2.5,
                label=method,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(DATASETS[dataset])
        axis.set_xlabel(r"Training density $n_{\mathrm{train}}/d$")
        axis.set_axisbelow(True)
        axis.yaxis.grid(True, color="0.88", linewidth=0.8)
        axis.xaxis.grid(False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines["left"].set_linewidth(1.8)
        axis.spines["bottom"].set_linewidth(1.8)
        axis.tick_params(width=1.8)
    axes[0].set_ylabel("Linear Fisher MAE")
    axes[0].legend(frameon=False, loc="best")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "toy_linear_fisher_error_vs_training_density"
    png, svg = stem.with_suffix(".png"), stem.with_suffix(".svg")
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def _aggregate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Path, Path, Path]:
    rows = _read_rows(args)
    grouped = _group_rows(rows)
    png, svg = _plot(grouped, args.output_dir)
    csv_path = args.output_dir / "toy_linear_fisher_density_summary.csv"
    fieldnames = list(grouped[0]) if grouped else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(grouped)
    summary = {
        "config": _json_ready(vars(args)),
        "rows": rows,
        "grouped": grouped,
        "artifacts": {"png": str(png), "svg": str(svg), "csv": str(csv_path)},
    }
    summary_path = args.output_dir / "toy_linear_fisher_density_summary.json"
    summary_path.write_text(
        json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8"
    )
    return grouped, png, svg, summary_path


def main() -> int:
    args = parse_args()
    if int(args.x_dim) < 1:
        raise ValueError("--x-dim must be positive.")
    if not 0.0 < float(args.train_fraction) < 1.0:
        raise ValueError("--train-fraction must be in (0, 1).")
    if int(args.theta_grid_size) < 3:
        raise ValueError("--theta-grid-size must be at least 3.")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = require_device(str(args.device))
    datasets = list(DATASETS) if args.dataset == "all" else [str(args.dataset)]
    started = time.perf_counter()
    if not args.aggregate_only:
        for dataset in datasets:
            for seed in args.seeds:
                for n_total in args.n_list:
                    print(
                        f"[density] dataset={dataset} seed={seed} N={n_total} "
                        f"train_density={round(args.train_fraction * n_total) / args.x_dim:.3g}",
                        flush=True,
                    )
                    result = _fit_case(
                        args,
                        dataset=dataset,
                        seed=int(seed),
                        n_total=int(n_total),
                        device=device,
                    )
                    print(
                        "[density:mae] "
                        + " ".join(
                            f"{method}={result['metadata']['mae'][method]:.6g}"
                            for method in METHODS
                        ),
                        flush=True,
                    )
    print(f"[density] runtime_seconds={time.perf_counter() - started:.3f}", flush=True)
    if args.skip_aggregate:
        return 0
    grouped, png, svg, summary_path = _aggregate(args)
    print(f"Saved: {summary_path}", flush=True)
    print(f"Saved: {png}", flush=True)
    print(f"Saved: {svg}", flush=True)
    if grouped:
        print(json.dumps(grouped, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
