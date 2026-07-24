#!/usr/bin/env python3
"""Refresh only GKR entries in cached linear-Fisher sweep archives."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.gkr import GKRConfig, TorchGKR, estimate_gkr_linear_fisher
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import require_device


DEFAULT_SAMPLE_RESULTS = (
    REPO_ROOT
    / "data"
    / "linear_fisher_xdim50_gkr_classical_flow_n500_1000_3000_5000_10000_r5"
    / "linear_fisher_gkr_classical_flow_results.npz"
)
DEFAULT_DIMENSION_RESULTS = (
    REPO_ROOT
    / "data"
    / "linear_fisher_n3000_dimension_sweep_xdim3_10_30_50_70_90_110_r5"
    / "linear_fisher_dimension_sweep_results.npz"
)
TRAIN_FRACTION = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--sample-results", type=Path, default=DEFAULT_SAMPLE_RESULTS)
    parser.add_argument(
        "--dimension-results", type=Path, default=DEFAULT_DIMENSION_RESULTS
    )
    parser.add_argument("--case-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def gkr_config(n_total: int) -> GKRConfig:
    return GKRConfig(
        mean_iterations=300,
        mean_learning_rate=0.05,
        mean_batch_size=3000 if int(n_total) > 10_000 else None,
        n_inducing=200,
        covariance_epochs=30,
        covariance_learning_rate=0.1,
        covariance_batch_size=3000,
        validation_fraction=0.33,
        covariance_jitter=1e-6,
        likelihood_jitter=1e-5,
        prediction_batch_size=3000,
        standardize_responses=True,
        covariance_kernel_parameterization="log-lambda",
        covariance_neighbors_per_effective_dimension=5.0,
        covariance_lambda_epsilon=1e-8,
        covariance_initialization_grid_size=256,
        log_every=25,
    )


def case_directory(
    case_root: Path,
    *,
    x_dim: int,
    n_total: int,
    seed: int,
) -> Path:
    suffix = "" if int(seed) == 7 else f"_datasetseed{int(seed)}"
    return case_root / f"gkr_fixed_xdim{x_dim}_n{n_total}_linear{suffix}"


def dataset_path(
    case_root: Path,
    *,
    x_dim: int,
    n_total: int,
    seed: int,
) -> Path:
    return case_directory(
        case_root, x_dim=x_dim, n_total=n_total, seed=seed
    ) / f"randamp_gaussian_sqrtd_xdim{x_dim}_n{n_total}.npz"


def cache_path(
    case_root: Path,
    *,
    x_dim: int,
    n_total: int,
    seed: int,
) -> Path:
    return case_directory(
        case_root, x_dim=x_dim, n_total=n_total, seed=seed
    ) / "gkr_log_lambda_h0p2_results.npz"


def fit_or_load_gkr(
    *,
    dataset_npz: Path,
    output_npz: Path,
    query: np.ndarray,
    truth: np.ndarray,
    device: torch.device,
    seed: int,
    force: bool,
) -> tuple[np.ndarray, float]:
    if output_npz.is_file() and not force:
        with np.load(output_npz, allow_pickle=False) as cached:
            estimate = np.asarray(cached["gkr_linear_fisher"], dtype=np.float64)
            cached_query = np.asarray(
                cached["theta_midpoints"], dtype=np.float64
            ).reshape(-1)
        if estimate.shape == truth.shape and np.allclose(
            cached_query, np.asarray(query, dtype=np.float64).reshape(-1)
        ):
            return estimate, float(np.mean(np.abs(estimate - truth)))

    bundle = load_shared_dataset_npz(dataset_npz)
    config = gkr_config(
        int(bundle.x_train.shape[0] + bundle.x_validation.shape[0])
    )
    model = TorchGKR(
        n_input=int(bundle.theta_train.shape[1]),
        n_output=int(bundle.x_train.shape[1]),
        config=config,
        dtype=torch.float64,
        device=device,
        seed=int(seed),
    )
    model.fit(bundle.x_train, bundle.theta_train)
    query_2d = np.asarray(query, dtype=np.float64).reshape(-1, 1)
    if query_2d.shape[0] < 2:
        raise ValueError("At least two query positions are required.")
    spacing = float(np.median(np.diff(query_2d[:, 0])))
    result = estimate_gkr_linear_fisher(
        model,
        query_2d,
        finite_difference_step=spacing,
        solve_jitter=1e-6,
    )
    estimate = np.asarray(result.linear_fisher, dtype=np.float64)
    if estimate.shape != truth.shape:
        raise ValueError(
            f"GKR shape {estimate.shape} does not match truth {truth.shape}."
        )
    metadata = model.covariance_kernel_metadata()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        theta_midpoints=query_2d,
        ground_truth_linear_fisher=np.asarray(truth, dtype=np.float64),
        gkr_linear_fisher=estimate,
        gkr_mean=result.mean,
        gkr_covariance=result.covariance,
        gkr_mean_jacobian=result.mean_jacobian,
        gkr_mean_loss=result.mean_loss,
        gkr_covariance_loss=result.covariance_loss,
        gkr_config_json=np.asarray(json.dumps(asdict(config), sort_keys=True)),
        covariance_kernel_metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True)
        ),
        dataset_npz=np.asarray(str(dataset_npz)),
    )
    return estimate, float(np.mean(np.abs(estimate - truth)))


def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def write_archive_preserving_non_gkr(
    path: Path,
    original: dict[str, np.ndarray],
    gkr: np.ndarray,
    gkr_mae: np.ndarray,
) -> Path:
    updated = dict(original)
    updated["gkr"] = np.asarray(gkr, dtype=np.float64)
    updated["gkr_mae"] = np.asarray(gkr_mae, dtype=np.float64)
    for key, value in original.items():
        if key not in {"gkr", "gkr_mae"} and not np.array_equal(
            value, updated[key], equal_nan=True
        ):
            raise AssertionError(f"Non-GKR array changed unexpectedly: {key}")

    backup = path.with_name(path.stem + "_pre_log_lambda_gkr.npz")
    if not backup.exists():
        shutil.copy2(path, backup)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.stem + "_", suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **updated)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return backup


def update_summary(path: Path, axis_name: str, axis_values: np.ndarray, mae: np.ndarray) -> None:
    if not path.is_file():
        return
    summary = json.loads(path.read_text(encoding="utf-8"))
    cases = summary.get("cases", [])
    by_value = {int(case[axis_name]): case for case in cases}
    for index, value in enumerate(axis_values):
        case = by_value.get(int(value))
        if case is None:
            continue
        values = np.asarray(mae[index], dtype=np.float64)
        case["gkr_mae_mean"] = float(values.mean())
        case["gkr_mae_std"] = (
            float(values.std(ddof=1)) if values.size > 1 else 0.0
        )
        for repeat_index, repeat in enumerate(case.get("repeats", [])):
            if repeat_index < values.size:
                repeat["gkr_mae"] = float(values[repeat_index])
    summary["gkr_estimator"] = {
        "covariance_kernel": "non-periodic Gaussian RBF",
        "bandwidth_parameterization": "ARD log-lambda",
        "condition_standardization": True,
        "initialization": "residual participation ratio and target kernel ESS",
        "uses_training_fraction": TRAIN_FRACTION,
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def update_csv(
    path: Path,
    axis_name: str,
    axis_values: np.ndarray,
    seeds: np.ndarray,
    mae: np.ndarray,
) -> None:
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0]) if rows else [axis_name, "seed", "method", "mae"]
    replacement = {
        (int(value), int(seed)): float(mae[i, j])
        for i, value in enumerate(axis_values)
        for j, seed in enumerate(seeds)
    }
    for row in rows:
        if row.get("method") == "GKR":
            row["mae"] = str(
                replacement[(int(row[axis_name]), int(row["seed"]))]
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def refresh_sample_archive(
    path: Path,
    *,
    case_root: Path,
    device: torch.device,
    force: bool,
) -> dict[str, object]:
    archive = load_archive(path)
    n_values = np.asarray(archive["n_values"], dtype=np.int64)
    seeds = np.asarray(archive["seeds"], dtype=np.int64)
    refreshed = np.empty_like(archive["gkr"], dtype=np.float64)
    mae = np.empty_like(archive["gkr_mae"], dtype=np.float64)
    for i, n_total in enumerate(n_values):
        for j, seed in enumerate(seeds):
            data = dataset_path(
                case_root,
                x_dim=50,
                n_total=int(n_total),
                seed=int(seed),
            )
            cache = cache_path(
                case_root,
                x_dim=50,
                n_total=int(n_total),
                seed=int(seed),
            )
            refreshed[i, j], mae[i, j] = fit_or_load_gkr(
                dataset_npz=data,
                output_npz=cache,
                query=archive["theta"][i, j],
                truth=archive["ground_truth"][i, j],
                device=device,
                seed=int(seed),
                force=force,
            )
            print(
                f"[sample] N={n_total} seed={seed} GKR MAE={mae[i, j]:.6f}",
                flush=True,
            )
    backup = write_archive_preserving_non_gkr(path, archive, refreshed, mae)
    update_summary(
        path.with_name("linear_fisher_gkr_classical_flow_summary.json"),
        "n_total",
        n_values,
        mae,
    )
    update_csv(
        path.with_name("linear_fisher_gkr_classical_flow_errors.csv"),
        "n_total",
        n_values,
        seeds,
        mae,
    )
    return {
        "results": str(path),
        "backup": str(backup),
        "gkr_mae_mean": mae.mean(axis=1).tolist(),
        "gkr_mae_std": mae.std(axis=1, ddof=1).tolist(),
    }


def refresh_dimension_archive(
    path: Path,
    *,
    case_root: Path,
    device: torch.device,
    force: bool,
) -> dict[str, object]:
    archive = load_archive(path)
    dimensions = np.asarray(archive["x_dims"], dtype=np.int64)
    seeds = np.asarray(archive["seeds"], dtype=np.int64)
    refreshed = np.empty_like(archive["gkr"], dtype=np.float64)
    mae = np.empty_like(archive["gkr_mae"], dtype=np.float64)
    for i, x_dim in enumerate(dimensions):
        for j, seed in enumerate(seeds):
            data = dataset_path(
                case_root,
                x_dim=int(x_dim),
                n_total=3000,
                seed=int(seed),
            )
            cache = cache_path(
                case_root,
                x_dim=int(x_dim),
                n_total=3000,
                seed=int(seed),
            )
            refreshed[i, j], mae[i, j] = fit_or_load_gkr(
                dataset_npz=data,
                output_npz=cache,
                query=archive["theta"][i, j],
                truth=archive["ground_truth"][i, j],
                device=device,
                seed=int(seed),
                force=force,
            )
            print(
                f"[dimension] d={x_dim} seed={seed} GKR MAE={mae[i, j]:.6f}",
                flush=True,
            )
    backup = write_archive_preserving_non_gkr(path, archive, refreshed, mae)
    update_summary(
        path.with_name("linear_fisher_dimension_sweep_summary.json"),
        "x_dim",
        dimensions,
        mae,
    )
    update_csv(
        path.with_name("linear_fisher_dimension_sweep_errors.csv"),
        "x_dim",
        dimensions,
        seeds,
        mae,
    )
    return {
        "results": str(path),
        "backup": str(backup),
        "gkr_mae_mean": mae.mean(axis=1).tolist(),
        "gkr_mae_std": mae.std(axis=1, ddof=1).tolist(),
    }


def main() -> None:
    args = parse_args()
    device = require_device(str(args.device))
    case_root = args.case_root.expanduser().resolve()
    sample_results = args.sample_results.expanduser().resolve()
    dimension_results = args.dimension_results.expanduser().resolve()
    if not sample_results.is_file() or not dimension_results.is_file():
        raise FileNotFoundError("Both aggregate result archives must exist.")

    sample = refresh_sample_archive(
        sample_results,
        case_root=case_root,
        device=device,
        force=bool(args.force),
    )
    dimension = refresh_dimension_archive(
        dimension_results,
        case_root=case_root,
        device=device,
        force=bool(args.force),
    )
    summary = {
        "device": str(device),
        "kernel": "non-periodic Gaussian RBF with learned ARD log-lambda",
        "sample_sweep": sample,
        "dimension_sweep": dimension,
    }
    summary_path = (
        sample_results.parent / "upgraded_gkr_cache_refresh_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
