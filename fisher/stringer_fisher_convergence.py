"""Stringer single-session Fisher convergence utilities."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from fisher.continuous_fisher_comparison import ContinuousFlowConfig, METHOD_CLASSICAL_LINEAR, METHOD_FLOW_LINEAR
from fisher.distance_comparison import save_flow_result_npz
from fisher.flow_matching_skl import (
    FlowSKLResult,
    build_flow_skl_model,
    estimate_affine_mixed_symmetric_kl_fisher,
    train_flow_skl_model,
)
from fisher.shared_dataset_io import SharedDatasetBundle

RESULTS_NPZ_NAME = "stringer_linear_fisher_convergence_results.npz"
CURVES_CSV_NAME = "stringer_linear_fisher_convergence_curves.csv"
SUMMARY_JSON_NAME = "stringer_linear_fisher_convergence_summary.json"
ABS_ERROR_SVG_NAME = "stringer_linear_fisher_convergence_abs_error.svg"
ABS_ERROR_PNG_NAME = "stringer_linear_fisher_convergence_abs_error.png"
CURVE_EXAMPLES_SVG_NAME = "stringer_linear_fisher_curve_examples.svg"
CURVE_EXAMPLES_PNG_NAME = "stringer_linear_fisher_curve_examples.png"

METHODS = (METHOD_CLASSICAL_LINEAR, METHOD_FLOW_LINEAR)
METRICS = ("mae", "rmse", "pearson", "area_normalized_l2")


@dataclass(frozen=True)
class PCAProjectionResult:
    x_all: np.ndarray
    explained_variance_ratio: np.ndarray
    singular_values: np.ndarray
    mean: np.ndarray
    components: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LinearFisherConvergenceResult:
    theta_grid: np.ndarray
    theta_midpoints: np.ndarray
    n_list: np.ndarray
    repeat_indices: np.ndarray
    references: dict[str, np.ndarray]
    curves: dict[str, np.ndarray]
    metrics: dict[str, dict[str, np.ndarray]]
    rows: list[dict[str, Any]]
    metadata: dict[str, Any]


def parse_int_list(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        out = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        out = [int(v) for v in value]
    if not out:
        raise ValueError("Expected at least one integer.")
    if any(v < 1 for v in out):
        raise ValueError("All entries must be positive.")
    return out


def theta_grid_periodic(period: float, theta_grid_size: int) -> np.ndarray:
    if int(theta_grid_size) < 2:
        raise ValueError("theta_grid_size must be >= 2.")
    if float(period) <= 0.0:
        raise ValueError("period must be positive.")
    return np.linspace(0.0, float(period), int(theta_grid_size), dtype=np.float64).reshape(-1, 1)


def theta_midpoints(theta_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(theta_grid, dtype=np.float64).reshape(-1)
    return (0.5 * (grid[:-1] + grid[1:])).reshape(-1, 1)


def circular_distance(theta: np.ndarray, center: float, period: float) -> np.ndarray:
    if float(period) <= 0.0:
        raise ValueError("period must be positive.")
    delta = np.abs(np.mod(np.asarray(theta, dtype=np.float64).reshape(-1) - float(center) + 0.5 * float(period), float(period)) - 0.5 * float(period))
    return delta.astype(np.float64)


def circular_endpoint_windows(
    *,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    theta_grid: np.ndarray,
    period: float,
    radius: float | None,
    min_endpoint_samples: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    th = np.mod(np.asarray(theta_all, dtype=np.float64).reshape(-1), float(period))
    x = np.asarray(x_all, dtype=np.float64)
    grid = np.asarray(theta_grid, dtype=np.float64).reshape(-1)
    if x.ndim != 2:
        raise ValueError("x_all must be a 2D array.")
    if th.shape[0] != x.shape[0]:
        raise ValueError("theta_all and x_all lengths must match.")
    if grid.shape[0] < 2:
        raise ValueError("theta_grid must contain at least two endpoints.")
    if radius is None:
        radius_val = 0.5 * float(np.min(np.diff(grid)))
    else:
        radius_val = float(radius)
    if radius_val <= 0.0:
        raise ValueError("window radius must be positive.")
    need = int(min_endpoint_samples)
    if need < 1:
        raise ValueError("min_endpoint_samples must be >= 1.")
    if need > int(th.shape[0]):
        raise ValueError("min_endpoint_samples exceeds the number of trials.")

    windows: list[tuple[np.ndarray, np.ndarray]] = []
    for endpoint in grid:
        dist = circular_distance(th, float(endpoint), float(period))
        idx = np.flatnonzero(dist <= radius_val)
        if int(idx.size) < need:
            idx = np.argsort(dist, kind="mergesort")[:need]
        windows.append((th[idx].reshape(-1, 1), x[idx]))
    return windows


def classical_linear_fisher_circular(
    *,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    theta_grid: np.ndarray,
    period: float,
    ridge: float = 1e-6,
    window_radius: float | None = None,
    min_endpoint_samples: int = 8,
) -> np.ndarray:
    grid = np.asarray(theta_grid, dtype=np.float64).reshape(-1)
    windows = circular_endpoint_windows(
        theta_all=theta_all,
        x_all=x_all,
        theta_grid=theta_grid,
        period=float(period),
        radius=window_radius,
        min_endpoint_samples=int(min_endpoint_samples),
    )
    d = int(np.asarray(x_all).shape[1])
    eye = np.eye(d, dtype=np.float64)
    out = np.full(grid.shape[0] - 1, np.nan, dtype=np.float64)
    for i in range(grid.shape[0] - 1):
        x_l = windows[i][1]
        x_r = windows[i + 1][1]
        dtheta = float(grid[i + 1] - grid[i])
        mu_prime = (np.mean(x_r, axis=0) - np.mean(x_l, axis=0)) / dtheta
        cov_l = np.cov(x_l, rowvar=False) if int(x_l.shape[0]) > 1 else np.zeros((d, d), dtype=np.float64)
        cov_r = np.cov(x_r, rowvar=False) if int(x_r.shape[0]) > 1 else np.zeros((d, d), dtype=np.float64)
        cov = 0.5 * (np.atleast_2d(cov_l) + np.atleast_2d(cov_r)) + float(ridge) * eye
        out[i] = max(0.0, float(mu_prime @ np.linalg.solve(cov, mu_prime)))
    return out


def fit_pca_projection(
    responses: np.ndarray,
    *,
    n_components: int = 50,
    random_state: int = 0,
    whiten: bool = True,
) -> PCAProjectionResult:
    x = np.asarray(responses, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("responses must be a 2D trial-by-feature array.")
    max_components = min(int(x.shape[0]), int(x.shape[1]))
    if int(n_components) < 1 or int(n_components) > max_components:
        raise ValueError(f"n_components must be in [1, {max_components}].")
    pca = PCA(n_components=int(n_components), whiten=bool(whiten), svd_solver="randomized", random_state=int(random_state))
    x_pca = pca.fit_transform(x).astype(np.float64, copy=False)
    return PCAProjectionResult(
        x_all=x_pca,
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_, dtype=np.float64),
        singular_values=np.asarray(pca.singular_values_, dtype=np.float64),
        mean=np.asarray(pca.mean_, dtype=np.float64),
        components=np.asarray(pca.components_, dtype=np.float64),
        metadata={
            "pca_dim": int(n_components),
            "pca_whiten": bool(whiten),
            "pca_svd_solver": "randomized",
            "pca_random_state": int(random_state),
            "pca_input_uses_orientation_labels": False,
            "pca_input": "neural_responses_only",
            "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        },
    )


def stratified_subset_indices(
    theta_all: np.ndarray,
    *,
    n_total: int,
    n_bins: int,
    period: float,
    seed: int,
) -> np.ndarray:
    theta = np.mod(np.asarray(theta_all, dtype=np.float64).reshape(-1), float(period))
    n = int(theta.shape[0])
    target = int(n_total)
    if target < 1:
        raise ValueError("n_total must be >= 1.")
    if target > n:
        raise ValueError(f"n_total={target} exceeds available trials={n}.")
    if int(n_bins) < 1:
        raise ValueError("n_bins must be >= 1.")
    bin_id = np.floor(theta / float(period) * int(n_bins)).astype(np.int64)
    bin_id = np.clip(bin_id, 0, int(n_bins) - 1)
    counts = np.bincount(bin_id, minlength=int(n_bins)).astype(np.int64)
    raw = target * counts.astype(np.float64) / float(n)
    quotas = np.minimum(np.floor(raw).astype(np.int64), counts)
    remaining = target - int(np.sum(quotas))
    fractions = raw - np.floor(raw)
    order = np.argsort(-fractions, kind="mergesort")
    while remaining > 0:
        progressed = False
        for b in order:
            if remaining <= 0:
                break
            if quotas[b] < counts[b]:
                quotas[b] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("Could not allocate stratified subset quotas.")

    rng = np.random.default_rng(int(seed))
    chosen: list[np.ndarray] = []
    for b in range(int(n_bins)):
        if quotas[b] <= 0:
            continue
        available = np.flatnonzero(bin_id == b)
        chosen.append(rng.choice(available, size=int(quotas[b]), replace=False))
    out = np.concatenate(chosen).astype(np.int64)
    rng.shuffle(out)
    if int(out.shape[0]) != target:
        raise RuntimeError(f"Expected {target} subset indices, got {out.shape[0]}.")
    return out


def split_train_validation(indices: np.ndarray, *, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if int(idx.shape[0]) < 2:
        raise ValueError("Need at least two samples for train/validation split.")
    frac = float(train_frac)
    if not (0.0 < frac < 1.0):
        raise ValueError("train_frac must be in (0, 1).")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(idx)
    n_train = int(frac * int(idx.shape[0]))
    n_train = min(max(n_train, 1), int(idx.shape[0]) - 1)
    return perm[:n_train].astype(np.int64), perm[n_train:].astype(np.int64)


def make_shared_bundle(
    *,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    meta: dict[str, Any],
) -> SharedDatasetBundle:
    theta = np.asarray(theta_all, dtype=np.float64).reshape(-1, 1)
    x = np.asarray(x_all, dtype=np.float64)
    tr = np.asarray(train_idx, dtype=np.int64).reshape(-1)
    va = np.asarray(validation_idx, dtype=np.int64).reshape(-1)
    return SharedDatasetBundle(
        meta=dict(meta),
        theta_all=theta,
        x_all=x,
        train_idx=tr,
        validation_idx=va,
        theta_train=theta[tr],
        x_train=x[tr],
        theta_validation=theta[va],
        x_validation=x[va],
    )


def train_flow_linear_curve(
    *,
    bundle: SharedDatasetBundle,
    theta_grid: np.ndarray,
    device: torch.device,
    config: ContinuousFlowConfig,
    seed: int,
    output_npz: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any], Path | None]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = build_flow_skl_model(
        velocity_family="condition_affine",
        theta_dim=1,
        x_dim=int(np.asarray(bundle.x_train).shape[1]),
        hidden_dim=int(config.hidden_dim),
        depth=int(config.depth),
        quadrature_steps=int(config.quadrature_steps),
        path_schedule=str(config.path_schedule),
        divergence_estimator=str(config.divergence_estimator),
        hutchinson_probes=int(config.hutchinson_probes),
        shared_affine_a_diag_jitter=float(config.shared_affine_a_diag_jitter),
    ).to(device)
    meta = train_flow_skl_model(
        model=model,
        theta_train=np.asarray(bundle.theta_train, dtype=np.float64).reshape(-1, 1),
        x_train=np.asarray(bundle.x_train, dtype=np.float64),
        theta_val=np.asarray(bundle.theta_validation, dtype=np.float64).reshape(-1, 1),
        x_val=np.asarray(bundle.x_validation, dtype=np.float64),
        device=device,
        velocity_family="condition_affine",
        path_schedule=str(config.path_schedule),
        epochs=int(config.epochs),
        batch_size=int(config.batch_size),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
        t_eps=float(config.t_eps),
        patience=int(config.early_patience),
        min_delta=float(config.early_min_delta),
        ema_alpha=float(config.early_ema_alpha),
        max_grad_norm=float(config.max_grad_norm),
        log_every=max(1, int(config.log_every)),
    )
    fd = estimate_affine_mixed_symmetric_kl_fisher(
        model=model,
        theta_all=theta_grid,
        device=device,
        ridge=float(config.affine_ridge),
        ode_steps=int(config.ode_steps),
    )
    saved: Path | None = None
    if output_npz is not None:
        output_npz = Path(output_npz)
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        result = FlowSKLResult(
            symmetric_kl_matrix=np.asarray(fd["symmetric_kl_matrix"], dtype=np.float64),
            canonical_metric_matrix=np.asarray(fd["canonical_metric_matrix"], dtype=np.float64),
            canonical_metric_name=str(fd["canonical_metric_name"]),
            fisher_theta_midpoints=fd["theta_midpoints"],
            fisher_linear=fd["fisher"],
            train_metadata=meta,
        )
        saved = save_flow_result_npz(output_npz, result=result, metric=METHOD_FLOW_LINEAR, theta_eval=theta_grid, velocity_family="condition_affine")
    return np.asarray(fd["fisher"], dtype=np.float64), meta, saved


def curve_metrics(curve: np.ndarray, reference: np.ndarray, theta_mid: np.ndarray | None = None) -> dict[str, float]:
    vals = np.asarray(curve, dtype=np.float64).reshape(-1)
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    if vals.shape != ref.shape:
        raise ValueError("curve and reference must have the same shape.")
    diff = vals - ref
    mae = float(np.nanmean(np.abs(diff)))
    rmse = float(np.sqrt(np.nanmean(diff**2)))
    if vals.shape[0] < 2 or np.nanstd(vals) <= 1e-12 or np.nanstd(ref) <= 1e-12:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(vals, ref)[0, 1])
    if theta_mid is None or vals.shape[0] < 2:
        num = float(np.nanmean(diff**2))
        den = float(np.nanmean(ref**2))
    else:
        th = np.asarray(theta_mid, dtype=np.float64).reshape(-1)
        num = float(np.trapezoid(diff**2, x=th))
        den = float(np.trapezoid(ref**2, x=th))
    area_l2 = float(math.sqrt(max(num, 0.0) / max(den, 1e-24)))
    return {"mae": mae, "rmse": rmse, "pearson": pearson, "area_normalized_l2": area_l2}


def _metric_arrays(n_count: int, n_repeats: int) -> dict[str, np.ndarray]:
    return {name: np.full((n_count, n_repeats), np.nan, dtype=np.float64) for name in METRICS}


def _compact_train_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    out = dict(meta)
    for key in ("train_losses", "val_losses", "val_monitor_losses"):
        arr = np.asarray(out.pop(key, []), dtype=np.float64)
        if arr.size:
            out[f"{key}_final"] = float(arr[-1])
            out[f"{key}_length"] = int(arr.size)
    return out


def run_linear_fisher_convergence(
    *,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    theta_grid: np.ndarray,
    period: float,
    n_list: Iterable[int],
    n_repeats: int,
    train_frac: float,
    seed: int,
    device: torch.device,
    flow_config: ContinuousFlowConfig,
    output_dir: Path,
    classical_ridge: float = 1e-6,
    classical_window_radius: float | None = None,
    classical_min_endpoint_samples: int = 8,
    save_flow_npz: bool = True,
    metadata: dict[str, Any] | None = None,
) -> LinearFisherConvergenceResult:
    theta = np.mod(np.asarray(theta_all, dtype=np.float64).reshape(-1), float(period))
    x = np.asarray(x_all, dtype=np.float64)
    grid = np.asarray(theta_grid, dtype=np.float64).reshape(-1, 1)
    mids = theta_midpoints(grid)
    n_values = np.asarray(list(n_list), dtype=np.int64)
    if n_values.ndim != 1 or int(n_values.shape[0]) < 1:
        raise ValueError("n_list must contain at least one n.")
    if np.any(n_values < 2):
        raise ValueError("All n_list values must be >= 2.")
    if int(n_repeats) < 1:
        raise ValueError("n_repeats must be >= 1.")
    if np.any(n_values > int(theta.shape[0])):
        raise ValueError("n_list contains values larger than the number of trials.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_indices = np.arange(int(theta.shape[0]), dtype=np.int64)
    ref_train, ref_val = split_train_validation(all_indices, train_frac=float(train_frac), seed=int(seed))
    ref_meta = {"dataset_family": "stringer_pca", "theta_low": 0.0, "theta_high": float(period), "n_total": int(theta.shape[0])}
    ref_meta.update(metadata or {})
    ref_bundle = make_shared_bundle(theta_all=theta, x_all=x, train_idx=ref_train, validation_idx=ref_val, meta=ref_meta)

    print("[stringer-convergence] computing all-data classical_linear reference", flush=True)
    references: dict[str, np.ndarray] = {
        METHOD_CLASSICAL_LINEAR: classical_linear_fisher_circular(
            theta_all=theta,
            x_all=x,
            theta_grid=grid,
            period=float(period),
            ridge=float(classical_ridge),
            window_radius=classical_window_radius,
            min_endpoint_samples=int(classical_min_endpoint_samples),
        )
    }
    print("[stringer-convergence] training all-data flow_linear reference", flush=True)
    flow_ref_npz = output_dir / "flow" / "reference_all_data_flow_linear_results.npz" if save_flow_npz else None
    flow_ref, flow_ref_train_meta, flow_ref_path = train_flow_linear_curve(
        bundle=ref_bundle,
        theta_grid=grid,
        device=device,
        config=flow_config,
        seed=int(seed),
        output_npz=flow_ref_npz,
    )
    references[METHOD_FLOW_LINEAR] = flow_ref

    n_count = int(n_values.shape[0])
    r_count = int(n_repeats)
    n_mid = int(mids.shape[0])
    curves = {
        METHOD_CLASSICAL_LINEAR: np.full((n_count, r_count, n_mid), np.nan, dtype=np.float64),
        METHOD_FLOW_LINEAR: np.full((n_count, r_count, n_mid), np.nan, dtype=np.float64),
    }
    metrics = {METHOD_CLASSICAL_LINEAR: _metric_arrays(n_count, r_count), METHOD_FLOW_LINEAR: _metric_arrays(n_count, r_count)}
    rows: list[dict[str, Any]] = []
    flow_paths: dict[str, str] = {}
    if flow_ref_path is not None:
        flow_paths["reference_all_data"] = str(flow_ref_path)

    for ni, n_total in enumerate(n_values):
        for repeat_idx in range(r_count):
            repeat_seed = int(seed) + 1000 * int(n_total) + int(repeat_idx)
            print(f"[stringer-convergence] n={int(n_total)} repeat={repeat_idx} seed={repeat_seed}", flush=True)
            subset = stratified_subset_indices(
                theta,
                n_total=int(n_total),
                n_bins=int(grid.shape[0] - 1),
                period=float(period),
                seed=repeat_seed,
            )
            tr, va = split_train_validation(subset, train_frac=float(train_frac), seed=repeat_seed + 17)
            sub_meta = dict(ref_meta)
            sub_meta.update({"n_total": int(n_total), "repeat_idx": int(repeat_idx), "repeat_seed": int(repeat_seed)})
            sub_bundle = make_shared_bundle(
                theta_all=theta[np.concatenate([tr, va])],
                x_all=x[np.concatenate([tr, va])],
                train_idx=np.arange(0, tr.shape[0], dtype=np.int64),
                validation_idx=np.arange(tr.shape[0], tr.shape[0] + va.shape[0], dtype=np.int64),
                meta=sub_meta,
            )
            classical_curve = classical_linear_fisher_circular(
                theta_all=theta[subset],
                x_all=x[subset],
                theta_grid=grid,
                period=float(period),
                ridge=float(classical_ridge),
                window_radius=classical_window_radius,
                min_endpoint_samples=int(classical_min_endpoint_samples),
            )
            flow_npz = output_dir / "flow" / f"n{int(n_total)}_repeat{repeat_idx:02d}_flow_linear_results.npz" if save_flow_npz else None
            flow_curve, _flow_train_meta, flow_path = train_flow_linear_curve(
                bundle=sub_bundle,
                theta_grid=grid,
                device=device,
                config=flow_config,
                seed=repeat_seed,
                output_npz=flow_npz,
            )
            if flow_path is not None:
                flow_paths[f"n{int(n_total)}_repeat{repeat_idx:02d}"] = str(flow_path)

            for method, curve, ref in (
                (METHOD_CLASSICAL_LINEAR, classical_curve, references[METHOD_CLASSICAL_LINEAR]),
                (METHOD_FLOW_LINEAR, flow_curve, references[METHOD_FLOW_LINEAR]),
            ):
                curves[method][ni, repeat_idx, :] = curve
                met = curve_metrics(curve, ref, mids)
                for metric_name, metric_value in met.items():
                    metrics[method][metric_name][ni, repeat_idx] = metric_value
                for ti, midpoint in enumerate(mids.reshape(-1)):
                    rows.append(
                        {
                            "method": method,
                            "n_total": int(n_total),
                            "repeat_idx": int(repeat_idx),
                            "repeat_seed": int(repeat_seed),
                            "theta_midpoint": float(midpoint),
                            "theta_left": float(grid[ti, 0]),
                            "theta_right": float(grid[ti + 1, 0]),
                            "fisher": float(curve[ti]),
                            "reference_fisher": float(ref[ti]),
                            "abs_error": float(abs(curve[ti] - ref[ti])),
                            "rmse": float(met["rmse"]),
                            "mae": float(met["mae"]),
                            "pearson": float(met["pearson"]),
                            "area_normalized_l2": float(met["area_normalized_l2"]),
                        }
                    )

    result_meta = dict(metadata or {})
    result_meta.update(
        {
            "methods": list(METHODS),
            "reference": "method_specific_all_data",
            "orientation_period": float(period),
            "theta_grid_size": int(grid.shape[0]),
            "n_trials_all_data": int(theta.shape[0]),
            "train_frac": float(train_frac),
            "seed": int(seed),
            "flow_reference_train_metadata": _json_ready(_compact_train_metadata(flow_ref_train_meta)),
            "flow_npz_paths": flow_paths,
        }
    )
    return LinearFisherConvergenceResult(
        theta_grid=grid,
        theta_midpoints=mids,
        n_list=n_values,
        repeat_indices=np.arange(r_count, dtype=np.int64),
        references=references,
        curves=curves,
        metrics=metrics,
        rows=rows,
        metadata=result_meta,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_results_npz(path: Path, result: LinearFisherConvergenceResult, pca: PCAProjectionResult | None = None) -> Path:
    fields: dict[str, Any] = {
        "theta_grid": np.asarray(result.theta_grid, dtype=np.float64),
        "theta_midpoints": np.asarray(result.theta_midpoints, dtype=np.float64),
        "n_list": np.asarray(result.n_list, dtype=np.int64),
        "repeat_indices": np.asarray(result.repeat_indices, dtype=np.int64),
    }
    for method in METHODS:
        fields[f"{method}_reference_fisher"] = np.asarray(result.references[method], dtype=np.float64)
        fields[f"{method}_subset_fisher"] = np.asarray(result.curves[method], dtype=np.float64)
        for metric_name, arr in result.metrics[method].items():
            fields[f"{method}_{metric_name}"] = np.asarray(arr, dtype=np.float64)
    if pca is not None:
        fields["pca_x_all"] = np.asarray(pca.x_all, dtype=np.float64)
        fields["pca_explained_variance_ratio"] = np.asarray(pca.explained_variance_ratio, dtype=np.float64)
        fields["pca_singular_values"] = np.asarray(pca.singular_values, dtype=np.float64)
        fields["pca_components"] = np.asarray(pca.components, dtype=np.float64)
        fields["pca_mean"] = np.asarray(pca.mean, dtype=np.float64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **fields)
    return path


def write_curves_csv(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    columns = (
        "method",
        "n_total",
        "repeat_idx",
        "repeat_seed",
        "theta_midpoint",
        "theta_left",
        "theta_right",
        "fisher",
        "reference_fisher",
        "abs_error",
        "mae",
        "rmse",
        "pearson",
        "area_normalized_l2",
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    return path


def write_summary_json(path: Path, result: LinearFisherConvergenceResult, *, extra: dict[str, Any] | None = None) -> Path:
    summary: dict[str, Any] = {
        "methods": list(METHODS),
        "metrics": list(METRICS),
        "metadata": _json_ready(result.metadata),
        "metric_summary": {},
    }
    for method in METHODS:
        summary["metric_summary"][method] = {}
        for metric_name, arr in result.metrics[method].items():
            summary["metric_summary"][method][metric_name] = {
                "mean_by_n": np.nanmean(arr, axis=1).tolist(),
                "sd_by_n": np.nanstd(arr, axis=1, ddof=1).tolist() if arr.shape[1] > 1 else np.zeros(arr.shape[0]).tolist(),
            }
    if extra:
        summary.update(_json_ready(extra))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def plot_abs_error(path_svg: Path, path_png: Path, result: LinearFisherConvergenceResult) -> tuple[Path, Path]:
    n_values = np.asarray(result.n_list, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.6, 4.8), layout="constrained")
    for method in METHODS:
        vals = np.asarray(result.metrics[method]["mae"], dtype=np.float64)
        mean = np.nanmean(vals, axis=1)
        sd = np.nanstd(vals, axis=1, ddof=1) if vals.shape[1] > 1 else np.zeros(vals.shape[0], dtype=np.float64)
        ax.errorbar(n_values, mean, yerr=sd, marker="o", linewidth=1.8, capsize=4, label=method)
    ax.set_xlabel("subset trials")
    ax.set_ylabel("MAE to method-specific all-data reference")
    ax.set_title("Stringer linear Fisher convergence")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    path_svg = Path(path_svg)
    path_png = Path(path_png)
    path_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_svg)
    fig.savefig(path_png, dpi=180)
    plt.close(fig)
    return path_svg, path_png


def plot_curve_examples(path_svg: Path, path_png: Path, result: LinearFisherConvergenceResult) -> tuple[Path, Path]:
    mids = np.asarray(result.theta_midpoints, dtype=np.float64).reshape(-1)
    n_values = np.asarray(result.n_list, dtype=np.int64)
    example_indices = [0] if len(n_values) == 1 else [0, len(n_values) - 1]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), layout="constrained", sharex=True)
    for ax, method in zip(axes, METHODS, strict=True):
        ax.plot(mids, result.references[method], linewidth=2.2, color="black", label="all-data reference")
        for ni in example_indices:
            ax.plot(mids, result.curves[method][ni, 0, :], linewidth=1.4, label=f"n={int(n_values[ni])}, repeat=0")
        ax.set_title(method)
        ax.set_xlabel("orientation")
        ax.set_ylabel("linear Fisher")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    path_svg = Path(path_svg)
    path_png = Path(path_png)
    path_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_svg)
    fig.savefig(path_png, dpi=180)
    plt.close(fig)
    return path_svg, path_png
