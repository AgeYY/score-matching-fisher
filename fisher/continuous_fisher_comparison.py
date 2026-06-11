"""Continuous PR Fisher comparison for scalar ``randamp_gaussian_sqrtd`` data."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fisher.distance_comparison import encode_with_pr_autoencoder, save_flow_result_npz
from fisher.flow_matching_skl import (
    FlowSKLResult,
    build_flow_skl_model,
    estimate_adjacent_model_jeffreys_fisher,
    estimate_affine_mixed_covariance_fisher,
    train_flow_skl_model,
)
from fisher.shared_dataset_io import SharedDatasetBundle, load_shared_dataset_npz
from fisher.shared_fisher_est import analytic_fisher_curve, build_dataset_from_meta

REPO_ROOT = Path(__file__).resolve().parent.parent

RESULTS_NPZ_NAME = "continuous_pr_fisher_results.npz"
CURVES_CSV_NAME = "continuous_pr_fisher_curves.csv"
SUMMARY_JSON_NAME = "continuous_pr_fisher_summary.json"
CURVES_SVG_NAME = "continuous_pr_fisher_curves.svg"
CURVES_PNG_NAME = "continuous_pr_fisher_curves.png"

METHOD_FLOW_LINEAR = "flow_linear"
METHOD_FLOW_FULL = "flow_full"
METHOD_CLASSICAL_LINEAR = "classical_linear"
METHOD_CLASSICAL_FULL = "classical_full"
METHOD_GT_NATIVE_FULL = "ground_truth_native_full"
METHOD_GT_NATIVE_LINEAR = "ground_truth_native_linear"
METHOD_GT_PR_LINEAR = "ground_truth_pr_linear"

CSV_COLUMNS = (
    "method",
    "theta_midpoint",
    "theta_left",
    "theta_right",
    "fisher",
    "reference",
    "abs_error",
    "rel_error",
)


@dataclass(frozen=True)
class ContinuousFlowConfig:
    epochs: int = 20_000
    early_patience: int = 1_000
    early_min_delta: float = 1e-4
    early_ema_alpha: float = 0.05
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 0.0
    hidden_dim: int = 256
    depth: int = 5
    path_schedule: str = "cosine"
    t_eps: float = 0.0005
    quadrature_steps: int = 64
    mc_jeffreys_sample: int = 4096
    ode_steps: int = 64
    ode_method: str = "midpoint"
    divergence_estimator: str = "exact"
    hutchinson_probes: int = 1
    shared_affine_a_diag_jitter: float = 1e-3
    solve_jitter: float = 1e-6
    max_grad_norm: float = 10.0
    log_every: int = 50
    affine_ridge: float = 1e-6


@dataclass(frozen=True)
class ClassicalConfig:
    linear_ridge: float = 1e-6
    window_radius: float | None = None
    min_endpoint_samples: int = 2
    skl_folds: int = 5
    skl_logistic_c: float = 1.0


@dataclass(frozen=True)
class ContinuousFisherResult:
    theta_grid: np.ndarray
    theta_midpoints: np.ndarray
    curves: dict[str, np.ndarray]
    references: dict[str, np.ndarray]
    errors: dict[str, dict[str, np.ndarray]]
    rows: list[dict[str, Any]]
    flow_npz_paths: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_pr_dim(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError("--pr-dim must be an integer, 'none', or 'null'.") from exc


def theta_grid_from_meta(meta: dict[str, Any], *, theta_grid_size: int) -> np.ndarray:
    n = int(theta_grid_size)
    if n < 2:
        raise ValueError("theta_grid_size must be >= 2.")
    return np.linspace(float(meta["theta_low"]), float(meta["theta_high"]), n, dtype=np.float64).reshape(-1, 1)


def theta_midpoints(theta_grid: np.ndarray) -> np.ndarray:
    th = np.asarray(theta_grid, dtype=np.float64).reshape(-1, 1)
    return (0.5 * (th[:-1, 0] + th[1:, 0])).reshape(-1, 1).astype(np.float64)


def native_linear_fisher_curve(theta: np.ndarray, dataset: Any) -> np.ndarray:
    t = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
    dmu = dataset.tuning_curve_derivative(t)
    cov = dataset.covariance(t)
    inv_cov = np.linalg.inv(cov)
    return np.einsum("bi,bij,bj->b", dmu, inv_cov, dmu).astype(np.float64)


def native_ground_truth_curves(theta_mid: np.ndarray, native_meta: dict[str, Any]) -> dict[str, np.ndarray]:
    dataset = build_dataset_from_meta(native_meta)
    return {
        METHOD_GT_NATIVE_FULL: analytic_fisher_curve(theta_mid, dataset),
        METHOD_GT_NATIVE_LINEAR: native_linear_fisher_curve(theta_mid, dataset),
    }


def _load_script_module(name: str, relative: str) -> Any:
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_native_dataset_npz(
    *,
    output_npz: Path,
    dataset_family: str,
    x_dim: int,
    n_total: int,
    train_frac: float,
    seed: int,
    force: bool = False,
) -> Path:
    output_npz = Path(output_npz)
    if output_npz.is_file() and not bool(force):
        return output_npz
    mod = _load_script_module("make_dataset", "bin/make_dataset.py")
    args = mod.parse_make_dataset_args(
        [
            "--dataset-family",
            str(dataset_family),
            "--x-dim",
            str(int(x_dim)),
            "--n-total",
            str(int(n_total)),
            "--train-frac",
            str(float(train_frac)),
            "--seed",
            str(int(seed)),
            "--output-npz",
            str(output_npz),
        ]
    )
    mod.validate_dataset_sample_args(args)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    dataset = mod.build_dataset_from_args(args)
    rng = np.random.default_rng(int(seed))
    theta_all, x_all = dataset.sample_joint(int(n_total))
    perm = rng.permutation(int(n_total))
    n_train = int(float(train_frac) * int(n_total))
    n_train = min(max(n_train, 1), int(n_total) - 1)
    tr_idx = perm[:n_train]
    va_idx = perm[n_train:]
    meta = mod.meta_dict_from_args(args)
    if str(dataset_family) in ("randamp_gaussian", "randamp_gaussian_sqrtd", "randamp_gaussian2d_sqrtd"):
        meta["randamp_mu_amp_per_dim"] = dataset._randamp_amp.tolist()
    mod.save_shared_dataset_npz(
        output_npz,
        meta=meta,
        theta_all=theta_all,
        x_all=x_all,
        train_idx=tr_idx.astype(np.int64),
        validation_idx=va_idx.astype(np.int64),
        theta_train=theta_all[tr_idx],
        x_train=x_all[tr_idx],
        theta_validation=theta_all[va_idx],
        x_validation=x_all[va_idx],
    )
    return output_npz


def project_pr_dataset_npz(
    *,
    input_npz: Path,
    output_npz: Path,
    pr_dim: int,
    device: str,
    seed: int,
    cache_dir: Path,
    use_cache: bool = False,
    force: bool = False,
    pr_train_epochs: int | None = None,
    pr_train_samples: int | None = None,
    pr_train_batch_size: int | None = None,
    pr_train_lr: float | None = None,
    skip_viz: bool = True,
) -> Path:
    output_npz = Path(output_npz)
    if output_npz.is_file() and not bool(force):
        return output_npz
    cmd = [
        sys.executable,
        str(REPO_ROOT / "bin" / "project_dataset_pr_autoencoder.py"),
        "--input-npz",
        str(input_npz),
        "--output-npz",
        str(output_npz),
        "--h-dim",
        str(int(pr_dim)),
        "--device",
        str(device),
        "--seed",
        str(int(seed)),
        "--cache-dir",
        str(cache_dir),
    ]
    if bool(use_cache):
        cmd.append("--use-cache")
    if bool(skip_viz):
        cmd.append("--skip-viz")
    if pr_train_epochs is not None:
        cmd.extend(["--pr-train-epochs", str(int(pr_train_epochs))])
    if pr_train_samples is not None:
        cmd.extend(["--pr-train-samples", str(int(pr_train_samples))])
    if pr_train_batch_size is not None:
        cmd.extend(["--pr-train-batch-size", str(int(pr_train_batch_size))])
    if pr_train_lr is not None:
        cmd.extend(["--pr-train-lr", str(float(pr_train_lr))])
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    return output_npz


def _endpoint_windows(
    theta_all: np.ndarray,
    x_all: np.ndarray,
    theta_grid: np.ndarray,
    *,
    radius: float | None,
    min_endpoint_samples: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    th = np.asarray(theta_all, dtype=np.float64).reshape(-1)
    x = np.asarray(x_all, dtype=np.float64)
    grid = np.asarray(theta_grid, dtype=np.float64).reshape(-1)
    if grid.shape[0] < 2:
        raise ValueError("theta_grid must contain at least two points.")
    if radius is None:
        radius_val = 0.5 * float(np.min(np.diff(grid)))
    else:
        radius_val = float(radius)
    if radius_val <= 0.0:
        raise ValueError("window radius must be positive.")
    windows: list[tuple[np.ndarray, np.ndarray]] = []
    need = int(min_endpoint_samples)
    for val in grid:
        idx = np.flatnonzero(np.abs(th - float(val)) <= radius_val)
        if int(idx.size) < need:
            idx = np.argsort(np.abs(th - float(val)), kind="mergesort")[:need]
        windows.append((th[idx].reshape(-1, 1), x[idx]))
    return windows


def classical_linear_fisher(
    *,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    theta_grid: np.ndarray,
    ridge: float = 1e-6,
    window_radius: float | None = None,
    min_endpoint_samples: int = 2,
) -> np.ndarray:
    grid = np.asarray(theta_grid, dtype=np.float64).reshape(-1)
    windows = _endpoint_windows(
        theta_all,
        x_all,
        theta_grid,
        radius=window_radius,
        min_endpoint_samples=int(min_endpoint_samples),
    )
    d = int(np.asarray(x_all).shape[1])
    out = np.full(grid.shape[0] - 1, np.nan, dtype=np.float64)
    eye = np.eye(d, dtype=np.float64)
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


def _cross_fitted_pair_jeffreys(
    x_l: np.ndarray,
    x_r: np.ndarray,
    *,
    n_splits: int,
    seed: int,
    logistic_c: float,
) -> float:
    if int(x_l.shape[0]) < 2 or int(x_r.shape[0]) < 2:
        raise ValueError("Need at least two samples per endpoint for cross-fitted logistic SKL.")
    x = np.vstack([x_l, x_r])
    y = np.concatenate([np.ones(int(x_l.shape[0]), dtype=np.int64), np.zeros(int(x_r.shape[0]), dtype=np.int64)])
    folds = min(int(n_splits), int(x_l.shape[0]), int(x_r.shape[0]))
    if folds < 2:
        raise ValueError("n_splits is too large for available endpoint samples.")
    logits = np.zeros(int(y.shape[0]), dtype=np.float64)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
    for train_idx, test_idx in cv.split(x, y):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=float(logistic_c), class_weight="balanced", max_iter=2000, solver="lbfgs"),
        )
        clf.fit(x[train_idx], y[train_idx])
        logits[test_idx] = np.asarray(clf.decision_function(x[test_idx]), dtype=np.float64)
    return max(0.0, float(np.mean(logits[y == 1], dtype=np.float64) - np.mean(logits[y == 0], dtype=np.float64)))


def classical_full_fisher(
    *,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    theta_grid: np.ndarray,
    window_radius: float | None = None,
    min_endpoint_samples: int = 2,
    n_splits: int = 5,
    seed: int = 7,
    logistic_c: float = 1.0,
) -> np.ndarray:
    grid = np.asarray(theta_grid, dtype=np.float64).reshape(-1)
    windows = _endpoint_windows(
        theta_all,
        x_all,
        theta_grid,
        radius=window_radius,
        min_endpoint_samples=max(int(min_endpoint_samples), 2),
    )
    out = np.full(grid.shape[0] - 1, np.nan, dtype=np.float64)
    for i in range(grid.shape[0] - 1):
        skl = _cross_fitted_pair_jeffreys(
            windows[i][1],
            windows[i + 1][1],
            n_splits=int(n_splits),
            seed=int(seed) + i,
            logistic_c=float(logistic_c),
        )
        dtheta = float(grid[i + 1] - grid[i])
        out[i] = skl / (dtheta**2)
    return out


def projected_mc_linear_fisher(
    *,
    native_meta: dict[str, Any],
    projected_meta: dict[str, Any],
    theta_grid: np.ndarray,
    device: torch.device,
    cache_dir: Path,
    samples_per_endpoint: int,
    seed: int,
    batch_size: int,
    ridge: float,
) -> np.ndarray:
    dataset = build_dataset_from_meta(native_meta)
    rng = np.random.default_rng(int(seed))
    if hasattr(dataset, "rng"):
        dataset.rng = rng
    grid = np.asarray(theta_grid, dtype=np.float64).reshape(-1)
    endpoints: list[np.ndarray] = []
    for th in grid:
        t = np.full((int(samples_per_endpoint), 1), float(th), dtype=np.float64)
        z = dataset.sample_x(t)
        endpoints.append(
            encode_with_pr_autoencoder(
                z,
                projected_meta=projected_meta,
                device=device,
                cache_dir=cache_dir,
                batch_size=int(batch_size),
            )
        )
    return classical_linear_fisher(
        theta_all=np.repeat(grid, int(samples_per_endpoint)).reshape(-1, 1),
        x_all=np.vstack(endpoints),
        theta_grid=theta_grid,
        ridge=float(ridge),
        window_radius=0.0 + 1e-12,
        min_endpoint_samples=int(samples_per_endpoint),
    )


def train_flow_fisher_curves(
    *,
    bundle: SharedDatasetBundle,
    theta_grid: np.ndarray,
    device: torch.device,
    output_dir: Path,
    config: ContinuousFlowConfig,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Path]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    output_dir.mkdir(parents=True, exist_ok=True)
    theta_dim = 1
    x_dim = int(np.asarray(bundle.x_train).shape[1])
    curves: dict[str, np.ndarray] = {}
    paths: dict[str, Path] = {}

    for method, family in ((METHOD_FLOW_LINEAR, "condition_affine"), (METHOD_FLOW_FULL, "nonlinear")):
        print(f"[continuous-fisher] training {method} velocity_family={family}", flush=True)
        model = build_flow_skl_model(
            velocity_family=family,
            theta_dim=theta_dim,
            x_dim=x_dim,
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
            velocity_family=family,
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
        if method == METHOD_FLOW_LINEAR:
            fd = estimate_affine_mixed_covariance_fisher(
                model=model,
                theta_all=theta_grid,
                device=device,
                ridge=float(config.affine_ridge),
                ode_steps=int(config.ode_steps),
            )
            curves[method] = fd["fisher"]
            result = FlowSKLResult(
                symmetric_kl_matrix=np.zeros((int(theta_grid.shape[0]), int(theta_grid.shape[0])), dtype=np.float64),
                canonical_metric_matrix=np.zeros((int(theta_grid.shape[0]), int(theta_grid.shape[0])), dtype=np.float64),
                canonical_metric_name="affine_mixed_covariance_linear_fisher",
                fisher_theta_midpoints=fd["theta_midpoints"],
                fisher_linear=fd["fisher"],
                train_metadata=meta,
            )
        else:
            fd = estimate_adjacent_model_jeffreys_fisher(
                model=model,
                theta_all=theta_grid,
                device=device,
                mc_jeffreys_sample=int(config.mc_jeffreys_sample),
                ode_steps=int(config.ode_steps),
                ode_method=str(config.ode_method),
                batch_size=int(config.batch_size),
                solve_jitter=float(config.solve_jitter),
                quadrature_steps=int(config.quadrature_steps),
            )
            curves[method] = fd["fisher"]
            result = FlowSKLResult(
                symmetric_kl_matrix=np.zeros((int(theta_grid.shape[0]), int(theta_grid.shape[0])), dtype=np.float64),
                canonical_metric_matrix=np.zeros((int(theta_grid.shape[0]), int(theta_grid.shape[0])), dtype=np.float64),
                canonical_metric_name="adjacent_model_jeffreys_sum",
                fisher_theta_midpoints=fd["theta_midpoints"],
                fisher_full=fd["fisher"],
                train_metadata=meta,
            )
        paths[method] = save_flow_result_npz(
            output_dir / f"{method}_flow_matching_skl_results.npz",
            result=result,
            metric=method,
            theta_eval=theta_grid,
            velocity_family=family,
        )
    return curves, paths


def assemble_rows(
    *,
    theta_grid: np.ndarray,
    curves: dict[str, np.ndarray],
    references: dict[str, np.ndarray],
    reference_by_method: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    grid = np.asarray(theta_grid, dtype=np.float64).reshape(-1)
    mids = 0.5 * (grid[:-1] + grid[1:])
    rows: list[dict[str, Any]] = []
    errors: dict[str, dict[str, np.ndarray]] = {}
    for method, vals_raw in curves.items():
        vals = np.asarray(vals_raw, dtype=np.float64).reshape(-1)
        ref_name = reference_by_method.get(method, METHOD_GT_NATIVE_FULL)
        ref = np.asarray(references[ref_name], dtype=np.float64).reshape(-1)
        abs_err = np.abs(vals - ref)
        rel_err = abs_err / np.maximum(np.abs(ref), 1e-12)
        errors[method] = {"abs_error": abs_err, "rel_error": rel_err}
        for i, val in enumerate(vals):
            rows.append(
                {
                    "method": method,
                    "theta_midpoint": float(mids[i]),
                    "theta_left": float(grid[i]),
                    "theta_right": float(grid[i + 1]),
                    "fisher": float(val),
                    "reference": ref_name,
                    "abs_error": float(abs_err[i]),
                    "rel_error": float(rel_err[i]),
                }
            )
    return rows, errors


def write_curves_csv(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})
    return path


def write_results_npz(path: Path, result: ContinuousFisherResult) -> Path:
    fields: dict[str, Any] = {
        "theta_grid": np.asarray(result.theta_grid, dtype=np.float64),
        "theta_midpoints": np.asarray(result.theta_midpoints, dtype=np.float64),
        "skl_convention": np.asarray(["jeffreys_sum"]),
    }
    for name, arr in result.curves.items():
        fields[f"{name}_fisher"] = np.asarray(arr, dtype=np.float64)
    for name, arr in result.references.items():
        fields[f"{name}_fisher"] = np.asarray(arr, dtype=np.float64)
    for name, err in result.errors.items():
        fields[f"{name}_abs_error"] = np.asarray(err["abs_error"], dtype=np.float64)
        fields[f"{name}_rel_error"] = np.asarray(err["rel_error"], dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **fields)
    return path


def write_summary_json(path: Path, result: ContinuousFisherResult, *, extra: dict[str, Any] | None = None) -> Path:
    summary: dict[str, Any] = {
        "skl_convention": "jeffreys_sum",
        "methods": sorted(result.curves.keys()),
        "references": sorted(result.references.keys()),
        "flow_npz_paths": {k: str(v) for k, v in result.flow_npz_paths.items()},
        "error_summary": {},
    }
    for method, err in result.errors.items():
        summary["error_summary"][method] = {
            "mae": float(np.nanmean(err["abs_error"])),
            "rmse": float(np.sqrt(np.nanmean(err["abs_error"] ** 2))),
            "mean_rel_error": float(np.nanmean(err["rel_error"])),
        }
    summary.update(result.metadata)
    if extra:
        summary.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def plot_curves(path_svg: Path, path_png: Path, result: ContinuousFisherResult) -> tuple[Path, Path]:
    mids = np.asarray(result.theta_midpoints, dtype=np.float64).reshape(-1)
    fig, ax = plt.subplots(figsize=(8.4, 5.2), layout="constrained")
    for name, vals in result.references.items():
        ax.plot(mids, vals, linewidth=2.0, linestyle="--", label=name)
    for name, vals in result.curves.items():
        ax.plot(mids, vals, linewidth=1.5, label=name)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("scalar Fisher")
    ax.set_title("Continuous PR Fisher sweep")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    path_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_svg)
    fig.savefig(path_png, dpi=180)
    plt.close(fig)
    return path_svg, path_png


def run_continuous_comparison(
    *,
    native_bundle: SharedDatasetBundle,
    work_bundle: SharedDatasetBundle,
    theta_grid_size: int,
    device: torch.device,
    output_dir: Path,
    flow_config: ContinuousFlowConfig,
    classical_config: ClassicalConfig,
    seed: int,
    pr_projected: bool,
    pr_cache_dir: Path,
    gt_pr_samples_per_endpoint: int,
    gt_batch_size: int,
) -> ContinuousFisherResult:
    output_dir = Path(output_dir)
    grid = theta_grid_from_meta(native_bundle.meta, theta_grid_size=int(theta_grid_size))
    mids = theta_midpoints(grid)
    references = native_ground_truth_curves(mids, dict(native_bundle.meta))
    if bool(pr_projected):
        references[METHOD_GT_PR_LINEAR] = projected_mc_linear_fisher(
            native_meta=dict(native_bundle.meta),
            projected_meta=dict(work_bundle.meta),
            theta_grid=grid,
            device=device,
            cache_dir=Path(pr_cache_dir),
            samples_per_endpoint=int(gt_pr_samples_per_endpoint),
            seed=int(seed) + 12345,
            batch_size=int(gt_batch_size),
            ridge=float(classical_config.linear_ridge),
        )

    print("[continuous-fisher] computing classical estimators", flush=True)
    curves: dict[str, np.ndarray] = {
        METHOD_CLASSICAL_LINEAR: classical_linear_fisher(
            theta_all=work_bundle.theta_all,
            x_all=work_bundle.x_all,
            theta_grid=grid,
            ridge=float(classical_config.linear_ridge),
            window_radius=classical_config.window_radius,
            min_endpoint_samples=int(classical_config.min_endpoint_samples),
        ),
        METHOD_CLASSICAL_FULL: classical_full_fisher(
            theta_all=work_bundle.theta_all,
            x_all=work_bundle.x_all,
            theta_grid=grid,
            window_radius=classical_config.window_radius,
            min_endpoint_samples=max(int(classical_config.min_endpoint_samples), int(classical_config.skl_folds)),
            n_splits=int(classical_config.skl_folds),
            seed=int(seed),
            logistic_c=float(classical_config.skl_logistic_c),
        ),
    }

    print("[continuous-fisher] training flow estimators", flush=True)
    flow_curves, flow_paths = train_flow_fisher_curves(
        bundle=work_bundle,
        theta_grid=grid,
        device=device,
        output_dir=output_dir / "flow",
        config=flow_config,
        seed=int(seed),
    )
    curves.update(flow_curves)

    linear_ref = METHOD_GT_PR_LINEAR if bool(pr_projected) else METHOD_GT_NATIVE_LINEAR
    ref_by_method = {
        METHOD_CLASSICAL_LINEAR: linear_ref,
        METHOD_FLOW_LINEAR: linear_ref,
        METHOD_CLASSICAL_FULL: METHOD_GT_NATIVE_FULL,
        METHOD_FLOW_FULL: METHOD_GT_NATIVE_FULL,
    }
    rows, errors = assemble_rows(theta_grid=grid, curves=curves, references=references, reference_by_method=ref_by_method)
    return ContinuousFisherResult(
        theta_grid=grid,
        theta_midpoints=mids,
        curves=curves,
        references=references,
        errors=errors,
        rows=rows,
        flow_npz_paths=flow_paths,
        metadata={"reference_by_method": ref_by_method},
    )
