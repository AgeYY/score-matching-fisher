"""Stringer grating-session identification from half-session Fisher curves."""

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
from fisher.stringer_dataset import StringerSessionInfo, load_stringer_session

RESULTS_NPZ_NAME = "stringer_session_identification_results.npz"
CURVES_CSV_NAME = "stringer_session_identification_curves.csv"
PAIRS_CSV_NAME = "stringer_session_identification_pairs.csv"
SUMMARY_JSON_NAME = "stringer_session_identification_summary.json"
HEATMAPS_SVG_NAME = "stringer_session_identification_primary_heatmaps.svg"
HEATMAPS_PNG_NAME = "stringer_session_identification_primary_heatmaps.png"
RANKS_SVG_NAME = "stringer_session_identification_ranks.svg"
RANKS_PNG_NAME = "stringer_session_identification_ranks.png"
ALL_DISTANCE_SUMMARY_SVG_NAME = "stringer_session_identification_all_distance_summary.svg"
ALL_DISTANCE_SUMMARY_PNG_NAME = "stringer_session_identification_all_distance_summary.png"

METHODS = (METHOD_CLASSICAL_LINEAR, METHOD_FLOW_LINEAR)
DISTANCE_PRIMARY = "log_correlation"
DISTANCE_AREA_L2 = "area_normalized_l2"
DISTANCE_RMSE = "raw_rmse"
DISTANCES = (DISTANCE_PRIMARY, DISTANCE_AREA_L2, DISTANCE_RMSE)
HALF_A = "A"
HALF_B = "B"
DIRECTION_A_TO_B = "A_to_B"


@dataclass(frozen=True)
class PCAProjectionResult:
    x_all: np.ndarray
    explained_variance_ratio: np.ndarray
    singular_values: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class HalfSplit:
    session_index: int
    session_key: str
    session_file: str
    half_label: str
    indices: np.ndarray


@dataclass(frozen=True)
class HalfCurveResult:
    session_index: int
    session_key: str
    session_file: str
    half_label: str
    n_trials: int
    n_neurons: int
    theta_grid: np.ndarray
    theta_midpoints: np.ndarray
    curves: dict[str, np.ndarray]
    pca_metadata: dict[str, Any]
    train_metadata: dict[str, Any]
    cache_path: Path
    flow_npz_path: Path | None


@dataclass(frozen=True)
class IdentificationResult:
    session_keys: list[str]
    theta_grid: np.ndarray
    theta_midpoints: np.ndarray
    half_results: list[HalfCurveResult]
    distances: dict[str, dict[str, dict[str, np.ndarray]]]
    pair_rows: list[dict[str, Any]]
    curve_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def parse_optional_int(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text in {"none", "null", "all", ""}:
        return None
    out = int(text)
    if out < 1:
        raise ValueError("integer value must be positive.")
    return out


def theta_grid_periodic(period: float, theta_grid_size: int) -> np.ndarray:
    if int(theta_grid_size) < 2:
        raise ValueError("theta_grid_size must be >= 2.")
    if float(period) <= 0.0:
        raise ValueError("period must be positive.")
    return np.linspace(0.0, float(period), int(theta_grid_size), dtype=np.float64).reshape(-1, 1)


def theta_midpoints(theta_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(theta_grid, dtype=np.float64).reshape(-1)
    if grid.shape[0] < 2:
        raise ValueError("theta_grid must have at least two endpoints.")
    return (0.5 * (grid[:-1] + grid[1:])).reshape(-1, 1)


def circular_distance(theta: np.ndarray, center: float, period: float) -> np.ndarray:
    if float(period) <= 0.0:
        raise ValueError("period must be positive.")
    delta = np.mod(np.asarray(theta, dtype=np.float64).reshape(-1) - float(center) + 0.5 * float(period), float(period))
    return np.abs(delta - 0.5 * float(period)).astype(np.float64)


def circular_endpoint_windows(
    *,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    theta_grid: np.ndarray,
    period: float,
    radius: float | None,
    min_endpoint_samples: int,
) -> list[np.ndarray]:
    th = np.mod(np.asarray(theta_all, dtype=np.float64).reshape(-1), float(period))
    x = np.asarray(x_all, dtype=np.float64)
    grid = np.asarray(theta_grid, dtype=np.float64).reshape(-1)
    if x.ndim != 2:
        raise ValueError("x_all must be 2D.")
    if th.shape[0] != x.shape[0]:
        raise ValueError("theta_all and x_all lengths must match.")
    if grid.shape[0] < 2:
        raise ValueError("theta_grid must contain at least two endpoints.")
    radius_val = 0.5 * float(np.min(np.diff(grid))) if radius is None else float(radius)
    if radius_val <= 0.0:
        raise ValueError("window radius must be positive.")
    need = int(min_endpoint_samples)
    if need < 1:
        raise ValueError("min_endpoint_samples must be >= 1.")
    if need > th.shape[0]:
        raise ValueError("min_endpoint_samples exceeds the number of observations.")
    windows: list[np.ndarray] = []
    for endpoint in grid:
        dist = circular_distance(th, float(endpoint), float(period))
        idx = np.flatnonzero(dist <= radius_val)
        if int(idx.size) < need:
            idx = np.argsort(dist, kind="mergesort")[:need]
        windows.append(idx.astype(np.int64))
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
    x = np.asarray(x_all, dtype=np.float64)
    grid = np.asarray(theta_grid, dtype=np.float64).reshape(-1)
    windows = circular_endpoint_windows(
        theta_all=theta_all,
        x_all=x,
        theta_grid=theta_grid,
        period=float(period),
        radius=window_radius,
        min_endpoint_samples=int(min_endpoint_samples),
    )
    d = int(x.shape[1])
    eye = np.eye(d, dtype=np.float64)
    out = np.full(grid.shape[0] - 1, np.nan, dtype=np.float64)
    for i in range(grid.shape[0] - 1):
        x_l = x[windows[i]]
        x_r = x[windows[i + 1]]
        dtheta = float(grid[i + 1] - grid[i])
        mu_prime = (np.mean(x_r, axis=0) - np.mean(x_l, axis=0)) / dtheta
        cov_l = np.cov(x_l, rowvar=False) if x_l.shape[0] > 1 else np.zeros((d, d), dtype=np.float64)
        cov_r = np.cov(x_r, rowvar=False) if x_r.shape[0] > 1 else np.zeros((d, d), dtype=np.float64)
        cov = 0.5 * (np.atleast_2d(cov_l) + np.atleast_2d(cov_r)) + float(ridge) * eye
        out[i] = max(0.0, float(mu_prime @ np.linalg.solve(cov, mu_prime)))
    return out


def stratified_half_split(
    theta_all: np.ndarray,
    *,
    n_bins: int,
    period: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.mod(np.asarray(theta_all, dtype=np.float64).reshape(-1), float(period))
    if int(n_bins) < 1:
        raise ValueError("n_bins must be >= 1.")
    bin_id = np.floor(theta / float(period) * int(n_bins)).astype(np.int64)
    bin_id = np.clip(bin_id, 0, int(n_bins) - 1)
    rng = np.random.default_rng(int(seed))
    half_a: list[np.ndarray] = []
    half_b: list[np.ndarray] = []
    give_extra_to_a = True
    for b in range(int(n_bins)):
        idx = np.flatnonzero(bin_id == b)
        rng.shuffle(idx)
        n_a = int(idx.shape[0] // 2)
        if idx.shape[0] % 2 == 1 and give_extra_to_a:
            n_a += 1
        give_extra_to_a = not give_extra_to_a
        half_a.append(idx[:n_a])
        half_b.append(idx[n_a:])
    a = np.concatenate(half_a).astype(np.int64)
    b = np.concatenate(half_b).astype(np.int64)
    rng.shuffle(a)
    rng.shuffle(b)
    if np.intersect1d(a, b).size:
        raise RuntimeError("Internal error: stratified halves overlap.")
    if a.size + b.size != theta.size:
        raise RuntimeError("Internal error: stratified halves do not cover all trials.")
    return a, b


def fit_half_pca(
    responses: np.ndarray,
    *,
    n_components: int,
    random_state: int,
    whiten: bool,
    session_key: str,
    half_label: str,
) -> PCAProjectionResult:
    x = np.asarray(responses, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("responses must be a 2D trial-by-neuron array.")
    max_components = min(int(x.shape[0]), int(x.shape[1]))
    if int(n_components) < 1 or int(n_components) > max_components:
        raise ValueError(f"n_components must be in [1, {max_components}].")
    pca = PCA(n_components=int(n_components), whiten=bool(whiten), svd_solver="randomized", random_state=int(random_state))
    z = pca.fit_transform(x).astype(np.float64, copy=False)
    return PCAProjectionResult(
        x_all=z,
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_, dtype=np.float64),
        singular_values=np.asarray(pca.singular_values_, dtype=np.float64),
        metadata={
            "session_key": str(session_key),
            "half_label": str(half_label),
            "pca_fit_scope": "half",
            "pca_input": "neural_responses_only",
            "pca_input_uses_orientation_labels": False,
            "pca_trial_averaging_before_fit": False,
            "pca_dim": int(n_components),
            "pca_whiten": bool(whiten),
            "pca_svd_solver": "randomized",
            "pca_random_state": int(random_state),
            "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            "pca_explained_variance_ratio": np.asarray(pca.explained_variance_ratio_, dtype=np.float64).tolist(),
        },
    )


def split_train_validation(n_total: int, *, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if int(n_total) < 2:
        raise ValueError("Need at least two observations for train/validation split.")
    frac = float(train_frac)
    if not (0.0 < frac < 1.0):
        raise ValueError("train_frac must be in (0, 1).")
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(int(n_total)).astype(np.int64)
    n_train = int(frac * int(n_total))
    n_train = min(max(n_train, 1), int(n_total) - 1)
    return idx[:n_train], idx[n_train:]


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
    output_npz: Path | None,
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
    train_meta = train_flow_skl_model(
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
            train_metadata=train_meta,
        )
        saved = save_flow_result_npz(
            output_npz,
            result=result,
            metric=METHOD_FLOW_LINEAR,
            theta_eval=theta_grid,
            velocity_family="condition_affine",
        )
    return np.asarray(fd["fisher"], dtype=np.float64), train_meta, saved


def compact_train_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    out = dict(meta)
    for key in ("train_losses", "val_losses", "val_monitor_losses"):
        arr = np.asarray(out.pop(key, []), dtype=np.float64)
        if arr.size:
            out[f"{key}_final"] = float(arr[-1])
            out[f"{key}_length"] = int(arr.size)
    return out


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def config_signature(config: dict[str, Any]) -> str:
    return json.dumps(json_ready(config), sort_keys=True, separators=(",", ":"))


def save_half_cache(path: Path, *, result: HalfCurveResult, signature: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        signature=np.asarray([signature]),
        theta_grid=np.asarray(result.theta_grid, dtype=np.float64),
        theta_midpoints=np.asarray(result.theta_midpoints, dtype=np.float64),
        classical_linear_fisher=np.asarray(result.curves[METHOD_CLASSICAL_LINEAR], dtype=np.float64),
        flow_linear_fisher=np.asarray(result.curves[METHOD_FLOW_LINEAR], dtype=np.float64),
        session_index=np.asarray([int(result.session_index)], dtype=np.int64),
        session_key=np.asarray([str(result.session_key)]),
        session_file=np.asarray([str(result.session_file)]),
        half_label=np.asarray([str(result.half_label)]),
        n_trials=np.asarray([int(result.n_trials)], dtype=np.int64),
        n_neurons=np.asarray([int(result.n_neurons)], dtype=np.int64),
        pca_metadata_json=np.asarray([json.dumps(json_ready(result.pca_metadata), sort_keys=True)]),
        train_metadata_json=np.asarray([json.dumps(json_ready(result.train_metadata), sort_keys=True)]),
        flow_npz_path=np.asarray(["" if result.flow_npz_path is None else str(result.flow_npz_path)]),
    )
    return path


def load_half_cache(path: Path, *, signature: str) -> HalfCurveResult | None:
    path = Path(path)
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=False)
    got = str(np.asarray(data["signature"]).reshape(-1)[0])
    if got != str(signature):
        return None
    flow_path_text = str(np.asarray(data["flow_npz_path"]).reshape(-1)[0])
    flow_path = None if flow_path_text == "" else Path(flow_path_text)
    return HalfCurveResult(
        session_index=int(np.asarray(data["session_index"]).reshape(-1)[0]),
        session_key=str(np.asarray(data["session_key"]).reshape(-1)[0]),
        session_file=str(np.asarray(data["session_file"]).reshape(-1)[0]),
        half_label=str(np.asarray(data["half_label"]).reshape(-1)[0]),
        n_trials=int(np.asarray(data["n_trials"]).reshape(-1)[0]),
        n_neurons=int(np.asarray(data["n_neurons"]).reshape(-1)[0]),
        theta_grid=np.asarray(data["theta_grid"], dtype=np.float64),
        theta_midpoints=np.asarray(data["theta_midpoints"], dtype=np.float64),
        curves={
            METHOD_CLASSICAL_LINEAR: np.asarray(data["classical_linear_fisher"], dtype=np.float64),
            METHOD_FLOW_LINEAR: np.asarray(data["flow_linear_fisher"], dtype=np.float64),
        },
        pca_metadata=json.loads(str(np.asarray(data["pca_metadata_json"]).reshape(-1)[0])),
        train_metadata=json.loads(str(np.asarray(data["train_metadata_json"]).reshape(-1)[0])),
        cache_path=path,
        flow_npz_path=flow_path,
    )


def estimate_half_curves(
    *,
    session_info: StringerSessionInfo,
    session_index: int,
    half_label: str,
    half_indices: np.ndarray,
    theta_grid: np.ndarray,
    period: float,
    pca_dim: int,
    pca_random_state: int,
    pca_whiten: bool,
    train_frac: float,
    seed: int,
    device: torch.device,
    flow_config: ContinuousFlowConfig,
    output_dir: Path,
    force: bool,
    save_flow_npz: bool,
    classical_ridge: float,
    classical_window_radius: float | None,
    classical_min_endpoint_samples: int,
) -> HalfCurveResult:
    session_key = Path(session_info.session_file).stem
    cache_path = Path(output_dir) / "half_curves" / f"{session_key}_{half_label}_curves.npz"
    signature = config_signature(
        {
            "session_file": str(session_info.session_file),
            "session_index": int(session_index),
            "half_label": str(half_label),
            "half_indices": np.asarray(half_indices, dtype=np.int64).tolist(),
            "theta_grid": np.asarray(theta_grid, dtype=np.float64).reshape(-1).tolist(),
            "period": float(period),
            "pca_dim": int(pca_dim),
            "pca_random_state": int(pca_random_state),
            "pca_whiten": bool(pca_whiten),
            "train_frac": float(train_frac),
            "seed": int(seed),
            "flow_config": json_ready(vars(flow_config)),
            "classical_ridge": float(classical_ridge),
            "classical_window_radius": classical_window_radius,
            "classical_min_endpoint_samples": int(classical_min_endpoint_samples),
        }
    )
    cached = None if force else load_half_cache(cache_path, signature=signature)
    if cached is not None:
        print(f"[stringer-identification] cache hit {session_key} half={half_label}", flush=True)
        return cached

    print(f"[stringer-identification] loading {session_key} half={half_label}", flush=True)
    session = load_stringer_session(session_info, orientation_period=float(period))
    idx = np.asarray(half_indices, dtype=np.int64)
    theta_half = np.asarray(session.grating_orientation, dtype=np.float64).reshape(-1)[idx]
    responses_half = np.asarray(session.neural_responses)[idx]
    pca = fit_half_pca(
        responses_half,
        n_components=int(pca_dim),
        random_state=int(pca_random_state),
        whiten=bool(pca_whiten),
        session_key=session_key,
        half_label=half_label,
    )
    classical = classical_linear_fisher_circular(
        theta_all=theta_half,
        x_all=pca.x_all,
        theta_grid=theta_grid,
        period=float(period),
        ridge=float(classical_ridge),
        window_radius=classical_window_radius,
        min_endpoint_samples=int(classical_min_endpoint_samples),
    )
    tr, va = split_train_validation(int(theta_half.shape[0]), train_frac=float(train_frac), seed=int(seed))
    bundle = make_shared_bundle(
        theta_all=theta_half,
        x_all=pca.x_all,
        train_idx=tr,
        validation_idx=va,
        meta={
            "dataset_family": "stringer_half_pca",
            "session_key": session_key,
            "session_file": str(session_info.session_file),
            "half_label": half_label,
            "theta_low": 0.0,
            "theta_high": float(period),
            "n_total": int(theta_half.shape[0]),
            "pca_metadata": pca.metadata,
        },
    )
    flow_npz = Path(output_dir) / "flow" / f"{session_key}_{half_label}_flow_linear_results.npz" if save_flow_npz else None
    print(f"[stringer-identification] training flow_linear {session_key} half={half_label}", flush=True)
    flow, train_meta, flow_path = train_flow_linear_curve(
        bundle=bundle,
        theta_grid=theta_grid,
        device=device,
        config=flow_config,
        seed=int(seed),
        output_npz=flow_npz,
    )
    result = HalfCurveResult(
        session_index=int(session_index),
        session_key=session_key,
        session_file=str(session_info.session_file),
        half_label=half_label,
        n_trials=int(theta_half.shape[0]),
        n_neurons=int(responses_half.shape[1]),
        theta_grid=np.asarray(theta_grid, dtype=np.float64),
        theta_midpoints=theta_midpoints(theta_grid),
        curves={METHOD_CLASSICAL_LINEAR: classical, METHOD_FLOW_LINEAR: flow},
        pca_metadata=pca.metadata,
        train_metadata=compact_train_metadata(train_meta),
        cache_path=cache_path,
        flow_npz_path=flow_path,
    )
    save_half_cache(cache_path, result=result, signature=signature)
    return result


def _zscore_log_curve(curve: np.ndarray, eps: float) -> np.ndarray:
    vals = np.log(np.maximum(np.asarray(curve, dtype=np.float64).reshape(-1), float(eps)))
    sd = float(np.nanstd(vals))
    if sd <= 1e-12:
        return np.zeros_like(vals)
    return (vals - float(np.nanmean(vals))) / sd


def curve_distance(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    theta_mid: np.ndarray,
    *,
    distance: str,
    eps: float = 1e-12,
) -> float:
    a = np.asarray(curve_a, dtype=np.float64).reshape(-1)
    b = np.asarray(curve_b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Curves must have matching shape.")
    if str(distance) == DISTANCE_PRIMARY:
        za = _zscore_log_curve(a, eps)
        zb = _zscore_log_curve(b, eps)
        corr = float(np.nanmean(za * zb))
        return float(1.0 - corr)
    if str(distance) == DISTANCE_AREA_L2:
        th = np.asarray(theta_mid, dtype=np.float64).reshape(-1)
        num = float(np.trapezoid((a - b) ** 2, x=th))
        den = 0.5 * (float(np.trapezoid(a**2, x=th)) + float(np.trapezoid(b**2, x=th)))
        return float(math.sqrt(max(num, 0.0) / max(den, eps)))
    if str(distance) == DISTANCE_RMSE:
        return float(np.sqrt(np.nanmean((a - b) ** 2)))
    raise ValueError(f"Unknown distance {distance!r}.")


def compute_identification(
    half_results: list[HalfCurveResult],
    *,
    theta_mid: np.ndarray,
    eps: float = 1e-12,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], list[dict[str, Any]], dict[str, Any]]:
    session_keys = sorted({h.session_key for h in half_results})
    n = len(session_keys)
    by_half = {
        HALF_A: {h.session_key: h for h in half_results if h.half_label == HALF_A},
        HALF_B: {h.session_key: h for h in half_results if h.half_label == HALF_B},
    }
    if any(set(by_half[label]) != set(session_keys) for label in (HALF_A, HALF_B)):
        raise ValueError("Need exactly one A and one B half result per session.")
    distances: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    pair_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for method in METHODS:
        distances[method] = {}
        summary[method] = {}
        for dist_name in DISTANCES:
            distances[method][dist_name] = {}
            direction = DIRECTION_A_TO_B
            q_label = HALF_A
            c_label = HALF_B
            mat = np.full((n, n), np.nan, dtype=np.float64)
            for qi, q_key in enumerate(session_keys):
                for ci, c_key in enumerate(session_keys):
                    mat[qi, ci] = curve_distance(
                        by_half[q_label][q_key].curves[method],
                        by_half[c_label][c_key].curves[method],
                        theta_mid,
                        distance=dist_name,
                        eps=float(eps),
                    )
            distances[method][dist_name][direction] = mat
            ranks = np.full(n, -1, dtype=np.int64)
            tie_counts = np.full(n, 0, dtype=np.int64)
            for qi, q_key in enumerate(session_keys):
                order = np.argsort(mat[qi], kind="mergesort")
                correct = session_keys.index(q_key)
                rank = int(np.flatnonzero(order == correct)[0]) + 1
                ranks[qi] = rank
                best = float(mat[qi, order[0]])
                tie_counts[qi] = int(np.sum(np.isclose(mat[qi], best, rtol=1e-12, atol=1e-12)))
                for candidate_rank, ci in enumerate(order, start=1):
                    pair_rows.append(
                        {
                            "method": method,
                            "distance": dist_name,
                            "direction": direction,
                            "query_half": q_label,
                            "candidate_half": c_label,
                            "query_session": q_key,
                            "candidate_session": session_keys[int(ci)],
                            "query_session_index": int(qi),
                            "candidate_session_index": int(ci),
                            "distance_value": float(mat[qi, int(ci)]),
                            "rank": int(candidate_rank),
                            "is_match": bool(int(ci) == correct),
                            "best_tie_count_for_query": int(tie_counts[qi]),
                        }
                    )
            summary[method][f"{dist_name}_{direction}"] = {
                "top1_accuracy": float(np.mean(ranks == 1)),
                "top2_accuracy": float(np.mean(ranks <= min(2, n))),
                "top3_accuracy": float(np.mean(ranks <= min(3, n))),
                "mean_reciprocal_rank": float(np.mean(1.0 / ranks.astype(np.float64))),
                "ranks": ranks.tolist(),
                "tie_counts": tie_counts.tolist(),
            }
    return distances, pair_rows, summary


def curve_rows_from_halves(half_results: list[HalfCurveResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for half in half_results:
        mids = half.theta_midpoints.reshape(-1)
        grid = half.theta_grid.reshape(-1)
        for method in METHODS:
            curve = np.asarray(half.curves[method], dtype=np.float64).reshape(-1)
            for i, val in enumerate(curve):
                rows.append(
                    {
                        "session_index": int(half.session_index),
                        "session_key": half.session_key,
                        "session_file": half.session_file,
                        "half_label": half.half_label,
                        "method": method,
                        "theta_midpoint": float(mids[i]),
                        "theta_left": float(grid[i]),
                        "theta_right": float(grid[i + 1]),
                        "fisher": float(val),
                        "n_trials_half": int(half.n_trials),
                        "n_neurons": int(half.n_neurons),
                        "cache_path": str(half.cache_path),
                        "flow_npz_path": "" if half.flow_npz_path is None else str(half.flow_npz_path),
                    }
                )
    return rows


def run_session_identification(
    *,
    sessions: list[StringerSessionInfo],
    theta_grid: np.ndarray,
    period: float,
    pca_dim: int,
    pca_random_state: int,
    pca_whiten: bool,
    train_frac: float,
    seed: int,
    device: torch.device,
    flow_config: ContinuousFlowConfig,
    output_dir: Path,
    force: bool = False,
    save_flow_npz: bool = True,
    classical_ridge: float = 1e-6,
    classical_window_radius: float | None = None,
    classical_min_endpoint_samples: int = 8,
) -> IdentificationResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    half_results: list[HalfCurveResult] = []
    for session_index, info in enumerate(sessions):
        session = load_stringer_session(info, orientation_period=float(period))
        a_idx, b_idx = stratified_half_split(
            session.grating_orientation,
            n_bins=int(np.asarray(theta_grid).reshape(-1).shape[0] - 1),
            period=float(period),
            seed=int(seed) + int(session_index),
        )
        for half_offset, (half_label, idx) in enumerate(((HALF_A, a_idx), (HALF_B, b_idx))):
            half_results.append(
                estimate_half_curves(
                    session_info=info,
                    session_index=session_index,
                    half_label=half_label,
                    half_indices=idx,
                    theta_grid=theta_grid,
                    period=float(period),
                    pca_dim=int(pca_dim),
                    pca_random_state=int(pca_random_state) + 100 * int(session_index) + int(half_offset),
                    pca_whiten=bool(pca_whiten),
                    train_frac=float(train_frac),
                    seed=int(seed) + 1000 * int(session_index) + int(half_offset),
                    device=device,
                    flow_config=flow_config,
                    output_dir=output_dir,
                    force=bool(force),
                    save_flow_npz=bool(save_flow_npz),
                    classical_ridge=float(classical_ridge),
                    classical_window_radius=classical_window_radius,
                    classical_min_endpoint_samples=int(classical_min_endpoint_samples),
                )
            )
    session_keys = [Path(info.session_file).stem for info in sessions]
    mids = theta_midpoints(theta_grid)
    distances, pair_rows, identification_summary = compute_identification(half_results, theta_mid=mids)
    curve_rows = curve_rows_from_halves(half_results)
    summary = {
        "session_keys": session_keys,
        "n_sessions": int(len(session_keys)),
        "n_half_results": int(len(half_results)),
        "methods": list(METHODS),
        "distances": list(DISTANCES),
        "primary_distance": DISTANCE_PRIMARY,
        "identification_direction": DIRECTION_A_TO_B,
        "query_half": HALF_A,
        "reference_half": HALF_B,
        "orientation_period": float(period),
        "theta_grid_size": int(np.asarray(theta_grid).reshape(-1).shape[0]),
        "pca_fit_scope": "half",
        "pca_label_blind": True,
        "pca_trial_averaging_before_fit": False,
        "identification": identification_summary,
        "half_metadata": [
            {
                "session_index": h.session_index,
                "session_key": h.session_key,
                "half_label": h.half_label,
                "n_trials": h.n_trials,
                "n_neurons": h.n_neurons,
                "pca_metadata": h.pca_metadata,
                "train_metadata": h.train_metadata,
                "cache_path": str(h.cache_path),
                "flow_npz_path": None if h.flow_npz_path is None else str(h.flow_npz_path),
            }
            for h in half_results
        ],
    }
    return IdentificationResult(
        session_keys=session_keys,
        theta_grid=np.asarray(theta_grid, dtype=np.float64),
        theta_midpoints=mids,
        half_results=half_results,
        distances=distances,
        pair_rows=pair_rows,
        curve_rows=curve_rows,
        summary=summary,
    )


def write_results_npz(path: Path, result: IdentificationResult) -> Path:
    fields: dict[str, Any] = {
        "session_keys": np.asarray(result.session_keys),
        "theta_grid": np.asarray(result.theta_grid, dtype=np.float64),
        "theta_midpoints": np.asarray(result.theta_midpoints, dtype=np.float64),
    }
    for method in METHODS:
        for dist_name in DISTANCES:
            for direction, mat in result.distances[method][dist_name].items():
                fields[f"{method}_{dist_name}_{direction}"] = np.asarray(mat, dtype=np.float64)
    for half in result.half_results:
        prefix = f"{half.session_key}_{half.half_label}"
        fields[f"{prefix}_classical_linear_fisher"] = np.asarray(half.curves[METHOD_CLASSICAL_LINEAR], dtype=np.float64)
        fields[f"{prefix}_flow_linear_fisher"] = np.asarray(half.curves[METHOD_FLOW_LINEAR], dtype=np.float64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **fields)
    return path


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: tuple[str, ...]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    return path


def write_curves_csv(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    return write_csv(
        path,
        rows,
        (
            "session_index",
            "session_key",
            "session_file",
            "half_label",
            "method",
            "theta_midpoint",
            "theta_left",
            "theta_right",
            "fisher",
            "n_trials_half",
            "n_neurons",
            "cache_path",
            "flow_npz_path",
        ),
    )


def write_pairs_csv(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    return write_csv(
        path,
        rows,
        (
            "method",
            "distance",
            "direction",
            "query_half",
            "candidate_half",
            "query_session",
            "candidate_session",
            "query_session_index",
            "candidate_session_index",
            "distance_value",
            "rank",
            "is_match",
            "best_tie_count_for_query",
        ),
    )


def write_summary_json(path: Path, result: IdentificationResult, *, extra: dict[str, Any] | None = None) -> Path:
    summary = dict(result.summary)
    if extra:
        summary.update(json_ready(extra))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n")
    return path


def plot_primary_heatmaps(path_svg: Path, path_png: Path, result: IdentificationResult) -> tuple[Path, Path]:
    labels = [str(i) for i in range(len(result.session_keys))]
    fig, axes = plt.subplots(len(METHODS), 1, figsize=(5.6, 4.8 * len(METHODS)), layout="constrained")
    axes_arr = np.asarray(axes).reshape(len(METHODS))
    for mi, method in enumerate(METHODS):
        ax = axes_arr[mi]
        mat = result.distances[method][DISTANCE_PRIMARY][DIRECTION_A_TO_B]
        im = ax.imshow(mat, cmap="viridis")
        ax.set_title(f"{method} {DIRECTION_A_TO_B}")
        ax.set_xlabel("candidate B session")
        ax.set_ylabel("query A session")
        ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(labels)), labels=labels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path_svg = Path(path_svg)
    path_png = Path(path_png)
    path_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_svg)
    fig.savefig(path_png, dpi=180)
    plt.close(fig)
    return path_svg, path_png


def plot_ranks(path_svg: Path, path_png: Path, result: IdentificationResult) -> tuple[Path, Path]:
    x = np.arange(len(result.session_keys), dtype=np.float64)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10.0, 5.2), layout="constrained")
    offsets = {
        METHOD_CLASSICAL_LINEAR: -0.5 * width,
        METHOD_FLOW_LINEAR: 0.5 * width,
    }
    for method, offset in offsets.items():
        ranks = result.summary["identification"][method][f"{DISTANCE_PRIMARY}_{DIRECTION_A_TO_B}"]["ranks"]
        ax.bar(x + offset, ranks, width=width, label=method)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("session index")
    ax.set_ylabel("correct-pair rank")
    ax.set_title("Stringer half-session identification ranks")
    ax.set_xticks(x, [str(i) for i in range(len(result.session_keys))])
    ax.set_ylim(0.5, len(result.session_keys) + 0.5)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    path_svg = Path(path_svg)
    path_png = Path(path_png)
    path_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_svg)
    fig.savefig(path_png, dpi=180)
    plt.close(fig)
    return path_svg, path_png


def plot_all_distance_summary(path_svg: Path, path_png: Path, result: IdentificationResult) -> tuple[Path, Path]:
    labels = [str(i) for i in range(len(result.session_keys))]
    n_rows = len(METHODS) * len(DISTANCES)
    fig_height = max(12.0, 2.7 * n_rows)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(14.0, fig_height),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.0, 0.8, 0.8]},
    )
    axes_arr = np.asarray(axes).reshape(n_rows, 3)
    x = np.arange(len(result.session_keys), dtype=np.float64)
    row = 0
    for method in METHODS:
        for distance_name in DISTANCES:
            row_axes = axes_arr[row]
            mats = result.distances[method][distance_name]
            mat = mats[DIRECTION_A_TO_B]
            ax = row_axes[0]
            im = ax.imshow(mat, cmap="viridis")
            ax.set_title(f"{method}\n{distance_name} {DIRECTION_A_TO_B}", fontsize=10)
            ax.set_xlabel("candidate B")
            ax.set_ylabel("query A")
            ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(labels)), labels=labels)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            rank_ax = row_axes[1]
            metric_ax = row_axes[2]
            width = 0.6
            ranks_a = np.asarray(
                result.summary["identification"][method][f"{distance_name}_{DIRECTION_A_TO_B}"]["ranks"],
                dtype=np.float64,
            )
            rank_ax.bar(x, ranks_a, width=width, label=DIRECTION_A_TO_B)
            rank_ax.axhline(1.0, color="black", linewidth=0.9, linestyle="--")
            rank_ax.set_title("correct-pair rank", fontsize=10)
            rank_ax.set_xlabel("session")
            rank_ax.set_ylabel("rank")
            rank_ax.set_xticks(x, labels)
            rank_ax.set_ylim(0.5, len(result.session_keys) + 0.5)
            rank_ax.grid(True, axis="y", alpha=0.25)

            summary_a = result.summary["identification"][method][f"{distance_name}_{DIRECTION_A_TO_B}"]
            metric_names = ("top1_accuracy", "top2_accuracy", "top3_accuracy", "mean_reciprocal_rank")
            metric_labels = ("top1", "top2", "top3", "MRR")
            vals_a = [float(summary_a[name]) for name in metric_names]
            mx = np.arange(len(metric_names), dtype=np.float64)
            metric_ax.bar(mx, vals_a, width=width)
            metric_ax.set_title("identification scores", fontsize=10)
            metric_ax.set_ylim(0.0, 1.05)
            metric_ax.set_xticks(mx, metric_labels, rotation=25, ha="right")
            metric_ax.grid(True, axis="y", alpha=0.25)
            row += 1

    fig.suptitle("Stringer session identification across Fisher-curve distances (A queries, B reference)", fontsize=16)
    path_svg = Path(path_svg)
    path_png = Path(path_png)
    path_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_svg)
    fig.savefig(path_png, dpi=180)
    plt.close(fig)
    return path_svg, path_png
