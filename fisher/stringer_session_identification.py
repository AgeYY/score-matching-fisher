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
SUBSAMPLE_RESULTS_NPZ_NAME = "stringer_session_identification_a_subsample_convergence_results.npz"
SUBSAMPLE_CURVES_CSV_NAME = "stringer_session_identification_a_subsample_convergence_curves.csv"
SUBSAMPLE_PAIRS_CSV_NAME = "stringer_session_identification_a_subsample_convergence_pairs.csv"
SUBSAMPLE_SUMMARY_JSON_NAME = "stringer_session_identification_a_subsample_convergence_summary.json"
SUBSAMPLE_SUMMARY_SVG_NAME = "stringer_session_identification_a_subsample_convergence_summary.svg"
SUBSAMPLE_SUMMARY_PNG_NAME = "stringer_session_identification_a_subsample_convergence_summary.png"
SUBSAMPLE_TOPK_ACCURACY_SVG_NAME = "stringer_session_identification_a_subsample_topk_accuracy.svg"
SUBSAMPLE_TOPK_ACCURACY_PNG_NAME = "stringer_session_identification_a_subsample_topk_accuracy.png"
SUBSAMPLE_LOGCORR_EXAMPLE_SVG_NAME = "stringer_session_identification_a_subsample_logcorr_example.svg"
SUBSAMPLE_LOGCORR_EXAMPLE_PNG_NAME = "stringer_session_identification_a_subsample_logcorr_example.png"

METHODS = (METHOD_CLASSICAL_LINEAR, METHOD_FLOW_LINEAR)
DISTANCE_PRIMARY = "log_correlation"
DISTANCE_AREA_L2 = "area_normalized_l2"
DISTANCE_RMSE = "raw_rmse"
DISTANCES = (DISTANCE_PRIMARY, DISTANCE_AREA_L2, DISTANCE_RMSE)
HALF_A = "A"
HALF_B = "B"
DIRECTION_A_TO_B = "A_to_B"
FLOW_ORIENTATION_ENCODING_SCALAR = "scalar"
FLOW_ORIENTATION_ENCODING_PERIODIC_SINCOS = "periodic-sincos"
FLOW_ORIENTATION_ENCODINGS = (FLOW_ORIENTATION_ENCODING_PERIODIC_SINCOS, FLOW_ORIENTATION_ENCODING_SCALAR)
ASUBSAMPLE_TASK_ENDPOINT_FULL_A = "endpoint_full_a"
ASUBSAMPLE_TASK_REFERENCE_B = "reference_b"
ASUBSAMPLE_TASK_SUBSET_A = "subset_a"
ASUBSAMPLE_TASK_KINDS = (
    ASUBSAMPLE_TASK_ENDPOINT_FULL_A,
    ASUBSAMPLE_TASK_REFERENCE_B,
    ASUBSAMPLE_TASK_SUBSET_A,
)


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


@dataclass(frozen=True)
class SubsampleConvergenceResult:
    session_keys: list[str]
    theta_grid: np.ndarray
    theta_midpoints: np.ndarray
    n_values: list[int]
    repeats: int
    sampling: str
    endpoint_result: IdentificationResult
    subset_matrices: dict[str, dict[str, np.ndarray]]
    pair_rows: list[dict[str, Any]]
    curve_rows: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass(frozen=True)
class ASubsampleCurveTask:
    kind: str
    session_index: int
    session_key: str
    subset_n: int | None = None
    repeat: int | None = None
    n_index: int | None = None

    @property
    def label(self) -> str:
        if self.kind == ASUBSAMPLE_TASK_ENDPOINT_FULL_A:
            return f"endpoint_full_a_session{int(self.session_index):03d}"
        if self.kind == ASUBSAMPLE_TASK_REFERENCE_B:
            return f"reference_b_session{int(self.session_index):03d}"
        if self.kind == ASUBSAMPLE_TASK_SUBSET_A:
            if self.subset_n is None or self.repeat is None:
                raise ValueError("subset_a tasks require subset_n and repeat.")
            return f"subset_a_n{int(self.subset_n):06d}_repeat{int(self.repeat):03d}_session{int(self.session_index):03d}"
        raise ValueError(f"Unknown A-subsample task kind {self.kind!r}.")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "session_index": int(self.session_index),
            "session_key": str(self.session_key),
            "subset_n": None if self.subset_n is None else int(self.subset_n),
            "repeat": None if self.repeat is None else int(self.repeat),
            "n_index": None if self.n_index is None else int(self.n_index),
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "ASubsampleCurveTask":
        return cls(
            kind=str(payload["kind"]),
            session_index=int(payload["session_index"]),
            session_key=str(payload["session_key"]),
            subset_n=None if payload.get("subset_n") is None else int(payload["subset_n"]),
            repeat=None if payload.get("repeat") is None else int(payload["repeat"]),
            n_index=None if payload.get("n_index") is None else int(payload["n_index"]),
        )


@dataclass(frozen=True)
class ResolvedASubsampleCurveTask:
    task: ASubsampleCurveTask
    session_info: StringerSessionInfo
    half_label: str
    half_indices: np.ndarray
    output_dir: Path
    pca_random_state: int
    seed: int
    cache_path: Path
    signature: str


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


def parse_positive_int_list(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
    else:
        parts = [str(v).strip() for v in value]
    if not parts:
        raise ValueError("integer list must contain at least one value.")
    out = sorted({int(p) for p in parts})
    if any(v < 1 for v in out):
        raise ValueError("integer list values must be positive.")
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


def orientation_bin_ids(theta_all: np.ndarray, *, n_bins: int, period: float) -> np.ndarray:
    theta = np.mod(np.asarray(theta_all, dtype=np.float64).reshape(-1), float(period))
    if int(n_bins) < 1:
        raise ValueError("n_bins must be >= 1.")
    bin_id = np.floor(theta / float(period) * int(n_bins)).astype(np.int64)
    return np.clip(bin_id, 0, int(n_bins) - 1)


def stratified_bootstrap_indices(
    theta_all: np.ndarray,
    *,
    n_samples: int,
    n_bins: int,
    period: float,
    seed: int,
    replace: bool = True,
) -> np.ndarray:
    theta = np.asarray(theta_all, dtype=np.float64).reshape(-1)
    n = int(n_samples)
    if n < 1:
        raise ValueError("n_samples must be >= 1.")
    if theta.shape[0] < 1:
        raise ValueError("theta_all must contain at least one observation.")
    bin_id = orientation_bin_ids(theta, n_bins=int(n_bins), period=float(period))
    counts = np.bincount(bin_id, minlength=int(n_bins)).astype(np.float64)
    expected = n * counts / float(theta.shape[0])
    per_bin = np.floor(expected).astype(np.int64)
    remainder = n - int(np.sum(per_bin))
    if remainder > 0:
        order = np.argsort(-(expected - per_bin), kind="mergesort")
        for b in order[:remainder]:
            if counts[int(b)] > 0:
                per_bin[int(b)] += 1
    rng = np.random.default_rng(int(seed))
    pieces: list[np.ndarray] = []
    for b, take in enumerate(per_bin):
        if int(take) <= 0:
            continue
        candidates = np.flatnonzero(bin_id == int(b))
        if candidates.size == 0:
            continue
        if not bool(replace) and int(take) > int(candidates.size):
            raise ValueError(
                f"Cannot sample {int(take)} observations without replacement from orientation bin {int(b)} "
                f"with only {int(candidates.size)} candidate(s)."
            )
        pieces.append(rng.choice(candidates, size=int(take), replace=bool(replace)).astype(np.int64))
    if not pieces:
        raise RuntimeError("Internal error: stratified bootstrap selected no samples.")
    out = np.concatenate(pieces).astype(np.int64)
    if out.size != n:
        raise RuntimeError("Internal error: stratified bootstrap sample count mismatch.")
    rng.shuffle(out)
    return out


def bootstrap_indices(
    theta_all: np.ndarray,
    *,
    n_samples: int,
    n_bins: int,
    period: float,
    seed: int,
    sampling: str,
    replace: bool = True,
) -> np.ndarray:
    theta = np.asarray(theta_all).reshape(-1)
    if not bool(replace) and int(n_samples) > int(theta.shape[0]):
        raise ValueError(
            f"Cannot sample {int(n_samples)} observations without replacement from {int(theta.shape[0])} candidate(s)."
        )
    if str(sampling) == "stratified":
        return stratified_bootstrap_indices(
            theta_all,
            n_samples=int(n_samples),
            n_bins=int(n_bins),
            period=float(period),
            seed=int(seed),
            replace=bool(replace),
        )
    if str(sampling) == "uniform":
        if theta.shape[0] < 1:
            raise ValueError("theta_all must contain at least one observation.")
        rng = np.random.default_rng(int(seed))
        if bool(replace):
            return rng.integers(0, int(theta.shape[0]), size=int(n_samples), endpoint=False, dtype=np.int64)
        return rng.choice(int(theta.shape[0]), size=int(n_samples), replace=False).astype(np.int64)
    raise ValueError(f"Unknown sampling mode {sampling!r}.")


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


def normalize_flow_orientation_encoding(encoding: str) -> str:
    enc = str(encoding).strip().lower()
    if enc not in FLOW_ORIENTATION_ENCODINGS:
        raise ValueError(f"flow_orientation_encoding must be one of {FLOW_ORIENTATION_ENCODINGS}, got {encoding!r}.")
    return enc


def encode_flow_orientation(theta: np.ndarray, *, period: float, encoding: str) -> np.ndarray:
    enc = normalize_flow_orientation_encoding(encoding)
    theta_col = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
    if enc == FLOW_ORIENTATION_ENCODING_SCALAR:
        return theta_col
    pp = float(period)
    if pp <= 0.0 or not math.isfinite(pp):
        raise ValueError("period must be finite and positive for periodic orientation encoding.")
    phase = 2.0 * math.pi * np.mod(theta_col[:, 0], pp) / pp
    return np.stack([np.cos(phase), np.sin(phase)], axis=1).astype(np.float64)


def estimate_affine_mixed_symmetric_kl_fisher_for_conditions(
    *,
    model: torch.nn.Module,
    theta_all: np.ndarray,
    condition_all: np.ndarray,
    device: torch.device,
    ridge: float = 1e-6,
    ode_steps: int = 64,
) -> dict[str, Any]:
    theta = np.asarray(theta_all, dtype=np.float64).reshape(-1, 1)
    cond = np.asarray(condition_all, dtype=np.float64)
    if cond.ndim == 1:
        cond = cond.reshape(-1, 1)
    if int(theta.shape[0]) != int(cond.shape[0]):
        raise ValueError("theta_all and condition_all must have the same number of rows.")
    if int(theta.shape[0]) < 2:
        raise ValueError("At least two theta points are required.")
    if not hasattr(model, "endpoint_mean") or not hasattr(model, "A"):
        raise ValueError("Affine mixed-covariance Fisher requires model.endpoint_mean() and model.A().")
    steps = int(ode_steps)
    if steps < 1:
        raise ValueError("ode_steps must be >= 1.")
    rr = float(ridge)
    if rr < 0.0 or not math.isfinite(rr):
        raise ValueError("ridge must be finite and nonnegative.")

    order = np.argsort(theta[:, 0], kind="mergesort")
    theta_s = theta[order]
    cond_s = cond[order]
    dtheta = np.diff(theta_s[:, 0])
    if np.any(dtheta <= 0.0):
        raise ValueError("theta_all must contain strictly increasing unique scalar values after sorting.")

    model.to(device)
    model.eval()
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        dtype = torch.float32
    if not dtype.is_floating_point:
        dtype = torch.float32
    x_dim = int(getattr(model, "x_dim"))
    n_theta = int(theta_s.shape[0])
    eye_np = np.eye(x_dim, dtype=np.float64)
    fish = np.zeros(n_theta - 1, dtype=np.float64)
    adjacent_skl = np.zeros(n_theta - 1, dtype=np.float64)
    skl_matrix = np.zeros((n_theta, n_theta), dtype=np.float64)
    mid_covs = np.zeros((n_theta - 1, x_dim, x_dim), dtype=np.float64)
    deltas = np.zeros((n_theta - 1, x_dim), dtype=np.float64)

    def _a_for(condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        try:
            return model.A(condition, t)
        except TypeError:
            return model.A(t)

    for i in range(n_theta - 1):
        cond_l = torch.from_numpy(cond_s[i : i + 1].astype(np.float32)).to(device=device, dtype=dtype)
        cond_r = torch.from_numpy(cond_s[i + 1 : i + 2].astype(np.float32)).to(device=device, dtype=dtype)
        mu_l = model.endpoint_mean(cond_l).detach().cpu().numpy().reshape(-1).astype(np.float64)
        mu_r = model.endpoint_mean(cond_r).detach().cpu().numpy().reshape(-1).astype(np.float64)
        delta = mu_r - mu_l
        sigma = eye_np.copy()
        dt = 1.0 / float(steps)
        for step in range(steps):
            t_val = (float(step) + 0.5) * dt
            tt = torch.full((1, 1), t_val, dtype=dtype, device=device)
            a_l = _a_for(cond_l, tt)
            a_r = _a_for(cond_r, tt)
            a_bar = (0.5 * (a_l + a_r)).detach().cpu().numpy().reshape(x_dim, x_dim).astype(np.float64)
            sigma = sigma + dt * (a_bar @ sigma + sigma @ a_bar.T)
            sigma = 0.5 * (sigma + sigma.T)
        cov = sigma + rr * eye_np
        skl = max(0.0, float(delta @ np.linalg.solve(cov, delta)))
        adjacent_skl[i] = skl
        skl_matrix[i, i + 1] = skl
        skl_matrix[i + 1, i] = skl
        fish[i] = skl / float(dtheta[i] ** 2)
        mid_covs[i] = sigma
        deltas[i] = delta

    return {
        "theta_midpoints": (0.5 * (theta_s[:-1, 0] + theta_s[1:, 0])).reshape(-1, 1).astype(np.float64),
        "theta_left": theta_s[:-1].astype(np.float64),
        "theta_right": theta_s[1:].astype(np.float64),
        "condition_left": cond_s[:-1].astype(np.float64),
        "condition_right": cond_s[1:].astype(np.float64),
        "dtheta": dtheta.astype(np.float64),
        "delta_mu": deltas,
        "mixed_covariance": mid_covs,
        "adjacent_symmetric_kl": adjacent_skl,
        "symmetric_kl_matrix": skl_matrix,
        "canonical_metric_matrix": skl_matrix.copy(),
        "canonical_metric_name": "mixed_affine_symmetric_kl",
        "fisher": fish,
    }


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
    period: float,
    flow_orientation_encoding: str,
    device: torch.device,
    config: ContinuousFlowConfig,
    seed: int,
    output_npz: Path | None,
) -> tuple[np.ndarray, dict[str, Any], Path | None]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    encoding = normalize_flow_orientation_encoding(flow_orientation_encoding)
    condition_train = encode_flow_orientation(bundle.theta_train, period=float(period), encoding=encoding)
    condition_val = encode_flow_orientation(bundle.theta_validation, period=float(period), encoding=encoding)
    condition_grid = encode_flow_orientation(theta_grid, period=float(period), encoding=encoding)
    model = build_flow_skl_model(
        velocity_family="condition_affine",
        theta_dim=int(condition_train.shape[1]),
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
        theta_train=condition_train,
        x_train=np.asarray(bundle.x_train, dtype=np.float64),
        theta_val=condition_val,
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
    train_meta = dict(train_meta)
    train_meta.update(
        {
            "flow_orientation_encoding": encoding,
            "orientation_period": float(period),
            "flow_condition_dim": int(condition_train.shape[1]),
        }
    )
    fd = estimate_affine_mixed_symmetric_kl_fisher_for_conditions(
        model=model,
        theta_all=theta_grid,
        condition_all=condition_grid,
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


def half_curve_cache_path(*, output_dir: Path, session_info: StringerSessionInfo, half_label: str) -> Path:
    session_key = Path(session_info.session_file).stem
    return Path(output_dir) / "half_curves" / f"{session_key}_{half_label}_curves.npz"


def half_curve_signature(
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
    flow_config: ContinuousFlowConfig,
    flow_orientation_encoding: str,
    classical_ridge: float,
    classical_window_radius: float | None,
    classical_min_endpoint_samples: int,
) -> str:
    return config_signature(
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
            "flow_orientation_encoding": normalize_flow_orientation_encoding(flow_orientation_encoding),
            "classical_ridge": float(classical_ridge),
            "classical_window_radius": classical_window_radius,
            "classical_min_endpoint_samples": int(classical_min_endpoint_samples),
        }
    )


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
    flow_orientation_encoding: str,
    output_dir: Path,
    force: bool,
    save_flow_npz: bool,
    classical_ridge: float,
    classical_window_radius: float | None,
    classical_min_endpoint_samples: int,
) -> HalfCurveResult:
    session_key = Path(session_info.session_file).stem
    cache_path = half_curve_cache_path(output_dir=output_dir, session_info=session_info, half_label=half_label)
    signature = half_curve_signature(
        session_info=session_info,
        session_index=int(session_index),
        half_label=str(half_label),
        half_indices=half_indices,
        theta_grid=theta_grid,
        period=float(period),
        pca_dim=int(pca_dim),
        pca_random_state=int(pca_random_state),
        pca_whiten=bool(pca_whiten),
        train_frac=float(train_frac),
        seed=int(seed),
        flow_config=flow_config,
        flow_orientation_encoding=flow_orientation_encoding,
        classical_ridge=float(classical_ridge),
        classical_window_radius=classical_window_radius,
        classical_min_endpoint_samples=int(classical_min_endpoint_samples),
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
        period=float(period),
        flow_orientation_encoding=flow_orientation_encoding,
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


def compute_query_reference_identification(
    *,
    query_results: list[HalfCurveResult],
    reference_results: list[HalfCurveResult],
    theta_mid: np.ndarray,
    session_keys: list[str],
    query_half: str,
    candidate_half: str,
    direction: str = DIRECTION_A_TO_B,
    eps: float = 1e-12,
    row_extra: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], list[dict[str, Any]], dict[str, Any]]:
    n = len(session_keys)
    by_query = {h.session_key: h for h in query_results}
    by_reference = {h.session_key: h for h in reference_results}
    if set(by_query) != set(session_keys) or set(by_reference) != set(session_keys):
        raise ValueError("Need exactly one query and one reference result per session.")
    distances: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    pair_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for method in METHODS:
        distances[method] = {}
        summary[method] = {}
        for dist_name in DISTANCES:
            distances[method][dist_name] = {}
            mat = np.full((n, n), np.nan, dtype=np.float64)
            for qi, q_key in enumerate(session_keys):
                for ci, c_key in enumerate(session_keys):
                    mat[qi, ci] = curve_distance(
                        by_query[q_key].curves[method],
                        by_reference[c_key].curves[method],
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
                    row = {
                        "method": method,
                        "distance": dist_name,
                        "direction": direction,
                        "query_half": query_half,
                        "candidate_half": candidate_half,
                        "query_session": q_key,
                        "candidate_session": session_keys[int(ci)],
                        "query_session_index": int(qi),
                        "candidate_session_index": int(ci),
                        "distance_value": float(mat[qi, int(ci)]),
                        "rank": int(candidate_rank),
                        "is_match": bool(int(ci) == correct),
                        "best_tie_count_for_query": int(tie_counts[qi]),
                    }
                    if row_extra:
                        row.update(row_extra)
                    pair_rows.append(row)
            summary[method][f"{dist_name}_{direction}"] = {
                "top1_accuracy": float(np.mean(ranks == 1)),
                "top2_accuracy": float(np.mean(ranks <= min(2, n))),
                "top3_accuracy": float(np.mean(ranks <= min(3, n))),
                "mean_reciprocal_rank": float(np.mean(1.0 / ranks.astype(np.float64))),
                "ranks": ranks.tolist(),
                "tie_counts": tie_counts.tolist(),
            }
    return distances, pair_rows, summary


def compute_identification(
    half_results: list[HalfCurveResult],
    *,
    theta_mid: np.ndarray,
    eps: float = 1e-12,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], list[dict[str, Any]], dict[str, Any]]:
    session_keys = sorted({h.session_key for h in half_results})
    by_half = {
        HALF_A: {h.session_key: h for h in half_results if h.half_label == HALF_A},
        HALF_B: {h.session_key: h for h in half_results if h.half_label == HALF_B},
    }
    if any(set(by_half[label]) != set(session_keys) for label in (HALF_A, HALF_B)):
        raise ValueError("Need exactly one A and one B half result per session.")
    return compute_query_reference_identification(
        query_results=list(by_half[HALF_A].values()),
        reference_results=list(by_half[HALF_B].values()),
        theta_mid=theta_mid,
        session_keys=session_keys,
        query_half=HALF_A,
        candidate_half=HALF_B,
        direction=DIRECTION_A_TO_B,
        eps=float(eps),
    )


def curve_rows_from_halves(half_results: list[HalfCurveResult], *, row_extra: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for half in half_results:
        mids = half.theta_midpoints.reshape(-1)
        grid = half.theta_grid.reshape(-1)
        for method in METHODS:
            curve = np.asarray(half.curves[method], dtype=np.float64).reshape(-1)
            for i, val in enumerate(curve):
                row = {
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
                if row_extra:
                    row.update(row_extra)
                rows.append(row)
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
    flow_orientation_encoding: str = FLOW_ORIENTATION_ENCODING_PERIODIC_SINCOS,
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
                    flow_orientation_encoding=flow_orientation_encoding,
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
        "flow_orientation_encoding": normalize_flow_orientation_encoding(flow_orientation_encoding),
        "flow_condition_dim": int(
            encode_flow_orientation(
                np.asarray(theta_grid, dtype=np.float64).reshape(-1),
                period=float(period),
                encoding=flow_orientation_encoding,
            ).shape[1]
        ),
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


def _metric_from_summary(summary: dict[str, Any], method: str, distance_name: str, metric_name: str) -> float:
    return float(summary[method][f"{distance_name}_{DIRECTION_A_TO_B}"][metric_name])


def plan_a_subsample_curve_tasks(
    *,
    sessions: list[StringerSessionInfo],
    n_values: list[int],
    repeats: int,
) -> list[ASubsampleCurveTask]:
    n_list = parse_positive_int_list(n_values)
    if int(repeats) < 1:
        raise ValueError("repeats must be >= 1.")
    session_keys = [Path(info.session_file).stem for info in sessions]
    tasks: list[ASubsampleCurveTask] = []
    for session_index, session_key in enumerate(session_keys):
        tasks.append(
            ASubsampleCurveTask(
                kind=ASUBSAMPLE_TASK_ENDPOINT_FULL_A,
                session_index=int(session_index),
                session_key=str(session_key),
            )
        )
    for session_index, session_key in enumerate(session_keys):
        tasks.append(
            ASubsampleCurveTask(
                kind=ASUBSAMPLE_TASK_REFERENCE_B,
                session_index=int(session_index),
                session_key=str(session_key),
            )
        )
    for n_index, n_subset in enumerate(n_list):
        for repeat in range(int(repeats)):
            for session_index, session_key in enumerate(session_keys):
                tasks.append(
                    ASubsampleCurveTask(
                        kind=ASUBSAMPLE_TASK_SUBSET_A,
                        session_index=int(session_index),
                        session_key=str(session_key),
                        subset_n=int(n_subset),
                        repeat=int(repeat),
                        n_index=int(n_index),
                    )
                )
    return tasks


def resolve_a_subsample_curve_task(
    task: ASubsampleCurveTask,
    *,
    sessions: list[StringerSessionInfo],
    theta_grid: np.ndarray,
    period: float,
    pca_dim: int,
    pca_random_state: int,
    pca_whiten: bool,
    train_frac: float,
    seed: int,
    flow_config: ContinuousFlowConfig,
    output_dir: Path,
    n_values: list[int],
    sampling: str,
    flow_orientation_encoding: str = FLOW_ORIENTATION_ENCODING_PERIODIC_SINCOS,
    replace: bool = True,
    classical_ridge: float = 1e-6,
    classical_window_radius: float | None = None,
    classical_min_endpoint_samples: int = 8,
) -> ResolvedASubsampleCurveTask:
    if str(task.kind) not in ASUBSAMPLE_TASK_KINDS:
        raise ValueError(f"Unknown A-subsample task kind {task.kind!r}.")
    n_list = parse_positive_int_list(n_values)
    if str(sampling) not in {"stratified", "uniform"}:
        raise ValueError("sampling must be 'stratified' or 'uniform'.")
    session_index = int(task.session_index)
    if session_index < 0 or session_index >= len(sessions):
        raise ValueError(f"session_index out of range: {session_index}")
    info = sessions[session_index]
    session_key = Path(info.session_file).stem
    if str(task.session_key) != session_key:
        raise ValueError(f"Task session_key {task.session_key!r} does not match session {session_key!r}.")

    n_bins = int(np.asarray(theta_grid).reshape(-1).shape[0] - 1)
    session = load_stringer_session(info, orientation_period=float(period))
    a_idx, b_idx = stratified_half_split(
        session.grating_orientation,
        n_bins=n_bins,
        period=float(period),
        seed=int(seed) + session_index,
    )
    base_output_dir = Path(output_dir)
    if task.kind == ASUBSAMPLE_TASK_ENDPOINT_FULL_A:
        half_label = HALF_A
        half_indices = a_idx
        task_output_dir = base_output_dir / "endpoint_full_a"
        task_pca_random_state = int(pca_random_state) + 100 * session_index
        task_seed = int(seed) + 1000 * session_index
    elif task.kind == ASUBSAMPLE_TASK_REFERENCE_B:
        half_label = HALF_B
        half_indices = b_idx
        task_output_dir = base_output_dir / "reference_b"
        task_pca_random_state = int(pca_random_state) + 100 * session_index + 1
        task_seed = int(seed) + 1000 * session_index + 1
    else:
        if task.subset_n is None or task.repeat is None:
            raise ValueError("subset_a tasks require subset_n and repeat.")
        if int(task.subset_n) not in n_list:
            raise ValueError(f"subset_n {task.subset_n} is not in n_values.")
        n_index = int(task.n_index) if task.n_index is not None else int(n_list.index(int(task.subset_n)))
        if n_list[n_index] != int(task.subset_n):
            raise ValueError("subset_a task n_index does not match subset_n.")
        theta_a = np.asarray(session.grating_orientation, dtype=np.float64).reshape(-1)[a_idx]
        local_idx = bootstrap_indices(
            theta_a,
            n_samples=int(task.subset_n),
            n_bins=n_bins,
            period=float(period),
            seed=int(seed) + 100000 * int(n_index + 1) + 10000 * int(task.repeat + 1) + session_index,
            sampling=str(sampling),
            replace=bool(replace),
        )
        half_label = HALF_A
        half_indices = a_idx[local_idx]
        task_output_dir = base_output_dir / "subset_a" / f"n_{int(task.subset_n):06d}" / f"repeat_{int(task.repeat):03d}"
        task_pca_random_state = int(pca_random_state) + 100000 * int(n_index + 1) + 10000 * int(task.repeat + 1) + 100 * session_index
        task_seed = int(seed) + 200000 * int(n_index + 1) + 10000 * int(task.repeat + 1) + session_index

    cache_path = half_curve_cache_path(output_dir=task_output_dir, session_info=info, half_label=half_label)
    signature = half_curve_signature(
        session_info=info,
        session_index=session_index,
        half_label=half_label,
        half_indices=half_indices,
        theta_grid=theta_grid,
        period=float(period),
        pca_dim=int(pca_dim),
        pca_random_state=int(task_pca_random_state),
        pca_whiten=bool(pca_whiten),
        train_frac=float(train_frac),
        seed=int(task_seed),
        flow_config=flow_config,
        flow_orientation_encoding=flow_orientation_encoding,
        classical_ridge=float(classical_ridge),
        classical_window_radius=classical_window_radius,
        classical_min_endpoint_samples=int(classical_min_endpoint_samples),
    )
    return ResolvedASubsampleCurveTask(
        task=task,
        session_info=info,
        half_label=half_label,
        half_indices=np.asarray(half_indices, dtype=np.int64),
        output_dir=task_output_dir,
        pca_random_state=int(task_pca_random_state),
        seed=int(task_seed),
        cache_path=cache_path,
        signature=signature,
    )


def a_subsample_curve_task_cache_hit(task: ASubsampleCurveTask, **kwargs: Any) -> bool:
    resolved = resolve_a_subsample_curve_task(task, **kwargs)
    return load_half_cache(resolved.cache_path, signature=resolved.signature) is not None


def estimate_a_subsample_curve_task(
    task: ASubsampleCurveTask,
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
    n_values: list[int],
    sampling: str,
    flow_orientation_encoding: str = FLOW_ORIENTATION_ENCODING_PERIODIC_SINCOS,
    replace: bool = True,
    force: bool = False,
    save_flow_npz: bool = True,
    classical_ridge: float = 1e-6,
    classical_window_radius: float | None = None,
    classical_min_endpoint_samples: int = 8,
) -> HalfCurveResult:
    resolved = resolve_a_subsample_curve_task(
        task,
        sessions=sessions,
        theta_grid=theta_grid,
        period=float(period),
        pca_dim=int(pca_dim),
        pca_random_state=int(pca_random_state),
        pca_whiten=bool(pca_whiten),
        train_frac=float(train_frac),
        seed=int(seed),
        flow_config=flow_config,
        flow_orientation_encoding=flow_orientation_encoding,
        output_dir=output_dir,
        n_values=n_values,
        sampling=str(sampling),
        replace=bool(replace),
        classical_ridge=float(classical_ridge),
        classical_window_radius=classical_window_radius,
        classical_min_endpoint_samples=int(classical_min_endpoint_samples),
    )
    print(f"[stringer-identification] worker task={task.label}", flush=True)
    return estimate_half_curves(
        session_info=resolved.session_info,
        session_index=int(task.session_index),
        half_label=resolved.half_label,
        half_indices=resolved.half_indices,
        theta_grid=theta_grid,
        period=float(period),
        pca_dim=int(pca_dim),
        pca_random_state=int(resolved.pca_random_state),
        pca_whiten=bool(pca_whiten),
        train_frac=float(train_frac),
        seed=int(resolved.seed),
        device=device,
        flow_config=flow_config,
        flow_orientation_encoding=flow_orientation_encoding,
        output_dir=resolved.output_dir,
        force=bool(force),
        save_flow_npz=bool(save_flow_npz),
        classical_ridge=float(classical_ridge),
        classical_window_radius=classical_window_radius,
        classical_min_endpoint_samples=int(classical_min_endpoint_samples),
    )


def run_a_subsample_convergence(
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
    n_values: list[int],
    repeats: int,
    sampling: str,
    flow_orientation_encoding: str = FLOW_ORIENTATION_ENCODING_PERIODIC_SINCOS,
    replace: bool = True,
    force: bool = False,
    save_flow_npz: bool = True,
    classical_ridge: float = 1e-6,
    classical_window_radius: float | None = None,
    classical_min_endpoint_samples: int = 8,
) -> SubsampleConvergenceResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_list = parse_positive_int_list(n_values)
    if int(repeats) < 1:
        raise ValueError("repeats must be >= 1.")
    if str(sampling) not in {"stratified", "uniform"}:
        raise ValueError("sampling must be 'stratified' or 'uniform'.")
    if any(int(n) < int(pca_dim) for n in n_list):
        raise ValueError("All A subset sizes must be >= pca_dim for fresh per-subset PCA.")

    n_bins = int(np.asarray(theta_grid).reshape(-1).shape[0] - 1)
    session_keys = [Path(info.session_file).stem for info in sessions]
    split_indices: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    full_a_results: list[HalfCurveResult] = []
    reference_b_results: list[HalfCurveResult] = []

    for session_index, info in enumerate(sessions):
        session = load_stringer_session(info, orientation_period=float(period))
        a_idx, b_idx = stratified_half_split(
            session.grating_orientation,
            n_bins=n_bins,
            period=float(period),
            seed=int(seed) + int(session_index),
        )
        session_key = session_keys[session_index]
        theta_a = np.asarray(session.grating_orientation, dtype=np.float64).reshape(-1)[a_idx]
        split_indices[session_key] = (a_idx, b_idx, theta_a)
        full_a_results.append(
            estimate_half_curves(
                session_info=info,
                session_index=session_index,
                half_label=HALF_A,
                half_indices=a_idx,
                theta_grid=theta_grid,
                period=float(period),
                pca_dim=int(pca_dim),
                pca_random_state=int(pca_random_state) + 100 * int(session_index),
                pca_whiten=bool(pca_whiten),
                train_frac=float(train_frac),
                seed=int(seed) + 1000 * int(session_index),
                device=device,
                flow_config=flow_config,
                flow_orientation_encoding=flow_orientation_encoding,
                output_dir=output_dir / "endpoint_full_a",
                force=bool(force),
                save_flow_npz=bool(save_flow_npz),
                classical_ridge=float(classical_ridge),
                classical_window_radius=classical_window_radius,
                classical_min_endpoint_samples=int(classical_min_endpoint_samples),
            )
        )
        reference_b_results.append(
            estimate_half_curves(
                session_info=info,
                session_index=session_index,
                half_label=HALF_B,
                half_indices=b_idx,
                theta_grid=theta_grid,
                period=float(period),
                pca_dim=int(pca_dim),
                pca_random_state=int(pca_random_state) + 100 * int(session_index) + 1,
                pca_whiten=bool(pca_whiten),
                train_frac=float(train_frac),
                seed=int(seed) + 1000 * int(session_index) + 1,
                device=device,
                flow_config=flow_config,
                flow_orientation_encoding=flow_orientation_encoding,
                output_dir=output_dir / "reference_b",
                force=bool(force),
                save_flow_npz=bool(save_flow_npz),
                classical_ridge=float(classical_ridge),
                classical_window_radius=classical_window_radius,
                classical_min_endpoint_samples=int(classical_min_endpoint_samples),
            )
        )

    mids = theta_midpoints(theta_grid)
    endpoint_distances, endpoint_pairs, endpoint_summary = compute_query_reference_identification(
        query_results=full_a_results,
        reference_results=reference_b_results,
        theta_mid=mids,
        session_keys=session_keys,
        query_half=HALF_A,
        candidate_half=HALF_B,
        direction=DIRECTION_A_TO_B,
        row_extra={"subset_n": "full", "repeat": -1, "sampling": "full_A"},
    )
    endpoint_curve_rows = curve_rows_from_halves(full_a_results + reference_b_results, row_extra={"subset_n": "full", "repeat": -1})
    endpoint_result = IdentificationResult(
        session_keys=session_keys,
        theta_grid=np.asarray(theta_grid, dtype=np.float64),
        theta_midpoints=mids,
        half_results=full_a_results + reference_b_results,
        distances=endpoint_distances,
        pair_rows=endpoint_pairs,
        curve_rows=endpoint_curve_rows,
        summary={
            "session_keys": session_keys,
            "identification": endpoint_summary,
            "identification_direction": DIRECTION_A_TO_B,
            "query_half": HALF_A,
            "reference_half": HALF_B,
            "orientation_period": float(period),
            "flow_orientation_encoding": normalize_flow_orientation_encoding(flow_orientation_encoding),
            "flow_condition_dim": int(
                encode_flow_orientation(
                    np.asarray(theta_grid, dtype=np.float64).reshape(-1),
                    period=float(period),
                    encoding=flow_orientation_encoding,
                ).shape[1]
            ),
        },
    )

    subset_matrices: dict[str, dict[str, np.ndarray]] = {
        method: {
            distance_name: np.full((len(n_list), int(repeats), len(session_keys), len(session_keys)), np.nan, dtype=np.float64)
            for distance_name in DISTANCES
        }
        for method in METHODS
    }
    pair_rows = list(endpoint_pairs)
    curve_rows = list(endpoint_curve_rows)
    per_run_summary: list[dict[str, Any]] = []

    for ni, n_subset in enumerate(n_list):
        for repeat in range(int(repeats)):
            query_results: list[HalfCurveResult] = []
            for session_index, info in enumerate(sessions):
                session_key = session_keys[session_index]
                a_idx, _b_idx, theta_a = split_indices[session_key]
                local_idx = bootstrap_indices(
                    theta_a,
                    n_samples=int(n_subset),
                    n_bins=n_bins,
                    period=float(period),
                    seed=int(seed) + 100000 * int(ni + 1) + 10000 * int(repeat + 1) + int(session_index),
                    sampling=str(sampling),
                    replace=bool(replace),
                )
                subset_idx = a_idx[local_idx]
                print(
                    f"[stringer-identification] A subset session={session_key} n={n_subset} "
                    f"repeat={repeat} sampling={sampling} replace={bool(replace)}",
                    flush=True,
                )
                query_results.append(
                    estimate_half_curves(
                        session_info=info,
                        session_index=session_index,
                        half_label=HALF_A,
                        half_indices=subset_idx,
                        theta_grid=theta_grid,
                        period=float(period),
                        pca_dim=int(pca_dim),
                        pca_random_state=int(pca_random_state)
                        + 100000 * int(ni + 1)
                        + 10000 * int(repeat + 1)
                        + 100 * int(session_index),
                        pca_whiten=bool(pca_whiten),
                        train_frac=float(train_frac),
                        seed=int(seed) + 200000 * int(ni + 1) + 10000 * int(repeat + 1) + int(session_index),
                        device=device,
                        flow_config=flow_config,
                        flow_orientation_encoding=flow_orientation_encoding,
                        output_dir=output_dir / "subset_a" / f"n_{int(n_subset):06d}" / f"repeat_{int(repeat):03d}",
                        force=bool(force),
                        save_flow_npz=bool(save_flow_npz),
                        classical_ridge=float(classical_ridge),
                        classical_window_radius=classical_window_radius,
                        classical_min_endpoint_samples=int(classical_min_endpoint_samples),
                    )
                )
            distances, rows, one_summary = compute_query_reference_identification(
                query_results=query_results,
                reference_results=reference_b_results,
                theta_mid=mids,
                session_keys=session_keys,
                query_half=HALF_A,
                candidate_half=HALF_B,
                direction=DIRECTION_A_TO_B,
                row_extra={"subset_n": int(n_subset), "repeat": int(repeat), "sampling": str(sampling)},
            )
            for method in METHODS:
                for distance_name in DISTANCES:
                    subset_matrices[method][distance_name][ni, repeat] = distances[method][distance_name][DIRECTION_A_TO_B]
            pair_rows.extend(rows)
            curve_rows.extend(
                curve_rows_from_halves(
                    query_results,
                    row_extra={"subset_n": int(n_subset), "repeat": int(repeat), "sampling": str(sampling)},
                )
            )
            per_run_summary.append({"subset_n": int(n_subset), "repeat": int(repeat), "identification": one_summary})

    convergence: dict[str, dict[str, dict[str, Any]]] = {}
    for method in METHODS:
        convergence[method] = {}
        for distance_name in DISTANCES:
            top1 = np.full((len(n_list), int(repeats)), np.nan, dtype=np.float64)
            top2 = np.full_like(top1, np.nan)
            top3 = np.full_like(top1, np.nan)
            mrr = np.full_like(top1, np.nan)
            for row in per_run_summary:
                ni = n_list.index(int(row["subset_n"]))
                ri = int(row["repeat"])
                summary = row["identification"]
                top1[ni, ri] = _metric_from_summary(summary, method, distance_name, "top1_accuracy")
                top2[ni, ri] = _metric_from_summary(summary, method, distance_name, "top2_accuracy")
                top3[ni, ri] = _metric_from_summary(summary, method, distance_name, "top3_accuracy")
                mrr[ni, ri] = _metric_from_summary(summary, method, distance_name, "mean_reciprocal_rank")
            endpoint = endpoint_summary[method][f"{distance_name}_{DIRECTION_A_TO_B}"]
            convergence[method][distance_name] = {
                "n_values": list(map(int, n_list)),
                "top1_by_repeat": top1.tolist(),
                "top2_by_repeat": top2.tolist(),
                "top3_by_repeat": top3.tolist(),
                "mrr_by_repeat": mrr.tolist(),
                "top1_mean": np.nanmean(top1, axis=1).tolist(),
                "top2_mean": np.nanmean(top2, axis=1).tolist(),
                "top3_mean": np.nanmean(top3, axis=1).tolist(),
                "mrr_mean": np.nanmean(mrr, axis=1).tolist(),
                "full_a": endpoint,
            }

    summary = {
        "session_keys": session_keys,
        "n_sessions": int(len(session_keys)),
        "methods": list(METHODS),
        "distances": list(DISTANCES),
        "primary_distance": DISTANCE_PRIMARY,
        "identification_direction": DIRECTION_A_TO_B,
        "query_half": HALF_A,
        "reference_half": HALF_B,
        "experiment": "a_subsample_convergence",
        "n_values": list(map(int, n_list)),
        "repeats": int(repeats),
        "sampling": str(sampling),
        "subsample_replace": bool(replace),
        "subsample_without_replacement": not bool(replace),
        "orientation_period": float(period),
        "flow_orientation_encoding": normalize_flow_orientation_encoding(flow_orientation_encoding),
        "flow_condition_dim": int(
            encode_flow_orientation(
                np.asarray(theta_grid, dtype=np.float64).reshape(-1),
                period=float(period),
                encoding=flow_orientation_encoding,
            ).shape[1]
        ),
        "theta_grid_size": int(np.asarray(theta_grid).reshape(-1).shape[0]),
        "pca_fit_scope": "A subset or B half",
        "pca_label_blind": True,
        "pca_trial_averaging_before_fit": False,
        "endpoint_identification": endpoint_summary,
        "subsample_runs": per_run_summary,
        "subsample_convergence": convergence,
    }
    return SubsampleConvergenceResult(
        session_keys=session_keys,
        theta_grid=np.asarray(theta_grid, dtype=np.float64),
        theta_midpoints=mids,
        n_values=list(map(int, n_list)),
        repeats=int(repeats),
        sampling=str(sampling),
        endpoint_result=endpoint_result,
        subset_matrices=subset_matrices,
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


def write_subsample_results_npz(path: Path, result: SubsampleConvergenceResult) -> Path:
    fields: dict[str, Any] = {
        "session_keys": np.asarray(result.session_keys),
        "theta_grid": np.asarray(result.theta_grid, dtype=np.float64),
        "theta_midpoints": np.asarray(result.theta_midpoints, dtype=np.float64),
        "n_values": np.asarray(result.n_values, dtype=np.int64),
        "repeats": np.asarray([int(result.repeats)], dtype=np.int64),
        "sampling": np.asarray([str(result.sampling)]),
        "subsample_replace": np.asarray([bool(result.summary.get("subsample_replace", True))]),
    }
    for method in METHODS:
        for dist_name in DISTANCES:
            fields[f"endpoint_{method}_{dist_name}_{DIRECTION_A_TO_B}"] = np.asarray(
                result.endpoint_result.distances[method][dist_name][DIRECTION_A_TO_B],
                dtype=np.float64,
            )
            fields[f"subset_{method}_{dist_name}_{DIRECTION_A_TO_B}"] = np.asarray(
                result.subset_matrices[method][dist_name],
                dtype=np.float64,
            )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **fields)
    return path


def load_subsample_results_npz(path: Path, summary: dict[str, Any]) -> SubsampleConvergenceResult:
    data = np.load(Path(path), allow_pickle=False)
    session_keys = [str(v) for v in np.asarray(data["session_keys"]).reshape(-1)]
    theta_grid = np.asarray(data["theta_grid"], dtype=np.float64)
    mids = np.asarray(data["theta_midpoints"], dtype=np.float64)
    n_values = [int(v) for v in np.asarray(data["n_values"], dtype=np.int64).reshape(-1)]
    repeats = int(np.asarray(data["repeats"], dtype=np.int64).reshape(-1)[0])
    sampling = str(np.asarray(data["sampling"]).reshape(-1)[0])
    if "subsample_replace" in data.files:
        summary.setdefault("subsample_replace", bool(np.asarray(data["subsample_replace"]).reshape(-1)[0]))
        summary.setdefault("subsample_without_replacement", not bool(np.asarray(data["subsample_replace"]).reshape(-1)[0]))
    endpoint_distances: dict[str, dict[str, dict[str, np.ndarray]]] = {method: {} for method in METHODS}
    subset_matrices: dict[str, dict[str, np.ndarray]] = {method: {} for method in METHODS}
    for method in METHODS:
        for dist_name in DISTANCES:
            endpoint_distances[method][dist_name] = {
                DIRECTION_A_TO_B: np.asarray(data[f"endpoint_{method}_{dist_name}_{DIRECTION_A_TO_B}"], dtype=np.float64)
            }
            subset_matrices[method][dist_name] = np.asarray(
                data[f"subset_{method}_{dist_name}_{DIRECTION_A_TO_B}"],
                dtype=np.float64,
            )
    endpoint_result = IdentificationResult(
        session_keys=session_keys,
        theta_grid=theta_grid,
        theta_midpoints=mids,
        half_results=[],
        distances=endpoint_distances,
        pair_rows=[],
        curve_rows=[],
        summary={
            "identification": summary.get("endpoint_identification", {}),
            "session_keys": session_keys,
            "identification_direction": DIRECTION_A_TO_B,
            "query_half": HALF_A,
            "reference_half": HALF_B,
        },
    )
    return SubsampleConvergenceResult(
        session_keys=session_keys,
        theta_grid=theta_grid,
        theta_midpoints=mids,
        n_values=n_values,
        repeats=repeats,
        sampling=sampling,
        endpoint_result=endpoint_result,
        subset_matrices=subset_matrices,
        pair_rows=[],
        curve_rows=[],
        summary=summary,
    )


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


def write_subsample_curves_csv(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    return write_csv(
        path,
        rows,
        (
            "subset_n",
            "repeat",
            "sampling",
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


def write_subsample_pairs_csv(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    return write_csv(
        path,
        rows,
        (
            "subset_n",
            "repeat",
            "sampling",
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


def write_subsample_summary_json(
    path: Path,
    result: SubsampleConvergenceResult,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
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


def plot_subsample_convergence_summary(
    path_svg: Path,
    path_png: Path,
    result: SubsampleConvergenceResult,
) -> tuple[Path, Path]:
    labels = [str(i) for i in range(len(result.session_keys))]
    n_rows = len(METHODS) * len(DISTANCES)
    fig_height = max(12.0, 2.8 * n_rows)
    fig = plt.figure(
        figsize=(28.0, fig_height),
        layout="constrained",
    )
    gs = fig.add_gridspec(n_rows, 6, width_ratios=[1.0, 0.85, 0.8, 1.25, 1.35, 1.35])
    session_x = np.arange(len(result.session_keys), dtype=np.float64)
    metric_names = ("top1_accuracy", "top2_accuracy", "top3_accuracy", "mean_reciprocal_rank")
    metric_labels = ("top1", "top2", "top3", "MRR")
    topk_keys = ("top1", "top2", "top3")
    topk_labels = ("top1", "top2", "top3")
    topk_colors = ("tab:blue", "tab:orange", "tab:green")
    method_labels = {
        METHOD_CLASSICAL_LINEAR: "classical",
        METHOD_FLOW_LINEAR: "flow matching",
    }
    method_colors = {
        METHOD_CLASSICAL_LINEAR: "tab:orange",
        METHOD_FLOW_LINEAR: "tab:blue",
    }
    n_sessions = max(1, len(result.session_keys))
    chance_by_topk = {
        "top1": 1.0 / float(n_sessions),
        "top3": min(3, n_sessions) / float(n_sessions),
    }
    x_labels = [str(n) for n in result.n_values]
    x = np.arange(len(x_labels), dtype=np.float64)
    old_x_labels = [str(n) for n in result.n_values] + ["full A"]
    old_x = np.arange(len(old_x_labels), dtype=np.float64)

    for distance_index, distance_name in enumerate(DISTANCES):
        first_row = 2 * int(distance_index)
        compare_axes = {
            "top1": fig.add_subplot(gs[first_row : first_row + 2, 4]),
            "top3": fig.add_subplot(gs[first_row : first_row + 2, 5]),
        }
        for method_offset, method in enumerate(METHODS):
            row = first_row + int(method_offset)
            endpoint_mat = result.endpoint_result.distances[method][distance_name][DIRECTION_A_TO_B]
            endpoint_summary = result.summary["endpoint_identification"][method][f"{distance_name}_{DIRECTION_A_TO_B}"]

            heat_ax = fig.add_subplot(gs[row, 0])
            im = heat_ax.imshow(endpoint_mat, cmap="viridis")
            heat_ax.set_title(f"{method}\n{distance_name} endpoint A -> B", fontsize=10)
            heat_ax.set_xlabel("candidate B")
            heat_ax.set_ylabel("query A")
            heat_ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
            heat_ax.set_yticks(np.arange(len(labels)), labels=labels)
            fig.colorbar(im, ax=heat_ax, fraction=0.046, pad=0.04)

            rank_ax = fig.add_subplot(gs[row, 1])
            ranks = np.asarray(endpoint_summary["ranks"], dtype=np.float64)
            rank_ax.bar(session_x, ranks, width=0.65)
            rank_ax.axhline(1.0, color="black", linewidth=0.9, linestyle="--")
            rank_ax.set_title("endpoint rank", fontsize=10)
            rank_ax.set_xlabel("session")
            rank_ax.set_ylabel("rank")
            rank_ax.set_xticks(session_x, labels)
            rank_ax.set_ylim(0.5, len(result.session_keys) + 0.5)
            rank_ax.grid(True, axis="y", alpha=0.25)

            bar_ax = fig.add_subplot(gs[row, 2])
            vals = [float(endpoint_summary[name]) for name in metric_names]
            mx = np.arange(len(metric_names), dtype=np.float64)
            bar_ax.bar(mx, vals, width=0.65)
            bar_ax.set_title("endpoint scores", fontsize=10)
            bar_ax.set_ylim(0.0, 1.05)
            bar_ax.set_xticks(mx, metric_labels, rotation=25, ha="right")
            bar_ax.grid(True, axis="y", alpha=0.25)

            conv = result.summary["subsample_convergence"][method][distance_name]
            conv_ax = fig.add_subplot(gs[row, 3])
            for key, label, color in zip(topk_keys, topk_labels, topk_colors):
                mean_vals = np.asarray(conv[f"{key}_mean"], dtype=np.float64)
                full_val = float(conv["full_a"][f"{key}_accuracy"])
                y = np.concatenate([mean_vals, np.asarray([full_val], dtype=np.float64)])
                conv_ax.plot(old_x, y, marker="o", linewidth=1.6, color=color, label=label)
                by_repeat = np.asarray(conv[f"{key}_by_repeat"], dtype=np.float64)
                if by_repeat.ndim == 2 and by_repeat.shape[1] > 1:
                    for ri in range(by_repeat.shape[1]):
                        yr = np.concatenate([by_repeat[:, ri], np.asarray([full_val], dtype=np.float64)])
                        conv_ax.plot(old_x, yr, color=color, alpha=0.18, linewidth=0.8)
            conv_ax.set_title("top-k vs A subset size", fontsize=10)
            conv_ax.set_xlabel("A subset size")
            conv_ax.set_ylabel("accuracy")
            conv_ax.set_ylim(0.0, 1.05)
            conv_ax.set_xticks(old_x, old_x_labels, rotation=30, ha="right")
            conv_ax.grid(True, axis="y", alpha=0.25)
            conv_ax.legend(fontsize=8, loc="lower right")

            for compare_key, compare_ax in compare_axes.items():
                mean_vals = np.asarray(conv[f"{compare_key}_mean"], dtype=np.float64)
                by_repeat = np.asarray(conv[f"{compare_key}_by_repeat"], dtype=np.float64)
                if by_repeat.ndim == 2 and by_repeat.shape[1] > 1:
                    sd_vals = np.nanstd(by_repeat, axis=1, ddof=1)
                else:
                    sd_vals = np.zeros_like(mean_vals, dtype=np.float64)
                compare_ax.errorbar(
                    x,
                    mean_vals,
                    yerr=sd_vals,
                    marker="o",
                    linewidth=1.8,
                    capsize=3.0,
                    color=method_colors[method],
                    label=method_labels[method],
                )
        for compare_key, compare_ax in compare_axes.items():
            compare_ax.axhline(
                chance_by_topk[compare_key],
                color="black",
                linewidth=1.0,
                linestyle="--",
                label="chance",
            )
            compare_ax.set_title(f"{distance_name} {compare_key} vs A subset size", fontsize=11)
            compare_ax.set_xlabel("A subset size")
            compare_ax.set_ylabel(f"{compare_key} accuracy")
            compare_ax.set_ylim(0.0, 1.05)
            compare_ax.set_xticks(x, x_labels, rotation=30, ha="right")
            compare_ax.grid(True, axis="y", alpha=0.25)
            compare_ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(
        "Stringer A-subset session identification convergence (A queries, fixed B reference)",
        fontsize=16,
    )
    path_svg = Path(path_svg)
    path_png = Path(path_png)
    path_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_svg)
    fig.savefig(path_png, dpi=180)
    plt.close(fig)
    return path_svg, path_png


def plot_subsample_topk_accuracy(
    path_svg: Path,
    path_png: Path,
    result: SubsampleConvergenceResult,
) -> tuple[Path, Path]:
    n_labels = [str(n) for n in result.n_values]
    x = np.arange(len(n_labels), dtype=np.float64)
    row_specs = (
        ("top1", "top1 accuracy"),
        ("top3", "top3 accuracy"),
    )
    method_labels = {
        METHOD_CLASSICAL_LINEAR: "classical",
        METHOD_FLOW_LINEAR: "flow matching",
    }
    method_colors = {
        METHOD_CLASSICAL_LINEAR: "tab:orange",
        METHOD_FLOW_LINEAR: "tab:blue",
    }
    n_sessions = max(1, len(result.session_keys))
    chance_by_topk = {
        "top1": 1.0 / float(n_sessions),
        "top3": min(3, n_sessions) / float(n_sessions),
    }

    fig, axes = plt.subplots(
        len(row_specs),
        len(DISTANCES),
        figsize=(11, 6.8),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for row_index, (topk_key, y_label) in enumerate(row_specs):
        for col_index, distance_name in enumerate(DISTANCES):
            ax = axes[row_index, col_index]
            for method in METHODS:
                conv = result.summary["subsample_convergence"][method][distance_name]
                mean_vals = np.asarray(conv[f"{topk_key}_mean"], dtype=np.float64)
                by_repeat = np.asarray(conv[f"{topk_key}_by_repeat"], dtype=np.float64)
                if by_repeat.ndim == 2 and by_repeat.shape[1] > 1:
                    sd_vals = np.nanstd(by_repeat, axis=1, ddof=1)
                else:
                    sd_vals = np.zeros_like(mean_vals, dtype=np.float64)
                ax.errorbar(
                    x,
                    mean_vals,
                    yerr=sd_vals,
                    marker="o",
                    linewidth=2.0,
                    capsize=3.0,
                    color=method_colors[method],
                    label=method_labels[method],
                )
            ax.axhline(
                chance_by_topk[topk_key],
                color="black",
                linewidth=1.0,
                linestyle="--",
                label="chance",
            )
            if row_index == 0:
                ax.set_title(distance_name, fontsize=11)
            if col_index == 0:
                ax.set_ylabel(y_label)
            ax.set_xlabel("A subset size")
            ax.set_xticks(x, n_labels, rotation=25, ha="right")
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_ylim(0.0, 1.05)
            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row_index == 0 and col_index == 0:
                ax.legend(loc="lower right", frameon=False, fontsize=9)

    path_svg = Path(path_svg)
    path_png = Path(path_png)
    path_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_svg, bbox_inches="tight")
    fig.savefig(path_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path_svg, path_png


def zscore_log_fisher_curve(fisher: np.ndarray, *, tiny: float = 1e-12) -> np.ndarray:
    values = np.asarray(fisher, dtype=np.float64).reshape(-1)
    safe = np.where(np.isfinite(values) & (values > float(tiny)), values, float(tiny))
    logged = np.log(safe)
    mean = float(np.mean(logged))
    sd = float(np.std(logged))
    if not np.isfinite(sd) or sd <= 0.0:
        return np.zeros_like(logged, dtype=np.float64)
    return (logged - mean) / sd


def select_logcorr_flow_advantage_example(
    summary: dict[str, Any],
    *,
    n_subset: int = 1550,
) -> dict[str, Any]:
    session_keys = [str(v) for v in summary.get("session_keys", [])]
    candidates: list[dict[str, Any]] = []
    for run_summary in summary.get("subsample_runs", []):
        if int(run_summary.get("subset_n")) != int(n_subset):
            continue
        identification = run_summary.get("identification", {})
        classical = identification[METHOD_CLASSICAL_LINEAR][f"{DISTANCE_PRIMARY}_{DIRECTION_A_TO_B}"]
        flow = identification[METHOD_FLOW_LINEAR][f"{DISTANCE_PRIMARY}_{DIRECTION_A_TO_B}"]
        classical_ranks = [int(v) for v in classical["ranks"]]
        flow_ranks = [int(v) for v in flow["ranks"]]
        for session_index, (classical_rank, flow_rank) in enumerate(zip(classical_ranks, flow_ranks)):
            if int(flow_rank) == 1 and int(classical_rank) > 1:
                candidates.append(
                    {
                        "subset_n": int(n_subset),
                        "repeat": int(run_summary["repeat"]),
                        "session_index": int(session_index),
                        "session_key": session_keys[int(session_index)],
                        "classical_rank": int(classical_rank),
                        "flow_rank": int(flow_rank),
                    }
                )
    if not candidates:
        raise ValueError(f"No flow-advantage log-correlation example found at A subset size {int(n_subset)}.")
    candidates = sorted(candidates, key=lambda row: (int(row["repeat"]), int(row["session_index"])))
    rng = np.random.default_rng(int(summary.get("seed", 0)))
    return candidates[int(rng.integers(0, len(candidates)))]


def _read_curve_rows_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def _curve_from_rows(
    rows: list[dict[str, str]],
    *,
    subset_n: int | str,
    repeat: int,
    session_index: int,
    half_label: str,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    subset_key = str(subset_n)
    selected = [
        row
        for row in rows
        if str(row.get("subset_n")) == subset_key
        and int(row.get("repeat", 0)) == int(repeat)
        and int(row.get("session_index", -1)) == int(session_index)
        and str(row.get("half_label")) == str(half_label)
        and str(row.get("method")) == str(method)
    ]
    if not selected:
        raise ValueError(
            "Missing curve rows for "
            f"subset_n={subset_key} repeat={int(repeat)} session_index={int(session_index)} "
            f"half={half_label} method={method}."
        )
    selected = sorted(selected, key=lambda row: float(row["theta_midpoint"]))
    theta = np.asarray([float(row["theta_midpoint"]) for row in selected], dtype=np.float64)
    fisher = np.asarray([float(row["fisher"]) for row in selected], dtype=np.float64)
    return theta, fisher


def _set_angle_ticks(ax: Any, period: float) -> None:
    ticks = [0.0, 0.25 * float(period), 0.5 * float(period), 0.75 * float(period)]
    labels = ["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$"] if math.isclose(float(period), math.pi) else None
    ax.set_xticks(ticks, labels)


def plot_subsample_logcorr_example(
    path_svg: Path,
    path_png: Path,
    result: SubsampleConvergenceResult,
    curves_csv_path: Path,
    *,
    n_subset: int = 1550,
) -> tuple[Path, Path]:
    curve_rows = _read_curve_rows_csv(curves_csv_path)
    example = select_logcorr_flow_advantage_example(result.summary, n_subset=int(n_subset))
    session_index = int(example["session_index"])
    repeat = int(example["repeat"])
    session_label = str(session_index)
    method_labels = {
        METHOD_CLASSICAL_LINEAR: "classical linear",
        METHOD_FLOW_LINEAR: "flow matching",
    }
    method_ranks = {
        METHOD_CLASSICAL_LINEAR: int(example["classical_rank"]),
        METHOD_FLOW_LINEAR: int(example["flow_rank"]),
    }

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 3.8),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]},
    )
    for ax, method in zip(axes[:2], METHODS):
        theta_a, fisher_a = _curve_from_rows(
            curve_rows,
            subset_n=int(n_subset),
            repeat=repeat,
            session_index=session_index,
            half_label=HALF_A,
            method=method,
        )
        theta_b, fisher_b = _curve_from_rows(
            curve_rows,
            subset_n="full",
            repeat=-1,
            session_index=session_index,
            half_label=HALF_B,
            method=method,
        )
        ax.plot(theta_b, zscore_log_fisher_curve(fisher_b), color="0.45", linewidth=2.0, linestyle="--", label="paired B")
        ax.plot(theta_a, zscore_log_fisher_curve(fisher_a), color="black", linewidth=2.0, marker="o", markersize=3.2, label="A subset")
        ax.set_title(f"{method_labels[method]}\nlog-corr rank {method_ranks[method]}", fontsize=11)
        ax.set_xlabel("orientation angle (rad)")
        ax.set_ylabel("z-score log Fisher")
        _set_angle_ticks(ax, float(result.summary.get("orientation_period", math.pi)))
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(loc="best", frameon=False, fontsize=9)

    ax = axes[2]
    n_labels = [str(n) for n in result.n_values]
    x = np.arange(len(n_labels), dtype=np.float64)
    method_colors = {
        METHOD_CLASSICAL_LINEAR: "tab:orange",
        METHOD_FLOW_LINEAR: "tab:blue",
    }
    method_short_labels = {
        METHOD_CLASSICAL_LINEAR: "classical",
        METHOD_FLOW_LINEAR: "flow matching",
    }
    for method in METHODS:
        conv = result.summary["subsample_convergence"][method][DISTANCE_PRIMARY]
        mean_vals = np.asarray(conv["top1_mean"], dtype=np.float64)
        by_repeat = np.asarray(conv["top1_by_repeat"], dtype=np.float64)
        sd_vals = np.nanstd(by_repeat, axis=1, ddof=1) if by_repeat.ndim == 2 and by_repeat.shape[1] > 1 else np.zeros_like(mean_vals)
        ax.errorbar(
            x,
            mean_vals,
            yerr=sd_vals,
            marker="o",
            linewidth=2.0,
            capsize=3.0,
            color=method_colors[method],
            label=method_short_labels[method],
        )
    chance = 1.0 / float(max(1, len(result.session_keys)))
    ax.axhline(chance, color="black", linewidth=1.0, linestyle="--", label="chance")
    ax.set_title("log-correlation top1", fontsize=11)
    ax.set_xlabel("A subset size")
    ax.set_ylabel("top1 accuracy")
    ax.set_xticks(x, n_labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    fig.suptitle(f"Example session {session_label}, A subset n={int(n_subset)}, repeat={repeat}", fontsize=12)
    path_svg = Path(path_svg)
    path_png = Path(path_png)
    path_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_svg, bbox_inches="tight")
    fig.savefig(path_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path_svg, path_png
