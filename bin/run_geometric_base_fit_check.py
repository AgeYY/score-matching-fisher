#!/usr/bin/env python3
"""Unified geometric-base flow diagnostic for line, square, and half-circle data."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import DATA_DIR, DEFAULT_DEVICE

from fisher.geometric_base_flow_skl import (
    HalfCircle3DBase,
    HalfCircleBase,
    LineSegmentBase,
    NoisyGeometricBase,
    SquarePerimeterBase,
    build_geometric_base_velocity_model,
    estimate_smoothed_curve_symmetric_kl,
    finetune_geometric_base_cnf_likelihood,
    push_base_curve,
    push_initial_points,
    train_geometric_base_affine_flow,
)
from fisher.noisy_half_circle_dataset import NoisyHalfCircleBoundaryDataset
from fisher.noisy_line_dataset import NoisyLineDataset
from fisher.noisy_square_dataset import NoisySquareBoundaryDataset
from fisher.shared_fisher_est import require_device


DATASET_CHOICES = (
    "two-line",
    "one-line",
    "two-square",
    "one-square",
    "two-half-circle",
    "one-half-circle",
    "two-half-circle-3d",
    "one-half-circle-3d",
)
VELOCITY_CHOICES = ("lie-affine-2d", "lie-similarity-2d", "lie-similarity-3d")
DATASET_DEFAULT_TARGET_SIGMA = {
    "two-line": 0.12,
    "one-line": 0.12,
    "two-square": 0.2,
    "one-square": 0.2,
    "two-half-circle": 0.2,
    "one-half-circle": 0.2,
    "two-half-circle-3d": 0.2,
    "one-half-circle-3d": 0.2,
}


class RotatedGeometricBase:
    """Rotate a 2D geometric base while preserving the intrinsic coordinate."""

    def __init__(self, base: Any, *, angle_degrees: float) -> None:
        angle = float(angle_degrees)
        if not math.isfinite(angle):
            raise ValueError("angle_degrees must be finite.")
        if int(getattr(base, "ambient_dim", 0)) != 2:
            raise ValueError("RotatedGeometricBase only supports 2D bases.")
        self.base = base
        self.angle_degrees = angle
        self.angle_radians = math.radians(angle)
        self.u_low = float(base.u_low)
        self.u_high = float(base.u_high)
        self.name = f"rotated_{getattr(base, 'name', 'geometric_base')}"

    @property
    def ambient_dim(self) -> int:
        return int(self.base.ambient_dim)

    @property
    def intrinsic_dim(self) -> int:
        return int(self.base.intrinsic_dim)

    def sample_u(self, n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.base.sample_u(int(n), device=device, dtype=dtype)

    def points_from_u(self, u: torch.Tensor) -> torch.Tensor:
        x = self.base.points_from_u(u)
        c = math.cos(self.angle_radians)
        s = math.sin(self.angle_radians)
        rot = torch.tensor([[c, -s], [s, c]], device=x.device, dtype=x.dtype)
        return x @ rot.T

    def sample(self, n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.points_from_u(self.sample_u(int(n), device=device, dtype=dtype))

    def sample_with_u(self, n: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.sample_u(int(n), device=device, dtype=dtype)
        return self.points_from_u(u), u


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASET_CHOICES, default="two-line")
    p.add_argument("--velocity-family", type=str, default="lie-affine-2d")
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--theta-values", type=str, default="")
    p.add_argument("--condition-values", type=str, default="0.0,1.0")
    p.add_argument("--ell", type=float, default=1.5)
    p.add_argument("--side-length", type=float, default=2.0)
    p.add_argument("--base-side-length", type=float, default=1.0)
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--base-radius", type=float, default=1.0)
    p.add_argument("--base-angle-deg", type=float, default=0.0)
    p.add_argument("--base-noise-sigma", type=float, default=0.1)
    p.add_argument("--target-sigma", type=float, default=None)
    p.add_argument("--shift-x", type=float, default=0.0)
    p.add_argument("--shift-y", type=float, default=0.0)
    p.add_argument("--center-x", type=float, default=0.0)
    p.add_argument("--center-y", type=float, default=0.0)
    p.add_argument("--left-center-x", type=float, default=-1.0)
    p.add_argument("--left-center-y", type=float, default=0.0)
    p.add_argument("--left-center-z", type=float, default=0.0)
    p.add_argument("--right-center-x", type=float, default=1.0)
    p.add_argument("--right-center-y", type=float, default=0.0)
    p.add_argument("--right-center-z", type=float, default=0.0)
    p.add_argument("--n-per-condition", "--n-per-theta", dest="n_per_condition", type=int, default=3000)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--max-test-plot-per-condition", "--max-test-plot-per-theta", dest="max_test_plot_per_condition", type=int, default=600)

    p.add_argument("--path-schedule", choices=("linear", "straight", "cosine"), default="cosine")
    p.add_argument("--smooth-sigma", type=float, default=0.12)
    p.add_argument("--mc-skl-samples", type=int, default=1024)
    p.add_argument("--density-mc-samples", type=int, default=512)
    p.add_argument("--ode-steps", type=int, default=64)
    p.add_argument("--ode-method", type=str, default="midpoint")
    p.add_argument("--curve-points", type=int, default=300)
    p.add_argument("--curve-points-per-edge", type=int, default=100)
    p.add_argument("--generated-samples-per-condition", "--generated-samples-per-theta", dest="generated_samples_per_condition", type=int, default=600)

    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--t-eps", type=float, default=0.0005)
    p.add_argument("--early-patience", type=int, default=1000)
    p.add_argument("--early-min-delta", type=float, default=1e-4)
    p.add_argument("--early-ema-alpha", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=100)

    p.add_argument("--nf-likelihood-finetune", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--nf-epochs", type=int, default=500)
    p.add_argument("--nf-batch-size", type=int, default=0, help="0 reuses --batch-size.")
    p.add_argument("--nf-lr", type=float, default=1e-4)
    p.add_argument("--nf-weight-decay", type=float, default=0.0)
    p.add_argument("--nf-density-points", type=int, default=512)
    p.add_argument("--nf-checkpoint-selection", choices=("last", "best"), default="last")
    p.add_argument("--nf-learn-base-noise", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--nf-sigma-min", type=float, default=1e-4)
    return p


def _parse_values(text: str, *, default: tuple[float, ...], name: str) -> np.ndarray:
    raw = str(text).strip()
    vals = list(default) if raw == "" else [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(vals) < 1:
        raise ValueError(f"{name} must contain at least one comma-separated value.")
    if not np.all(np.isfinite(vals)):
        raise ValueError(f"{name} must be finite.")
    return np.asarray(vals, dtype=np.float64).reshape(-1, 1)


def _select_condition_rows(values: np.ndarray, *, dataset: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    if dataset.startswith("two-"):
        if int(arr.shape[0]) != 2:
            raise ValueError(f"{dataset} requires exactly two condition values.")
        return arr
    if int(arr.shape[0]) == 1:
        return arr
    return arr[1:2]


def _condition_one_hot(n_conditions: int) -> np.ndarray:
    count = int(n_conditions)
    if count < 1:
        raise ValueError("At least one condition is required.")
    return np.eye(count, dtype=np.float64)


def _normalize_velocity_family(value: str) -> str:
    key = str(value).strip().lower().replace("_", "-")
    aliases = {
        "2d-affine": "lie-affine-2d",
        "2d affine": "lie-affine-2d",
        "affine-2d": "lie-affine-2d",
        "lie-affine-2d": "lie-affine-2d",
        "2d-similarity": "lie-similarity-2d",
        "2d similarity": "lie-similarity-2d",
        "similarity-2d": "lie-similarity-2d",
        "lie-similarity-2d": "lie-similarity-2d",
        "3d-similarity": "lie-similarity-3d",
        "3d similarity": "lie-similarity-3d",
        "similarity-3d": "lie-similarity-3d",
        "lie-similarity-3d": "lie-similarity-3d",
    }
    if key not in aliases:
        raise ValueError(f"--velocity-family must be one of {VELOCITY_CHOICES}; got {value!r}.")
    return aliases[key]


def validate_dataset_velocity(args: argparse.Namespace) -> str:
    dataset = str(args.dataset)
    velocity = _normalize_velocity_family(str(args.velocity_family))
    is_3d_dataset = dataset.endswith("-3d")
    if is_3d_dataset and velocity != "lie-similarity-3d":
        raise ValueError("3D half-circle datasets require --velocity-family lie-similarity-3d.")
    if velocity == "lie-similarity-3d" and not is_3d_dataset:
        raise ValueError("--velocity-family lie-similarity-3d is only valid for 3D half-circle datasets.")
    if not is_3d_dataset and velocity not in ("lie-affine-2d", "lie-similarity-2d"):
        raise ValueError("2D datasets require a 2D velocity family.")
    if bool(args.nf_likelihood_finetune) and float(args.base_noise_sigma) <= 0.0:
        raise ValueError("--base-noise-sigma must be > 0 when --nf-likelihood-finetune is enabled.")
    if (
        bool(args.nf_likelihood_finetune)
        and bool(args.nf_learn_base_noise)
        and float(args.base_noise_sigma) <= float(args.nf_sigma_min)
    ):
        raise ValueError("--base-noise-sigma must be greater than --nf-sigma-min when --nf-learn-base-noise is enabled.")
    return velocity


def _target_sigma(args: argparse.Namespace) -> float:
    sigma = DATASET_DEFAULT_TARGET_SIGMA[str(args.dataset)] if args.target_sigma is None else float(args.target_sigma)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("--target-sigma must be finite and nonnegative.")
    return float(sigma)


def _split_indices(
    n_total: int,
    *,
    train_frac: float,
    val_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(n_total)
    if count < 3:
        raise ValueError("--n-per-condition must be >= 3 so train/validation/test splits are non-empty.")
    trf = float(train_frac)
    vf = float(val_frac)
    if not (0.0 < trf < 1.0):
        raise ValueError("--train-frac must be in (0, 1).")
    if not (0.0 < vf < 1.0):
        raise ValueError("--val-frac must be in (0, 1).")
    if trf + vf >= 1.0:
        raise ValueError("--train-frac + --val-frac must be < 1.")
    idx = rng.permutation(count)
    n_train = min(max(int(round(trf * count)), 1), count - 2)
    n_val = min(max(int(round(vf * count)), 1), count - n_train - 1)
    return idx[:n_train].astype(np.int64), idx[n_train : n_train + n_val].astype(np.int64), idx[n_train + n_val :].astype(np.int64)


def _half_circle_3d_from_u(
    u: np.ndarray,
    *,
    radius: float,
    center: tuple[float, float, float],
    arc: str,
) -> np.ndarray:
    u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
    r = float(radius)
    if not math.isfinite(r) or r <= 0.0:
        raise ValueError("radius must be finite and positive.")
    arc_norm = str(arc).strip().lower()
    if arc_norm not in ("upper", "lower"):
        raise ValueError("arc must be upper or lower.")
    theta = math.pi * np.clip(u_arr, 0.0, 1.0)
    y_sign = 1.0 if arc_norm == "upper" else -1.0
    base = np.column_stack((r * np.cos(theta), y_sign * r * np.sin(theta), np.zeros_like(theta)))
    return base + np.asarray(center, dtype=np.float64).reshape(1, 3)


def _append_split(
    *,
    idx: int,
    condition_eval: np.ndarray,
    condition_value: float,
    x_all: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    plot_idx: np.ndarray,
    theta_train_parts: list[np.ndarray],
    x_train_parts: list[np.ndarray],
    theta_val_parts: list[np.ndarray],
    x_val_parts: list[np.ndarray],
    theta_test_parts: list[np.ndarray],
    x_test_parts: list[np.ndarray],
    theta_test_plot_parts: list[np.ndarray],
    theta_test_plot_scalar_parts: list[np.ndarray],
    x_test_plot_parts: list[np.ndarray],
) -> None:
    n_total = int(x_all.shape[0])
    theta_col = np.repeat(condition_eval[idx : idx + 1], n_total, axis=0)
    theta_scalar_col = np.full((n_total, 1), float(condition_value), dtype=np.float64)
    theta_train_parts.append(theta_col[train_idx])
    x_train_parts.append(x_all[train_idx])
    theta_val_parts.append(theta_col[val_idx])
    x_val_parts.append(x_all[val_idx])
    theta_test_parts.append(theta_col[test_idx])
    x_test_parts.append(x_all[test_idx])
    theta_test_plot_parts.append(theta_col[plot_idx])
    theta_test_plot_scalar_parts.append(theta_scalar_col[plot_idx])
    x_test_plot_parts.append(x_all[plot_idx])


def make_geometric_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset = str(args.dataset)
    n_total = int(args.n_per_condition)
    max_plot = int(args.max_test_plot_per_condition)
    if max_plot < 1:
        raise ValueError("--max-test-plot-per-condition must be >= 1.")
    target_sigma = _target_sigma(args)
    if dataset in ("two-line", "one-line"):
        default_values = (math.pi / 4.0, 3.0 * math.pi / 4.0)
        condition_values = _select_condition_rows(_parse_values(args.theta_values, default=default_values, name="--theta-values"), dataset=dataset)
        kind = "line"
    elif dataset in ("two-square", "one-square"):
        default_values = (0.0, math.pi / 4.0)
        condition_values = _select_condition_rows(_parse_values(args.theta_values, default=default_values, name="--theta-values"), dataset=dataset)
        kind = "square"
    elif dataset in ("two-half-circle", "one-half-circle", "two-half-circle-3d", "one-half-circle-3d"):
        condition_values = _select_condition_rows(_parse_values(args.condition_values, default=(0.0, 1.0), name="--condition-values"), dataset=dataset)
        kind = "half_circle_3d" if dataset.endswith("-3d") else "half_circle"
    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}.")

    condition_eval = _condition_one_hot(int(condition_values.shape[0]))
    split_rng = np.random.default_rng(int(args.seed) + 101)
    plot_rng = np.random.default_rng(int(args.seed) + 202)
    theta_train_parts: list[np.ndarray] = []
    x_train_parts: list[np.ndarray] = []
    theta_val_parts: list[np.ndarray] = []
    x_val_parts: list[np.ndarray] = []
    theta_test_parts: list[np.ndarray] = []
    x_test_parts: list[np.ndarray] = []
    theta_test_plot_parts: list[np.ndarray] = []
    theta_test_plot_scalar_parts: list[np.ndarray] = []
    x_test_plot_parts: list[np.ndarray] = []
    target_curves: list[np.ndarray] = []
    split_counts: dict[str, dict[str, int]] = {}
    condition_labels: list[str] = []

    if kind == "line":
        for idx, value in enumerate(condition_values[:, 0]):
            ds = NoisyLineDataset(
                theta=float(value),
                ell=float(args.ell),
                sigma=target_sigma,
                shift=(float(args.shift_x), float(args.shift_y)),
                seed=int(args.seed) + idx,
            )
            x_all = ds.sample(n_total).x1.astype(np.float64, copy=False)
            train_idx, val_idx, test_idx = _split_indices(n_total, train_frac=args.train_frac, val_frac=args.val_frac, rng=split_rng)
            plot_idx = test_idx if int(test_idx.size) <= max_plot else np.sort(plot_rng.choice(test_idx, size=max_plot, replace=False))
            _append_split(
                idx=idx,
                condition_eval=condition_eval,
                condition_value=float(value),
                x_all=x_all,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                plot_idx=plot_idx,
                theta_train_parts=theta_train_parts,
                x_train_parts=x_train_parts,
                theta_val_parts=theta_val_parts,
                x_val_parts=x_val_parts,
                theta_test_parts=theta_test_parts,
                x_test_parts=x_test_parts,
                theta_test_plot_parts=theta_test_plot_parts,
                theta_test_plot_scalar_parts=theta_test_plot_scalar_parts,
                x_test_plot_parts=x_test_plot_parts,
            )
            target_curves.append(ds.centerline(num=int(args.curve_points)))
            condition_labels.append(f"theta={float(value):.4g}")
            split_counts[f"condition_{idx}"] = {"train": int(train_idx.size), "validation": int(val_idx.size), "test": int(test_idx.size), "test_plotted": int(plot_idx.size)}
    elif kind == "square":
        for idx, value in enumerate(condition_values[:, 0]):
            ds = NoisySquareBoundaryDataset(
                theta=float(value),
                side_length=float(args.side_length),
                sigma=target_sigma,
                center=(float(args.center_x), float(args.center_y)),
                seed=int(args.seed) + idx,
            )
            x_all = ds.sample(n_total).x1.astype(np.float64, copy=False)
            train_idx, val_idx, test_idx = _split_indices(n_total, train_frac=args.train_frac, val_frac=args.val_frac, rng=split_rng)
            plot_idx = test_idx if int(test_idx.size) <= max_plot else np.sort(plot_rng.choice(test_idx, size=max_plot, replace=False))
            _append_split(
                idx=idx,
                condition_eval=condition_eval,
                condition_value=float(value),
                x_all=x_all,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                plot_idx=plot_idx,
                theta_train_parts=theta_train_parts,
                x_train_parts=x_train_parts,
                theta_val_parts=theta_val_parts,
                x_val_parts=x_val_parts,
                theta_test_parts=theta_test_parts,
                x_test_parts=x_test_parts,
                theta_test_plot_parts=theta_test_plot_parts,
                theta_test_plot_scalar_parts=theta_test_plot_scalar_parts,
                x_test_plot_parts=x_test_plot_parts,
            )
            target_curves.append(ds.boundary(points_per_edge=int(args.curve_points_per_edge)))
            condition_labels.append(f"theta={float(value):.4g}")
            split_counts[f"condition_{idx}"] = {"train": int(train_idx.size), "validation": int(val_idx.size), "test": int(test_idx.size), "test_plotted": int(plot_idx.size)}
    elif kind == "half_circle":
        all_centers = [(float(args.left_center_x), float(args.left_center_y)), (float(args.right_center_x), float(args.right_center_y))]
        all_arcs = ["upper", "lower"]
        start = 1 if dataset.startswith("one-") else 0
        centers = all_centers[start : start + int(condition_values.shape[0])]
        arcs = all_arcs[start : start + int(condition_values.shape[0])]
        for idx, value in enumerate(condition_values[:, 0]):
            ds = NoisyHalfCircleBoundaryDataset(
                radius=float(args.radius),
                sigma=target_sigma,
                center=centers[idx],
                arc=arcs[idx],
                seed=int(args.seed) + idx,
            )
            x_all = ds.sample(n_total).x1.astype(np.float64, copy=False)
            train_idx, val_idx, test_idx = _split_indices(n_total, train_frac=args.train_frac, val_frac=args.val_frac, rng=split_rng)
            plot_idx = test_idx if int(test_idx.size) <= max_plot else np.sort(plot_rng.choice(test_idx, size=max_plot, replace=False))
            _append_split(
                idx=idx,
                condition_eval=condition_eval,
                condition_value=float(value),
                x_all=x_all,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                plot_idx=plot_idx,
                theta_train_parts=theta_train_parts,
                x_train_parts=x_train_parts,
                theta_val_parts=theta_val_parts,
                x_val_parts=x_val_parts,
                theta_test_parts=theta_test_parts,
                x_test_parts=x_test_parts,
                theta_test_plot_parts=theta_test_plot_parts,
                theta_test_plot_scalar_parts=theta_test_plot_scalar_parts,
                x_test_plot_parts=x_test_plot_parts,
            )
            target_curves.append(ds.boundary(n_points=int(args.curve_points)))
            condition_labels.append(f"{arcs[idx]} half-circle")
            split_counts[f"condition_{idx}"] = {"train": int(train_idx.size), "validation": int(val_idx.size), "test": int(test_idx.size), "test_plotted": int(plot_idx.size)}
    else:
        all_centers_3d = [
            (float(args.left_center_x), float(args.left_center_y), float(args.left_center_z)),
            (float(args.right_center_x), float(args.right_center_y), float(args.right_center_z)),
        ]
        all_arcs = ["upper", "lower"]
        start = 1 if dataset.startswith("one-") else 0
        centers_3d = all_centers_3d[start : start + int(condition_values.shape[0])]
        arcs = all_arcs[start : start + int(condition_values.shape[0])]
        for idx, value in enumerate(condition_values[:, 0]):
            rng = np.random.default_rng(int(args.seed) + idx)
            u = rng.uniform(0.0, 1.0, size=(n_total, 1)).astype(np.float64, copy=False)
            clean = _half_circle_3d_from_u(u, radius=float(args.radius), center=centers_3d[idx], arc=arcs[idx])
            x_all = clean + target_sigma * rng.standard_normal(size=(n_total, 3)).astype(np.float64, copy=False)
            train_idx, val_idx, test_idx = _split_indices(n_total, train_frac=args.train_frac, val_frac=args.val_frac, rng=split_rng)
            plot_idx = test_idx if int(test_idx.size) <= max_plot else np.sort(plot_rng.choice(test_idx, size=max_plot, replace=False))
            _append_split(
                idx=idx,
                condition_eval=condition_eval,
                condition_value=float(value),
                x_all=x_all,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                plot_idx=plot_idx,
                theta_train_parts=theta_train_parts,
                x_train_parts=x_train_parts,
                theta_val_parts=theta_val_parts,
                x_val_parts=x_val_parts,
                theta_test_parts=theta_test_parts,
                x_test_parts=x_test_parts,
                theta_test_plot_parts=theta_test_plot_parts,
                theta_test_plot_scalar_parts=theta_test_plot_scalar_parts,
                x_test_plot_parts=x_test_plot_parts,
            )
            u_curve = np.linspace(0.0, 1.0, int(args.curve_points), dtype=np.float64)
            target_curves.append(_half_circle_3d_from_u(u_curve, radius=float(args.radius), center=centers_3d[idx], arc=arcs[idx]))
            condition_labels.append(f"{arcs[idx]} half-circle 3D")
            split_counts[f"condition_{idx}"] = {"train": int(train_idx.size), "validation": int(val_idx.size), "test": int(test_idx.size), "test_plotted": int(plot_idx.size)}

    theta_train = np.concatenate(theta_train_parts, axis=0)
    x_train = np.concatenate(x_train_parts, axis=0)
    theta_val = np.concatenate(theta_val_parts, axis=0)
    x_val = np.concatenate(x_val_parts, axis=0)
    shuffle_rng = np.random.default_rng(int(args.seed) + 303)
    perm = shuffle_rng.permutation(int(theta_train.shape[0]))
    val_perm = shuffle_rng.permutation(int(theta_val.shape[0]))
    return {
        "dataset_kind": kind,
        "ambient_dim": int(x_train.shape[1]),
        "condition_values": condition_values,
        "condition_eval": condition_eval,
        "theta_train": theta_train[perm],
        "x_train": x_train[perm],
        "theta_val": theta_val[val_perm],
        "x_val": x_val[val_perm],
        "theta_test": np.concatenate(theta_test_parts, axis=0),
        "x_test": np.concatenate(x_test_parts, axis=0),
        "theta_test_plot": np.concatenate(theta_test_plot_parts, axis=0),
        "theta_test_plot_scalar": np.concatenate(theta_test_plot_scalar_parts, axis=0),
        "x_test_plot": np.concatenate(x_test_plot_parts, axis=0),
        "target_curves": target_curves,
        "condition_labels": condition_labels,
        "target_sigma": target_sigma,
        "theta_encoding": "one_hot",
        "split_counts": split_counts,
    }


def make_base(args: argparse.Namespace, *, dataset_kind: str) -> NoisyGeometricBase:
    noise = float(args.base_noise_sigma)
    if not math.isfinite(noise) or noise < 0.0:
        raise ValueError("--base-noise-sigma must be finite and nonnegative.")
    if dataset_kind == "line":
        clean: Any = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0), u_low=-0.5, u_high=0.5)
    elif dataset_kind == "square":
        clean = SquarePerimeterBase(center=(0.0, 0.0), side_length=float(args.base_side_length))
    elif dataset_kind == "half_circle":
        clean = RotatedGeometricBase(HalfCircleBase(center=(0.0, 0.0), radius=float(args.base_radius)), angle_degrees=float(args.base_angle_deg))
    elif dataset_kind == "half_circle_3d":
        clean = HalfCircle3DBase(center=(0.0, 0.0, 0.0), radius=float(args.base_radius))
    else:
        raise ValueError(f"Unsupported dataset_kind: {dataset_kind!r}.")
    return NoisyGeometricBase(clean, sigma=noise)


def resolve_output_paths(output_dir: Path | None, *, dataset: str, velocity_family: str, nf_likelihood: bool) -> dict[str, Path]:
    tag = f"{dataset}__{velocity_family}__{'nf' if nf_likelihood else 'fm'}"
    out_dir = Path(DATA_DIR) / "geometric_base_fit_check" / tag if output_dir is None else Path(output_dir)
    out_dir = out_dir.expanduser().resolve()
    return {
        "output_dir": out_dir,
        "png": out_dir / "geometric_base_fit_check.png",
        "svg": out_dir / "geometric_base_fit_check.svg",
        "summary": out_dir / "geometric_base_fit_check_summary.json",
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _set_2d_equal_axes(ax: Any, arrays: list[np.ndarray]) -> None:
    xy = np.concatenate([np.asarray(arr, dtype=np.float64).reshape(-1, 2) for arr in arrays if np.asarray(arr).size > 0], axis=0)
    mins = np.min(xy, axis=0)
    maxs = np.max(xy, axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.5 * float(np.max(maxs - mins)), 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")


def _set_3d_equal_axes(ax: Any, arrays: list[np.ndarray]) -> None:
    xyz = np.concatenate([np.asarray(arr, dtype=np.float64).reshape(-1, 3) for arr in arrays if np.asarray(arr).size > 0], axis=0)
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.5 * float(np.max(maxs - mins)), 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _plot_loss_history(
    ax_loss: Any,
    *,
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    val_monitor_losses: np.ndarray,
    nf_likelihood_metadata: dict[str, Any] | None,
) -> None:
    epochs = np.arange(1, int(train_losses.size) + 1, dtype=np.int64)
    ax_loss.plot(epochs, train_losses, color="#4c78a8", linewidth=1.8, label="FM train loss")
    ax_loss.plot(epochs, val_losses, color="#f58518", linewidth=1.8, label="FM validation loss")
    if int(val_monitor_losses.size) == int(train_losses.size):
        ax_loss.plot(epochs, val_monitor_losses, color="#444444", linewidth=1.4, linestyle="--", label="FM validation EMA")
    if nf_likelihood_metadata is not None:
        nf_train = np.asarray(nf_likelihood_metadata["train_nll_losses"], dtype=np.float64)
        nf_val = np.asarray(nf_likelihood_metadata["val_nll_losses"], dtype=np.float64)
        nf_epochs = np.arange(1, int(nf_train.size) + 1, dtype=np.int64)
        ax_nf = ax_loss.twinx()
        ax_nf.plot(nf_epochs, nf_train, color="#b279a2", linewidth=1.2, alpha=0.75, label="NF train NLL")
        ax_nf.plot(nf_epochs, nf_val, color="#b279a2", linewidth=1.5, linestyle=":", label="NF validation NLL")
        ax_nf.set_ylabel("NF NLL")
        ax_nf.legend(frameon=False, loc="lower right", fontsize=8)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("FM loss")
    ax_loss.set_title("Training history")
    ax_loss.set_yscale("log")
    ax_loss.grid(alpha=0.25, linewidth=0.8)
    ax_loss.legend(frameon=False, loc="best", fontsize=8)


def _plot_two_line_overlay(
    *,
    png_path: Path,
    svg_path: Path,
    condition_values: np.ndarray,
    x_plot: np.ndarray,
    theta_plot_scalar: np.ndarray,
    base_curve: np.ndarray,
    base_samples: np.ndarray,
    fitted_curves: list[np.ndarray],
    generated_samples: list[np.ndarray],
    skl_value: float,
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    val_monitor_losses: np.ndarray,
    nf_likelihood_metadata: dict[str, Any] | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#4c78a8", "#f58518"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    ax_target, ax_fit, ax_loss = axes
    all_arrays = [x_plot, base_samples, base_curve, *generated_samples, *fitted_curves]

    for idx, condition_value in enumerate(condition_values[:, 0]):
        mask = np.isclose(theta_plot_scalar[:, 0], float(condition_value))
        color = colors[idx % len(colors)]
        ax_target.scatter(
            x_plot[mask, 0],
            x_plot[mask, 1],
            s=16,
            alpha=0.62,
            color=color,
            linewidths=0,
            label=f"Dataset {idx + 1}",
        )
    ax_target.set_title("Target data")
    ax_target.set_xlabel("x1")
    ax_target.set_ylabel("x2")
    ax_target.legend(frameon=False, loc="best", fontsize=8)
    _set_2d_equal_axes(ax_target, all_arrays)

    ax_fit.scatter(base_samples[:, 0], base_samples[:, 1], s=10, alpha=0.22, color="#2f2f2f", linewidths=0, label="base samples")
    ax_fit.plot(base_curve[:, 0], base_curve[:, 1], color="#2f2f2f", linewidth=1.8, linestyle="--", label="base mean")
    for idx, condition_value in enumerate(condition_values[:, 0]):
        del condition_value
        color = colors[idx % len(colors)]
        gen = np.asarray(generated_samples[idx], dtype=np.float64)
        curve = np.asarray(fitted_curves[idx], dtype=np.float64)
        ax_fit.scatter(gen[:, 0], gen[:, 1], s=13, alpha=0.36, color=color, linewidths=0, label=f"fitted samples {idx + 1}")
        ax_fit.plot(curve[:, 0], curve[:, 1], color=color, linewidth=2.5, label=f"fitted mean {idx + 1}")
    ax_fit.text(0.02, 0.98, f"SKL = {skl_value:.4g}", transform=ax_fit.transAxes, va="top", ha="left", fontsize=12)
    ax_fit.set_title("Fitted distribution")
    ax_fit.set_xlabel("x1")
    ax_fit.set_ylabel("x2")
    ax_fit.legend(frameon=False, loc="best", fontsize=7)
    _set_2d_equal_axes(ax_fit, all_arrays)

    _plot_loss_history(
        ax_loss,
        train_losses=train_losses,
        val_losses=val_losses,
        val_monitor_losses=val_monitor_losses,
        nf_likelihood_metadata=nf_likelihood_metadata,
    )

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)


def _plot_overlay(
    *,
    png_path: Path,
    svg_path: Path,
    dataset: str,
    velocity_family: str,
    condition_values: np.ndarray,
    condition_labels: list[str],
    x_plot: np.ndarray,
    theta_plot_scalar: np.ndarray,
    base_curve: np.ndarray,
    base_samples: np.ndarray,
    target_curves: list[np.ndarray],
    fitted_curves: list[np.ndarray],
    generated_samples: list[np.ndarray],
    skl_value: float,
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    val_monitor_losses: np.ndarray,
    nf_likelihood_metadata: dict[str, Any] | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dim = int(x_plot.shape[1])
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    fig = plt.figure(figsize=(12.4, 5.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d") if dim == 3 else fig.add_subplot(1, 2, 1)
    ax_loss = fig.add_subplot(1, 2, 2)
    all_arrays = [base_samples, base_curve, x_plot]

    if dim == 3:
        ax.scatter(base_samples[:, 0], base_samples[:, 1], base_samples[:, 2], s=10, alpha=0.22, color="#2f2f2f", label="base samples")
    else:
        ax.scatter(base_samples[:, 0], base_samples[:, 1], s=10, alpha=0.22, color="#2f2f2f", label="base samples")

    for idx, condition_value in enumerate(condition_values[:, 0]):
        mask = np.isclose(theta_plot_scalar[:, 0], float(condition_value))
        color = colors[idx % len(colors)]
        label = condition_labels[idx] if idx < len(condition_labels) else f"condition {idx + 1}"
        target = np.asarray(target_curves[idx], dtype=np.float64)
        curve = np.asarray(fitted_curves[idx], dtype=np.float64)
        gen = np.asarray(generated_samples[idx], dtype=np.float64)
        if dim == 3:
            ax.scatter(x_plot[mask, 0], x_plot[mask, 1], x_plot[mask, 2], s=14, alpha=0.50, color=color, linewidths=0, label=f"Dataset {idx + 1} ({label})")
            ax.plot(target[:, 0], target[:, 1], target[:, 2], color=color, linewidth=1.2, linestyle=":", label=f"target {idx + 1}")
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=color, linewidth=2.4, label=f"fitted {idx + 1}")
            ax.scatter(gen[:, 0], gen[:, 1], gen[:, 2], s=12, alpha=0.35, marker="x", color=color, linewidths=0.8, label=f"generated {idx + 1}")
        else:
            ax.scatter(x_plot[mask, 0], x_plot[mask, 1], s=14, alpha=0.50, color=color, linewidths=0, label=f"Dataset {idx + 1} ({label})")
            ax.plot(target[:, 0], target[:, 1], color=color, linewidth=1.2, linestyle=":", label=f"target {idx + 1}")
            ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=2.4, label=f"fitted {idx + 1}")
            ax.scatter(gen[:, 0], gen[:, 1], s=12, alpha=0.35, marker="x", color=color, linewidths=0.8, label=f"generated {idx + 1}")
        all_arrays.extend([target, curve, gen])

    if dim == 3:
        ax.plot(base_curve[:, 0], base_curve[:, 1], base_curve[:, 2], color="#2f2f2f", linewidth=2.0, linestyle="--", label="base geometry")
        ax.text2D(0.02, 0.98, f"SKL = {skl_value:.4g}", transform=ax.transAxes, va="top", ha="left", fontsize=13)
        _set_3d_equal_axes(ax, all_arrays)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.view_init(elev=22.0, azim=-58.0)
    else:
        ax.plot(base_curve[:, 0], base_curve[:, 1], color="#2f2f2f", linewidth=2.0, linestyle="--", label="base geometry")
        ax.text(0.02, 0.98, f"SKL = {skl_value:.4g}", transform=ax.transAxes, va="top", ha="left", fontsize=13)
        _set_2d_equal_axes(ax, all_arrays)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
    ax.set_title(f"{dataset} with {velocity_family}")
    ax.legend(frameon=False, loc="best", fontsize=7)

    _plot_loss_history(
        ax_loss,
        train_losses=train_losses,
        val_losses=val_losses,
        val_monitor_losses=val_monitor_losses,
        nf_likelihood_metadata=nf_likelihood_metadata,
    )

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    velocity_family = validate_dataset_velocity(args)
    dev = require_device(str(args.device))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    data = make_geometric_dataset(args)
    base = make_base(args, dataset_kind=str(data["dataset_kind"]))
    condition_eval = np.asarray(data["condition_eval"], dtype=np.float64)
    theta_train = np.asarray(data["theta_train"], dtype=np.float64)
    x_train = np.asarray(data["x_train"], dtype=np.float64)
    theta_val = np.asarray(data["theta_val"], dtype=np.float64)
    x_val = np.asarray(data["x_val"], dtype=np.float64)
    paths = resolve_output_paths(args.output_dir, dataset=str(args.dataset), velocity_family=velocity_family, nf_likelihood=bool(args.nf_likelihood_finetune))
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    generated_sample_count = int(args.generated_samples_per_condition)
    if generated_sample_count < 1:
        raise ValueError("--generated-samples-per-condition must be >= 1.")
    model = build_geometric_base_velocity_model(
        velocity_family=velocity_family,
        theta_dim=int(condition_eval.shape[1]),
        x_dim=int(data["ambient_dim"]),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        path_schedule=str(args.path_schedule),
    ).to(dev)
    train_meta = train_geometric_base_affine_flow(
        model=model,
        base=base,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        device=dev,
        path_schedule=str(args.path_schedule),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        t_eps=float(args.t_eps),
        patience=int(args.early_patience),
        min_delta=float(args.early_min_delta),
        ema_alpha=float(args.early_ema_alpha),
        max_grad_norm=float(args.max_grad_norm),
        log_every=max(1, int(args.log_every)),
    )
    nf_likelihood_meta = None
    if bool(args.nf_likelihood_finetune):
        nf_batch_size = int(args.nf_batch_size) if int(args.nf_batch_size) > 0 else int(args.batch_size)
        nf_likelihood_meta = finetune_geometric_base_cnf_likelihood(
            model=model,
            base=base,
            theta_train=theta_train,
            x_train=x_train,
            theta_val=theta_val,
            x_val=x_val,
            condition_eval=condition_eval,
            device=dev,
            epochs=int(args.nf_epochs),
            batch_size=nf_batch_size,
            lr=float(args.nf_lr),
            weight_decay=float(args.nf_weight_decay),
            density_points=int(args.nf_density_points),
            ode_steps=int(args.ode_steps),
            ode_method=str(args.ode_method),
            checkpoint_selection=str(args.nf_checkpoint_selection),
            learn_base_noise=bool(args.nf_learn_base_noise),
            sigma_min=float(args.nf_sigma_min),
            log_every=max(1, int(args.log_every)),
        )

    result = estimate_smoothed_curve_symmetric_kl(
        model=model,
        base=base,
        theta_all=condition_eval,
        device=dev,
        smooth_sigma=float(args.smooth_sigma),
        mc_skl_samples=int(args.mc_skl_samples),
        density_mc_samples=int(args.density_mc_samples),
        ode_steps=int(args.ode_steps),
        ode_method=str(args.ode_method),
        batch_size=int(args.batch_size),
        train_metadata=train_meta,
    )

    curve_u = torch.linspace(base.u_low, base.u_high, int(args.curve_points), dtype=torch.float32).reshape(-1, 1)
    if str(data["dataset_kind"]) == "square":
        curve_u = torch.linspace(base.u_low, base.u_high, 4 * int(args.curve_points_per_edge) + 1, dtype=torch.float32).reshape(-1, 1)
    base_curve = base.points_from_u(curve_u).detach().cpu().numpy().astype(np.float64)
    base_samples_t = base.sample(generated_sample_count, device=dev, dtype=torch.float32)
    base_samples = base_samples_t.detach().cpu().numpy().astype(np.float64)
    fitted_curves: list[np.ndarray] = []
    generated_samples: list[np.ndarray] = []
    selected_base_sigmas = None
    if nf_likelihood_meta is not None and "selected_base_noise_sigmas" in nf_likelihood_meta:
        selected_base_sigmas = np.asarray(nf_likelihood_meta["selected_base_noise_sigmas"], dtype=np.float64).reshape(-1)
    for condition_idx, theta_row in enumerate(condition_eval):
        curve, _ = push_base_curve(
            model=model,
            base=base,
            theta=theta_row.reshape(1, -1),
            device=dev,
            u=curve_u,
            ode_steps=int(args.ode_steps),
            ode_method=str(args.ode_method),
        )
        fitted_curves.append(curve.detach().cpu().numpy().astype(np.float64))
        if selected_base_sigmas is not None:
            u_gen = base.sample_u(generated_sample_count, device=dev, dtype=torch.float32)
            x0_gen = base.points_from_u(u_gen)
            sigma_gen = float(selected_base_sigmas[int(condition_idx)])
            if sigma_gen > 0.0:
                x0_gen = x0_gen + sigma_gen * torch.randn_like(x0_gen)
        else:
            x0_gen = base_samples_t
        pushed = push_initial_points(
            model=model,
            x0=x0_gen,
            theta=theta_row.reshape(1, -1),
            device=dev,
            ode_steps=int(args.ode_steps),
            ode_method=str(args.ode_method),
        )
        generated_samples.append(pushed.detach().cpu().numpy().astype(np.float64))
    skl_value = float(result.symmetric_kl_matrix[0, 1]) if int(condition_eval.shape[0]) > 1 else 0.0
    plot_common = {
        "png_path": paths["png"],
        "svg_path": paths["svg"],
        "condition_values": np.asarray(data["condition_values"], dtype=np.float64),
        "x_plot": np.asarray(data["x_test_plot"], dtype=np.float64),
        "theta_plot_scalar": np.asarray(data["theta_test_plot_scalar"], dtype=np.float64),
        "base_curve": base_curve,
        "base_samples": base_samples,
        "fitted_curves": fitted_curves,
        "generated_samples": generated_samples,
        "skl_value": skl_value,
        "train_losses": np.asarray(train_meta["train_losses"], dtype=np.float64),
        "val_losses": np.asarray(train_meta["val_losses"], dtype=np.float64),
        "val_monitor_losses": np.asarray(train_meta["val_monitor_losses"], dtype=np.float64),
        "nf_likelihood_metadata": nf_likelihood_meta,
    }
    if str(args.dataset) == "two-line":
        _plot_two_line_overlay(**plot_common)
    else:
        _plot_overlay(
            dataset=str(args.dataset),
            velocity_family=velocity_family,
            condition_labels=list(data["condition_labels"]),
            target_curves=[np.asarray(arr, dtype=np.float64) for arr in data["target_curves"]],
            **plot_common,
        )

    training_parameters = {
        "dataset": str(args.dataset),
        "dataset_kind": str(data["dataset_kind"]),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "velocity_family": velocity_family,
        "path_schedule": str(args.path_schedule),
        "t_eps": float(args.t_eps),
        "early_patience": int(args.early_patience),
        "early_min_delta": float(args.early_min_delta),
        "early_ema_alpha": float(args.early_ema_alpha),
        "max_grad_norm": float(args.max_grad_norm),
        "n_per_condition": int(args.n_per_condition),
        "train_frac": float(args.train_frac),
        "val_frac": float(args.val_frac),
        "condition_values": np.asarray(data["condition_values"], dtype=np.float64).reshape(-1),
        "theta_encoding": "one_hot",
        "condition_eval": condition_eval,
        "target_sigma": float(data["target_sigma"]),
        "base_noise_sigma": float(args.base_noise_sigma),
        "smooth_sigma": float(args.smooth_sigma),
        "mc_skl_samples": int(args.mc_skl_samples),
        "density_mc_samples": int(args.density_mc_samples),
        "ode_steps": int(args.ode_steps),
        "ode_method": str(args.ode_method),
        "curve_points": int(args.curve_points),
        "curve_points_per_edge": int(args.curve_points_per_edge),
        "generated_samples_per_condition": generated_sample_count,
        "nf_likelihood_finetune": bool(args.nf_likelihood_finetune),
        "nf_epochs": int(args.nf_epochs),
        "nf_batch_size": int(args.nf_batch_size) if int(args.nf_batch_size) > 0 else int(args.batch_size),
        "nf_lr": float(args.nf_lr),
        "nf_weight_decay": float(args.nf_weight_decay),
        "nf_density_points": int(args.nf_density_points),
        "nf_checkpoint_selection": str(args.nf_checkpoint_selection),
        "nf_learn_base_noise": bool(args.nf_learn_base_noise),
        "nf_sigma_min": float(args.nf_sigma_min),
    }
    summary = {
        "script": "bin/run_geometric_base_fit_check.py",
        "device": str(dev),
        "dataset": str(args.dataset),
        "velocity_family": velocity_family,
        "condition_values": np.asarray(data["condition_values"], dtype=np.float64).reshape(-1),
        "theta_encoding": "one_hot",
        "condition_eval": condition_eval,
        "training_parameters": training_parameters,
        "split_counts": data["split_counts"],
        "train_shape": list(theta_train.shape),
        "validation_shape": list(theta_val.shape),
        "test_shape": list(np.asarray(data["theta_test"], dtype=np.float64).shape),
        "target_sigma": float(data["target_sigma"]),
        "symmetric_kl_matrix": result.symmetric_kl_matrix,
        "skl_value": skl_value,
        "best_epoch": int(train_meta["best_epoch"]),
        "best_val_loss": float(train_meta["best_val_loss"]),
        "stopped_epoch": int(train_meta["stopped_epoch"]),
        "stopped_early": bool(train_meta["stopped_early"]),
        "nf_likelihood_finetune_metadata": nf_likelihood_meta,
        "base_samples_shape": list(base_samples.shape),
        "generated_sample_shapes": [list(arr.shape) for arr in generated_samples],
        "png": paths["png"],
        "svg": paths["svg"],
        "summary": paths["summary"],
    }
    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"png: {paths['png']}", flush=True)
    print(f"svg: {paths['svg']}", flush=True)
    print(f"summary_json: {paths['summary']}", flush=True)
    print(f"skl: {skl_value:.12g}", flush=True)
    print(f"best_epoch: {int(train_meta['best_epoch'])}", flush=True)
    print(f"best_val_loss: {float(train_meta['best_val_loss']):.12g}", flush=True)
    if nf_likelihood_meta is not None:
        print(
            f"nf_selected_epoch: {int(nf_likelihood_meta['selected_epoch'])} "
            f"nf_selected_val_nll: {float(nf_likelihood_meta['selected_val_nll']):.12g}",
            flush=True,
        )
        if "selected_base_noise_sigmas" in nf_likelihood_meta:
            sigmas = np.asarray(nf_likelihood_meta["selected_base_noise_sigmas"], dtype=np.float64)
            print(f"nf_selected_base_noise_sigmas: {np.array2string(sigmas, precision=6, separator=',')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
