"""Two-condition hidden nonlinear shear dataset for flow-SKL rank experiments."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fisher.shared_dataset_io import SHARED_DATASET_NPZ_VERSION, SharedDatasetBundle


@dataclass(frozen=True)
class ShearRankDataset:
    """Generated shared dataset plus analytic hidden-shear metadata."""

    bundle: SharedDatasetBundle
    q_matrix: np.ndarray
    u_star: np.ndarray
    condition_shear_a: np.ndarray
    condition_mean_offsets: np.ndarray
    mean_shift_dim: int
    nu: float
    true_skl_matrix: np.ndarray


def centered_cosine_nu(omega: float) -> float:
    """Variance of ``cos(omega Z) - E cos(omega Z)`` for ``Z ~ N(0, 1)``."""

    w = float(omega)
    return float(0.5 * (1.0 + math.exp(-2.0 * w * w)) - math.exp(-w * w))


def centered_cosine_feature(z: np.ndarray, *, omega: float) -> np.ndarray:
    """Evaluate the centered cosine feature used by the shear construction."""

    w = float(omega)
    return np.cos(w * np.asarray(z, dtype=np.float64)) - math.exp(-0.5 * w * w)


def random_orthogonal_matrix(dim: int, *, seed: int) -> np.ndarray:
    """Generate a deterministic Haar-like orthogonal matrix with stable column signs."""

    d = int(dim)
    if d < 1:
        raise ValueError("dim must be >= 1.")
    rng = np.random.default_rng(int(seed))
    raw = rng.standard_normal(size=(d, d))
    q, r = np.linalg.qr(raw)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q = q * signs.reshape(1, d)
    return q.astype(np.float64, copy=False)


def shear_kl(a: float, b: float, *, nu: float) -> float:
    """Analytic directional KL for one nonlinear shear pair."""

    aa = float(a)
    bb = float(b)
    vv = float(nu)
    sa = math.sqrt(1.0 + aa * aa * vv)
    sb = math.sqrt(1.0 + bb * bb * vv)
    ratio = sb / sa
    return float(math.log(sa / sb) + 0.5 * (ratio * ratio + vv * (aa * ratio - bb) ** 2 - 1.0))


def shear_symmetric_kl(a_values: np.ndarray, b_values: np.ndarray, *, nu: float) -> float:
    """Analytic Jeffreys SKL for independent shear pairs."""

    a = np.asarray(a_values, dtype=np.float64).reshape(-1)
    b = np.asarray(b_values, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("a_values and b_values must have the same shape.")
    total = 0.0
    for ai, bi in zip(a, b, strict=True):
        total += shear_kl(float(ai), float(bi), nu=float(nu))
        total += shear_kl(float(bi), float(ai), nu=float(nu))
    return float(total)


def _validate_config(*, x_dim: int, r_star: int, n_per_condition: int, train_frac: float) -> None:
    if int(x_dim) < 2:
        raise ValueError("x_dim must be >= 2.")
    if int(r_star) != 0 and (int(r_star) < 2 or int(r_star) > int(x_dim) or int(r_star) % 2):
        raise ValueError("r_star must be 0 or an even integer in [2, x_dim].")
    if int(n_per_condition) < 2:
        raise ValueError("n_per_condition must be >= 2.")
    if not (0.0 < float(train_frac) < 1.0):
        raise ValueError("train_frac must be in (0, 1).")


def _condition_mean_offsets(*, x_dim: int, r_star: int, mean_shift: float) -> tuple[np.ndarray, int]:
    d = int(x_dim)
    r = int(r_star)
    shift = float(mean_shift)
    offsets = np.zeros((2, d), dtype=np.float64)
    if shift == 0.0:
        return offsets, -1
    if r >= d:
        raise ValueError("mean_shift requires at least one non-shear dimension: x_dim must be > r_star.")
    shift_dim = r
    offsets[0, shift_dim] = -0.5 * shift
    offsets[1, shift_dim] = 0.5 * shift
    return offsets, shift_dim


def _condition_a_values(*, mode: str, amplitude: float, q_pairs: int) -> np.ndarray:
    mode_norm = str(mode).strip().lower().replace("-", "_")
    amp = float(amplitude)
    if mode_norm == "sign_flip":
        vals = np.vstack(
            [
                np.full(int(q_pairs), -amp, dtype=np.float64),
                np.full(int(q_pairs), amp, dtype=np.float64),
            ]
        )
    elif mode_norm == "null":
        vals = np.vstack(
            [
                np.full(int(q_pairs), amp, dtype=np.float64),
                np.full(int(q_pairs), amp, dtype=np.float64),
            ]
        )
    else:
        raise ValueError("mode must be one of: sign_flip, null.")
    return vals


def generate_shear_rank_dataset(
    *,
    n_per_condition: int,
    x_dim: int = 64,
    r_star: int = 8,
    amplitude: float = 0.7,
    mean_shift: float = 0.0,
    omega: float = 2.5,
    seed: int = 7,
    q_seed: int | None = None,
    train_frac: float = 0.8,
    mode: str = "sign_flip",
) -> ShearRankDataset:
    """Generate the two-condition hidden nonlinear shear dataset.

    ``n_per_condition`` is the total number of rows per condition before the
    stratified train/validation split.
    """

    _validate_config(
        x_dim=int(x_dim),
        r_star=int(r_star),
        n_per_condition=int(n_per_condition),
        train_frac=float(train_frac),
    )
    d = int(x_dim)
    r = int(r_star)
    q_pairs = r // 2
    n = int(n_per_condition)
    rng = np.random.default_rng(int(seed))
    q_matrix = random_orthogonal_matrix(d, seed=int(seed if q_seed is None else q_seed))
    nu = centered_cosine_nu(float(omega))
    a_by_condition = _condition_a_values(mode=mode, amplitude=float(amplitude), q_pairs=q_pairs)
    mean_offsets, mean_shift_dim = _condition_mean_offsets(
        x_dim=d,
        r_star=r,
        mean_shift=float(mean_shift),
    )

    x_parts: list[np.ndarray] = []
    theta_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for condition in range(2):
        z = rng.standard_normal(size=(n, d)).astype(np.float64, copy=False)
        y = z.copy()
        for pair_idx in range(q_pairs):
            even_col = 2 * pair_idx + 1
            odd_col = 2 * pair_idx
            a = float(a_by_condition[condition, pair_idx])
            scale = math.sqrt(1.0 + a * a * nu)
            y[:, even_col] = z[:, even_col]
            y[:, odd_col] = (
                z[:, odd_col] + a * centered_cosine_feature(z[:, even_col], omega=float(omega))
            ) / scale
        y += mean_offsets[condition].reshape(1, d)
        x_parts.append(y @ q_matrix.T)
        theta_parts.append(np.eye(2, dtype=np.float64)[np.full(n, condition, dtype=np.int64)])
        label_parts.append(np.full(n, condition, dtype=np.int64))

    x_all = np.vstack(x_parts).astype(np.float64, copy=False)
    theta_all = np.vstack(theta_parts).astype(np.float64, copy=False)
    labels = np.concatenate(label_parts).astype(np.int64, copy=False)

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for condition in range(2):
        idx = np.flatnonzero(labels == condition)
        rng.shuffle(idx)
        n_train = int(math.floor(float(train_frac) * float(idx.size)))
        n_train = min(max(n_train, 1), int(idx.size) - 1)
        train_parts.append(idx[:n_train])
        val_parts.append(idx[n_train:])
    train_idx = np.concatenate(train_parts).astype(np.int64, copy=False)
    validation_idx = np.concatenate(val_parts).astype(np.int64, copy=False)
    rng.shuffle(train_idx)
    rng.shuffle(validation_idx)

    true_skl = np.zeros((2, 2), dtype=np.float64)
    true_skl[0, 1] = true_skl[1, 0] = shear_symmetric_kl(
        a_by_condition[0],
        a_by_condition[1],
        nu=nu,
    ) + float(mean_shift) * float(mean_shift)
    meta: dict[str, Any] = {
        "version": SHARED_DATASET_NPZ_VERSION,
        "dataset_family": "two_condition_hidden_shear_rank",
        "theta_type": "categorical",
        "theta_encoding": "one_hot",
        "theta_dim": 2,
        "num_categories": 2,
        "x_dim": d,
        "n_total": int(2 * n),
        "n_per_condition": n,
        "train_frac": float(train_frac),
        "seed": int(seed),
        "q_seed": int(seed if q_seed is None else q_seed),
        "r_star": r,
        "q_pairs": q_pairs,
        "amplitude": float(amplitude),
        "mean_shift": float(mean_shift),
        "mean_shift_dim": int(mean_shift_dim),
        "omega": float(omega),
        "nu": float(nu),
        "mode": str(mode).strip().lower().replace("-", "_"),
        "true_skl": float(true_skl[0, 1]),
    }
    bundle = SharedDatasetBundle(
        meta=meta,
        theta_all=theta_all,
        x_all=x_all,
        train_idx=train_idx,
        validation_idx=validation_idx,
        theta_train=theta_all[train_idx],
        x_train=x_all[train_idx],
        theta_validation=theta_all[validation_idx],
        x_validation=x_all[validation_idx],
    )
    return ShearRankDataset(
        bundle=bundle,
        q_matrix=q_matrix,
        u_star=q_matrix[:, :r].copy(),
        condition_shear_a=a_by_condition,
        condition_mean_offsets=mean_offsets,
        mean_shift_dim=int(mean_shift_dim),
        nu=float(nu),
        true_skl_matrix=true_skl,
    )


def save_shear_rank_dataset_npz(path: str | Path, dataset: ShearRankDataset) -> Path:
    """Save a shear-rank dataset in shared NPZ format with extra truth arrays."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    bundle = dataset.bundle
    meta = dict(bundle.meta)
    meta["version"] = int(SHARED_DATASET_NPZ_VERSION)
    meta_utf8 = json.dumps(meta, sort_keys=True).encode("utf-8")
    np.savez_compressed(
        out,
        meta_json_utf8=np.frombuffer(meta_utf8, dtype=np.uint8),
        theta_all=np.asarray(bundle.theta_all, dtype=np.float64),
        x_all=np.asarray(bundle.x_all, dtype=np.float64),
        train_idx=np.asarray(bundle.train_idx, dtype=np.int64),
        validation_idx=np.asarray(bundle.validation_idx, dtype=np.int64),
        theta_train=np.asarray(bundle.theta_train, dtype=np.float64),
        x_train=np.asarray(bundle.x_train, dtype=np.float64),
        theta_validation=np.asarray(bundle.theta_validation, dtype=np.float64),
        x_validation=np.asarray(bundle.x_validation, dtype=np.float64),
        orthogonal_Q=np.asarray(dataset.q_matrix, dtype=np.float64),
        u_star=np.asarray(dataset.u_star, dtype=np.float64),
        condition_shear_a=np.asarray(dataset.condition_shear_a, dtype=np.float64),
        condition_mean_offsets=np.asarray(dataset.condition_mean_offsets, dtype=np.float64),
        mean_shift_dim=np.asarray([dataset.mean_shift_dim], dtype=np.int64),
        nu=np.asarray([dataset.nu], dtype=np.float64),
        true_skl_matrix=np.asarray(dataset.true_skl_matrix, dtype=np.float64),
    )
    return out
