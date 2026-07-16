"""CTSM-v estimators for pairwise Jeffreys divergence."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from global_setting import TRAINING_EARLY_STOPPING_PATIENCE, TRAINING_MAX_EPOCHS
from fisher.ctsm_models import (
    ToyBinaryTimeScoreNet,
    ToyPairConditionedTimeScoreNet,
    ToyPairConditionedTimeScoreNetFiLM,
)
from fisher.ctsm_objectives import estimate_log_ratio_trapz_pair
from fisher.shared_fisher_est import (
    estimate_binary_ctsm_v_log_ratio,
    train_binary_ctsm_v_model,
    train_pair_conditioned_ctsm_v_model,
)


@dataclass(frozen=True)
class CTSMVJeffreysConfig:
    epochs: int = TRAINING_MAX_EPOCHS
    batch_size: int = 512
    lr: float = 2e-3
    weight_decay: float = 0.0
    hidden_dim: int = 256
    architecture: str = "film"
    film_depth: int = 3
    gated_film: bool = False
    raw_time: bool = False
    m_scale: float = 1.0
    delta_scale: float = 0.5
    two_sb_var: float = 2.0
    path_schedule: str = "linear"
    path_eps: float = 1e-12
    factor: float = 1.0
    t_eps: float = 1e-4
    integration_steps: int = 300
    eval_batch_size: int = 4096
    early_patience: int = TRAINING_EARLY_STOPPING_PATIENCE
    early_min_delta: float = 1e-4
    early_ema_alpha: float = 0.05
    validation_batches_per_epoch: int = 8
    restore_best: bool = True
    normalize_x: bool = False
    normalize_x_eps: float = 1e-8
    log_every: int = 50


@dataclass
class CTSMVJeffreysResult:
    symmetric_kl_matrix: np.ndarray
    raw_symmetric_kl_matrix: np.ndarray
    directed_kl_matrix: np.ndarray
    condition_theta: np.ndarray
    train_metadata: dict[str, Any]


@dataclass(frozen=True)
class CTSMVBinaryJeffreysConfig:
    epochs: int = TRAINING_MAX_EPOCHS
    batch_size: int = 512
    lr: float = 2e-3
    weight_decay: float = 0.0
    hidden_dim: int = 256
    two_sb_var: float = 2.0
    path_schedule: str = "linear"
    path_eps: float = 1e-12
    factor: float = 1.0
    t_eps: float = 1e-4
    integration_steps: int = 300
    eval_batch_size: int = 4096
    early_patience: int = TRAINING_EARLY_STOPPING_PATIENCE
    early_min_delta: float = 1e-4
    early_ema_alpha: float = 0.05
    validation_batches_per_epoch: int = 8
    restore_best: bool = True
    normalize_x: bool = False
    normalize_x_eps: float = 1e-8
    log_every: int = 50


@dataclass
class CTSMVBinaryJeffreysResult:
    symmetric_kl_matrix: np.ndarray
    raw_symmetric_kl_matrix: np.ndarray
    directed_kl_matrix: np.ndarray
    pair_metadata: dict[str, dict[str, Any]]
    run_metadata: dict[str, Any]


def _as_2d_float32(value: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape [N, D].")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _validate_labels(labels: np.ndarray, *, n_rows: int, num_categories: int) -> np.ndarray:
    out = np.asarray(labels, dtype=np.int64).reshape(-1)
    if out.shape[0] != int(n_rows):
        raise ValueError("labels must contain one entry per evaluation row.")
    if np.any((out < 0) | (out >= int(num_categories))):
        raise ValueError(f"labels must be in [0, {int(num_categories) - 1}].")
    for condition in range(int(num_categories)):
        if not np.any(out == condition):
            raise ValueError(f"Condition {condition} has no evaluation samples.")
    return out


def _condition_theta(theta: np.ndarray, labels: np.ndarray, *, num_categories: int) -> np.ndarray:
    theta_arr = _as_2d_float32(theta, name="theta_eval")
    if theta_arr.shape[0] != labels.shape[0]:
        raise ValueError("theta_eval and labels must have the same number of rows.")
    endpoints = np.stack(
        [np.mean(theta_arr[labels == condition], axis=0) for condition in range(int(num_categories))],
        axis=0,
    ).astype(np.float32, copy=False)
    for condition in range(int(num_categories)):
        rows = theta_arr[labels == condition]
        if not np.allclose(rows, endpoints[condition], rtol=0.0, atol=1e-6):
            raise ValueError(f"theta_eval is not constant within condition {condition}.")
    return endpoints


def _fit_normalizer(x_train: np.ndarray, *, eps: float) -> tuple[np.ndarray, np.ndarray]:
    threshold = float(eps)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("normalize_x_eps must be finite and positive.")
    mean = np.mean(x_train, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(x_train, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < threshold, 1.0, std).astype(np.float32, copy=False)
    return mean, std


def _build_model(*, x_dim: int, theta_dim: int, config: CTSMVJeffreysConfig) -> torch.nn.Module:
    architecture = str(config.architecture).strip().lower()
    if architecture == "film":
        return ToyPairConditionedTimeScoreNetFiLM(
            dim=int(x_dim),
            hidden_dim=int(config.hidden_dim),
            depth=int(config.film_depth),
            theta_dim=int(theta_dim),
            m_scale=float(config.m_scale),
            delta_scale=float(config.delta_scale),
            use_logit_time=not bool(config.raw_time),
            gated_film=bool(config.gated_film),
        )
    if architecture == "mlp":
        return ToyPairConditionedTimeScoreNet(
            dim=int(x_dim),
            hidden_dim=int(config.hidden_dim),
            theta_dim=int(theta_dim),
            m_scale=float(config.m_scale),
            delta_scale=float(config.delta_scale),
        )
    raise ValueError("CTSM-v architecture must be 'mlp' or 'film'.")


@torch.no_grad()
def _estimate_log_ratio_batched(
    model: torch.nn.Module,
    x: np.ndarray,
    *,
    theta_a: np.ndarray,
    theta_b: np.ndarray,
    device: torch.device,
    batch_size: int,
    t_eps: float,
    integration_steps: int,
) -> np.ndarray:
    x_arr = _as_2d_float32(x, name="x_eval_pair")
    if int(batch_size) < 1:
        raise ValueError("CTSM-v eval_batch_size must be >= 1.")
    values: list[np.ndarray] = []
    model.eval()
    for start in range(0, int(x_arr.shape[0]), int(batch_size)):
        xb = torch.from_numpy(x_arr[start : start + int(batch_size)]).to(device)
        b = int(xb.shape[0])
        a_t = torch.from_numpy(np.broadcast_to(theta_a, (b, theta_a.size)).copy()).to(device)
        b_t = torch.from_numpy(np.broadcast_to(theta_b, (b, theta_b.size)).copy()).to(device)
        estimate = estimate_log_ratio_trapz_pair(
            model,
            xb,
            a_t,
            b_t,
            eps1=float(t_eps),
            eps2=float(t_eps),
            n_time=int(integration_steps),
        )
        values.append(estimate.detach().cpu().numpy().astype(np.float64, copy=False))
    return np.concatenate(values, axis=0)


def train_and_estimate_pairwise_binary_ctsm_v_jeffreys(
    *,
    x_train: np.ndarray,
    labels_train: np.ndarray,
    x_val: np.ndarray,
    labels_val: np.ndarray,
    x_eval: np.ndarray,
    labels_eval: np.ndarray,
    num_categories: int,
    device: torch.device,
    seed: int,
    config: CTSMVBinaryJeffreysConfig,
) -> tuple[dict[str, dict[str, torch.Tensor]], CTSMVBinaryJeffreysResult]:
    """Fit one binary CTSM-v model per condition pair and estimate Jeffreys divergence."""

    x_train_arr = _as_2d_float32(x_train, name="x_train")
    x_val_arr = _as_2d_float32(x_val, name="x_val")
    x_eval_arr = _as_2d_float32(x_eval, name="x_eval")
    if x_train_arr.shape[1] != x_val_arr.shape[1] or x_train_arr.shape[1] != x_eval_arr.shape[1]:
        raise ValueError("CTSM-v-binary train, validation, and evaluation x dimensions must match.")
    labels_train_arr = _validate_labels(
        labels_train,
        n_rows=x_train_arr.shape[0],
        num_categories=int(num_categories),
    )
    labels_val_arr = _validate_labels(
        labels_val,
        n_rows=x_val_arr.shape[0],
        num_categories=int(num_categories),
    )
    labels_eval_arr = _validate_labels(
        labels_eval,
        n_rows=x_eval_arr.shape[0],
        num_categories=int(num_categories),
    )

    normalization: dict[str, Any] = {"normalize_x": bool(config.normalize_x)}
    if bool(config.normalize_x):
        x_mean, x_std = _fit_normalizer(x_train_arr, eps=float(config.normalize_x_eps))
        x_train_arr = (x_train_arr - x_mean) / x_std
        x_val_arr = (x_val_arr - x_mean) / x_std
        x_eval_arr = (x_eval_arr - x_mean) / x_std
        normalization.update({"normalize_x_mean": x_mean, "normalize_x_std": x_std})

    directed = np.zeros((int(num_categories), int(num_categories)), dtype=np.float64)
    pair_metadata: dict[str, dict[str, Any]] = {}
    pair_state_dicts: dict[str, dict[str, torch.Tensor]] = {}
    for i in range(int(num_categories)):
        for j in range(i + 1, int(num_categories)):
            pair_key = f"{i}_{j}"
            pair_seed = int(seed) + 10_007 * i + 1_009 * j
            torch.manual_seed(pair_seed)
            np.random.seed(pair_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(pair_seed)
            model = ToyBinaryTimeScoreNet(
                dim=int(x_train_arr.shape[1]),
                hidden_dim=int(config.hidden_dim),
            ).to(device)
            n_train_i = int(np.sum(labels_train_arr == i))
            n_train_j = int(np.sum(labels_train_arr == j))
            n_val_i = int(np.sum(labels_val_arr == i))
            n_val_j = int(np.sum(labels_val_arr == j))
            print(
                f"[ctsm_v_binary_pairwise] pair=({i},{j}) "
                f"train=({n_train_i},{n_train_j}) val=({n_val_i},{n_val_j})",
                flush=True,
            )
            metadata = train_binary_ctsm_v_model(
                model=model,
                x0_train=x_train_arr[labels_train_arr == i],
                x1_train=x_train_arr[labels_train_arr == j],
                epochs=int(config.epochs),
                batch_size=int(config.batch_size),
                lr=float(config.lr),
                weight_decay=float(config.weight_decay),
                device=device,
                log_every=max(1, int(config.log_every)),
                two_sb_var=float(config.two_sb_var),
                path_schedule=str(config.path_schedule),
                path_eps=float(config.path_eps),
                factor=float(config.factor),
                t_eps=float(config.t_eps),
                x0_val=x_val_arr[labels_val_arr == i],
                x1_val=x_val_arr[labels_val_arr == j],
                early_stopping_patience=int(config.early_patience),
                early_stopping_min_delta=float(config.early_min_delta),
                early_stopping_ema_alpha=float(config.early_ema_alpha),
                restore_best=bool(config.restore_best),
                val_batches_per_epoch=int(config.validation_batches_per_epoch),
            )
            xi = x_eval_arr[labels_eval_arr == i]
            xj = x_eval_arr[labels_eval_arr == j]
            log_pj_minus_pi_i = estimate_binary_ctsm_v_log_ratio(
                model,
                xi,
                device=device,
                batch_size=int(config.eval_batch_size),
                eps1=float(config.t_eps),
                eps2=float(config.t_eps),
                n_time=int(config.integration_steps),
            )
            log_pj_minus_pi_j = estimate_binary_ctsm_v_log_ratio(
                model,
                xj,
                device=device,
                batch_size=int(config.eval_batch_size),
                eps1=float(config.t_eps),
                eps2=float(config.t_eps),
                n_time=int(config.integration_steps),
            )
            directed[i, j] = -float(np.mean(log_pj_minus_pi_i, dtype=np.float64))
            directed[j, i] = float(np.mean(log_pj_minus_pi_j, dtype=np.float64))
            pair_metadata[pair_key] = {
                **metadata,
                "condition_i": int(i),
                "condition_j": int(j),
                "seed": pair_seed,
                "n_train_i": n_train_i,
                "n_train_j": n_train_j,
                "n_val_i": n_val_i,
                "n_val_j": n_val_j,
                "n_eval_i": int(xi.shape[0]),
                "n_eval_j": int(xj.shape[0]),
            }
            pair_state_dicts[pair_key] = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            del model

    raw_symmetric = directed + directed.T
    np.fill_diagonal(raw_symmetric, 0.0)
    symmetric = np.maximum(raw_symmetric, 0.0)
    np.fill_diagonal(symmetric, 0.0)
    run_metadata = {
        **normalization,
        "config": asdict(config),
        "seed": int(seed),
        "eval_split": "all",
        "num_eval_rows": int(x_eval_arr.shape[0]),
        "num_pair_models": int(num_categories) * (int(num_categories) - 1) // 2,
    }
    return pair_state_dicts, CTSMVBinaryJeffreysResult(
        symmetric_kl_matrix=symmetric,
        raw_symmetric_kl_matrix=raw_symmetric,
        directed_kl_matrix=directed,
        pair_metadata=pair_metadata,
        run_metadata=run_metadata,
    )


def estimate_ctsm_v_jeffreys_matrix(
    model: torch.nn.Module,
    *,
    theta_eval: np.ndarray,
    x_eval: np.ndarray,
    labels: np.ndarray,
    num_categories: int,
    device: torch.device,
    config: CTSMVJeffreysConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate directed KL and Jeffreys matrices from CTSM-v log ratios."""

    x_arr = _as_2d_float32(x_eval, name="x_eval")
    labels_arr = _validate_labels(labels, n_rows=x_arr.shape[0], num_categories=int(num_categories))
    endpoints = _condition_theta(theta_eval, labels_arr, num_categories=int(num_categories))
    directed = np.zeros((int(num_categories), int(num_categories)), dtype=np.float64)
    for i in range(int(num_categories)):
        xi = x_arr[labels_arr == i]
        for j in range(int(num_categories)):
            if i == j:
                continue
            # CTSM-v integrates log p_j(x) - log p_i(x). Negating its
            # expectation under p_i gives KL(p_i || p_j).
            log_pj_minus_pi = _estimate_log_ratio_batched(
                model,
                xi,
                theta_a=endpoints[i],
                theta_b=endpoints[j],
                device=device,
                batch_size=int(config.eval_batch_size),
                t_eps=float(config.t_eps),
                integration_steps=int(config.integration_steps),
            )
            directed[i, j] = -float(np.mean(log_pj_minus_pi, dtype=np.float64))
    raw_symmetric = directed + directed.T
    np.fill_diagonal(raw_symmetric, 0.0)
    symmetric = np.maximum(raw_symmetric, 0.0)
    np.fill_diagonal(symmetric, 0.0)
    return symmetric, raw_symmetric, directed, endpoints


def train_and_estimate_ctsm_v_jeffreys(
    *,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    theta_eval: np.ndarray,
    x_eval: np.ndarray,
    labels_eval: np.ndarray,
    num_categories: int,
    device: torch.device,
    seed: int,
    config: CTSMVJeffreysConfig,
) -> tuple[torch.nn.Module, CTSMVJeffreysResult]:
    """Train pair-conditioned CTSM-v and estimate a Jeffreys matrix."""

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    theta_train_arr = _as_2d_float32(theta_train, name="theta_train")
    theta_val_arr = _as_2d_float32(theta_val, name="theta_val")
    x_train_arr = _as_2d_float32(x_train, name="x_train")
    x_val_arr = _as_2d_float32(x_val, name="x_val")
    x_eval_arr = _as_2d_float32(x_eval, name="x_eval")
    if theta_train_arr.shape[0] != x_train_arr.shape[0] or theta_val_arr.shape[0] != x_val_arr.shape[0]:
        raise ValueError("CTSM-v theta and x split row counts must match.")
    if theta_train_arr.shape[1] != theta_val_arr.shape[1]:
        raise ValueError("CTSM-v train and validation theta dimensions must match.")
    if x_train_arr.shape[1] != x_val_arr.shape[1] or x_train_arr.shape[1] != x_eval_arr.shape[1]:
        raise ValueError("CTSM-v train, validation, and evaluation x dimensions must match.")

    normalization: dict[str, Any] = {"normalize_x": bool(config.normalize_x)}
    if bool(config.normalize_x):
        x_mean, x_std = _fit_normalizer(x_train_arr, eps=float(config.normalize_x_eps))
        x_train_arr = (x_train_arr - x_mean) / x_std
        x_val_arr = (x_val_arr - x_mean) / x_std
        x_eval_arr = (x_eval_arr - x_mean) / x_std
        normalization.update({"normalize_x_mean": x_mean, "normalize_x_std": x_std})

    model = _build_model(
        x_dim=int(x_train_arr.shape[1]),
        theta_dim=int(theta_train_arr.shape[1]),
        config=config,
    ).to(device)
    train_metadata = train_pair_conditioned_ctsm_v_model(
        model=model,
        theta_train=theta_train_arr,
        x_train=x_train_arr,
        epochs=int(config.epochs),
        batch_size=int(config.batch_size),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
        device=device,
        log_every=max(1, int(config.log_every)),
        two_sb_var=float(config.two_sb_var),
        path_schedule=str(config.path_schedule),
        path_eps=float(config.path_eps),
        factor=float(config.factor),
        t_eps=float(config.t_eps),
        theta_val=theta_val_arr,
        x_val=x_val_arr,
        early_stopping_patience=int(config.early_patience),
        early_stopping_min_delta=float(config.early_min_delta),
        early_stopping_ema_alpha=float(config.early_ema_alpha),
        restore_best=bool(config.restore_best),
        val_batches_per_epoch=int(config.validation_batches_per_epoch),
    )
    symmetric, raw_symmetric, directed, endpoints = estimate_ctsm_v_jeffreys_matrix(
        model,
        theta_eval=theta_eval,
        x_eval=x_eval_arr,
        labels=labels_eval,
        num_categories=int(num_categories),
        device=device,
        config=config,
    )
    metadata = {
        **train_metadata,
        **normalization,
        "config": asdict(config),
        "seed": int(seed),
        "eval_split": "all",
        "num_eval_rows": int(x_eval_arr.shape[0]),
    }
    return model, CTSMVJeffreysResult(
        symmetric_kl_matrix=symmetric,
        raw_symmetric_kl_matrix=raw_symmetric,
        directed_kl_matrix=directed,
        condition_theta=endpoints,
        train_metadata=metadata,
    )


def save_ctsm_v_jeffreys_result(
    npz_path: str | Path,
    checkpoint_path: str | Path,
    *,
    model: torch.nn.Module,
    result: CTSMVJeffreysResult,
) -> tuple[Path, Path]:
    """Save CTSM-v matrices, losses, configuration, and model weights."""

    out_npz = Path(npz_path)
    out_checkpoint = Path(checkpoint_path)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(result.train_metadata)
    fields: dict[str, Any] = {
        "symmetric_kl_matrix": np.asarray(result.symmetric_kl_matrix, dtype=np.float64),
        "raw_symmetric_kl_matrix": np.asarray(result.raw_symmetric_kl_matrix, dtype=np.float64),
        "directed_kl_matrix": np.asarray(result.directed_kl_matrix, dtype=np.float64),
        "condition_theta": np.asarray(result.condition_theta, dtype=np.float64),
        "metadata_json": np.asarray([json.dumps(metadata, default=lambda value: np.asarray(value).tolist())]),
    }
    for key in ("train_losses", "val_losses", "val_monitor_losses"):
        if key in metadata:
            fields[key] = np.asarray(metadata[key], dtype=np.float64)
    for key in ("best_epoch", "stopped_epoch", "stopped_early", "best_val_loss"):
        if key in metadata:
            fields[key] = np.asarray([metadata[key]])
    np.savez_compressed(out_npz, **fields)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, out_checkpoint)
    return out_npz, out_checkpoint


def save_pairwise_binary_ctsm_v_jeffreys_result(
    npz_path: str | Path,
    checkpoint_path: str | Path,
    *,
    pair_state_dicts: dict[str, dict[str, torch.Tensor]],
    result: CTSMVBinaryJeffreysResult,
) -> tuple[Path, Path]:
    """Save pairwise CTSM-v-binary matrices, per-pair histories, and model weights."""

    out_npz = Path(npz_path)
    out_checkpoint = Path(checkpoint_path)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run": result.run_metadata,
        "pairs": {
            pair: {
                key: value
                for key, value in pair_metadata.items()
                if key not in {"train_losses", "val_losses", "val_monitor_losses"}
            }
            for pair, pair_metadata in result.pair_metadata.items()
        },
    }
    fields: dict[str, Any] = {
        "symmetric_kl_matrix": np.asarray(result.symmetric_kl_matrix, dtype=np.float64),
        "raw_symmetric_kl_matrix": np.asarray(result.raw_symmetric_kl_matrix, dtype=np.float64),
        "directed_kl_matrix": np.asarray(result.directed_kl_matrix, dtype=np.float64),
        "pair_keys": np.asarray(sorted(result.pair_metadata)),
        "metadata_json": np.asarray(
            [json.dumps(metadata, default=lambda value: np.asarray(value).tolist())]
        ),
    }
    for pair, pair_metadata in result.pair_metadata.items():
        for key in ("train_losses", "val_losses", "val_monitor_losses"):
            fields[f"pair_{pair}_{key}"] = np.asarray(pair_metadata[key], dtype=np.float64)
        for key in ("best_epoch", "stopped_epoch", "stopped_early", "best_val_loss"):
            fields[f"pair_{pair}_{key}"] = np.asarray([pair_metadata[key]])
    np.savez_compressed(out_npz, **fields)
    torch.save(
        {
            "pair_model_state_dicts": pair_state_dicts,
            "metadata": metadata,
        },
        out_checkpoint,
    )
    return out_npz, out_checkpoint
