"""Distance comparison helpers for the MoG5 PR dataset."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fisher.autoencoder_embedding import PRAutoencoderConfig, train_or_load_pr_autoencoder
from fisher.flow_matching_skl import (
    FlowSKLResult,
    build_flow_skl_model,
    estimate_model_symmetric_kl,
    flow_skl_result_to_npz_dict,
    train_flow_skl_model,
)
from fisher.shared_dataset_io import SharedDatasetBundle


METRIC_SQUARED_EUCLIDEAN = "squared_euclidean"
METRIC_COSINE = "cosine"
METRIC_CORRELATION = "correlation"
METRIC_MAHALANOBIS_SQ = "mahalanobis_sq"
METRIC_SYMMETRIC_KL = "symmetric_kl"

METRIC_NAMES = (
    METRIC_SQUARED_EUCLIDEAN,
    METRIC_COSINE,
    METRIC_CORRELATION,
    METRIC_MAHALANOBIS_SQ,
    METRIC_SYMMETRIC_KL,
)

FLOW_VELOCITY_FAMILY_BY_METRIC = {
    METRIC_SQUARED_EUCLIDEAN: "translation",
    METRIC_COSINE: "translation_fixed_norm",
    METRIC_CORRELATION: "translation_centered_fixed_norm",
    METRIC_MAHALANOBIS_SQ: "shared_affine",
    METRIC_SYMMETRIC_KL: "nonlinear",
}

CSV_COLUMNS = (
    "metric",
    "condition_i",
    "condition_j",
    "classical",
    "flow_matching",
    "ground_truth",
    "flow_velocity_family",
    "abs_error_classical",
    "abs_error_flow",
)


@dataclass(frozen=True)
class FlowComparisonConfig:
    """Training/evaluation defaults for flow-matching comparison runs."""

    epochs: int = 20_000
    early_patience: int = 1_000
    early_min_delta: float = 1e-4
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.0
    hidden_dim: int = 128
    depth: int = 3
    low_rank_dim: int = 4
    path_schedule: str = "cosine"
    t_eps: float = 0.05
    quadrature_steps: int = 64
    mc_jeffreys_sample: int = 4096
    ode_steps: int = 64
    ode_method: str = "midpoint"
    divergence_estimator: str = "exact"
    hutchinson_probes: int = 1
    solve_jitter: float = 1e-6
    max_grad_norm: float = 10.0
    log_every: int = 50
    radius: float = 1.0


@dataclass(frozen=True)
class DistanceComparisonResult:
    metrics: tuple[str, ...]
    condition_labels: tuple[str, ...]
    pair_indices: np.ndarray
    classical_matrices: dict[str, np.ndarray]
    flow_matrices: dict[str, np.ndarray]
    ground_truth_matrices: dict[str, np.ndarray]
    rows: list[dict[str, Any]]
    flow_npz_paths: dict[str, Path] = field(default_factory=dict)


def labels_from_theta(theta: np.ndarray, *, num_categories: int | None = None) -> np.ndarray:
    """Return integer labels from one-hot or integer categorical theta."""

    arr = np.asarray(theta)
    if arr.ndim == 2 and int(arr.shape[1]) > 1:
        vals = np.asarray(arr, dtype=np.float64)
        row_sums = vals.sum(axis=1)
        is_binary = np.all((np.abs(vals) <= 1e-6) | (np.abs(vals - 1.0) <= 1e-6), axis=1)
        if np.any(np.abs(row_sums - 1.0) > 1e-6) or not bool(np.all(is_binary)):
            raise ValueError("Expected one-hot categorical theta rows.")
        labels = np.argmax(vals, axis=1).astype(np.int64)
    else:
        vals = np.asarray(arr, dtype=np.float64).reshape(-1)
        labels = np.rint(vals).astype(np.int64)
        if np.any(np.abs(vals - labels.astype(np.float64)) > 1e-6):
            raise ValueError("Expected integer categorical theta labels.")
    if num_categories is not None:
        k = int(num_categories)
        if np.any((labels < 0) | (labels >= k)):
            raise ValueError(f"Categorical labels must be in [0, {k - 1}].")
    return labels


def condition_labels(num_categories: int) -> tuple[str, ...]:
    return tuple(f"category_{i}" for i in range(int(num_categories)))


def pair_indices(num_categories: int) -> np.ndarray:
    pairs = [(i, j) for i in range(int(num_categories)) for j in range(i + 1, int(num_categories))]
    return np.asarray(pairs, dtype=np.int64)


def class_means(x: np.ndarray, labels: np.ndarray, *, num_categories: int) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int64).reshape(-1)
    if x_arr.ndim != 2:
        raise ValueError("x must be a two-dimensional array.")
    if int(x_arr.shape[0]) != int(lab.shape[0]):
        raise ValueError("x and labels must have the same number of rows.")
    means = np.zeros((int(num_categories), int(x_arr.shape[1])), dtype=np.float64)
    for k in range(int(num_categories)):
        mask = lab == k
        if not np.any(mask):
            raise ValueError(f"Category {k} has no rows.")
        means[k] = np.mean(x_arr[mask], axis=0, dtype=np.float64)
    return means


def squared_euclidean_mean_distance_matrix(means: np.ndarray) -> np.ndarray:
    m = np.asarray(means, dtype=np.float64)
    diff = m[:, None, :] - m[None, :, :]
    out = np.sum(diff * diff, axis=2, dtype=np.float64)
    out = np.maximum(out, 0.0)
    np.fill_diagonal(out, 0.0)
    return out


def cosine_distance_matrix(means: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    m = np.asarray(means, dtype=np.float64)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    if np.any(norms <= float(eps)):
        raise ValueError("Cosine distance is undefined for a zero-norm class mean.")
    u = m / norms
    out = 1.0 - u @ u.T
    out = np.clip(out, 0.0, 2.0)
    np.fill_diagonal(out, 0.0)
    return out


def correlation_distance_matrix(means: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    m = np.asarray(means, dtype=np.float64)
    centered = m - np.mean(m, axis=1, keepdims=True, dtype=np.float64)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norms <= float(eps)):
        raise ValueError("Correlation distance is undefined for a zero-variance class mean.")
    u = centered / norms
    out = 1.0 - u @ u.T
    out = np.clip(out, 0.0, 2.0)
    np.fill_diagonal(out, 0.0)
    return out


def pooled_within_class_covariance(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    num_categories: int,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int64).reshape(-1)
    if x_arr.ndim != 2:
        raise ValueError("x must be a two-dimensional array.")
    if int(x_arr.shape[0]) != int(lab.shape[0]):
        raise ValueError("x and labels must have the same number of rows.")
    k = int(num_categories)
    dof = int(x_arr.shape[0]) - k
    if dof < 1:
        raise ValueError("Need more observations than categories for pooled covariance.")
    means = class_means(x_arr, lab, num_categories=k)
    scatter = np.zeros((int(x_arr.shape[1]), int(x_arr.shape[1])), dtype=np.float64)
    for cls in range(k):
        centered = x_arr[lab == cls] - means[cls]
        scatter += centered.T @ centered
    return scatter / float(dof)


def mahalanobis_sq_matrix(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    num_categories: int,
    ridge: float = 1e-6,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int64).reshape(-1)
    means = class_means(x_arr, lab, num_categories=int(num_categories))
    d = int(x_arr.shape[1])
    cov = pooled_within_class_covariance(x_arr, lab, num_categories=int(num_categories))
    cov = cov + float(ridge) * np.eye(d, dtype=np.float64)
    out = np.zeros((int(num_categories), int(num_categories)), dtype=np.float64)
    for i, j in pair_indices(num_categories):
        delta = means[int(i)] - means[int(j)]
        val = float(delta @ np.linalg.solve(cov, delta))
        out[int(i), int(j)] = out[int(j), int(i)] = max(0.0, val)
    return out


def logistic_density_ratio_skl_matrix(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    num_categories: int,
    n_splits: int = 5,
    seed: int = 7,
    logistic_c: float = 1.0,
) -> np.ndarray:
    """Estimate Jeffreys KL with cross-fitted logistic density ratios."""

    x_arr = np.asarray(x, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int64).reshape(-1)
    if x_arr.ndim != 2:
        raise ValueError("x must be two-dimensional.")
    if int(x_arr.shape[0]) != int(lab.shape[0]):
        raise ValueError("x and labels must have the same number of rows.")
    out = np.zeros((int(num_categories), int(num_categories)), dtype=np.float64)
    for i, j in pair_indices(num_categories):
        idx_i = np.flatnonzero(lab == int(i))
        idx_j = np.flatnonzero(lab == int(j))
        if int(idx_i.size) < 2 or int(idx_j.size) < 2:
            raise ValueError(f"Need at least two rows per class for cross-fitted SKL; pair=({i}, {j}).")
        pair_x = np.vstack([x_arr[idx_i], x_arr[idx_j]])
        pair_y = np.concatenate(
            [
                np.ones(int(idx_i.size), dtype=np.int64),
                np.zeros(int(idx_j.size), dtype=np.int64),
            ]
        )
        folds = min(int(n_splits), int(idx_i.size), int(idx_j.size))
        if folds < 2:
            raise ValueError(f"n_splits={n_splits} is too large for pair ({i}, {j}).")
        logits = np.zeros(int(pair_y.shape[0]), dtype=np.float64)
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
        for train_idx, test_idx in cv.split(pair_x, pair_y):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=float(logistic_c),
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                ),
            )
            clf.fit(pair_x[train_idx], pair_y[train_idx])
            logits[test_idx] = np.asarray(clf.decision_function(pair_x[test_idx]), dtype=np.float64)
        skl = float(np.mean(logits[pair_y == 1], dtype=np.float64) - np.mean(logits[pair_y == 0], dtype=np.float64))
        out[int(i), int(j)] = out[int(j), int(i)] = max(0.0, skl)
    return out


def classical_metric_matrices(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    num_categories: int,
    metrics: Iterable[str] = METRIC_NAMES,
    mahalanobis_ridge: float = 1e-6,
    skl_folds: int = 5,
    skl_seed: int = 7,
    skl_logistic_c: float = 1.0,
) -> dict[str, np.ndarray]:
    metric_tuple = tuple(str(m) for m in metrics)
    means = class_means(x, labels, num_categories=int(num_categories))
    out: dict[str, np.ndarray] = {}
    for metric in metric_tuple:
        if metric == METRIC_SQUARED_EUCLIDEAN:
            out[metric] = squared_euclidean_mean_distance_matrix(means)
        elif metric == METRIC_COSINE:
            out[metric] = cosine_distance_matrix(means)
        elif metric == METRIC_CORRELATION:
            out[metric] = correlation_distance_matrix(means)
        elif metric == METRIC_MAHALANOBIS_SQ:
            out[metric] = mahalanobis_sq_matrix(
                x,
                labels,
                num_categories=int(num_categories),
                ridge=float(mahalanobis_ridge),
            )
        elif metric == METRIC_SYMMETRIC_KL:
            out[metric] = logistic_density_ratio_skl_matrix(
                x,
                labels,
                num_categories=int(num_categories),
                n_splits=int(skl_folds),
                seed=int(skl_seed),
                logistic_c=float(skl_logistic_c),
            )
        else:
            raise ValueError(f"Unknown metric: {metric!r}.")
    return out


def analytic_diagonal_gaussian_skl_matrix(
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Analytic Jeffreys KL for diagonal Gaussian components."""

    mu = np.asarray(means, dtype=np.float64)
    var = np.asarray(variances, dtype=np.float64)
    if mu.ndim != 2 or var.shape != mu.shape:
        raise ValueError("means and variances must have matching shape [K, D].")
    if np.any(var <= 0.0):
        raise ValueError("variances must be strictly positive.")
    k = int(mu.shape[0])
    out = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(i + 1, k):
            diff2 = (mu[i] - mu[j]) ** 2
            kl_ij = 0.5 * np.sum(np.log(var[j] / var[i]) + (var[i] + diff2) / var[j] - 1.0)
            kl_ji = 0.5 * np.sum(np.log(var[i] / var[j]) + (var[j] + diff2) / var[i] - 1.0)
            out[i, j] = out[j, i] = float(kl_ij + kl_ji)
    return out


def _pr_config_from_projected_meta(meta: dict[str, Any]) -> PRAutoencoderConfig:
    return PRAutoencoderConfig(
        z_dim=int(meta.get("pr_autoencoder_z_dim", 2)),
        h_dim=int(meta["x_dim"]),
        hidden1=int(meta.get("pr_autoencoder_hidden1", 100)),
        hidden2=int(meta.get("pr_autoencoder_hidden2", 200)),
        train_samples=int(meta.get("pr_autoencoder_train_samples", 12000)),
        train_epochs=int(meta.get("pr_autoencoder_train_epochs", 200)),
        train_batch_size=int(meta.get("pr_autoencoder_train_batch_size", 512)),
        train_lr=float(meta.get("pr_autoencoder_train_lr", 1e-3)),
        lambda_pr=float(meta.get("pr_autoencoder_lambda_pr", 1e-2)),
        pr_eps=float(meta.get("pr_autoencoder_pr_eps", 1e-8)),
        adversarial_categorical=bool(meta.get("pr_autoencoder_adversarial_categorical", False)),
        lambda_adv=float(meta.get("pr_autoencoder_lambda_adv", 0.1)),
        adv_warmup_epochs=int(meta.get("pr_autoencoder_adv_warmup_epochs", 0)),
        adv_ramp_epochs=int(meta.get("pr_autoencoder_adv_ramp_epochs", 40)),
        adv_steps=int(meta.get("pr_autoencoder_adv_steps", 1)),
        adv_train_samples=int(meta.get("pr_autoencoder_adv_train_samples", 0)),
        adv_num_classes=int(meta.get("pr_autoencoder_adv_num_classes", 0)),
        adv_source_sha256=str(meta.get("pr_autoencoder_adv_source_sha256", "")),
    )


def encode_with_pr_autoencoder(
    z: np.ndarray,
    *,
    projected_meta: dict[str, Any],
    device: torch.device,
    cache_dir: str | Path,
    batch_size: int = 8192,
) -> np.ndarray:
    """Encode native MoG samples through the cached PR autoencoder."""

    meta = dict(projected_meta)
    if not bool(meta.get("pr_autoencoder_embedded", False)):
        raise ValueError("projected_meta must describe a PR-autoencoder embedded dataset.")
    cfg = _pr_config_from_projected_meta(meta)
    if bool(cfg.adversarial_categorical):
        raise ValueError("Adversarial categorical PR autoencoders need source labels and are not supported here.")
    built = train_or_load_pr_autoencoder(
        config=cfg,
        seed=int(meta.get("pr_autoencoder_seed", meta.get("seed", 7))),
        device=device,
        cache_dir=cache_dir,
        force_retrain=False,
        logger=None,
    )
    z_arr = np.asarray(z, dtype=np.float64)
    if z_arr.ndim != 2 or int(z_arr.shape[1]) != int(cfg.z_dim):
        raise ValueError(f"z must have shape [N, {cfg.z_dim}].")
    outs: list[np.ndarray] = []
    built.model.eval()
    for start in range(0, int(z_arr.shape[0]), int(batch_size)):
        zb = torch.from_numpy(z_arr[start : start + int(batch_size)].astype(np.float32, copy=False)).to(device)
        with torch.no_grad():
            hb, _ = built.model(zb)
        outs.append(hb.detach().cpu().numpy().astype(np.float64, copy=False))
    return np.concatenate(outs, axis=0)


def pr_autoencoder_ground_truth_matrices(
    *,
    native_meta: dict[str, Any],
    projected_meta: dict[str, Any],
    device: torch.device,
    cache_dir: str | Path,
    samples_per_class: int = 200_000,
    seed: int = 7,
    batch_size: int = 8192,
    mahalanobis_ridge: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Monte Carlo projected-coordinate ground truth plus analytic native SKL."""

    means = np.asarray(native_meta.get("mog_component_means"), dtype=np.float64)
    variances = np.asarray(native_meta.get("mog_component_variances"), dtype=np.float64)
    if means.ndim != 2 or variances.shape != means.shape:
        raise ValueError("native_meta must contain mog_component_means and mog_component_variances.")
    k, d = means.shape
    n = int(samples_per_class)
    if n < 2:
        raise ValueError("samples_per_class must be at least 2.")
    rng = np.random.default_rng(int(seed))
    z_parts: list[np.ndarray] = []
    labels = np.repeat(np.arange(k, dtype=np.int64), n)
    for cls in range(k):
        z = means[cls] + np.sqrt(variances[cls]) * rng.standard_normal(size=(n, d))
        z_parts.append(z.astype(np.float64, copy=False))
    z_all = np.vstack(z_parts)
    h_all = encode_with_pr_autoencoder(
        z_all,
        projected_meta=projected_meta,
        device=device,
        cache_dir=cache_dir,
        batch_size=int(batch_size),
    )
    out = classical_metric_matrices(
        h_all,
        labels,
        num_categories=k,
        metrics=(
            METRIC_SQUARED_EUCLIDEAN,
            METRIC_COSINE,
            METRIC_CORRELATION,
            METRIC_MAHALANOBIS_SQ,
        ),
        mahalanobis_ridge=float(mahalanobis_ridge),
    )
    out[METRIC_SYMMETRIC_KL] = analytic_diagonal_gaussian_skl_matrix(means, variances)
    return out


def _theta_eval_from_num_categories(num_categories: int) -> np.ndarray:
    return np.eye(int(num_categories), dtype=np.float64)


def _flow_npz_name(metric: str, pair: tuple[int, int] | None = None) -> str:
    if pair is None:
        return f"{metric}_flow_matching_skl_results.npz"
    return f"{metric}_pair_{int(pair[0])}_{int(pair[1])}_flow_matching_skl_results.npz"


def flow_skl_to_metric_readout(metric: str, symmetric_kl_matrix: np.ndarray, *, radius: float) -> np.ndarray:
    """Convert raw model flow Jeffreys SKL to the comparison-row metric."""

    mat = np.asarray(symmetric_kl_matrix, dtype=np.float64)
    out = mat.copy()
    if str(metric) in (METRIC_COSINE, METRIC_CORRELATION):
        r = float(radius)
        if not math.isfinite(r) or r <= 0.0:
            raise ValueError("radius must be finite and positive for fixed-norm flow readouts.")
        out = out / (2.0 * r * r)
    np.fill_diagonal(out, 0.0)
    return out


def train_and_estimate_flow(
    *,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    theta_eval: np.ndarray,
    velocity_family: str,
    device: torch.device,
    seed: int,
    config: FlowComparisonConfig,
) -> FlowSKLResult:
    """Train one flow model and return its endpoint metric matrix."""

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    theta_dim = int(theta_train.shape[1] if np.asarray(theta_train).ndim == 2 else 1)
    x_dim = int(x_train.shape[1] if np.asarray(x_train).ndim == 2 else 1)
    model = build_flow_skl_model(
        velocity_family=str(velocity_family),
        theta_dim=theta_dim,
        x_dim=x_dim,
        radius=float(config.radius),
        hidden_dim=int(config.hidden_dim),
        depth=int(config.depth),
        low_rank_dim=int(config.low_rank_dim),
        quadrature_steps=int(config.quadrature_steps),
        path_schedule=str(config.path_schedule),
        divergence_estimator=str(config.divergence_estimator),
        hutchinson_probes=int(config.hutchinson_probes),
    ).to(device)
    train_meta = train_flow_skl_model(
        model=model,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        device=device,
        velocity_family=str(velocity_family),
        path_schedule=str(config.path_schedule),
        epochs=int(config.epochs),
        batch_size=int(config.batch_size),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
        t_eps=float(config.t_eps),
        patience=int(config.early_patience),
        min_delta=float(config.early_min_delta),
        max_grad_norm=float(config.max_grad_norm),
        log_every=max(1, int(config.log_every)),
    )
    return estimate_model_symmetric_kl(
        model=model,
        theta_all=theta_eval,
        device=device,
        velocity_family=str(velocity_family),
        radius=float(config.radius),
        mc_jeffreys_sample=int(config.mc_jeffreys_sample),
        ode_steps=int(config.ode_steps),
        ode_method=str(config.ode_method),
        batch_size=int(config.batch_size),
        solve_jitter=float(config.solve_jitter),
        quadrature_steps=int(config.quadrature_steps),
        fisher_kind="none",
        train_metadata=train_meta,
    )


def save_flow_result_npz(
    path: str | Path,
    *,
    result: FlowSKLResult,
    metric: str,
    theta_eval: np.ndarray,
    velocity_family: str,
    pair: tuple[int, int] | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = flow_skl_result_to_npz_dict(result)
    fields.update(
        {
            "metric": np.asarray([str(metric)]),
            "velocity_family": np.asarray([str(velocity_family)]),
            "theta_eval": np.asarray(theta_eval, dtype=np.float64),
        }
    )
    if pair is not None:
        fields["condition_pair"] = np.asarray(pair, dtype=np.int64)
    meta = dict(result.train_metadata)
    for key in ("train_losses", "val_losses"):
        if key in meta:
            fields[key] = np.asarray(meta[key], dtype=np.float64)
    for key in ("best_val_loss", "best_epoch", "stopped_epoch", "stopped_early"):
        if key in meta:
            fields[key] = np.asarray([meta[key]])
    np.savez_compressed(out, **fields)
    return out


def flow_metric_matrices(
    *,
    bundle: SharedDatasetBundle,
    device: torch.device,
    output_dir: str | Path,
    config: FlowComparisonConfig,
    seed: int = 7,
    metrics: Iterable[str] = METRIC_NAMES,
) -> tuple[dict[str, np.ndarray], dict[str, Path]]:
    """Train flow models and assemble per-metric matrices."""

    meta = dict(bundle.meta)
    k = int(meta.get("num_categories", np.asarray(bundle.theta_all).shape[1]))
    x_train_all = np.asarray(bundle.x_train, dtype=np.float64)
    x_val_all = np.asarray(bundle.x_validation, dtype=np.float64)
    theta_train_all = np.asarray(bundle.theta_train, dtype=np.float64)
    theta_val_all = np.asarray(bundle.theta_validation, dtype=np.float64)
    flow_dir = Path(output_dir)
    matrices: dict[str, np.ndarray] = {}
    paths: dict[str, Path] = {}
    theta_eval = _theta_eval_from_num_categories(k)

    for metric in tuple(str(m) for m in metrics):
        family = FLOW_VELOCITY_FAMILY_BY_METRIC[metric]
        print(f"[distance-comparison] flow metric={metric} velocity_family={family}", flush=True)
        result = train_and_estimate_flow(
            theta_train=theta_train_all,
            x_train=x_train_all,
            theta_val=theta_val_all,
            x_val=x_val_all,
            theta_eval=theta_eval,
            velocity_family=family,
            device=device,
            seed=int(seed),
            config=config,
        )
        matrices[metric] = flow_skl_to_metric_readout(
            metric,
            result.symmetric_kl_matrix,
            radius=float(config.radius),
        )
        path = save_flow_result_npz(
            flow_dir / _flow_npz_name(metric),
            result=result,
            metric=metric,
            theta_eval=theta_eval,
            velocity_family=family,
        )
        paths[metric] = path
    return matrices, paths


def pair_rows(
    *,
    metrics: Iterable[str],
    labels: Iterable[str],
    classical: dict[str, np.ndarray],
    flow_matching: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    label_tuple = tuple(str(v) for v in labels)
    pairs = pair_indices(len(label_tuple))
    rows: list[dict[str, Any]] = []
    for metric in tuple(str(m) for m in metrics):
        cmat = np.asarray(classical[metric], dtype=np.float64)
        fmat = np.asarray(flow_matching[metric], dtype=np.float64)
        gmat = np.asarray(ground_truth[metric], dtype=np.float64)
        for i, j in pairs:
            i_int, j_int = int(i), int(j)
            cval = float(cmat[i_int, j_int])
            fval = float(fmat[i_int, j_int])
            gval = float(gmat[i_int, j_int])
            rows.append(
                {
                    "metric": metric,
                    "condition_i": label_tuple[i_int],
                    "condition_j": label_tuple[j_int],
                    "classical": cval,
                    "flow_matching": fval,
                    "ground_truth": gval,
                    "flow_velocity_family": FLOW_VELOCITY_FAMILY_BY_METRIC[metric],
                    "abs_error_classical": abs(cval - gval),
                    "abs_error_flow": abs(fval - gval),
                }
            )
    return rows


def assemble_comparison_result(
    *,
    metrics: Iterable[str],
    condition_names: Iterable[str],
    classical: dict[str, np.ndarray],
    flow_matching: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
    flow_npz_paths: dict[str, Path] | None = None,
) -> DistanceComparisonResult:
    metric_tuple = tuple(str(m) for m in metrics)
    label_tuple = tuple(str(v) for v in condition_names)
    rows = pair_rows(
        metrics=metric_tuple,
        labels=label_tuple,
        classical=classical,
        flow_matching=flow_matching,
        ground_truth=ground_truth,
    )
    return DistanceComparisonResult(
        metrics=metric_tuple,
        condition_labels=label_tuple,
        pair_indices=pair_indices(len(label_tuple)),
        classical_matrices={m: np.asarray(classical[m], dtype=np.float64) for m in metric_tuple},
        flow_matrices={m: np.asarray(flow_matching[m], dtype=np.float64) for m in metric_tuple},
        ground_truth_matrices={m: np.asarray(ground_truth[m], dtype=np.float64) for m in metric_tuple},
        rows=rows,
        flow_npz_paths={} if flow_npz_paths is None else dict(flow_npz_paths),
    )


def write_pairs_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in CSV_COLUMNS})
    return out


def write_results_npz(path: str | Path, result: DistanceComparisonResult) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    metric_names = np.asarray(result.metrics)
    stack_classical = np.stack([result.classical_matrices[m] for m in result.metrics], axis=0)
    stack_flow = np.stack([result.flow_matrices[m] for m in result.metrics], axis=0)
    stack_gt = np.stack([result.ground_truth_matrices[m] for m in result.metrics], axis=0)
    np.savez_compressed(
        out,
        metric_names=metric_names,
        condition_labels=np.asarray(result.condition_labels),
        pair_indices=result.pair_indices.astype(np.int64, copy=False),
        classical_matrices=stack_classical,
        flow_matching_matrices=stack_flow,
        ground_truth_matrices=stack_gt,
        abs_error_classical=np.abs(stack_classical - stack_gt),
        abs_error_flow=np.abs(stack_flow - stack_gt),
    )
    return out


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
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_summary_json(
    path: str | Path,
    *,
    result: DistanceComparisonResult,
    extra: dict[str, Any] | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    metric_summary: dict[str, Any] = {}
    for metric in result.metrics:
        c = np.asarray(result.classical_matrices[metric], dtype=np.float64)
        f = np.asarray(result.flow_matrices[metric], dtype=np.float64)
        g = np.asarray(result.ground_truth_matrices[metric], dtype=np.float64)
        pairs = result.pair_indices
        c_err = np.asarray([abs(float(c[i, j]) - float(g[i, j])) for i, j in pairs], dtype=np.float64)
        f_err = np.asarray([abs(float(f[i, j]) - float(g[i, j])) for i, j in pairs], dtype=np.float64)
        metric_summary[metric] = {
            "flow_velocity_family": FLOW_VELOCITY_FAMILY_BY_METRIC[metric],
            "mean_abs_error_classical": float(np.mean(c_err)),
            "mean_abs_error_flow": float(np.mean(f_err)),
            "max_abs_error_classical": float(np.max(c_err)),
            "max_abs_error_flow": float(np.max(f_err)),
        }
    summary = {
        "metrics": list(result.metrics),
        "condition_labels": list(result.condition_labels),
        "num_pairs": int(result.pair_indices.shape[0]),
        "metric_summary": metric_summary,
        "flow_npz_paths": {k: str(v) for k, v in result.flow_npz_paths.items()},
    }
    if extra is not None:
        summary.update(extra)
    out.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
