#!/usr/bin/env python3
"""Diagnose symmetric-KL flow endpoints on native or PR-projected MoG5 data."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import DATA_DIR

from fisher import flow_matching_skl as fms
from fisher.distance_comparison import analytic_diagonal_gaussian_skl_matrix, labels_from_theta, pair_indices
from fisher.flow_matching_skl import (
    build_flow_skl_model,
    estimate_model_symmetric_kl,
    sample_flow_endpoint,
    train_flow_skl_model,
)
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import require_device

NUM_CATEGORIES = 5
RESULTS_NPZ_NAME = "symmetric_kl_endpoint_diagnostics.npz"
SUMMARY_JSON_NAME = "symmetric_kl_endpoint_diagnostics_summary.json"
PAIR_ERRORS_CSV_NAME = "symmetric_kl_endpoint_pair_errors.csv"


def _load_make_mog5_module() -> Any:
    path = _REPO_ROOT / "bin" / "make_mog5_pr_dataset.py"
    spec = importlib.util.spec_from_file_location("make_mog5_pr_dataset", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _repo_data_dir() -> Path:
    repo_data = _REPO_ROOT / "data"
    data_dir = Path(DATA_DIR)
    if not data_dir.is_absolute():
        return _REPO_ROOT / data_dir
    try:
        if repo_data.exists() and repo_data.resolve() == data_dir.resolve():
            return repo_data
    except OSError:
        pass
    return data_dir


def parse_pr_dim(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.lower() in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--pr-dim must be an integer, 'none', or 'null'.") from exc


def default_output_dir(*, n_total: int, native_x_dim: int, pr_dim: int | None, seed: int) -> Path:
    mode = "native" if pr_dim is None else f"pr{int(pr_dim)}"
    return (
        _repo_data_dir()
        / "mog5_symmetric_kl_endpoint_diagnostics"
        / f"mog5_{mode}_xdim{int(native_x_dim)}_n{int(n_total)}_seed{int(seed)}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--n-total", type=int, default=1_000)
    p.add_argument("--native-x-dim", type=int, default=3)
    p.add_argument("--pr-dim", type=parse_pr_dim, default=None, help="Use 'none' or 'null' for native mode.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--native-template-npz", type=Path, default=None)
    p.add_argument("--force-dataset", action="store_true")
    p.add_argument("--dataset-use-cache", action="store_true")
    p.add_argument("--skip-dataset-viz", action="store_true")
    p.add_argument("--pr-cache-dir", type=Path, default=_REPO_ROOT / "data" / "pr_autoencoder_cache")

    p.add_argument("--epochs", type=int, default=20_000)
    p.add_argument("--early-patience", type=int, default=1_000)
    p.add_argument("--early-min-delta", type=float, default=1e-4)
    p.add_argument("--early-ema-alpha", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--t-eps", type=float, default=0.0005)
    p.add_argument("--ode-steps", type=int, default=64)
    p.add_argument("--ode-method", type=str, default="midpoint")
    p.add_argument("--mc-jeffreys-sample", dest="mc_jeffreys_sample", type=int, default=4096)
    p.add_argument("--mc-samples", dest="mc_jeffreys_sample", type=int, help=argparse.SUPPRESS)
    p.add_argument("--quadrature-steps", type=int, default=64)
    p.add_argument("--divergence-estimator", choices=("hutchinson", "exact"), default="exact")
    p.add_argument("--hutchinson-probes", type=int, default=1)
    p.add_argument("--solve-jitter", type=float, default=1e-6)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=50)

    p.add_argument("--endpoint-samples-per-class", type=int, default=4096)
    p.add_argument("--logprob-samples-per-class", type=int, default=4096)
    p.add_argument("--two-sample-max-points", type=int, default=512)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--skip-plots", action="store_true")
    return p


def validate_args(args: argparse.Namespace) -> None:
    if int(args.n_total) <= 0:
        raise ValueError("--n-total must be positive.")
    if int(args.native_x_dim) < 2:
        raise ValueError("--native-x-dim must be >= 2.")
    if args.pr_dim is not None and int(args.pr_dim) < int(args.native_x_dim):
        raise ValueError("--pr-dim must be >= --native-x-dim.")
    if int(args.endpoint_samples_per_class) < 2:
        raise ValueError("--endpoint-samples-per-class must be >= 2.")
    if int(args.logprob_samples_per_class) < 1:
        raise ValueError("--logprob-samples-per-class must be >= 1.")


def resolve_dataset_dir(args: argparse.Namespace) -> Path:
    if args.dataset_dir is not None:
        return Path(args.dataset_dir).expanduser()
    mod = _load_make_mog5_module()
    return Path(
        mod.default_output_dir(
            n_total=int(args.n_total),
            pr_dim=args.pr_dim,
            native_x_dim=int(args.native_x_dim),
        )
    )


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).expanduser()
    return default_output_dir(
        n_total=int(args.n_total),
        native_x_dim=int(args.native_x_dim),
        pr_dim=args.pr_dim,
        seed=int(args.seed),
    )


def _dataset_wrapper_args(args: argparse.Namespace, dataset_dir: Path) -> argparse.Namespace:
    mod = _load_make_mog5_module()
    argv = [
        "--n-total",
        str(int(args.n_total)),
        "--native-x-dim",
        str(int(args.native_x_dim)),
        "--pr-dim",
        "none" if args.pr_dim is None else str(int(args.pr_dim)),
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.device),
        "--output-dir",
        str(dataset_dir),
        "--pr-cache-dir",
        str(Path(args.pr_cache_dir)),
    ]
    if args.native_template_npz is not None:
        argv.extend(["--native-template-npz", str(Path(args.native_template_npz))])
    if bool(args.force_dataset):
        argv.append("--force")
    if bool(args.dataset_use_cache):
        argv.append("--use-cache")
    if bool(args.skip_dataset_viz):
        argv.append("--skip-viz")
    return mod.parse_args(argv)


def ensure_dataset(args: argparse.Namespace, dataset_dir: Path) -> tuple[Path, Path | None]:
    mod = _load_make_mog5_module()
    return mod.run(_dataset_wrapper_args(args, dataset_dir))


def _theta_eval(num_categories: int) -> np.ndarray:
    return np.eye(int(num_categories), dtype=np.float64)


def _native_params(meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray(meta.get("mog_component_means"), dtype=np.float64)
    variances = np.asarray(meta.get("mog_component_variances"), dtype=np.float64)
    if means.ndim != 2 or variances.shape != means.shape:
        raise ValueError("Native MoG metadata must contain mog_component_means and mog_component_variances.")
    if np.any(variances <= 0.0):
        raise ValueError("Native MoG variances must be positive.")
    return means, variances


def sample_native_class(means: np.ndarray, variances: np.ndarray, cls: int, n: int, rng: np.random.Generator) -> np.ndarray:
    return (
        means[int(cls)] + np.sqrt(variances[int(cls)]) * rng.standard_normal(size=(int(n), int(means.shape[1])))
    ).astype(np.float64, copy=False)


def diagonal_gaussian_log_prob(x: np.ndarray, means: np.ndarray, variances: np.ndarray, cls: int) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    mu = means[int(cls)]
    var = variances[int(cls)]
    delta = x_arr - mu.reshape(1, -1)
    return -0.5 * (
        float(mu.size) * np.log(2.0 * np.pi)
        + np.sum(np.log(var), dtype=np.float64)
        + np.sum(delta * delta / var.reshape(1, -1), axis=1, dtype=np.float64)
    )


def model_log_prob(
    *,
    model: torch.nn.Module,
    x: np.ndarray | torch.Tensor,
    theta: np.ndarray,
    device: torch.device,
    ode_steps: int,
    ode_method: str,
    batch_size: int,
    solve_jitter: float,
    quadrature_steps: int,
) -> np.ndarray:
    x_t = x if isinstance(x, torch.Tensor) else torch.from_numpy(np.asarray(x, dtype=np.float32))
    return fms._log_prob_model(
        model=model,
        x=x_t,
        theta=theta,
        device=device,
        ode_steps=int(ode_steps),
        ode_method=str(ode_method),
        batch_size=int(batch_size),
        solve_jitter=float(solve_jitter),
        quadrature_steps=int(quadrature_steps),
    )


def sample_model_class(
    *,
    model: torch.nn.Module,
    theta: np.ndarray,
    n: int,
    device: torch.device,
    ode_steps: int,
    ode_method: str,
) -> np.ndarray:
    x = sample_flow_endpoint(
        model=model,
        theta=theta,
        n_samples=int(n),
        device=device,
        ode_steps=int(ode_steps),
        ode_method=str(ode_method),
    )
    return x.detach().cpu().numpy().astype(np.float64, copy=False)


def _subsample_rows(x: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if int(max_points) <= 0 or int(arr.shape[0]) <= int(max_points):
        return arr
    idx = rng.choice(int(arr.shape[0]), size=int(max_points), replace=False)
    return arr[np.sort(idx)]


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    dxy = np.linalg.norm(x_arr[:, None, :] - y_arr[None, :, :], axis=2)
    dxx = np.linalg.norm(x_arr[:, None, :] - x_arr[None, :, :], axis=2)
    dyy = np.linalg.norm(y_arr[:, None, :] - y_arr[None, :, :], axis=2)
    return float(2.0 * np.mean(dxy, dtype=np.float64) - np.mean(dxx, dtype=np.float64) - np.mean(dyy, dtype=np.float64))


def rbf_mmd2(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    xy = np.vstack([x_arr, y_arr])
    sq = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=2, dtype=np.float64)
    tri = sq[np.triu_indices(int(sq.shape[0]), k=1)]
    positive = tri[tri > 0.0]
    bandwidth2 = float(np.median(positive)) if positive.size else 1.0
    bandwidth2 = max(bandwidth2, 1e-12)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dist2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2, dtype=np.float64)
        return np.exp(-0.5 * dist2 / bandwidth2)

    kxx = kernel(x_arr, x_arr)
    kyy = kernel(y_arr, y_arr)
    kxy = kernel(x_arr, y_arr)
    return float(np.mean(kxx, dtype=np.float64) + np.mean(kyy, dtype=np.float64) - 2.0 * np.mean(kxy, dtype=np.float64))


def classifier_two_sample_auc(x: np.ndarray, y: np.ndarray, *, seed: int) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n_min = min(int(x_arr.shape[0]), int(y_arr.shape[0]))
    if n_min < 4:
        return float("nan")
    data = np.vstack([x_arr, y_arr])
    labels = np.concatenate([np.zeros(int(x_arr.shape[0]), dtype=np.int64), np.ones(int(y_arr.shape[0]), dtype=np.int64)])
    folds = min(5, n_min)
    if folds < 2:
        return float("nan")
    scores = np.zeros(int(labels.shape[0]), dtype=np.float64)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
    for train_idx, test_idx in cv.split(data, labels):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs"),
        )
        clf.fit(data[train_idx], labels[train_idx])
        scores[test_idx] = np.asarray(clf.decision_function(data[test_idx]), dtype=np.float64)
    auc = float(roc_auc_score(labels, scores))
    return max(auc, 1.0 - auc)


def endpoint_sample_diagnostics(
    *,
    true_samples_by_class: list[np.ndarray],
    model_samples_by_class: list[np.ndarray],
    two_sample_max_points: int,
    seed: int,
) -> dict[str, np.ndarray]:
    k = len(true_samples_by_class)
    mean_error = np.zeros(k, dtype=np.float64)
    diag_var_error = np.zeros(k, dtype=np.float64)
    cov_fro_error = np.zeros(k, dtype=np.float64)
    auc = np.zeros(k, dtype=np.float64)
    energy = np.zeros(k, dtype=np.float64)
    mmd2 = np.zeros(k, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    for cls in range(k):
        true_x = np.asarray(true_samples_by_class[cls], dtype=np.float64)
        model_x = np.asarray(model_samples_by_class[cls], dtype=np.float64)
        true_cov = np.cov(true_x, rowvar=False)
        model_cov = np.cov(model_x, rowvar=False)
        true_cov = np.atleast_2d(true_cov)
        model_cov = np.atleast_2d(model_cov)
        mean_error[cls] = float(np.linalg.norm(np.mean(model_x, axis=0) - np.mean(true_x, axis=0)))
        diag_var_error[cls] = float(np.linalg.norm(np.diag(model_cov) - np.diag(true_cov)))
        cov_fro_error[cls] = float(np.linalg.norm(model_cov - true_cov, ord="fro"))
        true_small = _subsample_rows(true_x, int(two_sample_max_points), rng)
        model_small = _subsample_rows(model_x, int(two_sample_max_points), rng)
        auc[cls] = classifier_two_sample_auc(true_small, model_small, seed=int(seed) + cls)
        energy[cls] = energy_distance(true_small, model_small)
        mmd2[cls] = rbf_mmd2(true_small, model_small)
    return {
        "category_mean_error": mean_error,
        "category_diag_variance_error": diag_var_error,
        "category_covariance_frobenius_error": cov_fro_error,
        "category_classifier_two_sample_auc": auc,
        "category_energy_distance": energy,
        "category_rbf_mmd2": mmd2,
    }


def exact_native_kl_diagnostics(
    *,
    model: torch.nn.Module,
    theta_eval: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    device: torch.device,
    samples_per_class: int,
    seed: int,
    ode_steps: int,
    ode_method: str,
    batch_size: int,
    solve_jitter: float,
    quadrature_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k = int(theta_eval.shape[0])
    rng = np.random.default_rng(int(seed))
    kl_true_model = np.zeros(k, dtype=np.float64)
    kl_model_true = np.zeros(k, dtype=np.float64)
    true_sampled_model_skl = np.zeros((k, k), dtype=np.float64)

    true_samples = [sample_native_class(means, variances, cls, int(samples_per_class), rng) for cls in range(k)]
    model_samples = [
        sample_model_class(
            model=model,
            theta=theta_eval[cls : cls + 1],
            n=int(samples_per_class),
            device=device,
            ode_steps=int(ode_steps),
            ode_method=str(ode_method),
        )
        for cls in range(k)
    ]

    for i in range(k):
        log_true_i_on_true = diagonal_gaussian_log_prob(true_samples[i], means, variances, i)
        log_model_i_on_true = model_log_prob(
            model=model,
            x=true_samples[i],
            theta=theta_eval[i : i + 1],
            device=device,
            ode_steps=int(ode_steps),
            ode_method=str(ode_method),
            batch_size=int(batch_size),
            solve_jitter=float(solve_jitter),
            quadrature_steps=int(quadrature_steps),
        )
        kl_true_model[i] = float(np.mean(log_true_i_on_true - log_model_i_on_true, dtype=np.float64))

        log_model_i_on_model = model_log_prob(
            model=model,
            x=model_samples[i],
            theta=theta_eval[i : i + 1],
            device=device,
            ode_steps=int(ode_steps),
            ode_method=str(ode_method),
            batch_size=int(batch_size),
            solve_jitter=float(solve_jitter),
            quadrature_steps=int(quadrature_steps),
        )
        log_true_i_on_model = diagonal_gaussian_log_prob(model_samples[i], means, variances, i)
        kl_model_true[i] = float(np.mean(log_model_i_on_model - log_true_i_on_model, dtype=np.float64))

        for j in range(k):
            if i == j:
                continue
            log_model_j_on_true = model_log_prob(
                model=model,
                x=true_samples[i],
                theta=theta_eval[j : j + 1],
                device=device,
                ode_steps=int(ode_steps),
                ode_method=str(ode_method),
                batch_size=int(batch_size),
                solve_jitter=float(solve_jitter),
                quadrature_steps=int(quadrature_steps),
            )
            true_sampled_model_skl[i, j] = float(np.mean(log_model_i_on_true - log_model_j_on_true, dtype=np.float64))

    true_sampled_model_skl = true_sampled_model_skl + true_sampled_model_skl.T
    true_sampled_model_skl = np.maximum(true_sampled_model_skl, 0.0)
    np.fill_diagonal(true_sampled_model_skl, 0.0)
    return kl_true_model, kl_model_true, kl_true_model + kl_model_true, true_sampled_model_skl


def pair_error_rows(
    *,
    gt_skl: np.ndarray,
    model_skl: np.ndarray,
    true_sampled_model_skl: np.ndarray,
    names: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, j in pair_indices(int(gt_skl.shape[0])):
        ii, jj = int(i), int(j)
        true_sampled = float(true_sampled_model_skl[ii, jj])
        rows.append(
            {
                "condition_i": names[ii],
                "condition_j": names[jj],
                "analytic_gt_jeffreys_skl": float(gt_skl[ii, jj]),
                "model_sampled_jeffreys_skl": float(model_skl[ii, jj]),
                "true_sampled_model_logratio_jeffreys_skl": true_sampled,
                "abs_error_model_sampled": float(abs(model_skl[ii, jj] - gt_skl[ii, jj])),
                "abs_error_true_sampled_model_logratio": float(abs(true_sampled - gt_skl[ii, jj])),
            }
        )
    return rows


def write_pair_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    columns = (
        "condition_i",
        "condition_j",
        "analytic_gt_jeffreys_skl",
        "model_sampled_jeffreys_skl",
        "true_sampled_model_logratio_jeffreys_skl",
        "abs_error_model_sampled",
        "abs_error_true_sampled_model_logratio",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_summary(path: Path, summary: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def maybe_write_endpoint_plot(
    *,
    output_dir: Path,
    true_samples_by_class: list[np.ndarray],
    model_samples_by_class: list[np.ndarray],
    native: bool,
    skip_plots: bool,
) -> Path | None:
    if bool(skip_plots) or not bool(native):
        return None
    x_dim = int(np.asarray(true_samples_by_class[0]).shape[1])
    if x_dim < 2:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    path = output_dir / "symmetric_kl_endpoint_samples_x0_x1.png"
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for cls, (true_x, model_x) in enumerate(zip(true_samples_by_class, model_samples_by_class)):
        color = colors[cls % len(colors)] if colors else None
        true_small = true_x[: min(500, int(true_x.shape[0]))]
        model_small = model_x[: min(500, int(model_x.shape[0]))]
        ax.scatter(true_small[:, 0], true_small[:, 1], s=8, alpha=0.25, color=color, marker="o")
        ax.scatter(model_small[:, 0], model_small[:, 1], s=8, alpha=0.25, color=color, marker="x")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title("MoG5 true endpoints (o) vs symmetric-KL flow endpoints (x)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run(args: argparse.Namespace) -> dict[str, Path]:
    validate_args(args)
    dev = require_device(str(args.device))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    dataset_dir = resolve_dataset_dir(args)
    output_dir = resolve_output_dir(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    native_npz, projected_npz = ensure_dataset(args, dataset_dir)
    pr_projected = args.pr_dim is not None
    work_npz = projected_npz if pr_projected else native_npz
    if work_npz is None:
        raise RuntimeError("Native mode did not produce a work NPZ.")

    native_bundle = load_shared_dataset_npz(native_npz)
    work_bundle = load_shared_dataset_npz(work_npz)
    native_meta = dict(native_bundle.meta)
    work_meta = dict(work_bundle.meta)
    if str(native_meta.get("dataset_family", "")) != "random_mog_categorical":
        raise ValueError("Expected native random_mog_categorical dataset.")
    k = int(work_meta.get("num_categories", NUM_CATEGORIES))
    if k != NUM_CATEGORIES:
        raise ValueError(f"Expected MoG5 num_categories=5, got {k}.")
    theta_eval = _theta_eval(k)
    names = tuple(f"category_{i}" for i in range(k))
    means, variances = _native_params(native_meta)
    gt_skl = analytic_diagonal_gaussian_skl_matrix(means, variances)

    theta_train = np.asarray(work_bundle.theta_train, dtype=np.float64)
    x_train = np.asarray(work_bundle.x_train, dtype=np.float64)
    theta_val = np.asarray(work_bundle.theta_validation, dtype=np.float64)
    x_val = np.asarray(work_bundle.x_validation, dtype=np.float64)
    theta_dim = int(theta_train.shape[1] if theta_train.ndim == 2 else 1)
    x_dim = int(x_train.shape[1] if x_train.ndim == 2 else 1)

    print("[mog5-skl-endpoint] training symmetric_kl flow velocity_family=nonlinear", flush=True)
    model = build_flow_skl_model(
        velocity_family="nonlinear",
        theta_dim=theta_dim,
        x_dim=x_dim,
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        quadrature_steps=int(args.quadrature_steps),
        divergence_estimator=str(args.divergence_estimator),
        hutchinson_probes=int(args.hutchinson_probes),
    ).to(dev)
    train_meta = train_flow_skl_model(
        model=model,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        device=dev,
        velocity_family="nonlinear",
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
    model_skl_result = estimate_model_symmetric_kl(
        model=model,
        theta_all=theta_eval,
        device=dev,
        velocity_family="nonlinear",
        mc_jeffreys_sample=int(args.mc_jeffreys_sample),
        ode_steps=int(args.ode_steps),
        ode_method=str(args.ode_method),
        batch_size=int(args.batch_size),
        solve_jitter=float(args.solve_jitter),
        quadrature_steps=int(args.quadrature_steps),
        train_metadata=train_meta,
    )
    model_skl = np.asarray(model_skl_result.symmetric_kl_matrix, dtype=np.float64)

    rng = np.random.default_rng(int(args.seed) + 101)
    endpoint_n = int(args.endpoint_samples_per_class)
    model_endpoint_samples = [
        sample_model_class(
            model=model,
            theta=theta_eval[cls : cls + 1],
            n=endpoint_n,
            device=dev,
            ode_steps=int(args.ode_steps),
            ode_method=str(args.ode_method),
        )
        for cls in range(k)
    ]
    if not pr_projected:
        true_endpoint_samples = [sample_native_class(means, variances, cls, endpoint_n, rng) for cls in range(k)]
        exact_logprob_available = True
    else:
        labels = labels_from_theta(work_bundle.theta_all, num_categories=k)
        true_endpoint_samples = []
        for cls in range(k):
            cls_x = np.asarray(work_bundle.x_all, dtype=np.float64)[labels == cls]
            replace = int(cls_x.shape[0]) < endpoint_n
            idx = rng.choice(int(cls_x.shape[0]), size=endpoint_n, replace=replace)
            true_endpoint_samples.append(cls_x[idx].astype(np.float64, copy=False))
        exact_logprob_available = False

    cat_diag = endpoint_sample_diagnostics(
        true_samples_by_class=true_endpoint_samples,
        model_samples_by_class=model_endpoint_samples,
        two_sample_max_points=int(args.two_sample_max_points),
        seed=int(args.seed) + 202,
    )

    if exact_logprob_available:
        kl_true_model, kl_model_true, self_jeffreys, true_sampled_model_skl = exact_native_kl_diagnostics(
            model=model,
            theta_eval=theta_eval,
            means=means,
            variances=variances,
            device=dev,
            samples_per_class=int(args.logprob_samples_per_class),
            seed=int(args.seed) + 303,
            ode_steps=int(args.ode_steps),
            ode_method=str(args.ode_method),
            batch_size=int(args.batch_size),
            solve_jitter=float(args.solve_jitter),
            quadrature_steps=int(args.quadrature_steps),
        )
    else:
        kl_true_model = np.full(k, np.nan, dtype=np.float64)
        kl_model_true = np.full(k, np.nan, dtype=np.float64)
        self_jeffreys = np.full(k, np.nan, dtype=np.float64)
        true_sampled_model_skl = np.full((k, k), np.nan, dtype=np.float64)

    rows = pair_error_rows(
        gt_skl=gt_skl,
        model_skl=model_skl,
        true_sampled_model_skl=true_sampled_model_skl,
        names=names,
    )
    pair_csv = write_pair_csv(output_dir / PAIR_ERRORS_CSV_NAME, rows)
    plot_path = maybe_write_endpoint_plot(
        output_dir=output_dir,
        true_samples_by_class=true_endpoint_samples,
        model_samples_by_class=model_endpoint_samples,
        native=not pr_projected,
        skip_plots=bool(args.skip_plots),
    )

    results_npz = output_dir / RESULTS_NPZ_NAME
    np.savez_compressed(
        results_npz,
        theta_eval=theta_eval,
        pair_indices=pair_indices(k),
        analytic_gt_jeffreys_skl_matrix=gt_skl,
        model_sampled_jeffreys_skl_matrix=model_skl,
        true_sampled_model_logratio_jeffreys_skl_matrix=true_sampled_model_skl,
        abs_error_model_sampled_matrix=np.abs(model_skl - gt_skl),
        abs_error_true_sampled_model_logratio_matrix=np.abs(true_sampled_model_skl - gt_skl),
        kl_true_model_by_category=kl_true_model,
        kl_model_true_by_category=kl_model_true,
        self_jeffreys_by_category=self_jeffreys,
        train_losses=np.asarray(train_meta.get("train_losses", []), dtype=np.float64),
        val_losses=np.asarray(train_meta.get("val_losses", []), dtype=np.float64),
        val_monitor_losses=np.asarray(train_meta.get("val_monitor_losses", []), dtype=np.float64),
        best_epoch=np.asarray([int(train_meta.get("best_epoch", 0))], dtype=np.int64),
        best_val_loss=np.asarray([float(train_meta.get("best_val_loss", np.nan))], dtype=np.float64),
        **cat_diag,
    )

    summary = {
        "script": "bin/diagnose_mog5_symmetric_kl_endpoint.py",
        "device": str(dev),
        "velocity_family": "nonlinear",
        "metric": "symmetric_kl",
        "n_total": int(args.n_total),
        "native_x_dim": int(args.native_x_dim),
        "pr_projected": bool(pr_projected),
        "pr_dim": None if args.pr_dim is None else int(args.pr_dim),
        "seed": int(args.seed),
        "dataset_dir": str(dataset_dir.resolve()),
        "native_npz": str(Path(native_npz).resolve()),
        "work_npz": str(Path(work_npz).resolve()),
        "projected_npz": None if projected_npz is None else str(Path(projected_npz).resolve()),
        "output_dir": str(output_dir),
        "results_npz": str(results_npz),
        "pair_errors_csv": str(pair_csv),
        "endpoint_plot": None if plot_path is None else str(plot_path),
        "endpoint_samples_per_class": int(args.endpoint_samples_per_class),
        "logprob_samples_per_class": int(args.logprob_samples_per_class),
        "exact_logprob_available": bool(exact_logprob_available),
        "exact_logprob_unavailable_reason": None
        if exact_logprob_available
        else "PR-projected density has no exact analytic log-prob in this diagnostic.",
        "best_epoch": int(train_meta.get("best_epoch", 0)),
        "best_val_loss": float(train_meta.get("best_val_loss", np.nan)),
        "stopped_epoch": int(train_meta.get("stopped_epoch", 0)),
        "stopped_early": bool(train_meta.get("stopped_early", False)),
        "max_abs_error_model_sampled": float(np.nanmax(np.abs(model_skl - gt_skl))),
        "max_abs_error_true_sampled_model_logratio": float(np.nanmax(np.abs(true_sampled_model_skl - gt_skl)))
        if exact_logprob_available
        else None,
    }
    summary_json = write_summary(output_dir / SUMMARY_JSON_NAME, summary)

    print(f"results_npz: {results_npz}", flush=True)
    print(f"pair_errors_csv: {pair_csv}", flush=True)
    print(f"summary_json: {summary_json}", flush=True)
    if plot_path is not None:
        print(f"endpoint_plot: {plot_path}", flush=True)
    return {"output_dir": output_dir, "results_npz": results_npz, "pair_errors_csv": pair_csv, "summary_json": summary_json}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
