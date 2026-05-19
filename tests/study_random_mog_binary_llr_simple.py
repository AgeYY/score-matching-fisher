#!/usr/bin/env python3
"""Simple binary random_mog_categorical LLR diagnostic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import global_setting  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from fisher.ctsm_models import ToyBinaryTimeScoreNet, ToyLatentBeliefBinaryTimeScoreNet
from fisher.data import ToyCategoricalRandomMoGDataset
from fisher.dataset_visualization import pca_project
from fisher.pr_autoencoder_embedding import pr_autoencoder_config_from_namespace, project_x_through_pr_autoencoder
from fisher.shared_fisher_est import (
    estimate_binary_ctsm_v_log_ratio,
    estimate_latent_belief_ctsm_v_binary_log_ratio,
    require_device,
    train_binary_ctsm_v_model,
    train_latent_belief_binary_ctsm_v_model,
    train_latent_belief_binary_ctsm_v_inner_post_model,
)

_METHOD_ALIASES = {
    "ctsm-v-binary": "ctsm_v_binary",
    "ctsm_v_binary": "ctsm_v_binary",
    "latent-belief-ctsm-v-binary": "latent_belief_ctsm_v_binary",
    "latent_belief_ctsm_v_binary": "latent_belief_ctsm_v_binary",
    "latent-belief-ctsm-v-binary-inner-post": "latent_belief_ctsm_v_binary_inner_post",
    "latent_belief_ctsm_v_binary_inner_post": "latent_belief_ctsm_v_binary_inner_post",
    "latent-belief-ctsm-v-binary-innner-post": "latent_belief_ctsm_v_binary_inner_post",
    "latent_belief_ctsm_v_binary_innner_post": "latent_belief_ctsm_v_binary_inner_post",
}


def normalize_method(value: str) -> str:
    key = str(value).strip().lower()
    try:
        return _METHOD_ALIASES[key]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unknown method {value!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=_repo_root / "data" / "random_mog_binary_llr_simple")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--x-dim", type=int, default=2)
    p.add_argument("--num-categories", type=int, default=2)
    p.add_argument("--theta-dim", type=int, default=2)
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-val", type=int, default=300)
    p.add_argument("--n-test-per-class", type=int, default=400)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--clf-max-iter", type=int, default=1000)
    p.add_argument("--method", type=normalize_method, default="ctsm_v_binary")
    p.add_argument("--ctsm-binary-epochs", type=int, default=50000)
    p.add_argument("--ctsm-batch-size", type=int, default=256)
    p.add_argument("--ctsm-lr", type=float, default=2e-3)
    p.add_argument("--ctsm-hidden-dim", type=int, default=128)
    p.add_argument("--ctsm-weight-decay", type=float, default=0.0)
    p.add_argument("--ctsm-two-sb-var", type=float, default=2.0)
    p.add_argument("--ctsm-path-schedule", type=str, default="linear", choices=["linear", "cosine"])
    p.add_argument("--ctsm-path-eps", type=float, default=1e-12)
    p.add_argument("--ctsm-factor", type=float, default=1.0)
    p.add_argument("--ctsm-t-eps", type=float, default=0.01)
    p.add_argument("--ctsm-int-n-time", type=int, default=300)
    p.add_argument("--latent-h-dim", type=int, default=4)
    p.add_argument("--latent-n-mc-train", type=int, default=32)
    p.add_argument("--latent-n-mc-val", type=int, default=8)
    p.add_argument("--latent-n-mc-eval", type=int, default=16)
    p.add_argument("--latent-n-posterior-pairs", type=int, default=1)
    p.add_argument("--latent-precision-eps", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--early-stopping-patience", type=int, default=1000)
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    p.add_argument("--early-stopping-ema-alpha", type=float, default=0.05)
    p.add_argument("--no-restore-best", dest="restore_best", action="store_false")
    p.set_defaults(restore_best=True)
    p.add_argument("--mog-a-low", type=float, default=0.2)
    p.add_argument("--mog-a-high", type=float, default=2.0)
    p.add_argument("--mog-sigma-base", type=float, default=0.15)
    p.add_argument("--mog-alpha", type=float, default=0.15)
    p.add_argument("--mog-eps", type=float, default=1e-5)
    p.add_argument("--mog-mean-min-dist", type=float, default=-1.0)
    p.add_argument("--mog-mean-max-attempts", type=int, default=10000)
    p.add_argument("--pr-project", action="store_true")
    p.add_argument("--pr-dim", type=int, default=10)
    p.add_argument("--pr-use-cache", action="store_true")
    p.add_argument("--pr-cache-dir", type=str, default="data/pr_autoencoder_cache")
    p.add_argument("--pr-train-epochs", type=int, default=None)
    p.add_argument("--pr-train-samples", type=int, default=None)
    p.add_argument("--pr-train-batch-size", type=int, default=None)
    p.add_argument("--pr-train-lr", type=float, default=None)
    p.add_argument("--pr-lambda-pr", type=float, default=None)
    p.add_argument("--pr-eps", type=float, default=None)
    p.add_argument("--pr-hidden1", type=int, default=None)
    p.add_argument("--pr-hidden2", type=int, default=None)
    return p


def labels_from_one_hot(theta: np.ndarray) -> np.ndarray:
    arr = np.asarray(theta)
    if arr.ndim != 2:
        raise ValueError("theta must be a one-hot matrix.")
    return np.argmax(arr, axis=1).astype(np.int64)


def one_hot(labels: np.ndarray, num_categories: int = 2) -> np.ndarray:
    lab = np.asarray(labels, dtype=np.int64).reshape(-1)
    return np.eye(int(num_categories), dtype=np.float64)[lab]


def sample_per_class(
    dataset: ToyCategoricalRandomMoGDataset,
    n_per_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.repeat(np.arange(dataset.num_categories, dtype=np.int64), int(n_per_class))
    theta = one_hot(labels, dataset.num_categories)
    x = dataset.sample_x(theta)
    return theta, x, labels


def analytic_binary_llr(x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    mu = np.asarray(means, dtype=np.float64)
    var = np.asarray(variances, dtype=np.float64)
    if x_arr.ndim != 2:
        raise ValueError("x must have shape (N, d).")
    if mu.shape[0] < 2 or var.shape[0] < 2 or mu.shape != var.shape or mu.shape[1] != x_arr.shape[1]:
        raise ValueError("means and variances must have shape (K >= 2, d) matching x.")
    logp = []
    for k in (0, 1):
        delta = x_arr - mu[k]
        quad = np.sum((delta**2) / var[k], axis=1)
        logdet = np.sum(np.log(var[k]))
        logp.append(-0.5 * (x_arr.shape[1] * np.log(2.0 * np.pi) + logdet + quad))
    return np.asarray(logp[1] - logp[0], dtype=np.float64)


def regression_metrics(est: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    e = np.asarray(est, dtype=np.float64).reshape(-1)
    g = np.asarray(gt, dtype=np.float64).reshape(-1)
    if e.shape != g.shape:
        raise ValueError("est and gt must have matching shape.")
    rmse = float(np.sqrt(np.mean((e - g) ** 2)))
    corr = float(np.corrcoef(e, g)[0, 1]) if e.size > 1 and np.std(e) > 0 and np.std(g) > 0 else float("nan")
    return {"rmse": rmse, "corr": corr}


def _hist_bins_for(values: np.ndarray, bins: int) -> np.ndarray | int:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return int(bins)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        center = 0.0 if not np.isfinite(lo) else lo
        return np.linspace(center - 0.5, center + 0.5, int(bins) + 1)
    return int(bins)


def _metrics_label(name: str, metrics: dict[str, float]) -> str:
    return f"{name}: RMSE={metrics['rmse']:.3g}, corr={metrics['corr']:.3f}"


def _scatter_projection(
    *,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str, str]:
    train = np.asarray(x_train, dtype=np.float64)
    val = np.asarray(x_val, dtype=np.float64)
    test = np.asarray(x_test, dtype=np.float64)
    mu = np.asarray(means, dtype=np.float64)
    if train.ndim != 2 or val.ndim != 2 or test.ndim != 2 or mu.ndim != 2:
        raise ValueError("scatter inputs must be 2D arrays.")
    if train.shape[1] == 2 and val.shape[1] == 2 and test.shape[1] == 2 and mu.shape[1] == 2:
        return train, val, test, mu, "x0", "x1", "dataset"
    all_x = np.vstack([train, val, test])
    proj, center, basis = pca_project(all_x, n_components=2)
    n_train = int(train.shape[0])
    n_val = int(val.shape[0])
    train_proj = proj[:n_train]
    val_proj = proj[n_train : n_train + n_val]
    test_proj = proj[n_train + n_val :]
    if int(mu.shape[1]) != int(center.shape[0]):
        raise ValueError("component means must live in the same feature space as plotted samples.")
    means_proj = (mu - center.reshape(1, -1)) @ basis
    return train_proj, val_proj, test_proj, means_proj, "PC1", "PC2", f"dataset PCA projection (x_dim={train.shape[1]})"


def _pr_config_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        pr_autoencoder_z_dim=int(args.x_dim),
        pr_autoencoder_hidden1=100 if args.pr_hidden1 is None else int(args.pr_hidden1),
        pr_autoencoder_hidden2=200 if args.pr_hidden2 is None else int(args.pr_hidden2),
        pr_autoencoder_train_samples=12000 if args.pr_train_samples is None else int(args.pr_train_samples),
        pr_autoencoder_train_epochs=200 if args.pr_train_epochs is None else int(args.pr_train_epochs),
        pr_autoencoder_train_batch_size=512 if args.pr_train_batch_size is None else int(args.pr_train_batch_size),
        pr_autoencoder_train_lr=1e-3 if args.pr_train_lr is None else float(args.pr_train_lr),
        pr_autoencoder_lambda_pr=1e-2 if args.pr_lambda_pr is None else float(args.pr_lambda_pr),
        pr_autoencoder_pr_eps=1e-8 if args.pr_eps is None else float(args.pr_eps),
    )


@torch.no_grad()
def _encode_with_pr_model(model: torch.nn.Module, x: np.ndarray, *, device: torch.device) -> np.ndarray:
    x_t = torch.from_numpy(np.asarray(x, dtype=np.float64)).to(device=device, dtype=torch.float32)
    model.eval()
    h_t, _ = model(x_t)
    return h_t.detach().cpu().numpy().astype(np.float64, copy=False)


def prepare_work_features(
    args: argparse.Namespace,
    *,
    device: torch.device,
    x_train_native: np.ndarray,
    x_val_native: np.ndarray,
    x_test_native: np.ndarray,
    means_native: np.ndarray,
) -> dict[str, Any]:
    native_x_dim = int(np.asarray(x_train_native).shape[1])
    if not bool(args.pr_project):
        return {
            "x_train": np.asarray(x_train_native, dtype=np.float64),
            "x_validation": np.asarray(x_val_native, dtype=np.float64),
            "x_test": np.asarray(x_test_native, dtype=np.float64),
            "means": np.asarray(means_native, dtype=np.float64),
            "x_dim": native_x_dim,
            "pr_projected": False,
            "pr_dim": native_x_dim,
            "pr_cache_run_dir": "",
            "pr_loaded_from_cache": False,
            "pr_train_loss": np.asarray([], dtype=np.float64),
            "pr_train_recon": np.asarray([], dtype=np.float64),
            "pr_train_pr": np.asarray([], dtype=np.float64),
        }
    if int(args.pr_dim) <= native_x_dim:
        raise ValueError(f"--pr-dim must exceed native x_dim={native_x_dim}; got {args.pr_dim}.")
    cfg = pr_autoencoder_config_from_namespace(_pr_config_namespace(args), h_dim=int(args.pr_dim))
    x_all_native = np.vstack(
        [
            np.asarray(x_train_native, dtype=np.float64),
            np.asarray(x_val_native, dtype=np.float64),
            np.asarray(x_test_native, dtype=np.float64),
        ]
    )
    x_all_work, cache_run_dir, loaded_from_cache, metrics, model = project_x_through_pr_autoencoder(
        x_all_native,
        config=cfg,
        seed=int(args.seed),
        device=device,
        cache_dir=str(args.pr_cache_dir),
        force_retrain=not bool(args.pr_use_cache),
    )
    n_train = int(np.asarray(x_train_native).shape[0])
    n_val = int(np.asarray(x_val_native).shape[0])
    means_work = _encode_with_pr_model(model, np.asarray(means_native, dtype=np.float64), device=device)
    return {
        "x_train": x_all_work[:n_train],
        "x_validation": x_all_work[n_train : n_train + n_val],
        "x_test": x_all_work[n_train + n_val :],
        "means": means_work,
        "x_dim": int(args.pr_dim),
        "pr_projected": True,
        "pr_dim": int(args.pr_dim),
        "pr_cache_run_dir": str(Path(cache_run_dir).resolve()),
        "pr_loaded_from_cache": bool(loaded_from_cache),
        "pr_train_loss": np.asarray(metrics.get("loss", []), dtype=np.float64),
        "pr_train_recon": np.asarray(metrics.get("recon", []), dtype=np.float64),
        "pr_train_pr": np.asarray(metrics.get("pr", []), dtype=np.float64),
    }


def train_binary_classifier_llr(
    x_train: np.ndarray,
    labels_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
    max_iter: int,
) -> np.ndarray:
    labels = np.asarray(labels_train, dtype=np.int64).reshape(-1)
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("binary classifier expects labels 0/1.")
    clf = LogisticRegression(solver="lbfgs", random_state=int(seed), max_iter=int(max_iter))
    clf.fit(np.asarray(x_train, dtype=np.float64), labels)
    n0 = int(np.sum(labels == 0))
    n1 = int(np.sum(labels == 1))
    if n0 < 1 or n1 < 1:
        raise ValueError("binary classifier requires both classes in training data.")
    prior_log_odds = float(np.log(float(n1) / float(n0)))
    return np.asarray(clf.decision_function(np.asarray(x_test, dtype=np.float64)), dtype=np.float64) - prior_log_odds


def save_diagnostic_figure(
    *,
    output_base: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    means: np.ndarray,
    ctsm_train_losses: np.ndarray,
    ctsm_val_losses: np.ndarray,
    gt_llr: np.ndarray,
    binary_llr: np.ndarray,
    ctsm_llr: np.ndarray,
    ctsm_label: str = "ctsm_v_binary",
) -> tuple[Path, Path]:
    binary_metrics = regression_metrics(binary_llr, gt_llr)
    ctsm_metrics = regression_metrics(ctsm_llr, gt_llr)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    ax = axes[0, 0]
    x_train_plot, x_val_plot, x_test_plot, means_plot, xlabel, ylabel, dataset_title = _scatter_projection(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        means=means,
    )
    for name, x, y, marker, alpha in (
        ("train", x_train_plot, y_train, "o", 0.45),
        ("validation", x_val_plot, y_val, "^", 0.65),
        ("test", x_test_plot, y_test, ".", 0.25),
    ):
        for cls in (0, 1):
            m = np.asarray(y) == cls
            ax.scatter(np.asarray(x)[m, 0], np.asarray(x)[m, 1], s=14, marker=marker, alpha=alpha, label=f"{name} c{cls}")
    ax.scatter(means_plot[:2, 0], means_plot[:2, 1], s=110, c=["tab:blue", "tab:orange"], marker="X", edgecolor="black", label="means")
    ax.set_title(dataset_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    if len(ctsm_train_losses):
        ax.plot(np.arange(1, len(ctsm_train_losses) + 1), ctsm_train_losses, label="train")
    if len(ctsm_val_losses):
        ax.plot(np.arange(1, len(ctsm_val_losses) + 1), ctsm_val_losses, label="validation")
    ax.set_title("CTSM-v-binary loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()

    ax = axes[0, 2]
    ax.scatter(gt_llr, binary_llr, s=13, alpha=0.55, label="binary_classifier")
    ax.scatter(gt_llr, ctsm_llr, s=13, alpha=0.55, label=ctsm_label)
    lo = float(np.nanmin([np.min(gt_llr), np.min(binary_llr), np.min(ctsm_llr)]))
    hi = float(np.nanmax([np.max(gt_llr), np.max(binary_llr), np.max(ctsm_llr)]))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_title("estimated LLR vs analytic GT")
    ax.set_xlabel("analytic GT LLR")
    ax.set_ylabel("estimated LLR")
    ax.legend()
    ax.text(
        0.03,
        0.97,
        "\n".join(
            [
                _metrics_label("binary_classifier", binary_metrics),
                _metrics_label(ctsm_label, ctsm_metrics),
            ]
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 4},
    )

    ax = axes[1, 0]
    ax.scatter(binary_llr, ctsm_llr, s=13, alpha=0.55)
    lo = float(np.nanmin([np.min(binary_llr), np.min(ctsm_llr)]))
    hi = float(np.nanmax([np.max(binary_llr), np.max(ctsm_llr)]))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_title(f"binary_classifier vs {ctsm_label}")
    ax.set_xlabel("binary_classifier LLR")
    ax.set_ylabel(f"{ctsm_label} LLR")

    err = np.asarray(ctsm_llr, dtype=np.float64) - np.asarray(gt_llr, dtype=np.float64)
    ax = axes[1, 1]
    sc = ax.scatter(x_test_plot[:, 0], x_test_plot[:, 1], c=err, s=14, cmap="coolwarm")
    ax.scatter(means_plot[:2, 0], means_plot[:2, 1], s=110, c="none", marker="X", edgecolor="black")
    ax.set_title("CTSM residual")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 2]
    bins = max(10, min(60, int(np.sqrt(len(err)))))
    ax.hist(binary_llr - gt_llr, bins=_hist_bins_for(binary_llr - gt_llr, bins), alpha=0.55, label="binary_classifier")
    ax.hist(err, bins=_hist_bins_for(err, bins), alpha=0.55, label=ctsm_label)
    ax.set_title("LLR residuals")
    ax.set_xlabel("estimated - GT")
    ax.legend()
    ax.text(
        0.03,
        0.97,
        "\n".join(
            [
                _metrics_label("binary_classifier", binary_metrics),
                _metrics_label(ctsm_label, ctsm_metrics),
            ]
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 4},
    )

    fig.tight_layout()
    svg_path = output_base.with_suffix(".svg").resolve()
    png_path = output_base.with_suffix(".png").resolve()
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


def write_summary(
    *,
    path: Path,
    output_dir: Path,
    results_npz: Path,
    figure_svg: Path,
    figure_png: Path,
    binary_metrics: dict[str, float],
    ctsm_metrics: dict[str, float],
    ctsm_label: str = "ctsm_v_binary",
) -> Path:
    path = path.resolve()
    with path.open("w", encoding="utf-8") as f:
        f.write("study_random_mog_binary_llr_simple\n")
        f.write(f"output_dir: {output_dir.resolve()}\n")
        f.write(f"results_npz: {results_npz.resolve()}\n")
        f.write(f"simple_binary_llr_diagnostic.svg: {figure_svg.resolve()}\n")
        f.write(f"simple_binary_llr_diagnostic.png: {figure_png.resolve()}\n")
        f.write(f"binary_classifier_rmse: {binary_metrics['rmse']:.8g}\n")
        f.write(f"binary_classifier_corr: {binary_metrics['corr']:.8g}\n")
        f.write(f"{ctsm_label}_rmse: {ctsm_metrics['rmse']:.8g}\n")
        f.write(f"{ctsm_label}_corr: {ctsm_metrics['corr']:.8g}\n")
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.num_categories) != 2:
        raise ValueError("This diagnostic is intentionally binary: --num-categories must be 2.")
    dev = require_device(str(args.device))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = ToyCategoricalRandomMoGDataset(
        num_categories=2,
        x_dim=int(args.x_dim),
        theta_dim=int(args.theta_dim),
        mog_a_low=float(args.mog_a_low),
        mog_a_high=float(args.mog_a_high),
        mog_sigma_base=float(args.mog_sigma_base),
        mog_alpha=float(args.mog_alpha),
        mog_eps=float(args.mog_eps),
        mog_mean_min_dist=float(args.mog_mean_min_dist),
        mog_mean_max_attempts=int(args.mog_mean_max_attempts),
        seed=int(args.seed),
    )
    theta_train, x_train_native = dataset.sample_joint(int(args.n_train))
    theta_val, x_val_native = dataset.sample_joint(int(args.n_val))
    theta_test, x_test_native, y_test = sample_per_class(dataset, int(args.n_test_per_class))
    y_train = labels_from_one_hot(theta_train)
    y_val = labels_from_one_hot(theta_val)
    work = prepare_work_features(
        args,
        device=dev,
        x_train_native=x_train_native,
        x_val_native=x_val_native,
        x_test_native=x_test_native,
        means_native=dataset._mog_means,
    )
    x_train = np.asarray(work["x_train"], dtype=np.float64)
    x_val = np.asarray(work["x_validation"], dtype=np.float64)
    x_test = np.asarray(work["x_test"], dtype=np.float64)
    means_work = np.asarray(work["means"], dtype=np.float64)
    work_x_dim = int(work["x_dim"])

    gt_llr = analytic_binary_llr(x_test_native, dataset._mog_means, dataset._mog_variances)
    binary_llr = train_binary_classifier_llr(
        x_train,
        y_train,
        x_test,
        seed=int(args.seed),
        max_iter=int(args.clf_max_iter),
    )

    train0 = np.flatnonzero(y_train == 0)
    train1 = np.flatnonzero(y_train == 1)
    val0 = np.flatnonzero(y_val == 0)
    val1 = np.flatnonzero(y_val == 1)
    method = str(args.method)
    if method in {"latent_belief_ctsm_v_binary", "latent_belief_ctsm_v_binary_inner_post"}:
        model = ToyLatentBeliefBinaryTimeScoreNet(
            dim=work_x_dim,
            h_dim=int(args.latent_h_dim),
            hidden_dim=int(args.ctsm_hidden_dim),
            precision_eps=float(args.latent_precision_eps),
        ).to(dev)
        train_fn = (
            train_latent_belief_binary_ctsm_v_inner_post_model
            if method == "latent_belief_ctsm_v_binary_inner_post"
            else train_latent_belief_binary_ctsm_v_model
        )
        train_kwargs = {}
        if method == "latent_belief_ctsm_v_binary_inner_post":
            train_kwargs["latent_n_mc_train"] = int(args.latent_n_mc_train)
        else:
            train_kwargs["n_posterior_pairs"] = int(args.latent_n_posterior_pairs)
        train_out = train_fn(
            model=model,
            x0_train=x_train[train0],
            x1_train=x_train[train1],
            epochs=int(args.ctsm_binary_epochs),
            batch_size=int(args.ctsm_batch_size),
            lr=float(args.ctsm_lr),
            weight_decay=float(args.ctsm_weight_decay),
            device=dev,
            log_every=max(1, int(args.log_every)),
            two_sb_var=float(args.ctsm_two_sb_var),
            path_schedule=str(args.ctsm_path_schedule),
            path_eps=float(args.ctsm_path_eps),
            factor=float(args.ctsm_factor),
            t_eps=float(args.ctsm_t_eps),
            x0_val=x_val[val0],
            x1_val=x_val[val1],
            early_stopping_patience=int(args.early_stopping_patience),
            early_stopping_min_delta=float(args.early_stopping_min_delta),
            early_stopping_ema_alpha=float(args.early_stopping_ema_alpha),
            restore_best=bool(args.restore_best),
            n_mc_val=int(args.latent_n_mc_val),
            **train_kwargs,
        )
        ctsm_llr = estimate_latent_belief_ctsm_v_binary_log_ratio(
            model,
            x_test,
            device=dev,
            batch_size=int(args.ctsm_batch_size),
            eps1=float(args.ctsm_t_eps),
            eps2=float(args.ctsm_t_eps),
            n_time=int(args.ctsm_int_n_time),
            n_mc_eval=int(args.latent_n_mc_eval),
        )
    else:
        model = ToyBinaryTimeScoreNet(dim=work_x_dim, hidden_dim=int(args.ctsm_hidden_dim)).to(dev)
        train_out = train_binary_ctsm_v_model(
            model=model,
            x0_train=x_train[train0],
            x1_train=x_train[train1],
            epochs=int(args.ctsm_binary_epochs),
            batch_size=int(args.ctsm_batch_size),
            lr=float(args.ctsm_lr),
            weight_decay=float(args.ctsm_weight_decay),
            device=dev,
            log_every=max(1, int(args.log_every)),
            two_sb_var=float(args.ctsm_two_sb_var),
            path_schedule=str(args.ctsm_path_schedule),
            path_eps=float(args.ctsm_path_eps),
            factor=float(args.ctsm_factor),
            t_eps=float(args.ctsm_t_eps),
            x0_val=x_val[val0],
            x1_val=x_val[val1],
            early_stopping_patience=int(args.early_stopping_patience),
            early_stopping_min_delta=float(args.early_stopping_min_delta),
            early_stopping_ema_alpha=float(args.early_stopping_ema_alpha),
            restore_best=bool(args.restore_best),
        )
        ctsm_llr = estimate_binary_ctsm_v_log_ratio(
            model,
            x_test,
            device=dev,
            batch_size=int(args.ctsm_batch_size),
            eps1=float(args.ctsm_t_eps),
            eps2=float(args.ctsm_t_eps),
            n_time=int(args.ctsm_int_n_time),
        )

    binary_metrics = regression_metrics(binary_llr, gt_llr)
    ctsm_metrics = regression_metrics(ctsm_llr, gt_llr)
    results_npz = (out_dir / "simple_binary_llr_results.npz").resolve()
    extra_llr_payload = {}
    if method != "ctsm_v_binary":
        extra_llr_payload[f"{method}_llr"] = ctsm_llr
    np.savez_compressed(
        results_npz,
        theta_train=theta_train,
        x_train_native=x_train_native,
        x_train=x_train,
        y_train=y_train,
        theta_validation=theta_val,
        x_validation_native=x_val_native,
        x_validation=x_val,
        y_validation=y_val,
        theta_test=theta_test,
        x_test_native=x_test_native,
        x_test=x_test,
        y_test=y_test,
        mog_component_gains=dataset._mog_gains,
        mog_component_means=dataset._mog_means,
        mog_component_variances=dataset._mog_variances,
        gt_llr=gt_llr,
        binary_classifier_llr=binary_llr,
        ctsm_v_binary_llr=ctsm_llr,
        **extra_llr_payload,
        ctsm_train_losses=np.asarray(train_out["train_losses"], dtype=np.float64),
        ctsm_val_losses=np.asarray(train_out["val_losses"], dtype=np.float64),
        ctsm_val_monitor_losses=np.asarray(train_out["val_monitor_losses"], dtype=np.float64),
        method=np.asarray(method, dtype=np.str_),
        latent_h_dim=np.int64(int(args.latent_h_dim)),
        latent_n_mc_train=np.int64(int(args.latent_n_mc_train)),
        latent_n_mc_val=np.int64(int(args.latent_n_mc_val)),
        latent_n_mc_eval=np.int64(int(args.latent_n_mc_eval)),
        latent_n_posterior_pairs=np.int64(int(args.latent_n_posterior_pairs)),
        latent_precision_eps=np.float64(float(args.latent_precision_eps)),
        pr_projected=np.bool_(bool(work["pr_projected"])),
        pr_dim=np.int64(int(work["pr_dim"])),
        native_x_dim=np.int64(int(args.x_dim)),
        pr_cache_run_dir=np.asarray(str(work["pr_cache_run_dir"]), dtype=np.str_),
        pr_loaded_from_cache=np.bool_(bool(work["pr_loaded_from_cache"])),
        pr_train_loss=np.asarray(work["pr_train_loss"], dtype=np.float64),
        pr_train_recon=np.asarray(work["pr_train_recon"], dtype=np.float64),
        pr_train_pr=np.asarray(work["pr_train_pr"], dtype=np.float64),
        seed=int(args.seed),
        x_dim=work_x_dim,
        num_categories=2,
    )
    fig_svg, fig_png = save_diagnostic_figure(
        output_base=out_dir / "simple_binary_llr_diagnostic",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        means=means_work,
        ctsm_train_losses=np.asarray(train_out["train_losses"], dtype=np.float64),
        ctsm_val_losses=np.asarray(train_out["val_losses"], dtype=np.float64),
        gt_llr=gt_llr,
        binary_llr=binary_llr,
        ctsm_llr=ctsm_llr,
        ctsm_label=method,
    )
    summary = write_summary(
        path=out_dir / "simple_binary_llr_summary.txt",
        output_dir=out_dir,
        results_npz=results_npz,
        figure_svg=fig_svg,
        figure_png=fig_png,
        binary_metrics=binary_metrics,
        ctsm_metrics=ctsm_metrics,
        ctsm_label=method,
    )
    print(f"Saved results: {results_npz}", flush=True)
    print(f"Saved figure: {fig_svg}", flush=True)
    print(f"Saved figure: {fig_png}", flush=True)
    print(f"Saved summary: {summary}", flush=True)
    return {
        "results_npz": results_npz,
        "figure_svg": fig_svg,
        "figure_png": fig_png,
        "summary": summary,
        "binary_metrics": binary_metrics,
        "ctsm_metrics": ctsm_metrics,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
