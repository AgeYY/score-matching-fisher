#!/usr/bin/env python3
"""Embed a low-dimensional shared dataset NPZ into higher-dimensional x via PR-autoencoder.

Typical workflow:

1. ``python bin/make_dataset.py --dataset-family randamp_gaussian_sqrtd --x-dim 2 ...``
2. ``python bin/project_dataset_pr_autoencoder.py --input-npz <low.npz> --output-npz <high.npz> --h-dim 10``

For ``random_mog_categorical`` (or any non-``randamp_gaussian_sqrtd`` family), add
``--allow-non-randamp-sqrtd``. The embedded NPZ keeps one-hot ``theta`` unchanged; only ``x`` is lifted.

The output archive keeps the source ``dataset_family`` and ``theta`` arrays, sets ``meta['x_dim']``
to the embedded dimension ``h_dim``, and sets ``pr_autoencoder_embedded=True`` with
``pr_autoencoder_z_dim`` equal to the source latent dimension. Ground-truth helpers that need the
low-dimensional generative model use :func:`fisher.shared_fisher_est.build_dataset_from_meta`.

Unless ``--skip-viz``, writes ``pr_projection_summary.{png,svg}`` next to ``--output-npz``:
same two-panel native layout as ``make_dataset.py`` (tuning curves, binned empirical mean of
embedded samples with scatter overlaid in PCA), plus a third panel with PR-autoencoder training
loss vs epoch. The PCA scatter uses at most ``--pr-viz-scatter-max`` points (default 400).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import global_setting  # noqa: F401  # matplotlib rc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fisher.dataset_visualization import plot_joint_and_tuning_on_axes
from fisher.pr_autoencoder_embedding import pr_autoencoder_config_from_namespace, project_x_through_pr_autoencoder
from fisher.shared_dataset_io import load_shared_dataset_npz, save_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_meta


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _categorical_labels_from_theta(theta: np.ndarray, *, num_categories: int) -> np.ndarray:
    arr = np.asarray(theta)
    k = int(num_categories)
    if k < 2:
        raise ValueError("Adversarial categorical PR projection requires num_categories >= 2.")
    if arr.ndim == 2 and int(arr.shape[1]) == k:
        vals = np.asarray(arr, dtype=np.float64)
        row_sums = vals.sum(axis=1)
        is_binary = np.all((np.abs(vals) <= 1e-6) | (np.abs(vals - 1.0) <= 1e-6), axis=1)
        if np.any(np.abs(row_sums - 1.0) > 1e-6) or not bool(np.all(is_binary)):
            raise ValueError("Adversarial categorical PR projection requires one-hot categorical theta rows.")
        return np.argmax(vals, axis=1).astype(np.int64)
    vals = np.asarray(arr, dtype=np.float64).reshape(-1)
    labels = np.rint(vals).astype(np.int64)
    if np.any(np.abs(vals - labels.astype(np.float64)) > 1e-6):
        raise ValueError("Adversarial categorical PR projection requires integer categorical theta labels.")
    if np.any((labels < 0) | (labels >= k)):
        raise ValueError(f"Categorical labels must be in [0, {k - 1}].")
    return labels


def _stratified_label_subsample(labels: np.ndarray, *, n_samples: int, seed: int) -> np.ndarray:
    lab = np.asarray(labels, dtype=np.int64).reshape(-1)
    n = int(lab.shape[0])
    m = int(n_samples)
    if m <= 0 or m >= n:
        return np.arange(n, dtype=np.int64)
    classes = np.unique(lab)
    if m < int(classes.shape[0]):
        raise ValueError(
            f"--pr-adv-train-samples={m} is smaller than the number of observed classes ({classes.shape[0]})."
        )
    rng = np.random.default_rng(int(seed))
    chosen: list[np.ndarray] = []
    remaining = m
    for j, cls in enumerate(classes):
        cls_idx = np.flatnonzero(lab == cls)
        quota = int(round(float(m) * float(cls_idx.shape[0]) / float(n)))
        quota = max(1, min(int(cls_idx.shape[0]), quota))
        if j == int(classes.shape[0]) - 1:
            quota = max(1, min(int(cls_idx.shape[0]), remaining))
        chosen.append(rng.choice(cls_idx, size=quota, replace=False))
        remaining -= quota
    out = np.concatenate(chosen)
    if int(out.shape[0]) > m:
        out = rng.choice(out, size=m, replace=False)
    elif int(out.shape[0]) < m:
        mask = np.ones(n, dtype=bool)
        mask[out] = False
        extra = rng.choice(np.flatnonzero(mask), size=m - int(out.shape[0]), replace=False)
        out = np.concatenate([out, extra])
    rng.shuffle(out)
    return out.astype(np.int64, copy=False)


def _categorical_empirical_embedded_mean(
    theta_all: np.ndarray,
    x_embed: np.ndarray,
    num_categories: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-category empirical mean of embedded x at centers 0, 1, ..., K-1."""
    k = int(num_categories)
    if k < 2:
        raise ValueError("num_categories must be >= 2")
    th = np.asarray(theta_all, dtype=np.float64).reshape(-1)
    x = np.asarray(x_embed, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"x_embed must be 2D; got {x.shape}")
    bins = np.arange(-0.5, float(k), 1.0, dtype=np.float64)
    centers = np.arange(k, dtype=np.float64)
    idx = np.digitize(th, bins, right=False) - 1
    valid = (idx >= 0) & (idx < k)
    if not np.all(valid):
        bad = np.unique(th[~valid])
        raise ValueError(f"categorical empirical mean: labels outside [0, {k - 1}]: {bad}")
    ts: list[float] = []
    means: list[np.ndarray] = []
    for b in range(k):
        mask = idx == b
        if not np.any(mask):
            continue
        ts.append(float(centers[b]))
        means.append(x[mask].mean(axis=0))
    if not means:
        raise ValueError("categorical empirical embedded mean produced no non-empty categories")
    return np.asarray(ts, dtype=np.float64).reshape(-1, 1), np.vstack(means)


def _binned_empirical_embedded_mean(
    theta_all: np.ndarray,
    x_embed: np.ndarray,
    theta_low: float,
    theta_high: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin by theta and return (bin_center[:, None], mean embedded x per non-empty bin)."""
    th = np.asarray(theta_all, dtype=np.float64).reshape(-1)
    x = np.asarray(x_embed, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"x_embed must be 2D; got {x.shape}")
    nb = int(n_bins)
    if nb < 2:
        raise ValueError("n_bins must be >= 2")
    bins = np.linspace(float(theta_low), float(theta_high), nb + 1, dtype=np.float64)
    centers = 0.5 * (bins[:-1] + bins[1:])
    idx = np.digitize(th, bins) - 1
    valid = (idx >= 0) & (idx < nb)
    idx_v = idx[valid]
    x_v = x[valid]
    ts: list[float] = []
    means: list[np.ndarray] = []
    for b in range(nb):
        m = idx_v == b
        if not np.any(m):
            continue
        ts.append(float(centers[b]))
        means.append(x_v[m].mean(axis=0))
    if not means:
        raise ValueError("binned empirical embedded mean produced no non-empty bins")
    return np.asarray(ts, dtype=np.float64).reshape(-1, 1), np.vstack(means)


def _binned_empirical_embedded_mean_2d(
    theta_all: np.ndarray,
    x_embed: np.ndarray,
    theta_low: float,
    theta_high: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    th = np.asarray(theta_all, dtype=np.float64)
    x = np.asarray(x_embed, dtype=np.float64)
    if th.ndim != 2 or int(th.shape[1]) != 2:
        raise ValueError(f"theta_all must have shape (N, 2); got {th.shape}")
    nb = int(n_bins)
    bins = np.linspace(float(theta_low), float(theta_high), nb + 1, dtype=np.float64)
    centers = 0.5 * (bins[:-1] + bins[1:])
    i0 = np.digitize(th[:, 0], bins) - 1
    i1 = np.digitize(th[:, 1], bins) - 1
    grid_theta: list[list[float]] = []
    means: list[np.ndarray] = []
    for a in range(nb):
        for b in range(nb):
            mask = (i0 == a) & (i1 == b)
            if np.any(mask):
                grid_theta.append([float(centers[a]), float(centers[b])])
                means.append(x[mask].mean(axis=0))
    if not means:
        raise ValueError("2D binned empirical embedded mean produced no non-empty bins")
    return np.asarray(grid_theta, dtype=np.float64), np.vstack(means)


def _save_projection_summary_figure(
    theta_all: np.ndarray,
    x_embed: np.ndarray,
    train_metrics: dict[str, np.ndarray],
    out_dir: Path,
    base_dataset: Any,
    *,
    scatter_max_points: int | None = 400,
    scatter_subsample_seed: int = 0,
    pr_viz_mean_bins: int = 60,
    pr_viz_mean_smooth_window: int = 3,
) -> None:
    """Three panels: tuning + overlaid PCA manifold/scatter for embedded ``x``, then loss vs epoch.

    The manifold curve is the binned empirical mean of embedded samples (not ``AE(mu_z(theta))``).
    """
    th_raw = np.asarray(theta_all, dtype=np.float64)
    if getattr(base_dataset, "theta_type", "") == "categorical":
        if th_raw.ndim == 2 and int(th_raw.shape[1]) > 1:
            th_raw = np.argmax(th_raw, axis=1).astype(np.float64).reshape(-1, 1)
        elif th_raw.ndim == 1:
            th_raw = th_raw.reshape(-1, 1)
    th_all = th_raw.reshape(-1, 1) if not (th_raw.ndim == 2 and int(th_raw.shape[1]) == 2) else th_raw
    x_all = np.asarray(x_embed, dtype=np.float64)
    n = int(th_raw.shape[0])
    out_dir.mkdir(parents=True, exist_ok=True)

    if th_raw.ndim == 2 and int(th_raw.shape[1]) == 2:
        viz_theta = th_raw
        viz_x = x_all
        if scatter_max_points is not None and n > int(scatter_max_points):
            rng = np.random.default_rng(int(scatter_subsample_seed))
            pick = rng.choice(n, size=int(scatter_max_points), replace=False)
            viz_theta = viz_theta[pick]
            viz_x = viz_x[pick]
        grid_theta, grid_mean = _binned_empirical_embedded_mean_2d(
            th_raw,
            x_all,
            float(base_dataset.theta_low),
            float(base_dataset.theta_high),
            max(4, int(np.sqrt(max(4, int(pr_viz_mean_bins))))),
        )
        png = out_dir / "pr_projection_summary.png"
        fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.6), layout="constrained")
        n_heat = min(4, int(grid_mean.shape[1]))
        for j in range(n_heat):
            ax = axes.ravel()[j]
            sc = ax.scatter(grid_theta[:, 0], grid_theta[:, 1], c=grid_mean[:, j], s=24, cmap="viridis")
            ax.set_title(rf"binned mean $x_{{{j + 1}}}$")
            ax.set_xlabel(r"$\theta_1$")
            ax.set_ylabel(r"$\theta_2$")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        for j in range(n_heat, 4):
            axes.ravel()[j].set_axis_off()
        from fisher.dataset_visualization import pca_project

        proj, _, _ = pca_project(viz_x, n_components=2)
        for k, color_idx in enumerate((0, 1)):
            ax = axes.ravel()[4 + k]
            sc = ax.scatter(proj[:, 0], proj[:, 1], c=viz_theta[:, color_idx], s=8, alpha=0.45, cmap="viridis")
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_axis_off()
            ax.set_title(rf"PCA $x$ colored by $\theta_{color_idx + 1}$")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        loss_raw = train_metrics.get("loss")
        ax_loss = axes.ravel()[6]
        if loss_raw is not None and np.asarray(loss_raw).size >= 1:
            loss = np.asarray(loss_raw, dtype=np.float64).reshape(-1)
            ax_loss.plot(np.arange(1, loss.shape[0] + 1), loss, color="#4c78a8", linewidth=1.2)
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("loss")
        ax_loss.set_title("PR-autoencoder training loss")
        axes.ravel()[7].set_axis_off()
        fig.savefig(png, dpi=180, bbox_inches="tight")
        fig.savefig(png.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)
        print(f"[embed] Saved visualization: {png}")
        print(f"[embed] Saved visualization: {png.with_suffix('.svg')}")
        return

    theta_plot = th_all
    x_plot = x_all
    if scatter_max_points is not None and n > int(scatter_max_points):
        k = int(scatter_max_points)
        rng = np.random.default_rng(int(scatter_subsample_seed))
        pick = rng.choice(n, size=k, replace=False)
        theta_plot = theta_plot[pick]
        x_plot = x_plot[pick]

    is_categorical = getattr(base_dataset, "theta_type", "") == "categorical"
    if is_categorical:
        num_categories = int(getattr(base_dataset, "num_categories", 0))
        if num_categories < 2:
            raise ValueError("categorical projection summary requires num_categories >= 2")
        t_emp, mu_emp = _categorical_empirical_embedded_mean(th_all, x_all, num_categories)
        mean_smooth_window = 0
    else:
        t_emp, mu_emp = _binned_empirical_embedded_mean(
            th_all,
            x_all,
            float(base_dataset.theta_low),
            float(base_dataset.theta_high),
            int(pr_viz_mean_bins),
        )
        mean_smooth_window = int(pr_viz_mean_smooth_window)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.2), layout="constrained")
    ax_tune, ax_manifold, ax_loss = axes[0], axes[1], axes[2]
    plot_joint_and_tuning_on_axes(
        fig,
        ax_tune,
        ax_manifold,
        theta_plot,
        x_plot,
        base_dataset,
        mu_override=mu_emp,
        theta_grid_override=t_emp,
        n_theta_grid=500,
        smooth_mean_window=mean_smooth_window,
    )
    if is_categorical:
        ax_tune.set_xlabel("category")
        ax_tune.set_ylabel(r"Mean of $x|category$")

    loss_raw = train_metrics.get("loss")
    if loss_raw is not None and np.asarray(loss_raw).size >= 1:
        loss = np.asarray(loss_raw, dtype=np.float64).reshape(-1)
        epochs = np.arange(1, loss.shape[0] + 1, dtype=np.float64)
        ax_loss.plot(epochs, loss, color="#4c78a8", linewidth=1.2, label="loss")
        adv_acc_raw = train_metrics.get("adv_acc")
        if adv_acc_raw is not None and np.asarray(adv_acc_raw).size == loss.shape[0]:
            ax_loss.plot(
                epochs,
                np.asarray(adv_acc_raw, dtype=np.float64).reshape(-1),
                color="#f58518",
                linewidth=1.1,
                label="linear adv acc",
            )
            ax_loss.legend(frameon=False, fontsize=7)
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("loss")
        ax_loss.set_title("PR-autoencoder training loss")
    else:
        ax_loss.set_title("PR-autoencoder training loss")
        ax_loss.text(0.5, 0.5, "no training metrics", ha="center", va="center", transform=ax_loss.transAxes)

    png = out_dir / "pr_projection_summary.png"
    svg = out_dir / "pr_projection_summary.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"[embed] Saved visualization: {png}")
    print(f"[embed] Saved visualization: {svg}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Load a low-x-dim shared Fisher dataset .npz, train a PR-autoencoder (z_dim -> h_dim) "
            "unless --use-cache is set, then write a new .npz with embedded x and updated metadata."
        )
    )
    p.add_argument("--input-npz", type=str, required=True, help="Source shared dataset (low-dimensional x).")
    p.add_argument("--output-npz", type=str, required=True, help="Destination .npz path.")
    p.add_argument(
        "--h-dim",
        type=int,
        required=True,
        help="Target observation dimension after embedding (must be >= source x_dim / z_dim).",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="PR-autoencoder / projection seed; default: meta seed from the input NPZ.",
    )
    p.add_argument("--cache-dir", type=str, default="data/pr_autoencoder_cache")
    p.add_argument(
        "--use-cache",
        action="store_true",
        help=(
            "If a matching PR-autoencoder checkpoint exists under --cache-dir, load it instead of training. "
            "Default: always train (force_retrain) so each run fits a fresh encoder."
        ),
    )
    p.add_argument(
        "--pr-viz-scatter-max",
        type=int,
        default=400,
        metavar="N",
        help=(
            "Max number of (theta, x) rows subsampled for the PCA scatter in pr_projection_summary "
            "(uniform random subset when the dataset is larger). Use 0 for no subsampling (all points)."
        ),
    )
    p.add_argument(
        "--pr-viz-mean-bins",
        type=int,
        default=60,
        metavar="B",
        help=(
            "Number of theta bins for the empirical mean curve in pr_projection_summary "
            "(mean embedded x per bin; non-empty bins only)."
        ),
    )
    p.add_argument(
        "--pr-viz-mean-smooth-window",
        type=int,
        default=3,
        metavar="W",
        help=(
            "Odd-length moving-average window along binned theta for the empirical mean curve "
            "before PCA (only when using binned embedded means). Use 0, 1, or 2 to disable smoothing."
        ),
    )
    p.add_argument("--allow-non-randamp-sqrtd", action="store_true", help="Skip dataset_family check (expert).")
    p.add_argument(
        "--skip-viz",
        action="store_true",
        help="Do not write pr_projection_summary.{png,svg} (tuning + overlaid PCA manifold/scatter + loss).",
    )

    p.add_argument("--pr-hidden1", type=int, default=None)
    p.add_argument("--pr-hidden2", type=int, default=None)
    p.add_argument("--pr-train-samples", type=int, default=None)
    p.add_argument("--pr-train-epochs", type=int, default=None)
    p.add_argument("--pr-train-batch-size", type=int, default=None)
    p.add_argument("--pr-train-lr", type=float, default=None)
    p.add_argument("--pr-lambda-pr", type=float, default=None)
    p.add_argument("--pr-eps", type=float, default=None)
    p.add_argument(
        "--pr-adversarial-categorical",
        action="store_true",
        help=(
            "Enable adversarial PR projection with a gradient-reversal linear categorical classifier. "
            "Only valid when input meta theta_type is categorical."
        ),
    )
    p.add_argument("--pr-lambda-adv", type=float, default=0.1)
    p.add_argument("--pr-adv-warmup-epochs", type=int, default=0)
    p.add_argument(
        "--pr-adv-ramp-epochs",
        type=int,
        default=None,
        help="Epochs used to ramp the adversarial coefficient to --pr-lambda-adv; default train_epochs // 5.",
    )
    p.add_argument("--pr-adv-steps", type=int, default=1)
    p.add_argument(
        "--pr-adv-train-samples",
        type=int,
        default=0,
        help="Rows to stratified-subsample for adversarial PR training; 0 uses all input rows.",
    )
    return p.parse_args(argv)


def validate_h_dim(*, h_dim: int, z_dim: int) -> None:
    if int(h_dim) < int(z_dim):
        raise ValueError(f"--h-dim must be >= latent z_dim={int(z_dim)}; got {int(h_dim)}.")


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_npz).resolve()
    out_path = Path(args.output_npz).resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Input NPZ not found: {in_path}")

    bundle = load_shared_dataset_npz(in_path)
    meta = dict(bundle.meta)
    fam = str(meta.get("dataset_family", ""))
    if not args.allow_non_randamp_sqrtd and fam != "randamp_gaussian_sqrtd":
        raise ValueError(
            f"Expected dataset_family='randamp_gaussian_sqrtd' in input meta (got {fam!r}). "
            "Pass --allow-non-randamp-sqrtd for other native NPZs (e.g. random_mog_categorical)."
        )
    if bool(meta.get("pr_autoencoder_embedded", False)):
        raise ValueError(
            "Input already has pr_autoencoder_embedded=True; refuse to chain-embed. "
            "Start from a native low-dimensional archive from make_dataset.py."
        )

    base_dataset = build_dataset_from_meta(meta)

    z_dim = int(meta["x_dim"])
    x_all = np.asarray(bundle.x_all, dtype=np.float64)
    if x_all.ndim != 2 or int(x_all.shape[1]) != z_dim:
        raise ValueError(f"Input x_all shape {x_all.shape} inconsistent with meta x_dim={z_dim}.")

    h_dim = int(args.h_dim)
    validate_h_dim(h_dim=h_dim, z_dim=z_dim)

    device_name = str(args.device)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested (`--device cuda`) but CUDA is unavailable on this machine.")
    device = torch.device(device_name)

    seed = int(args.seed) if args.seed is not None else int(meta["seed"])
    source_sha256 = _file_sha256(in_path)
    adv_labels: np.ndarray | None = None
    adv_x_train = x_all
    adv_train_samples = int(args.pr_adv_train_samples)
    train_epochs = int(args.pr_train_epochs) if args.pr_train_epochs is not None else 200
    adv_ramp_epochs = (
        int(args.pr_adv_ramp_epochs)
        if args.pr_adv_ramp_epochs is not None
        else max(1, int(train_epochs) // 5)
    )
    if bool(args.pr_adversarial_categorical):
        if str(meta.get("theta_type", "")) != "categorical":
            raise ValueError(
                "--pr-adversarial-categorical only works when the input dataset has categorical labels "
                f"(meta theta_type='categorical'); got {meta.get('theta_type')!r}."
            )
        if float(args.pr_lambda_adv) < 0.0:
            raise ValueError("--pr-lambda-adv must be non-negative.")
        if int(args.pr_adv_warmup_epochs) < 0:
            raise ValueError("--pr-adv-warmup-epochs must be >= 0.")
        if int(adv_ramp_epochs) < 1:
            raise ValueError("--pr-adv-ramp-epochs must be >= 1.")
        if int(args.pr_adv_steps) < 1:
            raise ValueError("--pr-adv-steps must be >= 1.")
        if int(adv_train_samples) < 0:
            raise ValueError("--pr-adv-train-samples must be >= 0.")
        num_categories = int(meta.get("num_categories", 0))
        labels_all = _categorical_labels_from_theta(bundle.theta_all, num_categories=num_categories)
        if int(labels_all.shape[0]) != int(x_all.shape[0]):
            raise ValueError("Categorical theta row count does not match x_all row count.")
        adv_idx = _stratified_label_subsample(labels_all, n_samples=adv_train_samples, seed=seed)
        adv_x_train = x_all[adv_idx]
        adv_labels = labels_all[adv_idx]
        observed = np.unique(adv_labels)
        if int(observed.shape[0]) < num_categories:
            raise ValueError(
                "Adversarial categorical PR training sample does not include every category; "
                "increase --pr-adv-train-samples or use 0 for all rows."
            )

    cfg_ns = SimpleNamespace(
        pr_autoencoder_z_dim=z_dim,
        pr_autoencoder_hidden1=int(args.pr_hidden1) if args.pr_hidden1 is not None else 100,
        pr_autoencoder_hidden2=int(args.pr_hidden2) if args.pr_hidden2 is not None else 200,
        pr_autoencoder_train_samples=int(args.pr_train_samples) if args.pr_train_samples is not None else 12000,
        pr_autoencoder_train_epochs=train_epochs,
        pr_autoencoder_train_batch_size=(
            int(args.pr_train_batch_size) if args.pr_train_batch_size is not None else 512
        ),
        pr_autoencoder_train_lr=float(args.pr_train_lr) if args.pr_train_lr is not None else 1e-3,
        pr_autoencoder_lambda_pr=float(args.pr_lambda_pr) if args.pr_lambda_pr is not None else 1e-2,
        pr_autoencoder_pr_eps=float(args.pr_eps) if args.pr_eps is not None else 1e-8,
        pr_autoencoder_adversarial_categorical=bool(args.pr_adversarial_categorical),
        pr_autoencoder_lambda_adv=float(args.pr_lambda_adv),
        pr_autoencoder_adv_warmup_epochs=int(args.pr_adv_warmup_epochs),
        pr_autoencoder_adv_ramp_epochs=adv_ramp_epochs,
        pr_autoencoder_adv_steps=int(args.pr_adv_steps),
        pr_autoencoder_adv_train_samples=adv_train_samples,
        pr_autoencoder_adv_num_classes=int(meta.get("num_categories", 0)) if bool(args.pr_adversarial_categorical) else 0,
        pr_autoencoder_adv_source_sha256=source_sha256 if bool(args.pr_adversarial_categorical) else "",
    )
    cfg = pr_autoencoder_config_from_namespace(cfg_ns, h_dim=h_dim)

    x_embed_all, cache_run_dir, loaded_from_cache, train_metrics, ae_model = project_x_through_pr_autoencoder(
        adv_x_train if bool(args.pr_adversarial_categorical) else x_all,
        config=cfg,
        seed=seed,
        device=device,
        cache_dir=str(args.cache_dir),
        force_retrain=not bool(args.use_cache),
        train_y=adv_labels,
    )
    if bool(args.pr_adversarial_categorical):
        z_t = torch.from_numpy(x_all.astype(np.float32, copy=False)).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            h_t, _ = ae_model(z_t)
        x_embed_all = h_t.detach().cpu().numpy().astype(np.float64, copy=False)

    train_idx = np.asarray(bundle.train_idx, dtype=np.int64)
    val_idx = np.asarray(bundle.validation_idx, dtype=np.int64)
    theta_train = np.asarray(bundle.theta_train, dtype=np.float64)
    theta_validation = np.asarray(bundle.theta_validation, dtype=np.float64)
    x_train = x_embed_all[train_idx]
    x_validation = x_embed_all[val_idx]

    out_meta = dict(meta)
    out_meta["x_dim"] = h_dim
    out_meta["seed"] = int(meta["seed"])
    out_meta["pr_autoencoder_enabled"] = True
    out_meta["pr_autoencoder_embedded"] = True
    out_meta["pr_autoencoder_z_dim"] = z_dim
    out_meta["pr_autoencoder_hidden1"] = int(cfg.hidden1)
    out_meta["pr_autoencoder_hidden2"] = int(cfg.hidden2)
    out_meta["pr_autoencoder_train_samples"] = int(cfg.train_samples)
    out_meta["pr_autoencoder_train_epochs"] = int(cfg.train_epochs)
    out_meta["pr_autoencoder_train_batch_size"] = int(cfg.train_batch_size)
    out_meta["pr_autoencoder_train_lr"] = float(cfg.train_lr)
    out_meta["pr_autoencoder_lambda_pr"] = float(cfg.lambda_pr)
    out_meta["pr_autoencoder_pr_eps"] = float(cfg.pr_eps)
    out_meta["pr_autoencoder_adversarial_categorical"] = bool(cfg.adversarial_categorical)
    out_meta["pr_autoencoder_lambda_adv"] = float(cfg.lambda_adv)
    out_meta["pr_autoencoder_adv_warmup_epochs"] = int(cfg.adv_warmup_epochs)
    out_meta["pr_autoencoder_adv_ramp_epochs"] = int(cfg.adv_ramp_epochs)
    out_meta["pr_autoencoder_adv_steps"] = int(cfg.adv_steps)
    out_meta["pr_autoencoder_adv_train_samples"] = int(cfg.adv_train_samples)
    out_meta["pr_autoencoder_adv_num_classes"] = int(cfg.adv_num_classes)
    adv_acc = np.asarray(train_metrics.get("adv_acc", []), dtype=np.float64).reshape(-1)
    adv_ce = np.asarray(train_metrics.get("adv_ce", []), dtype=np.float64).reshape(-1)
    out_meta["pr_autoencoder_adv_final_linear_accuracy"] = (
        float(adv_acc[-1]) if bool(cfg.adversarial_categorical) and adv_acc.size else None
    )
    out_meta["pr_autoencoder_adv_final_ce"] = (
        float(adv_ce[-1]) if bool(cfg.adversarial_categorical) and adv_ce.size else None
    )
    out_meta["pr_autoencoder_seed"] = seed
    out_meta["pr_autoencoder_cache_key"] = str(cache_run_dir.name)
    out_meta["pr_autoencoder_source_npz"] = str(in_path)
    out_meta["pr_autoencoder_source_sha256"] = source_sha256
    out_meta["pr_autoencoder_projection_note"] = (
        "Embedded x via bin/project_dataset_pr_autoencoder.py; generative model dim is pr_autoencoder_z_dim."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_shared_dataset_npz(
        out_path,
        meta=out_meta,
        theta_all=np.asarray(bundle.theta_all, dtype=np.float64),
        x_all=x_embed_all,
        train_idx=train_idx,
        validation_idx=val_idx,
        theta_train=theta_train,
        x_train=x_train,
        theta_validation=theta_validation,
        x_validation=x_validation,
    )
    print(f"[embed] Wrote {out_path}  x_dim={h_dim} z_dim={z_dim}  cache_loaded={loaded_from_cache}")
    print(f"[embed] PR cache dir: {cache_run_dir}")

    if not args.skip_viz:
        if int(args.pr_viz_mean_bins) < 2:
            raise ValueError("--pr-viz-mean-bins must be >= 2.")
        if int(args.pr_viz_mean_smooth_window) < 0:
            raise ValueError("--pr-viz-mean-smooth-window must be >= 0.")
        viz_cap: int | None = None if int(args.pr_viz_scatter_max) <= 0 else int(args.pr_viz_scatter_max)
        _save_projection_summary_figure(
            np.asarray(bundle.theta_all, dtype=np.float64),
            x_embed_all,
            train_metrics,
            out_path.parent,
            base_dataset,
            scatter_max_points=viz_cap,
            pr_viz_mean_bins=int(args.pr_viz_mean_bins),
            pr_viz_mean_smooth_window=int(args.pr_viz_mean_smooth_window),
        )

    # sidecar JSON for humans (optional quick provenance)
    side = out_path.with_suffix(".projection_meta.json")
    side.write_text(
        json.dumps(
            {
                "source_npz": str(in_path),
                "source_sha256": out_meta["pr_autoencoder_source_sha256"],
                "output_npz": str(out_path),
                "z_dim": z_dim,
                "h_dim": h_dim,
                "pr_autoencoder_cache_key": out_meta["pr_autoencoder_cache_key"],
                "pr_autoencoder_adversarial_categorical": bool(cfg.adversarial_categorical),
                "pr_autoencoder_adv_final_linear_accuracy": out_meta[
                    "pr_autoencoder_adv_final_linear_accuracy"
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"[embed] Wrote {side}")


if __name__ == "__main__":
    main()
