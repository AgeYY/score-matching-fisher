#!/usr/bin/env python3
"""Embed a low-dimensional shared dataset NPZ into higher-dimensional x via PR-autoencoder.

Typical workflow:

1. ``python bin/make_dataset.py --dataset-family randamp_gaussian_sqrtd --x-dim 2 ...``
2. ``python bin/project_dataset_pr_autoencoder.py --input-npz <low.npz> --output-npz <high.npz> --h-dim 10``

The output archive keeps ``dataset_family='randamp_gaussian_sqrtd'`` (generative recipe), sets
``meta['x_dim']`` to the embedded dimension ``h_dim``, and sets ``pr_autoencoder_embedded=True``
with ``pr_autoencoder_z_dim`` equal to the source latent dimension. Ground-truth helpers that need
the low-dimensional generative model use :func:`fisher.shared_fisher_est.build_dataset_from_meta`.

Unless ``--skip-viz``, writes ``pr_projection_summary.{png,svg}`` next to ``--output-npz``:
same two-panel native layout as ``make_dataset.py`` (tuning curves, binned empirical mean of
embedded samples with scatter overlaid in PCA), plus a third panel with PR-autoencoder training
loss vs epoch.
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


def _save_projection_summary_figure(
    theta_all: np.ndarray,
    x_embed: np.ndarray,
    train_metrics: dict[str, np.ndarray],
    out_dir: Path,
    base_dataset: Any,
    *,
    scatter_max_points: int | None = 800,
    scatter_subsample_seed: int = 0,
    pr_viz_mean_bins: int = 60,
    pr_viz_mean_smooth_window: int = 3,
) -> None:
    """Three panels: tuning + overlaid PCA manifold/scatter for embedded ``x``, then loss vs epoch.

    The manifold curve is the binned empirical mean of embedded samples (not ``AE(mu_z(theta))``).
    """
    th_all = np.asarray(theta_all, dtype=np.float64).reshape(-1, 1)
    x_all = np.asarray(x_embed, dtype=np.float64)
    n = int(th_all.shape[0])
    out_dir.mkdir(parents=True, exist_ok=True)

    theta_plot = th_all
    x_plot = x_all
    if scatter_max_points is not None and n > int(scatter_max_points):
        k = int(scatter_max_points)
        rng = np.random.default_rng(int(scatter_subsample_seed))
        pick = rng.choice(n, size=k, replace=False)
        theta_plot = theta_plot[pick]
        x_plot = x_plot[pick]

    t_emp, mu_emp = _binned_empirical_embedded_mean(
        th_all,
        x_all,
        float(base_dataset.theta_low),
        float(base_dataset.theta_high),
        int(pr_viz_mean_bins),
    )

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
        smooth_mean_window=int(pr_viz_mean_smooth_window),
    )

    loss_raw = train_metrics.get("loss")
    if loss_raw is not None and np.asarray(loss_raw).size >= 1:
        loss = np.asarray(loss_raw, dtype=np.float64).reshape(-1)
        epochs = np.arange(1, loss.shape[0] + 1, dtype=np.float64)
        ax_loss.plot(epochs, loss, color="#4c78a8", linewidth=1.2)
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
        help="Target observation dimension after embedding (must be > source x_dim / z_dim).",
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
        default=800,
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
    return p.parse_args(argv)


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
            "Use --allow-non-randamp-sqrtd to override."
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
    if h_dim <= z_dim:
        raise ValueError(f"--h-dim must be > latent z_dim={z_dim}; got {h_dim}.")

    device_name = str(args.device)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested (`--device cuda`) but CUDA is unavailable on this machine.")
    device = torch.device(device_name)

    seed = int(args.seed) if args.seed is not None else int(meta["seed"])

    cfg_ns = SimpleNamespace(
        pr_autoencoder_z_dim=z_dim,
        pr_autoencoder_hidden1=int(args.pr_hidden1) if args.pr_hidden1 is not None else 100,
        pr_autoencoder_hidden2=int(args.pr_hidden2) if args.pr_hidden2 is not None else 200,
        pr_autoencoder_train_samples=int(args.pr_train_samples) if args.pr_train_samples is not None else 12000,
        pr_autoencoder_train_epochs=int(args.pr_train_epochs) if args.pr_train_epochs is not None else 200,
        pr_autoencoder_train_batch_size=(
            int(args.pr_train_batch_size) if args.pr_train_batch_size is not None else 512
        ),
        pr_autoencoder_train_lr=float(args.pr_train_lr) if args.pr_train_lr is not None else 1e-3,
        pr_autoencoder_lambda_pr=float(args.pr_lambda_pr) if args.pr_lambda_pr is not None else 1e-2,
        pr_autoencoder_pr_eps=float(args.pr_eps) if args.pr_eps is not None else 1e-8,
    )
    cfg = pr_autoencoder_config_from_namespace(cfg_ns, h_dim=h_dim)

    x_embed_all, cache_run_dir, loaded_from_cache, train_metrics, ae_model = project_x_through_pr_autoencoder(
        x_all,
        config=cfg,
        seed=seed,
        device=device,
        cache_dir=str(args.cache_dir),
        force_retrain=not bool(args.use_cache),
    )

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
    out_meta["pr_autoencoder_seed"] = seed
    out_meta["pr_autoencoder_cache_key"] = str(cache_run_dir.name)
    out_meta["pr_autoencoder_source_npz"] = str(in_path)
    out_meta["pr_autoencoder_source_sha256"] = _file_sha256(in_path)
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
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"[embed] Wrote {side}")


if __name__ == "__main__":
    main()
