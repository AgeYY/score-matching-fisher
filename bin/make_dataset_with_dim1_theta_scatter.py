#!/usr/bin/env python3
"""Like ``bin/make_dataset.py``, plus a scatter of first coordinate vs. $\\theta$.

Saves the same NPZ and ``joint_scatter_and_tuning_curve.{png,svg}``, and additionally writes
``theta_vs_x1_scatter.{png,svg}`` with a fixed-size random subsample (default 80 points) for a
clear view of the $x_1$ response vs. $\\theta$.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import global_setting  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

from global_setting import DATA_DIR
from fisher.cli_shared_fisher import add_dataset_arguments
from fisher.dataset_family_recipes import (
    assert_no_legacy_dataset_cli_flags,
    format_resolved_family_summary,
)
from fisher.dataset_visualization import plot_joint_and_tuning, summarize_dataset
from fisher.pr_autoencoder_embedding import build_randamp_gaussian_sqrtd_pr_autoencoder_dataset
from fisher.realnvp_embedding import build_randamp_gaussian_sqrtd_realnvp_dataset
from fisher.shared_dataset_io import meta_dict_from_args, save_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_args, validate_dataset_sample_args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sample a synthetic (theta, x) dataset, save .npz and joint/tuning figures like "
            "make_dataset.py, and emit an extra scatter of x_1 vs theta using a subsample of "
            "fixed size (default 80)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_dataset_arguments(p)
    p.add_argument(
        "--output-npz",
        type=str,
        default=str(Path(DATA_DIR) / "shared_fisher_dataset.npz"),
        help=(
            "Path for the shared dataset archive (theta_all, x_all, train/eval indices, meta). "
            "Prefer a path under your DATAROOT data directory."
        ),
    )
    p.add_argument(
        "--dim1-scatter-n",
        type=int,
        default=80,
        metavar="K",
        help="Number of scatter points in the theta vs x_1 figure (uniform subsample without replacement).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device used by embedding-based families (e.g., PR-autoencoder).",
    )
    return p.parse_args(argv)


def plot_theta_vs_x1_scatter(
    theta: np.ndarray,
    x: np.ndarray,
    out_path_png: str,
    *,
    n_points: int,
    rng: np.random.Generator,
) -> None:
    """Scatter of $\\theta$ vs first observation dimension, at most ``n_points`` samples."""
    theta = np.asarray(theta, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = int(theta.shape[0])
    k = int(min(max(n_points, 1), n))
    if n > k:
        pick = rng.choice(n, size=k, replace=False)
    else:
        pick = np.arange(n, dtype=np.int64)
    th = theta[pick].ravel()
    x1 = x[pick, 0]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sc = ax.scatter(th, x1, c=th, s=36, alpha=0.75, cmap="viridis", edgecolors="white", linewidths=0.35)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$x_1$")
    ax.set_title(rf"First dimension vs. $\theta$ ($n={k}$)")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    fig.colorbar(sc, ax=ax, label=r"$\theta$")
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=180, bbox_inches="tight")
    fig.savefig(str(Path(out_path_png).with_suffix(".svg")), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    argv = sys.argv[1:]
    assert_no_legacy_dataset_cli_flags(argv)
    args = parse_args(argv)
    validate_dataset_sample_args(args)
    if int(args.dim1_scatter_n) < 1:
        raise ValueError("--dim1-scatter-n must be >= 1.")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)) or ".", exist_ok=True)

    print("[data] Resolved family configuration:")
    print(format_resolved_family_summary(args))

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    rng_dim1 = np.random.default_rng(int(args.seed) + 913_409)

    n_total = int(args.n_total)
    if str(args.dataset_family) == "randamp_gaussian_sqrtd_realnvp":
        built = build_randamp_gaussian_sqrtd_realnvp_dataset(args)
        dataset = built.base_dataset
        theta_all = built.theta_all
        x_all = built.x_embed_all
    elif str(args.dataset_family) == "randamp_gaussian_sqrtd_pr_autoencoder":
        built = build_randamp_gaussian_sqrtd_pr_autoencoder_dataset(args)
        dataset = built.base_dataset
        theta_all = built.theta_all
        x_all = built.x_embed_all
    else:
        dataset = build_dataset_from_args(args)
        theta_all, x_all = dataset.sample_joint(n_total)
    perm = rng.permutation(n_total)
    tf = float(args.train_frac)
    if tf >= 1.0:
        n_train = n_total
    else:
        n_train = int(tf * n_total)
        n_train = min(max(n_train, 1), n_total - 1)

    tr_idx = perm[:n_train]
    ev_idx = perm[n_train:]
    theta_train, x_train = theta_all[tr_idx], x_all[tr_idx]
    theta_validation, x_validation = theta_all[ev_idx], x_all[ev_idx]

    meta = meta_dict_from_args(args)
    if str(args.dataset_family) in (
        "randamp_gaussian",
        "randamp_gaussian_sqrtd",
        "randamp_gaussian_sqrtd_realnvp",
        "randamp_gaussian_sqrtd_pr_autoencoder",
    ):
        meta["randamp_mu_amp_per_dim"] = dataset._randamp_amp.tolist()
    if str(args.dataset_family) == "randamp_gaussian_sqrtd_realnvp":
        meta["realnvp_enabled"] = True
        meta["realnvp_z_dim"] = int(built.embedder_config.z_dim)
        meta["realnvp_n_transforms"] = int(built.embedder_config.n_transforms)
        meta["realnvp_hidden_width"] = int(built.embedder_config.hidden_width)
        meta["realnvp_seed"] = int(args.seed)
        meta["realnvp_batch_norm_between_transforms"] = bool(
            built.embedder_config.batch_norm_between_transforms
        )
    if str(args.dataset_family) == "randamp_gaussian_sqrtd_pr_autoencoder":
        meta["pr_autoencoder_enabled"] = True
        meta["pr_autoencoder_z_dim"] = int(built.embedder_config.z_dim)
        meta["pr_autoencoder_hidden1"] = int(built.embedder_config.hidden1)
        meta["pr_autoencoder_hidden2"] = int(built.embedder_config.hidden2)
        meta["pr_autoencoder_train_samples"] = int(built.embedder_config.train_samples)
        meta["pr_autoencoder_train_epochs"] = int(built.embedder_config.train_epochs)
        meta["pr_autoencoder_train_batch_size"] = int(built.embedder_config.train_batch_size)
        meta["pr_autoencoder_train_lr"] = float(built.embedder_config.train_lr)
        meta["pr_autoencoder_lambda_pr"] = float(built.embedder_config.lambda_pr)
        meta["pr_autoencoder_pr_eps"] = float(built.embedder_config.pr_eps)
        meta["pr_autoencoder_seed"] = int(args.seed)
        meta["pr_autoencoder_cache_key"] = str(built.cache_run_dir.name)
    save_shared_dataset_npz(
        args.output_npz,
        meta=meta,
        theta_all=theta_all,
        x_all=x_all,
        train_idx=tr_idx.astype(np.int64),
        validation_idx=ev_idx.astype(np.int64),
        theta_train=theta_train,
        x_train=x_train,
        theta_validation=theta_validation,
        x_validation=x_validation,
    )

    print(
        f"[data] total={n_total} train={theta_train.shape[0]} validation={theta_validation.shape[0]}"
    )
    print(f"Saved shared dataset: {args.output_npz}")

    out_dir = Path(args.output_npz).resolve().parent
    joint_tuning_path = out_dir / "joint_scatter_and_tuning_curve.png"
    if str(args.dataset_family) in (
        "randamp_gaussian_sqrtd_realnvp",
        "randamp_gaussian_sqrtd_pr_autoencoder",
    ):
        print(
            "[data] Skipping summarize_dataset for embedded randamp family: "
            "embedded x_dim differs from base tuning dimension."
        )
    else:
        summarize_dataset(theta_all, x_all, dataset)
    plot_joint_and_tuning(theta_all, x_all, dataset, str(joint_tuning_path))
    print(f"Saved visualization: {joint_tuning_path}")
    print(f"Saved visualization: {joint_tuning_path.with_suffix('.svg')}")

    dim1_path = out_dir / "theta_vs_x1_scatter.png"
    plot_theta_vs_x1_scatter(
        theta_all,
        x_all,
        str(dim1_path),
        n_points=int(args.dim1_scatter_n),
        rng=rng_dim1,
    )
    print(f"Saved visualization: {dim1_path}")
    print(f"Saved visualization: {dim1_path.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
