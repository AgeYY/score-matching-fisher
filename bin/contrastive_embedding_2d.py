#!/usr/bin/env python3
"""Train soft contrastive (normalized_dot) with low feature_dim and visualize x-embeddings.

Uses ContrastiveNormalizedDotScorer from fisher.contrastive_llr: the x-branch maps x to
R^{feature_dim}, then L2-normalizes (encode_x). For feature_dim=2, points lie on the unit circle.

Example:
  mamba run -n geo_diffusion python bin/contrastive_embedding_2d.py \\
    --dataset-npz data/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha4x/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha4x_pr30d.npz \\
    --output-dir data/experiments/contrastive_embed2d_cosine_pr30 \\
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_repo_root = Path(__file__).resolve().parent.parent
_bin_dir = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_bin_dir) not in sys.path:
    sys.path.insert(0, str(_bin_dir))

from global_setting import DATA_DIR, apply_matplotlib_defaults

apply_matplotlib_defaults()

from fisher.contrastive_llr import ContrastiveNormalizedDotScorer, train_contrastive_soft_llr
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import require_device


def _theta_as_matrix(theta: np.ndarray) -> np.ndarray:
    t = np.asarray(theta, dtype=np.float64)
    if t.ndim == 1:
        return t.reshape(-1, 1)
    if t.ndim == 2:
        return t
    raise ValueError("theta must be 1D or 2D.")


def _encode_x_all(
    model: ContrastiveNormalizedDotScorer,
    x_norm: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    n = int(x_norm.shape[0])
    bs = max(1, int(batch_size))
    with torch.no_grad():
        for i0 in range(0, n, bs):
            i1 = min(n, i0 + bs)
            xt = torch.from_numpy(x_norm[i0:i1].astype(np.float32)).to(device)
            z = model.encode_x(xt).detach().cpu().numpy().astype(np.float64, copy=False)
            out.append(z)
    return np.concatenate(out, axis=0)


def _bin_indices(theta_flat: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    lo = float(np.min(theta_flat))
    hi = float(np.max(theta_flat))
    edges = np.linspace(lo, hi, int(n_bins) + 1)
    idx = np.searchsorted(edges, theta_flat, side="right") - 1
    idx = np.clip(idx, 0, int(n_bins) - 1)
    return idx.astype(np.int64), edges


def main(argv: list[str] | None = None) -> None:
    default_npz = (
        Path(DATA_DIR)
        / "cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha4x"
        / "cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha4x_pr30d.npz"
    )
    p = argparse.ArgumentParser(description="Contrastive soft normalized_dot embedding + 2D plot.")
    p.add_argument(
        "--dataset-npz",
        type=str,
        default=str(default_npz),
        help="Shared dataset .npz (PR-projected cosine PR30 default).",
    )
    p.add_argument("--output-dir", type=str, required=True, help="Directory for NPZ + figures.")
    p.add_argument("--device", type=str, default="cuda", help="cuda only per repo policy.")
    p.add_argument("--feature-dim", type=int, default=2, help="Encoder output dim (default 2).")
    p.add_argument("--encode-batch-size", type=int, default=4096, help="Batch size for encode_x.")
    p.add_argument("--num-theta-bins", type=int, default=10, help="Bins for secondary coloring.")
    # Training (aligned with study_h_decoding_convergence contrastive-soft defaults)
    p.add_argument("--contrastive-epochs", type=int, default=2000)
    p.add_argument("--contrastive-batch-size", type=int, default=256)
    p.add_argument("--contrastive-lr", type=float, default=1e-3)
    p.add_argument("--contrastive-hidden-dim", type=int, default=128)
    p.add_argument("--contrastive-depth", type=int, default=3)
    p.add_argument("--contrastive-weight-decay", type=float, default=0.0)
    p.add_argument("--contrastive-early-patience", type=int, default=300)
    p.add_argument("--contrastive-early-min-delta", type=float, default=1e-4)
    p.add_argument("--contrastive-early-ema-alpha", type=float, default=0.05)
    p.add_argument("--contrastive-max-grad-norm", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--contrastive-soft-bandwidth-bins", type=int, default=10)
    p.add_argument("--contrastive-soft-bandwidth-start", type=float, default=0.0)
    p.add_argument("--contrastive-soft-bandwidth-end", type=float, default=0.0)
    p.add_argument("--contrastive-soft-periodic", action="store_true")
    p.add_argument("--contrastive-soft-period", type=float, default=2.0 * np.pi)
    args = p.parse_args(argv)

    dev = require_device(str(args.device))
    out_dir = os.path.abspath(str(args.output_dir))
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.abspath(str(args.dataset_npz))
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"dataset npz not found: {npz_path}")

    bundle = load_shared_dataset_npz(npz_path)
    meta = dict(bundle.meta)
    theta_train = _theta_as_matrix(bundle.theta_train)
    theta_val = _theta_as_matrix(bundle.theta_validation)
    x_train = np.asarray(bundle.x_train, dtype=np.float64)
    x_val = np.asarray(bundle.x_validation, dtype=np.float64)
    theta_all = _theta_as_matrix(bundle.theta_all)
    x_all = np.asarray(bundle.x_all, dtype=np.float64)

    fdim = int(args.feature_dim)
    if fdim < 1:
        raise ValueError("--feature-dim must be >= 1.")

    model = ContrastiveNormalizedDotScorer(
        x_dim=int(x_all.shape[1]),
        theta_dim=1,
        feature_dim=fdim,
        hidden_dim=int(args.contrastive_hidden_dim),
        depth=int(args.contrastive_depth),
    ).to(dev)

    train_out = train_contrastive_soft_llr(
        model=model,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        device=dev,
        epochs=int(args.contrastive_epochs),
        batch_size=int(args.contrastive_batch_size),
        lr=float(args.contrastive_lr),
        bandwidth_bins=int(args.contrastive_soft_bandwidth_bins),
        bandwidth_start=float(args.contrastive_soft_bandwidth_start),
        bandwidth_end=float(args.contrastive_soft_bandwidth_end),
        periodic=bool(args.contrastive_soft_periodic),
        period=float(args.contrastive_soft_period),
        weight_decay=float(args.contrastive_weight_decay),
        patience=int(args.contrastive_early_patience),
        min_delta=float(args.contrastive_early_min_delta),
        ema_alpha=float(args.contrastive_early_ema_alpha),
        max_grad_norm=float(args.contrastive_max_grad_norm),
        log_every=max(1, int(args.log_every)),
        restore_best=True,
        contrastive_theta_fourier_k=0,
    )

    x_mean = np.asarray(train_out["x_mean"], dtype=np.float64)
    x_std = np.asarray(train_out["x_std"], dtype=np.float64)
    x_all_n = (x_all - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)
    z_unit = _encode_x_all(model, x_all_n, device=dev, batch_size=int(args.encode_batch_size))

    theta_flat = theta_all.reshape(-1)
    bin_idx, bin_edges = _bin_indices(theta_flat, int(args.num_theta_bins))

    meta_out = {
        "dataset_npz": npz_path,
        "dataset_family": str(meta.get("dataset_family", "")),
        "feature_dim": fdim,
        "best_epoch": int(train_out["best_epoch"]),
        "stopped_early": bool(train_out["stopped_early"]),
        "bandwidth_raw_final": float(train_out["bandwidth_raw"]),
        "bandwidth_normalized_final": float(train_out["bandwidth_normalized"]),
        "num_theta_bins_viz": int(args.num_theta_bins),
    }
    meta_json = json.dumps(meta_out, sort_keys=True)

    emb_npz = os.path.join(out_dir, "embedding_2d.npz")
    np.savez_compressed(
        emb_npz,
        z_unit=z_unit.astype(np.float64),
        theta_all=np.asarray(theta_flat, dtype=np.float64),
        theta_bin_index=bin_idx,
        theta_bin_edges=np.asarray(bin_edges, dtype=np.float64),
        x_mean=x_mean,
        x_std=x_std,
        meta_json_utf8=np.frombuffer(meta_json.encode("utf-8"), dtype=np.uint8),
    )

    # Figures: first two dims (for fdim>2, plot PC1/PC2 of z_unit if needed)
    if fdim >= 2:
        zx = z_unit[:, 0]
        zy = z_unit[:, 1]
    else:
        zx = z_unit[:, 0]
        zy = np.zeros_like(zx)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))
    theta_min = float(np.min(theta_flat))
    theta_max = float(np.max(theta_flat))
    sc0 = axes[0].scatter(zx, zy, c=theta_flat, cmap="viridis", s=6, alpha=0.75, linewidths=0)
    plt.colorbar(sc0, ax=axes[0], label=r"$\theta$")
    axes[0].set_title("Embedding (L2-normalized); color = continuous " + r"$\theta$")
    axes[0].set_xlabel(r"$z_1$")
    axes[0].set_ylabel(r"$z_2$" if fdim >= 2 else r"$z_1$ only")
    axes[0].set_aspect("equal", adjustable="box")
    if fdim == 2:
        circ = plt.Circle((0, 0), 1.0, fill=False, linestyle=":", color="gray", linewidth=1.0)
        axes[0].add_patch(circ)
        axes[0].set_xlim(-1.15, 1.15)
        axes[0].set_ylim(-1.15, 1.15)

    sc1 = axes[1].scatter(
        zx,
        zy,
        c=bin_idx,
        cmap="turbo",
        s=6,
        alpha=0.75,
        linewidths=0,
        vmin=0,
        vmax=max(int(args.num_theta_bins) - 1, 1),
    )
    plt.colorbar(sc1, ax=axes[1], label="theta bin index")
    axes[1].set_title(f"Equal-width bins on [{theta_min:.3g}, {theta_max:.3g}], K={int(args.num_theta_bins)}")
    axes[1].set_xlabel(r"$z_1$")
    axes[1].set_ylabel(r"$z_2$" if fdim >= 2 else r"$z_1$ only")
    axes[1].set_aspect("equal", adjustable="box")
    if fdim == 2:
        circ2 = plt.Circle((0, 0), 1.0, fill=False, linestyle=":", color="gray", linewidth=1.0)
        axes[1].add_patch(circ2)
        axes[1].set_xlim(-1.15, 1.15)
        axes[1].set_ylim(-1.15, 1.15)

    fig.tight_layout()
    svg_path = os.path.join(out_dir, "contrastive_embedding_2d.svg")
    png_path = os.path.join(out_dir, "contrastive_embedding_2d.png")
    fig.savefig(svg_path, format="svg")
    fig.savefig(png_path, format="png", dpi=160)
    plt.close(fig)

    summary_path = os.path.join(out_dir, "contrastive_embedding_2d_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(meta_json + "\n")
        f.write(f"embedding_npz: {os.path.abspath(emb_npz)}\n")
        f.write(f"figure_svg: {os.path.abspath(svg_path)}\n")
        f.write(f"figure_png: {os.path.abspath(png_path)}\n")

    print("[contrastive_embedding_2d] Saved:", flush=True)
    print(f"  - {os.path.abspath(emb_npz)}", flush=True)
    print(f"  - {os.path.abspath(svg_path)}", flush=True)
    print(f"  - {os.path.abspath(png_path)}", flush=True)
    print(f"  - {os.path.abspath(summary_path)}", flush=True)


if __name__ == "__main__":
    main()
