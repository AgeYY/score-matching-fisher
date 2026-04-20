#!/usr/bin/env python3
"""Executable low->high synthetic generator checks (PR-regularized autoencoder only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from fisher.autoencoder_embedding import (
    InputAutoencoder,
    PRAutoencoderConfig,
    embed_latents,
    participation_ratio,
    set_torch_seed,
    train_or_load_pr_autoencoder,
)

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run executable checks for low->high dimensional synthetic generation using a "
            "PR-regularized autoencoder (fisher/autoencoder_embedding.py)."
        )
    )
    p.add_argument("--z-dim", type=int, default=2)
    p.add_argument("--h-dim", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")

    p.add_argument("--enc-hidden1", type=int, default=100)
    p.add_argument("--enc-hidden2", type=int, default=200)
    p.add_argument("--train-samples", type=int, default=12000)
    p.add_argument("--train-epochs", type=int, default=200)
    p.add_argument("--train-batch-size", type=int, default=512)
    p.add_argument("--train-lr", type=float, default=1e-3)
    p.add_argument("--lambda-pr", type=float, default=1e-2)
    p.add_argument("--pr-eps", type=float, default=1e-8)
    p.add_argument("--recon-threshold", type=float, default=0.5)
    p.add_argument("--pr-threshold", type=float, default=2.0)
    p.add_argument("--cache-dir", type=str, default="data/pr_autoencoder_cache")
    p.add_argument("--force-retrain", action="store_true")

    p.add_argument("--save-viz", action="store_true", help="Save synthetic data visualizations.")
    p.add_argument("--viz-samples", type=int, default=2000, help="Number of synthetic samples for visualization.")
    p.add_argument(
        "--viz-output-dir",
        type=str,
        default="data/pr_autoencoder_low_to_high_synth_viz",
        help="Directory for visualization artifacts when --save-viz is set.",
    )
    return p.parse_args()


def ensure_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested (`--device cuda`) but CUDA is unavailable on this machine.")
    return torch.device(device)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _pca_project_2d(x: np.ndarray) -> np.ndarray:
    x0 = x - x.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(x0, full_matrices=False)
    basis = vh[:2].T
    return x0 @ basis


def _pr_config_from_args(args: argparse.Namespace) -> PRAutoencoderConfig:
    return PRAutoencoderConfig(
        z_dim=int(args.z_dim),
        h_dim=int(args.h_dim),
        hidden1=int(args.enc_hidden1),
        hidden2=int(args.enc_hidden2),
        train_samples=int(args.train_samples),
        train_epochs=int(args.train_epochs),
        train_batch_size=int(args.train_batch_size),
        train_lr=float(args.train_lr),
        lambda_pr=float(args.lambda_pr),
        pr_eps=float(args.pr_eps),
    )


def check_dimensionality_increase_pr(model: InputAutoencoder, args: argparse.Namespace, device: torch.device) -> None:
    _assert(args.h_dim > args.z_dim, "Expected dimensionality increase: h_dim must be > z_dim.")
    z = torch.randn(args.batch_size, args.z_dim, device=device)
    h, z_hat = model(z)
    _assert(tuple(z.shape) == (args.batch_size, args.z_dim), f"Unexpected z shape: {tuple(z.shape)}")
    _assert(tuple(h.shape) == (args.batch_size, args.h_dim), f"Unexpected h shape: {tuple(h.shape)}")
    _assert(tuple(z_hat.shape) == (args.batch_size, args.z_dim), f"Unexpected z_hat shape: {tuple(z_hat.shape)}")


def check_finite_outputs_pr(model: InputAutoencoder, args: argparse.Namespace, device: torch.device) -> None:
    z = torch.randn(args.batch_size, args.z_dim, device=device)
    h, z_hat = model(z)
    _assert(bool(torch.isfinite(h).all()), "Non-finite values found in h output.")
    _assert(bool(torch.isfinite(z_hat).all()), "Non-finite values found in z_hat output.")


def check_quality_pr(model: InputAutoencoder, args: argparse.Namespace, device: torch.device) -> None:
    z = torch.randn(max(args.batch_size * 4, 256), args.z_dim, device=device)
    with torch.no_grad():
        h, z_hat = model(z)
        recon = F.mse_loss(z_hat, z).item()
        pr = float(participation_ratio(h, eps=float(args.pr_eps)).item())
    _assert(recon <= float(args.recon_threshold), f"Reconstruction MSE too high: {recon:.6f} > {args.recon_threshold}")
    _assert(pr >= float(args.pr_threshold), f"Participation ratio too low: {pr:.6f} < {args.pr_threshold}")


def check_reproducibility_pr(args: argparse.Namespace, device: torch.device) -> None:
    cfg = _pr_config_from_args(args)
    out1 = train_or_load_pr_autoencoder(
        config=cfg,
        seed=int(args.seed),
        device=device,
        cache_dir=args.cache_dir,
        force_retrain=False,
    )
    out2 = train_or_load_pr_autoencoder(
        config=cfg,
        seed=int(args.seed),
        device=device,
        cache_dir=args.cache_dir,
        force_retrain=False,
    )

    set_torch_seed(args.seed)
    z1 = torch.randn(args.batch_size, args.z_dim, device=device)
    set_torch_seed(args.seed)
    z2 = torch.randn(args.batch_size, args.z_dim, device=device)
    _assert(torch.allclose(z1, z2), "Latent sampling is not reproducible for same seed.")

    with torch.no_grad():
        h1, zhat1 = out1.model(z1)
        h2, zhat2 = out2.model(z2)
    _assert(torch.allclose(h1, h2), "PR-autoencoder outputs are not reproducible for same cache/config.")
    _assert(torch.allclose(zhat1, zhat2), "PR-autoencoder recon outputs are not reproducible for same cache/config.")


def check_invalid_hdim_raises_pr(args: argparse.Namespace) -> None:
    try:
        _ = InputAutoencoder(
            z_dim=max(2, args.z_dim),
            h_dim=max(1, args.z_dim - 1),
            hidden1=args.enc_hidden1,
            hidden2=args.enc_hidden2,
        )
    except ValueError:
        return
    raise AssertionError("Expected ValueError when h_dim < z_dim, but no error was raised.")


def run_checks_pr(args: argparse.Namespace, device: torch.device) -> tuple[int, InputAutoencoder | None, dict[str, np.ndarray] | None, Path | None]:
    cfg = _pr_config_from_args(args)
    build = train_or_load_pr_autoencoder(
        config=cfg,
        seed=int(args.seed),
        device=device,
        cache_dir=args.cache_dir,
        force_retrain=bool(args.force_retrain),
    )
    model = build.model
    metrics = build.metrics
    cache_run_dir = build.cache_run_dir

    checks: list[tuple[str, Callable[[], None]]] = [
        ("dimensionality increase and output shapes", lambda: check_dimensionality_increase_pr(model, args, device)),
        ("outputs are finite", lambda: check_finite_outputs_pr(model, args, device)),
        ("quality checks (recon + PR)", lambda: check_quality_pr(model, args, device)),
        ("seed/cache reproducibility", lambda: check_reproducibility_pr(args, device)),
        ("invalid h_dim < z_dim raises", lambda: check_invalid_hdim_raises_pr(args)),
    ]

    failed: list[str] = []
    for name, fn in checks:
        try:
            fn()
            print(f"[pass] {name}")
        except Exception as exc:
            failed.append(f"{name}: {exc}")
            print(f"[fail] {name}: {exc}")

    if failed:
        print("\nSummary: FAIL")
        for item in failed:
            print(f" - {item}")
        return 1, model, metrics, cache_run_dir

    print("\nSummary: PASS")
    return 0, model, metrics, cache_run_dir


def save_visualizations(
    args: argparse.Namespace,
    device: torch.device,
    model: nn.Module,
    metrics: dict[str, np.ndarray] | None,
    cache_run_dir: Path | None,
) -> None:
    set_torch_seed(args.seed)
    model = model.to(device)
    model.eval()

    n = int(max(16, args.viz_samples))
    with torch.no_grad():
        z = torch.randn(n, args.z_dim, device=device)
        h, z_hat = embed_latents(model=model, z=z)

    z_np = z.detach().cpu().numpy()
    h_np = h.detach().cpu().numpy()
    aux_np = z_hat.detach().cpu().numpy()

    out_dir = Path(args.viz_output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    ax00, ax01, ax10, ax11 = axs.ravel()

    if args.z_dim >= 2:
        sc = ax00.scatter(z_np[:, 0], z_np[:, 1], c=np.linalg.norm(z_np, axis=1), s=8, alpha=0.55, cmap="viridis")
        fig.colorbar(sc, ax=ax00, label="||z||")
        ax00.set_xlabel("z1")
        ax00.set_ylabel("z2")
        ax00.set_title("Latent z (first 2 dims)")
    else:
        ax00.hist(z_np[:, 0], bins=60, color="#4c78a8", alpha=0.85)
        ax00.set_xlabel("z1")
        ax00.set_ylabel("count")
        ax00.set_title("Latent z histogram")

    if args.h_dim >= 2:
        sc_h = ax01.scatter(h_np[:, 0], h_np[:, 1], c=z_np[:, 0], s=8, alpha=0.55, cmap="plasma")
        fig.colorbar(sc_h, ax=ax01, label="z1")
        ax01.set_xlabel("h1")
        ax01.set_ylabel("h2")
        ax01.set_title("Embedded h (first 2 dims)")
    else:
        ax01.hist(h_np[:, 0], bins=60, color="#f58518", alpha=0.85)
        ax01.set_xlabel("h1")
        ax01.set_ylabel("count")
        ax01.set_title("Embedded h histogram")

    pca = _pca_project_2d(h_np) if args.h_dim >= 2 else np.concatenate([h_np, np.zeros_like(h_np)], axis=1)
    sc_pca = ax10.scatter(pca[:, 0], pca[:, 1], c=np.linalg.norm(h_np, axis=1), s=8, alpha=0.55, cmap="viridis")
    fig.colorbar(sc_pca, ax=ax10, label="||h||")
    ax10.set_xlabel("PC1")
    ax10.set_ylabel("PC2")
    ax10.set_title("PCA of h")

    recon_err = np.mean((aux_np - z_np) ** 2, axis=1)
    ax11.hist(recon_err, bins=60, color="#54a24b", alpha=0.85)
    ax11.set_xlabel("per-sample recon MSE")
    ax11.set_ylabel("count")
    ax11.set_title("Reconstruction error distribution")

    fig.tight_layout()
    prefix = "pr_autoencoder"
    panel_png = out_dir / f"{prefix}_overview.png"
    panel_svg = out_dir / f"{prefix}_overview.svg"
    fig.savefig(panel_png, dpi=180, bbox_inches="tight")
    fig.savefig(panel_svg, bbox_inches="tight")
    plt.close(fig)

    if metrics is not None:
        fig2, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        epochs = np.arange(1, metrics["loss"].shape[0] + 1)
        axes[0].plot(epochs, metrics["loss"], color="#4c78a8")
        axes[0].set_title("Total loss")
        axes[0].set_xlabel("epoch")
        axes[1].plot(epochs, metrics["recon"], color="#f58518")
        axes[1].set_title("Reconstruction MSE")
        axes[1].set_xlabel("epoch")
        axes[2].plot(epochs, metrics["pr"], color="#54a24b")
        axes[2].set_title("Participation ratio")
        axes[2].set_xlabel("epoch")
        fig2.tight_layout()
        curves_png = out_dir / "pr_autoencoder_training_curves.png"
        curves_svg = out_dir / "pr_autoencoder_training_curves.svg"
        fig2.savefig(curves_png, dpi=180, bbox_inches="tight")
        fig2.savefig(curves_svg, bbox_inches="tight")
        plt.close(fig2)
        print(f"[viz] Saved: {curves_png}")
        print(f"[viz] Saved: {curves_svg}")

    sample_npz = out_dir / f"{prefix}_samples.npz"
    save_kwargs = {
        "z": z_np.astype(np.float64, copy=False),
        "h": h_np.astype(np.float64, copy=False),
        "z_hat": aux_np.astype(np.float64, copy=False),
        "seed": np.asarray([args.seed], dtype=np.int64),
        "z_dim": np.asarray([args.z_dim], dtype=np.int64),
        "h_dim": np.asarray([args.h_dim], dtype=np.int64),
        "n_samples": np.asarray([n], dtype=np.int64),
    }
    if metrics is not None:
        save_kwargs["train_loss"] = metrics["loss"]
        save_kwargs["train_recon"] = metrics["recon"]
        save_kwargs["train_pr"] = metrics["pr"]
    np.savez_compressed(sample_npz, **save_kwargs)

    print(f"[viz] Saved: {panel_png}")
    print(f"[viz] Saved: {panel_svg}")
    print(f"[viz] Saved: {sample_npz}")
    if cache_run_dir is not None:
        print(f"[cache] artifacts: {cache_run_dir}")


def main() -> None:
    args = parse_args()
    device = ensure_device(args.device)

    code, model, metrics, cache_run_dir = run_checks_pr(args, device)

    if code == 0 and bool(args.save_viz) and model is not None:
        save_visualizations(args, device, model=model, metrics=metrics, cache_run_dir=cache_run_dir)
    sys.exit(code)


if __name__ == "__main__":
    main()
