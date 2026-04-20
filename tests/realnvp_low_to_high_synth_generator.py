#!/usr/bin/env python3
"""Executable RealNVP low->high synthetic generator checks (no training).

This script is intentionally not a unittest/pytest module. It runs a small set of
self-checks and exits nonzero if any check fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")

try:
    from glasflow import RealNVP
except Exception as exc:  # pragma: no cover - import-time environment check
    raise RuntimeError(
        "glasflow is required for this script. Install it in geo_diffusion, e.g.:\n"
        "  mamba run -n geo_diffusion pip install glasflow"
    ) from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run executable checks for low->high dimensional synthetic generation "
            "via zero-padding + RealNVP (Option B only, no training)."
        )
    )
    p.add_argument("--z-dim", type=int, default=2)
    p.add_argument("--h-dim", type=int, default=32)
    p.add_argument("--n-transforms", type=int, default=6)
    p.add_argument("--hidden-width", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--positive-output", action="store_true")
    p.add_argument("--save-viz", action="store_true", help="Save synthetic data visualizations.")
    p.add_argument("--viz-samples", type=int, default=2000, help="Number of synthetic samples for visualization.")
    p.add_argument(
        "--viz-output-dir",
        type=str,
        default="data/realnvp_low_to_high_synth_viz",
        help="Directory for visualization artifacts when --save-viz is set.",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested (`--device cuda`) but CUDA is unavailable on this machine.")
    return torch.device(device)


class LowToHighRealNVPGenerator(nn.Module):
    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        n_transforms: int,
        hidden_width: int,
        positive_output: bool = False,
    ) -> None:
        super().__init__()
        if h_dim < z_dim:
            raise ValueError(f"h_dim ({h_dim}) must be >= z_dim ({z_dim})")
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.positive_output = positive_output
        self.flow = RealNVP(
            n_inputs=h_dim,
            n_transforms=n_transforms,
            n_neurons=hidden_width,
            batch_norm_between_transforms=True,
        )

    def pad_latent(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.shape[0]
        if self.h_dim == self.z_dim:
            return z
        pad = torch.zeros(batch, self.h_dim - self.z_dim, device=z.device, dtype=z.dtype)
        return torch.cat([z, pad], dim=-1)

    def flow_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.flow.forward(x)
        if isinstance(out, tuple) and len(out) == 2:
            h, logdet = out
            return h, logdet
        if torch.is_tensor(out):
            h = out
            logdet = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
            return h, logdet
        raise RuntimeError(
            f"Unexpected RealNVP.forward output type: {type(out)}. "
            "Expected tensor or (tensor, logdet) tuple."
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.pad_latent(z)
        h, logdet = self.flow_forward(x0)
        if self.positive_output:
            h = F.softplus(h)
        return h, logdet


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def check_dimensionality_increase(model: LowToHighRealNVPGenerator, args: argparse.Namespace, device: torch.device) -> None:
    _assert(args.h_dim > args.z_dim, "Expected dimensionality increase: h_dim must be > z_dim.")
    z = torch.randn(args.batch_size, args.z_dim, device=device)
    h, logdet = model(z)
    _assert(tuple(z.shape) == (args.batch_size, args.z_dim), f"Unexpected z shape: {tuple(z.shape)}")
    _assert(tuple(h.shape) == (args.batch_size, args.h_dim), f"Unexpected h shape: {tuple(h.shape)}")
    _assert(tuple(logdet.shape) == (args.batch_size,), f"Unexpected logdet shape: {tuple(logdet.shape)}")


def check_padding_preserves_latent(model: LowToHighRealNVPGenerator, args: argparse.Namespace, device: torch.device) -> None:
    z = torch.randn(args.batch_size, args.z_dim, device=device)
    padded = model.pad_latent(z)
    _assert(tuple(padded.shape) == (args.batch_size, args.h_dim), f"Unexpected padded shape: {tuple(padded.shape)}")
    _assert(torch.allclose(padded[:, : args.z_dim], z), "First z_dim coordinates changed after padding.")
    tail = padded[:, args.z_dim :]
    _assert(torch.allclose(tail, torch.zeros_like(tail)), "Padding tail is not all zeros.")


def check_finite_outputs(model: LowToHighRealNVPGenerator, args: argparse.Namespace, device: torch.device) -> None:
    z = torch.randn(args.batch_size, args.z_dim, device=device)
    h, logdet = model(z)
    _assert(bool(torch.isfinite(h).all()), "Non-finite values found in h output.")
    _assert(bool(torch.isfinite(logdet).all()), "Non-finite values found in logdet output.")


def check_positive_output_branch(args: argparse.Namespace, device: torch.device) -> None:
    model = LowToHighRealNVPGenerator(
        z_dim=args.z_dim,
        h_dim=args.h_dim,
        n_transforms=args.n_transforms,
        hidden_width=args.hidden_width,
        positive_output=True,
    ).to(device)
    z = torch.randn(args.batch_size, args.z_dim, device=device)
    h, _ = model(z)
    _assert(bool((h >= 0).all()), "positive_output=True but negative values were produced.")


def check_reproducibility(args: argparse.Namespace, device: torch.device) -> None:
    set_seed(args.seed)
    m1 = LowToHighRealNVPGenerator(
        z_dim=args.z_dim,
        h_dim=args.h_dim,
        n_transforms=args.n_transforms,
        hidden_width=args.hidden_width,
        positive_output=args.positive_output,
    ).to(device)
    z1 = torch.randn(args.batch_size, args.z_dim, device=device)
    h1, l1 = m1(z1)

    set_seed(args.seed)
    m2 = LowToHighRealNVPGenerator(
        z_dim=args.z_dim,
        h_dim=args.h_dim,
        n_transforms=args.n_transforms,
        hidden_width=args.hidden_width,
        positive_output=args.positive_output,
    ).to(device)
    z2 = torch.randn(args.batch_size, args.z_dim, device=device)
    h2, l2 = m2(z2)

    _assert(torch.allclose(z1, z2), "Latent sampling is not reproducible for same seed.")
    _assert(torch.allclose(h1, h2), "Generator output is not reproducible for same seed.")
    _assert(torch.allclose(l1, l2), "logdet output is not reproducible for same seed.")


def check_invalid_hdim_raises(args: argparse.Namespace) -> None:
    try:
        _ = LowToHighRealNVPGenerator(
            z_dim=max(2, args.z_dim),
            h_dim=max(1, args.z_dim - 1),
            n_transforms=args.n_transforms,
            hidden_width=args.hidden_width,
            positive_output=False,
        )
    except ValueError:
        return
    raise AssertionError("Expected ValueError when h_dim < z_dim, but no error was raised.")


def run_checks(args: argparse.Namespace, device: torch.device) -> int:
    set_seed(args.seed)
    model = LowToHighRealNVPGenerator(
        z_dim=args.z_dim,
        h_dim=args.h_dim,
        n_transforms=args.n_transforms,
        hidden_width=args.hidden_width,
        positive_output=args.positive_output,
    ).to(device)

    checks: List[Tuple[str, Callable[[], None]]] = [
        ("dimensionality increase and output shapes", lambda: check_dimensionality_increase(model, args, device)),
        ("zero-padding preserves latent prefix", lambda: check_padding_preserves_latent(model, args, device)),
        ("outputs are finite", lambda: check_finite_outputs(model, args, device)),
        ("positive_output branch", lambda: check_positive_output_branch(args, device)),
        ("seed reproducibility", lambda: check_reproducibility(args, device)),
        ("invalid h_dim < z_dim raises", lambda: check_invalid_hdim_raises(args)),
    ]

    failed: List[str] = []
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
        return 1

    print("\nSummary: PASS")
    return 0


def _pca_project_2d(x: np.ndarray) -> np.ndarray:
    x0 = x - x.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(x0, full_matrices=False)
    basis = vh[:2].T
    return x0 @ basis


def save_visualizations(args: argparse.Namespace, device: torch.device) -> None:
    set_seed(args.seed)
    model = LowToHighRealNVPGenerator(
        z_dim=args.z_dim,
        h_dim=args.h_dim,
        n_transforms=args.n_transforms,
        hidden_width=args.hidden_width,
        positive_output=args.positive_output,
    ).to(device)
    model.eval()
    n = int(max(16, args.viz_samples))
    with torch.no_grad():
        z = torch.randn(n, args.z_dim, device=device)
        h, logdet = model(z)
    z_np = z.detach().cpu().numpy()
    h_np = h.detach().cpu().numpy()
    logdet_np = logdet.detach().cpu().numpy()

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

    ax11.hist(logdet_np, bins=60, color="#54a24b", alpha=0.85)
    ax11.set_xlabel("logdet")
    ax11.set_ylabel("count")
    ax11.set_title("Flow logdet distribution")

    fig.tight_layout()
    panel_png = out_dir / "realnvp_synth_overview.png"
    panel_svg = out_dir / "realnvp_synth_overview.svg"
    fig.savefig(panel_png, dpi=180, bbox_inches="tight")
    fig.savefig(panel_svg, bbox_inches="tight")
    plt.close(fig)

    sample_npz = out_dir / "realnvp_synth_samples.npz"
    np.savez_compressed(
        sample_npz,
        z=z_np.astype(np.float64, copy=False),
        h=h_np.astype(np.float64, copy=False),
        logdet=logdet_np.astype(np.float64, copy=False),
        seed=np.asarray([args.seed], dtype=np.int64),
        z_dim=np.asarray([args.z_dim], dtype=np.int64),
        h_dim=np.asarray([args.h_dim], dtype=np.int64),
        n_samples=np.asarray([n], dtype=np.int64),
    )
    print(f"[viz] Saved: {panel_png}")
    print(f"[viz] Saved: {panel_svg}")
    print(f"[viz] Saved: {sample_npz}")


def main() -> None:
    args = parse_args()
    device = ensure_device(args.device)
    code = run_checks(args, device)
    if code == 0 and bool(args.save_viz):
        save_visualizations(args, device)
    sys.exit(code)


if __name__ == "__main__":
    main()
