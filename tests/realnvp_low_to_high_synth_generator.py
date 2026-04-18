#!/usr/bin/env python3
"""Executable RealNVP low->high synthetic generator checks (no training).

This script is intentionally not a unittest/pytest module. It runs a small set of
self-checks and exits nonzero if any check fails.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def main() -> None:
    args = parse_args()
    device = ensure_device(args.device)
    code = run_checks(args, device)
    sys.exit(code)


if __name__ == "__main__":
    main()
