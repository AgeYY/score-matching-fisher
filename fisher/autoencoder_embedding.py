"""PR-regularized autoencoder low-to-high embedding utilities.

This module provides reusable components for learning a high-dimensional embedding
from low-dimensional latent variables via:

    loss = MSE(z_hat, z) - lambda_pr * participation_ratio(h)

where `h = encoder(z)` and `z_hat = decoder(h)`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PRAutoencoderConfig:
    z_dim: int = 2
    h_dim: int = 32
    hidden1: int = 100
    hidden2: int = 200
    train_samples: int = 12000
    train_epochs: int = 200
    train_batch_size: int = 512
    train_lr: float = 1e-3
    lambda_pr: float = 1e-2
    pr_eps: float = 1e-8


@dataclass(frozen=True)
class PRAutoencoderBuildResult:
    model: "InputAutoencoder"
    metrics: dict[str, np.ndarray]
    cache_run_dir: Path
    config: PRAutoencoderConfig
    loaded_from_cache: bool


class InputAutoencoder(nn.Module):
    def __init__(self, z_dim: int, h_dim: int, hidden1: int, hidden2: int) -> None:
        super().__init__()
        if h_dim < z_dim:
            raise ValueError(f"h_dim ({h_dim}) must be >= z_dim ({z_dim})")
        self.encoder = nn.Sequential(
            nn.Linear(z_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, h_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(h_dim, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, z_dim),
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(z)
        z_hat = self.decoder(h)
        return h, z_hat


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def participation_ratio(h: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    h_centered = h - h.mean(dim=0, keepdim=True)
    n = int(h_centered.shape[0])
    if n <= 1:
        return torch.zeros((), device=h.device, dtype=h.dtype)
    c = (h_centered.T @ h_centered) / float(n - 1)
    tr_c = torch.trace(c)
    tr_c2 = torch.trace(c @ c)
    return (tr_c * tr_c) / (tr_c2 + float(eps))


def config_cache_key(config: PRAutoencoderConfig, *, seed: int) -> str:
    payload = {
        "method": "pr_autoencoder",
        "seed": int(seed),
        "z_dim": int(config.z_dim),
        "h_dim": int(config.h_dim),
        "hidden1": int(config.hidden1),
        "hidden2": int(config.hidden2),
        "train_samples": int(config.train_samples),
        "train_epochs": int(config.train_epochs),
        "train_batch_size": int(config.train_batch_size),
        "train_lr": float(config.train_lr),
        "lambda_pr": float(config.lambda_pr),
        "pr_eps": float(config.pr_eps),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def train_or_load_pr_autoencoder(
    *,
    config: PRAutoencoderConfig,
    seed: int,
    device: torch.device,
    cache_dir: str | Path,
    force_retrain: bool = False,
    logger: Callable[[str], None] | None = print,
) -> PRAutoencoderBuildResult:
    """Build (or load) a PR-autoencoder and return model + metrics + cache artifacts."""
    set_torch_seed(seed)
    cache_root = Path(cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    run_dir = cache_root / f"pr_ae_{config_cache_key(config, seed=seed)}"
    encoder_path = run_dir / "encoder.pt"
    decoder_path = run_dir / "decoder.pt"
    metrics_path = run_dir / "train_metrics.npz"
    config_path = run_dir / "config.json"

    model = InputAutoencoder(
        z_dim=int(config.z_dim),
        h_dim=int(config.h_dim),
        hidden1=int(config.hidden1),
        hidden2=int(config.hidden2),
    ).to(device)

    if (
        (not bool(force_retrain))
        and encoder_path.exists()
        and decoder_path.exists()
        and metrics_path.exists()
        and config_path.exists()
    ):
        model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        model.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        with np.load(metrics_path, allow_pickle=False) as arr:
            metrics = {
                "loss": np.asarray(arr["loss"], dtype=np.float64),
                "recon": np.asarray(arr["recon"], dtype=np.float64),
                "pr": np.asarray(arr["pr"], dtype=np.float64),
            }
        if logger is not None:
            logger(f"[cache] hit: {run_dir}")
        model.eval()
        return PRAutoencoderBuildResult(
            model=model,
            metrics=metrics,
            cache_run_dir=run_dir,
            config=config,
            loaded_from_cache=True,
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    z_train = torch.randn(int(config.train_samples), int(config.z_dim), device=device)
    optim = torch.optim.Adam(model.parameters(), lr=float(config.train_lr))

    loss_hist: list[float] = []
    recon_hist: list[float] = []
    pr_hist: list[float] = []

    model.train()
    n = int(z_train.shape[0])
    b = int(config.train_batch_size)
    for epoch in range(int(config.train_epochs)):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_pr = 0.0
        n_batches = 0

        for i in range(0, n, b):
            idx = perm[i : i + b]
            batch = z_train[idx]
            h, z_hat = model(batch)
            recon = F.mse_loss(z_hat, batch)
            pr = participation_ratio(h, eps=float(config.pr_eps))
            loss = recon - float(config.lambda_pr) * pr

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            epoch_loss += float(loss.detach().cpu())
            epoch_recon += float(recon.detach().cpu())
            epoch_pr += float(pr.detach().cpu())
            n_batches += 1

        loss_hist.append(epoch_loss / max(1, n_batches))
        recon_hist.append(epoch_recon / max(1, n_batches))
        pr_hist.append(epoch_pr / max(1, n_batches))

        if logger is not None and ((epoch + 1) % max(1, int(config.train_epochs) // 10) == 0 or epoch == 0):
            logger(
                f"[train] epoch={epoch + 1:04d}/{config.train_epochs} "
                f"loss={loss_hist[-1]:.6f} recon={recon_hist[-1]:.6f} pr={pr_hist[-1]:.6f}"
            )

    model.eval()

    torch.save(model.encoder.state_dict(), encoder_path)
    torch.save(model.decoder.state_dict(), decoder_path)
    metrics = {
        "loss": np.asarray(loss_hist, dtype=np.float64),
        "recon": np.asarray(recon_hist, dtype=np.float64),
        "pr": np.asarray(pr_hist, dtype=np.float64),
    }
    np.savez_compressed(metrics_path, **metrics)

    config_json = {
        "seed": int(seed),
        "z_dim": int(config.z_dim),
        "h_dim": int(config.h_dim),
        "hidden1": int(config.hidden1),
        "hidden2": int(config.hidden2),
        "train_samples": int(config.train_samples),
        "train_epochs": int(config.train_epochs),
        "train_batch_size": int(config.train_batch_size),
        "train_lr": float(config.train_lr),
        "lambda_pr": float(config.lambda_pr),
        "pr_eps": float(config.pr_eps),
    }
    config_path.write_text(json.dumps(config_json, indent=2, sort_keys=True), encoding="utf-8")
    if logger is not None:
        logger(f"[cache] saved: {run_dir}")

    return PRAutoencoderBuildResult(
        model=model,
        metrics=metrics,
        cache_run_dir=run_dir,
        config=config,
        loaded_from_cache=False,
    )


def embed_latents(
    *,
    model: InputAutoencoder,
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(h, z_hat)` for latent batch `z` using a trained model."""
    with torch.no_grad():
        return model(z)
