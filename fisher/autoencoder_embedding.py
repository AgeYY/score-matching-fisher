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
    adversarial_categorical: bool = False
    lambda_adv: float = 0.1
    adv_warmup_epochs: int = 0
    adv_ramp_epochs: int = 40
    adv_steps: int = 1
    adv_train_samples: int = 0
    adv_num_classes: int = 0
    adv_source_sha256: str = ""


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


class LinearAdversary(nn.Module):
    def __init__(self, h_dim: int, num_classes: int) -> None:
        super().__init__()
        if int(num_classes) < 2:
            raise ValueError("LinearAdversary requires num_classes >= 2.")
        self.classifier = nn.Linear(int(h_dim), int(num_classes))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.classifier(h)


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -float(ctx.scale) * grad_output, None


def gradient_reversal(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, float(scale))


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
    if bool(config.adversarial_categorical):
        payload.update(
            {
                "adversarial_categorical": True,
                "lambda_adv": float(config.lambda_adv),
                "adv_warmup_epochs": int(config.adv_warmup_epochs),
                "adv_ramp_epochs": int(config.adv_ramp_epochs),
                "adv_steps": int(config.adv_steps),
                "adv_train_samples": int(config.adv_train_samples),
                "adv_num_classes": int(config.adv_num_classes),
                "adv_source_sha256": str(config.adv_source_sha256),
            }
        )
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def adversarial_lambda_for_epoch(config: PRAutoencoderConfig, epoch: int) -> float:
    """Return the effective adversarial coefficient for a zero-based epoch."""
    if not bool(config.adversarial_categorical):
        return 0.0
    e = int(epoch)
    warm = max(0, int(config.adv_warmup_epochs))
    ramp = max(1, int(config.adv_ramp_epochs))
    if e < warm:
        return 0.0
    return float(config.lambda_adv) * min(1.0, float(e - warm + 1) / float(ramp))


def _load_metrics(metrics_path: Path) -> dict[str, np.ndarray]:
    with np.load(metrics_path, allow_pickle=False) as arr:
        return {str(k): np.asarray(arr[k], dtype=np.float64) for k in arr.files}


def train_or_load_pr_autoencoder(
    *,
    config: PRAutoencoderConfig,
    seed: int,
    device: torch.device,
    cache_dir: str | Path,
    force_retrain: bool = False,
    train_z: np.ndarray | None = None,
    train_y: np.ndarray | None = None,
    logger: Callable[[str], None] | None = print,
) -> PRAutoencoderBuildResult:
    """Build (or load) a PR-autoencoder and return model + metrics + cache artifacts."""
    adv_enabled = bool(config.adversarial_categorical)
    if adv_enabled:
        if train_z is None or train_y is None:
            raise ValueError("Adversarial categorical PR training requires train_z and train_y.")
        if int(config.adv_num_classes) < 2:
            raise ValueError("Adversarial categorical PR training requires adv_num_classes >= 2.")
        if float(config.lambda_adv) < 0.0:
            raise ValueError("lambda_adv must be non-negative.")
        if int(config.adv_steps) < 1:
            raise ValueError("adv_steps must be >= 1.")
    set_torch_seed(seed)
    cache_root = Path(cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    run_dir = cache_root / f"pr_ae_{config_cache_key(config, seed=seed)}"
    encoder_path = run_dir / "encoder.pt"
    decoder_path = run_dir / "decoder.pt"
    adversary_path = run_dir / "adversary.pt"
    metrics_path = run_dir / "train_metrics.npz"
    config_path = run_dir / "config.json"

    model = InputAutoencoder(
        z_dim=int(config.z_dim),
        h_dim=int(config.h_dim),
        hidden1=int(config.hidden1),
        hidden2=int(config.hidden2),
    ).to(device)
    adversary = (
        LinearAdversary(h_dim=int(config.h_dim), num_classes=int(config.adv_num_classes)).to(device)
        if adv_enabled
        else None
    )

    if (
        (not bool(force_retrain))
        and encoder_path.exists()
        and decoder_path.exists()
        and metrics_path.exists()
        and config_path.exists()
        and ((not adv_enabled) or adversary_path.exists())
    ):
        model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        model.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        if adversary is not None:
            adversary.load_state_dict(torch.load(adversary_path, map_location=device))
        metrics = _load_metrics(metrics_path)
        if logger is not None:
            logger(f"[cache] hit: {run_dir}")
        model.eval()
        if adversary is not None:
            adversary.eval()
        return PRAutoencoderBuildResult(
            model=model,
            metrics=metrics,
            cache_run_dir=run_dir,
            config=config,
            loaded_from_cache=True,
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    if adv_enabled:
        z_arr = np.asarray(train_z, dtype=np.float32)
        y_arr = np.asarray(train_y, dtype=np.int64).reshape(-1)
        if z_arr.ndim != 2 or int(z_arr.shape[1]) != int(config.z_dim):
            raise ValueError(f"train_z must have shape (N, {config.z_dim}); got {z_arr.shape}.")
        if int(y_arr.shape[0]) != int(z_arr.shape[0]):
            raise ValueError("train_y length must match train_z rows.")
        if np.any((y_arr < 0) | (y_arr >= int(config.adv_num_classes))):
            raise ValueError("train_y contains labels outside [0, adv_num_classes - 1].")
        z_train = torch.from_numpy(z_arr).to(device=device, dtype=torch.float32)
        y_train = torch.from_numpy(y_arr).to(device=device, dtype=torch.long)
    else:
        z_train = torch.randn(int(config.train_samples), int(config.z_dim), device=device)
        y_train = None
    optim = torch.optim.Adam(model.parameters(), lr=float(config.train_lr))
    adv_optim = (
        torch.optim.Adam(adversary.parameters(), lr=float(config.train_lr)) if adversary is not None else None
    )

    loss_hist: list[float] = []
    recon_hist: list[float] = []
    pr_hist: list[float] = []
    adv_ce_hist: list[float] = []
    adv_acc_hist: list[float] = []
    lambda_adv_hist: list[float] = []

    model.train()
    if adversary is not None:
        adversary.train()
    n = int(z_train.shape[0])
    b = int(config.train_batch_size)
    for epoch in range(int(config.train_epochs)):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_pr = 0.0
        epoch_adv_ce = 0.0
        epoch_adv_acc = 0.0
        epoch_adv_count = 0
        n_batches = 0
        lambda_adv_eff = adversarial_lambda_for_epoch(config, epoch)

        for i in range(0, n, b):
            idx = perm[i : i + b]
            batch = z_train[idx]
            labels = y_train[idx] if y_train is not None else None

            if adversary is not None and adv_optim is not None and labels is not None and float(lambda_adv_eff) > 0.0:
                for _ in range(max(0, int(config.adv_steps) - 1)):
                    with torch.no_grad():
                        h_detached, _ = model(batch)
                    adv_logits = adversary(h_detached.detach())
                    adv_loss_only = F.cross_entropy(adv_logits, labels)
                    adv_optim.zero_grad(set_to_none=True)
                    adv_loss_only.backward()
                    adv_optim.step()

            h, z_hat = model(batch)
            recon = F.mse_loss(z_hat, batch)
            pr = participation_ratio(h, eps=float(config.pr_eps))
            loss = recon - float(config.lambda_pr) * pr
            adv_ce = None
            adv_acc = None
            if adversary is not None and labels is not None:
                logits = adversary(gradient_reversal(h, scale=1.0))
                adv_ce = F.cross_entropy(logits, labels)
                loss = loss + float(lambda_adv_eff) * adv_ce
                adv_acc = (logits.argmax(dim=1) == labels).to(torch.float32).mean()

            optim.zero_grad(set_to_none=True)
            if adv_optim is not None:
                adv_optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if adv_optim is not None:
                adv_optim.step()

            epoch_loss += float(loss.detach().cpu())
            epoch_recon += float(recon.detach().cpu())
            epoch_pr += float(pr.detach().cpu())
            if adv_ce is not None and adv_acc is not None:
                epoch_adv_ce += float(adv_ce.detach().cpu())
                epoch_adv_acc += float(adv_acc.detach().cpu())
                epoch_adv_count += 1
            n_batches += 1

        loss_hist.append(epoch_loss / max(1, n_batches))
        recon_hist.append(epoch_recon / max(1, n_batches))
        pr_hist.append(epoch_pr / max(1, n_batches))
        if adv_enabled:
            adv_ce_hist.append(epoch_adv_ce / max(1, epoch_adv_count))
            adv_acc_hist.append(epoch_adv_acc / max(1, epoch_adv_count))
            lambda_adv_hist.append(float(lambda_adv_eff))

        if logger is not None and ((epoch + 1) % max(1, int(config.train_epochs) // 10) == 0 or epoch == 0):
            msg = (
                f"[train] epoch={epoch + 1:04d}/{config.train_epochs} "
                f"loss={loss_hist[-1]:.6f} recon={recon_hist[-1]:.6f} pr={pr_hist[-1]:.6f}"
            )
            if adv_enabled:
                msg += (
                    f" adv_ce={adv_ce_hist[-1]:.6f} adv_acc={adv_acc_hist[-1]:.4f} "
                    f"lambda_adv={lambda_adv_hist[-1]:.6f}"
                )
            logger(msg)

    model.eval()
    if adversary is not None:
        adversary.eval()

    torch.save(model.encoder.state_dict(), encoder_path)
    torch.save(model.decoder.state_dict(), decoder_path)
    if adversary is not None:
        torch.save(adversary.state_dict(), adversary_path)
    metrics = {
        "loss": np.asarray(loss_hist, dtype=np.float64),
        "recon": np.asarray(recon_hist, dtype=np.float64),
        "pr": np.asarray(pr_hist, dtype=np.float64),
    }
    if adv_enabled:
        metrics.update(
            {
                "adv_ce": np.asarray(adv_ce_hist, dtype=np.float64),
                "adv_acc": np.asarray(adv_acc_hist, dtype=np.float64),
                "lambda_adv_eff": np.asarray(lambda_adv_hist, dtype=np.float64),
            }
        )
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
        "adversarial_categorical": bool(config.adversarial_categorical),
        "lambda_adv": float(config.lambda_adv),
        "adv_warmup_epochs": int(config.adv_warmup_epochs),
        "adv_ramp_epochs": int(config.adv_ramp_epochs),
        "adv_steps": int(config.adv_steps),
        "adv_train_samples": int(config.adv_train_samples),
        "adv_num_classes": int(config.adv_num_classes),
        "adv_source_sha256": str(config.adv_source_sha256),
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
