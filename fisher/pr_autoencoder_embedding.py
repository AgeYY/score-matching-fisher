"""PR-autoencoder low-to-high embedding utilities for synthetic datasets."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fisher.autoencoder_embedding import PRAutoencoderConfig, train_or_load_pr_autoencoder
from fisher.shared_fisher_est import build_dataset_from_args


@dataclass(frozen=True)
class PRAutoencoderDatasetBuildResult:
    theta_all: np.ndarray
    x_embed_all: np.ndarray
    base_dataset: Any
    embedder_config: PRAutoencoderConfig
    cache_run_dir: Path
    loaded_from_cache: bool


def _copy_namespace(ns: Any) -> Namespace:
    out = Namespace()
    for k, v in vars(ns).items():
        setattr(out, k, v)
    return out


def _config_from_namespace(ns: Any, *, x_dim_target: int) -> PRAutoencoderConfig:
    """Build config with fixed defaults (override-able only via non-public namespace attrs)."""
    return PRAutoencoderConfig(
        z_dim=int(getattr(ns, "pr_autoencoder_z_dim", 2)),
        h_dim=int(x_dim_target),
        hidden1=int(getattr(ns, "pr_autoencoder_hidden1", 100)),
        hidden2=int(getattr(ns, "pr_autoencoder_hidden2", 200)),
        train_samples=int(getattr(ns, "pr_autoencoder_train_samples", 12000)),
        train_epochs=int(getattr(ns, "pr_autoencoder_train_epochs", 200)),
        train_batch_size=int(getattr(ns, "pr_autoencoder_train_batch_size", 512)),
        train_lr=float(getattr(ns, "pr_autoencoder_train_lr", 1e-3)),
        lambda_pr=float(getattr(ns, "pr_autoencoder_lambda_pr", 1e-2)),
        pr_eps=float(getattr(ns, "pr_autoencoder_pr_eps", 1e-8)),
    )


def build_randamp_gaussian_sqrtd_pr_autoencoder_dataset(ns: Any) -> PRAutoencoderDatasetBuildResult:
    """Generate dataset by sampling base randamp_sqrtd in z_dim then embedding with PR-autoencoder."""
    x_dim_target = int(getattr(ns, "x_dim"))
    cfg = _config_from_namespace(ns, x_dim_target=x_dim_target)
    if x_dim_target < cfg.z_dim:
        raise ValueError(
            f"--x-dim must be >= {cfg.z_dim} for randamp_gaussian_sqrtd_pr_autoencoder; got {x_dim_target}."
        )

    device_name = str(getattr(ns, "device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested (`--device cuda`) but CUDA is unavailable on this machine.")
    device = torch.device(device_name)

    base_ns = _copy_namespace(ns)
    base_ns.dataset_family = "randamp_gaussian_sqrtd"
    base_ns.x_dim = int(cfg.z_dim)
    base_dataset = build_dataset_from_args(base_ns)

    n_total = int(getattr(ns, "n_total"))
    theta_all, x_base_all = base_dataset.sample_joint(n_total)

    cache_dir = str(getattr(ns, "pr_autoencoder_cache_dir", "data/pr_autoencoder_cache"))
    force_retrain = bool(getattr(ns, "pr_autoencoder_force_retrain", False))
    built = train_or_load_pr_autoencoder(
        config=cfg,
        seed=int(getattr(ns, "seed")),
        device=device,
        cache_dir=cache_dir,
        force_retrain=force_retrain,
    )

    z_t = torch.from_numpy(np.asarray(x_base_all, dtype=np.float64)).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        h_t, _ = built.model(z_t)
    x_embed_all = h_t.detach().cpu().numpy().astype(np.float64, copy=False)

    return PRAutoencoderDatasetBuildResult(
        theta_all=theta_all,
        x_embed_all=x_embed_all,
        base_dataset=base_dataset,
        embedder_config=cfg,
        cache_run_dir=built.cache_run_dir,
        loaded_from_cache=bool(built.loaded_from_cache),
    )
