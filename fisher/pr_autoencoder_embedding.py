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


def pr_autoencoder_config_from_namespace(ns: Any, *, h_dim: int) -> PRAutoencoderConfig:
    """Build :class:`PRAutoencoderConfig` from an argparse-like namespace (PR-* fields optional)."""
    return PRAutoencoderConfig(
        z_dim=int(getattr(ns, "pr_autoencoder_z_dim", getattr(ns, "z_dim", 2))),
        h_dim=int(h_dim),
        hidden1=int(getattr(ns, "pr_autoencoder_hidden1", 100)),
        hidden2=int(getattr(ns, "pr_autoencoder_hidden2", 200)),
        train_samples=int(getattr(ns, "pr_autoencoder_train_samples", 12000)),
        train_epochs=int(getattr(ns, "pr_autoencoder_train_epochs", 200)),
        train_batch_size=int(getattr(ns, "pr_autoencoder_train_batch_size", 512)),
        train_lr=float(getattr(ns, "pr_autoencoder_train_lr", 1e-3)),
        lambda_pr=float(getattr(ns, "pr_autoencoder_lambda_pr", 1e-2)),
        pr_eps=float(getattr(ns, "pr_autoencoder_pr_eps", 1e-8)),
    )


def project_x_through_pr_autoencoder(
    x_low: np.ndarray,
    *,
    config: PRAutoencoderConfig,
    seed: int,
    device: torch.device,
    cache_dir: str,
    force_retrain: bool,
) -> tuple[np.ndarray, Path, bool, dict[str, np.ndarray]]:
    """Map low-dimensional rows ``x_low`` (N, z_dim) to embedded ``x_embed`` (N, h_dim) via PR-AE encoder.

    Returns ``(x_embed, cache_run_dir, loaded_from_cache, train_metrics)`` where ``train_metrics`` has
    keys ``loss``, ``recon``, ``pr`` (per-epoch arrays from PR-autoencoder training or cache).
    """
    x_arr = np.asarray(x_low, dtype=np.float64)
    if x_arr.ndim != 2 or int(x_arr.shape[1]) != int(config.z_dim):
        raise ValueError(
            f"x_low must have shape (N, z_dim) with z_dim={config.z_dim}; got {x_arr.shape}."
        )
    built = train_or_load_pr_autoencoder(
        config=config,
        seed=int(seed),
        device=device,
        cache_dir=cache_dir,
        force_retrain=force_retrain,
    )
    z_t = torch.from_numpy(x_arr).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        h_t, _ = built.model(z_t)
    x_embed = h_t.detach().cpu().numpy().astype(np.float64, copy=False)
    metrics = {k: np.asarray(v, dtype=np.float64) for k, v in built.metrics.items()}
    return x_embed, built.cache_run_dir, bool(built.loaded_from_cache), metrics


def build_randamp_gaussian_sqrtd_pr_autoencoder_dataset(ns: Any) -> PRAutoencoderDatasetBuildResult:
    """Sample base ``randamp_gaussian_sqrtd`` in z_dim, then embed x with a PR-autoencoder to h_dim.

    Used by tests and reproducibility checks; production datasets should use ``make_dataset`` +
    ``bin/project_dataset_pr_autoencoder.py`` instead of the removed ``dataset_family`` token.
    """
    h_dim = int(getattr(ns, "x_dim"))
    cfg = pr_autoencoder_config_from_namespace(ns, h_dim=h_dim)
    if h_dim < cfg.z_dim:
        raise ValueError(f"Target x_dim (h_dim) must be >= z_dim={cfg.z_dim}; got {h_dim}.")

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
    x_embed_all, cache_run_dir, loaded_from_cache, _metrics = project_x_through_pr_autoencoder(
        x_base_all,
        config=cfg,
        seed=int(getattr(ns, "seed")),
        device=device,
        cache_dir=cache_dir,
        force_retrain=force_retrain,
    )

    return PRAutoencoderDatasetBuildResult(
        theta_all=theta_all,
        x_embed_all=x_embed_all,
        base_dataset=base_dataset,
        embedder_config=cfg,
        cache_run_dir=cache_run_dir,
        loaded_from_cache=loaded_from_cache,
    )
