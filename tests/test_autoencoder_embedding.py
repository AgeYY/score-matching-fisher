from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fisher.autoencoder_embedding import (
    PRAutoencoderConfig,
    participation_ratio,
    set_torch_seed,
    train_or_load_pr_autoencoder,
)


def test_train_and_cache_reload_smoke(tmp_path: Path) -> None:
    device = torch.device("cpu")
    cfg = PRAutoencoderConfig(
        z_dim=2,
        h_dim=10,
        hidden1=32,
        hidden2=64,
        train_samples=512,
        train_epochs=6,
        train_batch_size=128,
        train_lr=1e-3,
        lambda_pr=1e-2,
        pr_eps=1e-8,
    )

    out1 = train_or_load_pr_autoencoder(
        config=cfg,
        seed=11,
        device=device,
        cache_dir=tmp_path,
        force_retrain=True,
        logger=None,
    )
    assert not out1.loaded_from_cache
    assert out1.metrics["loss"].shape == (cfg.train_epochs,)
    assert out1.cache_run_dir.exists()

    set_torch_seed(11)
    z = torch.randn(128, cfg.z_dim, device=device)
    with torch.no_grad():
        h1, zhat1 = out1.model(z)
    assert h1.shape == (128, cfg.h_dim)
    assert zhat1.shape == (128, cfg.z_dim)
    assert torch.isfinite(h1).all()
    assert torch.isfinite(zhat1).all()

    pr1 = float(participation_ratio(h1, eps=cfg.pr_eps).item())
    assert np.isfinite(pr1)

    out2 = train_or_load_pr_autoencoder(
        config=cfg,
        seed=11,
        device=device,
        cache_dir=tmp_path,
        force_retrain=False,
        logger=None,
    )
    assert out2.loaded_from_cache

    set_torch_seed(11)
    z2 = torch.randn(128, cfg.z_dim, device=device)
    with torch.no_grad():
        h2, zhat2 = out2.model(z2)

    assert torch.allclose(z, z2)
    assert torch.allclose(h1, h2)
    assert torch.allclose(zhat1, zhat2)
