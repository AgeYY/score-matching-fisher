"""Fixed RealNVP low-to-high embedding utilities for synthetic datasets."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from fisher.shared_fisher_est import build_dataset_from_args


def _require_glasflow_realnvp() -> Any:
    try:
        from glasflow import RealNVP  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - import-time environment check
        raise RuntimeError(
            "glasflow is required for dataset_family='randamp_gaussian_sqrtd_realnvp'. Install it in "
            "geo_diffusion, e.g.:\n"
            "  mamba run -n geo_diffusion pip install glasflow"
        ) from exc
    return RealNVP


@dataclass(frozen=True)
class RealNVPLowToHighConfig:
    z_dim: int = 2
    n_transforms: int = 6
    hidden_width: int = 128
    positive_output: bool = False
    batch_norm_between_transforms: bool = True


class FixedRealNVPLowToHighEmbedder:
    """Untrained, deterministic RealNVP map from low-dimensional latent to high-dimensional output."""

    def __init__(self, *, h_dim: int, seed: int, config: RealNVPLowToHighConfig) -> None:
        if h_dim < config.z_dim:
            raise ValueError(f"h_dim ({h_dim}) must be >= z_dim ({config.z_dim}).")
        self.h_dim = int(h_dim)
        self.seed = int(seed)
        self.config = config
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        real_nvp_cls = _require_glasflow_realnvp()
        self.flow = real_nvp_cls(
            n_inputs=self.h_dim,
            n_transforms=int(config.n_transforms),
            n_neurons=int(config.hidden_width),
            batch_norm_between_transforms=bool(config.batch_norm_between_transforms),
        )
        self.flow.eval()
        for p in self.flow.parameters():
            p.requires_grad = False

    def _pad_latent(self, z: torch.Tensor) -> torch.Tensor:
        batch = int(z.shape[0])
        if self.h_dim == self.config.z_dim:
            return z
        pad = torch.zeros(batch, self.h_dim - self.config.z_dim, dtype=z.dtype, device=z.device)
        return torch.cat([z, pad], dim=-1)

    @torch.no_grad()
    def transform(self, z_np: np.ndarray) -> np.ndarray:
        z = np.asarray(z_np, dtype=np.float64)
        if z.ndim != 2 or int(z.shape[1]) != int(self.config.z_dim):
            raise ValueError(f"Expected z shape (N,{self.config.z_dim}), got {tuple(z.shape)}.")
        z_t = torch.from_numpy(z).to(dtype=torch.float32)
        x0 = self._pad_latent(z_t)
        out = self.flow.forward(x0)
        if isinstance(out, tuple):
            h_t = out[0]
        else:
            h_t = out
        if bool(self.config.positive_output):
            h_t = F.softplus(h_t)
        h = h_t.detach().cpu().numpy().astype(np.float64, copy=False)
        return h


@dataclass(frozen=True)
class RealNVPDatasetBuildResult:
    theta_all: np.ndarray
    x_embed_all: np.ndarray
    base_dataset: Any
    embedder_config: RealNVPLowToHighConfig


def _copy_namespace(ns: Any) -> Namespace:
    out = Namespace()
    for k, v in vars(ns).items():
        setattr(out, k, v)
    return out


def build_randamp_gaussian_sqrtd_realnvp_dataset(ns: Any) -> RealNVPDatasetBuildResult:
    """Generate dataset by sampling base randamp_sqrtd in z_dim then embedding with fixed RealNVP."""
    cfg = RealNVPLowToHighConfig()
    x_dim_target = int(getattr(ns, "x_dim"))
    if x_dim_target < cfg.z_dim:
        raise ValueError(
            f"--x-dim must be >= {cfg.z_dim} for randamp_gaussian_sqrtd_realnvp; got {x_dim_target}."
        )
    base_ns = _copy_namespace(ns)
    base_ns.dataset_family = "randamp_gaussian_sqrtd"
    base_ns.x_dim = int(cfg.z_dim)
    base_dataset = build_dataset_from_args(base_ns)
    theta_all, x_base_all = base_dataset.sample_joint(int(getattr(ns, "n_total")))
    embedder = FixedRealNVPLowToHighEmbedder(h_dim=x_dim_target, seed=int(getattr(ns, "seed")), config=cfg)
    x_embed_all = embedder.transform(x_base_all)
    return RealNVPDatasetBuildResult(
        theta_all=theta_all,
        x_embed_all=x_embed_all,
        base_dataset=base_dataset,
        embedder_config=cfg,
    )
