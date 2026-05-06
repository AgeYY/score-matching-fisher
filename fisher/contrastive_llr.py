from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _as_2d_float64(a: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D.")
    return arr


class ContrastiveLLRMLP(nn.Module):
    """Scalar scorer S(x, theta) for shuffled-theta contrastive learning."""

    def __init__(
        self,
        *,
        x_dim: int,
        theta_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if int(x_dim) < 1:
            raise ValueError("x_dim must be >= 1.")
        if int(theta_dim) < 1:
            raise ValueError("theta_dim must be >= 1.")
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be >= 1.")
        if int(depth) < 1:
            raise ValueError("depth must be >= 1.")
        self.x_dim = int(x_dim)
        self.theta_dim = int(theta_dim)
        layers: list[nn.Module] = []
        in_dim = self.x_dim + self.theta_dim
        for _ in range(int(depth)):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.SiLU())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        if x.shape[0] != theta.shape[0]:
            raise ValueError("x and theta batch sizes must match.")
        return self.net(torch.cat([x, theta], dim=1)).squeeze(-1)

    def score_matrix(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        bx = int(x.shape[0])
        bt = int(theta.shape[0])
        x_rep = x.repeat_interleave(bt, dim=0)
        theta_rep = theta.repeat(bx, 1)
        return self.forward(x_rep, theta_rep).reshape(bx, bt)


def _make_feature_mlp(*, in_dim: int, out_dim: int, hidden_dim: int, depth: int) -> nn.Sequential:
    if int(in_dim) < 1:
        raise ValueError("in_dim must be >= 1.")
    if int(out_dim) < 1:
        raise ValueError("out_dim must be >= 1.")
    if int(hidden_dim) < 1:
        raise ValueError("hidden_dim must be >= 1.")
    if int(depth) < 1:
        raise ValueError("depth must be >= 1.")
    layers: list[nn.Module] = []
    d = int(in_dim)
    for _ in range(int(depth)):
        layers.append(nn.Linear(d, int(hidden_dim)))
        layers.append(nn.SiLU())
        d = int(hidden_dim)
    layers.append(nn.Linear(d, int(out_dim)))
    return nn.Sequential(*layers)


class ContrastiveNormalizedDotScorer(nn.Module):
    """Scalar scorer S(x, theta) = normalize(g(x))^T normalize(a(theta))."""

    def __init__(
        self,
        *,
        x_dim: int,
        theta_dim: int,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        depth: int = 3,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if int(x_dim) < 1:
            raise ValueError("x_dim must be >= 1.")
        if int(theta_dim) < 1:
            raise ValueError("theta_dim must be >= 1.")
        if int(feature_dim) < 1:
            raise ValueError("feature_dim must be >= 1.")
        if not math.isfinite(float(eps)) or float(eps) <= 0.0:
            raise ValueError("eps must be finite and > 0.")
        self.x_dim = int(x_dim)
        self.theta_dim = int(theta_dim)
        self.feature_dim = int(feature_dim)
        self.eps = float(eps)
        self.rho = nn.Parameter(torch.zeros(()))
        self.g_net = _make_feature_mlp(
            in_dim=self.x_dim,
            out_dim=self.feature_dim,
            hidden_dim=int(hidden_dim),
            depth=int(depth),
        )
        self.a_net = _make_feature_mlp(
            in_dim=self.theta_dim,
            out_dim=self.feature_dim,
            hidden_dim=int(hidden_dim),
            depth=int(depth),
        )

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.rho)

    def encode_x(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return F.normalize(self.g_net(x), p=2.0, dim=-1, eps=float(self.eps))

    def encode_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """Theta features (optionally Fourier-augmented by the training caller)."""
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        return F.normalize(self.a_net(theta), p=2.0, dim=-1, eps=float(self.eps))

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        if x.shape[0] != theta.shape[0]:
            raise ValueError("x and theta batch sizes must match.")
        gx = self.encode_x(x)
        at = self.encode_theta(theta)
        return self.alpha * (gx * at).sum(dim=-1)

    def score_matrix(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        gx = self.encode_x(x)
        at = self.encode_theta(theta)
        return self.alpha * (gx @ at.transpose(0, 1))


class ContrastiveNormalizedDotBiasScorer(ContrastiveNormalizedDotScorer):
    """Normalized dot scorer S(x, theta) = alpha cos(g(x), a(theta)) + b(theta)."""

    def __init__(
        self,
        *,
        x_dim: int,
        theta_dim: int,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        depth: int = 3,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(
            x_dim=int(x_dim),
            theta_dim=int(theta_dim),
            feature_dim=int(feature_dim),
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            eps=float(eps),
        )
        self.b_net = _make_feature_mlp(
            in_dim=self.theta_dim,
            out_dim=1,
            hidden_dim=int(hidden_dim),
            depth=int(depth),
        )

    def theta_bias(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        return self.b_net(theta).squeeze(-1)

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return super().forward(x, theta) + self.theta_bias(theta)

    def score_matrix(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return super().score_matrix(x, theta) + self.theta_bias(theta).unsqueeze(0)


class ContrastiveAdditiveIndependentScorer(nn.Module):
    """Additive scalar scorer S(x, theta) = D^{-1} sum_d h_d(x_d)^T a(theta)."""

    def __init__(
        self,
        *,
        x_dim: int,
        theta_dim: int,
        feature_dim: int = 16,
        hidden_dim: int = 128,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if int(x_dim) < 1:
            raise ValueError("x_dim must be >= 1.")
        if int(theta_dim) < 1:
            raise ValueError("theta_dim must be >= 1.")
        if int(feature_dim) < 1:
            raise ValueError("feature_dim must be >= 1.")
        self.x_dim = int(x_dim)
        self.theta_dim = int(theta_dim)
        self.feature_dim = int(feature_dim)
        self.h_nets = nn.ModuleList(
            [
                _make_feature_mlp(
                    in_dim=1,
                    out_dim=self.feature_dim,
                    hidden_dim=int(hidden_dim),
                    depth=int(depth),
                )
                for _ in range(self.x_dim)
            ]
        )
        self.a_net = _make_feature_mlp(
            in_dim=self.theta_dim,
            out_dim=self.feature_dim,
            hidden_dim=int(hidden_dim),
            depth=int(depth),
        )

    def encode_x_by_dim(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if int(x.shape[1]) != self.x_dim:
            raise ValueError(f"x must have {self.x_dim} columns.")
        pieces = [net(x[:, d : d + 1]) for d, net in enumerate(self.h_nets)]
        return torch.stack(pieces, dim=1)

    def encode_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """``theta`` features may be Fourier-augmented by the caller (contrastive-soft)."""
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        return self.a_net(theta)

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        if x.shape[0] != theta.shape[0]:
            raise ValueError("x and theta batch sizes must match.")
        hx = self.encode_x_by_dim(x)
        at = self.encode_theta(theta)
        return (hx * at.unsqueeze(1)).sum(dim=-1).mean(dim=1)

    def score_matrix(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        hx = self.encode_x_by_dim(x)
        at = self.encode_theta(theta)
        return torch.einsum("bdf,tf->bt", hx, at) / float(self.x_dim)


class ContrastiveIndependentGaussianScorer(nn.Module):
    """Diagonal Gaussian scorer with theta-dependent mean and constant learned variance."""

    def __init__(
        self,
        *,
        x_dim: int,
        theta_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        logvar_min: float = -8.0,
        logvar_max: float = 5.0,
    ) -> None:
        super().__init__()
        if int(x_dim) < 1:
            raise ValueError("x_dim must be >= 1.")
        if int(theta_dim) < 1:
            raise ValueError("theta_dim must be >= 1.")
        if not math.isfinite(float(logvar_min)) or not math.isfinite(float(logvar_max)):
            raise ValueError("logvar_min/logvar_max must be finite.")
        if float(logvar_min) >= float(logvar_max):
            raise ValueError("logvar_min must be < logvar_max.")
        self.x_dim = int(x_dim)
        self.theta_dim = int(theta_dim)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.raw_logvar = nn.Parameter(torch.zeros(self.x_dim))
        self.theta_net = _make_feature_mlp(
            in_dim=self.theta_dim,
            out_dim=self.x_dim,
            hidden_dim=int(hidden_dim),
            depth=int(depth),
        )

    def theta_params(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        mu = self.theta_net(theta)
        logvar = torch.clamp(self.raw_logvar, min=self.logvar_min, max=self.logvar_max)
        return mu, logvar

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        if x.shape[0] != theta.shape[0]:
            raise ValueError("x and theta batch sizes must match.")
        if int(x.shape[1]) != self.x_dim:
            raise ValueError(f"x must have {self.x_dim} columns.")
        mu, logvar = self.theta_params(theta)
        inv_var = torch.exp(-logvar)
        return -0.5 * (((x - mu) ** 2) * inv_var + logvar).sum(dim=-1)

    def score_matrix(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if int(x.shape[1]) != self.x_dim:
            raise ValueError(f"x must have {self.x_dim} columns.")
        mu, logvar = self.theta_params(theta)
        inv_var = torch.exp(-logvar)
        x3 = x.unsqueeze(1)
        mu3 = mu.unsqueeze(0)
        logvar3 = logvar.unsqueeze(0)
        inv_var3 = inv_var.unsqueeze(0)
        return -0.5 * (((x3 - mu3) ** 2) * inv_var3 + logvar3).sum(dim=-1)


class ContrastiveIndependentDotProductScorer(nn.Module):
    """Coordinate-embedded additive dot scorer with learnable scale and theta bias."""

    def __init__(
        self,
        *,
        x_dim: int,
        theta_dim: int,
        feature_dim: int = 16,
        coord_embed_dim: int = 16,
        hidden_dim: int = 128,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if int(x_dim) < 1:
            raise ValueError("x_dim must be >= 1.")
        if int(theta_dim) < 1:
            raise ValueError("theta_dim must be >= 1.")
        if int(feature_dim) < 1:
            raise ValueError("feature_dim must be >= 1.")
        if int(coord_embed_dim) < 1:
            raise ValueError("coord_embed_dim must be >= 1.")
        self.x_dim = int(x_dim)
        self.theta_dim = int(theta_dim)
        self.feature_dim = int(feature_dim)
        self.coord_embed_dim = int(coord_embed_dim)
        self.rho = nn.Parameter(torch.zeros(()))
        self.coord_embedding = nn.Embedding(self.x_dim, self.coord_embed_dim)
        self.h_net = _make_feature_mlp(
            in_dim=1 + self.coord_embed_dim,
            out_dim=self.feature_dim,
            hidden_dim=int(hidden_dim),
            depth=int(depth),
        )
        self.a_net = _make_feature_mlp(
            in_dim=self.theta_dim,
            out_dim=self.feature_dim,
            hidden_dim=int(hidden_dim),
            depth=int(depth),
        )
        self.b_net = _make_feature_mlp(
            in_dim=self.theta_dim,
            out_dim=1,
            hidden_dim=int(hidden_dim),
            depth=int(depth),
        )

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.rho)

    def encode_x_by_dim(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if int(x.shape[1]) != self.x_dim:
            raise ValueError(f"x must have {self.x_dim} columns.")
        bsz = int(x.shape[0])
        coord_idx = torch.arange(self.x_dim, device=x.device)
        coord = self.coord_embedding(coord_idx).unsqueeze(0).expand(bsz, -1, -1)
        h_in = torch.cat([x.unsqueeze(-1), coord], dim=-1).reshape(bsz * self.x_dim, -1)
        return self.h_net(h_in).reshape(bsz, self.x_dim, self.feature_dim)

    def encode_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """``theta`` features may be Fourier-augmented by the caller (contrastive-soft)."""
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        return self.a_net(theta)

    def theta_bias(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        return self.b_net(theta).squeeze(-1)

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        if x.shape[0] != theta.shape[0]:
            raise ValueError("x and theta batch sizes must match.")
        hx = self.encode_x_by_dim(x)
        at = self.encode_theta(theta)
        b = self.theta_bias(theta)
        scale = self.alpha / math.sqrt(float(self.x_dim))
        return scale * (hx * at.unsqueeze(1)).sum(dim=(-1, -2)) + b

    def score_matrix(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        hx = self.encode_x_by_dim(x)
        at = self.encode_theta(theta)
        b = self.theta_bias(theta)
        scale = self.alpha / math.sqrt(float(self.x_dim))
        return scale * torch.einsum("bdf,tf->bt", hx, at) + b.unsqueeze(0)


class ContrastiveGaussianNetworkScorer(nn.Module):
    """Adapter exposing a diagonal Gaussian log p(x|theta) as a contrastive scalar scorer."""

    def __init__(self, gaussian_model: nn.Module) -> None:
        super().__init__()
        if not hasattr(gaussian_model, "x_dim") or not hasattr(gaussian_model, "theta_dim"):
            raise ValueError("gaussian_model must expose x_dim and theta_dim.")
        if not hasattr(gaussian_model, "log_prob"):
            raise ValueError("gaussian_model must expose log_prob(x, theta).")
        self.gaussian_model = gaussian_model
        self.x_dim = int(gaussian_model.x_dim)
        self.theta_dim = int(gaussian_model.theta_dim)

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        if x.shape[0] != theta.shape[0]:
            raise ValueError("x and theta batch sizes must match.")
        return self.gaussian_model.log_prob(x, theta)

    def score_matrix(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        bx = int(x.shape[0])
        bt = int(theta.shape[0])
        x_rep = x.repeat_interleave(bt, dim=0)
        theta_rep = theta.repeat(bx, 1)
        return self.forward(x_rep, theta_rep).reshape(bx, bt)


def dot_scorer_augmented_theta_dim(*, fourier_k: int, fourier_include_linear: bool) -> int:
    """Input width for contrastive dot-family theta branch: z-scored scalar plus optional Fourier block."""
    k = int(fourier_k)
    if k <= 0:
        return 1
    fourier_width = (1 if bool(fourier_include_linear) else 0) + 2 * k
    return 1 + fourier_width


def theta_scalar_fourier_columns(
    theta_values: np.ndarray,
    theta_ref: np.ndarray,
    k: int,
    period_mult: float,
    include_linear: bool,
) -> np.ndarray:
    """Sin/cos harmonics (and optional scaled linear) matching ``_build_theta_fourier_state`` for scalar θ (``d==1``)."""
    k = int(k)
    if k < 1:
        return np.zeros((int(np.asarray(theta_values).size), 0), dtype=np.float64)
    theta_all = np.asarray(theta_values, dtype=np.float64).reshape(-1)
    theta_ref_vec = np.asarray(theta_ref, dtype=np.float64).reshape(-1)
    if theta_all.size < 1 or theta_ref_vec.size < 1:
        raise ValueError("theta_scalar_fourier_columns requires non-empty theta and theta_ref.")
    ref_min = float(np.min(theta_ref_vec))
    ref_max = float(np.max(theta_ref_vec))
    ref_range = float(ref_max - ref_min)
    range_safe = max(ref_range, 1e-12)
    period = float(period_mult) * range_safe
    w0 = 2.0 * np.pi / period
    theta_center = 0.5 * (ref_min + ref_max)
    theta_shift = theta_all - theta_center
    cols: list[np.ndarray] = []
    if include_linear:
        cols.append((theta_shift / range_safe).reshape(-1, 1))
    for kk in range(1, k + 1):
        phase = (float(kk) * w0) * theta_shift
        cols.append(np.sin(phase).reshape(-1, 1))
        cols.append(np.cos(phase).reshape(-1, 1))
    return np.concatenate(cols, axis=1).astype(np.float64, copy=False)


def augment_scalar_theta_for_dot_scorer(
    th_norm: np.ndarray,
    th_raw: np.ndarray,
    theta_ref: np.ndarray,
    fourier_k: int,
    period_mult: float,
    fourier_include_linear: bool,
) -> np.ndarray:
    """Concatenate z-scored scalar θ with Fourier features of raw θ (train-ref period)."""
    th_norm = _as_2d_float64(th_norm, name="th_norm")
    th_raw = _as_2d_float64(th_raw, name="th_raw")
    fk = int(fourier_k)
    if fk <= 0:
        if int(th_norm.shape[1]) != 1:
            raise ValueError("Expected scalar normalized theta (N,1) when fourier_k<=0.")
        return th_norm.astype(np.float64, copy=False)
    if th_norm.shape[0] != th_raw.shape[0] or int(th_norm.shape[1]) != 1 or int(th_raw.shape[1]) != 1:
        raise ValueError("th_norm and th_raw must be (N,1) with matching N.")
    fou = theta_scalar_fourier_columns(
        th_raw.reshape(-1),
        theta_ref,
        fk,
        float(period_mult),
        bool(fourier_include_linear),
    )
    if fou.shape[0] != th_norm.shape[0]:
        raise ValueError("Fourier row count mismatch.")
    return np.concatenate([th_norm, fou], axis=1).astype(np.float64, copy=False)


def contrastive_soft_theta_fourier_supported(model: nn.Module) -> bool:
    """Architectures whose theta MLP branch accepts Fourier-augmented coordinates."""
    return isinstance(
        model,
        (
            ContrastiveNormalizedDotScorer,
            ContrastiveNormalizedDotBiasScorer,
            ContrastiveAdditiveIndependentScorer,
            ContrastiveIndependentDotProductScorer,
        ),
    )


def one_hot_bins(bin_idx: torch.Tensor, n_bins: int) -> torch.Tensor:
    b = bin_idx.reshape(-1).to(dtype=torch.long)
    if b.numel() < 1:
        raise ValueError("bin_idx must be non-empty.")
    if int(torch.min(b).detach().cpu()) < 0 or int(torch.max(b).detach().cpu()) >= int(n_bins):
        raise ValueError("bin_idx contains values outside [0, n_bins).")
    return F.one_hot(b, num_classes=int(n_bins)).to(dtype=torch.float32)


def normalize_theta_encoding(encoding: str) -> str:
    enc = str(encoding).strip().lower()
    aliases = {
        "one_hot": "one_hot_bin",
        "one-hot": "one_hot_bin",
        "one-hot-bin": "one_hot_bin",
        "one_hot_bin": "one_hot_bin",
        "integer": "integer_bin",
        "int": "integer_bin",
        "integer-bin": "integer_bin",
        "integer_bin": "integer_bin",
    }
    if enc not in aliases:
        raise ValueError("--contrastive-theta-encoding must be one of {'one_hot_bin','integer_bin'}.")
    return aliases[enc]


def theta_dim_for_encoding(n_bins: int, encoding: str) -> int:
    enc = normalize_theta_encoding(encoding)
    if enc == "one_hot_bin":
        return int(n_bins)
    if enc == "integer_bin":
        return 1
    raise AssertionError(f"unhandled contrastive theta encoding {enc!r}")


def encode_bins(bin_idx: torch.Tensor, n_bins: int, encoding: str) -> torch.Tensor:
    enc = normalize_theta_encoding(encoding)
    if enc == "one_hot_bin":
        return one_hot_bins(bin_idx, int(n_bins))
    b = bin_idx.reshape(-1).to(dtype=torch.long)
    if b.numel() < 1:
        raise ValueError("bin_idx must be non-empty.")
    if int(torch.min(b).detach().cpu()) < 0 or int(torch.max(b).detach().cpu()) >= int(n_bins):
        raise ValueError("bin_idx contains values outside [0, n_bins).")
    if int(n_bins) <= 1:
        raise ValueError("n_bins must be >= 2 for integer_bin encoding.")
    z = 2.0 * b.to(dtype=torch.float32) / float(int(n_bins) - 1) - 1.0
    return z.reshape(-1, 1)


def _sample_unique_bin_indices(bin_idx: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    """Sample at most one row per occupied bin for shuffled-batch negatives."""
    bins = torch.unique(bin_idx.reshape(-1).to(dtype=torch.long), sorted=False)
    if int(bins.numel()) < 2:
        raise ValueError("contrastive training requires at least two occupied theta bins.")
    bsz = min(int(batch_size), int(bins.numel()))
    chosen_bins = bins[torch.randperm(int(bins.numel()), device=bin_idx.device)[:bsz]]
    rows: list[torch.Tensor] = []
    flat = bin_idx.reshape(-1).to(dtype=torch.long)
    for b in chosen_bins:
        candidates = torch.nonzero(flat == b, as_tuple=False).reshape(-1)
        rows.append(candidates[torch.randint(0, int(candidates.numel()), (1,), device=bin_idx.device)])
    return torch.cat(rows, dim=0)


def _deterministic_unique_bin_indices(bin_idx: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    flat = bin_idx.reshape(-1).to(dtype=torch.long)
    bins = torch.unique(flat, sorted=True)
    if int(bins.numel()) < 2:
        raise ValueError("contrastive validation requires at least two occupied theta bins.")
    rows: list[torch.Tensor] = []
    for b in bins[: min(int(batch_size), int(bins.numel()))]:
        rows.append(torch.nonzero(flat == b, as_tuple=False).reshape(-1)[:1])
    return torch.cat(rows, dim=0)


def _contrastive_loss(model: ContrastiveLLRMLP, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    if int(x.shape[0]) < 2:
        raise ValueError("contrastive minibatch must contain at least two rows.")
    logits = model.score_matrix(x, theta)
    labels = torch.arange(int(x.shape[0]), device=x.device)
    return F.cross_entropy(logits, labels)


def _theta_pair_distance(
    theta_a: torch.Tensor,
    theta_b: torch.Tensor,
    *,
    periodic: bool,
    period: float,
) -> torch.Tensor:
    """Pairwise distance for soft targets on **z-scored** θ rows.

    Scalar ``(N, 1)`` with ``periodic=False`` matches legacy ``abs`` distance.
    Multi-dimensional θ uses Euclidean distance on the (whitened) coordinate vectors.
    Periodic wrapping is **scalar-only** (``d_theta == 1``).
    """
    if int(theta_a.shape[1]) != int(theta_b.shape[1]):
        raise ValueError("theta_a and theta_b must have the same theta feature dimension.")
    d_theta = int(theta_a.shape[1])
    if bool(periodic):
        if d_theta != 1:
            raise ValueError(
                "periodic contrastive-soft distance supports scalar theta only (d_theta=1); "
                "disable --contrastive-soft-periodic for multi-dimensional theta."
            )
        da = theta_a.reshape(-1, 1)
        db = theta_b.reshape(1, -1)
        d = torch.abs(da - db)
        p = float(period)
        if not math.isfinite(p) or p <= 0.0:
            raise ValueError("period must be finite and > 0 for periodic theta distance.")
        d = torch.remainder(d, p)
        d = torch.minimum(d, torch.as_tensor(p, dtype=d.dtype, device=d.device) - d)
        return d
    if d_theta == 1:
        return torch.abs(theta_a.reshape(-1, 1) - theta_b.reshape(1, -1))
    diff = theta_a.unsqueeze(1) - theta_b.unsqueeze(0)
    return torch.linalg.vector_norm(diff, dim=-1)


def _soft_contrastive_loss(
    model: ContrastiveLLRMLP,
    x: torch.Tensor,
    theta_score: torch.Tensor,
    *,
    theta_kernel: torch.Tensor | None = None,
    bandwidth: float,
    periodic: bool,
    period: float,
) -> torch.Tensor:
    if int(x.shape[0]) < 2:
        raise ValueError("soft contrastive minibatch must contain at least two rows.")
    h = float(bandwidth)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("soft contrastive bandwidth must be finite and > 0.")
    tk = theta_score if theta_kernel is None else theta_kernel
    logits = model.score_matrix(x, theta_score)
    dist = _theta_pair_distance(tk, tk, periodic=bool(periodic), period=float(period))
    log_w = -0.5 * (dist / h).pow(2)
    weights = torch.softmax(log_w, dim=1)
    log_probs = torch.log_softmax(logits, dim=1)
    return -(weights * log_probs).sum(dim=1).mean()


def _bidir_soft_contrastive_loss_parts(
    model: ContrastiveLLRMLP,
    x: torch.Tensor,
    theta_score: torch.Tensor,
    *,
    theta_kernel: torch.Tensor | None = None,
    bandwidth: float,
    periodic: bool,
    period: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(x.shape[0]) < 2:
        raise ValueError("bidirectional soft contrastive minibatch must contain at least two rows.")
    h = float(bandwidth)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("bidirectional soft contrastive bandwidth must be finite and > 0.")
    tk = theta_score if theta_kernel is None else theta_kernel
    logits = model.score_matrix(x, theta_score)
    dist = _theta_pair_distance(tk, tk, periodic=bool(periodic), period=float(period))
    log_w = -0.5 * (dist / h).pow(2)
    row_weights = torch.softmax(log_w, dim=1)
    col_weights = torch.softmax(log_w, dim=0)
    # Row objective: raw scores (standard softmax contrastive along theta candidates).
    row_log_probs = torch.log_softmax(logits, dim=1)
    # Column objective: row-centered scores remove per-sample additive offsets along theta,
    # stabilizing the column softmax without changing inference LLRs S(z,theta_i)-S(z,theta_j)
    # (those differences are unchanged by per-row constants).
    logits_row_centered = logits - logits.mean(dim=1, keepdim=True)
    col_log_probs = torch.log_softmax(logits_row_centered, dim=0)
    row_loss = -(row_weights * row_log_probs).sum(dim=1).mean()
    col_loss = -(col_weights * col_log_probs).sum(dim=0).mean()
    loss = 0.5 * (row_loss + col_loss)
    return loss, row_loss, col_loss


def _bidir_soft_contrastive_loss(
    model: ContrastiveLLRMLP,
    x: torch.Tensor,
    theta_score: torch.Tensor,
    *,
    theta_kernel: torch.Tensor | None = None,
    bandwidth: float,
    periodic: bool,
    period: float,
) -> torch.Tensor:
    loss, _, _ = _bidir_soft_contrastive_loss_parts(
        model,
        x,
        theta_score,
        theta_kernel=theta_kernel,
        bandwidth=float(bandwidth),
        periodic=bool(periodic),
        period=float(period),
    )
    return loss


def contrastive_soft_normalization_and_bandwidth_from_train(
    *,
    th_tr: np.ndarray,
    x_tr: np.ndarray,
    bandwidth_bins: int,
    bandwidth_start: float,
    bandwidth_end: float,
    periodic: bool,
    period: float,
) -> dict[str, Any]:
    """Train-set normalization and soft-contrastive bandwidth scale (matches ``train_contrastive_soft_llr``).

    When bandwidth annealing is disabled, a scalar raw scale is
    ``theta_range / (2 * bandwidth_bins)`` where ``theta_range`` is the train per-coordinate span
    (scalar: ``max-min`` on training θ; multi-dim: ``max_j (max θ_j - min θ_j)``).
    That scale is converted to normalized space using ``mean(theta_std)`` so the Gaussian
    kernel applies to Euclidean distance on z-scored θ.

    ``--contrastive-soft-periodic`` is only allowed for scalar θ (``d_theta == 1``).
    """
    x_mean = np.mean(x_tr, axis=0, dtype=np.float64)
    x_std = np.maximum(np.std(x_tr, axis=0, dtype=np.float64), 1e-6)
    theta_mean = np.mean(th_tr, axis=0, dtype=np.float64)
    theta_std = np.maximum(np.std(th_tr, axis=0, dtype=np.float64), 1e-6)
    th2 = _as_2d_float64(th_tr, name="th_tr")
    d_theta = int(th2.shape[1])
    if bool(periodic) and d_theta != 1:
        raise ValueError(
            "contrastive-soft periodic bandwidth/distance supports scalar theta only (d_theta=1)."
        )

    bw_start = float(bandwidth_start)
    bw_end = float(bandwidth_end)
    if int(bandwidth_bins) < 1:
        raise ValueError("contrastive-soft bandwidth_bins must be >= 1.")
    if not math.isfinite(bw_start) or not math.isfinite(bw_end):
        raise ValueError("contrastive-soft bandwidth start/end must be finite.")
    bandwidth_anneal_enabled = bool(bw_start > 0.0 or bw_end > 0.0)
    if bandwidth_anneal_enabled and not (bw_start > 0.0 and bw_end > 0.0):
        raise ValueError("contrastive-soft bandwidth annealing requires both start and end > 0.")

    theta_scale = float(np.mean(theta_std))
    if bandwidth_anneal_enabled:
        h_start = bw_start / theta_scale
        h_end = bw_end / theta_scale
    else:
        if d_theta == 1:
            th_flat = th2.reshape(-1)
            theta_range = float(np.max(th_flat) - np.min(th_flat))
        else:
            theta_range = float(np.max(np.ptp(th2, axis=0)))
        if not math.isfinite(theta_range) or theta_range <= 0.0:
            raise ValueError(
                "contrastive-soft bandwidth from bins requires a positive train theta span "
                "(per-coordinate range for multi-dimensional theta)."
            )
        h_raw = float(theta_range) / float(int(bandwidth_bins)) / 2.0
        h_start = h_raw / theta_scale
        h_end = h_start
    if (
        not math.isfinite(float(h_start))
        or not math.isfinite(float(h_end))
        or float(h_start) <= 0.0
        or float(h_end) <= 0.0
    ):
        raise ValueError("effective contrastive-soft bandwidth must be finite and > 0.")

    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "theta_mean": theta_mean,
        "theta_std": theta_std,
        "theta_scale": theta_scale,
        "bandwidth_anneal_enabled": bandwidth_anneal_enabled,
        "h_start_norm": float(h_start),
        "h_end_norm": float(h_end),
        "bandwidth_bins": int(bandwidth_bins),
    }


def contrastive_soft_metadata_without_training(
    *,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    bandwidth_bins: int = 10,
    bandwidth_start: float = 0.0,
    bandwidth_end: float = 0.0,
    periodic: bool = False,
    period: float = 2.0 * math.pi,
) -> dict[str, Any]:
    """Same keys as ``train_contrastive_soft_llr`` return dict, with empty loss traces (no fine-tuning)."""
    th_tr = _as_2d_float64(theta_train, name="theta_train")
    x_tr = _as_2d_float64(x_train, name="x_train")
    if int(th_tr.shape[1]) < 1:
        raise ValueError("contrastive-soft expects theta_train with shape (N, d_theta), d_theta >= 1.")
    if th_tr.shape[0] < 2:
        raise ValueError("contrastive-soft requires at least two training rows.")
    if th_tr.shape[0] != x_tr.shape[0]:
        raise ValueError("theta/x row count mismatch.")

    nb = contrastive_soft_normalization_and_bandwidth_from_train(
        th_tr=th_tr,
        x_tr=x_tr,
        bandwidth_bins=int(bandwidth_bins),
        bandwidth_start=float(bandwidth_start),
        bandwidth_end=float(bandwidth_end),
        periodic=bool(periodic),
        period=float(period),
    )
    x_mean = nb["x_mean"]
    x_std = nb["x_std"]
    theta_mean = nb["theta_mean"]
    theta_std = nb["theta_std"]
    theta_scale = float(nb["theta_scale"])
    bandwidth_anneal_enabled = bool(nb["bandwidth_anneal_enabled"])
    h_start = float(nb["h_start_norm"])
    h_end = float(nb["h_end_norm"])
    h_final_norm = float(h_end)
    bw_bins = int(nb["bandwidth_bins"])

    return {
        "train_losses": [],
        "val_losses": [],
        "val_monitor_losses": [],
        "best_epoch": 0,
        "stopped_epoch": 0,
        "stopped_early": False,
        "best_val_loss": float("nan"),
        "x_mean": x_mean,
        "x_std": x_std,
        "theta_mean": theta_mean,
        "theta_std": theta_std,
        "bandwidth_raw": float(h_final_norm * theta_scale),
        "bandwidth_normalized": float(h_final_norm),
        "bandwidth_auto": False,
        "bandwidth_anneal_enabled": bandwidth_anneal_enabled,
        "bandwidth_bins": bw_bins,
        "bandwidth_start_raw": float(h_start * theta_scale),
        "bandwidth_end_raw": float(h_end * theta_scale),
        "bandwidth_start_normalized": float(h_start),
        "bandwidth_end_normalized": float(h_end),
        "bandwidth_raw_schedule": [float(h_final_norm * theta_scale)],
        "bandwidth_normalized_schedule": [float(h_final_norm)],
        "lr_last": float("nan"),
        "n_clipped_steps": 0,
        "n_total_steps": 0,
        "effective_batch_size": 0,
        "contrastive_theta_fourier_k": 0,
        "contrastive_theta_fourier_period_mult": 2.0,
        "contrastive_theta_fourier_include_linear": False,
        "theta_fourier_ref_min": float(np.min(th_tr)),
        "theta_fourier_ref_max": float(np.max(th_tr)),
    }


def _eval_soft_contrastive_loss(
    model: ContrastiveLLRMLP,
    x: torch.Tensor,
    theta_score: torch.Tensor,
    *,
    theta_kernel: torch.Tensor | None = None,
    batch_size: int,
    bandwidth: float,
    periodic: bool,
    period: float,
) -> float:
    n = int(x.shape[0])
    if n < 2:
        return float("nan")
    bs = min(n, max(2, int(batch_size)))
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for i0 in range(0, n, bs):
            i1 = min(n, i0 + bs)
            if i1 - i0 < 2:
                continue
            if i1 - i0 != bs and losses:
                continue
            loss = _soft_contrastive_loss(
                model,
                x[i0:i1],
                theta_score[i0:i1],
                theta_kernel=None if theta_kernel is None else theta_kernel[i0:i1],
                bandwidth=float(bandwidth),
                periodic=bool(periodic),
                period=float(period),
            )
            losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


def _eval_bidir_soft_contrastive_loss(
    model: ContrastiveLLRMLP,
    x: torch.Tensor,
    theta_score: torch.Tensor,
    *,
    theta_kernel: torch.Tensor | None = None,
    batch_size: int,
    bandwidth: float,
    periodic: bool,
    period: float,
) -> tuple[float, float, float]:
    n = int(x.shape[0])
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    bs = min(n, max(2, int(batch_size)))
    losses: list[float] = []
    row_losses: list[float] = []
    col_losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for i0 in range(0, n, bs):
            i1 = min(n, i0 + bs)
            if i1 - i0 < 2:
                continue
            if i1 - i0 != bs and losses:
                continue
            loss, row_loss, col_loss = _bidir_soft_contrastive_loss_parts(
                model,
                x[i0:i1],
                theta_score[i0:i1],
                theta_kernel=None if theta_kernel is None else theta_kernel[i0:i1],
                bandwidth=float(bandwidth),
                periodic=bool(periodic),
                period=float(period),
            )
            losses.append(float(loss.detach().cpu().item()))
            row_losses.append(float(row_loss.detach().cpu().item()))
            col_losses.append(float(col_loss.detach().cpu().item()))
    if not losses:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(losses)), float(np.mean(row_losses)), float(np.mean(col_losses))


def _eval_contrastive_loss(
    model: ContrastiveLLRMLP,
    x: torch.Tensor,
    bin_idx: torch.Tensor,
    *,
    batch_size: int,
    n_bins: int,
    theta_encoding: str,
) -> float:
    if int(x.shape[0]) < 2:
        return float("nan")
    model.eval()
    with torch.no_grad():
        idx = _deterministic_unique_bin_indices(bin_idx, batch_size=max(2, int(batch_size)))
        theta_code = encode_bins(bin_idx[idx], int(n_bins), theta_encoding).to(device=x.device)
        loss = _contrastive_loss(model, x[idx], theta_code)
    return float(loss.detach().cpu().item())


def train_contrastive_llr(
    *,
    model: ContrastiveLLRMLP,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    bin_train: np.ndarray,
    bin_val: np.ndarray,
    n_bins: int,
    theta_encoding: str = "one_hot_bin",
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.0,
    patience: int = 300,
    min_delta: float = 1e-4,
    ema_alpha: float = 0.05,
    max_grad_norm: float = 10.0,
    log_every: int = 50,
    restore_best: bool = True,
) -> dict[str, Any]:
    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1.")
    if int(batch_size) < 2:
        raise ValueError("batch_size must be >= 2 for contrastive learning.")
    if int(n_bins) < 2:
        raise ValueError("n_bins must be >= 2 for binned contrastive learning.")
    theta_encoding_norm = normalize_theta_encoding(theta_encoding)
    if float(lr) <= 0.0:
        raise ValueError("lr must be > 0.")
    if float(weight_decay) < 0.0:
        raise ValueError("weight_decay must be >= 0.")
    if int(patience) < 0:
        raise ValueError("patience must be >= 0.")
    if float(min_delta) < 0.0:
        raise ValueError("min_delta must be >= 0.")
    if not (0.0 < float(ema_alpha) <= 1.0):
        raise ValueError("ema_alpha must be in (0, 1].")
    if not math.isfinite(float(max_grad_norm)) or float(max_grad_norm) < 0.0:
        raise ValueError("max_grad_norm must be finite and >= 0.")

    th_tr = _as_2d_float64(theta_train, name="theta_train")
    th_va = _as_2d_float64(theta_val, name="theta_val")
    x_tr = _as_2d_float64(x_train, name="x_train")
    x_va = _as_2d_float64(x_val, name="x_val")
    if th_tr.shape[0] < 2:
        raise ValueError("contrastive method requires at least two training rows.")
    if th_va.shape[0] < 2:
        raise ValueError("contrastive method requires at least two validation rows.")
    if th_tr.shape[0] != x_tr.shape[0] or th_va.shape[0] != x_va.shape[0]:
        raise ValueError("theta/x row count mismatch.")
    if int(x_tr.shape[1]) != model.x_dim or int(x_va.shape[1]) != model.x_dim:
        raise ValueError("x dimension does not match ContrastiveLLRMLP.x_dim.")
    expected_theta_dim = theta_dim_for_encoding(int(n_bins), theta_encoding_norm)
    if int(model.theta_dim) != int(expected_theta_dim):
        raise ValueError(
            f"ContrastiveLLRMLP.theta_dim={model.theta_dim} does not match "
            f"{theta_encoding_norm} expected dim={expected_theta_dim}."
        )
    b_tr = np.asarray(bin_train, dtype=np.int64).reshape(-1)
    b_va = np.asarray(bin_val, dtype=np.int64).reshape(-1)
    if b_tr.shape[0] != x_tr.shape[0] or b_va.shape[0] != x_va.shape[0]:
        raise ValueError("bin labels must match x/theta row counts.")
    if np.min(b_tr) < 0 or np.max(b_tr) >= int(n_bins) or np.min(b_va) < 0 or np.max(b_va) >= int(n_bins):
        raise ValueError("bin labels contain values outside [0, n_bins).")
    if np.unique(b_tr).size < 2:
        raise ValueError("contrastive method requires at least two occupied training theta bins.")
    if np.unique(b_va).size < 2:
        raise ValueError("contrastive method requires at least two occupied validation theta bins.")

    x_mean = np.mean(x_tr, axis=0, dtype=np.float64)
    x_std = np.maximum(np.std(x_tr, axis=0, dtype=np.float64), 1e-6)
    x_tr_n = (x_tr - x_mean) / x_std
    x_va_n = (x_va - x_mean) / x_std

    x_tr_t = torch.from_numpy(x_tr_n.astype(np.float32)).to(device)
    x_va_t = torch.from_numpy(x_va_n.astype(np.float32)).to(device)
    b_tr_t = torch.from_numpy(b_tr.astype(np.int64)).to(device)
    b_va_t = torch.from_numpy(b_va.astype(np.int64)).to(device)

    ntr = int(x_tr_t.shape[0])
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_ema_losses: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_ema = float("inf")
    best_epoch = 0
    bad = 0
    ema: float | None = None
    stopped_early = False
    stopped_epoch = int(epochs)
    n_clipped = 0
    n_total_steps = 0

    for ep in range(1, int(epochs) + 1):
        model.train()
        idx = _sample_unique_bin_indices(b_tr_t, batch_size=int(batch_size))
        theta_code = encode_bins(b_tr_t[idx], int(n_bins), theta_encoding_norm).to(device=device)
        loss = _contrastive_loss(model, x_tr_t[idx], theta_code)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if float(max_grad_norm) > 0.0:
            g = torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            if float(g.detach().cpu()) > float(max_grad_norm):
                n_clipped += 1
        opt.step()
        n_total_steps += 1

        tr = float(loss.detach().cpu().item())
        train_losses.append(tr)
        va = _eval_contrastive_loss(
            model,
            x_va_t,
            b_va_t,
            batch_size=int(batch_size),
            n_bins=int(n_bins),
            theta_encoding=theta_encoding_norm,
        )
        val_losses.append(va)
        ema = va if ema is None else (float(ema_alpha) * va + (1.0 - float(ema_alpha)) * float(ema))
        val_ema_losses.append(float(ema))
        if np.isfinite(float(ema)) and float(ema) < best_ema - float(min_delta):
            best_ema = float(ema)
            best_epoch = int(ep)
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        if ep == 1 or ep % max(1, int(log_every)) == 0 or ep == int(epochs):
            print(
                f"[contrastive {ep:4d}/{int(epochs)}] train_ce={tr:.6f} "
                f"val_ce={va:.6f} val_smooth={float(ema):.6f} best_smooth={best_ema:.6f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
        if int(patience) > 0 and bad >= int(patience):
            stopped_early = True
            stopped_epoch = int(ep)
            print(
                f"[contrastive early-stop] epoch={ep} best_epoch={best_epoch} "
                f"best_smooth={best_ema:.6f} patience={int(patience)}",
                flush=True,
            )
            break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_monitor_losses": val_ema_losses,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "stopped_early": stopped_early,
        "best_val_loss": best_ema,
        "x_mean": x_mean,
        "x_std": x_std,
        "theta_mean": np.asarray([], dtype=np.float64),
        "theta_std": np.asarray([], dtype=np.float64),
        "n_bins": int(n_bins),
        "theta_encoding": theta_encoding_norm,
        "lr_last": float(opt.param_groups[0]["lr"]),
        "n_clipped_steps": n_clipped,
        "n_total_steps": n_total_steps,
    }


def train_contrastive_soft_llr(
    *,
    model: ContrastiveLLRMLP,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    bandwidth_bins: int = 10,
    bandwidth_start: float = 0.0,
    bandwidth_end: float = 0.0,
    periodic: bool = False,
    period: float = 2.0 * math.pi,
    weight_decay: float = 0.0,
    patience: int = 300,
    min_delta: float = 1e-4,
    ema_alpha: float = 0.05,
    max_grad_norm: float = 10.0,
    log_every: int = 50,
    restore_best: bool = True,
    contrastive_theta_fourier_k: int = 4,
    contrastive_theta_fourier_period_mult: float = 2.0,
    contrastive_theta_fourier_include_linear: bool = False,
) -> dict[str, Any]:
    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1.")
    if int(batch_size) < 2:
        raise ValueError("batch_size must be >= 2 for soft contrastive learning.")
    if float(lr) <= 0.0:
        raise ValueError("lr must be > 0.")
    if float(weight_decay) < 0.0:
        raise ValueError("weight_decay must be >= 0.")
    if int(patience) < 0:
        raise ValueError("patience must be >= 0.")
    if float(min_delta) < 0.0:
        raise ValueError("min_delta must be >= 0.")
    if not (0.0 < float(ema_alpha) <= 1.0):
        raise ValueError("ema_alpha must be in (0, 1].")
    if not math.isfinite(float(max_grad_norm)) or float(max_grad_norm) < 0.0:
        raise ValueError("max_grad_norm must be finite and >= 0.")

    th_tr = _as_2d_float64(theta_train, name="theta_train")
    th_va = _as_2d_float64(theta_val, name="theta_val")
    x_tr = _as_2d_float64(x_train, name="x_train")
    x_va = _as_2d_float64(x_val, name="x_val")
    d_theta = int(th_tr.shape[1])
    if int(th_va.shape[1]) != d_theta:
        raise ValueError("theta_train and theta_val must have the same d_theta.")
    if th_tr.shape[0] < 2 or th_va.shape[0] < 2:
        raise ValueError("contrastive-soft requires at least two train and two validation rows.")
    if th_tr.shape[0] != x_tr.shape[0] or th_va.shape[0] != x_va.shape[0]:
        raise ValueError("theta/x row count mismatch.")
    if int(x_tr.shape[1]) != model.x_dim or int(x_va.shape[1]) != model.x_dim:
        raise ValueError("x dimension does not match ContrastiveLLRMLP.x_dim.")
    fk = int(contrastive_theta_fourier_k)
    pm = float(contrastive_theta_fourier_period_mult)
    inc_lin = bool(contrastive_theta_fourier_include_linear)
    if fk < 0:
        raise ValueError("contrastive_theta_fourier_k must be >= 0.")
    if fk > 0:
        if d_theta != 1:
            raise ValueError(
                "contrastive-soft Fourier theta features (--theta-flow-fourier-state) require scalar theta (d_theta=1)."
            )
        if not math.isfinite(pm) or pm <= 0.0:
            raise ValueError("contrastive_theta_fourier_period_mult must be finite and > 0 when fourier_k > 0.")
        if not contrastive_soft_theta_fourier_supported(model):
            raise ValueError(
                "Fourier theta features (contrastive_theta_fourier_k > 0) are only supported for "
                "ContrastiveNormalizedDotScorer, ContrastiveNormalizedDotBiasScorer, "
                "ContrastiveAdditiveIndependentScorer, and ContrastiveIndependentDotProductScorer; "
                f"got {type(model).__name__}."
            )
    if fk > 0:
        expected_theta_dim = int(dot_scorer_augmented_theta_dim(fourier_k=fk, fourier_include_linear=inc_lin))
    else:
        expected_theta_dim = int(d_theta)
    if int(model.theta_dim) != int(expected_theta_dim):
        raise ValueError(
            f"model.theta_dim={int(model.theta_dim)} does not match expected theta width {expected_theta_dim} "
            f"(d_theta={d_theta}, fourier_k={fk}, fourier_include_linear={inc_lin})."
        )
    ref_min = float(np.min(th_tr))
    ref_max = float(np.max(th_tr))

    nb = contrastive_soft_normalization_and_bandwidth_from_train(
        th_tr=th_tr,
        x_tr=x_tr,
        bandwidth_bins=int(bandwidth_bins),
        bandwidth_start=float(bandwidth_start),
        bandwidth_end=float(bandwidth_end),
        periodic=bool(periodic),
        period=float(period),
    )
    x_mean = nb["x_mean"]
    x_std = nb["x_std"]
    theta_mean = nb["theta_mean"]
    theta_std = nb["theta_std"]
    theta_scale = float(nb["theta_scale"])
    bandwidth_anneal_enabled = bool(nb["bandwidth_anneal_enabled"])
    bw_bins = int(nb["bandwidth_bins"])
    h_start = float(nb["h_start_norm"])
    h_end = float(nb["h_end_norm"])
    x_tr_n = (x_tr - x_mean) / x_std
    x_va_n = (x_va - x_mean) / x_std
    th_tr_n = (th_tr - theta_mean) / theta_std
    th_va_n = (th_va - theta_mean) / theta_std

    x_tr_t = torch.from_numpy(x_tr_n.astype(np.float32)).to(device)
    x_va_t = torch.from_numpy(x_va_n.astype(np.float32)).to(device)
    th_tr_kernel_t = torch.from_numpy(th_tr_n.astype(np.float32)).to(device)
    th_va_kernel_t = torch.from_numpy(th_va_n.astype(np.float32)).to(device)
    if fk > 0:
        th_tr_aug = augment_scalar_theta_for_dot_scorer(
            th_tr_n,
            th_tr,
            th_tr,
            fk,
            pm,
            inc_lin,
        )
        th_va_aug = augment_scalar_theta_for_dot_scorer(
            th_va_n,
            th_va,
            th_tr,
            fk,
            pm,
            inc_lin,
        )
        th_tr_t = torch.from_numpy(th_tr_aug.astype(np.float32)).to(device)
        th_va_t = torch.from_numpy(th_va_aug.astype(np.float32)).to(device)
    else:
        th_tr_t = th_tr_kernel_t
        th_va_t = th_va_kernel_t
    ntr = int(x_tr_t.shape[0])
    nva = int(x_va_t.shape[0])
    effective_batch_size = min(int(batch_size), ntr, nva)
    if effective_batch_size < 2:
        raise ValueError("soft contrastive requires effective train/validation batch size >= 2.")
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_ema_losses: list[float] = []
    bandwidth_schedule: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_ema = float("inf")
    best_epoch = 0
    bad = 0
    ema: float | None = None
    stopped_early = False
    stopped_epoch = int(epochs)
    n_clipped = 0
    n_total_steps = 0
    period_eff = float(period) / theta_scale if bool(periodic) else float(period)

    for ep in range(1, int(epochs) + 1):
        frac = 0.0 if int(epochs) <= 1 else float(ep - 1) / float(int(epochs) - 1)
        h = float(h_start) + frac * (float(h_end) - float(h_start))
        bandwidth_schedule.append(float(h))
        model.train()
        idx = torch.randperm(ntr, device=device)[:effective_batch_size]
        loss = _soft_contrastive_loss(
            model,
            x_tr_t[idx],
            th_tr_t[idx],
            theta_kernel=th_tr_kernel_t[idx],
            bandwidth=float(h),
            periodic=bool(periodic),
            period=float(period_eff),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if float(max_grad_norm) > 0.0:
            g = torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            if float(g.detach().cpu()) > float(max_grad_norm):
                n_clipped += 1
        opt.step()
        n_total_steps += 1

        tr = float(loss.detach().cpu().item())
        train_losses.append(tr)
        va = _eval_soft_contrastive_loss(
            model,
            x_va_t,
            th_va_t,
            theta_kernel=th_va_kernel_t,
            batch_size=effective_batch_size,
            bandwidth=float(h),
            periodic=bool(periodic),
            period=float(period_eff),
        )
        val_losses.append(va)
        ema = va if ema is None else (float(ema_alpha) * va + (1.0 - float(ema_alpha)) * float(ema))
        val_ema_losses.append(float(ema))
        if np.isfinite(float(ema)) and float(ema) < best_ema - float(min_delta):
            best_ema = float(ema)
            best_epoch = int(ep)
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        if ep == 1 or ep % max(1, int(log_every)) == 0 or ep == int(epochs):
            alpha_msg = ""
            if hasattr(model, "alpha"):
                alpha_msg = f" alpha={float(model.alpha.detach().cpu().item()):.6g}"
            print(
                f"[contrastive_soft {ep:4d}/{int(epochs)}] train_soft_ce={tr:.6f} "
                f"val_soft_ce={va:.6f} val_smooth={float(ema):.6f} best_smooth={best_ema:.6f} "
                f"best_epoch={best_epoch} h_norm={float(h):.6g} batch={effective_batch_size}{alpha_msg}",
                flush=True,
            )
        if int(patience) > 0 and bad >= int(patience):
            stopped_early = True
            stopped_epoch = int(ep)
            print(
                f"[contrastive_soft early-stop] epoch={ep} best_epoch={best_epoch} "
                f"best_smooth={best_ema:.6f} patience={int(patience)}",
                flush=True,
            )
            break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_monitor_losses": val_ema_losses,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "stopped_early": stopped_early,
        "best_val_loss": best_ema,
        "x_mean": x_mean,
        "x_std": x_std,
        "theta_mean": theta_mean,
        "theta_std": theta_std,
        "bandwidth_raw": float(bandwidth_schedule[-1] * theta_scale),
        "bandwidth_normalized": float(bandwidth_schedule[-1]),
        "bandwidth_auto": False,
        "bandwidth_anneal_enabled": bool(bandwidth_anneal_enabled),
        "bandwidth_bins": int(bw_bins),
        "bandwidth_start_raw": float(h_start * theta_scale),
        "bandwidth_end_raw": float(h_end * theta_scale),
        "bandwidth_start_normalized": float(h_start),
        "bandwidth_end_normalized": float(h_end),
        "bandwidth_raw_schedule": [float(v * theta_scale) for v in bandwidth_schedule],
        "bandwidth_normalized_schedule": bandwidth_schedule,
        "lr_last": float(opt.param_groups[0]["lr"]),
        "n_clipped_steps": n_clipped,
        "n_total_steps": n_total_steps,
        "effective_batch_size": int(effective_batch_size),
        "contrastive_theta_fourier_k": fk,
        "contrastive_theta_fourier_period_mult": pm,
        "contrastive_theta_fourier_include_linear": inc_lin,
        "theta_fourier_ref_min": ref_min,
        "theta_fourier_ref_max": ref_max,
    }


def train_bidir_contrastive_soft_llr(
    *,
    model: ContrastiveLLRMLP,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    bandwidth_bins: int = 10,
    bandwidth_start: float = 0.0,
    bandwidth_end: float = 0.0,
    periodic: bool = False,
    period: float = 2.0 * math.pi,
    weight_decay: float = 0.0,
    patience: int = 300,
    min_delta: float = 1e-4,
    ema_alpha: float = 0.05,
    max_grad_norm: float = 10.0,
    log_every: int = 50,
    restore_best: bool = True,
    contrastive_theta_fourier_k: int = 4,
    contrastive_theta_fourier_period_mult: float = 2.0,
    contrastive_theta_fourier_include_linear: bool = False,
) -> dict[str, Any]:
    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1.")
    if int(batch_size) < 2:
        raise ValueError("batch_size must be >= 2 for bidirectional soft contrastive learning.")
    if float(lr) <= 0.0:
        raise ValueError("lr must be > 0.")
    if float(weight_decay) < 0.0:
        raise ValueError("weight_decay must be >= 0.")
    if int(patience) < 0:
        raise ValueError("patience must be >= 0.")
    if float(min_delta) < 0.0:
        raise ValueError("min_delta must be >= 0.")
    if not (0.0 < float(ema_alpha) <= 1.0):
        raise ValueError("ema_alpha must be in (0, 1].")
    if not math.isfinite(float(max_grad_norm)) or float(max_grad_norm) < 0.0:
        raise ValueError("max_grad_norm must be finite and >= 0.")

    th_tr = _as_2d_float64(theta_train, name="theta_train")
    th_va = _as_2d_float64(theta_val, name="theta_val")
    x_tr = _as_2d_float64(x_train, name="x_train")
    x_va = _as_2d_float64(x_val, name="x_val")
    if int(th_tr.shape[1]) != 1 or int(th_va.shape[1]) != 1:
        raise ValueError("bidir-contrastive-soft v1 requires scalar theta.")
    if th_tr.shape[0] < 2 or th_va.shape[0] < 2:
        raise ValueError("bidir-contrastive-soft requires at least two train and two validation rows.")
    if th_tr.shape[0] != x_tr.shape[0] or th_va.shape[0] != x_va.shape[0]:
        raise ValueError("theta/x row count mismatch.")
    if int(x_tr.shape[1]) != model.x_dim or int(x_va.shape[1]) != model.x_dim:
        raise ValueError("x dimension does not match model.x_dim.")
    fk = int(contrastive_theta_fourier_k)
    pm = float(contrastive_theta_fourier_period_mult)
    inc_lin = bool(contrastive_theta_fourier_include_linear)
    if fk < 0:
        raise ValueError("contrastive_theta_fourier_k must be >= 0.")
    if fk > 0:
        if not math.isfinite(pm) or pm <= 0.0:
            raise ValueError("contrastive_theta_fourier_period_mult must be finite and > 0 when fourier_k > 0.")
        if not contrastive_soft_theta_fourier_supported(model):
            raise ValueError(
                "Fourier theta features (contrastive_theta_fourier_k > 0) are only supported for "
                "ContrastiveNormalizedDotScorer, ContrastiveNormalizedDotBiasScorer, "
                "ContrastiveAdditiveIndependentScorer, and ContrastiveIndependentDotProductScorer; "
                f"got {type(model).__name__}."
            )
    aug_dim = dot_scorer_augmented_theta_dim(fourier_k=fk, fourier_include_linear=inc_lin)
    if int(model.theta_dim) != int(aug_dim):
        raise ValueError(
            f"model.theta_dim={int(model.theta_dim)} does not match expected augmented width {aug_dim} "
            f"(fourier_k={fk}, fourier_include_linear={inc_lin})."
        )
    ref_min = float(np.min(th_tr))
    ref_max = float(np.max(th_tr))

    nb = contrastive_soft_normalization_and_bandwidth_from_train(
        th_tr=th_tr,
        x_tr=x_tr,
        bandwidth_bins=int(bandwidth_bins),
        bandwidth_start=float(bandwidth_start),
        bandwidth_end=float(bandwidth_end),
        periodic=bool(periodic),
        period=float(period),
    )
    x_mean = nb["x_mean"]
    x_std = nb["x_std"]
    theta_mean = nb["theta_mean"]
    theta_std = nb["theta_std"]
    theta_scale = float(nb["theta_scale"])
    bandwidth_anneal_enabled = bool(nb["bandwidth_anneal_enabled"])
    bw_bins = int(nb["bandwidth_bins"])
    h_start = float(nb["h_start_norm"])
    h_end = float(nb["h_end_norm"])
    x_tr_n = (x_tr - x_mean) / x_std
    x_va_n = (x_va - x_mean) / x_std
    th_tr_n = (th_tr - theta_mean) / theta_std
    th_va_n = (th_va - theta_mean) / theta_std

    x_tr_t = torch.from_numpy(x_tr_n.astype(np.float32)).to(device)
    x_va_t = torch.from_numpy(x_va_n.astype(np.float32)).to(device)
    th_tr_kernel_t = torch.from_numpy(th_tr_n.astype(np.float32)).to(device)
    th_va_kernel_t = torch.from_numpy(th_va_n.astype(np.float32)).to(device)
    if fk > 0:
        th_tr_aug = augment_scalar_theta_for_dot_scorer(
            th_tr_n,
            th_tr,
            th_tr,
            fk,
            pm,
            inc_lin,
        )
        th_va_aug = augment_scalar_theta_for_dot_scorer(
            th_va_n,
            th_va,
            th_tr,
            fk,
            pm,
            inc_lin,
        )
        th_tr_t = torch.from_numpy(th_tr_aug.astype(np.float32)).to(device)
        th_va_t = torch.from_numpy(th_va_aug.astype(np.float32)).to(device)
    else:
        th_tr_t = th_tr_kernel_t
        th_va_t = th_va_kernel_t
    ntr = int(x_tr_t.shape[0])
    nva = int(x_va_t.shape[0])
    effective_batch_size = min(int(batch_size), ntr, nva)
    if effective_batch_size < 2:
        raise ValueError("bidir-contrastive-soft requires effective train/validation batch size >= 2.")
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    train_losses: list[float] = []
    train_row_losses: list[float] = []
    train_col_losses: list[float] = []
    val_losses: list[float] = []
    val_row_losses: list[float] = []
    val_col_losses: list[float] = []
    val_ema_losses: list[float] = []
    bandwidth_schedule: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_ema = float("inf")
    best_epoch = 0
    bad = 0
    ema: float | None = None
    stopped_early = False
    stopped_epoch = int(epochs)
    n_clipped = 0
    n_total_steps = 0
    period_eff = float(period) / theta_scale if bool(periodic) else float(period)

    for ep in range(1, int(epochs) + 1):
        frac = 0.0 if int(epochs) <= 1 else float(ep - 1) / float(int(epochs) - 1)
        h = float(h_start) + frac * (float(h_end) - float(h_start))
        bandwidth_schedule.append(float(h))
        model.train()
        idx = torch.randperm(ntr, device=device)[:effective_batch_size]
        loss, row_loss, col_loss = _bidir_soft_contrastive_loss_parts(
            model,
            x_tr_t[idx],
            th_tr_t[idx],
            theta_kernel=th_tr_kernel_t[idx],
            bandwidth=float(h),
            periodic=bool(periodic),
            period=float(period_eff),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if float(max_grad_norm) > 0.0:
            g = torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            if float(g.detach().cpu()) > float(max_grad_norm):
                n_clipped += 1
        opt.step()
        n_total_steps += 1

        tr = float(loss.detach().cpu().item())
        tr_row = float(row_loss.detach().cpu().item())
        tr_col = float(col_loss.detach().cpu().item())
        train_losses.append(tr)
        train_row_losses.append(tr_row)
        train_col_losses.append(tr_col)
        va, va_row, va_col = _eval_bidir_soft_contrastive_loss(
            model,
            x_va_t,
            th_va_t,
            theta_kernel=th_va_kernel_t,
            batch_size=effective_batch_size,
            bandwidth=float(h),
            periodic=bool(periodic),
            period=float(period_eff),
        )
        val_losses.append(va)
        val_row_losses.append(va_row)
        val_col_losses.append(va_col)
        ema = va if ema is None else (float(ema_alpha) * va + (1.0 - float(ema_alpha)) * float(ema))
        val_ema_losses.append(float(ema))
        if np.isfinite(float(ema)) and float(ema) < best_ema - float(min_delta):
            best_ema = float(ema)
            best_epoch = int(ep)
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        if ep == 1 or ep % max(1, int(log_every)) == 0 or ep == int(epochs):
            alpha_msg = ""
            if hasattr(model, "alpha"):
                alpha_msg = f" alpha={float(model.alpha.detach().cpu().item()):.6g}"
            print(
                f"[bidir_contrastive_soft {ep:4d}/{int(epochs)}] train_bi_ce={tr:.6f} "
                f"train_row={tr_row:.6f} train_col={tr_col:.6f} val_bi_ce={va:.6f} "
                f"val_row={va_row:.6f} val_col={va_col:.6f} val_smooth={float(ema):.6f} "
                f"best_smooth={best_ema:.6f} best_epoch={best_epoch} h_norm={float(h):.6g} "
                f"batch={effective_batch_size}{alpha_msg}",
                flush=True,
            )
        if int(patience) > 0 and bad >= int(patience):
            stopped_early = True
            stopped_epoch = int(ep)
            print(
                f"[bidir_contrastive_soft early-stop] epoch={ep} best_epoch={best_epoch} "
                f"best_smooth={best_ema:.6f} patience={int(patience)}",
                flush=True,
            )
            break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses": train_losses,
        "train_row_losses": train_row_losses,
        "train_col_losses": train_col_losses,
        "val_losses": val_losses,
        "val_row_losses": val_row_losses,
        "val_col_losses": val_col_losses,
        "val_monitor_losses": val_ema_losses,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "stopped_early": stopped_early,
        "best_val_loss": best_ema,
        "x_mean": x_mean,
        "x_std": x_std,
        "theta_mean": theta_mean,
        "theta_std": theta_std,
        "bandwidth_raw": float(bandwidth_schedule[-1] * theta_scale),
        "bandwidth_normalized": float(bandwidth_schedule[-1]),
        "bandwidth_auto": False,
        "bandwidth_anneal_enabled": bool(bandwidth_anneal_enabled),
        "bandwidth_bins": int(bw_bins),
        "bandwidth_start_raw": float(h_start * theta_scale),
        "bandwidth_end_raw": float(h_end * theta_scale),
        "bandwidth_start_normalized": float(h_start),
        "bandwidth_end_normalized": float(h_end),
        "bandwidth_raw_schedule": [float(v * theta_scale) for v in bandwidth_schedule],
        "bandwidth_normalized_schedule": bandwidth_schedule,
        "lr_last": float(opt.param_groups[0]["lr"]),
        "n_clipped_steps": n_clipped,
        "n_total_steps": n_total_steps,
        "effective_batch_size": int(effective_batch_size),
        "contrastive_theta_fourier_k": fk,
        "contrastive_theta_fourier_period_mult": pm,
        "contrastive_theta_fourier_include_linear": inc_lin,
        "theta_fourier_ref_min": ref_min,
        "theta_fourier_ref_max": ref_max,
    }


def compute_contrastive_c_matrix(
    *,
    model: ContrastiveLLRMLP,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    bin_all: np.ndarray,
    n_bins: int,
    theta_encoding: str = "one_hot_bin",
    device: torch.device,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    pair_batch_size: int = 65536,
) -> np.ndarray:
    theta = _as_2d_float64(theta_all, name="theta_all")
    x = _as_2d_float64(x_all, name="x_all")
    if theta.shape[0] != x.shape[0]:
        raise ValueError("theta_all and x_all row counts must match.")
    if int(x.shape[1]) != model.x_dim:
        raise ValueError("x dimension does not match ContrastiveLLRMLP.x_dim.")
    theta_encoding_norm = normalize_theta_encoding(theta_encoding)
    expected_theta_dim = theta_dim_for_encoding(int(n_bins), theta_encoding_norm)
    if int(model.theta_dim) != int(expected_theta_dim):
        raise ValueError(
            f"ContrastiveLLRMLP.theta_dim={model.theta_dim} does not match "
            f"{theta_encoding_norm} expected dim={expected_theta_dim}."
        )
    n = int(x.shape[0])
    if n < 1:
        raise ValueError("Need at least one row to compute contrastive C matrix.")
    bins = np.asarray(bin_all, dtype=np.int64).reshape(-1)
    if bins.shape[0] != n:
        raise ValueError("bin_all length must match theta_all/x_all rows.")
    if np.min(bins) < 0 or np.max(bins) >= int(n_bins):
        raise ValueError("bin_all contains values outside [0, n_bins).")
    pb = max(1, int(pair_batch_size))
    row_bs = max(1, min(n, pb // n))

    x_n = (x - np.asarray(x_mean, dtype=np.float64).reshape(1, -1)) / np.asarray(x_std, dtype=np.float64).reshape(1, -1)
    x_t = torch.from_numpy(x_n.astype(np.float32))
    bin_t = torch.from_numpy(bins.astype(np.int64))
    c = np.empty((n, n), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        theta_dev = encode_bins(bin_t.to(device), int(n_bins), theta_encoding_norm).to(device=device)
        for i0 in range(0, n, row_bs):
            i1 = min(n, i0 + row_bs)
            logits = model.score_matrix(x_t[i0:i1].to(device), theta_dev)
            c[i0:i1, :] = logits.detach().cpu().numpy().astype(np.float64, copy=False)
    return c


def compute_contrastive_soft_c_matrix(
    *,
    model: ContrastiveLLRMLP,
    theta_all: np.ndarray,
    x_all: np.ndarray,
    device: torch.device,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    theta_mean: np.ndarray,
    theta_std: np.ndarray,
    pair_batch_size: int = 65536,
    contrastive_theta_fourier_k: int = 0,
    contrastive_theta_fourier_period_mult: float = 2.0,
    contrastive_theta_fourier_include_linear: bool = False,
    theta_fourier_ref: np.ndarray | None = None,
) -> np.ndarray:
    theta = _as_2d_float64(theta_all, name="theta_all")
    x = _as_2d_float64(x_all, name="x_all")
    if theta.shape[0] != x.shape[0]:
        raise ValueError("theta_all and x_all row counts must match.")
    d_theta = int(theta.shape[1])
    if int(x.shape[1]) != model.x_dim:
        raise ValueError("x dimension does not match ContrastiveLLRMLP.x_dim.")
    fk = int(contrastive_theta_fourier_k)
    pm = float(contrastive_theta_fourier_period_mult)
    inc_lin = bool(contrastive_theta_fourier_include_linear)
    if fk > 0:
        if d_theta != 1:
            raise ValueError(
                "compute_contrastive_soft_c_matrix with Fourier features requires scalar theta (d_theta=1)."
            )
        expected_theta_dim = int(dot_scorer_augmented_theta_dim(fourier_k=fk, fourier_include_linear=inc_lin))
    else:
        expected_theta_dim = int(d_theta)
    if int(model.theta_dim) != int(expected_theta_dim):
        raise ValueError(
            f"model.theta_dim={int(model.theta_dim)} does not match expected theta width {expected_theta_dim} "
            f"(d_theta={d_theta}, fourier_k={fk}, fourier_include_linear={inc_lin})."
        )
    if fk > 0:
        if theta_fourier_ref is None:
            raise ValueError(
                "compute_contrastive_soft_c_matrix requires theta_fourier_ref (training theta) when fourier_k > 0."
            )
        if not math.isfinite(pm) or pm <= 0.0:
            raise ValueError("contrastive_theta_fourier_period_mult must be finite and > 0 when fourier_k > 0.")
    n = int(x.shape[0])
    if n < 1:
        raise ValueError("Need at least one row to compute contrastive-soft C matrix.")
    pb = max(1, int(pair_batch_size))
    row_bs = max(1, min(n, pb // n))
    x_n = (x - np.asarray(x_mean, dtype=np.float64).reshape(1, -1)) / np.asarray(x_std, dtype=np.float64).reshape(1, -1)
    th_n = (theta - np.asarray(theta_mean, dtype=np.float64).reshape(1, -1)) / np.asarray(theta_std, dtype=np.float64).reshape(1, -1)
    if fk > 0:
        th_feat = augment_scalar_theta_for_dot_scorer(
            th_n,
            theta,
            _as_2d_float64(theta_fourier_ref, name="theta_fourier_ref"),
            fk,
            pm,
            inc_lin,
        )
    else:
        th_feat = th_n.astype(np.float64, copy=False)
    x_t = torch.from_numpy(x_n.astype(np.float32))
    th_t = torch.from_numpy(th_feat.astype(np.float32))
    c = np.empty((n, n), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        theta_dev = th_t.to(device)
        for i0 in range(0, n, row_bs):
            i1 = min(n, i0 + row_bs)
            logits = model.score_matrix(x_t[i0:i1].to(device), theta_dev)
            c[i0:i1, :] = logits.detach().cpu().numpy().astype(np.float64, copy=False)
    return c


def h_directed_from_delta_l(delta_l: np.ndarray) -> np.ndarray:
    """One-sided H^2 estimate from per-row log likelihood ratios."""
    d = np.asarray(delta_l, dtype=np.float64)
    z = np.clip(0.5 * d, -60.0, 60.0)
    h = 1.0 - np.exp(z)
    np.fill_diagonal(h, 0.0)
    return h
