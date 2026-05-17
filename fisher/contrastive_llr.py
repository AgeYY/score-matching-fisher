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
            ContrastiveAdditiveIndependentScorer,
        ),
    )


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
    model: nn.Module,
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


def categorical_soft_targets(labels: torch.Tensor, n_classes: int, beta: float = 0.0) -> torch.Tensor:
    """Class-level categorical soft targets: true class weight 1, off-class weight beta."""
    y = labels.reshape(-1).to(dtype=torch.long)
    if y.numel() < 1:
        raise ValueError("labels must be non-empty.")
    k = int(n_classes)
    if k < 2:
        raise ValueError("n_classes must be >= 2.")
    if int(torch.min(y).detach().cpu()) < 0 or int(torch.max(y).detach().cpu()) >= k:
        raise ValueError("labels contain values outside [0, n_classes).")
    b = float(beta)
    if not math.isfinite(b) or b < 0.0:
        raise ValueError("beta must be finite and >= 0.")
    weights = torch.full((int(y.numel()), k), b, dtype=torch.float32, device=y.device)
    weights.scatter_(1, y.reshape(-1, 1), 1.0)
    return weights / weights.sum(dim=1, keepdim=True)


def _soft_categorical_contrastive_loss(
    model: nn.Module,
    x: torch.Tensor,
    labels: torch.Tensor,
    class_codes: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    if int(x.shape[0]) < 1:
        raise ValueError("categorical soft contrastive minibatch must contain at least one row.")
    logits = model.score_matrix(x, class_codes)
    weights = categorical_soft_targets(labels, int(class_codes.shape[0]), beta=float(beta)).to(dtype=logits.dtype)
    log_probs = torch.log_softmax(logits, dim=1)
    return -(weights * log_probs).sum(dim=1).mean()


def contrastive_soft_normalization_and_bandwidth_from_train(
    *,
    th_tr: np.ndarray,
    x_tr: np.ndarray,
    bandwidth_bins: int,
    periodic: bool,
    period: float,
) -> dict[str, Any]:
    """Train-set normalization and fixed soft-contrastive bandwidth (matches ``train_contrastive_soft_llr``).

    Raw bandwidth is ``theta_range / (2 * bandwidth_bins)`` where ``theta_range`` is the train
    per-coordinate span (scalar: ``max-min`` on training θ; multi-dim: ``max_j (max θ_j - min θ_j)``).
    That scale is converted to normalized space using ``mean(theta_std)`` so the Gaussian kernel
    applies to Euclidean distance on z-scored θ.

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

    if int(bandwidth_bins) < 1:
        raise ValueError("contrastive-soft bandwidth_bins must be >= 1.")

    theta_scale = float(np.mean(theta_std))
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
    h_norm = h_raw / theta_scale
    if not math.isfinite(float(h_norm)) or float(h_norm) <= 0.0:
        raise ValueError("effective contrastive-soft bandwidth must be finite and > 0.")

    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "theta_mean": theta_mean,
        "theta_std": theta_std,
        "theta_scale": theta_scale,
        "h_norm": float(h_norm),
        "bandwidth_bins": int(bandwidth_bins),
    }


def _eval_soft_contrastive_loss(
    model: nn.Module,
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


def _eval_soft_categorical_contrastive_loss(
    model: nn.Module,
    x: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_codes: torch.Tensor,
    batch_size: int,
    beta: float,
) -> float:
    n = int(x.shape[0])
    if n < 1:
        return float("nan")
    bs = min(n, max(1, int(batch_size)))
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for i0 in range(0, n, bs):
            i1 = min(n, i0 + bs)
            if i1 - i0 != bs and losses:
                continue
            loss = _soft_categorical_contrastive_loss(
                model,
                x[i0:i1],
                labels[i0:i1],
                class_codes,
                beta=float(beta),
            )
            losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


def train_contrastive_soft_llr(
    *,
    model: nn.Module,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    bandwidth_bins: int = 10,
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
        raise ValueError("x dimension does not match model.x_dim.")
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
                "ContrastiveNormalizedDotScorer and ContrastiveAdditiveIndependentScorer; "
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
        periodic=bool(periodic),
        period=float(period),
    )
    x_mean = nb["x_mean"]
    x_std = nb["x_std"]
    theta_mean = nb["theta_mean"]
    theta_std = nb["theta_std"]
    theta_scale = float(nb["theta_scale"])
    bw_bins = int(nb["bandwidth_bins"])
    h = float(nb["h_norm"])
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
        "bandwidth_raw": float(h * theta_scale),
        "bandwidth_normalized": float(h),
        "bandwidth_auto": True,
        "bandwidth_bins": int(bw_bins),
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


def train_contrastive_soft_categorical_llr(
    *,
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    beta: float = 0.0,
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
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1 for categorical soft contrastive learning.")
    if int(n_classes) < 2:
        raise ValueError("n_classes must be >= 2 for categorical soft contrastive learning.")
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
    if not math.isfinite(float(beta)) or float(beta) < 0.0:
        raise ValueError("beta must be finite and >= 0.")

    x_tr = _as_2d_float64(x_train, name="x_train")
    x_va = _as_2d_float64(x_val, name="x_val")
    y_tr = np.asarray(y_train, dtype=np.int64).reshape(-1)
    y_va = np.asarray(y_val, dtype=np.int64).reshape(-1)
    if x_tr.shape[0] < 2 or x_va.shape[0] < 2:
        raise ValueError("contrastive-soft-categorical requires at least two train and two validation rows.")
    if y_tr.shape[0] != x_tr.shape[0] or y_va.shape[0] != x_va.shape[0]:
        raise ValueError("labels must match x row counts.")
    if int(x_tr.shape[1]) != model.x_dim or int(x_va.shape[1]) != model.x_dim:
        raise ValueError("x dimension does not match model.x_dim.")
    if int(model.theta_dim) != int(n_classes):
        raise ValueError(f"model.theta_dim={model.theta_dim} does not match n_classes={int(n_classes)}.")
    if np.min(y_tr) < 0 or np.max(y_tr) >= int(n_classes) or np.min(y_va) < 0 or np.max(y_va) >= int(n_classes):
        raise ValueError("labels contain values outside [0, n_classes).")
    if np.unique(y_tr).size < 2:
        raise ValueError("contrastive-soft-categorical requires at least two occupied training classes.")
    if np.unique(y_va).size < 2:
        raise ValueError("contrastive-soft-categorical requires at least two occupied validation classes.")

    x_mean = np.mean(x_tr, axis=0, dtype=np.float64)
    x_std = np.maximum(np.std(x_tr, axis=0, dtype=np.float64), 1e-6)
    x_tr_n = (x_tr - x_mean) / x_std
    x_va_n = (x_va - x_mean) / x_std

    x_tr_t = torch.from_numpy(x_tr_n.astype(np.float32)).to(device)
    x_va_t = torch.from_numpy(x_va_n.astype(np.float32)).to(device)
    y_tr_t = torch.from_numpy(y_tr.astype(np.int64)).to(device)
    y_va_t = torch.from_numpy(y_va.astype(np.int64)).to(device)
    class_codes = torch.eye(int(n_classes), dtype=torch.float32, device=device)
    class_priors = np.bincount(y_tr, minlength=int(n_classes)).astype(np.float64)
    class_priors = class_priors / np.maximum(float(class_priors.sum()), 1.0)

    ntr = int(x_tr_t.shape[0])
    nva = int(x_va_t.shape[0])
    effective_batch_size = min(int(batch_size), ntr, nva)
    if effective_batch_size < 1:
        raise ValueError("categorical soft contrastive requires effective batch size >= 1.")
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
        idx = torch.randperm(ntr, device=device)[:effective_batch_size]
        loss = _soft_categorical_contrastive_loss(
            model,
            x_tr_t[idx],
            y_tr_t[idx],
            class_codes,
            beta=float(beta),
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
        va = _eval_soft_categorical_contrastive_loss(
            model,
            x_va_t,
            y_va_t,
            class_codes=class_codes,
            batch_size=effective_batch_size,
            beta=float(beta),
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
                f"[contrastive_soft_categorical {ep:4d}/{int(epochs)}] train_soft_ce={tr:.6f} "
                f"val_soft_ce={va:.6f} val_smooth={float(ema):.6f} best_smooth={best_ema:.6f} "
                f"best_epoch={best_epoch} beta={float(beta):.6g} batch={effective_batch_size}{alpha_msg}",
                flush=True,
            )
        if int(patience) > 0 and bad >= int(patience):
            stopped_early = True
            stopped_epoch = int(ep)
            print(
                f"[contrastive_soft_categorical early-stop] epoch={ep} best_epoch={best_epoch} "
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
        "n_classes": int(n_classes),
        "beta": float(beta),
        "class_priors": class_priors,
        "lr_last": float(opt.param_groups[0]["lr"]),
        "n_clipped_steps": n_clipped,
        "n_total_steps": n_total_steps,
        "effective_batch_size": int(effective_batch_size),
    }


def compute_contrastive_soft_c_matrix(
    *,
    model: nn.Module,
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
        raise ValueError("x dimension does not match model.x_dim.")
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


def compute_contrastive_soft_categorical_c_matrix(
    *,
    model: nn.Module,
    x_all: np.ndarray,
    y_all: np.ndarray,
    n_classes: int,
    device: torch.device,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    pair_batch_size: int = 65536,
) -> np.ndarray:
    x = _as_2d_float64(x_all, name="x_all")
    if int(x.shape[1]) != model.x_dim:
        raise ValueError("x dimension does not match model.x_dim.")
    k = int(n_classes)
    if k < 2:
        raise ValueError("n_classes must be >= 2.")
    if int(model.theta_dim) != k:
        raise ValueError(f"model.theta_dim={int(model.theta_dim)} does not match n_classes={k}.")
    n = int(x.shape[0])
    if n < 1:
        raise ValueError("Need at least one row to compute contrastive-soft-categorical C matrix.")
    labels = np.asarray(y_all, dtype=np.int64).reshape(-1)
    if labels.shape[0] != n:
        raise ValueError("y_all length must match x_all rows.")
    if np.min(labels) < 0 or np.max(labels) >= k:
        raise ValueError("y_all contains values outside [0, n_classes).")

    pb = max(1, int(pair_batch_size))
    row_bs = max(1, min(n, pb // k))
    x_n = (x - np.asarray(x_mean, dtype=np.float64).reshape(1, -1)) / np.asarray(x_std, dtype=np.float64).reshape(1, -1)
    x_t = torch.from_numpy(x_n.astype(np.float32))
    class_codes = torch.eye(k, dtype=torch.float32, device=device)
    class_scores = np.empty((n, k), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for i0 in range(0, n, row_bs):
            i1 = min(n, i0 + row_bs)
            logits = model.score_matrix(x_t[i0:i1].to(device), class_codes)
            class_scores[i0:i1, :] = logits.detach().cpu().numpy().astype(np.float64, copy=False)
    return class_scores[:, labels]


def h_directed_from_delta_l(delta_l: np.ndarray) -> np.ndarray:
    """One-sided H^2 estimate from per-row log likelihood ratios."""
    d = np.asarray(delta_l, dtype=np.float64)
    z = np.clip(0.5 * d, -60.0, 60.0)
    h = 1.0 - np.exp(z)
    np.fill_diagonal(h, 0.0)
    return h
