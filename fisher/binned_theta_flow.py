"""Binned theta-flow posterior mixture utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from fisher.h_matrix import HMatrixEstimator
from fisher.models import ConditionalThetaFlowVelocity
from fisher.trainers import train_conditional_theta_flow_model, train_weighted_conditional_theta_flow_model


@dataclass(frozen=True)
class ThetaBinSpec:
    edges: np.ndarray

    @property
    def n_bins(self) -> int:
        return int(self.edges.size - 1)

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])


@dataclass(frozen=True)
class ThetaSoftBinSpec:
    """RBF mixture centers and bandwidth (scalar theta, non-periodic)."""

    centers: np.ndarray
    sigma: float
    center_mode: str

    @property
    def n_experts(self) -> int:
        return int(np.asarray(self.centers).size)


def derive_soft_sigma_from_centers(centers: np.ndarray, theta_range: tuple[float, float] | None = None) -> float:
    """Default σ = mean(Δμ)/4 for adjacent sorted centers μ; K==1 uses half θ-range."""
    c = np.asarray(centers, dtype=np.float64).reshape(-1)
    if c.size < 1:
        raise ValueError("centers must be non-empty.")
    if c.size == 1:
        if theta_range is None:
            raise ValueError("theta_range is required when there is a single center.")
        lo, hi = float(theta_range[0]), float(theta_range[1])
        span = max(hi - lo, 1e-12)
        return float(0.5 * span)
    c_sorted = np.sort(c)
    spacing = float(np.mean(np.diff(c_sorted)))
    return float(max(0.25 * spacing, 1e-12))


def make_soft_theta_bins(
    theta: np.ndarray,
    k_experts: int,
    *,
    center_mode: str = "uniform",
    sigma: float | None = None,
) -> ThetaSoftBinSpec:
    """Build K Gaussian RBF centers (equal-width bin midpoints or quantile-bin midpoints)."""
    th = np.asarray(theta, dtype=np.float64).reshape(-1)
    if th.size < 2:
        raise ValueError("Need at least two theta values to build soft bins.")
    if int(k_experts) < 2:
        raise ValueError("smooth_binned_theta_flow requires at least two mixture components.")
    lo = float(np.min(th))
    hi = float(np.max(th))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("theta range must be finite and non-degenerate.")
    mode = str(center_mode).strip().lower()
    k = int(k_experts)
    if mode == "uniform":
        centers = np.asarray(make_equal_width_theta_bins(th, k).centers, dtype=np.float64).reshape(-1)
    elif mode == "quantile":
        qs = np.linspace(0.0, 1.0, k + 1, dtype=np.float64)
        edges = np.asarray(np.quantile(th, qs), dtype=np.float64).reshape(-1)
        centers = (0.5 * (edges[:-1] + edges[1:])).astype(np.float64)
    else:
        raise ValueError("center_mode must be one of: uniform, quantile.")
    sig = float(sigma) if sigma is not None else -1.0
    if sig <= 0.0:
        sig = derive_soft_sigma_from_centers(centers, theta_range=(lo, hi))
    if not np.isfinite(sig) or sig <= 0.0:
        raise ValueError("sigma must be finite and positive after derivation.")
    return ThetaSoftBinSpec(centers=centers, sigma=float(sig), center_mode=mode)


def soft_theta_responsibilities(theta: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    """Row-normalized Gaussian RBF weights exp(-0.5 * ((θ-μ)/σ)²), stabilized."""
    th = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
    mu = np.asarray(centers, dtype=np.float64).reshape(1, -1)
    if float(sigma) <= 0.0 or not np.isfinite(float(sigma)):
        raise ValueError("sigma must be finite and positive.")
    z = (th - mu) / float(sigma)
    logw = -0.5 * z * z
    log_max = np.max(logw, axis=1, keepdims=True)
    w = np.exp(logw - log_max)
    r = w / np.sum(w, axis=1, keepdims=True)
    return r.astype(np.float64)


def mixture_log_density_matrix(*, log_pi: np.ndarray, log_q_experts: list[np.ndarray]) -> np.ndarray:
    """C[i,j] = logsumexp_k ( log π_k(x_i) + log q_k(θ_j | x_i) )."""
    lp = np.asarray(log_pi, dtype=np.float64)
    if lp.ndim != 2:
        raise ValueError("log_pi must be (N, K).")
    k = lp.shape[1]
    if len(log_q_experts) != k:
        raise ValueError("log_q_experts length must match log_pi's K dimension.")
    stack = np.stack(
        [lp[:, kk].reshape(-1, 1) + np.asarray(log_q_experts[kk], dtype=np.float64) for kk in range(k)],
        axis=0,
    )
    m = np.max(stack, axis=0)
    return m + np.log(np.sum(np.exp(stack - m), axis=0))


class BinPosteriorClassifierMLP(nn.Module):
    """MLP classifier for pi(k | x)."""

    def __init__(self, *, x_dim: int, n_bins: int, hidden_dim: int = 128, depth: int = 2) -> None:
        super().__init__()
        if int(x_dim) < 1:
            raise ValueError("x_dim must be >= 1.")
        if int(n_bins) < 2:
            raise ValueError("n_bins must be >= 2.")
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be >= 1.")
        if int(depth) < 1:
            raise ValueError("depth must be >= 1.")
        layers: list[nn.Module] = []
        in_dim = int(x_dim)
        for _ in range(int(depth)):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.SiLU())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, int(n_bins)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_equal_width_theta_bins(theta: np.ndarray, n_bins: int) -> ThetaBinSpec:
    th = np.asarray(theta, dtype=np.float64).reshape(-1)
    if th.size < 2:
        raise ValueError("Need at least two theta values to build bins.")
    if int(n_bins) < 2:
        raise ValueError("binned theta-flow requires at least two bins.")
    lo = float(np.min(th))
    hi = float(np.max(th))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("theta range must be finite and non-degenerate.")
    edges = np.linspace(lo, hi, int(n_bins) + 1, dtype=np.float64)
    return ThetaBinSpec(edges=edges)


def assign_theta_bins(theta: np.ndarray, spec: ThetaBinSpec) -> np.ndarray:
    th = np.asarray(theta, dtype=np.float64).reshape(-1)
    labels = np.searchsorted(spec.edges, th, side="right") - 1
    return np.clip(labels, 0, spec.n_bins - 1).astype(np.int64)


def normalize_theta_in_bins(theta: np.ndarray, labels: np.ndarray, spec: ThetaBinSpec) -> np.ndarray:
    th = np.asarray(theta, dtype=np.float64).reshape(-1, 1)
    lab = np.asarray(labels, dtype=np.int64).reshape(-1)
    if th.shape[0] != lab.size:
        raise ValueError("theta and labels length mismatch.")
    left = spec.edges[lab].reshape(-1, 1)
    width = spec.widths[lab].reshape(-1, 1)
    return (th - left) / width


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, *, n_bins: int = 10) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if p.ndim != 2 or p.shape[0] != y.size:
        raise ValueError("probs must be shape (N,K) and labels length N.")
    conf = np.max(p, axis=1)
    pred = np.argmax(p, axis=1)
    correct = (pred == y).astype(np.float64)
    ece = 0.0
    for b in range(int(n_bins)):
        lo = b / float(n_bins)
        hi = (b + 1) / float(n_bins)
        mask = (conf >= lo) & (conf <= hi if b == int(n_bins) - 1 else conf < hi)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(correct[mask])) - float(np.mean(conf[mask])))
    return float(ece)


def _classifier_loader(x: np.ndarray, y: np.ndarray, batch_size: int, *, shuffle: bool) -> DataLoader:
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64).reshape(-1))
    return DataLoader(TensorDataset(xt, yt), batch_size=int(batch_size), shuffle=bool(shuffle), drop_last=False)


def _eval_classifier_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    losses: list[float] = []
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            losses.append(float(criterion(model(xb), yb).item()))
    return float(np.mean(losses)) if losses else float("nan")


def train_bin_posterior_classifier(
    *,
    model: BinPosteriorClassifierMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    early_patience: int,
    early_min_delta: float,
    early_ema_alpha: float,
    log_every: int,
) -> dict[str, Any]:
    if int(epochs) < 1:
        raise ValueError("classifier epochs must be >= 1.")
    train_loader = _classifier_loader(x_train, y_train, batch_size, shuffle=True)
    val_loader = _classifier_loader(x_val, y_val, batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_monitor_losses: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = 0
    patience = 0
    val_ema: float | None = None
    stopped_epoch = int(epochs)
    stopped_early = False
    for epoch in range(1, int(epochs) + 1):
        model.train()
        epoch_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss = criterion(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.item()))
        train_loss = float(np.mean(epoch_losses))
        val_loss = _eval_classifier_loss(model, val_loader, device)
        val_ema = val_loss if val_ema is None else float(early_ema_alpha) * val_loss + (1.0 - float(early_ema_alpha)) * val_ema
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_monitor_losses.append(float(val_ema))
        if val_ema < best_val - float(early_min_delta):
            best_val = float(val_ema)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if epoch == 1 or epoch % max(1, int(log_every)) == 0 or epoch == int(epochs):
            print(f"[binned_theta_flow cls epoch {epoch:4d}/{epochs}] train={train_loss:.6f} val={val_loss:.6f} val_smooth={val_ema:.6f}", flush=True)
        if int(early_patience) > 0 and patience >= int(early_patience):
            stopped_epoch = int(epoch)
            stopped_early = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    probs_val = predict_bin_log_probs(model=model, x=x_val, device=device, batch_size=batch_size)
    probs_val_exp = np.exp(probs_val)
    yv = np.asarray(y_val, dtype=np.int64).reshape(-1)
    nll = -float(np.mean(probs_val[np.arange(yv.size), yv]))
    onehot = np.eye(probs_val.shape[1], dtype=np.float64)[yv]
    brier = float(np.mean(np.sum((probs_val_exp - onehot) ** 2, axis=1)))
    acc = float(np.mean(np.argmax(probs_val_exp, axis=1) == yv))
    ece = expected_calibration_error(probs_val_exp, yv)
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_monitor_losses": val_monitor_losses,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "stopped_early": stopped_early,
        "val_nll": nll,
        "val_brier": brier,
        "val_accuracy": acc,
        "val_ece": ece,
    }


def _soft_classifier_loader(x: np.ndarray, r: np.ndarray, batch_size: int, *, shuffle: bool) -> DataLoader:
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
    rt = torch.from_numpy(np.asarray(r, dtype=np.float32))
    if int(xt.shape[0]) != int(rt.shape[0]):
        raise ValueError("x and soft targets r must have the same number of rows.")
    return DataLoader(TensorDataset(xt, rt), batch_size=int(batch_size), shuffle=bool(shuffle), drop_last=False)


def _soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    return -(target * logp).sum(dim=-1).mean()


def _eval_soft_classifier_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for xb, rb in loader:
            xb = xb.to(device, non_blocking=True)
            rb = rb.to(device, non_blocking=True)
            losses.append(float(_soft_cross_entropy(model(xb), rb).item()))
    return float(np.mean(losses)) if losses else float("nan")


def train_soft_label_classifier(
    *,
    model: BinPosteriorClassifierMLP,
    x_train: np.ndarray,
    r_train: np.ndarray,
    x_val: np.ndarray,
    r_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    early_patience: int,
    early_min_delta: float,
    early_ema_alpha: float,
    log_every: int,
) -> dict[str, Any]:
    if int(epochs) < 1:
        raise ValueError("classifier epochs must be >= 1.")
    train_loader = _soft_classifier_loader(x_train, r_train, batch_size, shuffle=True)
    val_loader = _soft_classifier_loader(x_val, r_val, batch_size, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_monitor_losses: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = 0
    patience = 0
    val_ema: float | None = None
    stopped_epoch = int(epochs)
    stopped_early = False
    for epoch in range(1, int(epochs) + 1):
        model.train()
        epoch_losses: list[float] = []
        for xb, rb in train_loader:
            xb = xb.to(device, non_blocking=True)
            rb = rb.to(device, non_blocking=True)
            loss = _soft_cross_entropy(model(xb), rb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.item()))
        train_loss = float(np.mean(epoch_losses))
        val_loss = _eval_soft_classifier_loss(model, val_loader, device)
        val_ema = val_loss if val_ema is None else float(early_ema_alpha) * val_loss + (1.0 - float(early_ema_alpha)) * val_ema
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_monitor_losses.append(float(val_ema))
        if val_ema < best_val - float(early_min_delta):
            best_val = float(val_ema)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if epoch == 1 or epoch % max(1, int(log_every)) == 0 or epoch == int(epochs):
            print(
                f"[smooth_binned_theta_flow cls epoch {epoch:4d}/{epochs}] train={train_loss:.6f} val={val_loss:.6f} val_smooth={val_ema:.6f}",
                flush=True,
            )
        if int(early_patience) > 0 and patience >= int(early_patience):
            stopped_epoch = int(epoch)
            stopped_early = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    logp_val = predict_bin_log_probs(model=model, x=x_val, device=device, batch_size=batch_size)
    p_val = np.exp(logp_val)
    rv = np.asarray(r_val, dtype=np.float64)
    if rv.ndim != 2 or rv.shape != p_val.shape:
        raise ValueError("r_val shape must match softmax probabilities.")
    soft_brier = float(np.mean(np.sum((p_val - rv) ** 2, axis=1)))
    soft_ce = -float(np.mean(np.sum(rv * logp_val, axis=1)))
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_monitor_losses": val_monitor_losses,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "stopped_early": stopped_early,
        "val_soft_cross_entropy": soft_ce,
        "val_soft_brier": soft_brier,
        "val_nll": soft_ce,
        "val_brier": soft_brier,
        "val_accuracy": float("nan"),
        "val_ece": float("nan"),
    }


def predict_bin_log_probs(
    *,
    model: BinPosteriorClassifierMLP,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(np.asarray(x, dtype=np.float32))), batch_size=int(batch_size), shuffle=False)
    outs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            outs.append(torch.log_softmax(model(xb), dim=-1).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(outs, axis=0)


class BinPairBinaryClassifierMLP(nn.Module):
    """Binary logistic head: one logit scoping approx log π(a|x)/π(b|x) for a fixed bin pair (a,b), a<b."""

    def __init__(self, *, x_dim: int, hidden_dim: int = 128, depth: int = 2) -> None:
        super().__init__()
        if int(x_dim) < 1:
            raise ValueError("x_dim must be >= 1.")
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be >= 1.")
        if int(depth) < 1:
            raise ValueError("depth must be >= 1.")
        layers: list[nn.Module] = []
        in_dim = int(x_dim)
        for _ in range(int(depth)):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.SiLU())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def unordered_bin_pairs(K: int) -> list[tuple[int, int]]:
    """All unordered pairs (a,b) with a < b; row order matches ``build_pairwise_ls_reduction_matrix``."""
    kk = int(K)
    if kk < 2:
        raise ValueError("K must be >= 2.")
    pairs: list[tuple[int, int]] = []
    for a in range(kk):
        for b in range(a + 1, kk):
            pairs.append((a, b))
    return pairs


def build_pairwise_ls_reduction_matrix(K: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Matrix B such that B @ z ≈ s encodes (ℓ_a - ℓ_b) with gauge ℓ_0 = 0 and z_k = ℓ_{k+1}.

    Rows follow ``unordered_bin_pairs(K)`` order: one row per pair (a,b), a<b.
    """
    pairs = unordered_bin_pairs(K)
    P = len(pairs)
    kk = int(K)
    B = np.zeros((P, kk - 1), dtype=np.float64)
    for row, (a, b) in enumerate(pairs):
        if a == 0:
            B[row, b - 1] = -1.0
        else:
            B[row, a - 1] = 1.0
            B[row, b - 1] -= 1.0
    return B, pairs


def assemble_log_pi_pairwise_ls(
    pairwise_logits_corrected: np.ndarray,
    *,
    K: int,
    ridge: float,
    pair_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Map pairwise log-ratio estimates to full ``log π_k(x)`` via ridge LSQ + ``log_softmax``.

    ``pairwise_logits_corrected[i,p]`` ≈ log π(a|x_i)/π(b|x_i) for pair p=(a,b) (same order as ``unordered_bin_pairs``).
    Gauge: ℓ_0 = 0; optimize ℓ_1..ℓ_{K-1} with ridge on reduced coordinates.
    """
    S = np.asarray(pairwise_logits_corrected, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError("pairwise_logits_corrected must be 2D (N, P).")
    kk = int(K)
    B, _pairs = build_pairwise_ls_reduction_matrix(kk)
    P = int(B.shape[0])
    if int(S.shape[1]) != P:
        raise ValueError(f"Expected P={P} pairwise columns for K={kk}, got {S.shape[1]}.")
    n = int(S.shape[0])
    w = np.ones(P, dtype=np.float64) if pair_weights is None else np.asarray(pair_weights, dtype=np.float64).reshape(P)
    if w.shape[0] != P:
        raise ValueError("pair_weights length must match number of pairs.")
    if np.any(w < 0.0) or not np.all(np.isfinite(w)):
        raise ValueError("pair_weights must be finite and nonnegative.")
    lam = float(ridge)
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError("ridge must be finite and >= 0.")
    W = np.diag(w)
    G = B.T @ W @ B + lam * np.eye(kk - 1, dtype=np.float64)
    # z_i = G^{-1} B^T W s_i  →  Z = S @ W B @ G^{-1} ... check: B.T shape (K-1,P), W (P,P)
    # rhs_i = B.T @ W @ s_i  shape (K-1,)
    # Z = S @ W @ B @ inv(G) is wrong dim.
    # z_i = solve(G, B.T @ W @ s_i). Stack: columns are rhs for each i -> R = B.T @ W @ S.T  shape (K-1, N)
    rhs = B.T @ W @ S.T
    z_rows = np.linalg.solve(G, rhs).T
    ell = np.zeros((n, kk), dtype=np.float64)
    ell[:, 0] = 0.0
    ell[:, 1:] = z_rows
    m = np.max(ell, axis=1, keepdims=True)
    ex = np.exp(ell - m)
    denom = np.sum(ex, axis=1, keepdims=True)
    return (ell - m) - np.log(denom)


def empirical_bin_priors(labels: np.ndarray, K: int) -> np.ndarray:
    """Smoothed empirical class priors p(k) from hard labels (training set)."""
    c = np.bincount(np.asarray(labels, dtype=np.int64).reshape(-1), minlength=int(K)).astype(np.float64)
    s = float(np.sum(c))
    if s <= 0.0:
        raise ValueError("empty labels for prior computation.")
    return c / s


def train_bin_pair_binary_classifier(
    *,
    model: BinPairBinaryClassifierMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    early_patience: int,
    early_min_delta: float,
    early_ema_alpha: float,
    log_every: int,
    balance_classes: bool,
    pair_tag: str,
) -> dict[str, Any]:
    """Binary BCE training; optional balanced sampling via ``WeightedRandomSampler``."""
    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1.")
    xt = torch.from_numpy(np.asarray(x_train, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y_train, dtype=np.float64).reshape(-1))
    train_ds = TensorDataset(xt, yt)
    if bool(balance_classes):
        y_np = np.asarray(y_train, dtype=np.int64).reshape(-1)
        n_pos = int(np.sum(y_np == 1))
        n_neg = int(np.sum(y_np == 0))
        if n_pos < 1 or n_neg < 1:
            raise ValueError(f"{pair_tag}: balanced training requires both classes in training subset.")
        w_pair = np.where(y_np == 1, 1.0 / (2.0 * n_pos), 1.0 / (2.0 * n_neg)).astype(np.float64)
        sampler = WeightedRandomSampler(torch.from_numpy(w_pair), num_samples=len(w_pair), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=int(batch_size), sampler=sampler, drop_last=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, drop_last=False)
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.asarray(x_val, dtype=np.float32)),
            torch.from_numpy(np.asarray(y_val, dtype=np.float64).reshape(-1)),
        ),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_monitor_losses: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = 0
    patience = 0
    val_ema: float | None = None
    stopped_epoch = int(epochs)
    stopped_early = False
    for epoch in range(1, int(epochs) + 1):
        model.train()
        epoch_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, reduction="mean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.item()))
        train_loss = float(np.mean(epoch_losses))
        model.eval()
        vlosses: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                vlosses.append(float(F.binary_cross_entropy_with_logits(logits, yb, reduction="mean").item()))
        val_loss = float(np.mean(vlosses)) if vlosses else float("nan")
        val_ema = val_loss if val_ema is None else float(early_ema_alpha) * val_loss + (1.0 - float(early_ema_alpha)) * val_ema
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_monitor_losses.append(float(val_ema))
        if val_ema < best_val - float(early_min_delta):
            best_val = float(val_ema)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if epoch == 1 or epoch % max(1, int(log_every)) == 0 or epoch == int(epochs):
            print(
                f"[binary_btf {pair_tag} cls epoch {epoch:4d}/{epochs}] train={train_loss:.6f} val={val_loss:.6f} val_smooth={val_ema:.6f}",
                flush=True,
            )
        if int(early_patience) > 0 and patience >= int(early_patience):
            stopped_epoch = int(epoch)
            stopped_early = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    # validation binary accuracy / brier on val subset
    model.eval()
    probs_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            pr = torch.sigmoid(model(xb)).detach().cpu().numpy().astype(np.float64)
            probs_list.append(pr.reshape(-1))
            y_list.append(yb.numpy().astype(np.float64).reshape(-1))
    pv = np.concatenate(probs_list, axis=0)
    yv = np.concatenate(y_list, axis=0)
    brier = float(np.mean((pv - yv) ** 2))
    acc = float(np.mean((pv >= 0.5).astype(np.float64) == yv))
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_monitor_losses": val_monitor_losses,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "stopped_early": stopped_early,
        "val_brier": brier,
        "val_accuracy": acc,
        "val_nll": float("nan"),
        "val_ece": float("nan"),
    }


def predict_bin_pair_logit(
    *,
    model: BinPairBinaryClassifierMLP,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(np.asarray(x, dtype=np.float32))), batch_size=int(batch_size), shuffle=False)
    outs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            outs.append(model(xb).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(outs, axis=0)


def multiclass_metrics_from_log_pi(log_pi: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """NLL / Brier / acc / ECE vs hard labels when rows are log-prob vectors."""
    lp = np.asarray(log_pi, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if lp.ndim != 2 or lp.shape[0] != y.size:
        raise ValueError("log_pi must be (N,K) and labels length N.")
    pr = np.exp(lp - np.max(lp, axis=1, keepdims=True))
    pr = pr / np.sum(pr, axis=1, keepdims=True)
    nll = -float(np.mean(lp[np.arange(y.size), y]))
    onehot = np.eye(pr.shape[1], dtype=np.float64)[y]
    brier = float(np.mean(np.sum((pr - onehot) ** 2, axis=1)))
    acc = float(np.mean(np.argmax(pr, axis=1) == y))
    ece = expected_calibration_error(pr, y)
    return {"val_nll": nll, "val_brier": brier, "val_accuracy": acc, "val_ece": ece}


def make_local_theta_flow_model(*, x_dim: int, hidden_dim: int, depth: int) -> ConditionalThetaFlowVelocity:
    return ConditionalThetaFlowVelocity(
        x_dim=int(x_dim),
        hidden_dim=int(hidden_dim),
        depth=int(depth),
        use_logit_time=True,
        theta_dim=1,
    )


def _make_solver(model: nn.Module) -> Any:
    from flow_matching.solver.ode_solver import ODESolver

    def velocity(x: torch.Tensor, t: torch.Tensor, **extras: Any) -> torch.Tensor:
        x_cond = extras.get("x_cond")
        if x_cond is None:
            raise ValueError("local theta-flow likelihood requires x_cond.")
        if t.ndim == 0:
            t_col = t.to(device=x.device, dtype=x.dtype).expand(x.shape[0]).unsqueeze(-1)
        elif t.ndim == 1:
            t_col = t.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
            if t_col.shape[0] == 1:
                t_col = t_col.expand(x.shape[0], 1)
        else:
            t_col = t.to(device=x.device, dtype=x.dtype)
        return model(x, x_cond, t_col)

    return ODESolver(velocity_model=velocity)


def _log_std_normal(z: torch.Tensor) -> torch.Tensor:
    flat = z.reshape(z.shape[0], -1)
    return -0.5 * (flat.pow(2).sum(dim=1) + flat.shape[1] * math.log(2.0 * math.pi))


def local_flow_log_prob_matrix(
    *,
    model: nn.Module,
    theta_eval: np.ndarray,
    x_all: np.ndarray,
    device: torch.device,
    pair_batch_size: int,
    ode_steps: int,
    exact_divergence: bool,
) -> np.ndarray:
    theta_col = np.asarray(theta_eval, dtype=np.float32).reshape(-1, 1)
    x_arr = np.asarray(x_all, dtype=np.float32)
    n_rows = int(x_arr.shape[0])
    n_cols = int(theta_col.shape[0])
    row_block = max(1, int(pair_batch_size) // max(1, n_cols))
    out = np.zeros((n_rows, n_cols), dtype=np.float64)
    solver = _make_solver(model)
    model.eval()
    for i0 in range(0, n_rows, row_block):
        i1 = min(n_rows, i0 + row_block)
        xb = x_arr[i0:i1]
        b = int(i1 - i0)
        theta_tile = np.tile(theta_col, (b, 1))
        x_rep = np.repeat(xb, repeats=n_cols, axis=0)
        theta_t = torch.from_numpy(theta_tile).to(device)
        x_t = torch.from_numpy(x_rep).to(device)
        time_grid = torch.linspace(1.0, 0.0, int(ode_steps) + 1, device=device, dtype=theta_t.dtype)
        _, logp = solver.compute_likelihood(
            x_1=theta_t,
            log_p0=_log_std_normal,
            step_size=None,
            method="midpoint",
            time_grid=time_grid,
            exact_divergence=bool(exact_divergence),
            enable_grad=False,
            x_cond=x_t,
        )
        out[i0:i1, :] = logp.reshape(b, n_cols).detach().cpu().numpy().astype(np.float64)
    return out


def compute_h_from_c_matrix(c_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta_l = HMatrixEstimator.compute_delta_l(np.asarray(c_matrix, dtype=np.float64))
    h_sym = HMatrixEstimator.symmetrize(HMatrixEstimator.compute_h_directed(delta_l))
    return delta_l, h_sym


def train_local_flows(
    *,
    spec: ThetaBinSpec,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    labels_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    labels_val: np.ndarray,
    device: torch.device,
    normalize_local: bool,
    min_bin_count: int,
    x_dim: int,
    hidden_dim: int,
    depth: int,
    epochs: int,
    batch_size: int,
    lr: float,
    log_every: int,
    early_patience: int,
    early_min_delta: float,
    early_ema_alpha: float,
    restore_best: bool,
    scheduler_name: str,
    endpoint_loss_weight: float,
    endpoint_ode_steps: int,
    fm_t_eps: float,
) -> tuple[list[nn.Module], list[dict[str, Any]], np.ndarray, np.ndarray]:
    models: list[nn.Module] = []
    train_outs: list[dict[str, Any]] = []
    counts_train = np.bincount(np.asarray(labels_train, dtype=np.int64), minlength=spec.n_bins)
    counts_val = np.bincount(np.asarray(labels_val, dtype=np.int64), minlength=spec.n_bins)
    for k in range(spec.n_bins):
        if int(counts_train[k]) < int(min_bin_count):
            raise ValueError(f"binned_theta_flow bin {k} has {int(counts_train[k])} training rows; need >= {int(min_bin_count)}.")
        tr_mask = np.asarray(labels_train) == k
        va_mask = np.asarray(labels_val) == k
        if not np.any(va_mask):
            va_mask = tr_mask
            theta_val_k = theta_train
            x_val_k = x_train
            labels_val_k = labels_train
        else:
            theta_val_k = theta_val
            x_val_k = x_val
            labels_val_k = labels_val
        theta_tr = normalize_theta_in_bins(theta_train[tr_mask], labels_train[tr_mask], spec) if normalize_local else np.asarray(theta_train[tr_mask], dtype=np.float64).reshape(-1, 1)
        theta_va = normalize_theta_in_bins(theta_val_k[va_mask], labels_val_k[va_mask], spec) if normalize_local else np.asarray(theta_val_k[va_mask], dtype=np.float64).reshape(-1, 1)
        model = make_local_theta_flow_model(x_dim=x_dim, hidden_dim=hidden_dim, depth=depth).to(device)
        print(f"[binned_theta_flow] train local bin={k} train={int(counts_train[k])} val={int(counts_val[k])}", flush=True)
        train_out = train_conditional_theta_flow_model(
            model=model,
            theta_train=theta_tr,
            x_train=np.asarray(x_train[tr_mask], dtype=np.float64),
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(lr),
            device=device,
            log_every=max(1, int(log_every)),
            theta_val=theta_va,
            x_val=np.asarray(x_val_k[va_mask], dtype=np.float64),
            early_stopping_patience=int(early_patience),
            early_stopping_min_delta=float(early_min_delta),
            early_stopping_ema_alpha=float(early_ema_alpha),
            restore_best=bool(restore_best),
            scheduler_name=str(scheduler_name),
            endpoint_loss_weight=float(endpoint_loss_weight),
            endpoint_ode_steps=int(endpoint_ode_steps),
            fm_t_eps=float(fm_t_eps),
        )
        models.append(model)
        train_outs.append(train_out)
    return models, train_outs, counts_train.astype(np.int64), counts_val.astype(np.int64)


def train_smooth_weighted_local_flows(
    *,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    r_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    r_val: np.ndarray,
    device: torch.device,
    x_dim: int,
    hidden_dim: int,
    depth: int,
    epochs: int,
    batch_size: int,
    lr: float,
    log_every: int,
    early_patience: int,
    early_min_delta: float,
    early_ema_alpha: float,
    restore_best: bool,
    scheduler_name: str,
    endpoint_loss_weight: float,
    endpoint_ode_steps: int,
    fm_t_eps: float,
) -> tuple[list[nn.Module], list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Train K conditional theta-flow experts with FM loss weighted by soft RBF mass r_{ik}."""
    rtr = np.asarray(r_train, dtype=np.float64)
    rva = np.asarray(r_val, dtype=np.float64)
    if rtr.ndim != 2 or rva.ndim != 2:
        raise ValueError("r_train and r_val must be 2D (N, K).")
    if int(rtr.shape[1]) != int(rva.shape[1]):
        raise ValueError("r_train and r_val must have the same number of mixture columns K.")
    th_tr = np.asarray(theta_train, dtype=np.float64).reshape(-1)
    th_va = np.asarray(theta_val, dtype=np.float64).reshape(-1)
    if int(rtr.shape[0]) != int(th_tr.shape[0]):
        raise ValueError("r_train rows must match theta_train rows.")
    if int(rva.shape[0]) != int(th_va.shape[0]):
        raise ValueError("r_val rows must match theta_val rows.")
    x_tr = np.asarray(x_train, dtype=np.float64)
    x_va = np.asarray(x_val, dtype=np.float64)
    if int(x_tr.shape[0]) != int(th_tr.shape[0]):
        raise ValueError("x_train rows must match theta_train rows.")
    if int(x_va.shape[0]) != int(th_va.shape[0]):
        raise ValueError("x_val rows must match theta_val rows.")
    k_mix = int(rtr.shape[1])
    models: list[nn.Module] = []
    train_outs: list[dict[str, Any]] = []
    mass_train = np.sum(rtr, axis=0)
    mass_val = np.sum(rva, axis=0)
    for k in range(k_mix):
        w_tr = rtr[:, k]
        w_va = rva[:, k]
        if float(np.sum(w_tr)) < 1e-14:
            w_tr = np.ones_like(w_tr, dtype=np.float64) / float(len(w_tr))
        if float(np.sum(w_va)) < 1e-14:
            w_va = np.ones_like(w_va, dtype=np.float64) / float(len(w_va))
        model = make_local_theta_flow_model(x_dim=x_dim, hidden_dim=hidden_dim, depth=depth).to(device)
        print(
            f"[smooth_binned_theta_flow] train expert k={k} responsibility_mass_train={float(mass_train[k]):.6f} "
            f"mass_val={float(mass_val[k]):.6f}",
            flush=True,
        )
        train_out = train_weighted_conditional_theta_flow_model(
            model=model,
            theta_train=np.asarray(theta_train, dtype=np.float64).reshape(-1, 1),
            x_train=np.asarray(x_train, dtype=np.float64),
            weight_train=w_tr,
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(lr),
            device=device,
            log_every=max(1, int(log_every)),
            theta_val=np.asarray(theta_val, dtype=np.float64).reshape(-1, 1),
            x_val=np.asarray(x_val, dtype=np.float64),
            weight_val=w_va,
            early_stopping_patience=int(early_patience),
            early_stopping_min_delta=float(early_min_delta),
            early_stopping_ema_alpha=float(early_ema_alpha),
            restore_best=bool(restore_best),
            scheduler_name=str(scheduler_name),
            endpoint_loss_weight=float(endpoint_loss_weight),
            endpoint_ode_steps=int(endpoint_ode_steps),
            fm_t_eps=float(fm_t_eps),
        )
        models.append(model)
        train_outs.append(train_out)
    return models, train_outs, mass_train.astype(np.float64), mass_val.astype(np.float64)
