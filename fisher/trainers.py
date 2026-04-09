from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fisher.models import (
    ConditionalScore1D,
    ConditionalScore1DFiLMPerLayer,
    ConditionalThetaFlowVelocity,
    ConditionalXFlowVelocity,
    ConditionalXFlowVelocityFiLMPerLayer,
    ConditionalXScore,
    LocalDecoderLogit,
    PriorThetaFlowVelocity,
    PriorScore1D,
    PriorScore1DFiLMPerLayer,
    UnconditionalXFlowVelocity,
    UnconditionalXFlowVelocityFiLMPerLayer,
    UnconditionalXScore,
)


def to_score_loader(theta: np.ndarray, x: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    t = torch.from_numpy(theta.astype(np.float32))
    xx = torch.from_numpy(x.astype(np.float32))
    ds = TensorDataset(t, xx)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def to_x_loader(x: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    xx = torch.from_numpy(x.astype(np.float32))
    ds = TensorDataset(xx)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def to_decoder_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    xt = torch.from_numpy(x.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.float32)).reshape(-1, 1)
    ds = TensorDataset(xt, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def to_prior_loader(theta: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    t = torch.from_numpy(theta.astype(np.float32))
    ds = TensorDataset(t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _ema_update_val_monitor(prev_ema: float | None, mean_val_loss: float, ema_alpha: float) -> float:
    """Exponential moving average of per-epoch validation loss for early-stopping monitor."""
    if prev_ema is None:
        return float(mean_val_loss)
    return float(ema_alpha * mean_val_loss + (1.0 - ema_alpha) * prev_ema)


def geometric_sigma_schedule(sigma_min: float, sigma_max: float, n_levels: int, descending: bool = True) -> np.ndarray:
    if n_levels < 2:
        raise ValueError("n_levels must be >= 2 for geometric sigma schedule.")
    if sigma_min <= 0.0 or sigma_max <= 0.0:
        raise ValueError("sigma_min and sigma_max must be positive.")
    if sigma_min > sigma_max:
        sigma_min, sigma_max = sigma_max, sigma_min
    sigmas = np.geomspace(sigma_min, sigma_max, num=n_levels, endpoint=True).astype(np.float64)
    return sigmas[::-1] if descending else sigmas


def sample_continuous_geometric_sigmas(
    batch_size: int,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
) -> torch.Tensor:
    if sigma_min <= 0.0 or sigma_max <= 0.0:
        raise ValueError("sigma_min and sigma_max must be positive.")
    lo = min(float(sigma_min), float(sigma_max))
    hi = max(float(sigma_min), float(sigma_max))
    u = torch.rand((batch_size, 1), device=device)
    log_sigma = np.log(lo) + u * (np.log(hi) - np.log(lo))
    return torch.exp(log_sigma)


def train_score_model(
    model: ConditionalScore1D | ConditionalScore1DFiLMPerLayer,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    sigma_values: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    log_every: int,
    theta_val: np.ndarray | None = None,
    x_val: np.ndarray | None = None,
    early_stopping_patience: int = 30,
    early_stopping_min_delta: float = 1e-4,
    early_stopping_ema_alpha: float = 0.05,
    restore_best: bool = True,
) -> dict[str, float | int | bool | list[float]]:
    loader = to_score_loader(theta_train, x_train, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sigma_values_t = torch.from_numpy(sigma_values.astype(np.float32)).to(device)
    has_val = theta_val is not None and x_val is not None and len(theta_val) > 0
    val_loader = (
        to_score_loader(theta_val, x_val, batch_size=batch_size, shuffle=False)
        if has_val
        else None
    )
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_monitor_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    stopped_early = False
    stopped_epoch = epochs
    val_ema: float | None = None
    alpha = float(early_stopping_ema_alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("early_stopping_ema_alpha must be in (0, 1].")

    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        model.train()
        for tb, xb in loader:
            tb = tb.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            sigma_idx = torch.randint(low=0, high=sigma_values_t.numel(), size=(tb.shape[0],), device=tb.device)
            sigma = sigma_values_t[sigma_idx].unsqueeze(-1)
            eps = torch.randn_like(tb)
            theta_tilde = tb + sigma * eps
            target = -(theta_tilde - tb) / (sigma**2)
            pred = model(theta_tilde, xb, sigma)
            loss = torch.mean((pred - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        mean_train_loss = float(np.mean(epoch_losses))
        train_losses.append(mean_train_loss)

        mean_val_loss = float("nan")
        if has_val and val_loader is not None:
            model.eval()
            val_epoch_losses: list[float] = []
            with torch.no_grad():
                for tb, xb in val_loader:
                    tb = tb.to(device, non_blocking=True)
                    xb = xb.to(device, non_blocking=True)
                    sigma_idx = torch.randint(low=0, high=sigma_values_t.numel(), size=(tb.shape[0],), device=tb.device)
                    sigma = sigma_values_t[sigma_idx].unsqueeze(-1)
                    eps = torch.randn_like(tb)
                    theta_tilde = tb + sigma * eps
                    target = -(theta_tilde - tb) / (sigma**2)
                    pred = model(theta_tilde, xb, sigma)
                    val_loss = torch.mean((pred - target) ** 2)
                    val_epoch_losses.append(float(val_loss.item()))
            mean_val_loss = float(np.mean(val_epoch_losses))
            val_ema = _ema_update_val_monitor(val_ema, mean_val_loss, alpha)
            smooth_val_loss = val_ema
            if smooth_val_loss < (best_val_loss - early_stopping_min_delta):
                best_val_loss = smooth_val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            val_monitor_losses.append(smooth_val_loss)
        else:
            val_monitor_losses.append(float("nan"))
        val_losses.append(mean_val_loss)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            if has_val:
                print(
                    f"[epoch {epoch:4d}/{epochs}] train_loss={mean_train_loss:.6f} "
                    f"val_loss={mean_val_loss:.6f} val_smooth={val_monitor_losses[-1]:.6f} "
                    f"best_smooth={best_val_loss:.6f} best_epoch={best_epoch}"
                )
            else:
                print(f"[epoch {epoch:4d}/{epochs}] train_loss={mean_train_loss:.6f}")

        if has_val and patience_counter >= early_stopping_patience:
            stopped_early = True
            stopped_epoch = epoch
            print(
                f"[early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_smooth={best_val_loss:.6f} patience={early_stopping_patience}"
            )
            break

    if has_val and restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[restore-best] restored epoch={best_epoch} val_smooth={best_val_loss:.6f}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_monitor_losses": val_monitor_losses,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
    }


def train_score_model_ncsm_continuous(
    model: ConditionalScore1D | ConditionalScore1DFiLMPerLayer,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    sigma_min: float,
    sigma_max: float,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    log_every: int,
    theta_val: np.ndarray | None = None,
    x_val: np.ndarray | None = None,
    early_stopping_patience: int = 30,
    early_stopping_min_delta: float = 1e-4,
    early_stopping_ema_alpha: float = 0.05,
    restore_best: bool = True,
) -> dict[str, float | int | bool | list[float]]:
    loader = to_score_loader(theta_train, x_train, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    has_val = theta_val is not None and x_val is not None and len(theta_val) > 0
    val_loader = (
        to_score_loader(theta_val, x_val, batch_size=batch_size, shuffle=False)
        if has_val
        else None
    )
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_monitor_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    stopped_early = False
    stopped_epoch = epochs
    val_ema: float | None = None
    alpha = float(early_stopping_ema_alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("early_stopping_ema_alpha must be in (0, 1].")

    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        model.train()
        for tb, xb in loader:
            tb = tb.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            sigma = sample_continuous_geometric_sigmas(
                batch_size=tb.shape[0],
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                device=tb.device,
            )
            eps = torch.randn_like(tb)
            theta_tilde = tb + sigma * eps
            pred = model(theta_tilde, xb, sigma)
            loss = torch.mean((sigma * pred + eps) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        mean_train_loss = float(np.mean(epoch_losses))
        train_losses.append(mean_train_loss)

        mean_val_loss = float("nan")
        if has_val and val_loader is not None:
            model.eval()
            val_epoch_losses: list[float] = []
            with torch.no_grad():
                for tb, xb in val_loader:
                    tb = tb.to(device, non_blocking=True)
                    xb = xb.to(device, non_blocking=True)
                    sigma = sample_continuous_geometric_sigmas(
                        batch_size=tb.shape[0],
                        sigma_min=sigma_min,
                        sigma_max=sigma_max,
                        device=tb.device,
                    )
                    eps = torch.randn_like(tb)
                    theta_tilde = tb + sigma * eps
                    pred = model(theta_tilde, xb, sigma)
                    val_loss = torch.mean((sigma * pred + eps) ** 2)
                    val_epoch_losses.append(float(val_loss.item()))
            mean_val_loss = float(np.mean(val_epoch_losses))
            val_ema = _ema_update_val_monitor(val_ema, mean_val_loss, alpha)
            smooth_val_loss = val_ema
            if smooth_val_loss < (best_val_loss - early_stopping_min_delta):
                best_val_loss = smooth_val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            val_monitor_losses.append(smooth_val_loss)
        else:
            val_monitor_losses.append(float("nan"))
        val_losses.append(mean_val_loss)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            if has_val:
                print(
                    f"[epoch {epoch:4d}/{epochs}] ncsm_train={mean_train_loss:.6f} "
                    f"val_loss={mean_val_loss:.6f} val_smooth={val_monitor_losses[-1]:.6f} "
                    f"best_smooth={best_val_loss:.6f} best_epoch={best_epoch}"
                )
            else:
                print(f"[epoch {epoch:4d}/{epochs}] ncsm_loss={mean_train_loss:.6f}")

        if has_val and patience_counter >= early_stopping_patience:
            stopped_early = True
            stopped_epoch = epoch
            print(
                f"[early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_smooth={best_val_loss:.6f} patience={early_stopping_patience}"
            )
            break

    if has_val and restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[restore-best] restored epoch={best_epoch} val_smooth={best_val_loss:.6f}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_monitor_losses": val_monitor_losses,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
    }


def train_conditional_x_score_model_ncsm_continuous(
    model: ConditionalXScore,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    sigma_min: float,
    sigma_max: float,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    log_every: int,
    theta_val: np.ndarray | None = None,
    x_val: np.ndarray | None = None,
    early_stopping_patience: int = 30,
    early_stopping_min_delta: float = 1e-4,
    early_stopping_ema_alpha: float = 0.05,
    restore_best: bool = True,
) -> dict[str, float | int | bool | list[float]]:
    loader = to_score_loader(theta_train, x_train, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    has_val = theta_val is not None and x_val is not None and len(theta_val) > 0
    val_loader = (
        to_score_loader(theta_val, x_val, batch_size=batch_size, shuffle=False)
        if has_val
        else None
    )
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_monitor_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    stopped_early = False
    stopped_epoch = epochs
    val_ema: float | None = None
    alpha = float(early_stopping_ema_alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("early_stopping_ema_alpha must be in (0, 1].")

    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        model.train()
        for tb, xb in loader:
            tb = tb.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            sigma = sample_continuous_geometric_sigmas(
                batch_size=tb.shape[0],
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                device=tb.device,
            )
            eps = torch.randn_like(xb)
            x_tilde = xb + sigma * eps
            pred = model(x_tilde, tb, sigma)
            loss = torch.mean((sigma * pred + eps) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        mean_train_loss = float(np.mean(epoch_losses))
        train_losses.append(mean_train_loss)

        mean_val_loss = float("nan")
        if has_val and val_loader is not None:
            model.eval()
            val_epoch_losses: list[float] = []
            with torch.no_grad():
                for tb, xb in val_loader:
                    tb = tb.to(device, non_blocking=True)
                    xb = xb.to(device, non_blocking=True)
                    sigma = sample_continuous_geometric_sigmas(
                        batch_size=tb.shape[0],
                        sigma_min=sigma_min,
                        sigma_max=sigma_max,
                        device=tb.device,
                    )
                    eps = torch.randn_like(xb)
                    x_tilde = xb + sigma * eps
                    pred = model(x_tilde, tb, sigma)
                    val_loss = torch.mean((sigma * pred + eps) ** 2)
                    val_epoch_losses.append(float(val_loss.item()))
            mean_val_loss = float(np.mean(val_epoch_losses))
            val_ema = _ema_update_val_monitor(val_ema, mean_val_loss, alpha)
            smooth_val_loss = val_ema
            if smooth_val_loss < (best_val_loss - early_stopping_min_delta):
                best_val_loss = smooth_val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            val_monitor_losses.append(smooth_val_loss)
        else:
            val_monitor_losses.append(float("nan"))
        val_losses.append(mean_val_loss)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            if has_val:
                print(
                    f"[epoch {epoch:4d}/{epochs}] x_ncsm_train={mean_train_loss:.6f} "
                    f"val_loss={mean_val_loss:.6f} val_smooth={val_monitor_losses[-1]:.6f} "
                    f"best_smooth={best_val_loss:.6f} best_epoch={best_epoch}"
                )
            else:
                print(f"[epoch {epoch:4d}/{epochs}] x_ncsm_loss={mean_train_loss:.6f}")

        if has_val and patience_counter >= early_stopping_patience:
            stopped_early = True
            stopped_epoch = epoch
            print(
                f"[early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_smooth={best_val_loss:.6f} patience={early_stopping_patience}"
            )
            break

    if has_val and restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[restore-best] restored epoch={best_epoch} val_smooth={best_val_loss:.6f}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_monitor_losses": val_monitor_losses,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
    }


def train_unconditional_x_score_model_ncsm_continuous(
    model: UnconditionalXScore,
    x_train: np.ndarray,
    sigma_min: float,
    sigma_max: float,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    log_every: int,
    x_val: np.ndarray | None = None,
    early_stopping_patience: int = 30,
    early_stopping_min_delta: float = 1e-4,
    early_stopping_ema_alpha: float = 0.05,
    restore_best: bool = True,
) -> dict[str, float | int | bool | list[float]]:
    """NCSM denoising score matching on x only (no theta in the network)."""
    loader = to_x_loader(x_train, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    has_val = x_val is not None and len(x_val) > 0
    val_loader = to_x_loader(x_val, batch_size=batch_size, shuffle=False) if has_val else None
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_monitor_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    stopped_early = False
    stopped_epoch = epochs
    val_ema: float | None = None
    alpha = float(early_stopping_ema_alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("early_stopping_ema_alpha must be in (0, 1].")

    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        model.train()
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            sigma = sample_continuous_geometric_sigmas(
                batch_size=xb.shape[0],
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                device=xb.device,
            )
            eps = torch.randn_like(xb)
            x_tilde = xb + sigma * eps
            pred = model(x_tilde, sigma)
            loss = torch.mean((sigma * pred + eps) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        mean_train_loss = float(np.mean(epoch_losses))
        train_losses.append(mean_train_loss)

        mean_val_loss = float("nan")
        if has_val and val_loader is not None:
            model.eval()
            val_epoch_losses: list[float] = []
            with torch.no_grad():
                for (xb,) in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    sigma = sample_continuous_geometric_sigmas(
                        batch_size=xb.shape[0],
                        sigma_min=sigma_min,
                        sigma_max=sigma_max,
                        device=xb.device,
                    )
                    eps = torch.randn_like(xb)
                    x_tilde = xb + sigma * eps
                    pred = model(x_tilde, sigma)
                    val_loss = torch.mean((sigma * pred + eps) ** 2)
                    val_epoch_losses.append(float(val_loss.item()))
            mean_val_loss = float(np.mean(val_epoch_losses))
            val_ema = _ema_update_val_monitor(val_ema, mean_val_loss, alpha)
            smooth_val_loss = val_ema
            if smooth_val_loss < (best_val_loss - early_stopping_min_delta):
                best_val_loss = smooth_val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            val_monitor_losses.append(smooth_val_loss)
        else:
            val_monitor_losses.append(float("nan"))
        val_losses.append(mean_val_loss)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            if has_val:
                print(
                    f"[epoch {epoch:4d}/{epochs}] x_ncsm_train={mean_train_loss:.6f} "
                    f"val_loss={mean_val_loss:.6f} val_smooth={val_monitor_losses[-1]:.6f} "
                    f"best_smooth={best_val_loss:.6f} best_epoch={best_epoch}"
                )
            else:
                print(f"[epoch {epoch:4d}/{epochs}] x_ncsm_loss={mean_train_loss:.6f}")

        if has_val and patience_counter >= early_stopping_patience:
            stopped_early = True
            stopped_epoch = epoch
            print(
                f"[early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_smooth={best_val_loss:.6f} patience={early_stopping_patience}"
            )
            break

    if has_val and restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[restore-best] restored epoch={best_epoch} val_smooth={best_val_loss:.6f}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_monitor_losses": val_monitor_losses,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
    }


def _make_flow_matching_path(scheduler_name: str):
    name = str(scheduler_name).strip().lower()
    try:
        from flow_matching.path import AffineProbPath
        from flow_matching.path.scheduler import CosineScheduler, LinearVPScheduler, VPScheduler
    except ImportError as e:
        raise ImportError(
            "Flow matching training requires the `flow_matching` package. "
            "Install it in your environment before using --method flow."
        ) from e

    scheduler_lookup = {
        "cosine": CosineScheduler,
        "vp": VPScheduler,
        "linear_vp": LinearVPScheduler,
    }
    if name not in scheduler_lookup:
        supported = ", ".join(sorted(scheduler_lookup.keys()))
        raise ValueError(f"Unknown flow scheduler '{scheduler_name}'. Supported: {supported}.")
    return AffineProbPath(scheduler=scheduler_lookup[name]())


def train_conditional_x_flow_model(
    model: ConditionalXFlowVelocity | ConditionalXFlowVelocityFiLMPerLayer,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    log_every: int,
    theta_val: np.ndarray | None = None,
    x_val: np.ndarray | None = None,
    early_stopping_patience: int = 30,
    early_stopping_min_delta: float = 1e-4,
    early_stopping_ema_alpha: float = 0.05,
    restore_best: bool = True,
    scheduler_name: str = "cosine",
) -> dict[str, float | int | bool | list[float]]:
    path = _make_flow_matching_path(scheduler_name=scheduler_name)
    loader = to_score_loader(theta_train, x_train, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    has_val = theta_val is not None and x_val is not None and len(theta_val) > 0
    val_loader = (
        to_score_loader(theta_val, x_val, batch_size=batch_size, shuffle=False)
        if has_val
        else None
    )
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_monitor_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    stopped_early = False
    stopped_epoch = epochs
    val_ema: float | None = None
    alpha = float(early_stopping_ema_alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("early_stopping_ema_alpha must be in (0, 1].")

    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        model.train()
        for tb, xb in loader:
            tb = tb.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            x0 = torch.randn_like(xb)
            t = torch.rand(xb.shape[0], device=xb.device)
            path_sample = path.sample(t=t, x_0=x0, x_1=xb)
            pred = model(path_sample.x_t, tb, path_sample.t)
            loss = torch.mean((pred - path_sample.dx_t) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        mean_train_loss = float(np.mean(epoch_losses))
        train_losses.append(mean_train_loss)

        mean_val_loss = float("nan")
        if has_val and val_loader is not None:
            model.eval()
            val_epoch_losses: list[float] = []
            with torch.no_grad():
                for tb, xb in val_loader:
                    tb = tb.to(device, non_blocking=True)
                    xb = xb.to(device, non_blocking=True)
                    x0 = torch.randn_like(xb)
                    t = torch.rand(xb.shape[0], device=xb.device)
                    path_sample = path.sample(t=t, x_0=x0, x_1=xb)
                    pred = model(path_sample.x_t, tb, path_sample.t)
                    val_loss = torch.mean((pred - path_sample.dx_t) ** 2)
                    val_epoch_losses.append(float(val_loss.item()))
            mean_val_loss = float(np.mean(val_epoch_losses))
            val_ema = _ema_update_val_monitor(val_ema, mean_val_loss, alpha)
            smooth_val_loss = val_ema
            if smooth_val_loss < (best_val_loss - early_stopping_min_delta):
                best_val_loss = smooth_val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            val_monitor_losses.append(smooth_val_loss)
        else:
            val_monitor_losses.append(float("nan"))
        val_losses.append(mean_val_loss)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            if has_val:
                print(
                    f"[epoch {epoch:4d}/{epochs}] flow_train={mean_train_loss:.6f} "
                    f"val_loss={mean_val_loss:.6f} val_smooth={val_monitor_losses[-1]:.6f} "
                    f"best_smooth={best_val_loss:.6f} best_epoch={best_epoch}"
                )
            else:
                print(f"[epoch {epoch:4d}/{epochs}] flow_loss={mean_train_loss:.6f}")

        if has_val and patience_counter >= early_stopping_patience:
            stopped_early = True
            stopped_epoch = epoch
            print(
                f"[early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_smooth={best_val_loss:.6f} patience={early_stopping_patience}"
            )
            break

    if has_val and restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[restore-best] restored epoch={best_epoch} val_smooth={best_val_loss:.6f}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_monitor_losses": val_monitor_losses,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
    }


def train_conditional_theta_flow_model(
    model: ConditionalThetaFlowVelocity,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    log_every: int,
    theta_val: np.ndarray | None = None,
    x_val: np.ndarray | None = None,
    early_stopping_patience: int = 30,
    early_stopping_min_delta: float = 1e-4,
    early_stopping_ema_alpha: float = 0.05,
    restore_best: bool = True,
    scheduler_name: str = "cosine",
) -> dict[str, float | int | bool | list[float]]:
    path = _make_flow_matching_path(scheduler_name=scheduler_name)
    loader = to_score_loader(theta_train, x_train, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    has_val = theta_val is not None and x_val is not None and len(theta_val) > 0
    val_loader = (
        to_score_loader(theta_val, x_val, batch_size=batch_size, shuffle=False)
        if has_val
        else None
    )
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_monitor_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    stopped_early = False
    stopped_epoch = epochs
    val_ema: float | None = None
    alpha = float(early_stopping_ema_alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("early_stopping_ema_alpha must be in (0, 1].")

    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        model.train()
        for tb, xb in loader:
            tb = tb.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            t = torch.rand(tb.shape[0], device=tb.device)
            theta0 = torch.randn_like(tb)
            path_sample = path.sample(t=t, x_0=theta0, x_1=tb)
            pred = model(path_sample.x_t, xb, path_sample.t)
            loss = torch.mean((pred - path_sample.dx_t) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        mean_train_loss = float(np.mean(epoch_losses))
        train_losses.append(mean_train_loss)

        mean_val_loss = float("nan")
        if has_val and val_loader is not None:
            model.eval()
            val_epoch_losses: list[float] = []
            with torch.no_grad():
                for tb, xb in val_loader:
                    tb = tb.to(device, non_blocking=True)
                    xb = xb.to(device, non_blocking=True)
                    t = torch.rand(tb.shape[0], device=tb.device)
                    theta0 = torch.randn_like(tb)
                    path_sample = path.sample(t=t, x_0=theta0, x_1=tb)
                    pred = model(path_sample.x_t, xb, path_sample.t)
                    val_loss = torch.mean((pred - path_sample.dx_t) ** 2)
                    val_epoch_losses.append(float(val_loss.item()))
            mean_val_loss = float(np.mean(val_epoch_losses))
            val_ema = _ema_update_val_monitor(val_ema, mean_val_loss, alpha)
            smooth_val_loss = val_ema
            if smooth_val_loss < (best_val_loss - early_stopping_min_delta):
                best_val_loss = smooth_val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            val_monitor_losses.append(smooth_val_loss)
        else:
            val_monitor_losses.append(float("nan"))
        val_losses.append(mean_val_loss)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            if has_val:
                print(
                    f"[epoch {epoch:4d}/{epochs}] theta_flow_train={mean_train_loss:.6f} "
                    f"val_loss={mean_val_loss:.6f} val_smooth={val_monitor_losses[-1]:.6f} "
                    f"best_smooth={best_val_loss:.6f} best_epoch={best_epoch}"
                )
            else:
                print(f"[epoch {epoch:4d}/{epochs}] theta_flow_loss={mean_train_loss:.6f}")

        if has_val and patience_counter >= early_stopping_patience:
            stopped_early = True
            stopped_epoch = epoch
            print(
                f"[early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_smooth={best_val_loss:.6f} patience={early_stopping_patience}"
            )
            break

    if has_val and restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[restore-best] restored epoch={best_epoch} val_smooth={best_val_loss:.6f}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_monitor_losses": val_monitor_losses,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
    }


def train_prior_theta_flow_model(
    model: PriorThetaFlowVelocity,
    theta_train: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    log_every: int,
    theta_val: np.ndarray | None = None,
    early_stopping_patience: int = 30,
    early_stopping_min_delta: float = 1e-4,
    early_stopping_ema_alpha: float = 0.05,
    restore_best: bool = True,
    scheduler_name: str = "cosine",
) -> dict[str, float | int | bool | list[float]]:
    path = _make_flow_matching_path(scheduler_name=scheduler_name)
    loader = to_prior_loader(theta_train, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    has_val = theta_val is not None and len(theta_val) > 0
    val_loader = to_prior_loader(theta_val, batch_size=batch_size, shuffle=False) if has_val else None
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_monitor_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    stopped_early = False
    stopped_epoch = epochs
    val_ema: float | None = None
    alpha = float(early_stopping_ema_alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("early_stopping_ema_alpha must be in (0, 1].")

    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        model.train()
        for (tb,) in loader:
            tb = tb.to(device, non_blocking=True)
            t = torch.rand(tb.shape[0], device=tb.device)
            theta0 = torch.randn_like(tb)
            path_sample = path.sample(t=t, x_0=theta0, x_1=tb)
            pred = model(path_sample.x_t, path_sample.t)
            loss = torch.mean((pred - path_sample.dx_t) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        mean_train_loss = float(np.mean(epoch_losses))
        train_losses.append(mean_train_loss)

        mean_val_loss = float("nan")
        if has_val and val_loader is not None:
            model.eval()
            val_epoch_losses: list[float] = []
            with torch.no_grad():
                for (tb,) in val_loader:
                    tb = tb.to(device, non_blocking=True)
                    t = torch.rand(tb.shape[0], device=tb.device)
                    theta0 = torch.randn_like(tb)
                    path_sample = path.sample(t=t, x_0=theta0, x_1=tb)
                    pred = model(path_sample.x_t, path_sample.t)
                    val_loss = torch.mean((pred - path_sample.dx_t) ** 2)
                    val_epoch_losses.append(float(val_loss.item()))
            mean_val_loss = float(np.mean(val_epoch_losses))
            val_ema = _ema_update_val_monitor(val_ema, mean_val_loss, alpha)
            smooth_val_loss = val_ema
            if smooth_val_loss < (best_val_loss - early_stopping_min_delta):
                best_val_loss = smooth_val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            val_monitor_losses.append(smooth_val_loss)
        else:
            val_monitor_losses.append(float("nan"))
        val_losses.append(mean_val_loss)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            if has_val:
                print(
                    f"[prior flow {epoch:4d}/{epochs}] train={mean_train_loss:.6f} "
                    f"val_loss={mean_val_loss:.6f} val_smooth={val_monitor_losses[-1]:.6f} "
                    f"best_smooth={best_val_loss:.6f} best_epoch={best_epoch}"
                )
            else:
                print(f"[prior flow {epoch:4d}/{epochs}] train={mean_train_loss:.6f}")

        if has_val and patience_counter >= early_stopping_patience:
            stopped_early = True
            stopped_epoch = epoch
            print(
                f"[prior early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_smooth={best_val_loss:.6f} patience={early_stopping_patience}"
            )
            break

    if has_val and restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[prior restore-best] restored epoch={best_epoch} val_smooth={best_val_loss:.6f}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_monitor_losses": val_monitor_losses,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
    }


def train_unconditional_x_flow_model(
    model: UnconditionalXFlowVelocity | UnconditionalXFlowVelocityFiLMPerLayer,
    x_train: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    log_every: int,
    x_val: np.ndarray | None = None,
    early_stopping_patience: int = 30,
    early_stopping_min_delta: float = 1e-4,
    early_stopping_ema_alpha: float = 0.05,
    restore_best: bool = True,
    scheduler_name: str = "cosine",
) -> dict[str, float | int | bool | list[float]]:
    """Flow matching on x only: learn v(x_t, t) with no theta in the network."""
    path = _make_flow_matching_path(scheduler_name=scheduler_name)
    loader = to_x_loader(x_train, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    has_val = x_val is not None and len(x_val) > 0
    val_loader = to_x_loader(x_val, batch_size=batch_size, shuffle=False) if has_val else None
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_monitor_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    stopped_early = False
    stopped_epoch = epochs
    val_ema: float | None = None
    alpha = float(early_stopping_ema_alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("early_stopping_ema_alpha must be in (0, 1].")

    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        model.train()
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            x0 = torch.randn_like(xb)
            t = torch.rand(xb.shape[0], device=xb.device)
            path_sample = path.sample(t=t, x_0=x0, x_1=xb)
            pred = model(path_sample.x_t, path_sample.t)
            loss = torch.mean((pred - path_sample.dx_t) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        mean_train_loss = float(np.mean(epoch_losses))
        train_losses.append(mean_train_loss)

        mean_val_loss = float("nan")
        if has_val and val_loader is not None:
            model.eval()
            val_epoch_losses: list[float] = []
            with torch.no_grad():
                for (xb,) in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    x0 = torch.randn_like(xb)
                    t = torch.rand(xb.shape[0], device=xb.device)
                    path_sample = path.sample(t=t, x_0=x0, x_1=xb)
                    pred = model(path_sample.x_t, path_sample.t)
                    val_loss = torch.mean((pred - path_sample.dx_t) ** 2)
                    val_epoch_losses.append(float(val_loss.item()))
            mean_val_loss = float(np.mean(val_epoch_losses))
            val_ema = _ema_update_val_monitor(val_ema, mean_val_loss, alpha)
            smooth_val_loss = val_ema
            if smooth_val_loss < (best_val_loss - early_stopping_min_delta):
                best_val_loss = smooth_val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            val_monitor_losses.append(smooth_val_loss)
        else:
            val_monitor_losses.append(float("nan"))
        val_losses.append(mean_val_loss)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            if has_val:
                print(
                    f"[epoch {epoch:4d}/{epochs}] flow_train={mean_train_loss:.6f} "
                    f"val_loss={mean_val_loss:.6f} val_smooth={val_monitor_losses[-1]:.6f} "
                    f"best_smooth={best_val_loss:.6f} best_epoch={best_epoch}"
                )
            else:
                print(f"[epoch {epoch:4d}/{epochs}] flow_loss={mean_train_loss:.6f}")

        if has_val and patience_counter >= early_stopping_patience:
            stopped_early = True
            stopped_epoch = epoch
            print(
                f"[early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_smooth={best_val_loss:.6f} patience={early_stopping_patience}"
            )
            break

    if has_val and restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[restore-best] restored epoch={best_epoch} val_smooth={best_val_loss:.6f}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_monitor_losses": val_monitor_losses,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
    }


def train_prior_score_model(
    model: PriorScore1D | PriorScore1DFiLMPerLayer,
    theta_train: np.ndarray,
    sigma_values: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    log_every: int,
    theta_val: np.ndarray | None = None,
    early_stopping_patience: int = 30,
    early_stopping_min_delta: float = 1e-4,
    early_stopping_ema_alpha: float = 0.05,
    restore_best: bool = True,
) -> dict[str, float | int | bool | list[float]]:
    loader = to_prior_loader(theta_train, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sigma_values_t = torch.from_numpy(sigma_values.astype(np.float32)).to(device)
    has_val = theta_val is not None and len(theta_val) > 0
    val_loader = to_prior_loader(theta_val, batch_size=batch_size, shuffle=False) if has_val else None
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_monitor_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    stopped_early = False
    stopped_epoch = epochs
    val_ema: float | None = None
    alpha = float(early_stopping_ema_alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("early_stopping_ema_alpha must be in (0, 1].")

    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        model.train()
        for (tb,) in loader:
            tb = tb.to(device, non_blocking=True)
            sigma_idx = torch.randint(low=0, high=sigma_values_t.numel(), size=(tb.shape[0],), device=tb.device)
            sigma = sigma_values_t[sigma_idx].unsqueeze(-1)
            eps = torch.randn_like(tb)
            theta_tilde = tb + sigma * eps
            target = -(theta_tilde - tb) / (sigma**2)
            pred = model(theta_tilde, sigma)
            loss = torch.mean((pred - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        mean_train_loss = float(np.mean(epoch_losses))
        train_losses.append(mean_train_loss)

        mean_val_loss = float("nan")
        if has_val and val_loader is not None:
            model.eval()
            val_epoch_losses: list[float] = []
            with torch.no_grad():
                for (tb,) in val_loader:
                    tb = tb.to(device, non_blocking=True)
                    sigma_idx = torch.randint(low=0, high=sigma_values_t.numel(), size=(tb.shape[0],), device=tb.device)
                    sigma = sigma_values_t[sigma_idx].unsqueeze(-1)
                    eps = torch.randn_like(tb)
                    theta_tilde = tb + sigma * eps
                    target = -(theta_tilde - tb) / (sigma**2)
                    pred = model(theta_tilde, sigma)
                    val_loss = torch.mean((pred - target) ** 2)
                    val_epoch_losses.append(float(val_loss.item()))
            mean_val_loss = float(np.mean(val_epoch_losses))
            val_ema = _ema_update_val_monitor(val_ema, mean_val_loss, alpha)
            smooth_val_loss = val_ema
            if smooth_val_loss < (best_val_loss - early_stopping_min_delta):
                best_val_loss = smooth_val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            val_monitor_losses.append(smooth_val_loss)
        else:
            val_monitor_losses.append(float("nan"))
        val_losses.append(mean_val_loss)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            if has_val:
                print(
                    f"[prior epoch {epoch:4d}/{epochs}] train_loss={mean_train_loss:.6f} "
                    f"val_loss={mean_val_loss:.6f} val_smooth={val_monitor_losses[-1]:.6f} "
                    f"best_smooth={best_val_loss:.6f} best_epoch={best_epoch}"
                )
            else:
                print(f"[prior epoch {epoch:4d}/{epochs}] train_loss={mean_train_loss:.6f}")

        if has_val and patience_counter >= early_stopping_patience:
            stopped_early = True
            stopped_epoch = epoch
            print(
                f"[prior early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_smooth={best_val_loss:.6f} patience={early_stopping_patience}"
            )
            break

    if has_val and restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[prior restore-best] restored epoch={best_epoch} val_smooth={best_val_loss:.6f}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_monitor_losses": val_monitor_losses,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
    }


def train_prior_score_model_ncsm_continuous(
    model: PriorScore1D | PriorScore1DFiLMPerLayer,
    theta_train: np.ndarray,
    sigma_min: float,
    sigma_max: float,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    log_every: int,
    theta_val: np.ndarray | None = None,
    early_stopping_patience: int = 30,
    early_stopping_min_delta: float = 1e-4,
    early_stopping_ema_alpha: float = 0.05,
    restore_best: bool = True,
) -> dict[str, float | int | bool | list[float]]:
    loader = to_prior_loader(theta_train, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    has_val = theta_val is not None and len(theta_val) > 0
    val_loader = to_prior_loader(theta_val, batch_size=batch_size, shuffle=False) if has_val else None
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_monitor_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    stopped_early = False
    stopped_epoch = epochs
    val_ema: float | None = None
    alpha = float(early_stopping_ema_alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("early_stopping_ema_alpha must be in (0, 1].")

    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        model.train()
        for (tb,) in loader:
            tb = tb.to(device, non_blocking=True)
            sigma = sample_continuous_geometric_sigmas(
                batch_size=tb.shape[0],
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                device=tb.device,
            )
            eps = torch.randn_like(tb)
            theta_tilde = tb + sigma * eps
            pred = model(theta_tilde, sigma)
            loss = torch.mean((sigma * pred + eps) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        mean_train_loss = float(np.mean(epoch_losses))
        train_losses.append(mean_train_loss)

        mean_val_loss = float("nan")
        if has_val and val_loader is not None:
            model.eval()
            val_epoch_losses: list[float] = []
            with torch.no_grad():
                for (tb,) in val_loader:
                    tb = tb.to(device, non_blocking=True)
                    sigma = sample_continuous_geometric_sigmas(
                        batch_size=tb.shape[0],
                        sigma_min=sigma_min,
                        sigma_max=sigma_max,
                        device=tb.device,
                    )
                    eps = torch.randn_like(tb)
                    theta_tilde = tb + sigma * eps
                    pred = model(theta_tilde, sigma)
                    val_loss = torch.mean((sigma * pred + eps) ** 2)
                    val_epoch_losses.append(float(val_loss.item()))
            mean_val_loss = float(np.mean(val_epoch_losses))
            val_ema = _ema_update_val_monitor(val_ema, mean_val_loss, alpha)
            smooth_val_loss = val_ema
            if smooth_val_loss < (best_val_loss - early_stopping_min_delta):
                best_val_loss = smooth_val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            val_monitor_losses.append(smooth_val_loss)
        else:
            val_monitor_losses.append(float("nan"))
        val_losses.append(mean_val_loss)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            if has_val:
                print(
                    f"[prior ncsm {epoch:4d}/{epochs}] train={mean_train_loss:.6f} "
                    f"val_loss={mean_val_loss:.6f} val_smooth={val_monitor_losses[-1]:.6f} "
                    f"best_smooth={best_val_loss:.6f} best_epoch={best_epoch}"
                )
            else:
                print(f"[prior ncsm {epoch:4d}/{epochs}] train={mean_train_loss:.6f}")

        if has_val and patience_counter >= early_stopping_patience:
            stopped_early = True
            stopped_epoch = epoch
            print(
                f"[prior early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_smooth={best_val_loss:.6f} patience={early_stopping_patience}"
            )
            break

    if has_val and restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[prior restore-best] restored epoch={best_epoch} val_smooth={best_val_loss:.6f}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_monitor_losses": val_monitor_losses,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
    }


def train_local_decoder(
    model: LocalDecoderLogit,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 1e-4,
    early_stopping_ema_alpha: float = 0.2,
    restore_best: bool = True,
    log_every: int = 20,
) -> dict[str, float | int | bool | list[float]]:
    loader = to_decoder_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    has_val = x_val is not None and y_val is not None and len(x_val) > 0 and len(y_val) > 0
    val_loader = to_decoder_loader(x_val, y_val, batch_size=batch_size, shuffle=False) if has_val else None
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    stopped_early = False
    stopped_epoch = epochs
    val_losses: list[float] = []
    val_monitor_losses: list[float] = []
    val_ema: float | None = None
    alpha = float(early_stopping_ema_alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("early_stopping_ema_alpha must be in (0, 1].")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        mean_train_loss = float(np.mean(epoch_losses))
        losses.append(mean_train_loss)

        mean_val_loss = float("nan")
        if has_val and val_loader is not None:
            model.eval()
            val_epoch_losses: list[float] = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    logits = model(xb)
                    val_loss = criterion(logits, yb)
                    val_epoch_losses.append(float(val_loss.item()))
            mean_val_loss = float(np.mean(val_epoch_losses))
            val_ema = _ema_update_val_monitor(val_ema, mean_val_loss, alpha)
            smooth_val_loss = val_ema
            if smooth_val_loss < (best_val_loss - early_stopping_min_delta):
                best_val_loss = smooth_val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            val_monitor_losses.append(smooth_val_loss)
        else:
            val_monitor_losses.append(float("nan"))
        val_losses.append(mean_val_loss)

        if epoch == 1 or epoch % max(1, log_every) == 0 or epoch == epochs:
            if has_val:
                print(
                    f"[decoder epoch {epoch:4d}/{epochs}] train={mean_train_loss:.6f} "
                    f"val={mean_val_loss:.6f} val_smooth={val_monitor_losses[-1]:.6f} "
                    f"best_smooth={best_val_loss:.6f} best_epoch={best_epoch}"
                )
            else:
                print(f"[decoder epoch {epoch:4d}/{epochs}] train={mean_train_loss:.6f}")

        if has_val and early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            stopped_early = True
            stopped_epoch = epoch
            print(
                f"[decoder early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_smooth={best_val_loss:.6f} patience={early_stopping_patience}"
            )
            break

    if has_val and restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[decoder restore-best] restored epoch={best_epoch} val_smooth={best_val_loss:.6f}")

    return {
        "train_losses": losses,
        "val_losses": val_losses,
        "val_monitor_losses": val_monitor_losses,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
    }
