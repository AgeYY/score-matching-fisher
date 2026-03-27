from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fisher.models import ConditionalScore1D, LocalDecoderLogit


def to_score_loader(theta: np.ndarray, x: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    t = torch.from_numpy(theta.astype(np.float32))
    xx = torch.from_numpy(x.astype(np.float32))
    ds = TensorDataset(t, xx)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def to_decoder_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    xt = torch.from_numpy(x.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.float32)).reshape(-1, 1)
    ds = TensorDataset(xt, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


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
    model: ConditionalScore1D,
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
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    stopped_early = False
    stopped_epoch = epochs

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
            if mean_val_loss < (best_val_loss - early_stopping_min_delta):
                best_val_loss = mean_val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
        val_losses.append(mean_val_loss)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            if has_val:
                print(
                    f"[epoch {epoch:4d}/{epochs}] train_loss={mean_train_loss:.6f} "
                    f"val_loss={mean_val_loss:.6f} best_val={best_val_loss:.6f} best_epoch={best_epoch}"
                )
            else:
                print(f"[epoch {epoch:4d}/{epochs}] train_loss={mean_train_loss:.6f}")

        if has_val and patience_counter >= early_stopping_patience:
            stopped_early = True
            stopped_epoch = epoch
            print(
                f"[early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_val={best_val_loss:.6f} patience={early_stopping_patience}"
            )
            break

    if has_val and restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[restore-best] restored epoch={best_epoch} val_loss={best_val_loss:.6f}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "stopped_early": bool(stopped_early),
    }


def train_score_model_ncsm_continuous(
    model: ConditionalScore1D,
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
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    stopped_early = False
    stopped_epoch = epochs

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
            if mean_val_loss < (best_val_loss - early_stopping_min_delta):
                best_val_loss = mean_val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
        val_losses.append(mean_val_loss)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            if has_val:
                print(
                    f"[epoch {epoch:4d}/{epochs}] ncsm_train={mean_train_loss:.6f} "
                    f"val_loss={mean_val_loss:.6f} best_val={best_val_loss:.6f} best_epoch={best_epoch}"
                )
            else:
                print(f"[epoch {epoch:4d}/{epochs}] ncsm_loss={mean_train_loss:.6f}")

        if has_val and patience_counter >= early_stopping_patience:
            stopped_early = True
            stopped_epoch = epoch
            print(
                f"[early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"best_val={best_val_loss:.6f} patience={early_stopping_patience}"
            )
            break

    if has_val and restore_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[restore-best] restored epoch={best_epoch} val_loss={best_val_loss:.6f}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
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
) -> list[float]:
    loader = to_decoder_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    for _ in range(epochs):
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
        losses.append(float(np.mean(epoch_losses)))
    return losses
