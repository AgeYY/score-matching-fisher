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
) -> list[float]:
    loader = to_score_loader(theta_train, x_train, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sigma_values_t = torch.from_numpy(sigma_values.astype(np.float32)).to(device)
    losses: list[float] = []

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
        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(f"[epoch {epoch:4d}/{epochs}] train_loss={mean_loss:.6f}")
    return losses


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
) -> list[float]:
    loader = to_score_loader(theta_train, x_train, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []

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
        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(f"[epoch {epoch:4d}/{epochs}] ncsm_loss={mean_loss:.6f}")
    return losses


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
