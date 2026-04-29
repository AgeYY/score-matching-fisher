"""Small helpers for model-parameter exponential moving averages."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn


def init_model_weight_ema(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def update_model_weight_ema(ema_state: dict[str, torch.Tensor], model: nn.Module, *, decay: float) -> None:
    d = float(decay)
    with torch.no_grad():
        for k, v in model.state_dict().items():
            cur = v.detach()
            if k not in ema_state:
                ema_state[k] = cur.clone()
                continue
            if torch.is_floating_point(cur) or torch.is_complex(cur):
                ema_state[k].mul_(d).add_(cur, alpha=1.0 - d)
            else:
                ema_state[k].copy_(cur)


def clone_model_weight_ema(ema_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in ema_state.items()}


def scalar_val_ema_update(prev: float | None, mean_val_loss: float, ema_alpha: float) -> float:
    """Exponential moving average of a per-epoch scalar validation loss (monitor / early-stop)."""
    if prev is None:
        return float(mean_val_loss)
    a = float(ema_alpha)
    return float(a * float(mean_val_loss) + (1.0 - a) * float(prev))


def load_model_weights_from_ema_state(model: nn.Module, ema_state: dict[str, torch.Tensor]) -> None:
    """Load a weight-EMA snapshot into ``model`` in-place.

    ``ema_state`` may hold CPU tensors; they are cast to each parameter/buffer's device and dtype.
    Used after training for downstream eval/sampling when the eval checkpoint is EMA weights only.
    """
    cur = model.state_dict()
    load_sd: dict[str, torch.Tensor] = {}
    for k in cur.keys():
        if k not in ema_state:
            raise KeyError(f"EMA state_dict missing key {k!r}")
        load_sd[k] = ema_state[k].to(device=cur[k].device, dtype=cur[k].dtype)
    model.load_state_dict(load_sd, strict=True)


@contextmanager
def evaluate_with_weight_ema(model: nn.Module, ema_state: dict[str, torch.Tensor]):
    """Temporarily load EMA weights for evaluation, then restore raw training weights.

    Training must keep optimizing the raw ``model`` parameters; use this only around
    validation / scoring blocks. Restores even if the wrapped block raises.
    """
    raw_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    try:
        load_model_weights_from_ema_state(model, ema_state)
        yield
    finally:
        cur = model.state_dict()
        restore_sd = {k: raw_cpu[k].to(device=cur[k].device, dtype=cur[k].dtype) for k in cur.keys()}
        model.load_state_dict(restore_sd, strict=True)
