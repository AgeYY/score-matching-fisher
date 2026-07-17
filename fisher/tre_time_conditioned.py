"""Time-conditioned telescoping density-ratio estimation for EEG RDMs."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from fisher.tre_distance import (
    TREDensityRatioConfig,
    TelescopingDensityRatio,
    build_tre_waymarks,
)


@dataclass(frozen=True)
class TimeConditionedTRETrainingResult:
    train_losses: np.ndarray
    validation_losses: np.ndarray
    best_epoch: int
    best_validation_loss: float
    stopped_epoch: int
    training_seconds: float
    config: dict[str, Any]


def _as_trial_time_tensor(
    values: np.ndarray | torch.Tensor,
    *,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    if tensor.ndim != 3 or min(tensor.shape) < 1:
        raise ValueError(f"{name} must have shape [trials, time, features].")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values.")
    return tensor


def _conditioned_waymarks(
    x0: torch.Tensor,
    x1: torch.Tensor,
    times: torch.Tensor,
    *,
    num_bridges: int,
    schedule: str,
) -> torch.Tensor:
    waymarks = build_tre_waymarks(
        x0,
        x1,
        num_bridges=num_bridges,
        schedule=schedule,
    )
    time_column = times[None, :, None].expand(waymarks.shape[0], -1, 1)
    return torch.cat((waymarks, time_column), dim=-1)


def _conditioned_binary_loss(
    model: TelescopingDensityRatio,
    waymarks: torch.Tensor,
) -> torch.Tensor:
    bridges = model.num_bridges
    batch = int(waymarks.shape[1])
    lower_all = model.adjacent_logits(waymarks[:-1].reshape(bridges * batch, -1))
    upper_all = model.adjacent_logits(waymarks[1:].reshape(bridges * batch, -1))
    head_index = torch.arange(bridges, device=waymarks.device).repeat_interleave(batch)
    row_index = torch.arange(bridges * batch, device=waymarks.device)
    lower = lower_all[row_index, head_index]
    upper = upper_all[row_index, head_index]
    return 0.5 * (
        F.binary_cross_entropy_with_logits(lower, torch.ones_like(lower))
        + F.binary_cross_entropy_with_logits(upper, torch.zeros_like(upper))
    )


def train_time_conditioned_tre_density_ratio(
    *,
    x0_train: np.ndarray | torch.Tensor,
    x1_train: np.ndarray | torch.Tensor,
    x0_validation: np.ndarray | torch.Tensor,
    x1_validation: np.ndarray | torch.Tensor,
    times: np.ndarray | torch.Tensor,
    device: torch.device,
    seed: int,
    config: TREDensityRatioConfig,
) -> tuple[TelescopingDensityRatio, TimeConditionedTRETrainingResult]:
    """Fit ``log p0(x|t)-log p1(x|t)`` using same-time TRE waymarks."""

    config.validate()
    device = torch.device(device)
    x0_tr = _as_trial_time_tensor(x0_train, name="x0_train", device=device)
    x1_tr = _as_trial_time_tensor(x1_train, name="x1_train", device=device)
    x0_val = _as_trial_time_tensor(x0_validation, name="x0_validation", device=device)
    x1_val = _as_trial_time_tensor(x1_validation, name="x1_validation", device=device)
    time_tensor = torch.as_tensor(times, dtype=torch.float32, device=device).reshape(-1)
    n_time = int(time_tensor.numel())
    feature_dim = int(x0_tr.shape[-1])
    for value in (x1_tr, x0_val, x1_val):
        if int(value.shape[1]) != n_time or int(value.shape[2]) != feature_dim:
            raise ValueError("TRE endpoints must share time and feature dimensions.")

    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    joined = torch.cat((x0_tr.reshape(-1, feature_dim), x1_tr.reshape(-1, feature_dim)))
    if config.standardize:
        x_mean = joined.mean(dim=0)
        x_scale = joined.std(dim=0, unbiased=False).clamp_min(1e-6)
        t_mean = time_tensor.mean().reshape(1)
        t_scale = time_tensor.std(unbiased=False).clamp_min(1e-6).reshape(1)
    else:
        x_mean = torch.zeros(feature_dim, device=device)
        x_scale = torch.ones(feature_dim, device=device)
        t_mean = torch.zeros(1, device=device)
        t_scale = torch.ones(1, device=device)
    model = TelescopingDensityRatio(
        input_dim=feature_dim + 1,
        num_bridges=config.num_bridges,
        architecture=config.architecture,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        input_mean=torch.cat((x_mean, t_mean)),
        input_scale=torch.cat((x_scale, t_scale)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    generator = torch.Generator(device=device).manual_seed(int(seed) + 1)
    val_generator = torch.Generator(device=device).manual_seed(int(seed) + 2)

    def sample_waymarks(
        left: torch.Tensor,
        right: torch.Tensor,
        count: int,
        rng: torch.Generator,
    ) -> torch.Tensor:
        time_index = torch.randint(n_time, (count,), generator=rng, device=device)
        left_trial = torch.randint(int(left.shape[0]), (count,), generator=rng, device=device)
        right_trial = torch.randint(int(right.shape[0]), (count,), generator=rng, device=device)
        return _conditioned_waymarks(
            left[left_trial, time_index],
            right[right_trial, time_index],
            time_tensor[time_index],
            num_bridges=config.num_bridges,
            schedule=config.waymark_schedule,
        )

    fixed_validation = sample_waymarks(
        x0_val, x1_val, config.validation_pairs, val_generator
    )
    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_loss = math.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        waymarks = sample_waymarks(x0_tr, x1_tr, config.batch_size, generator)
        optimizer.zero_grad(set_to_none=True)
        loss = _conditioned_binary_loss(model, waymarks)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite time-conditioned TRE loss at epoch {epoch}.")
        loss.backward()
        if config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        train_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(_conditioned_binary_loss(model, fixed_validation))
        validation_losses.append(validation_loss)
        if validation_loss < best_loss - config.early_min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % config.log_every == 0:
            print(
                f"[time-TRE] epoch={epoch}/{config.epochs} "
                f"train={float(loss.detach()):.6f} "
                f"validation={validation_loss:.6f} best_epoch={best_epoch}",
                flush=True,
            )
        if config.early_patience and epochs_without_improvement >= config.early_patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, TimeConditionedTRETrainingResult(
        train_losses=np.asarray(train_losses),
        validation_losses=np.asarray(validation_losses),
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        stopped_epoch=len(train_losses),
        training_seconds=time.perf_counter() - started,
        config=asdict(config),
    )


@torch.no_grad()
def evaluate_time_conditioned_log_ratio(
    model: TelescopingDensityRatio,
    values: np.ndarray | torch.Tensor,
    times: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Return trial-by-time estimates of ``log p0(x|t)-log p1(x|t)``."""

    device = torch.device(device)
    tensor = _as_trial_time_tensor(values, name="values", device=device)
    time_tensor = torch.as_tensor(times, dtype=torch.float32, device=device).reshape(-1)
    if int(tensor.shape[1]) != int(time_tensor.numel()):
        raise ValueError("Evaluation values and times do not match.")
    joined = torch.cat(
        (
            tensor,
            time_tensor[None, :, None].expand(int(tensor.shape[0]), -1, 1),
        ),
        dim=-1,
    ).reshape(-1, int(tensor.shape[-1]) + 1)
    chunks: list[torch.Tensor] = []
    for start in range(0, int(joined.shape[0]), int(batch_size)):
        chunks.append(model.log_ratio(joined[start : start + batch_size]))
    return (
        torch.cat(chunks)
        .reshape(int(tensor.shape[0]), int(tensor.shape[1]))
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
    )
