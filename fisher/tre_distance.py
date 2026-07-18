"""Torch telescoping density-ratio estimation (TRE).

The model estimates ``log p0(x) - log p1(x)`` by summing binary-classifier
logits over adjacent waymark distributions. Classifier head ``k`` receives
label one for waymark ``k`` and label zero for waymark ``k + 1``.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


TRE_WAYMARK_SCHEDULES = ("angle", "linear_alpha")
TRE_ARCHITECTURES = ("linear", "mlp")


@dataclass(frozen=True)
class TREDensityRatioConfig:
    """Training defaults for one endpoint pair."""

    num_bridges: int = 8
    waymark_schedule: str = "angle"
    architecture: str = "mlp"
    hidden_dim: int = 128
    depth: int = 3
    epochs: int = 1_000
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.0
    early_patience: int = 100
    early_min_delta: float = 1e-5
    max_grad_norm: float = 10.0
    validation_pairs: int = 2_048
    standardize: bool = True
    log_every: int = 50

    def validate(self) -> None:
        if int(self.num_bridges) < 1:
            raise ValueError("num_bridges must be >= 1.")
        if self.waymark_schedule not in TRE_WAYMARK_SCHEDULES:
            raise ValueError(f"waymark_schedule must be one of {TRE_WAYMARK_SCHEDULES}.")
        if self.architecture not in TRE_ARCHITECTURES:
            raise ValueError(f"architecture must be one of {TRE_ARCHITECTURES}.")
        if int(self.hidden_dim) < 1 or int(self.depth) < 1:
            raise ValueError("hidden_dim and depth must be >= 1.")
        if int(self.epochs) < 1 or int(self.batch_size) < 1:
            raise ValueError("epochs and batch_size must be >= 1.")
        if float(self.lr) <= 0.0 or float(self.weight_decay) < 0.0:
            raise ValueError("lr must be positive and weight_decay must be nonnegative.")
        if int(self.early_patience) < 0:
            raise ValueError("early_patience must be >= 0; zero disables early stopping.")
        if float(self.early_min_delta) < 0.0:
            raise ValueError("early_min_delta must be nonnegative.")
        if float(self.max_grad_norm) < 0.0:
            raise ValueError("max_grad_norm must be nonnegative; zero disables clipping.")
        if int(self.validation_pairs) < 1:
            raise ValueError("validation_pairs must be >= 1.")
        if int(self.log_every) < 1:
            raise ValueError("log_every must be >= 1.")


@dataclass(frozen=True)
class TRETrainingResult:
    train_losses: np.ndarray
    validation_losses: np.ndarray
    best_epoch: int
    best_validation_loss: float
    stopped_epoch: int
    stopped_early: bool
    training_seconds: float
    config: dict[str, Any]


@dataclass(frozen=True)
class PairwiseTREJeffreysResult:
    """Pairwise TRE estimates and per-pair optimization histories."""

    symmetric_kl_matrix: np.ndarray
    raw_symmetric_kl_matrix: np.ndarray
    directed_kl_matrix: np.ndarray
    pair_histories: dict[str, dict[str, Any]]
    pair_metadata: dict[str, dict[str, Any]]
    run_metadata: dict[str, Any]


@dataclass(frozen=True)
class BinnedTREFisherResult:
    """Adjacent-bin TRE Jeffreys estimates converted to scalar Fisher."""

    fisher: np.ndarray
    jeffreys: np.ndarray
    raw_jeffreys: np.ndarray
    pair_histories: dict[str, dict[str, Any]]
    pair_metadata: dict[str, dict[str, Any]]
    run_metadata: dict[str, Any]


def tre_waymark_coefficients(
    num_bridges: int,
    *,
    schedule: str = "angle",
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return endpoint coefficients for ``num_bridges + 1`` waymarks.

    Every waymark has the form ``left[k] * x0 + right[k] * x1``. Both
    schedules satisfy ``left**2 + right**2 == 1``, preserving covariance when
    the endpoints are independent Gaussians with the same covariance.
    """

    bridges = int(num_bridges)
    if bridges < 1:
        raise ValueError("num_bridges must be >= 1.")
    if schedule == "angle":
        angle = torch.linspace(0.0, 0.5 * math.pi, bridges + 1, device=device, dtype=dtype)
        left = torch.cos(angle)
        right = torch.sin(angle)
    elif schedule == "linear_alpha":
        right = torch.linspace(0.0, 1.0, bridges + 1, device=device, dtype=dtype)
        left = torch.sqrt(torch.clamp(1.0 - right.square(), min=0.0))
    else:
        raise ValueError(f"schedule must be one of {TRE_WAYMARK_SCHEDULES}.")
    left[0], right[0] = 1.0, 0.0
    left[-1], right[-1] = 0.0, 1.0
    return left, right


def build_tre_waymarks(
    x0: torch.Tensor,
    x1: torch.Tensor,
    *,
    num_bridges: int,
    schedule: str = "angle",
) -> torch.Tensor:
    """Construct paired waymarks with shape ``[num_bridges + 1, B, D]``."""

    if x0.ndim != 2 or x1.ndim != 2:
        raise ValueError("x0 and x1 must have shape [B, D].")
    if x0.shape != x1.shape:
        raise ValueError("x0 and x1 must have matching shapes.")
    left, right = tre_waymark_coefficients(
        int(num_bridges),
        schedule=str(schedule),
        device=x0.device,
        dtype=x0.dtype,
    )
    return left[:, None, None] * x0[None, :, :] + right[:, None, None] * x1[None, :, :]


class TelescopingDensityRatio(nn.Module):
    """Shared classifier whose heads estimate adjacent waymark log ratios."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_bridges: int,
        architecture: str = "mlp",
        hidden_dim: int = 128,
        depth: int = 3,
        input_mean: torch.Tensor | np.ndarray | None = None,
        input_scale: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_bridges = int(num_bridges)
        self.architecture = str(architecture)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        if self.input_dim < 1 or self.num_bridges < 1:
            raise ValueError("input_dim and num_bridges must be >= 1.")
        if self.architecture not in TRE_ARCHITECTURES:
            raise ValueError(f"architecture must be one of {TRE_ARCHITECTURES}.")

        mean = torch.zeros(self.input_dim, dtype=torch.float32) if input_mean is None else torch.as_tensor(input_mean)
        scale = torch.ones(self.input_dim, dtype=torch.float32) if input_scale is None else torch.as_tensor(input_scale)
        mean = mean.detach().to(dtype=torch.float32).reshape(-1)
        scale = scale.detach().to(dtype=torch.float32).reshape(-1)
        if mean.numel() != self.input_dim or scale.numel() != self.input_dim:
            raise ValueError("input_mean and input_scale must have shape [input_dim].")
        if torch.any(scale <= 0.0):
            raise ValueError("input_scale entries must be positive.")
        self.register_buffer("input_mean", mean.clone())
        self.register_buffer("input_scale", scale.clone())

        if self.architecture == "linear":
            self.trunk = nn.Identity()
            self.heads = nn.Linear(self.input_dim, self.num_bridges)
        else:
            layers: list[nn.Module] = []
            in_dim = self.input_dim
            for _ in range(self.depth):
                layers.extend((nn.Linear(in_dim, self.hidden_dim), nn.SiLU()))
                in_dim = self.hidden_dim
            self.trunk = nn.Sequential(*layers)
            self.heads = nn.Linear(self.hidden_dim, self.num_bridges)

    def adjacent_logits(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or int(x.shape[1]) != self.input_dim:
            raise ValueError(f"x must have shape [B, {self.input_dim}].")
        normalized = (x - self.input_mean) / self.input_scale
        return self.heads(self.trunk(normalized))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adjacent_logits(x)

    def log_ratio(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate ``log p0(x) - log p1(x)``."""

        return self.adjacent_logits(x).sum(dim=-1)


def _as_2d_tensor(x: np.ndarray | torch.Tensor, *, name: str, device: torch.device) -> torch.Tensor:
    out = torch.as_tensor(x, dtype=torch.float32, device=device)
    if out.ndim != 2 or int(out.shape[0]) < 1 or int(out.shape[1]) < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional array.")
    if not bool(torch.isfinite(out).all()):
        raise ValueError(f"{name} contains non-finite values.")
    return out


def _selected_adjacent_logits(model: TelescopingDensityRatio, waymarks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bridges = model.num_bridges
    batch = int(waymarks.shape[1])
    lower_all = model.adjacent_logits(waymarks[:-1].reshape(bridges * batch, -1))
    upper_all = model.adjacent_logits(waymarks[1:].reshape(bridges * batch, -1))
    head_index = torch.arange(bridges, device=waymarks.device).repeat_interleave(batch)
    row_index = torch.arange(bridges * batch, device=waymarks.device)
    return lower_all[row_index, head_index], upper_all[row_index, head_index]


def _tre_binary_loss(model: TelescopingDensityRatio, waymarks: torch.Tensor) -> torch.Tensor:
    lower_logits, upper_logits = _selected_adjacent_logits(model, waymarks)
    positive_loss = F.binary_cross_entropy_with_logits(lower_logits, torch.ones_like(lower_logits))
    negative_loss = F.binary_cross_entropy_with_logits(upper_logits, torch.zeros_like(upper_logits))
    return 0.5 * (positive_loss + negative_loss)


def train_tre_density_ratio(
    *,
    x0_train: np.ndarray | torch.Tensor,
    x1_train: np.ndarray | torch.Tensor,
    x0_validation: np.ndarray | torch.Tensor,
    x1_validation: np.ndarray | torch.Tensor,
    device: torch.device,
    seed: int = 7,
    config: TREDensityRatioConfig | None = None,
) -> tuple[TelescopingDensityRatio, TRETrainingResult]:
    """Fit one TRE model and restore its best validation-BCE checkpoint."""

    cfg = TREDensityRatioConfig() if config is None else config
    cfg.validate()
    device = torch.device(device)
    x0_tr = _as_2d_tensor(x0_train, name="x0_train", device=device)
    x1_tr = _as_2d_tensor(x1_train, name="x1_train", device=device)
    x0_val = _as_2d_tensor(x0_validation, name="x0_validation", device=device)
    x1_val = _as_2d_tensor(x1_validation, name="x1_validation", device=device)
    input_dim = int(x0_tr.shape[1])
    if any(int(x.shape[1]) != input_dim for x in (x1_tr, x0_val, x1_val)):
        raise ValueError("All TRE endpoint arrays must have the same feature dimension.")

    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    train_joined = torch.cat((x0_tr, x1_tr), dim=0)
    if bool(cfg.standardize):
        input_mean = train_joined.mean(dim=0)
        input_scale = train_joined.std(dim=0, unbiased=False).clamp_min(1e-6)
    else:
        input_mean = torch.zeros(input_dim, device=device)
        input_scale = torch.ones(input_dim, device=device)

    model = TelescopingDensityRatio(
        input_dim=input_dim,
        num_bridges=int(cfg.num_bridges),
        architecture=str(cfg.architecture),
        hidden_dim=int(cfg.hidden_dim),
        depth=int(cfg.depth),
        input_mean=input_mean,
        input_scale=input_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + 1)
    val_generator = torch.Generator(device=device)
    val_generator.manual_seed(int(seed) + 2)
    val_pairs = int(cfg.validation_pairs)
    val_idx0 = torch.randint(int(x0_val.shape[0]), (val_pairs,), generator=val_generator, device=device)
    val_idx1 = torch.randint(int(x1_val.shape[0]), (val_pairs,), generator=val_generator, device=device)
    fixed_val_waymarks = build_tre_waymarks(
        x0_val[val_idx0],
        x1_val[val_idx1],
        num_bridges=int(cfg.num_bridges),
        schedule=str(cfg.waymark_schedule),
    )

    batches_per_epoch = max(1, math.ceil(max(int(x0_tr.shape[0]), int(x1_tr.shape[0])) / int(cfg.batch_size)))
    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_loss = math.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    stopped_early = False
    started = time.perf_counter()

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        loss_sum = 0.0
        for _ in range(batches_per_epoch):
            idx0 = torch.randint(
                int(x0_tr.shape[0]),
                (int(cfg.batch_size),),
                generator=generator,
                device=device,
            )
            idx1 = torch.randint(
                int(x1_tr.shape[0]),
                (int(cfg.batch_size),),
                generator=generator,
                device=device,
            )
            waymarks = build_tre_waymarks(
                x0_tr[idx0],
                x1_tr[idx1],
                num_bridges=int(cfg.num_bridges),
                schedule=str(cfg.waymark_schedule),
            )
            optimizer.zero_grad(set_to_none=True)
            loss = _tre_binary_loss(model, waymarks)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite TRE training loss at epoch {epoch}.")
            loss.backward()
            if float(cfg.max_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.max_grad_norm))
            optimizer.step()
            loss_sum += float(loss.detach())
        train_loss = loss_sum / float(batches_per_epoch)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            validation_loss = float(_tre_binary_loss(model, fixed_val_waymarks))
        if not math.isfinite(validation_loss):
            raise FloatingPointError(f"Non-finite TRE validation loss at epoch {epoch}.")
        validation_losses.append(validation_loss)

        if validation_loss < best_loss - float(cfg.early_min_delta):
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % int(cfg.log_every) == 0:
            print(
                f"[TRE] epoch={epoch}/{cfg.epochs} train_bce={train_loss:.6f} "
                f"validation_bce={validation_loss:.6f} best_epoch={best_epoch}",
                flush=True,
            )
        if int(cfg.early_patience) > 0 and epochs_without_improvement >= int(cfg.early_patience):
            stopped_early = True
            break

    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    result = TRETrainingResult(
        train_losses=np.asarray(train_losses, dtype=np.float64),
        validation_losses=np.asarray(validation_losses, dtype=np.float64),
        best_epoch=int(best_epoch),
        best_validation_loss=float(best_loss),
        stopped_epoch=int(len(train_losses)),
        stopped_early=bool(stopped_early),
        training_seconds=float(elapsed),
        config=asdict(cfg),
    )
    return model, result


@torch.no_grad()
def estimate_tre_log_ratio(
    model: TelescopingDensityRatio,
    x: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 4_096,
) -> np.ndarray:
    """Evaluate ``log p0(x) - log p1(x)`` in batches."""

    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1.")
    device = torch.device(device)
    x_tensor = _as_2d_tensor(x, name="x", device=device)
    if int(x_tensor.shape[1]) != model.input_dim:
        raise ValueError(f"Expected x feature dimension {model.input_dim}, got {x_tensor.shape[1]}.")
    model = model.to(device)
    model.eval()
    chunks: list[np.ndarray] = []
    for start in range(0, int(x_tensor.shape[0]), int(batch_size)):
        values = model.log_ratio(x_tensor[start : start + int(batch_size)])
        chunks.append(values.detach().cpu().numpy().astype(np.float64, copy=False))
    return np.concatenate(chunks, axis=0)


def tre_jeffreys_from_log_ratios(log_ratio_x0: np.ndarray, log_ratio_x1: np.ndarray) -> float:
    """Estimate Jeffreys divergence for a ``log p0 - log p1`` estimator."""

    ratio0 = np.asarray(log_ratio_x0, dtype=np.float64).reshape(-1)
    ratio1 = np.asarray(log_ratio_x1, dtype=np.float64).reshape(-1)
    if ratio0.size < 1 or ratio1.size < 1:
        raise ValueError("Both endpoint log-ratio arrays must be non-empty.")
    if not np.all(np.isfinite(ratio0)) or not np.all(np.isfinite(ratio1)):
        raise ValueError("Endpoint log-ratio arrays must be finite.")
    return float(np.mean(ratio0, dtype=np.float64) - np.mean(ratio1, dtype=np.float64))


def _theta_endpoint_windows(
    theta: np.ndarray,
    x: np.ndarray,
    theta_grid: np.ndarray,
    *,
    radius: float | None,
    min_samples: int,
) -> list[np.ndarray]:
    theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    x_arr = np.asarray(x, dtype=np.float32)
    grid = np.asarray(theta_grid, dtype=np.float64).reshape(-1)
    if x_arr.ndim != 2 or int(x_arr.shape[0]) != int(theta_arr.shape[0]):
        raise ValueError("theta and x must have matching rows.")
    if grid.size < 2 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("theta_grid must contain at least two increasing values.")
    radius_value = 0.5 * float(np.min(np.diff(grid))) if radius is None else float(radius)
    if radius_value <= 0.0:
        raise ValueError("window radius must be positive.")
    required = int(min_samples)
    if required < 1:
        raise ValueError("min_samples must be positive.")
    required = min(required, int(theta_arr.size))
    windows: list[np.ndarray] = []
    for value in grid:
        index = np.flatnonzero(np.abs(theta_arr - float(value)) <= radius_value)
        if int(index.size) < required:
            index = np.argsort(np.abs(theta_arr - float(value)), kind="mergesort")[:required]
        windows.append(x_arr[index])
    return windows


def train_and_estimate_binned_tre_fisher(
    *,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_validation: np.ndarray,
    x_validation: np.ndarray,
    theta_eval: np.ndarray,
    x_eval: np.ndarray,
    theta_grid: np.ndarray,
    device: torch.device,
    seed: int = 7,
    config: TREDensityRatioConfig | None = None,
    window_radius: float | None = None,
    min_train_samples: int = 2,
    min_validation_samples: int = 2,
    min_eval_samples: int = 2,
    eval_batch_size: int = 4_096,
) -> tuple[dict[str, dict[str, torch.Tensor]], BinnedTREFisherResult]:
    """Fit independent TRE models between adjacent local theta windows."""

    cfg = TREDensityRatioConfig() if config is None else config
    cfg.validate()
    grid = np.asarray(theta_grid, dtype=np.float64).reshape(-1)
    train_windows = _theta_endpoint_windows(
        theta_train,
        x_train,
        grid,
        radius=window_radius,
        min_samples=int(min_train_samples),
    )
    validation_windows = _theta_endpoint_windows(
        theta_validation,
        x_validation,
        grid,
        radius=window_radius,
        min_samples=int(min_validation_samples),
    )
    eval_windows = _theta_endpoint_windows(
        theta_eval,
        x_eval,
        grid,
        radius=window_radius,
        min_samples=int(min_eval_samples),
    )
    n_pairs = int(grid.size - 1)
    raw_jeffreys = np.empty(n_pairs, dtype=np.float64)
    jeffreys = np.empty(n_pairs, dtype=np.float64)
    fisher = np.empty(n_pairs, dtype=np.float64)
    state_dicts: dict[str, dict[str, torch.Tensor]] = {}
    pair_histories: dict[str, dict[str, Any]] = {}
    pair_metadata: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for pair_index in range(n_pairs):
        pair_key = f"{pair_index}_{pair_index + 1}"
        pair_seed = int(seed) + 10_007 * pair_index
        print(
            f"[binned-TRE] fitting pair={pair_key} seed={pair_seed} "
            f"bridges={cfg.num_bridges}",
            flush=True,
        )
        model, training = train_tre_density_ratio(
            x0_train=train_windows[pair_index],
            x1_train=train_windows[pair_index + 1],
            x0_validation=validation_windows[pair_index],
            x1_validation=validation_windows[pair_index + 1],
            device=torch.device(device),
            seed=pair_seed,
            config=cfg,
        )
        log_ratio_left = estimate_tre_log_ratio(
            model,
            eval_windows[pair_index],
            device=torch.device(device),
            batch_size=int(eval_batch_size),
        )
        log_ratio_right = estimate_tre_log_ratio(
            model,
            eval_windows[pair_index + 1],
            device=torch.device(device),
            batch_size=int(eval_batch_size),
        )
        raw_value = tre_jeffreys_from_log_ratios(log_ratio_left, log_ratio_right)
        clipped_value = max(0.0, raw_value)
        spacing = float(grid[pair_index + 1] - grid[pair_index])
        raw_jeffreys[pair_index] = raw_value
        jeffreys[pair_index] = clipped_value
        fisher[pair_index] = clipped_value / spacing**2
        state_dicts[pair_key] = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        pair_histories[pair_key] = {
            "train_losses": training.train_losses,
            "validation_losses": training.validation_losses,
            "best_epoch": int(training.best_epoch),
            "best_validation_loss": float(training.best_validation_loss),
            "stopped_epoch": int(training.stopped_epoch),
            "stopped_early": bool(training.stopped_early),
            "training_seconds": float(training.training_seconds),
        }
        pair_metadata[pair_key] = {
            "theta_left": float(grid[pair_index]),
            "theta_right": float(grid[pair_index + 1]),
            "spacing": spacing,
            "pair_seed": pair_seed,
            "n_train_left": int(train_windows[pair_index].shape[0]),
            "n_train_right": int(train_windows[pair_index + 1].shape[0]),
            "n_validation_left": int(validation_windows[pair_index].shape[0]),
            "n_validation_right": int(validation_windows[pair_index + 1].shape[0]),
            "n_eval_left": int(eval_windows[pair_index].shape[0]),
            "n_eval_right": int(eval_windows[pair_index + 1].shape[0]),
            "raw_jeffreys": raw_value,
            "jeffreys": clipped_value,
            "fisher": float(fisher[pair_index]),
        }
    result = BinnedTREFisherResult(
        fisher=fisher,
        jeffreys=jeffreys,
        raw_jeffreys=raw_jeffreys,
        pair_histories=pair_histories,
        pair_metadata=pair_metadata,
        run_metadata={
            "seed": int(seed),
            "num_pairs": n_pairs,
            "theta_grid": grid.tolist(),
            "window_radius": (
                0.5 * float(np.min(np.diff(grid)))
                if window_radius is None
                else float(window_radius)
            ),
            "eval_batch_size": int(eval_batch_size),
            "total_training_seconds": float(time.perf_counter() - started),
            "config": asdict(cfg),
        },
    )
    return state_dicts, result


def train_and_estimate_pairwise_tre_jeffreys(
    *,
    x_train: np.ndarray,
    labels_train: np.ndarray,
    x_validation: np.ndarray,
    labels_validation: np.ndarray,
    x_eval: np.ndarray,
    labels_eval: np.ndarray,
    num_categories: int,
    device: torch.device,
    seed: int = 7,
    config: TREDensityRatioConfig | None = None,
    eval_batch_size: int = 4_096,
) -> tuple[dict[str, dict[str, torch.Tensor]], PairwiseTREJeffreysResult]:
    """Fit one TRE model per unordered condition pair and estimate Jeffreys KL."""

    cfg = TREDensityRatioConfig() if config is None else config
    cfg.validate()
    x_train_arr = np.asarray(x_train, dtype=np.float32)
    x_validation_arr = np.asarray(x_validation, dtype=np.float32)
    x_eval_arr = np.asarray(x_eval, dtype=np.float32)
    labels_train_arr = np.asarray(labels_train, dtype=np.int64).reshape(-1)
    labels_validation_arr = np.asarray(labels_validation, dtype=np.int64).reshape(-1)
    labels_eval_arr = np.asarray(labels_eval, dtype=np.int64).reshape(-1)
    arrays = (
        ("train", x_train_arr, labels_train_arr),
        ("validation", x_validation_arr, labels_validation_arr),
        ("evaluation", x_eval_arr, labels_eval_arr),
    )
    for split_name, x_arr, label_arr in arrays:
        if x_arr.ndim != 2 or int(x_arr.shape[0]) != int(label_arr.shape[0]):
            raise ValueError(f"TRE {split_name} x and labels must have matching rows.")
        if not np.all(np.isfinite(x_arr)):
            raise ValueError(f"TRE {split_name} x contains non-finite values.")
    if not (
        int(x_train_arr.shape[1])
        == int(x_validation_arr.shape[1])
        == int(x_eval_arr.shape[1])
    ):
        raise ValueError("TRE train, validation, and evaluation feature dimensions must match.")
    categories = int(num_categories)
    if categories < 2:
        raise ValueError("num_categories must be >= 2.")
    if int(eval_batch_size) < 1:
        raise ValueError("eval_batch_size must be >= 1.")
    for split_name, _, label_arr in arrays:
        if np.any((label_arr < 0) | (label_arr >= categories)):
            raise ValueError(f"TRE {split_name} labels must be in [0, {categories - 1}].")
        counts = np.bincount(label_arr, minlength=categories)
        if np.any(counts == 0):
            raise ValueError(f"TRE {split_name} split must contain every category; counts={counts.tolist()}.")

    symmetric = np.zeros((categories, categories), dtype=np.float64)
    raw_symmetric = np.zeros_like(symmetric)
    directed = np.zeros_like(symmetric)
    state_dicts: dict[str, dict[str, torch.Tensor]] = {}
    pair_histories: dict[str, dict[str, Any]] = {}
    pair_metadata: dict[str, dict[str, Any]] = {}
    pair_counter = 0
    run_started = time.perf_counter()

    for condition_i in range(categories):
        for condition_j in range(condition_i + 1, categories):
            pair_key = f"{condition_i}_{condition_j}"
            pair_seed = int(seed) + 10_007 * pair_counter
            print(
                f"[TRE] fitting pair={pair_key} seed={pair_seed} "
                f"bridges={cfg.num_bridges}",
                flush=True,
            )
            model, training = train_tre_density_ratio(
                x0_train=x_train_arr[labels_train_arr == condition_i],
                x1_train=x_train_arr[labels_train_arr == condition_j],
                x0_validation=x_validation_arr[labels_validation_arr == condition_i],
                x1_validation=x_validation_arr[labels_validation_arr == condition_j],
                device=torch.device(device),
                seed=pair_seed,
                config=cfg,
            )
            eval_i = x_eval_arr[labels_eval_arr == condition_i]
            eval_j = x_eval_arr[labels_eval_arr == condition_j]
            log_ratio_i = estimate_tre_log_ratio(
                model,
                eval_i,
                device=torch.device(device),
                batch_size=int(eval_batch_size),
            )
            log_ratio_j = estimate_tre_log_ratio(
                model,
                eval_j,
                device=torch.device(device),
                batch_size=int(eval_batch_size),
            )
            kl_i_j = float(np.mean(log_ratio_i, dtype=np.float64))
            kl_j_i = float(-np.mean(log_ratio_j, dtype=np.float64))
            raw_value = kl_i_j + kl_j_i
            symmetric_value = max(0.0, raw_value)
            directed[condition_i, condition_j] = kl_i_j
            directed[condition_j, condition_i] = kl_j_i
            raw_symmetric[condition_i, condition_j] = raw_symmetric[condition_j, condition_i] = raw_value
            symmetric[condition_i, condition_j] = symmetric[condition_j, condition_i] = symmetric_value
            state_dicts[pair_key] = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            pair_histories[pair_key] = {
                "train_losses": training.train_losses,
                "validation_losses": training.validation_losses,
                "best_epoch": int(training.best_epoch),
                "best_validation_loss": float(training.best_validation_loss),
                "stopped_epoch": int(training.stopped_epoch),
                "stopped_early": bool(training.stopped_early),
                "training_seconds": float(training.training_seconds),
            }
            pair_metadata[pair_key] = {
                "condition_i": int(condition_i),
                "condition_j": int(condition_j),
                "pair_seed": int(pair_seed),
                "n_train_i": int(np.sum(labels_train_arr == condition_i)),
                "n_train_j": int(np.sum(labels_train_arr == condition_j)),
                "n_validation_i": int(np.sum(labels_validation_arr == condition_i)),
                "n_validation_j": int(np.sum(labels_validation_arr == condition_j)),
                "n_eval_i": int(eval_i.shape[0]),
                "n_eval_j": int(eval_j.shape[0]),
                "kl_i_j": kl_i_j,
                "kl_j_i": kl_j_i,
                "raw_jeffreys": raw_value,
                "jeffreys": symmetric_value,
            }
            pair_counter += 1

    result = PairwiseTREJeffreysResult(
        symmetric_kl_matrix=symmetric,
        raw_symmetric_kl_matrix=raw_symmetric,
        directed_kl_matrix=directed,
        pair_histories=pair_histories,
        pair_metadata=pair_metadata,
        run_metadata={
            "num_categories": categories,
            "input_dim": int(x_train_arr.shape[1]),
            "num_pair_models": int(pair_counter),
            "seed": int(seed),
            "eval_batch_size": int(eval_batch_size),
            "total_training_seconds": float(time.perf_counter() - run_started),
            "config": asdict(cfg),
        },
    )
    return state_dicts, result


def save_pairwise_tre_jeffreys_result(
    npz_path: str | Path,
    checkpoint_path: str | Path,
    *,
    pair_state_dicts: dict[str, dict[str, torch.Tensor]],
    result: PairwiseTREJeffreysResult,
) -> tuple[Path, Path]:
    """Serialize pairwise TRE matrices, histories, metadata, and model states."""

    npz_out = Path(npz_path).expanduser()
    checkpoint_out = Path(checkpoint_path).expanduser()
    npz_out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    pair_keys = tuple(sorted(result.pair_histories))
    fields: dict[str, np.ndarray] = {
        "symmetric_kl_matrix": np.asarray(result.symmetric_kl_matrix, dtype=np.float64),
        "raw_symmetric_kl_matrix": np.asarray(result.raw_symmetric_kl_matrix, dtype=np.float64),
        "directed_kl_matrix": np.asarray(result.directed_kl_matrix, dtype=np.float64),
        "pair_keys": np.asarray(pair_keys),
        "run_metadata_json": np.asarray([json.dumps(result.run_metadata, sort_keys=True)]),
        "pair_metadata_json": np.asarray([json.dumps(result.pair_metadata, sort_keys=True)]),
    }
    for pair_key in pair_keys:
        history = result.pair_histories[pair_key]
        fields[f"pair_{pair_key}_train_losses"] = np.asarray(history["train_losses"], dtype=np.float64)
        fields[f"pair_{pair_key}_validation_losses"] = np.asarray(
            history["validation_losses"], dtype=np.float64
        )
    np.savez_compressed(npz_out, **fields)
    torch.save(
        {
            "pair_state_dicts": pair_state_dicts,
            "pair_metadata": result.pair_metadata,
            "run_metadata": result.run_metadata,
        },
        checkpoint_out,
    )
    return npz_out, checkpoint_out
