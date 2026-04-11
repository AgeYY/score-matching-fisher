from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fisher.models import (
    ConditionalScore1D,
    ConditionalScore1DFiLMPerLayer,
    ConditionalThetaEDM,
    ConditionalThetaFlowVelocity,
    ConditionalXFlowVelocity,
    ConditionalXFlowVelocityFiLMPerLayer,
    ConditionalXScore,
    LocalDecoderLogit,
    PriorThetaEDM,
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


def _early_stop_val_smooth(
    epoch: int,
    mean_val_loss: float,
    val_ema: float | None,
    ema_alpha: float,
    warmup_epochs: int,
) -> tuple[float, float | None]:
    """Warmup: use raw val loss and keep EMA state unset. After warmup: standard EMA update."""
    w = int(warmup_epochs)
    if w > 0 and epoch <= w:
        return float(mean_val_loss), None
    new_ema = _ema_update_val_monitor(val_ema, mean_val_loss, ema_alpha)
    return new_ema, new_ema


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
    *,
    mode: str = "uniform_log",
    beta_param: float = 2.0,
) -> torch.Tensor:
    if sigma_min <= 0.0 or sigma_max <= 0.0:
        raise ValueError("sigma_min and sigma_max must be positive.")
    lo = min(float(sigma_min), float(sigma_max))
    hi = max(float(sigma_min), float(sigma_max))
    _mode = str(mode).lower()
    if _mode not in ("uniform_log", "beta_log"):
        raise ValueError("mode must be one of {'uniform_log', 'beta_log'}.")
    if _mode == "beta_log":
        if float(beta_param) <= 0.0:
            raise ValueError("beta_param must be positive for beta_log sigma sampling.")
        # u in (0,1): Beta(beta, 1) biases toward larger log-sigmas when beta>1.
        dist = torch.distributions.Beta(
            torch.tensor(float(beta_param), device=device),
            torch.tensor(1.0, device=device),
        )
        u = dist.sample((batch_size, 1))
    else:
        u = torch.rand((batch_size, 1), device=device)
    log_sigma = np.log(lo) + u * (np.log(hi) - np.log(lo))
    return torch.exp(log_sigma)


def sample_edm_sigmas(
    batch_size: int,
    p_mean: float,
    p_std: float,
    device: torch.device,
) -> torch.Tensor:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    if float(p_std) <= 0.0:
        raise ValueError("p_std must be positive.")
    rnd = torch.randn((int(batch_size), 1), device=device)
    return torch.exp(rnd * float(p_std) + float(p_mean))


def _build_optimizer(
    model: nn.Module,
    lr: float,
    optimizer_name: str,
    weight_decay: float,
) -> torch.optim.Optimizer:
    name = str(optimizer_name).lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(weight_decay))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(weight_decay))
    raise ValueError("optimizer_name must be one of {'adam', 'adamw'}.")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    epochs: int,
    warmup_frac: float,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    name = str(scheduler_name).lower()
    if name == "none":
        return None
    if not (0.0 <= float(warmup_frac) < 1.0):
        raise ValueError("lr_warmup_frac must be in [0,1).")
    warmup_epochs = int(max(0, round(float(warmup_frac) * int(epochs))))
    if name == "cosine":
        if warmup_epochs <= 0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)))

        def _lr_lambda(ep: int) -> float:
            e = int(ep) + 1
            if e <= warmup_epochs:
                return float(e) / float(max(1, warmup_epochs))
            rem = max(1, int(epochs) - warmup_epochs)
            p = float(e - warmup_epochs) / float(rem)
            p = min(max(p, 0.0), 1.0)
            return 0.5 * (1.0 + np.cos(np.pi * p))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    raise ValueError("lr_scheduler must be one of {'none', 'cosine'}.")


def _loss_reduce(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str,
    huber_delta: float,
) -> torch.Tensor:
    typ = str(loss_type).lower()
    if typ == "mse":
        return torch.mean((pred - target) ** 2)
    if typ == "huber":
        if float(huber_delta) <= 0.0:
            raise ValueError("huber_delta must be positive when loss_type='huber'.")
        return torch.nn.functional.huber_loss(pred, target, delta=float(huber_delta), reduction="mean")
    raise ValueError("loss_type must be one of {'mse', 'huber'}.")


def _score_matching_loss(
    pred: torch.Tensor,
    eps: torch.Tensor,
    sigma: torch.Tensor,
    *,
    loss_type: str,
    huber_delta: float,
    normalize_by_sigma: bool,
) -> torch.Tensor:
    # Continuous NCSM target: sigma * score + eps -> 0.
    residual = sigma * pred + eps
    if normalize_by_sigma:
        residual = residual / torch.clamp(sigma, min=1e-6)
    return _loss_reduce(
        residual,
        torch.zeros_like(residual),
        loss_type=loss_type,
        huber_delta=huber_delta,
    )


def _edm_theta_loss(
    denoised: torch.Tensor,
    clean_theta: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float,
    *,
    loss_type: str,
    huber_delta: float,
) -> torch.Tensor:
    if float(sigma_data) <= 0.0:
        raise ValueError("sigma_data must be positive.")
    if sigma.ndim == 1:
        sigma = sigma.unsqueeze(-1)
    if torch.any(sigma <= 0):
        raise ValueError("sigma must be strictly positive in EDM loss.")
    weight = (sigma**2 + float(sigma_data) ** 2) / torch.clamp((sigma * float(sigma_data)) ** 2, min=1e-8)
    residual = denoised - clean_theta
    weighted_residual = torch.sqrt(weight) * residual
    return _loss_reduce(
        weighted_residual,
        torch.zeros_like(weighted_residual),
        loss_type=loss_type,
        huber_delta=huber_delta,
    )


def _finite_grad_norm(model: nn.Module) -> float:
    vals: list[float] = []
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if torch.isfinite(g).all():
            vals.append(float(torch.linalg.vector_norm(g).item()))
    if not vals:
        return float("nan")
    return float(np.sqrt(np.sum(np.asarray(vals, dtype=np.float64) ** 2)))


def _param_norm(model: nn.Module) -> float:
    vals: list[float] = []
    for p in model.parameters():
        v = p.detach()
        if torch.isfinite(v).all():
            vals.append(float(torch.linalg.vector_norm(v).item()))
    if not vals:
        return float("nan")
    return float(np.sqrt(np.sum(np.asarray(vals, dtype=np.float64) ** 2)))


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
    early_stopping_ema_warmup_epochs: int = 0,
    restore_best: bool = True,
    optimizer_name: str = "adamw",
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    lr_scheduler: str = "cosine",
    lr_warmup_frac: float = 0.05,
    loss_type: str = "huber",
    huber_delta: float = 1.0,
    normalize_by_sigma: bool = False,
    abort_on_nonfinite: bool = True,
) -> dict[str, float | int | bool | list[float]]:
    if int(early_stopping_ema_warmup_epochs) < 0:
        raise ValueError("early_stopping_ema_warmup_epochs must be >= 0.")
    loader = to_score_loader(theta_train, x_train, batch_size=batch_size, shuffle=True)
    optimizer = _build_optimizer(model, lr=lr, optimizer_name=optimizer_name, weight_decay=weight_decay)
    scheduler = _build_scheduler(
        optimizer,
        scheduler_name=lr_scheduler,
        epochs=epochs,
        warmup_frac=lr_warmup_frac,
    )
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
    ema_warmup = int(early_stopping_ema_warmup_epochs)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("early_stopping_ema_alpha must be in (0, 1].")
    grad_norms: list[float] = []
    n_clipped_steps = 0
    total_steps = 0
    has_nonfinite = False

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
            residual = pred - target
            if normalize_by_sigma:
                residual = sigma * residual
            loss = _loss_reduce(residual, torch.zeros_like(residual), loss_type=loss_type, huber_delta=huber_delta)
            if not torch.isfinite(loss):
                has_nonfinite = True
                if abort_on_nonfinite:
                    print(f"[nonfinite] score train loss became non-finite at epoch={epoch}")
                    break
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(max_grad_norm) > 0.0:
                gn_before = _finite_grad_norm(model)
                if np.isfinite(gn_before):
                    grad_norms.append(float(gn_before))
                    if gn_before > float(max_grad_norm):
                        n_clipped_steps += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
            optimizer.step()
            total_steps += 1
            epoch_losses.append(float(loss.item()))
        if has_nonfinite and abort_on_nonfinite:
            stopped_early = True
            stopped_epoch = epoch
            break
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
                    residual = pred - target
                    if normalize_by_sigma:
                        residual = sigma * residual
                    val_loss = _loss_reduce(
                        residual,
                        torch.zeros_like(residual),
                        loss_type=loss_type,
                        huber_delta=huber_delta,
                    )
                    if not torch.isfinite(val_loss):
                        has_nonfinite = True
                        if abort_on_nonfinite:
                            print(f"[nonfinite] score val loss became non-finite at epoch={epoch}")
                            break
                        continue
                    val_epoch_losses.append(float(val_loss.item()))
            if has_nonfinite and abort_on_nonfinite:
                stopped_early = True
                stopped_epoch = epoch
                break
            mean_val_loss = float(np.mean(val_epoch_losses))
            smooth_val_loss, val_ema = _early_stop_val_smooth(
                epoch, mean_val_loss, val_ema, alpha, ema_warmup
            )
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
        if scheduler is not None:
            scheduler.step()

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
        "has_nonfinite": bool(has_nonfinite),
        "grad_norm_mean": float(np.nanmean(np.asarray(grad_norms, dtype=np.float64))) if grad_norms else float("nan"),
        "grad_norm_max": float(np.nanmax(np.asarray(grad_norms, dtype=np.float64))) if grad_norms else float("nan"),
        "param_norm_final": float(_param_norm(model)),
        "n_clipped_steps": int(n_clipped_steps),
        "n_total_steps": int(total_steps),
        "lr_last": float(optimizer.param_groups[0]["lr"]),
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
    early_stopping_ema_warmup_epochs: int = 0,
    restore_best: bool = True,
    optimizer_name: str = "adamw",
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    lr_scheduler: str = "cosine",
    lr_warmup_frac: float = 0.05,
    loss_type: str = "huber",
    huber_delta: float = 1.0,
    normalize_by_sigma: bool = False,
    abort_on_nonfinite: bool = True,
    sigma_sample_mode: str = "uniform_log",
    sigma_sample_beta: float = 2.0,
) -> dict[str, float | int | bool | list[float]]:
    if int(early_stopping_ema_warmup_epochs) < 0:
        raise ValueError("early_stopping_ema_warmup_epochs must be >= 0.")
    loader = to_score_loader(theta_train, x_train, batch_size=batch_size, shuffle=True)
    optimizer = _build_optimizer(model, lr=lr, optimizer_name=optimizer_name, weight_decay=weight_decay)
    scheduler = _build_scheduler(
        optimizer,
        scheduler_name=lr_scheduler,
        epochs=epochs,
        warmup_frac=lr_warmup_frac,
    )
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
    ema_warmup = int(early_stopping_ema_warmup_epochs)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("early_stopping_ema_alpha must be in (0, 1].")
    grad_norms: list[float] = []
    n_clipped_steps = 0
    total_steps = 0
    has_nonfinite = False

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
                mode=sigma_sample_mode,
                beta_param=sigma_sample_beta,
            )
            eps = torch.randn_like(tb)
            theta_tilde = tb + sigma * eps
            pred = model(theta_tilde, xb, sigma)
            loss = _score_matching_loss(
                pred,
                eps,
                sigma,
                loss_type=loss_type,
                huber_delta=huber_delta,
                normalize_by_sigma=normalize_by_sigma,
            )
            if not torch.isfinite(loss):
                has_nonfinite = True
                if abort_on_nonfinite:
                    print(f"[nonfinite] score ncsm train loss became non-finite at epoch={epoch}")
                    break
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(max_grad_norm) > 0.0:
                gn_before = _finite_grad_norm(model)
                if np.isfinite(gn_before):
                    grad_norms.append(float(gn_before))
                    if gn_before > float(max_grad_norm):
                        n_clipped_steps += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
            optimizer.step()
            total_steps += 1
            epoch_losses.append(float(loss.item()))
        if has_nonfinite and abort_on_nonfinite:
            stopped_early = True
            stopped_epoch = epoch
            break
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
                        mode=sigma_sample_mode,
                        beta_param=sigma_sample_beta,
                    )
                    eps = torch.randn_like(tb)
                    theta_tilde = tb + sigma * eps
                    pred = model(theta_tilde, xb, sigma)
                    val_loss = _score_matching_loss(
                        pred,
                        eps,
                        sigma,
                        loss_type=loss_type,
                        huber_delta=huber_delta,
                        normalize_by_sigma=normalize_by_sigma,
                    )
                    if not torch.isfinite(val_loss):
                        has_nonfinite = True
                        if abort_on_nonfinite:
                            print(f"[nonfinite] score ncsm val loss became non-finite at epoch={epoch}")
                            break
                        continue
                    val_epoch_losses.append(float(val_loss.item()))
            if has_nonfinite and abort_on_nonfinite:
                stopped_early = True
                stopped_epoch = epoch
                break
            mean_val_loss = float(np.mean(val_epoch_losses))
            smooth_val_loss, val_ema = _early_stop_val_smooth(
                epoch, mean_val_loss, val_ema, alpha, ema_warmup
            )
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
        if scheduler is not None:
            scheduler.step()

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
        "has_nonfinite": bool(has_nonfinite),
        "grad_norm_mean": float(np.nanmean(np.asarray(grad_norms, dtype=np.float64))) if grad_norms else float("nan"),
        "grad_norm_max": float(np.nanmax(np.asarray(grad_norms, dtype=np.float64))) if grad_norms else float("nan"),
        "param_norm_final": float(_param_norm(model)),
        "n_clipped_steps": int(n_clipped_steps),
        "n_total_steps": int(total_steps),
        "lr_last": float(optimizer.param_groups[0]["lr"]),
    }


def train_theta_edm_model(
    model: ConditionalThetaEDM,
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
    early_stopping_ema_warmup_epochs: int = 0,
    restore_best: bool = True,
    optimizer_name: str = "adamw",
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    lr_scheduler: str = "cosine",
    lr_warmup_frac: float = 0.05,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    abort_on_nonfinite: bool = True,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    sigma_data: float = 0.5,
) -> dict[str, float | int | bool | list[float]]:
    if int(early_stopping_ema_warmup_epochs) < 0:
        raise ValueError("early_stopping_ema_warmup_epochs must be >= 0.")
    loader = to_score_loader(theta_train, x_train, batch_size=batch_size, shuffle=True)
    optimizer = _build_optimizer(model, lr=lr, optimizer_name=optimizer_name, weight_decay=weight_decay)
    scheduler = _build_scheduler(
        optimizer,
        scheduler_name=lr_scheduler,
        epochs=epochs,
        warmup_frac=lr_warmup_frac,
    )
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
    ema_warmup = int(early_stopping_ema_warmup_epochs)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("early_stopping_ema_alpha must be in (0, 1].")
    grad_norms: list[float] = []
    n_clipped_steps = 0
    total_steps = 0
    has_nonfinite = False

    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        model.train()
        for tb, xb in loader:
            tb = tb.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            sigma = sample_edm_sigmas(batch_size=tb.shape[0], p_mean=p_mean, p_std=p_std, device=tb.device)
            eps = torch.randn_like(tb)
            theta_tilde = tb + sigma * eps
            denoised = model(theta_tilde, xb, sigma)
            loss = _edm_theta_loss(
                denoised=denoised,
                clean_theta=tb,
                sigma=sigma,
                sigma_data=sigma_data,
                loss_type=loss_type,
                huber_delta=huber_delta,
            )
            if not torch.isfinite(loss):
                has_nonfinite = True
                if abort_on_nonfinite:
                    print(f"[nonfinite] score edm train loss became non-finite at epoch={epoch}")
                    break
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(max_grad_norm) > 0.0:
                gn_before = _finite_grad_norm(model)
                if np.isfinite(gn_before):
                    grad_norms.append(float(gn_before))
                    if gn_before > float(max_grad_norm):
                        n_clipped_steps += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
            optimizer.step()
            total_steps += 1
            epoch_losses.append(float(loss.item()))
        if has_nonfinite and abort_on_nonfinite:
            stopped_early = True
            stopped_epoch = epoch
            break
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
                    sigma = sample_edm_sigmas(
                        batch_size=tb.shape[0],
                        p_mean=p_mean,
                        p_std=p_std,
                        device=tb.device,
                    )
                    eps = torch.randn_like(tb)
                    theta_tilde = tb + sigma * eps
                    denoised = model(theta_tilde, xb, sigma)
                    val_loss = _edm_theta_loss(
                        denoised=denoised,
                        clean_theta=tb,
                        sigma=sigma,
                        sigma_data=sigma_data,
                        loss_type=loss_type,
                        huber_delta=huber_delta,
                    )
                    if not torch.isfinite(val_loss):
                        has_nonfinite = True
                        if abort_on_nonfinite:
                            print(f"[nonfinite] score edm val loss became non-finite at epoch={epoch}")
                            break
                        continue
                    val_epoch_losses.append(float(val_loss.item()))
            if has_nonfinite and abort_on_nonfinite:
                stopped_early = True
                stopped_epoch = epoch
                break
            mean_val_loss = float(np.mean(val_epoch_losses))
            smooth_val_loss, val_ema = _early_stop_val_smooth(
                epoch, mean_val_loss, val_ema, alpha, ema_warmup
            )
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
        if scheduler is not None:
            scheduler.step()

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            if has_val:
                print(
                    f"[epoch {epoch:4d}/{epochs}] edm_train={mean_train_loss:.6f} "
                    f"val_loss={mean_val_loss:.6f} val_smooth={val_monitor_losses[-1]:.6f} "
                    f"best_smooth={best_val_loss:.6f} best_epoch={best_epoch}"
                )
            else:
                print(f"[epoch {epoch:4d}/{epochs}] edm_loss={mean_train_loss:.6f}")

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
        "has_nonfinite": bool(has_nonfinite),
        "grad_norm_mean": float(np.nanmean(np.asarray(grad_norms, dtype=np.float64))) if grad_norms else float("nan"),
        "grad_norm_max": float(np.nanmax(np.asarray(grad_norms, dtype=np.float64))) if grad_norms else float("nan"),
        "param_norm_final": float(_param_norm(model)),
        "n_clipped_steps": int(n_clipped_steps),
        "n_total_steps": int(total_steps),
        "lr_last": float(optimizer.param_groups[0]["lr"]),
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
    early_stopping_ema_warmup_epochs: int = 0,
    restore_best: bool = True,
    optimizer_name: str = "adamw",
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    lr_scheduler: str = "cosine",
    lr_warmup_frac: float = 0.05,
    loss_type: str = "huber",
    huber_delta: float = 1.0,
    normalize_by_sigma: bool = False,
    abort_on_nonfinite: bool = True,
) -> dict[str, float | int | bool | list[float]]:
    if int(early_stopping_ema_warmup_epochs) < 0:
        raise ValueError("early_stopping_ema_warmup_epochs must be >= 0.")
    loader = to_prior_loader(theta_train, batch_size=batch_size, shuffle=True)
    optimizer = _build_optimizer(model, lr=lr, optimizer_name=optimizer_name, weight_decay=weight_decay)
    scheduler = _build_scheduler(
        optimizer,
        scheduler_name=lr_scheduler,
        epochs=epochs,
        warmup_frac=lr_warmup_frac,
    )
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
    ema_warmup = int(early_stopping_ema_warmup_epochs)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("early_stopping_ema_alpha must be in (0, 1].")
    grad_norms: list[float] = []
    n_clipped_steps = 0
    total_steps = 0
    has_nonfinite = False

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
            residual = pred - target
            if normalize_by_sigma:
                residual = sigma * residual
            loss = _loss_reduce(residual, torch.zeros_like(residual), loss_type=loss_type, huber_delta=huber_delta)
            if not torch.isfinite(loss):
                has_nonfinite = True
                if abort_on_nonfinite:
                    print(f"[nonfinite] prior train loss became non-finite at epoch={epoch}")
                    break
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(max_grad_norm) > 0.0:
                gn_before = _finite_grad_norm(model)
                if np.isfinite(gn_before):
                    grad_norms.append(float(gn_before))
                    if gn_before > float(max_grad_norm):
                        n_clipped_steps += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
            optimizer.step()
            total_steps += 1
            epoch_losses.append(float(loss.item()))
        if has_nonfinite and abort_on_nonfinite:
            stopped_early = True
            stopped_epoch = epoch
            break
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
                    residual = pred - target
                    if normalize_by_sigma:
                        residual = sigma * residual
                    val_loss = _loss_reduce(
                        residual,
                        torch.zeros_like(residual),
                        loss_type=loss_type,
                        huber_delta=huber_delta,
                    )
                    if not torch.isfinite(val_loss):
                        has_nonfinite = True
                        if abort_on_nonfinite:
                            print(f"[nonfinite] prior val loss became non-finite at epoch={epoch}")
                            break
                        continue
                    val_epoch_losses.append(float(val_loss.item()))
            if has_nonfinite and abort_on_nonfinite:
                stopped_early = True
                stopped_epoch = epoch
                break
            mean_val_loss = float(np.mean(val_epoch_losses))
            smooth_val_loss, val_ema = _early_stop_val_smooth(
                epoch, mean_val_loss, val_ema, alpha, ema_warmup
            )
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
        if scheduler is not None:
            scheduler.step()

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
        "has_nonfinite": bool(has_nonfinite),
        "grad_norm_mean": float(np.nanmean(np.asarray(grad_norms, dtype=np.float64))) if grad_norms else float("nan"),
        "grad_norm_max": float(np.nanmax(np.asarray(grad_norms, dtype=np.float64))) if grad_norms else float("nan"),
        "param_norm_final": float(_param_norm(model)),
        "n_clipped_steps": int(n_clipped_steps),
        "n_total_steps": int(total_steps),
        "lr_last": float(optimizer.param_groups[0]["lr"]),
    }


def train_prior_theta_edm_model(
    model: PriorThetaEDM,
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
    early_stopping_ema_warmup_epochs: int = 0,
    restore_best: bool = True,
    optimizer_name: str = "adamw",
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    lr_scheduler: str = "cosine",
    lr_warmup_frac: float = 0.05,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    abort_on_nonfinite: bool = True,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    sigma_data: float = 0.5,
) -> dict[str, float | int | bool | list[float]]:
    if int(early_stopping_ema_warmup_epochs) < 0:
        raise ValueError("early_stopping_ema_warmup_epochs must be >= 0.")
    loader = to_prior_loader(theta_train, batch_size=batch_size, shuffle=True)
    optimizer = _build_optimizer(model, lr=lr, optimizer_name=optimizer_name, weight_decay=weight_decay)
    scheduler = _build_scheduler(
        optimizer,
        scheduler_name=lr_scheduler,
        epochs=epochs,
        warmup_frac=lr_warmup_frac,
    )
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
    ema_warmup = int(early_stopping_ema_warmup_epochs)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("early_stopping_ema_alpha must be in (0, 1].")
    grad_norms: list[float] = []
    n_clipped_steps = 0
    total_steps = 0
    has_nonfinite = False

    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        model.train()
        for (tb,) in loader:
            tb = tb.to(device, non_blocking=True)
            sigma = sample_edm_sigmas(batch_size=tb.shape[0], p_mean=p_mean, p_std=p_std, device=tb.device)
            eps = torch.randn_like(tb)
            theta_tilde = tb + sigma * eps
            denoised = model(theta_tilde, sigma)
            loss = _edm_theta_loss(
                denoised=denoised,
                clean_theta=tb,
                sigma=sigma,
                sigma_data=sigma_data,
                loss_type=loss_type,
                huber_delta=huber_delta,
            )
            if not torch.isfinite(loss):
                has_nonfinite = True
                if abort_on_nonfinite:
                    print(f"[nonfinite] prior edm train loss became non-finite at epoch={epoch}")
                    break
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(max_grad_norm) > 0.0:
                gn_before = _finite_grad_norm(model)
                if np.isfinite(gn_before):
                    grad_norms.append(float(gn_before))
                    if gn_before > float(max_grad_norm):
                        n_clipped_steps += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
            optimizer.step()
            total_steps += 1
            epoch_losses.append(float(loss.item()))
        if has_nonfinite and abort_on_nonfinite:
            stopped_early = True
            stopped_epoch = epoch
            break
        mean_train_loss = float(np.mean(epoch_losses))
        train_losses.append(mean_train_loss)

        mean_val_loss = float("nan")
        if has_val and val_loader is not None:
            model.eval()
            val_epoch_losses: list[float] = []
            with torch.no_grad():
                for (tb,) in val_loader:
                    tb = tb.to(device, non_blocking=True)
                    sigma = sample_edm_sigmas(
                        batch_size=tb.shape[0],
                        p_mean=p_mean,
                        p_std=p_std,
                        device=tb.device,
                    )
                    eps = torch.randn_like(tb)
                    theta_tilde = tb + sigma * eps
                    denoised = model(theta_tilde, sigma)
                    val_loss = _edm_theta_loss(
                        denoised=denoised,
                        clean_theta=tb,
                        sigma=sigma,
                        sigma_data=sigma_data,
                        loss_type=loss_type,
                        huber_delta=huber_delta,
                    )
                    if not torch.isfinite(val_loss):
                        has_nonfinite = True
                        if abort_on_nonfinite:
                            print(f"[nonfinite] prior edm val loss became non-finite at epoch={epoch}")
                            break
                        continue
                    val_epoch_losses.append(float(val_loss.item()))
            if has_nonfinite and abort_on_nonfinite:
                stopped_early = True
                stopped_epoch = epoch
                break
            mean_val_loss = float(np.mean(val_epoch_losses))
            smooth_val_loss, val_ema = _early_stop_val_smooth(
                epoch, mean_val_loss, val_ema, alpha, ema_warmup
            )
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
        if scheduler is not None:
            scheduler.step()

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            if has_val:
                print(
                    f"[prior edm {epoch:4d}/{epochs}] train={mean_train_loss:.6f} "
                    f"val_loss={mean_val_loss:.6f} val_smooth={val_monitor_losses[-1]:.6f} "
                    f"best_smooth={best_val_loss:.6f} best_epoch={best_epoch}"
                )
            else:
                print(f"[prior edm {epoch:4d}/{epochs}] train={mean_train_loss:.6f}")

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
        "has_nonfinite": bool(has_nonfinite),
        "grad_norm_mean": float(np.nanmean(np.asarray(grad_norms, dtype=np.float64))) if grad_norms else float("nan"),
        "grad_norm_max": float(np.nanmax(np.asarray(grad_norms, dtype=np.float64))) if grad_norms else float("nan"),
        "param_norm_final": float(_param_norm(model)),
        "n_clipped_steps": int(n_clipped_steps),
        "n_total_steps": int(total_steps),
        "lr_last": float(optimizer.param_groups[0]["lr"]),
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
    early_stopping_ema_warmup_epochs: int = 0,
    restore_best: bool = True,
    optimizer_name: str = "adamw",
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    lr_scheduler: str = "cosine",
    lr_warmup_frac: float = 0.05,
    loss_type: str = "huber",
    huber_delta: float = 1.0,
    normalize_by_sigma: bool = False,
    abort_on_nonfinite: bool = True,
    sigma_sample_mode: str = "uniform_log",
    sigma_sample_beta: float = 2.0,
) -> dict[str, float | int | bool | list[float]]:
    if int(early_stopping_ema_warmup_epochs) < 0:
        raise ValueError("early_stopping_ema_warmup_epochs must be >= 0.")
    loader = to_prior_loader(theta_train, batch_size=batch_size, shuffle=True)
    optimizer = _build_optimizer(model, lr=lr, optimizer_name=optimizer_name, weight_decay=weight_decay)
    scheduler = _build_scheduler(
        optimizer,
        scheduler_name=lr_scheduler,
        epochs=epochs,
        warmup_frac=lr_warmup_frac,
    )
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
    ema_warmup = int(early_stopping_ema_warmup_epochs)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("early_stopping_ema_alpha must be in (0, 1].")
    grad_norms: list[float] = []
    n_clipped_steps = 0
    total_steps = 0
    has_nonfinite = False

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
                mode=sigma_sample_mode,
                beta_param=sigma_sample_beta,
            )
            eps = torch.randn_like(tb)
            theta_tilde = tb + sigma * eps
            pred = model(theta_tilde, sigma)
            loss = _score_matching_loss(
                pred,
                eps,
                sigma,
                loss_type=loss_type,
                huber_delta=huber_delta,
                normalize_by_sigma=normalize_by_sigma,
            )
            if not torch.isfinite(loss):
                has_nonfinite = True
                if abort_on_nonfinite:
                    print(f"[nonfinite] prior ncsm train loss became non-finite at epoch={epoch}")
                    break
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(max_grad_norm) > 0.0:
                gn_before = _finite_grad_norm(model)
                if np.isfinite(gn_before):
                    grad_norms.append(float(gn_before))
                    if gn_before > float(max_grad_norm):
                        n_clipped_steps += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
            optimizer.step()
            total_steps += 1
            epoch_losses.append(float(loss.item()))
        if has_nonfinite and abort_on_nonfinite:
            stopped_early = True
            stopped_epoch = epoch
            break
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
                        mode=sigma_sample_mode,
                        beta_param=sigma_sample_beta,
                    )
                    eps = torch.randn_like(tb)
                    theta_tilde = tb + sigma * eps
                    pred = model(theta_tilde, sigma)
                    val_loss = _score_matching_loss(
                        pred,
                        eps,
                        sigma,
                        loss_type=loss_type,
                        huber_delta=huber_delta,
                        normalize_by_sigma=normalize_by_sigma,
                    )
                    if not torch.isfinite(val_loss):
                        has_nonfinite = True
                        if abort_on_nonfinite:
                            print(f"[nonfinite] prior ncsm val loss became non-finite at epoch={epoch}")
                            break
                        continue
                    val_epoch_losses.append(float(val_loss.item()))
            if has_nonfinite and abort_on_nonfinite:
                stopped_early = True
                stopped_epoch = epoch
                break
            mean_val_loss = float(np.mean(val_epoch_losses))
            smooth_val_loss, val_ema = _early_stop_val_smooth(
                epoch, mean_val_loss, val_ema, alpha, ema_warmup
            )
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
        if scheduler is not None:
            scheduler.step()

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
        "has_nonfinite": bool(has_nonfinite),
        "grad_norm_mean": float(np.nanmean(np.asarray(grad_norms, dtype=np.float64))) if grad_norms else float("nan"),
        "grad_norm_max": float(np.nanmax(np.asarray(grad_norms, dtype=np.float64))) if grad_norms else float("nan"),
        "param_norm_final": float(_param_norm(model)),
        "n_clipped_steps": int(n_clipped_steps),
        "n_total_steps": int(total_steps),
        "lr_last": float(optimizer.param_groups[0]["lr"]),
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
