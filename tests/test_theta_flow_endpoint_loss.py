from __future__ import annotations

import numpy as np
import torch

from fisher.models import ConditionalThetaFlowVelocity, PriorThetaFlowVelocity
from fisher.trainers import (
    _theta_flow_conditional_nll_aux_loss,
    train_conditional_theta_flow_likelihood_finetune,
    train_conditional_theta_flow_model,
    train_prior_theta_flow_likelihood_finetune,
)


def _make_toy_dataset(n: int = 16) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    theta = rng.standard_normal((n, 1)).astype(np.float32)
    x = rng.standard_normal((n, 2)).astype(np.float32)
    return theta, x


def test_theta_flow_endpoint_nll_is_differentiable() -> None:
    torch.manual_seed(0)
    model = ConditionalThetaFlowVelocity(x_dim=2, hidden_dim=8, depth=1)
    theta = torch.randn(8, 1)
    x_cond = torch.randn(8, 2)
    loss = _theta_flow_conditional_nll_aux_loss(
        model=model,
        theta_target=theta,
        x_cond=x_cond,
        n_steps=4,
        enable_grad=True,
    )
    loss.backward()
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += float(torch.norm(p.grad).item())
    assert grad_norm > 0.0


def test_train_theta_flow_endpoint_loss_disabled_matches_backward_compatible_shape() -> None:
    torch.manual_seed(0)
    theta, x = _make_toy_dataset(n=16)
    model = ConditionalThetaFlowVelocity(x_dim=2, hidden_dim=16, depth=2).to(torch.device("cpu"))
    out = train_conditional_theta_flow_model(
        model=model,
        theta_train=theta,
        x_train=x,
        epochs=2,
        batch_size=8,
        lr=1e-3,
        device=torch.device("cpu"),
        log_every=10,
        endpoint_loss_weight=0.0,
        endpoint_ode_steps=3,
    )
    assert len(out["train_losses"]) == 2
    assert len(out["train_fm_losses"]) == 2
    assert len(out["train_endpoint_losses"]) == 2
    assert all(abs(float(v)) < 1e-12 for v in out["train_endpoint_losses"])
    assert len(out["val_losses"]) == 2
    assert len(out["val_fm_losses"]) == 2
    assert len(out["val_endpoint_losses"]) == 2
    assert all(np.isnan(float(v)) for v in out["val_losses"])


def test_train_theta_flow_endpoint_loss_enabled_tracks_component_losses() -> None:
    torch.manual_seed(0)
    theta, x = _make_toy_dataset(n=20)
    theta_tr, x_tr = theta[:12], x[:12]
    theta_va, x_va = theta[12:], x[12:]
    model = ConditionalThetaFlowVelocity(x_dim=2, hidden_dim=16, depth=2).to(torch.device("cpu"))
    out = train_conditional_theta_flow_model(
        model=model,
        theta_train=theta_tr,
        x_train=x_tr,
        epochs=2,
        batch_size=6,
        lr=1e-3,
        device=torch.device("cpu"),
        log_every=10,
        theta_val=theta_va,
        x_val=x_va,
        endpoint_loss_weight=0.2,
        endpoint_ode_steps=4,
    )
    assert len(out["train_losses"]) == 2
    assert len(out["train_fm_losses"]) == 2
    assert len(out["train_endpoint_losses"]) == 2
    assert np.all(np.isfinite(np.asarray(out["train_endpoint_losses"], dtype=np.float64)))
    assert np.any(np.abs(np.asarray(out["train_endpoint_losses"], dtype=np.float64)) > 1e-6)
    assert len(out["val_losses"]) == 2
    assert len(out["val_fm_losses"]) == 2
    assert len(out["val_endpoint_losses"]) == 2
    assert np.all(np.isfinite(np.asarray(out["val_losses"], dtype=np.float64)))
    assert np.all(np.isfinite(np.asarray(out["val_endpoint_losses"], dtype=np.float64)))


def test_train_theta_flow_likelihood_finetune_tracks_nll_losses() -> None:
    torch.manual_seed(0)
    theta, x = _make_toy_dataset(n=18)
    theta_tr, x_tr = theta[:12], x[:12]
    theta_va, x_va = theta[12:], x[12:]
    model = ConditionalThetaFlowVelocity(x_dim=2, hidden_dim=12, depth=1).to(torch.device("cpu"))
    out = train_conditional_theta_flow_likelihood_finetune(
        model=model,
        theta_train=theta_tr,
        x_train=x_tr,
        epochs=2,
        batch_size=6,
        lr=1e-4,
        device=torch.device("cpu"),
        log_every=10,
        theta_val=theta_va,
        x_val=x_va,
        ode_steps=3,
    )
    assert len(out["train_losses"]) == 2
    assert len(out["val_losses"]) == 2
    assert len(out["val_monitor_losses"]) == 2
    assert np.all(np.isfinite(np.asarray(out["train_losses"], dtype=np.float64)))
    assert np.all(np.isfinite(np.asarray(out["val_losses"], dtype=np.float64)))


def test_train_prior_theta_flow_likelihood_finetune_tracks_nll_losses() -> None:
    torch.manual_seed(0)
    theta, _ = _make_toy_dataset(n=18)
    theta_tr = theta[:12]
    theta_va = theta[12:]
    model = PriorThetaFlowVelocity(hidden_dim=12, depth=1, theta_dim=1).to(torch.device("cpu"))
    out = train_prior_theta_flow_likelihood_finetune(
        model=model,
        theta_train=theta_tr,
        epochs=2,
        batch_size=6,
        lr=1e-4,
        device=torch.device("cpu"),
        log_every=10,
        theta_val=theta_va,
        ode_steps=3,
    )
    assert len(out["train_losses"]) == 2
    assert len(out["val_losses"]) == 2
    assert len(out["val_monitor_losses"]) == 2
    assert np.all(np.isfinite(np.asarray(out["train_losses"], dtype=np.float64)))
    assert np.all(np.isfinite(np.asarray(out["val_losses"], dtype=np.float64)))
