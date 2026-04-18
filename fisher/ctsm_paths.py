"""
Two-sample affine bridge paths for CTSM-v with pluggable schedules.

Default behavior remains the original linear TwoSB bridge.
"""

from __future__ import annotations

import math
from typing import Protocol

import torch


class Schedule(Protocol):
    def value(self, t: torch.Tensor) -> torch.Tensor:
        """Map physical time t in [0,1] to interpolation clock u in [0,1]."""

    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        """Time derivative du/dt."""


class LinearScheduler:
    def value(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)


class CosineScheduler:
    def value(self, t: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 - torch.cos(math.pi * t))

    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        return 0.5 * math.pi * torch.sin(math.pi * t)


SCHEDULER_REGISTRY: dict[str, type[Schedule]] = {
    "linear": LinearScheduler,
    "cosine": CosineScheduler,
}


def build_scheduler(scheduler: str | Schedule) -> Schedule:
    if isinstance(scheduler, str):
        key = scheduler.strip().lower()
        if key not in SCHEDULER_REGISTRY:
            supported = ", ".join(sorted(SCHEDULER_REGISTRY.keys()))
            raise ValueError(f"Unknown CTSM path scheduler {scheduler!r}. Supported: {supported}.")
        return SCHEDULER_REGISTRY[key]()
    if not hasattr(scheduler, "value") or not hasattr(scheduler, "derivative"):
        raise TypeError("Scheduler object must implement value(t) and derivative(t).")
    return scheduler


class TwoEndpointBridge:
    """
    General two-endpoint Gaussian affine bridge with schedule u=s(t):

        x_t = (1-u) x_0 + u x_1 + sigma * sqrt(u (1-u)) * epsilon

    where sigma^2 = var and epsilon ~ N(0, I).
    """

    def __init__(
        self,
        dim: int,
        var: float = 2.0,
        scheduler: str | Schedule = "linear",
        eps: float = 1e-12,
    ):
        if float(var) <= 0.0:
            raise ValueError("var must be positive.")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive.")
        self.dim = dim
        self.var = float(var)
        self.sigma = math.sqrt(self.var)
        self.sqrt2 = math.sqrt(2.0)
        self.eps = float(eps)
        self.scheduler = build_scheduler(scheduler)
        self.scheduler_name = scheduler.strip().lower() if isinstance(scheduler, str) else scheduler.__class__.__name__

    @staticmethod
    def _ensure_t_shape(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            return t.unsqueeze(-1)
        return t

    def _schedule_terms(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self._ensure_t_shape(t)
        u = self.scheduler.value(t)
        du_dt = self.scheduler.derivative(t)
        return u, du_dt

    def marginal_prob(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        u, _ = self._schedule_terms(t)
        mean = (1.0 - u) * x0 + u * x1
        var = torch.clamp(u * (1.0 - u) * self.var, min=self.eps)
        std = torch.sqrt(var)
        return mean, std, var

    def raw_vector_target(
        self,
        epsilon: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized conditional time-score target from journal/notes/path.tex:
          - (kdot / 2k)
          + (mudot / sqrt(k)) * epsilon
          + (kdot / 2k) * epsilon^2
        """
        u, du_dt = self._schedule_terms(t)
        u1m = torch.clamp(u * (1.0 - u), min=self.eps)
        one_minus_2u = 1.0 - 2.0 * u

        # k_t = sigma^2 u(1-u), kdot_t = sigma^2 du/dt (1-2u)
        coef = du_dt * one_minus_2u / (2.0 * u1m)
        mu_dot_over_sqrtk = (du_dt / (torch.sqrt(u1m) * self.sigma)) * (x1 - x0)
        return -coef + mu_dot_over_sqrtk * epsilon + coef * torch.square(epsilon)

    def time_score_normalization(self, t: torch.Tensor, factor: float = 1.0) -> torch.Tensor:
        """
        Schedule-generalized CTSM-v weighting:
          lambda_t = sqrt(2) u(1-u) / sqrt((1-2u)^2 + 2*factor*u(1-u))
        """
        u, _ = self._schedule_terms(t)
        u1m = torch.clamp(u * (1.0 - u), min=self.eps)
        denom = torch.sqrt(torch.clamp((1.0 - 2.0 * u) ** 2 + 2.0 * factor * u1m, min=self.eps))
        return self.sqrt2 * u1m / denom

    def full_epsilon_target(
        self,
        epsilon: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        factor: float = 1.0,
    ):
        """
        Vector-valued CTSM-v target:
          (lambda_t, lambda_t * raw_vector_target)
        """
        lambda_t = self.time_score_normalization(t=t, factor=factor)
        targets = lambda_t * self.raw_vector_target(epsilon=epsilon, x0=x0, x1=x1, t=t)
        return lambda_t, targets


# Backward compatibility alias used throughout the codebase.
TwoSB = TwoEndpointBridge
