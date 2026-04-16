"""
Two-sample Schrödinger-bridge-style path (TwoSB) for CTSM-v.

Minimal port of prob_path_lib.TwoSB used by two-sample conditional time score matching.
"""

from __future__ import annotations

import math

import torch


class TwoSB:
    """
    Minimal version of prob_path_lib.TwoSB for the two-sample CTSM-v case.

    The conditional path is
        x_t = (1 - t) x_0 + t x_1 + sqrt(t (1 - t) * var) * epsilon,
    where x_0 ~ p, x_1 ~ q, epsilon ~ N(0, I).
    """

    def __init__(self, dim: int, var: float = 2.0):
        self.dim = dim
        self.var = var
        self.sigma = math.sqrt(var)
        self.sqrt2 = math.sqrt(2.0)

    def marginal_prob(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        mean = (1.0 - t) * x0 + t * x1
        var = t * (1.0 - t) * self.var
        std = torch.sqrt(var)
        return mean, std, var

    def full_epsilon_target(
        self,
        epsilon: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        factor: float = 1.0,
    ):
        """
        Vector-valued CTSM-v target adapted from prob_path_lib.TwoSB.full_epsilon_target.
        """
        temp1 = torch.sqrt(
            1.0 - 4.0 * t + 4.0 * t**2 + 2.0 * factor * t - 2.0 * factor * t**2
        )
        lambda_t = self.sqrt2 * t * (1.0 - t) / temp1
        temp2 = (1.0 - 2.0 * t) / temp1
        mut_d = x1 - x0

        targets = (
            -temp2 / self.sqrt2
            + temp2 / self.sqrt2 * torch.square(epsilon)
            + self.sqrt2
            * torch.sqrt(t * (1.0 - t))
            / temp1
            / self.sigma
            * epsilon
            * mut_d
        )
        return lambda_t, targets
