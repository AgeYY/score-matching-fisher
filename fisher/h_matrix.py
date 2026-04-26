"""H-matrix estimation from learned posterior/prior theta scores."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from fisher.ctsm_models import PairConditionedTimeScoreNetBase
from fisher.ctsm_objectives import estimate_log_ratio_trapz_pair
from fisher.models import (
    ConditionalScore1D,
    ConditionalScore1DFiLMPerLayer,
    ConditionalThetaFlowVelocity,
    ConditionalThetaFlowVelocityFiLMPerLayer,
    ConditionalThetaFlowVelocityThetaFourierMLP,
    ConditionalXFlowVelocity,
    ConditionalXFlowVelocityFiLMPerLayer,
    ConditionalXFlowVelocityIndependentMLP,
    ConditionalXFlowVelocityIndependentThetaFourierMLP,
    ConditionalXFlowVelocityThetaFourierFiLMPerLayer,
    ConditionalXFlowVelocityThetaFourierMLP,
    PriorScore1D,
    PriorScore1DFiLMPerLayer,
    PriorThetaFlowVelocity,
    PriorThetaFlowVelocityFiLMPerLayer,
    PriorThetaFlowVelocityThetaFourierMLP,
)


@dataclass
class HMatrixResult:
    theta_used: np.ndarray
    theta_sorted: np.ndarray
    perm: np.ndarray
    inv_perm: np.ndarray
    g_matrix: np.ndarray
    c_matrix: np.ndarray
    delta_l_matrix: np.ndarray
    h_directed: np.ndarray
    h_sym: np.ndarray
    sigma_eval: float
    field_method: str
    eval_scalar_name: str
    order_mode: str
    delta_diag_max_abs: float
    h_sym_max_asym_abs: float
    flow_scheduler: str | None
    flow_score_mode: str | None
    theta_flow_log_post_matrix: np.ndarray | None
    theta_flow_log_prior_matrix: np.ndarray | None


def _make_flow_matching_path(scheduler_name: str) -> Any:
    name = str(scheduler_name).strip().lower()
    try:
        from flow_matching.path import AffineProbPath
        from flow_matching.path.scheduler import CosineScheduler, LinearVPScheduler, VPScheduler
    except ImportError as e:
        raise ImportError(
            "Flow score conversion requires the `flow_matching` package. "
            "Install it in your environment before using --theta-field-method theta_path_integral."
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


def _make_flow_ode_solver(velocity_model: Any) -> Any:
    try:
        from flow_matching.solver.ode_solver import ODESolver
    except ImportError as e:
        raise ImportError(
            "Flow likelihood estimation requires the `flow_matching` package. "
            "Install it in your environment before using flow ODE likelihood methods "
            "(theta_flow / flow_x_likelihood)."
        ) from e
    return ODESolver(velocity_model=velocity_model)


class HMatrixEstimator:
    """Estimate directed and symmetric H-matrices from score models."""

    def __init__(
        self,
        *,
        model_post: ConditionalScore1D
        | ConditionalScore1DFiLMPerLayer
        | ConditionalThetaFlowVelocity
        | ConditionalThetaFlowVelocityFiLMPerLayer
        | ConditionalThetaFlowVelocityThetaFourierMLP
        | ConditionalXFlowVelocity
        | ConditionalXFlowVelocityFiLMPerLayer
        | ConditionalXFlowVelocityIndependentMLP
        | ConditionalXFlowVelocityIndependentThetaFourierMLP
        | ConditionalXFlowVelocityThetaFourierFiLMPerLayer
        | ConditionalXFlowVelocityThetaFourierMLP
        | PairConditionedTimeScoreNetBase,
        model_prior: PriorScore1D
        | PriorScore1DFiLMPerLayer
        | PriorThetaFlowVelocity
        | PriorThetaFlowVelocityFiLMPerLayer
        | PriorThetaFlowVelocityThetaFourierMLP
        | None = None,
        sigma_eval: float,
        device: torch.device,
        pair_batch_size: int = 65536,
        field_method: str = "dsm",
        flow_scheduler: str = "cosine",
        flow_ode_steps: int = 64,
        ctsm_int_n_time: int = 300,
        ctsm_t_eps: float = 1e-5,
    ) -> None:
        if pair_batch_size < 1:
            raise ValueError("pair_batch_size must be >= 1.")
        self.model_post = model_post
        self.model_prior = model_prior
        self.sigma_eval = float(sigma_eval)
        self.device = device
        self.pair_batch_size = int(pair_batch_size)
        method = str(field_method).strip().lower()
        if method not in ("dsm", "theta_path_integral", "theta_flow", "flow_x_likelihood", "ctsm_v"):
            raise ValueError(
                "field_method must be one of "
                "{'dsm', 'theta_path_integral', 'theta_flow', 'flow_x_likelihood', 'ctsm_v'}."
            )
        if method == "dsm" and sigma_eval <= 0.0:
            raise ValueError("sigma_eval must be positive for DSM mode.")
        if method in ("theta_path_integral", "theta_flow", "flow_x_likelihood") and not (0.0 <= sigma_eval <= 1.0):
            raise ValueError("For flow-based methods, t_eval (passed via sigma_eval) must be in [0, 1].")
        if int(flow_ode_steps) < 2:
            raise ValueError("flow_ode_steps must be >= 2.")
        if method == "flow_x_likelihood":
            if model_prior is not None:
                raise ValueError("flow_x_likelihood expects model_prior=None.")
            if not isinstance(
                model_post,
                (
                    ConditionalXFlowVelocity,
                    ConditionalXFlowVelocityFiLMPerLayer,
                    ConditionalXFlowVelocityIndependentMLP,
                    ConditionalXFlowVelocityIndependentThetaFourierMLP,
                    ConditionalXFlowVelocityThetaFourierFiLMPerLayer,
                    ConditionalXFlowVelocityThetaFourierMLP,
                ),
            ):
                raise TypeError(
                    "flow_x_likelihood requires model_post to be ConditionalXFlowVelocity, "
                    "ConditionalXFlowVelocityFiLMPerLayer, "
                    "ConditionalXFlowVelocityIndependentMLP, "
                    "ConditionalXFlowVelocityIndependentThetaFourierMLP, "
                    "ConditionalXFlowVelocityThetaFourierFiLMPerLayer, or ConditionalXFlowVelocityThetaFourierMLP."
                )
        elif method == "ctsm_v":
            if model_prior is not None:
                raise ValueError("ctsm_v expects model_prior=None.")
            if not isinstance(model_post, PairConditionedTimeScoreNetBase):
                raise TypeError("ctsm_v requires model_post to be a PairConditionedTimeScoreNetBase subclass.")
        elif model_prior is None:
            raise ValueError(f"field_method={method!r} requires a non-None model_prior.")
        self.field_method = method
        self.flow_scheduler = str(flow_scheduler).strip().lower()
        self.flow_score_mode = "velocity_to_epsilon" if self.field_method == "theta_path_integral" else None
        self.flow_ode_steps = int(flow_ode_steps)
        if int(ctsm_int_n_time) < 2:
            raise ValueError("ctsm_int_n_time must be >= 2.")
        if not (0.0 <= float(ctsm_t_eps) < 0.5):
            raise ValueError("ctsm_t_eps must be in [0, 0.5).")
        self.ctsm_int_n_time = int(ctsm_int_n_time)
        self.ctsm_t_eps = float(ctsm_t_eps)
        self.flow_likelihood_method = "midpoint"
        self._flow_path = (
            _make_flow_matching_path(self.flow_scheduler)
            if self.field_method in ("theta_path_integral", "theta_flow")
            else None
        )
        self._flow_likelihood_solver_post = None
        self._flow_likelihood_solver_prior = None
        self._flow_x_likelihood_solver = None
        self._theta_flow_log_post_matrix: np.ndarray | None = None
        self._theta_flow_log_prior_matrix: np.ndarray | None = None
        if self.field_method == "theta_flow":
            self._flow_likelihood_solver_post = _make_flow_ode_solver(self._post_velocity_for_likelihood)
            self._flow_likelihood_solver_prior = _make_flow_ode_solver(self._prior_velocity_for_likelihood)
        if self.field_method == "flow_x_likelihood":
            self._flow_x_likelihood_solver = _make_flow_ode_solver(self._x_post_velocity_for_likelihood)

    def _velocity_to_score(self, velocity: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self._flow_path is None:
            raise RuntimeError("_velocity_to_score called outside flow mode.")
        try:
            eps = self._flow_path.velocity_to_epsilon(velocity=velocity, x_t=x_t, t=t)
        except TypeError:
            eps = self._flow_path.velocity_to_epsilon(velocity, x_t, t)
        schedule_state = self._flow_path.scheduler(t)
        sigma_t = getattr(schedule_state, "sigma_t", None)
        if sigma_t is None:
            raise RuntimeError("Flow scheduler state is missing sigma_t for velocity-to-score conversion.")
        while sigma_t.ndim < x_t.ndim:
            sigma_t = sigma_t.unsqueeze(-1)
        return -eps / torch.clamp(sigma_t, min=1e-8)

    @staticmethod
    def _standard_normal_log_prob(theta: torch.Tensor) -> torch.Tensor:
        theta_flat = theta.reshape(theta.shape[0], -1)
        return -0.5 * (theta_flat.pow(2).sum(dim=1) + theta_flat.shape[1] * math.log(2.0 * math.pi))

    @staticmethod
    def _time_to_batch_column(t: torch.Tensor | float, ref: torch.Tensor) -> torch.Tensor:
        t_tensor = t if torch.is_tensor(t) else torch.tensor(float(t), device=ref.device, dtype=ref.dtype)
        t_tensor = t_tensor.to(device=ref.device, dtype=ref.dtype)
        batch = int(ref.shape[0])
        if t_tensor.ndim == 0:
            return t_tensor.expand(batch).unsqueeze(-1)
        if t_tensor.ndim == 1:
            if t_tensor.shape[0] == 1:
                return t_tensor.expand(batch).unsqueeze(-1)
            if t_tensor.shape[0] != batch:
                raise ValueError("ODE solver provided 1D time tensor with mismatched batch size.")
            return t_tensor.unsqueeze(-1)
        if t_tensor.ndim == 2:
            if t_tensor.shape[0] == 1:
                return t_tensor.expand(batch, t_tensor.shape[1])
            if t_tensor.shape[0] != batch:
                raise ValueError("ODE solver provided 2D time tensor with mismatched batch size.")
            return t_tensor
        raise ValueError("Unsupported time tensor rank from ODE solver.")

    def _post_velocity_for_likelihood(self, x: torch.Tensor, t: torch.Tensor, **model_extras: Any) -> torch.Tensor:
        x_cond = model_extras.get("x_cond", None)
        if x_cond is None:
            raise ValueError("theta_flow posterior ODE call requires model_extras['x_cond'].")
        return self.model_post(x, x_cond, self._time_to_batch_column(t, x))

    def _prior_velocity_for_likelihood(self, x: torch.Tensor, t: torch.Tensor, **model_extras: Any) -> torch.Tensor:
        _ = model_extras
        if self.model_prior is None:
            raise RuntimeError("_prior_velocity_for_likelihood called without model_prior.")
        return self.model_prior(x, self._time_to_batch_column(t, x))

    def _x_post_velocity_for_likelihood(self, x: torch.Tensor, t: torch.Tensor, **model_extras: Any) -> torch.Tensor:
        """ODE state is ``x_t``; ``ODESolver`` invokes ``velocity_model(x=..., t=..., **extras)``."""
        theta_cond = model_extras.get("theta_cond", None)
        if theta_cond is None:
            raise ValueError("flow_x_likelihood ODE call requires model_extras['theta_cond'].")
        return self.model_post(x, theta_cond, self._time_to_batch_column(t, x))

    @staticmethod
    def _theta_as_matrix(theta: np.ndarray) -> np.ndarray:
        th = np.asarray(theta, dtype=np.float64)
        if th.ndim == 1:
            return th.reshape(-1, 1)
        if th.ndim != 2:
            raise ValueError("theta must be 1D or 2D.")
        return th

    @staticmethod
    def _theta_is_one_hot(theta_mat: np.ndarray, *, atol: float = 1e-6) -> bool:
        th = np.asarray(theta_mat, dtype=np.float64)
        if th.ndim != 2 or th.shape[1] < 2:
            return False
        row_sum_ok = np.allclose(np.sum(th, axis=1), 1.0, atol=atol, rtol=0.0)
        in_range = bool(np.all(th >= -atol) and np.all(th <= 1.0 + atol))
        near_binary = bool(np.all((np.abs(th) <= atol) | (np.abs(th - 1.0) <= atol)))
        return bool(row_sum_ok and in_range and near_binary)

    @staticmethod
    def sort_by_theta(theta: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        theta_col = HMatrixEstimator._theta_as_matrix(theta)
        x2 = np.asarray(x, dtype=np.float64)
        if x2.ndim != 2:
            raise ValueError("x must be a 2D array.")
        if theta_col.shape[0] != x2.shape[0]:
            raise ValueError("theta and x must have the same number of rows.")
        if theta_col.shape[1] == 1:
            perm = np.argsort(theta_col[:, 0], kind="mergesort")
        elif HMatrixEstimator._theta_is_one_hot(theta_col):
            # One-hot states: deterministic order by active bin index.
            perm = np.argsort(np.argmax(theta_col, axis=1), kind="mergesort")
        else:
            # General multidim theta: lexicographic column order.
            keys = [theta_col[:, j] for j in range(theta_col.shape[1] - 1, -1, -1)]
            perm = np.lexsort(keys)
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(perm.size, dtype=perm.dtype)
        theta_sorted = theta_col[perm]
        x_sorted = x2[perm]
        return theta_sorted, x_sorted, perm, inv_perm

    def compute_g_matrix(self, theta_sorted: np.ndarray, x_sorted: np.ndarray) -> np.ndarray:
        theta_grid_col = self._theta_as_matrix(theta_sorted)
        if theta_grid_col.shape[1] != 1:
            raise ValueError("compute_g_matrix requires scalar theta (shape (N,1)).")
        n = int(theta_grid_col.shape[0])
        if n < 1:
            raise ValueError("Need at least one sample to compute H-matrix.")
        row_block = max(1, int(self.pair_batch_size // n))
        theta_grid_col = np.asarray(theta_grid_col, dtype=np.float32)
        g = np.zeros((n, n), dtype=np.float64)
        self.model_post.eval()
        if self.model_prior is not None:
            self.model_prior.eval()
        with torch.no_grad():
            for i0 in range(0, n, row_block):
                i1 = min(n, i0 + row_block)
                xb = np.asarray(x_sorted[i0:i1], dtype=np.float32)
                b = int(i1 - i0)
                theta_tile = np.tile(theta_grid_col, (b, 1))
                x_rep = np.repeat(xb, repeats=n, axis=0)
                theta_t = torch.from_numpy(theta_tile).to(self.device)
                x_t = torch.from_numpy(x_rep).to(self.device)
                if self.field_method == "theta_path_integral":
                    t_eval = torch.full(
                        (theta_t.shape[0], 1),
                        fill_value=float(self.sigma_eval),
                        dtype=theta_t.dtype,
                        device=theta_t.device,
                    )
                    v_post = self.model_post.predict_velocity(theta_t, x_t, t_eval=float(self.sigma_eval))
                    if self.model_prior is None:
                        raise RuntimeError("compute_g_matrix requires model_prior.")
                    v_prior = self.model_prior.predict_velocity(theta_t, t_eval=float(self.sigma_eval))
                    f_post = self._velocity_to_score(v_post, theta_t, t_eval).cpu().numpy().reshape(b, n)
                    f_prior = self._velocity_to_score(v_prior, theta_t, t_eval).cpu().numpy().reshape(b, n)
                else:
                    f_post = (
                        self.model_post.predict_score(theta_t, x_t, sigma_eval=self.sigma_eval)
                        .cpu()
                        .numpy()
                        .reshape(b, n)
                    )
                    if self.model_prior is None:
                        raise RuntimeError("compute_g_matrix requires model_prior.")
                    f_prior = (
                        self.model_prior.predict_score(theta_t, sigma_eval=self.sigma_eval)
                        .cpu()
                        .numpy()
                        .reshape(b, n)
                    )
                g[i0:i1, :] = (f_post - f_prior).astype(np.float64)
        return g

    def compute_log_ratio_matrix(self, theta_sorted: np.ndarray, x_sorted: np.ndarray) -> np.ndarray:
        """Directly estimate per-pair log-likelihood ratio matrix via flow ODE likelihoods.

        r[i,j] = log p(theta_j | x_i) - log p(theta_j).
        """
        theta_grid_col = self._theta_as_matrix(theta_sorted)
        n = int(theta_grid_col.shape[0])
        if n < 1:
            raise ValueError("Need at least one sample to compute H-matrix.")
        row_block = max(1, int(self.pair_batch_size // n))
        theta_grid_col = np.asarray(theta_grid_col, dtype=np.float32)
        r = np.zeros((n, n), dtype=np.float64)
        log_post_matrix = np.zeros((n, n), dtype=np.float64)
        log_prior_matrix = np.zeros((n, n), dtype=np.float64)
        self.model_post.eval()
        if self.model_prior is not None:
            self.model_prior.eval()
        if self._flow_likelihood_solver_post is None or self._flow_likelihood_solver_prior is None:
            raise RuntimeError("theta_flow ODE solvers are not initialized.")
        for i0 in range(0, n, row_block):
            i1 = min(n, i0 + row_block)
            xb = np.asarray(x_sorted[i0:i1], dtype=np.float32)
            b = int(i1 - i0)
            theta_tile = np.tile(theta_grid_col, (b, 1))
            x_rep = np.repeat(xb, repeats=n, axis=0)
            theta_t = torch.from_numpy(theta_tile).to(self.device)
            x_t = torch.from_numpy(x_rep).to(self.device)
            time_grid = torch.linspace(1.0, 0.0, self.flow_ode_steps + 1, device=theta_t.device, dtype=theta_t.dtype)
            _, log_post = self._flow_likelihood_solver_post.compute_likelihood(
                x_1=theta_t,
                log_p0=self._standard_normal_log_prob,
                step_size=None,
                method=self.flow_likelihood_method,
                time_grid=time_grid,
                exact_divergence=False,
                enable_grad=False,
                x_cond=x_t,
            )
            _, log_prior = self._flow_likelihood_solver_prior.compute_likelihood(
                x_1=theta_t,
                log_p0=self._standard_normal_log_prob,
                step_size=None,
                method=self.flow_likelihood_method,
                time_grid=time_grid,
                exact_divergence=False,
                enable_grad=False,
            )
            log_post_block = log_post.reshape(b, n).detach().cpu().numpy().astype(np.float64)
            log_prior_block = log_prior.reshape(b, n).detach().cpu().numpy().astype(np.float64)
            log_post_matrix[i0:i1, :] = log_post_block
            log_prior_matrix[i0:i1, :] = log_prior_block
            r[i0:i1, :] = log_post_block - log_prior_block
        self._theta_flow_log_post_matrix = log_post_matrix
        self._theta_flow_log_prior_matrix = log_prior_matrix
        return r

    def compute_x_conditional_loglik_matrix(self, theta_sorted: np.ndarray, x_sorted: np.ndarray) -> np.ndarray:
        """Estimate C_ij = log p(x_i | theta_j) via conditional x-flow ODE likelihood (one solver call per block)."""
        theta_grid_col = self._theta_as_matrix(theta_sorted)
        if theta_grid_col.shape[1] != 1:
            raise ValueError("compute_x_conditional_loglik_matrix requires scalar theta (shape (N,1)).")
        n = int(theta_grid_col.shape[0])
        if n < 1:
            raise ValueError("Need at least one sample to compute H-matrix.")
        row_block = max(1, int(self.pair_batch_size // n))
        theta_grid_col = np.asarray(theta_grid_col, dtype=np.float32)
        c = np.zeros((n, n), dtype=np.float64)
        self.model_post.eval()
        if self._flow_x_likelihood_solver is None:
            raise RuntimeError("flow_x_likelihood ODE solver is not initialized.")
        for i0 in range(0, n, row_block):
            i1 = min(n, i0 + row_block)
            xb = np.asarray(x_sorted[i0:i1], dtype=np.float32)
            b = int(i1 - i0)
            x_rep = np.repeat(xb, repeats=n, axis=0)
            theta_tile = np.tile(theta_grid_col, (b, 1))
            x_t = torch.from_numpy(x_rep).to(self.device)
            theta_t = torch.from_numpy(theta_tile).to(self.device)
            time_grid = torch.linspace(1.0, 0.0, self.flow_ode_steps + 1, device=x_t.device, dtype=x_t.dtype)
            _, log_p = self._flow_x_likelihood_solver.compute_likelihood(
                x_1=x_t,
                log_p0=self._standard_normal_log_prob,
                step_size=None,
                method=self.flow_likelihood_method,
                time_grid=time_grid,
                exact_divergence=False,
                enable_grad=False,
                theta_cond=theta_t,
            )
            c[i0:i1, :] = log_p.reshape(b, n).detach().cpu().numpy().astype(np.float64)
        return c

    def compute_ctsm_delta_l_matrix(self, theta_sorted: np.ndarray, x_sorted: np.ndarray) -> np.ndarray:
        """Estimate direct DeltaL_ij = log p(x_i|theta_j) - log p(x_i|theta_i) via pair-conditioned CTSM-v."""
        theta_grid_col = self._theta_as_matrix(theta_sorted)
        if theta_grid_col.shape[1] != 1:
            raise ValueError("compute_ctsm_delta_l_matrix requires scalar theta (shape (N,1)).")
        n = int(theta_grid_col.shape[0])
        if n < 1:
            raise ValueError("Need at least one sample to compute H-matrix.")
        row_block = max(1, int(self.pair_batch_size // n))
        theta_grid_col = np.asarray(theta_grid_col, dtype=np.float32)
        delta_l = np.zeros((n, n), dtype=np.float64)
        self.model_post.eval()
        with torch.no_grad():
            for i0 in range(0, n, row_block):
                i1 = min(n, i0 + row_block)
                xb = np.asarray(x_sorted[i0:i1], dtype=np.float32)
                theta_i = np.asarray(theta_sorted[i0:i1], dtype=np.float32).reshape(-1, 1)
                b = int(i1 - i0)
                x_rep = np.repeat(xb, repeats=n, axis=0)
                a_rep = np.repeat(theta_i, repeats=n, axis=0)
                b_rep = np.tile(theta_grid_col, (b, 1))
                x_t = torch.from_numpy(x_rep).to(self.device)
                a_t = torch.from_numpy(a_rep.reshape(-1)).to(self.device)
                b_t = torch.from_numpy(b_rep.reshape(-1)).to(self.device)
                delta_block = estimate_log_ratio_trapz_pair(
                    self.model_post,
                    x_t,
                    a_t,
                    b_t,
                    eps1=self.ctsm_t_eps,
                    eps2=self.ctsm_t_eps,
                    n_time=self.ctsm_int_n_time,
                )
                delta_l[i0:i1, :] = delta_block.reshape(b, n).detach().cpu().numpy().astype(np.float64)
        np.fill_diagonal(delta_l, 0.0)
        return delta_l

    @staticmethod
    def compute_c_matrix(theta_sorted: np.ndarray, g_matrix: np.ndarray) -> np.ndarray:
        theta_arr = np.asarray(theta_sorted, dtype=np.float64)
        if theta_arr.ndim == 2 and theta_arr.shape[1] != 1:
            raise ValueError("compute_c_matrix requires scalar theta (shape (N,1)).")
        theta_flat = theta_arr.reshape(-1)
        if g_matrix.ndim != 2 or g_matrix.shape[0] != theta_flat.size or g_matrix.shape[1] != theta_flat.size:
            raise ValueError("g_matrix must have shape (N, N) matching theta_sorted.")
        dtheta = np.diff(theta_flat).reshape(1, -1)
        trapezoids = 0.5 * dtheta * (g_matrix[:, :-1] + g_matrix[:, 1:])
        c = np.zeros_like(g_matrix, dtype=np.float64)
        if trapezoids.shape[1] > 0:
            c[:, 1:] = np.cumsum(trapezoids, axis=1)
        return c

    @staticmethod
    def compute_delta_l(c_matrix: np.ndarray) -> np.ndarray:
        if c_matrix.ndim != 2 or c_matrix.shape[0] != c_matrix.shape[1]:
            raise ValueError("c_matrix must be square.")
        diag = np.diag(c_matrix).reshape(-1, 1)
        return c_matrix - diag

    @staticmethod
    def compute_h_directed(delta_l_matrix: np.ndarray) -> np.ndarray:
        z = 0.5 * np.asarray(delta_l_matrix, dtype=np.float64)
        z = np.clip(z, -60.0, 60.0)
        h_directed = 1.0 - (1.0 / np.cosh(z))
        np.fill_diagonal(h_directed, 0.0)
        return h_directed

    @staticmethod
    def symmetrize(h_directed: np.ndarray) -> np.ndarray:
        return 0.5 * (h_directed + h_directed.T)

    @staticmethod
    def _permute_back(mat_sorted: np.ndarray, inv_perm: np.ndarray) -> np.ndarray:
        return mat_sorted[np.ix_(inv_perm, inv_perm)]

    def run(self, theta: np.ndarray, x: np.ndarray, *, restore_original_order: bool = False) -> HMatrixResult:
        theta_col = self._theta_as_matrix(theta)
        theta_sorted, x_sorted, perm, inv_perm = self.sort_by_theta(theta_col, x)
        if theta_sorted.shape[1] == 1 and np.any(np.diff(theta_sorted[:, 0]) < 0.0):
            raise ValueError("sort_by_theta produced a non-monotone theta sequence.")

        if self.field_method == "theta_flow":
            c_sorted = self.compute_log_ratio_matrix(theta_sorted, x_sorted)
            delta_sorted = self.compute_delta_l(c_sorted)
            g_sorted = np.zeros_like(c_sorted, dtype=np.float64)
            theta_flow_log_post_sorted = self._theta_flow_log_post_matrix
            theta_flow_log_prior_sorted = self._theta_flow_log_prior_matrix
        elif self.field_method == "flow_x_likelihood":
            c_sorted = self.compute_x_conditional_loglik_matrix(theta_sorted, x_sorted)
            delta_sorted = self.compute_delta_l(c_sorted)
            g_sorted = np.zeros_like(c_sorted, dtype=np.float64)
            theta_flow_log_post_sorted = None
            theta_flow_log_prior_sorted = None
        elif self.field_method == "ctsm_v":
            delta_sorted = self.compute_ctsm_delta_l_matrix(theta_sorted, x_sorted)
            c_sorted = np.asarray(delta_sorted, dtype=np.float64)
            g_sorted = np.zeros_like(c_sorted, dtype=np.float64)
            theta_flow_log_post_sorted = None
            theta_flow_log_prior_sorted = None
        else:
            g_sorted = self.compute_g_matrix(theta_sorted, x_sorted)
            c_sorted = self.compute_c_matrix(theta_sorted, g_sorted)
            delta_sorted = self.compute_delta_l(c_sorted)
            theta_flow_log_post_sorted = None
            theta_flow_log_prior_sorted = None
        h_dir_sorted = self.compute_h_directed(delta_sorted)
        h_sym_sorted = self.symmetrize(h_dir_sorted)

        delta_diag_max_abs = float(np.max(np.abs(np.diag(delta_sorted))))
        h_sym_max_asym_abs = float(np.max(np.abs(h_sym_sorted - h_sym_sorted.T)))

        if restore_original_order:
            theta_used = theta_col.reshape(-1) if theta_col.shape[1] == 1 else theta_col.copy()
            g_used = self._permute_back(g_sorted, inv_perm)
            c_used = self._permute_back(c_sorted, inv_perm)
            delta_used = self._permute_back(delta_sorted, inv_perm)
            h_dir_used = self._permute_back(h_dir_sorted, inv_perm)
            h_sym_used = self._permute_back(h_sym_sorted, inv_perm)
            theta_flow_log_post_used = (
                self._permute_back(theta_flow_log_post_sorted, inv_perm)
                if theta_flow_log_post_sorted is not None
                else None
            )
            theta_flow_log_prior_used = (
                self._permute_back(theta_flow_log_prior_sorted, inv_perm)
                if theta_flow_log_prior_sorted is not None
                else None
            )
            order_mode = "original"
        else:
            theta_used = theta_sorted.reshape(-1) if theta_sorted.shape[1] == 1 else theta_sorted.copy()
            g_used = g_sorted
            c_used = c_sorted
            delta_used = delta_sorted
            h_dir_used = h_dir_sorted
            h_sym_used = h_sym_sorted
            theta_flow_log_post_used = theta_flow_log_post_sorted
            theta_flow_log_prior_used = theta_flow_log_prior_sorted
            order_mode = "sorted"

        return HMatrixResult(
            theta_used=theta_used,
            theta_sorted=(theta_sorted.reshape(-1) if theta_sorted.shape[1] == 1 else theta_sorted.copy()),
            perm=perm,
            inv_perm=inv_perm,
            g_matrix=g_used,
            c_matrix=c_used,
            delta_l_matrix=delta_used,
            h_directed=h_dir_used,
            h_sym=h_sym_used,
            sigma_eval=self.sigma_eval,
            field_method=self.field_method,
            eval_scalar_name=(
                "t_eval"
                if self.field_method == "theta_path_integral"
                else (
                    "flow_ode_t_span"
                    if self.field_method in ("theta_flow", "flow_x_likelihood")
                    else ("ctsm_t_eps" if self.field_method == "ctsm_v" else "sigma_eval")
                )
            ),
            order_mode=order_mode,
            delta_diag_max_abs=delta_diag_max_abs,
            h_sym_max_asym_abs=h_sym_max_asym_abs,
            flow_scheduler=(
                self.flow_scheduler
                if self.field_method in ("theta_path_integral", "theta_flow", "flow_x_likelihood")
                else None
            ),
            flow_score_mode=(
                self.flow_score_mode
                if self.flow_score_mode is not None
                else (
                    "direct_ode_likelihood"
                    if self.field_method == "theta_flow"
                    else (
                        "direct_ode_x_cond_likelihood"
                        if self.field_method == "flow_x_likelihood"
                        else (
                            f"pair_conditioned_ctsm_v_trapz_n_time={self.ctsm_int_n_time}"
                            if self.field_method == "ctsm_v"
                            else None
                        )
                    )
                )
            ),
            theta_flow_log_post_matrix=theta_flow_log_post_used,
            theta_flow_log_prior_matrix=theta_flow_log_prior_used,
        )
