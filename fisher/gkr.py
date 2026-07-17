"""Torch implementation of Gaussian Process with Kernel Regression (GKR).

Adapted from the authors' ``GKR_demo_torch_colab.ipynb`` at
https://github.com/AgeYY/speed_grid_cell_information (MIT license). GKR fits
the conditional mean with Gaussian processes and estimates a smooth local
covariance by kernel averaging residual outer products.

The estimator supports linear Fisher information, ``J.T @ Sigma^-1 @ J``.
It does not include the covariance-derivative term in full Gaussian Fisher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import gpytorch
import numpy as np
import torch

TensorLike = np.ndarray | torch.Tensor
CircularPeriod = float | Sequence[float | None] | None


def _as_2d(value: TensorLike, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 2:
        raise ValueError(f"Expected a two-dimensional array, got {tuple(tensor.shape)}.")
    return tensor


def _periods(period: CircularPeriod, n_input: int) -> tuple[float | None, ...]:
    if period is None:
        return (None,) * int(n_input)
    if np.isscalar(period):
        return (float(period),) * int(n_input)
    values = tuple(None if value is None else float(value) for value in period)
    if len(values) != int(n_input):
        raise ValueError("circular_period must have one entry per input dimension.")
    return values


def _wrapped_difference(diff: torch.Tensor, period: CircularPeriod) -> torch.Tensor:
    values = _periods(period, int(diff.shape[-1]))
    if all(value is None for value in values):
        return diff
    result = diff.clone()
    for dim, value in enumerate(values):
        if value is not None:
            result[..., dim] = torch.sin(torch.pi * result[..., dim] / value).square()
    return result


def gaussian_residual_log_likelihood(
    residuals: torch.Tensor,
    covariance: torch.Tensor,
    *,
    jitter: float = 1e-5,
) -> torch.Tensor:
    """Average zero-mean Gaussian log likelihood, omitting constants."""

    residuals = torch.as_tensor(residuals, dtype=covariance.dtype, device=covariance.device)
    if residuals.ndim != 2 or covariance.ndim != 3:
        raise ValueError("residuals must be [n, d] and covariance must be [n, d, d].")
    if covariance.shape != (residuals.shape[0], residuals.shape[1], residuals.shape[1]):
        raise ValueError("Residual and covariance shapes do not agree.")
    eye = torch.eye(covariance.shape[-1], dtype=covariance.dtype, device=covariance.device)
    chol = torch.linalg.cholesky(covariance + float(jitter) * eye.unsqueeze(0))
    log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
    whitened = torch.linalg.solve_triangular(chol, residuals.unsqueeze(-1), upper=False)
    quadratic = whitened.square().sum(dim=(-2, -1))
    return -0.5 * (log_det + quadratic).mean()


def _dimension_kernel(
    period: float | None,
    active_dim: int,
    *,
    batch_shape: torch.Size,
) -> gpytorch.kernels.Kernel:
    kwargs: dict[str, Any] = {"active_dims": (active_dim,), "batch_shape": batch_shape}
    if period is None:
        return gpytorch.kernels.RBFKernel(**kwargs)
    kernel = gpytorch.kernels.PeriodicKernel(**kwargs)
    kernel.period_length = float(period)
    return kernel


def _create_gp_kernel(
    *,
    n_input: int,
    n_output: int,
    circular_period: CircularPeriod,
) -> gpytorch.kernels.Kernel:
    batch_shape = torch.Size([int(n_output)])
    periods = _periods(circular_period, int(n_input))
    if all(value is None for value in periods):
        base = gpytorch.kernels.RBFKernel(ard_num_dims=int(n_input), batch_shape=batch_shape)
    else:
        factors = [
            _dimension_kernel(value, dim, batch_shape=batch_shape)
            for dim, value in enumerate(periods)
        ]
        base = factors[0]
        for factor in factors[1:]:
            base = base * factor
    return gpytorch.kernels.ScaleKernel(base, batch_shape=batch_shape)


class _BatchedExactGP(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        kernel: gpytorch.kernels.Kernel,
        batch_shape: torch.Size,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = kernel

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class _BatchedVariationalGP(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        inducing_points: torch.Tensor,
        kernel: gpytorch.kernels.Kernel,
        batch_shape: torch.Size,
    ) -> None:
        distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.shape[-2], batch_shape=batch_shape
        )
        strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            distribution,
            learn_inducing_locations=True,
        )
        super().__init__(strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = kernel

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


@dataclass(frozen=True)
class GKRConfig:
    """Training configuration for :class:`TorchGKR`."""

    mean_iterations: int = 300
    mean_learning_rate: float = 0.05
    mean_batch_size: int | None = None
    n_inducing: int | None = 200
    covariance_epochs: int = 30
    covariance_learning_rate: float = 0.1
    covariance_batch_size: int = 3000
    validation_fraction: float = 0.33
    covariance_jitter: float = 1e-6
    likelihood_jitter: float = 1e-5
    prediction_batch_size: int = 3000
    standardize_responses: bool = True
    log_every: int = 25


@dataclass(frozen=True)
class GKRFisherResult:
    query: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    mean_jacobian: np.ndarray
    covariance_jacobian: np.ndarray
    fisher_matrix: np.ndarray
    linear_fisher: np.ndarray
    covariance_fisher_matrix: np.ndarray
    covariance_fisher: np.ndarray
    full_fisher_matrix: np.ndarray
    full_fisher: np.ndarray
    covariance_loss: np.ndarray
    mean_loss: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class TorchKernelCovariance(torch.nn.Module):
    """Normalized kernel regression over residual outer products."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        *,
        circular_period: CircularPeriod = None,
        jitter: float = 1e-6,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.n_input = int(n_input)
        self.n_output = int(n_output)
        self.circular_period = circular_period
        self.jitter = float(jitter)
        self.dtype = dtype
        self.device = torch.device(device)
        self.kernel_precision_cholesky = torch.nn.Parameter(
            torch.eye(self.n_input, dtype=dtype, device=self.device)
        )
        self.register_buffer(
            "train_residuals", torch.empty(0, self.n_output, dtype=dtype, device=self.device)
        )
        self.register_buffer(
            "train_inputs", torch.empty(0, self.n_input, dtype=dtype, device=self.device)
        )

    def set_data(self, residuals: TensorLike, inputs: TensorLike) -> None:
        residuals_t = _as_2d(residuals, dtype=self.dtype, device=self.device)
        inputs_t = _as_2d(inputs, dtype=self.dtype, device=self.device)
        if residuals_t.shape[0] != inputs_t.shape[0]:
            raise ValueError("Residuals and inputs must contain the same number of samples.")
        if residuals_t.shape[1] != self.n_output or inputs_t.shape[1] != self.n_input:
            raise ValueError("Training data dimensions do not match the covariance model.")
        self.train_residuals = residuals_t
        self.train_inputs = inputs_t

    def precision(self) -> torch.Tensor:
        lower = torch.tril(self.kernel_precision_cholesky)
        return lower @ lower.transpose(-1, -2)

    def forward(self, query: TensorLike, *, batch_size: int = 3000) -> torch.Tensor:
        if self.train_inputs.numel() == 0:
            raise RuntimeError("Call set_data before predicting covariance.")
        query_t = _as_2d(query, dtype=self.dtype, device=self.device)
        precision = self.precision()
        numerator = torch.zeros(
            query_t.shape[0], self.n_output, self.n_output,
            dtype=self.dtype, device=self.device,
        )
        denominator = torch.zeros(query_t.shape[0], dtype=self.dtype, device=self.device)
        for start in range(0, self.train_inputs.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), self.train_inputs.shape[0])
            inputs = self.train_inputs[start:stop]
            residuals = self.train_residuals[start:stop]
            diff = _wrapped_difference(inputs[:, None, :] - query_t[None, :, :], self.circular_period)
            squared_distance = torch.einsum("bqi,ij,bqj->bq", diff, precision, diff)
            weights = torch.exp(-0.5 * squared_distance)
            grams = torch.einsum("bi,bj->bij", residuals, residuals)
            numerator = numerator + torch.einsum("bq,bij->qij", weights, grams)
            denominator = denominator + weights.sum(dim=0)
        covariance = numerator / denominator.clamp_min(1e-12)[:, None, None]
        eye = torch.eye(self.n_output, dtype=self.dtype, device=self.device)
        return covariance + self.jitter * eye.unsqueeze(0)


class TorchGKR:
    """Fit a GKR conditional Gaussian model using Torch and GPyTorch."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        *,
        circular_period: CircularPeriod = None,
        config: GKRConfig | None = None,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        self.n_input = int(n_input)
        self.n_output = int(n_output)
        self.circular_period = circular_period
        self.config = config or GKRConfig()
        self.dtype = dtype
        self.device = torch.device(device)
        self.seed = int(seed)
        self.output_mean = torch.zeros(self.n_output, dtype=dtype, device=self.device)
        self.output_std = torch.ones(self.n_output, dtype=dtype, device=self.device)
        self.mean_model: gpytorch.models.GP | None = None
        self.mean_likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        self.mean_loss_history: list[float] = []
        self.covariance_loss_history: list[float] = []
        self.covariance_model = TorchKernelCovariance(
            self.n_input,
            self.n_output,
            circular_period=circular_period,
            jitter=self.config.covariance_jitter,
            dtype=dtype,
            device=self.device,
        )
        self._generator = torch.Generator(device="cpu").manual_seed(self.seed)

    def _select_inducing_points(self, inputs: torch.Tensor, count: int) -> torch.Tensor:
        if count >= inputs.shape[0]:
            return inputs.clone()
        order = torch.argsort(inputs[:, 0])
        positions = torch.linspace(0, inputs.shape[0] - 1, count, device=self.device).round().long()
        return inputs[order[positions]]

    def _fit_mean(self, inputs: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        if self.config.standardize_responses:
            self.output_mean = responses.mean(dim=0)
            self.output_std = responses.std(dim=0).clamp_min(1e-8)
            targets = (responses - self.output_mean) / self.output_std
        else:
            targets = responses
        batch_shape = torch.Size([self.n_output])
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=batch_shape).to(
            self.device, self.dtype
        )
        kernel = _create_gp_kernel(
            n_input=self.n_input,
            n_output=self.n_output,
            circular_period=self.circular_period,
        ).to(self.device, self.dtype)
        batched_targets = targets.transpose(0, 1).contiguous()
        n_inducing = self.config.n_inducing
        mean_batch_size = self.config.mean_batch_size
        if mean_batch_size is not None and int(mean_batch_size) < 1:
            raise ValueError("mean_batch_size must be positive when provided.")
        if n_inducing is None or int(n_inducing) >= inputs.shape[0]:
            if mean_batch_size is not None and int(mean_batch_size) < int(inputs.shape[0]):
                raise ValueError("Mean minibatching requires a variational GP with inducing points.")
            model: gpytorch.models.GP = _BatchedExactGP(
                inputs, batched_targets, likelihood, kernel, batch_shape
            ).to(self.device, self.dtype)
            objective: gpytorch.mlls.MarginalLogLikelihood = (
                gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            )
        else:
            inducing = self._select_inducing_points(inputs, int(n_inducing))
            inducing = inducing.unsqueeze(0).repeat(self.n_output, 1, 1)
            model = _BatchedVariationalGP(inducing, kernel, batch_shape).to(self.device, self.dtype)
            objective = gpytorch.mlls.VariationalELBO(
                likelihood, model, num_data=inputs.shape[0]
            )
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.mean_learning_rate)
        self.mean_loss_history = []
        for iteration in range(int(self.config.mean_iterations)):
            if mean_batch_size is None or int(mean_batch_size) >= int(inputs.shape[0]):
                batch_inputs = inputs
                batch_targets = batched_targets
            else:
                index = torch.randperm(
                    int(inputs.shape[0]), generator=self._generator
                )[: int(mean_batch_size)].to(self.device)
                batch_inputs = inputs[index]
                batch_targets = batched_targets[:, index]
            optimizer.zero_grad(set_to_none=True)
            loss_values = -objective(model(batch_inputs), batch_targets)
            loss = loss_values.sum() if loss_values.ndim else loss_values
            loss.backward()
            optimizer.step()
            value = float(loss_values.detach().mean().cpu())
            self.mean_loss_history.append(value)
            if self.config.log_every > 0 and (
                iteration == 0
                or (iteration + 1) % self.config.log_every == 0
                or iteration + 1 == self.config.mean_iterations
            ):
                print(
                    f"[gkr:mean] iteration={iteration + 1}/{self.config.mean_iterations} "
                    f"loss={value:.6f}", flush=True,
                )
        self.mean_model = model
        self.mean_likelihood = likelihood
        return responses - self.predict_mean(inputs)

    def predict_mean(self, query: TensorLike) -> torch.Tensor:
        if self.mean_model is None or self.mean_likelihood is None:
            raise RuntimeError("Fit GKR before predicting the mean.")
        query_t = _as_2d(query, dtype=self.dtype, device=self.device)
        self.mean_model.eval()
        self.mean_likelihood.eval()
        batch_size = int(self.config.prediction_batch_size)
        if batch_size < 1:
            raise ValueError("prediction_batch_size must be positive.")
        means: list[torch.Tensor] = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for start in range(0, int(query_t.shape[0]), batch_size):
                posterior = self.mean_model(query_t[start : start + batch_size])
                means.append(posterior.mean.transpose(0, 1).contiguous())
        mean = torch.cat(means, dim=0)
        return mean * self.output_std + self.output_mean

    def _split_covariance_batch(
        self, residuals: torch.Tensor, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = residuals.shape[0]
        if n < 2:
            return residuals, residuals, inputs, inputs
        n_valid = min(n - 1, max(1, int(round(n * self.config.validation_fraction))))
        indices = torch.randperm(n, generator=self._generator)
        valid = indices[:n_valid].to(self.device)
        train = indices[n_valid:].to(self.device)
        return residuals[train], residuals[valid], inputs[train], inputs[valid]

    def fit(self, responses: TensorLike, inputs: TensorLike) -> "TorchGKR":
        responses_t = _as_2d(responses, dtype=self.dtype, device=self.device)
        inputs_t = _as_2d(inputs, dtype=self.dtype, device=self.device)
        if responses_t.shape != (inputs_t.shape[0], self.n_output):
            raise ValueError("Response shape does not match n_output or the number of labels.")
        if inputs_t.shape[1] != self.n_input:
            raise ValueError("Input shape does not match n_input.")
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        residuals = self._fit_mean(inputs_t, responses_t).detach()
        optimizer = torch.optim.Adam(
            self.covariance_model.parameters(), lr=self.config.covariance_learning_rate
        )
        self.covariance_loss_history = []
        batch_size = int(self.config.covariance_batch_size)
        for epoch in range(int(self.config.covariance_epochs)):
            order = torch.randperm(residuals.shape[0], generator=self._generator).to(self.device)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, residuals.shape[0], batch_size):
                index = order[start : start + batch_size]
                r_train, r_valid, x_train, x_valid = self._split_covariance_batch(
                    residuals[index], inputs_t[index]
                )
                optimizer.zero_grad(set_to_none=True)
                self.covariance_model.set_data(r_train, x_train)
                covariance = self.covariance_model(
                    x_valid, batch_size=self.config.prediction_batch_size
                )
                loss = -gaussian_residual_log_likelihood(
                    r_valid, covariance, jitter=self.config.likelihood_jitter
                )
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                n_batches += 1
            epoch_loss /= max(n_batches, 1)
            self.covariance_loss_history.append(epoch_loss)
            if self.config.log_every > 0 and (
                epoch == 0
                or (epoch + 1) % self.config.log_every == 0
                or epoch + 1 == self.config.covariance_epochs
            ):
                print(
                    f"[gkr:covariance] epoch={epoch + 1}/{self.config.covariance_epochs} "
                    f"loss={epoch_loss:.6f}", flush=True,
                )
        self.covariance_model.set_data(residuals, inputs_t)
        return self

    def predict(self, query: TensorLike) -> tuple[np.ndarray, np.ndarray]:
        query_t = _as_2d(query, dtype=self.dtype, device=self.device)
        mean = self.predict_mean(query_t)
        with torch.no_grad():
            covariance = self.covariance_model(
                query_t, batch_size=self.config.prediction_batch_size
            )
        return mean.cpu().numpy(), covariance.cpu().numpy()


def estimate_gkr_linear_fisher(
    model: TorchGKR,
    query: TensorLike,
    *,
    finite_difference_step: float | np.ndarray,
    solve_jitter: float = 1e-6,
) -> GKRFisherResult:
    """Estimate GKR Fisher information using explicit local label separations."""

    query_np = np.asarray(query, dtype=np.float64)
    if query_np.ndim == 1:
        query_np = query_np[:, None]
    if query_np.ndim != 2:
        raise ValueError("query must be a two-dimensional array.")
    step_value = np.asarray(finite_difference_step, dtype=np.float64)
    if step_value.ndim == 0:
        step = np.full(query_np.shape, float(step_value), dtype=np.float64)
    elif step_value.shape == (query_np.shape[1],):
        step = np.broadcast_to(step_value.reshape(1, -1), query_np.shape).copy()
    elif step_value.shape == query_np.shape:
        step = step_value.copy()
    else:
        raise ValueError(
            "finite_difference_step must be scalar, [n_input], or [n_query, n_input]."
        )
    if np.any(~np.isfinite(step)) or np.any(step <= 0.0):
        raise ValueError("finite_difference_step must contain finite positive values.")
    mean, covariance = model.predict(query_np)
    jacobian = np.empty((query_np.shape[0], model.n_output, model.n_input), dtype=np.float64)
    covariance_jacobian = np.empty(
        (query_np.shape[0], model.n_output, model.n_output, model.n_input),
        dtype=np.float64,
    )
    for dim in range(model.n_input):
        plus = query_np.copy()
        minus = query_np.copy()
        plus[:, dim] += 0.5 * step[:, dim]
        minus[:, dim] -= 0.5 * step[:, dim]
        mean_plus, covariance_plus = model.predict(plus)
        mean_minus, covariance_minus = model.predict(minus)
        jacobian[:, :, dim] = (mean_plus - mean_minus) / step[:, dim, None]
        covariance_jacobian[:, :, :, dim] = (
            covariance_plus - covariance_minus
        ) / step[:, dim, None, None]
    eye = np.eye(model.n_output, dtype=np.float64)
    stabilized = covariance + float(solve_jitter) * eye[None, :, :]
    solved = np.linalg.solve(stabilized, jacobian)
    linear_fisher_matrix = np.einsum("noi,noj->nij", jacobian, solved)
    linear_fisher = np.trace(linear_fisher_matrix, axis1=1, axis2=2)

    n_query, n_output, _, n_input = covariance_jacobian.shape
    covariance_rhs = covariance_jacobian.reshape(n_query, n_output, n_output * n_input)
    precision_times_derivative = np.linalg.solve(stabilized, covariance_rhs).reshape(
        n_query, n_output, n_output, n_input
    )
    covariance_fisher_matrix = 0.5 * np.einsum(
        "nabi,nbaj->nij", precision_times_derivative, precision_times_derivative
    )
    covariance_fisher = np.trace(covariance_fisher_matrix, axis1=1, axis2=2)
    full_fisher_matrix = linear_fisher_matrix + covariance_fisher_matrix
    full_fisher = np.trace(full_fisher_matrix, axis1=1, axis2=2)
    return GKRFisherResult(
        query=query_np,
        mean=np.asarray(mean, dtype=np.float64),
        covariance=np.asarray(covariance, dtype=np.float64),
        mean_jacobian=jacobian,
        covariance_jacobian=covariance_jacobian,
        fisher_matrix=linear_fisher_matrix,
        linear_fisher=np.maximum(linear_fisher, 0.0),
        covariance_fisher_matrix=covariance_fisher_matrix,
        covariance_fisher=np.maximum(covariance_fisher, 0.0),
        full_fisher_matrix=full_fisher_matrix,
        full_fisher=np.maximum(full_fisher, 0.0),
        covariance_loss=np.asarray(model.covariance_loss_history, dtype=np.float64),
        mean_loss=np.asarray(model.mean_loss_history, dtype=np.float64),
        metadata={
            "finite_difference_step": step.copy(),
            "solve_jitter": float(solve_jitter),
            "linear_fisher_only": False,
            "full_gaussian_fisher": True,
        },
    )
