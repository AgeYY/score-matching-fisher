from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.flow_matching_skl import (
    FlowSKLResult,
    VELOCITY_FAMILIES,
    build_flow_skl_model,
    centered_radius_normalize,
    estimate_model_symmetric_kl,
    estimate_scalar_fisher_from_skl,
    flow_skl_result_to_npz_dict,
)
from fisher.llr_divergence import symmetric_kl_gaussian_full_matrix


class TableTranslationModel(nn.Module):
    def __init__(self, means: np.ndarray, *, velocity_family: str = "translation", radius: float = 1.0) -> None:
        super().__init__()
        self.velocity_family = velocity_family
        self.radius = float(radius)
        self.x_dim = int(means.shape[1])
        self.register_buffer("means", torch.as_tensor(means, dtype=torch.float32))

    def endpoint_mean(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        return theta.to(self.means.dtype) @ self.means


class CenteredTableTranslationModel(TableTranslationModel):
    def endpoint_mean(self, theta: torch.Tensor) -> torch.Tensor:
        raw = super().endpoint_mean(theta)
        return centered_radius_normalize(raw, self.radius)


class ScalarLinearTranslationModel(nn.Module):
    def __init__(self, slope: np.ndarray) -> None:
        super().__init__()
        self.velocity_family = "translation"
        self.x_dim = int(np.asarray(slope).size)
        self.register_buffer("slope", torch.as_tensor(slope, dtype=torch.float32).reshape(1, -1))

    def endpoint_mean(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        return theta.to(self.slope.dtype) @ self.slope


class GaussianEndpointModel(nn.Module):
    def __init__(
        self,
        means: np.ndarray,
        covariance: np.ndarray,
        *,
        velocity_family: str,
    ) -> None:
        super().__init__()
        self.velocity_family = velocity_family
        self.x_dim = int(means.shape[1])
        self.register_buffer("means", torch.as_tensor(means, dtype=torch.float64))
        self.register_buffer("covariance", torch.as_tensor(covariance, dtype=torch.float64))

    def _mean(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        return theta.to(self.means.dtype) @ self.means

    def endpoint_mean_covariance(self, theta: torch.Tensor, *, solve_jitter: float = 1e-6, quadrature_steps=None):
        del solve_jitter, quadrature_steps
        mu = self._mean(theta)
        cov = self.covariance
        if cov.ndim == 2:
            cov = cov.reshape(1, self.x_dim, self.x_dim).expand(int(mu.shape[0]), self.x_dim, self.x_dim)
        return mu, cov


class ConstantNet(nn.Module):
    def __init__(self, values: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("values", values.reshape(1, -1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.values.to(dtype=inp.dtype, device=inp.device).expand(int(inp.shape[0]), -1)


class OneHotConditionNet(nn.Module):
    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("table", table)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        table = self.table.to(dtype=inp.dtype, device=inp.device)
        theta = inp[:, 1 : 1 + int(table.shape[0])]
        return theta @ table


class FirstReducedCoordinateNet(nn.Module):
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return inp[:, :1]


class StandardNormalMCFlowModel(nn.Module):
    def __init__(self, *, velocity_family: str, x_dim: int) -> None:
        super().__init__()
        self.velocity_family = velocity_family
        self.x_dim = int(x_dim)

    def forward(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del theta, t
        return torch.zeros_like(x)

    def log_prob_normalized(
        self,
        x_norm: torch.Tensor,
        theta: torch.Tensor,
        *,
        solve_jitter: float = 1e-6,
        quadrature_steps=None,
        ode_steps: int = 32,
    ) -> torch.Tensor:
        del theta, solve_jitter, quadrature_steps, ode_steps
        if x_norm.ndim == 1:
            x_norm = x_norm.unsqueeze(0)
        d = int(x_norm.shape[1])
        return -0.5 * (torch.sum(x_norm**2, dim=1) + float(d) * np.log(2.0 * np.pi))


def _one_hot(n: int) -> np.ndarray:
    return np.eye(int(n), dtype=np.float64)


def _patch_table_b(model: nn.Module, table: torch.Tensor) -> None:
    def b(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        table_t = table.to(dtype=theta.dtype, device=theta.device)
        return theta @ table_t

    model.b = types.MethodType(b, model)  # type: ignore[method-assign]


def _patch_shared_A(model: nn.Module, matrix: torch.Tensor) -> None:
    def A(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        mat = matrix.to(dtype=t.dtype, device=t.device)
        return mat.reshape(1, *mat.shape).expand(int(t.shape[0]), *mat.shape)

    model.A = types.MethodType(A, model)  # type: ignore[method-assign]


def _patch_condition_A(model: nn.Module, matrices: torch.Tensor) -> None:
    def A(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        mats = matrices.to(dtype=theta.dtype, device=theta.device)
        return torch.einsum("bc,cij->bij", theta, mats)

    model.A = types.MethodType(A, model)  # type: ignore[method-assign]


def test_build_flow_skl_model_constructs_restricted_velocity_families() -> None:
    expected = {
        "shared_affine_scalar": "CenteredSharedAffineScalarFlowSKLModel",
        "shared_affine_diag": "CenteredSharedAffineDiagFlowSKLModel",
        "condition_affine_scalar": "CenteredConditionAffineScalarFlowSKLModel",
        "condition_affine_diag": "CenteredConditionAffineDiagFlowSKLModel",
        "shared_affine_low_rank_scalar": "CenteredSharedAffineLowRankScalarFlowSKLModel",
        "shared_affine_low_rank_diag": "CenteredSharedAffineLowRankDiagFlowSKLModel",
    }
    for family, class_name in expected.items():
        assert family in VELOCITY_FAMILIES
        model = build_flow_skl_model(
            velocity_family=family,
            theta_dim=2,
            x_dim=3,
            hidden_dim=4,
            depth=1,
            low_rank_dim=1,
            path_schedule="linear",
        )
        assert model.velocity_family == family
        assert type(model).__name__ == class_name


def test_translation_skl_equals_squared_euclidean() -> None:
    means = np.array([[0.0, 0.0], [1.0, 2.0], [-2.0, 1.0]], dtype=np.float64)
    model = TableTranslationModel(means)
    result = estimate_model_symmetric_kl(
        model=model,
        theta_all=_one_hot(3),
        device=torch.device("cpu"),
        velocity_family="translation",
    )
    diff = means[:, None, :] - means[None, :, :]
    expected = np.sum(diff * diff, axis=2)
    np.testing.assert_allclose(result.symmetric_kl_matrix, expected, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(result.canonical_metric_matrix, expected, rtol=1e-7, atol=1e-7)
    assert result.canonical_metric_name == "squared_euclidean"


def test_fixed_norm_translation_canonical_metric_is_cosine_distance() -> None:
    radius = 2.5
    means = radius * np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    model = TableTranslationModel(means, velocity_family="translation_fixed_norm", radius=radius)
    result = estimate_model_symmetric_kl(
        model=model,
        theta_all=_one_hot(3),
        device=torch.device("cpu"),
        velocity_family="translation_fixed_norm",
        radius=radius,
    )
    cosine_distance = 1.0 - (means @ means.T) / (radius * radius)
    np.fill_diagonal(cosine_distance, 0.0)
    np.testing.assert_allclose(result.canonical_metric_matrix, cosine_distance, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(result.symmetric_kl_matrix / (2.0 * radius * radius), cosine_distance, rtol=1e-7, atol=1e-7)


def test_centered_fixed_norm_translation_canonical_metric_is_correlation_distance() -> None:
    radius = 1.75
    raw = np.array([[1.0, 2.0, 5.0], [3.0, 6.0, 9.0], [5.0, 2.0, 1.0]], dtype=np.float64)
    model = CenteredTableTranslationModel(raw, velocity_family="translation_centered_fixed_norm", radius=radius)
    result = estimate_model_symmetric_kl(
        model=model,
        theta_all=_one_hot(3),
        device=torch.device("cpu"),
        velocity_family="translation_centered_fixed_norm",
        radius=radius,
    )
    centered = raw - raw.mean(axis=1, keepdims=True)
    centered = radius * centered / np.linalg.norm(centered, axis=1, keepdims=True)
    corr_distance = 1.0 - (centered @ centered.T) / (radius * radius)
    np.fill_diagonal(corr_distance, 0.0)
    np.testing.assert_allclose(result.canonical_metric_matrix, corr_distance, rtol=1e-6, atol=1e-6)


def test_shared_affine_forward_uses_centered_residual() -> None:
    model = build_flow_skl_model(
        velocity_family="shared_affine",
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
        path_schedule="linear",
    ).double()
    means = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float64)
    a_mat = torch.tensor([[0.2, -0.4], [0.7, 0.1]], dtype=torch.float64)
    _patch_table_b(model, means)
    _patch_shared_A(model, a_mat)

    x = torch.tensor([[2.0, 1.0], [-1.0, 4.0]], dtype=torch.float64)
    theta = torch.eye(2, dtype=torch.float64)
    t = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
    beta = t
    beta_dot = torch.ones_like(t)
    b = theta @ means
    expected = beta_dot * b + (x - beta * b) @ a_mat.T
    torch.testing.assert_close(model(x, theta, t), expected, rtol=1e-12, atol=1e-12)


def test_shared_affine_scalar_forward_uses_scalar_identity_base() -> None:
    model = build_flow_skl_model(
        velocity_family="shared_affine_scalar",
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
        path_schedule="linear",
    ).double()
    means = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float64)
    scale = torch.tensor([0.35], dtype=torch.float64)
    _patch_table_b(model, means)
    model.a_net = ConstantNet(scale)

    x = torch.tensor([[2.0, 1.0], [-1.0, 4.0]], dtype=torch.float64)
    theta = torch.eye(2, dtype=torch.float64)
    t = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
    b = theta @ means
    centered = x - t * b
    expected = b + scale.reshape(1, 1) * centered
    expected_a = scale.reshape(1, 1, 1) * torch.eye(2, dtype=torch.float64).reshape(1, 2, 2).expand(2, 2, 2)
    torch.testing.assert_close(model.A(t), expected_a, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(model(x, theta, t), expected, rtol=1e-12, atol=1e-12)


def test_shared_affine_diag_forward_uses_diagonal_base() -> None:
    model = build_flow_skl_model(
        velocity_family="shared_affine_diag",
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
        path_schedule="linear",
    ).double()
    means = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float64)
    diag = torch.tensor([0.2, -0.4], dtype=torch.float64)
    _patch_table_b(model, means)
    model.a_net = ConstantNet(diag)

    x = torch.tensor([[2.0, 1.0], [-1.0, 4.0]], dtype=torch.float64)
    theta = torch.eye(2, dtype=torch.float64)
    t = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
    b = theta @ means
    centered = x - t * b
    expected = b + centered * diag.reshape(1, -1)
    expected_a = torch.diag_embed(diag.reshape(1, -1).expand(2, -1))
    torch.testing.assert_close(model.A(t), expected_a, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(model(x, theta, t), expected, rtol=1e-12, atol=1e-12)


def test_condition_affine_forward_uses_centered_residual_and_condition_matrix() -> None:
    model = build_flow_skl_model(
        velocity_family="condition_affine",
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
        path_schedule="linear",
    ).double()
    means = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float64)
    a_mats = torch.tensor(
        [
            [[0.2, -0.4], [0.7, 0.1]],
            [[-0.3, 0.6], [0.25, 0.4]],
        ],
        dtype=torch.float64,
    )
    _patch_table_b(model, means)
    _patch_condition_A(model, a_mats)

    x = torch.tensor([[2.0, 1.0], [-1.0, 4.0]], dtype=torch.float64)
    theta = torch.eye(2, dtype=torch.float64)
    t = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
    beta = t
    beta_dot = torch.ones_like(t)
    b = theta @ means
    centered = x - beta * b
    expected = beta_dot * b + torch.bmm(a_mats, centered.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(model(x, theta, t), expected, rtol=1e-12, atol=1e-12)


def test_condition_affine_scalar_forward_uses_condition_scalar_identity_base() -> None:
    model = build_flow_skl_model(
        velocity_family="condition_affine_scalar",
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
        path_schedule="linear",
    ).double()
    means = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float64)
    scales = torch.tensor([[0.2], [-0.3]], dtype=torch.float64)
    _patch_table_b(model, means)
    model.a_net = OneHotConditionNet(scales)

    x = torch.tensor([[2.0, 1.0], [-1.0, 4.0]], dtype=torch.float64)
    theta = torch.eye(2, dtype=torch.float64)
    t = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
    b = theta @ means
    centered = x - t * b
    row_scales = theta @ scales
    expected = b + row_scales * centered
    expected_a = row_scales.reshape(2, 1, 1) * torch.eye(2, dtype=torch.float64).reshape(1, 2, 2)
    torch.testing.assert_close(model.A(theta, t), expected_a, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(model(x, theta, t), expected, rtol=1e-12, atol=1e-12)


def test_condition_affine_diag_forward_uses_condition_diagonal_base() -> None:
    model = build_flow_skl_model(
        velocity_family="condition_affine_diag",
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
        path_schedule="linear",
    ).double()
    means = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float64)
    diag_table = torch.tensor([[0.2, -0.4], [-0.3, 0.6]], dtype=torch.float64)
    _patch_table_b(model, means)
    model.a_net = OneHotConditionNet(diag_table)

    x = torch.tensor([[2.0, 1.0], [-1.0, 4.0]], dtype=torch.float64)
    theta = torch.eye(2, dtype=torch.float64)
    t = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
    b = theta @ means
    centered = x - t * b
    diag_rows = theta @ diag_table
    expected = b + centered * diag_rows
    expected_a = torch.diag_embed(diag_rows)
    torch.testing.assert_close(model.A(theta, t), expected_a, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(model(x, theta, t), expected, rtol=1e-12, atol=1e-12)


def test_shared_affine_low_rank_forward_centers_low_rank_correction() -> None:
    class IdentityReducedNet(nn.Module):
        def forward(self, inp: torch.Tensor) -> torch.Tensor:
            return inp[:, :1]

    model = build_flow_skl_model(
        velocity_family="shared_affine_low_rank",
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
        low_rank_dim=1,
        path_schedule="linear",
    ).double()
    means = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float64)
    _patch_table_b(model, means)
    _patch_shared_A(model, torch.zeros(2, 2, dtype=torch.float64))
    model.h_net = IdentityReducedNet()

    x = torch.tensor([[2.0, 1.0], [-1.0, 4.0]], dtype=torch.float64)
    theta = torch.eye(2, dtype=torch.float64)
    t = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
    beta = t
    beta_dot = torch.ones_like(t)
    b = theta @ means
    centered = x - beta * b
    u = model.U.detach()
    expected = beta_dot * b + (centered @ u) @ u.T
    torch.testing.assert_close(model(x, theta, t), expected, rtol=1e-12, atol=1e-12)


def test_shared_affine_low_rank_scalar_forward_includes_scalar_base_and_correction() -> None:
    model = build_flow_skl_model(
        velocity_family="shared_affine_low_rank_scalar",
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
        low_rank_dim=1,
        path_schedule="linear",
    ).double()
    means = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float64)
    scale = torch.tensor([0.25], dtype=torch.float64)
    _patch_table_b(model, means)
    model.a_net = ConstantNet(scale)
    model.h_net = FirstReducedCoordinateNet()

    x = torch.tensor([[2.0, 1.0], [-1.0, 4.0]], dtype=torch.float64)
    theta = torch.eye(2, dtype=torch.float64)
    t = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
    b = theta @ means
    centered = x - t * b
    u = model.U.detach()
    expected = b + scale.reshape(1, 1) * centered + (centered @ u) @ u.T
    torch.testing.assert_close(model(x, theta, t), expected, rtol=1e-12, atol=1e-12)


def test_shared_affine_low_rank_diag_forward_includes_diagonal_base_and_correction() -> None:
    model = build_flow_skl_model(
        velocity_family="shared_affine_low_rank_diag",
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
        low_rank_dim=1,
        path_schedule="linear",
    ).double()
    means = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float64)
    diag = torch.tensor([0.25, -0.1], dtype=torch.float64)
    _patch_table_b(model, means)
    model.a_net = ConstantNet(diag)
    model.h_net = FirstReducedCoordinateNet()

    x = torch.tensor([[2.0, 1.0], [-1.0, 4.0]], dtype=torch.float64)
    theta = torch.eye(2, dtype=torch.float64)
    t = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
    b = theta @ means
    centered = x - t * b
    u = model.U.detach()
    expected = b + centered * diag.reshape(1, -1) + (centered @ u) @ u.T
    torch.testing.assert_close(model(x, theta, t), expected, rtol=1e-12, atol=1e-12)


def test_shared_affine_skl_reduces_to_mahalanobis_squared() -> None:
    means = np.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 0.5]], dtype=np.float64)
    cov = np.array([[2.0, 0.3], [0.3, 0.8]], dtype=np.float64)
    model = GaussianEndpointModel(means, cov, velocity_family="shared_affine")
    result = estimate_model_symmetric_kl(
        model=model,
        theta_all=_one_hot(3),
        device=torch.device("cpu"),
        velocity_family="shared_affine",
    )
    inv = np.linalg.inv(cov)
    expected = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            d = means[i] - means[j]
            expected[i, j] = float(d @ inv @ d)
    np.testing.assert_allclose(result.symmetric_kl_matrix, expected, rtol=2e-6, atol=1e-5)
    np.testing.assert_allclose(result.canonical_metric_matrix, expected, rtol=2e-6, atol=1e-5)
    assert result.canonical_metric_name == "mahalanobis_sq"
    assert result.endpoint_covariance is not None
    assert result.endpoint_covariance.shape == (2, 2)


def test_restricted_shared_affine_endpoint_covariance_is_shared_full_matrix() -> None:
    means = np.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 0.5]], dtype=np.float64)
    cov = np.array([[2.0, 0.3], [0.3, 0.8]], dtype=np.float64)
    for family in ("shared_affine_scalar", "shared_affine_diag"):
        model = GaussianEndpointModel(means, cov, velocity_family=family)
        result = estimate_model_symmetric_kl(
            model=model,
            theta_all=_one_hot(3),
            device=torch.device("cpu"),
            velocity_family=family,
        )
        assert result.endpoint_covariance is not None
        assert result.endpoint_covariance.shape == (2, 2)
        assert result.canonical_metric_name == "mahalanobis_sq"


def test_condition_affine_skl_uses_full_gaussian_helper_with_jeffreys_scaling() -> None:
    means = np.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 0.5]], dtype=np.float64)
    covs = np.array(
        [
            [[1.0, 0.2], [0.2, 2.0]],
            [[0.7, -0.15], [-0.15, 1.5]],
            [[2.5, 0.4], [0.4, 0.9]],
        ],
        dtype=np.float64,
    )
    model = GaussianEndpointModel(means, covs, velocity_family="condition_affine")
    result = estimate_model_symmetric_kl(
        model=model,
        theta_all=_one_hot(3),
        device=torch.device("cpu"),
        velocity_family="condition_affine",
    )
    helper = symmetric_kl_gaussian_full_matrix(means, covs, jitter=1e-6)
    np.testing.assert_allclose(result.symmetric_kl_matrix, 2.0 * helper, rtol=1e-6, atol=1e-6)
    assert result.endpoint_covariance is not None
    assert result.endpoint_covariance.shape == (3, 2, 2)
    assert result.canonical_metric_name == "gaussian_symmetric_kl"


def test_restricted_condition_affine_endpoint_covariance_is_batched_full_matrix() -> None:
    means = np.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 0.5]], dtype=np.float64)
    covs = np.array(
        [
            [[1.0, 0.2], [0.2, 2.0]],
            [[0.7, -0.15], [-0.15, 1.5]],
            [[2.5, 0.4], [0.4, 0.9]],
        ],
        dtype=np.float64,
    )
    for family in ("condition_affine_scalar", "condition_affine_diag"):
        model = GaussianEndpointModel(means, covs, velocity_family=family)
        result = estimate_model_symmetric_kl(
            model=model,
            theta_all=_one_hot(3),
            device=torch.device("cpu"),
            velocity_family=family,
        )
        assert result.endpoint_covariance is not None
        assert result.endpoint_covariance.shape == (3, 2, 2)
        assert result.canonical_metric_name == "gaussian_symmetric_kl"


def test_restricted_low_rank_affine_endpoint_uses_mc_without_gaussian_covariance() -> None:
    theta = np.eye(2, dtype=np.float64)
    for family in ("shared_affine_low_rank_scalar", "shared_affine_low_rank_diag"):
        model = StandardNormalMCFlowModel(velocity_family=family, x_dim=2)
        result = estimate_model_symmetric_kl(
            model=model,
            theta_all=theta,
            device=torch.device("cpu"),
            velocity_family=family,
            mc_samples=8,
            ode_steps=2,
            batch_size=4,
        )
        assert result.endpoint_mean is None
        assert result.endpoint_covariance is None
        assert result.canonical_metric_name == "model_symmetric_kl_mc"


def test_flow_skl_result_npz_uses_endpoint_covariance_field() -> None:
    result = FlowSKLResult(
        symmetric_kl_matrix=np.zeros((1, 1), dtype=np.float64),
        canonical_metric_matrix=np.zeros((1, 1), dtype=np.float64),
        canonical_metric_name="gaussian_symmetric_kl",
        endpoint_mean=np.zeros((1, 2), dtype=np.float64),
        endpoint_covariance=np.eye(2, dtype=np.float64),
    )
    fields = flow_skl_result_to_npz_dict(result)
    assert "endpoint_covariance" in fields
    assert "endpoint_cov_or_diag" not in fields
    assert "endpoint_is_diag" not in fields


def test_scalar_fisher_finite_differences_recover_linear_gaussian_case() -> None:
    theta = np.array([[0.0], [0.1], [0.2], [0.35]], dtype=np.float64)
    slope = np.array([2.0, -1.0], dtype=np.float64)
    model = ScalarLinearTranslationModel(slope)
    result = estimate_model_symmetric_kl(
        model=model,
        theta_all=theta,
        device=torch.device("cpu"),
        velocity_family="translation",
        fisher_kind="full",
    )
    expected_fisher = float(slope @ slope)
    np.testing.assert_allclose(result.fisher_full, expected_fisher, rtol=1e-6, atol=1e-6)
    fd = estimate_scalar_fisher_from_skl(theta, result.symmetric_kl_matrix)
    np.testing.assert_allclose(fd["fisher"], expected_fisher, rtol=1e-6, atol=1e-6)
