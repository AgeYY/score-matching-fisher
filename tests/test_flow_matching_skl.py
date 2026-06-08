from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher import flow_matching_skl as fms
from fisher.flow_matching_skl import (
    FlowSKLResult,
    VELOCITY_FAMILIES,
    build_flow_skl_model,
    centered_radius_normalize,
    estimate_model_symmetric_kl,
    estimate_scalar_fisher_from_skl,
    flow_endpoint_log_prob,
    flow_skl_result_to_npz_dict,
    sample_flow_endpoint,
    train_flow_skl_model,
)


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


def _one_hot(n: int) -> np.ndarray:
    return np.eye(int(n), dtype=np.float64)


def _sentinel_skl(n: int, *, offset: float = 10.0) -> np.ndarray:
    raw = np.arange(int(n) * int(n), dtype=np.float64).reshape(int(n), int(n)) + float(offset)
    out = raw + raw.T
    np.fill_diagonal(out, 0.0)
    return out


def _patch_model_jeffreys(
    monkeypatch: pytest.MonkeyPatch,
    sentinel: np.ndarray,
    calls: list[dict[str, object]] | None = None,
) -> None:
    def fake_estimate_model_jeffreys(**kwargs):
        if calls is not None:
            calls.append(dict(kwargs))
        theta = np.asarray(kwargs["theta_all"], dtype=np.float64)
        assert theta.shape[0] == sentinel.shape[0]
        assert "theta_data" not in kwargs
        assert "x_data" not in kwargs
        return sentinel.copy()

    monkeypatch.setattr(fms, "_estimate_model_jeffreys", fake_estimate_model_jeffreys)


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


def test_shared_affine_full_A_is_symmetric_for_arbitrary_t() -> None:
    torch.manual_seed(123)
    model = build_flow_skl_model(
        velocity_family="shared_affine",
        theta_dim=2,
        x_dim=4,
        hidden_dim=8,
        depth=2,
        path_schedule="linear",
    ).double()
    t = torch.tensor([[0.0], [0.17], [0.5], [0.91]], dtype=torch.float64)
    a_t = model.A(t)
    torch.testing.assert_close(a_t, a_t.transpose(-1, -2), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("family", "net_values", "base_a"),
    (
        ("shared_affine", torch.zeros(4, dtype=torch.float64), torch.zeros(2, 2, dtype=torch.float64)),
        ("shared_affine_scalar", torch.tensor([0.25], dtype=torch.float64), 0.25 * torch.eye(2, dtype=torch.float64)),
        ("shared_affine_diag", torch.tensor([0.2, -0.4], dtype=torch.float64), torch.diag(torch.tensor([0.2, -0.4], dtype=torch.float64))),
    ),
)
def test_shared_affine_default_A_includes_diagonal_jitter(
    family: str,
    net_values: torch.Tensor,
    base_a: torch.Tensor,
) -> None:
    model = build_flow_skl_model(
        velocity_family=family,
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
        path_schedule="linear",
    ).double()
    model.a_net = ConstantNet(net_values)

    t = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
    expected = (base_a + 1e-3 * torch.eye(2, dtype=torch.float64)).reshape(1, 2, 2).expand(2, 2, 2)
    assert model.a_diag_jitter == pytest.approx(1e-3)
    torch.testing.assert_close(model.A(t), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("family", "net_values", "base_a"),
    (
        ("shared_affine_low_rank", torch.zeros(4, dtype=torch.float64), torch.zeros(2, 2, dtype=torch.float64)),
        ("shared_affine_low_rank_scalar", torch.tensor([0.25], dtype=torch.float64), 0.25 * torch.eye(2, dtype=torch.float64)),
        ("shared_affine_low_rank_diag", torch.tensor([0.2, -0.4], dtype=torch.float64), torch.diag(torch.tensor([0.2, -0.4], dtype=torch.float64))),
    ),
)
def test_shared_affine_low_rank_default_base_A_includes_diagonal_jitter(
    family: str,
    net_values: torch.Tensor,
    base_a: torch.Tensor,
) -> None:
    model = build_flow_skl_model(
        velocity_family=family,
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
        low_rank_dim=1,
        path_schedule="linear",
    ).double()
    model.a_net = ConstantNet(net_values)

    t = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
    expected = (base_a + 1e-3 * torch.eye(2, dtype=torch.float64)).reshape(1, 2, 2).expand(2, 2, 2)
    assert model.a_diag_jitter == pytest.approx(1e-3)
    torch.testing.assert_close(model.A(t), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("bad_jitter", (-1e-6, float("inf"), float("nan")))
def test_shared_affine_a_diag_jitter_must_be_finite_nonnegative(bad_jitter: float) -> None:
    with pytest.raises(ValueError):
        build_flow_skl_model(
            velocity_family="shared_affine",
            theta_dim=2,
            x_dim=2,
            hidden_dim=4,
            depth=1,
            path_schedule="linear",
            shared_affine_a_diag_jitter=bad_jitter,
        )


def test_condition_affine_full_A_is_symmetric_for_arbitrary_theta_and_t() -> None:
    torch.manual_seed(123)
    model = build_flow_skl_model(
        velocity_family="condition_affine",
        theta_dim=3,
        x_dim=4,
        hidden_dim=8,
        depth=2,
        path_schedule="linear",
    ).double()
    theta = torch.tensor(
        [[1.0, 0.0, 0.0], [0.25, 0.5, -0.75], [-1.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    t = torch.tensor([[0.0], [0.37], [0.91]], dtype=torch.float64)
    a_t = model.A(theta, t)
    torch.testing.assert_close(a_t, a_t.transpose(-1, -2), rtol=1e-12, atol=1e-12)


def test_translation_families_report_model_jeffreys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    radius_fixed = 2.5
    fixed_means = radius_fixed * np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    radius_centered = 1.75
    centered_raw = np.array([[1.0, 2.0, 5.0], [3.0, 6.0, 9.0], [5.0, 2.0, 1.0]], dtype=np.float64)
    cases = (
        (
            "translation",
            TableTranslationModel(np.array([[0.0, 0.0], [1.0, 2.0], [-2.0, 1.0]], dtype=np.float64)),
            None,
        ),
        (
            "translation_fixed_norm",
            TableTranslationModel(fixed_means, velocity_family="translation_fixed_norm", radius=radius_fixed),
            radius_fixed,
        ),
        (
            "translation_centered_fixed_norm",
            CenteredTableTranslationModel(
                centered_raw,
                velocity_family="translation_centered_fixed_norm",
                radius=radius_centered,
            ),
            radius_centered,
        ),
    )

    for idx, (family, model, radius) in enumerate(cases):
        theta_eval = _one_hot(3)
        sentinel = _sentinel_skl(3, offset=10.0 * (idx + 1))
        calls: list[dict[str, object]] = []
        _patch_model_jeffreys(monkeypatch, sentinel, calls)
        result = estimate_model_symmetric_kl(
            model=model,
            theta_all=theta_eval,
            device=torch.device("cpu"),
            velocity_family=family,
            radius=radius,
            mc_jeffreys_sample=17,
            ode_steps=5,
            batch_size=4,
            solve_jitter=3e-5,
            quadrature_steps=7,
            ode_method="heun3",
        )

        np.testing.assert_allclose(result.symmetric_kl_matrix, sentinel)
        np.testing.assert_allclose(result.canonical_metric_matrix, sentinel)
        assert result.canonical_metric_name == "model_jeffreys_symmetric_kl"
        assert calls and calls[0]["mc_jeffreys_sample"] == 17
        assert "normalization" not in calls[0]
        assert calls[0]["ode_steps"] == 5
        assert calls[0]["batch_size"] == 4
        assert calls[0]["solve_jitter"] == 3e-5
        assert calls[0]["quadrature_steps"] == 7
        assert calls[0]["ode_method"] == "heun3"


def test_translation_training_uses_flow_matching_forward_loss() -> None:
    model = build_flow_skl_model(
        velocity_family="translation",
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
        path_schedule="linear",
    )
    calls: list[dict[str, object]] = []
    original_forward = model.forward

    def spy_forward(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del self
        calls.append({"x_shape": tuple(x.shape), "t_ndim": int(t.ndim)})
        return original_forward(x, theta, t)

    model.forward = types.MethodType(spy_forward, model)  # type: ignore[method-assign]
    theta = np.eye(2, dtype=np.float64)[[0, 1, 0, 1]]
    x = np.array([[0.0, 0.0], [1.0, -1.0], [0.2, 0.1], [1.2, -0.8]], dtype=np.float64)

    out = train_flow_skl_model(
        model=model,
        theta_train=theta,
        x_train=x,
        theta_val=theta,
        x_val=x,
        device=torch.device("cpu"),
        velocity_family="translation",
        path_schedule="linear",
        epochs=1,
        batch_size=2,
        t_eps=0.1,
        log_every=999,
    )

    assert out["n_total_steps"] == 2
    assert calls
    assert all(c["t_ndim"] == 1 for c in calls)


def test_translation_log_prob_uses_package_likelihood_against_shifted_normal() -> None:
    model = build_flow_skl_model(
        velocity_family="translation",
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
        path_schedule="linear",
        divergence_estimator="exact",
    ).double()
    mean = torch.tensor([1.0, -2.0], dtype=torch.float64)
    model.mean_net = ConstantNet(mean)

    x = torch.tensor([[2.0, 1.0]], dtype=torch.float64)
    theta = torch.eye(2, dtype=torch.float64)[:1]
    got = flow_endpoint_log_prob(model, x, theta, ode_steps=1, ode_method="midpoint")
    x0 = x - mean.reshape(1, -1)
    expected = -0.5 * (torch.sum(x0**2, dim=1) + 2.0 * np.log(2.0 * np.pi))
    torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)


def test_model_jeffreys_matches_translation_model_skl_with_deterministic_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_flow_skl_model(
        velocity_family="translation",
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
        path_schedule="linear",
        divergence_estimator="exact",
    ).double()
    means = torch.tensor([[0.0, 0.0], [1.0, -2.0]], dtype=torch.float64)
    mean_net = nn.Linear(2, 2, bias=False).double()
    with torch.no_grad():
        mean_net.weight.copy_(means.T)
    model.mean_net = mean_net

    theta_eval = np.eye(2, dtype=np.float64)

    def fake_sample_flow_endpoint(**kwargs):
        theta = torch.as_tensor(kwargs["theta"], dtype=torch.float64)
        mean = theta @ means
        base = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=torch.float64)
        return base + mean

    monkeypatch.setattr(fms, "sample_flow_endpoint", fake_sample_flow_endpoint)
    result = estimate_model_symmetric_kl(
        model=model,
        theta_all=theta_eval,
        device=torch.device("cpu"),
        velocity_family="translation",
        mc_jeffreys_sample=4,
        ode_steps=1,
        ode_method="midpoint",
        batch_size=2,
    )

    expected = np.array([[0.0, 5.0], [5.0, 0.0]], dtype=np.float64)
    np.testing.assert_allclose(result.symmetric_kl_matrix, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(result.canonical_metric_matrix, expected, rtol=1e-10, atol=1e-10)
    assert result.canonical_metric_name == "model_jeffreys_symmetric_kl"


def test_x_independent_velocity_likelihood_has_zero_divergence() -> None:
    model = build_flow_skl_model(
        velocity_family="translation",
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
        path_schedule="linear",
        divergence_estimator="exact",
    ).double()
    mean = torch.tensor([0.0, 0.0], dtype=torch.float64)
    model.mean_net = ConstantNet(mean)

    x = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    theta = torch.eye(2, dtype=torch.float64)[:1]
    got = flow_endpoint_log_prob(model, x, theta, ode_steps=2, ode_method="midpoint")
    expected = -0.5 * (torch.sum(x**2, dim=1) + 2.0 * np.log(2.0 * np.pi))
    torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)


def test_sample_flow_endpoint_uses_package_solver_for_constant_velocity() -> None:
    model = build_flow_skl_model(
        velocity_family="translation",
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
        path_schedule="linear",
    )
    mean = torch.tensor([0.5, -1.25], dtype=torch.float32)
    model.mean_net = ConstantNet(mean)

    theta = np.eye(2, dtype=np.float64)[:1]
    torch.manual_seed(123)
    expected_x0 = torch.randn(4, 2, dtype=torch.float32)
    torch.manual_seed(123)
    got = sample_flow_endpoint(
        model=model,
        theta=theta,
        n_samples=4,
        device=torch.device("cpu"),
        ode_steps=3,
        ode_method="midpoint",
    )
    torch.testing.assert_close(got, expected_x0 + mean.reshape(1, -1), rtol=1e-6, atol=1e-6)


def test_shared_affine_forward_uses_centered_residual() -> None:
    model = build_flow_skl_model(
        velocity_family="shared_affine",
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
        path_schedule="linear",
        shared_affine_a_diag_jitter=0.0,
    ).double()
    means = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float64)
    a_mat = torch.tensor([[0.2, -0.4], [-0.4, 0.1]], dtype=torch.float64)
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
        shared_affine_a_diag_jitter=0.0,
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
        shared_affine_a_diag_jitter=0.0,
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
            [[0.2, -0.4], [-0.4, 0.1]],
            [[-0.3, 0.6], [0.6, 0.4]],
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
        shared_affine_a_diag_jitter=0.0,
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
        shared_affine_a_diag_jitter=0.0,
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
        shared_affine_a_diag_jitter=0.0,
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


def test_shared_affine_families_report_model_jeffreys_without_endpoint_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for idx, family in enumerate(("shared_affine", "shared_affine_scalar", "shared_affine_diag")):
        model = StandardNormalMCFlowModel(velocity_family=family, x_dim=2)
        theta_eval = _one_hot(3)
        sentinel = _sentinel_skl(3, offset=20.0 * (idx + 1))
        _patch_model_jeffreys(monkeypatch, sentinel)
        result = estimate_model_symmetric_kl(
            model=model,
            theta_all=theta_eval,
            device=torch.device("cpu"),
            velocity_family=family,
        )
        np.testing.assert_allclose(result.symmetric_kl_matrix, sentinel)
        np.testing.assert_allclose(result.canonical_metric_matrix, sentinel)
        assert result.canonical_metric_name == "model_jeffreys_symmetric_kl"


def test_condition_affine_families_report_model_jeffreys_without_endpoint_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for idx, family in enumerate(("condition_affine", "condition_affine_scalar", "condition_affine_diag")):
        model = StandardNormalMCFlowModel(velocity_family=family, x_dim=2)
        theta_eval = _one_hot(3)
        sentinel = _sentinel_skl(3, offset=30.0 * (idx + 1))
        _patch_model_jeffreys(monkeypatch, sentinel)
        result = estimate_model_symmetric_kl(
            model=model,
            theta_all=theta_eval,
            device=torch.device("cpu"),
            velocity_family=family,
        )
        np.testing.assert_allclose(result.symmetric_kl_matrix, sentinel)
        np.testing.assert_allclose(result.canonical_metric_matrix, sentinel)
        assert result.canonical_metric_name == "model_jeffreys_symmetric_kl"


def test_restricted_low_rank_affine_endpoint_uses_model_jeffreys_without_endpoint_diagnostics() -> None:
    theta = np.eye(2, dtype=np.float64)
    for family in ("shared_affine_low_rank_scalar", "shared_affine_low_rank_diag"):
        model = StandardNormalMCFlowModel(velocity_family=family, x_dim=2)
        result = estimate_model_symmetric_kl(
            model=model,
            theta_all=theta,
            device=torch.device("cpu"),
            velocity_family=family,
            mc_jeffreys_sample=8,
            ode_steps=2,
            batch_size=4,
        )
        assert result.canonical_metric_name == "model_jeffreys_symmetric_kl"
        np.testing.assert_allclose(result.canonical_metric_matrix, result.symmetric_kl_matrix)


def test_flow_skl_result_npz_omits_endpoint_distribution_fields() -> None:
    result = FlowSKLResult(
        symmetric_kl_matrix=np.zeros((1, 1), dtype=np.float64),
        canonical_metric_matrix=np.zeros((1, 1), dtype=np.float64),
        canonical_metric_name="model_jeffreys_symmetric_kl",
    )
    fields = flow_skl_result_to_npz_dict(result)
    assert "endpoint_mean" not in fields
    assert "endpoint_covariance" not in fields
    assert "endpoint_cov_or_diag" not in fields
    assert "endpoint_is_diag" not in fields


def test_scalar_fisher_finite_differences_use_model_jeffreys_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    theta = np.array([[0.0], [0.1], [0.2], [0.35]], dtype=np.float64)
    slope = np.array([2.0, -1.0], dtype=np.float64)
    model = ScalarLinearTranslationModel(slope)
    expected_fisher = float(slope @ slope)
    diff = theta[:, None, 0] - theta[None, :, 0]
    sentinel = expected_fisher * diff * diff
    _patch_model_jeffreys(monkeypatch, sentinel)
    result = estimate_model_symmetric_kl(
        model=model,
        theta_all=theta,
        device=torch.device("cpu"),
        velocity_family="translation",
        fisher_kind="full",
    )
    np.testing.assert_allclose(result.symmetric_kl_matrix, sentinel)
    np.testing.assert_allclose(result.canonical_metric_matrix, sentinel)
    assert result.canonical_metric_name == "model_jeffreys_symmetric_kl"
    np.testing.assert_allclose(result.fisher_full, expected_fisher, rtol=1e-6, atol=1e-6)
    fd = estimate_scalar_fisher_from_skl(theta, result.symmetric_kl_matrix)
    np.testing.assert_allclose(fd["fisher"], expected_fisher, rtol=1e-6, atol=1e-6)
