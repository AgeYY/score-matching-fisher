from __future__ import annotations

import importlib.util
import inspect
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


def _load_run_flow_matching_skl_module():
    path = _REPO_ROOT / "bin" / "run_flow_matching_skl.py"
    spec = importlib.util.spec_from_file_location("run_flow_matching_skl", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def test_flow_skl_default_t_eps_is_small_endpoint_clamp() -> None:
    sig = inspect.signature(train_flow_skl_model)
    assert sig.parameters["t_eps"].default == pytest.approx(0.0005)
    assert sig.parameters["ema_alpha"].default == pytest.approx(0.05)

    mod = _load_run_flow_matching_skl_module()
    args = mod.build_parser().parse_args([])
    assert args.t_eps == pytest.approx(0.0005)
    assert args.early_ema_alpha == pytest.approx(0.05)


def test_centered_fixed_radius_normalize_behavior_is_unchanged() -> None:
    raw = torch.tensor([[1.0, 2.0, 5.0], [3.0, 6.0, 9.0]], dtype=torch.float64)
    got = centered_radius_normalize(raw, radius=2.0)
    centered = raw - raw.mean(dim=1, keepdim=True)
    expected = 2.0 * centered / torch.linalg.norm(centered, dim=1, keepdim=True)
    torch.testing.assert_close(got, expected)


def test_train_flow_skl_early_stopping_uses_ema_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    monitor_values = [10.0, 9.0, 11.0, 12.0]
    ema_calls: list[tuple[float | None, float, float]] = []

    def fake_scalar_val_ema_update(prev: float | None, mean_val_loss: float, ema_alpha: float) -> float:
        ema_calls.append((prev, float(mean_val_loss), float(ema_alpha)))
        return monitor_values[len(ema_calls) - 1]

    monkeypatch.setattr(fms, "scalar_val_ema_update", fake_scalar_val_ema_update)
    model = build_flow_skl_model(
        velocity_family="translation",
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
        path_schedule="linear",
    )
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
        epochs=10,
        batch_size=2,
        patience=2,
        min_delta=0.0,
        ema_alpha=0.25,
        log_every=999,
    )

    np.testing.assert_allclose(out["val_monitor_losses"], monitor_values)
    assert out["best_val_loss"] == pytest.approx(9.0)
    assert out["best_epoch"] == 2
    assert out["stopped_epoch"] == 4
    assert out["stopped_early"] is True
    assert out["early_ema_alpha"] == pytest.approx(0.25)
    assert len(out["val_losses"]) == 4
    assert [call[2] for call in ema_calls] == [pytest.approx(0.25)] * 4


def test_train_flow_skl_rejects_invalid_ema_alpha() -> None:
    model = build_flow_skl_model(
        velocity_family="translation",
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
    )
    theta = np.eye(2, dtype=np.float64)
    x = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="ema_alpha"):
        train_flow_skl_model(
            model=model,
            theta_train=theta,
            x_train=x,
            theta_val=theta,
            x_val=x,
            device=torch.device("cpu"),
            velocity_family="translation",
            ema_alpha=0.0,
        )


def test_conditioned_film_net_preserves_shape_dtype_and_uses_linear_theta_embedding() -> None:
    torch.manual_seed(123)
    net = fms._ConditionedFiLMNet(  # type: ignore[attr-defined]
        trunk_dim=3,
        theta_dim=2,
        out_dim=2,
        hidden_dim=5,
        depth=2,
        final_gain=0.01,
    ).double()
    x = torch.randn(4, 3, dtype=torch.float64)
    theta = torch.randn(4, 2, dtype=torch.float64)
    t = torch.rand(4, 1, dtype=torch.float64)
    y = net(x, theta, t)
    assert y.shape == (4, 2)
    assert y.dtype == torch.float64
    assert net.network_architecture == "film"
    assert isinstance(net.theta_embedding, nn.Linear)
    assert isinstance(net.condition_mlp, nn.Sequential)
    for block in net.blocks:
        assert isinstance(block.norm, nn.LayerNorm)
        assert isinstance(block.branch, nn.Sequential)
        torch.testing.assert_close(block.film.weight, torch.zeros_like(block.film.weight))
        torch.testing.assert_close(block.film.bias, torch.zeros_like(block.film.bias))


def _first_linear_in_features(module: nn.Sequential) -> int:
    first = module[0]
    assert isinstance(first, nn.Linear)
    return int(first.in_features)


def test_build_flow_skl_model_uses_mlp_heads_and_film_nonlinear_subnets() -> None:
    for family in VELOCITY_FAMILIES:
        model = build_flow_skl_model(
            velocity_family=family,
            theta_dim=2,
            x_dim=3,
            hidden_dim=4,
            depth=1,
            low_rank_dim=1,
            path_schedule="linear",
        )
        if family == "condition_quadratic":
            assert getattr(model, "network_architecture") == "quadratic_conditioned_mlp"
        elif family == "condition_tanh":
            assert getattr(model, "network_architecture") == "tanh_conditioned_shared_trunk"
        elif family == "condition_tanh_linear":
            assert getattr(model, "network_architecture") == "tanh_linear_conditioned_shared_trunk"
        else:
            assert getattr(model, "network_architecture") == "film"
        if family in (
            "translation",
            "translation_fixed_norm",
            "translation_centered_fixed_norm",
        ):
            assert isinstance(model.mean_net, nn.Sequential)
            assert _first_linear_in_features(model.mean_net) == 2
        elif family == "nonlinear":
            assert type(model).__name__ == "ConditionalNonlinearXFlowFiLM"
            assert isinstance(model.net, fms._ConditionedFiLMNet)  # type: ignore[attr-defined]
            assert model.net.trunk_dim == 3
            assert model.net.theta_dim == 2
            assert isinstance(model.net.theta_embedding, nn.Linear)
        elif family == "condition_quadratic":
            assert isinstance(model.b_net, nn.Sequential)
            assert _first_linear_in_features(model.b_net) == 3
            assert isinstance(model.a_net, nn.Sequential)
            assert _first_linear_in_features(model.a_net) == 3
            assert isinstance(model.q_net, nn.Sequential)
            assert _first_linear_in_features(model.q_net) == 3
            assert model.n_quadratic_features == 6
        elif family in ("condition_tanh", "condition_tanh_linear"):
            assert isinstance(model.trunk, nn.Sequential)
            assert _first_linear_in_features(model.trunk) == 3
            assert isinstance(model.b_head, nn.Linear)
            assert isinstance(model.u_head, nn.Linear)
            assert isinstance(model.w_head, nn.Linear)
            assert isinstance(model.d_head, nn.Linear)
            assert model.b_head.in_features == 4
            assert model.u_head.in_features == 4
            assert model.w_head.in_features == 4
            assert model.d_head.in_features == 4
            assert model.b_head.out_features == 3
            assert model.u_head.out_features == 3
            assert model.w_head.out_features == 3
            assert model.d_head.out_features == 1
            if family == "condition_tanh_linear":
                assert isinstance(model.a_net, nn.Sequential)
                assert _first_linear_in_features(model.a_net) == 1
        else:
            assert isinstance(model.b_net, nn.Sequential)
            assert _first_linear_in_features(model.b_net) == 2
            assert isinstance(model.a_net, nn.Sequential)
            if family.startswith("condition_affine"):
                assert _first_linear_in_features(model.a_net) == 3
            else:
                assert _first_linear_in_features(model.a_net) == 1
            if "low_rank" in family:
                assert isinstance(model.h_net, fms._ConditionedFiLMNet)  # type: ignore[attr-defined]
                assert model.h_net.trunk_dim == 1
                assert model.h_net.theta_dim == 2
                assert isinstance(model.h_net.theta_embedding, nn.Linear)
                assert all(isinstance(block.norm, nn.LayerNorm) for block in model.h_net.blocks)


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
    assert out["network_architecture"] == "film"
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


def test_shared_affine_low_rank_can_use_fixed_oracle_basis() -> None:
    basis = np.asarray([[1.0], [0.0]], dtype=np.float64)
    model = build_flow_skl_model(
        velocity_family="shared_affine_low_rank",
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
        low_rank_dim=1,
        path_schedule="linear",
        low_rank_basis=basis,
    ).double()

    assert model.low_rank_basis_mode == "fixed"
    assert "fixed_u" in dict(model.named_buffers())
    assert not any(name.startswith("u_layer") for name, _ in model.named_parameters())
    torch.testing.assert_close(model.U, torch.as_tensor(basis, dtype=torch.float64))


def test_shared_affine_low_rank_rejects_nonorthonormal_fixed_basis() -> None:
    with pytest.raises(ValueError, match="orthonormal"):
        build_flow_skl_model(
            velocity_family="shared_affine_low_rank",
            theta_dim=2,
            x_dim=2,
            hidden_dim=4,
            depth=1,
            low_rank_dim=1,
            path_schedule="linear",
            low_rank_basis=np.asarray([[2.0], [0.0]], dtype=np.float64),
        )


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


def test_condition_tanh_forward_uses_shared_trunk_and_heads() -> None:
    model = build_flow_skl_model(
        velocity_family="condition_tanh",
        theta_dim=2,
        x_dim=2,
        hidden_dim=5,
        depth=1,
        path_schedule="linear",
        divergence_estimator="exact",
    ).double()
    assert type(model).__name__ == "ConditionTanhFlowSKLModel"
    assert isinstance(model.trunk, nn.Sequential)
    assert all(isinstance(getattr(model, name), nn.Linear) for name in ("b_head", "u_head", "w_head", "d_head"))

    x = torch.tensor([[0.3, -0.4], [1.0, 0.5], [-0.2, 0.7]], dtype=torch.float64)
    theta = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    t = torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.float64)
    y = model(x, theta, t)
    assert y.shape == x.shape
    assert y.dtype == torch.float64

    b, u, w, d = model.parameters_for(theta, t)
    expected = b + u * torch.tanh(torch.sum(w * x, dim=1, keepdim=True) + d)
    torch.testing.assert_close(y, expected, rtol=1e-12, atol=1e-12)


def test_condition_tanh_endpoint_likelihood_is_finite() -> None:
    model = build_flow_skl_model(
        velocity_family="condition_tanh",
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
        path_schedule="linear",
        divergence_estimator="exact",
    )
    x = torch.tensor([[0.1, -0.2], [0.4, 0.3]], dtype=torch.float32)
    theta = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    logp = flow_endpoint_log_prob(model, x, theta, ode_steps=2, ode_method="midpoint")
    assert logp.shape == (2,)
    assert torch.isfinite(logp).all()


def test_condition_tanh_linear_forward_adds_shared_time_linear_term() -> None:
    model = build_flow_skl_model(
        velocity_family="condition_tanh_linear",
        theta_dim=2,
        x_dim=2,
        hidden_dim=5,
        depth=1,
        path_schedule="linear",
        divergence_estimator="exact",
    ).double()
    assert type(model).__name__ == "ConditionTanhLinearFlowSKLModel"
    assert isinstance(model.trunk, nn.Sequential)
    assert isinstance(model.a_net, nn.Sequential)

    x = torch.tensor([[0.3, -0.4], [1.0, 0.5], [-0.2, 0.7]], dtype=torch.float64)
    theta = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    t = torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.float64)
    y = model(x, theta, t)
    assert y.shape == x.shape
    assert y.dtype == torch.float64

    b, u, w, d = model.parameters_for(theta, t)
    expected_tanh = b + u * torch.tanh(torch.sum(w * x, dim=1, keepdim=True) + d)
    expected = expected_tanh + torch.bmm(model.A(t), x.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(y, expected, rtol=1e-12, atol=1e-12)


def test_condition_tanh_linear_endpoint_likelihood_is_finite() -> None:
    model = build_flow_skl_model(
        velocity_family="condition_tanh_linear",
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
        path_schedule="linear",
        divergence_estimator="exact",
    )
    x = torch.tensor([[0.1, -0.2], [0.4, 0.3]], dtype=torch.float32)
    theta = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    logp = flow_endpoint_log_prob(model, x, theta, ode_steps=2, ode_method="midpoint")
    assert logp.shape == (2,)
    assert torch.isfinite(logp).all()


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
    assert fields["network_architecture"][0] == "film"


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


class ScalarAffineIdentityCovModel(nn.Module):
    def __init__(self, slope: np.ndarray) -> None:
        super().__init__()
        self.velocity_family = "condition_affine"
        self.x_dim = int(np.asarray(slope).size)
        self.register_buffer("slope", torch.as_tensor(slope, dtype=torch.float32).reshape(1, -1))

    def endpoint_mean(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        return theta.to(self.slope.dtype) @ self.slope

    def A(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del theta
        batch = int(t.reshape(-1, 1).shape[0])
        return torch.zeros(batch, self.x_dim, self.x_dim, dtype=self.slope.dtype, device=self.slope.device)


def test_affine_mixed_covariance_fisher_zero_a_closed_form() -> None:
    model = ScalarAffineIdentityCovModel(np.asarray([2.0, -1.0], dtype=np.float64))
    theta = np.asarray([0.0, 0.5, 1.0], dtype=np.float64).reshape(-1, 1)

    got = fms.estimate_affine_mixed_symmetric_kl_fisher(
        model=model,
        theta_all=theta,
        device=torch.device("cpu"),
        ridge=0.0,
        ode_steps=4,
    )

    expected_skl = np.asarray([1.25, 1.25], dtype=np.float64)
    expected_matrix = np.asarray(
        [
            [0.0, 1.25, 0.0],
            [1.25, 0.0, 1.25],
            [0.0, 1.25, 0.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(got["adjacent_symmetric_kl"], expected_skl, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got["symmetric_kl_matrix"], expected_matrix, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got["canonical_metric_matrix"], expected_matrix, rtol=1e-12, atol=1e-12)
    assert got["canonical_metric_name"] == "mixed_affine_symmetric_kl"
    np.testing.assert_allclose(got["fisher"], np.asarray([5.0, 5.0]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got["mixed_covariance"], np.repeat(np.eye(2)[None, :, :], 2, axis=0))

    compat = fms.estimate_affine_mixed_covariance_fisher(
        model=model,
        theta_all=theta,
        device=torch.device("cpu"),
        ridge=0.0,
        ode_steps=4,
    )
    np.testing.assert_allclose(compat["fisher"], got["fisher"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(compat["symmetric_kl_matrix"], got["symmetric_kl_matrix"], rtol=1e-12, atol=1e-12)


def test_adjacent_model_jeffreys_fisher_uses_only_adjacent_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyModel(nn.Module):
        x_dim = 1

    calls: list[tuple[str, float]] = []

    def fake_sample_flow_endpoint(*, model, theta, n_samples, device, ode_steps, ode_method):
        del model, n_samples, device, ode_steps, ode_method
        val = float(np.asarray(theta).reshape(-1)[0])
        calls.append(("sample", val))
        return torch.full((3, 1), val, dtype=torch.float32)

    def fake_log_prob_model(*, model, x, theta, device, ode_steps, batch_size, solve_jitter, quadrature_steps, ode_method):
        del model, x, device, ode_steps, batch_size, solve_jitter, quadrature_steps, ode_method
        val = float(np.asarray(theta).reshape(-1)[0])
        calls.append(("logp", val))
        return np.full(3, -val, dtype=np.float64)

    monkeypatch.setattr(fms, "sample_flow_endpoint", fake_sample_flow_endpoint)
    monkeypatch.setattr(fms, "_log_prob_model", fake_log_prob_model)

    theta = np.asarray([0.0, 2.0, 5.0], dtype=np.float64).reshape(-1, 1)
    got = fms.estimate_adjacent_model_jeffreys_fisher(
        model=DummyModel(),
        theta_all=theta,
        device=torch.device("cpu"),
        mc_jeffreys_sample=3,
    )

    np.testing.assert_allclose(got["adjacent_jeffreys"], np.asarray([0.0, 0.0]))
    np.testing.assert_allclose(got["fisher"], np.asarray([0.0, 0.0]))
    sample_thetas = [val for kind, val in calls if kind == "sample"]
    assert sample_thetas == [0.0, 2.0, 2.0, 5.0]
