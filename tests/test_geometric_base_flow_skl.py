from __future__ import annotations

import importlib.util
import inspect
import math
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

from fisher import geometric_base_flow_skl as gb
from fisher.geometric_base_flow_skl import (
    ConditionTimeAffineVelocity,
    ConditionTimeLieAffine2DVelocity,
    ConditionTimeLieSimilarity2DVelocity,
    ConditionTimeLieSimilarity3DVelocity,
    HalfCircle3DBase,
    HalfCircleBase,
    LineSegmentBase,
    NoisyGeometricBase,
    SquarePerimeterBase,
    StandardNormalBase,
    build_geometric_base_velocity_model,
    estimate_pushed_base_symmetric_kl,
    estimate_smoothed_curve_symmetric_kl,
    finetune_geometric_base_cnf_likelihood,
    geometric_base_cnf_log_prob,
    log_noisy_geometric_base_density,
    log_smoothed_curve_density,
    push_base_curve,
    push_initial_points,
    train_geometric_base_affine_flow,
)


def _load_cli_module():
    path = _REPO_ROOT / "bin" / "run_geometric_base_flow_skl.py"
    spec = importlib.util.spec_from_file_location("run_geometric_base_flow_skl", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_unified_fit_check_module():
    path = _REPO_ROOT / "bin" / "run_geometric_base_fit_check.py"
    spec = importlib.util.spec_from_file_location("run_geometric_base_fit_check", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_line_fit_check_module():
    path = _REPO_ROOT / "bin" / "run_geometric_base_line_fit_check.py"
    spec = importlib.util.spec_from_file_location("run_geometric_base_line_fit_check", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_square_fit_check_module():
    path = _REPO_ROOT / "bin" / "run_geometric_base_square_fit_check.py"
    spec = importlib.util.spec_from_file_location("run_geometric_base_square_fit_check", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_half_circle_fit_check_module():
    path = _REPO_ROOT / "bin" / "run_geometric_base_half_circle_fit_check.py"
    spec = importlib.util.spec_from_file_location("run_geometric_base_half_circle_fit_check", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_half_circle_3d_fit_check_module():
    path = _REPO_ROOT / "bin" / "run_geometric_base_half_circle_3d_fit_check.py"
    spec = importlib.util.spec_from_file_location("run_geometric_base_half_circle_3d_fit_check", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class IdentitySolver:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def sample(self, *, x_init: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return x_init


class FakePath:
    def __init__(self) -> None:
        self.x0_seen: list[torch.Tensor] = []

    def sample(self, *, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor):
        self.x0_seen.append(x_0.detach().cpu())
        t_col = t.reshape(-1, 1)
        return types.SimpleNamespace(x_t=(1.0 - t_col) * x_0 + t_col * x_1, dx_t=x_1 - x_0, t=t)


class ConstantTranslationAffineVelocity(nn.Module):
    velocity_family = "constant_translation_affine"
    network_architecture = "test_constant"

    def __init__(self, shift: tuple[float, float] = (0.0, 0.0)) -> None:
        super().__init__()
        self.x_dim = 2
        self.register_buffer("shift", torch.tensor(shift, dtype=torch.float32))

    def affine_params(self, theta: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del t
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        batch = int(theta.shape[0])
        a = torch.zeros(batch, 2, 2, dtype=theta.dtype, device=theta.device)
        b = self.shift.to(device=theta.device, dtype=theta.dtype).reshape(1, 2).expand(batch, 2)
        return a, b

    def forward(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del theta, t
        return self.shift.to(device=x.device, dtype=x.dtype).reshape(1, 2).expand_as(x)


class ConstantScalingAffineVelocity(nn.Module):
    velocity_family = "constant_scaling_affine"
    network_architecture = "test_constant_scaling"

    def __init__(self, lam: float = 0.0) -> None:
        super().__init__()
        self.x_dim = 2
        self.register_buffer("lam", torch.tensor(float(lam), dtype=torch.float32))

    def affine_params(self, theta: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del t
        if theta.ndim == 1:
            theta = theta.unsqueeze(-1)
        batch = int(theta.shape[0])
        lam = self.lam.to(device=theta.device, dtype=theta.dtype)
        a = torch.eye(2, dtype=theta.dtype, device=theta.device).reshape(1, 2, 2).expand(batch, 2, 2).clone() * lam
        b = torch.zeros(batch, 2, dtype=theta.dtype, device=theta.device)
        return a, b

    def forward(self, x: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del theta, t
        return self.lam.to(device=x.device, dtype=x.dtype) * x


class SentinelBase:
    name = "sentinel_base"
    ambient_dim = 2
    intrinsic_dim = 1
    anchor = np.asarray([0.0, 0.0], dtype=np.float64)
    direction = np.asarray([1.0, 0.0], dtype=np.float64)
    u_low = -0.5
    u_high = 0.5

    def __init__(self) -> None:
        self.sample_calls = 0

    def sample(self, n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        self.sample_calls += 1
        out = torch.zeros(int(n), 2, device=device, dtype=dtype)
        out[:, 1] = 123.0
        return out


def test_line_segment_base_samples_centered_line() -> None:
    torch.manual_seed(0)
    base = LineSegmentBase(anchor=(1.0, -2.0), direction=(2.0, 0.5), u_low=-0.5, u_high=0.5)
    x, u = base.sample_with_u(128, device=torch.device("cpu"), dtype=torch.float64)

    assert x.shape == (128, 2)
    assert u.shape == (128, 1)
    assert torch.all(u >= -0.5)
    assert torch.all(u <= 0.5)
    expected = torch.tensor([[1.0, -2.0]], dtype=torch.float64) + u * torch.tensor([[2.0, 0.5]], dtype=torch.float64)
    torch.testing.assert_close(x, expected)


def test_half_circle_base_maps_unit_upper_arc() -> None:
    base = HalfCircleBase(center=(0.0, 0.0), radius=1.0)
    u = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)

    got = base.points_from_u(u)
    expected = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(got, expected, atol=1e-15, rtol=1e-15)


def test_half_circle_3d_base_maps_unit_upper_arc_in_xy_plane() -> None:
    base = HalfCircle3DBase(center=(1.0, -2.0, 0.5), radius=2.0)
    u = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)

    got = base.points_from_u(u)
    expected = torch.tensor([[3.0, -2.0, 0.5], [1.0, 0.0, 0.5], [-1.0, -2.0, 0.5]], dtype=torch.float64)

    assert base.ambient_dim == 3
    assert base.intrinsic_dim == 1
    torch.testing.assert_close(got, expected, atol=1e-12, rtol=1e-12)


def test_half_circle_base_samples_on_upper_arc() -> None:
    torch.manual_seed(0)
    base = HalfCircleBase(center=(1.0, -2.0), radius=2.0)
    x, u = base.sample_with_u(128, device=torch.device("cpu"), dtype=torch.float64)

    assert x.shape == (128, 2)
    assert u.shape == (128, 1)
    centered = x - torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    assert torch.all(u >= 0.0)
    assert torch.all(u <= 1.0)
    assert torch.all(centered[:, 1] >= 0.0)
    torch.testing.assert_close(torch.linalg.norm(centered, dim=1), 2.0 * torch.ones(128, dtype=torch.float64))


def test_noisy_geometric_base_zero_sigma_matches_clean_geometry() -> None:
    torch.manual_seed(0)
    clean = HalfCircleBase(center=(0.25, -0.5), radius=1.5)
    noisy = NoisyGeometricBase(clean, sigma=0.0)

    x, u = noisy.sample_with_u(128, device=torch.device("cpu"), dtype=torch.float64)

    torch.testing.assert_close(x, clean.points_from_u(u))
    assert noisy.ambient_dim == clean.ambient_dim
    assert noisy.intrinsic_dim == clean.intrinsic_dim
    assert noisy.u_low == pytest.approx(clean.u_low)
    assert noisy.u_high == pytest.approx(clean.u_high)


def test_noisy_geometric_base_adds_ambient_noise_only() -> None:
    torch.manual_seed(0)
    clean = HalfCircleBase(center=(0.0, 0.0), radius=1.0)
    noisy = NoisyGeometricBase(clean, sigma=0.1)
    u = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)

    torch.testing.assert_close(noisy.points_from_u(u), clean.points_from_u(u))
    x, got_u = noisy.sample_with_u(512, device=torch.device("cpu"), dtype=torch.float64)
    clean_x = clean.points_from_u(got_u)

    assert x.shape == clean_x.shape
    assert float(torch.mean(torch.linalg.norm(x - clean_x, dim=1))) > 0.05
    assert "half_circle" in noisy.name


def test_standard_normal_base_samples_full_dimensional_points() -> None:
    torch.manual_seed(0)
    base = StandardNormalBase(ambient_dim=3)
    x = base.sample(128, device=torch.device("cpu"), dtype=torch.float64)

    assert base.ambient_dim == 3
    assert base.intrinsic_dim == 3
    assert base.name == "standard_normal"
    assert x.shape == (128, 3)
    assert x.dtype == torch.float64


def test_square_perimeter_base_maps_unit_square_boundary() -> None:
    base = SquarePerimeterBase(center=(0.0, 0.0), side_length=1.0)
    u = torch.tensor([[0.0], [0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0]], dtype=torch.float64)

    got = base.points_from_u(u)
    expected = torch.tensor(
        [
            [-0.5, -0.5],
            [0.0, -0.5],
            [0.5, -0.5],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.0, 0.5],
            [-0.5, 0.5],
            [-0.5, 0.0],
            [-0.5, -0.5],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(got, expected)


def test_square_perimeter_base_samples_on_boundary() -> None:
    torch.manual_seed(0)
    base = SquarePerimeterBase(center=(1.0, -2.0), side_length=2.0)
    x, u = base.sample_with_u(128, device=torch.device("cpu"), dtype=torch.float64)

    assert x.shape == (128, 2)
    assert u.shape == (128, 1)
    centered = x - torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    assert torch.all(u >= 0.0)
    assert torch.all(u <= 4.0)
    torch.testing.assert_close(torch.max(torch.abs(centered), dim=1).values, torch.ones(128, dtype=torch.float64))


def test_condition_time_affine_velocity_is_affine_in_x() -> None:
    model = ConditionTimeAffineVelocity(theta_dim=1, x_dim=2, hidden_dim=4, depth=1)
    theta = torch.tensor([[0.25]], dtype=torch.float32)
    t = torch.tensor([[0.4]], dtype=torch.float32)
    x_a = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    x_b = torch.tensor([[-0.5, 3.0]], dtype=torch.float32)

    a, b = model.affine_params(theta, t)
    got_a = model(x_a, theta, t)
    got_b = model(x_b, theta, t)

    torch.testing.assert_close(got_a - got_b, torch.bmm(a, (x_a - x_b).unsqueeze(-1)).squeeze(-1))
    assert b.shape == (1, 2)


def test_condition_time_lie_affine_2d_velocity_uses_lie_basis_and_center() -> None:
    model = ConditionTimeLieAffine2DVelocity(theta_dim=2, x_dim=2, hidden_dim=4, depth=1)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.net[-1].bias.copy_(torch.tensor([1.0, -2.0, 0.3, 0.4, 0.5, -0.2, 2.0, -1.0]))
    theta = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    t = torch.tensor([[0.4]], dtype=torch.float32)
    x = torch.tensor([[3.0, 5.0]], dtype=torch.float32)

    v, a, c = model.lie_params(theta, t)
    got_a, got_b = model.affine_params(theta, t)
    got = model(x, theta, t)

    expected_v = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    expected_a = torch.tensor([[[0.9, -0.5], [0.1, -0.1]]], dtype=torch.float32)
    expected_c = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
    expected_b = expected_v - torch.bmm(expected_a, expected_c.unsqueeze(-1)).squeeze(-1)
    expected = expected_v + torch.bmm(expected_a, (x - expected_c).unsqueeze(-1)).squeeze(-1)

    torch.testing.assert_close(v, expected_v)
    torch.testing.assert_close(a, expected_a)
    torch.testing.assert_close(c, expected_c)
    torch.testing.assert_close(got_a, expected_a)
    torch.testing.assert_close(got_b, expected_b)
    torch.testing.assert_close(got, expected)


def test_condition_time_lie_affine_2d_velocity_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="x_dim == 2"):
        ConditionTimeLieAffine2DVelocity(theta_dim=1, x_dim=3)


def test_condition_time_lie_similarity_2d_velocity_has_no_strain_terms() -> None:
    model = ConditionTimeLieSimilarity2DVelocity(theta_dim=2, x_dim=2, hidden_dim=4, depth=1)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.net[-1].bias.copy_(torch.tensor([1.0, -2.0, 0.3, 0.4, 2.0, -1.0]))
    theta = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    t = torch.tensor([[0.4]], dtype=torch.float32)
    x = torch.tensor([[3.0, 5.0]], dtype=torch.float32)

    v, a, c = model.lie_params(theta, t)
    got_a, got_b = model.affine_params(theta, t)
    got = model(x, theta, t)

    expected_v = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    expected_a = torch.tensor([[[0.4, -0.3], [0.3, 0.4]]], dtype=torch.float32)
    expected_c = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
    expected_b = expected_v - torch.bmm(expected_a, expected_c.unsqueeze(-1)).squeeze(-1)
    expected = expected_v + torch.bmm(expected_a, (x - expected_c).unsqueeze(-1)).squeeze(-1)

    torch.testing.assert_close(v, expected_v)
    torch.testing.assert_close(a, expected_a)
    torch.testing.assert_close(c, expected_c)
    torch.testing.assert_close(got_a[:, 0, 0], got_a[:, 1, 1])
    torch.testing.assert_close(got_a[:, 0, 1], -got_a[:, 1, 0])
    torch.testing.assert_close(got_b, expected_b)
    torch.testing.assert_close(got, expected)


def test_condition_time_lie_similarity_2d_velocity_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="x_dim == 2"):
        ConditionTimeLieSimilarity2DVelocity(theta_dim=1, x_dim=3)


def test_condition_time_lie_similarity_3d_velocity_uses_skew_rotation_and_scale() -> None:
    model = ConditionTimeLieSimilarity3DVelocity(theta_dim=2, x_dim=3, hidden_dim=4, depth=1)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.net[-1].bias.copy_(
            torch.tensor([1.0, -2.0, 0.5, 0.2, -0.3, 0.4, 0.7, 2.0, -1.0, 0.25], dtype=torch.float32)
        )
    theta = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    t = torch.tensor([[0.4]], dtype=torch.float32)
    x = torch.tensor([[3.0, 5.0, -2.0]], dtype=torch.float32)

    v, a, c = model.lie_params(theta, t)
    got_a, got_b = model.affine_params(theta, t)
    got = model(x, theta, t)

    expected_v = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32)
    expected_a = torch.tensor([[[0.7, -0.4, -0.3], [0.4, 0.7, -0.2], [0.3, 0.2, 0.7]]], dtype=torch.float32)
    expected_c = torch.tensor([[2.0, -1.0, 0.25]], dtype=torch.float32)
    expected_b = expected_v - torch.bmm(expected_a, expected_c.unsqueeze(-1)).squeeze(-1)
    expected = expected_v + torch.bmm(expected_a, (x - expected_c).unsqueeze(-1)).squeeze(-1)

    torch.testing.assert_close(v, expected_v)
    torch.testing.assert_close(a, expected_a)
    torch.testing.assert_close(c, expected_c)
    torch.testing.assert_close(got_a, expected_a)
    torch.testing.assert_close(got_b, expected_b)
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(torch.diagonal(got_a, dim1=-2, dim2=-1).sum(dim=1), torch.tensor([2.1]))


def test_condition_time_lie_similarity_3d_velocity_rejects_non_3d() -> None:
    with pytest.raises(ValueError, match="x_dim == 3"):
        ConditionTimeLieSimilarity3DVelocity(theta_dim=1, x_dim=2)


def test_build_geometric_base_velocity_model_selects_lie_default_and_centered_fallback() -> None:
    lie = build_geometric_base_velocity_model(theta_dim=2, x_dim=2, hidden_dim=4, depth=1)
    similarity = build_geometric_base_velocity_model(
        velocity_family="lie-similarity-2d",
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
    )
    centered = build_geometric_base_velocity_model(
        velocity_family="centered-affine",
        theta_dim=2,
        x_dim=2,
        hidden_dim=4,
        depth=1,
    )
    similarity3d = build_geometric_base_velocity_model(
        velocity_family="lie-similarity-3d",
        theta_dim=2,
        x_dim=3,
        hidden_dim=4,
        depth=1,
    )

    assert isinstance(lie, ConditionTimeLieAffine2DVelocity)
    assert getattr(lie, "velocity_family") == "lie_affine_2d"
    assert isinstance(similarity, ConditionTimeLieSimilarity2DVelocity)
    assert getattr(similarity, "velocity_family") == "lie_similarity_2d"
    assert isinstance(similarity3d, ConditionTimeLieSimilarity3DVelocity)
    assert getattr(similarity3d, "velocity_family") == "lie_similarity_3d"
    assert getattr(centered, "velocity_family") == "condition_affine"


def test_training_loop_draws_source_from_base(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_path = FakePath()
    monkeypatch.setattr(gb, "_make_flow_matching_affine_path", lambda path_schedule: (fake_path, "linear"))
    base = SentinelBase()
    model = ConditionTimeAffineVelocity(theta_dim=1, x_dim=2, hidden_dim=4, depth=1)
    theta = np.asarray([[0.0], [0.0], [1.0], [1.0]], dtype=np.float64)
    x = np.asarray([[0.0, 0.0], [0.1, 0.0], [1.0, 0.2], [1.1, 0.3]], dtype=np.float64)

    train_geometric_base_affine_flow(
        model=model,
        base=base,  # type: ignore[arg-type]
        theta_train=theta,
        x_train=x,
        theta_val=theta,
        x_val=x,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=2,
        log_every=999,
    )

    assert base.sample_calls > 0
    assert fake_path.x0_seen
    for x0 in fake_path.x0_seen:
        torch.testing.assert_close(x0[:, 1], torch.full((int(x0.shape[0]),), 123.0))


def test_push_base_curve_zero_ode_leaves_points_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gb, "_make_flow_ode_solver", lambda model: IdentitySolver(model))
    model = ConditionTimeAffineVelocity(theta_dim=1, x_dim=2, hidden_dim=4, depth=1)
    base = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0))
    u = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.float32)

    got, got_u = push_base_curve(
        model=model,
        base=base,
        theta=np.asarray([[0.0]], dtype=np.float64),
        device=torch.device("cpu"),
        u=u,
        ode_steps=2,
    )

    torch.testing.assert_close(got, torch.tensor([[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]], dtype=torch.float32))
    torch.testing.assert_close(got_u, u)


def test_push_initial_points_constant_translation() -> None:
    model = ConstantTranslationAffineVelocity((0.25, -0.5))
    x0 = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.float32)

    got = push_initial_points(
        model=model,
        x0=x0,
        theta=np.asarray([[0.0]], dtype=np.float64),
        device=torch.device("cpu"),
        ode_steps=4,
        ode_method="midpoint",
    )

    torch.testing.assert_close(got, x0 + torch.tensor([[0.25, -0.5]], dtype=torch.float32))


def test_log_smoothed_curve_density_matches_logsumexp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gb, "_make_flow_ode_solver", lambda model: IdentitySolver(model))
    model = ConditionTimeAffineVelocity(theta_dim=1, x_dim=2, hidden_dim=4, depth=1)
    base = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0))
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    support_u = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
    sigma = 0.5

    got = log_smoothed_curve_density(
        model=model,
        base=base,
        x=x,
        theta=np.asarray([[0.0]], dtype=np.float64),
        smooth_sigma=sigma,
        density_mc_samples=2,
        device=torch.device("cpu"),
        support_u=support_u,
    )

    centers = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    log_norm = -0.5 * 2 * math.log(2.0 * math.pi * sigma * sigma)
    sq = torch.sum((x[:, None, :] - centers[None, :, :]) ** 2, dim=-1)
    expected = torch.logsumexp(log_norm - 0.5 * sq / (sigma * sigma), dim=1) - math.log(2.0)
    torch.testing.assert_close(got, expected)


def test_log_noisy_geometric_base_density_matches_logsumexp() -> None:
    clean = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0), u_low=-1.0, u_high=1.0)
    base = NoisyGeometricBase(clean, sigma=0.5)
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    support_u = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)

    got = log_noisy_geometric_base_density(x, base=base, support_u=support_u)

    centers = clean.points_from_u(support_u)
    log_norm = -0.5 * 2 * math.log(2.0 * math.pi * 0.5 * 0.5)
    sq = torch.sum((x[:, None, :] - centers[None, :, :]) ** 2, dim=-1)
    expected = torch.logsumexp(log_norm - 0.5 * sq / (0.5 * 0.5), dim=1) - math.log(2.0)
    torch.testing.assert_close(got, expected)


def test_log_noisy_geometric_base_density_accepts_explicit_sigma() -> None:
    clean = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0), u_low=-1.0, u_high=1.0)
    base = NoisyGeometricBase(clean, sigma=0.5)
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    support_u = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
    sigma = torch.tensor([0.25, 0.75], dtype=torch.float32, requires_grad=True)

    got = log_noisy_geometric_base_density(x, base=base, support_u=support_u, sigma=sigma)

    centers = clean.points_from_u(support_u)
    sq = torch.sum((x[:, None, :] - centers[None, :, :]) ** 2, dim=-1)
    log_norm = -0.5 * 2 * (math.log(2.0 * math.pi) + 2.0 * torch.log(sigma))
    expected = torch.logsumexp(log_norm[:, None] - 0.5 * sq / (sigma[:, None] * sigma[:, None]), dim=1) - math.log(2.0)
    torch.testing.assert_close(got, expected)
    (-got.mean()).backward()
    assert sigma.grad is not None
    assert torch.all(torch.isfinite(sigma.grad))


def test_geometric_base_cnf_log_prob_zero_velocity_is_base_density() -> None:
    model = ConstantTranslationAffineVelocity((0.0, 0.0))
    clean = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0), u_low=-1.0, u_high=1.0)
    base = NoisyGeometricBase(clean, sigma=0.5)
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    support_u = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)

    got = geometric_base_cnf_log_prob(
        model=model,
        base=base,
        x=x,
        theta=torch.tensor([[0.0]], dtype=torch.float32).expand(2, 1),
        support_u=support_u,
        device=torch.device("cpu"),
        ode_steps=2,
    )
    expected = log_noisy_geometric_base_density(x, base=base, support_u=support_u)

    torch.testing.assert_close(got, expected)


def test_geometric_base_cnf_log_prob_constant_translation() -> None:
    model = ConstantTranslationAffineVelocity((0.25, -0.5))
    clean = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0), u_low=-1.0, u_high=1.0)
    base = NoisyGeometricBase(clean, sigma=0.5)
    x = torch.tensor([[0.25, -0.5], [1.25, -0.5]], dtype=torch.float32)
    support_u = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)

    got = geometric_base_cnf_log_prob(
        model=model,
        base=base,
        x=x,
        theta=torch.tensor([[0.0]], dtype=torch.float32).expand(2, 1),
        support_u=support_u,
        device=torch.device("cpu"),
        ode_steps=4,
    )
    expected = log_noisy_geometric_base_density(x - torch.tensor([[0.25, -0.5]], dtype=torch.float32), base=base, support_u=support_u)

    torch.testing.assert_close(got, expected)


def test_geometric_base_cnf_log_prob_scaling_uses_backward_divergence_sign() -> None:
    lam = 0.1
    model = ConstantScalingAffineVelocity(lam)
    clean = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0), u_low=-1.0, u_high=1.0)
    base = NoisyGeometricBase(clean, sigma=0.5)
    x = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    support_u = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)

    got = geometric_base_cnf_log_prob(
        model=model,
        base=base,
        x=x,
        theta=torch.tensor([[0.0]], dtype=torch.float32),
        support_u=support_u,
        device=torch.device("cpu"),
        ode_steps=1,
        ode_method="euler",
    )
    expected_z0 = (1.0 - lam) * x
    expected = log_noisy_geometric_base_density(expected_z0, base=base, support_u=support_u) - 2.0 * lam

    torch.testing.assert_close(got, expected)


def test_geometric_base_cnf_log_prob_requires_noisy_base() -> None:
    model = ConstantTranslationAffineVelocity((0.0, 0.0))
    clean = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0), u_low=-1.0, u_high=1.0)
    x = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    support_u = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="requires NoisyGeometricBase"):
        geometric_base_cnf_log_prob(
            model=model,
            base=clean,  # type: ignore[arg-type]
            x=x,
            theta=torch.tensor([[0.0]], dtype=torch.float32),
            support_u=support_u,
            device=torch.device("cpu"),
            ode_steps=1,
        )


def test_geometric_base_cnf_likelihood_finetune_records_metadata() -> None:
    torch.manual_seed(0)
    model = ConditionTimeAffineVelocity(theta_dim=1, x_dim=2, hidden_dim=4, depth=1)
    clean = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0), u_low=-1.0, u_high=1.0)
    base = NoisyGeometricBase(clean, sigma=0.5)
    theta = np.asarray([[1.0], [1.0], [1.0], [1.0]], dtype=np.float64)
    x = np.asarray([[-0.4, 0.0], [-0.1, 0.0], [0.2, 0.0], [0.45, 0.0]], dtype=np.float64)

    meta = finetune_geometric_base_cnf_likelihood(
        model=model,
        base=base,
        theta_train=theta,
        x_train=x,
        theta_val=theta,
        x_val=x,
        condition_eval=np.asarray([[1.0]], dtype=np.float64),
        device=torch.device("cpu"),
        epochs=1,
        batch_size=2,
        lr=1e-3,
        density_points=4,
        ode_steps=2,
        initial_base_noise_sigmas=np.asarray([0.25], dtype=np.float64),
        epoch_offset=10,
        checkpoint_selection="best",
        log_every=999,
    )

    assert meta["enabled"] is True
    assert meta["epochs"] == 1
    assert meta["epoch_offset"] == 10
    assert meta["total_epochs"] == 11
    assert meta["density_points"] == 4
    assert meta["base_noise_sigma"] == pytest.approx(0.5)
    assert meta["base_noise_sigma_init"] == pytest.approx(0.5)
    np.testing.assert_allclose(meta["initial_base_noise_sigmas"], np.asarray([0.25]))
    assert meta["learn_base_noise"] is True
    assert meta["sigma_min"] == pytest.approx(1e-4)
    assert meta["checkpoint_selection"] == "best"
    assert meta["selected_epoch"] == 11
    assert np.asarray(meta["selected_base_noise_sigmas"]).shape == (1,)
    assert np.asarray(meta["best_base_noise_sigmas"]).shape == (1,)
    assert float(np.asarray(meta["selected_base_noise_sigmas"])[0]) > 0.0
    assert np.asarray(meta["train_nll_losses"]).shape == (1,)
    assert np.asarray(meta["val_nll_losses"]).shape == (1,)


def test_geometric_base_cnf_likelihood_finetune_can_keep_fixed_noise() -> None:
    torch.manual_seed(0)
    model = ConditionTimeAffineVelocity(theta_dim=1, x_dim=2, hidden_dim=4, depth=1)
    clean = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0), u_low=-1.0, u_high=1.0)
    base = NoisyGeometricBase(clean, sigma=0.5)
    theta = np.asarray([[1.0], [1.0]], dtype=np.float64)
    x = np.asarray([[-0.4, 0.0], [0.45, 0.0]], dtype=np.float64)

    meta = finetune_geometric_base_cnf_likelihood(
        model=model,
        base=base,
        theta_train=theta,
        x_train=x,
        theta_val=theta,
        x_val=x,
        condition_eval=np.asarray([[1.0]], dtype=np.float64),
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        density_points=4,
        ode_steps=1,
        learn_base_noise=False,
        log_every=999,
    )

    assert meta["learn_base_noise"] is False
    np.testing.assert_allclose(meta["selected_base_noise_sigmas"], np.asarray([0.5]))


def test_geometric_base_cnf_likelihood_finetune_requires_noisy_base() -> None:
    model = ConstantTranslationAffineVelocity((0.0, 0.0))
    base = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0))
    theta = np.asarray([[1.0], [1.0]], dtype=np.float64)
    x = np.asarray([[0.0, 0.0], [0.1, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="requires NoisyGeometricBase"):
        finetune_geometric_base_cnf_likelihood(
            model=model,
            base=base,  # type: ignore[arg-type]
            theta_train=theta,
            x_train=x,
            theta_val=theta,
            x_val=x,
            condition_eval=np.asarray([[1.0]], dtype=np.float64),
            device=torch.device("cpu"),
            epochs=1,
            batch_size=1,
        )


def test_smoothed_curve_skl_identical_conditions_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gb, "_make_flow_ode_solver", lambda model: IdentitySolver(model))
    torch.manual_seed(0)
    model = ConditionTimeAffineVelocity(theta_dim=1, x_dim=2, hidden_dim=4, depth=1)
    base = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0))

    result = estimate_smoothed_curve_symmetric_kl(
        model=model,
        base=base,
        theta_all=np.asarray([[0.0], [0.0]], dtype=np.float64),
        device=torch.device("cpu"),
        smooth_sigma=0.2,
        mc_skl_samples=8,
        density_mc_samples=8,
        ode_steps=2,
    )

    np.testing.assert_allclose(result.symmetric_kl_matrix, np.zeros((2, 2)), atol=1e-7)
    assert result.canonical_metric_name == gb.SMOOTHED_LINE_CURVE_METRIC


def test_pushed_base_skl_identical_noisy_geometric_conditions_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gb, "_make_flow_ode_solver", lambda model: IdentitySolver(model))
    torch.manual_seed(0)
    model = ConstantTranslationAffineVelocity((0.0, 0.0))
    clean = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0), u_low=-1.0, u_high=1.0)
    base = NoisyGeometricBase(clean, sigma=0.2)

    result = estimate_pushed_base_symmetric_kl(
        model=model,
        base=base,
        theta_all=np.asarray([[0.0], [0.0]], dtype=np.float64),
        device=torch.device("cpu"),
        base_noise_sigmas=np.asarray([0.2, 0.2], dtype=np.float64),
        mc_skl_samples=8,
        density_mc_samples=8,
        ode_steps=2,
    )

    np.testing.assert_allclose(result.symmetric_kl_matrix, np.zeros((2, 2)), atol=1e-7)
    assert result.canonical_metric_name == gb.PUSHED_BASE_DISTRIBUTION_METRIC
    np.testing.assert_allclose(result.train_metadata["base_noise_sigmas"], np.asarray([0.2, 0.2]))


def test_pushed_base_skl_validates_per_condition_base_sigmas() -> None:
    model = ConstantTranslationAffineVelocity((0.0, 0.0))
    clean = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0), u_low=-1.0, u_high=1.0)
    base = NoisyGeometricBase(clean, sigma=0.2)

    with pytest.raises(ValueError, match="one value per theta row"):
        estimate_pushed_base_symmetric_kl(
            model=model,
            base=base,
            theta_all=np.asarray([[0.0], [1.0]], dtype=np.float64),
            device=torch.device("cpu"),
            base_noise_sigmas=np.asarray([0.2], dtype=np.float64),
            mc_skl_samples=2,
            density_mc_samples=2,
            ode_steps=1,
        )


def test_geometric_base_flow_skl_cli_defaults() -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args([])

    assert args.toy is None
    assert args.path_schedule == "cosine"
    assert args.smooth_sigma == pytest.approx(0.12)
    assert args.mc_skl_samples == 4096
    assert args.density_mc_samples == 1024


def test_unified_geometric_base_fit_check_defaults() -> None:
    mod = _load_unified_fit_check_module()
    args = mod.build_parser().parse_args([])
    paths = mod.resolve_output_paths(None, dataset=args.dataset, velocity_family=mod.validate_dataset_velocity(args), nf_likelihood=True)

    assert args.dataset == "two-line"
    assert args.velocity_family == "lie-affine-2d"
    assert args.base_noise_sigma == pytest.approx(0.1)
    assert args.target_sigma is None
    assert args.path_schedule == "cosine"
    assert args.epochs == 50000
    assert args.early_patience == 1000
    assert args.n_per_condition == 3000
    assert args.nf_likelihood_finetune is True
    assert mod.build_parser().parse_args(["--no-nf-likelihood-finetune"]).nf_likelihood_finetune is False
    assert args.nf_epochs == 500
    assert args.nf_batch_size == 0
    assert args.nf_lr == pytest.approx(1e-4)
    assert args.nf_weight_decay == pytest.approx(0.0)
    assert args.nf_density_points == 512
    assert args.nf_checkpoint_selection == "last"
    assert args.nf_learn_base_noise is True
    assert args.nf_epoch_offset == 0
    assert mod.build_parser().parse_args(["--no-nf-learn-base-noise"]).nf_learn_base_noise is False
    assert args.nf_sigma_min == pytest.approx(1e-4)
    assert args.init_model_checkpoint is None
    assert args.skip_fm_training is False
    assert args.max_test_plot_per_condition == 600
    assert args.generated_samples_per_condition == 600
    assert args.smooth_sigma == pytest.approx(0.12)
    assert paths["png"].name == "geometric_base_fit_check.png"
    assert paths["summary"].name == "geometric_base_fit_check_summary.json"
    assert "estimate_smoothed_curve_symmetric_kl" not in Path(mod.__file__).read_text(encoding="utf-8")


def test_geometric_base_training_exposes_no_matched_source_path() -> None:
    sig = inspect.signature(train_geometric_base_affine_flow)
    assert "x0_train" not in sig.parameters
    assert "x0_val" not in sig.parameters

    mod = _load_unified_fit_check_module()
    args = mod.build_parser().parse_args(["--dataset", "two-line", "--n-per-condition", "6", "--max-test-plot-per-condition", "2"])
    data = mod.make_geometric_dataset(args)

    assert "x0_train" not in data
    assert "uses_matched_source" not in data
    assert data["theta_train"].shape[0] > 0
    assert data["theta_train"].shape[1] == 2
    assert data["theta_val"].shape[0] > 0
    assert data["theta_val"].shape[1] == 2
    assert data["theta_test"].shape[0] > 0
    assert data["theta_test"].shape[1] == 2
    assert data["theta_test_plot"].shape[0] > 0
    assert data["theta_test_plot"].shape[1] == 2
    assert data["theta_test_plot_scalar"].shape[1] == 1
    assert data["theta_train"].shape[0] + data["theta_val"].shape[0] + data["theta_test"].shape[0] == 12


@pytest.mark.parametrize(
    ("dataset", "expected_dim", "expected_conditions"),
    [
        ("two-line", 2, 2),
        ("one-line", 2, 1),
        ("two-square", 2, 2),
        ("one-square", 2, 1),
        ("two-half-circle", 2, 2),
        ("one-half-circle", 2, 1),
        ("two-half-circle-3d", 3, 2),
        ("one-half-circle-3d", 3, 1),
    ],
)
def test_unified_geometric_base_fit_check_builds_dataset(dataset: str, expected_dim: int, expected_conditions: int) -> None:
    mod = _load_unified_fit_check_module()
    args = mod.build_parser().parse_args(
        [
            "--dataset",
            dataset,
            "--n-per-condition",
            "12",
            "--max-test-plot-per-condition",
            "2",
        ]
    )
    data = mod.make_geometric_dataset(args)

    assert data["ambient_dim"] == expected_dim
    assert data["theta_encoding"] == "one_hot"
    assert data["condition_eval"].shape == (expected_conditions, expected_conditions)
    assert data["theta_train"].shape[1] == expected_conditions
    assert data["x_train"].shape[1] == expected_dim
    assert data["target_curves"][0].shape[1] == expected_dim


def test_unified_geometric_base_fit_check_one_condition_uses_second_preset() -> None:
    mod = _load_unified_fit_check_module()

    line = mod.make_geometric_dataset(mod.build_parser().parse_args(["--dataset", "one-line", "--n-per-condition", "6"]))
    square = mod.make_geometric_dataset(mod.build_parser().parse_args(["--dataset", "one-square", "--n-per-condition", "6"]))
    half = mod.make_geometric_dataset(mod.build_parser().parse_args(["--dataset", "one-half-circle", "--n-per-condition", "6"]))
    half3d = mod.make_geometric_dataset(mod.build_parser().parse_args(["--dataset", "one-half-circle-3d", "--n-per-condition", "6"]))

    np.testing.assert_allclose(line["condition_values"], np.asarray([[3.0 * math.pi / 4.0]]))
    np.testing.assert_allclose(square["condition_values"], np.asarray([[math.pi / 4.0]]))
    assert half["condition_labels"] == ["lower half-circle"]
    assert half3d["condition_labels"] == ["lower half-circle 3D"]
    np.testing.assert_allclose(line["condition_eval"], np.ones((1, 1)))
    np.testing.assert_allclose(square["condition_eval"], np.ones((1, 1)))


def test_unified_geometric_base_fit_check_velocity_validation() -> None:
    mod = _load_unified_fit_check_module()

    args = mod.build_parser().parse_args(["--dataset", "two-square", "--velocity-family", "2D similarity"])
    assert mod.validate_dataset_velocity(args) == "lie-similarity-2d"

    with pytest.raises(ValueError, match="3D half-circle datasets require"):
        mod.validate_dataset_velocity(mod.build_parser().parse_args(["--dataset", "two-half-circle-3d", "--velocity-family", "lie-similarity-2d"]))
    with pytest.raises(ValueError, match="only valid for 3D half-circle"):
        mod.validate_dataset_velocity(mod.build_parser().parse_args(["--dataset", "two-square", "--velocity-family", "lie-similarity-3d"]))
    with pytest.raises(ValueError, match="base-noise-sigma"):
        mod.validate_dataset_velocity(mod.build_parser().parse_args(["--dataset", "two-square", "--base-noise-sigma", "0.0"]))


def test_unified_geometric_base_fit_check_base_geometry_override() -> None:
    mod = _load_unified_fit_check_module()

    args = mod.build_parser().parse_args(["--dataset", "two-square", "--base-geometry", "standard-normal", "--base-noise-sigma", "0.0"])
    assert mod.validate_dataset_velocity(args) == "lie-affine-2d"
    data = mod.make_geometric_dataset(args)
    base = mod.make_base(args, dataset_kind=str(data["dataset_kind"]), ambient_dim=int(data["ambient_dim"]))
    assert isinstance(base, StandardNormalBase)
    assert base.ambient_dim == 2

    inferred_args = mod.build_parser().parse_args(["--dataset", "two-square"])
    inferred_data = mod.make_geometric_dataset(inferred_args)
    inferred_base = mod.make_base(
        inferred_args,
        dataset_kind=str(inferred_data["dataset_kind"]),
        ambient_dim=int(inferred_data["ambient_dim"]),
    )
    assert isinstance(inferred_base, NoisyGeometricBase)
    assert isinstance(inferred_base.base, SquarePerimeterBase)


def test_geometric_base_fit_check_wrappers_route_to_unified_datasets() -> None:
    assert _load_line_fit_check_module().DEFAULT_DATASET == "two-line"
    assert _load_square_fit_check_module().DEFAULT_DATASET == "two-square"
    assert _load_half_circle_fit_check_module().DEFAULT_DATASET == "two-half-circle"
    assert _load_half_circle_3d_fit_check_module().DEFAULT_DATASET == "two-half-circle-3d"
