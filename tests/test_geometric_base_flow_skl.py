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
    LineSegmentBase,
    SquarePerimeterBase,
    estimate_smoothed_curve_symmetric_kl,
    finetune_geometric_base_nll,
    geometric_smoothed_curve_nll_loss,
    log_smoothed_curve_density,
    push_base_curve,
    train_geometric_base_affine_flow,
)


def _load_cli_module():
    path = _REPO_ROOT / "bin" / "run_geometric_base_flow_skl.py"
    spec = importlib.util.spec_from_file_location("run_geometric_base_flow_skl", path)
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


def test_push_base_curve_affine_map_zero_velocity_leaves_points_unchanged() -> None:
    model = ConstantTranslationAffineVelocity((0.0, 0.0))
    base = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0))
    u = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.float32)

    got, got_u = gb._push_base_curve_affine_map(
        model=model,
        base=base,
        theta=np.asarray([[0.0], [1.0]], dtype=np.float64),
        device=torch.device("cpu"),
        u=u,
        ode_steps=4,
        ode_method="midpoint",
    )

    expected = torch.tensor(
        [[[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]], [[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(got_u, u)


def test_push_base_curve_affine_map_constant_translation() -> None:
    model = ConstantTranslationAffineVelocity((1.25, -0.5))
    base = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0))
    u = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.float32)

    got, _ = gb._push_base_curve_affine_map(
        model=model,
        base=base,
        theta=np.asarray([[0.0]], dtype=np.float64),
        device=torch.device("cpu"),
        u=u,
        ode_steps=5,
        ode_method="euler",
    )

    expected = torch.tensor([[[0.75, -0.5], [1.25, -0.5], [1.75, -0.5]]], dtype=torch.float32)
    torch.testing.assert_close(got, expected)


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


def test_geometric_smoothed_curve_nll_loss_matches_logsumexp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gb, "_make_flow_ode_solver", lambda model: IdentitySolver(model))
    model = ConditionTimeAffineVelocity(theta_dim=1, x_dim=2, hidden_dim=4, depth=1)
    base = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0))
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    support_u = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
    sigma_min = 1e-4
    sigma = 0.5
    raw_sigma = torch.tensor(math.log(math.expm1(sigma - sigma_min)), dtype=torch.float32)

    got, got_sigma = geometric_smoothed_curve_nll_loss(
        model=model,
        base=base,
        x=x,
        theta=np.asarray([[0.0]], dtype=np.float64),
        raw_sigma=raw_sigma,
        sigma_min=sigma_min,
        u_grid=support_u,
        device=torch.device("cpu"),
        ode_steps=2,
    )

    centers = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    log_norm = -0.5 * 2 * math.log(2.0 * math.pi * sigma * sigma)
    sq = torch.sum((x[:, None, :] - centers[None, :, :]) ** 2, dim=-1)
    log_prob = torch.logsumexp(log_norm - 0.5 * sq / (sigma * sigma), dim=1) - math.log(2.0)
    torch.testing.assert_close(got, -log_prob.mean())
    torch.testing.assert_close(got_sigma, torch.tensor(sigma, dtype=torch.float32))


def test_affine_map_nll_matches_particle_ode_for_constant_translation() -> None:
    model = ConstantTranslationAffineVelocity((0.25, -0.1))
    base = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0))
    x = torch.tensor([[0.0, 0.0], [1.0, -0.1]], dtype=torch.float32)
    support_u = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float32)
    sigma_min = 1e-4
    sigma = 0.5
    raw_sigma = torch.tensor(math.log(math.expm1(sigma - sigma_min)), dtype=torch.float32)

    particle, particle_sigma = geometric_smoothed_curve_nll_loss(
        model=model,
        base=base,
        x=x,
        theta=np.asarray([[0.0]], dtype=np.float64),
        raw_sigma=raw_sigma,
        sigma_min=sigma_min,
        u_grid=support_u,
        device=torch.device("cpu"),
        ode_steps=4,
        ode_method="midpoint",
        endpoint_solver="particle-ode",
    )
    fast, fast_sigma = geometric_smoothed_curve_nll_loss(
        model=model,
        base=base,
        x=x,
        theta=np.asarray([[0.0]], dtype=np.float64),
        raw_sigma=raw_sigma,
        sigma_min=sigma_min,
        u_grid=support_u,
        device=torch.device("cpu"),
        ode_steps=4,
        ode_method="midpoint",
        endpoint_solver="affine-map",
    )

    torch.testing.assert_close(fast, particle)
    torch.testing.assert_close(fast_sigma, particle_sigma)


def test_geometric_base_nll_finetune_records_metadata_and_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(gb, "_make_flow_ode_solver", lambda model: IdentitySolver(model))
    model = ConditionTimeAffineVelocity(theta_dim=1, x_dim=2, hidden_dim=4, depth=1)
    base = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0))
    theta = np.asarray([[1.0], [1.0], [1.0], [1.0]], dtype=np.float64)
    x = np.asarray([[-0.4, 0.0], [-0.1, 0.0], [0.2, 0.0], [0.45, 0.0]], dtype=np.float64)

    meta = finetune_geometric_base_nll(
        model=model,
        base=base,
        theta_train=theta,
        x_train=x,
        theta_val=theta,
        x_val=x,
        condition_eval=np.asarray([[1.0]], dtype=np.float64),
        device=torch.device("cpu"),
        epochs=2,
        batch_size=2,
        lr=1e-3,
        sigma_init=0.2,
        n_particles=4,
        ode_steps=2,
        save_checkpoints=True,
        checkpoint_dir=tmp_path / "nll_checkpoints",
        checkpoint_every=1,
        log_every=999,
    )

    assert meta["enabled"] is True
    assert meta["epochs"] == 2
    assert meta["selected_epoch"] == 2
    assert np.asarray(meta["train_nll_losses"]).shape == (2,)
    assert np.asarray(meta["val_nll_losses"]).shape == (2,)
    assert np.asarray(meta["learned_sigmas"]).shape == (1,)
    assert float(np.asarray(meta["learned_sigmas"])[0]) > 0.0
    assert meta["save_checkpoints"] is True
    assert meta["checkpoint_every"] == 1
    checkpoint_paths = [Path(p) for p in meta["checkpoint_paths"]]
    assert [p.name for p in checkpoint_paths] == ["nll_epoch_000001.pt", "nll_epoch_000002.pt"]
    for path in checkpoint_paths:
        assert path.is_file()
    checkpoint = torch.load(checkpoint_paths[-1], map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 2
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert checkpoint["raw_sigma"].shape == (1,)
    assert np.asarray(checkpoint["train_nll_losses"]).shape == (2,)
    assert np.asarray(checkpoint["val_nll_losses"]).shape == (2,)
    assert np.asarray(checkpoint["learned_sigmas"]).shape == (1,)
    assert checkpoint["config"]["n_particles"] == 4
    assert checkpoint["config"]["nll_endpoint_solver"] == "particle_ode"
    assert checkpoint["config"]["checkpoint_every"] == 1


def test_geometric_base_nll_finetune_records_affine_map_solver() -> None:
    torch.manual_seed(0)
    model = ConditionTimeAffineVelocity(theta_dim=1, x_dim=2, hidden_dim=4, depth=1)
    base = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0))
    theta = np.asarray([[1.0], [1.0], [1.0], [1.0]], dtype=np.float64)
    x = np.asarray([[-0.4, 0.0], [-0.1, 0.0], [0.2, 0.0], [0.45, 0.0]], dtype=np.float64)

    meta = finetune_geometric_base_nll(
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
        sigma_init=0.2,
        n_particles=4,
        ode_steps=2,
        nll_endpoint_solver="affine-map",
        log_every=999,
    )

    assert meta["nll_endpoint_solver"] == "affine_map"
    assert np.asarray(meta["train_nll_losses"]).shape == (1,)
    assert np.asarray(meta["val_nll_losses"]).shape == (1,)


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


def test_geometric_base_flow_skl_cli_defaults() -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args([])

    assert args.toy is None
    assert args.path_schedule == "cosine"
    assert args.smooth_sigma == pytest.approx(0.12)
    assert args.mc_skl_samples == 4096
    assert args.density_mc_samples == 1024


def test_geometric_base_line_fit_check_defaults() -> None:
    mod = _load_line_fit_check_module()
    args = mod.build_parser().parse_args([])
    paths = mod.resolve_output_paths(args.output_dir)

    assert args.theta_values == "0.7853981633974483,2.356194490192345"
    assert args.path_schedule == "cosine"
    assert args.epochs == 50000
    assert args.early_patience == 1000
    assert args.n_per_theta == 3000
    assert args.train_frac == pytest.approx(0.7)
    assert args.val_frac == pytest.approx(0.15)
    assert args.nll_finetune is False
    assert args.nll_epochs == 2000
    assert args.nll_batch_size == 0
    assert args.nll_lr == pytest.approx(1e-4)
    assert args.nll_particles == 128
    assert args.nll_sigma_init == pytest.approx(0.1)
    assert args.nll_endpoint_solver == "particle-ode"
    assert args.nll_checkpoint_selection == "last"
    assert args.max_test_plot_per_theta == 500
    assert args.smooth_sigma == pytest.approx(0.12)
    assert paths["png"].name == "geometric_base_line_fit_check.png"
    assert paths["svg"].name == "geometric_base_line_fit_check.svg"
    assert paths["summary"].name == "geometric_base_line_fit_check_summary.json"
    theta = mod._parse_theta_values(args.theta_values)
    data = mod._make_noisy_line_data(args, theta)
    np.testing.assert_allclose(data["condition_eval"], np.eye(2))
    assert data["theta_encoding"] == "one_hot"
    assert data["theta_train"].shape[1] == 2


def test_geometric_base_training_exposes_no_matched_source_path() -> None:
    sig = inspect.signature(train_geometric_base_affine_flow)
    assert "x0_train" not in sig.parameters
    assert "x0_val" not in sig.parameters

    mod = _load_line_fit_check_module()
    args = mod.build_parser().parse_args(["--n-per-theta", "6", "--max-test-plot-per-theta", "2"])
    theta = mod._parse_theta_values(args.theta_values)
    data = mod._make_noisy_line_data(args, theta)

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


def test_geometric_base_square_fit_check_defaults() -> None:
    mod = _load_square_fit_check_module()
    args = mod.build_parser().parse_args([])
    paths = mod.resolve_output_paths(args.output_dir)

    assert args.theta_values == "0.0,0.7853981633974483"
    assert args.side_length == pytest.approx(2.0)
    assert args.base_side_length == pytest.approx(1.0)
    assert args.target_sigma == pytest.approx(0.2)
    assert args.path_schedule == "cosine"
    assert args.epochs == 50000
    assert args.early_patience == 1000
    assert args.n_per_theta == 3000
    assert args.nll_finetune is False
    assert args.nll_epochs == 2000
    assert args.nll_particles == 128
    assert args.nll_sigma_init == pytest.approx(0.1)
    assert args.nll_endpoint_solver == "particle-ode"
    assert args.nll_checkpoint_selection == "last"
    assert args.nll_save_checkpoints is True
    assert args.nll_checkpoint_every == 0
    assert args.nll_checkpoint_dir is None
    assert args.curve_points_per_edge == 100
    assert paths["png"].name == "geometric_base_square_fit_check.png"
    assert paths["svg"].name == "geometric_base_square_fit_check.svg"
    assert paths["summary"].name == "geometric_base_square_fit_check_summary.json"
    theta = mod._parse_theta_values(args.theta_values)
    data = mod._make_noisy_square_data(args, theta)
    np.testing.assert_allclose(data["condition_eval"], np.eye(2))
    assert data["theta_encoding"] == "one_hot"
    assert data["theta_train"].shape[1] == 2


def test_geometric_base_square_fit_check_allows_single_condition() -> None:
    mod = _load_square_fit_check_module()
    args = mod.build_parser().parse_args(["--theta-values", "0.7853981633974483", "--n-per-theta", "12"])
    theta = mod._parse_theta_values(args.theta_values)
    data = mod._make_noisy_square_data(args, theta)

    assert theta.shape == (1, 1)
    np.testing.assert_allclose(data["condition_eval"], np.ones((1, 1)))
    assert data["theta_train"].shape[1] == 1
    assert data["theta_val"].shape[1] == 1
    assert data["theta_test"].shape[1] == 1
