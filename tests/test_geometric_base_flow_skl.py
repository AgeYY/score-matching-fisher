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
    estimate_smoothed_curve_symmetric_kl,
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
    path = _REPO_ROOT / "tests" / "run_geometric_base_line_fit_check.py"
    spec = importlib.util.spec_from_file_location("run_geometric_base_line_fit_check", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_ot_compare_module():
    path = _REPO_ROOT / "tests" / "run_geometric_base_line_fit_ot_compare.py"
    spec = importlib.util.spec_from_file_location("run_geometric_base_line_fit_ot_compare", path)
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


def test_exact_ot_plan_sampler_matches_assignment_plan() -> None:
    x0 = torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float32)
    x1 = torch.tensor([[9.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

    sampler = gb.MinibatchOTPlanSampler(method="exact")
    pi, plan_cost = sampler.get_map(x0, x1)

    np.testing.assert_allclose(pi, np.asarray([[0.0, 0.5], [0.5, 0.0]]), atol=1e-12)
    assert plan_cost == pytest.approx(1.0)


def test_torch_sinkhorn_plan_has_balanced_marginals() -> None:
    x0 = torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float32)
    x1 = torch.tensor([[9.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

    sampler = gb.MinibatchOTPlanSampler(method="sinkhorn", reg=1.0, sinkhorn_iters=200)
    pi, plan_cost = sampler.get_torch_map(x0, x1)

    assert sampler.method == "torch_sinkhorn"
    assert pi.device == x0.device
    assert torch.all(torch.isfinite(pi))
    assert torch.all(pi >= 0.0)
    torch.testing.assert_close(torch.sum(pi), torch.tensor(1.0, dtype=pi.dtype), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.sum(pi, dim=1), torch.full((2,), 0.5, dtype=pi.dtype), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(torch.sum(pi, dim=0), torch.full((2,), 0.5, dtype=pi.dtype), atol=1e-4, rtol=1e-4)
    assert plan_cost == pytest.approx(1.0, abs=1e-3)


def test_torch_sinkhorn_pair_sampler_keeps_indices_on_device() -> None:
    torch.manual_seed(0)
    x0 = torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float32)
    x1 = torch.tensor([[9.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

    x0_ot, x1_ot, target_idx, plan_cost = gb._ot_pair_source_to_target_batch(
        x0,
        x1,
        ot_method="sinkhorn",
        ot_reg=1.0,
        ot_sinkhorn_iters=200,
    )

    assert x0_ot.device == x0.device
    assert x1_ot.device == x1.device
    assert target_idx.device == x1.device
    assert target_idx.shape == (2,)
    assert plan_cost == pytest.approx(1.0, abs=1e-3)


def test_ot_pair_source_to_target_batch_samples_plan_indices() -> None:
    np.random.seed(0)
    x0 = torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float32)
    x1 = torch.tensor([[9.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

    x0_ot, x1_ot, target_idx, plan_cost = gb._ot_pair_source_to_target_batch(x0, x1, ot_method="exact")

    assert x0_ot.shape == x0.shape
    assert x1_ot.shape == x1.shape
    assert target_idx.shape == (2,)
    for source, target in zip(x0_ot, x1_ot):
        assert torch.sum((source - target) ** 2).item() == pytest.approx(1.0)
    assert plan_cost == pytest.approx(1.0)


def test_training_loop_records_ot_source_pairing() -> None:
    base = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0))
    model = ConditionTimeAffineVelocity(theta_dim=1, x_dim=2, hidden_dim=4, depth=1)
    theta = np.asarray([[0.0], [0.0], [1.0], [1.0]], dtype=np.float64)
    x = np.asarray([[0.0, 0.0], [0.1, 0.0], [1.0, 0.2], [1.1, 0.3]], dtype=np.float64)

    meta = train_geometric_base_affine_flow(
        model=model,
        base=base,
        theta_train=theta,
        x_train=x,
        theta_val=theta,
        x_val=x,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=2,
        source_pairing="ot",
        log_every=999,
    )

    assert meta["source_pairing"] == "ot"
    assert meta["ot_method"] == "torch_sinkhorn"
    assert meta["ot_sinkhorn_iters"] == 100
    assert meta["checkpoint_selection"] == "best"
    assert meta["selected_epoch"] == meta["best_epoch"]
    assert meta["selected_val_loss"] == pytest.approx(meta["best_val_loss"])
    assert np.asarray(meta["train_pairing_costs"]).shape == (1,)
    assert np.asarray(meta["val_pairing_costs"]).shape == (1,)


def test_training_loop_can_keep_last_checkpoint() -> None:
    base = LineSegmentBase(anchor=(0.0, 0.0), direction=(1.0, 0.0))
    model = ConditionTimeAffineVelocity(theta_dim=1, x_dim=2, hidden_dim=4, depth=1)
    theta = np.asarray([[0.0], [0.0], [1.0], [1.0]], dtype=np.float64)
    x = np.asarray([[0.0, 0.0], [0.1, 0.0], [1.0, 0.2], [1.1, 0.3]], dtype=np.float64)

    meta = train_geometric_base_affine_flow(
        model=model,
        base=base,
        theta_train=theta,
        x_train=x,
        theta_val=theta,
        x_val=x,
        device=torch.device("cpu"),
        epochs=2,
        batch_size=2,
        checkpoint_selection="last",
        log_every=999,
    )

    assert meta["checkpoint_selection"] == "last"
    assert meta["stopped_early"] is False
    assert meta["stopped_epoch"] == 2
    assert meta["selected_epoch"] == 2
    assert meta["selected_val_loss"] == pytest.approx(np.asarray(meta["val_monitor_losses"])[-1])


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
    assert args.source_pairing == "random"
    assert args.ot_method == "torch_sinkhorn"
    assert args.ot_reg == pytest.approx(0.05)
    assert args.ot_sinkhorn_iters == 100
    assert args.checkpoint_selection == "best"
    assert args.smooth_sigma == pytest.approx(0.12)
    assert args.mc_skl_samples == 4096
    assert args.density_mc_samples == 1024


def test_geometric_base_line_fit_check_defaults() -> None:
    mod = _load_line_fit_check_module()
    args = mod.build_parser().parse_args([])
    paths = mod.resolve_output_paths(args.output_dir)

    assert args.theta_values == "0.7853981633974483,2.356194490192345"
    assert args.path_schedule == "cosine"
    assert args.source_pairing == "random"
    assert args.ot_method == "torch_sinkhorn"
    assert args.ot_reg == pytest.approx(0.05)
    assert args.ot_sinkhorn_iters == 100
    assert args.checkpoint_selection == "best"
    assert mod.build_parser().parse_args(["--ot-method", "pot_sinkhorn"]).ot_method == "pot_sinkhorn"
    assert args.epochs == 50000
    assert args.early_patience == 1000
    assert args.n_per_theta == 3000
    assert args.train_frac == pytest.approx(0.7)
    assert args.val_frac == pytest.approx(0.15)
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


def test_geometric_base_line_fit_ot_compare_defaults() -> None:
    mod = _load_ot_compare_module()
    args = mod.build_parser().parse_args([])
    paths = mod.resolve_output_paths(args.output_dir)

    assert args.path_schedule == "cosine"
    assert args.run_method == "both"
    assert args.ot_method == "torch_sinkhorn"
    assert args.ot_reg == pytest.approx(0.05)
    assert args.ot_sinkhorn_iters == 100
    assert args.checkpoint_selection == "best"
    assert mod.build_parser().parse_args(["--ot-method", "torch_sinkhorn"]).ot_method == "torch_sinkhorn"
    assert mod.build_parser().parse_args(["--ot-method", "pot_sinkhorn"]).ot_method == "pot_sinkhorn"
    assert args.epochs == 50000
    assert args.early_patience == 1000
    assert args.n_per_theta == 3000
    assert args.batch_size == 1024
    assert args.lr == pytest.approx(1e-3)
    assert args.max_test_plot_per_theta == 500
    assert paths["png"].name == "geometric_base_line_fit_ot_compare.png"
    assert paths["svg"].name == "geometric_base_line_fit_ot_compare.svg"
    assert paths["summary"].name == "geometric_base_line_fit_ot_compare_summary.json"

    assert mod._selected_methods("regular") == [("Regular CFM", "random")]
    assert mod._selected_methods("ot") == [("OT-CFM", "ot")]
    assert mod._selected_methods("both") == [("Regular CFM", "random"), ("OT-CFM", "ot")]
