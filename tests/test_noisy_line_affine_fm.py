from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def _load_cli_module():
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "bin" / "run_noisy_line_affine_fm.py"
    spec = importlib.util.spec_from_file_location("run_noisy_line_affine_fm", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_noisy_line_batch_uses_matched_latent_and_noiseless_base() -> None:
    mod = _load_cli_module()
    rng = np.random.default_rng(123)
    batch = mod.generate_noisy_line_batch(
        num=64,
        theta=np.pi / 6.0,
        ell=1.5,
        sigma=0.12,
        shift=(0.25, -0.1),
        rng=rng,
    )

    assert batch.x0.shape == (64, 2)
    assert batch.x1.shape == (64, 2)
    np.testing.assert_allclose(batch.x0[:, 0], batch.u)
    np.testing.assert_allclose(batch.x0[:, 1], 0.0)

    q, n = mod.noisy_line_basis(np.pi / 6.0)
    expected_x1 = np.asarray([[0.25, -0.1]]) + 1.5 * batch.u[:, None] * q + 0.12 * batch.eta[:, None] * n
    np.testing.assert_allclose(batch.x1, expected_x1)


def test_linear_schedule_boundaries_and_derivatives() -> None:
    mod = _load_cli_module()
    schedule = mod.path_schedule_from_name("linear")
    t = torch.tensor([[0.0], [1.0]])
    alpha, beta, alpha_dot, beta_dot = schedule.ab_ad_bd(t)

    torch.testing.assert_close(alpha, torch.tensor([[1.0], [0.0]]))
    torch.testing.assert_close(beta, torch.tensor([[0.0], [1.0]]))
    torch.testing.assert_close(alpha_dot, torch.tensor([[-1.0], [-1.0]]))
    torch.testing.assert_close(beta_dot, torch.tensor([[1.0], [1.0]]))


def test_time_affine_velocity_is_affine_in_x() -> None:
    mod = _load_cli_module()
    torch.manual_seed(5)
    model = mod.TimeAffineVelocity(hidden_dim=8, depth=1)
    x_a = torch.tensor([[0.4, -0.2]], dtype=torch.float32)
    x_b = torch.tensor([[-0.1, 0.7]], dtype=torch.float32)
    x_mid = 0.5 * (x_a + x_b)
    t = torch.tensor([[0.35]], dtype=torch.float32)

    v_a = model(x_a, t)
    v_b = model(x_b, t)
    v_mid = model(x_mid, t)

    torch.testing.assert_close(v_mid, 0.5 * (v_a + v_b), atol=1e-6, rtol=1e-6)


def test_zero_velocity_ode_returns_base_points() -> None:
    mod = _load_cli_module()
    model = mod.TimeAffineVelocity(hidden_dim=8, depth=1)
    for param in model.parameters():
        torch.nn.init.zeros_(param)
    x0 = np.asarray([[0.0, 0.0], [0.2, 0.0], [-0.3, 0.0]], dtype=np.float64)

    out = mod.sample_flow_endpoint(model=model, x0=x0, device=torch.device("cpu"), ode_steps=8)

    np.testing.assert_allclose(out, x0, atol=1e-7)


def test_parser_standard_defaults() -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args([])

    assert args.device == "cuda:1"
    assert args.output_dir == Path("./data/") / "noisy_line_affine_fm_standard"
    assert args.theta == np.pi / 6.0
    assert args.ell == 1.5
    assert args.sigma == 0.12
    assert args.train_n == 16_384
    assert args.val_n == 4_096
    assert args.plot_n == 4_000
    assert args.steps == 5_000
    assert args.batch_size == 1_024
    assert args.ode_steps == 256
