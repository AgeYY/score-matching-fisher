from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from fisher.gkr import GKRConfig


def _load_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "bin"
        / "refit_stringer_gkr_conventional_kernel.py"
    )
    spec = importlib.util.spec_from_file_location(
        "refit_stringer_gkr_conventional_kernel", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conventional_periodic_kernel_uses_exp_negative_distance() -> None:
    module = _load_script()
    estimator = module.ConventionalPeriodicKernelCovariance(
        n_input=1,
        n_output=1,
        circular_period=np.pi,
        jitter=0.0,
        dtype=torch.float64,
        device="cpu",
    )
    estimator.set_data(
        np.asarray([[1.0], [3.0]]),
        np.asarray([[0.0], [0.5 * np.pi]]),
    )
    with torch.no_grad():
        estimator.kernel_precision_cholesky.fill_(1.0)
    observed = float(estimator(np.asarray([[0.0]])).item())
    expected = (1.0 + np.exp(-1.0) * 9.0) / (1.0 + np.exp(-1.0))
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_conventional_kernel_is_locally_quadratic_not_quartic() -> None:
    module = _load_script()
    estimator = module.ConventionalPeriodicKernelCovariance(
        n_input=1,
        n_output=1,
        circular_period=np.pi,
        jitter=0.0,
        dtype=torch.float64,
        device="cpu",
    )
    estimator.set_data(
        np.asarray([[1.0], [2.0]]),
        np.asarray([[0.0], [0.1]]),
    )
    with torch.no_grad():
        estimator.kernel_precision_cholesky.fill_(2.0)
    precision = float(estimator.precision().item())
    expected_weight = np.exp(-precision * np.sin(0.1) ** 2)
    observed = float(estimator(np.asarray([[0.0]])).item())
    expected = (1.0 + expected_weight * 4.0) / (1.0 + expected_weight)
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_default_covariance_training_is_extended_to_300_epochs(
    monkeypatch,
) -> None:
    module = _load_script()
    monkeypatch.setattr(
        "sys.argv",
        ["refit_stringer_gkr_conventional_kernel.py", "--device", "cuda:0"],
    )
    args = module.parse_args()
    assert args.covariance_epochs == 300
    assert args.standardize_responses is True
    assert args.mean_learning_rate == 0.05
    assert args.covariance_learning_rate == 0.1


def test_training_ablation_flags_are_explicit(monkeypatch) -> None:
    module = _load_script()
    monkeypatch.setattr(
        "sys.argv",
        [
            "refit_stringer_gkr_conventional_kernel.py",
            "--device",
            "cuda:0",
            "--no-standardize-responses",
            "--mean-learning-rate",
            "0.01",
            "--covariance-learning-rate",
            "0.02",
        ],
    )
    args = module.parse_args()
    assert args.standardize_responses is False
    assert args.mean_learning_rate == 0.01
    assert args.covariance_learning_rate == 0.02


def test_checkpoint_payload_preserves_response_scaling() -> None:
    module = _load_script()
    model = SimpleNamespace(
        mean_model=torch.nn.Linear(1, 1),
        mean_likelihood=torch.nn.Linear(1, 1),
        covariance_model=torch.nn.Linear(1, 1),
        output_mean=torch.tensor([2.0, -3.0]),
        output_std=torch.tensor([4.0, 5.0]),
        mean_loss_history=[1.5],
        covariance_loss_history=[2.5],
        config=GKRConfig(),
        n_input=1,
        n_output=2,
        circular_period=np.pi,
        seed=7,
    )

    checkpoint = module.gkr_checkpoint(model)

    torch.testing.assert_close(checkpoint["output_mean"], model.output_mean)
    torch.testing.assert_close(checkpoint["output_std"], model.output_std)
