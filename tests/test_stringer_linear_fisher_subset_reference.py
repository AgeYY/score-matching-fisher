from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_script():
    path = Path(__file__).resolve().parents[1] / "bin" / "compare_stringer_linear_fisher_subset_reference.py"
    spec = importlib.util.spec_from_file_location("compare_stringer_linear_fisher_subset_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_linear_fisher_subsets_are_nested() -> None:
    module = _load_script()
    theta = (np.arange(160, dtype=np.float64) + 0.5) * np.pi / 160.0
    subsets = module._nested_subsets(theta, [32, 80, 144], n_strata=16, seed=37)
    assert [subset.size for subset in subsets] == [32, 80, 144]
    assert set(subsets[0]).issubset(set(subsets[1]))
    assert set(subsets[1]).issubset(set(subsets[2]))


def test_linear_fisher_rows_keep_condition_curves() -> None:
    module = _load_script()
    fit = {
        "flow_fisher": np.array([3.0, 5.0]),
        "gkr_fisher": np.array([2.0, 4.0]),
        "ole_fisher": np.array([1.0, 3.0]),
        "metadata": {"flow_selected_epoch": 12},
    }
    rows = module._rows(fit, reference=np.array([2.0, 4.0]), n=100)
    assert [row["method"] for row in rows] == list(module.METHODS)
    assert rows[0]["mean_fisher"] == 4.0
    assert rows[1]["curve_normalized_rmse_to_full_ole"] == 0.0
    assert rows[0]["flow_selected_epoch"] == 12


def test_linear_fisher_defaults_to_centered_pca_and_256_step_readout(monkeypatch) -> None:
    module = _load_script()
    monkeypatch.setattr("sys.argv", ["compare_stringer_linear_fisher_subset_reference.py", "--device", "cuda:0"])
    args = module.parse_args()
    assert args.pca_whiten is False
    assert args.ode_steps == 256


def test_training_signature_ignores_only_readout_configuration() -> None:
    module = _load_script()
    cached = {"n": 500, "lr": 1e-4, "ode_steps": 64}
    current = {
        "n": 500,
        "lr": 1e-4,
        "ode_steps": 256,
        "covariance_integrator": "midpoint_matrix_exponential",
    }
    assert module._same_training_signature(cached, current)
    assert not module._same_training_signature(cached, current | {"lr": 2e-4})
