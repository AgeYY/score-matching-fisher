from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "bin"
        / "diagnose_stringer_gkr_covariance_likelihood.py"
    )
    spec = importlib.util.spec_from_file_location(
        "diagnose_stringer_gkr_covariance_likelihood", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_covariance_likelihood_matches_expanded_covariances() -> None:
    module = _load_script()
    rng = np.random.default_rng(7)
    x = rng.normal(size=(12, 3))
    means = rng.normal(size=(12, 3))
    groups = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    matrices = rng.normal(size=(3, 3, 3))
    covariance = np.einsum("nij,nkj->nik", matrices, matrices) + 0.5 * np.eye(3)
    expected = module._gaussian_log_likelihood(
        x,
        means,
        covariance[groups],
        jitter=1e-6,
    )
    observed = module._shared_covariance_log_likelihood(
        x,
        means,
        groups,
        covariance,
        jitter=1e-6,
    )
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_relative_values_use_paired_session_baseline() -> None:
    module = _load_script()
    values = np.asarray(
        [
            [-12.0, -10.0, -8.0],
            [-22.0, -17.0, -16.0],
        ]
    )
    observed = module._relative_to_baseline(values, baseline_index=1)
    np.testing.assert_allclose(
        observed,
        [[-2.0, 0.0, 2.0], [-5.0, 0.0, 1.0]],
    )
