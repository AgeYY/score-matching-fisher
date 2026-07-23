from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "bin"
        / "summarize_stringer_gkr_training_ablations.py"
    )
    spec = importlib.util.spec_from_file_location(
        "summarize_stringer_gkr_training_ablations", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalized_mahalanobis_matches_identity_case() -> None:
    module = _load_script()
    x = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    mean = np.zeros_like(x)
    covariance = np.repeat(np.eye(2)[None], 2, axis=0)

    observed = module._normalized_mahalanobis(
        x, mean, covariance, jitter=0.0
    )

    assert observed == 0.5


def test_relative_likelihood_uses_paired_session_baseline() -> None:
    module = _load_script()
    values = np.asarray([[-10.0, -8.0], [-20.0, -15.0]])
    baseline = np.asarray([-9.0, -18.0])

    observed = module._relative_to_baseline(values, baseline)

    np.testing.assert_allclose(observed, [[-1.0, 1.0], [-2.0, 3.0]])
