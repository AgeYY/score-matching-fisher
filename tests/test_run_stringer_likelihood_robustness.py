from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "bin"
        / "run_stringer_likelihood_robustness.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_stringer_likelihood_robustness", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_requested_robustness_configurations_change_one_factor() -> None:
    module = _load_script()
    observed = {
        config.key: (config.fit_fraction, config.pca_dim)
        for config in module.CONFIGS
    }
    assert observed == {
        "fit40_pca82": (0.4, 82),
        "fit60_pca82": (0.6, 82),
        "fit80_pca30": (0.8, 30),
        "fit80_pca130": (0.8, 130),
    }


def test_axis_limits_cover_sessions_without_forcing_zero() -> None:
    module = _load_script()
    values = np.asarray(
        [
            [-540.0, -530.0, -520.0, -500.0],
            [-550.0, -535.0, -525.0, -505.0],
            [-545.0, -532.0, -522.0, -502.0],
        ]
    )
    sem = np.asarray([2.0, 1.5, 1.0, 2.5])
    lower, upper = module._axis_limits(values, sem)
    assert lower < np.min(values)
    assert upper > np.max(values)
    assert upper < 0.0


def test_relative_likelihood_uses_paired_bin_lw_baseline() -> None:
    module = _load_script()
    values = np.asarray(
        [
            [-10.0, -8.0, -7.0, -5.0],
            [-20.0, -17.0, -16.0, -12.0],
        ]
    )
    relative = module._relative_to_bin_lw(values)
    np.testing.assert_allclose(
        relative,
        [
            [0.0, 2.0, 3.0, 5.0],
            [0.0, 3.0, 4.0, 8.0],
        ],
    )


def test_method_order_and_bar_style_match_requested_figure() -> None:
    module = _load_script()
    assert module.METHOD_KEYS == (
        "binned_test_log_likelihood",
        "gkr_test_log_likelihood",
        "affine_test_log_likelihood",
        "nonlinear_test_log_likelihood",
    )
    assert module.METHOD_LABELS[:2] == ("Bin+LW", "GKR")
    assert module.BAR_COLOR == "0.65"
