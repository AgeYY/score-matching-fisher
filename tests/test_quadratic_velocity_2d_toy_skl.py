from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_cli_module():
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "bin" / "run_quadratic_velocity_2d_toy_skl.py"
    spec = importlib.util.spec_from_file_location("run_quadratic_velocity_2d_toy_skl", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_low_rank_x1_models_parse_and_map_to_fixed_x1_basis() -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(["--models", "low_rank_x1,shared_low_rank_x1", "--x-dim", "2"])
    specs = mod._model_specs(args.models)

    assert [spec.name for spec in specs] == ["low_rank_x1", "shared_low_rank_x1"]
    assert [spec.velocity_family for spec in specs] == [
        "condition_fixed_input_low_rank",
        "shared_affine_low_rank",
    ]
    assert [spec.low_rank_axis for spec in specs] == [0, 0]
    np.testing.assert_allclose(mod._fixed_axis_basis(x_dim=2, axis=0), np.asarray([[1.0], [0.0]]))


def test_low_rank_x2_models_parse_and_map_to_fixed_x2_basis() -> None:
    mod = _load_cli_module()
    args = mod.build_parser().parse_args(["--models", "low_rank_x2,shared_low_rank_x2", "--x-dim", "2"])
    specs = mod._model_specs(args.models)

    assert [spec.name for spec in specs] == ["low_rank_x2", "shared_low_rank_x2"]
    assert [spec.velocity_family for spec in specs] == [
        "condition_fixed_input_low_rank",
        "shared_affine_low_rank",
    ]
    assert [spec.low_rank_axis for spec in specs] == [1, 1]
    np.testing.assert_allclose(mod._fixed_axis_basis(x_dim=2, axis=1), np.asarray([[0.0], [1.0]]))


def test_low_rank_fixed_axis_models_require_two_dimensional_toy_and_known_axis() -> None:
    mod = _load_cli_module()

    with pytest.raises(ValueError, match="--x-dim 2"):
        mod._fixed_axis_basis(x_dim=4, axis=1)
    with pytest.raises(ValueError, match="axis in"):
        mod._fixed_axis_basis(x_dim=2, axis=2)
