"""CLI validation for segmented theta-flow in ``bin/study_h_decoding_convergence.py``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def test_validate_rejects_segmented_without_theta_flow() -> None:
    repo = Path(__file__).resolve().parent.parent
    path = repo / "bin" / "study_h_decoding_convergence.py"
    spec = importlib.util.spec_from_file_location("study_h_decoding_convergence_segcli", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = mod.build_parser()
    args = p.parse_args(
        [
            "--dataset-npz",
            str(repo / "pyproject.toml"),
            "--theta-flow-segmented",
            "--theta-field-method",
            "x_flow",
            "--flow-arch",
            "mlp",
        ]
    )
    with pytest.raises(ValueError, match="theta.flow"):
        mod._validate_cli(args)


def test_theta_segment_ids_study_delegates_to_viz() -> None:
    repo = Path(__file__).resolve().parent.parent
    path = repo / "bin" / "study_h_decoding_convergence.py"
    spec = importlib.util.spec_from_file_location("study_h_decoding_convergence_deleg", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    th = np.linspace(-1.0, 1.0, 8, dtype=np.float64)
    a, e1 = mod.theta_segment_ids_equal_width(th, 2)
    b, e2 = mod.vhb.theta_segment_ids_equal_width(th, 2)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(e1, e2)


def test_validate_rejects_onehot_without_theta_flow() -> None:
    repo = Path(__file__).resolve().parent.parent
    path = repo / "bin" / "study_h_decoding_convergence.py"
    spec = importlib.util.spec_from_file_location("study_h_decoding_convergence_onehot_method", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = mod.build_parser()
    args = p.parse_args(
        [
            "--dataset-npz",
            str(repo / "pyproject.toml"),
            "--theta-flow-onehot-state",
            "--theta-field-method",
            "x_flow",
            "--flow-arch",
            "mlp",
        ]
    )
    with pytest.raises(ValueError, match="theta.flow"):
        mod._validate_cli(args)


def test_validate_rejects_onehot_non_mlp_arch() -> None:
    repo = Path(__file__).resolve().parent.parent
    path = repo / "bin" / "study_h_decoding_convergence.py"
    spec = importlib.util.spec_from_file_location("study_h_decoding_convergence_onehot_arch", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = mod.build_parser()
    args = p.parse_args(
        [
            "--dataset-npz",
            str(repo / "pyproject.toml"),
            "--theta-flow-onehot-state",
            "--theta-field-method",
            "theta_flow",
            "--flow-arch",
            "iid_soft",
        ]
    )
    with pytest.raises(ValueError, match="flow.arch mlp"):
        mod._validate_cli(args)
