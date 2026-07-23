from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "bin"
        / "aggregate_stringer_pca_four_methods.py"
    )
    spec = importlib.util.spec_from_file_location(
        "aggregate_stringer_pca_four_methods", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_case_and_session_means(tmp_path: Path) -> None:
    module = _load_script()
    case = tmp_path / "session_00_GT1"
    case.mkdir()
    (case / "summary.json").write_text(
        json.dumps({"session_index": 0, "session_label": "GT1"}) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        case / "four_method_results.npz",
        affine_test_log_likelihood=np.asarray([-1.0, -3.0]),
        gkr_test_log_likelihood=np.asarray([-2.0, -4.0]),
        binned_test_log_likelihood=np.asarray([-3.0, -5.0]),
        nonlinear_test_log_likelihood=np.asarray([-4.0, -6.0]),
    )
    loaded = module._load_case(case)
    np.testing.assert_allclose(
        module._session_mean_likelihoods(loaded), [-2.0, -3.0, -4.0, -5.0]
    )
