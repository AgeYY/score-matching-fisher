from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "bin"
        / "aggregate_stringer_pca_flow_gkr_sessions.py"
    )
    spec = importlib.util.spec_from_file_location(
        "aggregate_stringer_pca_flow_gkr_sessions", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_case_reads_summary_and_likelihoods(tmp_path: Path) -> None:
    module = _load_script()
    case = tmp_path / "session_00_GT1"
    case.mkdir()
    (case / "summary.json").write_text(
        json.dumps({"session_index": 0, "session_label": "GT1"}) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        case / "selected_theta_moments.npz",
        flow_test_log_likelihood=np.asarray([-1.0, -2.0]),
        gkr_test_log_likelihood=np.asarray([-2.0, -3.0]),
    )
    loaded = module._load_case(case)
    assert loaded["summary"]["session_label"] == "GT1"
    np.testing.assert_allclose(
        loaded["arrays"]["flow_test_log_likelihood"], [-1.0, -2.0]
    )
