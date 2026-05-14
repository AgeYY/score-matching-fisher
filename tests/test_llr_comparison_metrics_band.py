"""Unit tests for band-restricted LLR metrics in ``debug_categorical_xflow_llr``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def dbg_cat():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "bin" / "debug_categorical_xflow_llr.py"
    spec = importlib.util.spec_from_file_location("_dbg_cat_xflow_llr_test", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_llr_band_metrics_match_manual_filter(dbg_cat):
    """Off-diagonal GT∈[-8,8] RMSE/r match filtering ``t_off`` the same way as the implementation."""
    true = np.array(
        [
            [0.0, 1.0, 50.0],
            [2.0, 0.0, 3.0],
            [4.0, 5.0, 0.0],
        ],
        dtype=np.float64,
    )
    est = np.array(
        [
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 3.0],
            [4.0, 100.0, 0.0],
        ],
        dtype=np.float64,
    )
    n = true.shape[0]
    mask = ~np.eye(n, dtype=bool)
    t_off = true[mask]
    e_off = est[mask]
    d_off = e_off - t_off
    lo, hi = dbg_cat.LLR_GT_METRIC_BAND_LO, dbg_cat.LLR_GT_METRIC_BAND_HI
    band_m = (t_off >= lo) & (t_off <= hi)
    e_b = e_off[band_m]
    t_b = t_off[band_m]
    d_b = e_b - t_b
    expect_rmse_band = float(np.sqrt(np.mean(d_b**2)))
    expect_r_band = dbg_cat._pearson_r(e_b, t_b)

    m = dbg_cat._llr_comparison_metrics(est, true)
    assert m["llr_rmse_offdiag"] == pytest.approx(float(np.sqrt(np.mean(d_off**2))))
    assert m["llr_pearson_r_offdiag"] == pytest.approx(dbg_cat._pearson_r(e_off, t_off))
    assert m["llr_rmse_offdiag_true_in_m8_p8"] == pytest.approx(expect_rmse_band)
    assert m["llr_pearson_r_offdiag_true_in_m8_p8"] == pytest.approx(expect_r_band)


def test_llr_band_all_nan_when_no_gt_in_band(dbg_cat):
    true = np.array([[0.0, 100.0], [100.0, 0.0]], dtype=np.float64)
    est = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    m = dbg_cat._llr_comparison_metrics(est, true)
    assert np.isnan(m["llr_rmse_offdiag_true_in_m8_p8"])
    assert np.isnan(m["llr_pearson_r_offdiag_true_in_m8_p8"])


def test_llr_band_n_le_1(dbg_cat):
    m = dbg_cat._llr_comparison_metrics(np.array([[0.0]]), np.array([[0.0]]))
    assert np.isnan(m["llr_rmse_offdiag_true_in_m8_p8"])
    assert np.isnan(m["llr_pearson_r_offdiag_true_in_m8_p8"])
