from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fisher.continuous_fisher_comparison import METHOD_CLASSICAL_LINEAR, METHOD_FLOW_LINEAR
from fisher.stringer_gkr_comparison import (
    COMPARISON_METHODS,
    METHOD_GKR_LINEAR,
    StringerGKRComparisonResult,
    load_baseline_arrays,
    plot_topk_comparison,
    summarize_identification_matrix,
)
from fisher.stringer_session_identification import DISTANCES, split_train_validation


def test_summarize_identification_matrix_ranks_diagonal_matches() -> None:
    summary = summarize_identification_matrix(
        np.asarray(
            [
                [0.1, 0.2, 0.3],
                [0.1, 0.3, 0.2],
                [0.3, 0.2, 0.1],
            ]
        )
    )

    assert summary["ranks"] == [1, 3, 1]
    assert summary["top1_accuracy"] == 2.0 / 3.0
    assert summary["top3_accuracy"] == 1.0


def test_gkr_uses_the_same_deterministic_training_split_as_flow() -> None:
    flow_train, flow_validation = split_train_validation(2000, train_frac=0.8, seed=17)
    gkr_train, gkr_validation = split_train_validation(2000, train_frac=0.8, seed=17)

    np.testing.assert_array_equal(gkr_train, flow_train)
    np.testing.assert_array_equal(gkr_validation, flow_validation)
    assert gkr_train.size == 1600
    assert gkr_validation.size == 400


def test_load_baseline_arrays_slices_n_repeats_and_sessions(tmp_path: Path) -> None:
    n_values = [200, 650]
    session_keys = ["s0", "s1", "s2"]
    summary = {
        "n_values": n_values,
        "repeats": 3,
        "session_keys": session_keys,
    }
    (tmp_path / "stringer_session_identification_a_subsample_convergence_summary.json").write_text(
        json.dumps(summary)
    )
    arrays = {
        "theta_grid": np.linspace(0.0, np.pi, 4)[:, None],
        "theta_midpoints": np.linspace(0.1, 3.0, 3)[:, None],
    }
    for method in (METHOD_CLASSICAL_LINEAR, METHOD_FLOW_LINEAR):
        for distance in DISTANCES:
            arrays[f"endpoint_{method}_{distance}_A_to_B"] = np.ones((3, 3))
            arrays[f"subset_{method}_{distance}_A_to_B"] = np.ones((2, 3, 3, 3))
    np.savez_compressed(
        tmp_path / "stringer_session_identification_a_subsample_convergence_results.npz",
        **arrays,
    )

    _summary, loaded = load_baseline_arrays(
        tmp_path,
        n_values=[650],
        repeats=2,
        max_sessions=2,
    )

    assert loaded["n_values"] == [650]
    assert loaded["repeats"] == 2
    assert loaded["session_keys"] == ["s0", "s1"]
    assert loaded["subset"][METHOD_FLOW_LINEAR][DISTANCES[0]].shape == (1, 2, 2, 2)


def test_plot_topk_comparison_writes_vector_and_raster(tmp_path: Path) -> None:
    convergence = {}
    endpoint = {}
    subset = {}
    for method in COMPARISON_METHODS:
        convergence[method] = {}
        endpoint[method] = {}
        subset[method] = {}
        for distance in DISTANCES:
            convergence[method][distance] = {
                "top1_by_repeat": [[0.5, 1.0], [1.0, 1.0]],
                "top3_by_repeat": [[1.0, 1.0], [1.0, 1.0]],
            }
            endpoint[method][distance] = np.eye(2)
            subset[method][distance] = np.zeros((2, 2, 2, 2))
    result = StringerGKRComparisonResult(
        session_keys=["s0", "s1"],
        theta_grid=np.linspace(0.0, np.pi, 4)[:, None],
        theta_midpoints=np.linspace(0.1, 3.0, 3)[:, None],
        n_values=[200, 650],
        repeats=2,
        endpoint_matrices=endpoint,
        subset_matrices=subset,
        summary={"subsample_convergence": convergence},
    )

    svg, png = plot_topk_comparison(tmp_path / "plot.svg", tmp_path / "plot.png", result)

    assert svg.is_file()
    assert png.is_file()
    assert METHOD_GKR_LINEAR in COMPARISON_METHODS
