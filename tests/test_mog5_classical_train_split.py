from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parent.parent / "bin" / "compare_mog5_classical_train_split.py"
    spec = importlib.util.spec_from_file_location("compare_mog5_classical_train_split", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_rows_averages_pairs_then_repeats() -> None:
    module = _load_module()
    rows = pd.DataFrame(
        [
            {"metric": "cosine", "estimator": "classical_train", "n_total": 50, "n_train": 40, "repeat_idx": 0, "abs_error": 1.0, "rel_error": 0.5},
            {"metric": "cosine", "estimator": "classical_train", "n_total": 50, "n_train": 40, "repeat_idx": 0, "abs_error": 3.0, "rel_error": 1.5},
            {"metric": "cosine", "estimator": "classical_train", "n_total": 50, "n_train": 40, "repeat_idx": 1, "abs_error": 5.0, "rel_error": 2.0},
            {"metric": "cosine", "estimator": "classical_train", "n_total": 50, "n_train": 40, "repeat_idx": 1, "abs_error": 7.0, "rel_error": 4.0},
        ]
    )

    summary = module.summarize_rows(rows)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["mae_mean"] == 4.0
    assert row["mrae_mean"] == 2.0
    assert row["n_repeats"] == 2


def test_case_rows_uses_training_split_for_added_classical_estimator(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    result_dir = tmp_path / "dataset" / "comparison"
    result_dir.mkdir(parents=True)
    result_path = result_dir / "mog5_pr_distance_comparison_results.npz"
    metric_names = np.asarray(["squared_euclidean"])
    pair_indices = np.asarray([[0, 1]], dtype=np.int64)
    matrix = np.asarray([[[0.0, 1.0], [1.0, 0.0]]], dtype=np.float64)
    np.savez(
        result_path,
        metric_names=metric_names,
        condition_labels=np.asarray(["a", "b"]),
        pair_indices=pair_indices,
        classical_matrices=matrix,
        flow_matching_matrices=matrix,
        flow_matching_nll_finetuned_matrices=matrix,
        ground_truth_matrices=matrix,
    )
    bundle = type(
        "Bundle",
        (),
        {
            "meta": {"num_categories": 2},
            "theta_train": np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float64),
            "x_train": np.asarray([[0.0], [2.0], [4.0], [6.0]], dtype=np.float64),
        },
    )()
    monkeypatch.setattr(module, "load_shared_dataset_npz", lambda path: bundle)

    rows = module._case_rows(
        result_path=result_path,
        n_total=5,
        repeat_idx=0,
        repeat_seed=7,
        mahalanobis_ridge=1e-6,
    )

    train_row = next(row for row in rows if row["estimator"] == "classical_train")
    assert train_row["estimate"] == 16.0
    assert train_row["n_train"] == 4
