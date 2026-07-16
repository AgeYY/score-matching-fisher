from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from fisher.bci_iv_2a_temporal_rdm_identification import (
    correct_match_margins,
    correct_match_ranks,
    exact_sign_flip_paired,
    native_evaluation_indices,
    shuffled_half_split_indices,
    temporal_rdm_score_matrix,
    vectorize_temporal_rdm,
)


ROOT = Path(__file__).resolve().parents[1]


def test_native_evaluation_grid_has_sixty_center_samples() -> None:
    times = -2.0 + np.arange(1500, dtype=np.float64) / 250.0
    indices = native_evaluation_indices(times, step_seconds=0.1)
    assert indices.shape == (60,)
    assert indices[0] == 12
    assert np.all(np.diff(indices) == 25)
    np.testing.assert_allclose(np.diff(times[indices]), 0.1, atol=1e-12)


def test_shuffled_half_split_is_deterministic_disjoint_and_exhaustive() -> None:
    query, reference = shuffled_half_split_indices(69, seed=23)
    repeated_query, repeated_reference = shuffled_half_split_indices(69, seed=23)
    np.testing.assert_array_equal(query, repeated_query)
    np.testing.assert_array_equal(reference, repeated_reference)
    assert query.size == 34
    assert reference.size == 35
    assert np.intersect1d(query, reference).size == 0
    np.testing.assert_array_equal(
        np.sort(np.concatenate([query, reference])),
        np.arange(69),
    )


def test_shuffled_half_split_changes_with_seed() -> None:
    first_query, _ = shuffled_half_split_indices(62, seed=1)
    second_query, _ = shuffled_half_split_indices(62, seed=2)
    assert first_query.size == second_query.size == 31
    assert not np.array_equal(first_query, second_query)


def test_vectorize_fid_uses_square_root_and_interval() -> None:
    times = np.array([-1.0, -0.5, 0.0, 0.5])
    rdm = np.array(
        [
            [0.0, 1.0, 4.0, 9.0],
            [1.0, 0.0, 16.0, 25.0],
            [4.0, 16.0, 0.0, 36.0],
            [9.0, 25.0, 36.0, 0.0],
        ]
    )
    got = vectorize_temporal_rdm(rdm, times, interval=(-1.0, 0.0), sqrt_values=True)
    np.testing.assert_allclose(got, [1.0])


def test_score_matrix_and_correct_ranks_identify_matching_patterns() -> None:
    times = np.arange(4, dtype=np.float64)
    base = np.array(
        [
            [0.0, 1.0, 2.0, 4.0],
            [1.0, 0.0, 3.0, 5.0],
            [2.0, 3.0, 0.0, 6.0],
            [4.0, 5.0, 6.0, 0.0],
        ]
    )
    alternative = base**2
    query = [{"euclidean": base}, {"euclidean": alternative}]
    reference = [{"euclidean": base + 0.01}, {"euclidean": alternative + 0.02}]
    for item in reference:
        np.fill_diagonal(item["euclidean"], 0.0)
    scores = temporal_rdm_score_matrix(query, reference, times, metric="euclidean")
    np.testing.assert_array_equal(correct_match_ranks(scores), [1, 1])
    assert np.all(correct_match_margins(scores) > 0.0)


def test_exact_sign_flip_paired_all_positive_has_expected_resolution() -> None:
    assert exact_sign_flip_paired(np.ones(5)) == 2.0 / 32.0


def test_cli_defaults_define_five_way_left_hand_pilot() -> None:
    path = ROOT / "bin/compare_bci_iv_2a_temporal_rdm_identification.py"
    spec = importlib.util.spec_from_file_location("temporal_rdm_identification_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args(["--device", "cuda:0"])
    assert args.recordings == ["A01T", "A02T", "A03T", "A04T", "A05T"]
    assert args.class_name == "left_hand"
    assert args.epochs == 20_000
    assert args.patience == 1_000
    assert args.batch_size == 16_384
    assert args.split_mode == "run_disjoint"
    assert args.metrics == ["correlation", "cosine", "euclidean", "fid"]


def test_cli_accepts_shuffled_cosine_w2_experiment() -> None:
    path = ROOT / "bin/compare_bci_iv_2a_temporal_rdm_identification.py"
    spec = importlib.util.spec_from_file_location("temporal_rdm_identification_split_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args(
        [
            "--device",
            "cuda:1",
            "--split-mode",
            "shuffled_half",
            "--metrics",
            "cosine",
            "fid",
        ]
    )
    assert args.split_mode == "shuffled_half"
    assert args.metrics == ["cosine", "fid"]


def test_cli_json_serializer_handles_numpy_training_metadata() -> None:
    path = ROOT / "bin/compare_bci_iv_2a_temporal_rdm_identification.py"
    spec = importlib.util.spec_from_file_location("temporal_rdm_identification_json", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    got = module._json_safe(
        {
            "losses": np.array([1.0, 0.5]),
            "epoch": np.int64(2),
            "nested": (np.float32(0.25),),
        }
    )
    assert got == {"losses": [1.0, 0.5], "epoch": 2, "nested": [0.25]}


def test_summary_reports_top2_accuracy_and_paired_difference() -> None:
    path = ROOT / "bin/compare_bci_iv_2a_temporal_rdm_identification.py"
    spec = importlib.util.spec_from_file_location("temporal_rdm_identification_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args(["--device", "cuda:0"])
    descending_rows = np.tile(np.arange(5.0, 0.0, -1.0), (5, 1))
    ascending_rows = np.tile(np.arange(1.0, 6.0), (5, 1))
    scores = {
        "full": {
            "classical": {"cosine": descending_rows},
            "flow": {"cosine": ascending_rows},
        }
    }
    summary = module._build_summary(
        args=args,
        evaluation_times=np.arange(5, dtype=np.float64),
        trial_counts={},
        split_metadata={},
        fit_summaries={},
        scores=scores,
        metrics=("cosine",),
    )
    assert summary["chance_top2"] == 0.4
    assert summary["intervals"]["full"]["classical"]["cosine"]["top2_accuracy"] == 0.4
    assert summary["intervals"]["full"]["flow"]["cosine"]["top2_accuracy"] == 0.4
    assert summary["paired_flow_minus_classical"]["full"]["cosine"]["top2_accuracy_difference"] == 0.0
