from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "bin/compare_bci_iv_2a_euclidean_subsample_identification.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("bci_euclidean_subsample", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_n_labels_require_unique_finite_sizes_followed_by_all() -> None:
    module = _load_module()
    assert module._validate_n_labels(["4", "8", "all"]) == ("4", "8", "all")
    with pytest.raises(ValueError):
        module._validate_n_labels(["4", "8"])
    with pytest.raises(ValueError):
        module._validate_n_labels(["4", "4", "all"])


def test_literal_euclidean_sequence_takes_square_root_before_flattening() -> None:
    module = _load_module()
    squared = np.zeros((2, 4, 4), dtype=np.float64)
    upper = np.triu_indices(4, k=1)
    squared[:, upper[0], upper[1]] = np.asarray(
        [[1.0, 4.0, 9.0, 16.0, 25.0, 36.0], [4.0, 9.0, 16.0, 25.0, 36.0, 49.0]]
    )
    squared[:, upper[1], upper[0]] = squared[:, upper[0], upper[1]]
    got = module._literal_euclidean_sequence(
        squared,
        np.asarray([0.0, 0.004]),
        interval=(0.0, 0.004),
    )
    np.testing.assert_allclose(
        got,
        np.asarray([1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7], dtype=np.float64),
    )


def test_rank_score_matrix_uses_rows_as_queries() -> None:
    module = _load_module()
    scores = np.asarray(
        [
            [0.8, 0.1, 0.2],
            [0.7, 0.6, 0.2],
            [0.1, 0.3, 0.9],
        ]
    )
    ranks, predictions, margins = module._rank_score_matrix(scores)
    np.testing.assert_array_equal(ranks, [1, 2, 1])
    np.testing.assert_array_equal(predictions, [0, 0, 2])
    np.testing.assert_allclose(margins, [0.6, -0.1, 0.6])


def test_cache_roundtrip_is_atomic_and_finite(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "fit.npz"
    rdms = np.zeros((3, 4, 4), dtype=np.float64)
    module._save_cache(path, rdms, {"seed": 7})
    got, metadata = module._load_cache(path)
    np.testing.assert_array_equal(got, rdms)
    assert metadata == {"seed": 7}
    assert not path.with_suffix(".npz.tmp").exists()
