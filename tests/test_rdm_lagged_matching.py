from __future__ import annotations

import numpy as np
import pytest

from fisher.rdm_lagged_matching import (
    lagged_pearson_similarity,
    rdm_upper_triangle_sequence,
)


def test_lagged_correlation_recovers_known_positive_shift() -> None:
    rng = np.random.default_rng(12)
    reference = rng.normal(size=(80, 6))
    query = rng.normal(size=(80, 6))
    query[5:-5] = reference[8:-2]
    result = lagged_pearson_similarity(query, reference, max_lag_samples=5)
    assert result.lag_samples == 3
    assert result.score == pytest.approx(1.0)
    assert result.n_core_time_points == 70
    assert result.zero_lag_same_core_score < 0.2


def test_lagged_correlation_uses_zero_lag_for_identical_sequences() -> None:
    rng = np.random.default_rng(4)
    sequence = rng.normal(size=(20, 6))
    result = lagged_pearson_similarity(sequence, sequence, max_lag_samples=4)
    assert result.lag_samples == 0
    assert result.score == pytest.approx(1.0)
    assert result.zero_lag_same_core_score == pytest.approx(1.0)


def test_lag_limit_must_leave_two_common_time_points() -> None:
    sequence = np.ones((10, 6))
    with pytest.raises(ValueError, match="fewer than two"):
        lagged_pearson_similarity(sequence, sequence, max_lag_samples=5)


def test_rdm_sequence_extracts_six_entries_at_selected_times() -> None:
    rdms = np.zeros((3, 4, 4), dtype=np.float64)
    upper = np.triu_indices(4, k=1)
    rdms[:, upper[0], upper[1]] = np.arange(18).reshape(3, 6)
    rdms += rdms.transpose(0, 2, 1)
    sequence, times = rdm_upper_triangle_sequence(
        rdms, np.array([0.0, 0.1, 0.2]), interval=(0.1, 0.2)
    )
    np.testing.assert_array_equal(sequence, np.arange(6, 18).reshape(2, 6))
    np.testing.assert_allclose(times, [0.1, 0.2])
