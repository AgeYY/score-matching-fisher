from __future__ import annotations

import numpy as np
import pytest

from fisher.stringer_session_identification import DISTANCE_RMSE
from fisher.toy_fisher_identification import (
    METHOD_FLOW,
    evaluate_identification,
    fisher_mae,
    identification_matrix,
    identification_summary,
)


def test_fisher_mae_preserves_leading_axes() -> None:
    truth = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    estimates = np.stack([truth, truth + 2.0], axis=0)
    np.testing.assert_allclose(fisher_mae(estimates, truth), [[0.0, 0.0], [2.0, 2.0]])


def test_identification_matrix_and_summary_recover_matching_sessions() -> None:
    theta = np.asarray([0.0, 1.0, 2.0])
    half_a = np.asarray([[0.0, 0.5, 1.0], [3.0, 2.0, 1.0]])
    half_b = half_a + np.asarray([[0.01], [-0.01]])
    matrix = identification_matrix(half_a, half_b, theta, distance=DISTANCE_RMSE)
    summary = identification_summary(matrix)
    assert matrix.shape == (2, 2)
    assert summary["top1_accuracy"] == 1.0
    assert np.all(np.asarray(summary["correct_minus_best_wrong_margin"]) > 0.0)


def test_evaluate_identification_requires_two_halves() -> None:
    with pytest.raises(ValueError, match="shape"):
        evaluate_identification({METHOD_FLOW: np.zeros((3, 1, 4))}, np.arange(4))
