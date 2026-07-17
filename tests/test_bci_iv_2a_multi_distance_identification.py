from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "bin"
        / "compare_bci_iv_2a_multi_distance_session_identification.py"
    )
    spec = importlib.util.spec_from_file_location("bci_multi_distance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mean_metric_rdms_have_expected_geometry() -> None:
    module = _load_module()
    means = np.asarray(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        ]
    )
    cosine = module._metric_from_means(means, "cosine")
    euclidean = module._metric_from_means(means, "euclidean")
    np.testing.assert_allclose(cosine[0, 0, 1], 1.0)
    np.testing.assert_allclose(cosine[0, 0, 2], 2.0)
    np.testing.assert_allclose(euclidean[0, 0, 1], np.sqrt(2.0))
    np.testing.assert_allclose(euclidean[0, 0, 2], 2.0)


def test_rank_metrics_uses_smallest_mse() -> None:
    module = _load_module()
    recordings = ["A01T", "A02T", "A03T"]
    mse = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.2, 0.1],
            [0.3, 0.4, 0.1],
        ]
    )
    result = module._rank_metrics(mse, recordings)
    assert result["ranks"] == [1, 2, 1]
    assert result["top1_accuracy"] == 2.0 / 3.0
