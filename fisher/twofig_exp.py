"""Small compatibility helpers for twofig experiments."""

from __future__ import annotations

import numpy as np

from fisher.h_binned_visualization import pairwise_bin_logistic_accuracy_train_val


def _pairwise_decode_accuracy_and_hellinger_train_val(
    x_train: np.ndarray,
    bin_train: np.ndarray,
    x_eval: np.ndarray,
    bin_eval: np.ndarray,
    n_bins: int,
    *,
    min_class_count: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pairwise decoding accuracy and a bounded Hellinger-like matrix.

    This compatibility shim is intentionally lightweight: it reuses the shared
    pairwise logistic decoder and maps accuracy above chance to ``[0, 1]``.
    Missing/unsupported pairs remain NaN, matching the decoder matrix.
    """
    acc, _, _, _ = pairwise_bin_logistic_accuracy_train_val(
        x_train,
        bin_train,
        x_eval,
        bin_eval,
        int(n_bins),
        min_class_count=int(min_class_count),
        random_state=int(random_state),
    )
    h = np.asarray(2.0 * np.asarray(acc, dtype=np.float64) - 1.0, dtype=np.float64)
    h = np.clip(h, 0.0, 1.0)
    np.fill_diagonal(h, np.nan)
    return np.asarray(acc, dtype=np.float64), h

