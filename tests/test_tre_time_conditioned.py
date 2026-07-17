from __future__ import annotations

import numpy as np
import torch

from fisher.tre_distance import TREDensityRatioConfig
from fisher.tre_time_conditioned import (
    _conditioned_waymarks,
    evaluate_time_conditioned_log_ratio,
    train_time_conditioned_tre_density_ratio,
)


def test_conditioned_waymarks_keep_time_fixed_across_bridges() -> None:
    x0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    times = torch.tensor([-0.5, 1.25])
    waymarks = _conditioned_waymarks(
        x0, x1, times, num_bridges=4, schedule="angle"
    )
    torch.testing.assert_close(waymarks[:, :, -1], times.expand(5, -1))
    torch.testing.assert_close(waymarks[0, :, :-1], x0)
    torch.testing.assert_close(waymarks[-1, :, :-1], x1)


def test_tiny_time_conditioned_tre_returns_finite_trial_time_ratios() -> None:
    rng = np.random.default_rng(19)
    times = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    x0 = rng.normal(-0.5, 0.3, size=(8, times.size, 2)).astype(np.float32)
    x1 = rng.normal(0.5, 0.3, size=(8, times.size, 2)).astype(np.float32)
    model, result = train_time_conditioned_tre_density_ratio(
        x0_train=x0[:6],
        x1_train=x1[:6],
        x0_validation=x0[6:],
        x1_validation=x1[6:],
        times=times,
        device=torch.device("cpu"),
        seed=23,
        config=TREDensityRatioConfig(
            num_bridges=2,
            architecture="linear",
            epochs=3,
            batch_size=8,
            early_patience=0,
            validation_pairs=8,
            log_every=10,
        ),
    )
    ratios = evaluate_time_conditioned_log_ratio(
        model, x0, times, device=torch.device("cpu"), batch_size=7
    )
    assert ratios.shape == (8, 5)
    assert np.isfinite(ratios).all()
    assert np.isfinite(result.validation_losses).all()
