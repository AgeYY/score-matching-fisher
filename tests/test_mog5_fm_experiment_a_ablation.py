from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    path = _REPO_ROOT / "bin" / "ablate_mog5_fm_experiment_a.py"
    spec = importlib.util.spec_from_file_location("ablate_mog5_fm_experiment_a", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ablation_defaults_use_requested_small_mlp_and_two_repeats() -> None:
    module = _load_module()
    args = module.build_parser().parse_args([])
    assert args.hidden_dim == 128
    assert args.depth == 3
    assert args.n_repeats == 2
    assert args.device == "cuda:0"


def test_ablation_trajectories_isolate_experiment_a_factors() -> None:
    module = _load_module()
    specs = {spec.key: spec for spec in module.TRAJECTORIES}
    full = specs["experiment_a"]
    assert full.epochs == 20_000
    assert full.lr_schedule == "cosine"
    assert full.checkpoint_selection == "last"
    assert full.fixed_validation is True
    assert full.patience == 0
    assert full.retain_best_state is True
    short = specs["short_2k"]
    assert short.epochs == 2_000
    assert short.lr_schedule_epochs == full.lr_schedule_epochs
    assert specs["constant_lr"].lr_schedule == "constant"
    assert specs["resampled_best"].fixed_validation is False
    assert specs["early_stop_fixed"].patience == 1_000


def test_effect_rows_use_declared_pairwise_references() -> None:
    module = _load_module()
    rows = []
    values = {
        "experiment_a_last": 1.0,
        "experiment_a_best": 1.1,
        "short_2k": 1.2,
        "constant_lr": 1.3,
        "resampled_best": 1.4,
        "early_stop_fixed": 1.5,
        "previous_like": 1.6,
    }
    for metric in module.METRICS:
        for variant, value in values.items():
            rows.append(
                {
                    "repeat_idx": 0,
                    "metric": metric,
                    "variant": variant,
                    "relative_error_to_ground_truth": value,
                    "relative_gap_to_train_optimum": 2.0 * value,
                }
            )
    effects = module._effect_rows(pd.DataFrame(rows)).set_index(["effect", "metric"])
    metric = module.METRICS[0]
    assert effects.loc[("shorter_training", metric), "delta_error_to_ground_truth"] == pytest.approx(0.2)
    assert effects.loc[("best_instead_of_last", metric), "delta_error_to_ground_truth"] == pytest.approx(0.1)
    assert effects.loc[("resampled_instead_of_fixed_validation", metric), "delta_error_to_ground_truth"] == pytest.approx(0.3)
