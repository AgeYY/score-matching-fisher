from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


_ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = _ROOT / "bin" / "diagnose_mog5_jeffreys_minimal_scaling.py"
    spec = importlib.util.spec_from_file_location("diagnose_mog5_jeffreys_minimal_scaling", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_defaults_define_requested_minimal_experiment() -> None:
    module = _load()
    args = module.build_parser().parse_args([])
    assert args.device == "cuda:0"
    assert args.n_list == [3000, 10_000]
    assert args.n_repeats == 5
    assert args.epochs == 20_000
    assert args.early_patience == 1000
    assert args.fixed_validation_paths == 10
    config = module._flow_config(args)
    assert config.lr_schedule == "constant"
    assert config.early_patience == 1000
    assert config.checkpoint_selection == "best"
    assert config.best_checkpoint_metric == "flow_matching"
    assert config.fixed_validation_paths == 10
    assert config.likelihood_finetune_epochs == 0


def test_validation_nll_checkpoint_selection_flags_propagate() -> None:
    module = _load()
    args = module.build_parser().parse_args(
        [
            "--best-checkpoint-metric",
            "validation_nll",
            "--likelihood-validation-every",
            "25",
            "--likelihood-validation-ode-steps",
            "16",
        ]
    )
    config = module._flow_config(args)
    assert config.best_checkpoint_metric == "validation_nll"
    assert config.likelihood_validation_every == 25
    assert config.likelihood_validation_ode_steps == 16


def test_fixed_validation_path_count_propagates() -> None:
    module = _load()
    args = module.build_parser().parse_args(["--fixed-validation-paths", "10"])
    config = module._flow_config(args)
    assert config.fixed_validation_paths == 10


def test_summary_averages_repeats_by_n_and_estimator() -> None:
    module = _load()
    rows = pd.DataFrame(
        {
            "n_total": [3000, 3000, 3000, 3000],
            "repeat_idx": [0, 1, 0, 1],
            "estimator": ["classical", "classical", "flow_matching", "flow_matching"],
            "mae": [1.0, 3.0, 2.0, 4.0],
            "mrae": [0.1, 0.3, 0.2, 0.4],
            "runtime_seconds": [0.01, 0.02, 1.0, 2.0],
        }
    )
    summary = module._summarize(rows)
    classical = summary[summary["estimator"].eq("classical")].iloc[0]
    assert classical["mean_mae"] == 2.0
    assert classical["mean_mrae"] == 0.2
    assert classical["n_repeats"] == 2


def test_combined_diagnostic_plot_contains_error_and_loss_panels(tmp_path: Path) -> None:
    module = _load()
    rows = pd.DataFrame(
        {
            "n_total": [3000, 10_000] * 4,
            "repeat_idx": [0, 0, 1, 1] * 2,
            "estimator": ["classical"] * 4 + ["flow_matching"] * 4,
            "mae": [4.0, 3.0, 4.2, 3.2, 2.0, 1.5, 2.2, 1.7],
            "mrae": [0.4, 0.3, 0.42, 0.32, 0.2, 0.15, 0.22, 0.17],
            "runtime_seconds": [0.1] * 4 + [1.0] * 4,
        }
    )
    histories = {
        3000: [
            (np.asarray([2.0, 1.0]), np.asarray([2.1, 1.1])),
            (np.asarray([2.0, 1.1, 0.9]), np.asarray([2.1, 1.2, 1.0])),
        ],
        10_000: [
            (np.asarray([2.0, 0.9]), np.asarray([2.1, 1.0])),
            (np.asarray([2.0]), np.asarray([2.1])),
        ],
    }

    png, svg = module._plot_combined_diagnostics(
        rows,
        module._summarize(rows),
        histories,
        tmp_path,
    )

    assert png.is_file()
    assert svg.is_file()
