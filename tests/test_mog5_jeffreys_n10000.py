from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


_ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = _ROOT / "bin" / "investigate_mog5_jeffreys_n10000.py"
    spec = importlib.util.spec_from_file_location("investigate_mog5_jeffreys_n10000", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_defaults_match_unified_fm_and_baseline_nll() -> None:
    module = _load()
    args = module.build_parser().parse_args([])
    assert args.device == "cuda:0"
    assert args.seed == 7
    assert args.n_total == 10_000
    assert args.n_repeats == 5
    assert args.fm_epochs == 20_000
    assert args.nll_configs == ["baseline"]
    fm = module._fm_config(args)
    assert fm.lr_schedule == "cosine"
    assert fm.min_lr == 1e-6
    assert fm.early_patience == 0
    assert fm.fixed_validation is True
    assert fm.checkpoint_selection == "best"


def test_nll_screen_varies_lr_batch_size_and_weight_decay() -> None:
    module = _load()
    configs = module._config_map()
    assert configs["baseline"].lr == 3e-5
    assert configs["wd1e2_lr1e5"].weight_decay == pytest.approx(1e-2)
    assert configs["steps64_lr1e5"].ode_steps == 64
    assert configs["responsive_lr1e5"].min_delta == 0.0
    assert configs["responsive_lr1e5"].ema_alpha == 1.0
    assert configs["lr1e5"].lr == 1e-5
    assert configs["lr3e6"].lr == 3e-6
    assert configs["fullbatch_lr1e5"].batch_size == 10_000
    assert configs["wd1e4_lr1e5"].weight_decay == 1e-4


def test_trend_artifacts_append_n10000_rows(tmp_path: Path) -> None:
    module = _load()
    prior_path = tmp_path / "prior.csv"
    pd.DataFrame(
        {
            "metric": ["symmetric_kl"] * 3,
            "n_total": [100] * 3,
            "repeat_idx": [0] * 3,
            "estimator": ["classical", "flow_matching", "flow_matching_nll_finetuned"],
            "abs_error": [3.0, 2.0, 1.0],
            "rel_error": [0.3, 0.2, 0.1],
        }
    ).to_csv(prior_path, index=False)
    current = pd.DataFrame(
        {
            "repeat_idx": [0, 0, 0],
            "variant": ["classical", "flow_matching", "nll_baseline"],
            "mae": [0.3, 0.2, 0.1],
            "mrae": [0.03, 0.02, 0.01],
        }
    )

    csv_path, png_path, svg_path = module._write_trend_artifacts(
        prior_errors_csv=prior_path,
        current_rows=current,
        output_dir=tmp_path,
    )

    trend = pd.read_csv(csv_path)
    assert set(trend["n_total"]) == {100, 10_000}
    assert len(trend) == 6
    assert png_path.is_file()
    assert svg_path.is_file()
