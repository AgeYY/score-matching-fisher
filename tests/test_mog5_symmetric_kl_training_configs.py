from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    path = _REPO_ROOT / "bin" / "compare_mog5_symmetric_kl_training_configs.py"
    spec = importlib.util.spec_from_file_location("compare_mog5_symmetric_kl_training_configs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_defaults_use_two_repeats_on_cuda_zero() -> None:
    module = _load_module()
    args = module.build_parser().parse_args([])
    assert args.n_repeats == 2
    assert args.device == "cuda:0"
    assert args.batch_size == 3000
    assert args.lr == 1e-3


def test_configs_compare_previous_and_long_cosine_protocols() -> None:
    module = _load_module()
    configs = {spec.key: spec for spec in module.CONFIGS}
    previous = configs["previous"]
    assert (previous.hidden_dim, previous.depth) == (256, 5)
    assert previous.lr_schedule == "constant"
    assert previous.patience == 1_000
    assert previous.fixed_validation is False
    assert previous.checkpoint_selection == "best"
    new = configs["long_cosine"]
    assert (new.hidden_dim, new.depth) == (128, 3)
    assert new.epochs == 20_000
    assert new.lr_schedule == "cosine"
    assert new.min_lr == 1e-6
    assert new.patience == 0
    assert new.fixed_validation is True
    assert new.retain_best_state is True


def test_new_trajectory_reports_best_and_last() -> None:
    module = _load_module()
    assert module.VARIANT_ORDER == (
        "previous_best",
        "long_cosine_best",
        "long_cosine_last",
    )
