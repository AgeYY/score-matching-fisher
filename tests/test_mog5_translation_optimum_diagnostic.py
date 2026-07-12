from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    path = Path(__file__).resolve().parent.parent / "bin" / "diagnose_mog5_translation_optima.py"
    spec = importlib.util.spec_from_file_location("diagnose_mog5_translation_optima", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_defaults_match_selected_diagnostic() -> None:
    module = _load_module()
    args = module.build_parser().parse_args([])

    assert args.n_list == [3000]
    assert args.n_repeats == 10
    assert args.seed == 19
    assert args.device == "cuda:0"
    assert args.batch_size == 3000
    assert args.nll_batch_size == 3000
    assert args.nll_epochs == 500


def test_relative_pair_error_averages_upper_triangle_ratios() -> None:
    module = _load_module()
    reference = np.asarray([[0.0, 2.0, 4.0], [2.0, 0.0, 8.0], [4.0, 8.0, 0.0]])
    estimate = np.asarray([[0.0, 3.0, 2.0], [3.0, 0.0, 12.0], [2.0, 12.0, 0.0]])

    assert module.relative_pair_error(estimate, reference) == pytest.approx(0.5)


def test_relative_pair_error_rejects_mismatched_shapes() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="matching shapes"):
        module.relative_pair_error(np.zeros((2, 2)), np.zeros((3, 3)))
