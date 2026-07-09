#!/usr/bin/env python3
"""Compatibility wrapper for the unified geometric-base diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path


DEFAULT_DATASET = "two-square"
_UNIFIED_PATH = Path(__file__).resolve().with_name("run_geometric_base_fit_check.py")


def _load_unified_module():
    spec = importlib.util.spec_from_file_location("run_geometric_base_fit_check", _UNIFIED_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load unified runner from {_UNIFIED_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    mod = _load_unified_module()
    return int(mod.main(["--dataset", DEFAULT_DATASET, *(argv or [])]))


if __name__ == "__main__":
    raise SystemExit(main())
