#!/usr/bin/env python3
"""CLI wrapper for :mod:`fisher.h_binned_visualization`.

This file intentionally re-exports the package module's helpers because other
scripts historically import ``visualize_h_matrix_binned`` from the ``bin`` path.
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from fisher.h_binned_visualization import *  # noqa: F401,F403
from fisher.h_binned_visualization import main


if __name__ == "__main__":
    main()
