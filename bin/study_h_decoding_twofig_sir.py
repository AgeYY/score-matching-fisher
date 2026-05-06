#!/usr/bin/env python3
"""CLI wrapper for :mod:`fisher.h_decoding_twofig_sir`."""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from fisher.h_decoding_twofig_sir import *  # noqa: F401,F403
from fisher.h_decoding_twofig_sir import main


if __name__ == "__main__":
    main()
