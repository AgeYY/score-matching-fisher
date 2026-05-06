#!/usr/bin/env python3
"""CLI wrapper for :mod:`fisher.h_decoding_convergence`.

The implementation lives in ``fisher`` so it can be imported and tested as
package code. This wrapper preserves the historical script path and plain
``import study_h_decoding_convergence as conv`` usage from other ``bin`` tools.
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from fisher import h_decoding_convergence as _impl

globals().update({name: value for name, value in vars(_impl).items() if not name.startswith("__")})


if __name__ == "__main__":
    main()
