#!/usr/bin/env python3
"""CLI wrapper: categorical H-decoding twofig with validation-only learned H / GT-LLR diagnostics.

Same as ``bin/study_h_decoding_categorical_twofig.py`` but defaults to
``--hellinger-eval-split validation`` (pass ``--hellinger-eval-split all`` to match the base script).
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from fisher.h_decoding_categorical_twofig import main


def _argv_with_default_validation(argv: list[str] | None) -> list[str]:
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)
    if not any(a == "--hellinger-eval-split" or a.startswith("--hellinger-eval-split=") for a in argv):
        argv = ["--hellinger-eval-split", "validation"] + argv
    return argv


if __name__ == "__main__":
    main(_argv_with_default_validation(None))
