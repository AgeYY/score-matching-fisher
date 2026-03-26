#!/usr/bin/env python3
"""Compatibility wrapper for legacy Step 3 entrypoint.

Use `python run_fisher.py score ...` for the unified CLI.
"""

from __future__ import annotations

import sys

from run_fisher import main as unified_main


def main() -> None:
    print("[deprecated] step3_direct_fisher_score.py -> run_fisher.py score")
    unified_main(["score", *sys.argv[1:]])


if __name__ == "__main__":
    main()
