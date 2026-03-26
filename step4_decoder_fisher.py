#!/usr/bin/env python3
"""Compatibility wrapper for legacy Step 4 entrypoint.

Use `python run_fisher.py decoder ...` for the unified CLI.
"""

from __future__ import annotations

import sys

from run_fisher import main as unified_main


def main() -> None:
    print("[deprecated] step4_decoder_fisher.py -> run_fisher.py decoder")
    unified_main(["decoder", *sys.argv[1:]])


if __name__ == "__main__":
    main()
