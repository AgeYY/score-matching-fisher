"""One-off: run tests/ctsm.py main() and save matplotlib figures instead of plt.show()."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path(__file__).resolve().parents[4]
FIG_DIR = Path(__file__).resolve().parent
FIG_DIR.mkdir(parents=True, exist_ok=True)

_counter = 0


def _save_show() -> None:
    global _counter
    _counter += 1
    out = FIG_DIR / f"figure_{_counter}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("wrote", out, flush=True)


def main() -> None:
    plt.show = _save_show
    spec = importlib.util.spec_from_file_location("ctsm", _REPO / "tests" / "ctsm.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ctsm"] = mod
    spec.loader.exec_module(mod)
    mod.main()
    print("saved_count", _counter, flush=True)


if __name__ == "__main__":
    main()
