"""Project-wide paths: default data root for datasets and run outputs."""

from __future__ import annotations

import os

# All default paths in bin/ and fisher.cli_shared_fisher resolve under this directory.
# Override at runtime: export SCORE_MATCHING_FISHER_DATAROOT=/path/to/data
DATAROOT = os.environ.get(
    "SCORE_MATCHING_FISHER_DATAROOT",
    "/data/zeyuan/score-matching-fisher",
)

# Default datasets and run outputs live directly under DATAROOT (same tree as the repo `data/` symlink).
DATA_DIR = DATAROOT

# Score matching: when using train_split validation, this fraction of the score pool is held out for val.
SCORE_VAL_FRACTION = 0.2


def apply_matplotlib_defaults() -> None:
    """Set matplotlib rcParams for bin/ scripts: larger tick labels, thicker spines, no top/right spines.

    Idempotent: safe to call multiple times. Importing this module runs it once.
    """
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "xtick.major.width": 2.0,
            "ytick.major.width": 2.0,
            "axes.labelsize": 20,
            "axes.titlesize": 18,
            "axes.linewidth": 3.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# Run once on import so ``from global_setting import DATA_DIR`` activates styling for bin/ tools.
apply_matplotlib_defaults()
