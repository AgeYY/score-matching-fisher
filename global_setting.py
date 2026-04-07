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
