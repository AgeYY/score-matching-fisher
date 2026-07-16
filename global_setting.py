"""Project-wide paths: default data root for datasets and run outputs."""

from __future__ import annotations

import os

# All default paths in bin/ and fisher.cli_shared_fisher resolve under this directory.
# Override at runtime: export SCORE_MATCHING_FISHER_DATAROOT=/path/to/data
DATAROOT = os.environ.get(
    "SCORE_MATCHING_FISHER_DATAROOT",
    # "/data/zeyuan/score-matching-fisher",
    "./data/",
)

# Default datasets and run outputs live directly under DATAROOT (same tree as the repo `data/` symlink).
DATA_DIR = DATAROOT

# Local Stringer et al. 2019 mouse visual cortex dataset.
# Override at runtime:
# export SCORE_MATCHING_FISHER_STRINGER_DATA_DIR=/path/to/stringer-et-al-2019
STRINGER_DATA_DIR = os.environ.get(
    "SCORE_MATCHING_FISHER_STRINGER_DATA_DIR",
    "/storage/zeyuan/stringer-et-al-2019",
)

# Canonical example recording session for quick Stringer experiments.
# This is one full-field static grating session from database.npy.
STRINGER_EXAMPLE_SESSION_FILE = os.environ.get(
    "SCORE_MATCHING_FISHER_STRINGER_EXAMPLE_SESSION_FILE",
    "gratings_static_GT1_2019_04_17_1.npy",
)

# Default execution device for training/evaluation scripts.
# Override at runtime: export SCORE_MATCHING_FISHER_DEFAULT_DEVICE=cuda:1
DEFAULT_DEVICE = os.environ.get("SCORE_MATCHING_FISHER_DEFAULT_DEVICE", "cuda:0")

# Project-wide training defaults. Individual commands may still override these
# values explicitly for smoke tests or deliberately shorter/longer studies.
TRAINING_MAX_EPOCHS = 20_000
TRAINING_EARLY_STOPPING_PATIENCE = 1_000


def _default_cuda_device_id(device_name: str) -> int:
    text = str(device_name).strip().lower()
    if text == "cuda":
        return 0
    if text.startswith("cuda:"):
        suffix = text.split(":", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return 0


# Default physical CUDA ids for subprocess launchers that use CUDA_VISIBLE_DEVICES.
DEFAULT_CUDA_DEVICE_ID = _default_cuda_device_id(DEFAULT_DEVICE)
DEFAULT_CUDA_DEVICE_IDS = [DEFAULT_CUDA_DEVICE_ID]

# Local Hugging Face builder cache location for the EcoSet validation Arrow table.
# Override at runtime:
# export SCORE_MATCHING_FISHER_ECOSET_VALIDATION_DIR=/path/to/ecoset-validation.arrow
ECOSET_VALIDATION_DIR = os.environ.get(
    "SCORE_MATCHING_FISHER_ECOSET_VALIDATION_DIR",
    os.path.join(DATA_DIR, "ecoset", "hf_cache", "kietzmannlab___ecoset", "Full"),
)

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
