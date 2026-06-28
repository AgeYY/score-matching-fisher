"""Print a small example from the local Stringer dataset loader.

Run from the repo root:
    mamba run -n geo_diffusion python tests/inspect_stringer_dataset.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.stringer_dataset import (
    available_stringer_stimuli_types,
    list_stringer_sessions,
    load_stringer_session,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--session-stimuli-type", default="gratings_static")
    parser.add_argument("--session-index", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--n-neurons", type=int, default=5)
    args = parser.parse_args()

    counts = available_stringer_stimuli_types(data_dir=args.data_dir)
    sessions = list_stringer_sessions(args.session_stimuli_type, data_dir=args.data_dir)
    loaded = load_stringer_session(
        session_stimuli_type=args.session_stimuli_type,
        session_index=args.session_index,
        data_dir=args.data_dir,
    )

    n_trials = min(int(args.n_trials), loaded.neural_responses.shape[0])
    n_neurons = min(int(args.n_neurons), loaded.neural_responses.shape[1])

    print("Stringer dataset loader example")
    print(f"dataset_root: {loaded.meta['data_dir']}")
    print(f"available_session_stimuli_types: {counts}")
    print(f"selected_session_count_for_type: {len(sessions)}")
    print(f"session_file: {loaded.session_file}")
    print(f"session_stimuli_type: {loaded.session_stimuli_type}")
    print(f"neural_responses_shape: {loaded.neural_responses.shape}")
    print(f"grating_orientation_shape: {loaded.grating_orientation.shape}")
    print(f"grating_orientation_first_{n_trials}: {loaded.grating_orientation[:n_trials]}")
    print(
        f"neural_responses_first_{n_trials}x{n_neurons}: "
        f"{loaded.neural_responses[:n_trials, :n_neurons]}"
    )
    print("meta:")
    print(json.dumps({k: _jsonable(v) for k, v in loaded.meta.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
