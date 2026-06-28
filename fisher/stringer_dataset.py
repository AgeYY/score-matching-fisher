"""Small loader for the local Stringer et al. 2019 visual cortex sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from global_setting import STRINGER_DATA_DIR, STRINGER_EXAMPLE_SESSION_FILE


@dataclass(frozen=True)
class StringerSessionInfo:
    """Catalog entry for one Stringer session file."""

    session_file: Path
    session_stimuli_type: str
    mouse_name: str
    date: str
    block: str
    depth: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class StringerSession:
    """Loaded Stringer session arrays in trial-major form."""

    neural_responses: np.ndarray
    grating_orientation: np.ndarray
    session_stimuli_type: str
    session_file: Path
    meta: dict[str, Any]


def _stringer_root(data_dir: str | Path | None = None) -> Path:
    return Path(STRINGER_DATA_DIR if data_dir is None else data_dir).expanduser()


def _session_filename(entry: dict[str, Any]) -> str:
    return f"{entry['expt']}_{entry['mouse_name']}_{entry['date']}_{entry['block']}.npy"


def load_stringer_database(data_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Load ``database.npy`` as a plain list of session metadata dicts."""

    root = _stringer_root(data_dir)
    db_path = root / "database.npy"
    if not db_path.exists():
        raise FileNotFoundError(
            f"Stringer database not found: {db_path}. "
            "Set SCORE_MATCHING_FISHER_STRINGER_DATA_DIR to the dataset directory."
        )
    return [dict(entry) for entry in np.load(db_path, allow_pickle=True)]


def list_stringer_sessions(
    session_stimuli_type: str | None = None,
    *,
    data_dir: str | Path | None = None,
) -> list[StringerSessionInfo]:
    """Return available sessions, optionally filtered by stimulus/session type."""

    root = _stringer_root(data_dir)
    infos: list[StringerSessionInfo] = []
    for entry in load_stringer_database(root):
        stype = str(entry["expt"])
        if session_stimuli_type is not None and stype != session_stimuli_type:
            continue
        infos.append(
            StringerSessionInfo(
                session_file=root / _session_filename(entry),
                session_stimuli_type=stype,
                mouse_name=str(entry["mouse_name"]),
                date=str(entry["date"]),
                block=str(entry["block"]),
                depth=int(entry["depth"]) if "depth" in entry else None,
                notes=str(entry.get("notes", "")),
            )
        )
    return infos


def available_stringer_stimuli_types(data_dir: str | Path | None = None) -> dict[str, int]:
    """Count sessions by ``session_stimuli_type``."""

    counts: dict[str, int] = {}
    for session in list_stringer_sessions(data_dir=data_dir):
        counts[session.session_stimuli_type] = counts.get(session.session_stimuli_type, 0) + 1
    return dict(sorted(counts.items()))


def _resolve_session(
    session: StringerSessionInfo | str | Path | None,
    *,
    session_stimuli_type: str,
    session_index: int,
    data_dir: str | Path | None,
) -> StringerSessionInfo:
    if isinstance(session, StringerSessionInfo):
        return session

    root = _stringer_root(data_dir)
    if session is not None:
        session_file = Path(session).expanduser()
        if not session_file.is_absolute():
            session_file = root / session_file
        for info in list_stringer_sessions(data_dir=root):
            if info.session_file.name == session_file.name:
                return StringerSessionInfo(
                    session_file=session_file,
                    session_stimuli_type=info.session_stimuli_type,
                    mouse_name=info.mouse_name,
                    date=info.date,
                    block=info.block,
                    depth=info.depth,
                    notes=info.notes,
                )
        return StringerSessionInfo(
            session_file=session_file,
            session_stimuli_type=session_file.stem,
            mouse_name="",
            date="",
            block="",
        )

    sessions = list_stringer_sessions(session_stimuli_type, data_dir=root)
    if not sessions:
        raise ValueError(f"No Stringer sessions found for session_stimuli_type={session_stimuli_type!r}.")
    try:
        return sessions[session_index]
    except IndexError as exc:
        raise IndexError(
            f"session_index={session_index} out of range for {session_stimuli_type!r}; "
            f"available sessions: {len(sessions)}"
        ) from exc


def load_stringer_session(
    session: StringerSessionInfo | str | Path | None = STRINGER_EXAMPLE_SESSION_FILE,
    *,
    session_stimuli_type: str = "gratings_static",
    session_index: int = 0,
    data_dir: str | Path | None = None,
    orientation_period: float | None = np.pi,
    trim_mismatch: bool = True,
) -> StringerSession:
    """Load neural responses, grating orientation, and session stimulus type.

    ``neural_responses`` is returned as ``(n_trials, n_neurons)``. The raw files
    store ``sresp`` as ``(n_neurons, n_trials)``.
    """

    info = _resolve_session(
        session,
        session_stimuli_type=session_stimuli_type,
        session_index=session_index,
        data_dir=data_dir,
    )
    if not info.session_file.exists():
        raise FileNotFoundError(f"Stringer session file not found: {info.session_file}")

    raw = np.load(info.session_file, allow_pickle=True).item()
    if "sresp" not in raw or "istim" not in raw:
        raise KeyError(f"{info.session_file} must contain 'sresp' and 'istim'.")

    sresp = np.asarray(raw["sresp"])
    orientation = np.asarray(raw["istim"], dtype=np.float64).reshape(-1)
    if sresp.ndim != 2:
        raise ValueError(f"Expected 2D sresp in {info.session_file}, got shape {sresp.shape}.")

    if sresp.shape[1] == orientation.shape[0]:
        responses = sresp.T
    elif sresp.shape[0] == orientation.shape[0]:
        responses = sresp
    elif trim_mismatch:
        n_trials = min(sresp.shape[1], orientation.shape[0])
        responses = sresp[:, :n_trials].T
        orientation = orientation[:n_trials]
    else:
        raise ValueError(
            f"sresp shape {sresp.shape} and istim shape {orientation.shape} do not share a trial axis."
        )

    if orientation_period is not None:
        orientation = np.mod(orientation, float(orientation_period))

    meta: dict[str, Any] = {
        "data_dir": str(info.session_file.parent),
        "session_file": str(info.session_file),
        "session_stimuli_type": info.session_stimuli_type,
        "mouse_name": info.mouse_name,
        "date": info.date,
        "block": info.block,
        "depth": info.depth,
        "notes": info.notes,
        "raw_keys": sorted(str(k) for k in raw.keys()),
        "raw_sresp_shape": tuple(int(v) for v in sresp.shape),
        "raw_istim_shape": tuple(int(v) for v in np.asarray(raw["istim"]).shape),
        "n_trials": int(responses.shape[0]),
        "n_neurons": int(responses.shape[1]),
        "orientation_period": None if orientation_period is None else float(orientation_period),
        "orientation_min": float(np.min(orientation)) if orientation.size else None,
        "orientation_max": float(np.max(orientation)) if orientation.size else None,
        "orientation_unique_count": int(np.unique(orientation).size),
        "dropped_response_columns": int(max(0, sresp.shape[1] - responses.shape[0])),
        "dropped_orientation_values": int(max(0, np.asarray(raw["istim"]).reshape(-1).shape[0] - responses.shape[0])),
    }

    return StringerSession(
        neural_responses=responses,
        grating_orientation=orientation,
        session_stimuli_type=info.session_stimuli_type,
        session_file=info.session_file,
        meta=meta,
    )
