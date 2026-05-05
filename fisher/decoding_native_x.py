"""Resolve pre-PR native NPZ and slice native x for pairwise decoding aligned with subset bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fisher.shared_dataset_io import SharedDatasetBundle, load_shared_dataset_npz


def _train_rows_from_n_and_meta(n: int, meta: dict[str, Any]) -> int:
    """Match ``_subset_bundle`` train row count for pool size ``n``."""
    n = int(n)
    tf = float(meta["train_frac"])
    if tf >= 1.0:
        return n
    n_train = int(tf * n)
    return int(min(max(n_train, 1), n - 1))


def decoding_x_train_all_from_native(
    native_bundle: SharedDatasetBundle,
    perm: np.ndarray,
    n: int,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Slice native ``x`` with the same ``perm[:n]`` and ``train_frac`` split as ``_subset_bundle``."""
    n = int(n)
    sub_perm = np.asarray(perm[:n], dtype=np.int64)
    x_all = np.asarray(native_bundle.x_all[sub_perm], dtype=np.float64)
    n_train = _train_rows_from_n_and_meta(n, meta)
    x_train = np.asarray(x_all[:n_train], dtype=np.float64)
    return x_train, x_all


def resolve_pr_source_npz_path(
    meta: dict[str, Any],
    embedded_npz_path: str | Path,
    override: str | None,
) -> tuple[Path | None, list[str]]:
    """Resolve pre-embedding NPZ path for PR datasets; return ``None`` if not PR-embedded."""
    if not bool(meta.get("pr_autoencoder_embedded")):
        return None, []
    emb_p = Path(embedded_npz_path).resolve()
    tried: list[str] = []
    candidates: list[Path] = []

    if override:
        o = Path(str(override)).expanduser()
        candidates.append(o if o.is_absolute() else (Path.cwd() / o).resolve())

    raw = meta.get("pr_autoencoder_source_npz")
    if raw:
        p = Path(str(raw)).expanduser()
        candidates.append(p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve())
        candidates.append((emb_p.parent / p.name).resolve())

    try:
        from global_setting import DATA_DIR

        if raw:
            candidates.append((Path(DATA_DIR) / Path(str(raw)).name).resolve())
    except Exception:
        pass

    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        tried.append(key)
        if c.is_file():
            return c, tried
    return None, tried


def verify_native_matches_embedded(
    embedded: SharedDatasetBundle,
    native: SharedDatasetBundle,
    meta: dict[str, Any],
) -> None:
    ne = int(embedded.theta_all.shape[0])
    nn = int(native.theta_all.shape[0])
    if ne != nn:
        raise ValueError(f"Native NPZ row count {nn} does not match embedded {ne}.")
    th_e = np.asarray(embedded.theta_all, dtype=np.float64)
    th_n = np.asarray(native.theta_all, dtype=np.float64)
    if th_e.shape != th_n.shape:
        raise ValueError(
            f"Native theta_all shape {th_n.shape} does not match embedded {th_e.shape}."
        )
    if not np.allclose(th_e, th_n, rtol=0.0, atol=1e-9):
        raise ValueError("Native NPZ theta_all does not match embedded NPZ (row order or draws differ).")
    zd = int(meta.get("pr_autoencoder_z_dim", native.x_all.shape[1]))
    if int(native.x_all.shape[1]) != zd:
        raise ValueError(
            f"Native x second dim {native.x_all.shape[1]} != meta pr_autoencoder_z_dim={zd}."
        )


def load_native_bundle_for_pr_gt_decoding(
    embedded_bundle: SharedDatasetBundle,
    meta: dict[str, Any],
    embedded_npz_path: str | Path,
    override: str | None,
) -> tuple[SharedDatasetBundle | None, list[str]]:
    """Load and verify native bundle; return ``None`` if path did not resolve."""
    resolved, tried = resolve_pr_source_npz_path(meta, embedded_npz_path, override)
    if resolved is None:
        return None, tried
    native = load_shared_dataset_npz(resolved)
    verify_native_matches_embedded(embedded_bundle, native, meta)
    return native, tried
