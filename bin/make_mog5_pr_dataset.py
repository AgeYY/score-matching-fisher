#!/usr/bin/env python3
"""Generate the fixed K=5 categorical MoG native dataset and PR embedding.

This is a small orchestration wrapper around:

1. ``bin/make_dataset.py`` for the native 2D ``random_mog_categorical`` NPZ.
2. ``bin/project_dataset_pr_autoencoder.py`` for the higher-dimensional PR embedding.

The default output directory is ``<repo-root>/data/mog_5pr{pr_dim}_n{n_total}/``.
Existing native and projected NPZs are reused by default after metadata validation; pass
``--force`` to rerun both stages.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import DATA_DIR
from fisher.shared_dataset_io import SharedDatasetBundle, load_shared_dataset_npz

DATASET_FAMILY = "random_mog_categorical"
NUM_CATEGORIES = 5
NATIVE_X_DIM = 2


Runner = Callable[[Sequence[str]], None]


def _repo_data_dir() -> Path:
    """Return the repo-visible data directory when it aliases DATA_DIR."""
    repo_data = _REPO_ROOT / "data"
    data_dir = Path(DATA_DIR)
    if not data_dir.is_absolute():
        return _REPO_ROOT / data_dir
    try:
        if repo_data.exists() and repo_data.resolve() == data_dir.resolve():
            return repo_data
    except OSError:
        pass
    return data_dir


def default_output_dir(*, n_total: int, pr_dim: int) -> Path:
    return _repo_data_dir() / f"mog_5pr{int(pr_dim)}_n{int(n_total)}"


def native_npz_path(output_dir: Path) -> Path:
    return Path(output_dir) / "random_mog_categorical.npz"


def projected_npz_path(output_dir: Path, *, pr_dim: int) -> Path:
    return Path(output_dir) / f"random_mog_categorical_pr{int(pr_dim)}.npz"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate a fixed K=5 native 2D random_mog_categorical dataset and PR-autoencoder "
            "embedding. Existing NPZs are validated and reused unless --force is passed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-total", type=int, default=1_000, help="Total joint samples.")
    p.add_argument("--pr-dim", type=int, default=5, help="Target PR-embedded x dimension; must be >= 2.")
    p.add_argument("--seed", type=int, default=7, help="Dataset and PR-autoencoder seed.")
    p.add_argument("--train-frac", type=float, default=0.8, help="Fraction of rows assigned to train_idx.")
    p.add_argument("--device", type=str, default="cuda", help="Device passed to the PR projection script.")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <repo-root>/data/mog_5pr{pr_dim}_n{n_total}/.",
    )
    p.add_argument("--force", action="store_true", help="Regenerate both native and PR-embedded NPZs.")
    p.add_argument(
        "--use-cache",
        action="store_true",
        help="Allow the PR projection script to reuse a matching PR-autoencoder checkpoint.",
    )
    p.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip PR projection figures and remove native figures generated during this wrapper run.",
    )
    p.add_argument("--pr-train-epochs", type=int, default=None, help="Override PR-autoencoder epochs.")
    p.add_argument("--pr-train-samples", type=int, default=None, help="Override PR-autoencoder train samples.")
    p.add_argument(
        "--pr-train-batch-size",
        type=int,
        default=None,
        help="Override PR-autoencoder train batch size.",
    )
    p.add_argument("--pr-hidden1", type=int, default=None, help="Override first PR-autoencoder hidden width.")
    p.add_argument("--pr-hidden2", type=int, default=None, help="Override second PR-autoencoder hidden width.")
    p.add_argument(
        "--pr-cache-dir",
        type=str,
        default=None,
        help="Cache directory passed to project_dataset_pr_autoencoder.py as --cache-dir.",
    )
    return p.parse_args(argv)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    return default_output_dir(n_total=int(args.n_total), pr_dim=int(args.pr_dim))


def validate_args(args: argparse.Namespace) -> None:
    if int(args.pr_dim) < NATIVE_X_DIM:
        raise ValueError(f"--pr-dim must be >= native x_dim={NATIVE_X_DIM}; got {args.pr_dim}.")
    if int(args.n_total) <= 0:
        raise ValueError(f"--n-total must be positive; got {args.n_total}.")
    tf = float(args.train_frac)
    if not (0.0 < tf <= 1.0):
        raise ValueError(f"--train-frac must be in (0, 1]; got {args.train_frac}.")


def _labels_from_theta(theta: np.ndarray, *, num_categories: int = NUM_CATEGORIES) -> np.ndarray:
    arr = np.asarray(theta)
    k = int(num_categories)
    if arr.ndim == 2 and int(arr.shape[1]) == k:
        vals = np.asarray(arr, dtype=np.float64)
        row_sums = vals.sum(axis=1)
        is_binary = np.all((np.abs(vals) <= 1e-6) | (np.abs(vals - 1.0) <= 1e-6), axis=1)
        if np.any(np.abs(row_sums - 1.0) > 1e-6) or not bool(np.all(is_binary)):
            raise ValueError("Expected one-hot categorical theta rows for K=5.")
        return np.argmax(vals, axis=1).astype(np.int64)

    vals = np.asarray(arr, dtype=np.float64).reshape(-1)
    labels = np.rint(vals).astype(np.int64)
    if np.any(np.abs(vals - labels.astype(np.float64)) > 1e-6):
        raise ValueError("Expected integer categorical theta labels or one-hot rows for K=5.")
    if np.any((labels < 0) | (labels >= k)):
        raise ValueError(f"Categorical labels must be in [0, {k - 1}].")
    return labels


def _require_all_categories(bundle: SharedDatasetBundle, *, path: Path) -> None:
    labels = _labels_from_theta(bundle.theta_all, num_categories=NUM_CATEGORIES)
    observed = set(int(v) for v in np.unique(labels))
    expected = set(range(NUM_CATEGORIES))
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(
            f"{path} does not contain all {NUM_CATEGORIES} categories; missing {missing}. "
            "Try a larger --n-total, a different --seed, or rerun with --force."
        )


def _validate_common_npz(path: Path, *, n_total: int, expected_x_dim: int) -> SharedDatasetBundle:
    bundle = load_shared_dataset_npz(path)
    meta = dict(bundle.meta)
    if str(meta.get("dataset_family", "")) != DATASET_FAMILY:
        raise ValueError(f"{path} has dataset_family={meta.get('dataset_family')!r}; expected {DATASET_FAMILY!r}.")
    if int(meta.get("num_categories", -1)) != NUM_CATEGORIES:
        raise ValueError(f"{path} has num_categories={meta.get('num_categories')!r}; expected {NUM_CATEGORIES}.")
    if int(meta.get("x_dim", -1)) != int(expected_x_dim):
        raise ValueError(f"{path} has x_dim={meta.get('x_dim')!r}; expected {expected_x_dim}.")

    x_all = np.asarray(bundle.x_all)
    theta_all = np.asarray(bundle.theta_all)
    if x_all.ndim != 2 or int(x_all.shape[1]) != int(expected_x_dim):
        raise ValueError(f"{path} x_all has shape {x_all.shape}; expected second dimension {expected_x_dim}.")
    if int(x_all.shape[0]) != int(n_total) or int(theta_all.shape[0]) != int(n_total):
        raise ValueError(f"{path} row count does not match --n-total={n_total}.")
    _require_all_categories(bundle, path=path)
    return bundle


def validate_native_npz(path: Path, *, n_total: int) -> SharedDatasetBundle:
    bundle = _validate_common_npz(path, n_total=n_total, expected_x_dim=NATIVE_X_DIM)
    if bool(bundle.meta.get("pr_autoencoder_embedded", False)):
        raise ValueError(f"{path} is already PR-embedded; expected a native 2D NPZ.")
    return bundle


def validate_projected_npz(path: Path, *, n_total: int, pr_dim: int) -> SharedDatasetBundle:
    bundle = _validate_common_npz(path, n_total=n_total, expected_x_dim=int(pr_dim))
    if not bool(bundle.meta.get("pr_autoencoder_embedded", False)):
        raise ValueError(f"{path} is not marked pr_autoencoder_embedded=True.")
    if int(bundle.meta.get("pr_autoencoder_z_dim", -1)) != NATIVE_X_DIM:
        raise ValueError(f"{path} has pr_autoencoder_z_dim={bundle.meta.get('pr_autoencoder_z_dim')!r}; expected 2.")
    return bundle


def build_native_command(args: argparse.Namespace, native_npz: Path) -> list[str]:
    return [
        sys.executable,
        str(_REPO_ROOT / "bin" / "make_dataset.py"),
        "--dataset-family",
        DATASET_FAMILY,
        "--num-categories",
        str(NUM_CATEGORIES),
        "--x-dim",
        str(NATIVE_X_DIM),
        "--n-total",
        str(int(args.n_total)),
        "--train-frac",
        str(float(args.train_frac)),
        "--seed",
        str(int(args.seed)),
        "--output-npz",
        str(native_npz),
    ]


def build_project_command(args: argparse.Namespace, native_npz: Path, projected_npz: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "bin" / "project_dataset_pr_autoencoder.py"),
        "--input-npz",
        str(native_npz),
        "--output-npz",
        str(projected_npz),
        "--h-dim",
        str(int(args.pr_dim)),
        "--allow-non-randamp-sqrtd",
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed)),
    ]
    if bool(args.use_cache):
        cmd.append("--use-cache")
    if bool(args.skip_viz):
        cmd.append("--skip-viz")
    if args.pr_cache_dir is not None:
        cmd.extend(["--cache-dir", str(args.pr_cache_dir)])
    for attr, flag in (
        ("pr_train_epochs", "--pr-train-epochs"),
        ("pr_train_samples", "--pr-train-samples"),
        ("pr_train_batch_size", "--pr-train-batch-size"),
        ("pr_hidden1", "--pr-hidden1"),
        ("pr_hidden2", "--pr-hidden2"),
    ):
        value = getattr(args, attr)
        if value is not None:
            cmd.extend([flag, str(int(value))])
    return cmd


def _run_command(cmd: Sequence[str]) -> None:
    print("[mog5-pr] Running:", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(list(cmd), check=True)


def _remove_native_viz(output_dir: Path) -> None:
    for name in ("joint_scatter_and_tuning_curve.png", "joint_scatter_and_tuning_curve.svg"):
        path = Path(output_dir) / name
        if path.exists():
            path.unlink()


def run(args: argparse.Namespace, *, runner: Runner = _run_command) -> tuple[Path, Path]:
    validate_args(args)
    output_dir = resolve_output_dir(args)
    native_npz = native_npz_path(output_dir)
    projected_npz = projected_npz_path(output_dir, pr_dim=int(args.pr_dim))
    output_dir.mkdir(parents=True, exist_ok=True)

    if native_npz.is_file() and not bool(args.force):
        validate_native_npz(native_npz, n_total=int(args.n_total))
        print(f"[mog5-pr] Reusing native NPZ: {native_npz}", flush=True)
    else:
        runner(build_native_command(args, native_npz))
        if bool(args.skip_viz):
            _remove_native_viz(output_dir)
        validate_native_npz(native_npz, n_total=int(args.n_total))

    if projected_npz.is_file() and not bool(args.force):
        validate_projected_npz(projected_npz, n_total=int(args.n_total), pr_dim=int(args.pr_dim))
        print(f"[mog5-pr] Reusing projected NPZ: {projected_npz}", flush=True)
    else:
        runner(build_project_command(args, native_npz, projected_npz))
        validate_projected_npz(projected_npz, n_total=int(args.n_total), pr_dim=int(args.pr_dim))

    print(f"[mog5-pr] Native NPZ: {native_npz}", flush=True)
    print(f"[mog5-pr] Projected NPZ: {projected_npz}", flush=True)
    return native_npz, projected_npz


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
