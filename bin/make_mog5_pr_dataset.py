#!/usr/bin/env python3
"""Generate the fixed K=5 categorical MoG native dataset and PR embedding.

This is a small orchestration wrapper around:

1. ``bin/make_dataset.py`` for the native ``random_mog_categorical`` NPZ.
2. ``bin/project_dataset_pr_autoencoder.py`` for the higher-dimensional PR embedding.

The default output directory is ``<repo-root>/data/mog_5native_xdim3_n{n_total}/``.
Explicit native 2D runs keep the legacy ``mog_5native_n{n_total}`` and
``mog_5pr{pr_dim}_n{n_total}`` names.
Existing native and projected NPZs are reused by default after metadata validation; pass
``--force`` to rerun both stages.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_setting import DATA_DIR, DEFAULT_DEVICE
from fisher.data import ToyCategoricalRandomMoGDataset
from fisher.shared_dataset_io import SharedDatasetBundle, load_shared_dataset_npz, save_shared_dataset_npz

DATASET_FAMILY = "random_mog_categorical"
NUM_CATEGORIES = 5
NATIVE_X_DIM = 2
DEFAULT_NATIVE_X_DIM = 3


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


def parse_pr_dim(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.lower() in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--pr-dim must be an integer, 'none', or 'null'.") from exc


def default_output_dir(*, n_total: int, pr_dim: int | None, native_x_dim: int = DEFAULT_NATIVE_X_DIM) -> Path:
    native_x_dim = int(native_x_dim)
    if native_x_dim < 2:
        raise ValueError(f"--native-x-dim must be >= 2; got {native_x_dim}.")
    if pr_dim is None:
        if native_x_dim == NATIVE_X_DIM:
            return _repo_data_dir() / f"mog_5native_n{int(n_total)}"
        return _repo_data_dir() / f"mog_5native_xdim{native_x_dim}_n{int(n_total)}"
    if native_x_dim == NATIVE_X_DIM:
        return _repo_data_dir() / f"mog_5pr{int(pr_dim)}_n{int(n_total)}"
    return _repo_data_dir() / f"mog_5native_xdim{native_x_dim}_pr{int(pr_dim)}_n{int(n_total)}"


def native_npz_path(output_dir: Path) -> Path:
    return Path(output_dir) / "random_mog_categorical.npz"


def projected_npz_path(output_dir: Path, *, pr_dim: int) -> Path:
    return Path(output_dir) / f"random_mog_categorical_pr{int(pr_dim)}.npz"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate a fixed K=5 native random_mog_categorical dataset and PR-autoencoder "
            "embedding. Existing NPZs are validated and reused unless --force is passed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-total", type=int, default=1_000, help="Total joint samples.")
    p.add_argument(
        "--native-x-dim",
        type=int,
        default=DEFAULT_NATIVE_X_DIM,
        help="Native random_mog_categorical x dimension before optional PR projection.",
    )
    p.add_argument(
        "--pr-dim",
        type=parse_pr_dim,
        default=None,
        help="Target PR-embedded x dimension; must be >= --native-x-dim. Use 'none' or 'null' for native mode.",
    )
    p.add_argument("--seed", type=int, default=7, help="Dataset and PR-autoencoder seed.")
    p.add_argument("--train-frac", type=float, default=0.8, help="Fraction of rows assigned to train_idx.")
    p.add_argument(
        "--obs-noise-scale",
        type=float,
        default=1.0,
        help="Scale factor for the random-MoG baseline observation noise.",
    )
    p.add_argument(
        "--cov-theta-amp-scale",
        type=float,
        default=1.0,
        help="Scale factor for the mean-dependent diagonal variance term.",
    )
    p.add_argument(
        "--mog-mean-min-dist",
        type=float,
        default=None,
        help="Minimum pairwise component-mean distance; omitted uses 0.5*sqrt(native_x_dim).",
    )
    p.add_argument(
        "--native-template-npz",
        type=Path,
        default=None,
        help="Optional native MoG5 NPZ whose fixed component gains, means, and variances are reused.",
    )
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device passed to the PR projection script.")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to dimension-aware <repo-root>/data/mog_5... path.",
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
    return default_output_dir(n_total=int(args.n_total), pr_dim=args.pr_dim, native_x_dim=int(args.native_x_dim))


def validate_args(args: argparse.Namespace) -> None:
    native_x_dim = int(args.native_x_dim)
    if native_x_dim < 2:
        raise ValueError(f"--native-x-dim must be >= 2; got {args.native_x_dim}.")
    if args.pr_dim is not None and int(args.pr_dim) < native_x_dim:
        raise ValueError(f"--pr-dim must be >= native x_dim={native_x_dim}; got {args.pr_dim}.")
    if int(args.n_total) <= 0:
        raise ValueError(f"--n-total must be positive; got {args.n_total}.")
    tf = float(args.train_frac)
    if not (0.0 < tf <= 1.0):
        raise ValueError(f"--train-frac must be in (0, 1]; got {args.train_frac}.")
    for name in ("obs_noise_scale", "cov_theta_amp_scale"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive; got {value}.")
    if args.mog_mean_min_dist is not None:
        value = float(args.mog_mean_min_dist)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--mog-mean-min-dist must be finite and non-negative; got {value}.")
    if args.native_template_npz is not None:
        template = validate_native_npz(
            Path(args.native_template_npz),
            n_total=int(load_shared_dataset_npz(Path(args.native_template_npz)).x_all.shape[0]),
            native_x_dim=native_x_dim,
        )
        meta = dict(template.meta)
        for key in ("mog_component_gains", "mog_component_means", "mog_component_variances"):
            if meta.get(key) is None:
                raise ValueError(f"--native-template-npz is missing {key}.")


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


def validate_native_npz(path: Path, *, n_total: int, native_x_dim: int = DEFAULT_NATIVE_X_DIM) -> SharedDatasetBundle:
    bundle = _validate_common_npz(path, n_total=n_total, expected_x_dim=int(native_x_dim))
    if bool(bundle.meta.get("pr_autoencoder_embedded", False)):
        raise ValueError(f"{path} is already PR-embedded; expected a native NPZ.")
    return bundle


def validate_projected_npz(
    path: Path,
    *,
    n_total: int,
    pr_dim: int,
    native_x_dim: int = DEFAULT_NATIVE_X_DIM,
) -> SharedDatasetBundle:
    bundle = _validate_common_npz(path, n_total=n_total, expected_x_dim=int(pr_dim))
    if not bool(bundle.meta.get("pr_autoencoder_embedded", False)):
        raise ValueError(f"{path} is not marked pr_autoencoder_embedded=True.")
    if int(bundle.meta.get("pr_autoencoder_z_dim", -1)) != int(native_x_dim):
        raise ValueError(
            f"{path} has pr_autoencoder_z_dim={bundle.meta.get('pr_autoencoder_z_dim')!r}; expected {int(native_x_dim)}."
        )
    return bundle


def build_native_command(args: argparse.Namespace, native_npz: Path) -> list[str]:
    command = [
        sys.executable,
        str(_REPO_ROOT / "bin" / "make_dataset.py"),
        "--dataset-family",
        DATASET_FAMILY,
        "--num-categories",
        str(NUM_CATEGORIES),
        "--x-dim",
        str(int(args.native_x_dim)),
        "--n-total",
        str(int(args.n_total)),
        "--train-frac",
        str(float(args.train_frac)),
        "--obs-noise-scale",
        str(float(args.obs_noise_scale)),
        "--cov-theta-amp-scale",
        str(float(args.cov_theta_amp_scale)),
        "--seed",
        str(int(args.seed)),
        "--output-npz",
        str(native_npz),
    ]
    if args.mog_mean_min_dist is not None:
        command.extend(["--mog-mean-min-dist", str(float(args.mog_mean_min_dist))])
    return command


def validate_native_generation_config(
    bundle: SharedDatasetBundle,
    args: argparse.Namespace,
    *,
    path: Path,
) -> None:
    """Reject a cached native dataset generated under incompatible public controls."""

    if args.native_template_npz is not None:
        return
    meta = dict(bundle.meta)
    expected = {
        "seed": float(args.seed),
        "train_frac": float(args.train_frac),
        "obs_noise_scale": float(args.obs_noise_scale),
        "cov_theta_amp_scale": float(args.cov_theta_amp_scale),
        "mog_mean_min_dist": (
            0.5 * math.sqrt(float(args.native_x_dim))
            if args.mog_mean_min_dist is None
            else float(args.mog_mean_min_dist)
        ),
    }
    mismatches: list[str] = []
    for key, wanted in expected.items():
        found = meta.get(key)
        if found is None or not math.isclose(float(found), wanted, rel_tol=1e-10, abs_tol=1e-12):
            mismatches.append(f"{key}: cached={found!r}, requested={wanted!r}")
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"{path} has incompatible generation metadata ({details}); rerun with --force.")


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


def write_native_from_template(args: argparse.Namespace, native_npz: Path) -> None:
    template_path = Path(args.native_template_npz)
    template = load_shared_dataset_npz(template_path)
    meta_template = dict(template.meta)
    if str(meta_template.get("dataset_family", "")) != DATASET_FAMILY:
        raise ValueError(
            f"--native-template-npz has dataset_family={meta_template.get('dataset_family')!r}; "
            f"expected {DATASET_FAMILY!r}."
        )
    if int(meta_template.get("num_categories", -1)) != NUM_CATEGORIES:
        raise ValueError(
            f"--native-template-npz has num_categories={meta_template.get('num_categories')!r}; "
            f"expected {NUM_CATEGORIES}."
        )
    if int(meta_template.get("x_dim", -1)) != int(args.native_x_dim):
        raise ValueError(
            f"--native-template-npz has x_dim={meta_template.get('x_dim')!r}; "
            f"expected {int(args.native_x_dim)}."
        )

    gains = np.asarray(meta_template.get("mog_component_gains"), dtype=np.float64)
    means = np.asarray(meta_template.get("mog_component_means"), dtype=np.float64)
    variances = np.asarray(meta_template.get("mog_component_variances"), dtype=np.float64)
    expected_shape = (NUM_CATEGORIES, int(args.native_x_dim))
    for name, arr in (
        ("mog_component_gains", gains),
        ("mog_component_means", means),
        ("mog_component_variances", variances),
    ):
        if arr.shape != expected_shape:
            raise ValueError(f"--native-template-npz {name} has shape {arr.shape}; expected {expected_shape}.")

    dataset = ToyCategoricalRandomMoGDataset(
        x_dim=int(args.native_x_dim),
        num_categories=NUM_CATEGORIES,
        mog_a_low=float(meta_template.get("mog_a_low", 0.2)),
        mog_a_high=float(meta_template.get("mog_a_high", 2.0)),
        mog_sigma_base=float(meta_template.get("mog_sigma_base", 0.15)),
        mog_alpha=float(meta_template.get("mog_alpha", 0.15)),
        mog_eps=float(meta_template.get("mog_eps", 1e-5)),
        mog_mean_min_dist=meta_template.get("mog_mean_min_dist", None),
        mog_mean_max_attempts=int(meta_template.get("mog_mean_max_attempts", 10_000)),
        mog_component_gains=gains,
        mog_component_means=means,
        mog_component_variances=variances,
        seed=int(args.seed),
    )
    theta_all, x_all = dataset.sample_joint(int(args.n_total))
    rng = np.random.default_rng(int(args.seed))
    perm = rng.permutation(int(args.n_total))
    if float(args.train_frac) >= 1.0:
        n_train = int(args.n_total)
    else:
        n_train = int(float(args.train_frac) * int(args.n_total))
        n_train = min(max(n_train, 1), int(args.n_total) - 1)
    train_idx = perm[:n_train].astype(np.int64)
    validation_idx = perm[n_train:].astype(np.int64)

    meta = dict(meta_template)
    meta.update(
        {
            "dataset_family": DATASET_FAMILY,
            "theta_type": "categorical",
            "theta_encoding": "one_hot",
            "num_categories": NUM_CATEGORIES,
            "x_dim": int(args.native_x_dim),
            "n_total": int(args.n_total),
            "train_frac": float(args.train_frac),
            "seed": int(args.seed),
            "pr_autoencoder_embedded": False,
            "mog_component_gains": gains.tolist(),
            "mog_component_means": means.tolist(),
            "mog_component_variances": variances.tolist(),
            "native_template_npz": str(template_path),
        }
    )
    save_shared_dataset_npz(
        native_npz,
        meta=meta,
        theta_all=theta_all,
        x_all=x_all,
        train_idx=train_idx,
        validation_idx=validation_idx,
        theta_train=theta_all[train_idx],
        x_train=x_all[train_idx],
        theta_validation=theta_all[validation_idx],
        x_validation=x_all[validation_idx],
    )


def run(args: argparse.Namespace, *, runner: Runner = _run_command) -> tuple[Path, Path | None]:
    validate_args(args)
    output_dir = resolve_output_dir(args)
    native_npz = native_npz_path(output_dir)
    projected_npz = None if args.pr_dim is None else projected_npz_path(output_dir, pr_dim=int(args.pr_dim))
    output_dir.mkdir(parents=True, exist_ok=True)

    if native_npz.is_file() and not bool(args.force):
        native_bundle = validate_native_npz(
            native_npz,
            n_total=int(args.n_total),
            native_x_dim=int(args.native_x_dim),
        )
        if native_bundle is not None:
            validate_native_generation_config(native_bundle, args, path=native_npz)
        print(f"[mog5-pr] Reusing native NPZ: {native_npz}", flush=True)
    else:
        if args.native_template_npz is None:
            runner(build_native_command(args, native_npz))
            if bool(args.skip_viz):
                _remove_native_viz(output_dir)
        else:
            write_native_from_template(args, native_npz)
        native_bundle = validate_native_npz(
            native_npz,
            n_total=int(args.n_total),
            native_x_dim=int(args.native_x_dim),
        )
        if native_bundle is not None:
            validate_native_generation_config(native_bundle, args, path=native_npz)

    print(f"[mog5-pr] Native NPZ: {native_npz}", flush=True)
    if projected_npz is None:
        print("[mog5-pr] Native mode requested; skipping PR projection.", flush=True)
        return native_npz, None

    if projected_npz.is_file() and not bool(args.force):
        validate_projected_npz(
            projected_npz,
            n_total=int(args.n_total),
            pr_dim=int(args.pr_dim),
            native_x_dim=int(args.native_x_dim),
        )
        print(f"[mog5-pr] Reusing projected NPZ: {projected_npz}", flush=True)
    else:
        runner(build_project_command(args, native_npz, projected_npz))
        validate_projected_npz(
            projected_npz,
            n_total=int(args.n_total),
            pr_dim=int(args.pr_dim),
            native_x_dim=int(args.native_x_dim),
        )

    print(f"[mog5-pr] Projected NPZ: {projected_npz}", flush=True)
    return native_npz, projected_npz


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
