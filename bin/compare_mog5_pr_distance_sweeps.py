#!/usr/bin/env python3
"""Run MoG5 PR distance comparisons across sample-size sweeps."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.distance_comparison import METRIC_NAMES
from fisher.dataset_visualization import plot_mog5_native_scatter_covariance

RESULTS_NAME = "mog5_pr_distance_comparison_results.npz"
SWEEP_NPZ_NAME = "mog5_pr_distance_sweep_results.npz"
SWEEP_CSV_NAME = "mog5_pr_distance_sweep_errors.csv"
SWEEP_SUMMARY_NAME = "mog5_pr_distance_sweep_summary.json"
SWEEP_SVG_NAME = "mog5_pr_distance_sweep_abs_error.svg"
SWEEP_PNG_NAME = "mog5_pr_distance_sweep_abs_error.png"
SWEEP_REL_SVG_NAME = "mog5_pr_distance_sweep_rel_error.svg"
SWEEP_REL_PNG_NAME = "mog5_pr_distance_sweep_rel_error.png"
SWEEP_FLOW_LOSS_SVG_NAME = "mog5_pr_distance_sweep_flow_loss_vs_epoch.svg"
SWEEP_FLOW_LOSS_PNG_NAME = "mog5_pr_distance_sweep_flow_loss_vs_epoch.png"
GROUND_TRUTH_RDMS_SVG_NAME = "mog5_pr_distance_ground_truth_rdms.svg"
GROUND_TRUTH_RDMS_PNG_NAME = "mog5_pr_distance_ground_truth_rdms.png"
MOG5_DATASET_SVG_NAME = "mog5_native_dataset_scatter_covariance.svg"
MOG5_DATASET_PNG_NAME = "mog5_native_dataset_scatter_covariance.png"
REL_ERROR_DENOM_FLOOR = 1e-12
GROUND_TRUTH_RDM_METRIC_ORDER = (
    "correlation",
    "cosine",
    "squared_euclidean",
    "mahalanobis_sq",
    "symmetric_kl",
)
GROUND_TRUTH_RDM_METRIC_TITLES = {
    "correlation": "Correlation",
    "cosine": "Cosine",
    "squared_euclidean": "Squared Euclidean",
    "mahalanobis_sq": "Squared Mahalanobis",
    "symmetric_kl": "Jeffreys divergence",
}


def _load_single_case_module() -> Any:
    path = _REPO_ROOT / "bin" / "compare_mog5_pr_distances.py"
    spec = importlib.util.spec_from_file_location("compare_mog5_pr_distances", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_int_list(value: str) -> list[int]:
    vals = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer.")
    return vals


def _repo_data_dir() -> Path:
    return _REPO_ROOT / "data"


def default_output_dir(*, pr_dim: int | None = 5, native_x_dim: int = 3) -> Path:
    native_x_dim = int(native_x_dim)
    if native_x_dim == 2:
        return _repo_data_dir() / "mog5_pr_distance_sweeps"
    return _repo_data_dir() / f"mog5_native_xdim{native_x_dim}_pr{int(pr_dim)}_distance_sweeps"


def default_native_output_dir(*, native_x_dim: int = 3) -> Path:
    native_x_dim = int(native_x_dim)
    if native_x_dim == 2:
        return _repo_data_dir() / "mog5_native_distance_sweeps"
    return _repo_data_dir() / f"mog5_native_xdim{native_x_dim}_distance_sweeps"


def _pr_projected(pr_dim: int | None) -> bool:
    return pr_dim is not None


def _pr_dim_storage(pr_dim: int | None) -> int:
    return -1 if pr_dim is None else int(pr_dim)


def _pr_dim_label(pr_dim: int | None) -> str:
    return "native" if pr_dim is None else f"pr{int(pr_dim)}"


def _case_key(n_total: int, pr_dim: int | None) -> tuple[int, int]:
    return (int(n_total), _pr_dim_storage(pr_dim))


def _repeat_case_key(n_total: int, pr_dim: int | None, repeat_idx: int) -> tuple[int, int, int]:
    return (int(n_total), _pr_dim_storage(pr_dim), int(repeat_idx))


def _n_repeats(args: argparse.Namespace) -> int:
    return int(getattr(args, "n_repeats", 1))


def _repeat_seed(args: argparse.Namespace, repeat_idx: int) -> int:
    return int(args.seed) + int(repeat_idx)


def build_parser() -> argparse.ArgumentParser:
    single = _load_single_case_module()
    p = single.build_parser()
    p.description = __doc__
    p.set_defaults(n_total=100_000, pr_dim=None, output_dir=default_native_output_dir(native_x_dim=3))
    for action in p._actions:
        if action.dest == "output_dir":
            action.help = "Aggregate sweep output directory."
    p.add_argument(
        "--n-list",
        type=_parse_int_list,
        default=[100, 550, 1000, 1550],
        help="Comma-separated sample-size sweep values.",
    )
    p.add_argument(
        "--case-output-name",
        type=str,
        default="distance_comparison_flow_skl",
        help="Per-case output directory name under each MoG5 PR dataset directory.",
    )
    p.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Independent finite-sample repeats per n_total value.",
    )
    p.add_argument(
        "--force-comparison",
        action="store_true",
        help="Rerun single-case comparisons even when the per-case result NPZ exists.",
    )
    p.add_argument(
        "--visualization-only",
        action="store_true",
        help="Only rebuild aggregate tables and figures from cached per-case result NPZs.",
    )
    p.add_argument("--yscale", choices=("log", "linear"), default="linear", help="Y-axis scale for the error figure.")
    p.add_argument(
        "--loss-yscale",
        choices=("log", "linear"),
        default="linear",
        help="Y-axis scale for the aggregate flow loss-vs-epoch figure.",
    )
    original_parse_args = p.parse_args

    def parse_args(args=None, namespace=None):
        parsed = original_parse_args(args, namespace)
        argv = sys.argv[1:] if args is None else list(args)
        output_was_explicit = any(str(arg) == "--output-dir" or str(arg).startswith("--output-dir=") for arg in argv)
        if not output_was_explicit:
            parsed.output_dir = (
                default_native_output_dir(native_x_dim=int(parsed.native_x_dim))
                if parsed.pr_dim is None
                else default_output_dir(pr_dim=parsed.pr_dim, native_x_dim=int(parsed.native_x_dim))
            )
        return parsed

    p.parse_args = parse_args  # type: ignore[method-assign]
    return p


def case_output_dir(
    *,
    n_total: int,
    pr_dim: int | None,
    case_output_name: str,
    native_x_dim: int = 3,
    repeat_idx: int = 0,
    n_repeats: int = 1,
) -> Path:
    single = _load_single_case_module()
    dataset_dir = single.default_dataset_dir(n_total=int(n_total), pr_dim=pr_dim, native_x_dim=int(native_x_dim))
    if int(n_repeats) > 1:
        return dataset_dir / f"repeat_{int(repeat_idx):02d}" / str(case_output_name)
    return dataset_dir / str(case_output_name)


def case_results_npz(
    *,
    n_total: int,
    pr_dim: int | None,
    case_output_name: str,
    native_x_dim: int = 3,
    repeat_idx: int = 0,
    n_repeats: int = 1,
) -> Path:
    return (
        case_output_dir(
            n_total=n_total,
            pr_dim=pr_dim,
            case_output_name=case_output_name,
            native_x_dim=int(native_x_dim),
            repeat_idx=int(repeat_idx),
            n_repeats=int(n_repeats),
        )
        / RESULTS_NAME
    )


def case_flow_loss_npz(
    *,
    n_total: int,
    pr_dim: int | None,
    case_output_name: str,
    metric: str,
    native_x_dim: int = 3,
    repeat_idx: int = 0,
    n_repeats: int = 1,
) -> Path:
    return (
        case_output_dir(
            n_total=n_total,
            pr_dim=pr_dim,
            case_output_name=case_output_name,
            native_x_dim=int(native_x_dim),
            repeat_idx=int(repeat_idx),
            n_repeats=int(n_repeats),
        )
        / "flow"
        / f"{metric}_flow_matching_skl_results.npz"
    )


def representative_native_npz(
    *,
    args: argparse.Namespace,
    case_paths: dict[tuple[int, int, int], Path],
) -> Path:
    n_total = max(int(v) for v in args.n_list)
    case = _repeat_case_key(n_total, args.pr_dim, 0)
    if case in case_paths:
        return Path(case_paths[case]).parent.parent / "random_mog_categorical.npz"
    single = _load_single_case_module()
    return (
        Path(single.default_dataset_dir(n_total=n_total, pr_dim=args.pr_dim, native_x_dim=int(args.native_x_dim)))
        / "random_mog_categorical.npz"
    )


def maybe_plot_representative_dataset(
    *,
    args: argparse.Namespace,
    case_paths: dict[tuple[int, int, int], Path],
    output_dir: Path,
) -> tuple[Path, Path] | None:
    if bool(getattr(args, "skip_dataset_viz", False)):
        return None
    native_npz = representative_native_npz(args=args, case_paths=case_paths)
    if not native_npz.is_file():
        print(
            f"[sweep] warning: skipped native MoG5 dataset figure; missing representative NPZ: {native_npz}",
            flush=True,
        )
        return None
    return plot_mog5_native_scatter_covariance(
        native_npz,
        svg_path=output_dir / MOG5_DATASET_SVG_NAME,
        png_path=output_dir / MOG5_DATASET_PNG_NAME,
        max_points=500,
    )


def _unique_cases(args: argparse.Namespace) -> list[tuple[int, int | None, int]]:
    cases: list[tuple[int, int | None, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for n_total in args.n_list:
        for repeat_idx in range(_n_repeats(args)):
            case = (int(n_total), args.pr_dim, int(repeat_idx))
            key = _repeat_case_key(int(n_total), args.pr_dim, int(repeat_idx))
            if key not in seen:
                cases.append(case)
                seen.add(key)
    return cases


def resolve_metric_names(args: argparse.Namespace) -> tuple[str, ...]:
    single = _load_single_case_module()
    if hasattr(single, "resolve_metric_names"):
        return tuple(str(m) for m in single.resolve_metric_names(args))
    metric = str(getattr(args, "metric", "all"))
    if metric == "all":
        return tuple(METRIC_NAMES)
    return (metric,)


def validate_args(args: argparse.Namespace) -> None:
    native_x_dim = int(args.native_x_dim)
    if native_x_dim < 2:
        raise ValueError(f"--native-x-dim must be >= 2; got {args.native_x_dim}.")
    if args.pr_dim is not None and int(args.pr_dim) < native_x_dim:
        raise ValueError(f"--pr-dim must be >= native x_dim={native_x_dim}; got {args.pr_dim}.")
    if _n_repeats(args) < 1:
        raise ValueError(f"--n-repeats must be >= 1; got {getattr(args, 'n_repeats', None)}.")


def compute_baseline_ground_truth_rdms(args: argparse.Namespace, metrics: tuple[str, ...]) -> dict[str, Any]:
    single = _load_single_case_module()
    case_args = _single_case_args(
        args,
        n_total=int(args.n_total),
        pr_dim=args.pr_dim,
        output_dir=case_output_dir(
            n_total=int(args.n_total),
            pr_dim=args.pr_dim,
            case_output_name=str(args.case_output_name),
            native_x_dim=int(args.native_x_dim),
            repeat_idx=0,
            n_repeats=1,
        ),
    )
    dev = single.require_device(str(case_args.device))
    dataset_dir = single.resolve_dataset_dir(case_args)
    native_npz, projected_npz = single.ensure_dataset(case_args, dataset_dir)
    pr_projected = case_args.pr_dim is not None
    work_npz = projected_npz if pr_projected else native_npz
    if work_npz is None:
        raise RuntimeError("Native mode did not produce a work NPZ.")

    native_bundle = single.load_shared_dataset_npz(native_npz)
    work_bundle = single.load_shared_dataset_npz(work_npz)
    single._validate_bundle(
        work_bundle,
        n_total=int(case_args.n_total),
        pr_dim=case_args.pr_dim,
        native_x_dim=int(case_args.native_x_dim),
        pr_projected=pr_projected,
    )
    k = int(work_bundle.meta.get("num_categories", 5))
    names = single.condition_labels(k)

    if pr_projected:
        print("[sweep] computing projected-coordinate ground-truth RDMs for baseline n_total", flush=True)
        ground_truth = single.pr_autoencoder_ground_truth_matrices(
            native_meta=dict(native_bundle.meta),
            projected_meta=dict(work_bundle.meta),
            device=dev,
            cache_dir=Path(case_args.pr_cache_dir),
            samples_per_class=int(case_args.gt_samples_per_class),
            seed=int(case_args.seed) + 12345,
            batch_size=int(case_args.gt_batch_size),
            mahalanobis_ridge=float(case_args.mahalanobis_ridge),
            metrics=metrics,
        )
    else:
        print("[sweep] computing native-coordinate ground-truth RDMs for baseline n_total", flush=True)
        ground_truth = single.native_mog_ground_truth_matrices(
            native_meta=dict(native_bundle.meta),
            samples_per_class=int(case_args.gt_samples_per_class),
            seed=int(case_args.seed) + 12345,
            mahalanobis_ridge=float(case_args.mahalanobis_ridge),
            metrics=metrics,
        )

    return {
        "metric_names": tuple(metrics),
        "condition_labels": tuple(names),
        "ground_truth_matrices": np.stack(
            [np.asarray(ground_truth[str(metric)], dtype=np.float64) for metric in metrics],
            axis=0,
        ),
        "n_total": int(case_args.n_total),
        "native_x_dim": int(case_args.native_x_dim),
        "pr_dim": None if case_args.pr_dim is None else int(case_args.pr_dim),
        "pr_projected": bool(pr_projected),
        "pr_dim_label": _pr_dim_label(case_args.pr_dim),
        "native_npz": str(native_npz),
        "work_npz": str(work_npz),
    }


def _single_case_args(
    args: argparse.Namespace,
    *,
    n_total: int,
    pr_dim: int | None,
    output_dir: Path,
    repeat_idx: int = 0,
    native_template_npz: Path | None = None,
) -> argparse.Namespace:
    single = _load_single_case_module()
    case_args = single.build_parser().parse_args([])
    for key, value in vars(args).items():
        if hasattr(case_args, key):
            setattr(case_args, key, value)
    case_args.n_total = int(n_total)
    case_args.pr_dim = pr_dim
    case_args.seed = _repeat_seed(args, int(repeat_idx))
    case_args.dataset_dir = Path(output_dir).parent
    case_args.output_dir = Path(output_dir)
    if hasattr(case_args, "native_template_npz"):
        case_args.native_template_npz = None if native_template_npz is None else Path(native_template_npz)
    return case_args


def _load_case_cache(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing cached comparison results: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {
            "metric_names": tuple(str(v) for v in data["metric_names"].tolist()),
            "condition_labels": tuple(str(v) for v in data["condition_labels"].tolist()),
            "pair_indices": np.asarray(data["pair_indices"], dtype=np.int64),
            "classical_matrices": np.asarray(data["classical_matrices"], dtype=np.float64),
            "flow_matching_matrices": np.asarray(data["flow_matching_matrices"], dtype=np.float64),
            "ground_truth_matrices": np.asarray(data["ground_truth_matrices"], dtype=np.float64),
        }


def _load_flow_loss_cache(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing cached flow loss results: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = ("train_losses", "val_losses")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"Cached flow loss results {path} are missing: {', '.join(missing)}")
        out: dict[str, Any] = {
            "train_losses": np.asarray(data["train_losses"], dtype=np.float64),
            "val_losses": np.asarray(data["val_losses"], dtype=np.float64),
        }
        if "val_monitor_losses" in data.files:
            out["val_monitor_losses"] = np.asarray(data["val_monitor_losses"], dtype=np.float64)
        for key in ("best_epoch", "stopped_epoch", "stopped_early"):
            if key in data.files:
                value = np.asarray(data[key]).reshape(-1)
                if value.size:
                    out[key] = value[0].item()
        return out


def _filter_case_metrics(data: dict[str, Any], metrics: tuple[str, ...], *, path: Path | None = None) -> dict[str, Any]:
    available = tuple(str(v) for v in data["metric_names"])
    missing = [metric for metric in metrics if metric not in available]
    if missing:
        where = "" if path is None else f" in {path}"
        raise ValueError(f"Cached comparison results{where} are missing requested metric(s): {', '.join(missing)}")
    indices = [available.index(metric) for metric in metrics]
    return {
        "metric_names": tuple(metrics),
        "condition_labels": tuple(data["condition_labels"]),
        "pair_indices": np.asarray(data["pair_indices"], dtype=np.int64),
        "classical_matrices": np.asarray(data["classical_matrices"], dtype=np.float64)[indices],
        "flow_matching_matrices": np.asarray(data["flow_matching_matrices"], dtype=np.float64)[indices],
        "ground_truth_matrices": np.asarray(data["ground_truth_matrices"], dtype=np.float64)[indices],
    }


def ensure_case_results(
    args: argparse.Namespace,
    *,
    n_total: int,
    pr_dim: int | None,
    repeat_idx: int = 0,
    native_template_npz: Path | None = None,
) -> tuple[Path, bool]:
    output_dir = case_output_dir(
        n_total=n_total,
        pr_dim=pr_dim,
        case_output_name=str(args.case_output_name),
        native_x_dim=int(args.native_x_dim),
        repeat_idx=int(repeat_idx),
        n_repeats=_n_repeats(args),
    )
    result_path = output_dir / RESULTS_NAME
    if result_path.is_file() and not bool(args.force_comparison):
        requested_metrics = resolve_metric_names(args)
        try:
            _filter_case_metrics(_load_case_cache(result_path), requested_metrics, path=result_path)
            print(
                f"[sweep] cache hit n_total={n_total} repeat={int(repeat_idx)} "
                f"pr_dim={_pr_dim_label(pr_dim)}: {result_path}",
                flush=True,
            )
            return result_path, True
        except ValueError:
            if bool(args.visualization_only):
                raise
            print(
                f"[sweep] cache missing requested metrics; rerunning n_total={n_total} "
                f"repeat={int(repeat_idx)} pr_dim={_pr_dim_label(pr_dim)}",
                flush=True,
            )
    if bool(args.visualization_only):
        raise FileNotFoundError(f"--visualization-only requires cached results: {result_path}")

    single = _load_single_case_module()
    print(
        f"[sweep] running comparison n_total={n_total} repeat={int(repeat_idx)} "
        f"seed={_repeat_seed(args, int(repeat_idx))} pr_dim={_pr_dim_label(pr_dim)}",
        flush=True,
    )
    paths = single.run(
        _single_case_args(
            args,
            n_total=n_total,
            pr_dim=pr_dim,
            output_dir=output_dir,
            repeat_idx=int(repeat_idx),
            native_template_npz=native_template_npz,
        )
    )
    return Path(paths["results_npz"]), False


def _mean_pair_abs_error(est: np.ndarray, gt: np.ndarray, pairs: np.ndarray) -> float:
    vals = [abs(float(est[int(i), int(j)]) - float(gt[int(i), int(j)])) for i, j in pairs]
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def _relative_error(abs_error: float, ground_truth: float) -> float:
    return float(abs_error) / max(abs(float(ground_truth)), REL_ERROR_DENOM_FLOOR)


def _mean_pair_error(est: np.ndarray, gt: np.ndarray, pairs: np.ndarray, *, relative: bool) -> float:
    vals = []
    for i, j in pairs:
        ci, cj = int(i), int(j)
        truth = float(gt[ci, cj])
        abs_error = abs(float(est[ci, cj]) - truth)
        vals.append(_relative_error(abs_error, truth) if relative else abs_error)
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def _mean_pair_error_curve(
    matrices: np.ndarray,
    gt: np.ndarray,
    pair_indices: np.ndarray,
    *,
    metric_idx: int,
    relative: bool,
) -> np.ndarray:
    arr = np.asarray(matrices, dtype=np.float64)
    gt_arr = np.asarray(gt, dtype=np.float64)
    if arr.ndim == 5:
        return np.asarray(
            [
                [
                    _mean_pair_error(
                        np.asarray(arr[n_idx, repeat_idx, int(metric_idx)], dtype=np.float64),
                        np.asarray(gt_arr[n_idx, repeat_idx, int(metric_idx)], dtype=np.float64),
                        pair_indices,
                        relative=bool(relative),
                    )
                    for repeat_idx in range(int(arr.shape[1]))
                ]
                for n_idx in range(int(arr.shape[0]))
            ],
            dtype=np.float64,
        )
    return np.asarray(
        [
            _mean_pair_error(
                np.asarray(arr[row_idx, int(metric_idx)], dtype=np.float64),
                np.asarray(gt_arr[row_idx, int(metric_idx)], dtype=np.float64),
                pair_indices,
                relative=bool(relative),
            )
            for row_idx in range(int(arr.shape[0]))
        ],
        dtype=np.float64,
    )


def aggregate_sweeps(
    *,
    args: argparse.Namespace,
    case_data: dict[tuple[int, int, int], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    first_n_total, first_pr_dim, first_repeat_idx = _unique_cases(args)[0]
    first_key3 = _repeat_case_key(first_n_total, first_pr_dim, first_repeat_idx)
    first = case_data[first_key3] if first_key3 in case_data else case_data[_case_key(first_n_total, first_pr_dim)]  # type: ignore[index]
    metric_names = tuple(first["metric_names"])
    condition_labels = tuple(first["condition_labels"])
    pair_indices = np.asarray(first["pair_indices"], dtype=np.int64)

    for case, data in case_data.items():
        if tuple(data["metric_names"]) != metric_names:
            raise ValueError(f"Metric names differ for case {case}.")
        if tuple(data["condition_labels"]) != condition_labels:
            raise ValueError(f"Condition labels differ for case {case}.")
        np.testing.assert_array_equal(np.asarray(data["pair_indices"], dtype=np.int64), pair_indices)

    repeat_indices = np.arange(_n_repeats(args), dtype=np.int64)

    def get_case(n_total: int, repeat_idx: int) -> dict[str, Any]:
        key3 = _repeat_case_key(int(n_total), args.pr_dim, int(repeat_idx))
        if key3 in case_data:
            return case_data[key3]
        key2 = _case_key(int(n_total), args.pr_dim)
        return case_data[key2]  # type: ignore[index]

    def stack_repeat(key: str) -> np.ndarray:
        return np.stack(
            [
                np.stack(
                    [np.asarray(get_case(int(n_total), int(repeat_idx))[key], dtype=np.float64) for repeat_idx in repeat_indices],
                    axis=0,
                )
                for n_total in args.n_list
            ],
            axis=0,
        )

    n_repeat_classical = stack_repeat("classical_matrices")
    n_repeat_flow = stack_repeat("flow_matching_matrices")
    n_repeat_ground_truth = stack_repeat("ground_truth_matrices")

    aggregate = {
        "metric_names": metric_names,
        "condition_labels": condition_labels,
        "pair_indices": pair_indices,
        "n_list": np.asarray(args.n_list, dtype=np.int64),
        "pr_dim": None if args.pr_dim is None else int(args.pr_dim),
        "pr_projected": _pr_projected(args.pr_dim),
        "pr_dim_label": _pr_dim_label(args.pr_dim),
        "pr_dim_storage": _pr_dim_storage(args.pr_dim),
        "native_x_dim": int(args.native_x_dim),
        "n_total": int(args.n_total),
        "n_repeats": _n_repeats(args),
        "repeat_indices": repeat_indices,
        "repeat_seeds": np.asarray([_repeat_seed(args, int(r)) for r in repeat_indices], dtype=np.int64),
        "n_repeat_classical_matrices": n_repeat_classical,
        "n_repeat_flow_matching_matrices": n_repeat_flow,
        "n_repeat_ground_truth_matrices": n_repeat_ground_truth,
        "n_sweep_classical_matrices": np.mean(n_repeat_classical, axis=1),
        "n_sweep_flow_matching_matrices": np.mean(n_repeat_flow, axis=1),
        "n_sweep_ground_truth_matrices": np.mean(n_repeat_ground_truth, axis=1),
    }

    rows: list[dict[str, Any]] = []
    for n_total in args.n_list:
        for repeat_idx in repeat_indices:
            data = get_case(int(n_total), int(repeat_idx))
            for metric_idx, metric in enumerate(metric_names):
                gt = np.asarray(data["ground_truth_matrices"][metric_idx], dtype=np.float64)
                for i, j in pair_indices:
                    ci, cj = int(i), int(j)
                    for estimator, matrix_key in (
                        ("classical", "classical_matrices"),
                        ("flow_matching", "flow_matching_matrices"),
                    ):
                        est = float(np.asarray(data[matrix_key][metric_idx], dtype=np.float64)[ci, cj])
                        truth = float(gt[ci, cj])
                        abs_error = abs(est - truth)
                        rows.append(
                            {
                                "axis": "n_total",
                                "n_total": int(n_total),
                                "repeat_idx": int(repeat_idx),
                                "repeat_seed": _repeat_seed(args, int(repeat_idx)),
                                "pr_dim": None if args.pr_dim is None else _pr_dim_storage(args.pr_dim),
                                "pr_projected": _pr_projected(args.pr_dim),
                                "pr_dim_label": _pr_dim_label(args.pr_dim),
                                "native_x_dim": int(args.native_x_dim),
                                "metric": str(metric),
                                "estimator": estimator,
                                "condition_i": condition_labels[ci],
                                "condition_j": condition_labels[cj],
                                "estimate": est,
                                "ground_truth": truth,
                                "abs_error": abs_error,
                                "rel_error": _relative_error(abs_error, truth),
                            }
                        )
    return aggregate, rows


def write_aggregate_npz(path: Path, aggregate: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metric_names=np.asarray(aggregate["metric_names"]),
        condition_labels=np.asarray(aggregate["condition_labels"]),
        pair_indices=np.asarray(aggregate["pair_indices"], dtype=np.int64),
        n_list=np.asarray(aggregate["n_list"], dtype=np.int64),
        pr_dim=np.asarray([int(aggregate["pr_dim_storage"])], dtype=np.int64),
        pr_projected=np.asarray([bool(aggregate["pr_projected"])]),
        pr_dim_label=np.asarray([str(aggregate["pr_dim_label"])]),
        native_x_dim=np.asarray([int(aggregate["native_x_dim"])], dtype=np.int64),
        n_total=np.asarray([int(aggregate["n_total"])], dtype=np.int64),
        n_repeats=np.asarray([int(aggregate["n_repeats"])], dtype=np.int64),
        repeat_indices=np.asarray(aggregate["repeat_indices"], dtype=np.int64),
        repeat_seeds=np.asarray(aggregate["repeat_seeds"], dtype=np.int64),
        n_sweep_classical_matrices=np.asarray(aggregate["n_sweep_classical_matrices"], dtype=np.float64),
        n_sweep_flow_matching_matrices=np.asarray(aggregate["n_sweep_flow_matching_matrices"], dtype=np.float64),
        n_sweep_ground_truth_matrices=np.asarray(aggregate["n_sweep_ground_truth_matrices"], dtype=np.float64),
        n_repeat_classical_matrices=np.asarray(aggregate["n_repeat_classical_matrices"], dtype=np.float64),
        n_repeat_flow_matching_matrices=np.asarray(aggregate["n_repeat_flow_matching_matrices"], dtype=np.float64),
        n_repeat_ground_truth_matrices=np.asarray(aggregate["n_repeat_ground_truth_matrices"], dtype=np.float64),
    )
    return path


def write_errors_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "axis",
        "n_total",
        "repeat_idx",
        "repeat_seed",
        "pr_dim",
        "pr_projected",
        "pr_dim_label",
        "native_x_dim",
        "metric",
        "estimator",
        "condition_i",
        "condition_j",
        "estimate",
        "ground_truth",
        "abs_error",
        "rel_error",
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_sweep_error(
    aggregate: dict[str, Any],
    *,
    svg_path: Path,
    png_path: Path,
    yscale: str,
    relative: bool,
) -> tuple[Path, Path]:
    metric_names = tuple(str(v) for v in aggregate["metric_names"])
    if not metric_names:
        raise ValueError("aggregate must contain at least one metric.")
    pair_indices = np.asarray(aggregate["pair_indices"], dtype=np.int64)
    estimator_styles = {
        "classical": {"color": "tab:blue", "marker": "o", "linestyle": "-", "label": "classical"},
        "flow_matching": {"color": "tab:orange", "marker": "s", "linestyle": "--", "label": "flow matching"},
    }
    row_specs = (
        ("Classical + flow", ("classical", "flow_matching")),
        ("Flow only", ("flow_matching",)),
    )
    xvals = np.asarray(aggregate["n_list"], dtype=np.int64)
    gt = np.asarray(aggregate.get("n_repeat_ground_truth_matrices", aggregate["n_sweep_ground_truth_matrices"]), dtype=np.float64)
    matrices_by_estimator = {
        "classical": np.asarray(
            aggregate.get("n_repeat_classical_matrices", aggregate["n_sweep_classical_matrices"]),
            dtype=np.float64,
        ),
        "flow_matching": np.asarray(
            aggregate.get("n_repeat_flow_matching_matrices", aggregate["n_sweep_flow_matching_matrices"]),
            dtype=np.float64,
        ),
    }
    error_label = "Mean relative absolute error" if relative else "Mean absolute error"

    fig_width = max(7.0, 3.6 * len(metric_names))
    fig, axes = plt.subplots(2, len(metric_names), figsize=(fig_width, 8.0), squeeze=False, constrained_layout=True)
    handles_by_label: dict[str, Any] = {}
    for metric_idx, metric in enumerate(metric_names):
        axes[0, metric_idx].set_title(metric)
        for row_idx, (row_label, estimators) in enumerate(row_specs):
            ax = axes[row_idx, metric_idx]
            for estimator in estimators:
                matrices = matrices_by_estimator[estimator]
                yvals = _mean_pair_error_curve(
                    matrices,
                    gt,
                    pair_indices,
                    metric_idx=metric_idx,
                    relative=bool(relative),
                )
                style = estimator_styles[estimator]
                y_arr = np.asarray(yvals, dtype=np.float64)
                if y_arr.ndim == 1:
                    y_repeat = y_arr[:, None]
                else:
                    y_repeat = y_arr
                for repeat_idx in range(int(y_repeat.shape[1])):
                    scatter = ax.scatter(
                        xvals,
                        y_repeat[:, repeat_idx],
                        color=style["color"],
                        marker=style["marker"],
                        s=18,
                        alpha=0.45,
                        linewidths=0.0,
                        label=f"{style['label']} repeats" if repeat_idx == 0 else None,
                        zorder=2,
                    )
                    if repeat_idx == 0 and scatter.get_label():
                        handles_by_label[scatter.get_label()] = scatter
                y_mean = np.mean(y_repeat, axis=1)
                (line,) = ax.plot(
                    xvals,
                    y_mean,
                    color=style["color"],
                    marker=None,
                    linestyle=style["linestyle"],
                    linewidth=1.8,
                    label=f"{style['label']} mean",
                    zorder=3,
                )
                handles_by_label[line.get_label()] = line
            ax.set_xlabel("n_total")
            ax.set_ylabel(error_label)
            if metric_idx == 0:
                ax.text(
                    -0.36,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10,
                    fontweight="semibold",
                )
            ax.set_xticks(xvals)
            ax.grid(True, which="both", alpha=0.25)
            if str(yscale) == "log":
                ax.set_yscale("log")
                ymin = min(
                    (
                        line.get_ydata()[line.get_ydata() > 0].min()
                        for line in ax.lines
                        if np.any(line.get_ydata() > 0)
                    ),
                    default=1e-12,
                )
                ax.set_ylim(bottom=max(float(ymin) * 0.5, 1e-12))
    fig.legend(
        handles_by_label.values(),
        handles_by_label.keys(),
        loc="lower center",
        ncol=max(1, len(handles_by_label)),
        frameon=False,
        fontsize=8,
    )
    title_kind = "relative absolute error" if relative else "absolute error"
    fig.suptitle(f"MoG5 PR distance comparison {title_kind} ({aggregate['pr_dim_label']})", fontsize=13)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Description": (
            f"layout=2x{len(metric_names)};metrics={','.join(metric_names)};"
            "rows=classical+flow,flow_only;points=repeats;lines=repeat_means"
        )
    }
    fig.savefig(svg_path, metadata=metadata)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return svg_path, png_path


def plot_abs_error(aggregate: dict[str, Any], *, svg_path: Path, png_path: Path, yscale: str) -> tuple[Path, Path]:
    return plot_sweep_error(aggregate, svg_path=svg_path, png_path=png_path, yscale=yscale, relative=False)


def plot_ground_truth_rdms(
    ground_truth: dict[str, Any],
    *,
    svg_path: Path,
    png_path: Path,
) -> tuple[Path, Path]:
    metric_names = tuple(str(v) for v in ground_truth["metric_names"])
    condition_labels = tuple(str(v) for v in ground_truth["condition_labels"])
    matrices = np.asarray(ground_truth["ground_truth_matrices"], dtype=np.float64)
    if matrices.ndim != 3:
        raise ValueError("ground_truth_matrices must have shape [metric, condition, condition].")
    if matrices.shape[0] != len(metric_names):
        raise ValueError("ground_truth_matrices metric axis does not match metric_names.")
    if matrices.shape[1] != len(condition_labels) or matrices.shape[2] != len(condition_labels):
        raise ValueError("ground_truth_matrices condition axes do not match condition_labels.")

    ordered_metric_names = tuple(metric for metric in GROUND_TRUTH_RDM_METRIC_ORDER if metric in metric_names)
    ordered_metric_names += tuple(metric for metric in metric_names if metric not in ordered_metric_names)
    ordered_indices = [metric_names.index(metric) for metric in ordered_metric_names]
    ordered_matrices = matrices[ordered_indices]

    fig_width = max(4.8, 3.8 * len(ordered_metric_names))
    fig, axes_obj = plt.subplots(1, len(ordered_metric_names), figsize=(fig_width, 4.6), squeeze=False, constrained_layout=True)
    axes = axes_obj[0]
    images = []
    ticks = np.arange(len(condition_labels), dtype=np.int64)
    for panel_idx, (ax, metric, matrix) in enumerate(zip(axes, ordered_metric_names, ordered_matrices, strict=True)):
        arr = np.asarray(matrix, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
        else:
            vmin, vmax = 0.0, 1.0
        if vmin == vmax:
            vmax = vmin + 1.0
        image = ax.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        images.append(image)
        ax.set_title(GROUND_TRUTH_RDM_METRIC_TITLES.get(str(metric), str(metric)))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(condition_labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("condition")
        if panel_idx == 0:
            ax.set_yticklabels(condition_labels, fontsize=8)
            ax.set_ylabel("condition")
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"MoG5 ground-truth RDMs ({ground_truth['pr_dim_label']}, n_total={int(ground_truth['n_total'])})",
        fontsize=13,
    )
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path, metadata={"Description": "metrics=" + ",".join(ordered_metric_names)})
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return svg_path, png_path


def plot_flow_loss_sweep(
    *,
    loss_data: dict[tuple[int, int, int, str], dict[str, Any]],
    n_list: list[int],
    pr_dim: int | None,
    metrics: tuple[str, ...],
    svg_path: Path,
    png_path: Path,
    yscale: str,
) -> tuple[Path, Path] | None:
    if not loss_data:
        return None

    n_set = {int(n) for n in n_list}
    metric_tuple = tuple(
        str(metric)
        for metric in metrics
        if any(
            int(key[0]) in n_set and int(key[1]) == _pr_dim_storage(pr_dim) and str(key[3]) == str(metric)
            for key in loss_data
        )
    )
    if not metric_tuple:
        return None

    fig_width = max(6.5, 5.2 * len(metric_tuple))
    fig, axes_obj = plt.subplots(1, len(metric_tuple), figsize=(fig_width, 4.9), squeeze=False, constrained_layout=True)
    axes = axes_obj[0]
    loss_styles = {
        "train": {"color": "tab:blue", "linestyle": "-", "label": "train"},
        "val": {"color": "tab:orange", "linestyle": "--", "label": "val"},
        "val_monitor": {"color": "tab:green", "linestyle": ":", "label": "val EMA"},
    }
    handles_by_label: dict[str, Any] = {}

    for ax, metric in zip(axes, metric_tuple):
        for n_total in n_list:
            items = [
                item
                for key, item in loss_data.items()
                if int(key[0]) == int(n_total) and int(key[1]) == _pr_dim_storage(pr_dim) and str(key[3]) == str(metric)
            ]
            if not items:
                continue

            def mean_curve(name: str) -> np.ndarray:
                curves = [np.asarray(item.get(name, []), dtype=np.float64).reshape(-1) for item in items]
                curves = [curve for curve in curves if curve.size > 0]
                if not curves:
                    return np.asarray([], dtype=np.float64)
                min_len = min(int(curve.size) for curve in curves)
                return np.mean(np.stack([curve[:min_len] for curve in curves], axis=0), axis=0)

            train_losses = mean_curve("train_losses")
            val_losses = mean_curve("val_losses")
            val_monitor_losses = mean_curve("val_monitor_losses")
            if train_losses.size == 0 and val_losses.size == 0 and val_monitor_losses.size == 0:
                continue
            if train_losses.size:
                style = loss_styles["train"]
                epochs = np.arange(1, train_losses.size + 1, dtype=np.int64)
                (line,) = ax.plot(
                    epochs,
                    train_losses,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.4,
                    label=f"n={int(n_total)} {style['label']}",
                )
                handles_by_label[line.get_label()] = line
            if val_losses.size:
                style = loss_styles["val"]
                epochs = np.arange(1, val_losses.size + 1, dtype=np.int64)
                (line,) = ax.plot(
                    epochs,
                    val_losses,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.4,
                    label=f"n={int(n_total)} {style['label']}",
                )
                handles_by_label[line.get_label()] = line
            if val_monitor_losses.size:
                style = loss_styles["val_monitor"]
                epochs = np.arange(1, val_monitor_losses.size + 1, dtype=np.int64)
                (line,) = ax.plot(
                    epochs,
                    val_monitor_losses,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.4,
                    label=f"n={int(n_total)} {style['label']}",
                )
                handles_by_label[line.get_label()] = line
        ax.set_title(str(metric))
        ax.set_xlabel("epoch")
        ax.set_ylabel("flow matching loss")
        ax.grid(True, which="both", alpha=0.25)
        if str(yscale) == "log":
            ax.set_yscale("log")
            positive = [
                float(value)
                for line in ax.lines
                for value in np.asarray(line.get_ydata(), dtype=np.float64)
                if np.isfinite(value) and value > 0.0
            ]
            if positive:
                ax.set_ylim(bottom=max(min(positive) * 0.5, 1e-12))

    if not handles_by_label:
        plt.close(fig)
        return None

    fig.legend(
        handles_by_label.values(),
        handles_by_label.keys(),
        loc="lower center",
        ncol=min(4, max(1, len(handles_by_label))),
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(f"MoG5 PR flow training loss sweep ({_pr_dim_label(pr_dim)})", fontsize=13)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return svg_path, png_path


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    case_paths: dict[tuple[int, int, int], Path],
    cache_hits: dict[tuple[int, int, int], bool],
    outputs: dict[str, Path],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = resolve_metric_names(args)
    payload = {
        "script": "bin/compare_mog5_pr_distance_sweeps.py",
        "config": {
            "n_list": [int(v) for v in args.n_list],
            "pr_dim": None if args.pr_dim is None else int(args.pr_dim),
            "pr_projected": _pr_projected(args.pr_dim),
            "pr_dim_label": _pr_dim_label(args.pr_dim),
            "native_x_dim": int(args.native_x_dim),
            "n_total": int(args.n_total),
            "n_repeats": _n_repeats(args),
            "seed": int(args.seed),
            "repeat_seeds": [_repeat_seed(args, r) for r in range(_n_repeats(args))],
            "device": str(args.device),
            "case_output_name": str(args.case_output_name),
            "force_comparison": bool(args.force_comparison),
            "visualization_only": bool(args.visualization_only),
            "metric": str(args.metric),
            "metrics": list(metrics),
            "yscale": str(args.yscale),
            "abs_error_yscale": str(args.yscale),
            "rel_error_yscale": "linear",
            "loss_yscale": str(args.loss_yscale),
        },
        "case_paths": {
            f"n{int(n_total)}_repeat{int(repeat_idx):02d}_{'native' if int(pr_dim) == -1 else f'pr{int(pr_dim)}'}": str(path)
            for (n_total, pr_dim, repeat_idx), path in sorted(case_paths.items())
        },
        "cache_hits": {
            f"n{int(n_total)}_repeat{int(repeat_idx):02d}_{'native' if int(pr_dim) == -1 else f'pr{int(pr_dim)}'}": bool(hit)
            for (n_total, pr_dim, repeat_idx), hit in sorted(cache_hits.items())
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> dict[str, Path]:
    validate_args(args)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = resolve_metric_names(args)
    ground_truth = compute_baseline_ground_truth_rdms(args, metrics)
    gt_svg_path, gt_png_path = plot_ground_truth_rdms(
        ground_truth,
        svg_path=output_dir / GROUND_TRUTH_RDMS_SVG_NAME,
        png_path=output_dir / GROUND_TRUTH_RDMS_PNG_NAME,
    )

    native_template_npz = Path(str(ground_truth["native_npz"]))
    case_paths: dict[tuple[int, int, int], Path] = {}
    cache_hits: dict[tuple[int, int, int], bool] = {}
    case_data: dict[tuple[int, int, int], dict[str, Any]] = {}
    for n_total, pr_dim, repeat_idx in _unique_cases(args):
        path, cache_hit = ensure_case_results(
            args,
            n_total=n_total,
            pr_dim=pr_dim,
            repeat_idx=int(repeat_idx),
            native_template_npz=native_template_npz,
        )
        case = _repeat_case_key(int(n_total), pr_dim, int(repeat_idx))
        case_paths[case] = Path(path)
        cache_hits[case] = bool(cache_hit)
        case_data[case] = _filter_case_metrics(_load_case_cache(Path(path)), metrics, path=Path(path))

    flow_loss_data: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    flow_loss_warnings: list[str] = []
    for n_total, pr_dim, repeat_idx in _unique_cases(args):
        for metric in metrics:
            loss_path = case_flow_loss_npz(
                n_total=int(n_total),
                pr_dim=pr_dim,
                case_output_name=str(args.case_output_name),
                metric=str(metric),
                native_x_dim=int(args.native_x_dim),
                repeat_idx=int(repeat_idx),
                n_repeats=_n_repeats(args),
            )
            try:
                key = _repeat_case_key(int(n_total), pr_dim, int(repeat_idx))
                flow_loss_data[(key[0], key[1], key[2], str(metric))] = _load_flow_loss_cache(loss_path)
            except (FileNotFoundError, KeyError, ValueError) as exc:
                flow_loss_warnings.append(str(exc))

    aggregate, rows = aggregate_sweeps(args=args, case_data=case_data)
    npz_path = write_aggregate_npz(output_dir / SWEEP_NPZ_NAME, aggregate)
    csv_path = write_errors_csv(output_dir / SWEEP_CSV_NAME, rows)
    svg_path, png_path = plot_abs_error(
        aggregate,
        svg_path=output_dir / SWEEP_SVG_NAME,
        png_path=output_dir / SWEEP_PNG_NAME,
        yscale=str(args.yscale),
    )
    rel_svg_path, rel_png_path = plot_sweep_error(
        aggregate,
        svg_path=output_dir / SWEEP_REL_SVG_NAME,
        png_path=output_dir / SWEEP_REL_PNG_NAME,
        yscale="linear",
        relative=True,
    )
    loss_paths = plot_flow_loss_sweep(
        loss_data=flow_loss_data,
        n_list=[int(v) for v in args.n_list],
        pr_dim=args.pr_dim,
        metrics=metrics,
        svg_path=output_dir / SWEEP_FLOW_LOSS_SVG_NAME,
        png_path=output_dir / SWEEP_FLOW_LOSS_PNG_NAME,
        yscale=str(args.loss_yscale),
    )
    dataset_paths = maybe_plot_representative_dataset(
        args=args,
        case_paths=case_paths,
        output_dir=output_dir,
    )
    outputs = {
        "results_npz": npz_path,
        "errors_csv": csv_path,
        "figure_svg": svg_path,
        "figure_png": png_path,
        "abs_error_figure_svg": svg_path,
        "abs_error_figure_png": png_path,
        "rel_error_figure_svg": rel_svg_path,
        "rel_error_figure_png": rel_png_path,
        "ground_truth_rdms_figure_svg": gt_svg_path,
        "ground_truth_rdms_figure_png": gt_png_path,
    }
    if loss_paths is not None:
        outputs["flow_loss_figure_svg"] = loss_paths[0]
        outputs["flow_loss_figure_png"] = loss_paths[1]
    if dataset_paths is not None:
        outputs["dataset_figure_svg"] = dataset_paths[0]
        outputs["dataset_figure_png"] = dataset_paths[1]
    summary_path = write_summary(
        output_dir / SWEEP_SUMMARY_NAME,
        args=args,
        case_paths=case_paths,
        cache_hits=cache_hits,
        outputs=outputs,
    )
    outputs["summary_json"] = summary_path
    print(f"results_npz: {npz_path}", flush=True)
    print(f"errors_csv: {csv_path}", flush=True)
    print(f"figure_svg: {svg_path}", flush=True)
    print(f"figure_png: {png_path}", flush=True)
    print(f"rel_error_figure_svg: {rel_svg_path}", flush=True)
    print(f"rel_error_figure_png: {rel_png_path}", flush=True)
    print(f"ground_truth_rdms_figure_svg: {gt_svg_path}", flush=True)
    print(f"ground_truth_rdms_figure_png: {gt_png_path}", flush=True)
    if loss_paths is None:
        if flow_loss_warnings:
            print(
                f"[sweep] warning: no usable flow loss histories found; skipped flow loss figure. First issue: {flow_loss_warnings[0]}",
                flush=True,
            )
        else:
            print("[sweep] warning: no usable flow loss histories found; skipped flow loss figure.", flush=True)
    else:
        print(f"flow_loss_figure_svg: {loss_paths[0]}", flush=True)
        print(f"flow_loss_figure_png: {loss_paths[1]}", flush=True)
        if flow_loss_warnings:
            print(
                f"[sweep] warning: skipped {len(flow_loss_warnings)} missing or incomplete flow loss cache(s). First issue: {flow_loss_warnings[0]}",
                flush=True,
            )
    if dataset_paths is not None:
        print(f"dataset_figure_svg: {dataset_paths[0]}", flush=True)
        print(f"dataset_figure_png: {dataset_paths[1]}", flush=True)
    print(f"summary_json: {summary_path}", flush=True)
    return outputs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
