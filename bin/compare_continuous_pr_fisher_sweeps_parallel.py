#!/usr/bin/env python3
"""Run continuous PR Fisher sample-size sweeps in parallel across CUDA devices."""

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.dataset_visualization import plot_joint_and_tuning, plot_tuning_and_covariance_on_axes
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_meta


def _load_single_module() -> Any:
    path = _REPO_ROOT / "bin" / "compare_continuous_pr_fisher.py"
    spec = importlib.util.spec_from_file_location("compare_continuous_pr_fisher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


single = _load_single_module()

SWEEP_NPZ_NAME = "continuous_pr_fisher_sweep_results.npz"
SWEEP_CSV_NAME = "continuous_pr_fisher_sweep_errors.csv"
SWEEP_SUMMARY_NAME = "continuous_pr_fisher_sweep_summary.json"
SWEEP_SVG_NAME = "continuous_pr_fisher_sweep_abs_error.svg"
SWEEP_PNG_NAME = "continuous_pr_fisher_sweep_abs_error.png"
DATASET_VIZ_PNG_NAME = "continuous_pr_fisher_representative_dataset.png"
DATASET_VIZ_SVG_NAME = "continuous_pr_fisher_representative_dataset.svg"
COMPOSITE_PNG_NAME = "continuous_pr_fisher_composite.png"
COMPOSITE_SVG_NAME = "continuous_pr_fisher_composite.svg"
COMPOSITE_FISHER_EXAMPLE_N_TARGET = 1550


METHOD_LABELS = {
    "ground_truth_native_full": "GT full",
    "ground_truth_native_linear": "GT linear",
    "classical_linear": "classical linear",
    "classical_full": "classical full",
    "flow_linear": "flow linear",
    "flow_full": "flow full",
}

METHOD_COLORS = {
    "ground_truth_native_linear": "black",
    "ground_truth_native_full": "black",
    "flow_linear": "C0",
    "flow_full": "C0",
    "classical_linear": "C1",
    "classical_full": "C1",
}

METHOD_LINESTYLES = {
    "ground_truth_native_linear": "--",
    "ground_truth_native_full": "--",
    "classical_linear": "-",
    "classical_full": "-",
    "flow_linear": "-",
    "flow_full": "-",
}

METHOD_MARKERS = {
    "ground_truth_native_linear": None,
    "ground_truth_native_full": None,
    "classical_linear": "s",
    "classical_full": "s",
    "flow_linear": "o",
    "flow_full": "o",
}


def _parse_int_list(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, (list, tuple)):
        vals = [int(v) for v in value]
    else:
        vals = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer.")
    return vals


def _parse_pr_dims(value: str | list[int | None] | tuple[int | None, ...]) -> list[int | None]:
    if isinstance(value, (list, tuple)):
        vals = [None if v is None else int(v) for v in value]
    else:
        vals = []
        for part in str(value).split(","):
            text = part.strip()
            if not text:
                continue
            vals.append(single.parse_pr_dim(text))
    if not vals:
        raise argparse.ArgumentTypeError("Expected comma-separated PR dims, e.g. none,30.")
    return vals


def _parse_gpu_ids(value: str | list[int] | tuple[int, ...]) -> list[int]:
    vals = _parse_int_list(value)
    if any(v < 0 for v in vals):
        raise argparse.ArgumentTypeError("CUDA device ids must be non-negative.")
    return vals


def _positive_int(value: str | int) -> int:
    out = int(value)
    if out < 1:
        raise argparse.ArgumentTypeError("Expected an integer >= 1.")
    return out


def _cpu_threads_auto(total_workers: int) -> int:
    return max(1, int(os.cpu_count() or 1) // max(1, int(total_workers)))


def _resolve_cpu_threads(value: str | int, total_workers: int) -> int:
    if str(value).strip().lower() == "auto":
        return _cpu_threads_auto(total_workers)
    return _positive_int(value)


def _pr_label(pr_dim: int | None) -> str:
    return "native" if pr_dim is None else f"pr{int(pr_dim)}"


def default_output_dir(*, dataset_family: str = "randamp_gaussian_sqrtd", native_x_dim: int = 3) -> Path:
    return _REPO_ROOT / "data" / f"{dataset_family}_xdim{int(native_x_dim)}_pr_fisher_sweeps"


def build_parser() -> argparse.ArgumentParser:
    p = single.build_parser()
    p.description = __doc__
    p.set_defaults(n_total=1000, output_dir=None, device="cuda")
    p.add_argument("--n-list", type=_parse_int_list, default=[50, 500, 1000, 1500, 2000, 3000])
    p.add_argument("--pr-dims", type=_parse_pr_dims, default=[None])
    p.add_argument("--n-repeats", type=_positive_int, default=1)
    p.add_argument("--case-output-name", type=str, default="continuous_pr_fisher")
    p.add_argument("--force-comparison", action="store_true")
    p.add_argument("--visualization-only", action="store_true")
    p.add_argument("--gpu-ids", type=_parse_gpu_ids, default=[0])
    p.add_argument("--jobs-per-gpu", type=_positive_int, default=1)
    p.add_argument("--cpu-threads-per-job", default="auto")
    p.add_argument("--parallel-log-dir", type=Path, default=None)
    p.add_argument("--yscale", choices=("linear", "log"), default="linear")
    p.add_argument("--skip-dataset-viz", action="store_true")
    original_parse_args = p.parse_args

    def parse_args(args=None, namespace=None):
        parsed = original_parse_args(args, namespace)
        parsed.n_list = _parse_int_list(parsed.n_list)
        parsed.pr_dims = _parse_pr_dims(parsed.pr_dims)
        parsed.gpu_ids = _parse_gpu_ids(parsed.gpu_ids)
        parsed.jobs_per_gpu = _positive_int(parsed.jobs_per_gpu)
        total_workers = len(parsed.gpu_ids) * int(parsed.jobs_per_gpu)
        parsed.cpu_threads_per_job = _resolve_cpu_threads(parsed.cpu_threads_per_job, total_workers)
        argv = sys.argv[1:] if args is None else list(args)
        output_was_explicit = any(str(arg) == "--output-dir" or str(arg).startswith("--output-dir=") for arg in argv)
        if parsed.output_dir is None or not output_was_explicit:
            parsed.output_dir = default_output_dir(
                dataset_family=str(parsed.dataset_family),
                native_x_dim=int(parsed.native_x_dim),
            )
        if parsed.parallel_log_dir is None:
            parsed.parallel_log_dir = Path(parsed.output_dir) / "parallel_logs"
        return parsed

    p.parse_args = parse_args  # type: ignore[method-assign]
    return p


@dataclass(frozen=True)
class CaseTask:
    n_total: int
    pr_dim: int | None
    repeat_idx: int
    seed: int
    dataset_dir: Path
    output_dir: Path
    result_path: Path

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.n_total, -1 if self.pr_dim is None else int(self.pr_dim), self.repeat_idx)

    @property
    def label(self) -> str:
        return f"n{self.n_total}_repeat{self.repeat_idx:02d}_{_pr_label(self.pr_dim)}"


@dataclass
class RunningCase:
    task: CaseTask
    gpu_id: int
    process: subprocess.Popen
    stdout_path: Path
    stderr_path: Path


def case_dataset_dir(args: argparse.Namespace, *, n_total: int, pr_dim: int | None, repeat_idx: int) -> Path:
    base = Path(args.output_dir) / f"n{int(n_total)}" / _pr_label(pr_dim)
    if int(args.n_repeats) > 1:
        base = base / f"repeat_{int(repeat_idx):02d}"
    return base


def case_output_dir(args: argparse.Namespace, *, n_total: int, pr_dim: int | None, repeat_idx: int) -> Path:
    return case_dataset_dir(args, n_total=n_total, pr_dim=pr_dim, repeat_idx=repeat_idx) / str(args.case_output_name)


def plan_cases(args: argparse.Namespace) -> list[CaseTask]:
    tasks: list[CaseTask] = []
    for n_total in args.n_list:
        for pr_dim in args.pr_dims:
            for repeat_idx in range(int(args.n_repeats)):
                output_dir = case_output_dir(args, n_total=int(n_total), pr_dim=pr_dim, repeat_idx=repeat_idx)
                tasks.append(
                    CaseTask(
                        n_total=int(n_total),
                        pr_dim=pr_dim,
                        repeat_idx=int(repeat_idx),
                        seed=int(args.seed) + int(repeat_idx),
                        dataset_dir=case_dataset_dir(args, n_total=int(n_total), pr_dim=pr_dim, repeat_idx=repeat_idx),
                        output_dir=output_dir,
                        result_path=output_dir / single.RESULTS_NPZ_NAME,
                    )
                )
    return tasks


def _arg_to_cli_name(dest: str) -> str:
    return "--" + str(dest).replace("_", "-")


def _stringify_value(value: Any) -> str:
    if value is None:
        raise ValueError("Cannot stringify None.")
    return str(value)


def build_case_command(args: argparse.Namespace, task: CaseTask) -> list[str]:
    case_args = argparse.Namespace(**vars(args))
    case_args.n_total = int(task.n_total)
    case_args.pr_dim = task.pr_dim
    case_args.seed = int(task.seed)
    case_args.dataset_dir = task.dataset_dir
    case_args.output_dir = task.output_dir
    case_args.device = "cuda"
    command = [sys.executable, str(_REPO_ROOT / "bin" / "compare_continuous_pr_fisher.py")]
    parser = single.build_parser()
    skip = {
        "help",
        "n_list",
        "pr_dims",
        "n_repeats",
        "case_output_name",
        "force_comparison",
        "visualization_only",
        "gpu_ids",
        "jobs_per_gpu",
        "cpu_threads_per_job",
        "parallel_log_dir",
        "yscale",
        "skip_dataset_viz",
    }
    for action in parser._actions:
        dest = getattr(action, "dest", None)
        if not dest or dest in skip or not hasattr(case_args, dest):
            continue
        value = getattr(case_args, dest)
        if value is None:
            continue
        option = _arg_to_cli_name(dest)
        if isinstance(action, argparse._StoreTrueAction):
            if bool(value):
                command.append(option)
            continue
        if isinstance(action, argparse._StoreFalseAction):
            if not bool(value):
                command.append(option)
            continue
        if dest == "pr_dim":
            value = "none" if value is None else str(int(value))
        command.extend([option, _stringify_value(value)])
    return command


def build_case_env(base_env: dict[str, str], *, gpu_id: int, cpu_threads_per_job: int) -> dict[str, str]:
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    env["PYTHONUNBUFFERED"] = "1"
    threads = str(int(cpu_threads_per_job))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = threads
    return env


def _cache_is_usable(path: Path) -> bool:
    if not Path(path).is_file():
        return False
    with np.load(path, allow_pickle=False) as data:
        return "theta_midpoints" in data.files and "flow_full_fisher" in data.files


def select_tasks_to_run(tasks: list[CaseTask], args: argparse.Namespace) -> tuple[list[CaseTask], dict[tuple[int, int, int], bool]]:
    to_run: list[CaseTask] = []
    hits: dict[tuple[int, int, int], bool] = {}
    for task in tasks:
        hit = (not bool(args.force_comparison)) and _cache_is_usable(task.result_path)
        if hit:
            print(f"[continuous-parallel] cache hit {task.label}: {task.result_path}", flush=True)
            hits[task.key] = True
        elif bool(args.visualization_only):
            raise FileNotFoundError(f"--visualization-only requires cached result: {task.result_path}")
        else:
            hits[task.key] = False
            to_run.append(task)
    return to_run, hits


def run_cases_parallel(tasks: list[CaseTask], args: argparse.Namespace) -> None:
    if not tasks:
        return
    log_dir = Path(args.parallel_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    slots = [gpu for gpu in args.gpu_ids for _ in range(int(args.jobs_per_gpu))]
    pending = list(tasks)
    running: list[RunningCase] = []
    base_env = os.environ.copy()
    while pending or running:
        while pending and len(running) < len(slots):
            used = {rc.gpu_id: 0 for rc in running}
            for rc in running:
                used[rc.gpu_id] = used.get(rc.gpu_id, 0) + 1
            gpu_id = min(slots, key=lambda g: used.get(g, 0))
            task = pending.pop(0)
            stdout_path = log_dir / f"{task.label}.stdout.log"
            stderr_path = log_dir / f"{task.label}.stderr.log"
            cmd = build_case_command(args, task)
            env = build_case_env(base_env, gpu_id=gpu_id, cpu_threads_per_job=int(args.cpu_threads_per_job))
            print(f"[continuous-parallel] launch {task.label} gpu={gpu_id}", flush=True)
            stdout_f = stdout_path.open("w")
            stderr_f = stderr_path.open("w")
            proc = subprocess.Popen(cmd, cwd=str(_REPO_ROOT), env=env, stdout=stdout_f, stderr=stderr_f)
            stdout_f.close()
            stderr_f.close()
            running.append(RunningCase(task=task, gpu_id=int(gpu_id), process=proc, stdout_path=stdout_path, stderr_path=stderr_path))
        time.sleep(2.0)
        still: list[RunningCase] = []
        for rc in running:
            code = rc.process.poll()
            if code is None:
                still.append(rc)
                continue
            if int(code) != 0:
                raise RuntimeError(
                    f"Case {rc.task.label} failed with exit code {code}. "
                    f"stdout={rc.stdout_path} stderr={rc.stderr_path}"
                )
            print(f"[continuous-parallel] done {rc.task.label}", flush=True)
        running = still


def representative_native_npz(tasks: list[CaseTask], args: argparse.Namespace) -> Path:
    n_total = max(int(v) for v in args.n_list)
    repeat_idx = 0
    native_tasks = [
        task
        for task in tasks
        if task.n_total == n_total and task.repeat_idx == repeat_idx and task.pr_dim is None
    ]
    if native_tasks:
        task = native_tasks[0]
    else:
        candidates = [task for task in tasks if task.n_total == n_total and task.repeat_idx == repeat_idx]
        if not candidates:
            raise ValueError("No representative continuous sweep case is available.")
        task = candidates[0]
    return task.dataset_dir / f"{args.dataset_family}_xdim{int(args.native_x_dim)}_native.npz"


def plot_representative_dataset(
    native_npz: Path,
    *,
    output_dir: Path,
    scatter_max_points: int = 500,
) -> tuple[Path, Path]:
    bundle = load_shared_dataset_npz(native_npz)
    dataset = build_dataset_from_meta(dict(bundle.meta))
    png_path = Path(output_dir) / DATASET_VIZ_PNG_NAME
    svg_path = Path(output_dir) / DATASET_VIZ_SVG_NAME
    plot_joint_and_tuning(
        bundle.theta_all,
        bundle.x_all,
        dataset,
        str(png_path),
        scatter_max_points=int(scatter_max_points),
    )
    return svg_path, png_path


def maybe_plot_representative_dataset(
    tasks: list[CaseTask],
    args: argparse.Namespace,
    *,
    output_dir: Path,
) -> tuple[Path, Path] | None:
    if bool(getattr(args, "skip_dataset_viz", False)):
        return None
    native_npz = representative_native_npz(tasks, args)
    if not native_npz.is_file():
        print(
            f"[continuous-parallel] warning: skipped representative dataset figure; missing NPZ: {native_npz}",
            flush=True,
        )
        return None
    return plot_representative_dataset(native_npz, output_dir=output_dir, scatter_max_points=500)


def representative_native_task(tasks: list[CaseTask], args: argparse.Namespace) -> CaseTask:
    n_total = max(int(v) for v in args.n_list)
    repeat_idx = 0
    native_tasks = [
        task
        for task in tasks
        if task.n_total == n_total and task.repeat_idx == repeat_idx and task.pr_dim is None
    ]
    if native_tasks:
        return native_tasks[0]
    candidates = [task for task in tasks if task.n_total == n_total and task.repeat_idx == repeat_idx]
    if not candidates:
        raise ValueError("No representative continuous sweep case is available.")
    return candidates[0]


def composite_fisher_example_task(tasks: list[CaseTask], args: argparse.Namespace) -> CaseTask:
    native_tasks = [
        task
        for task in tasks
        if task.repeat_idx == 0 and task.pr_dim is None and Path(task.result_path).is_file()
    ]
    if native_tasks:
        return min(
            native_tasks,
            key=lambda task: (abs(int(task.n_total) - COMPOSITE_FISHER_EXAMPLE_N_TARGET), int(task.n_total)),
        )
    return representative_native_task(tasks, args)


def _method_label(name: str) -> str:
    return METHOD_LABELS.get(str(name), str(name).replace("_", " "))


def _fisher_family_methods(family: str) -> tuple[str, str, str]:
    fam = str(family)
    if fam == "full":
        return ("ground_truth_native_full", "flow_full", "classical_full")
    if fam == "linear":
        return ("ground_truth_native_linear", "flow_linear", "classical_linear")
    raise ValueError(f"Unknown Fisher family: {family!r}.")


def _plot_composite_fisher_curves(ax: Any, result_path: Path, *, family: str) -> tuple[float, float]:
    y_vals: list[np.ndarray] = []
    methods = _fisher_family_methods(str(family))
    with np.load(result_path, allow_pickle=False) as data:
        mids = np.asarray(data["theta_midpoints"], dtype=np.float64).reshape(-1)
        for name in methods:
            key = f"{name}_fisher"
            if key in data.files:
                vals = np.asarray(data[key], dtype=np.float64).reshape(-1)
                y_vals.append(vals[np.isfinite(vals)])
                ax.plot(
                    mids,
                    vals,
                    color=METHOD_COLORS.get(name),
                    linewidth=2.2 if name.startswith("ground_truth") else 1.7,
                    linestyle=METHOD_LINESTYLES.get(name, "-"),
                    marker=METHOD_MARKERS.get(name),
                    markevery=4,
                    markersize=4,
                    label=_method_label(name),
                )
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Fisher information")
    ax.legend(fontsize=7)
    if not y_vals:
        return 0.0, 1.0
    merged = np.concatenate(y_vals)
    if merged.size == 0:
        return 0.0, 1.0
    return float(np.nanmin(merged)), float(np.nanmax(merged))


def _plot_composite_error_vs_n(ax: Any, rows: list[dict[str, Any]], *, family: str, yscale: str) -> None:
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        if str(row["pr_dim"]) != "none":
            continue
        grouped.setdefault(str(row["method"]), []).append((int(row["n_total"]), float(row["mae_abs_error"])))
    _, flow_method, classical_method = _fisher_family_methods(str(family))
    for method in (flow_method, classical_method):
        vals = grouped.get(method, [])
        if not vals:
            continue
        ns = sorted({n for n, _ in vals})
        means = [float(np.mean([err for n, err in vals if n == nn])) for nn in ns]
        ax.plot(
            ns,
            means,
            color=METHOD_COLORS.get(method),
            linestyle=METHOD_LINESTYLES.get(method, "-"),
            marker=METHOD_MARKERS.get(method) or "o",
            linewidth=1.6,
            label=_method_label(method),
        )
    ax.set_xlabel("number of data points")
    ax.set_ylabel("mean absolute Fisher error")
    ax.set_yscale(str(yscale))
    ax.legend(fontsize=7)


def plot_composite_figure(
    *,
    native_npz: Path,
    representative_result_path: Path,
    representative_n_total: int | None = None,
    rows: list[dict[str, Any]],
    output_dir: Path,
    yscale: str = "linear",
) -> tuple[Path, Path]:
    bundle = load_shared_dataset_npz(native_npz)
    dataset = build_dataset_from_meta(dict(bundle.meta))
    fig, axes = plt.subplots(3, 2, figsize=(12.2, 12.2), layout="constrained")
    ax_tune, ax_cov = axes[0]
    ax_full_fisher, ax_linear_fisher = axes[1]
    ax_full_error, ax_linear_error = axes[2]
    plot_tuning_and_covariance_on_axes(fig, ax_tune, ax_cov, dataset)
    ax_tune.set_title("Tuning curves")
    ax_cov.set_title("PCA covariance ellipses")
    full_ylim = _plot_composite_fisher_curves(ax_full_fisher, representative_result_path, family="full")
    linear_ylim = _plot_composite_fisher_curves(ax_linear_fisher, representative_result_path, family="linear")
    y_min = min(0.0, full_ylim[0], linear_ylim[0])
    y_max = max(full_ylim[1], linear_ylim[1])
    if not np.isfinite(y_max) or y_max <= y_min:
        y_max = y_min + 1.0
    pad = 0.05 * max(y_max - y_min, 1e-12)
    ax_full_fisher.set_ylim(y_min, y_max + pad)
    ax_linear_fisher.set_ylim(y_min, y_max + pad)
    if representative_n_total is None:
        ax_full_fisher.set_title(r"Full Fisher vs. $\theta$")
        ax_linear_fisher.set_title(r"Linear Fisher vs. $\theta$")
    else:
        ax_full_fisher.set_title(rf"Full Fisher vs. $\theta$ ($n={int(representative_n_total)}$)")
        ax_linear_fisher.set_title(rf"Linear Fisher vs. $\theta$ ($n={int(representative_n_total)}$)")
    _plot_composite_error_vs_n(ax_full_error, rows, family="full", yscale=str(yscale))
    _plot_composite_error_vs_n(ax_linear_error, rows, family="linear", yscale=str(yscale))
    ax_full_error.set_title("Full Fisher error vs. data size")
    ax_linear_error.set_title("Linear Fisher error vs. data size")
    svg = Path(output_dir) / COMPOSITE_SVG_NAME
    png = Path(output_dir) / COMPOSITE_PNG_NAME
    svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg)
    fig.savefig(png, dpi=180)
    plt.close(fig)
    return svg, png


def maybe_plot_composite_figure(
    tasks: list[CaseTask],
    args: argparse.Namespace,
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> tuple[Path, Path] | None:
    if bool(getattr(args, "skip_dataset_viz", False)):
        return None
    task = composite_fisher_example_task(tasks, args)
    native_npz = representative_native_npz(tasks, args)
    if not native_npz.is_file() or not task.result_path.is_file():
        print(
            "[continuous-parallel] warning: skipped composite figure; "
            f"missing representative inputs: native_npz={native_npz} result={task.result_path}",
            flush=True,
        )
        return None
    return plot_composite_figure(
        native_npz=native_npz,
        representative_result_path=task.result_path,
        representative_n_total=int(task.n_total),
        rows=rows,
        output_dir=output_dir,
        yscale=str(args.yscale),
    )


def aggregate_results(
    tasks: list[CaseTask],
    output_dir: Path,
    *,
    args: argparse.Namespace,
    dataset_paths: tuple[Path, Path] | None = None,
) -> tuple[Path, Path, Path, Path, Path, tuple[Path, Path] | None]:
    rows: list[dict[str, Any]] = []
    n_values: list[int] = []
    pr_values: list[int] = []
    repeat_values: list[int] = []
    methods: set[str] = set()
    abs_errors: dict[tuple[int, int, int, str], float] = {}
    for task in tasks:
        with np.load(task.result_path, allow_pickle=False) as data:
            for key in data.files:
                if not key.endswith("_abs_error"):
                    continue
                method = key[: -len("_abs_error")]
                vals = np.asarray(data[key], dtype=np.float64).reshape(-1)
                mae = float(np.nanmean(vals))
                methods.add(method)
                pr_store = -1 if task.pr_dim is None else int(task.pr_dim)
                abs_errors[(task.n_total, pr_store, task.repeat_idx, method)] = mae
                rows.append(
                    {
                        "n_total": task.n_total,
                        "pr_dim": "none" if task.pr_dim is None else int(task.pr_dim),
                        "repeat_idx": task.repeat_idx,
                        "method": method,
                        "mae_abs_error": mae,
                        "result_path": str(task.result_path),
                    }
                )
                n_values.append(task.n_total)
                pr_values.append(pr_store)
                repeat_values.append(task.repeat_idx)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / SWEEP_CSV_NAME
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("n_total", "pr_dim", "repeat_idx", "method", "mae_abs_error", "result_path"))
        writer.writeheader()
        writer.writerows(rows)
    npz_path = out / SWEEP_NPZ_NAME
    np.savez_compressed(
        npz_path,
        n_total=np.asarray(n_values, dtype=np.int64),
        pr_dim=np.asarray(pr_values, dtype=np.int64),
        repeat_idx=np.asarray(repeat_values, dtype=np.int64),
        methods=np.asarray(sorted(methods), dtype=str),
        mae_abs_error=np.asarray([r["mae_abs_error"] for r in rows], dtype=np.float64),
    )
    svg_path, png_path = plot_sweep_errors(rows, out, yscale=str(args.yscale))
    composite_paths = maybe_plot_composite_figure(tasks, args, output_dir=out, rows=rows)
    outputs: dict[str, str] = {
        "sweep_npz": str(npz_path),
        "sweep_csv": str(csv_path),
        "sweep_svg": str(svg_path),
        "sweep_png": str(png_path),
    }
    if dataset_paths is not None:
        outputs["dataset_figure_svg"] = str(dataset_paths[0])
        outputs["dataset_figure_png"] = str(dataset_paths[1])
    if composite_paths is not None:
        outputs["composite_svg"] = str(composite_paths[0])
        outputs["composite_png"] = str(composite_paths[1])
    summary_path = out / SWEEP_SUMMARY_NAME
    summary_path.write_text(
        json.dumps(
            {
                "rows": len(rows),
                "methods": sorted(methods),
                "csv": str(csv_path),
                "npz": str(npz_path),
                "config": {
                    "n_list": [int(v) for v in args.n_list],
                    "pr_dims": [None if v is None else int(v) for v in args.pr_dims],
                    "n_repeats": int(args.n_repeats),
                    "dataset_family": str(args.dataset_family),
                    "native_x_dim": int(args.native_x_dim),
                    "skip_dataset_viz": bool(getattr(args, "skip_dataset_viz", False)),
                },
                "outputs": outputs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return npz_path, csv_path, summary_path, svg_path, png_path, composite_paths


def plot_sweep_errors(rows: list[dict[str, Any]], output_dir: Path, *, yscale: str = "linear") -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(8.0, 5.0), layout="constrained")
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), str(row["pr_dim"])), []).append((int(row["n_total"]), float(row["mae_abs_error"])))
    for (method, pr_dim), vals in sorted(grouped.items()):
        ns = sorted({v[0] for v in vals})
        means = [float(np.mean([e for n, e in vals if n == nn])) for nn in ns]
        ax.plot(ns, means, marker="o", label=f"{method} {pr_dim}")
    ax.set_xlabel("n_total")
    ax.set_ylabel("mean absolute Fisher error")
    ax.set_yscale(str(yscale))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    svg = Path(output_dir) / SWEEP_SVG_NAME
    png = Path(output_dir) / SWEEP_PNG_NAME
    fig.savefig(svg)
    fig.savefig(png, dpi=180)
    plt.close(fig)
    return svg, png


def run(args: argparse.Namespace) -> dict[str, Path]:
    tasks = plan_cases(args)
    to_run, _ = select_tasks_to_run(tasks, args)
    run_cases_parallel(to_run, args)
    output_dir = Path(args.output_dir)
    dataset_paths = maybe_plot_representative_dataset(tasks, args, output_dir=output_dir)
    npz_path, csv_path, summary_path, svg_path, png_path, composite_paths = aggregate_results(
        tasks,
        output_dir,
        args=args,
        dataset_paths=dataset_paths,
    )
    print(f"sweep_npz: {npz_path}", flush=True)
    print(f"sweep_csv: {csv_path}", flush=True)
    print(f"sweep_svg: {svg_path}", flush=True)
    print(f"sweep_png: {png_path}", flush=True)
    if dataset_paths is not None:
        print(f"dataset_figure_svg: {dataset_paths[0]}", flush=True)
        print(f"dataset_figure_png: {dataset_paths[1]}", flush=True)
    if composite_paths is not None:
        print(f"composite_svg: {composite_paths[0]}", flush=True)
        print(f"composite_png: {composite_paths[1]}", flush=True)
    print(f"summary_json: {summary_path}", flush=True)
    outputs = {
        "sweep_npz": npz_path,
        "sweep_csv": csv_path,
        "summary_json": summary_path,
        "sweep_svg": svg_path,
        "sweep_png": png_path,
    }
    if dataset_paths is not None:
        outputs["dataset_figure_svg"] = dataset_paths[0]
        outputs["dataset_figure_png"] = dataset_paths[1]
    if composite_paths is not None:
        outputs["composite_svg"] = composite_paths[0]
        outputs["composite_png"] = composite_paths[1]
    return outputs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
