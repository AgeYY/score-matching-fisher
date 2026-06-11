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


def aggregate_results(tasks: list[CaseTask], output_dir: Path) -> tuple[Path, Path, Path, Path]:
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
    summary_path = out / SWEEP_SUMMARY_NAME
    summary_path.write_text(
        json.dumps(
            {
                "rows": len(rows),
                "methods": sorted(methods),
                "csv": str(csv_path),
                "npz": str(npz_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    svg_path, png_path = plot_sweep_errors(rows, out)
    return npz_path, csv_path, summary_path, svg_path


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
    npz_path, csv_path, summary_path, svg_path = aggregate_results(tasks, Path(args.output_dir))
    print(f"sweep_npz: {npz_path}", flush=True)
    print(f"sweep_csv: {csv_path}", flush=True)
    print(f"summary_json: {summary_path}", flush=True)
    return {"sweep_npz": npz_path, "sweep_csv": csv_path, "summary_json": summary_path, "sweep_svg": svg_path}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
