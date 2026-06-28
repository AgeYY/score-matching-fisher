#!/usr/bin/env python3
"""Run MoG5 PR distance sample-size sweeps in parallel across CUDA devices."""

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_sweep_module() -> Any:
    path = _REPO_ROOT / "bin" / "compare_mog5_pr_distance_sweeps.py"
    spec = importlib.util.spec_from_file_location("compare_mog5_pr_distance_sweeps", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sweep = _load_sweep_module()


def _parse_gpu_ids(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, (list, tuple)):
        vals = [int(v) for v in value]
    else:
        vals = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated CUDA device id.")
    if any(v < 0 for v in vals):
        raise argparse.ArgumentTypeError("CUDA device ids must be non-negative integers.")
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


def build_parser() -> argparse.ArgumentParser:
    p = sweep.build_parser()
    p.description = __doc__
    p.set_defaults(device="cuda")
    p.add_argument(
        "--gpu-ids",
        type=_parse_gpu_ids,
        default=[0, 1],
        help="Comma-separated CUDA device ids used for parallel case workers.",
    )
    p.add_argument(
        "--jobs-per-gpu",
        type=_positive_int,
        default=1,
        help="Concurrent comparison subprocesses per CUDA device.",
    )
    p.add_argument(
        "--cpu-threads-per-job",
        default="auto",
        help="CPU thread count per child process, or 'auto' for os.cpu_count() // total_workers.",
    )
    p.add_argument(
        "--parallel-log-dir",
        type=Path,
        default=None,
        help="Directory for per-case stdout/stderr logs. Defaults to <output-dir>/parallel_logs.",
    )
    original_parse_args = p.parse_args

    def parse_args(args=None, namespace=None):
        parsed = original_parse_args(args, namespace)
        parsed.gpu_ids = _parse_gpu_ids(parsed.gpu_ids)
        parsed.jobs_per_gpu = _positive_int(parsed.jobs_per_gpu)
        total_workers = len(parsed.gpu_ids) * int(parsed.jobs_per_gpu)
        parsed.cpu_threads_per_job = _resolve_cpu_threads(parsed.cpu_threads_per_job, total_workers)
        if parsed.parallel_log_dir is None:
            parsed.parallel_log_dir = Path(parsed.output_dir).expanduser() / "parallel_logs"
        else:
            parsed.parallel_log_dir = Path(parsed.parallel_log_dir).expanduser()
        return parsed

    p.parse_args = parse_args  # type: ignore[method-assign]
    return p


@dataclass(frozen=True)
class CaseTask:
    n_total: int
    pr_dim: int | None
    repeat_idx: int
    result_path: Path
    output_dir: Path
    dataset_dir: Path
    seed: int

    @property
    def key(self) -> tuple[int, int, int]:
        return sweep._repeat_case_key(self.n_total, self.pr_dim, self.repeat_idx)

    @property
    def label(self) -> str:
        return f"n{self.n_total}_repeat{self.repeat_idx:02d}_{sweep._pr_dim_label(self.pr_dim)}"


@dataclass
class RunningCase:
    task: CaseTask
    gpu_id: int
    process: Any
    stdout_path: Path
    stderr_path: Path


def plan_cases(args: argparse.Namespace) -> list[CaseTask]:
    tasks: list[CaseTask] = []
    for n_total, pr_dim, repeat_idx in sweep._unique_cases(args):
        output_dir = sweep.case_output_dir(
            n_total=int(n_total),
            pr_dim=pr_dim,
            case_output_name=str(args.case_output_name),
            native_x_dim=int(args.native_x_dim),
            repeat_idx=int(repeat_idx),
            n_repeats=sweep._n_repeats(args),
        )
        tasks.append(
            CaseTask(
                n_total=int(n_total),
                pr_dim=pr_dim,
                repeat_idx=int(repeat_idx),
                result_path=output_dir / sweep.RESULTS_NAME,
                output_dir=output_dir,
                dataset_dir=output_dir.parent,
                seed=sweep._repeat_seed(args, int(repeat_idx)),
            )
        )
    return tasks


def _cache_is_usable(path: Path, metrics: tuple[str, ...]) -> bool:
    if not Path(path).is_file():
        return False
    sweep._filter_case_metrics(sweep._load_case_cache(Path(path)), metrics, path=Path(path))
    return True


def preflight_visualization_only(tasks: list[CaseTask], metrics: tuple[str, ...]) -> None:
    missing: list[str] = []
    for task in tasks:
        try:
            if not _cache_is_usable(task.result_path, metrics):
                missing.append(f"{task.label}: {task.result_path}")
        except (FileNotFoundError, KeyError, ValueError) as exc:
            missing.append(f"{task.label}: {task.result_path} ({exc})")
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"--visualization-only requires cached results for every repeat:\n{joined}")


def select_tasks_to_run(
    tasks: list[CaseTask],
    *,
    args: argparse.Namespace,
    metrics: tuple[str, ...],
) -> tuple[list[CaseTask], dict[tuple[int, int, int], bool]]:
    to_run: list[CaseTask] = []
    cache_hits: dict[tuple[int, int, int], bool] = {}
    for task in tasks:
        if not bool(args.force_comparison):
            try:
                if _cache_is_usable(task.result_path, metrics):
                    print(f"[parallel-sweep] cache hit {task.label}: {task.result_path}", flush=True)
                    cache_hits[task.key] = True
                    continue
            except ValueError:
                if bool(args.visualization_only):
                    raise
                print(f"[parallel-sweep] cache missing requested metrics; rerunning {task.label}", flush=True)
        if bool(args.visualization_only):
            raise FileNotFoundError(f"--visualization-only requires cached results: {task.result_path}")
        cache_hits[task.key] = False
        to_run.append(task)
    return to_run, cache_hits


def _arg_to_cli_name(dest: str) -> str:
    special = {
        "optimizer_name": "--flow-optimizer",
        "lr_scheduler": "--flow-lr-scheduler",
    }
    if str(dest) in special:
        return special[str(dest)]
    return "--" + str(dest).replace("_", "-")


def _stringify_value(value: Any) -> str:
    if value is None:
        raise ValueError("Cannot stringify None.")
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_case_command(
    args: argparse.Namespace,
    task: CaseTask,
    *,
    native_template_npz: Path | None,
) -> list[str]:
    single = sweep._load_single_case_module()
    case_args = sweep._single_case_args(
        args,
        n_total=task.n_total,
        pr_dim=task.pr_dim,
        output_dir=task.output_dir,
        repeat_idx=task.repeat_idx,
        native_template_npz=native_template_npz,
    )
    case_args.device = "cuda"
    command = [sys.executable, str(_REPO_ROOT / "bin" / "compare_mog5_pr_distances.py")]
    parser = single.build_parser()
    for action in parser._actions:
        dest = getattr(action, "dest", None)
        if not dest or dest == "help" or not hasattr(case_args, dest):
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


def _launch_case_process(
    command: list[str],
    *,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.Popen:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_f = stdout_path.open("w", encoding="utf-8")
    stderr_f = stderr_path.open("w", encoding="utf-8")
    try:
        return subprocess.Popen(command, cwd=_REPO_ROOT, env=env, stdout=stdout_f, stderr=stderr_f)
    except Exception:
        stdout_f.close()
        stderr_f.close()
        raise


def run_parallel_cases(
    args: argparse.Namespace,
    tasks: list[CaseTask],
    *,
    native_template_npz: Path | None,
) -> dict[tuple[int, int, int], dict[str, Any]]:
    if not tasks:
        return {}
    log_dir = Path(args.parallel_log_dir).expanduser()
    total_workers = len(args.gpu_ids) * int(args.jobs_per_gpu)
    free_slots = [int(gpu_id) for gpu_id in args.gpu_ids for _ in range(int(args.jobs_per_gpu))]
    pending = list(tasks)
    running: list[RunningCase] = []
    failures: list[RunningCase] = []
    logs: dict[tuple[int, int, int], dict[str, Any]] = {}
    stop_scheduling = False

    while pending or running:
        while pending and free_slots and not stop_scheduling:
            task = pending.pop(0)
            gpu_id = free_slots.pop(0)
            stdout_path = log_dir / f"{task.label}.stdout.log"
            stderr_path = log_dir / f"{task.label}.stderr.log"
            command = build_case_command(args, task, native_template_npz=native_template_npz)
            env = build_case_env(os.environ, gpu_id=gpu_id, cpu_threads_per_job=int(args.cpu_threads_per_job))
            print(f"[parallel-sweep] launching {task.label} on CUDA_VISIBLE_DEVICES={gpu_id}", flush=True)
            proc = _launch_case_process(command, env=env, stdout_path=stdout_path, stderr_path=stderr_path)
            running.append(
                RunningCase(
                    task=task,
                    gpu_id=gpu_id,
                    process=proc,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
            )
            logs[task.key] = {
                "gpu_id": int(gpu_id),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "command": command,
            }

        if not running:
            break

        progressed = False
        for item in list(running):
            rc = item.process.poll()
            if rc is None:
                continue
            progressed = True
            running.remove(item)
            free_slots.append(int(item.gpu_id))
            logs[item.task.key]["returncode"] = int(rc)
            if int(rc) != 0:
                failures.append(item)
                stop_scheduling = True
                print(f"[parallel-sweep] failed {item.task.label}; waiting for running cases", flush=True)
            else:
                print(f"[parallel-sweep] finished {item.task.label}", flush=True)

        if not progressed and running:
            time.sleep(1.0)

        if stop_scheduling and not running:
            break

    if failures:
        details = "\n".join(
            f"{item.task.label} rc={logs[item.task.key].get('returncode')} "
            f"stdout={item.stdout_path} stderr={item.stderr_path}"
            for item in failures
        )
        raise RuntimeError(f"Parallel comparison case(s) failed:\n{details}")

    return logs


def run(args: argparse.Namespace) -> dict[str, Path]:
    sweep.validate_args(args)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.parallel_log_dir).mkdir(parents=True, exist_ok=True)

    metrics = sweep.resolve_metric_names(args)
    tasks = plan_cases(args)
    if bool(args.visualization_only):
        preflight_visualization_only(tasks, metrics)

    ground_truth = sweep.compute_baseline_ground_truth_rdms(args, metrics)
    gt_svg_path, gt_png_path = sweep.plot_ground_truth_rdms(
        ground_truth,
        svg_path=output_dir / sweep.GROUND_TRUTH_RDMS_SVG_NAME,
        png_path=output_dir / sweep.GROUND_TRUTH_RDMS_PNG_NAME,
    )
    native_template_npz = Path(str(ground_truth["native_npz"]))

    to_run, cache_hits = select_tasks_to_run(tasks, args=args, metrics=metrics)
    worker_logs = run_parallel_cases(args, to_run, native_template_npz=native_template_npz)

    case_paths: dict[tuple[int, int, int], Path] = {}
    case_data: dict[tuple[int, int, int], dict[str, Any]] = {}
    for task in tasks:
        case_paths[task.key] = task.result_path
        case_data[task.key] = sweep._filter_case_metrics(
            sweep._load_case_cache(task.result_path),
            metrics,
            path=task.result_path,
        )

    worker_count = len(args.gpu_ids) * int(args.jobs_per_gpu)
    return sweep.finalize_sweep_outputs(
        args=args,
        output_dir=output_dir,
        metrics=metrics,
        case_paths=case_paths,
        cache_hits=cache_hits,
        case_data=case_data,
        gt_svg_path=gt_svg_path,
        gt_png_path=gt_png_path,
        summary_script="bin/compare_mog5_pr_distance_sweeps_parallel.py",
        summary_extra_config={
            "gpu_ids": [int(v) for v in args.gpu_ids],
            "jobs_per_gpu": int(args.jobs_per_gpu),
            "worker_count": int(worker_count),
            "cpu_threads_per_job": int(args.cpu_threads_per_job),
            "parallel_log_dir": str(Path(args.parallel_log_dir)),
        },
        summary_extra_payload={
            "parallel_case_logs": {
                f"n{int(k[0])}_repeat{int(k[2]):02d}_{'native' if int(k[1]) == -1 else f'pr{int(k[1])}'}": v
                for k, v in sorted(worker_logs.items())
            }
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
