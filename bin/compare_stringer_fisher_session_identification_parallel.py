#!/usr/bin/env python3
"""Parallel cache builder for Stringer session-identification Fisher curves."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.shared_fisher_est import require_device
from fisher.stringer_dataset import list_stringer_sessions
from fisher.stringer_session_identification import (
    ASubsampleCurveTask,
    estimate_a_subsample_curve_task,
    parse_positive_int_list,
    plan_a_subsample_curve_tasks,
    theta_grid_periodic,
)
from global_setting import DEFAULT_CUDA_DEVICE_IDS, DEFAULT_DEVICE


def _load_serial_module() -> Any:
    path = _REPO_ROOT / "bin" / "compare_stringer_fisher_session_identification.py"
    spec = importlib.util.spec_from_file_location("compare_stringer_fisher_session_identification", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


serial = _load_serial_module()


def _parse_int_list(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, (list, tuple)):
        vals = [int(v) for v in value]
    else:
        vals = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer.")
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


def resolve_output_dir(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser() if args.output_dir is not None else serial.default_output_dir(str(args.session_stimuli_type))
    if bool(args.subsample_a_convergence) and args.output_dir is None:
        output_dir = output_dir / "a_subsample_convergence"
    if not output_dir.is_absolute():
        output_dir = (_REPO_ROOT / output_dir).resolve()
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    p = serial.build_parser()
    p.description = __doc__
    p.set_defaults(device=DEFAULT_DEVICE)
    p.add_argument("--gpu-ids", type=_parse_gpu_ids, default=list(DEFAULT_CUDA_DEVICE_IDS))
    p.add_argument("--jobs-per-gpu", type=_positive_int, default=5)
    p.add_argument("--cpu-threads-per-job", default="auto")
    p.add_argument("--parallel-log-dir", type=Path, default=None)
    p.add_argument("--worker-task-json", type=Path, default=None, help=argparse.SUPPRESS)
    original_parse_args = p.parse_args

    def parse_args(args=None, namespace=None):
        parsed = original_parse_args(args, namespace)
        parsed.gpu_ids = _parse_gpu_ids(parsed.gpu_ids)
        parsed.jobs_per_gpu = _positive_int(parsed.jobs_per_gpu)
        total_workers = len(parsed.gpu_ids) * int(parsed.jobs_per_gpu)
        parsed.cpu_threads_per_job = _resolve_cpu_threads(parsed.cpu_threads_per_job, total_workers)
        parsed.output_dir = resolve_output_dir(parsed)
        if parsed.parallel_log_dir is None:
            parsed.parallel_log_dir = Path(parsed.output_dir) / "parallel_logs"
        elif not Path(parsed.parallel_log_dir).is_absolute():
            parsed.parallel_log_dir = (_REPO_ROOT / Path(parsed.parallel_log_dir)).resolve()
        return parsed

    p.parse_args = parse_args  # type: ignore[method-assign]
    return p


@dataclass
class RunningTask:
    task: ASubsampleCurveTask
    gpu_id: int
    process: subprocess.Popen
    stdout_path: Path
    stderr_path: Path


def build_worker_env(base_env: dict[str, str], *, gpu_id: int, cpu_threads_per_job: int) -> dict[str, str]:
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    env["PYTHONUNBUFFERED"] = "1"
    threads = str(int(cpu_threads_per_job))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = threads
    return env


def _arg_to_cli_name(dest: str) -> str:
    return "--" + str(dest).replace("_", "-")


def _stringify_value(value: Any) -> str:
    if value is None:
        raise ValueError("Cannot stringify None.")
    return str(value)


def _serial_command_from_args(args: argparse.Namespace, *, worker_task_json: Path | None = None) -> list[str]:
    command = [sys.executable, str(_REPO_ROOT / "bin" / "compare_stringer_fisher_session_identification_parallel.py")]
    case_args = argparse.Namespace(**vars(args))
    case_args.device = "cuda"
    parser = serial.build_parser()
    skip = {
        "help",
        "gpu_ids",
        "jobs_per_gpu",
        "cpu_threads_per_job",
        "parallel_log_dir",
        "worker_task_json",
    }
    for action in parser._actions:
        dest = getattr(action, "dest", None)
        if not dest or dest in skip or not hasattr(case_args, dest):
            continue
        value = getattr(case_args, dest)
        if value is None:
            continue
        option = _arg_to_cli_name(dest)
        if isinstance(action, argparse.BooleanOptionalAction):
            command.append(option if bool(value) else "--no-" + option[2:])
            continue
        if isinstance(action, argparse._StoreTrueAction):
            if bool(value):
                command.append(option)
            continue
        if isinstance(action, argparse._StoreFalseAction):
            if not bool(value):
                command.append(option)
            continue
        command.extend([option, _stringify_value(value)])
    if worker_task_json is not None:
        command.extend(["--worker-task-json", str(worker_task_json)])
    return command


def build_worker_command(args: argparse.Namespace, task_json: Path) -> list[str]:
    return _serial_command_from_args(args, worker_task_json=task_json)


def build_aggregation_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(_REPO_ROOT / "bin" / "compare_stringer_fisher_session_identification.py")]
    case_args = argparse.Namespace(**vars(args))
    case_args.device = "cuda"
    case_args.force = False
    parser = serial.build_parser()
    skip = {
        "help",
        "gpu_ids",
        "jobs_per_gpu",
        "cpu_threads_per_job",
        "parallel_log_dir",
        "worker_task_json",
    }
    for action in parser._actions:
        dest = getattr(action, "dest", None)
        if not dest or dest in skip or not hasattr(case_args, dest):
            continue
        value = getattr(case_args, dest)
        if value is None:
            continue
        option = _arg_to_cli_name(dest)
        if isinstance(action, argparse.BooleanOptionalAction):
            command.append(option if bool(value) else "--no-" + option[2:])
            continue
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


def write_task_json(task: ASubsampleCurveTask, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(task.to_json_dict(), sort_keys=True, indent=2) + "\n")
    return path


def load_task_json(path: Path) -> ASubsampleCurveTask:
    return ASubsampleCurveTask.from_json_dict(json.loads(Path(path).read_text()))


def _task_kwargs(args: argparse.Namespace, *, sessions: list[Any], grid: np.ndarray) -> dict[str, Any]:
    return {
        "sessions": sessions,
        "theta_grid": grid,
        "period": float(args.orientation_period),
        "pca_dim": int(args.pca_dim),
        "pca_random_state": int(args.pca_random_state),
        "pca_whiten": not bool(args.no_pca_whiten),
        "train_frac": float(args.train_frac),
        "seed": int(args.seed),
        "flow_config": serial.flow_config_from_args(args),
        "output_dir": Path(args.output_dir),
        "n_values": parse_positive_int_list(str(args.subsample_a_n_list)),
        "sampling": str(args.subsample_a_sampling),
        "replace": not bool(args.subsample_a_without_replacement),
        "classical_ridge": float(args.classical_linear_ridge),
        "classical_window_radius": args.classical_window_radius,
        "classical_min_endpoint_samples": int(args.classical_min_endpoint_samples),
    }


def select_tasks_to_run(
    tasks: list[ASubsampleCurveTask],
    args: argparse.Namespace,
    *,
    sessions: list[Any],
    grid: np.ndarray,
) -> list[ASubsampleCurveTask]:
    _task_kwargs(args, sessions=sessions, grid=grid)
    if bool(args.visualization_only):
        return []
    return list(tasks)


def validate_cuda_devices(gpu_ids: list[int]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run Stringer parallel workers on CPU.")
    count = int(torch.cuda.device_count())
    if not gpu_ids:
        raise ValueError("At least one GPU id is required.")
    missing = [int(g) for g in gpu_ids if int(g) >= count]
    if missing:
        raise RuntimeError(f"Requested CUDA device ids {missing}, but torch sees {count} CUDA device(s).")


def run_tasks_parallel(tasks: list[ASubsampleCurveTask], args: argparse.Namespace) -> None:
    if not tasks:
        return
    log_dir = Path(args.parallel_log_dir)
    task_dir = log_dir / "tasks"
    log_dir.mkdir(parents=True, exist_ok=True)
    slots = [int(gpu) for gpu in args.gpu_ids for _ in range(int(args.jobs_per_gpu))]
    pending = list(tasks)
    running: list[RunningTask] = []
    base_env = os.environ.copy()
    while pending or running:
        while pending and len(running) < len(slots):
            used: dict[int, int] = {}
            for rt in running:
                used[rt.gpu_id] = used.get(rt.gpu_id, 0) + 1
            gpu_id = min(slots, key=lambda g: used.get(int(g), 0))
            task = pending.pop(0)
            task_json = write_task_json(task, task_dir / f"{task.label}.json")
            stdout_path = log_dir / f"{task.label}.stdout.log"
            stderr_path = log_dir / f"{task.label}.stderr.log"
            cmd = build_worker_command(args, task_json)
            env = build_worker_env(base_env, gpu_id=int(gpu_id), cpu_threads_per_job=int(args.cpu_threads_per_job))
            print(f"[stringer-identification-parallel] launch {task.label} gpu={gpu_id}", flush=True)
            stdout_f = stdout_path.open("w")
            stderr_f = stderr_path.open("w")
            proc = subprocess.Popen(cmd, cwd=str(_REPO_ROOT), env=env, stdout=stdout_f, stderr=stderr_f)
            stdout_f.close()
            stderr_f.close()
            running.append(
                RunningTask(
                    task=task,
                    gpu_id=int(gpu_id),
                    process=proc,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
            )
        time.sleep(2.0)
        still: list[RunningTask] = []
        for rt in running:
            code = rt.process.poll()
            if code is None:
                still.append(rt)
                continue
            if int(code) != 0:
                raise RuntimeError(
                    f"Task {rt.task.label} failed with exit code {code}. "
                    f"stdout={rt.stdout_path} stderr={rt.stderr_path}"
                )
            print(f"[stringer-identification-parallel] done {rt.task.label}", flush=True)
        running = still


def run_worker(args: argparse.Namespace) -> dict[str, Path]:
    serial.validate_args(args)
    device = require_device(str(args.device))
    task = load_task_json(Path(args.worker_task_json))
    sessions = list_stringer_sessions(str(args.session_stimuli_type), data_dir=args.data_dir)
    if args.max_sessions is not None:
        sessions = sessions[: int(args.max_sessions)]
    grid = theta_grid_periodic(float(args.orientation_period), int(args.theta_grid_size))
    estimate_a_subsample_curve_task(
        task,
        sessions=sessions,
        theta_grid=grid,
        period=float(args.orientation_period),
        pca_dim=int(args.pca_dim),
        pca_random_state=int(args.pca_random_state),
        pca_whiten=not bool(args.no_pca_whiten),
        train_frac=float(args.train_frac),
        seed=int(args.seed),
        device=device,
        flow_config=serial.flow_config_from_args(args),
        output_dir=Path(args.output_dir),
        n_values=parse_positive_int_list(str(args.subsample_a_n_list)),
        sampling=str(args.subsample_a_sampling),
        replace=not bool(args.subsample_a_without_replacement),
        force=bool(args.force),
        save_flow_npz=not bool(args.skip_flow_npz),
        classical_ridge=float(args.classical_linear_ridge),
        classical_window_radius=args.classical_window_radius,
        classical_min_endpoint_samples=int(args.classical_min_endpoint_samples),
    )
    return {"output_dir": Path(args.output_dir)}


def run_aggregation(args: argparse.Namespace) -> None:
    gpu_id = int(args.gpu_ids[0])
    env = build_worker_env(os.environ.copy(), gpu_id=gpu_id, cpu_threads_per_job=int(args.cpu_threads_per_job))
    cmd = build_aggregation_command(args)
    print(f"[stringer-identification-parallel] aggregate gpu={gpu_id}", flush=True)
    subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env, check=True)


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.worker_task_json is not None:
        return run_worker(args)
    if not bool(args.subsample_a_convergence):
        raise ValueError("The parallel wrapper currently supports --subsample-a-convergence only.")
    serial.validate_args(args)
    validate_cuda_devices([int(g) for g in args.gpu_ids])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions = list_stringer_sessions(str(args.session_stimuli_type), data_dir=args.data_dir)
    if args.max_sessions is not None:
        sessions = sessions[: int(args.max_sessions)]
    if len(sessions) < 2:
        raise ValueError("Session identification requires at least two sessions.")
    grid = theta_grid_periodic(float(args.orientation_period), int(args.theta_grid_size))
    tasks = plan_a_subsample_curve_tasks(
        sessions=sessions,
        n_values=parse_positive_int_list(str(args.subsample_a_n_list)),
        repeats=int(args.subsample_a_repeats),
    )
    to_run = select_tasks_to_run(tasks, args, sessions=sessions, grid=grid)
    print(
        f"[stringer-identification-parallel] tasks={len(tasks)} to_run={len(to_run)} "
        f"gpus={args.gpu_ids} jobs_per_gpu={args.jobs_per_gpu} output_dir={output_dir}",
        flush=True,
    )
    run_tasks_parallel(to_run, args)
    run_aggregation(args)
    return {"output_dir": output_dir}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
