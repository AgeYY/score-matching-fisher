from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from global_setting import DEFAULT_DEVICE

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_parallel_module():
    path = _REPO_ROOT / "bin" / "compare_stringer_fisher_session_identification_parallel.py"
    spec = importlib.util.spec_from_file_location("compare_stringer_fisher_session_identification_parallel", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parallel_parser_defaults_jobs_per_gpu_five(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_parallel_module()
    monkeypatch.setattr(mod.os, "cpu_count", lambda: 40)

    args = mod.build_parser().parse_args(
        [
            "--subsample-a-convergence",
            "--gpu-ids",
            "0,1",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert args.gpu_ids == [0, 1]
    assert args.jobs_per_gpu == 5
    assert args.cpu_threads_per_job == 4
    assert args.device == DEFAULT_DEVICE
    assert args.output_dir == tmp_path / "out"
    assert args.parallel_log_dir == tmp_path / "out" / "parallel_logs"


def test_worker_env_masks_cuda_and_threads() -> None:
    mod = _load_parallel_module()

    env = mod.build_worker_env({"PATH": "/bin"}, gpu_id=1, cpu_threads_per_job=3)

    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["OMP_NUM_THREADS"] == "3"
    assert env["MKL_NUM_THREADS"] == "3"
    assert env["OPENBLAS_NUM_THREADS"] == "3"
    assert env["NUMEXPR_NUM_THREADS"] == "3"


def test_worker_and_aggregation_commands_use_logical_cuda_and_skip_parallel_flags(tmp_path: Path) -> None:
    mod = _load_parallel_module()
    args = mod.build_parser().parse_args(
        [
            "--subsample-a-convergence",
            "--output-dir",
            str(tmp_path / "out"),
            "--epochs",
            "7",
            "--batch-size",
            "32",
            "--gpu-ids",
            "0,1",
            "--jobs-per-gpu",
            "5",
            "--force",
            "--subsample-a-without-replacement",
            "--flow-orientation-encoding",
            "scalar",
        ]
    )
    task_json = tmp_path / "task.json"

    worker = mod.build_worker_command(args, task_json)
    aggregation = mod.build_aggregation_command(args)
    worker_joined = " ".join(worker)
    aggregation_joined = " ".join(aggregation)

    assert worker[1].endswith("bin/compare_stringer_fisher_session_identification_parallel.py")
    assert aggregation[1].endswith("bin/compare_stringer_fisher_session_identification.py")
    assert "--device cuda" in worker_joined
    assert "--device cuda" in aggregation_joined
    assert "--worker-task-json" in worker
    assert str(task_json) in worker
    assert "--gpu-ids" not in worker
    assert "--jobs-per-gpu" not in worker
    assert "--force" in worker
    assert "--force" not in aggregation
    assert "--subsample-a-without-replacement" in worker
    assert "--subsample-a-without-replacement" in aggregation
    assert "--flow-orientation-encoding scalar" in worker_joined
    assert "--flow-orientation-encoding scalar" in aggregation_joined
    assert "True" not in worker
    assert "False" not in worker
    assert "--epochs 7" in worker_joined
    assert "--batch-size 32" in worker_joined

    replacement_args = mod.build_parser().parse_args(
        [
            "--subsample-a-convergence",
            "--output-dir",
            str(tmp_path / "out2"),
            "--no-subsample-a-without-replacement",
        ]
    )
    replacement_worker = mod.build_worker_command(replacement_args, task_json)
    assert "--no-subsample-a-without-replacement" in replacement_worker
    assert "False" not in replacement_worker


def test_a_subsample_task_planning_labels_and_json_roundtrip() -> None:
    from fisher.stringer_session_identification import ASubsampleCurveTask, plan_a_subsample_curve_tasks

    sessions = [SimpleNamespace(session_file=f"session_{idx}.npy") for idx in range(2)]

    tasks = plan_a_subsample_curve_tasks(sessions=sessions, n_values=[64, 128], repeats=2)

    assert len(tasks) == 12
    assert tasks[0].label == "endpoint_full_a_session000"
    assert tasks[1].label == "endpoint_full_a_session001"
    assert tasks[2].label == "reference_b_session000"
    assert tasks[4].label == "subset_a_n000064_repeat000_session000"
    assert tasks[-1].label == "subset_a_n000128_repeat001_session001"
    assert ASubsampleCurveTask.from_json_dict(tasks[-1].to_json_dict()) == tasks[-1]
