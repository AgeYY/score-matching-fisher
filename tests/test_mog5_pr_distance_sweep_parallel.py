from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ALL_METRICS = (
    "squared_euclidean",
    "cosine",
    "correlation",
    "mahalanobis_sq",
    "symmetric_kl",
)


def _load_cli_module():
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "bin" / "compare_mog5_pr_distance_sweeps_parallel.py"
    spec = importlib.util.spec_from_file_location("compare_mog5_pr_distance_sweeps_parallel", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_case_npz(path: Path, *, offset: float = 0.0, metrics: tuple[str, ...] = ALL_METRICS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(["category_0", "category_1", "category_2"])
    pairs = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    classical = np.zeros((len(metrics), 3, 3), dtype=np.float64)
    flow = np.zeros((len(metrics), 3, 3), dtype=np.float64)
    flow_finetuned = np.zeros((len(metrics), 3, 3), dtype=np.float64)
    gt = np.zeros((len(metrics), 3, 3), dtype=np.float64)
    for metric_idx in range(len(metrics)):
        for i, j in pairs:
            truth = offset + 10.0 + metric_idx
            classical[metric_idx, i, j] = classical[metric_idx, j, i] = truth + 1.0
            flow[metric_idx, i, j] = flow[metric_idx, j, i] = truth - 2.0
            flow_finetuned[metric_idx, i, j] = flow_finetuned[metric_idx, j, i] = truth - 1.0
            gt[metric_idx, i, j] = gt[metric_idx, j, i] = truth
    np.savez_compressed(
        path,
        metric_names=np.asarray(metrics),
        condition_labels=labels,
        pair_indices=pairs,
        classical_matrices=classical,
        flow_matching_matrices=flow,
        flow_matching_nll_finetuned_matrices=flow_finetuned,
        ground_truth_matrices=gt,
    )
    return path


def _write_flow_loss_npz(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_losses=np.asarray([1.0, 0.5], dtype=np.float64),
        val_losses=np.asarray([1.1, 0.7], dtype=np.float64),
        val_monitor_losses=np.asarray([1.1, 0.8], dtype=np.float64),
    )
    return path


def _fake_ground_truth_payload(metrics: tuple[str, ...] = ALL_METRICS) -> dict[str, object]:
    labels = ("category_0", "category_1", "category_2")
    matrices = np.zeros((len(metrics), len(labels), len(labels)), dtype=np.float64)
    for metric_idx in range(len(metrics)):
        matrices[metric_idx] = np.asarray(
            [[0.0, 1.0 + metric_idx, 2.0 + metric_idx], [1.0 + metric_idx, 0.0, 3.0], [2.0, 3.0, 0.0]],
            dtype=np.float64,
        )
    return {
        "metric_names": tuple(metrics),
        "condition_labels": labels,
        "ground_truth_matrices": matrices,
        "n_total": 1000,
        "native_x_dim": 3,
        "pr_dim": None,
        "pr_projected": False,
        "pr_dim_label": "native",
        "native_npz": "random_mog_categorical.npz",
    }


class FakeProcess:
    def __init__(self, returncode: int = 0):
        self.returncode = int(returncode)

    def poll(self):
        return self.returncode


def _patch_case_output_dir(monkeypatch: pytest.MonkeyPatch, mod, root: Path) -> None:
    monkeypatch.setattr(
        mod.sweep,
        "case_output_dir",
        lambda *, n_total, pr_dim, case_output_name, native_x_dim=3, repeat_idx=0, n_repeats=1: (
            root / f"case_{int(n_total)}_{mod.sweep._pr_dim_label(pr_dim)}" / f"repeat_{int(repeat_idx):02d}"
        ),
    )


def test_parser_defaults_and_cpu_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_cli_module()
    monkeypatch.setattr(mod.os, "cpu_count", lambda: 16)

    args = mod.build_parser().parse_args([])

    assert args.gpu_ids == [0]
    assert args.jobs_per_gpu == 1
    assert args.device == "cuda:0"
    assert args.flow_likelihood_finetune_epochs == 20_000
    assert args.seed == 19
    assert args.n_list == [100, 1000, 2000, 3000]
    assert args.n_repeats == 10
    assert args.dataset_cov_theta_amp_scale == pytest.approx(1.0)
    assert args.batch_size == 3000
    assert args.lr == pytest.approx(1e-4)
    assert args.hidden_dim == 128
    assert args.depth == 3
    assert args.fixed_validation is True
    assert args.fixed_validation_paths == 10
    assert args.flow_likelihood_finetune_batch_size == 3000
    assert args.flow_likelihood_finetune_lr == pytest.approx(3e-5)
    assert args.flow_likelihood_finetune_ode_steps == 32
    assert args.flow_likelihood_finetune_patience == 1_000
    assert args.flow_likelihood_finetune_checkpoint_selection == "best"
    assert args.tre_num_bridges == 8
    assert args.tre_architecture == "mlp"
    assert args.cpu_threads_per_job == 16
    assert args.parallel_log_dir == args.output_dir / "parallel_logs"

    args = mod.build_parser().parse_args(["--gpu-ids", "2,3", "--jobs-per-gpu", "2", "--cpu-threads-per-job", "3"])
    assert args.gpu_ids == [2, 3]
    assert args.cpu_threads_per_job == 3


def test_case_planning_paths_and_repeat_seeds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    _patch_case_output_dir(monkeypatch, mod, tmp_path)
    args = mod.build_parser().parse_args(["--n-list", "100,200", "--n-repeats", "2", "--seed", "11"])

    tasks = mod.plan_cases(args)

    assert [(t.n_total, t.repeat_idx, t.seed) for t in tasks] == [
        (100, 0, 11),
        (100, 1, 12),
        (200, 0, 11),
        (200, 1, 12),
    ]
    assert tasks[1].result_path == tmp_path / "case_100_native" / "repeat_01" / mod.sweep.RESULTS_NAME
    assert tasks[1].key == (100, -1, 1)


def test_case_planning_supports_isolated_dataset_root(tmp_path: Path) -> None:
    mod = _load_cli_module()
    root = tmp_path / "seed7_cases"
    args = mod.build_parser().parse_args(
        [
            "--n-list",
            "100,1000",
            "--n-repeats",
            "2",
            "--pr-dim",
            "none",
            "--case-output-name",
            "comparison",
            "--case-dataset-root",
            str(root),
        ]
    )

    tasks = mod.plan_cases(args)

    assert tasks[0].dataset_dir == root / "n100_native" / "repeat_00"
    assert tasks[0].output_dir == tasks[0].dataset_dir / "comparison"
    assert tasks[-1].dataset_dir == root / "n1000_native" / "repeat_01"


def test_worker_command_and_env_forwarding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    _patch_case_output_dir(monkeypatch, mod, tmp_path)
    args = mod.build_parser().parse_args(
        [
            "--n-list",
            "100",
            "--n-repeats",
            "1",
            "--metric",
            "symmetric_kl",
            "--epochs",
            "7",
            "--batch-size",
            "32",
            "--include-tre",
            "--tre-augment-existing",
            "--native-template-npz",
            str(tmp_path / "ignored.npz"),
            "--output-dir",
            str(tmp_path / "sweep"),
            "--cpu-threads-per-job",
            "4",
        ]
    )
    task = mod.plan_cases(args)[0]

    command = mod.build_case_command(args, task, native_template_npz=tmp_path / "template.npz")
    env = mod.build_case_env({"PATH": "/bin"}, gpu_id=3, cpu_threads_per_job=args.cpu_threads_per_job)

    assert command[1].endswith("bin/compare_mog5_pr_distances.py")
    joined = " ".join(command)
    assert "--native-template-npz" in command
    assert str(tmp_path / "template.npz") in command
    assert "--dataset-dir" in command
    assert str(task.dataset_dir) in command
    assert "--output-dir" in command
    assert str(task.output_dir) in command
    assert "--device cuda" in joined
    assert "--metric symmetric_kl" in joined
    assert "--epochs 7" in joined
    assert "--batch-size 32" in joined
    assert "--include-tre" in command
    assert "--tre-augment-existing" in command
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["PYTHONUNBUFFERED"] == "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert env[name] == "4"


def test_cache_hits_and_force_behavior(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    _patch_case_output_dir(monkeypatch, mod, tmp_path)
    args = mod.build_parser().parse_args(["--n-list", "100", "--n-repeats", "1"])
    task = mod.plan_cases(args)[0]
    _write_case_npz(task.result_path)

    to_run, cache_hits = mod.select_tasks_to_run([task], args=args, metrics=ALL_METRICS)

    assert to_run == []
    assert cache_hits[task.key] is True

    force_args = mod.build_parser().parse_args(["--n-list", "100", "--n-repeats", "1", "--force-comparison"])
    force_task = mod.plan_cases(force_args)[0]
    to_run, cache_hits = mod.select_tasks_to_run([force_task], args=force_args, metrics=ALL_METRICS)
    assert to_run == [force_task]
    assert cache_hits[force_task.key] is False

    force_dataset_args = mod.build_parser().parse_args(
        ["--n-list", "100", "--n-repeats", "1", "--force-dataset"]
    )
    force_dataset_task = mod.plan_cases(force_dataset_args)[0]
    to_run, cache_hits = mod.select_tasks_to_run(
        [force_dataset_task], args=force_dataset_args, metrics=ALL_METRICS
    )
    assert to_run == [force_dataset_task]
    assert cache_hits[force_dataset_task.key] is False


def test_visualization_only_preflight_fails_for_missing_repeat(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    _patch_case_output_dir(monkeypatch, mod, tmp_path)
    args = mod.build_parser().parse_args(["--visualization-only", "--n-list", "100", "--n-repeats", "2"])
    tasks = mod.plan_cases(args)
    _write_case_npz(tasks[0].result_path)

    with pytest.raises(FileNotFoundError, match="repeat01|repeat_01|repeat=1"):
        mod.preflight_visualization_only(tasks, ALL_METRICS)


def test_finalization_with_fake_child_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_cli_module()
    _patch_case_output_dir(monkeypatch, mod, tmp_path)
    monkeypatch.setattr(
        mod.sweep,
        "compute_baseline_ground_truth_rdms",
        lambda args, metrics: _fake_ground_truth_payload(tuple(str(metric) for metric in metrics)),
    )

    launches: list[dict[str, object]] = []

    def fake_launch(command, *, env, stdout_path, stderr_path):
        output_dir = Path(command[command.index("--output-dir") + 1])
        _write_case_npz(output_dir / mod.sweep.RESULTS_NAME, metrics=("squared_euclidean",))
        _write_flow_loss_npz(output_dir / "flow" / "squared_euclidean_flow_matching_skl_results.npz")
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("ok\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        launches.append({"command": command, "env": env, "stdout": stdout_path, "stderr": stderr_path})
        return FakeProcess(0)

    monkeypatch.setattr(mod, "_launch_case_process", fake_launch)
    args = mod.build_parser().parse_args(
        [
            "--n-list",
            "100,200",
            "--n-repeats",
            "2",
            "--metric",
            "squared_euclidean",
            "--output-dir",
            str(tmp_path / "sweep"),
            "--parallel-log-dir",
            str(tmp_path / "logs"),
            "--gpu-ids",
            "0,1",
            "--jobs-per-gpu",
            "1",
            "--cpu-threads-per-job",
            "2",
            "--skip-dataset-viz",
        ]
    )

    outputs = mod.run(args)

    assert len(launches) == 4
    assert {launch["env"]["CUDA_VISIBLE_DEVICES"] for launch in launches} == {"0", "1"}
    assert outputs["results_npz"].is_file()
    assert outputs["errors_csv"].is_file()
    assert outputs["figure_svg"].is_file()
    assert outputs["figure_png"].is_file()
    assert outputs["rel_error_figure_svg"].is_file()
    assert outputs["flow_loss_figure_svg"].is_file()
    with np.load(outputs["results_npz"], allow_pickle=False) as data:
        assert data["n_repeat_classical_matrices"].shape == (2, 2, 1, 3, 3)
        assert data["n_repeat_flow_matching_nll_finetuned_matrices"].shape == (2, 2, 1, 3, 3)
    with outputs["errors_csv"].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {row["repeat_idx"] for row in rows} == {"0", "1"}
    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))
    assert summary["script"] == "bin/compare_mog5_pr_distance_sweeps_parallel.py"
    assert summary["config"]["gpu_ids"] == [0, 1]
    assert summary["config"]["worker_count"] == 2
    assert summary["config"]["cpu_threads_per_job"] == 2
    assert summary["config"]["parallel_log_dir"] == str(tmp_path / "logs")
    assert len(summary["parallel_case_logs"]) == 4
    assert all("stdout" in item and "stderr" in item for item in summary["parallel_case_logs"].values())
