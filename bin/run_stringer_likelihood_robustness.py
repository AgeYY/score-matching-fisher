#!/usr/bin/env python3
"""Run and plot four Stringer held-out likelihood robustness configurations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.stringer_dataset import list_stringer_sessions
from global_setting import DATA_DIR


@dataclass(frozen=True)
class RobustnessConfig:
    key: str
    title: str
    pca_dim: int
    fit_fraction: float


CONFIGS = (
    RobustnessConfig("fit40_pca82", "40% fit, PCA 82", 82, 0.4),
    RobustnessConfig("fit60_pca82", "60% fit, PCA 82", 82, 0.6),
    RobustnessConfig("fit80_pca30", "80% fit, PCA 30", 30, 0.8),
    RobustnessConfig("fit80_pca130", "80% fit, PCA 130", 130, 0.8),
)
METHOD_KEYS = (
    "binned_test_log_likelihood",
    "gkr_test_log_likelihood",
    "affine_test_log_likelihood",
    "nonlinear_test_log_likelihood",
)
METHOD_LABELS = ("Bin+LW", "GKR", "Affine Flow", "Uncon. Flow")
BAR_COLOR = "0.65"
BAR_EDGE_COLOR = "0.35"


def _csv_strings(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return result


def _csv_ints(value: str) -> list[int]:
    try:
        result = [int(item) for item in _csv_strings(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from exc
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--devices",
        type=_csv_strings,
        help="CUDA devices assigned round-robin, for example cuda:0,cuda:1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "stringer_likelihood_robustness",
    )
    parser.add_argument(
        "--configs",
        type=_csv_strings,
        default=[config.key for config in CONFIGS],
        help="Subset of configuration keys to run.",
    )
    parser.add_argument("--session-indices", type=_csv_ints, default=list(range(6)))
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--refit-gkr-only",
        action="store_true",
        help="Reuse existing base/four-method fits and only refit upgraded GKR.",
    )
    parser.add_argument("--gkr-covariance-epochs", type=int, default=100)
    parser.add_argument("--gkr-mean-learning-rate", type=float, default=0.01)
    parser.add_argument("--gkr-covariance-learning-rate", type=float, default=0.01)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _selected_configs(keys: list[str]) -> list[RobustnessConfig]:
    by_key = {config.key: config for config in CONFIGS}
    unknown = sorted(set(keys) - set(by_key))
    if unknown:
        raise ValueError(f"Unknown configurations: {unknown}")
    return [config for config in CONFIGS if config.key in keys]


def _run_command(command: list[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== command started {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
        log.write(" ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        log.write(
            f"=== command finished code={completed.returncode} "
            f"{time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n"
        )
    if completed.returncode:
        raise RuntimeError(
            f"Command failed with code {completed.returncode}; see {log_path}."
        )


def _run_case(
    *,
    config: RobustnessConfig,
    session_index: int,
    device: str,
    output_dir: Path,
    force: bool,
    refit_gkr_only: bool,
    gkr_covariance_epochs: int,
    gkr_mean_learning_rate: float,
    gkr_covariance_learning_rate: float,
) -> None:
    session = list_stringer_sessions("gratings_static")[int(session_index)]
    case_name = f"session_{int(session_index):02d}_{session.mouse_name}"
    config_dir = output_dir / config.key
    base_dir = config_dir / "base" / case_name
    result_dir = config_dir / "four_methods" / case_name
    gkr_dir = config_dir / "log_lambda_gkr" / case_name
    log_path = result_dir / "run.log"
    gkr_log_path = gkr_dir / "run.log"
    common_force = ["--force"] if force else []
    print(
        f"[start] {config.key} {case_name} device={device} log={log_path}",
        flush=True,
    )
    if not refit_gkr_only:
        _run_command(
            [
                sys.executable,
                str(REPO_ROOT / "bin" / "visualize_stringer_pca_flow_gkr.py"),
                "--device",
                device,
                "--session-index",
                str(int(session_index)),
                "--pca-dim",
                str(int(config.pca_dim)),
                "--train-fraction",
                str(float(config.fit_fraction)),
                "--flow-validation-fraction",
                "0.2",
                "--output-dir",
                str(base_dir),
                *common_force,
            ],
            log_path=log_path,
        )
        _run_command(
            [
                sys.executable,
                str(REPO_ROOT / "bin" / "visualize_stringer_pca_four_methods.py"),
                "--device",
                device,
                "--base-result-dir",
                str(base_dir),
                "--output-dir",
                str(result_dir),
                *common_force,
            ],
            log_path=log_path,
        )
    _run_command(
        [
            sys.executable,
            str(REPO_ROOT / "bin" / "refit_stringer_gkr_conventional_kernel.py"),
            "--device",
            device,
            "--base-result-dir",
            str(base_dir),
            "--output-dir",
            str(gkr_dir),
            "--kernel-parameterization",
            "log-lambda",
            "--covariance-epochs",
            str(int(gkr_covariance_epochs)),
            "--mean-learning-rate",
            str(float(gkr_mean_learning_rate)),
            "--covariance-learning-rate",
            str(float(gkr_covariance_learning_rate)),
            "--standardize-responses",
            *common_force,
        ],
        log_path=gkr_log_path,
    )
    print(f"[complete] {config.key} {case_name}", flush=True)


def _run_worker(
    device: str,
    tasks: list[tuple[RobustnessConfig, int]],
    *,
    output_dir: Path,
    force: bool,
    refit_gkr_only: bool,
    gkr_covariance_epochs: int,
    gkr_mean_learning_rate: float,
    gkr_covariance_learning_rate: float,
) -> None:
    for config, session_index in tasks:
        _run_case(
            config=config,
            session_index=session_index,
            device=device,
            output_dir=output_dir,
            force=force,
            refit_gkr_only=refit_gkr_only,
            gkr_covariance_epochs=gkr_covariance_epochs,
            gkr_mean_learning_rate=gkr_mean_learning_rate,
            gkr_covariance_learning_rate=gkr_covariance_learning_rate,
        )


def _load_config_results(
    config: RobustnessConfig,
    *,
    output_dir: Path,
) -> tuple[list[str], np.ndarray]:
    root = output_dir / config.key / "four_methods"
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for session_index, session in enumerate(list_stringer_sessions("gratings_static")):
        case_dir = root / f"session_{session_index:02d}_{session.mouse_name}"
        result_path = case_dir / "four_method_results.npz"
        gkr_path = (
            output_dir
            / config.key
            / "log_lambda_gkr"
            / f"session_{session_index:02d}_{session.mouse_name}"
            / "conventional_gkr_results.npz"
        )
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        if not gkr_path.is_file():
            raise FileNotFoundError(gkr_path)
        with np.load(result_path, allow_pickle=False) as saved:
            binned = float(np.mean(saved["binned_test_log_likelihood"]))
            affine = float(np.mean(saved["affine_test_log_likelihood"]))
            nonlinear = float(np.mean(saved["nonlinear_test_log_likelihood"]))
        with np.load(gkr_path, allow_pickle=False) as saved:
            gkr = float(np.mean(saved["conventional_test_log_likelihood"]))
        rows.append(np.asarray([binned, gkr, affine, nonlinear], dtype=np.float64))
        labels.append(str(session.mouse_name))
    return labels, np.vstack(rows)


def _axis_limits(values: np.ndarray, sem: np.ndarray) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    errors = np.asarray(sem, dtype=np.float64)
    means = np.mean(data, axis=0)
    low = float(min(np.min(data), np.min(means - errors)))
    high = float(max(np.max(data), np.max(means + errors)))
    span = max(high - low, 1.0)
    return low - 0.1 * span, high + 0.1 * span


def _relative_to_bin_lw(values: np.ndarray) -> np.ndarray:
    """Subtract each session's Bin+LW likelihood from every method."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(METHOD_KEYS):
        raise ValueError("values must have shape [sessions, methods].")
    baseline_index = METHOD_KEYS.index("binned_test_log_likelihood")
    return array - array[:, baseline_index : baseline_index + 1]


def _plot(
    results: list[tuple[RobustnessConfig, list[str], np.ndarray]],
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.grid": False,
        }
    )
    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(3.5 * len(results) + 0.5, 3.5),
        constrained_layout=True,
        squeeze=False,
    )
    positions = np.arange(len(METHOD_KEYS), dtype=np.float64)
    for panel_index, (axis, (config, _, values)) in enumerate(
        zip(axes[0], results, strict=True)
    ):
        relative_values = _relative_to_bin_lw(values)
        means = np.mean(relative_values, axis=0)
        sem = np.std(relative_values, axis=0, ddof=1) / np.sqrt(
            relative_values.shape[0]
        )
        axis.bar(
            positions,
            means,
            yerr=sem,
            width=0.68,
            color=BAR_COLOR,
            alpha=0.50,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=1.8,
            capsize=3.5,
            error_kw={
                "ecolor": "black",
                "elinewidth": 1.6,
                "capthick": 1.6,
            },
            zorder=2,
        )
        for session_values in relative_values:
            axis.plot(
                positions,
                session_values,
                color="0.45",
                linewidth=1.2,
                alpha=0.45,
                zorder=3,
            )
            axis.scatter(
                positions,
                session_values,
                color="black",
                edgecolor="white",
                linewidth=0.45,
                s=34,
                zorder=4,
            )
        axis.set_xticks(positions, METHOD_LABELS, rotation=25, ha="right")
        axis.set_title(config.title)
        axis.set_ylim(*_axis_limits(relative_values, sem))
        axis.axhline(
            0.0,
            color="0.35",
            linewidth=1.2,
            linestyle="--",
            zorder=1,
        )
        if panel_index == 0:
            axis.set_ylabel(
                r"Orientation $\theta$" "\n"
                "test log likelihood relative\n"
                "to Bin + LW"
            )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(1.8)
        axis.spines["bottom"].set_linewidth(1.8)
        axis.tick_params(width=1.8)
        axis.set_axisbelow(True)
        axis.yaxis.grid(True, color="0.88", linewidth=0.8)
        axis.xaxis.grid(False)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "stringer_likelihood_robustness.png"
    svg = figure_dir / "stringer_likelihood_robustness.svg"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    return png, svg


def _aggregate(
    configs: list[RobustnessConfig],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    if len(configs) != len(CONFIGS):
        raise ValueError("The final robustness figure requires all four configurations.")
    results = []
    summary_configs: dict[str, Any] = {}
    for config in configs:
        labels, values = _load_config_results(config, output_dir=output_dir)
        results.append((config, labels, values))
        means = np.mean(values, axis=0)
        sem = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
        summary_configs[config.key] = {
            "title": config.title,
            "pca_dim": config.pca_dim,
            "fit_fraction": config.fit_fraction,
            "flow_train_fraction": 0.8 * config.fit_fraction,
            "flow_validation_fraction": 0.2 * config.fit_fraction,
            "test_fraction": 1.0 - config.fit_fraction,
            "gkr": {
                "kernel_parameterization": "log-lambda",
                "bandwidth_initialization": (
                    "participation-ratio target effective sample size"
                ),
                "neighbors_per_effective_dimension": 5.0,
                "covariance_epochs": 100,
                "mean_learning_rate": 0.01,
                "covariance_learning_rate": 0.01,
                "standardize_responses": True,
            },
            "session_labels": labels,
            "session_mean_log_likelihood": {
                method: values[:, index]
                for index, method in enumerate(METHOD_LABELS)
            },
            "session_log_likelihood_relative_to_bin_lw": {
                method: _relative_to_bin_lw(values)[:, index]
                for index, method in enumerate(METHOD_LABELS)
            },
            "across_session_mean": {
                method: means[index]
                for index, method in enumerate(METHOD_LABELS)
            },
            "across_session_sem": {
                method: sem[index]
                for index, method in enumerate(METHOD_LABELS)
            },
            "across_session_mean_relative_to_bin_lw": {
                method: np.mean(_relative_to_bin_lw(values), axis=0)[index]
                for index, method in enumerate(METHOD_LABELS)
            },
            "across_session_sem_relative_to_bin_lw": {
                method: (
                    np.std(_relative_to_bin_lw(values), axis=0, ddof=1)
                    / np.sqrt(values.shape[0])
                )[index]
                for index, method in enumerate(METHOD_LABELS)
            },
        }
    png, svg = _plot(results, output_dir=output_dir)
    summary = {
        "comparison": (
            "Each panel changes only outer fit fraction or PCA dimension relative "
            "to the original 80%-fit, PCA-82 protocol. The figure shows paired "
            "session-wise test log likelihood differences relative to Bin+LW."
        ),
        "likelihood_comparability": (
            "Method rankings are comparable within every panel. Absolute joint "
            "log likelihoods are not directly comparable across PCA dimensions."
        ),
        "configurations": summary_configs,
        "artifacts": {"png": png, "svg": svg},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved: {summary_path}", flush=True)
    print(f"Saved: {png}", flush=True)
    print(f"Saved: {svg}", flush=True)
    return summary


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    configs = _selected_configs(args.configs)
    session_indices = [int(index) for index in args.session_indices]
    if sorted(set(session_indices)) != sorted(session_indices):
        raise ValueError("session-indices must be unique.")
    if any(index < 0 or index >= 6 for index in session_indices):
        raise ValueError("session-indices must be between 0 and 5.")
    if not args.aggregate_only:
        if not args.devices:
            raise ValueError("--devices is required unless --aggregate-only is used.")
        devices = list(args.devices)
        tasks = [
            (config, session_index)
            for config in configs
            for session_index in session_indices
        ]
        assigned = [tasks[index :: len(devices)] for index in range(len(devices))]
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [
                executor.submit(
                    _run_worker,
                    device,
                    device_tasks,
                    output_dir=output_dir,
                    force=bool(args.force),
                    refit_gkr_only=bool(args.refit_gkr_only),
                    gkr_covariance_epochs=int(args.gkr_covariance_epochs),
                    gkr_mean_learning_rate=float(args.gkr_mean_learning_rate),
                    gkr_covariance_learning_rate=float(
                        args.gkr_covariance_learning_rate
                    ),
                )
                for device, device_tasks in zip(devices, assigned, strict=True)
            ]
            for future in futures:
                future.result()
    if (
        len(configs) == len(CONFIGS)
        and session_indices == list(range(6))
    ):
        _aggregate(configs, output_dir=output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
