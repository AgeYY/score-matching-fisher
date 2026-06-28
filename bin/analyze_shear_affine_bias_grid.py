#!/usr/bin/env python3
"""Fast Gaussian-affine bias diagnostic for the hidden-shear dataset.

This script does not train flow-matching networks.  It estimates the affine
Gaussian baseline directly from sample means and covariances, which isolates the
finite-sample behavior of the affine model family from optimizer noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
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

from global_setting import DATA_DIR
from fisher.shear_rank_dataset import centered_cosine_nu, generate_shear_rank_dataset


@dataclass(frozen=True)
class DatasetConfig:
    x_dim: int
    r_star: int
    amplitude: float
    mean_shift: float

    @property
    def label(self) -> str:
        return f"d={self.x_dim}, r*={self.r_star}, A={self.amplitude:g}, m={self.mean_shift:g}"


def _parse_int_list(value: str) -> list[int]:
    vals = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer.")
    return vals


def _parse_float_list(value: str) -> list[float]:
    vals = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float.")
    return vals


def _cov_mle(x: np.ndarray) -> np.ndarray:
    xc = np.asarray(x, dtype=np.float64) - np.mean(x, axis=0, keepdims=True)
    return (xc.T @ xc) / float(max(1, x.shape[0]))


def gaussian_symmetric_kl(
    x0: np.ndarray,
    x1: np.ndarray,
    *,
    jitter: float,
) -> float:
    """Jeffreys KL between plug-in Gaussian fits to two samples."""

    a = np.asarray(x0, dtype=np.float64)
    b = np.asarray(x1, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("x0 and x1 must be two-dimensional arrays with matching feature dimension.")
    d = int(a.shape[1])
    mu0 = np.mean(a, axis=0)
    mu1 = np.mean(b, axis=0)
    s0 = _cov_mle(a) + float(jitter) * np.eye(d, dtype=np.float64)
    s1 = _cov_mle(b) + float(jitter) * np.eye(d, dtype=np.float64)
    delta = (mu1 - mu0).reshape(d, 1)
    inv0 = np.linalg.solve(s0, np.eye(d, dtype=np.float64))
    inv1 = np.linalg.solve(s1, np.eye(d, dtype=np.float64))
    mean_term = float((delta.T @ (inv0 + inv1) @ delta).item())
    value = 0.5 * (
        np.trace(inv1 @ s0)
        + np.trace(inv0 @ s1)
        + mean_term
        - 2.0 * float(d)
    )
    return float(max(0.0, value))


def _format_float_for_path(value: float) -> str:
    text = f"{float(value):g}".replace("-", "neg").replace(".", "p")
    return text


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path(DATA_DIR) / "shear_affine_bias_grid")
    p.add_argument("--force", action="store_true")
    p.add_argument("--n-list", type=_parse_int_list, default=[50, 100, 200, 500, 1000, 2000, 5000])
    p.add_argument("--n-seeds", type=int, default=200)
    p.add_argument("--seed", type=int, default=7000)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--omega", type=float, default=2.5)
    p.add_argument("--q-seed", type=int, default=12345)
    p.add_argument("--jitter", type=float, default=1e-6)
    p.add_argument(
        "--split",
        choices=("train", "all"),
        default="train",
        help="Estimate plug-in Gaussians from the train split or all generated samples.",
    )
    p.add_argument(
        "--config",
        action="append",
        default=[],
        help=(
            "Dataset config as x_dim,r_star,amplitude,mean_shift. "
            "May be repeated. If omitted, a default diagnostic grid is used."
        ),
    )
    p.add_argument("--workers", type=int, default=1)
    return p


def _default_configs() -> list[DatasetConfig]:
    return [
        DatasetConfig(x_dim=8, r_star=0, amplitude=0.0, mean_shift=6.0),
        DatasetConfig(x_dim=2, r_star=2, amplitude=0.7, mean_shift=0.0),
        DatasetConfig(x_dim=8, r_star=2, amplitude=0.7, mean_shift=0.0),
        DatasetConfig(x_dim=8, r_star=2, amplitude=0.7, mean_shift=1.2),
        DatasetConfig(x_dim=8, r_star=2, amplitude=0.7, mean_shift=6.0),
        DatasetConfig(x_dim=8, r_star=2, amplitude=2.0, mean_shift=0.0),
        DatasetConfig(x_dim=8, r_star=2, amplitude=2.0, mean_shift=1.2),
        DatasetConfig(x_dim=8, r_star=2, amplitude=2.0, mean_shift=6.0),
        DatasetConfig(x_dim=16, r_star=2, amplitude=2.0, mean_shift=6.0),
    ]


def _parse_configs(values: list[str]) -> list[DatasetConfig]:
    if not values:
        return _default_configs()
    configs: list[DatasetConfig] = []
    for raw in values:
        parts = [part.strip() for part in str(raw).split(",")]
        if len(parts) != 4:
            raise argparse.ArgumentTypeError("--config must have four comma-separated fields.")
        configs.append(
            DatasetConfig(
                x_dim=int(parts[0]),
                r_star=int(parts[1]),
                amplitude=float(parts[2]),
                mean_shift=float(parts[3]),
            )
        )
    return configs


def _case_rows(task: tuple[DatasetConfig, int, int, argparse.Namespace]) -> list[dict[str, Any]]:
    cfg, n_per_condition, repeat_idx, args = task
    seed = int(args.seed) + int(repeat_idx)
    dataset = generate_shear_rank_dataset(
        n_per_condition=int(n_per_condition),
        x_dim=int(cfg.x_dim),
        r_star=int(cfg.r_star),
        amplitude=float(cfg.amplitude),
        mean_shift=float(cfg.mean_shift),
        omega=float(args.omega),
        seed=seed,
        q_seed=int(args.q_seed),
        train_frac=float(args.train_frac),
        mode="sign_flip",
    )
    bundle = dataset.bundle
    if str(args.split) == "train":
        idx = bundle.train_idx
    else:
        idx = np.arange(bundle.x_all.shape[0], dtype=np.int64)
    labels = np.argmax(bundle.theta_all[idx], axis=1)
    x = bundle.x_all[idx]
    x0 = x[labels == 0]
    x1 = x[labels == 1]
    estimate = gaussian_symmetric_kl(x0, x1, jitter=float(args.jitter))
    true_skl = float(dataset.true_skl_matrix[0, 1])
    nonlinear_skl = float(dataset.nu) * 4.0 * float(cfg.amplitude) ** 2 * (float(cfg.r_star) / 2.0)
    population_affine_skl = float(cfg.mean_shift) ** 2
    return [
        {
            "x_dim": int(cfg.x_dim),
            "r_star": int(cfg.r_star),
            "amplitude": float(cfg.amplitude),
            "mean_shift": float(cfg.mean_shift),
            "config_label": cfg.label,
            "n_per_condition": int(n_per_condition),
            "repeat_idx": int(repeat_idx),
            "seed": seed,
            "split": str(args.split),
            "n_fit_per_condition": int(x0.shape[0]),
            "estimate_skl": estimate,
            "true_skl": true_skl,
            "population_affine_skl": population_affine_skl,
            "nonlinear_skl": nonlinear_skl,
            "bias_vs_true": estimate - true_skl,
            "bias_vs_population_affine": estimate - population_affine_skl,
            "relative_error": abs(estimate - true_skl) / true_skl if true_skl > 0.0 else float("nan"),
        }
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "x_dim",
        "r_star",
        "amplitude",
        "mean_shift",
        "config_label",
        "n_per_condition",
        "repeat_idx",
        "seed",
        "split",
        "n_fit_per_condition",
        "estimate_skl",
        "true_skl",
        "population_affine_skl",
        "nonlinear_skl",
        "bias_vs_true",
        "bias_vs_population_affine",
        "relative_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            int(row["x_dim"]),
            int(row["r_star"]),
            float(row["amplitude"]),
            float(row["mean_shift"]),
            str(row["config_label"]),
            int(row["n_per_condition"]),
        )
        grouped.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for key, vals in grouped.items():
        biases = np.asarray([float(v["bias_vs_true"]) for v in vals], dtype=np.float64)
        est = np.asarray([float(v["estimate_skl"]) for v in vals], dtype=np.float64)
        rel = np.asarray([float(v["relative_error"]) for v in vals], dtype=np.float64)
        n = int(len(vals))
        summary.append(
            {
                "x_dim": key[0],
                "r_star": key[1],
                "amplitude": key[2],
                "mean_shift": key[3],
                "config_label": key[4],
                "n_per_condition": key[5],
                "n_repeats": n,
                "mean_estimate_skl": float(np.mean(est)),
                "sem_estimate_skl": float(np.std(est, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
                "true_skl": float(vals[0]["true_skl"]),
                "population_affine_skl": float(vals[0]["population_affine_skl"]),
                "nonlinear_skl": float(vals[0]["nonlinear_skl"]),
                "mean_bias_vs_true": float(np.mean(biases)),
                "sd_bias_vs_true": float(np.std(biases, ddof=1)) if n > 1 else 0.0,
                "sem_bias_vs_true": float(np.std(biases, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
                "positive_bias_fraction": float(np.mean(biases > 0.0)),
                "mean_relative_error": float(np.mean(rel)),
                "sd_relative_error": float(np.std(rel, ddof=1)) if n > 1 else 0.0,
                "sem_relative_error": float(np.std(rel, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
            }
        )
    summary.sort(
        key=lambda r: (
            int(r["x_dim"]),
            int(r["r_star"]),
            float(r["amplitude"]),
            float(r["mean_shift"]),
            int(r["n_per_condition"]),
        )
    )
    return summary


def _write_summary_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "x_dim",
        "r_star",
        "amplitude",
        "mean_shift",
        "config_label",
        "n_per_condition",
        "n_repeats",
        "mean_estimate_skl",
        "sem_estimate_skl",
        "true_skl",
        "population_affine_skl",
        "nonlinear_skl",
        "mean_bias_vs_true",
        "sd_bias_vs_true",
        "sem_bias_vs_true",
        "positive_bias_fraction",
        "mean_relative_error",
        "sd_relative_error",
        "sem_relative_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def _plot_summary(path: Path, summary: list[dict[str, Any]]) -> None:
    configs = []
    seen = set()
    for row in summary:
        label = str(row["config_label"])
        if label not in seen:
            configs.append(label)
            seen.add(label)
    block_cols = 3
    n_rows = int(math.ceil(len(configs) / block_cols))
    n_cols = 2 * block_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.1 * n_cols, 3.5 * n_rows), squeeze=False)
    by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in configs}
    for row in summary:
        by_label[str(row["config_label"])].append(row)
    for idx, label in enumerate(configs):
        row_idx = idx // block_cols
        col_idx = idx % block_cols
        bias_ax = axes[row_idx, col_idx]
        rel_ax = axes[row_idx, col_idx + block_cols]
        vals = sorted(by_label[label], key=lambda r: int(r["n_per_condition"]))
        n = np.asarray([int(v["n_per_condition"]) for v in vals], dtype=np.float64)
        bias = np.asarray([float(v["mean_bias_vs_true"]) for v in vals], dtype=np.float64)
        bias_sd = np.asarray([float(v.get("sd_bias_vs_true", v["sem_bias_vs_true"])) for v in vals], dtype=np.float64)
        frac = np.asarray([float(v["positive_bias_fraction"]) for v in vals], dtype=np.float64)
        rel = np.asarray([float(v["mean_relative_error"]) for v in vals], dtype=np.float64)
        rel_sd = np.asarray([float(v.get("sd_relative_error", v.get("sem_relative_error", 0.0))) for v in vals], dtype=np.float64)
        pop_bias = float(vals[0]["population_affine_skl"]) - float(vals[0]["true_skl"])
        asymptotic_rel = abs(pop_bias) / float(vals[0]["true_skl"]) if float(vals[0]["true_skl"]) > 0.0 else np.nan

        bias_ax.axhline(0.0, color="0.2", linewidth=1.0)
        bias_ax.axhline(pop_bias, color="#8c564b", linewidth=1.0, linestyle="--", label="population affine bias")
        bias_ax.errorbar(n, bias, yerr=bias_sd, marker="o", linewidth=1.6, capsize=2.5, color="#1f77b4")
        bias_ax.set_xscale("log")
        bias_ax.set_title(label, fontsize=10)
        bias_ax.set_xlabel("N per condition")
        if col_idx == 0:
            bias_ax.set_ylabel("Bias vs true SKL\nmean +/- SD")
        bias_ax.grid(True, which="major", alpha=0.25)
        ax2 = bias_ax.twinx()
        ax2.plot(n, frac, marker="s", linewidth=1.0, color="#ff7f0e", alpha=0.75)
        ax2.set_ylim(-0.05, 1.05)
        if col_idx == block_cols - 1:
            ax2.set_ylabel("P(bias > 0)", color="#ff7f0e")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")

        rel_ax.axhline(0.0, color="0.2", linewidth=1.0)
        if np.isfinite(asymptotic_rel):
            rel_ax.axhline(
                asymptotic_rel,
                color="#8c564b",
                linewidth=1.0,
                linestyle="--",
                label="population affine rel. error",
            )
        rel_ax.errorbar(n, rel, yerr=rel_sd, marker="o", linewidth=1.6, capsize=2.5, color="#2ca02c")
        rel_ax.set_xscale("log")
        rel_ax.set_ylim(bottom=0.0)
        rel_ax.set_title(label, fontsize=10)
        rel_ax.set_xlabel("N per condition")
        if col_idx == 0:
            rel_ax.set_ylabel("Relative abs error\nmean +/- SD")
        rel_ax.grid(True, which="major", alpha=0.25)
    for idx in range(len(configs), n_rows * block_cols):
        row_idx = idx // block_cols
        col_idx = idx % block_cols
        axes[row_idx, col_idx].axis("off")
        axes[row_idx, col_idx + block_cols].axis("off")
    fig.text(0.25, 0.965, "Bias vs true SKL", ha="center", va="top", fontsize=13, weight="bold")
    fig.text(0.75, 0.965, "Relative absolute error vs N", ha="center", va="top", fontsize=13, weight="bold")
    fig.suptitle("Gaussian-affine plug-in diagnostics across hidden-shear parameterizations", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0.02, 0.0, 0.985, 0.935), w_pad=2.2, h_pad=2.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    configs = _parse_configs(args.config)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = output_dir / "affine_gaussian_bias_rows.csv"
    summary_csv = output_dir / "affine_gaussian_bias_summary.csv"
    summary_json = output_dir / "affine_gaussian_bias_summary.json"
    figure_png = output_dir / "affine_gaussian_bias_grid.png"

    if rows_csv.is_file() and summary_csv.is_file() and summary_json.is_file() and not bool(args.force):
        print(f"[affine-bias] Reusing existing outputs in {output_dir}")
        return

    tasks = [
        (cfg, int(n), int(repeat_idx), args)
        for cfg in configs
        for n in args.n_list
        for repeat_idx in range(int(args.n_seeds))
    ]
    print(
        f"[affine-bias] Running {len(tasks)} cases: configs={len(configs)} "
        f"N={list(args.n_list)} seeds={int(args.n_seeds)} split={args.split}",
        flush=True,
    )
    if int(args.workers) > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=int(args.workers)) as pool:
            nested = pool.map(_case_rows, tasks)
    else:
        nested = [_case_rows(task) for task in tasks]
    rows = [row for part in nested for row in part]
    rows.sort(
        key=lambda r: (
            int(r["x_dim"]),
            int(r["r_star"]),
            float(r["amplitude"]),
            float(r["mean_shift"]),
            int(r["n_per_condition"]),
            int(r["repeat_idx"]),
        )
    )
    summary = _summarize(rows)
    _write_csv(rows_csv, rows)
    _write_summary_csv(summary_csv, summary)
    _plot_summary(figure_png, summary)

    n_values = sorted({int(row["n_per_condition"]) for row in rows})
    smallest_n = min(n_values)
    largest_n = max(n_values)
    sign_table = []
    for cfg in configs:
        cfg_rows = [
            row
            for row in summary
            if int(row["x_dim"]) == cfg.x_dim
            and int(row["r_star"]) == cfg.r_star
            and float(row["amplitude"]) == float(cfg.amplitude)
            and float(row["mean_shift"]) == float(cfg.mean_shift)
        ]
        small = next(row for row in cfg_rows if int(row["n_per_condition"]) == smallest_n)
        large = next(row for row in cfg_rows if int(row["n_per_condition"]) == largest_n)
        sign_table.append(
            {
                "config": cfg.label,
                "true_skl": float(small["true_skl"]),
                "population_affine_skl": float(small["population_affine_skl"]),
                "nonlinear_skl": float(small["nonlinear_skl"]),
                f"mean_bias_N{smallest_n}": float(small["mean_bias_vs_true"]),
                f"positive_fraction_N{smallest_n}": float(small["positive_bias_fraction"]),
                f"mean_bias_N{largest_n}": float(large["mean_bias_vs_true"]),
                f"positive_fraction_N{largest_n}": float(large["positive_bias_fraction"]),
            }
        )

    meta = {
        "script": "bin/analyze_shear_affine_bias_grid.py",
        "output_dir": str(output_dir),
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
        "figure_png": str(figure_png),
        "figure_svg": str(figure_png.with_suffix(".svg")),
        "n_list": [int(v) for v in args.n_list],
        "n_seeds": int(args.n_seeds),
        "seed": int(args.seed),
        "split": str(args.split),
        "train_frac": float(args.train_frac),
        "omega": float(args.omega),
        "nu": centered_cosine_nu(float(args.omega)),
        "jitter": float(args.jitter),
        "configs": [cfg.__dict__ for cfg in configs],
        "sign_table": sign_table,
    }
    summary_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[affine-bias] Wrote rows: {rows_csv}")
    print(f"[affine-bias] Wrote summary: {summary_csv}")
    print(f"[affine-bias] Wrote figure: {figure_png}")
    print("[affine-bias] Sign table:")
    for row in sign_table:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
