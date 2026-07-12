#!/usr/bin/env python3
"""Investigate MoG5 Jeffreys convergence and NLL tuning at N=10000."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.distance_comparison import (
    FlowComparisonConfig,
    METRIC_SYMMETRIC_KL,
    _build_and_train_flow_model,
    _estimate_trained_flow,
    _seed_flow_rng,
    classical_metric_matrices,
    labels_from_theta,
    native_mog_ground_truth_matrices,
)
from fisher.flow_matching_skl import build_flow_skl_model, finetune_flow_skl_cnf_likelihood
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import require_device
from global_setting import DATA_DIR


@dataclass(frozen=True)
class NLLConfig:
    key: str
    label: str
    epochs: int = 500
    batch_size: int = 2048
    lr: float = 3e-5
    weight_decay: float = 0.0
    ode_steps: int = 32
    patience: int = 150
    min_delta: float = 1e-4
    ema_alpha: float = 0.05


NLL_CONFIGS = (
    NLLConfig("baseline", "Baseline", lr=3e-5, batch_size=2048),
    NLLConfig("lr1e5", "LR 1e-5", lr=1e-5, batch_size=2048),
    NLLConfig("lr3e6", "LR 3e-6", lr=3e-6, batch_size=2048),
    NLLConfig("fullbatch_lr1e5", "Full batch, LR 1e-5", lr=1e-5, batch_size=10000),
    NLLConfig("wd1e4_lr1e5", "WD 1e-4, LR 1e-5", lr=1e-5, batch_size=2048, weight_decay=1e-4),
    NLLConfig("wd1e2_lr1e5", "WD 1e-2, LR 1e-5", lr=1e-5, batch_size=2048, weight_decay=1e-2),
    NLLConfig("steps64_lr1e5", "64 steps, LR 1e-5", lr=1e-5, batch_size=2048, ode_steps=64),
    NLLConfig(
        "responsive_lr1e5",
        "Responsive monitor, LR 1e-5",
        lr=1e-5,
        batch_size=2048,
        min_delta=0.0,
        ema_alpha=1.0,
    ),
    NLLConfig(
        "responsive_lr3e6",
        "Responsive monitor, LR 3e-6",
        lr=3e-6,
        batch_size=2048,
        min_delta=0.0,
        ema_alpha=1.0,
    ),
)


def _parse_int_list(value: str) -> list[int]:
    out = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer.")
    return out


def _parse_str_list(value: str) -> list[str]:
    out = [part.strip() for part in str(value).split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated name.")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-total", type=int, default=10_000)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--repeat-indices", type=_parse_int_list, default=[0, 1, 2, 3, 4])
    parser.add_argument("--nll-configs", type=_parse_str_list, default=["baseline"])
    parser.add_argument("--fm-epochs", type=int, default=20_000)
    parser.add_argument("--fm-batch-size", type=int, default=3000)
    parser.add_argument("--fm-lr", type=float, default=1e-3)
    parser.add_argument("--fm-min-lr", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--mc-jeffreys-sample", type=int, default=4096)
    parser.add_argument("--ode-steps", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--force-fm", action="store_true")
    parser.add_argument("--force-nll", action="store_true")
    parser.add_argument("--skip-dataset-generation", action="store_true")
    parser.add_argument(
        "--template-npz",
        type=Path,
        default=Path(DATA_DIR) / "mog5_seed7_jeffreys_template_n100000" / "random_mog_categorical.npz",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(DATA_DIR) / "mog5_seed7_jeffreys_n10000_cases",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "mog5_seed7_jeffreys_n10000_investigation",
    )
    parser.add_argument(
        "--prior-errors-csv",
        type=Path,
        default=Path(DATA_DIR)
        / "mog5_seed7_jeffreys_unified_n100-3000_r5"
        / "mog5_pr_distance_sweep_errors.csv",
    )
    return parser


def _config_map() -> dict[str, NLLConfig]:
    return {config.key: config for config in NLL_CONFIGS}


def _fm_config(args: argparse.Namespace) -> FlowComparisonConfig:
    return FlowComparisonConfig(
        epochs=int(args.fm_epochs),
        early_patience=0,
        early_min_delta=1e-4,
        early_ema_alpha=0.05,
        batch_size=int(args.fm_batch_size),
        lr=float(args.fm_lr),
        lr_schedule="cosine",
        min_lr=float(args.fm_min_lr),
        weight_decay=0.0,
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        network_architecture="mlp",
        path_schedule="cosine",
        t_eps=5e-4,
        quadrature_steps=64,
        mc_jeffreys_sample=int(args.mc_jeffreys_sample),
        ode_steps=int(args.ode_steps),
        ode_method="midpoint",
        divergence_estimator="exact",
        max_grad_norm=10.0,
        log_every=int(args.log_every),
        checkpoint_selection="best",
        fixed_validation=True,
        likelihood_finetune_epochs=0,
    )


def _dataset_path(args: argparse.Namespace, repeat_idx: int) -> Path:
    return Path(args.dataset_root) / f"repeat_{int(repeat_idx):02d}" / "random_mog_categorical.npz"


def _ensure_dataset(args: argparse.Namespace, repeat_idx: int) -> Path:
    path = _dataset_path(args, repeat_idx)
    if path.is_file():
        return path
    if bool(args.skip_dataset_generation):
        raise FileNotFoundError(f"Missing dataset with generation disabled: {path}")
    output_dir = path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(_REPO_ROOT / "bin" / "make_mog5_pr_dataset.py"),
        "--n-total",
        str(int(args.n_total)),
        "--native-x-dim",
        "3",
        "--pr-dim",
        "none",
        "--seed",
        str(int(args.seed) + int(repeat_idx)),
        "--train-frac",
        "0.8",
        "--native-template-npz",
        str(Path(args.template_npz)),
        "--output-dir",
        str(output_dir),
        "--device",
        str(args.device),
        "--skip-viz",
    ]
    subprocess.run(command, cwd=_REPO_ROOT, check=True)
    if not path.is_file():
        raise RuntimeError(f"Dataset generation did not create {path}")
    return path


def _mean_pair_error(estimate: np.ndarray, reference: np.ndarray, *, relative: bool) -> float:
    rows, cols = np.triu_indices(int(reference.shape[0]), k=1)
    error = np.abs(np.asarray(estimate)[rows, cols] - np.asarray(reference)[rows, cols])
    if relative:
        error = error / np.maximum(np.abs(np.asarray(reference)[rows, cols]), 1e-12)
    return float(np.mean(error))


def _plain_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key != "best_state_dict" and isinstance(value, (str, int, float, bool, np.ndarray))
    }


def _build_model(args: argparse.Namespace, theta_dim: int, x_dim: int, device: torch.device) -> torch.nn.Module:
    return build_flow_skl_model(
        velocity_family="nonlinear",
        theta_dim=int(theta_dim),
        x_dim=int(x_dim),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        path_schedule="cosine",
        divergence_estimator="exact",
    ).to(device)


def _load_or_train_fm(
    *,
    args: argparse.Namespace,
    bundle: Any,
    repeat_idx: int,
    repeat_dir: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], float]:
    checkpoint_path = repeat_dir / "fm_best_checkpoint.pt"
    config = _fm_config(args)
    model = _build_model(args, bundle.theta_train.shape[1], bundle.x_train.shape[1], device)
    if checkpoint_path.is_file() and not bool(args.force_fm):
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("config") != asdict(config):
            raise ValueError(f"Cached FM configuration mismatch: {checkpoint_path}")
        model.load_state_dict(saved["state_dict"])
        return model, saved["metadata"], 0.0

    start = time.perf_counter()
    model, metadata = _build_and_train_flow_model(
        theta_train=bundle.theta_train,
        x_train=bundle.x_train,
        theta_val=bundle.theta_validation,
        x_val=bundle.x_validation,
        velocity_family="nonlinear",
        device=device,
        seed=int(args.seed) + int(repeat_idx),
        config=config,
    )
    runtime = time.perf_counter() - start
    plain = _plain_metadata(metadata)
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "metadata": plain,
            "config": asdict(config),
        },
        checkpoint_path,
    )
    return model, plain, runtime


def _estimate(
    *,
    model: torch.nn.Module,
    metadata: dict[str, Any],
    args: argparse.Namespace,
    repeat_idx: int,
    device: torch.device,
) -> np.ndarray:
    result = _estimate_trained_flow(
        model=model,
        theta_eval=np.eye(5, dtype=np.float64),
        velocity_family="nonlinear",
        device=device,
        seed=int(args.seed) + int(repeat_idx) + 100_000,
        config=_fm_config(args),
        train_metadata=metadata,
    )
    return np.asarray(result.symmetric_kl_matrix, dtype=np.float64)


def _load_or_run_nll(
    *,
    config: NLLConfig,
    fm_state: dict[str, torch.Tensor],
    fm_metadata: dict[str, Any],
    args: argparse.Namespace,
    bundle: Any,
    repeat_idx: int,
    repeat_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any], float]:
    path = repeat_dir / f"nll_{config.key}.npz"
    if path.is_file() and not bool(args.force_nll):
        with np.load(path, allow_pickle=False) as saved:
            return (
                np.asarray(saved["matrix"], dtype=np.float64),
                {
                    "best_epoch": int(saved["best_epoch"][0]),
                    "selected_epoch": int(saved["selected_epoch"][0]),
                    "stopped_epoch": int(saved["stopped_epoch"][0]),
                    "best_val_nll": float(saved["best_val_nll"][0]),
                    "initial_val_nll": float(saved["initial_val_nll"][0])
                    if "initial_val_nll" in saved
                    else float("nan"),
                },
                0.0,
            )

    model = _build_model(args, bundle.theta_train.shape[1], bundle.x_train.shape[1], device)
    model.load_state_dict(fm_state)
    _seed_flow_rng(int(args.seed) + int(repeat_idx) + 200_000, device)
    start = time.perf_counter()
    metadata = finetune_flow_skl_cnf_likelihood(
        model=model,
        theta_train=bundle.theta_train,
        x_train=bundle.x_train,
        theta_val=bundle.theta_validation,
        x_val=bundle.x_validation,
        device=device,
        epochs=int(config.epochs),
        batch_size=min(int(config.batch_size), int(bundle.x_train.shape[0])),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
        ode_steps=int(config.ode_steps),
        ode_method="midpoint",
        patience=int(config.patience),
        min_delta=float(config.min_delta),
        ema_alpha=float(config.ema_alpha),
        max_grad_norm=10.0,
        checkpoint_selection="best",
        log_every=int(args.log_every),
    )
    runtime = time.perf_counter() - start
    matrix = _estimate(
        model=model,
        metadata={**fm_metadata, "likelihood_finetune_metadata": metadata},
        args=args,
        repeat_idx=repeat_idx,
        device=device,
    )
    np.savez_compressed(
        path,
        matrix=matrix,
        train_nll_losses=np.asarray(metadata["train_nll_losses"], dtype=np.float64),
        val_nll_losses=np.asarray(metadata["val_nll_losses"], dtype=np.float64),
        val_monitor_nll_losses=np.asarray(metadata["val_monitor_nll_losses"], dtype=np.float64),
        best_epoch=np.asarray([metadata["best_epoch"]], dtype=np.int64),
        selected_epoch=np.asarray([metadata["selected_epoch"]], dtype=np.int64),
        stopped_epoch=np.asarray([metadata["stopped_epoch"]], dtype=np.int64),
        best_val_nll=np.asarray([metadata["best_val_nll"]], dtype=np.float64),
        initial_val_nll=np.asarray([metadata["initial_val_nll"]], dtype=np.float64),
    )
    return matrix, metadata, runtime


def _plot(summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    variants = summary.sort_values("mean_mrae")
    with plt.rc_context(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 13,
            "axes.grid": False,
        }
    ):
        fig, axis = plt.subplots(figsize=(8.0, 5.0))
        y = np.arange(len(variants))
        axis.errorbar(
            variants["mean_mrae"],
            y,
            xerr=variants["std_mrae"],
            fmt="o",
            color="#4C72B0",
            ecolor="black",
            capsize=3,
        )
        axis.set_yticks(y)
        axis.set_yticklabels(
            [
                f"{label} (r={int(count)})"
                for label, count in zip(variants["variant_label"], variants["n_repeats"])
            ]
        )
        axis.set_xlabel("Mean relative absolute error")
        axis.set_title("MoG5 Jeffreys divergence, N=10000")
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_linewidth(1.8)
        axis.tick_params(width=1.8)
        fig.tight_layout()
    png = output_dir / "jeffreys_n10000_nll_config_comparison.png"
    svg = output_dir / "jeffreys_n10000_nll_config_comparison.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


def _write_trend_artifacts(
    *,
    prior_errors_csv: Path,
    current_rows: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    prior = pd.read_csv(prior_errors_csv)
    prior = prior[prior["metric"].eq(METRIC_SYMMETRIC_KL)].copy()
    prior_repeat = (
        prior.groupby(["n_total", "repeat_idx", "estimator"], as_index=False)
        .agg(mae=("abs_error", "mean"), mrae=("rel_error", "mean"))
    )
    current_map = {
        "classical": "classical",
        "flow_matching": "flow_matching",
        "nll_baseline": "flow_matching_nll_finetuned",
    }
    current = current_rows[current_rows["variant"].isin(current_map)].copy()
    current["estimator"] = current["variant"].map(current_map)
    current["n_total"] = 10_000
    combined = pd.concat(
        [
            prior_repeat[["n_total", "repeat_idx", "estimator", "mae", "mrae"]],
            current[["n_total", "repeat_idx", "estimator", "mae", "mrae"]],
        ],
        ignore_index=True,
    )
    trend = (
        combined.groupby(["n_total", "estimator"], as_index=False)
        .agg(
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_mrae=("mrae", "mean"),
            std_mrae=("mrae", "std"),
            n_repeats=("repeat_idx", "nunique"),
        )
        .sort_values(["estimator", "n_total"])
    )
    csv_path = output_dir / "jeffreys_n100_to_n10000_trend.csv"
    trend.to_csv(csv_path, index=False)

    labels = {
        "classical": "Classical",
        "flow_matching": "Flow matching",
        "flow_matching_nll_finetuned": "Flow matching + NLL",
    }
    colors = {
        "classical": "#4C72B0",
        "flow_matching": "#DD8452",
        "flow_matching_nll_finetuned": "#55A868",
    }
    with plt.rc_context(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "axes.grid": False,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))
        for estimator, group in trend.groupby("estimator"):
            group = group.sort_values("n_total")
            for axis, mean_key, std_key, ylabel in (
                (axes[0], "mean_mrae", "std_mrae", "MRAE"),
                (axes[1], "mean_mae", "std_mae", "MAE"),
            ):
                axis.errorbar(
                    group["n_total"],
                    group[mean_key],
                    yerr=group[std_key],
                    marker="o",
                    capsize=3,
                    linewidth=1.8,
                    color=colors[estimator],
                    label=labels[estimator],
                )
                axis.set_xscale("log")
                axis.set_xlabel("Sample size $N$")
                axis.set_ylabel(ylabel)
                axis.grid(False)
                for spine in axis.spines.values():
                    spine.set_linewidth(1.8)
                axis.tick_params(width=1.8)
        axes[0].legend(frameon=False)
        fig.tight_layout()
    png_path = output_dir / "jeffreys_n100_to_n10000_trend.png"
    svg_path = output_dir / "jeffreys_n100_to_n10000_trend.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return csv_path, png_path, svg_path


def _write_common_screen_summary(current: pd.DataFrame, output_dir: Path) -> Path:
    tuned_variants = sorted(
        variant
        for variant in current["variant"].unique()
        if variant.startswith("nll_") and variant != "nll_baseline"
    )
    variants = ["classical", "flow_matching", "nll_baseline", *tuned_variants]
    repeat_sets = [
        set(current.loc[current["variant"].eq(variant), "repeat_idx"].astype(int))
        for variant in variants
    ]
    common_repeats = sorted(set.intersection(*repeat_sets)) if repeat_sets else []
    if not common_repeats:
        raise ValueError("NLL screen configurations have no common repeats.")
    common = current[
        current["variant"].isin(variants) & current["repeat_idx"].isin(common_repeats)
    ]
    summary = (
        common.groupby(["variant", "variant_label"], as_index=False)
        .agg(
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_mrae=("mrae", "mean"),
            std_mrae=("mrae", "std"),
            mean_selected_epoch=("selected_epoch", "mean"),
            n_repeats=("repeat_idx", "nunique"),
        )
        .sort_values("mean_mrae")
    )
    path = output_dir / "jeffreys_n10000_nll_screen_common_repeats.csv"
    summary.to_csv(path, index=False)
    return path


def _plot_baseline_nll_losses(output_dir: Path) -> tuple[Path, Path]:
    histories: list[tuple[np.ndarray, np.ndarray, float]] = []
    for path in sorted(output_dir.glob("repeat_*/nll_baseline.npz")):
        with np.load(path) as saved:
            histories.append(
                (
                    np.asarray(saved["train_nll_losses"], dtype=np.float64),
                    np.asarray(saved["val_nll_losses"], dtype=np.float64),
                    float(saved["initial_val_nll"][0]),
                )
            )
    if not histories:
        raise FileNotFoundError("No baseline NLL histories are available.")
    length = min(min(len(train), len(val)) for train, val, _ in histories)
    train_stack = np.stack([train[:length] for train, _, _ in histories])
    val_stack = np.stack(
        [np.concatenate([[initial], val[:length]]) for _, val, initial in histories]
    )
    with plt.rc_context(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 13,
            "axes.grid": False,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))
        for values in train_stack:
            axes[0].plot(np.arange(1, length + 1), values, color="#999999", alpha=0.35)
        axes[0].plot(
            np.arange(1, length + 1), train_stack.mean(axis=0), color="#4C72B0", linewidth=2.0
        )
        for values in val_stack:
            axes[1].plot(np.arange(length + 1), values, color="#999999", alpha=0.35)
        axes[1].plot(
            np.arange(length + 1), val_stack.mean(axis=0), color="#DD8452", linewidth=2.0
        )
        axes[0].set_ylabel("Training NLL")
        axes[1].set_ylabel("Validation NLL")
        for axis in axes:
            axis.set_xlabel("Fine-tuning epoch")
            axis.grid(False)
            for spine in axis.spines.values():
                spine.set_linewidth(1.8)
            axis.tick_params(width=1.8)
        fig.tight_layout()
    png_path = output_dir / "jeffreys_n10000_baseline_nll_losses.png"
    svg_path = output_dir / "jeffreys_n10000_baseline_nll_losses.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def run(args: argparse.Namespace) -> dict[str, Path]:
    device = require_device(str(args.device))
    config_map = _config_map()
    unknown = sorted(set(args.nll_configs) - set(config_map))
    if unknown:
        raise ValueError(f"Unknown NLL configuration(s): {unknown}; choices={sorted(config_map)}")
    if any(index < 0 or index >= int(args.n_repeats) for index in args.repeat_indices):
        raise ValueError("--repeat-indices must lie in [0, n_repeats).")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    template = load_shared_dataset_npz(Path(args.template_npz))
    if int(template.meta.get("seed", -1)) != int(args.seed):
        raise ValueError("Template seed does not match --seed.")
    ground_truth = native_mog_ground_truth_matrices(
        native_meta=dict(template.meta),
        samples_per_class=100_000,
        seed=int(args.seed) + 12_345,
        mahalanobis_ridge=1e-6,
        metrics=(METRIC_SYMMETRIC_KL,),
    )[METRIC_SYMMETRIC_KL]

    rows: list[dict[str, Any]] = []
    run_start = time.perf_counter()
    for repeat_idx in args.repeat_indices:
        dataset_path = _ensure_dataset(args, repeat_idx)
        bundle = load_shared_dataset_npz(dataset_path)
        repeat_dir = output_dir / f"repeat_{repeat_idx:02d}"
        repeat_dir.mkdir(parents=True, exist_ok=True)
        repeat_seed = int(args.seed) + int(repeat_idx)
        labels = labels_from_theta(bundle.theta_all, num_categories=5)
        classical = classical_metric_matrices(
            bundle.x_all,
            labels,
            num_categories=5,
            metrics=(METRIC_SYMMETRIC_KL,),
            mahalanobis_ridge=1e-6,
        )[METRIC_SYMMETRIC_KL]
        rows.append(
            {
                "repeat_idx": repeat_idx,
                "repeat_seed": repeat_seed,
                "variant": "classical",
                "variant_label": "Classical",
                "mae": _mean_pair_error(classical, ground_truth, relative=False),
                "mrae": _mean_pair_error(classical, ground_truth, relative=True),
                "selected_epoch": 0,
                "runtime_seconds": 0.0,
            }
        )

        print(f"[n10000] repeat={repeat_idx} seed={repeat_seed} FM", flush=True)
        model, fm_metadata, fm_runtime = _load_or_train_fm(
            args=args,
            bundle=bundle,
            repeat_idx=repeat_idx,
            repeat_dir=repeat_dir,
            device=device,
        )
        fm_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        fm_matrix_path = repeat_dir / "fm_matrix.npy"
        if fm_matrix_path.is_file() and not bool(args.force_fm):
            fm_matrix = np.load(fm_matrix_path)
        else:
            fm_matrix = _estimate(
                model=model,
                metadata=fm_metadata,
                args=args,
                repeat_idx=repeat_idx,
                device=device,
            )
            np.save(fm_matrix_path, fm_matrix)
        rows.append(
            {
                "repeat_idx": repeat_idx,
                "repeat_seed": repeat_seed,
                "variant": "flow_matching",
                "variant_label": "Flow matching",
                "mae": _mean_pair_error(fm_matrix, ground_truth, relative=False),
                "mrae": _mean_pair_error(fm_matrix, ground_truth, relative=True),
                "selected_epoch": int(fm_metadata["best_epoch"]),
                "runtime_seconds": fm_runtime,
            }
        )

        for key in args.nll_configs:
            config = config_map[key]
            print(f"[n10000] repeat={repeat_idx} seed={repeat_seed} NLL={key}", flush=True)
            matrix, metadata, runtime = _load_or_run_nll(
                config=config,
                fm_state=fm_state,
                fm_metadata=fm_metadata,
                args=args,
                bundle=bundle,
                repeat_idx=repeat_idx,
                repeat_dir=repeat_dir,
                device=device,
            )
            rows.append(
                {
                    "repeat_idx": repeat_idx,
                    "repeat_seed": repeat_seed,
                    "variant": f"nll_{key}",
                    "variant_label": f"FM + NLL: {config.label}",
                    "mae": _mean_pair_error(matrix, ground_truth, relative=False),
                    "mrae": _mean_pair_error(matrix, ground_truth, relative=True),
                    "selected_epoch": int(metadata["selected_epoch"]),
                    "stopped_epoch": int(metadata["stopped_epoch"]),
                    "initial_val_nll": float(metadata.get("initial_val_nll", float("nan"))),
                    "selected_val_nll": float(metadata["best_val_nll"]),
                    "runtime_seconds": runtime,
                }
            )

    current = pd.DataFrame(rows)
    rows_path = output_dir / "jeffreys_n10000_rows.csv"
    if rows_path.is_file():
        previous = pd.read_csv(rows_path)
        current = pd.concat([previous, current], ignore_index=True)
        current = current.drop_duplicates(["repeat_idx", "variant"], keep="last")
    current = current.sort_values(["repeat_idx", "variant"])
    current.to_csv(rows_path, index=False)
    summary = (
        current.groupby(["variant", "variant_label"], as_index=False)
        .agg(
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_mrae=("mrae", "mean"),
            std_mrae=("mrae", "std"),
            mean_selected_epoch=("selected_epoch", "mean"),
            n_repeats=("repeat_idx", "nunique"),
            runtime_seconds=("runtime_seconds", "sum"),
        )
    )
    summary_path = output_dir / "jeffreys_n10000_summary.csv"
    summary.to_csv(summary_path, index=False)
    png, svg = _plot(summary, output_dir)
    trend_csv, trend_png, trend_svg = _write_trend_artifacts(
        prior_errors_csv=Path(args.prior_errors_csv),
        current_rows=current,
        output_dir=output_dir,
    )
    screen_csv = _write_common_screen_summary(current, output_dir)
    nll_loss_png, nll_loss_svg = _plot_baseline_nll_losses(output_dir)
    config_path = output_dir / "jeffreys_n10000_config.json"
    config_path.write_text(
        json.dumps(
            {
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "fm_config": asdict(_fm_config(args)),
                "requested_nll_configs": [asdict(config_map[key]) for key in args.nll_configs],
                "available_nll_configs": [asdict(config) for config in NLL_CONFIGS],
                "elapsed_seconds_this_run": time.perf_counter() - run_start,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "rows_csv": rows_path,
        "summary_csv": summary_path,
        "config_json": config_path,
        "figure_png": png,
        "figure_svg": svg,
        "trend_csv": trend_csv,
        "trend_figure_png": trend_png,
        "trend_figure_svg": trend_svg,
        "screen_common_repeats_csv": screen_csv,
        "nll_loss_figure_png": nll_loss_png,
        "nll_loss_figure_svg": nll_loss_svg,
    }


def main() -> None:
    outputs = run(build_parser().parse_args())
    for key, path in outputs.items():
        print(f"{key}: {path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
