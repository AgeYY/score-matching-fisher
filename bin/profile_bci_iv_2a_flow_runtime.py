#!/usr/bin/env python3
"""Profile dense-time BCI IV-2a affine-flow runtime and speedup levers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher.bci_iv_2a_dataset import load_features_npz  # noqa: E402
from fisher.bci_iv_2a_session_identification import (  # noqa: E402
    N_CLASSES,
    REFERENCE_RUNS,
    _stratified_validation_trials,
    _time_conditioned_endpoint_covariances,
    condition_design,
    rdms_from_means_and_precisions,
    select_half,
)
from fisher.flow_matching_skl import (  # noqa: E402
    _make_flow_matching_affine_path,
    build_flow_skl_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-file",
        type=Path,
        default=ROOT / "data/bci_iv_2a/processed/native_voltage_all_timepoints_20uv/A01T.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/bci_iv_2a/flow_runtime_profile_A01T",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=21260715)
    parser.add_argument("--profile-epochs", type=int, default=5)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192],
    )
    parser.add_argument("--time-subsample", type=int, default=128)
    return parser.parse_args()


def _build_model(device: torch.device) -> torch.nn.Module:
    return build_flow_skl_model(
        velocity_family="covariate_affine",
        theta_dim=5,
        x_dim=22,
        hidden_dim=64,
        depth=2,
        quadrature_steps=32,
        path_schedule="cosine",
        divergence_estimator="exact",
        affine_condition_indices=(N_CLASSES,),
    ).to(device)


def _flatten(
    values: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = values[indices].reshape(-1, values.shape[-1])
    y = np.repeat(labels[indices], times.size)
    t = np.tile(times, indices.size)
    return condition_design(y, t).astype(np.float32), x.astype(np.float32)


def _fixed_validation_batches(
    theta: np.ndarray,
    x: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    path, _ = _make_flow_matching_affine_path("cosine")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    batches = []
    for start in range(0, x.shape[0], int(batch_size)):
        stop = min(x.shape[0], start + int(batch_size))
        tb = torch.from_numpy(theta[start:stop]).to(device)
        x1 = torch.from_numpy(x[start:stop]).to(device)
        raw_t = torch.rand(stop - start, device=device, generator=generator)
        flow_t = 0.0005 + (1.0 - 0.001) * raw_t
        x0 = torch.randn(x1.shape, device=device, generator=generator)
        sample = path.sample(x_0=x0, x_1=x1, t=flow_t)
        batches.append((tb, sample.x_t, sample.t, sample.dx_t))
    return batches


def _measure_current_style(
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    *,
    batch_size: int,
    epochs: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(int(seed))
    model = _build_model(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-5)
    path, _ = _make_flow_matching_affine_path("cosine")
    loader = DataLoader(
        TensorDataset(torch.from_numpy(theta_train), torch.from_numpy(x_train)),
        batch_size=int(batch_size),
        shuffle=True,
    )
    val_batches = _fixed_validation_batches(
        theta_val,
        x_val,
        batch_size=batch_size,
        device=device,
        seed=seed + 10_000,
    )

    def run_epoch(measure: bool) -> tuple[float, float, float, float]:
        torch.cuda.synchronize(device)
        train_start = time.perf_counter()
        scalar_sync_seconds = 0.0
        model.train()
        for tb, x1 in loader:
            tb = tb.to(device)
            x1 = x1.to(device)
            raw_t = torch.rand(x1.shape[0], device=device)
            flow_t = 0.0005 + (1.0 - 0.001) * raw_t
            x0 = torch.randn_like(x1)
            sample = path.sample(x_0=x0, x_1=x1, t=flow_t)
            loss = torch.mean((model(sample.x_t, tb, sample.t) - sample.dx_t) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            sync_start = time.perf_counter()
            _ = float(grad_norm)
            scalar_sync_seconds += time.perf_counter() - sync_start
            optimizer.step()
            sync_start = time.perf_counter()
            _ = float(loss.detach().cpu())
            scalar_sync_seconds += time.perf_counter() - sync_start
        torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - train_start

        validation_start = time.perf_counter()
        model.eval()
        with torch.no_grad():
            for tb, x_t, flow_t, dx_t in val_batches:
                loss = torch.mean((model(x_t, tb, flow_t) - dx_t) ** 2)
                sync_start = time.perf_counter()
                _ = float(loss.detach().cpu())
                scalar_sync_seconds += time.perf_counter() - sync_start
        torch.cuda.synchronize(device)
        validation_seconds = time.perf_counter() - validation_start
        return (
            train_seconds,
            validation_seconds,
            scalar_sync_seconds,
            float(theta_train.shape[0] / train_seconds),
        )

    run_epoch(measure=False)
    measured = [run_epoch(measure=True) for _ in range(int(epochs))]
    values = np.asarray(measured, dtype=np.float64)
    return {
        "train_seconds_per_epoch": float(np.mean(values[:, 0])),
        "validation_seconds_per_epoch": float(np.mean(values[:, 1])),
        "total_seconds_per_epoch": float(np.mean(values[:, 0] + values[:, 1])),
        "scalar_sync_seconds_per_epoch": float(np.mean(values[:, 2])),
        "training_examples_per_second": float(np.mean(values[:, 3])),
        "train_batches": int(math.ceil(theta_train.shape[0] / int(batch_size))),
        "validation_batches": int(math.ceil(theta_val.shape[0] / int(batch_size))),
    }


def _measure_optimized_style(
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    *,
    batch_size: int,
    epochs: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(int(seed))
    model = _build_model(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-5)
    path, _ = _make_flow_matching_affine_path("cosine")
    theta_gpu = torch.from_numpy(theta_train).to(device)
    x_gpu = torch.from_numpy(x_train).to(device)
    val_batches = _fixed_validation_batches(
        theta_val,
        x_val,
        batch_size=batch_size,
        device=device,
        seed=seed + 10_000,
    )

    def run_epoch() -> tuple[float, float, float]:
        torch.cuda.synchronize(device)
        train_start = time.perf_counter()
        permutation = torch.randperm(theta_gpu.shape[0], device=device)
        loss_sum = torch.zeros((), device=device)
        model.train()
        for start in range(0, theta_gpu.shape[0], int(batch_size)):
            selected = permutation[start : start + int(batch_size)]
            tb = theta_gpu[selected]
            x1 = x_gpu[selected]
            raw_t = torch.rand(x1.shape[0], device=device)
            flow_t = 0.0005 + (1.0 - 0.001) * raw_t
            x0 = torch.randn_like(x1)
            sample = path.sample(x_0=x0, x_1=x1, t=flow_t)
            loss = torch.mean((model(sample.x_t, tb, sample.t) - sample.dx_t) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()
            loss_sum += loss.detach()
        _ = float(loss_sum)
        torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - train_start

        validation_start = time.perf_counter()
        val_sum = torch.zeros((), device=device)
        model.eval()
        with torch.no_grad():
            for tb, x_t, flow_t, dx_t in val_batches:
                val_sum += torch.mean((model(x_t, tb, flow_t) - dx_t) ** 2)
        _ = float(val_sum)
        torch.cuda.synchronize(device)
        validation_seconds = time.perf_counter() - validation_start
        return train_seconds, validation_seconds, float(theta_train.shape[0] / train_seconds)

    run_epoch()
    measured = [run_epoch() for _ in range(int(epochs))]
    values = np.asarray(measured, dtype=np.float64)
    return {
        "train_seconds_per_epoch": float(np.mean(values[:, 0])),
        "validation_seconds_per_epoch": float(np.mean(values[:, 1])),
        "total_seconds_per_epoch": float(np.mean(values[:, 0] + values[:, 1])),
        "training_examples_per_second": float(np.mean(values[:, 2])),
        "train_batches": int(math.ceil(theta_train.shape[0] / int(batch_size))),
        "validation_batches": int(math.ceil(theta_val.shape[0] / int(batch_size))),
    }


def _measure_execution_variant(
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_val: np.ndarray,
    x_val: np.ndarray,
    *,
    batch_size: int,
    epochs: int,
    device: torch.device,
    seed: int,
    gpu_resident: bool,
    defer_scalar_reads: bool,
) -> dict[str, float]:
    """Isolate GPU residency from per-minibatch scalar synchronization."""

    torch.manual_seed(int(seed))
    model = _build_model(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-5)
    path, _ = _make_flow_matching_affine_path("cosine")
    loader = None
    theta_gpu = None
    x_gpu = None
    if gpu_resident:
        theta_gpu = torch.from_numpy(theta_train).to(device)
        x_gpu = torch.from_numpy(x_train).to(device)
    else:
        loader = DataLoader(
            TensorDataset(torch.from_numpy(theta_train), torch.from_numpy(x_train)),
            batch_size=int(batch_size),
            shuffle=True,
        )
    val_batches = _fixed_validation_batches(
        theta_val,
        x_val,
        batch_size=batch_size,
        device=device,
        seed=seed + 10_000,
    )

    def training_batches():
        if gpu_resident:
            if theta_gpu is None or x_gpu is None:
                raise RuntimeError("GPU-resident tensors were not initialized.")
            permutation = torch.randperm(theta_gpu.shape[0], device=device)
            for start in range(0, theta_gpu.shape[0], int(batch_size)):
                selected = permutation[start : start + int(batch_size)]
                yield theta_gpu[selected], x_gpu[selected]
        else:
            if loader is None:
                raise RuntimeError("CPU DataLoader was not initialized.")
            for tb, x1 in loader:
                yield tb.to(device), x1.to(device)

    def run_epoch() -> tuple[float, float]:
        torch.cuda.synchronize(device)
        train_start = time.perf_counter()
        loss_sum = torch.zeros((), device=device)
        model.train()
        for tb, x1 in training_batches():
            raw_t = torch.rand(x1.shape[0], device=device)
            flow_t = 0.0005 + (1.0 - 0.001) * raw_t
            x0 = torch.randn_like(x1)
            sample = path.sample(x_0=x0, x_1=x1, t=flow_t)
            loss = torch.mean((model(sample.x_t, tb, sample.t) - sample.dx_t) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()
            if defer_scalar_reads:
                loss_sum += loss.detach()
            else:
                _ = float(grad_norm)
                _ = float(loss.detach().cpu())
        if defer_scalar_reads:
            _ = float(loss_sum)
        torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - train_start

        validation_start = time.perf_counter()
        val_sum = torch.zeros((), device=device)
        model.eval()
        with torch.no_grad():
            for tb, x_t, flow_t, dx_t in val_batches:
                loss = torch.mean((model(x_t, tb, flow_t) - dx_t) ** 2)
                if defer_scalar_reads:
                    val_sum += loss
                else:
                    _ = float(loss.detach().cpu())
        if defer_scalar_reads:
            _ = float(val_sum)
        torch.cuda.synchronize(device)
        validation_seconds = time.perf_counter() - validation_start
        return train_seconds, validation_seconds

    run_epoch()
    values = np.asarray([run_epoch() for _ in range(int(epochs))], dtype=np.float64)
    return {
        "gpu_resident": bool(gpu_resident),
        "defer_scalar_reads": bool(defer_scalar_reads),
        "train_seconds_per_epoch": float(np.mean(values[:, 0])),
        "validation_seconds_per_epoch": float(np.mean(values[:, 1])),
        "total_seconds_per_epoch": float(np.mean(np.sum(values, axis=1))),
        "training_examples_per_second": float(
            theta_train.shape[0] / np.mean(values[:, 0])
        ),
    }


def _measure_readout(
    times: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, float]:
    model = _build_model(device)
    labels = np.repeat(np.arange(N_CLASSES, dtype=np.int64), times.size)
    grid_times = np.tile(times, N_CLASSES)
    conditions = condition_design(labels, grid_times)
    condition_tensor = torch.from_numpy(conditions.astype(np.float32)).to(device)

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        means_flat = model.endpoint_mean(condition_tensor).detach().cpu().numpy()
    endpoint_mean_seconds = time.perf_counter() - start
    means = means_flat.reshape(N_CLASSES, times.size, -1).transpose(1, 0, 2)

    time_conditions = condition_design(np.zeros(times.size, dtype=np.int64), times)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    covariances = _time_conditioned_endpoint_covariances(
        model,
        time_conditions,
        device=device,
        steps=48,
        ridge=1e-5,
    )
    covariance_integration_seconds = time.perf_counter() - start

    start = time.perf_counter()
    precisions = np.linalg.inv(
        covariances + 1e-5 * np.eye(covariances.shape[-1])[None]
    )
    precision_inversion_seconds = time.perf_counter() - start

    start = time.perf_counter()
    _ = rdms_from_means_and_precisions(means, precisions)
    rdm_construction_seconds = time.perf_counter() - start
    return {
        "endpoint_mean_seconds": endpoint_mean_seconds,
        "covariance_integration_seconds": covariance_integration_seconds,
        "precision_inversion_seconds": precision_inversion_seconds,
        "rdm_construction_seconds": rdm_construction_seconds,
    }


def _measure_forward_components(device: torch.device, batch_size: int = 8192) -> dict[str, float]:
    model = _build_model(device).eval()
    theta = torch.randn(batch_size, 5, device=device)
    theta[:, :4] = 0.0
    theta[torch.arange(batch_size, device=device), torch.arange(batch_size, device=device) % 4] = 1.0
    x = torch.randn(batch_size, 22, device=device)
    flow_t = torch.rand(batch_size, device=device)

    def measure(function, repetitions: int = 50) -> float:
        with torch.no_grad():
            for _ in range(5):
                function()
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(repetitions):
                function()
            torch.cuda.synchronize(device)
        return 1e3 * (time.perf_counter() - start) / repetitions

    return {
        "mean_network_ms_per_batch": measure(lambda: model.b(theta)),
        "covariance_matrix_network_ms_per_batch": measure(lambda: model.A(theta, flow_t)),
        "full_velocity_ms_per_batch": measure(lambda: model(x, theta, flow_t)),
        "batch_size": int(batch_size),
        "mean_network_parameters": int(sum(p.numel() for p in model.b_net.parameters())),
        "covariance_matrix_network_parameters": int(sum(p.numel() for p in model.a_net.parameters())),
    }


def _plot_profile(output_dir: Path, summary: dict) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 16,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))
    baseline = summary["current_style_batch_sweep"]["1024"]
    axes[0].bar(
        ["Training", "Validation"],
        [baseline["train_seconds_per_epoch"], baseline["validation_seconds_per_epoch"]],
        color=["#4477AA", "#CC6677"],
    )
    axes[0].set_ylabel("Seconds per epoch")
    axes[0].set_title("Current epoch")

    batch_sizes = [int(value) for value in summary["batch_sizes"]]
    current = [summary["current_style_batch_sweep"][str(value)]["total_seconds_per_epoch"] for value in batch_sizes]
    optimized = [summary["optimized_style_batch_sweep"][str(value)]["total_seconds_per_epoch"] for value in batch_sizes]
    axes[1].plot(batch_sizes, current, color="#4477AA", marker="o", linewidth=2.0, label="Current")
    axes[1].plot(batch_sizes, optimized, color="#228833", marker="o", linewidth=2.0, label="GPU-resident")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(batch_sizes, [str(value) for value in batch_sizes])
    axes[1].set_xlabel("Batch size")
    axes[1].set_ylabel("Seconds per epoch")
    axes[1].set_title("Batch-size sweep")
    axes[1].legend(frameon=False, loc="best", fontsize=12)

    readout = summary["readout_timing_seconds"]
    labels = ["Means", "Cov. ODE", "Inverse", "RDM"]
    values = [
        readout["endpoint_mean_seconds"],
        readout["covariance_integration_seconds"],
        readout["precision_inversion_seconds"],
        readout["rdm_construction_seconds"],
    ]
    axes[2].bar(labels, values, color="#AA4499")
    axes[2].set_ylabel("Seconds per fit")
    axes[2].set_title("Post-training readout")
    axes[2].tick_params(axis="x", labelrotation=30)
    for axis in axes:
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_linewidth(1.8)
        axis.tick_params(width=1.8)
    figure.tight_layout()
    figure.savefig(output_dir / "flow_runtime_profile.png", dpi=300)
    figure.savefig(output_dir / "flow_runtime_profile.svg")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This profiler requires CUDA; no CPU fallback is permitted.")
    torch.cuda.set_device(0 if device.index is None else device.index)
    dataset = load_features_npz(args.feature_file)
    values, labels, _ = select_half(dataset, REFERENCE_RUNS)
    times = np.asarray(dataset.time_centers, dtype=np.float64)
    train_trials, val_trials = _stratified_validation_trials(labels, 0.2, args.seed)
    theta_train, x_train = _flatten(values, labels, times, train_trials)
    theta_val, x_val = _flatten(values, labels, times, val_trials)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "experiment": "A01T dense-time affine-flow runtime profile",
        "device": args.device,
        "gpu": torch.cuda.get_device_name(device),
        "feature_file": str(args.feature_file.resolve()),
        "n_times": int(times.size),
        "n_train_trials": int(train_trials.size),
        "n_validation_trials": int(val_trials.size),
        "n_train_rows": int(theta_train.shape[0]),
        "n_validation_rows": int(theta_val.shape[0]),
        "batch_sizes": [int(value) for value in args.batch_sizes],
        "profile_epochs": int(args.profile_epochs),
        "current_style_batch_sweep": {},
        "optimized_style_batch_sweep": {},
    }
    for batch_size in args.batch_sizes:
        print(f"[profile] current batch_size={batch_size}", flush=True)
        summary["current_style_batch_sweep"][str(batch_size)] = _measure_current_style(
            theta_train,
            x_train,
            theta_val,
            x_val,
            batch_size=batch_size,
            epochs=args.profile_epochs,
            device=device,
            seed=args.seed + batch_size,
        )
        print(f"[profile] optimized batch_size={batch_size}", flush=True)
        summary["optimized_style_batch_sweep"][str(batch_size)] = _measure_optimized_style(
            theta_train,
            x_train,
            theta_val,
            x_val,
            batch_size=batch_size,
            epochs=args.profile_epochs,
            device=device,
            seed=args.seed + 100_000 + batch_size,
        )

    if not (1 <= args.time_subsample <= times.size):
        raise ValueError("--time-subsample must be between 1 and the number of time points.")
    selected_times = np.linspace(0, times.size - 1, args.time_subsample, dtype=np.int64)
    sub_times = times[selected_times]
    sub_values = values[:, selected_times]
    sub_theta_train, sub_x_train = _flatten(sub_values, labels, sub_times, train_trials)
    sub_theta_val, sub_x_val = _flatten(sub_values, labels, sub_times, val_trials)
    largest_batch = int(max(args.batch_sizes))
    summary["time_subsample_profile"] = {
        "n_times": int(args.time_subsample),
        "row_reduction_factor": float(times.size / args.time_subsample),
        "batch_size": largest_batch,
        **_measure_optimized_style(
            sub_theta_train,
            sub_x_train,
            sub_theta_val,
            sub_x_val,
            batch_size=largest_batch,
            epochs=args.profile_epochs,
            device=device,
            seed=args.seed + 200_000,
        ),
    }
    summary["readout_timing_seconds"] = _measure_readout(times, device=device)
    summary["forward_component_timing"] = _measure_forward_components(device)

    baseline_batch = 1024
    summary["execution_isolation_batch_1024"] = {
        "cpu_data_per_batch_scalars": summary["current_style_batch_sweep"][
            str(baseline_batch)
        ],
        "cpu_data_deferred_scalars": _measure_execution_variant(
            theta_train,
            x_train,
            theta_val,
            x_val,
            batch_size=baseline_batch,
            epochs=args.profile_epochs,
            device=device,
            seed=args.seed + 300_000,
            gpu_resident=False,
            defer_scalar_reads=True,
        ),
        "gpu_data_per_batch_scalars": _measure_execution_variant(
            theta_train,
            x_train,
            theta_val,
            x_val,
            batch_size=baseline_batch,
            epochs=args.profile_epochs,
            device=device,
            seed=args.seed + 400_000,
            gpu_resident=True,
            defer_scalar_reads=False,
        ),
        "gpu_data_deferred_scalars": summary["optimized_style_batch_sweep"][
            str(baseline_batch)
        ],
    }

    baseline = summary["current_style_batch_sweep"]["1024"]
    optimized = summary["optimized_style_batch_sweep"][str(largest_batch)]
    isolation = summary["execution_isolation_batch_1024"]
    cpu_deferred = isolation["cpu_data_deferred_scalars"]
    gpu_synced = isolation["gpu_data_per_batch_scalars"]
    gpu_deferred = isolation["gpu_data_deferred_scalars"]
    summary["derived"] = {
        "validation_fraction_of_current_epoch": float(
            baseline["validation_seconds_per_epoch"] / baseline["total_seconds_per_epoch"]
        ),
        "validation_every_25_only_speedup_current": float(
            baseline["total_seconds_per_epoch"]
            / (baseline["train_seconds_per_epoch"] + baseline["validation_seconds_per_epoch"] / 25.0)
        ),
        "defer_scalar_reads_only_speedup_cpu_data": float(
            baseline["total_seconds_per_epoch"] / cpu_deferred["total_seconds_per_epoch"]
        ),
        "gpu_residency_only_speedup_per_batch_scalars": float(
            baseline["total_seconds_per_epoch"] / gpu_synced["total_seconds_per_epoch"]
        ),
        "gpu_residency_only_speedup_deferred_scalars": float(
            cpu_deferred["total_seconds_per_epoch"] / gpu_deferred["total_seconds_per_epoch"]
        ),
        "gpu_resident_deferred_batch8192_speedup": float(
            baseline["total_seconds_per_epoch"] / optimized["total_seconds_per_epoch"]
        ),
        "combined_batch8192_validate_every_25_speedup": float(
            baseline["total_seconds_per_epoch"]
            / (optimized["train_seconds_per_epoch"] + optimized["validation_seconds_per_epoch"] / 25.0)
        ),
        "time_subsampling_incremental_speedup": float(
            optimized["total_seconds_per_epoch"]
            / summary["time_subsample_profile"]["total_seconds_per_epoch"]
        ),
        "combined_time_subsample_and_validate_every_25_speedup": float(
            baseline["total_seconds_per_epoch"]
            / (
                summary["time_subsample_profile"]["train_seconds_per_epoch"]
                + summary["time_subsample_profile"]["validation_seconds_per_epoch"] / 25.0
            )
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = []
    for style_key in ("current_style_batch_sweep", "optimized_style_batch_sweep"):
        for batch_size, metrics in summary[style_key].items():
            rows.append({"style": style_key, "batch_size": int(batch_size), **metrics})
    with (args.output_dir / "batch_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot_profile(args.output_dir, summary)
    print(json.dumps(summary["derived"], indent=2, sort_keys=True), flush=True)
    print(f"[profile] output={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
