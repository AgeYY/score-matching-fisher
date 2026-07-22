#!/usr/bin/env python3
"""Run held-out decoding and known-increment validation on Fisher toy data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.continuous_fisher_comparison import make_native_dataset_npz, theta_grid_from_meta
from fisher.fisher_validation import (
    append_paired_probe,
    calibration_metrics,
    decoder_directions,
    evaluate_endpoint_decoders,
    finite_grid_probe_increment,
    fit_flow_direction_estimator,
    fit_gkr_direction_estimator,
    gkr_checkpoint,
    population_linear_moments,
)
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_meta, require_device
from global_setting import DATA_DIR, DEFAULT_EARLY_STOPPING_PATIENCE, DEFAULT_TRAINING_MAX_EPOCHS


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in _csv_strings(value)]


def _csv_floats(value: str) -> list[float]:
    return [float(item) for item in _csv_strings(value)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(DATA_DIR) / "fisher_validation_pilot" / "toy")
    parser.add_argument("--families", type=_csv_strings, default=["randamp_gaussian_sqrtd", "cosine_gmm"])
    parser.add_argument("--seeds", type=_csv_ints, default=[7, 19])
    parser.add_argument("--x-dim", type=int, default=50)
    parser.add_argument("--n-total", type=int, default=3000)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--theta-grid-size", type=int, default=61)
    parser.add_argument("--endpoint-eval-samples", type=int, default=2000)
    parser.add_argument("--probe-peaks", type=_csv_floats, default=[0.25, 1.0, 4.0])
    parser.add_argument("--probe-phase", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_MAX_EPOCHS)
    parser.add_argument("--early-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--ode-steps", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _jsonable_training(training: dict[str, object]) -> dict[str, object]:
    keys = (
        "selected_epoch",
        "stopped_epoch",
        "best_epoch",
        "best_val_loss",
        "checkpoint_selection",
        "fixed_validation",
        "fixed_validation_paths",
    )
    return {key: training[key] for key in keys if key in training}


def _fit_case(
    *,
    case_dir: Path,
    label: str,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_validation: np.ndarray,
    x_validation: np.ndarray,
    theta_grid: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    fit_dir = case_dir / label
    fit_dir.mkdir(parents=True, exist_ok=True)
    result_path = fit_dir / "estimates.npz"
    flow_path = fit_dir / "flow_selected_model.pt"
    gkr_path = fit_dir / "gkr_model.pt"
    metadata_path = fit_dir / "metadata.json"
    if result_path.is_file() and flow_path.is_file() and gkr_path.is_file() and not args.force:
        with np.load(result_path) as saved:
            return {key: np.asarray(saved[key]) for key in saved.files} | {
                "metadata": json.loads(metadata_path.read_text(encoding="utf-8"))
            }

    flow_model, flow_training, flow_estimate, flow_direction = fit_flow_direction_estimator(
        theta_train=theta_train,
        x_train=x_train,
        theta_validation=theta_validation,
        x_validation=x_validation,
        theta_grid=theta_grid,
        device=device,
        seed=seed,
        epochs=args.epochs,
        patience=args.early_patience,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        ode_steps=args.ode_steps,
    )
    gkr_model, gkr_estimate, gkr_direction = fit_gkr_direction_estimator(
        theta_train=theta_train,
        x_train=x_train,
        theta_grid=theta_grid,
        device=device,
        seed=seed,
    )
    arrays = {
        "theta_midpoints": np.asarray(flow_estimate["theta_midpoints"], dtype=np.float64).reshape(-1),
        "theta_left": np.asarray(flow_estimate["theta_left"], dtype=np.float64).reshape(-1),
        "theta_right": np.asarray(flow_estimate["theta_right"], dtype=np.float64).reshape(-1),
        "dtheta": np.asarray(flow_estimate["dtheta"], dtype=np.float64),
        "flow_fisher": np.asarray(flow_estimate["fisher"], dtype=np.float64),
        "flow_direction": flow_direction,
        "flow_delta_mu": np.asarray(flow_estimate["delta_mu"], dtype=np.float64),
        "flow_covariance": np.asarray(flow_estimate["mixed_covariance"], dtype=np.float64),
        "flow_train_loss": np.asarray(flow_training["train_losses"], dtype=np.float64),
        "flow_validation_loss": np.asarray(flow_training["val_losses"], dtype=np.float64),
        "gkr_fisher": np.asarray(gkr_estimate.linear_fisher, dtype=np.float64),
        "gkr_direction": gkr_direction,
        "gkr_mean": np.asarray(gkr_estimate.mean, dtype=np.float64),
        "gkr_mean_jacobian": np.asarray(gkr_estimate.mean_jacobian[:, :, 0], dtype=np.float64),
        "gkr_covariance": np.asarray(gkr_estimate.covariance, dtype=np.float64),
        "gkr_mean_loss": np.asarray(gkr_estimate.mean_loss, dtype=np.float64),
        "gkr_covariance_loss": np.asarray(gkr_estimate.covariance_loss, dtype=np.float64),
    }
    np.savez_compressed(result_path, **arrays)
    torch.save({key: value.detach().cpu() for key, value in flow_model.state_dict().items()}, flow_path)
    torch.save(gkr_checkpoint(gkr_model), gkr_path)
    metadata = {
        "label": label,
        "seed": int(seed),
        "n_train": int(x_train.shape[0]),
        "n_validation": int(x_validation.shape[0]),
        "flow_training": _jsonable_training(flow_training),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    del flow_model, gkr_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays | {"metadata": metadata}


def _evaluation_samples(population: object, theta_grid: np.ndarray, n: int, seed: int) -> np.ndarray:
    population.rng = np.random.default_rng(int(seed))
    return np.stack(
        [population.sample_x(np.full((int(n), 1), value, dtype=np.float64)) for value in theta_grid[:, 0]],
        axis=0,
    )


def main() -> None:
    args = parse_args()
    device = require_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    heldout_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    omega = 2.0 * np.pi / 12.0

    for family in args.families:
        for seed in args.seeds:
            case_dir = output_dir / family / f"seed{seed}"
            case_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = case_dir / "dataset.npz"
            make_native_dataset_npz(
                output_npz=dataset_path,
                dataset_family=family,
                x_dim=args.x_dim,
                n_total=args.n_total,
                train_frac=args.train_fraction,
                seed=seed,
                force=args.force,
            )
            bundle = load_shared_dataset_npz(dataset_path)
            grid = theta_grid_from_meta(bundle.meta, theta_grid_size=args.theta_grid_size)
            baseline = _fit_case(
                case_dir=case_dir,
                label="baseline",
                theta_train=bundle.theta_train,
                x_train=bundle.x_train,
                theta_validation=bundle.theta_validation,
                x_validation=bundle.x_validation,
                theta_grid=grid,
                args=args,
                seed=seed,
                device=device,
            )
            population = build_dataset_from_meta(dict(bundle.meta))
            endpoint = _evaluation_samples(
                population, grid, args.endpoint_eval_samples, seed + 1_000_000
            )
            midpoint = 0.5 * (grid[:-1] + grid[1:])
            mean_mid, derivative_mid, covariance_mid = population_linear_moments(population, midpoint)
            mean_left, _, _ = population_linear_moments(population, grid[:-1])
            mean_right, _, _ = population_linear_moments(population, grid[1:])
            oracle_direction = decoder_directions(mean_right - mean_left, covariance_mid)
            analytic_fisher = np.einsum(
                "ki,kij,kj->k", derivative_mid, np.linalg.inv(covariance_mid), derivative_mid
            )
            for method, direction in (
                ("Flow matching", baseline["flow_direction"]),
                ("GKR", baseline["gkr_direction"]),
                ("Oracle", oracle_direction),
            ):
                evaluation = evaluate_endpoint_decoders(
                    np.asarray(direction), endpoint[:-1], endpoint[1:], np.diff(grid[:, 0])
                )
                heldout_rows.append(
                    {
                        "dataset": family,
                        "seed": int(seed),
                        "method": method,
                        "mean_achieved_fisher": float(np.mean(evaluation.achieved_fisher_raw)),
                        "median_achieved_fisher": float(np.median(evaluation.achieved_fisher_raw)),
                        "mean_auc": float(np.mean(evaluation.roc_auc)),
                    }
                )
                np.savez_compressed(
                    case_dir / f"heldout_{method.lower().replace(' ', '_')}.npz",
                    theta_midpoints=midpoint[:, 0],
                    achieved_fisher_raw=evaluation.achieved_fisher_raw,
                    achieved_fisher_display=evaluation.achieved_fisher_display,
                    roc_auc=evaluation.roc_auc,
                    analytic_linear_fisher=analytic_fisher,
                )

            noise_train = np.random.default_rng(seed + 200_000).standard_normal(bundle.x_train.shape[0])
            noise_validation = np.random.default_rng(seed + 300_000).standard_normal(bundle.x_validation.shape[0])
            control_train, _, _ = append_paired_probe(
                bundle.x_train,
                bundle.theta_train,
                peak_fisher=0.0,
                omega=omega,
                phase=args.probe_phase,
                seed=seed,
                noise=noise_train,
            )
            control_validation, _, _ = append_paired_probe(
                bundle.x_validation,
                bundle.theta_validation,
                peak_fisher=0.0,
                omega=omega,
                phase=args.probe_phase,
                seed=seed,
                noise=noise_validation,
            )
            control = _fit_case(
                case_dir=case_dir,
                label="probe_control",
                theta_train=bundle.theta_train,
                x_train=control_train,
                theta_validation=bundle.theta_validation,
                x_validation=control_validation,
                theta_grid=grid,
                args=args,
                seed=seed + 50_000,
                device=device,
            )
            for peak in args.probe_peaks:
                _, probe_train, _ = append_paired_probe(
                    bundle.x_train,
                    bundle.theta_train,
                    peak_fisher=peak,
                    omega=omega,
                    phase=args.probe_phase,
                    seed=seed,
                    noise=noise_train,
                )
                _, probe_validation, _ = append_paired_probe(
                    bundle.x_validation,
                    bundle.theta_validation,
                    peak_fisher=peak,
                    omega=omega,
                    phase=args.probe_phase,
                    seed=seed,
                    noise=noise_validation,
                )
                probe = _fit_case(
                    case_dir=case_dir,
                    label=f"probe_peak_{peak:g}_phase_{args.probe_phase:g}",
                    theta_train=bundle.theta_train,
                    x_train=probe_train,
                    theta_validation=bundle.theta_validation,
                    x_validation=probe_validation,
                    theta_grid=grid,
                    args=args,
                    seed=seed + 50_000,
                    device=device,
                )
                target = finite_grid_probe_increment(
                    grid[:-1, 0],
                    grid[1:, 0],
                    peak_fisher=peak,
                    omega=omega,
                    phase=args.probe_phase,
                )
                for method, key in (("Flow matching", "flow_fisher"), ("GKR", "gkr_fisher")):
                    increment = np.asarray(probe[key]) - np.asarray(control[key])
                    metrics = calibration_metrics(increment, target)
                    calibration_rows.append(
                        {
                            "dataset": family,
                            "seed": int(seed),
                            "method": method,
                            "peak_fisher": float(peak),
                            **metrics,
                            "target": target.tolist(),
                            "estimated": increment.tolist(),
                        }
                    )
            (case_dir / "case_metadata.json").write_text(
                json.dumps(
                    {
                        "dataset_family": family,
                        "seed": seed,
                        "x_dim": args.x_dim,
                        "n_total": args.n_total,
                        "train_fraction": args.train_fraction,
                        "theta_grid_size": args.theta_grid_size,
                        "endpoint_eval_samples_per_endpoint": args.endpoint_eval_samples,
                        "probe_omega": omega,
                        "probe_phase": args.probe_phase,
                        "probe_peaks": args.probe_peaks,
                        "analytic_linear_fisher": analytic_fisher.tolist(),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    summary = {
        "scope": "toy",
        "device": str(device),
        "heldout": heldout_rows,
        "calibration": calibration_rows,
        "config": vars(args) | {"output_dir": str(output_dir)},
    }
    summary_path = output_dir / "toy_fisher_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
