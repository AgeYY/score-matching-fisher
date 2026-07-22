#!/usr/bin/env python3
"""Run held-out decoding and known-increment validation on one Stringer session."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.fisher_validation import (
    append_paired_probe,
    calibration_metrics,
    evaluate_windowed_decoders,
    finite_grid_probe_increment,
    fit_flow_direction_estimator,
    fit_gkr_direction_estimator,
    gkr_checkpoint,
    stratified_train_validation_test_split,
)
from fisher.shared_fisher_est import require_device
from fisher.stringer_dataset import load_stringer_session
from fisher.stringer_session_identification import encode_flow_orientation
from global_setting import DATA_DIR, DEFAULT_EARLY_STOPPING_PATIENCE, DEFAULT_TRAINING_MAX_EPOCHS


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(DATA_DIR) / "fisher_validation_pilot" / "stringer")
    parser.add_argument("--session-index", type=int, default=0)
    parser.add_argument("--seeds", type=_csv_ints, default=[7, 19])
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--theta-grid-size", type=int, default=17)
    parser.add_argument("--train-fraction", type=float, default=0.64)
    parser.add_argument("--validation-fraction", type=float, default=0.16)
    parser.add_argument("--probe-peaks", type=_csv_floats, default=[0.25, 1.0, 4.0])
    parser.add_argument("--probe-phase", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_MAX_EPOCHS)
    parser.add_argument("--early-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--ode-steps", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _fit_case(
    *,
    case_dir: Path,
    label: str,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    theta_validation: np.ndarray,
    x_validation: np.ndarray,
    theta_grid: np.ndarray,
    period: float,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    fit_dir = case_dir / label
    fit_dir.mkdir(parents=True, exist_ok=True)
    result_path = fit_dir / "estimates.npz"
    metadata_path = fit_dir / "metadata.json"
    flow_path = fit_dir / "flow_selected_model.pt"
    gkr_path = fit_dir / "gkr_model.pt"
    if all(path.is_file() for path in (result_path, metadata_path, flow_path, gkr_path)) and not args.force:
        with np.load(result_path) as saved:
            return {key: np.asarray(saved[key]) for key in saved.files} | {
                "metadata": json.loads(metadata_path.read_text(encoding="utf-8"))
            }

    condition_train = encode_flow_orientation(theta_train, period=period, encoding="periodic-rbf")
    condition_validation = encode_flow_orientation(theta_validation, period=period, encoding="periodic-rbf")
    condition_grid = encode_flow_orientation(theta_grid, period=period, encoding="periodic-rbf")
    flow_model, flow_training, flow_estimate, flow_direction = fit_flow_direction_estimator(
        theta_train=theta_train,
        x_train=x_train,
        theta_validation=theta_validation,
        x_validation=x_validation,
        theta_grid=theta_grid,
        condition_train=condition_train,
        condition_validation=condition_validation,
        condition_grid=condition_grid,
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
        circular_period=period,
    )
    arrays = {
        "theta_midpoints": np.asarray(flow_estimate["theta_midpoints"], dtype=np.float64).reshape(-1),
        "theta_left": np.asarray(flow_estimate["theta_left"], dtype=np.float64).reshape(-1),
        "theta_right": np.asarray(flow_estimate["theta_right"], dtype=np.float64).reshape(-1),
        "dtheta": np.asarray(flow_estimate["dtheta"], dtype=np.float64),
        "flow_fisher": np.asarray(flow_estimate["fisher"], dtype=np.float64),
        "flow_direction": flow_direction,
        "flow_train_loss": np.asarray(flow_training["train_losses"], dtype=np.float64),
        "flow_validation_loss": np.asarray(flow_training["val_losses"], dtype=np.float64),
        "gkr_fisher": np.asarray(gkr_estimate.linear_fisher, dtype=np.float64),
        "gkr_direction": gkr_direction,
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
        "flow_selected_epoch": int(flow_training["selected_epoch"]),
        "flow_stopped_epoch": int(flow_training["stopped_epoch"]),
        "flow_orientation_encoding": "periodic-rbf",
        "flow_orientation_rbf_centers": 8,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    del flow_model, gkr_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays | {"metadata": metadata}


def main() -> None:
    args = parse_args()
    device = require_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session = load_stringer_session(
        None,
        session_stimuli_type="gratings_static",
        session_index=args.session_index,
        orientation_period=np.pi,
    )
    period = float(np.pi)
    theta_all = np.asarray(session.grating_orientation, dtype=np.float64)
    response_all = np.asarray(session.neural_responses, dtype=np.float64)
    grid = np.linspace(0.0, period, args.theta_grid_size, dtype=np.float64).reshape(-1, 1)
    half_width = 0.5 * float(np.diff(grid[:, 0])[0])
    omega = 2.0
    heldout_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        case_dir = output_dir / f"session{args.session_index}" / f"seed{seed}"
        case_dir.mkdir(parents=True, exist_ok=True)
        split = stratified_train_validation_test_split(
            theta_all,
            n_strata=args.theta_grid_size - 1,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            seed=seed,
            period=period,
        )
        pca = PCA(
            n_components=args.pca_dim,
            whiten=True,
            svd_solver="randomized",
            random_state=seed,
        )
        pca.fit(response_all[split.train])
        x_train = pca.transform(response_all[split.train]).astype(np.float64)
        x_validation = pca.transform(response_all[split.validation]).astype(np.float64)
        x_test = pca.transform(response_all[split.test]).astype(np.float64)
        theta_train = theta_all[split.train]
        theta_validation = theta_all[split.validation]
        theta_test = theta_all[split.test]
        np.savez_compressed(
            case_dir / "split_and_pca.npz",
            train_index=split.train,
            validation_index=split.validation,
            test_index=split.test,
            stratum=split.stratum,
            pca_components=pca.components_,
            pca_mean=pca.mean_,
            pca_explained_variance=pca.explained_variance_,
            pca_explained_variance_ratio=pca.explained_variance_ratio_,
        )

        baseline = _fit_case(
            case_dir=case_dir,
            label="baseline",
            theta_train=theta_train,
            x_train=x_train,
            theta_validation=theta_validation,
            x_validation=x_validation,
            theta_grid=grid,
            period=period,
            args=args,
            seed=seed,
            device=device,
        )
        for method, direction in (
            ("Flow matching", baseline["flow_direction"]),
            ("GKR", baseline["gkr_direction"]),
        ):
            evaluation = evaluate_windowed_decoders(
                np.asarray(direction),
                x_test,
                theta_test,
                grid[:-1, 0],
                grid[1:, 0],
                half_width=half_width,
                period=period,
            )
            heldout_rows.append(
                {
                    "dataset": "Stringer",
                    "seed": int(seed),
                    "method": method,
                    "mean_achieved_fisher": float(np.mean(evaluation.achieved_fisher_raw)),
                    "median_achieved_fisher": float(np.median(evaluation.achieved_fisher_raw)),
                    "mean_auc": float(np.mean(evaluation.roc_auc)),
                }
            )
            np.savez_compressed(
                case_dir / f"heldout_{method.lower().replace(' ', '_')}.npz",
                theta_midpoints=0.5 * (grid[:-1, 0] + grid[1:, 0]),
                achieved_fisher_raw=evaluation.achieved_fisher_raw,
                achieved_fisher_display=evaluation.achieved_fisher_display,
                roc_auc=evaluation.roc_auc,
                n_left=evaluation.n_left,
                n_right=evaluation.n_right,
            )

        noise_train = np.random.default_rng(seed + 200_000).standard_normal(x_train.shape[0])
        noise_validation = np.random.default_rng(seed + 300_000).standard_normal(x_validation.shape[0])
        control_train, _, _ = append_paired_probe(
            x_train,
            theta_train,
            peak_fisher=0.0,
            omega=omega,
            phase=args.probe_phase,
            seed=seed,
            noise=noise_train,
        )
        control_validation, _, _ = append_paired_probe(
            x_validation,
            theta_validation,
            peak_fisher=0.0,
            omega=omega,
            phase=args.probe_phase,
            seed=seed,
            noise=noise_validation,
        )
        control = _fit_case(
            case_dir=case_dir,
            label="probe_control",
            theta_train=theta_train,
            x_train=control_train,
            theta_validation=theta_validation,
            x_validation=control_validation,
            theta_grid=grid,
            period=period,
            args=args,
            seed=seed + 50_000,
            device=device,
        )
        for peak in args.probe_peaks:
            _, probe_train, _ = append_paired_probe(
                x_train,
                theta_train,
                peak_fisher=peak,
                omega=omega,
                phase=args.probe_phase,
                seed=seed,
                noise=noise_train,
            )
            _, probe_validation, _ = append_paired_probe(
                x_validation,
                theta_validation,
                peak_fisher=peak,
                omega=omega,
                phase=args.probe_phase,
                seed=seed,
                noise=noise_validation,
            )
            probe = _fit_case(
                case_dir=case_dir,
                label=f"probe_peak_{peak:g}_phase_{args.probe_phase:g}",
                theta_train=theta_train,
                x_train=probe_train,
                theta_validation=theta_validation,
                x_validation=probe_validation,
                theta_grid=grid,
                period=period,
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
                calibration_rows.append(
                    {
                        "dataset": "Stringer",
                        "seed": int(seed),
                        "method": method,
                        "peak_fisher": float(peak),
                        **calibration_metrics(increment, target),
                        "target": target.tolist(),
                        "estimated": increment.tolist(),
                    }
                )

        (case_dir / "case_metadata.json").write_text(
            json.dumps(
                {
                    "session_file": str(session.session_file),
                    "session_index": args.session_index,
                    "seed": seed,
                    "pca_dim": args.pca_dim,
                    "pca_fit_scope": "training_only",
                    "pca_whiten": True,
                    "n_train": int(split.train.size),
                    "n_validation": int(split.validation.size),
                    "n_test": int(split.test.size),
                    "theta_grid_size": args.theta_grid_size,
                    "endpoint_window_half_width": half_width,
                    "probe_omega": omega,
                    "probe_phase": args.probe_phase,
                    "probe_peaks": args.probe_peaks,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    summary = {
        "scope": "stringer",
        "device": str(device),
        "session_file": str(session.session_file),
        "heldout": heldout_rows,
        "calibration": calibration_rows,
        "config": {**vars(args), "output_dir": str(output_dir)},
    }
    summary["config"]["output_dir"] = str(output_dir)
    summary_path = output_dir / "stringer_fisher_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
