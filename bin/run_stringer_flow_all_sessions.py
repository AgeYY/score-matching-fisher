#!/usr/bin/env python3
"""Estimate Flow Matching linear or full Fisher curves for Stringer sessions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher.fisher_validation import (
    fit_flow_direction_estimator,
    stratified_disjoint_subset_indices,
)
from fisher.flow_matching_skl import (
    DEFAULT_AFFINE_COVARIANCE_ODE_STEPS,
    build_flow_skl_model,
    estimate_adjacent_model_jeffreys_fisher,
    finetune_flow_skl_cnf_likelihood,
    train_flow_skl_model,
)
from fisher.shared_fisher_est import require_device
from fisher.stringer_dataset import list_stringer_sessions, load_stringer_session
from fisher.stringer_session_identification import (
    encode_flow_orientation,
    estimate_affine_mixed_symmetric_kl_fisher_for_conditions,
)
from global_setting import (
    DATA_DIR,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_TRAINING_MAX_EPOCHS,
)

PERIOD = float(np.pi)


def _csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("Expected at least one session index.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--fisher-kind",
        choices=("linear", "full"),
        default="linear",
        help="Linear affine readout or nonlinear adjacent-Jeffreys full Fisher.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATA_DIR) / "stringer_flow_all_sessions_n100_pca82_uncentered",
    )
    parser.add_argument("--n-trials-per-session", type=int, default=100)
    parser.add_argument(
        "--use-full-dataset",
        action="store_true",
        help="Use every available trial in each session instead of subsampling.",
    )
    parser.add_argument("--pca-dim", type=int, default=82)
    parser.add_argument("--theta-grid-size", type=int, default=17)
    parser.add_argument("--flow-validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_MAX_EPOCHS)
    parser.add_argument(
        "--early-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument(
        "--ode-steps", type=int, default=DEFAULT_AFFINE_COVARIANCE_ODE_STEPS
    )
    parser.add_argument("--full-mc-jeffreys-samples", type=int, default=4096)
    parser.add_argument("--full-hutchinson-probes", type=int, default=4)
    parser.add_argument("--full-likelihood-batch-size", type=int, default=1024)
    parser.add_argument(
        "--nll-epochs",
        type=int,
        default=0,
        help="CNF likelihood fine-tuning epochs; zero disables fine-tuning.",
    )
    parser.add_argument("--nll-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--nll-batch-size", type=int, default=2048)
    parser.add_argument("--nll-lr", type=float, default=1e-3)
    parser.add_argument("--nll-ode-steps", type=int, default=32)
    parser.add_argument(
        "--session-indices",
        type=_csv_ints,
        default=None,
        help="Optional comma-separated zero-based session indices.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Fit requested sessions without writing shared summary/figure files.",
    )
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


def _select_trial_indices(
    theta: np.ndarray,
    *,
    n_trials: int,
    n_strata: int,
    seed: int,
) -> np.ndarray:
    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    if int(n_trials) > values.size:
        raise ValueError(
            f"Requested {n_trials} trials, but the session contains only {values.size}."
        )
    return stratified_disjoint_subset_indices(
        values,
        int(n_trials),
        n_subsets=1,
        n_strata=int(n_strata),
        seed=int(seed),
        period=PERIOD,
    )[0]


def _session_trial_indices(
    theta: np.ndarray,
    *,
    use_full_dataset: bool,
    n_trials: int,
    n_strata: int,
    seed: int,
) -> np.ndarray:
    values = np.asarray(theta).reshape(-1)
    if bool(use_full_dataset):
        return np.arange(values.size, dtype=np.int64)
    return _select_trial_indices(
        values,
        n_trials=int(n_trials),
        n_strata=int(n_strata),
        seed=int(seed),
    )


def _train_validation_indices(
    theta: np.ndarray,
    *,
    validation_fraction: float,
    n_strata: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    n_train = min(
        max(int(round((1.0 - float(validation_fraction)) * values.size)), 1),
        values.size - 1,
    )
    train = _select_trial_indices(
        values,
        n_trials=n_train,
        n_strata=int(n_strata),
        seed=int(seed),
    )
    validation = np.setdiff1d(np.arange(values.size, dtype=np.int64), train)
    return train, validation


def _uncentered_projection(
    responses: np.ndarray,
    *,
    n_components: int,
    random_state: int,
) -> tuple[np.ndarray, TruncatedSVD]:
    x = np.asarray(responses)
    if x.ndim != 2:
        raise ValueError("responses must be a trial-by-neuron matrix.")
    if int(n_components) < 1 or int(n_components) >= min(x.shape):
        raise ValueError("Uncentered projection dimension must be below min(responses.shape).")
    projector = TruncatedSVD(
        n_components=int(n_components),
        algorithm="randomized",
        random_state=int(random_state),
    )
    return projector.fit_transform(x).astype(np.float64), projector


def _session_label(info: Any) -> str:
    return str(info.mouse_name)


def _signature(args: argparse.Namespace, *, session_file: Path, session_index: int) -> dict[str, Any]:
    return {
        "session_file": str(session_file),
        "fisher_kind": str(args.fisher_kind),
        "use_full_dataset": bool(args.use_full_dataset),
        "n_trials_per_session": int(args.n_trials_per_session),
        "trial_selection": "orientation_stratified_without_replacement",
        "pca_dim": int(args.pca_dim),
        "projection": "truncated_svd_uncentered_unwhitened",
        "theta_grid_size": int(args.theta_grid_size),
        "flow_validation_fraction": float(args.flow_validation_fraction),
        "epochs": int(args.epochs),
        "early_patience": int(args.early_patience),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "ode_steps": int(args.ode_steps),
        "full_mc_jeffreys_samples": int(args.full_mc_jeffreys_samples),
        "full_hutchinson_probes": int(args.full_hutchinson_probes),
        "full_likelihood_batch_size": int(args.full_likelihood_batch_size),
        "nll_epochs": int(args.nll_epochs),
        "nll_patience": int(args.nll_patience),
        "nll_batch_size": int(args.nll_batch_size),
        "nll_lr": float(args.nll_lr),
        "nll_ode_steps": int(args.nll_ode_steps),
        "seed": int(args.seed) + int(session_index),
    }


def _fit_session(
    args: argparse.Namespace,
    *,
    session_index: int,
    theta_grid: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    info = list_stringer_sessions("gratings_static")[session_index]
    label = _session_label(info)
    case_dir = args.output_dir / f"session_{session_index:02d}_{label}"
    case_dir.mkdir(parents=True, exist_ok=True)
    estimates_path = case_dir / f"flow_{args.fisher_kind}_fisher.npz"
    metadata_path = case_dir / f"{args.fisher_kind}_metadata.json"
    signature = _signature(
        args, session_file=Path(info.session_file), session_index=int(session_index)
    )
    if estimates_path.is_file() and metadata_path.is_file() and not args.force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            with np.load(estimates_path, allow_pickle=False) as saved:
                return {key: np.asarray(saved[key]) for key in saved.files} | {
                    "metadata": metadata
                }

    session = load_stringer_session(
        None,
        session_stimuli_type="gratings_static",
        session_index=int(session_index),
        orientation_period=PERIOD,
    )
    theta_full = np.asarray(session.grating_orientation, dtype=np.float64)
    responses_full = np.asarray(session.neural_responses)
    selection_seed = int(args.seed) + 10_000 * int(session_index)
    trial_indices = _session_trial_indices(
        theta_full,
        use_full_dataset=bool(args.use_full_dataset),
        n_trials=int(args.n_trials_per_session),
        n_strata=int(args.theta_grid_size) - 1,
        seed=selection_seed,
    )
    theta = theta_full[trial_indices]
    responses = responses_full[trial_indices]
    x, projector = _uncentered_projection(
        responses,
        n_components=int(args.pca_dim),
        random_state=int(args.seed) + int(session_index),
    )
    train, validation = _train_validation_indices(
        theta,
        validation_fraction=float(args.flow_validation_fraction),
        n_strata=int(args.theta_grid_size) - 1,
        seed=int(args.seed) + 100_000 + int(session_index),
    )
    condition = encode_flow_orientation(theta, period=PERIOD, encoding="periodic-rbf")
    condition_grid = encode_flow_orientation(
        theta_grid, period=PERIOD, encoding="periodic-rbf"
    )
    fit_seed = int(args.seed) + int(session_index)
    if str(args.fisher_kind) == "linear":
        model, training, fm_estimate, _ = fit_flow_direction_estimator(
            theta_train=theta[train],
            x_train=x[train],
            theta_validation=theta[validation],
            x_validation=x[validation],
            theta_grid=theta_grid,
            condition_train=condition[train],
            condition_validation=condition[validation],
            condition_grid=condition_grid,
            device=device,
            seed=fit_seed,
            epochs=int(args.epochs),
            patience=int(args.early_patience),
            batch_size=int(args.batch_size),
            learning_rate=float(args.lr),
            hidden_dim=int(args.hidden_dim),
            depth=int(args.depth),
            ode_steps=int(args.ode_steps),
        )
        nll_training: dict[str, Any] | None = None
        estimate = fm_estimate
    else:
        torch.manual_seed(fit_seed)
        np.random.seed(fit_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(fit_seed)
        model = build_flow_skl_model(
            velocity_family="nonlinear",
            theta_dim=int(condition.shape[1]),
            x_dim=int(x.shape[1]),
            hidden_dim=int(args.hidden_dim),
            depth=int(args.depth),
            path_schedule="cosine",
            divergence_estimator="hutchinson",
            hutchinson_probes=int(args.full_hutchinson_probes),
            theta_embedding="identity",
        ).to(device)
        training = train_flow_skl_model(
            model=model,
            theta_train=condition[train],
            x_train=x[train],
            theta_val=condition[validation],
            x_val=x[validation],
            device=device,
            velocity_family="nonlinear",
            path_schedule="cosine",
            epochs=int(args.epochs),
            batch_size=min(int(args.batch_size), int(train.size)),
            lr=float(args.lr),
            lr_schedule="constant",
            weight_decay=0.0,
            t_eps=5e-4,
            patience=int(args.early_patience),
            min_delta=1e-4,
            ema_alpha=0.05,
            max_grad_norm=10.0,
            log_every=50,
            checkpoint_selection="last",
            best_checkpoint_metric="flow_matching",
            fixed_validation=True,
            fixed_validation_paths=10,
            validation_seed=fit_seed + 10_000,
            retain_best_state=True,
        )
        best_state = training.pop("best_state_dict")
        torch.save(model.state_dict(), case_dir / "flow_full_model_last.pt")
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), case_dir / "flow_full_model_best.pt")
        training["selected_epoch"] = int(training["best_epoch"])
        estimate = estimate_adjacent_model_jeffreys_fisher(
            model=model,
            theta_all=theta_grid,
            condition_all=condition_grid,
            device=device,
            mc_jeffreys_sample=int(args.full_mc_jeffreys_samples),
            ode_steps=int(args.ode_steps),
            ode_method="midpoint",
            batch_size=int(args.full_likelihood_batch_size),
            solve_jitter=1e-6,
            quadrature_steps=64,
        )
        fm_estimate = estimate
        nll_training = None
    if str(args.fisher_kind) == "linear" and int(args.nll_epochs) > 0:
        torch.manual_seed(fit_seed + 200_000)
        np.random.seed(fit_seed + 200_000)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(fit_seed + 200_000)
        print(f"[session:{label}] CNF NLL fine-tuning", flush=True)
        nll_training = finetune_flow_skl_cnf_likelihood(
            model=model,
            theta_train=condition[train],
            x_train=x[train],
            theta_val=condition[validation],
            x_val=x[validation],
            device=device,
            epochs=int(args.nll_epochs),
            batch_size=min(int(args.nll_batch_size), int(train.size)),
            lr=float(args.nll_lr),
            weight_decay=0.0,
            ode_steps=int(args.nll_ode_steps),
            ode_method="midpoint",
            patience=int(args.nll_patience),
            min_delta=1e-4,
            ema_alpha=0.05,
            max_grad_norm=10.0,
            checkpoint_selection="best",
            log_every=50,
        )
        estimate = estimate_affine_mixed_symmetric_kl_fisher_for_conditions(
            model=model,
            theta_all=theta_grid,
            condition_all=condition_grid,
            device=device,
            ridge=1e-6,
            ode_steps=int(args.ode_steps),
        )
    fisher = np.asarray(estimate["fisher"], dtype=np.float64).reshape(-1)
    arrays = {
        "theta_midpoints": np.asarray(estimate["theta_midpoints"], dtype=np.float64).reshape(-1),
        f"{args.fisher_kind}_fisher": fisher,
        "train_losses": np.asarray(training["train_losses"], dtype=np.float64),
        "validation_losses": np.asarray(training["val_losses"], dtype=np.float64),
        "nll_train_losses": np.asarray(
            [] if nll_training is None else nll_training["train_nll_losses"],
            dtype=np.float64,
        ),
        "nll_validation_losses": np.asarray(
            [] if nll_training is None else nll_training["val_nll_losses"],
            dtype=np.float64,
        ),
        "trial_indices": trial_indices,
        "train_indices": train,
        "validation_indices": validation,
    }
    if str(args.fisher_kind) == "linear":
        arrays.update(
            {
                "fm_linear_fisher": np.asarray(
                    fm_estimate["fisher"], dtype=np.float64
                ).reshape(-1),
                "delta_mu": np.asarray(estimate["delta_mu"], dtype=np.float64),
                "mixed_covariance": np.asarray(
                    estimate["mixed_covariance"], dtype=np.float64
                ),
            }
        )
    else:
        arrays.update(
            {
                "adjacent_jeffreys": np.asarray(
                    estimate["adjacent_jeffreys"], dtype=np.float64
                ),
                "dtheta": np.asarray(estimate["dtheta"], dtype=np.float64),
            }
        )
    np.savez_compressed(estimates_path, **arrays)
    if str(args.fisher_kind) == "linear":
        torch.save(
            {key: value.detach().cpu() for key, value in model.state_dict().items()},
            case_dir / "flow_selected_model.pt",
        )
    np.savez_compressed(
        case_dir / "pca82_uncentered.npz",
        theta=theta,
        x=x,
        trial_indices=trial_indices,
        components=projector.components_,
        singular_values=projector.singular_values_,
        explained_variance=projector.explained_variance_,
        explained_variance_ratio=projector.explained_variance_ratio_,
    )
    standardizer = model._response_standardizer  # type: ignore[attr-defined]
    metadata = {
        "signature": signature,
        "label": label,
        "session_index": int(session_index),
        "n_observations": int(theta.size),
        "n_observations_full": int(theta_full.size),
        "uses_full_dataset": bool(args.use_full_dataset),
        "n_train": int(train.size),
        "n_validation": int(validation.size),
        "n_neurons": int(responses.shape[1]),
        "projection_explained_variance_ratio_sum": float(
            np.sum(projector.explained_variance_ratio_)
        ),
        "response_standardizer_fitted_on": "flow_training_split_only",
        "response_standardizer_mean_norm": float(
            torch.linalg.vector_norm(standardizer.mean).detach().cpu().item()
        ),
        "response_standardizer_scale_mean": float(
            standardizer.scale.mean().detach().cpu().item()
        ),
        "selected_epoch": int(training["selected_epoch"]),
        "best_epoch": int(training["best_epoch"]),
        "stopped_epoch": int(training["stopped_epoch"]),
        "nll_finetuned": nll_training is not None,
        "nll_initial_validation": (
            None if nll_training is None else float(nll_training["initial_val_nll"])
        ),
        "nll_selected_validation": (
            None if nll_training is None else float(nll_training["selected_val_nll"])
        ),
        "nll_selected_epoch": (
            None if nll_training is None else int(nll_training["selected_epoch"])
        ),
        "nll_stopped_epoch": (
            None if nll_training is None else int(nll_training["stopped_epoch"])
        ),
        "fisher_kind": str(args.fisher_kind),
        f"mean_{args.fisher_kind}_fisher": float(np.mean(fisher)),
    }
    if str(args.fisher_kind) == "linear":
        metadata["mean_fm_linear_fisher"] = float(
            np.mean(np.asarray(fm_estimate["fisher"], dtype=np.float64))
        )
    metadata_path.write_text(
        json.dumps(_json_ready(metadata), indent=2) + "\n", encoding="utf-8"
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays | {"metadata": metadata}


def _plot(
    results: list[dict[str, Any]],
    output_dir: Path,
    fisher_kind: str = "linear",
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )
    fig, axis = plt.subplots(figsize=(5.8, 3.5), constrained_layout=True)
    for result in results:
        axis.plot(
            result["theta_midpoints"],
            result[f"{fisher_kind}_fisher"],
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label=result["metadata"]["label"],
        )
    axis.set_xlabel(r"Orientation $\theta$")
    axis.set_ylabel(f"{fisher_kind.title()} Fisher")
    method_label = "Flow Matching + NLL" if any(
        bool(result["metadata"].get("nll_finetuned")) for result in results
    ) else "Flow Matching"
    axis.set_title(f"Stringer sessions, {method_label} {fisher_kind.title()} Fisher")
    axis.set_xlim(0.0, np.pi)
    axis.set_xticks(
        [0.0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi],
        [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"],
    )
    axis.legend(
        frameon=False,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
        fontsize=12,
    )
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color="0.88", linewidth=0.8)
    axis.xaxis.grid(False)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"stringer_flow_all_sessions_{fisher_kind}_fisher"
    png, svg = stem.with_suffix(".png"), stem.with_suffix(".svg")
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def main() -> int:
    args = parse_args()
    if not 0.0 < float(args.flow_validation_fraction) < 1.0:
        raise ValueError("--flow-validation-fraction must be in (0, 1).")
    if int(args.nll_epochs) < 0:
        raise ValueError("--nll-epochs must be nonnegative.")
    if str(args.fisher_kind) == "full" and int(args.nll_epochs) > 0:
        raise ValueError("--nll-epochs is currently supported only for linear Fisher.")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = require_device(str(args.device))
    sessions = list_stringer_sessions("gratings_static")
    if not sessions:
        raise ValueError("No available gratings_static Stringer sessions.")
    session_indices = (
        list(range(len(sessions)))
        if args.session_indices is None
        else [int(index) for index in args.session_indices]
    )
    if len(set(session_indices)) != len(session_indices):
        raise ValueError("--session-indices must be unique.")
    if any(index < 0 or index >= len(sessions) for index in session_indices):
        raise ValueError("--session-indices contains an out-of-range index.")
    theta_grid = np.linspace(
        0.0, PERIOD, int(args.theta_grid_size), dtype=np.float64
    ).reshape(-1, 1)
    started = time.perf_counter()
    results = []
    for position, session_index in enumerate(session_indices):
        info = sessions[session_index]
        print(
            f"[session] {position + 1}/{len(session_indices)} {_session_label(info)} ",
            f"(global index {session_index})",
            flush=True,
        )
        results.append(
            _fit_session(
                args,
                session_index=session_index,
                theta_grid=theta_grid,
                device=device,
            )
        )
    if args.skip_aggregate:
        print(f"[sessions] runtime_seconds={time.perf_counter() - started:.3f}", flush=True)
        return 0
    png, svg = _plot(results, args.output_dir / "figures", str(args.fisher_kind))
    summary = {
        "n_sessions": len(results),
        "session_indices": session_indices,
        "n_trials_per_session": (
            "all_available" if args.use_full_dataset else int(args.n_trials_per_session)
        ),
        "pca_dim": int(args.pca_dim),
        "projection": "truncated_svd_uncentered_unwhitened",
        "fisher_kind": str(args.fisher_kind),
        "runtime_seconds": float(time.perf_counter() - started),
        "sessions": [result["metadata"] for result in results],
        "artifacts": {"png": str(png), "svg": str(svg)},
    }
    summary_path = args.output_dir / f"stringer_flow_all_sessions_{args.fisher_kind}_summary.json"
    summary_path.write_text(
        json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved: {summary_path}", flush=True)
    print(f"Saved: {png}", flush=True)
    print(f"Saved: {svg}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
