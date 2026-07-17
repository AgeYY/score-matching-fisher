#!/usr/bin/env python3
"""Resume CNF NLL fine-tuning from a saved continuous-Fisher FM checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.distance_comparison import save_flow_result_npz
from fisher.flow_matching_skl import (
    FlowSKLResult,
    build_flow_skl_model,
    estimate_adjacent_model_jeffreys_fisher,
    estimate_affine_mixed_symmetric_kl_fisher,
    finetune_flow_skl_cnf_likelihood,
)
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import require_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--result-npz", type=Path, required=True)
    parser.add_argument("--flow-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("linear", "full"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--quadrature-steps", type=int, default=64)
    parser.add_argument("--path-schedule", default="cosine")
    parser.add_argument("--divergence-estimator", default="hutchinson")
    parser.add_argument("--hutchinson-probes", type=int, default=4)
    parser.add_argument("--shared-affine-a-diag-jitter", type=float, default=1e-3)
    parser.add_argument("--nll-epochs", type=int, default=500)
    parser.add_argument("--nll-batch-size", type=int, default=128)
    parser.add_argument("--nll-lr", type=float, default=1e-3)
    parser.add_argument("--nll-weight-decay", type=float, default=0.0)
    parser.add_argument("--nll-ode-steps", type=int, default=32)
    parser.add_argument("--nll-ode-method", default="midpoint")
    parser.add_argument("--nll-patience", type=int, default=100)
    parser.add_argument("--nll-min-delta", type=float, default=1e-4)
    parser.add_argument("--nll-ema-alpha", type=float, default=0.05)
    parser.add_argument("--ode-steps", type=int, default=64)
    parser.add_argument("--ode-method", default="midpoint")
    parser.add_argument("--mc-jeffreys-sample", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--solve-jitter", type=float, default=1e-6)
    parser.add_argument("--affine-ridge", type=float, default=1e-6)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def load_fm_metadata(path: Path) -> dict[str, object]:
    metadata: dict[str, object] = {}
    with np.load(path, allow_pickle=False) as saved:
        for key in ("train_losses", "val_losses", "val_monitor_losses", "learning_rates"):
            if key in saved.files:
                metadata[key] = np.asarray(saved[key])
        for key in (
            "best_val_loss",
            "best_epoch",
            "selected_epoch",
            "stopped_epoch",
            "stopped_early",
            "early_ema_alpha",
        ):
            if key in saved.files:
                metadata[key] = saved[key].reshape(-1)[0].item()
    return metadata


def merge_nll_curves(result_path: Path, flow_dir: Path) -> None:
    with np.load(result_path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    for family in ("linear", "full"):
        nll_path = flow_dir / f"flow_{family}_nll_flow_matching_skl_results.npz"
        if not nll_path.is_file():
            continue
        with np.load(nll_path, allow_pickle=False) as nll:
            fisher_key = f"fisher_{family}"
            if fisher_key not in nll.files:
                continue
            estimate = np.asarray(nll[fisher_key], dtype=np.float64)
        output_key = f"flow_{family}_nll_fisher"
        truth = np.asarray(
            arrays[f"ground_truth_native_{family}_fisher"], dtype=np.float64
        )
        absolute_error = np.abs(estimate - truth)
        arrays[output_key] = estimate
        arrays[f"flow_{family}_nll_abs_error"] = absolute_error
        arrays[f"flow_{family}_nll_rel_error"] = absolute_error / np.maximum(
            np.abs(truth), 1e-12
        )
    np.savez_compressed(result_path, **arrays)


def main() -> None:
    args = parse_args()
    device = require_device(str(args.device))
    torch.manual_seed(int(args.seed) + 200_000)
    np.random.seed(int(args.seed) + 200_000)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed) + 200_000)

    bundle = load_shared_dataset_npz(args.dataset_npz.resolve())
    with np.load(args.result_npz.resolve(), allow_pickle=False) as result:
        theta_grid = np.asarray(result["theta_grid"], dtype=np.float64)
    family = "condition_affine" if args.method == "linear" else "nonlinear"
    method_key = f"flow_{args.method}"
    nll_method_key = f"{method_key}_nll"
    flow_dir = args.flow_dir.resolve()
    checkpoint_path = flow_dir / f"{method_key}_selected_model.pt"
    baseline_path = flow_dir / f"{method_key}_flow_matching_skl_results.npz"

    model = build_flow_skl_model(
        velocity_family=family,
        theta_dim=int(np.asarray(bundle.theta_train).shape[1]),
        x_dim=int(np.asarray(bundle.x_train).shape[1]),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        quadrature_steps=int(args.quadrature_steps),
        path_schedule=str(args.path_schedule),
        divergence_estimator=str(args.divergence_estimator),
        hutchinson_probes=int(args.hutchinson_probes),
        shared_affine_a_diag_jitter=float(args.shared_affine_a_diag_jitter),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    fm_metadata = load_fm_metadata(baseline_path)
    fine_metadata = finetune_flow_skl_cnf_likelihood(
        model=model,
        theta_train=np.asarray(bundle.theta_train, dtype=np.float64),
        x_train=np.asarray(bundle.x_train, dtype=np.float64),
        theta_val=np.asarray(bundle.theta_validation, dtype=np.float64),
        x_val=np.asarray(bundle.x_validation, dtype=np.float64),
        device=device,
        epochs=int(args.nll_epochs),
        batch_size=int(args.nll_batch_size),
        lr=float(args.nll_lr),
        weight_decay=float(args.nll_weight_decay),
        ode_steps=int(args.nll_ode_steps),
        ode_method=str(args.nll_ode_method),
        patience=int(args.nll_patience),
        min_delta=float(args.nll_min_delta),
        ema_alpha=float(args.nll_ema_alpha),
        checkpoint_selection="best",
        log_every=int(args.log_every),
    )
    train_metadata = {
        **fm_metadata,
        "likelihood_finetuned": True,
        "likelihood_finetune_metadata": fine_metadata,
    }
    if int(fine_metadata["selected_epoch"]) == 0:
        with np.load(baseline_path, allow_pickle=False) as baseline:
            flow_result = FlowSKLResult(
                symmetric_kl_matrix=np.asarray(
                    baseline["symmetric_kl_matrix"], dtype=np.float64
                ),
                canonical_metric_matrix=np.asarray(
                    baseline["canonical_metric_matrix"], dtype=np.float64
                ),
                canonical_metric_name=(
                    "affine_mixed_symmetric_kl"
                    if args.method == "linear"
                    else "adjacent_model_jeffreys_sum"
                ),
                fisher_theta_midpoints=np.asarray(
                    baseline["fisher_theta_midpoints"], dtype=np.float64
                ),
                fisher_linear=(
                    np.asarray(baseline["fisher_linear"], dtype=np.float64)
                    if "fisher_linear" in baseline.files
                    else None
                ),
                fisher_full=(
                    np.asarray(baseline["fisher_full"], dtype=np.float64)
                    if "fisher_full" in baseline.files
                    else None
                ),
                train_metadata=train_metadata,
            )
        print(
            "[continuous-nll] selected epoch 0; reusing the baseline Fisher curve",
            flush=True,
        )
    elif args.method == "linear":
        estimate = estimate_affine_mixed_symmetric_kl_fisher(
            model=model,
            theta_all=theta_grid,
            device=device,
            ridge=float(args.affine_ridge),
            ode_steps=int(args.ode_steps),
        )
        flow_result = FlowSKLResult(
            symmetric_kl_matrix=np.asarray(estimate["symmetric_kl_matrix"]),
            canonical_metric_matrix=np.asarray(estimate["canonical_metric_matrix"]),
            canonical_metric_name=str(estimate["canonical_metric_name"]),
            fisher_theta_midpoints=estimate["theta_midpoints"],
            fisher_linear=estimate["fisher"],
            train_metadata=train_metadata,
        )
    else:
        estimate = estimate_adjacent_model_jeffreys_fisher(
            model=model,
            theta_all=theta_grid,
            device=device,
            mc_jeffreys_sample=int(args.mc_jeffreys_sample),
            ode_steps=int(args.ode_steps),
            ode_method=str(args.ode_method),
            batch_size=int(args.batch_size),
            solve_jitter=float(args.solve_jitter),
            quadrature_steps=int(args.quadrature_steps),
        )
        matrix_shape = (int(theta_grid.shape[0]), int(theta_grid.shape[0]))
        flow_result = FlowSKLResult(
            symmetric_kl_matrix=np.zeros(matrix_shape, dtype=np.float64),
            canonical_metric_matrix=np.zeros(matrix_shape, dtype=np.float64),
            canonical_metric_name="adjacent_model_jeffreys_sum",
            fisher_theta_midpoints=estimate["theta_midpoints"],
            fisher_full=estimate["fisher"],
            train_metadata=train_metadata,
        )

    nll_path = flow_dir / f"{nll_method_key}_flow_matching_skl_results.npz"
    save_flow_result_npz(
        nll_path,
        result=flow_result,
        metric=nll_method_key,
        theta_eval=theta_grid,
        velocity_family=family,
        estimator="flow_matching_nll_finetuned",
    )
    torch.save(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        flow_dir / f"{nll_method_key}_selected_model.pt",
    )
    merge_nll_curves(args.result_npz.resolve(), flow_dir)
    print(f"nll_result_npz: {nll_path}")
    print(f"updated_result_npz: {args.result_npz.resolve()}")


if __name__ == "__main__":
    main()
