#!/usr/bin/env python3
"""Compare continuous scalar Fisher estimators on native or PR-embedded randamp data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.continuous_fisher_comparison import (
    CURVES_CSV_NAME,
    CURVES_PNG_NAME,
    CURVES_SVG_NAME,
    RESULTS_NPZ_NAME,
    SUMMARY_JSON_NAME,
    ClassicalConfig,
    ContinuousFlowConfig,
    make_native_dataset_npz,
    parse_pr_dim,
    plot_curves,
    project_pr_dataset_npz,
    run_continuous_comparison,
    write_curves_csv,
    write_results_npz,
    write_summary_json,
)
from global_setting import DEFAULT_DEVICE
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import require_device


def default_dataset_dir(*, dataset_family: str = "randamp_gaussian_sqrtd", native_x_dim: int = 3, n_total: int = 1000, pr_dim: int | None = None) -> Path:
    suffix = "native" if pr_dim is None else f"pr{int(pr_dim)}"
    return _REPO_ROOT / "data" / f"{dataset_family}_xdim{int(native_x_dim)}_{suffix}_n{int(n_total)}"


def default_output_dir(*, dataset_family: str = "randamp_gaussian_sqrtd", native_x_dim: int = 3, n_total: int = 1000, pr_dim: int | None = None) -> Path:
    return default_dataset_dir(
        dataset_family=dataset_family,
        native_x_dim=int(native_x_dim),
        n_total=int(n_total),
        pr_dim=pr_dim,
    ) / "continuous_pr_fisher"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset-family", default="randamp_gaussian_sqrtd")
    p.add_argument("--native-x-dim", type=int, default=3)
    p.add_argument("--n-total", type=int, default=1000)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--theta-grid-size", type=int, default=31)
    p.add_argument("--pr-dim", type=parse_pr_dim, default=None, help="Use 'none' for native mode or an integer PR h_dim.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--force-dataset", action="store_true")
    p.add_argument("--dataset-use-cache", action="store_true")
    p.add_argument("--pr-cache-dir", type=Path, default=_REPO_ROOT / "data" / "pr_autoencoder_cache")
    p.add_argument("--pr-train-epochs", type=int, default=None)
    p.add_argument("--pr-train-samples", type=int, default=None)
    p.add_argument("--pr-train-batch-size", type=int, default=None)
    p.add_argument("--pr-train-lr", type=float, default=None)
    p.add_argument("--skip-pr-viz", action="store_true", default=True)
    p.add_argument("--gt-pr-samples-per-endpoint", type=int, default=20_000)
    p.add_argument("--gt-batch-size", type=int, default=8192)

    p.add_argument("--classical-window-radius", type=float, default=None)
    p.add_argument("--classical-min-endpoint-samples", type=int, default=8)
    p.add_argument("--classical-linear-ridge", type=float, default=1e-6)
    p.add_argument("--skl-folds", type=int, default=5)
    p.add_argument("--skl-logistic-c", type=float, default=1.0)

    p.add_argument("--epochs", type=int, default=20_000)
    p.add_argument("--early-patience", type=int, default=1_000)
    p.add_argument("--early-min-delta", type=float, default=1e-4)
    p.add_argument("--early-ema-alpha", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--path-schedule", choices=("cosine", "linear", "straight"), default="cosine")
    p.add_argument("--t-eps", type=float, default=0.0005)
    p.add_argument("--quadrature-steps", type=int, default=64)
    p.add_argument("--mc-jeffreys-sample", type=int, default=4096)
    p.add_argument("--ode-steps", type=int, default=64)
    p.add_argument("--ode-method", type=str, default="midpoint")
    p.add_argument("--divergence-estimator", choices=("hutchinson", "exact"), default="exact")
    p.add_argument("--hutchinson-probes", type=int, default=1)
    p.add_argument("--shared-affine-a-diag-jitter", type=float, default=1e-3)
    p.add_argument("--solve-jitter", type=float, default=1e-6)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--affine-ridge", type=float, default=1e-6)
    return p


def _flow_config_from_args(args: argparse.Namespace) -> ContinuousFlowConfig:
    return ContinuousFlowConfig(
        epochs=int(args.epochs),
        early_patience=int(args.early_patience),
        early_min_delta=float(args.early_min_delta),
        early_ema_alpha=float(args.early_ema_alpha),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        path_schedule=str(args.path_schedule),
        t_eps=float(args.t_eps),
        quadrature_steps=int(args.quadrature_steps),
        mc_jeffreys_sample=int(args.mc_jeffreys_sample),
        ode_steps=int(args.ode_steps),
        ode_method=str(args.ode_method),
        divergence_estimator=str(args.divergence_estimator),
        hutchinson_probes=int(args.hutchinson_probes),
        shared_affine_a_diag_jitter=float(args.shared_affine_a_diag_jitter),
        solve_jitter=float(args.solve_jitter),
        max_grad_norm=float(args.max_grad_norm),
        log_every=int(args.log_every),
        affine_ridge=float(args.affine_ridge),
    )


def _classical_config_from_args(args: argparse.Namespace) -> ClassicalConfig:
    return ClassicalConfig(
        linear_ridge=float(args.classical_linear_ridge),
        window_radius=args.classical_window_radius,
        min_endpoint_samples=int(args.classical_min_endpoint_samples),
        skl_folds=int(args.skl_folds),
        skl_logistic_c=float(args.skl_logistic_c),
    )


def validate_args(args: argparse.Namespace) -> None:
    if str(args.dataset_family) != "randamp_gaussian_sqrtd":
        raise ValueError("This benchmark currently expects --dataset-family randamp_gaussian_sqrtd.")
    if int(args.native_x_dim) < 1:
        raise ValueError("--native-x-dim must be >= 1.")
    if args.pr_dim is not None and int(args.pr_dim) < int(args.native_x_dim):
        raise ValueError("--pr-dim must be >= --native-x-dim.")
    if not (0.0 < float(args.train_frac) < 1.0):
        raise ValueError("--train-frac must be in (0, 1).")


def resolve_dataset_dir(args: argparse.Namespace) -> Path:
    if args.dataset_dir is not None:
        return Path(args.dataset_dir).expanduser()
    return default_dataset_dir(
        dataset_family=str(args.dataset_family),
        native_x_dim=int(args.native_x_dim),
        n_total=int(args.n_total),
        pr_dim=args.pr_dim,
    )


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).expanduser()
    return resolve_dataset_dir(args) / "continuous_pr_fisher"


def ensure_dataset(args: argparse.Namespace, dataset_dir: Path) -> tuple[Path, Path | None]:
    native_npz = dataset_dir / f"{args.dataset_family}_xdim{int(args.native_x_dim)}_native.npz"
    make_native_dataset_npz(
        output_npz=native_npz,
        dataset_family=str(args.dataset_family),
        x_dim=int(args.native_x_dim),
        n_total=int(args.n_total),
        train_frac=float(args.train_frac),
        seed=int(args.seed),
        force=bool(args.force_dataset),
    )
    if args.pr_dim is None:
        return native_npz, None
    projected_npz = dataset_dir / f"{args.dataset_family}_xdim{int(args.native_x_dim)}_pr{int(args.pr_dim)}d.npz"
    project_pr_dataset_npz(
        input_npz=native_npz,
        output_npz=projected_npz,
        pr_dim=int(args.pr_dim),
        device=str(args.device),
        seed=int(args.seed),
        cache_dir=Path(args.pr_cache_dir),
        use_cache=bool(args.dataset_use_cache),
        force=bool(args.force_dataset),
        pr_train_epochs=args.pr_train_epochs,
        pr_train_samples=args.pr_train_samples,
        pr_train_batch_size=args.pr_train_batch_size,
        pr_train_lr=args.pr_train_lr,
        skip_viz=bool(args.skip_pr_viz),
    )
    return native_npz, projected_npz


def run(args: argparse.Namespace) -> dict[str, Path]:
    validate_args(args)
    dev = require_device(str(args.device))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    dataset_dir = resolve_dataset_dir(args)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    native_npz, projected_npz = ensure_dataset(args, dataset_dir)
    work_npz = projected_npz if args.pr_dim is not None else native_npz
    if work_npz is None:
        raise RuntimeError("No work dataset was produced.")

    native_bundle = load_shared_dataset_npz(native_npz)
    work_bundle = load_shared_dataset_npz(work_npz)
    result = run_continuous_comparison(
        native_bundle=native_bundle,
        work_bundle=work_bundle,
        theta_grid_size=int(args.theta_grid_size),
        device=dev,
        output_dir=output_dir,
        flow_config=_flow_config_from_args(args),
        classical_config=_classical_config_from_args(args),
        seed=int(args.seed),
        pr_projected=args.pr_dim is not None,
        pr_cache_dir=Path(args.pr_cache_dir),
        gt_pr_samples_per_endpoint=int(args.gt_pr_samples_per_endpoint),
        gt_batch_size=int(args.gt_batch_size),
    )
    results_npz = write_results_npz(output_dir / RESULTS_NPZ_NAME, result)
    curves_csv = write_curves_csv(output_dir / CURVES_CSV_NAME, result.rows)
    curves_svg, curves_png = plot_curves(output_dir / CURVES_SVG_NAME, output_dir / CURVES_PNG_NAME, result)
    summary_json = write_summary_json(
        output_dir / SUMMARY_JSON_NAME,
        result,
        extra={
            "script": "bin/compare_continuous_pr_fisher.py",
            "device": str(dev),
            "dataset_family": str(args.dataset_family),
            "native_x_dim": int(args.native_x_dim),
            "n_total": int(args.n_total),
            "train_frac": float(args.train_frac),
            "theta_grid_size": int(args.theta_grid_size),
            "pr_projected": args.pr_dim is not None,
            "pr_dim": None if args.pr_dim is None else int(args.pr_dim),
            "seed": int(args.seed),
            "dataset_dir": str(dataset_dir),
            "native_npz": str(native_npz),
            "work_npz": str(work_npz),
            "projected_npz": None if projected_npz is None else str(projected_npz),
            "output_dir": str(output_dir),
            "results_npz": str(results_npz),
            "curves_csv": str(curves_csv),
            "curves_svg": str(curves_svg),
            "curves_png": str(curves_png),
            "flow_defaults": vars(_flow_config_from_args(args)),
            "classical_defaults": vars(_classical_config_from_args(args)),
        },
    )
    print(f"results_npz: {results_npz}", flush=True)
    print(f"curves_csv: {curves_csv}", flush=True)
    print(f"summary_json: {summary_json}", flush=True)
    return {
        "output_dir": output_dir,
        "results_npz": results_npz,
        "curves_csv": curves_csv,
        "summary_json": summary_json,
        "curves_svg": curves_svg,
        "curves_png": curves_png,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
