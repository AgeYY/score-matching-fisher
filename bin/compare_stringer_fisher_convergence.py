#!/usr/bin/env python3
"""Compare Stringer single-session linear Fisher convergence estimators."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.continuous_fisher_comparison import ContinuousFlowConfig
from fisher.shared_fisher_est import require_device
from fisher.stringer_dataset import load_stringer_session
from fisher.stringer_fisher_convergence import (
    ABS_ERROR_PNG_NAME,
    ABS_ERROR_SVG_NAME,
    CURVE_EXAMPLES_PNG_NAME,
    CURVE_EXAMPLES_SVG_NAME,
    CURVES_CSV_NAME,
    RESULTS_NPZ_NAME,
    SUMMARY_JSON_NAME,
    fit_pca_projection,
    parse_int_list,
    plot_abs_error,
    plot_curve_examples,
    run_linear_fisher_convergence,
    theta_grid_periodic,
    write_curves_csv,
    write_results_npz,
    write_summary_json,
)
from global_setting import DATA_DIR, DEFAULT_DEVICE, STRINGER_EXAMPLE_SESSION_FILE


def _default_data_root() -> Path:
    root = Path(DATA_DIR).expanduser()
    if not root.is_absolute():
        root = _REPO_ROOT / root
    return root.resolve()


def default_output_dir(session_file: str = STRINGER_EXAMPLE_SESSION_FILE) -> Path:
    return _default_data_root() / "stringer_fisher_convergence" / Path(session_file).stem


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--session", type=str, default=STRINGER_EXAMPLE_SESSION_FILE)
    p.add_argument("--session-stimuli-type", default="gratings_static")
    p.add_argument("--session-index", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--orientation-period", type=float, default=float(np.pi))
    p.add_argument("--theta-grid-size", type=int, default=17)
    p.add_argument("--pca-dim", type=int, default=50)
    p.add_argument("--pca-random-state", type=int, default=0)
    p.add_argument("--no-pca-whiten", action="store_true")
    p.add_argument("--n-list", type=parse_int_list, default=parse_int_list("1000,1500,2000,3000,4000"))
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--classical-window-radius", type=float, default=None)
    p.add_argument("--classical-min-endpoint-samples", type=int, default=8)
    p.add_argument("--classical-linear-ridge", type=float, default=1e-6)
    p.add_argument("--skip-flow-npz", action="store_true")

    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--early-patience", type=int, default=1000)
    p.add_argument("--early-min-delta", type=float, default=1e-4)
    p.add_argument("--early-ema-alpha", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--path-schedule", choices=("cosine", "linear", "straight"), default="cosine")
    p.add_argument("--t-eps", type=float, default=0.0005)
    p.add_argument("--quadrature-steps", type=int, default=64)
    p.add_argument("--ode-steps", type=int, default=64)
    p.add_argument("--divergence-estimator", choices=("hutchinson", "exact"), default="exact")
    p.add_argument("--hutchinson-probes", type=int, default=1)
    p.add_argument("--shared-affine-a-diag-jitter", type=float, default=1e-3)
    p.add_argument("--solve-jitter", type=float, default=1e-6)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--affine-ridge", type=float, default=1e-6)
    return p


def flow_config_from_args(args: argparse.Namespace) -> ContinuousFlowConfig:
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
        mc_jeffreys_sample=0,
        ode_steps=int(args.ode_steps),
        ode_method="midpoint",
        divergence_estimator=str(args.divergence_estimator),
        hutchinson_probes=int(args.hutchinson_probes),
        shared_affine_a_diag_jitter=float(args.shared_affine_a_diag_jitter),
        solve_jitter=float(args.solve_jitter),
        max_grad_norm=float(args.max_grad_norm),
        log_every=int(args.log_every),
        affine_ridge=float(args.affine_ridge),
    )


def validate_args(args: argparse.Namespace) -> None:
    if int(args.theta_grid_size) < 2:
        raise ValueError("--theta-grid-size must be >= 2.")
    if int(args.pca_dim) < 1:
        raise ValueError("--pca-dim must be >= 1.")
    if int(args.n_repeats) < 1:
        raise ValueError("--n-repeats must be >= 1.")
    if not (0.0 < float(args.train_frac) < 1.0):
        raise ValueError("--train-frac must be in (0, 1).")
    if float(args.orientation_period) <= 0.0:
        raise ValueError("--orientation-period must be positive.")


def run(args: argparse.Namespace) -> dict[str, Path]:
    validate_args(args)
    device = require_device(str(args.device))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    output_dir = Path(args.output_dir).expanduser() if args.output_dir is not None else default_output_dir(str(args.session))
    if not output_dir.is_absolute():
        output_dir = (_REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stringer-convergence] loading session {args.session}", flush=True)
    session = load_stringer_session(
        session=args.session,
        session_stimuli_type=str(args.session_stimuli_type),
        session_index=int(args.session_index),
        data_dir=args.data_dir,
        orientation_period=float(args.orientation_period),
    )
    theta = np.mod(np.asarray(session.grating_orientation, dtype=np.float64).reshape(-1), float(args.orientation_period))
    print(f"[stringer-convergence] responses={session.neural_responses.shape} pca_dim={int(args.pca_dim)}", flush=True)
    pca = fit_pca_projection(
        session.neural_responses,
        n_components=int(args.pca_dim),
        random_state=int(args.pca_random_state),
        whiten=not bool(args.no_pca_whiten),
    )
    grid = theta_grid_periodic(float(args.orientation_period), int(args.theta_grid_size))

    metadata = {
        "script": "bin/compare_stringer_fisher_convergence.py",
        "session_file": str(session.session_file),
        "session_stimuli_type": str(session.session_stimuli_type),
        "stringer_meta": session.meta,
        "feature_space": "pca",
        "pca": pca.metadata,
        "pca_explained_variance_ratio": pca.explained_variance_ratio.tolist(),
        "pca_label_blind": True,
        "pca_trial_averaging_before_fit": False,
        "subsampling": "orientation-stratified",
        "reference": "method-specific all-data reference",
    }
    result = run_linear_fisher_convergence(
        theta_all=theta,
        x_all=pca.x_all,
        theta_grid=grid,
        period=float(args.orientation_period),
        n_list=args.n_list,
        n_repeats=int(args.n_repeats),
        train_frac=float(args.train_frac),
        seed=int(args.seed),
        device=device,
        flow_config=flow_config_from_args(args),
        output_dir=output_dir,
        classical_ridge=float(args.classical_linear_ridge),
        classical_window_radius=args.classical_window_radius,
        classical_min_endpoint_samples=int(args.classical_min_endpoint_samples),
        save_flow_npz=not bool(args.skip_flow_npz),
        metadata=metadata,
    )

    results_npz = write_results_npz(output_dir / RESULTS_NPZ_NAME, result, pca=pca)
    curves_csv = write_curves_csv(output_dir / CURVES_CSV_NAME, result.rows)
    error_svg, error_png = plot_abs_error(output_dir / ABS_ERROR_SVG_NAME, output_dir / ABS_ERROR_PNG_NAME, result)
    curves_svg, curves_png = plot_curve_examples(output_dir / CURVE_EXAMPLES_SVG_NAME, output_dir / CURVE_EXAMPLES_PNG_NAME, result)
    summary_json = write_summary_json(
        output_dir / SUMMARY_JSON_NAME,
        result,
        extra={
            "device": str(device),
            "output_dir": str(output_dir),
            "results_npz": str(results_npz),
            "curves_csv": str(curves_csv),
            "abs_error_svg": str(error_svg),
            "abs_error_png": str(error_png),
            "curve_examples_svg": str(curves_svg),
            "curve_examples_png": str(curves_png),
            "flow_config": vars(flow_config_from_args(args)),
            "classical_config": {
                "linear_ridge": float(args.classical_linear_ridge),
                "window_radius": args.classical_window_radius,
                "min_endpoint_samples": int(args.classical_min_endpoint_samples),
            },
            "n_list": [int(v) for v in args.n_list],
            "n_repeats": int(args.n_repeats),
        },
    )
    for label, path in (
        ("results_npz", results_npz),
        ("curves_csv", curves_csv),
        ("summary_json", summary_json),
        ("abs_error_svg", error_svg),
        ("curve_examples_svg", curves_svg),
    ):
        print(f"{label}: {path}", flush=True)
    return {
        "output_dir": output_dir,
        "results_npz": results_npz,
        "curves_csv": curves_csv,
        "summary_json": summary_json,
        "abs_error_svg": error_svg,
        "abs_error_png": error_png,
        "curve_examples_svg": curves_svg,
        "curve_examples_png": curves_png,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
