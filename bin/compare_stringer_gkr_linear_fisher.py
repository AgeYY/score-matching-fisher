#!/usr/bin/env python3
"""Compare GKR with classical and flow-based Stringer Fisher identification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fisher.gkr import GKRConfig
from fisher.shared_fisher_est import require_device
from fisher.stringer_dataset import list_stringer_sessions
from fisher.stringer_gkr_comparison import (
    RESULTS_NPZ_NAME,
    SUMMARY_JSON_NAME,
    TOPK_PNG_NAME,
    TOPK_SVG_NAME,
    plot_topk_comparison,
    run_gkr_comparison,
    save_comparison,
    save_summary,
)
from fisher.stringer_session_identification import parse_optional_int, parse_positive_int_list
from global_setting import DATA_DIR, DEFAULT_DEVICE


def _default_baseline_dir() -> Path:
    return (
        Path(DATA_DIR)
        / "stringer_fisher_session_identification"
        / "gratings_static"
        / "a_subsample_convergence_periodic_sincos_fmval10paths_matched80_r5"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--baseline-dir", type=Path, default=_default_baseline_dir())
    parser.add_argument("--output-dir", type=Path, default=Path(DATA_DIR) / "stringer_gkr_linear_fisher_comparison")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--session-stimuli-type", default="gratings_static")
    parser.add_argument("--max-sessions", type=parse_optional_int, default=None)
    parser.add_argument("--n-list", default=None, help="Subset of baseline A-half sample sizes; default uses all.")
    parser.add_argument(
        "--repeats",
        type=parse_optional_int,
        default=None,
        help="Use the first R baseline repeats; default uses all.",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--gkr-mean-iterations", type=int, default=300)
    parser.add_argument("--gkr-mean-lr", type=float, default=0.05)
    parser.add_argument("--gkr-mean-batch-size", type=parse_optional_int, default=None)
    parser.add_argument("--gkr-n-inducing", type=parse_optional_int, default=200)
    parser.add_argument("--gkr-cov-epochs", type=int, default=30)
    parser.add_argument("--gkr-cov-lr", type=float, default=0.1)
    parser.add_argument("--gkr-cov-batch-size", type=int, default=3000)
    parser.add_argument("--gkr-validation-fraction", type=float, default=0.33)
    parser.add_argument("--gkr-cov-jitter", type=float, default=1e-6)
    parser.add_argument("--gkr-likelihood-jitter", type=float, default=1e-5)
    parser.add_argument("--gkr-prediction-batch-size", type=int, default=3000)
    parser.add_argument("--gkr-solve-jitter", type=float, default=1e-6)
    parser.add_argument("--gkr-log-every", type=int, default=25)
    parser.add_argument("--gkr-standardize-responses", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _validate(args: argparse.Namespace) -> None:
    if args.gkr_mean_iterations < 1 or args.gkr_cov_epochs < 1:
        raise ValueError("GKR iteration and epoch counts must be positive.")
    if args.gkr_cov_batch_size < 1 or args.gkr_prediction_batch_size < 1:
        raise ValueError("GKR batch sizes must be positive.")
    if not 0.0 < args.gkr_validation_fraction < 1.0:
        raise ValueError("--gkr-validation-fraction must be in (0, 1).")


def run(args: argparse.Namespace) -> dict[str, Path]:
    _validate(args)
    device = require_device(str(args.device))
    sessions = list_stringer_sessions(str(args.session_stimuli_type), data_dir=args.data_dir)
    n_values = None if args.n_list is None else parse_positive_int_list(str(args.n_list))
    config = GKRConfig(
        mean_iterations=int(args.gkr_mean_iterations),
        mean_learning_rate=float(args.gkr_mean_lr),
        mean_batch_size=args.gkr_mean_batch_size,
        n_inducing=args.gkr_n_inducing,
        covariance_epochs=int(args.gkr_cov_epochs),
        covariance_learning_rate=float(args.gkr_cov_lr),
        covariance_batch_size=int(args.gkr_cov_batch_size),
        validation_fraction=float(args.gkr_validation_fraction),
        covariance_jitter=float(args.gkr_cov_jitter),
        likelihood_jitter=float(args.gkr_likelihood_jitter),
        prediction_batch_size=int(args.gkr_prediction_batch_size),
        standardize_responses=bool(args.gkr_standardize_responses),
        log_every=int(args.gkr_log_every),
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_gkr_comparison(
        sessions=sessions,
        baseline_dir=Path(args.baseline_dir).expanduser().resolve(),
        output_dir=output_dir,
        device=device,
        gkr_config=config,
        gkr_solve_jitter=float(args.gkr_solve_jitter),
        n_values=n_values,
        repeats=args.repeats,
        max_sessions=args.max_sessions,
        force=bool(args.force),
    )
    results_npz = save_comparison(output_dir / RESULTS_NPZ_NAME, result)
    summary_json = save_summary(output_dir / SUMMARY_JSON_NAME, result)
    figure_svg, figure_png = plot_topk_comparison(output_dir / TOPK_SVG_NAME, output_dir / TOPK_PNG_NAME, result)
    outputs = {
        "output_dir": output_dir,
        "results_npz": results_npz,
        "summary_json": summary_json,
        "figure_svg": figure_svg,
        "figure_png": figure_png,
    }
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2), flush=True)
    return outputs


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
