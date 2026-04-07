"""Argparse helpers for shared-dataset Fisher scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from global_setting import DATA_DIR


def add_dataset_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--dataset-family",
        type=str,
        default="gmm_non_gauss",
        choices=["gaussian", "gmm_non_gauss", "cos_sin_piecewise_noise", "linear_piecewise_noise"],
    )
    p.add_argument(
        "--tuning-curve-family",
        type=str,
        default="cosine",
        choices=["cosine", "von_mises_raw"],
        help="Mean tuning curve: cosine (default) or raw Von Mises form A*exp(kappa*cos(omega*theta-phi_j)).",
    )
    p.add_argument(
        "--vm-mu-amp",
        type=float,
        default=1.0,
        help="Amplitude A for von_mises_raw tuning curves (ignored for cosine).",
    )
    p.add_argument(
        "--vm-kappa",
        type=float,
        default=1.0,
        help="Concentration kappa >= 0 for von_mises_raw tuning curves (ignored for cosine).",
    )
    p.add_argument(
        "--vm-omega",
        type=float,
        default=1.0,
        help="Angular frequency omega for von_mises_raw tuning curves (ignored for cosine).",
    )
    p.add_argument("--theta-low", type=float, default=-3.0)
    p.add_argument("--theta-high", type=float, default=3.0)
    p.add_argument("--x-dim", type=int, default=2)
    p.add_argument("--sigma-x1", type=float, default=0.30)
    p.add_argument("--sigma-x2", type=float, default=0.30, help="Default matches --sigma-x1 so baseline noise is constant across dims.")
    p.add_argument("--rho", type=float, default=0.15)
    p.add_argument("--cov-theta-amp1", type=float, default=0.35)
    p.add_argument("--cov-theta-amp2", type=float, default=0.30)
    p.add_argument("--cov-theta-amp-rho", type=float, default=0.30)
    p.add_argument("--cov-theta-freq1", type=float, default=0.90)
    p.add_argument("--cov-theta-freq2", type=float, default=0.75)
    p.add_argument("--cov-theta-freq-rho", type=float, default=1.10)
    p.add_argument("--cov-theta-phase1", type=float, default=0.20)
    p.add_argument("--cov-theta-phase2", type=float, default=-0.35)
    p.add_argument("--cov-theta-phase-rho", type=float, default=0.40)
    p.add_argument("--rho-clip", type=float, default=0.85)
    p.add_argument("--gmm-sep-scale", type=float, default=1.10)
    p.add_argument("--gmm-sep-freq", type=float, default=0.85)
    p.add_argument("--gmm-sep-phase", type=float, default=0.35)
    p.add_argument("--gmm-mix-logit-scale", type=float, default=1.40)
    p.add_argument("--gmm-mix-bias", type=float, default=0.00)
    p.add_argument("--gmm-mix-freq", type=float, default=0.95)
    p.add_argument("--gmm-mix-phase", type=float, default=-0.20)
    p.add_argument(
        "--sigma-piecewise-low",
        type=float,
        default=0.30,
        help="Piecewise scalar std for cos_sin/linear piecewise_noise when theta is on the low-noise side.",
    )
    p.add_argument(
        "--sigma-piecewise-high",
        type=float,
        default=0.90,
        help="Piecewise scalar std for cos_sin/linear piecewise_noise when theta is on the high-noise side.",
    )
    p.add_argument(
        "--linear-k",
        type=float,
        default=1.0,
        help="For linear_piecewise_noise: first component mean is linear_k * theta (second is theta).",
    )
    p.add_argument(
        "--theta-zero-to-low",
        action="store_true",
        default=True,
        help="For cos_sin/linear piecewise_noise: include theta=0 in the low-noise side (theta<=0 low, theta>0 high).",
    )
    p.add_argument(
        "--no-theta-zero-to-low",
        action="store_false",
        dest="theta_zero_to_low",
        help="For cos_sin/linear piecewise_noise: put theta=0 in the high-noise side (theta<0 low, theta>=0 high).",
    )
    p.add_argument("--n-total", type=int, default=3000)
    p.add_argument(
        "--train-frac",
        type=float,
        default=1.0,
        help="Fraction of n_total in train_idx; 1.0 means no held-out theta_eval/x_eval split.",
    )


def add_estimation_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--gt-mc-samples-per-bin", type=int, default=6000)
    p.add_argument(
        "--theta-field-method",
        type=str,
        default="dsm",
        choices=["dsm", "flow"],
        help="Scalar field for theta-derivative pipeline: denoising score model (dsm) or flow velocity model (flow).",
    )

    p.add_argument("--score-epochs", type=int, default=10000)
    p.add_argument("--score-batch-size", type=int, default=256)
    p.add_argument("--score-lr", type=float, default=1e-3)
    p.add_argument("--score-hidden-dim", type=int, default=128)
    p.add_argument("--score-depth", type=int, default=3)
    p.add_argument(
        "--score-data-mode",
        type=str,
        default="full",
        choices=["split", "full"],
        help="split: train score on theta_train only; full: train score on all (theta_all, x_all).",
    )
    p.add_argument(
        "--score-fisher-eval-data",
        type=str,
        default="full",
        choices=["score_eval", "full"],
        help="Data split used for score-based Fisher evaluation after training.",
    )
    p.add_argument("--score-val-source", type=str, default="train_split", choices=["train_split", "eval_set"])
    p.add_argument("--score-early-patience", type=int, default=2000)
    p.add_argument("--score-early-min-delta", type=float, default=1e-4)
    p.add_argument(
        "--score-early-ema-alpha",
        type=float,
        default=0.05,
        help="EMA smoothing factor α in (0,1] for validation loss monitor used by score early stopping.",
    )
    p.add_argument("--score-restore-best", action="store_true", default=True)
    p.add_argument("--no-score-restore-best", action="store_false", dest="score_restore_best")
    p.add_argument("--score-noise-mode", type=str, default="continuous", choices=["discrete", "continuous"])
    p.add_argument(
        "--score-sigma-scale-mode",
        type=str,
        default="theta_std",
        choices=["theta_std", "posterior_proxy", "fixed"],
    )
    p.add_argument("--score-sigma-alpha-list", type=float, nargs="+", default=[0.08, 0.06, 0.045, 0.03, 0.02])
    p.add_argument("--score-sigma-min-alpha", type=float, default=0.01)
    p.add_argument("--score-sigma-max-alpha", type=float, default=0.25)
    p.add_argument("--score-eval-sigmas", type=int, default=12)
    p.add_argument("--score-proxy-l2", type=float, default=1e-3)
    p.add_argument("--score-proxy-min-mult", type=float, default=0.1)
    p.add_argument("--score-proxy-max-mult", type=float, default=2.0)
    p.add_argument("--score-fixed-sigma", type=float, default=0.02)
    p.add_argument("--flow-epochs", type=int, default=10000)
    p.add_argument("--flow-batch-size", type=int, default=256)
    p.add_argument("--flow-lr", type=float, default=1e-3)
    p.add_argument("--flow-hidden-dim", type=int, default=128)
    p.add_argument("--flow-depth", type=int, default=3)
    p.add_argument("--flow-scheduler", type=str, default="cosine", choices=["cosine", "vp", "linear_vp"])
    p.add_argument(
        "--flow-eval-t",
        type=float,
        default=0.5,
        help="Fixed time t used to evaluate theta-flow velocity field for H-matrix (flow mode).",
    )
    p.add_argument("--flow-early-patience", type=int, default=1000)
    p.add_argument("--flow-early-min-delta", type=float, default=1e-4)
    p.add_argument(
        "--flow-early-ema-alpha",
        type=float,
        default=0.05,
        help="EMA smoothing factor α in (0,1] for flow validation monitor used by early stopping.",
    )
    p.add_argument("--flow-restore-best", action="store_true", default=True)
    p.add_argument("--no-flow-restore-best", action="store_false", dest="flow_restore_best")

    p.add_argument(
        "--no-prior-score",
        action="store_false",
        dest="prior_enable",
        default=True,
        help="Disable unconditional prior score DSM; use posterior-only Fisher (legacy behavior).",
    )
    p.add_argument(
        "--fisher-score-mode",
        type=str,
        default="posterior_minus_prior",
        choices=["posterior_only", "posterior_minus_prior"],
        help="When prior score is trained: which curve to treat as primary vs GT (combined uses s_post - s_prior).",
    )
    p.add_argument("--prior-epochs", type=int, default=10000)
    p.add_argument("--prior-batch-size", type=int, default=256)
    p.add_argument("--prior-lr", type=float, default=1e-3)
    p.add_argument("--prior-hidden-dim", type=int, default=128)
    p.add_argument("--prior-depth", type=int, default=3)
    p.add_argument("--prior-early-patience", type=int, default=1000)
    p.add_argument("--prior-early-min-delta", type=float, default=1e-4)
    p.add_argument(
        "--prior-early-ema-alpha",
        type=float,
        default=0.05,
        help="EMA α for prior model validation monitor (early stopping).",
    )
    p.add_argument("--prior-restore-best", action="store_true", default=True)
    p.add_argument("--no-prior-restore-best", action="store_false", dest="prior_restore_best")

    p.add_argument("--n-bins", type=int, default=35)
    p.add_argument("--eval-margin", type=float, default=0.30)
    p.add_argument("--score-min-bin-count", type=int, default=10)
    p.add_argument("--fd-delta", type=float, default=0.03)

    p.add_argument("--decoder-epsilon", type=float, default=0.12)
    p.add_argument("--decoder-bandwidth", type=float, default=0.10)
    p.add_argument("--decoder-epochs", type=int, default=80)
    p.add_argument("--decoder-batch-size", type=int, default=256)
    p.add_argument("--decoder-lr", type=float, default=1e-3)
    p.add_argument("--decoder-hidden-dim", type=int, default=64)
    p.add_argument("--decoder-depth", type=int, default=2)
    p.add_argument(
        "--decoder-min-class-count",
        type=int,
        default=5,
        help="Minimum balanced samples per class in train AND eval windows; limited by eval mass near theta±epsilon/2.",
    )
    p.add_argument("--decoder-train-cap", type=int, default=1200)
    p.add_argument("--decoder-eval-cap", type=int, default=1200)
    p.add_argument("--decoder-val-frac", type=float, default=0.15)
    p.add_argument(
        "--decoder-min-val-class-size",
        type=int,
        default=10,
        help="Minimum validation samples per class after holdout; lower values ease nfit>=min_class_count when ntr is modest.",
    )
    p.add_argument("--decoder-early-patience", type=int, default=100)
    p.add_argument("--decoder-early-min-delta", type=float, default=1e-4)
    p.add_argument(
        "--decoder-early-ema-alpha",
        type=float,
        default=0.2,
        help="EMA smoothing factor α in (0,1] for validation loss monitor used by decoder early stopping.",
    )
    p.add_argument("--decoder-restore-best", action="store_true", default=True)
    p.add_argument("--no-decoder-restore-best", action="store_false", dest="decoder_restore_best")
    p.add_argument(
        "--decoder-debug-bins",
        action="store_true",
        default=False,
        help="Log per-center decoder skip details (counts and reasons) to stdout.",
    )
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(DATA_DIR) / "outputs_step6_shared_dataset"),
    )
    p.add_argument(
        "--compute-h-matrix",
        action="store_true",
        default=False,
        help="Estimate the sample-wise H matrix from trained posterior/prior score models.",
    )
    p.add_argument(
        "--h-sigma-eval",
        type=float,
        default=-1.0,
        help="Sigma used for h-matrix score evaluation; <=0 means use min(eval sigma grid).",
    )
    p.add_argument(
        "--h-batch-size",
        type=int,
        default=65536,
        help="Max pair evaluations per h-matrix score forward chunk.",
    )
    p.add_argument(
        "--h-restore-original-order",
        action="store_true",
        default=False,
        help="Return H matrices in original dataset order instead of theta-sorted order.",
    )
    p.add_argument(
        "--h-save-intermediates",
        action="store_true",
        default=False,
        help="Save intermediate matrices G, C, and DeltaL to h-matrix npz output.",
    )


def parse_full_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shared-dataset score-vs-decoder comparison with analytic GT.")
    add_dataset_arguments(p)
    add_estimation_arguments(p)
    return p.parse_args(argv)


def parse_dataset_only_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate and save a shared (theta, x) train/eval dataset.")
    add_dataset_arguments(p)
    p.add_argument(
        "--output-npz",
        type=str,
        default=str(Path(DATA_DIR) / "shared_fisher_dataset.npz"),
        help="Path to write the shared dataset .npz (under DATAROOT recommended).",
    )
    return p.parse_args(argv)


def parse_estimate_only_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fisher estimation from a saved shared dataset .npz.")
    add_estimation_arguments(p)
    p.add_argument(
        "--dataset-npz",
        type=str,
        required=True,
        help="Path to a shared dataset .npz produced by fisher_make_dataset.py.",
    )
    return p.parse_args(argv)
