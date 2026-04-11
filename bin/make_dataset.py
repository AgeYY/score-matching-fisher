#!/usr/bin/env python3
"""Generate a shared (theta, x) dataset, save .npz, and write diagnostic figures (same dataset object).

Dataset CLI flags (including sample count --n-total / --num-samples) are defined in
fisher.cli_shared_fisher.add_dataset_arguments; run this script with --help for the full list.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np

from global_setting import DATA_DIR
from fisher.cli_shared_fisher import add_dataset_arguments
from fisher.dataset_visualization import plot_joint_and_tuning, summarize_dataset
from fisher.shared_dataset_io import meta_dict_from_args, save_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_args, validate_dataset_sample_args

_MAKE_DATASET_PARAMETER_REFERENCE = """
Parameter reference (all available flags in this script):

  Randomness / size / split:
    --seed
      RNG seed used for joint sampling and train/eval permutation. Default: 7.
    --n-total, --num-samples
      Number of joint (theta, x) samples before split. Default: 3000.
    --train-frac
      Fraction assigned to training; if < 1, a held-out eval split is created. Default: 1.0.

  Core domain / shape:
    --theta-low, --theta-high
      Uniform theta sampling range [theta-low, theta-high]. Defaults: -6.0, 6.0.
    --x-dim
      Observation dimension. Default: 2.

  Family selection:
    --dataset-family
      One of:
      gaussian
        Conditional Gaussian x|theta with theta-modulated covariance.
      gaussian_sqrtd
        Same generative structure as gaussian, but observation noise std scales by sqrt(x_dim)
        (variance scaled by x_dim) to avoid extreme SNR when dimension is large.
      gmm_non_gauss
        Theta-dependent two-component GMM (non-Gaussian conditional).
      cos_sin_piecewise_noise
        Cosine or von-Mises tuning means + piecewise noise std by theta sign.
      linear_piecewise_noise
        Linear tuning means + observation std vs theta (default: linear from low to high across
        [theta-low, theta-high]; optional sigmoid schedule).
      Default: gaussian.

  Tuning-curve mean (used where applicable):
    --tuning-curve-family
      cosine, von_mises_raw, or gaussian_raw. Default: cosine.
    --vm-mu-amp, --vm-kappa, --vm-omega
      von_mises_raw only: amplitude A, concentration kappa, and omega in
      A*exp(kappa*cos(omega*(theta-theta_j))). Centers theta_j are uniform on [theta-low, theta-high].
      Ignored for cosine and gaussian_raw. Defaults: 1, 1, 1.
    --gauss-mu-amp, --gauss-kappa, --gauss-omega
      gaussian_raw only: amplitude A, precision kappa, and omega in
      A*exp(-kappa*(omega*(theta-theta_j))^2). Centers theta_j uniform on [theta-low, theta-high].
      Ignored for cosine and von_mises_raw.
      Defaults: 1, 0.2, 1.
      Not used for --dataset-family linear_piecewise_noise (fixed linear tuning) or
      cos_sin_piecewise_noise.

  Baseline covariance/noise (Gaussian and GMM families):
    --sigma-x1, --sigma-x2
      Baseline per-axis observation std. Defaults: 0.30, 0.30.
    --rho
      Baseline correlation before theta modulation. Default: 0.15.
    --rho-clip
      Clamp on |rho(theta)| after modulation for numerical stability. Default: 0.85.

  Theta-modulated covariance (gaussian / gaussian_sqrtd families):
    --cov-theta-amp1, --cov-theta-amp2, --cov-theta-amp-rho
      Modulation amplitudes for variance1, variance2, and correlation.
      Defaults: 0.35, 0.30, 0.30.
    --cov-theta-freq1, --cov-theta-freq2, --cov-theta-freq-rho
      Modulation angular frequencies. Defaults: 0.90, 0.75, 1.10.
    --cov-theta-phase1, --cov-theta-phase2, --cov-theta-phase-rho
      Modulation phase offsets. Defaults: 0.20, -0.35, 0.40.

  Mixture controls (gmm_non_gauss family):
    --gmm-sep-scale, --gmm-sep-freq, --gmm-sep-phase
      Theta-dependent component-mean separation controls.
      Defaults: 1.10, 0.85, 0.35.
    --gmm-mix-logit-scale, --gmm-mix-bias, --gmm-mix-freq, --gmm-mix-phase
      Theta-dependent mixture weight/logit controls.
      Defaults: 1.40, 0.00, 0.95, -0.20.

  Piecewise noise controls (cos_sin_piecewise_noise / linear_piecewise_noise):
    --sigma-piecewise-low, --sigma-piecewise-high
      Endpoints for observation std (cos_sin: low/high sides of theta=0; linear: std at theta-low
      and theta-high when using --linear-sigma-schedule linear, or sigmoid endpoints when sigmoid).
      Defaults: 0.1, 0.1.
    --theta-zero-to-low / --no-theta-zero-to-low
      cos_sin: whether theta=0 is on the low-noise side. linear: if False, flip which end of the
      theta range gets low vs high noise. Default: --theta-zero-to-low (True).
    --linear-k
      Linear slope parameter for linear_piecewise_noise mean construction. Default: 1.0.
    --linear-sigma-schedule
      linear_piecewise_noise: linear or sigmoid std vs theta. Default: linear.
    --linear-sigma-sigmoid-center, --linear-sigma-sigmoid-steepness
      linear_piecewise_noise with --linear-sigma-schedule sigmoid: center and steepness.
      Defaults: 0.0, 2.0.

  Output:
    --output-npz
      Output archive path containing all/eval/train arrays and meta.
      Default: DATA_DIR/shared_fisher_dataset.npz (resolved from global_setting.DATA_DIR).

Notes:
  - Flags that are not relevant to the selected --dataset-family are accepted but ignored.
  - Saved meta stores the effective flag values for reproducibility.
"""

def parse_make_dataset_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sample a synthetic conditional dataset (theta, x), save a shared .npz for Fisher/score "
            "pipelines, and emit joint scatter + tuning-curve figures. "
            "Set the number of samples with --n-total / --num-samples. "
            "All sampling hyperparameters are recorded in the NPZ meta. "
            "Run with --help to view per-argument defaults."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_dataset_arguments(p)
    p.add_argument(
        "--output-npz",
        type=str,
        default=str(Path(DATA_DIR) / "shared_fisher_dataset.npz"),
        help=(
            "Path for the shared dataset archive (theta_all, x_all, train/eval indices, meta). "
            "Prefer a path under your DATAROOT data directory."
        ),
    )
    return p.parse_args(argv)


def main() -> None:
    args = parse_make_dataset_args()
    validate_dataset_sample_args(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)) or ".", exist_ok=True)

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dataset = build_dataset_from_args(args)
    n_total = int(args.n_total)
    theta_all, x_all = dataset.sample_joint(n_total)
    perm = rng.permutation(n_total)
    tf = float(args.train_frac)
    if tf >= 1.0:
        n_train = n_total
    else:
        n_train = int(tf * n_total)
        n_train = min(max(n_train, 1), n_total - 1)

    tr_idx = perm[:n_train]
    ev_idx = perm[n_train:]
    theta_train, x_train = theta_all[tr_idx], x_all[tr_idx]
    theta_eval, x_eval = theta_all[ev_idx], x_all[ev_idx]

    meta = meta_dict_from_args(args)
    save_shared_dataset_npz(
        args.output_npz,
        meta=meta,
        theta_all=theta_all,
        x_all=x_all,
        train_idx=tr_idx.astype(np.int64),
        eval_idx=ev_idx.astype(np.int64),
        theta_train=theta_train,
        x_train=x_train,
        theta_eval=theta_eval,
        x_eval=x_eval,
    )

    print(f"[data] total={n_total} train={theta_train.shape[0]} eval={theta_eval.shape[0]}")
    print(f"Saved shared dataset: {args.output_npz}")

    out_dir = Path(args.output_npz).resolve().parent
    joint_tuning_path = out_dir / "joint_scatter_and_tuning_curve.png"
    summarize_dataset(theta_all, x_all, dataset)
    plot_joint_and_tuning(theta_all, x_all, dataset, str(joint_tuning_path))
    print(f"Saved visualization: {joint_tuning_path}")
    print(f"Saved visualization: {joint_tuning_path.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
