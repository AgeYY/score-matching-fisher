#!/usr/bin/env python3
"""Generate a shared (theta, x) dataset, save .npz, and write diagnostic figures (same dataset object).

CLI reference (this script only)
==================================
Arguments come from ``fisher.cli_shared_fisher.add_dataset_arguments`` plus ``--output-npz`` below.
Run ``python bin/make_dataset.py --help`` for argparse defaults and exact wording.

**What you can set (user-facing)**

- ``--dataset-family`` (required behavior choice; default ``cosine_gaussian``)
    Selects a *fixed* generative recipe. Names follow ``{tuning_curve}_{noise_model}``. Full numerics
    live in ``fisher.dataset_family_recipes.family_recipe_dict`` / ``apply_family_recipe_to_namespace``.

    Choices (``--dataset-family`` token — summary):

    - ``cosine_gaussian`` — Cosine means; theta-modulated diagonal Gaussian noise (baseline sigmas 0.50).
    - ``cosine_gaussian_const_noise`` — Cosine means; constant diagonal Gaussian noise (no
      activity-coupled variance modulation; baseline sigmas 0.50).
    - ``cosine_gaussian_sqrtd`` — Same cosine means; noise variance scaled by ``x_dim`` (baseline
      sigmas 0.50) so std ~ sqrt(d).
    - ``cosine_gaussian_sqrtd_rand_tune`` — Like ``cosine_gaussian_sqrtd``, but each coordinate's
      cosine mean amplitude is multiplied by an independent factor drawn once from ``Uniform(0.5, 1.5)``
      (stored in NPZ meta as ``cosine_tune_amp_per_dim``).
    - ``randamp_gaussian`` — Random-amplitude Gaussian bump means (per-dim amplitudes drawn once);
      Gaussian observation noise (baseline 0.30). Realized amplitudes in NPZ meta as
      ``randamp_mu_amp_per_dim``.
    - ``randamp_gaussian_sqrtd`` — Same bump tuning as ``randamp_gaussian`` with sqrt-d noise scaling
      (baseline 0.20). For high-dimensional observation space via a PR-autoencoder, generate this
      family at low ``--x-dim``, then run ``bin/project_dataset_pr_autoencoder.py``.
    - ``cosine_gmm`` — Cosine-like mean branch inside a theta-dependent two-component mixture (see
      ``ToyConditionalGMMNonGaussianDataset``).
    - ``cos_sin_piecewise`` — Means ``(cos θ, sin θ)``; scalar observation std piecewise in ``θ``
      (sign-based). **Requires** ``--x-dim 2``.
    - ``linear_piecewise`` — Linear mean construction in 2D; observation std vs ``θ`` (piecewise /
      scheduled). **Requires** ``--x-dim 2``.

- ``--seed`` (default 7) — NumPy RNG seed for joint sampling and for the train/validation permutation.

- ``--theta-low``, ``--theta-high`` (defaults -6.0, 6.0) — θ is uniform on ``[theta-low, theta-high]``.

- ``--x-dim`` (default 2) — Observation dimension (length of ``x``). Must be ≥ 2; piecewise
  families above must use ``x_dim == 2``.

- ``--n-total`` / ``--num-samples`` (default 3000) — Number of joint ``(θ, x)`` draws before splitting.

- ``--train-frac`` (default 0.7) — Fraction of indices assigned to ``train_idx``. The remainder is
  ``validation_idx`` (held-out). Must be in ``(0, 1]``. Values ``< 1`` are required for shared Fisher /
  H-matrix / pairwise-CLF pipelines that need a non-empty validation slice.

- ``--obs-noise-scale`` (default 1.0) — Multiplies the family-fixed baseline ``sigma_x1`` / ``sigma_x2``
  (e.g. ``0.5`` halves observation noise relative to the default recipe).

- ``--output-npz`` — Path to the written archive (default under ``global_setting.DATA_DIR`` /
  ``SCORE_MATCHING_FISHER_DATAROOT``). Contains ``theta_all``, ``x_all``, indices, splits, and
  ``meta_json_utf8`` with the *resolved* recipe fields for reproducibility.

**What you cannot set here**

Older composition flags (e.g. ``--tuning-curve-family``, ``--sigma-x1``, ``--randamp-*``,
``--cov-theta-*``, piecewise/GMM knobs) were removed from the public CLI. Passing them raises
``ValueError`` from ``assert_no_legacy_dataset_cli_flags`` with remediation text.

**Outputs**

- NPZ at ``--output-npz``.
- ``joint_scatter_and_tuning_curve.png`` and ``.svg`` next to that NPZ (same directory).

**Implementation notes**

- Before sampling, this script prints ``format_resolved_family_summary`` so the effective internal
  recipe is visible in the log.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Matplotlib rcParams (tick sizes, spines) apply when ``global_setting`` is imported — before pyplot.
import global_setting  # noqa: F401

import numpy as np

from global_setting import DATA_DIR
from fisher.cli_shared_fisher import add_dataset_arguments
from fisher.dataset_family_recipes import (
    assert_no_legacy_dataset_cli_flags,
    format_resolved_family_summary,
)
from fisher.dataset_visualization import plot_joint_and_tuning, summarize_dataset
from fisher.shared_dataset_io import meta_dict_from_args, save_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_args, validate_dataset_sample_args


def parse_make_dataset_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sample a synthetic conditional dataset (theta, x), save a shared .npz for Fisher/score "
            "pipelines, and emit joint scatter + tuning-curve figures. "
            "Set the number of samples with --n-total / --num-samples. "
            "Choose the generative model with --dataset-family only; tuning and noise are fixed internally. "
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
    argv = sys.argv[1:]
    assert_no_legacy_dataset_cli_flags(argv)
    args = parse_make_dataset_args(argv)
    validate_dataset_sample_args(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)) or ".", exist_ok=True)

    print("[data] Resolved family configuration:")
    print(format_resolved_family_summary(args))

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    n_total = int(args.n_total)
    dataset = build_dataset_from_args(args)
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
    theta_validation, x_validation = theta_all[ev_idx], x_all[ev_idx]

    meta = meta_dict_from_args(args)
    if str(args.dataset_family) in (
        "randamp_gaussian",
        "randamp_gaussian_sqrtd",
    ):
        meta["randamp_mu_amp_per_dim"] = dataset._randamp_amp.tolist()
    if str(args.dataset_family) == "cosine_gaussian_sqrtd_rand_tune":
        meta["cosine_tune_amp_per_dim"] = dataset._cosine_tune_amp.tolist()
    save_shared_dataset_npz(
        args.output_npz,
        meta=meta,
        theta_all=theta_all,
        x_all=x_all,
        train_idx=tr_idx.astype(np.int64),
        validation_idx=ev_idx.astype(np.int64),
        theta_train=theta_train,
        x_train=x_train,
        theta_validation=theta_validation,
        x_validation=x_validation,
    )

    print(
        f"[data] total={n_total} train={theta_train.shape[0]} "
        f"validation={theta_validation.shape[0]}"
    )
    print(f"Saved shared dataset: {args.output_npz}")

    out_dir = Path(args.output_npz).resolve().parent
    joint_tuning_path = out_dir / "joint_scatter_and_tuning_curve.png"
    summarize_dataset(theta_all, x_all, dataset)
    plot_joint_and_tuning(theta_all, x_all, dataset, str(joint_tuning_path))
    print(f"Saved visualization: {joint_tuning_path}")
    print(f"Saved visualization: {joint_tuning_path.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
