#!/usr/bin/env python3
"""Generate a shared (theta, x) dataset, save .npz, and write diagnostic figures (same dataset object)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np

from fisher.cli_shared_fisher import parse_dataset_only_args
from fisher.dataset_visualization import plot_joint_and_tuning, summarize_dataset
from fisher.shared_dataset_io import meta_dict_from_args, save_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_args, validate_dataset_sample_args


def main() -> None:
    args = parse_dataset_only_args()
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


if __name__ == "__main__":
    main()
