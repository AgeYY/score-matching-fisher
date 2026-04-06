#!/usr/bin/env python3
"""Load a saved shared dataset and run score + decoder Fisher estimation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import torch

from fisher.cli_shared_fisher import parse_estimate_only_args
from fisher.shared_dataset_io import load_shared_dataset_npz
from fisher.shared_fisher_est import (
    build_dataset_from_meta,
    merge_meta_into_args,
    run_shared_fisher_estimation,
    validate_estimation_args,
)


def main() -> None:
    est_args = parse_estimate_only_args()
    validate_estimation_args(est_args)

    bundle = load_shared_dataset_npz(est_args.dataset_npz)
    meta = bundle.meta
    full_args = merge_meta_into_args(meta, est_args)

    np.random.seed(int(meta["seed"]))
    torch.manual_seed(int(meta["seed"]))
    rng = np.random.default_rng(int(meta["seed"]))

    dataset = build_dataset_from_meta(meta)

    print(
        f"[data] loaded npz={est_args.dataset_npz} "
        f"total={bundle.theta_all.shape[0]} train={bundle.theta_train.shape[0]} eval={bundle.theta_eval.shape[0]}"
    )

    run_shared_fisher_estimation(
        full_args,
        dataset,
        theta_all=bundle.theta_all,
        x_all=bundle.x_all,
        theta_train=bundle.theta_train,
        x_train=bundle.x_train,
        theta_eval=bundle.theta_eval,
        x_eval=bundle.x_eval,
        rng=rng,
    )


if __name__ == "__main__":
    main()
