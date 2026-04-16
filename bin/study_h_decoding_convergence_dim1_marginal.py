#!/usr/bin/env python3
"""Run ``study_h_decoding_convergence`` on the first K coordinates, with marginal GT Hellinger.

Loads a *full-dimensional* shared dataset NPZ (e.g. cosine ``cosine_gaussian_sqrtd`` with ``x_dim=10``),
writes a sliced NPZ with ``x`` replaced by ``x[:, :K]`` and ``meta['x_dim']=K`` (default ``K=1``).
``meta['marginal_first_dim_gt_original_x_dim']`` stores the original ambient dimension; GT Hellinger MC
uses the diagonal **marginal** :math:`p(x_1,\\ldots,x_K\\mid\\theta)` implied by the full model.

Wrapper options are parsed first; all other flags are forwarded to ``study_h_decoding_convergence``.
The sliced NPZ path is injected as ``--dataset-npz`` (any user ``--dataset-npz`` is ignored).

Example (``flow_x_likelihood``; paths under repo ``data/``):

.. code-block:: bash

    mamba run -n geo_diffusion python bin/study_h_decoding_convergence_dim1_marginal.py \\
      --marginal-source-npz data/cosine_gaussian_sqrtd_xdim10_n3000_seed7/shared_fisher_dataset.npz \\
      --dataset-family cosine_gaussian_sqrtd \\
      --theta-field-method flow_x_likelihood \\
      --output-dir data/h_decoding_dim1_marginal_test \\
      --n-ref 2500 --n-list 80,200 \\
      --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_bin_dir = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_bin_dir) not in sys.path:
    sys.path.insert(0, str(_bin_dir))

import numpy as np

from fisher.marginal_first_dim_wrapper import MarginalLeadingDimsGaussianWrapper
from fisher.shared_dataset_io import load_shared_dataset_npz, save_shared_dataset_npz
from fisher.shared_fisher_est import build_dataset_from_meta as _build_dataset_from_meta_orig


def _slice_meta_for_marginal(bundle, *, original_x_dim: int, leading_dims: int) -> dict:
    meta = dict(bundle.meta)
    meta["marginal_first_dim_gt_original_x_dim"] = int(original_x_dim)
    meta["marginal_leading_dims_k"] = int(leading_dims)
    meta["x_dim"] = int(leading_dims)
    return meta


def _write_sliced_npz(dest: str, bundle, meta: dict, *, leading_dims: int) -> None:
    k = int(leading_dims)

    def _sl(x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)[:, :k]

    save_shared_dataset_npz(
        dest,
        meta=meta,
        theta_all=bundle.theta_all,
        x_all=_sl(bundle.x_all),
        train_idx=bundle.train_idx,
        eval_idx=bundle.eval_idx,
        theta_train=bundle.theta_train,
        x_train=_sl(bundle.x_train),
        theta_eval=bundle.theta_eval,
        x_eval=_sl(bundle.x_eval),
    )


def _patch_build_dataset_from_meta() -> None:
    def _wrapped(meta: dict) -> object:
        if meta.get("marginal_first_dim_gt_original_x_dim") is not None:
            d0 = int(meta["marginal_first_dim_gt_original_x_dim"])
            if d0 < 2:
                raise ValueError("marginal_first_dim_gt_original_x_dim must be >= 2.")
            k = int(meta.get("marginal_leading_dims_k", 1))
            if k < 1 or k > d0:
                raise ValueError(f"marginal_leading_dims_k={k} invalid for original_x_dim={d0}.")
            full_meta = dict(meta)
            full_meta["x_dim"] = d0
            full_meta.pop("marginal_first_dim_gt_original_x_dim", None)
            full_meta.pop("marginal_leading_dims_k", None)
            full_ds = _build_dataset_from_meta_orig(full_meta)
            return MarginalLeadingDimsGaussianWrapper(full_ds, k)
        return _build_dataset_from_meta_orig(meta)

    import fisher.shared_fisher_est as sfe

    sfe.build_dataset_from_meta = _wrapped  # noqa: PLW0603

    import study_h_decoding_convergence as sdc
    import visualize_h_matrix_binned as vhb

    sdc.build_dataset_from_meta = _wrapped
    vhb.build_dataset_from_meta = _wrapped


def _strip_dataset_npz_flag(argv: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--dataset-npz":
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def main() -> None:
    argv = sys.argv[1:]
    wp = argparse.ArgumentParser(
        description=(
            "Slice dataset to the first k coordinates and run study_h_decoding_convergence with "
            "marginal GT (full-dimensional generative model in meta)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    wp.add_argument(
        "--marginal-source-npz",
        type=str,
        required=True,
        help="Original full-x_dim shared_fisher_dataset.npz (e.g. 10D sqrtd cosine).",
    )
    wp.add_argument(
        "--leading-dims",
        type=int,
        default=1,
        metavar="K",
        help="Number of leading coordinates to keep (x[:, :K]); GT uses marginal p(x_1,...,x_K|theta).",
    )
    wp.add_argument(
        "--sliced-output-npz",
        type=str,
        default="",
        help="Where to write the sliced NPZ (default: next to source, dim{K}_marginal/shared_fisher_dataset.npz).",
    )
    wp.add_argument(
        "--original-x-dim",
        type=int,
        default=0,
        help="Override source x_dim (default: read from source meta).",
    )
    args, study_argv = wp.parse_known_args(argv)

    if not study_argv:
        print(
            "Add study_h_decoding_convergence flags after the wrapper options, e.g.\n"
            "  --marginal-source-npz data/.../shared_fisher_dataset.npz \\\n"
            "  --dataset-family cosine_gaussian_sqrtd --theta-field-method flow_x_likelihood \\\n"
            "  --output-dir data/... --n-ref 2500 --n-list 80,200 --device cuda",
            file=sys.stderr,
        )
        wp.print_help()
        raise SystemExit(2)

    src = os.path.abspath(str(args.marginal_source_npz))
    bundle = load_shared_dataset_npz(src)
    ox = int(args.original_x_dim) if int(args.original_x_dim) > 0 else int(bundle.meta["x_dim"])
    k = int(args.leading_dims)
    if ox < 2:
        raise ValueError("Source dataset must have x_dim >= 2 for marginal-from-full GT.")
    if k < 1 or k > ox:
        raise ValueError(f"--leading-dims K={k} must satisfy 1 <= K <= source x_dim={ox}.")
    if bundle.x_all.shape[1] < ox:
        raise ValueError(f"Source x_all second dim {bundle.x_all.shape[1]} < original_x_dim={ox}.")

    if str(args.sliced_output_npz).strip():
        dest = os.path.abspath(str(args.sliced_output_npz))
    else:
        dest = os.path.join(os.path.dirname(src), f"dim{k}_marginal", "shared_fisher_dataset.npz")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)

    meta = _slice_meta_for_marginal(bundle, original_x_dim=ox, leading_dims=k)
    _write_sliced_npz(dest, bundle, meta, leading_dims=k)
    print(f"[marginal-leading] wrote sliced NPZ: {dest}")
    print(f"[marginal-leading] original_x_dim={ox}  leading_dims={k}  meta x_dim={k}  marginal GT key set.")

    _patch_build_dataset_from_meta()

    import study_h_decoding_convergence as sdc

    study_argv = _strip_dataset_npz_flag(study_argv)
    study_argv = ["--dataset-npz", dest] + study_argv

    sdc.main(study_argv)


if __name__ == "__main__":
    main()
