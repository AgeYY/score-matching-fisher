#!/usr/bin/env python3
"""SIR-first two-figure H/decoding convergence study.

This entrypoint mirrors :mod:`fisher.h_decoding_twofig`, but for each nested
subset size ``n`` it fits Sliced Inverse Regression only on that subset's
training split and projects train/validation/all observations before estimator
fitting or sweep decoding.

Example benchmark-1D PR-30D runs:

.. code-block:: bash

   mamba run -n geo_diffusion python bin/study_h_decoding_twofig_sir.py \
     --dataset-npz data/randamp_gaussian_sqrtd_xdim5/randamp_gaussian_sqrtd_xdim5_pr30d.npz \
     --dataset-family randamp_gaussian_sqrtd \
     --theta-field-rows theta_path_integral,theta_flow,x_flow,linear_x_flow_t,contrastive,bin_gaussian \
     --n-list 80,200,400,600 \
     --device cuda:1 \
     --output-dir data/experiments/h_decoding_twofig_sir_pr30d_linearbench_<TAG>

   mamba run -n geo_diffusion python bin/study_h_decoding_twofig_sir.py \
     --dataset-npz data/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha4x/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha4x_pr30d.npz \
     --dataset-family cosine_gaussian_sqrtd_rand_tune_additive \
     --theta-field-rows theta_path_integral,theta_flow,x_flow,linear_x_flow_t,contrastive,bin_gaussian \
     --n-list 80,200,400,600 \
     --device cuda:1 \
     --output-dir data/experiments/h_decoding_twofig_sir_pr30d_cosinebench_noise2x_alpha4x_<TAG>
"""

from __future__ import annotations

from fisher.h_decoding_twofig import *  # noqa: F401,F403
from fisher.h_decoding_twofig import main as _main


def main(argv: list[str] | None = None) -> None:
    _main(argv, sir_first_default=True)


if __name__ == "__main__":
    main()
