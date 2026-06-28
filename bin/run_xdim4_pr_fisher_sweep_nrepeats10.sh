#!/usr/bin/env bash
# Continuous PR Fisher x-dim 4 sweep: mixed-affine SKL flow_linear, 10 repeats.
# Same specs as n-repeats=1 run except --n-repeats 10 and a separate output dir.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/data/randamp_gaussian_sqrtd_xdim4_pr_fisher_sweeps_trainfrac08_lr1e-4_n1500_3500_5500_7500_9500_nrepeats10"
LOG="${OUT_DIR}/sweep_nrepeats10.log"
mkdir -p "${OUT_DIR}"
export PYTHONUNBUFFERED=1
exec mamba run -n geo_diffusion python "${REPO_ROOT}/bin/compare_continuous_pr_fisher_sweeps_parallel.py" \
  --n-list 1500,3500,5500,7500,9500 \
  --n-repeats 10 \
  --seed 7 \
  --native-x-dim 4 \
  --train-frac 0.8 \
  --lr 1e-4 \
  --output-dir "${OUT_DIR}" \
  --gpu-ids 1 \
  --jobs-per-gpu 1 \
  --device cuda:1 \
  2>&1 | tee "${LOG}"
