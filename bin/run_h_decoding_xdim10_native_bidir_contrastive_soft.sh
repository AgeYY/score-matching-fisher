#!/usr/bin/env bash
# Native 10D H-decoding convergence: cosine_gaussian_sqrtd vs randamp_gaussian_sqrtd.
#
# Uses bidir-contrastive-soft with normalized-dot+bias scorer (--contrastive-soft-score-arch normalized_dot).
# Gaussian theta-kernel width: --contrastive-soft-bandwidth 0.5 (raw theta units).
#
# Requires NPZs under data/datasets/ (override paths below if you regenerate data).
# Nested training sizes: --n-list 2000 (max n must be <= --n-ref).
# Usage (from repo root):
#   bash bin/run_h_decoding_xdim10_native_bidir_contrastive_soft.sh
#
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1

TS="$(date +%Y%m%d-%H%M%S)"

COS_NPZ="${COS_NPZ:-data/datasets/cosine_gaussian_sqrtd_xdim10_n8000_20260428-143830.npz}"
RA_NPZ="${RA_NPZ:-data/datasets/randamp_gaussian_sqrtd_xdim10_n8000_20260428-093928.npz}"

OUT_COS="${OUT_COS:-data/h_decoding_conv_cosine_sqrtd_xdim10_bidir_contrastive_soft_bw0p5_normdot_n2000_${TS}}"
OUT_RA="${OUT_RA:-data/h_decoding_conv_randamp_sqrtd_xdim10_bidir_contrastive_soft_bw0p5_normdot_n2000_${TS}}"
mkdir -p "$OUT_COS" "$OUT_RA"

echo "=== start cosine 10D bidir-contrastive-soft (normalized_dot, h=0.5, n_list=2000) $(date -Is) ===" | tee -a "${OUT_COS}/run.log"
mamba run -n geo_diffusion python bin/study_h_decoding_convergence.py \
  --dataset-npz "$COS_NPZ" \
  --dataset-family cosine_gaussian_sqrtd \
  --theta-field-method bidir-contrastive-soft \
  --contrastive-soft-score-arch normalized_dot \
  --contrastive-soft-bandwidth 0.5 \
  --contrastive-soft-bandwidth-k 5 \
  --contrastive-soft-dot-dim 128 \
  --contrastive-hidden-dim 256 \
  --contrastive-depth 4 \
  --contrastive-lr 5e-4 \
  --contrastive-batch-size 256 \
  --contrastive-epochs 4000 \
  --contrastive-early-patience 500 \
  --contrastive-weight-decay 1e-4 \
  --n-ref 5000 \
  --n-list 2000 \
  --num-theta-bins 10 \
  --keep-intermediate \
  --device cuda \
  --output-dir "$OUT_COS" \
  2>&1 | tee -a "${OUT_COS}/run.log"

echo "=== cosine done $(date -Is) ===" | tee -a "${OUT_COS}/run.log"

echo "=== start randamp_gaussian_sqrtd 10D bidir-contrastive-soft (normalized_dot, h=0.5, n_list=2000) $(date -Is) ===" | tee -a "${OUT_RA}/run.log"
mamba run -n geo_diffusion python bin/study_h_decoding_convergence.py \
  --dataset-npz "$RA_NPZ" \
  --dataset-family randamp_gaussian_sqrtd \
  --theta-field-method bidir-contrastive-soft \
  --contrastive-soft-score-arch normalized_dot \
  --contrastive-soft-bandwidth 0.5 \
  --contrastive-soft-bandwidth-k 5 \
  --contrastive-soft-dot-dim 128 \
  --contrastive-hidden-dim 256 \
  --contrastive-depth 4 \
  --contrastive-lr 5e-4 \
  --contrastive-batch-size 256 \
  --contrastive-epochs 4000 \
  --contrastive-early-patience 500 \
  --contrastive-weight-decay 1e-4 \
  --n-ref 5000 \
  --n-list 2000 \
  --num-theta-bins 10 \
  --keep-intermediate \
  --device cuda \
  --output-dir "$OUT_RA" \
  2>&1 | tee -a "${OUT_RA}/run.log"

echo "=== randamp done $(date -Is) ===" | tee -a "${OUT_RA}/run.log"
echo "=== pipeline done $(date -Is) ==="
