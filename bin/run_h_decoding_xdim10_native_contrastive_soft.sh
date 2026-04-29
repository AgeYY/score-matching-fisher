#!/usr/bin/env bash
# Native 10D H-decoding convergence: cosine_gaussian_sqrtd vs randamp_gaussian_sqrtd.
#
# Uses contrastive-soft with normalized_dot (CLI default), fixed Gaussian theta kernel
# bandwidth --contrastive-soft-bandwidth 1 (CLI default), and larger dot / MLP capacity for x_dim=10.
#
# Requires NPZs under data/datasets/ (override paths below if you regenerate data).
# Usage (from repo root):
#   bash bin/run_h_decoding_xdim10_native_contrastive_soft.sh
#
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1

TS="$(date +%Y%m%d-%H%M%S)"

# Dataset archives (8000 samples, seed 7 — matches journal naming)
COS_NPZ="${COS_NPZ:-data/datasets/cosine_gaussian_sqrtd_xdim10_n8000_20260428-143830.npz}"
RA_NPZ="${RA_NPZ:-data/datasets/randamp_gaussian_sqrtd_xdim10_n8000_20260428-093928.npz}"

OUT_COS="${OUT_COS:-data/h_decoding_conv_cosine_sqrtd_xdim10_contrastive_soft_normdot_bw1_${TS}}"
OUT_RA="${OUT_RA:-data/h_decoding_conv_randamp_sqrtd_xdim10_contrastive_soft_normdot_bw1_${TS}}"
mkdir -p "$OUT_COS" "$OUT_RA"

echo "=== start cosine 10D $(date -Is) ===" | tee -a "${OUT_COS}/run.log"
mamba run -n geo_diffusion python bin/study_h_decoding_convergence.py \
  --dataset-npz "$COS_NPZ" \
  --dataset-family cosine_gaussian_sqrtd \
  --theta-field-method contrastive-soft \
  --contrastive-soft-score-arch normalized_dot \
  --contrastive-soft-bandwidth 1 \
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
  --n-list 80,200,400,600 \
  --num-theta-bins 10 \
  --keep-intermediate \
  --device cuda \
  --output-dir "$OUT_COS" \
  2>&1 | tee -a "${OUT_COS}/run.log"

echo "=== cosine done $(date -Is) ===" | tee -a "${OUT_COS}/run.log"

echo "=== start randamp_gaussian_sqrtd 10D $(date -Is) ===" | tee -a "${OUT_RA}/run.log"
mamba run -n geo_diffusion python bin/study_h_decoding_convergence.py \
  --dataset-npz "$RA_NPZ" \
  --dataset-family randamp_gaussian_sqrtd \
  --theta-field-method contrastive-soft \
  --contrastive-soft-score-arch normalized_dot \
  --contrastive-soft-bandwidth 1 \
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
  --n-list 80,200,400,600 \
  --num-theta-bins 10 \
  --keep-intermediate \
  --device cuda \
  --output-dir "$OUT_RA" \
  2>&1 | tee -a "${OUT_RA}/run.log"

echo "=== randamp done $(date -Is) ===" | tee -a "${OUT_RA}/run.log"
echo "=== pipeline done $(date -Is) ==="
