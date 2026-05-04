---
name: xflow-benchmark-1d
description: >-
  Score-matching-fisher: benchmark-1D twofig on PR30 linearbench and cosinebench with a fixed
  four-row panel — x_flow, linear_x_flow_t, linear_x_flow_low_rank_t, bin_gaussian — same n-list
  and lxfs budget as benchmark-1D; pass --lxf-low-rank-dim for the low-rank row. Use when the user
  says xflow-benchmark-1d, xflow bench 1D, or wants this x_flow + scheduled LXF + bin panel on both
  1D benches.
---

# xflow-benchmark-1D

## Meaning

Same **reproducible tier** as [**benchmark-1D**](../benchmark-1d/SKILL.md) (PR-30D **linearbench** / **cosinebench**, `theta1` binning, `--n-ref 5000`, `--n-list 80,400,1000`, `mamba run -n geo_diffusion`, `--device cuda`, outputs under **`data/experiments/...`**), but the **theta-field row list is fixed** to:

`x_flow,linear_x_flow_t,linear_x_flow_low_rank_t,bin_gaussian`

| Item | Value |
|------|--------|
| **linearbench** NPZ | `data/randamp_gaussian_sqrtd_xdim5/randamp_gaussian_sqrtd_xdim5_pr30d.npz` |
| **linearbench** `--dataset-family` | `randamp_gaussian_sqrtd` |
| **cosinebench** NPZ | `data/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x_pr30d.npz` |
| **cosinebench** `--dataset-family` | `cosine_gaussian_sqrtd_rand_tune_additive` |
| **Rows** | `x_flow,linear_x_flow_t,linear_x_flow_low_rank_t,bin_gaussian` |
| **Low-rank** | `--lxf-low-rank-dim 4` (required for `linear_x_flow_low_rank_t`) |
| **Scheduled LXF** | `--lxfs-path-schedule cosine`, `--lxfs-epochs 50000`, `--lxfs-early-patience 1000` |

`x_flow` uses the **conditional x-space flow** path (not `lxfs_*`); `linear_x_flow_t` and `linear_x_flow_low_rank_t` use **`lxfs_*`** training inside the convergence pipeline. Defaults for `x_flow` / `lxf_*` come from [`bin/study_h_decoding_convergence.py`](bin/study_h_decoding_convergence.py) unless you override them.

## Example commands (repo root)

Replace `<TAG>` (e.g. `20260504_xflow_bench1d`). Use **`PYTHONUNBUFFERED=1`** and **`tee`** per **`AGENTS.md`**.

**Linearbench** (`CUDA_VISIBLE_DEVICES=0` if splitting GPUs):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 mamba run -n geo_diffusion python bin/study_h_decoding_twofig.py \
  --dataset-npz data/randamp_gaussian_sqrtd_xdim5/randamp_gaussian_sqrtd_xdim5_pr30d.npz \
  --dataset-family randamp_gaussian_sqrtd \
  --theta-field-rows x_flow,linear_x_flow_t,linear_x_flow_low_rank_t,bin_gaussian \
  --lxf-low-rank-dim 4 \
  --n-ref 5000 --n-list 80,400,1000 \
  --lxfs-path-schedule cosine \
  --lxfs-epochs 50000 \
  --lxfs-early-patience 1000 \
  --device cuda \
  --output-dir data/experiments/h_decoding_twofig_pr30d_linearbench_<TAG> \
  2>&1 | tee data/experiments/h_decoding_twofig_pr30d_linearbench_<TAG>/run.log
```

**Cosinebench**:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 mamba run -n geo_diffusion python bin/study_h_decoding_twofig.py \
  --dataset-npz data/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x_pr30d.npz \
  --dataset-family cosine_gaussian_sqrtd_rand_tune_additive \
  --theta-field-rows x_flow,linear_x_flow_t,linear_x_flow_low_rank_t,bin_gaussian \
  --lxf-low-rank-dim 4 \
  --n-ref 5000 --n-list 80,400,1000 \
  --lxfs-path-schedule cosine \
  --lxfs-epochs 50000 \
  --lxfs-early-patience 1000 \
  --device cuda \
  --output-dir data/experiments/h_decoding_twofig_pr30d_cosinebench_<TAG> \
  2>&1 | tee data/experiments/h_decoding_twofig_pr30d_cosinebench_<TAG>/run.log
```

## Expected artifacts

Same as benchmark-1D: `h_decoding_twofig_results.npz`, `h_decoding_twofig_summary.txt`, sweep/GT/corr/NMSE/training-loss SVGs, `training_losses/`, optional `run.log`.

## Agent behavior

1. Resolve dataset paths via [**linearbench**](../linearbench/SKILL.md) and [**cosinebench**](../cosinebench/SKILL.md); do not change recipe hyperparameters unless the user asks.
2. Report full paths as **`<repo-root>/data/...`** for outputs under `DATAROOT` (**`AGENTS.md`**).
