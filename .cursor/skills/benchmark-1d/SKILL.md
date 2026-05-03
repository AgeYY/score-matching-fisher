---
name: benchmark-1d
description: >-
  Score-matching-fisher: canonical 1D-theta H-decoding twofig workflow on PR30 linearbench and
  cosinebench (datasets, n-list, optional long LXFS budget, two-GPU split). Example row list uses
  bin_gaussian plus scheduled linear_x_flow *_t methods; the user may substitute different
  --theta-field-rows or --theta-field-methods. Use when the user says benchmark-1D, benchmark 1D,
  or 1D bench twofig on linearbench/cosinebench together.
---

# benchmark-1D

## Meaning

**"benchmark-1D"** names a **reproducible twofig tier** in this repo: run [`bin/study_h_decoding_twofig.py`](bin/study_h_decoding_twofig.py) on the **canonical PR-30D** **linearbench** and **cosinebench** datasets (resolve paths with skills [**`linearbench`**](../linearbench/SKILL.md) and [**`cosinebench`**](../cosinebench/SKILL.md)). The default geometry is **1D $\theta$ binning** in twofig (e.g. `theta_binning_mode=theta1`, `n_bins=10`); this is the **embedded 5D-native** bench, not the native-2D `theta2_grid` skills.

## What is fixed vs what you may change

| Fixed by this skill | User may override |
|---------------------|-------------------|
| Dataset NPZ + `--dataset-family` for linearbench / cosinebench | Any other `--dataset-npz` / family (then this skill does not apply) |
| Nested subset sweep **`--n-list 80,400,1000`** (aligns with [**`lxf-bench-h-decoding-twofig`**](../lxf-bench-h-decoding-twofig/SKILL.md)) | Other `--n-list`; shrink `--n-ref` / `--n-list` if $n_{\mathrm{total}}$ is smaller |
| **`mamba run -n geo_diffusion`**, **`--device cuda`**, reporting under **`data/...`** | CPU-only runs violate **`AGENTS.md`** unless the user accepts that constraint |

**Theta-field methods are not fixed.** The examples below use **`bin_gaussian`** plus every token in **`_TIME_LXF_METHODS`** in [`bin/study_h_decoding_convergence.py`](bin/study_h_decoding_convergence.py) (`linear_x_flow_t`, `linear_x_flow_scalar_t`, `linear_x_flow_diagonal_t`, `linear_x_flow_diagonal_theta_t`, `linear_x_flow_low_rank_t`, `linear_x_flow_low_rank_randb_t`). For other comparisons, pass your own comma-separated **`--theta-field-rows`** (highest precedence) or **`--theta-field-methods`** / **`--theta-field-method`**. If you include **low-rank** rows, pass **`--lxf-low-rank-dim`** as required by the script.

## Scheduled time-dependent LXF (`*_t`) — use `lxfs_*` hyperparameters

For methods in **`_TIME_LXF_METHODS`**, convergence training uses the **`lxfs`** prefix (epochs, LR, batch size, early patience, path schedule), not **`lxf_epochs`**. See the `lxf_prefix = "lxfs"` branch in [`bin/study_h_decoding_convergence.py`](bin/study_h_decoding_convergence.py). Example long-run defaults used in repo experiments:

- **`--lxfs-epochs 50000`** (early stopping often finishes earlier)
- **`--lxfs-early-patience 1000`**
- **`--lxfs-path-schedule cosine`**

Omitting **`--lxfs-epochs`** uses the CLI default (shorter cap). Adjust **`--lxfs-lr`**, **`--lxfs-batch-size`**, etc. if needed for memory or stability.

## Two-GPU split (optional)

When the user has two GPUs and wants both benches in parallel:

- **`CUDA_VISIBLE_DEVICES=0`** → linearbench job  
- **`CUDA_VISIBLE_DEVICES=1`** → cosinebench job  

Each job needs a **distinct `--output-dir`** under e.g. **`data/experiments/...`**. Use **`PYTHONUNBUFFERED=1`** and redirect to **`run.log`** (e.g. `tee`) per **`AGENTS.md`** long-run notes.

## Example commands (repo root)

**Linearbench** — illustrative **full `_t` sweep** + `bin_gaussian` (replace `--theta-field-rows` if the user wants a different method set):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 mamba run -n geo_diffusion python bin/study_h_decoding_twofig.py \
  --dataset-npz data/randamp_gaussian_sqrtd_xdim5/randamp_gaussian_sqrtd_xdim5_pr30d.npz \
  --dataset-family randamp_gaussian_sqrtd \
  --theta-field-rows bin_gaussian,linear_x_flow_t,linear_x_flow_scalar_t,linear_x_flow_diagonal_t,linear_x_flow_diagonal_theta_t,linear_x_flow_low_rank_t,linear_x_flow_low_rank_randb_t \
  --lxf-low-rank-dim 4 \
  --n-list 80,400,1000 \
  --lxfs-path-schedule cosine \
  --lxfs-epochs 50000 \
  --lxfs-early-patience 1000 \
  --device cuda \
  --output-dir data/experiments/h_decoding_twofig_pr30d_linearbench_<TAG> \
  2>&1 | tee data/experiments/h_decoding_twofig_pr30d_linearbench_<TAG>/run.log
```

**Cosinebench** — same pattern; swap NPZ, family, output dir:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 mamba run -n geo_diffusion python bin/study_h_decoding_twofig.py \
  --dataset-npz data/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x_pr30d.npz \
  --dataset-family cosine_gaussian_sqrtd_rand_tune_additive \
  --theta-field-rows bin_gaussian,linear_x_flow_t,linear_x_flow_scalar_t,linear_x_flow_diagonal_t,linear_x_flow_diagonal_theta_t,linear_x_flow_low_rank_t,linear_x_flow_low_rank_randb_t \
  --lxf-low-rank-dim 4 \
  --n-list 80,400,1000 \
  --lxfs-path-schedule cosine \
  --lxfs-epochs 50000 \
  --lxfs-early-patience 1000 \
  --device cuda \
  --output-dir data/experiments/h_decoding_twofig_pr30d_cosinebench_<TAG> \
  2>&1 | tee data/experiments/h_decoding_twofig_pr30d_cosinebench_<TAG>/run.log
```

Replace `<TAG>` with a short run label (e.g. date + method slug).

## Expected artifacts

Under each **`--output-dir`**: `h_decoding_twofig_results.npz`, `h_decoding_twofig_summary.txt`, `h_decoding_twofig_{sweep,gt,corr_vs_n,nmse_vs_n,training_losses_panel}.svg`, plus **`run.log`** if you tee as above.

## Agent behavior

1. Resolve **linearbench** / **cosinebench** only via their skills; do not silently change noise scales, $n_{\mathrm{total}}$, or dimensions.
2. When the user asks for **different methods**, keep the **dataset + n-list + (if applicable) lxfs/lxf knobs** from this skill unless they say otherwise.
3. Report paths as **`<repo-root>/data/...`** when outputs live under `DATAROOT` (see **`AGENTS.md`**).
