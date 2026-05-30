---
name: benchmark-1d
description: >-
  Score-matching-fisher: canonical 1D-theta H-decoding convergence workflow on PR30 linearbench and
  cosine (noise2x_alpha4x PR30 bundle; cov_theta_amp_scale 4), n-list, optional long LXFS budget,
  two-GPU split. Example method list uses bin_gaussian plus scheduled linear_x_flow *_t methods; the
  user may substitute different --methods. Use when the user says benchmark-1D, benchmark 1D, or
  1D bench on linearbench and cosine (noise2x_alpha4x) together.
---

# benchmark-1D

## Meaning

**"benchmark-1D"** names a **reproducible convergence tier** in this repo: run [`bin/study_h_decoding_convergence.py`](bin/study_h_decoding_convergence.py) on the **canonical PR-30D** **linearbench** dataset and a **cosine** PR30 dataset in the same generative family as [**`cosinebench`**](../cosinebench/SKILL.md) but with **`--cov-theta-amp-scale 4`** (directory token **`noise2x_alpha4x`**; see **Cosine side** below). Resolve **linearbench** paths via [**`linearbench`**](../linearbench/SKILL.md). The script delegates to the shared continuous twofig compute/render pipeline ([`fisher/h_decoding_convergence_twofig.py`](fisher/h_decoding_convergence_twofig.py)); default geometry is **1D $\theta$ binning** (e.g. `theta_binning_mode=theta1`, `n_bins=10`). This is the **embedded 5D-native** bench, not the native-2D `theta2_grid` skills.

### Cosine side (benchmark-1D default)

- **PR-30D NPZ:** `data/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha4x/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha4x_pr30d.npz`
- **Regeneration:** same `make_dataset.py` / `project_dataset_pr_autoencoder.py` flow as [**`cosinebench`**](../cosinebench/SKILL.md), but pass **`--cov-theta-amp-scale 4`** and matching **`noise2x_alpha4x`** output paths (keep **`--obs-noise-scale 0.5`**, **`--x-dim 5`**, **`--n-total 10000`**, family `cosine_gaussian_sqrtd_rand_tune_additive`).
- **Note:** the **`cosinebench`** skill alias still documents the older **`noise2x_alpha2x`** bundle for historical runs; **benchmark-1D** cosine examples use **alpha4x** unless the user says otherwise. For stronger theta–variance coupling experiments, regenerate with **`--cov-theta-amp-scale 8`** and a **`noise2x_alpha8x`** output tree.

## What is fixed vs what you may change

| Fixed by this skill | User may override |
|---------------------|-------------------|
| Dataset NPZ + `--dataset-family`: linearbench per **`linearbench`**; cosine PR30 under **`data/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha4x/`** (`--cov-theta-amp-scale 4`) | Any other `--dataset-npz` / family (then this skill does not apply) |
| Nested subset sweep **`--n-list 80,400,1000`** (scheduled LXF runs often pair with **`--lxfs-early-patience 1000`**) | Other `--n-list`; shrink `--n-ref` / `--n-list` if $n_{\mathrm{total}}$ is smaller |
| **`mamba run -n geo_diffusion`**, **`--device cuda`**, reporting under **`data/...`** | CPU-only runs violate **`AGENTS.md`** unless the user accepts that constraint |

**Methods are not fixed.** Pass a comma-separated **`--methods`** list (no per-method `:arch` suffixes; use global **`--flow-arch`** when needed). The historical full scheduled-LXF sweep is **`bin_gaussian`** plus the core **`_TIME_LXF_METHODS`** tokens below. For other comparisons, substitute your own **`--methods`** string. Low-rank rows use **`--lxf-low-rank-dim`** (CLI default **3**); add **`--lxf-low-rank-divergence-estimator`** / **`--lxf-hutchinson-probes`** when relevant. **`xflow_sir_*`** rows also use **`--sir-num-bins`** / **`--sir-ridge`** (see comments in [`fisher/h_decoding_twofig.py`](fisher/h_decoding_twofig.py)).

### Scheduled LXF (`*_t`, `xflow_sir_*`, …) — `lxfs_*` hyperparameters

Methods in **`_TIME_LXF_METHODS`** in [`fisher/h_decoding_convergence_methods.py`](fisher/h_decoding_convergence_methods.py) train with the **`lxfs`** CLI prefix (epochs, LR, batch size, early patience, path schedule), not **`lxf_*`**. See the `lxf_prefix = "lxfs"` branch in that module. Current set:

`linear_x_flow_t`, `linear_x_flow_scalar_t`, `linear_x_flow_diagonal_t`, `linear_x_flow_diagonal_theta_t`, `linear_x_flow_low_rank_t`, `linear_x_flow_pure_low_rank_t`, `linear_x_flow_pure_cond_low_rank_t`, `linear_x_flow_lr_t_ts`, `linear_x_flow_low_rank_randb_t`, `xflow_sir_lrank`, `xflow_sir_lrank_dia`, `xflow_sir_lrank_dia_theta`, `xflow_sir_lrank_scalar`, `xflow_sir_lrank_scalar_theta`, `xflow_sir_pure_lrank`

Example long-run defaults used in repo experiments:

- **`--lxfs-epochs 50000`** (early stopping often finishes earlier)
- **`--lxfs-early-patience 1000`**
- **`--lxfs-path-schedule cosine`**

Omitting **`--lxfs-epochs`** uses the CLI default (shorter cap). Adjust **`--lxfs-lr`**, **`--lxfs-batch-size`**, etc. if needed for memory or stability.

## Two-GPU split (optional)

When the user has two GPUs and wants both benches in parallel:

- **`CUDA_VISIBLE_DEVICES=0`** → linearbench job  
- **`CUDA_VISIBLE_DEVICES=1`** → cosine (noise2x_alpha4x) job  

Each job needs a **distinct `--output-dir`** under e.g. **`data/experiments/...`**. Use **`PYTHONUNBUFFERED=1`** and redirect to **`run.log`** (e.g. `tee`) per **`AGENTS.md`** long-run notes.

## Example commands (repo root)

**Linearbench** — illustrative **core scheduled-LXF sweep** + `bin_gaussian` (replace **`--methods`** if the user wants a different set, e.g. all **`_TIME_LXF_METHODS`** tokens or SIR variants):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 mamba run -n geo_diffusion python bin/study_h_decoding_convergence.py \
  --dataset-npz data/randamp_gaussian_sqrtd_xdim5/randamp_gaussian_sqrtd_xdim5_pr30d.npz \
  --dataset-family randamp_gaussian_sqrtd \
  --methods bin_gaussian,linear_x_flow_t,linear_x_flow_scalar_t,linear_x_flow_diagonal_t,linear_x_flow_diagonal_theta_t,linear_x_flow_low_rank_t,linear_x_flow_low_rank_randb_t \
  --n-list 80,400,1000 \
  --lxfs-path-schedule cosine \
  --lxfs-epochs 50000 \
  --lxfs-early-patience 1000 \
  --device cuda \
  --output-dir data/experiments/h_decoding_convergence_pr30d_linearbench_<TAG> \
  2>&1 | tee data/experiments/h_decoding_convergence_pr30d_linearbench_<TAG>/run.log
```

**Cosine (noise2x / alpha4x)** — same pattern; swap NPZ, family, output dir:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 mamba run -n geo_diffusion python bin/study_h_decoding_convergence.py \
  --dataset-npz data/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha4x/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha4x_pr30d.npz \
  --dataset-family cosine_gaussian_sqrtd_rand_tune_additive \
  --methods bin_gaussian,linear_x_flow_t,linear_x_flow_scalar_t,linear_x_flow_diagonal_t,linear_x_flow_diagonal_theta_t,linear_x_flow_low_rank_t,linear_x_flow_low_rank_randb_t \
  --n-list 80,400,1000 \
  --lxfs-path-schedule cosine \
  --lxfs-epochs 50000 \
  --lxfs-early-patience 1000 \
  --device cuda \
  --output-dir data/experiments/h_decoding_convergence_pr30d_cosinebench_<TAG> \
  2>&1 | tee data/experiments/h_decoding_convergence_pr30d_cosinebench_<TAG>/run.log
```

Replace `<TAG>` with a short run label (e.g. date + method slug).

## Expected artifacts

Under each **`--output-dir`** (primary names written by the convergence entrypoint):

- `h_decoding_convergence_results.npz`
- `h_decoding_convergence_summary.txt`
- `h_decoding_convergence_sweep.svg`, `h_decoding_convergence_corr_nmse.svg`, `h_decoding_convergence_training_losses_panel.svg`
- `h_decoding_convergence_all_columns.png` (when the combined PNG step succeeds)
- **`run.log`** if you tee as above

The run may also leave intermediate twofig-prefixed copies (`h_decoding_twofig_results.npz`, `h_decoding_twofig_*.svg`) in the same directory; prefer the **`h_decoding_convergence_*`** paths for reporting.

## Agent behavior

1. Resolve **linearbench** via its skill. For **cosine** in this tier, use the **`noise2x_alpha4x`** PR30 NPZ above ( **`--cov-theta-amp-scale 4`** when regenerating); do not substitute the **`cosinebench`** skill’s **alpha2x** paths unless the user explicitly asks for that older bundle. If the user asks for the legacy stronger-coupling cosine bundle, use **`noise2x_alpha8x`** (`--cov-theta-amp-scale 8`). Do not silently change observation noise, $n_{\mathrm{total}}$, or dimensions relative to this skill’s cosine recipe unless the user opts into a different bundle.
2. Use **`bin/study_h_decoding_convergence.py`** with **`--methods`**, not the removed **`bin/study_h_decoding_twofig.py`** / **`--theta-field-rows`**. When the user asks for **different methods**, keep the **dataset + n-list + (if applicable) lxfs/lxf knobs** from this skill unless they say otherwise.
3. Report paths as **`<repo-root>/data/...`** when outputs live under `DATAROOT` (see **`AGENTS.md`**).
