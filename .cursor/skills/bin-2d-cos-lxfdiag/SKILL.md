---
name: bin-2d-cos-lxfdiag
description: >-
  Score-matching-fisher minimal H-decoding twofig on native gridcos 2D PR30D: bin_gaussian +
  linear_x_flow_diagonal_t, --theta-binning-mode theta2_grid (10×10 grid), --n-ref 10000,
  --lxfs-epochs 50000, --lxfs-early-patience 1000, --n-list 80,400,1000, --device cuda,
  dataset gridcos_gaussian2d_sqrtd_rand_tune_additive (noise2x alpha2x PR30D bundle).
  Use when the user says bin-2d-cos-lxfdiag, gridcos bin vs lxfdiag minimal, or wants the same
  twofig baseline as bin-2d-lin-lxfdiag but on the gridcos 2D native benchmark.
---

# Minimal twofig: `bin_gaussian` + `linear_x_flow_diagonal_t` (native gridcos 2D PR30D)

Same **minimal twofig** contract as **`bin-2d-lin-lxfdiag`**, but on the **grid-cosine** native 2D-$\theta$ benchmark (`gridcos_gaussian2d_sqrtd_rand_tune_additive`) embedded to **30D**. Hellinger / training binning uses **`theta2_grid`** (flattened 2D cells), not a $\theta_1$-only slice.

Background: **`journal/notes/2026-05-02-native-2d-theta-benchmark-datasets.md`** (gridcos recipe) and **`journal/notes/2026-05-02-native-2d-theta-twofig-h-decoding-pipeline.md`** (twofig commands for gridcos PR30D).

## Locked choices

| Setting | Value |
|--------|--------|
| Dataset NPZ | **`data/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x_pr30d.npz`** |
| `--dataset-family` | **`gridcos_gaussian2d_sqrtd_rand_tune_additive`** |
| Methods | **`bin_gaussian,linear_x_flow_diagonal_t`** (order preserved for figure rows) |
| $\theta$ binning | **`--theta-binning-mode theta2_grid`** · **`--num-theta-bins 10`** · **`--num-theta-bins-y 10`** → **100** flattened grid cells |
| LXF training cap | **`--lxfs-epochs 50000`** (max epochs per scheduled LXF fit) |
| LXF early stopping | **`--lxfs-early-patience 1000`** (0 disables) |
| Nested $n$ | **`--n-list 80,400,1000`** |
| Reference prefix size | **`--n-ref 10000`** (GT MC: `n_mc = n_ref // (num_theta_bins * num_theta_bins_y)` → **100** samples per grid cell here) |
| Runtime | **`mamba run -n geo_diffusion`** · **`--device cuda`** per **`AGENTS.md`** |

Override **`--output-dir`** each run (timestamp or experiment slug) so results do not collide.

## Canonical command (repo root)

```bash
PYTHONUNBUFFERED=1 mamba run -n geo_diffusion python bin/study_h_decoding_twofig.py \
  --dataset-npz data/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x_pr30d.npz \
  --dataset-family gridcos_gaussian2d_sqrtd_rand_tune_additive \
  --theta-field-methods bin_gaussian,linear_x_flow_diagonal_t \
  --theta-binning-mode theta2_grid \
  --num-theta-bins 10 \
  --num-theta-bins-y 10 \
  --lxfs-epochs 50000 \
  --lxfs-early-patience 1000 \
  --n-list 80,400,1000 \
  --n-ref 10000 \
  --device cuda \
  --output-dir data/experiments/native2d_gridcos_pr30d_bin_vs_lxf_diag_minimal_<suffix>
```

## Agent behavior

1. If the NPZ is missing, point to **`bin/make_dataset.py`** + **`bin/project_dataset_pr_autoencoder.py`** with the **`gridcos_gaussian2d_sqrtd_rand_tune_additive`** recipe (see **`journal/notes/2026-05-02-native-2d-theta-benchmark-datasets.md`**).
2. Report artifact paths under **`data/...`** when outputs live under `DATAROOT` (**`AGENTS.md`**).
