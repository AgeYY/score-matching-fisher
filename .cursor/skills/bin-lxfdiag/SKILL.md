---
name: bin-lxfdiag
description: >-
  Score-matching-fisher minimal H-decoding twofig on native randamp 2D PR30D: bin_gaussian +
  linear_x_flow_diagonal, --n-ref 10000, --lxf-early-patience 1000, --n-list 80,400,1000,
  --device cuda, dataset randamp_gaussian2d_sqrtd. Use when the user says bin-lxfdiag,
  bin vs lxfdiag minimal, or wants to rerun the bin_gaussian vs linear_x_flow_diagonal native 2D PR30 twofig baseline.
---

# Minimal twofig: `bin_gaussian` + `linear_x_flow_diagonal` (native randamp 2D PR30D)

Fixed experimental bundle for **`bin/study_h_decoding_twofig.py`** — isolate **binned Gaussian** vs **diagonal linear X-flow matching** on **`randamp_gaussian2d_sqrtd`** embedded to **30D**.

Journal write-up: **`journal/notes/2026-05-02-native2d-randamp-bin-vs-linear-x-flow-diagonal-minimal-twofig.md`**.

## Locked choices

| Setting | Value |
|--------|--------|
| Dataset NPZ | **`data/randamp_gaussian2d_sqrtd_xdim5/randamp_gaussian2d_sqrtd_xdim5_pr30d.npz`** |
| `--dataset-family` | **`randamp_gaussian2d_sqrtd`** |
| Methods | **`bin_gaussian,linear_x_flow_diagonal`** (order preserved for figure rows) |
| LXF patience | **`--lxf-early-patience 1000`** |
| Nested $n$ | **`--n-list 80,400,1000`** |
| Reference prefix size | **`--n-ref 10000`** (full pool; GT MC uses `n_mc = n_ref // num_theta_bins`, e.g. **1000**/row at default 10 bins) |
| Runtime | **`mamba run -n geo_diffusion`** · **`--device cuda`** per **`AGENTS.md`** |

Override **`--output-dir`** each run (timestamp or experiment slug) so results do not collide.

## Canonical command (repo root)

```bash
PYTHONUNBUFFERED=1 mamba run -n geo_diffusion python bin/study_h_decoding_twofig.py \
  --dataset-npz data/randamp_gaussian2d_sqrtd_xdim5/randamp_gaussian2d_sqrtd_xdim5_pr30d.npz \
  --dataset-family randamp_gaussian2d_sqrtd \
  --theta-field-methods bin_gaussian,linear_x_flow_diagonal \
  --lxf-early-patience 1000 \
  --n-list 80,400,1000 \
  --n-ref 10000 \
  --device cuda \
  --output-dir data/experiments/native2d_randamp_pr30d_bin_vs_lxf_diag_minimal_<suffix>
```

Replace **`<suffix>`** (e.g. date or `run003`). Example completed run from before this skill pinned **`n_ref=10000`** (used **`--n-ref 5000`**):

`/grad/zeyuan/score-matching-fisher/data/experiments/native2d_randamp_pr30d_bin_vs_lxf_diag_minimal_20260502/`

## Agent behavior

1. If the NPZ is missing, point to **`bin/make_dataset.py`** + **`bin/project_dataset_pr_autoencoder.py`** (see **`journal/notes/2026-05-02-native-2d-theta-benchmark-datasets.md`**).
2. Report artifact paths under **`data/...`** when outputs live under `DATAROOT` (**`AGENTS.md`**).
