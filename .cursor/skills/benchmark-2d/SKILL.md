---
name: benchmark-2d
description: >-
  Score-matching-fisher native 2D-θ PR30D H-decoding twofig: theta2_grid (10×10), n_ref 10000,
  n-list 80,400,1000, randamp_gaussian2d_sqrtd or gridcos_gaussian2d_sqrtd_rand_tune_additive NPZ.
  Method rows are configurable (--theta-field-rows / --theta-field-methods); use lxfs_* for
  scheduled *_t LXF and lxf_* for time-independent LXF. See journal note for bin_gaussian + all *_t run.
  Use when the user says benchmark-2d, benchmark 2D, native 2D theta twofig, or 2D theta2grid PR30.
---

# benchmark-2d (native 2D $\theta$, PR30D, `theta2_grid`)

**“benchmark-2d”** names the **geometry and data bundle** for H-decoding [`bin/study_h_decoding_twofig.py`](../../../bin/study_h_decoding_twofig.py) on **native 2D $\theta$** observations embedded to **30D**, with Hellinger binning on a **flattened $(\theta_1,\theta_2)$ grid** (`theta2_grid`), **not** a $\theta_1$-only marginal.

**Methods are not fixed by this alias.** Swap `--theta-field-rows` (or `--theta-field-methods`) for whatever comparison you want; only adjust training flags to match the method family (see below).

| Stage | What |
|-------|------|
| Observation | Native **5D** $x$, PR-autoencoder embedded to **30D** (`*_pr30d.npz`) |
| $\theta$ | **2D** (columns of `theta` in the NPZ) |
| Binning | **`--theta-binning-mode theta2_grid`** · **`--num-theta-bins 10`** · **`--num-theta-bins-y 10`** → **100** cells |
| Nested subsets | **`--n-list 80,400,1000`** |
| GT / reference prefix | **`--n-ref 10000`** → MC samples per cell: $\lfloor n_{\mathrm{ref}}/(10\cdot 10)\rfloor = 100$ |
| Runtime | **`mamba run -n geo_diffusion`** · **`--device cuda`** per [**`AGENTS.md`**](../../../AGENTS.md) |

## Dataset families (pick one per run)

**Randamp 2D**

| | |
|--|--|
| Family | `randamp_gaussian2d_sqrtd` |
| PR30D NPZ | **`data/randamp_gaussian2d_sqrtd_xdim5/randamp_gaussian2d_sqrtd_xdim5_pr30d.npz`** |

**Gridcos 2D**

| | |
|--|--|
| Family | `gridcos_gaussian2d_sqrtd_rand_tune_additive` |
| PR30D NPZ | **`data/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x_pr30d.npz`** |

Background: **`journal/notes/2026-05-02-native-2d-theta-benchmark-datasets.md`**.

## Method rows (you choose)

Pass **`--theta-field-rows tok1,tok2,...`** (highest precedence) or **`--theta-field-methods ...`**.

Examples:

- Minimal diagonal baseline (skills **`bin-2d-lin-lxfdiag`**, **`bin-2d-cos-lxfdiag`**): `bin_gaussian,linear_x_flow_diagonal` with **`--lxf-epochs`**, **`--lxf-early-patience`**.
- Full scheduled **`_t`** family + baseline (documented run):  
  `bin_gaussian,linear_x_flow_t,linear_x_flow_scalar_t,linear_x_flow_diagonal_t,linear_x_flow_diagonal_theta_t,linear_x_flow_low_rank_t,linear_x_flow_low_rank_randb_t`  
  with **`--lxfs-path-schedule`**, **`--lxfs-epochs`**, **`--lxfs-early-patience`**, and **`--lxf-low-rank-dim`** for low-rank rows.

**Training-flag routing:** methods in `_TIME_LXF_METHODS` in [`bin/study_h_decoding_convergence.py`](../../../bin/study_h_decoding_convergence.py) use the **`lxfs_*`** CLI prefix; time-independent linear X-flow uses **`lxf_*`**.

## Example completed run (all `*_t` + `bin_gaussian`, LXFS 50k cap)

Journal: **`journal/notes/2026-05-02-native2d-theta2grid-bin-plus-all-t-lxf.md`**.

- Randamp 2D: **`data/experiments/native2d_randamp_pr30d_bin_plus_all_t_lxf_50k_20260502/`**
- Gridcos 2D: **`data/experiments/native2d_gridcos_pr30d_bin_plus_all_t_lxf_50k_20260502/`**

## Canonical command templates (swap methods / flags)

**Randamp 2D — replace `<METHOD_ARGS>` and `<TRAIN_FLAGS>`**

```bash
PYTHONUNBUFFERED=1 mamba run -n geo_diffusion python bin/study_h_decoding_twofig.py \
  --dataset-npz data/randamp_gaussian2d_sqrtd_xdim5/randamp_gaussian2d_sqrtd_xdim5_pr30d.npz \
  --dataset-family randamp_gaussian2d_sqrtd \
  --theta-binning-mode theta2_grid \
  --num-theta-bins 10 \
  --num-theta-bins-y 10 \
  --n-ref 10000 \
  --n-list 80,400,1000 \
  <METHOD_ARGS> \
  <TRAIN_FLAGS> \
  --device cuda \
  --output-dir data/experiments/native2d_randamp_pr30d_<your_slug>
```

**Gridcos 2D**

```bash
PYTHONUNBUFFERED=1 mamba run -n geo_diffusion python bin/study_h_decoding_twofig.py \
  --dataset-npz data/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x_pr30d.npz \
  --dataset-family gridcos_gaussian2d_sqrtd_rand_tune_additive \
  --theta-binning-mode theta2_grid \
  --num-theta-bins 10 \
  --num-theta-bins-y 10 \
  --n-ref 10000 \
  --n-list 80,400,1000 \
  <METHOD_ARGS> \
  <TRAIN_FLAGS> \
  --device cuda \
  --output-dir data/experiments/native2d_gridcos_pr30d_<your_slug>
```

Example substitutions:

- **`<METHOD_ARGS>`** — e.g. `--theta-field-methods bin_gaussian,linear_x_flow_diagonal` or `--theta-field-rows bin_gaussian,linear_x_flow_diagonal_t,...`
- **`<TRAIN_FLAGS>`** — e.g. `--lxf-epochs 50000 --lxf-early-patience 1000` **or** `--lxfs-epochs 50000 --lxfs-early-patience 1000 --lxfs-path-schedule cosine --lxf-low-rank-dim 4`

Always set a fresh **`--output-dir`** so runs do not overwrite each other.

## Agent behavior

1. Resolve **benchmark-2d** to the **theta2_grid + 10×10 + `n_ref=10000` + `n-list 80,400,1000`** contract and the **two NPZ families** above; **do not** silently swap dataset or binning unless the user asks.
2. **Do not** assume a fixed method list: take **`--theta-field-rows`** / **`--theta-field-methods`** from the user or from an attached skill/note; align **`lxf_*` vs `lxfs_*`** with the chosen methods.
3. If an NPZ is missing, point to **`bin/make_dataset.py`** + **`bin/project_dataset_pr_autoencoder.py`** (see **`journal/notes/2026-05-02-native-2d-theta-benchmark-datasets.md`**).
4. Report artifact paths under **`data/...`** when outputs live under `DATAROOT` (**`AGENTS.md`**).
