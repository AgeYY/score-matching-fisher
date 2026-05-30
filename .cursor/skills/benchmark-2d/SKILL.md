---
name: benchmark-2d
description: >-
  Score-matching-fisher native 2D-θ PR30D H-decoding convergence: theta2_grid (10×10), n_ref 10000,
  n-list 80,400,1000, randamp_gaussian2d_sqrtd or gridcos_gaussian2d_sqrtd_rand_tune_additive NPZ.
  Method list is configurable via --methods; scheduled LXF rows use lxfs_* flags. See journal note
  for bin_gaussian + core *_t run. Use when the user says benchmark-2d, benchmark 2D, native 2D
  theta convergence, or 2D theta2grid PR30.
---

# benchmark-2d (native 2D $\theta$, PR30D, `theta2_grid`)

**“benchmark-2d”** names the **geometry and data bundle** for H-decoding [`bin/study_h_decoding_convergence.py`](../../../bin/study_h_decoding_convergence.py) on **native 2D $\theta$** observations embedded to **30D**, with Hellinger binning on a **flattened $(\theta_1,\theta_2)$ grid** (`theta2_grid`), **not** a $\theta_1$-only marginal. The script uses the same continuous twofig pipeline as 1D benches ([`fisher/h_decoding_convergence_twofig.py`](../../../fisher/h_decoding_convergence_twofig.py)).

**Methods are not fixed by this alias.** Pass **`--methods tok1,tok2,...`** for whatever comparison you want; align training flags with the method family (see below). Per-method `:arch` suffixes are **not** supported on the convergence entrypoint—use global **`--flow-arch`** when needed.

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

Pass **`--methods tok1,tok2,...`**.

Examples:

- Minimal diagonal baseline (randamp vs gridcos 2D PR30): `bin_gaussian,linear_x_flow_diagonal_t` with **`--lxfs-epochs`**, **`--lxfs-early-patience`**.
- Full **core** scheduled **`_t`** family + baseline (documented 2026-05-02 run):  
  `bin_gaussian,linear_x_flow_t,linear_x_flow_scalar_t,linear_x_flow_diagonal_t,linear_x_flow_diagonal_theta_t,linear_x_flow_low_rank_t,linear_x_flow_low_rank_randb_t`  
  with **`--lxfs-path-schedule`**, **`--lxfs-epochs`**, **`--lxfs-early-patience`**, and optionally **`--lxf-low-rank-dim`** for low-rank rows (CLI default rank **3**).

**Training-flag routing:** methods in **`_TIME_LXF_METHODS`** in [`fisher/h_decoding_convergence_methods.py`](../../../fisher/h_decoding_convergence_methods.py) use the **`lxfs_*`** CLI prefix (`lxf_prefix = "lxfs"` in that module). The set now also includes `linear_x_flow_pure_low_rank_t`, `linear_x_flow_pure_cond_low_rank_t`, `linear_x_flow_lr_t_ts`, and **`xflow_sir_*`** variants—add them to **`--methods`** when needed; SIR rows may require **`--sir-num-bins`** / **`--sir-ridge`**.

## Example completed run (core `*_t` + `bin_gaussian`, LXFS 50k cap)

Journal: **`journal/notes/2026-05-02-native2d-theta2grid-bin-plus-all-t-lxf.md`**.

- Randamp 2D: **`data/experiments/native2d_randamp_pr30d_bin_plus_all_t_lxf_50k_20260502/`**
- Gridcos 2D: **`data/experiments/native2d_gridcos_pr30d_bin_plus_all_t_lxf_50k_20260502/`**

(Those directories may contain legacy **`h_decoding_twofig_*`** names from before the convergence entrypoint; new runs should produce **`h_decoding_convergence_*`** artifacts.)

## Canonical command templates (swap methods / flags)

**Randamp 2D — replace `<METHODS>` and `<TRAIN_FLAGS>`**

```bash
PYTHONUNBUFFERED=1 mamba run -n geo_diffusion python bin/study_h_decoding_convergence.py \
  --dataset-npz data/randamp_gaussian2d_sqrtd_xdim5/randamp_gaussian2d_sqrtd_xdim5_pr30d.npz \
  --dataset-family randamp_gaussian2d_sqrtd \
  --theta-binning-mode theta2_grid \
  --num-theta-bins 10 \
  --num-theta-bins-y 10 \
  --n-ref 10000 \
  --n-list 80,400,1000 \
  --methods <METHODS> \
  <TRAIN_FLAGS> \
  --device cuda \
  --output-dir data/experiments/native2d_randamp_pr30d_<your_slug>
```

**Gridcos 2D**

```bash
PYTHONUNBUFFERED=1 mamba run -n geo_diffusion python bin/study_h_decoding_convergence.py \
  --dataset-npz data/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x_pr30d.npz \
  --dataset-family gridcos_gaussian2d_sqrtd_rand_tune_additive \
  --theta-binning-mode theta2_grid \
  --num-theta-bins 10 \
  --num-theta-bins-y 10 \
  --n-ref 10000 \
  --n-list 80,400,1000 \
  --methods <METHODS> \
  <TRAIN_FLAGS> \
  --device cuda \
  --output-dir data/experiments/native2d_gridcos_pr30d_<your_slug>
```

Example substitutions:

- **`<METHODS>`** — e.g. `bin_gaussian,linear_x_flow_diagonal_t,...` (comma-separated, no `:arch` suffixes)
- **`<TRAIN_FLAGS>`** — e.g. `--lxf-epochs 50000 --lxf-early-patience 1000` **or** `--lxfs-epochs 50000 --lxfs-early-patience 1000 --lxfs-path-schedule cosine` (add **`--lxf-low-rank-dim`** only if you want a rank other than the default **3**)

Always set a fresh **`--output-dir`** so runs do not overwrite each other.

## Expected artifacts

Same as benchmark-1D convergence runs: **`h_decoding_convergence_results.npz`**, **`h_decoding_convergence_summary.txt`**, **`h_decoding_convergence_{sweep,corr_nmse,training_losses_panel}.svg`**, optional **`h_decoding_convergence_all_columns.png`**.

## Agent behavior

1. Resolve **benchmark-2d** to the **theta2_grid + 10×10 + `n_ref=10000` + `n-list 80,400,1000`** contract and the **two NPZ families** above; **do not** silently swap dataset or binning unless the user asks.
2. Use **`bin/study_h_decoding_convergence.py`** with **`--methods`**, not **`bin/study_h_decoding_twofig.py`** / **`--theta-field-rows`**. Take the method list from the user or this skill; align **`lxf_*` vs `lxfs_*`** with the chosen methods per **`_TIME_LXF_METHODS`** in [`fisher/h_decoding_convergence_methods.py`](../../../fisher/h_decoding_convergence_methods.py).
3. If an NPZ is missing, point to **`bin/make_dataset.py`** + **`bin/project_dataset_pr_autoencoder.py`** (see **`journal/notes/2026-05-02-native-2d-theta-benchmark-datasets.md`**).
4. Report artifact paths under **`data/...`** when outputs live under `DATAROOT` (**`AGENTS.md`**).
