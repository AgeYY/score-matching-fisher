---
name: lxf-bench-h-decoding-twofig
description: >-
  Score-matching-fisher: when running bin/study_h_decoding_twofig.py with scheduled linear_x_flow_t
  variants on cosinebench or linearbench, use --lxfs-early-patience 1000 and
  --n-list 80,400,1000 (with script defaults for n-ref and benches at n_total 10000). Use when
  the user mentions these methods on cosinebench, linearbench, or H-decoding twofig LXF bench runs.
---

# LXF H-decoding twofig on cosinebench / linearbench

When the user asks to run supported scheduled LXF rows such as **`linear_x_flow_t`** or **`linear_x_flow_diagonal_t`** via **`bin/study_h_decoding_twofig.py`** on **`/cosinebench`** or **`/linearbench`**, apply this fixed hyperparameter bundle unless they explicitly override it.

## Locked choices

| Setting | Value |
|--------|--------|
| Early stopping patience | **`1000`** → pass **`--lxfs-early-patience 1000`**. |
| Nested subset sweep | **`--n-list 80,400,1000`** |

Keep **`--n-ref 5000`** (script default) when the dataset pool has at least 5000 samples (canonical cosinebench / linearbench use **`--n-total 10000`**). If the on-disk NPZ is smaller, lower **`--n-ref`** and/or **`--n-list`** so `n_total ≥ max(n_ref, max(n_list))`.

## Dataset resolution

Resolve **`/cosinebench`** and **`/linearbench`** via the repo skills **`cosinebench`** and **`linearbench`**: PR **`…_pr30d.npz`**, matching **`--dataset-family`**, **`mamba run -n geo_diffusion`**, **`--device cuda`** per **`AGENTS.md`**.

## Example commands (repo root)

**Linearbench** — scheduled full and diagonal rows:

```bash
mamba run -n geo_diffusion python bin/study_h_decoding_twofig.py \
  --dataset-npz data/randamp_gaussian_sqrtd_xdim5/randamp_gaussian_sqrtd_xdim5_pr30d.npz \
  --dataset-family randamp_gaussian_sqrtd \
  --theta-field-methods linear-x-flow-t,linear-x-flow-diagonal-t \
  --lxfs-early-patience 1000 \
  --n-list 80,400,1000 \
  --device cuda \
  --output-dir data/randamp_gaussian_sqrtd_xdim5/h_decoding_twofig_lxfs_t_diag_t_n80_400_1000
```

**Cosinebench** (`noise2x_alpha2x` canonical NPZ; see **`cosinebench`** skill):

```bash
mamba run -n geo_diffusion python bin/study_h_decoding_twofig.py \
  --dataset-npz data/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x_pr30d.npz \
  --dataset-family cosine_gaussian_sqrtd_rand_tune_additive \
  --theta-field-methods linear-x-flow-t,linear-x-flow-diagonal-t \
  --lxfs-early-patience 1000 \
  --n-list 80,400,1000 \
  --device cuda \
  --output-dir data/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/h_decoding_twofig_lxfs_t_diag_t_n80_400_1000
```

Use **`--theta-field-method linear-x-flow-t`** alone when only the full scheduled LXF row is requested.

## Agent behavior

1. Prefer **`./data/...`** paths from repo root when reporting locations (see **`AGENTS.md`**).
2. If **`--lxfs-early-patience`** is omitted, the convergence CLI default is still available; pass **`1000`** when following this skill so runs are explicit and reproducible.
