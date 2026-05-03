---
name: benchmark-1d-cosine-n80200
description: >-
  Score-matching-fisher: 1D-theta H-decoding twofig on PR30 cosinebench only, nested subset sweep
  --n-list 80,200 (no linearbench). Same lxfs / method-row conventions as benchmark-1d. Use when
  the user says cosine-only 1D twofig n 80/200, benchmark-1D cosine n80200, or faster cosinebench
  twofig with two n points.
---

# benchmark-1D cosine (n = 80, 200)

## Meaning

Derived from [**`benchmark-1d`**](../benchmark-1d/SKILL.md): run [`bin/study_h_decoding_twofig.py`](bin/study_h_decoding_twofig.py) on the **canonical PR-30D cosinebench** dataset only (resolve paths and semantics with [**`cosinebench`**](../cosinebench/SKILL.md)). Geometry is **1D $\theta$ binning** (e.g. `theta_binning_mode=theta1`, `n_bins=10`); this is the **embedded 5D-native** bench, not native-2D `theta2_grid`.

**This skill does not include linearbench / randamp.**

## What is fixed vs what you may change

| Fixed by this skill | User may override |
|---------------------|-------------------|
| **Cosinebench** NPZ + `--dataset-family cosine_gaussian_sqrtd_rand_tune_additive` | Other `--dataset-npz` / family (then this skill does not apply) |
| Nested subset sweep **`--n-list 80,200`** | Other `--n-list`; shrink `--n-ref` / `--n-list` if $n_{\mathrm{total}}$ is smaller |
| **`mamba run -n geo_diffusion`**, **`--device cuda`**, reporting under **`data/...`** | CPU-only runs violate **`AGENTS.md`** unless the user accepts that constraint |

**Theta-field methods are not fixed.** Example rows match **`benchmark-1d`**: **`bin_gaussian`** plus **`_TIME_LXF_METHODS`** tokens from [`bin/study_h_decoding_convergence.py`](bin/study_h_decoding_convergence.py) (`linear_x_flow_t`, `linear_x_flow_scalar_t`, `linear_x_flow_diagonal_t`, `linear_x_flow_diagonal_theta_t`, `linear_x_flow_low_rank_t`, `linear_x_flow_low_rank_randb_t`). Pass your own **`--theta-field-rows`** / **`--theta-field-methods`** as needed. Low-rank rows require **`--lxf-low-rank-dim`**.

## Scheduled time-dependent LXF (`*_t`) — use `lxfs_*` hyperparameters

Same as **`benchmark-1d`**: for **`_TIME_LXF_METHODS`**, use the **`lxfs`** prefix in [`bin/study_h_decoding_convergence.py`](bin/study_h_decoding_convergence.py) (`lxf_prefix = "lxfs"`).

- **`--lxfs-epochs 50000`**
- **`--lxfs-early-patience 1000`**
- **`--lxfs-path-schedule cosine`**

## Example command (repo root)

**Cosinebench** — illustrative full **`_t`** sweep + `bin_gaussian` (replace `--theta-field-rows` if you want a different method set):

```bash
PYTHONUNBUFFERED=1 mamba run -n geo_diffusion python bin/study_h_decoding_twofig.py \
  --dataset-npz data/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x_pr30d.npz \
  --dataset-family cosine_gaussian_sqrtd_rand_tune_additive \
  --theta-field-rows bin_gaussian,linear_x_flow_t,linear_x_flow_scalar_t,linear_x_flow_diagonal_t,linear_x_flow_diagonal_theta_t,linear_x_flow_low_rank_t,linear_x_flow_low_rank_randb_t \
  --lxf-low-rank-dim 4 \
  --n-list 80,200 \
  --lxfs-path-schedule cosine \
  --lxfs-epochs 50000 \
  --lxfs-early-patience 1000 \
  --device cuda \
  --output-dir data/experiments/h_decoding_twofig_pr30d_cosinebench_<TAG> \
  2>&1 | tee data/experiments/h_decoding_twofig_pr30d_cosinebench_<TAG>/run.log
```

Replace `<TAG>` with a short run label (e.g. date + method slug). Create the output directory before `tee` if needed (`mkdir -p`).

## Expected artifacts

Under **`--output-dir`**: `h_decoding_twofig_results.npz`, `h_decoding_twofig_summary.txt`, `h_decoding_twofig_{sweep,gt,corr_vs_n,nmse_vs_n,training_losses_panel}.svg`, plus **`run.log`** if you tee as above.

## Agent behavior

1. Resolve **cosinebench** only via [**`cosinebench`**](../cosinebench/SKILL.md); do not silently change noise scales, $n_{\mathrm{total}}$, or dimensions.
2. When the user asks for **different methods**, keep the **cosine dataset + `--n-list 80,200` + (if applicable) lxfs/lxf knobs** from this skill unless they say otherwise.
3. Report paths as **`<repo-root>/data/...`** when outputs live under `DATAROOT` (see **`AGENTS.md`**).
