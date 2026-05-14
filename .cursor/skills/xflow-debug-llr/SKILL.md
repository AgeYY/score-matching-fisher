---
name: xflow-debug-llr
description: >-
  Score-matching-fisher: quick categorical H-decoding / LLR debug via `bin/debug_llr.py` on **mog-5pr5**
  only — fixed nested sample size **`--n-list 200`**, methods **`binary_classifier`** and **`x_flow`** only,
  native 2D NPZ + PR-5 embedding. Use when the user says xflow-debug-llr, debug LLR mog-5pr5, or
  categorical twofig smoke with binary_classifier + x_flow on PR5 MoG.
---

# xflow-debug-llr

## What it is

A **minimal** run of [`bin/debug_llr.py`](bin/debug_llr.py) (CLI entry for [`fisher/h_decoding_categorical_twofig.py`](fisher/h_decoding_categorical_twofig.py)) to debug **LLR / Hellinger** paths on the **mog-5pr5** bundle (see [mog-5pr5](../mog-5pr5/SKILL.md)):

| Knob | Fixed value |
|------|-------------|
| Dataset | **mog-5pr5**: native `random_mog_categorical.npz` + PR-embedded **5D** observation space |
| `--num-categories` | **5** |
| `--n-list` | **200** only (single nested-$n$ point) |
| `--methods` | **`binary_classifier,x_flow`** (no `linear_x_flow_t`, `xflow_sir_lrank`, etc.) |
| Device | **`--device cuda`** per [`AGENTS.md`](../../AGENTS.md) |

Default `--n-ref` is **10000**; with `n_list=200` this satisfies `max(n_list) <= n_ref`.

## Canonical command (repo root)

Use env **`geo_diffusion`**:

```bash
mamba run -n geo_diffusion python bin/debug_llr.py \
  --num-categories 5 \
  --dataset-npz data/mog_5pr5_default/random_mog_categorical.npz \
  --pr-project \
  --pr-dim 5 \
  --pr-output-npz data/mog_5pr5_default/random_mog_categorical_pr5.npz \
  --n-list 200 \
  --methods binary_classifier,x_flow \
  --output-dir data/cate-exp/mog_5pr5_xflow_debug_llr_n200 \
  --device cuda
```

- **`--dataset-npz`**: native 2D joint NPZ under **`data/mog_5pr5_default/`** (same inode as `DATAROOT` when `data/` is symlinked).
- **`--pr-project` / `--pr-dim 5` / `--pr-output-npz`**: train or reuse the canonical **PR5** embedded NPZ; matches mog-5pr5’s `random_mog_categorical_pr5.npz` filename used in that tree.
- **`--output-dir`**: example dedicated run root under **`data/cate-exp/`**; change freely.

If the native or PR NPZ is missing, the driver can call `bin/make_dataset.py` / `bin/project_dataset_pr_autoencoder.py` (see mog-5pr5 skill). Use **`--force-regenerate`** only when you intend to rebuild data.

## Method aliases (optional)

The parser accepts aliases (e.g. `x-flow`, `binary-classifier`); the canonical spellings above match internal names **`x_flow`** and **`binary_classifier`**.

## Outputs (typical)

Under `--output-dir`: sweep / correlation NMSE SVGs, optional training-loss panel, NPZ, and unless `--no-scatter-diagnostics` is set, **`llr_est_vs_true_all`** / **`hellinger_est_vs_gt_all`** figures. Exact filenames are written beside the run (see `run_summary.txt` in that directory).

## Agent behavior

1. Resolve **mog-5pr5** to **`data/mog_5pr5_default/`** and **$K{=}5$**, **PR $h{=}5$** — not mog-5pr30 / benchmark-cate PR30.
2. Keep **`--n-list 200`** and **two methods only** unless the user explicitly expands scope.
3. Report artifact paths as **`<repo-root>/data/...`** per [`AGENTS.md`](../../AGENTS.md).
