---
name: mog-5pr30
description: >-
  Score-matching-fisher: shorthand **mog-5pr30** = `random_mog_categorical` with K=5 one-hot
  categories, native observation **x_dim 2**, then PR-autoencoder embed **x** to **h_dim 30**
  (`--allow-non-randamp-sqrtd`). Evaluation on mog-5pr30 means the **30D embedded** NPZ unless the
  user asks for native 2D. Canonical tree `data/mog_5pr30_default/`. Same numerics as **benchmark-cate**
  (see `.cursor/skills/benchmark-cate/SKILL.md`) but mog-5pr30 names this path layout. Use when the user
  says mog-5pr30, mog 5 pr 30, MoG 5PR30, or categorical MoG 2D→PR30.
---

# mog-5pr30

## Meaning

**"mog-5pr30"** names this **fixed dataset bundle**:

| Piece | Value |
|-------|--------|
| `dataset_family` | `random_mog_categorical` (uniform categorical diagonal Gaussian mixture; one-hot $\theta$) |
| Categories $K$ | **5** (`--num-categories 5`) |
| Native $x$ | **2D** (`--x-dim 2`) |
| $\theta$ storage | One-hot rows, shape $(N, 5)$ |
| Joint samples (canonical) | **50000** (`--n-total 50000`); **`--train-frac 0.7`** unless overridden |
| Embedded $x$ | **30D** via PR-autoencoder: `bin/project_dataset_pr_autoencoder.py` with **`--h-dim 30`** |
| Projector gate | **`--allow-non-randamp-sqrtd`** (required; script default only allows `randamp_gaussian_sqrtd`) |

**Relation to benchmark-cate:** Same $K$, native dim, embedded dim, and default `n_total` as the **benchmark-cate** skill. That skill’s canonical directory is **`data/random_mog_categorical_xdim2_default/`**; **mog-5pr30** uses **`data/mog_5pr30_default/`** here so the **mog-$K$pr$D$** naming pattern matches **mog-5pr5**. Use whichever tree the user names; do not assume both directories exist on disk until generated.

**Not** part of the name: PR hidden widths, training epochs, and cache paths use `bin/project_dataset_pr_autoencoder.py` defaults unless the user overrides them.

## Canonical artifacts (repo `data/`)

Directory: **`data/mog_5pr30_default/`** (full path: `<repo-root>/data/mog_5pr30_default/`).

| Role | File |
|------|------|
| Native 2D NPZ | `random_mog_categorical.npz` |
| PR-embedded **30D** NPZ (default observation space for methods) | `random_mog_categorical_pr30.npz` |
| Projection sidecar | `random_mog_categorical_pr30.projection_meta.json` |
| Native figures | `joint_scatter_and_tuning_curve.png` / `.svg` from `bin/make_dataset.py` |
| Embed figures | `pr_projection_summary.png` / `.svg` from `bin/project_dataset_pr_autoencoder.py` (unless `--skip-viz`) |

## `bin/study_h_decoding_twofig.py`

On this categorical PR30 setup, use **`--n-list 80,200,400,600`** and **`--n-ref` ≥ 600** (same as **benchmark-cate**).

## Regenerating

From repo root, **`mamba` env `geo_diffusion`**. `bin/make_dataset.py` has no `--device`; use **`--device cuda`** on the projector per **`AGENTS.md`**.

```bash
mamba run -n geo_diffusion python bin/make_dataset.py \
  --dataset-family random_mog_categorical \
  --num-categories 5 \
  --x-dim 2 \
  --n-total 50000 \
  --output-npz data/mog_5pr30_default/random_mog_categorical.npz

mamba run -n geo_diffusion python bin/project_dataset_pr_autoencoder.py \
  --input-npz data/mog_5pr30_default/random_mog_categorical.npz \
  --output-npz data/mog_5pr30_default/random_mog_categorical_pr30.npz \
  --h-dim 30 \
  --allow-non-randamp-sqrtd \
  --device cuda
```

Smaller `--n-total` is fine for smoke tests; then it is still the same **recipe** but not the full canonical sample count.

## Agent behavior

1. When the user says **mog-5pr30** (or close variants), resolve to **$K{=}5$, native $x{\in}\mathbb{R}^2$, embedded $x{\in}\mathbb{R}^{30}$** — not **mog-5pr5** (5D embed).
2. For **training/evaluation**, prefer **`random_mog_categorical_pr30.npz`** under **`data/mog_5pr30_default/`** unless the user explicitly wants native 2D $x$.
3. If the user says **benchmark-cate**, use **`benchmark-cate`** skill paths (`data/random_mog_categorical_xdim2_default/`); if they say **mog-5pr30**, use **`data/mog_5pr30_default/`** unless they point at another NPZ.
4. Report paths as **`<repo-root>/data/mog_5pr30_default/...`** when using this tree (see **`AGENTS.md`** / `DATAROOT` symlink rules).
