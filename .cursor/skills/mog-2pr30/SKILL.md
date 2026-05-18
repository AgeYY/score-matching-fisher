---
name: mog-2pr30
description: >-
  Score-matching-fisher: shorthand **mog-2pr30** = `random_mog_categorical` with K=2 one-hot
  categories, native observation **x_dim 2**, then PR-autoencoder embed **x** to **h_dim 30**
  (`--allow-non-randamp-sqrtd`). Evaluation on mog-2pr30 means the **30D embedded** NPZ unless the
  user asks for native 2D. Canonical tree `data/mog_2pr30_default/`. Use when the user says mog-2pr30,
  mog 2 pr 30, MoG 2PR30, or categorical MoG 2D→PR30 with K=2.
---

# mog-2pr30

## Meaning

**"mog-2pr30"** names this **fixed dataset bundle**:

| Piece | Value |
|-------|--------|
| `dataset_family` | `random_mog_categorical` (uniform categorical diagonal Gaussian mixture; one-hot $\theta$) |
| Categories $K$ | **2** (`--num-categories 2`) |
| Native $x$ | **2D** (`--x-dim 2`) |
| $\theta$ storage | One-hot rows, shape $(N, 2)$ |
| Joint samples (canonical) | **50000** (`--n-total 50000`); **`--train-frac 0.7`** unless overridden |
| Embedded $x$ | **30D** via PR-autoencoder: `bin/project_dataset_pr_autoencoder.py` with **`--h-dim 30`** |
| Projector gate | **`--allow-non-randamp-sqrtd`** (required; script default only allows `randamp_gaussian_sqrtd`) |

**Relation to legacy K=2 path:** Some scripts default to **`data/random_mog_categorical_xdim2_k2/`**, which may hold a **small** smoke NPZ (e.g. $N{=}200$). **mog-2pr30** uses **`data/mog_2pr30_default/`** with the full canonical **`n_total=50000`** recipe. Use whichever tree the user names; do not assume **`mog_2pr30_default`** exists on disk until generated.

**Not** part of the name: PR hidden widths, training epochs, and cache paths use `bin/project_dataset_pr_autoencoder.py` defaults unless the user overrides them.

## Canonical artifacts (repo `data/`)

Directory: **`data/mog_2pr30_default/`** (full path: `<repo-root>/data/mog_2pr30_default/`).

| Role | File |
|------|------|
| Native 2D NPZ | `random_mog_categorical.npz` |
| PR-embedded **30D** NPZ (default observation space for methods) | `random_mog_categorical_pr30.npz` |
| Projection sidecar | `random_mog_categorical_pr30.projection_meta.json` |
| Native figures | `joint_scatter_and_tuning_curve.png` / `.svg` from `bin/make_dataset.py` |
| Embed figures | `pr_projection_summary.png` / `.svg` from `bin/project_dataset_pr_autoencoder.py` (unless `--skip-viz`) |

## Regenerating

From repo root, **`mamba` env `geo_diffusion`**. `bin/make_dataset.py` has no `--device`; use **`--device cuda`** on the projector per **`AGENTS.md`**.

```bash
mamba run -n geo_diffusion python bin/make_dataset.py \
  --dataset-family random_mog_categorical \
  --num-categories 2 \
  --x-dim 2 \
  --n-total 50000 \
  --output-npz data/mog_2pr30_default/random_mog_categorical.npz

mamba run -n geo_diffusion python bin/project_dataset_pr_autoencoder.py \
  --input-npz data/mog_2pr30_default/random_mog_categorical.npz \
  --output-npz data/mog_2pr30_default/random_mog_categorical_pr30.npz \
  --h-dim 30 \
  --allow-non-randamp-sqrtd \
  --device cuda
```

Smaller `--n-total` is fine for smoke tests; then it is still the same **recipe** but not the full canonical sample count.

## Agent behavior

1. When the user says **mog-2pr30** (or close variants), resolve to **$K{=}2$, native $x{\in}\mathbb{R}^2$, embedded $x{\in}\mathbb{R}^{30}$** — not **mog-5pr30** ($K{=}5$) or **mog-2pr5** (5D embed).
2. For **training/evaluation**, prefer **`random_mog_categorical_pr30.npz`** under **`data/mog_2pr30_default/`** unless the user explicitly wants native 2D $x$.
3. Pass **`--num-categories 2`** when wiring categorical twofig / LLR scatter CLIs to this bundle.
4. Report paths as **`<repo-root>/data/mog_2pr30_default/...`** when using this tree (see **`AGENTS.md`** / `DATAROOT` symlink rules).
