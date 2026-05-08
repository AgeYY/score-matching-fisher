---
name: benchmark-cate
description: >-
  Score-matching-fisher: generate random_mog_categorical (uniform categorical diagonal Gaussian
  mixture, one-hot θ) with x_dim 2, n_total 50000, num_categories 5, then PR-autoencoder embed to 30D
  via --allow-non-randamp-sqrtd. Evaluation on benchmark-cate means the 30D NPZ only. For
  bin/study_h_decoding_twofig.py use --n-list 80,200,400,600. Canonical tree
  data/random_mog_categorical_xdim2_default/. Use when the user says benchmark-cate, benchmark cate,
  categorical MoG PR30 dataset, or mog_categorical default make_dataset + project to 30.
---

# benchmark-cate

## Meaning

**"benchmark-cate"** names a **fixed benchmark bundle** in this repo:

| Stage | What |
|-------|------|
| Generative family | `random_mog_categorical` (uniform mixture over $K$ categories; one-hot $\theta$; per-category diagonal Gaussian $x \mid k$; component means in NPZ meta) |
| Native observation dim | **2** (`--x-dim 2`, **default** in `bin/make_dataset.py` if omitted) |
| Categories | **5** (`--num-categories` default) |
| Joint samples | **50000** (`--n-total 50000`); **`--train-frac 0.7`** → 35000 train / 15000 validation |
| Other CLI | **Defaults** where not overridden above: `seed=7`, `obs_noise_scale=1.0`, family recipe from `fisher.dataset_family_recipes` |
| Embedded observation dim | **30** (PR-autoencoder; `--h-dim 30`) |
| Projector flag | **`--allow-non-randamp-sqrtd`** — required because `bin/project_dataset_pr_autoencoder.py` otherwise accepts only `randamp_gaussian_sqrtd` inputs |

$\theta$ is stored as **one-hot** rows of shape $(N, K)$; the embedded NPZ keeps the same `theta_*` arrays and `dataset_family`.

### Evaluation

When the task is to **evaluate a method on benchmark-cate**, use the **PR-embedded 30D data only**: `random_mog_categorical_pr30.npz`. Do not treat the native 2D NPZ (`random_mog_categorical.npz`) as the benchmark observation space unless the user explicitly asks for native-$x$ experiments.

### `bin/study_h_decoding_twofig.py`

For H-decoding twofig sweeps on benchmark-cate, use **`--n-list 80,200,400,600`** (nested subset columns). Ensure **`--n-ref`** is at least **600** so `max(n-list) <= n-ref` holds.

## Canonical artifacts (repo `data/`)

Use **`data/random_mog_categorical_xdim2_default/`** (paths from repo root: `<repo>/data/random_mog_categorical_xdim2_default/...`).

**Low-dim (2D) NPZ** (intermediate for projection; not the benchmark observation space for method evaluation)

- `random_mog_categorical.npz`

**PR-embedded (30D) NPZ** (this is **benchmark-cate** for training/evaluation scripts)

- `random_mog_categorical_pr30.npz`
- Sidecar: `random_mog_categorical_pr30.projection_meta.json`

**Figures** (same directory)

- `joint_scatter_and_tuning_curve.png` / `.svg` — `bin/make_dataset.py`
- `pr_projection_summary.png` / `.svg` — `bin/project_dataset_pr_autoencoder.py` (unless `--skip-viz`)

## Regenerating

From repo root, `mamba` env **`geo_diffusion`**, CUDA per **`AGENTS.md`**:

```bash
mamba run -n geo_diffusion python bin/make_dataset.py \
  --dataset-family random_mog_categorical \
  --n-total 50000 \
  --output-npz data/random_mog_categorical_xdim2_default/random_mog_categorical.npz \
  --device cuda

mamba run -n geo_diffusion python bin/project_dataset_pr_autoencoder.py \
  --input-npz data/random_mog_categorical_xdim2_default/random_mog_categorical.npz \
  --output-npz data/random_mog_categorical_xdim2_default/random_mog_categorical_pr30.npz \
  --h-dim 30 \
  --allow-non-randamp-sqrtd \
  --device cuda
```

Omit explicit **`--x-dim`** and **`--num-categories`** to stay on defaults; keep **`--n-total 50000`** for the canonical recipe. Change output paths if you need a variant directory (then it is no longer the canonical **benchmark-cate** tree unless the user redefines the alias).

## Agent behavior

1. Resolve **benchmark-cate** to the dataset generation + PR30 projection flow above; do not swap in **`randamp_gaussian_sqrtd`** or drop **`--allow-non-randamp-sqrtd`** for this alias unless the user changes the task.
2. For **evaluation**, point methods at **`random_mog_categorical_pr30.npz`** (30D embeddings only), not the 2D native NPZ.
3. When invoking **`bin/study_h_decoding_twofig.py`** on this benchmark, pass **`--n-list 80,200,400,600`** unless the user overrides it.
4. Report artifact paths as **`<repo-root>/data/random_mog_categorical_xdim2_default/...`** when using the canonical tree (see **`AGENTS.md`** / `DATAROOT` symlink rules).
