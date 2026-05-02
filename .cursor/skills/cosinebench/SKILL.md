---
name: cosinebench
description: >-
  In the score-matching-fisher repo, the user’s shorthand "cosinebench" names a fixed synthetic
  dataset: cosine_gaussian_sqrtd_rand_tune_additive at x_dim 5 with obs_noise_scale 0.25 and
  n_total 10000 joint draws, then PR-autoencoder embedded to 30D; artifacts under
  data/cosine_sqrtd_rand_tune_additive_xdim5/. Use when the user says cosinebench, cosine bench,
  or that benchmark dataset.
---

# cosinebench

## Meaning

**"cosinebench"** means this exact dataset (not a family name in code—only a user alias):

| Stage | What |
|-------|------|
| Generative family | `cosine_gaussian_sqrtd_rand_tune_additive` (additive $|{\mu}|$ sqrt-$d$ variance law) |
| Noise scaling | `--obs-noise-scale 0.25` (recipe $\sigma_{x1}=\sigma_{x2}=0.5$ → stored **0.125** each) |
| Native observation dim | 5 (`--x-dim 5`) |
| Embedded observation dim | 30 (PR-autoencoder; `--h-dim 30`) |
| Joint samples | `--n-total 10000` (default `--train-frac 0.7` → 7000 train / 3000 validation) |

## Canonical artifacts (repo `data/`)

All of these live in **`data/cosine_sqrtd_rand_tune_additive_xdim5/`** (full paths: `<repo>/data/cosine_sqrtd_rand_tune_additive_xdim5/...`).

**Low-dim (5D) NPZ**

- `cosine_sqrtd_rand_tune_additive_xdim5.npz`

**PR-embedded (30D) NPZ** (Fisher / score on high-dimensional $x$)

- `cosine_sqrtd_rand_tune_additive_xdim5_pr30d.npz`  
- Sidecar: `cosine_sqrtd_rand_tune_additive_xdim5_pr30d.projection_meta.json`

**Figures** (same directory)

- `joint_scatter_and_tuning_curve.png` / `.svg` — `bin/make_dataset.py` (tuning curves + PCA scatter of native $x$)
- `pr_projection_summary.png` / `.svg` — `bin/project_dataset_pr_autoencoder.py` (tuning, binned embedded mean + scatter, PR loss)

## Regenerating (if needed)

From repo root, `mamba` env `geo_diffusion`, projection on CUDA per `AGENTS.md`:

```bash
mamba run -n geo_diffusion python bin/make_dataset.py \
  --dataset-family cosine_gaussian_sqrtd_rand_tune_additive \
  --x-dim 5 \
  --obs-noise-scale 0.25 \
  --n-total 10000 \
  --output-npz data/cosine_sqrtd_rand_tune_additive_xdim5/cosine_sqrtd_rand_tune_additive_xdim5.npz

mamba run -n geo_diffusion python bin/project_dataset_pr_autoencoder.py \
  --input-npz data/cosine_sqrtd_rand_tune_additive_xdim5/cosine_sqrtd_rand_tune_additive_xdim5.npz \
  --output-npz data/cosine_sqrtd_rand_tune_additive_xdim5/cosine_sqrtd_rand_tune_additive_xdim5_pr30d.npz \
  --h-dim 30 \
  --allow-non-randamp-sqrtd \
  --device cuda
```

The projector expects `dataset_family == randamp_gaussian_sqrtd` unless **`--allow-non-randamp-sqrtd`** is set (required for this cosine family).

## Agent behavior

When the user says **cosinebench**, resolve paths and semantics to the above; do not substitute a different `--dataset-family`, **`--obs-noise-scale`**, **`--n-total`**, or dimensions unless they explicitly change the alias.
