---
name: linearbench
description: >-
  In the score-matching-fisher repo, the user’s shorthand "linearbench" names a fixed synthetic
  dataset: randamp_gaussian_sqrtd at x_dim 5 with default obs_noise_scale 1.0 and n_total 10000
  joint draws, then PR-autoencoder embedded to 30D; artifacts under data/randamp_gaussian_sqrtd_xdim5/.
  Use when the user says linearbench, linear bench, linear-bench, or that benchmark dataset.
---

# linearbench

## Meaning

**"linearbench"** means this exact dataset (not a family name in code—only a user alias):

| Stage | What |
|-------|------|
| Generative family | `randamp_gaussian_sqrtd` (random-amplitude Gaussian bumps in $\theta$; sqrt-$d$ diagonal noise, additive $|{\mu}|$ law in meta) |
| Noise scaling | `--obs-noise-scale 1.0` (default; recipe $\sigma_{x1}=\sigma_{x2}=0.2/\sqrt{2}$ as fixed by the family) |
| Native observation dim | 5 (`--x-dim 5`) |
| Embedded observation dim | 30 (PR-autoencoder; `--h-dim 30`) |
| Joint samples | `--n-total 10000` (default `--train-frac 0.7` → 7000 train / 3000 validation) |

## Canonical artifacts (repo `data/`)

All of these live in **`data/randamp_gaussian_sqrtd_xdim5/`** (full paths: `<repo>/data/randamp_gaussian_sqrtd_xdim5/...`).

**Low-dim (5D) NPZ**

- `randamp_gaussian_sqrtd_xdim5.npz`

**PR-embedded (30D) NPZ** (Fisher / score on high-dimensional $x$)

- `randamp_gaussian_sqrtd_xdim5_pr30d.npz`  
- Sidecar: `randamp_gaussian_sqrtd_xdim5_pr30d.projection_meta.json`

**Figures** (same directory)

- `joint_scatter_and_tuning_curve.png` / `.svg` — `bin/make_dataset.py` (tuning curves + PCA scatter of native $x$)
- `pr_projection_summary.png` / `.svg` — `bin/project_dataset_pr_autoencoder.py` (tuning, binned embedded mean + scatter, PR loss)

## Regenerating (if needed)

From repo root, `mamba` env `geo_diffusion`, projection on CUDA per `AGENTS.md`:

```bash
mamba run -n geo_diffusion python bin/make_dataset.py \
  --dataset-family randamp_gaussian_sqrtd \
  --x-dim 5 \
  --n-total 10000 \
  --output-npz data/randamp_gaussian_sqrtd_xdim5/randamp_gaussian_sqrtd_xdim5.npz

mamba run -n geo_diffusion python bin/project_dataset_pr_autoencoder.py \
  --input-npz data/randamp_gaussian_sqrtd_xdim5/randamp_gaussian_sqrtd_xdim5.npz \
  --output-npz data/randamp_gaussian_sqrtd_xdim5/randamp_gaussian_sqrtd_xdim5_pr30d.npz \
  --h-dim 30 \
  --device cuda
```

Do **not** pass **`--allow-non-randamp-sqrtd`** here: the projector’s default check matches **`randamp_gaussian_sqrtd`**.

## Agent behavior

When the user says **linearbench** (or similar), resolve paths and semantics to the above; do not substitute a different `--dataset-family`, **`--obs-noise-scale`**, **`--n-total`**, or dimensions unless they explicitly change the alias.
