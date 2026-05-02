---
name: cosinebench
description: >-
  In the score-matching-fisher repo, the user’s shorthand "cosinebench" names a fixed synthetic
  dataset: cosine_gaussian_sqrtd_rand_tune_additive at x_dim 5 with obs_noise_scale 0.5 and
  cov_theta_amp_scale 2 (noise and |mu|-driven variance coupling doubled vs the legacy 0.25 / 1.0
  recipe), n_total 10000, then PR-autoencoder embedded to 30D; canonical artifacts under
  data/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/. Use when the user says cosinebench,
  cosine bench, or that benchmark dataset.
---

# cosinebench

## Meaning

**"cosinebench"** means this exact dataset (not a family name in code—only a user alias):

| Stage | What |
|-------|------|
| Generative family | `cosine_gaussian_sqrtd_rand_tune_additive` (additive $|{\mu}|$ sqrt-$d$ variance law) |
| Noise scaling | **`--obs-noise-scale 0.5`** — doubles the legacy cosinebench baseline (**0.25**): recipe $\sigma_{x1}=\sigma_{x2}=0.5$ → stored **0.25** each after scaling |
| Activity coupling (“alpha”) | **`--cov-theta-amp-scale 2`** — doubles recipe `cov_theta_amp1` / `cov_theta_amp2` (**0.7** / **0.6** → **1.4** / **1.2**); mean activity $\alpha_{\mathrm{mean}}=\tfrac{1}{2}(\texttt{amp1}+\texttt{amp2})\approx 1.3$ vs **0.65** before doubling |
| Native observation dim | 5 (`--x-dim 5`) |
| Embedded observation dim | 30 (PR-autoencoder; `--h-dim 30`) |
| Joint samples | `--n-total 10000` (default `--train-frac 0.7` → 7000 train / 3000 validation) |

## Canonical artifacts (repo `data/`)

All of these live in **`data/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/`** (full paths: `<repo>/data/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/...`).

**Low-dim (5D) NPZ**

- `cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x.npz`

**PR-embedded (30D) NPZ** (Fisher / score on high-dimensional $x$)

- `cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x_pr30d.npz`  
- Sidecar: `cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x_pr30d.projection_meta.json`

**Figures** (same directory)

- `joint_scatter_and_tuning_curve.png` / `.svg` — `bin/make_dataset.py` (tuning curves + PCA scatter of native $x$)
- `pr_projection_summary.png` / `.svg` — `bin/project_dataset_pr_autoencoder.py` (tuning, binned embedded mean + scatter, PR loss)

**Legacy note:** The older half-noise / unit–activity-coupling files remain under `data/cosine_sqrtd_rand_tune_additive_xdim5/` (`--obs-noise-scale 0.25`, default `--cov-theta-amp-scale 1`) if you need historical repro only; they are **not** what “cosinebench” refers to anymore.

## Regenerating (if needed)

From repo root, `mamba` env `geo_diffusion`, projection on CUDA per `AGENTS.md`:

```bash
mamba run -n geo_diffusion python bin/make_dataset.py \
  --dataset-family cosine_gaussian_sqrtd_rand_tune_additive \
  --x-dim 5 \
  --obs-noise-scale 0.5 \
  --cov-theta-amp-scale 2 \
  --n-total 10000 \
  --output-npz data/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x.npz

mamba run -n geo_diffusion python bin/project_dataset_pr_autoencoder.py \
  --input-npz data/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x.npz \
  --output-npz data/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/cosine_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x_pr30d.npz \
  --h-dim 30 \
  --allow-non-randamp-sqrtd \
  --device cuda
```

The projector expects `dataset_family == randamp_gaussian_sqrtd` unless **`--allow-non-randamp-sqrtd`** is set (required for this cosine family).

## Agent behavior

When the user says **cosinebench**, resolve paths and semantics to the above; do not substitute a different `--dataset-family`, **`--obs-noise-scale`**, **`--cov-theta-amp-scale`**, **`--n-total`**, or dimensions unless they explicitly change the alias.
