# cosinebench2d

Use this skill to generate the native 2D-$\theta$ grid-cosine benchmark and its PR-autoencoder 30D embedding.

Run from `/nfshome/zeyuan/score-matching-fisher`.

```bash
mamba run -n geo_diffusion python bin/make_dataset.py \
  --dataset-family gridcos_gaussian2d_sqrtd_rand_tune_additive \
  --x-dim 5 \
  --obs-noise-scale 0.5 \
  --cov-theta-amp-scale 2 \
  --n-total 10000 \
  --output-npz data/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x.npz
```

Native outputs:

- `/nfshome/zeyuan/score-matching-fisher/data/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x.npz`
- `/nfshome/zeyuan/score-matching-fisher/data/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/joint_scatter_and_tuning_curve.png`
- `/nfshome/zeyuan/score-matching-fisher/data/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/joint_scatter_and_tuning_curve.svg`

```bash
mamba run -n geo_diffusion python bin/project_dataset_pr_autoencoder.py \
  --input-npz data/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x.npz \
  --output-npz data/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x_pr30d.npz \
  --h-dim 30 \
  --allow-non-randamp-sqrtd \
  --device cuda
```

PR outputs:

- `/nfshome/zeyuan/score-matching-fisher/data/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x_pr30d.npz`
- `/nfshome/zeyuan/score-matching-fisher/data/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/pr_projection_summary.png`
- `/nfshome/zeyuan/score-matching-fisher/data/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/pr_projection_summary.svg`
