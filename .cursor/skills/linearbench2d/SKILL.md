# linearbench2d

Use this skill to generate the native 2D-$\theta$ linear benchmark and its PR-autoencoder 30D embedding.

Run from `/nfshome/zeyuan/score-matching-fisher`.

```bash
mamba run -n geo_diffusion python bin/make_dataset.py \
  --dataset-family randamp_gaussian2d_sqrtd \
  --x-dim 5 \
  --n-total 10000 \
  --output-npz data/randamp_gaussian2d_sqrtd_xdim5/randamp_gaussian2d_sqrtd_xdim5.npz
```

Native outputs:

- `/nfshome/zeyuan/score-matching-fisher/data/randamp_gaussian2d_sqrtd_xdim5/randamp_gaussian2d_sqrtd_xdim5.npz`
- `/nfshome/zeyuan/score-matching-fisher/data/randamp_gaussian2d_sqrtd_xdim5/joint_scatter_and_tuning_curve.png`
- `/nfshome/zeyuan/score-matching-fisher/data/randamp_gaussian2d_sqrtd_xdim5/joint_scatter_and_tuning_curve.svg`

```bash
mamba run -n geo_diffusion python bin/project_dataset_pr_autoencoder.py \
  --input-npz data/randamp_gaussian2d_sqrtd_xdim5/randamp_gaussian2d_sqrtd_xdim5.npz \
  --output-npz data/randamp_gaussian2d_sqrtd_xdim5/randamp_gaussian2d_sqrtd_xdim5_pr30d.npz \
  --h-dim 30 \
  --allow-non-randamp-sqrtd \
  --device cuda
```

PR outputs:

- `/nfshome/zeyuan/score-matching-fisher/data/randamp_gaussian2d_sqrtd_xdim5/randamp_gaussian2d_sqrtd_xdim5_pr30d.npz`
- `/nfshome/zeyuan/score-matching-fisher/data/randamp_gaussian2d_sqrtd_xdim5/pr_projection_summary.png`
- `/nfshome/zeyuan/score-matching-fisher/data/randamp_gaussian2d_sqrtd_xdim5/pr_projection_summary.svg`
