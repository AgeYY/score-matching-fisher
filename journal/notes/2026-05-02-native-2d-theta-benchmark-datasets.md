# 2026-05-02 Native 2D-$\theta$ Benchmark Datasets

## Question

The existing `linearbench` and `cosinebench` aliases use scalar $\theta$ and then PR-autoencoder embedding to make a higher-dimensional observation space. This note records the new native two-coordinate source datasets that keep $\theta=(\theta_1,\theta_2)$ throughout data generation:

- `randamp_gaussian2d_sqrtd`
- `gridcos_gaussian2d_sqrtd_rand_tune_additive`

Both are intended as 2D-$\theta$ analogues of the scalar benchmarks: native $x \in \mathbb{R}^5$, $N=10000$, `train_frac=0.7`, shared bounds $\theta_1,\theta_2 \in [-6,6]$, then PR-autoencoder embedding to $x \in \mathbb{R}^{30}$ on CUDA.

## Method

For `randamp_gaussian2d_sqrtd`, each observation coordinate has a realized center $c_j \in [-6,6]^2$ and amplitude $a_j \sim \mathrm{Uniform}(0.2,2.0)$. The conditional mean is

$$
\mu_j(\theta)=a_j \exp(-0.2\|\theta-c_j\|^2).
$$

For `gridcos_gaussian2d_sqrtd_rand_tune_additive`, each coordinate has amplitude $a_j \sim \mathrm{Uniform}(0.2,2.0)$, orientation offset $\rho_j \sim \mathrm{Uniform}(0,\pi/3)$, three phases $\phi_{j,m} \sim \mathrm{Uniform}(0,2\pi)$, and fixed $\omega_j=1$. With three wavevectors $k_{j,m}$ separated by $\pi/3$,

$$
\mu_j(\theta)=\frac{a_j}{3}\sum_{m=1}^3 \cos(k_{j,m}^{\top}\theta+\phi_{j,m}).
$$

Both families use diagonal Gaussian observation noise with the additive $\sqrt{d}$ law

$$
v_j(\theta)=d\,\sigma_{\mathrm{base},j}^2+\alpha_j|\mu_j(\theta)|+10^{-8},
$$

where $d=x_{\dim}$. The default `randamp_gaussian2d_sqrtd` recipe uses $\sigma_{\mathrm{base}}=0.2/\sqrt{2}$ and activity amplitudes `0.70/0.60`. The grid-cosine benchmark command applies `--obs-noise-scale 0.5` and `--cov-theta-amp-scale 2`, giving effective $\sigma_{\mathrm{base}}=0.25$ and activity amplitudes `1.4/1.2`.

The NPZ metadata stores `theta_dim=2`, `theta_low_vec=[-6,-6]`, `theta_high_vec=[6,6]`, and the realized random parameters needed to reconstruct the exact generative model.

## Implementation

The core implementation lives in:

- `fisher/data.py`: `ToyConditionalGaussianRandamp2DSqrtdDataset` and `ToyConditionalGaussianGridcos2DSqrtdDataset`
- `fisher/dataset_family_recipes.py`: fixed recipes for both family tokens
- `fisher/shared_dataset_io.py`: metadata keys for 2D bounds and realized random parameters
- `fisher/shared_fisher_est.py`: metadata reconstruction via `build_dataset_from_meta`
- `fisher/dataset_visualization.py`: 2D heatmap/PCA diagnostics
- `bin/make_dataset.py`: native dataset generation and diagnostic figures
- `bin/project_dataset_pr_autoencoder.py`: PR embedding and 2D-aware projection summary

Focused tests were added to `tests/test_gaussian_tuning_curve.py` for shape checks, positive variances, deterministic parameter round trip, and finite-difference checks of the 2D tuning gradients.

## Reproduction

Run from `/nfshome/zeyuan/score-matching-fisher`.

### Linearbench2d source dataset

```bash
mamba run -n geo_diffusion python bin/make_dataset.py \
  --dataset-family randamp_gaussian2d_sqrtd \
  --x-dim 5 \
  --n-total 10000 \
  --output-npz data/randamp_gaussian2d_sqrtd_xdim5/randamp_gaussian2d_sqrtd_xdim5.npz
```

### Linearbench2d PR 30D embedding

```bash
mamba run -n geo_diffusion python bin/project_dataset_pr_autoencoder.py \
  --input-npz data/randamp_gaussian2d_sqrtd_xdim5/randamp_gaussian2d_sqrtd_xdim5.npz \
  --output-npz data/randamp_gaussian2d_sqrtd_xdim5/randamp_gaussian2d_sqrtd_xdim5_pr30d.npz \
  --h-dim 30 \
  --allow-non-randamp-sqrtd \
  --device cuda
```

### Cosinebench2d source dataset

```bash
mamba run -n geo_diffusion python bin/make_dataset.py \
  --dataset-family gridcos_gaussian2d_sqrtd_rand_tune_additive \
  --x-dim 5 \
  --obs-noise-scale 0.5 \
  --cov-theta-amp-scale 2 \
  --n-total 10000 \
  --output-npz data/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x.npz
```

### Cosinebench2d PR 30D embedding

```bash
mamba run -n geo_diffusion python bin/project_dataset_pr_autoencoder.py \
  --input-npz data/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x.npz \
  --output-npz data/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x_pr30d.npz \
  --h-dim 30 \
  --allow-non-randamp-sqrtd \
  --device cuda
```

## Figures

The figures below are smoke-run diagnostics with `n_total=40` and a short CPU PR embedding for visual verification. They are not the full 10k CUDA benchmark artifacts.

![Native `randamp_gaussian2d_sqrtd` smoke diagnostic](figs/2026-05-02-2d-theta-native-datasets/randamp2d_native_smoke.svg)

The native Gaussian-bump diagnostic shows four small multiples of $\mu_j(\theta_1,\theta_2)$ and PCA projections of sampled $x$ colored by each coordinate of $\theta$.

![Native `gridcos_gaussian2d_sqrtd_rand_tune_additive` smoke diagnostic](figs/2026-05-02-2d-theta-native-datasets/gridcos2d_native_smoke.svg)

The grid-cosine diagnostic has periodic two-coordinate structure in the tuning heatmaps, with the PCA scatter colored separately by $\theta_1$ and $\theta_2$.

![PR embedding smoke diagnostic for `randamp_gaussian2d_sqrtd`](figs/2026-05-02-2d-theta-native-datasets/randamp2d_pr_smoke.svg)

The PR summary replaces scalar-theta binning with a 2D empirical grid over non-empty bins, then shows PCA scatter of the embedded $x$ colored by each coordinate and the short PR training loss.

## Artifacts

Canonical full-run output paths:

- `/nfshome/zeyuan/score-matching-fisher/data/randamp_gaussian2d_sqrtd_xdim5/randamp_gaussian2d_sqrtd_xdim5.npz`
- `/nfshome/zeyuan/score-matching-fisher/data/randamp_gaussian2d_sqrtd_xdim5/randamp_gaussian2d_sqrtd_xdim5_pr30d.npz`
- `/nfshome/zeyuan/score-matching-fisher/data/randamp_gaussian2d_sqrtd_xdim5/joint_scatter_and_tuning_curve.png`
- `/nfshome/zeyuan/score-matching-fisher/data/randamp_gaussian2d_sqrtd_xdim5/pr_projection_summary.png`
- `/nfshome/zeyuan/score-matching-fisher/data/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x.npz`
- `/nfshome/zeyuan/score-matching-fisher/data/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x_pr30d.npz`
- `/nfshome/zeyuan/score-matching-fisher/data/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/joint_scatter_and_tuning_curve.png`
- `/nfshome/zeyuan/score-matching-fisher/data/gridcos_gaussian2d_sqrtd_rand_tune_additive_xdim5_noise2x_alpha2x/pr_projection_summary.png`

Journal figure copies from the smoke checks:

- `/nfshome/zeyuan/score-matching-fisher/journal/notes/figs/2026-05-02-2d-theta-native-datasets/randamp2d_native_smoke.svg`
- `/nfshome/zeyuan/score-matching-fisher/journal/notes/figs/2026-05-02-2d-theta-native-datasets/gridcos2d_native_smoke.svg`
- `/nfshome/zeyuan/score-matching-fisher/journal/notes/figs/2026-05-02-2d-theta-native-datasets/randamp2d_pr_smoke.svg`

## Verification

The implementation was checked with:

```bash
mamba run -n geo_diffusion python -m pytest tests/test_gaussian_tuning_curve.py tests/test_project_dataset_pr_autoencoder.py -q
```

This passed with `33 passed`. Additional smoke checks generated both native 2D families and one CPU PR projection, confirming that NPZ writing and SVG/PNG diagnostics work.

## Takeaway

The new native 2D-$\theta$ families provide benchmark sources where both coordinates influence $x$ before PR embedding. They preserve the scalar benchmark conventions for $x_{\dim}=5$, $N=10000$, train/validation split, and PR 30D embedding, while recording enough metadata to reconstruct the exact random tuning functions later.
