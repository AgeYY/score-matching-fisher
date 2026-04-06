# Gaussian High-Noise Fisher Comparison (x-dim = 10, n = 20000)

This note documents the exact setup and results for comparing:

1. score-based Fisher estimation, and
2. decoder-based local classification Fisher estimation,

on the same Gaussian toy dataset.

The score-based method in this run uses **direct evaluation at $\sigma_{\min}$** (no $\sigma\to 0$ extrapolation).

## 1. Reproducibility command

```bash
mamba run -n geo_diffusion python bin/fisher_make_dataset.py \
  --dataset-family gaussian \
  --x-dim 10 \
  --sigma-x1 2.40 \
  --sigma-x2 1.76 \
  --n-total 20000 \
  --output-npz data/shared_fisher_dataset_gaussian_xdim10.npz

mamba run -n geo_diffusion python bin/fisher_estimate_from_dataset.py \
  --dataset-npz data/shared_fisher_dataset_gaussian_xdim10.npz \
  --output-dir data/outputs_step6_shared_dataset \
  --device cuda
```

## 2. Dataset definition

We sample

- $\theta \sim \mathrm{Uniform}[\theta_{\mathrm{low}}, \theta_{\mathrm{high}}]$ with $\theta_{\mathrm{low}}=-3$, $\theta_{\mathrm{high}}=3$,
- $x\mid\theta \sim \mathcal N(\mu(\theta), \Sigma(\theta))$, with $x\in\mathbb R^{10}$.
- random seed: `7`

### 2.1 Mean (tuning curve)

For dimension $j\in\{1,\dots,10\}$,
$$
\mu_j(\theta)
= A^{\sin}_j\sin(\omega^{\sin}_j\theta+\phi_j)
+ A^{\cos}_j\cos(\omega^{\cos}_j\theta-0.5\phi_j)
+ b_j\theta + c_j\theta^2,
$$
where coefficients are dimension-dependent and fixed by `ToyConditionalGaussianDataset`.

### 2.2 Covariance

The covariance is diagonal for $x_{\text{dim}}>2$:
$$
\Sigma(\theta)=\mathrm{diag}(s_1^2(\theta),\dots,s_{10}^2(\theta)),
$$
with
$$
s_j(\theta)=\sigma^{\text{base}}_j\left[1+a_{1,j}\sin(f_{1,j}\theta+p_{1,j})+a_{2,j}\cos(f_{2,j}\theta+p_{2,j})\right],
$$
clipped to stay positive.

In this run, baseline noise levels are large:

- `sigma_x1 = 2.40`
- `sigma_x2 = 1.76`

Other covariance-shape parameters (defaults):

- `cov_theta_amp1 = 0.35`, `cov_theta_amp2 = 0.30`, `cov_theta_amp_rho = 0.30`
- `cov_theta_freq1 = 0.90`, `cov_theta_freq2 = 0.75`, `cov_theta_freq_rho = 1.10`
- `cov_theta_phase1 = 0.20`, `cov_theta_phase2 = -0.35`, `cov_theta_phase_rho = 0.40`
- `rho_clip = 0.85`

## 3. Methods

## 3.1 Score-based Fisher method

Train a noise-conditional score network $s_\phi(x,\theta,\sigma)$ by denoising score matching over a geometric noise schedule.

For each sample:

- pick a noise level $\sigma$,
- sample $z\sim\mathcal N(0,I)$,
- set $\tilde x = x + \sigma z$,
- optimize
$$
\mathcal L = \|\sigma s_\phi(\tilde x,\theta,\sigma) + z\|_2^2.
$$

After training, Fisher is estimated by
$$
\widehat I_{\text{score}}(\theta)=\mathbb E\left[s_\phi(x,\theta,\sigma_{\min})^2\mid\theta\right],
$$
using bin-wise averages (minimum bin count = 10).

Important: this run uses **$\sigma_{\min}$ direct evaluation** (no extrapolation).

### Score hyperparameters used

- model: `ConditionalScore1D(hidden_dim=128, depth=3)`
- optimizer LR: `1e-3`
- batch size: `256`
- max epochs: `10000`
- early stopping:
  - patience `5000`
  - min delta `1e-4`
  - smoothing window `20`
  - restore best model: `True`
- score data split for training: train/eval = `14000/6000`
- score fit/val split inside train: `11900/2100`
- score fisher evaluation set: **full dataset** (`n=20000`)

Noise schedule (continuous geometric):

- scale mode: `theta_std`
- $\theta$ std on score-fit set: `1.7320085401758998`
- `alpha_min = 0.01`, `alpha_max = 0.25`
- therefore
  - $\sigma_{\min}=0.017320085401758997$
  - $\sigma_{\max}=0.43300213504397495$
- number of eval levels: `12`

## 3.2 Decoder local-classification Fisher

For each target $\theta_0$, construct two local classes centered at

- $\theta_+ = \theta_0 + \epsilon/2$,
- $\theta_- = \theta_0 - \epsilon/2$,

using samples whose $|\theta-\theta_\pm|\le h$.

Train a local binary decoder $g_{\theta_0}(x)$, then estimate
$$
\widehat I_{\text{decoder}}(\theta_0) = \mathbb E\left[\frac{g_{\theta_0}(x)^2}{\epsilon^2}\right].
$$

### Decoder hyperparameters used

- `decoder_epsilon = 0.12`
- `decoder_bandwidth = 0.10`
- `decoder_epochs = 80`
- `decoder_batch_size = 256`
- `decoder_lr = 1e-3`
- `decoder_hidden_dim = 64`
- `decoder_depth = 2`
- `decoder_min_class_count = 60`
- `decoder_train_cap = 1200`
- `decoder_eval_cap = 1200`

### Shared evaluation settings

- `train_frac = 0.7`
- `n_bins = 35`
- `eval_margin = 0.30`
- `score_min_bin_count = 10`

## 3.3 Ground-truth Fisher

For Gaussian $x\mid\theta \sim \mathcal N(\mu(\theta),\Sigma(\theta))$, the analytic Fisher for scalar $\theta$ is
$$
I(\theta)
= \mu'(\theta)^\top\Sigma(\theta)^{-1}\mu'(\theta)
+ \frac12\,\mathrm{tr}\!\left(\Sigma(\theta)^{-1}\Sigma'(\theta)\Sigma(\theta)^{-1}\Sigma'(\theta)\right).
$$

## 4. Result (this run)

From `metrics_vs_gt_theta_cov.txt`:

- Score vs GT: `valid=35/35`, `rmse=1.912829`, `mae=1.508876`, `corr=0.863989`
- Decoder vs GT: `valid=35/35`, `rmse=35.123873`, `mae=32.352165`, `corr=-0.353771`

So under this high-noise setting, score-based Fisher estimation is much closer to analytic GT than decoder local classification.

## 5. Figures

### Fisher comparison

![Fisher comparison (Gaussian high-noise, x-dim=10, n=20000)](figures/gaussian_high_noise_xdim10_n20000_compare_sigma_min_direct.png)

### Score loss vs epoch

![Score training loss vs epoch (same run)](figures/gaussian_high_noise_xdim10_n20000_score_loss_sigma_min_direct.png)
