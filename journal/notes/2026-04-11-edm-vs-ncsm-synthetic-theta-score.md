# 2026-04-11 EDM vs NCSM (synthetic conditional theta-score benchmark)

## Goal

Run a controlled side-by-side benchmark between:

- baseline continuous NCSM score training, and
- new EDM denoiser training (`score_train_objective=edm` path),

using the same synthetic dataset, seed, model capacity, optimizer, and training length.

## Setup

- Repo: `/grad/zeyuan/score-matching-fisher`
- Environment: `mamba run -n geo_diffusion`
- Device: `cuda`
- Seed: `11`

Synthetic data:

- latent: `theta ~ N(0, 1)`
- observation:
  - `x1 = theta + eps`, `eps ~ N(0, 0.35^2)`
  - `x2 ~ N(0, 1)` (nuisance, independent of `theta`)
- sizes:
  - train: `4096`
  - val: `512`
  - eval: `2048`

Model/training (both runs):

- backbone: `ConditionalScore1D(x_dim=2, hidden_dim=64, depth=3)`
- optimizer: AdamW, `lr=1e-3`, `weight_decay=1e-4`
- scheduler: cosine, warmup `0.05`
- `epochs=80`, `batch_size=256`, `max_grad_norm=1.0`

Objective-specific settings:

- **NCSM**: `sigma_min=0.03`, `sigma_max=0.35`, Huber loss
- **EDM**: `p_mean=-1.2`, `p_std=1.2`, `sigma_data=0.5`, MSE loss

Evaluation:

- Compare predicted score vs analytic posterior score at `sigma_eval=0.05`.

## Results

Metrics (`score_pred` vs analytic score, eval set):

- **NCSM**
  - MSE: `0.13150805234909058`
  - MAE: `0.2920549213886261`
  - Corr: `0.9937930994037671`
- **EDM**
  - MSE: `13.936158180236816`
  - MAE: `2.9058949947357178`
  - Corr: `0.9711107953292909`

Observed in this setup:

- EDM converges stably (no non-finite events) but is clearly underperforming NCSM on absolute score accuracy.
- Correlation remains high, so EDM learns a roughly aligned field but scale/calibration is poor under this initial hyperparameter set.
- This matches the expected next tuning axis: EDM needs sigma-scale / standardization tuning before replacing NCSM defaults.

## Artifacts

- Summary JSON:
  `/grad/zeyuan/score-matching-fisher/journal/notes/figs/2026-04-11-edm-vs-ncsm-synthetic/edm_vs_ncsm_summary.json`
- Metrics + trajectories NPZ:
  `/grad/zeyuan/score-matching-fisher/journal/notes/figs/2026-04-11-edm-vs-ncsm-synthetic/edm_vs_ncsm_metrics.npz`
- Loss curves:
  `/grad/zeyuan/score-matching-fisher/journal/notes/figs/2026-04-11-edm-vs-ncsm-synthetic/edm_vs_ncsm_loss_curves.png`
- Predicted-vs-true score scatter:
  `/grad/zeyuan/score-matching-fisher/journal/notes/figs/2026-04-11-edm-vs-ncsm-synthetic/edm_vs_ncsm_score_scatter.png`

## Next actions

1. Standardize `theta` before EDM training and keep `sigma_data=0.5`.
2. Sweep `p_mean` / `p_std` for this scalar setup.
3. Evaluate `sigma_eval` sensitivity for score recovery from denoiser.
