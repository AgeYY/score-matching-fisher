# H-decoding convergence on 3D `cosine_gaussian_sqrtd` with `theta_flow_gaussian_scaffold`

## Question / context

We wanted to test the new posterior-source variant of theta-flow on a small but nontrivial observation space: **3D** `cosine_gaussian_sqrtd`. The new method, `theta_flow_gaussian_scaffold`, keeps the learned prior theta-flow unchanged, but replaces the standard-normal flow-matching source with a source sampled from a **Gaussian posterior scaffold** built from a fitted binned Gaussian estimate of $p_g(x\mid\theta)$.

This run answers a simple question: does the scaffolded source still let the H-decoding pipeline improve with more data, and does it do so without breaking the existing H-matrix / decoding study?

## Method

### What the new method does

The new posterior method is implemented in:

- `fisher/theta_gaussian_scaffold.py`
- `fisher/trainers.py`
- `fisher/h_matrix.py`
- `fisher/shared_fisher_est.py`
- `bin/study_h_decoding_convergence.py`

The pipeline is:

1. Fit a binned diagonal Gaussian estimate $p_g(x\mid\theta)$ on the training split.
2. On a dense scalar $\theta$ grid, form an approximate posterior
   $$
   q_0(\theta\mid x) \propto p_g(x\mid\theta)\,p(\theta),
   $$
   with a uniform prior over the $\theta$ range.
3. Decompose $q_0(\theta\mid x)$ into branches around local modes / valleys.
4. During posterior theta-flow training, sample the source $\theta_0$ from the branch-conditioned scaffold rather than from $\mathcal N(0, I)$.
5. During H-matrix likelihood evaluation, compute the posterior ODE log-density using the scaffold base term $\log q_0(\theta_0\mid x)$, then subtract the learned prior flow log density to form the Bayes-ratio matrix.

The default scaffold hyperparameters used here were the current defaults:

- `--theta-gaussian-scaffold-bin-n-bins 10`
- `--theta-gaussian-scaffold-grid-size 512`
- `--theta-gaussian-scaffold-variance-floor 1e-6`
- `--theta-gaussian-scaffold-min-branch-mass 1e-4`
- `--theta-gaussian-scaffold-source-eps 1e-6`

### Study setup

- **Dataset:** `cosine_gaussian_sqrtd`, `x_dim=3`
- **Dataset NPZ:** `data/datasets/cosine_gaussian_sqrtd_xdim3_n8000_20260426-174214.npz`
- **Study script:** `bin/study_h_decoding_convergence.py`
- **Method:** `--theta-field-method theta_flow_gaussian_scaffold`
- **Sweep sizes:** `--n-list 80,200,400,600`
- **Reference size:** `--n-ref 5000`
- **Intermediate artifacts:** `--keep-intermediate`
- **Device:** `cuda`

The exact run command was:

```bash
mamba run -n geo_diffusion python bin/study_h_decoding_convergence.py \
  --dataset-npz data/datasets/cosine_gaussian_sqrtd_xdim3_n8000_20260426-174214.npz \
  --dataset-family cosine_gaussian_sqrtd \
  --theta-field-method theta_flow_gaussian_scaffold \
  --keep-intermediate \
  --output-dir data/h_decoding_conv_cosine_gaussian_sqrtd_xdim3_theta_flow_gaussian_scaffold_20260426-234000 \
  --device cuda
```

## Results

The run completed for all four sweep sizes. The saved CSV reports:

| n | `corr_h_binned_vs_gt_mc` | `corr_clf_vs_ref` | `corr_llr_binned_vs_gt_mc` | wall seconds |
|---:|-------------------------:|------------------:|---------------------------:|-------------:|
| 80  | 0.6898 | 0.8215 | 0.4280 | 93.4 |
| 200 | 0.7785 | 0.8753 | 0.0465 | 295.0 |
| 400 | 0.8894 | 0.9127 | 0.0256 | 478.3 |
| 600 | 0.9175 | 0.9447 | 0.0630 | 454.9 |

The main signal is encouraging:

- the **H track** improves steadily with $n$ and reaches `corr_h_binned_vs_gt_mc ≈ 0.917` at $n=600$,
- the **decoding track** is also strong at the top end, `corr_clf_vs_ref ≈ 0.945`,
- the **LLR track** stays weak, which is not surprising because the new method is designed to improve the posterior-source geometry for theta-flow, not to directly optimize the x-space mean LLR alignment.

The fixed-$x$ posterior diagnostic embedded in the combined figure also looks reasonable for this run: it is being reconstructed from the just-trained posterior / learned prior artifacts, not from a fallback standard-normal prior.

## Figure

The combined figure below includes the H-decoding matrices, correlation curves, H-vs-GT scatter, training-loss panel, and the fixed-$x$ posterior diagnostic.

![H-decoding convergence, 3D `cosine_gaussian_sqrtd`, `theta_flow_gaussian_scaffold`](figs/2026-04-27-h-decoding-theta-flow-gaussian-scaffold-3d-cosine-sqrtd/h_decoding_convergence_combined.png)

## Artifacts

- Run directory: `./data/h_decoding_conv_cosine_gaussian_sqrtd_xdim3_theta_flow_gaussian_scaffold_20260426-234000/`
- Results CSV: `./data/h_decoding_conv_cosine_gaussian_sqrtd_xdim3_theta_flow_gaussian_scaffold_20260426-234000/h_decoding_convergence_results.csv`
- Results NPZ: `./data/h_decoding_conv_cosine_gaussian_sqrtd_xdim3_theta_flow_gaussian_scaffold_20260426-234000/h_decoding_convergence_results.npz`
- Combined figure: `./data/h_decoding_conv_cosine_gaussian_sqrtd_xdim3_theta_flow_gaussian_scaffold_20260426-234000/h_decoding_convergence_combined.png`
- Combined SVG: `./data/h_decoding_conv_cosine_gaussian_sqrtd_xdim3_theta_flow_gaussian_scaffold_20260426-234000/h_decoding_convergence_combined.svg`
- Sweep artifacts: `./data/h_decoding_conv_cosine_gaussian_sqrtd_xdim3_theta_flow_gaussian_scaffold_20260426-234000/sweep_runs/n_000080/` … `n_000600/`
- Per-run scaffold payload: `./data/h_decoding_conv_cosine_gaussian_sqrtd_xdim3_theta_flow_gaussian_scaffold_20260426-234000/sweep_runs/n_000600/theta_gaussian_scaffold.npz`
- Per-run H artifact: `./data/h_decoding_conv_cosine_gaussian_sqrtd_xdim3_theta_flow_gaussian_scaffold_20260426-234000/sweep_runs/n_000600/h_matrix_results_theta_cov.npz`
- Per-run training losses: `./data/h_decoding_conv_cosine_gaussian_sqrtd_xdim3_theta_flow_gaussian_scaffold_20260426-234000/sweep_runs/n_000600/score_prior_training_losses.npz`

The H-matrix NPZ includes `theta_flow_log_base_matrix`, `theta_flow_log_post_matrix`, and `theta_flow_log_prior_matrix`, which is the part needed for the fixed-$x$ posterior reconstruction.

## Takeaway

This is the first run of the scaffolded theta-flow source on a 3D cosine-sqrtd dataset, and it looks materially better than a random or degenerate source would be: H agreement and decoding both rise with data, while the posterior scaffold remains compatible with the existing H-matrix pipeline.

The main limitation is that the LLR track does not move in step with the H track, so the scaffold should be read as a posterior-geometry improvement for theta-flow, not as a universal fix for every diagnostic.
