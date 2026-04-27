# H-decoding convergence on 10D `cosine_gaussian_sqrtd` with `theta_flow_gaussian_scaffold` (`n=200`, `k=3`)

## Question / context

We wanted a single-point check of the revised posterior-source theta-flow method on a harder observation space: **10D** `cosine_gaussian_sqrtd` at **`n=200`**. The implementation now uses a **10-bin posterior** over theta, then fits a **k-component Gaussian mixture** to that posterior for each conditioning $x$. The branch assignment for a real target $\theta_i$ is chosen by maximum responsibility under that mixture, and the selected branch provides the source for flow matching.

This run asks whether that compact posterior-mixture scaffold still supports the H-decoding pipeline on a 10D cosine-sqrtd problem without changing the learned prior theta-flow.

## Method

### Scaffolded posterior source

The method is implemented in:

- `fisher/theta_gaussian_scaffold.py`
- `fisher/trainers.py`
- `fisher/h_matrix.py`
- `fisher/shared_fisher_est.py`
- `bin/study_h_decoding_convergence.py`

The construction is:

1. Fit a binned diagonal Gaussian estimate $p_g(x\mid\theta)$ on the training split.
2. Compute the posterior on the **10 theta bins**:
   $$
   q_0(b\mid x) \propto p_g(x\mid b),
   $$
   with a uniform prior over bins.
3. Fit a **1D Gaussian mixture** with `k=3` components to the 10-bin posterior mass using weighted EM.
4. For each target $\theta_i$, select the branch $r$ with maximum responsibility under the fitted mixture.
5. Sample the flow-matching source $\theta_0$ from that branch’s Gaussian component, truncated to the theta range.
6. During H-matrix likelihood evaluation, use the same mixture as the posterior-flow base density and subtract the learned prior flow log-density as before.

This run used the following scaffold settings:

- `--theta-gaussian-scaffold-bin-n-bins 10`
- `--theta-gaussian-scaffold-n-components 3`
- `--theta-gaussian-scaffold-em-steps 20`
- `--theta-gaussian-scaffold-grid-size 512` was accepted as a compatibility flag, but the posterior construction itself is now bin-based, not grid-based.
- `--theta-gaussian-scaffold-variance-floor 1e-6`
- `--theta-gaussian-scaffold-min-branch-mass 1e-4`
- `--theta-gaussian-scaffold-source-eps 1e-6`

### Study setup

- **Dataset:** `cosine_gaussian_sqrtd`, `x_dim=10`
- **Dataset NPZ:** `data/dataset_cosine_gaussian_sqrtd_xdim10_trainfrac07/shared_dataset.npz`
- **Study script:** `bin/study_h_decoding_convergence.py`
- **Method:** `--theta-field-method theta_flow_gaussian_scaffold`
- **Sweep sizes:** `--n-list 200`
- **Reference size:** `--n-ref 5000`
- **Device:** `cuda`
- **Intermediate artifacts:** `--keep-intermediate`

The exact run command was:

```bash
mamba run -n geo_diffusion python bin/study_h_decoding_convergence.py \
  --dataset-npz data/dataset_cosine_gaussian_sqrtd_xdim10_trainfrac07/shared_dataset.npz \
  --dataset-family cosine_gaussian_sqrtd \
  --theta-field-method theta_flow_gaussian_scaffold \
  --theta-gaussian-scaffold-n-components 3 \
  --theta-gaussian-scaffold-em-steps 20 \
  --n-list 200 \
  --keep-intermediate \
  --output-dir data/h_decoding_conv_cosine_gaussian_sqrtd_xdim10_theta_flow_gaussian_scaffold_k3_n200_20260427-001500 \
  --device cuda
```

## Results

The run completed cleanly. The saved CSV reports:

| n | `corr_h_binned_vs_gt_mc` | `corr_clf_vs_ref` | `corr_llr_binned_vs_gt_mc` | wall seconds |
|---:|-------------------------:|------------------:|---------------------------:|-------------:|
| 200 | 0.5871 | 0.8084 | 0.4966 | 240.0 |

Observations:

- The **decoding** correlation remains solid at `n=200`.
- The **H** correlation is weaker than in the 3D cosine-sqrtd run, which is consistent with the harder 10D posterior geometry.
- The **LLR** track is the strongest of the three here, but this is still an indirect diagnostic and should not be overread as a guarantee of better posterior geometry.

The fixed-$x$ diagnostic embedded in the combined figure is based on the saved posterior-mixture scaffold and learned prior artifacts from the same run.

## Figure

The combined figure below includes the H-decoding matrices, correlation curves, H-vs-GT scatter, training-loss panel, and the fixed-$x$ posterior diagnostic.

![H-decoding convergence, 10D `cosine_gaussian_sqrtd`, `theta_flow_gaussian_scaffold`, `n=200`](figs/2026-04-27-h-decoding-theta-flow-gaussian-scaffold-10d-cosine-sqrtd-n200/h_decoding_convergence_combined.png)

## Artifacts

- Run directory: `./data/h_decoding_conv_cosine_gaussian_sqrtd_xdim10_theta_flow_gaussian_scaffold_k3_n200_20260427-001500/`
- Results CSV: `./data/h_decoding_conv_cosine_gaussian_sqrtd_xdim10_theta_flow_gaussian_scaffold_k3_n200_20260427-001500/h_decoding_convergence_results.csv`
- Results NPZ: `./data/h_decoding_conv_cosine_gaussian_sqrtd_xdim10_theta_flow_gaussian_scaffold_k3_n200_20260427-001500/h_decoding_convergence_results.npz`
- Combined figure: `./data/h_decoding_conv_cosine_gaussian_sqrtd_xdim10_theta_flow_gaussian_scaffold_k3_n200_20260427-001500/h_decoding_convergence_combined.png`
- Combined SVG: `./data/h_decoding_conv_cosine_gaussian_sqrtd_xdim10_theta_flow_gaussian_scaffold_k3_n200_20260427-001500/h_decoding_convergence_combined.svg`
- Fixed-$x$ diagnostic: `./data/h_decoding_conv_cosine_gaussian_sqrtd_xdim10_theta_flow_gaussian_scaffold_k3_n200_20260427-001500/sweep_runs/n_000200/diagnostics/theta_flow_single_x_posterior_hist.png`
- Scaffold payload: `./data/h_decoding_conv_cosine_gaussian_sqrtd_xdim10_theta_flow_gaussian_scaffold_k3_n200_20260427-001500/sweep_runs/n_000200/theta_gaussian_scaffold.npz`
- H artifact: `./data/h_decoding_conv_cosine_gaussian_sqrtd_xdim10_theta_flow_gaussian_scaffold_k3_n200_20260427-001500/sweep_runs/n_000200/h_matrix_results_theta_cov.npz`
- Training losses: `./data/h_decoding_conv_cosine_gaussian_sqrtd_xdim10_theta_flow_gaussian_scaffold_k3_n200_20260427-001500/sweep_runs/n_000200/score_prior_training_losses.npz`

The H-matrix artifact stores `theta_flow_log_base_matrix`, `theta_flow_log_post_matrix`, and `theta_flow_log_prior_matrix`, which are the saved pieces used for posterior reconstruction and diagnostics.

## Takeaway

The revised scaffold is functioning as intended on the 10D cosine-sqrtd case: the posterior source is now a compact `k=3` Gaussian mixture fit to the 10-bin posterior, and the pipeline runs end-to-end without needing the earlier grid-branch heuristic.

This `n=200` run is not yet strong on the H track, but it is a valid baseline for the revised scaffold and gives a concrete place to compare against future `k` or EM-step sweeps.
