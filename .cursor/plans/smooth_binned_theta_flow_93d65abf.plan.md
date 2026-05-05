---
name: Smooth Binned Theta Flow
overview: Add `smooth_binned_theta_flow` as a new row method that replaces hard theta-bin assignment with non-periodic RBF responsibilities, trains a soft-label gating classifier and weighted local theta-flow experts, then evaluates the posterior by log-sum-exp over all experts.
todos:
  - id: extend-helper-soft-bins
    content: Extend binned theta-flow helper utilities with soft RBF responsibilities, soft-label gate training, weighted flow training, and log-sum-exp mixture evaluation.
    status: completed
  - id: wire-smooth-method-cli
    content: Expose `smooth_binned_theta_flow` and related CLI flags in convergence and twofig parsing.
    status: completed
  - id: implement-smooth-estimate
    content: Add the `_estimate_one` branch that trains the soft gate, weighted experts, computes mixture C matrix, and saves compatible artifacts.
    status: completed
  - id: validate-smooth-method
    content: Add focused tests and run unit, compile, and CLI smoke validation.
    status: completed
isProject: false
---

# Smooth Binned Theta-Flow Plan

## Goal

Add a new method token, `smooth_binned_theta_flow`, without changing the existing `binned_theta_flow`. It will keep separate local theta-flow experts, but remove discontinuities by using soft RBF responsibilities and mixture evaluation over all experts.

## Implementation Plan

1. Extend [`/grad/zeyuan/score-matching-fisher/fisher/binned_theta_flow.py`](/grad/zeyuan/score-matching-fisher/fisher/binned_theta_flow.py):
   - Add a `ThetaSoftBinSpec` or extend `ThetaBinSpec` with RBF centers and bandwidth.
   - Add `make_soft_theta_bins(theta, K, mode="uniform")` with default non-periodic uniform centers over the theta range.
   - Add `soft_theta_responsibilities(theta, centers, sigma)` implementing row-normalized non-periodic RBF weights.
   - Add soft-label classifier training using `-(r * log_softmax(logits)).sum(dim=-1).mean()`.
   - Add weighted flow-matching training for each expert, preferably as a new trainer helper that samples batches from the full dataset and weights per-sample FM loss by `r_ik`.

2. Add mixture density evaluation helpers:
   - Reuse `local_flow_log_prob_matrix(...)` to compute `log q_k(theta_j | x_i)` for every expert `k` over every test theta column.
   - Compute `log pi_k(x_i)` once from the gate.
   - Fill the pairwise matrix with:
     `C[i,j] = logsumexp_k(log_pi[i,k] + log_q_k[i,j])`.
   - Feed `C` through the existing `compute_delta_l`, `compute_h_directed`, and `h_sym` path.

3. Wire CLI and validation in [`/grad/zeyuan/score-matching-fisher/bin/study_h_decoding_convergence.py`](/grad/zeyuan/score-matching-fisher/bin/study_h_decoding_convergence.py):
   - Accept aliases: `smooth_binned_theta_flow`, `smooth-binned-theta-flow`, `sbtf`.
   - Add CLI flags:
     - `--smooth-binned-theta-flow-bins`, default `2`.
     - `--smooth-binned-theta-flow-sigma`, default derived from center spacing when `<=0`.
     - `--smooth-binned-theta-flow-center-mode`, default `uniform`, optional `quantile`.
     - Reuse existing `--btf-cls-*` and `--flow-*` knobs where possible.
   - Keep v1 scalar-theta only and `--flow-arch mlp` only, matching current `binned_theta_flow` constraints.

4. Add an `_estimate_one` branch for `smooth_binned_theta_flow`:
   - Build centers and sigma from `theta_all`.
   - Compute responsibilities for train, validation, and all rows.
   - Train the soft-label gate on `(x_train, r_train)`.
   - Train each expert with weighted FM loss using the full train/validation arrays and that expert’s responsibility column.
   - Evaluate all experts for all theta columns and combine with log-sum-exp.
   - Save `h_matrix_results_theta_cov.npz` metadata: centers, sigma, center mode, train/val responsibility mass per expert, gate soft-label losses/calibration-like diagnostics, and mixture log components if not too large.
   - Save `score_prior_training_losses.npz` with gate losses and per-expert weighted flow losses so the existing twofig loss panel remains usable.

5. Wire twofig parsing in [`/grad/zeyuan/score-matching-fisher/bin/study_h_decoding_twofig.py`](/grad/zeyuan/score-matching-fisher/bin/study_h_decoding_twofig.py):
   - Accept the new method token in `_normalize_theta_field_method_local`.
   - Update help text so `--theta-field-rows bin_gaussian,smooth_binned_theta_flow` works.
   - Do not add it to `_FLOW_BASED_METHODS` in v1, so `smooth_binned_theta_flow:film` is rejected explicitly rather than half-supported.

6. Validation:
   - Add helper tests in [`/grad/zeyuan/score-matching-fisher/tests/test_binned_theta_flow.py`](/grad/zeyuan/score-matching-fisher/tests/test_binned_theta_flow.py) for RBF responsibility row sums, non-periodic behavior, sigma derivation, soft-label CE shape, weighted loss finite behavior, and log-sum-exp mixture continuity on a toy log-density grid.
   - Run `mamba run -n geo_diffusion python -m unittest discover -s tests -p 'test_binned_theta_flow.py' -v`.
   - Run `mamba run -n geo_diffusion python -m py_compile fisher/binned_theta_flow.py bin/study_h_decoding_convergence.py bin/study_h_decoding_twofig.py`.
   - Run a lightweight CLI normalization smoke for `--theta-field-method smooth_binned_theta_flow`.

## Key Design Choices

- Keep `binned_theta_flow` unchanged so hard-bin and smooth-bin results remain directly comparable.
- Use non-periodic RBF distances only, as requested.
- Use log-sum-exp over all experts at evaluation; never choose a single bin at test time.
- Do not add explicit smoothness regularization in v1.
- Default `K=2`, mirroring the hard-binned method, with sigma defaulting to roughly one center spacing unless overridden.