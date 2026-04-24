# Journal index

Collected markdown notes in `journal/notes/`: long-form chapters (moved from the former `note/` folder) and dated experiment logs.

## Chapters

1. **[Ideas](notes/idea.md)** — Motivation for estimating the $\theta$-score (posterior score) and conditional denoising score matching, rather than modeling $p(x\mid\theta)$ directly.

2. **[Score matching and Fisher information (toy study)](notes/score_matching_fisher_note.md)** — Gaussian toy setup and comparison of score-matching, decoder-based, and analytic Fisher curves.

3. **[Non-Gaussian Fisher estimation](notes/non_gaussian_fisher_note.md)** — Mixture-model toy, continuous noise-conditional score matching vs decoder.

4. **[Gaussian high-noise Fisher (x-dim 10)](notes/gaussian_high_noise_xdim10_sigma_min_direct_note.md)** — Larger Gaussian experiment: score-based vs decoder Fisher at $\sigma_{\min}$, reproducibility commands, and results.

5. **[Executable scripts (`bin/`)](notes/exec_code.md)** — What `visualize_dataset`, `fisher_make_dataset`, and `fisher_estimate_from_dataset` do and how to run them from the repo root.

## 2026-04

- [2026-04-24 H-decoding convergence on 50D `randamp_gaussian_sqrtd` with `theta_flow` + MLP (default `n_list`), including classifier-init method details and reproducible command](notes/2026-04-24-h-decoding-randamp-sqrtd-50d-theta-flow-mlp-defaultn-classifier-init.md)

- [2026-04-23 x-flow on 10D `cosine_gaussian_sqrtd`: vanilla vs accuracy-MDS theta embedding (n=200 noise=0.5 and n=100 noise=1.0), reproducible commands + matrix figures](notes/2026-04-23-xflow-mds-vs-vanilla-cosine-sqrtd-xdim10.md)

- [2026-04-22 Theta-flow **Fourier + Soft-MoE posterior** on 2D `cosine_gaussian_sqrtd`: H-decoding metrics vs MLP / Soft-MoE; repro + figure](notes/2026-04-22-theta-flow-fourier-softmoe-posterior-2d-h-decoding.md)
- [2026-04-22 Theta-flow **Soft-MoE posterior** vs MLP on 2D `cosine_gaussian_sqrtd` H-decoding (`corr_h` / `corr_llr`); repro commands + figures](notes/2026-04-22-theta-flow-soft-moe-posterior-2d-h-decoding.md)
- [2026-04-22 OT-CFM vs CFM 2D bimodal benchmark (`tests/ot_cfm_torchcfm.py`): TorchCFM matchers, data split, MMD, Markdown + commands](notes/2026-04-22-ot-cfm-bimodal-2d-benchmark.md)
- [2026-04-22 Fixed-$x$ posterior panel: model curve (softmax + KDE) vs “GT” (generative $x\mid\theta$ + uniform prior); methods detail + `study_h_decoding_convergence` pointers](notes/2026-04-22-fixed-x-posterior-model-vs-approx-gt.md)
- [2026-04-21 H-decoding: generative mean LLR vs binned model $\Delta L$ (10D `cosine_gaussian_sqrtd`, $n=200`, repro; Markdown + commands)](notes/2026-04-21-h-decoding-llr-scatter-generative-vs-model-delta-l.md)
- [2026-04-20 PR-autoencoder embedding for `randamp_gaussian_sqrtd`: theta-flow H-decoding convergence with 2D latent mapped to 10D and 50D](notes/2026-04-20-randamp-sqrtd-pr-autoencoder-hdecoding-thetaflow.md)
- [2026-04-18 Rename: `theta_flow` = θ-space ODE Bayes-ratio; `theta_path_integral` = velocity→score + θ-axis integral (hard break)](notes/2026-04-18-theta-flow-rename-and-bayes-ratio.md)
- [2026-04-18 CTSM-v log-likelihood ratio on continuous parametric Gaussian family $p_t(x)=\mathcal{N}(2t-1,1)$: $(x,t)$ dataset, random held-out $(t_i,t_j)$ pairs, and reproducibility with `--ctsm-max-epochs 10000 --ctsm-early-patience 1000`](notes/2026-04-18-ctsm-v-parametric-gaussian-t-log-ratio.md)
- [2026-04-18 CTSM-v log-likelihood ratio between two 1D Gaussians: pair-conditioned FiLM (`a=-1`, `b=+1`), estimated vs GT scatter, and reproducibility command](notes/2026-04-18-ctsm-v-gaussian-two-dist-log-ratio.md)
- [2026-04-16 Marginal leading-$K$ dims for `flow_x_likelihood` H-decoding (10D `cosine_gaussian_sqrtd`): GT marginal, MLP x-flow, cosine scheduler, no two-stage / no Fourier (defaults in cited runs)](notes/2026-04-16-marginal-leading-dims-flow-x-h-decoding.md)
- [2026-04-16 Pair-conditioned CTSM-v (`tests/ctsm_pair_conditioned.py`): discrete $\theta$ grid, conditional GMM variance, $\log p(x|b)-\log p(x|a)$ + reproducibility](notes/2026-04-16-pair-conditioned-ctsm-v-discrete-theta-gmm.md)
- [2026-04-16 CTSM-v toy (`tests/ctsm.py`): `TwoSB` two-sample bridge, bimodal GMM $p$/$q$, $\log q/p$ by trapezoid $t$-integration + figures](notes/2026-04-16-ctsm-v-toy-gaussian-log-ratio.md)
- [2026-04-13 `theta_fourier_mlp`: Fourier + linear $\theta$ features for conditional x-flow (`flow_x_likelihood`) + cosine $\sqrt{d}$ benchmarks; addendum: theta-space **theta_fourier_mlp** for **`theta_path_integral`** (`velocity`→score) vs **`theta_flow`** (ODE log-density)](notes/2026-04-13-flow-x-likelihood-h-matrix-randamp-sqrtd-50d.md)
- [2026-04-13 `flow_x_likelihood`: x-space ODE $\log p(x\mid\theta)$ for H-matrix + `randamp_gaussian_sqrtd` 50D & 2D convergence](notes/2026-04-13-flow-x-likelihood-h-matrix-randamp-sqrtd-50d.md)
- [2026-04-13 Cosine Gaussian noise: base vs $\sqrt{d}$ scaling (`cosine_gaussian` / `cosine_gaussian_sqrtd`) + dataset figure](notes/2026-04-13-cosine-gaussian-noise-sqrtd-vs-base.md)
- [2026-04-13 `theta_flow` (formerly `flow_likelihood`): direct flow ODE log-density for $\theta$ H-matrix (math + algorithm)](notes/2026-04-13-flow-ode-direct-likelihood-theta-h-matrix.md)
- [2026-04-13 H-decoding: weak binned-\(H\) vs GT on periodic cosine means (2D vs 3D `cosine_gaussian`)](notes/2026-04-13-h-decoding-periodic-cosine-weak.md)
- [2026-04-11 EDM vs NCSM: synthetic score fit and 50D H-decoding (EDM underperforms)](notes/2026-04-11-edm-underperforms-ncsm-hdecoding-synthetic.md)
- [2026-04-11 EDM vs NCSM benchmark: synthetic conditional theta-score (Layer A sanity check)](notes/2026-04-11-edm-vs-ncsm-synthetic-theta-score.md)
- [2026-04-11 DSM vs flow on H-decoding convergence: `randamp_gaussian_sqrtd` in 2D vs 50D (fresh runs)](notes/2026-04-11-dsm-vs-flow-hdecoding-randamp-sqrtd-2d-50d.md)
- [2026-04-11 H-decoding convergence (Exp. 1): Gaussian tuning — 2D, 10D, and 50D with $\sqrt{d}$-scaled observation noise (`cosine_gaussian_sqrtd`)](notes/2026-04-11-h-decoding-convergence-gaussian-tuning-exp1.md)
- [2026-04-10 H-decoding convergence: circular `cos_sin_piecewise` (weak binned-H agreement)](notes/2026-04-10-h-decoding-convergence-cos-sin-piecewise.md)
- [2026-04-09 H-decoding convergence: linear piecewise dataset, DSM $\sigma_{\min}$ at 5%](notes/2026-04-09-h-decoding-convergence-linear-piecewise-minalpha05.md)
- [2026-04-05 Fisher estimation: Gaussian $x \in \mathbb{R}^2$, $N=4000$](notes/2026-04-05-fisher-gaussian-xdim2-n4000.md)
- [2026-04-05 Fisher at high dimension: Gaussian $x \in \mathbb{R}^{100}$, $N=4000$ (poor fit)](notes/2026-04-05-fisher-gaussian-xdim100-n4000.md)
- [2026-04-06 Score distance + MDS: Gaussian–von Mises (2D vs 10D, noise sweep)](notes/2026-04-06-score-distance-mds-comparison.md)
