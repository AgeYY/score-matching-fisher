# Journal index

Collected markdown notes in `journal/notes/`: long-form chapters (moved from the former `note/` folder) and dated experiment logs.

## Chapters

1. **[Ideas](notes/idea.md)** — Motivation for estimating the $\theta$-score (posterior score) and conditional denoising score matching, rather than modeling $p(x\mid\theta)$ directly.

2. **[Score matching and Fisher information (toy study)](notes/score_matching_fisher_note.md)** — Gaussian toy setup and comparison of score-matching, decoder-based, and analytic Fisher curves.

3. **[Non-Gaussian Fisher estimation](notes/non_gaussian_fisher_note.md)** — Mixture-model toy, continuous noise-conditional score matching vs decoder.

4. **[Gaussian high-noise Fisher (x-dim 10)](notes/gaussian_high_noise_xdim10_sigma_min_direct_note.md)** — Larger Gaussian experiment: score-based vs decoder Fisher at $\sigma_{\min}$, reproducibility commands, and results.

5. **[Executable scripts (`bin/`)](notes/exec_code.md)** — What `visualize_dataset`, `fisher_make_dataset`, and `fisher_estimate_from_dataset` do and how to run them from the repo root.

## 2026-04

- [2026-04-11 DSM vs flow on H-decoding convergence: `gaussian_randamp_sqrtd` in 2D vs 50D (fresh runs)](notes/2026-04-11-dsm-vs-flow-hdecoding-randamp-sqrtd-2d-50d.md)
- [2026-04-11 H-decoding convergence (Exp. 1): Gaussian tuning — 2D, 10D, and 50D with $\sqrt{d}$-scaled observation noise (`gaussian_sqrtd`)](notes/2026-04-11-h-decoding-convergence-gaussian-tuning-exp1.md)
- [2026-04-10 H-decoding convergence: circular `cos_sin_piecewise_noise` (weak binned-H agreement)](notes/2026-04-10-h-decoding-convergence-cos-sin-piecewise.md)
- [2026-04-09 H-decoding convergence: linear piecewise dataset, DSM $\sigma_{\min}$ at 5%](notes/2026-04-09-h-decoding-convergence-linear-piecewise-minalpha05.md)
- [2026-04-05 Fisher estimation: Gaussian $x \in \mathbb{R}^2$, $N=4000$](notes/2026-04-05-fisher-gaussian-xdim2-n4000.md)
- [2026-04-05 Fisher at high dimension: Gaussian $x \in \mathbb{R}^{100}$, $N=4000$ (poor fit)](notes/2026-04-05-fisher-gaussian-xdim100-n4000.md)
- [2026-04-06 Score distance + MDS: Gaussian–von Mises (2D vs 10D, noise sweep)](notes/2026-04-06-score-distance-mds-comparison.md)
