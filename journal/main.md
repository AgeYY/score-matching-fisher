# Journal index

Collected markdown notes in `journal/notes/`: long-form chapters (moved from the former `note/` folder) and dated experiment logs.

## Chapters

1. **[Ideas](notes/idea.md)** — Motivation for estimating the $\theta$-score (posterior score) and conditional denoising score matching, rather than modeling $p(x\mid\theta)$ directly.

2. **[Score matching and Fisher information (toy study)](notes/score_matching_fisher_note.md)** — Gaussian toy setup and comparison of score-matching, decoder-based, and analytic Fisher curves.

3. **[Non-Gaussian Fisher estimation](notes/non_gaussian_fisher_note.md)** — Mixture-model toy, continuous noise-conditional score matching vs decoder.

4. **[Gaussian high-noise Fisher (x-dim 10)](notes/gaussian_high_noise_xdim10_sigma_min_direct_note.md)** — Larger Gaussian experiment: score-based vs decoder Fisher at $\sigma_{\min}$, reproducibility commands, and results.

5. **[Executable scripts (`bin/`)](notes/exec_code.md)** — What `visualize_dataset`, `fisher_make_dataset`, and `fisher_estimate_from_dataset` do and how to run them from the repo root.

## 2026-04

- [2026-04-05 Fisher estimation: Gaussian $x \in \mathbb{R}^2$, $N=1000$](notes/2026-04-05-fisher-gaussian-xdim2-n1000.md)
