# Flow matching with analytical diagonal Gaussian velocity prior: 2D banana toy (tests)

## Question / context

We asked whether adding an **analytical diagonal Gaussian velocity prior** on top of standard flow matching improves learning on a **small**, **noisy** synthetic target. The distributional setup and evaluation plan are spelled out in [toy_experiment_plan_fm_gaussian_velocity_prior.md](toy_experiment_plan_fm_gaussian_velocity_prior.md). The loss and prior field are derived in [gaussian_diag_analytical_velocity_prior_affine_flow_matching.md](gaussian_diag_analytical_velocity_prior_affine_flow_matching.md).

**Takeaway (this run):** With $\lambda_0=0.1$ versus the baseline $\lambda_0=0$, the regularized model achieves **lower RBF MMD$^2$** and **lower sliced Wasserstein** distance to held-out data at both training sizes we tried, and **better agreement** with the banana on several shape statistics (e.g. banana-coefficient error, mean error, correlation error). The **mean test log-probability** under the learned model is **higher (better) at $N=256$** with the prior; at **$N=64$** it is slightly **lower** with the prior while sample-quality metrics still improve—so the headline “better” is clearest for **distributional / geometric** measures, with **mean test NLL** mixed at the smallest $N$.

---

## Method (pointers)

- **Objective:** $\mathcal{L}=\mathcal{L}_{\mathrm{FM}}+\lambda_0\,\mathcal{L}_{\mathrm{prior}}$, with the **prior term** built from the **analytical** Gaussian-prior velocity (not from noisy target velocities). See the derivations in [gaussian_diag_analytical_velocity_prior_affine_flow_matching.md](gaussian_diag_analytical_velocity_prior_affine_flow_matching.md), §1–2 and the prior term definition.
- **Data:** 2D **correlated banana** with $\rho=0.7$, $\beta=0.3$ as in the plan note [toy_experiment_plan_fm_gaussian_velocity_prior.md](toy_experiment_plan_fm_gaussian_velocity_prior.md), §2.1.
- **Implementation:** Standalone script `tests/fm_gaussian_velocity_prior.py` (CondOT + `AffineProbPath` from the `flow_matching` stack, MLP velocity, training and evaluation in one file).

---

## Reproduction (commands and scripts)

From the repo root, in the `geo_diffusion` environment:

```bash
mamba run -n geo_diffusion python tests/fm_gaussian_velocity_prior.py --device cuda
```

Defaults in the current script (if unchanged) include: seeds `0`, train sizes `64` and `256`, $\lambda_0\in\{0,0.1\}$, `train_steps=5000`, `test_size=2000`, `n_gen=1024`, `ode_steps=100`, `rho=0.7`, `beta=0.3`.

**Outputs (under `DATAROOT`, visible under the clone as `./data/…`):**

- Aggregated table: `/grad/zeyuan/score-matching-fisher/data/tests/fm_gaussian_velocity_prior/metrics.csv`
- Per-run JSON and sample panels: `/grad/zeyuan/score-matching-fisher/data/tests/fm_gaussian_velocity_prior/`
- Run log (if you redirect): `/grad/zeyuan/score-matching-fisher/data/tests/fm_gaussian_velocity_prior_run.log`

If `mamba run`’s stdio is piped and your log file stays empty, run with the environment’s `python` binary directly (same as `mamba run -n geo_diffusion which python`).

---

## Results

Table below is from **`metrics.csv`** after a completed run: seed **0**, **$\beta=0.3$**, **$\rho=0.7$**, training steps **5000** (one row per $(N,\lambda_0)$).

| train $N$ | $\lambda_0$ | MMD$^2$ (↓) | Sliced W (↓) | mean `test_logp` (↑) | $\Delta$NLL to truth (↓) | banana coef. error (↓) |
|----------|-------------|------------|--------------|----------------------|-------------------------|------------------------|
| 64 | 0.0   | 0.0183 | 0.162 | −2.751 | 0.232 | 0.0478 |
| 64 | 0.1   | 0.00626 | 0.114 | −2.812 | 0.293 | 0.00735 |
| 256 | 0.0  | 0.0122 | 0.131 | −2.624 | 0.105 | 0.0902 |
| 256 | 0.1  | 0.00595 | 0.109 | −2.570 | 0.0505 | 0.0145 |

**Observations:** MMD$^2$ and sliced Wasserstein improve with the prior at both $N$. At $N=256$, mean test log-probability and $\Delta$NLL (to the true mean test log-prob) also improve. At $N=64$, the prior strongly improves MMD and structure errors, but mean `test_logp` is a bit **worse** than the baseline—worth revisiting with more seeds or different $\lambda_0$ if the primary metric is single-seed mean test NLL at very small $N$.

**Conclusion (cautious):** For this single-seed run, the prior is **unambiguously helpful** for **sample–data discrepancy** and **geometric** diagnostics; for **calibrated mean test log-likelihood**, it is **helpful at $N=256$** and **not** on that scalar at $N=64$.

---

## Figure

2D scatter comparison for **$N=64$**, **seed 0**: held-out test points, baseline ($\lambda_0=0$), and regularized ($\lambda_0=0.1$), as produced by the script and copied into this note’s figure folder.

![Test vs. baseline vs. prior (N=64, seed 0): generated samples follow the banana more closely with $\lambda_0=0.1$ (lower MMD$^2$ in `metrics.csv`).](figs/2026-04-25-fm-gaussian-velocity-prior-banana/samples_n_64_seed_0.png)

The panel is the script output `samples_n_64_seed_0.png` (same data as under `data/tests/fm_gaussian_velocity_prior/` in the clone).

---

## Artifacts (absolute)

- **Metrics:** `/grad/zeyuan/score-matching-fisher/data/tests/fm_gaussian_velocity_prior/metrics.csv`
- **Summary (full JSON array):** `/grad/zeyuan/score-matching-fisher/data/tests/fm_gaussian_velocity_prior/summary.json`
- **Figure in note:** `/grad/zeyuan/score-matching-fisher/journal/notes/figs/2026-04-25-fm-gaussian-velocity-prior-banana/samples_n_64_seed_0.png`

---

## Method references (repo)

- [gaussian_diag_analytical_velocity_prior_affine_flow_matching.md](gaussian_diag_analytical_velocity_prior_affine_flow_matching.md) — full objective and analytical prior velocity.
- [toy_experiment_plan_fm_gaussian_velocity_prior.md](toy_experiment_plan_fm_gaussian_velocity_prior.md) — experimental questions, banana model, and intended comparisons.
