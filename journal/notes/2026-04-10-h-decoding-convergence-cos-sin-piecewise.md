# 2026-04-10 H-decoding convergence: circular (`cos_sin_piecewise_noise`) dataset — weak agreement with reference

This note records the same **H-matrix decoding convergence** protocol as the linear piecewise run ([2026-04-09 note](2026-04-09-h-decoding-convergence-linear-piecewise-minalpha05.md)), but on the **circular** synthetic family `cos_sin_piecewise_noise`: mean $(\cos\theta,\sin\theta)$ with **isotropic** observation noise. Here the **default dataset settings** use uniform $\theta\in[-6,6]$ and **constant** observation std (`--sigma-piecewise-low` = `--sigma-piecewise-high` = $0.1$), so the conditional is single-mode and the geometry is a noisy ring; there is no cross-theta noise contrast.

**Bottom line:** Relative to the large-$n$ reference, **binned H** and **Hellinger LB** (which tracks binned H by construction) show **poor** off-diagonal correlation at small and moderate nested sizes $n$, and only approach the reference toward the **largest** sweep size. This is much weaker than on the **linear piecewise** dataset at the same $n$ (see comparison below). Pairwise decoding and Bayes-opt (C) correlations are higher throughout, so the failure mode is concentrated in the **H-derived** rows of the study, not in every metric.

---

## 1. Data: `cos_sin_piecewise_noise`

- **Family:** `cos_sin_piecewise_noise` (`ToyCosSinPiecewiseNoiseDataset` in `fisher/data.py`).
- **Means:** $(\cos\theta,\sin\theta)$.
- **Noise:** scalar std $\sigma(\theta)$ per axis; with defaults above, $\sigma$ is **constant** $0.1$ on $[-6,6]$.
- **Size / split:** `--n-total 6000`, `--train-frac 1.0` (full pool in NPZ; convergence script builds nested subsets internally).

**Reproduction (dataset NPZ):**

```bash
mamba run -n geo_diffusion python bin/make_dataset.py \
  --dataset-family cos_sin_piecewise_noise \
  --n-total 6000 \
  --train-frac 1.0 \
  --output-npz /grad/zeyuan/score-matching-fisher/data/cos_sin_piecewise_h_decoding_n6000/shared_fisher_dataset.npz
```

---

## 2. Convergence study (same script / defaults as linear note)

**Script:** `bin/study_h_decoding_convergence.py`  
**Reference:** $n_{\mathrm{ref}}=5000$, **sweep:** $n\in\{80,160,240,320,400\}$, **bins:** 10, permutation seed = dataset meta seed $7$.

```bash
mamba run -n geo_diffusion python bin/study_h_decoding_convergence.py \
  --dataset-npz /grad/zeyuan/score-matching-fisher/data/cos_sin_piecewise_h_decoding_n6000/shared_fisher_dataset.npz \
  --output-dir /grad/zeyuan/score-matching-fisher/data/h_decoding_convergence_cos_sin_piecewise \
  --device cuda
```

(Uses repo defaults: posterior **FiLM**, prior **MLP**, continuous DSM $\sigma$ scaled by $\mathrm{std}(\theta)$ on the fit pool, etc.)

---

## 3. Results (numeric)

From `h_decoding_convergence_results.csv`:

| $n$ | corr binned H | corr pairwise decoding | corr Hellinger LB | corr Bayes (C) |
|----:|---------------:|------------------------:|------------------:|---------------:|
| 80 | 0.639 | 0.807 | 0.639 | 0.925 |
| 160 | 0.748 | 0.809 | 0.748 | 0.942 |
| 240 | 0.858 | 0.850 | 0.858 | 0.978 |
| 320 | 0.903 | 0.935 | 0.903 | 0.987 |
| 400 | 0.929 | 0.956 | 0.929 | 0.980 |

**Contrast (same metric, linear piecewise run at $n=80$):** binned H $\approx 0.97$ vs **0.64** here — the circular setting is **much worse** for H-alignment at small $n$.

---

## 4. Figure

Combined line plot + matrix panel written by the script:

`/grad/zeyuan/score-matching-fisher/data/h_decoding_convergence_cos_sin_piecewise/h_decoding_convergence_combined.png`

---

## 5. Output files

- **Summary:** `data/h_decoding_convergence_cos_sin_piecewise/h_decoding_convergence_summary.txt`
- **Tables:** `h_decoding_convergence_results.{csv,npz}`, `h_decoding_convergence_reference.npz`
- **Figures:** `h_decoding_convergence.{png,svg}`, `h_decoding_matrices_panel.{png,svg}`, `h_decoding_convergence_combined.{png,svg}`
- **Reference run:** `data/h_decoding_convergence_cos_sin_piecewise/reference/`

---

## 6. Short interpretation

On this **circular**, **homoscedastic** ring dataset, **finite-$n$** DSM estimates yield **binned H** matrices that stay **far** from the $n_{\mathrm{ref}}=5000$ reference until $n$ is large, unlike the linear piecewise experiment where binned H already matched the reference at $n=80$. Possible contributing factors include **periodicity / wrapping** of the mean in $\theta$ over a **long** interval (multiple periods on $[-6,6]$), **thin** annulus geometry with small $\sigma$ relative to bin widths, and **score/H** sensitivity that is not fully captured by pairwise decoding or C-based Bayes metrics. Treat this regime as **not good** for H-based convergence diagnostics at modest $n$ unless further tuning or problem-specific analysis is added.
