# 2026-04-09 H-decoding convergence: linear piecewise dataset, DSM $\sigma_{\min}$ at 5% of $\mathrm{std}(\theta)$

This note documents the **H-matrix decoding convergence** pipeline on a **linear piecewise noise** synthetic dataset, with continuous denoising score matching (DSM) using `**--score-sigma-min-alpha 0.05`** (σ range low end = 5% of $\mathrm{std}(\theta)$ on the score fit pool; default max remains 25%). We compare nested subset sizes $n \in 80,160,240,320,400$ to a **reference** run of size $n_{\mathrm{ref}}=5000$ by **off-diagonal Pearson correlation** of several matrices derived from binned $\theta$ and trained scores. The combined figure below matches the panel layout produced by the study script (line plot + matrix panel).

---

## 1. Data: linear piecewise noise

- **Family:** `linear_piecewise` (`bin/make_dataset.py`).
- **Observation noise:** scalar std $\sigma(\theta)$ per axis, linear in $\theta$ from `**--sigma-piecewise-low 0.1`** at `**--theta-low**` to `**--sigma-piecewise-high 2.0**` at `**--theta-high**` (defaults; endpoints are the **dataset** observation noise, unrelated to DSM score-matching $\sigma$).
- **Means:** first component mean $\propto \theta$ (`--linear-k 1.0`), second component tracks $\theta$ as in the toy `ToyLinearPiecewiseNoiseDataset` implementation.
- **Size / split:** `--n-total 6000`, `--train-frac 1.0` (all samples in train for this NPZ; the convergence script re-splits subsets internally).
2026-04-10 21:41:05.317 [info] Using configured platform linux for remote host xuexin-gpu
2026-04-10 21:41:05.318 [info] Using askpass script: c:\Users\Zeyuacursor\extensions\anysphere.remote-ssh-1.0.48\dist\scripts\launchSSHAskpass.bat with javascript file c:\Users\Zeyuacursor\extensions\anysphere.remote-ssh-1.0.48\dist\scripts\sshAskClient.js. Askpass handle: 60148
2026-04-10 21:41:05.341 [info] Launching SSH server via shell with command: type "C:\Users\Zeyua\AppData\Local\Temp\cursor_remote_install_5b635d56-91e6-4856-89eb-449b537d30c2.sh" | ssh -T -D 60149 xuexin-gpu bash --login -c bash
2026-04-10 21:41:05.341 [info] Establishing SSH connection: type "C:\Users\Zeyua\AppData\Local\Temp\cursor_remote_install_5b635d56-91e6-4856-89eb-449b537d30c2.sh" | ssh -T -D 60149 xuexin-gpu bash --login -c bash
2026-04-10 21:41:05.342 [info] Started installation script. Waiting for it to finish...
2026-04-10 21:41:05.342 [info] Waiting for SSH handshake (timeout: 120s). Install timeout: 30s.
2026-04-10 21:41:05.399 [info] (ssh_tunnel) stderr: C:UsersZeyua/.ssh/config: line 17: Bad configuration option: part
C:UsersZeyua/.ssh/config: terminating, 1 bad configuration options

2026-04-10 21:41:05.400 [info] (ssh_tunnel) stderr: The process tried to write to a nonexistent pipe.

2026-04-10 21:41:05.409 [error] SSH process exited (code 255) before connection was established (after 65ms)
2026-04-10 21:41:05.409 [error] Pre-connection stderr: C:UsersZeyua/.ssh/config: line 17: Bad configuration option: part
C:UsersZeyua/.ssh/config: terminating, 1 bad configuration options
The process tried to write to a nonexistent pipe.

2026-04-10 21:41:05.412 [error] Error installing server: Failed to connect to the remote SSH host. Please check the logs for more details.
2026-04-10 21:41:05.412 [info] Deleting local script C:\Users\Zeyua\AppData\Local\Temp\cursor_remote_install_5b635d56-91e6-4856-89eb-449b537d30c2.sh
2026-04-10 21:41:05.422 [error] Error resolving SSH authority Failed to connect to the remote SSH host. Please check the logs for more details.
**Reproduction (dataset NPZ + joint scatter):**

```bash
mamba run -n geo_diffusion python bin/make_dataset.py \
  --dataset-family linear_piecewise \
  --n-total 6000 \
  --output-npz /path/to/linear_piecewise_h_decoding_n6000/shared_fisher_dataset.npz
```

Saved dataset used here:  
`/grad/zeyuan/score-matching-fisher/data/linear_piecewise_h_decoding_n6000/shared_fisher_dataset.npz`

---

## 2. Models and score matching (DSM)

- **Posterior score:** FiLM architecture (`--score-arch film`, `--score-depth 3`), continuous noise-conditioned score matching (NCSM).
- **Prior score:** MLP (`--prior-score-arch mlp`, `--prior-depth 3`).
- **Noise schedule (training):** `--score-noise-mode continuous` with **log-uniform** $\sigma$ in $[\sigma_{\min}, \sigma_{\max}]$, where  
$\sigma_{\min} = \alpha_{\min}\mathrm{std}(\theta_{\mathrm{fit}})$,  
$\sigma_{\max} = \alpha_{\max}\mathrm{std}(\theta_{\mathrm{fit}})$,  
with `**--score-sigma-scale-mode theta_std`**, `**--score-sigma-min-alpha 0.05**`, `**--score-sigma-max-alpha 0.25**` (this experiment explicitly sets 5% for the low end).

**Reproduction (full convergence study):**

```bash
mamba run -n geo_diffusion python bin/study_h_decoding_convergence.py \
  --dataset-npz /grad/zeyuan/score-matching-fisher/data/linear_piecewise_h_decoding_n6000/shared_fisher_dataset.npz \
  --output-dir /grad/zeyuan/score-matching-fisher/data/h_decoding_convergence_linear_piecewise_minalpha05 \
  --score-sigma-min-alpha 0.05 \
  --device cuda
```

(With current repo defaults, `--score-sigma-min-alpha 0.05` is redundant if the CLI default is already 0.05.)

---

## 3. Metrics and reference construction

**Script:** `bin/study_h_decoding_convergence.py` (uses `visualize_h_matrix_binned` helpers).

**Global permutation:** indices are permuted with seed `meta["seed"] + subset_seed_offset` (here `7 + 0 = 7`). Nested subsets take the first $n$ indices in this order; **reference** uses the first $n_{\mathrm{ref}}=5000$.

**$\theta$ bins:** `num_theta_bins = 10` equal-width bins; **bin edges are fixed** from the reference subset’s $\theta$ range so all $n$ use the same binning.

### 3.1 Symmetric H matrix and binning

Train posterior and prior DSMs, evaluate the **sample H matrix** $H_{ij}$ (symmetric; see `HMatrixEstimator` in-repo), restrict to rows used for the H estimate, then **average entries** within each $(\theta_a,\theta_b)$ bin pair to obtain **binned H** matrices $\bar H_{ab}$ (same for each $n$ and for reference).

### 3.2 Hellinger lower bound from binned $H^2$

Treat binned symmetric entries as $H^2$ and map to a Hellinger accuracy **lower bound** matrix:

$$
A^{\mathrm{H\text{-}LB}}*{ab} = \tfrac{1}{2}\bigl(1 + \mathrm{clip}((\bar H^2)*{ab},0,1)\bigr),\quad a\neq b
$$

(diagonal set to NaN). Implementation: `hellinger_acc_lb_from_binned_h_squared` in `bin/visualize_h_matrix_binned.py`.

### 3.3 Pairwise logistic decoding

For each bin pair $(a,b)$ with enough samples, fit a **logistic regression** on $x$ to classify bin $a$ vs $b$, with stratified train/test split (`clf_test_frac=0.3`, `clf_min_class_count=5`, random state = dataset seed). Record test accuracy in a symmetric matrix (diagonal NaN). Implementation: `pairwise_bin_logistic_accuracy_matrix`.

### 3.4 Bayes-opt accuracy from C-matrix bin means

With `**h_save_intermediates`**, the H pipeline saves the integrated **C matrix** from the score-matching construction. Bin-mean differences induce a pairwise **Bayes-optimal** accuracy matrix (see docstring in `_c_matrix_bayes_opt_accuracy_matrix` in `study_h_decoding_convergence.py`).

### 3.5 Convergence score: off-diagonal correlation

For each metric matrix $M^{(n)}$ at subset size $n$ and reference $M^{(\mathrm{ref})}$:

$$
\rho_{\mathrm{off}}(M^{(n)}, M^{(\mathrm{ref})}) = \mathrm{corr}\bigl( M^{(n)}*{ab}, M^{(\mathrm{ref})}*{ab} \bigr)_{(a,b): a\neq b,\ \text{both finite}}
$$

Implementation: `matrix_corr_offdiag` (Pearson correlation over off-diagonal entries where both matrices are finite).

---

## 4. Results (numeric)

From `h_decoding_convergence_results.csv` in the output directory:


| $n$ | corr binned H | corr pairwise decoding | corr Hellinger LB | corr Bayes (C) |
| --- | ------------- | ---------------------- | ----------------- | -------------- |
| 80  | 0.967         | 0.712                  | 0.967             | 0.882          |
| 160 | 0.944         | 0.884                  | 0.944             | 0.953          |
| 240 | 0.984         | 0.864                  | 0.984             | 0.941          |
| 320 | 0.978         | 0.901                  | 0.978             | 0.968          |
| 400 | 0.982         | 0.930                  | 0.982             | 0.972          |


**Qualitative:** Binned H and Hellinger LB track the reference very closely (correlation $\gtrsim 0.94$). Pairwise decoding converges more slowly from $n=80$ but improves toward $\sim 0.93$ at $n=400$. Bayes-opt (C) correlation is high throughout and ends near $0.97$.

---

## 5. Figure (combined panel A + B)

**Panel A.** Off-diagonal Pearson correlation to the $n_{\mathrm{ref}}=5000$ reference vs nested subset size $n$ for: binned H, pairwise logistic decoding, Hellinger LB from binned $H^2$, and Bayes-opt accuracy from C-matrix bin means. **Panel B.** Four rows (binned H, pairwise decoding, Hellinger LB, Bayes-opt from C) and columns $n=80,\ldots,400$ plus reference column $n_{\mathrm{ref}}=5000$; viridis color scales per row block in the script output.

Source PNG (user-composed side-by-side from the study outputs): copied to `journal/notes/figs/2026-04-09-h-decoding-linear-minalpha05/h_decoding_convergence_combined.png`. Individual script outputs (PNG/SVG):  
`/grad/zeyuan/score-matching-fisher/data/h_decoding_convergence_linear_piecewise_minalpha05/`.

---

## 6. Output files

- **Summary:** `data/h_decoding_convergence_linear_piecewise_minalpha05/h_decoding_convergence_summary.txt`
- **Tables:** `h_decoding_convergence_results.{npz,csv}`, `h_decoding_convergence_reference.npz`
- **Figures:** `h_decoding_convergence.{png,svg}`, `h_decoding_matrices_panel.{png,svg}`
- **Reference run artifacts:** `data/h_decoding_convergence_linear_piecewise_minalpha05/reference/` (loss curves, H matrices, etc.)

---

## 7. Interpretation

The experiment isolates **finite-$n$** behavior of DSM-based **binned H** and derived **decoding** metrics relative to a large-sample reference on the **same** global permutation. The **Hellinger LB** line tracks **binned H** closely by construction. **Pairwise decoding** (a direct $x$-based test) is noisier at small $n$ but aligns better with the reference as $n$ grows. The **C-based Bayes** row reflects the integrated score pipeline and can behave differently from the raw binned-H geometry; here it stays strongly correlated with the reference across $n$.