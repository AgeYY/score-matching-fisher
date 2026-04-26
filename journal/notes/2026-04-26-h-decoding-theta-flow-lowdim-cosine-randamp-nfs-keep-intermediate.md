# H-decoding convergence: low-dimensional `theta_flow` sweeps (cosine $\sqrt{d}$ vs `randamp_gaussian`) and NFS `--keep-intermediate`

## Question / context

We wanted quick **`theta_flow`** convergence studies on small observation spaces: **3D** `cosine_gaussian_sqrtd`, **2D** and **3D** `randamp_gaussian`, using `bin/study_h_decoding_convergence.py` with default `n_list=80,200,400,600` and `n_ref=5000`. A first attempt on NFS failed during **temporary-directory cleanup** after a successful per-$n$ train; this note records the workaround and points to full artifact paths.

For how the **fixed-$x$ posterior** panel (blue model vs red generative reference) is computed, see the dedicated methods note (avoids duplicating equations here): [2026-04-22 fixed-$x$ posterior panel](2026-04-22-fixed-x-posterior-model-vs-approx-gt.md).

## Method

- **Data:** `bin/make_dataset.py` writes a shared NPZ (`theta_all`, `x_all`, train/val indices, `meta_json_utf8`). Families: `cosine_gaussian_sqrtd` (noise std scaled like $\sqrt{d}$ on top of a fixed per-dim baseline) and `randamp_gaussian` (random-amplitude bump means, baseline diagonal noise).
- **Study:** `bin/study_h_decoding_convergence.py` loads the NPZ, builds nested subsets for each $n$, trains the score + $\theta$-space flows for **`theta_field_method=theta_flow`**, evaluates binned $H$ vs GT Hellinger MC, pairwise decoding, and writes `h_decoding_convergence_combined.{png,svg}` plus per-$n$ artifacts under `sweep_runs/n_*/` when intermediates are kept.
- **NFS / temp dirs:** With default settings the per-$n$ run directory is a `tempfile.TemporaryDirectory` under `--output-dir`. On this filesystem, `tmp_ctx.cleanup()` sometimes raised `OSError: [Errno 39] Directory not empty` even after training finished. Passing **`--keep-intermediate`** uses a stable path `output_dir/sweep_runs/n_XXXXXX/` instead, avoiding that cleanup path.

## Reproduction (commands and scripts)

From the repo root, environment `geo_diffusion`, GPU (`--device cuda`). Stamp `TS=$(date +%Y%m%d-%H%M%S)` if you want fresh paths.

**1) Datasets (8000 samples, seed 7, 70/30 train/val split)**

```bash
mamba run -n geo_diffusion python bin/make_dataset.py \
  --dataset-family cosine_gaussian_sqrtd --x-dim 3 --n-total 8000 --seed 7 \
  --output-npz "data/datasets/cosine_gaussian_sqrtd_xdim3_n8000_${TS}.npz"

mamba run -n geo_diffusion python bin/make_dataset.py \
  --dataset-family randamp_gaussian --x-dim 2 --n-total 8000 --seed 7 \
  --output-npz "data/datasets/randamp_gaussian_xdim2_n8000_${TS}.npz"

mamba run -n geo_diffusion python bin/make_dataset.py \
  --dataset-family randamp_gaussian --x-dim 3 --n-total 8000 --seed 7 \
  --output-npz "data/datasets/randamp_gaussian_xdim3_n8000_${TS}.npz"
```

**2) Convergence studies (`theta_flow`, keep sweep artifacts)**

```bash
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=0 mamba run -n geo_diffusion python bin/study_h_decoding_convergence.py \
  --dataset-npz data/datasets/cosine_gaussian_sqrtd_xdim3_n8000_20260426-174214.npz \
  --dataset-family cosine_gaussian_sqrtd \
  --theta-field-method theta_flow --keep-intermediate \
  --output-dir data/h_decoding_conv_cosine_sqrtd_xdim3_theta_flow_20260426-174214 \
  --device cuda

CUDA_VISIBLE_DEVICES=0 mamba run -n geo_diffusion python bin/study_h_decoding_convergence.py \
  --dataset-npz data/datasets/randamp_gaussian_xdim2_n8000_20260426-174214.npz \
  --dataset-family randamp_gaussian \
  --theta-field-method theta_flow --keep-intermediate \
  --output-dir data/h_decoding_conv_randamp_gaussian_xdim2_theta_flow_20260426-174214 \
  --device cuda

CUDA_VISIBLE_DEVICES=0 mamba run -n geo_diffusion python bin/study_h_decoding_convergence.py \
  --dataset-npz data/datasets/randamp_gaussian_xdim3_n8000_20260426-180145.npz \
  --dataset-family randamp_gaussian \
  --theta-field-method theta_flow --keep-intermediate \
  --output-dir data/h_decoding_conv_randamp_gaussian_xdim3_theta_flow_20260426-180145 \
  --device cuda
```

**Bundled driver (optional):** `data/h_decoding_conv_theta_flow_runs_20260426-174214/run_both_theta_flow.sh` runs the cosine-3D and randamp-2D pair with `--keep-intermediate` (paths inside the script are tied to `TS=20260426-174214`).

**Key source files**

- `bin/study_h_decoding_convergence.py` — sweep, fixed-$x$ diagnostic (`_write_fixed_x_posterior_diagnostic`, `_plot_fixed_x_column`).
- `bin/make_dataset.py` — dataset NPZ generation.
- `fisher/evaluation.py` — `log_p_x_given_theta` for the red “GT” curve in the diagnostic.

## Results

- **3D `cosine_gaussian_sqrtd`:** Training completed for all $n$; combined figure and `h_decoding_convergence_results.npz` written under the output dir below. (After an initial failure without `--keep-intermediate`, the run was repeated with `--keep-intermediate`.)
- **2D / 3D `randamp_gaussian`:** Same pipeline; the 3D run showed strong matrix agreement at $n=600$ in the log (e.g. `corr_h≈0.96`, `corr_clf≈0.98`, `corr_llr≈0.94` for that configuration—numbers refer to that single run’s stdout, not a sweep across seeds).

**Observation vs conclusion:** Low-dimensional `randamp_gaussian` in this setup can yield **high** correlation between binned flow $H$ and GT MC $H$ at the largest $n$; the 3D cosine $\sqrt{d}$ run from the same day is a separate geometry/noise scaling and should be read from its own `h_decoding_convergence_results.csv`, not assumed identical.

## Figure

Fixed-$x$ posterior diagnostic embedded from the **3D `randamp_gaussian`** run at `n=600` (two deterministic row indices from the study’s seed logic). Blue: softmax on the flow $C$ row plus Gaussian prior add-back, then KDE on the training $\theta$ grid; red: generative $\log p(x\mid\theta)$ plus **uniform** prior on $[\theta_{\mathrm{low}},\theta_{\mathrm{high}}]$ (see linked 2026-04-22 note).

![Fixed-$x$ posterior diagnostic (`theta_flow`, 3D randamp Gaussian, $n=600$ sweep)](figs/2026-04-26-h-decoding-theta-flow-lowdim/theta_flow_single_x_posterior_hist.png)

## Artifacts (absolute paths under repo `data/`)

| Run | Output directory |
|-----|------------------|
| Cosine $\sqrt{d}$, $x_{\mathrm{dim}}=3$ | `/grad/zeyuan/score-matching-fisher/data/h_decoding_conv_cosine_sqrtd_xdim3_theta_flow_20260426-174214/` |
| `randamp_gaussian`, $x_{\mathrm{dim}}=2$ | `/grad/zeyuan/score-matching-fisher/data/h_decoding_conv_randamp_gaussian_xdim2_theta_flow_20260426-174214/` |
| `randamp_gaussian`, $x_{\mathrm{dim}}=3$ | `/grad/zeyuan/score-matching-fisher/data/h_decoding_conv_randamp_gaussian_xdim3_theta_flow_20260426-180145/` |

Each directory contains `h_decoding_convergence_combined.svg` (full layout), `h_decoding_convergence_results.{npz,csv}`, `h_decoding_convergence_summary.txt`, `training_losses/`, and with `--keep-intermediate`, `sweep_runs/n_000080/` … `n_000600/` with checkpoints and `diagnostics/theta_flow_single_x_posterior_hist.{png,svg}`.

**Datasets**

- `/grad/zeyuan/score-matching-fisher/data/datasets/cosine_gaussian_sqrtd_xdim3_n8000_20260426-174214.npz`
- `/grad/zeyuan/score-matching-fisher/data/datasets/randamp_gaussian_xdim2_n8000_20260426-174214.npz`
- `/grad/zeyuan/score-matching-fisher/data/datasets/randamp_gaussian_xdim3_n8000_20260426-180145.npz`

## Takeaway

- Use **`--keep-intermediate`** for `study_h_decoding_convergence.py` on NFS-like storage when temp-dir cleanup errors appear after training.
- The embedded diagnostic is a **qualitative** check; the blue and red curves use **different priors** in the `theta_flow` path (Gaussian add-back vs uniform generative box)—see [2026-04-22 note](2026-04-22-fixed-x-posterior-model-vs-approx-gt.md) for the precise construction.
