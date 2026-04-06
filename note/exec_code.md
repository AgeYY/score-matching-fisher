# Executable scripts (`bin/`)

Top-level driver scripts live under [`bin/`](../bin/). Run them from the **repository root** so default output paths like `data/outputs_step2` (dataset visuals) / `data/outputs_step6_shared_dataset` (`fisher_est` comparison runs) resolve correctly and imports from the `fisher` package work.

Use the project environment and CUDA as in [`AGENTS.md`](../AGENTS.md):

```bash
mamba run -n geo_diffusion python bin/<script>.py ... --device cuda
```

(`visualize_dataset` uses CPU-oriented NumPy/Matplotlib only; `fisher_est` follows the repo rule of not silently falling back when `--device cuda` is requested but unavailable.)

---

## `bin/visualize_dataset.py`

**Role:** **Dataset visualization** for the **uniform-\(\theta\)** toys used in later Fisher experiments.

**What it does:**

- Builds either `ToyConditionalGaussianDataset` or `ToyConditionalGMMNonGaussianDataset` from `fisher.data` (same parameterization as the shared comparison script).
- Draws a joint sample \((\theta, x)\), prints summary statistics (shapes, marginal stats, binned check of \(\mathbb E[x\mid\theta]\) vs. the tuning curve).
- Saves under `--output-dir` (default `data/outputs_step2`):
  - `joint_scatter_theta_color.png` — scatter of \(x\) (PCA projection if \(x\) is high-dimensional) colored by \(\theta\),
  - `tuning_curve.png` — mean \(\mu(\theta)\) vs. \(\theta\),
  - `conditional_slices.png` — conditional \(x\) slices at several \(\theta\) values.

**Notable flags:** `--dataset-family` (`gaussian` / `gmm_non_gauss`), `--x-dim`, covariance and GMM hyperparameters, `--n-joint`, `--output-dir`.

---

## `bin/fisher_est.py`

**Role:** **End-to-end Fisher comparison** on a **single shared dataset**: score-based estimator vs. decoder-based local classification vs. **ground truth** (analytic Gaussian Fisher or Monte Carlo score-squared for the mixture).

**What it does:**

1. Samples one joint dataset from \(p(\theta)p(x\mid\theta)\), then splits train / eval.
2. Trains the **noise-conditional score model** (discrete or continuous \(\sigma\) schedule) on the train split and evaluates **score-based Fisher** along a \(\theta\) bin grid.
3. Trains **local binary classifiers** on \(\theta\pm\epsilon/2\) neighborhoods (decoder path) and evaluates **decoder Fisher** with standard errors.
4. Computes **analytic** Fisher (Gaussian family) or **MC ground truth** (non-Gaussian mixture), then plots all curves and writes metrics under `--output-dir` (default `data/outputs_step6_shared_dataset`).

**Notable flags:** `--dataset-family`, `--x-dim`, `--n-total`, `--train-frac`, score training (`--score-noise-mode`, `--score-sigma-*`, early stopping), decoder (`--decoder-epsilon`, `--decoder-bandwidth`, …), evaluation grid (`--n-bins`, `--eval-margin`), `--device`, `--output-dir`.

This is the main reproducibility entry point referenced in the longer notes (Gaussian and mixture experiments).
