# Executable scripts (`bin/`)

Top-level driver scripts live under [`bin/`](../../bin/). Run them from the **repository root** so default output paths like `data/outputs_step2` (dataset visuals) / `data/outputs_step6_shared_dataset` (shared-dataset Fisher runs) resolve correctly and imports from the `fisher` package work.

Use the project environment and CUDA as in [`AGENTS.md`](../../AGENTS.md):

```bash
mamba run -n geo_diffusion python bin/<script>.py ... --device cuda
```

(`visualize_dataset` uses CPU-oriented NumPy/Matplotlib only. Score/decoder scripts follow the repo rule of not silently falling back when `--device cuda` is requested but unavailable.)

---

## `bin/fisher_make_dataset.py`

**Role:** **Stage 1 — generate and save** the shared joint dataset used by the Fisher comparison (train/eval split + metadata).

**What it does:**

- Builds the toy dataset family used by the shared Fisher pipeline (`--dataset-family`, covariance / GMM flags, etc.).
- Samples once, splits train/eval with `--seed`, and writes a compressed `.npz` under **`--output-npz`** (default `data/shared_fisher_dataset.npz`).
- The file contains arrays (`theta_all`, `x_all`, splits, indices) and JSON metadata (`meta_json_utf8`) so stage 2 can reconstruct the model and match ground-truth Fisher without resampling data.

**Typical run:**

```bash
mamba run -n geo_diffusion python bin/fisher_make_dataset.py --output-npz data/shared_fisher_dataset.npz
```

**Notable flags:** dataset hyperparameters (`--dataset-family`, `--theta-*`, `--x-dim`, `--n-total`, `--train-frac`, …), `--seed`, `--output-npz`.

---

## `bin/fisher_estimate_from_dataset.py`

**Role:** **Stage 2 — Fisher estimation** from a `.npz` produced by `fisher_make_dataset.py`.

**What it does:**

- Loads the saved arrays and metadata, instantiates the matching `ToyConditional*` dataset for **ground truth** (analytic or MC).
- Runs score training, score-based Fisher on bins, decoder local classifiers, plots, curve `.npz`, and `metrics_vs_gt_*.txt` under **`--output-dir`**.

**Typical run:**

```bash
mamba run -n geo_diffusion python bin/fisher_estimate_from_dataset.py \
  --dataset-npz data/shared_fisher_dataset.npz \
  --output-dir data/outputs_step6_shared_dataset \
  --device cuda
```

**Notable flags:** **`--dataset-npz`** (required), all score/decoder/evaluation flags (`--score-*`, `--decoder-*`, `--n-bins`, `--gt-mc-samples-per-bin`, …), `--device`, `--output-dir`.

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
