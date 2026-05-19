---
name: run-llr-bench
description: >-
  Score-matching-fisher: run the binary random-MoG LLR diagnostic
  tests/study_random_mog_binary_llr_simple.py with native x_dim 4, n_train 600, and PR
  projection to pr_dim 5 and 30 (two runs). Default methods are binary_classifier and
  ctsm_v_binary (both always computed by the script). Use when the user says run-llr-bench,
  run llr bench, LLR bench simple, or binary MoG LLR diagnostic on xdim4 n600 pr5/pr30.
---

# run-llr-bench

## Meaning

**"run-llr-bench"** runs the **simple binary random-MoG log-likelihood-ratio diagnostic** in `tests/study_random_mog_binary_llr_simple.py` with this **fixed recipe** unless the user overrides fields:

| Piece | Canonical value |
|-------|-----------------|
| Script | `tests/study_random_mog_binary_llr_simple.py` |
| Native $x$ dim | **4** (`--x-dim 4`) |
| Training samples | **600** (`--n-train 600`) |
| Categories | **2** (binary; script default `--num-categories 2`) |
| PR embed dims | **5** and **30** — **two separate runs** with `--pr-project` |
| Methods (default) | **`binary_classifier`**, **`ctsm_v_binary`** |
| Device | **CUDA** (`--device cuda`) per **`AGENTS.md`** |

Ground-truth LLR is **analytic** on **native** $x$; both estimators train on **work** features (PR-embedded $x$ when `--pr-project` is set).

### Methods

The script **always trains and evaluates both** methods in one invocation (no `--methods` flag). Keys in `simple_binary_llr_results.npz` and the summary:

| Method token | NPZ key | Role |
|--------------|---------|------|
| `binary_classifier` | `binary_classifier_llr` | sklearn `LogisticRegression` log-odds minus training prior log-odds |
| `ctsm_v_binary` | `ctsm_v_binary_llr` | binary CTSM-$v$ time-score net + path integral |

**User override:** If the user names a subset (e.g. only `ctsm_v_binary`), still run the script once and **report only** the requested methods from `simple_binary_llr_summary.txt` / NPZ. If they name methods this script does not implement, say so and ask whether to extend the script.

**Other CLI:** Keep script defaults for `--n-val`, `--n-test-per-class`, CTSM training, MoG sampling, and PR-autoencoder hyperparameters unless the user overrides them.

## Canonical output trees (repo `data/`)

Under **`<repo-root>/data/random_mog_binary_llr_bench/`**:

| Run | `--output-dir` |
|-----|----------------|
| PR5 | `data/random_mog_binary_llr_bench/xdim4_n600_pr5/` |
| PR30 | `data/random_mog_binary_llr_bench/xdim4_n600_pr30/` |

Each run writes:

- `simple_binary_llr_results.npz`
- `simple_binary_llr_diagnostic.svg` / `.png`
- `simple_binary_llr_summary.txt` (RMSE/corr for both methods vs analytic GT)

## Commands

From repo root, **`mamba` env `geo_diffusion`**, **`PYTHONUNBUFFERED=1`** for long CTSM training:

```bash
REPO=/grad/zeyuan/score-matching-fisher
cd "${REPO}"

mamba run -n geo_diffusion env PYTHONUNBUFFERED=1 python tests/study_random_mog_binary_llr_simple.py \
  --x-dim 4 \
  --n-train 600 \
  --pr-project \
  --pr-dim 5 \
  --pr-use-cache \
  --output-dir data/random_mog_binary_llr_bench/xdim4_n600_pr5 \
  --device cuda

mamba run -n geo_diffusion env PYTHONUNBUFFERED=1 python tests/study_random_mog_binary_llr_simple.py \
  --x-dim 4 \
  --n-train 600 \
  --pr-project \
  --pr-dim 30 \
  --pr-use-cache \
  --output-dir data/random_mog_binary_llr_bench/xdim4_n600_pr30 \
  --device cuda
```

`--pr-dim` must be **greater than** native `--x-dim` (4). PR cache: `data/pr_autoencoder_cache/` (default `--pr-cache-dir`).

## Agent behavior

1. On **run-llr-bench** (or close variants), run **both** PR5 and PR30 commands above unless the user restricts `pr-dim` or skips PR (native-only: omit `--pr-project`; not the default bench).
2. Apply user overrides to **`--x-dim`**, **`--n-train`**, **`--pr-dim`**, **`--output-dir`**, CTSM/PR flags, or **method names to emphasize in the report** — keep everything else at script defaults.
3. After each run, read **`simple_binary_llr_summary.txt`** and report **full paths** under **`<repo-root>/data/...`** (see **`AGENTS.md`**).
4. If CUDA is unavailable, **stop** and report; do not silently use CPU for this bench.
5. For a quick smoke test only when the user asks: lower `--ctsm-binary-epochs` and/or `--n-test-per-class`; label results as smoke, not canonical bench.

## Completion signal

Each run prints `Saved results:`, `Saved figure:`, `Saved summary:`. Poll those paths or log lines; do not use `pgrep -f` loops (see **`AGENTS.md`**).
