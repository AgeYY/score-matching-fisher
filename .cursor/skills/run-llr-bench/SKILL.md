---
name: run-llr-bench
description: >-
  Score-matching-fisher: run the binary random-MoG LLR diagnostic
  bin/bench-binary-llr.py with native x_dim 4, n_train 600, and PR
  projection to pr_dim 5 and 30 in parallel on cuda:0 (PR5) and cuda:1 (PR30). Default
  methods are binary_classifier and ctsm_v_binary (both always computed by the script).
  Use when the user says run-llr-bench, run llr bench, LLR bench simple, or binary MoG
  LLR diagnostic on xdim4 n600 pr5/pr30.
---

# run-llr-bench

## Meaning

**"run-llr-bench"** runs the **simple binary random-MoG log-likelihood-ratio diagnostic** in `bin/bench-binary-llr.py` with this **fixed recipe** unless the user overrides fields:

| Piece | Canonical value |
|-------|-----------------|
| Script | `bin/bench-binary-llr.py` |
| Native $x$ dim | **4** (`--x-dim 4`) |
| Training samples | **600** (`--n-train 600`) |
| Categories | **2** (binary; script default `--num-categories 2`) |
| PR embed dims | **5** and **30** — **two runs in parallel** with `--pr-project` |
| Methods (default) | **`binary_classifier`**, **`ctsm_v_binary`** |
| GPUs | **PR5 → `cuda:0`**, **PR30 → `cuda:1`** (`--device cuda:0` / `--device cuda:1`) |

Ground-truth LLR is **analytic** on **native** $x$; both estimators train on **work** features (PR-embedded $x$ when `--pr-project` is set).

### Two-GPU split (canonical)

Launch **both** jobs **at once** so wall time is ~one CTSM train, not two sequential runs:

| Run | `--pr-dim` | `--device` | `--output-dir` |
|-----|------------|------------|----------------|
| PR5 | 5 | **`cuda:0`** | `data/random_mog_binary_llr_bench/xdim4_n600_pr5/` |
| PR30 | 30 | **`cuda:1`** | `data/random_mog_binary_llr_bench/xdim4_n600_pr30/` |

Each process must use a **distinct** `--output-dir` and **distinct** log file. Capture both PIDs and `wait` on them (see **Commands**); do not rely on `pgrep -f` for completion.

**Fallback (one GPU):** If only one GPU is available, run sequentially (PR5 then PR30) on `cuda:0`, or ask the user. Do not map both jobs to the same device in parallel.

**Alternative pinning:** `CUDA_VISIBLE_DEVICES=0` / `=1` with `--device cuda` also works (see **benchmark-1d** skill); the canonical bench here uses explicit **`cuda:0`** / **`cuda:1`**.

### Methods

The script **always trains and evaluates both** methods in one invocation (no `--methods` flag). Keys in `simple_binary_llr_results.npz` and the summary:

| Method token | NPZ key | Role |
|--------------|---------|------|
| `binary_classifier` | `binary_classifier_llr` | sklearn `LogisticRegression` log-odds minus training prior log-odds |
| `ctsm_v_binary` | `ctsm_v_binary_llr` | binary CTSM-$v$ time-score net + path integral |

**User override:** If the user names a subset (e.g. only `binary_classifier`), still run the script once per PR dim and **report only** the requested methods from `simple_binary_llr_summary.txt` / NPZ. If they name methods this script does not implement, say so and ask whether to extend the script.

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
- `run.log` (recommended: redirect stdout/stderr when launching in background)

## Commands

From repo root, **`mamba` env `geo_diffusion`**, **`PYTHONUNBUFFERED=1`** for long CTSM training. **Launch both lines below in parallel** (same shell block):

```bash
REPO=/grad/zeyuan/score-matching-fisher
cd "${REPO}"

mamba run -n geo_diffusion env PYTHONUNBUFFERED=1 python bin/bench-binary-llr.py \
  --x-dim 4 \
  --n-train 600 \
  --pr-project \
  --pr-dim 5 \
  --pr-use-cache \
  --output-dir data/random_mog_binary_llr_bench/xdim4_n600_pr5 \
  --device cuda:0 \
  > data/random_mog_binary_llr_bench/xdim4_n600_pr5/run.log 2>&1 &
pid_pr5=$!

mamba run -n geo_diffusion env PYTHONUNBUFFERED=1 python bin/bench-binary-llr.py \
  --x-dim 4 \
  --n-train 600 \
  --pr-project \
  --pr-dim 30 \
  --pr-use-cache \
  --output-dir data/random_mog_binary_llr_bench/xdim4_n600_pr30 \
  --device cuda:1 \
  > data/random_mog_binary_llr_bench/xdim4_n600_pr30/run.log 2>&1 &
pid_pr30=$!

wait "${pid_pr5}" "${pid_pr30}"
```

`--pr-dim` must be **greater than** native `--x-dim` (4). PR cache: `data/pr_autoencoder_cache/` (default `--pr-cache-dir`).

**Pre-flight:** Confirm at least two GPUs (`nvidia-smi` or `torch.cuda.device_count() >= 2`) before parallel launch.

## Agent behavior

1. On **run-llr-bench**, launch **PR5 on `cuda:0` and PR30 on `cuda:1` in parallel** unless the user restricts `pr-dim`, requests a single GPU, or skips PR (native-only: omit `--pr-project`; not the default bench).
2. Capture **`pid_pr5`** and **`pid_pr30`** at start; poll completion via **`wait`**, **`run.log`** (`Saved summary:`), or artifact mtimes — not `pgrep -f` loops.
3. Apply user overrides to **`--x-dim`**, **`--n-train`**, **`--pr-dim`**, **`--output-dir`**, device/GPU assignment, CTSM/PR flags, or **method names to emphasize in the report** — keep everything else at script defaults.
4. After **both** runs finish, read each **`simple_binary_llr_summary.txt`** and report **full paths** under **`<repo-root>/data/...`** (see **`AGENTS.md`**).
5. If CUDA is unavailable, **stop** and report; do not silently use CPU for this bench.
6. For a quick smoke test only when the user asks: lower `--ctsm-binary-epochs` and/or `--n-test-per-class`; label results as smoke, not canonical bench.

## Completion signal

Each run prints `Saved results:`, `Saved figure:`, `Saved summary:` (in `run.log` when redirected). Poll those paths or log lines; bench is done when **both** output dirs have fresh summaries.
