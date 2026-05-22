---
name: run-llr-bench-5d
description: >-
  Score-matching-fisher: PR5-only slice of run-llr-bench — binary random-MoG LLR diagnostic
  bin/bench-binary-llr.py with native x_dim 4, n_train 600, PR projection
  to pr_dim 5 on cuda:0. Same methods as run-llr-bench (binary_classifier, ctsm_v_binary).
  Use when the user says run-llr-bench-5d, run llr bench 5d, LLR bench pr5 only, or xdim4
  n600 pr5 LLR diagnostic without pr30.
---

# run-llr-bench-5d

## Meaning

**"run-llr-bench-5d"** is the **PR5-only** variant of [**run-llr-bench**](../run-llr-bench/SKILL.md): one invocation of `bin/bench-binary-llr.py` with **`--pr-dim 5`** only (no PR30 run).

| Piece | Canonical value |
|-------|-----------------|
| Script | `bin/bench-binary-llr.py` |
| Native $x$ dim | **4** (`--x-dim 4`) |
| Training samples | **600** (`--n-train 600`) |
| Categories | **2** (binary; script default `--num-categories 2`) |
| PR embed | **`--pr-dim 5`** with `--pr-project` |
| Methods (default) | **`binary_classifier`**, **`ctsm_v_binary`** (both always computed; see parent skill) |
| Device | **`cuda:0`** (`--device cuda:0`) per **`AGENTS.md`** |

Ground-truth LLR is **analytic** on **native** $x$; estimators train on **PR5 work** features.

For **PR30**, method details, smoke-test overrides, and the full two-GPU bench, use [**run-llr-bench**](../run-llr-bench/SKILL.md).

### Methods

Same as **run-llr-bench** — the script has no `--methods` flag; both `binary_classifier` and `ctsm_v_binary` run every time. NPZ keys: `binary_classifier_llr`, `ctsm_v_binary_llr`.

**User override:** If the user names a subset (e.g. only `binary_classifier`), run once and **report only** those lines from `simple_binary_llr_summary.txt` / NPZ.

**Other CLI:** Keep script defaults for `--n-val`, `--n-test-per-class`, CTSM training, MoG sampling, and PR-autoencoder hyperparameters unless the user overrides them.

## Canonical output tree (repo `data/`)

**`<repo-root>/data/random_mog_binary_llr_bench/xdim4_n600_pr5/`**

Writes:

- `simple_binary_llr_results.npz`
- `simple_binary_llr_diagnostic.svg` / `.png`
- `simple_binary_llr_summary.txt` (RMSE/corr for both methods vs analytic GT)
- `run.log` (recommended when running in background)

## Commands

From repo root, **`mamba` env `geo_diffusion`**, **`PYTHONUNBUFFERED=1`**:

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
  2>&1 | tee data/random_mog_binary_llr_bench/xdim4_n600_pr5/run.log
```

Foreground `tee` is fine for a single job; for background:

```bash
mamba run -n geo_diffusion env PYTHONUNBUFFERED=1 python bin/bench-binary-llr.py \
  --x-dim 4 \
  --n-train 600 \
  --pr-project \
  --pr-dim 5 \
  --pr-use-cache \
  --output-dir data/random_mog_binary_llr_bench/xdim4_n600_pr5 \
  --device cuda:0 \
  > data/random_mog_binary_llr_bench/xdim4_n600_pr5/run.log 2>&1 &
pid=$!
wait "${pid}"
```

`--pr-dim` must be **greater than** native `--x-dim` (4). PR cache: `data/pr_autoencoder_cache/` (default `--pr-cache-dir`).

**Pre-flight:** Confirm CUDA is available; default GPU is **`cuda:0`**. Override with `--device cuda:1` only if the user asks.

**Alternative pinning:** `CUDA_VISIBLE_DEVICES=0` with `--device cuda` also works (see **benchmark-1d** skill).

## Agent behavior

1. On **run-llr-bench-5d**, run **only** the PR5 command above — do **not** launch PR30 unless the user explicitly asks for the full [**run-llr-bench**](../run-llr-bench/SKILL.md).
2. Capture **`pid`** if backgrounded; poll via **`wait`**, **`run.log`** (`Saved summary:`), or artifact mtimes — not `pgrep -f` loops.
3. Apply user overrides to **`--x-dim`**, **`--n-train`**, **`--output-dir`**, **`--device`**, CTSM/PR flags, or **method names to emphasize** — keep everything else at script defaults.
4. After the run finishes, read **`simple_binary_llr_summary.txt`** and report **full paths** under **`<repo-root>/data/...`** (see **`AGENTS.md`**).
5. If CUDA is unavailable, **stop** and report; do not silently use CPU.
6. Smoke tests: lower `--ctsm-binary-epochs` / `--n-test-per-class` only when the user asks; label as smoke.

## Completion signal

Prints `Saved results:`, `Saved figure:`, `Saved summary:` (in `run.log` when redirected). Bench is done when `xdim4_n600_pr5/simple_binary_llr_summary.txt` is fresh.
