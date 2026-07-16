# AGENTS Instructions

## Runtime Environment (Mandatory)

- Always run Python/code commands inside the `mamba` environment `geo_diffusion`.
- Before every new GPU training/evaluation run, compare free memory on CUDA
  devices 0 and 1 and use whichever device has more free memory. Break an exact
  tie in favor of CUDA device 0.
- Keep the selected device fixed for the complete run or multi-stage pipeline;
  do not switch devices between stages of the same run.
- If neither CUDA device 0 nor CUDA device 1 is available, stop and report that
  constraint instead of silently switching to CPU.

## Training Budget (Mandatory)

- Unless the user explicitly requests different values in the current prompt,
  always use the project-wide training defaults from `global_setting.py`:
  `TRAINING_MAX_EPOCHS = 20_000` and
  `EARLY_STOPPING_PATIENCE = 1_000`.
- Training scripts and configuration objects must use these global constants as
  their defaults instead of introducing smaller local epoch or patience
  defaults. Explicit command-line overrides requested by the user still take
  precedence.

Select the device immediately before starting a GPU run:

```bash
GPU_INDEX="$(
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F',' '
      {
        gsub(/[[:space:]]/, "", $1)
        gsub(/[[:space:]]/, "", $2)
      }
      $1 == "0" || $1 == "1" {
        if (!found || ($2 + 0) > best_free) {
          best_index = $1
          best_free = $2 + 0
          found = 1
        }
      }
      END { if (found) print best_index }
    '
)"
if [[ -z "${GPU_INDEX}" ]]; then
  echo "Neither CUDA device 0 nor CUDA device 1 is available." >&2
  exit 1
fi
DEVICE="cuda:${GPU_INDEX}"
echo "Selected ${DEVICE}"
```

## Standard Command Pattern

Use this pattern for all project runs:

```bash
mamba run -n geo_diffusion python <script>.py ... --device "${DEVICE}"
```

For the unified CLI:

```bash
mamba run -n geo_diffusion python run_fisher.py score ... --device "${DEVICE}"
mamba run -n geo_diffusion python run_fisher.py decoder ... --device "${DEVICE}"
```

## Data layout

- Default datasets and run outputs use `DATA_DIR = DATAROOT` from `global_setting.py` (override `DATAROOT` via `SCORE_MATCHING_FISHER_DATAROOT`). The repo `data/` path is a symlink to `DATAROOT` when present.

### Repo-root `journal/` and `report/` (notes sibling)

- Git tracks `journal` and `report` as symlinks into `../score-matching-fisher-note/score-matching-fisher-note-repo/`.
- On **Windows**, default Git checkout (`core.symlinks=false`) leaves **plain text files** with the target path; IDE clicks open the stub instead of the notes tree. Fix after clone:

```bash
mamba run -n geo_diffusion python bin/setup_repo_symlinks.py
```

- Do **not** `git restore journal report` on Windows after fixing links — that restores the broken stubs. `git status` may show those paths as deleted; ignore unless you intend to change the symlink targets in Git.

### PR-autoencoder high-`x_dim` datasets

- Generate low-dimensional `randamp_gaussian_sqrtd` with `bin/make_dataset.py`, then embed with `bin/project_dataset_pr_autoencoder.py` (see `docs/dataset_pr_autoencoder_workflow.md`). The `randamp_gaussian_sqrtd_pr_autoencoder` dataset-family token is removed.

### Reporting paths to humans (prefer `data/`)

- When a file or directory lives under `DATAROOT`, **report it via the repo symlink** so paths match the tree users browse in the clone: **`./data/...`** (from repo root) or **`<repo-root>/data/...`** as a full absolute path.
- **Do not** spell the same location using the bare resolved `DATAROOT` path (e.g. `/data/zeyuan/score-matching-fisher/...`) when `./data/` is available—those are the same inode, but `data/` is stable in docs and matches IDE file trees.
- Scripts may still read/write using `DATAROOT` internally; this rule is for **agent replies and documentation**, not for changing code.

## Markdown math (`journal/notes/`, AGENTS-facing docs)

- Research journal Markdown under `journal/notes/` lives in the **sibling notes workspace** (see `.cursor/skills/write-md-journal/SKILL.md` for `{NOTES_ROOT}`), not in this repo’s tree by default.
- Use **dollar delimiters** so math renders consistently in Markdown viewers (e.g. GitHub, many IDEs):
  - **Inline:** `$...$` (e.g. `$\mu(\theta)$`, `$\sqrt{d}$`).
  - **Display (own line):** `$$` on a line, equation, then `$$` on a line.
- Avoid `\(...\)` and `\[...\]` in project Markdown unless a specific pipeline requires them.

## Notes

- Free GPU memory at selection time is the availability criterion. Record the
  selected device in the run log when possible.

## Background jobs (agents)

When polling until a long-running process finishes, **do not** use a loop like:

`while pgrep -f "some_script.py"; do sleep 30; done`

`pgrep -f` matches the **full command line** of every process. The shell running that `while` loop often includes `some_script.py` in its own argv, so the match never clears and the wait runs forever even after the real job exited.

**Preferred:** capture the PID when starting, then `wait` or poll the PID only:

```bash
mamba run -n geo_diffusion python bin/some_script.py ... --device "${DEVICE}" &
pid=$!
wait "${pid}"
# or: while kill -0 "${pid}" 2>/dev/null; do sleep 30; done
```

## Monitoring long runs (agents)

When the user asks to **monitor until a job finishes**, use a **completion signal** that cannot match your own shell, not `pgrep -f` loops (see **Background jobs** above).

**Good ways to detect completion:**

1. **Sentinel line in a log** the job appends to (e.g. wrap the command with `echo "=== phase done $(date -Is) ==="` after each stage).
2. **Final artifact file** exists and is non-empty, e.g. `h_decoding_convergence_results.npz` under the run’s `--output-dir`, or `[convergence] Saved:` in `run.log`.
3. **Poll a known PID**: if you started the job with `pid=$!`, use `while kill -0 "$pid" 2>/dev/null; do sleep 30; done` (refresh the PID if you did not capture it at start).

**Polling pattern (safe):**

```bash
MASTER=/path/to/pipeline.log   # or check a results file
while ! grep -q '100D done' "$MASTER" 2>/dev/null; do sleep 45; done
```

Prefer `grep` on a **unique marker** you or the script wrote, not on a substring that appears in the monitor’s argv.

**Logging:** For long `mamba run` jobs, set `PYTHONUNBUFFERED=1` and/or redirect to a file so progress is visible; otherwise logs may buffer until exit.

**When reporting to the user:** state that the pipeline finished, cite the **absolute path** to the master log and each output directory (prefer **`<repo-root>/data/...`** when outputs are under `DATAROOT`), and mention key artifacts (e.g. `h_decoding_convergence_combined.svg`, `h_decoding_convergence_results.npz`).

## Output paths (agent replies)

- When reporting where a script wrote files (datasets, figures, logs, run directories, NPZ/CSV/PNG, etc.), **always state a full path** the user can open. Prefer **`<repo-root>/data/...`** for anything under `DATAROOT` (see **Data layout → Reporting paths**). Avoid giving only repo-relative `data/foo` with no root, and **avoid** spelling the same file using the resolved `DATAROOT` path alone (e.g. `/data/zeyuan/score-matching-fisher/...`) when `./data/` is present.
