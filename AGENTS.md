# AGENTS Instructions

## Runtime Environment (Mandatory)

- Always run Python/code commands inside the `mamba` environment `geo_diffusion`.
- Default execution device is CUDA (`--device cuda`) for training/evaluation scripts.

## Standard Command Pattern

Use this pattern for all project runs:

```bash
mamba run -n geo_diffusion python <script>.py ... --device cuda
```

For the unified CLI:

```bash
mamba run -n geo_diffusion python run_fisher.py score ... --device cuda
mamba run -n geo_diffusion python run_fisher.py decoder ... --device cuda
```

## Data layout

- Default datasets and run outputs use `DATA_DIR = DATAROOT` from `global_setting.py` (override `DATAROOT` via `SCORE_MATCHING_FISHER_DATAROOT`). The repo `data/` path is a symlink to `DATAROOT` when present.

## Notes

- If CUDA is unavailable on the current machine, stop and report that constraint instead of silently switching to CPU.

## Background jobs (agents)

When polling until a long-running process finishes, **do not** use a loop like:

`while pgrep -f "some_script.py"; do sleep 30; done`

`pgrep -f` matches the **full command line** of every process. The shell running that `while` loop often includes `some_script.py` in its own argv, so the match never clears and the wait runs forever even after the real job exited.

**Preferred:** capture the PID when starting, then `wait` or poll the PID only:

```bash
mamba run -n geo_diffusion python bin/some_script.py ... --device cuda &
pid=$!
wait "${pid}"
# or: while kill -0 "${pid}" 2>/dev/null; do sleep 30; done
```

## Output paths (agent replies)

- When reporting where a script wrote files (datasets, figures, logs, run directories, NPZ/CSV/PNG, etc.), **always state the full absolute path** to the artifact or directory (e.g. `/grad/zeyuan/score-matching-fisher/data/...` or the resolved `DATAROOT` path from `global_setting.py`). Do not rely on repo-relative paths alone (`data/foo`) as the only location the user sees.
