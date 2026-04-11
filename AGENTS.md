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

## Output paths (agent replies)

- When reporting where a script wrote files (datasets, figures, logs, run directories, NPZ/CSV/PNG, etc.), **always state the full absolute path** to the artifact or directory (e.g. `/grad/zeyuan/score-matching-fisher/data/...` or the resolved `DATAROOT` path from `global_setting.py`). Do not rely on repo-relative paths alone (`data/foo`) as the only location the user sees.
