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

- Default datasets and run outputs use `DATA_DIR = os.path.join(DATAROOT, "data")` from `global_setting.py` (override `DATAROOT` via `SCORE_MATCHING_FISHER_DATAROOT`). The repo `data/` path is a symlink to that directory when present.

## Notes

- If CUDA is unavailable on the current machine, stop and report that constraint instead of silently switching to CPU.
