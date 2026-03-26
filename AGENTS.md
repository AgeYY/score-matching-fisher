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

## Notes

- If CUDA is unavailable on the current machine, stop and report that constraint instead of silently switching to CPU.
