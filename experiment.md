# Experiment Task

## Goal

In the current repo, the dataset, focusing on two datasets, /grad/zeyuan/score-matching-fisher/.cursor/skills/cosinebench/SKILL.md and /grad/zeyuan/score-matching-fisher/.cursor/skills/linearbench/SKILL.md, are one dimensional, meaning that \theta is one-dimensional. I would like you to create two-dimensional \theta, mimicking the one-dimensional version. Please:

1. Create 2D versions
2. Benchmarking the following methods on this 2D version: bin_gaussian;theta_path_integral;theta_flow;x_flow;linear_x_flow;linear_x_flow_nonlinear_pca;linear_x_flow_diagonal;, using ./bin/study_h_decoding_twofig.py
3. Make sure you write the final report to me.

## Max loops

3

## Output folder

./data/experiments/<short_experiment_name>/

## Constraints

- Use existing repo code if possible.
- Use fixed random seeds.
- Each loop should save metrics, figures, logs, and notes.
- Each loop should be runnable with `mamba run -n geo_diffusion python ... --device cuda`.
- do not use internet
- if the current result looks good, you are allowed to terminate loop and write the final report.

## Required outputs

For each loop:

- `run.py`
- `metrics.json`
- `notes.md`
- `run.log`

At the end:

- `report.md`