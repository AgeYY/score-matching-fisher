---
name: write-md-journal
description: Write or update Markdown research journal notes with full reproducibility. Use when the user wants experiment documentation that may include shell commands, code snippets, and explicit references to repo scripts or modules—while still keeping prose clear and figures embedded under journal/notes/.
---

# Write Markdown Journal

Create **Markdown** journal notes (`journal/notes/`) that explain methods and results **and** allow **code, commands, and script paths** for reproducibility. This extends the generic journal workflow: prose stays readable, but you **may** include fenced code blocks, CLI invocations, and pointers to `bin/…`, `fisher/…`, or other source files when they help others rerun the work.

## When to use

- The user wants **copy-pasteable** reproduction (conda/mamba commands, script names, key flags).
- The note should link **implementation** to **results** (e.g. “trained with `run_fisher.py score …`”).
- The audience includes **readers who will run code**, not only narrative readers.

## Workflow

1. **Find journal structure**
   - Confirm `journal/notes/` and `journal/main.md`.
   - Follow repo conventions (`AGENTS.md`, `journal/readme.md` if present).

2. **Create or update the note**
   - Filename: `YYYY-MM-DD-topic.md`.
   - Suggested sections: **Question/Context**, **Method**, **Reproduction (commands & scripts)**, **Results**, **Figure**, **Artifacts**, **Takeaway**.
   - Explain the method in plain language first; then add **commands/code** where useful.

3. **Code and script references (allowed and encouraged when relevant)**
   - Full **shell commands** for dataset generation, training, and evaluation (e.g. `mamba run -n geo_diffusion python …`).
   - **Fenced code blocks** for short snippets or multi-line commands.
   - **Repo paths** to scripts and modules (e.g. `bin/make_dataset.py`, `fisher/shared_fisher_est.py`) when they clarify what was run.
   - Prefer **one** canonical command block per pipeline step over huge dumps of source code; link paths for long files.

4. **Math**
   - In Markdown notes, use `$...$` inline and `$$...$$` for display, consistent with other project notes—unless the user asks for LaTeX-only output (then use **write-tex-journal**).

5. **Figures**
   - Place figures under `journal/notes/figs/<note-slug>/`.
   - Copy or generate at least one representative figure; embed with a relative link, e.g. `![caption](figs/<note-slug>/figure.png)`.
   - Add 1–2 sentences interpreting the figure.

6. **Reproducibility block**
   - List **exact** commands used, **parameter values**, and **paths** to saved outputs (NPZ, CSV, PNG).
   - Prefer **absolute** artifact paths when listing outputs on disk.

7. **Index**
   - Add an entry to `journal/main.md` under the correct month; title should mention Markdown + reproducibility if helpful.

## Writing rules

- Balance **narrative** and **reproducibility**: never replace explanation with only code.
- Define symbols on first use.
- Separate **observations** from **conclusions**.
- If numbers are reported, state **which run/config** they came from.

## Quality checklist

- Note answers the user’s question and can be **re-run** from documented commands/paths.
- At least one figure is embedded with interpretation.
- Artifact paths are valid.
- `journal/main.md` is updated.
