---
name: write-tex-journal
description: Write LaTeX journal write-ups for non-coder readers under report/notes, wired into report/main.tex; narrative methods and results with full mathematical notation, no code blocks and no script or CLI references in the .tex. Use when the user wants a .tex note or section for the score-matching-fisher report PDF.
---

# Write LaTeX Journal

Produce **LaTeX** fragments that explain research context, methods, and results for **non-coder** readers. **Do not** include shell commands, code listings, script paths, module names, or repository file paths **inside the `.tex` note**. Convey reproducibility only through **mathematical and verbal** description (distributions, dimensions, hyperparameters named in words, etc.).

## When to use

- The audience is **paper / thesis / lecture** readers who will not open the codebase.
- The user wants **equations** typeset in proper LaTeX (`amsmath`, etc.).
- The user explicitly asks for **LaTeX** output, not Markdown.

## Output location and naming (mandatory for this repo)

- **Always** write new notes under `report/notes/` using filenames `YYYY-MM-DD-topic.tex`.
- Treat each note as a **section-like fragment** compiled via **`report/main.tex`**: do **not** rely on `journal/tex/` for new material unless the user explicitly overrides this skill.
- **Always** register the new file in **`report/main.tex`** by adding a line
  `\include{notes/YYYY-MM-DD-topic.tex}`
  (match existing style in that file; order is typically chronological / logical reading order at the end of the `\include` list).
- Standalone `\documentclass{article}` files are **out of scope** unless the user explicitly asks for a one-off document outside the report.

## Workflow

1. **Clarify scope**
   - Default deliverable: an `\include`-able fragment under `report/notes/`.
   - Confirm notation conventions if the project macro setup changes; follow `report/main.tex` preamble.

2. **Structure the write-up**
   - Suggested sections: **Context / question**, **Setup and notation**, **Model or estimator (math)**, **Metrics**, **Results (narrative + tables)**, **Figures**, **Takeaway**.
   - Use **prose** to describe procedures (e.g. “we train a noise-conditional score network with Adam”) without naming executable scripts.

3. **Forbidden content (inside the `.tex` note)**
   - No `verbatim`, `lstlisting`, or minted blocks for **code**.
   - No `\texttt{}` or inline paths for **files** (e.g. `bin/foo.py`).
   - No copy-paste **CLI** lines (`python`, `mamba`, `conda`, `bash`).
   - If “reproducibility” is needed, state **hyperparameters and data sizes in words and symbols** inside the narrative or in a table environment.

4. **Mathematics (required where relevant)**
   - Use LaTeX display environments: `equation`, `align`, `gather`, etc.
   - Define every symbol when it first appears.
   - Use `amsmath`; add `amssymb` / `bm` if the project standard includes them.

5. **Figures**
   - Place assets under `report/notes/figures/<note-slug>/` (or the existing `report/notes/figures/` layout) and reference them with `\includegraphics` using paths **relative to `report/`** (e.g. `notes/figures/...`) so `main.tex` builds cleanly.
   - The **LaTeX note** should not discuss **how** the figure was generated in code—only what it shows.
   - Always include a **caption** interpreting axes and the main message.

6. **Cross-linking Markdown journal (optional)**
   - If the repo keeps `journal/main.md`, you may add a one-line pointer to the companion `\include` under `report/notes/…` **without** duplicating code—only if the user wants index updates.

## Writing rules

- **Audience:** readers who understand math and scientific language but not the codebase.
- Distinguish **observations** from **conclusions**.
- Keep the TeX **free of implementation detail**; depth lives in **equations and definitions**.

## Quality checklist

- No code, no shell commands, no script paths anywhere in the `.tex` **body** of the note.
- New file lives in **`report/notes/`** and is listed in **`report/main.tex`** via `\include{notes/...}`.
- Equations are correct, labeled where useful, and symbols are defined.
- At least one figure or table has interpretive caption text (when figures/tables are used).
- `report/main.tex` builds successfully after the change.
