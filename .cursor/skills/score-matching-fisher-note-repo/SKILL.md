---
name: score-matching-fisher-note-repo
description: >-
  Points the agent at the sibling score-matching-fisher-note repository (notes and prose,
  separate from the coding repo). Use when the user invokes this skill, says notes repo,
  sibling note folder, ../score-matching-fisher-note, or wants reads/writes outside the
  code tree in the dedicated notes clone.
disable-model-invocation: true
---

# score-matching-fisher-note-repo

## What this is

The **coding** project is `score-matching-fisher` (this workspace). Long-form notes, journals, and non-code artifacts the user keeps elsewhere live in a **sibling** Git repo:

| Role | Typical path |
|------|----------------|
| Notes repo (read/write here when this skill applies) | **`/grad/zeyuan/score-matching-fisher-note/`** |
| Same place, relative from code repo root | **`../score-matching-fisher-note/`** |
| **`journal/`** and **`report/`** trees (canonical) | **`/grad/zeyuan/score-matching-fisher-note/score-matching-fisher-note-repo/journal/`** and **`.../report/`** |

Resolve the notes root as: parent directory of the `score-matching-fisher` clone + `score-matching-fisher-note`. If the clone lives elsewhere on disk, substitute that parent; the folder name stays `score-matching-fisher-note`.

In the coding clone, **`journal`** and **`report`** at the repo root are **symlinks** into `score-matching-fisher-note-repo/` so scripts and docs that use `journal/...` or `report/...` keep working.

## Agent behavior when this skill is active

1. **Treat the notes repo as in-scope** for Read, Write, StrReplace, Grep, Glob, and terminal commands—same as the code repo unless the user restricts a task to one side only.
2. **Prefer absolute paths** under `/grad/zeyuan/score-matching-fisher-note/` when running tools, so paths do not depend on the shell cwd.
3. **Git**: commits and branches in the notes repo are separate from the code repo; run `git` with `working_directory` (or `cd`) set to the notes root when the user asks for version control there.
4. **Do not** assume a fixed internal layout inside the notes repo; discover files with Glob/Grep after landing in that root.

## Journal and report

Markdown journal lives under **`.../score-matching-fisher-note-repo/journal/`** (e.g. `journal/notes/`). Report PDF sources live under **`.../score-matching-fisher-note-repo/report/`** (e.g. `report/notes/*.tex`, `report/main.tex`). Prefer those absolute paths when this skill is active; edits land in the **notes** Git history.
