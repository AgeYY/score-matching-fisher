---
name: check-note
description: >-
  In score-matching-fisher, searches and applies Markdown notes under journal/notes/ and
  report/notes/ before answering questions about experiments, methods, benchmarks, or prior
  write-ups. Use when the user asks to check journal notes, report notes, prior documentation,
  what was written about a topic, or to align answers with existing .md notes; also when
  discussing results that may already be summarized in those directories.
---

# check-note

## Scope

| Location | Role |
|----------|------|
| `journal/notes/` | Dated and topic `.md` experiment logs, methods, plans (primary). |
| `report/notes/` | Same intent for report-facing material; **many entries are `.tex`** (PDF narrative). Search **`*.md` here** when present; if nothing matches, say so and optionally point to a related `.tex` by topic/date. |

Paths from repo root: `journal/notes/`, `report/notes/`.

## Search workflow (agent)

1. **Narrow the query**: extract keywords (method names, dataset aliases, script names, `bin/...`, dates `YYYY-MM-DD`).
2. **Search both trees** (prefer fast exact match, then semantic search if needed):
   - `rg -i '<keywords>' journal/notes report/notes --glob '*.md'`
   - Or Glob `journal/notes/**/*.md` / `report/notes/**/*.md` and filter by filename (dated notes: `2026-*-*.md`).
3. **Open the best 1–3 files** (Read tool); prefer the newest dated note when several cover the same experiment.
4. **Use the content**: summarize or quote what the note already established; **cite the file path** so the user can open it (`journal/notes/...` or `report/notes/...`).
5. **If no `.md` hits**: state that clearly; for `report/notes/`, mention that narrative may live in `.tex` only and search `report/notes/*.tex` if the user cares about the PDF-side write-up.

## Conventions (this repo)

- Journal Markdown math: inline `$...$`, display `$$` on their own lines (see `AGENTS.md`).
- Dated note filenames often start with `YYYY-MM-DD-` for chronological scanning.

## Do not

- Invent citations to notes that were not found in those directories.
- Skip searching when the user’s question is clearly about “what we documented” or reproducing a prior study—run the search first, then answer.
