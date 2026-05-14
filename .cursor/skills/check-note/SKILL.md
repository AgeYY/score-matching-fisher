---
name: check-note
description: >-
  In score-matching-fisher, searches Markdown notes under journal/notes/ and
  report/notes/ in the sibling notes repo (or legacy in-repo journal/) before
  answering questions about experiments, methods, benchmarks, or prior write-ups.
---

# check-note

## Scope

| Location | Role |
|----------|------|
| `{NOTES_ROOT}/journal/notes/` | Dated and topic `.md` experiment logs, methods, plans (primary). |
| `{NOTES_ROOT}/report/notes/` | Same intent for report-facing material; **many entries are `.tex`** (PDF narrative). Search **`*.md` here** when present; if nothing matches, say so and optionally point to a related `.tex` by topic/date. |

**Resolve `{NOTES_ROOT}`** using the same rules as **write-md-journal** (parent of `journal/`): monorepo if `<workspace>/journal` exists; else sibling `../score-matching-fisher-note/score-matching-fisher-note-repo` when that tree exists; else `score-matching-fisher-note-repo` under the outer note workspace.

## Search workflow (agent)

1. **Narrow the query**: extract keywords (method names, dataset aliases, script names, `bin/...`, dates `YYYY-MM-DD`).
2. **Search both trees** (prefer fast exact match, then semantic search if needed):
   - `rg -i '<keywords>' "${NOTES_ROOT}/journal/notes" "${NOTES_ROOT}/report/notes" --glob '*.md'`
   - Or Glob `{NOTES_ROOT}/journal/notes/**/*.md` / `{NOTES_ROOT}/report/notes/**/*.md` and filter by filename (dated notes: `2026-*-*.md`).
3. **Open the best 1–3 files** (Read tool); prefer the newest dated note when several cover the same experiment.
4. **Use the content**: summarize or quote what the note already established; **cite the full path** so the user can open it (e.g. `/grad/zeyuan/score-matching-fisher-note/score-matching-fisher-note-repo/journal/notes/...`).
5. **If no `.md` hits**: state that clearly; for `report/notes/`, mention that narrative may live in `.tex` only and search `{NOTES_ROOT}/report/notes/*.tex` if the user cares about the PDF-side write-up.

## Conventions (this repo)

- Journal Markdown math: inline `$...$`, display `$$` on their own lines (see `AGENTS.md` in the **code** repo).
- Dated note filenames often start with `YYYY-MM-DD-` for chronological scanning.

## Do not

- Invent citations to notes that were not found in those directories.
- Skip searching when the user’s question is clearly about “what we documented” or reproducing a prior study—run the search first, then answer.
