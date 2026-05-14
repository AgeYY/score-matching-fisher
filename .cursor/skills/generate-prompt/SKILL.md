---
name: generate-prompt
description: >-
  When the user wants a handoff for another AI, produce one Markdown code fence (tag text) they can paste in
  one action. Default: a short, self-contained **Goal** only—no chat history, no long repo tours—so
  the other model does not need this conversation. Add repo/env/file hints only if the user
  explicitly asks for extra context. Use when the user says generate prompt, handoff prompt, or
  prompt for another model.
---

# generate-prompt

## When to apply

The user wants a **copy-paste prompt** for **another AI** (or a new session). The prompt should stand alone: **state the goal clearly** so the reader never needs this chat or prior turns.

## Output shape

1. **One markdown code fence** labeled `text` (or plain triple-backtick fence) containing the **entire** prompt to copy in one selection.
2. **Do not** put list markers on the same line as the opening fence (breaks rendering).
3. Inside the block: **plain text**. Be concise; no long code dumps unless the user pasted them.

## What to put inside the prompt (default)

**Goal only (preferred).** A few imperative sentences that fully describe what to do or deliver: scope, inputs/outputs, naming (e.g. method or flag names), and any definition that would otherwise live in chat. Write so a stranger can act without asking follow-ups.

Optional **one line** after the goal only if the user asked for it or the goal truly cannot be understood without it—for example: `Repo: score-matching-fisher.` or `Use mamba run -n geo_diffusion per AGENTS.md.` Do **not** add repo tours, “current behavior”, file inventories, acceptance-criteria headings, or conversation recap unless the user **explicitly** requests that level of detail.

## What to omit (unless the user asks)

- This conversation, prior messages, and “as we discussed”.
- Long lists of files, CLIs, or symbols (the other model can search the repo).
- Separate sections for constraints, acceptance criteria, or environment—unless the user wants them.

## Style

- Imperative (“Implement…”, “Add…”, “Define…”).
- Prefer one tight **Goal** block over many small headings inside the fence.

## Example (goal-only)

```text
Goal: …
```

## After delivering

- Say the fence is **ready to copy in one selection**.
- If the user **only** asked for a prompt, **do not** edit application code or unrelated docs—output the block only.
