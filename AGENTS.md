# AGENTS.md — AI Assistant Rules (local only, not tracked in git)

## Git Rules

- **NEVER push to remote** unless the user explicitly says "push", "push now", "open a PR", or similar.
- **NEVER create a PR** unless explicitly asked.
- **NEVER force-push** unless explicitly asked.
- **NEVER commit (git commit)** unless the user explicitly says "commit", "commit now", or similar.
- Ask before committing, pushing, or opening a PR.
- Never use emojis

## Workflow

- Default branch for new work: branch off `dev`.
- PR flow (when asked): feature-branch → `dev` → `main`.
- Keep commits atomic and well-described.

## Cloud Cost Rules

- **NEVER run cloud API benchmarks at scale without explicit user approval.** Document AI costs $1.50/1K pages + Gemini API per-token charges — running on 100 images × 10 runs = significant cost.
- **Before any live cloud run, estimate and report the cost** (e.g., "This will process N pages at ~$X. Proceed?").
- **Limit live runs** to `--num-runs 1` and a small subset of test images unless the user says otherwise.
