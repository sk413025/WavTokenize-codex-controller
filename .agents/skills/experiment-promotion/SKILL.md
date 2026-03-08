---
name: experiment-promotion
description: Review a completed run and decide whether it should become a family baseline.
---

Use this skill after a run has passed execution gates and has concrete evidence worth comparing.

Workflow:
1. Read `docs/project_goal.md`, run artifacts, and any existing baseline notes.
2. Compare the run against the current family baseline.
3. Record a Markdown conclusion:
   - promote
   - keep as candidate
   - reject as regression
4. Update `knowledge/best_runs.json` only after explicit review.

Checks:
- Promotion must be evidence-based.
- The runtime should not auto-promote for you.
