---
name: followup-generation
description: Turn a run outcome into a bounded next experiment step without building another controller layer.
---

Use this skill only after `run-diagnosis` or `result-comparison` has already produced a bounded conclusion.

Workflow:
1. Read `docs/project_goal.md`, `docs/experiment_policy.md`, the run summary, the diagnosis or comparison conclusion, and the relevant manifest.
2. Decide whether the next step is:
   - rerun with a bounded manifest change
   - move to the next ladder tier
   - stop and record the lesson
   - propose a small experiment-code change in the same family
3. Write the proposal as Markdown first.
4. Only edit a manifest if the next step is concrete and execution-specific.

Checks:
- Do not perform first-pass diagnosis or first-pass result judgment here; consume those inputs from `run-diagnosis` or `result-comparison`.
- Do not auto-generate a new controller protocol.
- Do not propose generic infrastructure work as a follow-up.
