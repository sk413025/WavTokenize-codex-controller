---
name: result-comparison
description: Compare a completed run against prior family evidence and summarize whether it is worth promoting or iterating.
---

Use this skill after a run has completed cleanly enough to produce usable evidence.

Workflow:
1. Read `docs/project_goal.md`, `analysis.json`, `monitor_report.json`, `metrics.json`, and any run-specific comparison artifacts such as `baseline_comparison.json`, `objective_eval.json`, or family notes when they exist.
2. Compare the run against the current family notes and any maintained baseline or comparison artifact.
3. Write a concise Markdown conclusion:
   - what improved
   - what regressed
   - whether the run is only an execution success or a real research candidate
4. If more evidence is required, say exactly what is missing.

Checks:
- Use this skill for successful or otherwise usable completed evidence, not for stalled or failed runs.
- Do not turn comparison into automatic promotion logic.
- Keep the conclusion tied to concrete artifacts and logs.
