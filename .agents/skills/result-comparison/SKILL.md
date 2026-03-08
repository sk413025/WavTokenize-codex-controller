---
name: result-comparison
description: Compare a completed run against prior family evidence and summarize whether it is worth promoting or iterating.
---

Use this skill after a run has completed and produced logs, monitor output, and any metrics.

Workflow:
1. Read `docs/project_goal.md`, `analysis.json`, `monitor_report.json`, and any run-specific metrics.
2. Compare the run against the current family notes and any manually maintained baseline.
3. Write a concise Markdown conclusion:
   - what improved
   - what regressed
   - whether the run is only an execution success or a real research candidate
4. If more evidence is required, say exactly what is missing.

Checks:
- Do not turn comparison into automatic promotion logic.
- Keep the conclusion tied to concrete artifacts and logs.
