---
name: run-diagnosis
description: Diagnose persisted runs and turn failures or weak results into bounded next actions.
---

Use this skill after a failed run, a stalled run, or a completed run with questionable results.

Workflow:
1. Read `state.json`, `analysis.json`, `diagnosis.json`, `monitor_report.json`, and stage logs.
2. Classify the outcome:
   - execution failure
   - stalled or missing artifact
   - completed but weak result
   - healthy candidate
3. Separate symptom, probable cause, and next action.
4. Prefer a Markdown conclusion or manifest change before proposing new runtime code.
5. If a code change is needed, keep it bounded to the experiment family or adapter.

Checks:
- Diagnosis should be evidence-based.
- The proposed fix should stay project-specific.
