---
name: run-diagnosis
description: Diagnose controller-managed runs and turn failures or weak results into patch-ready findings.
---

Use this skill when reviewing `controller_runs/<run_id>/` after failure, unmet criteria, or suspicious output.

Workflow:
1. Read `state.json`, `analysis.json`, `diagnosis.json`, `decision_log.jsonl`, and stage logs.
2. Classify the outcome:
   - execution failure
   - artifact failure
   - weak result against baseline
   - successful candidate
3. Separate symptom, probable cause, and required action.
4. If a patch is warranted, produce a bounded patch request tied to the run.
5. If a rerun or new experiment is better, specify the manifest delta instead of vague advice.
6. Promote durable findings into `knowledge/` only when they generalize beyond one run.

Checks:
- Diagnosis distinguishes controller failure from model-result failure.
- Recommended actions are executable by a worker or maintainer agent.
- Baseline comparison is explicit when available.
