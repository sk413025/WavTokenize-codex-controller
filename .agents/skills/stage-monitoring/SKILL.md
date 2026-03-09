---
name: stage-monitoring
description: Observe stage logs and artifacts for an active run and summarize the result for Codex.
---

Use this skill when a run is active or when a completed run needs a concrete monitor summary.

Workflow:
1. Collect live evidence first when available:
   - process state
   - GPU/process occupancy
   - durable run artifacts such as `state.json`, `monitor_report.json`, stage logs, `history.json`, and checkpoint paths
2. If evidence conflicts, prefer live process state and durable artifacts over prior sub-agent summaries.
3. Classify each stage as planned, running, stalled, failed, or completed.
4. Record concrete evidence:
   - log markers
   - missing artifacts
   - checkpoint readiness
   - stalled log timing
5. When monitoring a paired long-running launch, keep watching until the run reaches a terminal class or explicit sequencing loss.
6. When the run reaches a decision boundary, emit a user-visible event summary that includes:
   - event type
   - source run or artifact path
   - concrete live evidence
   - whether the current launch contract allows one bounded next step
   - the next required owner (`default`, `run-diagnosis`, or `analyst`)
7. Return a concise Markdown summary for `default` or `analyst`.

Checks:
- Monitoring reports facts only.
- For `clean_success`, the monitor may say that one declared next step is now eligible, but it must not launch that step itself.
- When a run looks stalled or ambiguous, say so explicitly and point `default` at `run-diagnosis`.
- If expected sequencing is no longer live, report `sequencing no longer active` as a control-surface fact.
- Promotion or follow-up decisions remain with `Codex(default)`.
