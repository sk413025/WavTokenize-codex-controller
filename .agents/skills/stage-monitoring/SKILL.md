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
5. Return a concise Markdown summary for `default` or `analyst`.

Checks:
- Monitoring reports facts only.
- When a run looks stalled or ambiguous, say so explicitly and point `default` at `run-diagnosis`.
- Promotion or follow-up decisions remain with `Codex(default)`.
