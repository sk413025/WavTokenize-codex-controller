---
name: stage-monitoring
description: Observe stage logs and artifacts for an active run and summarize the result for Codex.
---

Use this skill when a run is active or when a completed run needs a concrete monitor summary.

Workflow:
1. Read `state.json`, `monitor_report.json` if present, stage logs, and required artifact paths.
2. Classify each stage as planned, running, stalled, failed, or completed.
3. Record concrete evidence:
   - log markers
   - missing artifacts
   - checkpoint readiness
   - stalled log timing
4. Return a concise Markdown summary for `default` or `analyst`.

Checks:
- Monitoring reports facts only.
- Promotion or follow-up decisions remain with `Codex(default)`.
