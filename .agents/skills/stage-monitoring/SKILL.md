---
name: stage-monitoring
description: Observe controller stages, logs, and artifacts and report status through packet-driven monitor updates.
---

Use this skill when a run is active and a `monitor` or `default` role needs to summarize stage health without taking queue ownership.

Workflow:
1. Identify the active run, expected stage outputs, and the log or artifact paths to watch.
2. Use `emit-packets` to define the monitoring scope if the monitor task is being delegated.
3. Read stage logs, `state.json`, and any required artifacts before making conclusions.
4. Classify the stage state as:
   - running as expected
   - stalled or missing signal
   - failed with actionable signature
   - completed and ready to advance
5. Use `ingest-agent-result` to record the monitor report, failure signature, or completion evidence.
6. Let `default` or the designated controller owner call `advance-run` when the state is clear.

Checks:
- Monitoring output is evidence-based and points to concrete logs or artifacts.
- Stage status is separated from diagnosis and from queue decisions.
- The monitor role does not silently become the queue owner.
