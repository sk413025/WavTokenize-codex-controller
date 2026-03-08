---
name: dispatch-handoff
description: Hand off controller work through packet-driven dispatch without breaking default-role queue ownership.
---

Use this skill when a controller task needs to move from `default` to another native role and back through a documented packet boundary.

Workflow:
1. Confirm the current run, controller node, and why a handoff is needed.
2. Use `prepare-run` if run state, snapshots, or ownership records are missing or stale.
3. Use `emit-packets` to create a bounded packet that names:
   - target role
   - stage or node
   - required artifacts
   - completion signal
4. Keep queue intent explicit but advisory unless the acting role is `default`.
5. Use `ingest-agent-result` to merge the returned packet into controller-owned state and decisions.
6. Use `advance-run` only after the packet result is complete enough to move the graph forward.

Checks:
- The packet boundary is clear enough for another role to execute without hidden context.
- Handoff records distinguish execution output from queue decisions.
- Queue mutation stays with `default`.
