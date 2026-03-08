---
name: controller-decomposition
description: Decompose controller or experiment work into native Codex multi-agent tasks before implementation.
---

Use this skill when a task touches controller architecture, multiple experiment families, or repo-wide workflow changes.

Workflow:
1. Identify the blocking deliverable and the minimum integration path.
2. Split the task into parallelizable read-only exploration, bounded implementation, and validation work.
3. Prefer native role mapping:
   - `planner` or `default` for top-level decomposition
   - `explorer` for repo scans, manifest impact, and legacy family inventory
   - `worker` or `maintainer` for bounded file edits
   - `monitor` for long-running logs or training observation
4. Assign ownership by file or module before delegating.
5. Keep `AGENTS.md` as repo policy and `codex_controller` as lifecycle state, not as a replacement for native harness.
6. Close with an integration pass that states what each sub-agent changed or learned.

Checks:
- At least one concrete parallelization opportunity was considered.
- Repeated workflows are redirected into repo-local skills under `.agents/skills/`.
- Internal `agents/registry.json` is treated as controller metadata, not the runtime execution engine.
