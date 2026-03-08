---
name: experiment-decomposition
description: Decompose experiment work into native Codex multi-agent tasks before implementation or execution.
---

Use this skill when work spans multiple files, experiment families, or decision points.

Workflow:
1. State the smallest useful deliverable.
2. Split the task into native Codex roles:
   - `default` for final integration
   - `explorer` for read-only repo or run inspection
   - `worker` for bounded edits or execution
   - `monitor` for active runs
   - `analyst` for result comparison or diagnosis
3. Prefer parallel exploration before implementation.
4. Keep ownership explicit by file, family, or run.
5. End with an integration note that says what each delegated role returned.

Checks:
- The split should reduce coordination, not add protocol overhead.
- If Markdown or a skill is enough, do not add runtime code.
