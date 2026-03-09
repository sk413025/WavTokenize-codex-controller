---
name: native-handoff
description: Hand off work between native Codex roles without inventing repo-local orchestration protocols.
---

Use this skill when work needs a clean boundary between `default` and another native role.

Workflow:
1. State the task, the owner role, and the expected output.
2. Point the receiving role at the exact files, runs, or manifests it must inspect.
3. Keep the handoff bounded to evidence or edits, not control ownership.
4. For long-running or sequential experiment work, hand off facts, artifacts, or bounded fixes only; keep sequencing and final decisions with `default`.
5. Return a concise result that `default` can integrate directly.

Checks:
- The handoff should work through native Codex collaboration alone.
- Do not introduce repo-side packet artifacts or routing metadata.
