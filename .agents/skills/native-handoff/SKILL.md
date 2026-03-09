---
name: native-handoff
description: Hand off work between native Codex roles without inventing repo-local orchestration protocols.
---

Use this skill when work needs a clean boundary between `default` and another native role.

Workflow:
1. State the task, the owner role, and the expected output.
2. Point the receiving role at the exact files, runs, or manifests it must inspect.
3. Keep the handoff bounded to evidence or edits, not control ownership.
4. For any long-running launch, create a paired-monitor handoff at launch time instead of waiting for a later manual check.
5. The paired-monitor handoff must state:
   - the run reference (run dir, manifest run id, or process id)
   - the autonomy window
   - the one allowed next step, if any
   - that terminal decisions still return to `default`
6. For long-running or sequential experiment work, state whether the current task is inside an active autonomy window and whether that window ends with this handoff.
7. Hand off facts, artifacts, or bounded fixes only; keep sequencing and final decisions with `default`.
6. Return a concise result that `default` can integrate directly.

Checks:
- The handoff should work through native Codex collaboration alone.
- Do not introduce repo-side packet artifacts or routing metadata.
- Do not let the receiving role infer a new next step that was not declared in the launch contract.
