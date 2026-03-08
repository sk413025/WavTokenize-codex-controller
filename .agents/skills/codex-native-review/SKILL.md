---
name: codex-native-review
description: Review a proposed controller, workflow, or agent-native change to ensure it stays Codex-first, uses native harness first, and avoids rebuilding native capabilities.
---

Use this skill before adding or expanding controller, orchestration, or agent-native architecture in this repo.

Workflow:
1. State the proposed change in one sentence.
2. Identify which native Codex capability is already available:
   - `default` control
   - sub-agents
   - multi-agent orchestration
   - skills
   - existing repo policy or manifest support
3. Fill these five fields:
   - `native_capability_used`
   - `why_native_is_not_enough`
   - `repo_specific_gap`
   - `minimal_added_surface`
   - `what_not_to_build`
4. Add two required outcome fields:
   - `core_goal_alignment`
   - `decision`
5. Reject the change if it mainly recreates native planning, routing, approval, queueing, or monitoring.
6. Prefer the smallest acceptable layer:
   - `AGENTS.md`
   - skill
   - Markdown checklist or workflow note
   - manifest or adapter
   - thin runtime code
7. End with one of these outcomes:
   - `policy_only`
   - `skill_only`
   - `manifest_or_adapter_only`
   - `thin_runtime_change_allowed`
   - `reject_as_overengineering`
8. After any code change, apply this checklist during Codex native review in the Review pane or with `/review`.

Checks:
- `Codex(default)` stays the top-level controller.
- The repo start path still begins with `AGENTS.md`.
- The proposal does not create another agent platform inside the repo.
- The added surface is project-specific and minimal.
- The change still serves the research goal instead of adding generic infrastructure.
- If code was added, it is either project-specific experiment code or clearly in support of a repo-local skill with matching Markdown documentation.
- If Markdown plus a skill would have been enough, reject the code change as overengineering.
