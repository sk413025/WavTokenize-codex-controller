# Architecture Guardrails

Start in `AGENTS.md`.

This document defines the long-term boundary between native Codex capabilities and repo-local code.

## Layer Model

### Layer 1: Native Codex Control
This is the default place for behavior.
- `Codex(default)` as top-level controller
- native roles and sub-agents
- native multi-agent orchestration
- native skill triggering and selection

Do not reimplement this layer in the repo.

### Layer 2: Repo Policy And Skills
This layer tells Codex how to work in this project.
- repo `AGENTS.md`
- `.codex/config.toml`
- `.agents/skills/*`

Use this layer for workflow rules, decomposition patterns, preflight rules, diagnosis patterns, and review checklists.

### Layer 3: Thin Runtime
This layer persists project-specific research state.
- `codex_controller/*`
- `experiments/manifests/*.json`
- `experiments/adapters/*.json`
- `controller_runs/<run_id>/`
- `knowledge/*`

Use this layer only for durable experiment state, manifests, adapters, and project-specific lifecycle records.

## What The Repo May Add
Allowed additions:
- project-specific manifests and adapters
- project-specific preflight and diagnosis entrypoints
- project-specific skills
- thin run ledger and analysis artifacts
- experiment-family migration logic

## What The Repo Must Not Add
Forbidden additions:
- another agent platform
- another queue owner model
- generic approval frameworks
- generic planner or scheduler runtimes
- generic sub-agent lifecycle systems
- generic monitor frameworks when native monitor usage is enough
- repo-local wrappers for native skills or role selection that add no project-specific value

## Decision Rule
For any new controller or agent-native feature, answer:
- `native_capability_used`
- `why_native_is_not_enough`
- `repo_specific_gap`
- `minimal_added_surface`
- `what_not_to_build`

If the answer is weak, the change should not become new runtime code.

## Preferred Escalation Path
When Codex needs more structure, escalate in this order:
1. tighten `AGENTS.md`
2. add or improve a repo-local skill
3. update a manifest or adapter
4. add the smallest possible thin-runtime change

This order is the main defense against overengineering.
