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
- Markdown workflow notes and checklists

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
- Markdown instructions that help Codex operate the repo without new runtime code
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

## Markdown-First Policy
Prefer repo-local Markdown plus skills over Python wherever possible.

Use Markdown for:
- workflow rules
- review checklists
- decomposition guidance
- migration playbooks
- experiment operating notes

Only add Python when Markdown, `AGENTS.md`, and skills are not enough for Codex to operate the project correctly.

If Python is still required, it should either:
- support a project-specific experiment path, or
- support a repo-local skill and be documented by that skill

## Preferred Escalation Path
When Codex needs more structure, escalate in this order:
1. tighten `AGENTS.md`
2. add or improve a repo-local skill
3. update a manifest or adapter
4. add the smallest possible thin-runtime change

This order is the main defense against overengineering.

## Native Review
All code changes must go through Codex native review in the Review pane or with `/review`.

This repo adds only a project-specific checklist through `codex-native-review`.
It must not add a second review platform, a review ledger, or a mechanical review gate that duplicates Codex native review.
