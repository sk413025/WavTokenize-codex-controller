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
- `experiments/registry.json`
- run-local ledger directories when an official execution persists them
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
- small project-specific utility modules that support experiment families without taking over control flow

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

## Acceptable Tooling Code
Some repo code exists primarily as a thin tool surface. This is acceptable when it stays project-specific and does not replace native Codex control.

Reasonable examples in this repo include:
- `codex_controller/*` for official manifest execution facts, status, and persisted run records
- `experiments/registry.json` as a thin index of official, dependency, eval, and compatibility families
- `utils/hypothesis_reporting.py` for thin run-local summaries on one-off hypothesis lines
- `utils/train_runtime.py` and `utils/audio_losses.py` for shared experiment-family helpers that reduce duplicated training code without introducing workflow logic

These helpers should not become:
- a second orchestration layer
- a policy engine for sequencing, promotion, or approvals
- a generic training framework that hides family-specific research behavior

## Tooling Boundary Gate
Treat code-as-tool as acceptable only when all of these stay true:
- the tool persists project-specific facts or removes repeated family boilerplate
- the next step is still chosen by `AGENTS.md`, docs, skills, manifests, or `Codex(default)`
- the tool does not own role lifecycle, routing, queueing, approval, or generic sequencing
- the tool can be justified more narrowly than "this would be convenient to automate"

Treat the tooling boundary as failed when a helper starts to:
- infer or choose follow-up actions on its own
- encode session policy that belongs in Markdown or a skill
- generalize into a repo-wide framework with weak ties to a specific experiment path
- compete with native multi-agent orchestration or native monitoring

## Governance Completion Rule
For a governance refinement pass, do not stop at "the wording feels cleaner."

Treat the pass as complete only when all of these are true:
- the start path is still singular and `AGENTS.md`-first
- research-success review is still more important than execution success alone
- core skills remain bounded Markdown playbooks with clear ownership limits
- code-as-tool surfaces are still thin and project-specific

Treat the pass as incomplete when:
- one of these areas is still only `partial`
- the strongest counterargument has not been answered
- the monitor cannot show that docs, skills, and tooling were all audited together

When incomplete, prefer another bounded docs or skills refinement before considering new runtime code.

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
