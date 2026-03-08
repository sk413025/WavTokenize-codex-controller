# AGENTS

This file is the official starting point for work in this repo.

Start here first:
- Read `AGENTS.md` before using `codex_controller`, controller docs, or repo-local skills.
- Treat `AGENTS.md` as repo policy.
- Treat `docs/codex_controller.md` as implementation notes.

Higher-priority system, developer, and user instructions still apply.

## AGENTS-First
- This repository is Codex-first and agent-native.
- The official control surface starts with `AGENTS.md`, then `.codex/config.toml`, then `.agents/skills/`, then `codex_controller/*`.
- If `AGENTS.md` conflicts with lower-level repo docs or metadata, `AGENTS.md` wins.

## Single Golden Session
- Official work begins from a Codex-started session that reads `AGENTS.md` first.
- That session is the single golden session for planning, delegation, integration, and final queue decisions.
- Delegation is allowed, but the golden session stays authoritative:
  - `default` owns queue mutation and official run finalization
  - other roles return bounded results through packets and handoffs
  - controller state records the work, but does not replace the Codex-started session as the source of operational truth

The concise reference path is in `docs/golden_session.md`.

## Native Harness
The official Codex-native harness for this repo is:
- `AGENTS.md`
- `.codex/config.toml`
- `.agents/skills/*/SKILL.md`
- `codex_controller/*`

`codex_controller` is the experiment lifecycle layer. It does not replace native Codex roles, repo-local skills, or higher-level Codex instructions.

## Native-First Decision Policy
Before adding any controller, workflow, or agent-native feature, decide in this order:
1. Can `AGENTS.md` solve it as policy?
2. Can native Codex roles or sub-agents solve it?
3. Can an existing repo-local skill solve it?
4. Does it require thin persistent repo state for this research project?

Only when the first three are insufficient may the repo grow new controller code.

Default preference order:
- policy in `AGENTS.md`
- native Codex harness
- repo-local skill
- manifest or adapter contract
- thin runtime change in `codex_controller`

## Top-Level Control
- `Codex` running in the `default` role is the top-level controller, integrator, and sole queue owner for official workflows.
- Before substantial work, decompose it into agent-executable units and prefer parallel delegation when ownership boundaries are clear.
- Only `default` may queue, dequeue, or finalize official follow-up work.
- Native Codex roles should be used first:
  - `default`: top-level controller, integrator, queue owner
  - `planner`: decomposition and parallelization specialist
  - `explorer`: read-only repo scans, impact analysis, baseline review
  - `worker`: bounded implementation
  - `maintainer`: bounded diagnosis-driven patches
  - `monitor`: long-running run, log, and artifact observation
  - `analyst`: result classification, baseline comparison, next-step recommendations

## Do Not Rebuild Native Capabilities
Do not implement repo-local replacements for capabilities that Codex already provides natively:
- top-level planning that should stay with `default`
- sub-agent lifecycle management
- generic queue or approval systems
- generic monitor schedulers
- generic role routers
- generic skill dispatchers
- another agent platform layered over Codex

Allowed repo-local additions are limited to project-specific items:
- research manifests
- experiment adapters
- run state and analysis artifacts
- project-specific skills
- project-specific preflight and diagnosis logic

## Core Skills First
Use repo-local skills under `.agents/skills/` for repeated workflows.

Start with these core five:
- `controller-decomposition`
- `dispatch-handoff`
- `stage-monitoring`
- `run-diagnosis`
- `followup-generation`

Use these when the task specifically needs them:
- `manifest-authoring`
- `legacy-adapter-onboarding`
- `experiment-promotion`
- `codex-native-review`

Existing `.claude/skills/*` assets are compatibility references only.

## Dispatch Kernel Policy
Official native dispatch is packet-driven. The policy-stable command set is:
- `prepare-run`
- `emit-packets`
- `ingest-agent-result`
- `advance-run`
- `finalize-run`

Packet rules:
- Packets are the official boundary for controller handoff, stage status, and result ingestion.
- Non-`default` roles may emit observations and proposals, but queue mutation remains `default`-owned.
- `queue_next` remains a visible controller node, but only `default` may complete it for official runs.

## Internal Controller Agents
`agents/registry.json` defines controller-layer agent identities for run state, handoff records, and decision logs.

Internal controller agents must map onto native Codex roles instead of replacing them:
- `codex` -> `default`
- `planner` -> `planner` when available, otherwise `explorer`
- `executor` -> `worker`
- `monitor` -> `monitor`
- `analyst` -> `analyst` when available, otherwise `explorer`
- `maintainer` -> `maintainer` when available, otherwise `worker`

If `AGENTS.md` and `agents/registry.json` drift, `AGENTS.md` is the repo policy source of truth and `agents/registry.json` must be updated.

## Official Control Surface
Official work starts from these four layers:
- `AGENTS.md`
- `.codex/config.toml`
- `.agents/skills/`
- `codex_controller/*`

Secondary controller inputs and records are used after that start:
- `experiments/manifests/*.json`
- `experiments/adapters/*.json`
- `knowledge/*`
- `controller_runs/<run_id>/`

Controller metadata such as `agents/registry.json` supports runtime mapping but is not a primary entrypoint.

## Change Acceptance Gate
Any controller or agent-native architecture change must record these five fields in the proposal, plan, or review:
- `native_capability_used`
- `why_native_is_not_enough`
- `repo_specific_gap`
- `minimal_added_surface`
- `what_not_to_build`

If the change cannot justify all five fields, prefer a policy, skill, or manifest update instead of new runtime code.

## Closed-Loop Workflow
Official runs follow the controller graph:
- `plan`
- `prepare`
- `execute`
- `monitor`
- `analyze`
- `diagnose`
- `patch`
- `propose_next`
- `queue_next`

Graph nodes may be skipped by policy, but they must remain visible in controller state.

## Runtime Contract
Every official run must have:
- `state.json`
- `analysis.json`
- `diagnosis.json`
- `patch_request.json`
- `next_manifest.json`
- `controller_actions.jsonl`
- `agent_packets/`
- `agent_results/`
- `events.jsonl`
- `decision_log.jsonl`
- `agent_handoffs.jsonl`
- `manifest.snapshot.json`
- per-stage logs

`dispatch_plan.json` may exist as an internal debug artifact. It is not part of the primary repo-facing start path.

## Mutation Policy
Codex may directly modify:
- `AGENTS.md`
- `.codex/*`
- `.agents/*`
- `agents/*`
- `knowledge/*`
- `codex_controller/*`
- `experiments/*`
- explicitly onboarded experiment families referenced by official manifests or adapter contracts

Codex should not mutate unrelated legacy experiment families unless a manifest or diagnosis policy explicitly points to them.

## Legacy Migration Policy
A legacy experiment family becomes official only after it has:
- an adapter contract in `experiments/adapters/`
- a manifest in `experiments/manifests/`
- clear artifact and completion rules
- controller-owned run state
- a path to be operated through native Codex planning and skills

Until then, treat it as legacy and do not assume safe full automation.
