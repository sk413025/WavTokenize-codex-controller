# AGENTS

This file is the official starting point for work in this repo.

Start here first:
- read `AGENTS.md`
- read `docs/project_goal.md`
- read `docs/experiment_policy.md`
- read `docs/research_loop.md`
- for a real official run, use `official-run-ladder`
- treat `docs/codex_controller.md` as implementation notes only

Higher-priority system, developer, and user instructions still apply.

## AGENTS-First
- This repository is Codex-first and agent-native.
- The official start path is: `AGENTS.md` -> repo docs -> repo skills -> manifests/adapters -> `codex_controller` when execution or status persistence is needed.
- If `AGENTS.md` conflicts with lower-level repo docs or metadata, `AGENTS.md` wins.

## Single Golden Session
- Official work begins from a Codex-started session that reads `AGENTS.md` first.
- `Codex(default)` is the top-level controller and final integrator.
- Native multi-agent delegation is allowed, but the Codex session stays authoritative.
- `codex_controller` records run facts only; it does not become the operational source of truth.

The concise reference path is in `docs/golden_session.md`.

## Native-First Decision Policy
Before adding any controller, workflow, or agent-native feature, decide in this order:
1. Can `AGENTS.md` solve it as policy?
2. Can native Codex roles or sub-agents solve it?
3. Can an existing repo-local skill solve it?
4. Can a Markdown document or checklist explain the workflow well enough for Codex to follow it without new code?
5. Does the repo truly need thin persistent experiment state?

Only when the first four are insufficient may the repo grow new controller code.

Default preference order:
- policy in `AGENTS.md`
- repo docs
- native Codex harness
- repo-local skill
- manifest or adapter contract
- thin runtime change in `codex_controller`

## Markdown-First Rule
Prefer:
- `AGENTS.md`
- repo docs
- repo-local skills
- Markdown rules, checklists, and workflow notes

Prefer explaining the workflow to Codex over encoding the workflow in new Python.
If Codex can reliably operate the project through policy, docs, a skill, and a Markdown checklist, do not add runtime code.

## Top-Level Control
Use native Codex roles first:
- `default`: top-level controller and final decision-maker
- `planner`: decomposition and parallelization
- `explorer`: read-only repo and run inspection
- `worker`: bounded implementation or execution
- `monitor`: run/log/artifact observation
- `analyst`: result comparison and diagnosis
- `maintainer`: bounded diagnosis-driven fixes

## Do Not Rebuild Native Capabilities
Do not implement repo-local replacements for capabilities that Codex already provides natively:
- top-level planning
- sub-agent lifecycle management
- generic queue or approval systems
- generic monitor schedulers
- generic role routers
- generic skill dispatchers
- another agent platform layered over Codex

Allowed repo-local additions are limited to project-specific items:
- research manifests and adapters
- project-specific preflight and diagnosis entrypoints
- thin run state and monitor facts
- project-specific experiment code
- project-specific skills
- durable family notes and run summaries

Code additions must meet at least one of these conditions:
- they are directly required for a project-specific experiment pipeline
- they are directly required to support a repo-local skill and are described by that skill's `SKILL.md`
- they provide thin durable experiment state that cannot be replaced by policy, docs, skills, or Markdown guidance

## Core Skills First
Use repo-local skills under `.agents/skills/` for repeated workflows.

Core skills:
- `experiment-decomposition`
- `official-run-ladder`
- `stage-monitoring`
- `run-diagnosis`
- `result-comparison`
- `followup-generation`
- `codex-native-review`

Supporting skills:
- `manifest-authoring`
- `legacy-adapter-onboarding`
- `experiment-promotion`
- `native-handoff`

Compatibility skills may remain on disk, but the list above is the official surface.

## Official Control Surface
Official work starts from these layers:
- `AGENTS.md`
- `docs/project_goal.md`
- `docs/experiment_policy.md`
- `docs/research_loop.md`
- `.agents/skills/`
- `experiments/manifests/*.json`
- `experiments/adapters/*.json`

`codex_controller` is a helper after that start. Use it only for manifest validation, execution, monitoring, resume, and status inspection.

Secondary machine-readable records:
- `knowledge/*`
- `controller_runs/<run_id>/`

## Change Acceptance Gate
Any controller or agent-native architecture change must record these five fields in the proposal, plan, or review:
- `native_capability_used`
- `why_native_is_not_enough`
- `repo_specific_gap`
- `minimal_added_surface`
- `what_not_to_build`

After any code-changing turn, `Codex(default)` must also record:
- `core_goal_alignment`
- `decision`

The post-change flow is:
1. run `codex-native-review`
2. run Codex native review in the Review pane or with `/review`
3. use this repo's checklist to judge whether the change stays Codex-first
4. only then treat the work as complete

Review outcomes that must fail in native review:
- the change creates another primary start path besides `AGENTS.md`
- the change rebuilds native Codex capabilities
- the change adds generic infrastructure that does not serve the research goal
- the change encodes workflow logic in Python when `AGENTS.md`, a skill, or Markdown guidance would have been sufficient

## Run Lifecycle
Use this as a Codex-side checklist, not a repo-side state machine:
- `plan`
- `prepare`
- `execute`
- `monitor`
- `analyze`
- `diagnose`
- `propose_next`

## Runtime Contract
Every official run should keep a small durable record:
- `state.json`
- `manifest.snapshot.json`
- per-stage logs
- `events.jsonl`
- `monitor_report.json`
- `metrics.json`
- `analysis.json`
- `diagnosis.json`

Anything beyond that is optional and should not become another control layer.

## Mutation Policy
Codex may directly modify:
- `AGENTS.md`
- `.codex/*`
- `.agents/*`
- `docs/*`
- `knowledge/*`
- `codex_controller/*`
- `experiments/*`
- explicitly onboarded experiment families referenced by official manifests or adapter contracts

Codex should not mutate unrelated legacy experiment families unless a manifest or diagnosis path explicitly points to them.

## Legacy Migration Policy
A legacy experiment family becomes official only after it has:
- an adapter contract in `experiments/adapters/`
- a manifest in `experiments/manifests/`
- clear artifact and completion rules
- run-local state and logs
- a path to be operated through native Codex planning and skills

Until then, treat it as legacy and do not assume safe full automation.
