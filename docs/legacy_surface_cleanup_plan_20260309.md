# Legacy Surface Cleanup Plan

Start in `AGENTS.md`.

Date: 2026-03-09
Scope: stable worktree active surface cleanup for historical family experiment code
Owner: `Codex(default)`

## Goal-First Contract

This plan does not target blind deletion of old family code.
It targets a narrower outcome:

- reduce planning contamination from historical experiment surfaces
- preserve active import, checkpoint, and evaluation continuity
- keep historical evidence reproducible through archive and documented pointers
- keep completion evidence-based instead of appearance-based

## Success Goals

`goal_1_active_surface_reduced`
- historical experiment residue that is not part of the live execution surface is moved out of the active planning surface or explicitly marked as archived

`goal_2_live_dependencies_preserved`
- official, dependency, and eval families continue to resolve imports, checkpoints, and evaluator entrypoints after cleanup

`goal_3_history_preserved`
- removed or hidden historical material remains recoverable via archive paths, git history, or migration notes

`goal_4_evidence_complete`
- each cleanup step is backed by direct repo evidence, not only by preference for a cleaner tree

## Current Evidence Baseline

Confirmed live dependencies still exist:

- `families/official/material_generalization/data_material_aug.py`
- `families/deps/encoder_aug/train_augmented.py`
- `families/deps/no_vq_core/models_no_vq.py`
- `families/eval/decoder_lora_eval/generate_test_samples.py`

Historical compat removals remain recoverable through git history at tag
`pre-compat-legacy-cleanup`.

## Native Role Split

`default`
- sets goals
- owns final stop/go decisions
- integrates role outputs

`explorer-dependency-map`
- maps live import, checkpoint, and evaluator edges into legacy surfaces
- identifies safe and unsafe cleanup boundaries

`worker-phase1-cleanup`
- performs only safe cleanup of non-imported residue
- allowed targets: archived cold artifacts, stray logs, and other evidence-backed non-code residue
- blocked from touching active compat, deps, eval, or checkpoint-default surfaces

`worker-phase2-extraction`
- extracts still-imported legacy primitives into minimal dependency modules
- runs only after dependency mapping confirms a bounded extraction target

`analyst-regression-check`
- verifies cleanup did not break import, checkpoint, or eval continuity

`monitor-completeness`
- tracks goal state and warning conditions
- cannot close the task, but can require continuation

## Warning Conditions

The monitor must warn if any of the following happen:

- work starts before goals are explicit
- any goal remains `at-risk`, `unmet`, or `blocked`
- a worker changes active code without dependency evidence
- a role claims completion without file-level evidence
- cleanup is treated as success while live import or eval continuity is unverified
- contradictory role outputs remain unresolved by `default`

## Escalation Rules

- If dependency evidence is missing, reuse or spawn `explorer`.
- If removal safety is disputed, reuse or spawn `analyst`.
- If safe cleanup is blocked by active imports, do not delete; escalate to `worker-phase2-extraction`.
- If extraction lands but continuity is unverified, require `analyst-regression-check` before completion.
- If two rounds still leave a goal unmet, `monitor` must emit `continue required` and send control back to `default`.

## Phase Boundaries

`phase_1_safe_cleanup`
- archive or remove only clearly cold residue
- examples: old `nohup*.log`, already archived historical outputs, other non-imported leftovers

`phase_2_dependency_extraction`
- move still-imported primitives out of legacy training-script shells
- update importers to the minimal replacement
- only then retire the obsolete shell or wrapper

`phase_3_review`
- verify goals
- run native review before declaring the work complete

## Explicit Non-Goals

Do not build:

- a repo-local controller
- a cleanup scheduler
- a generic approval system
- a new archive runtime
- an automatic promotion engine for legacy removal

## Completion Rule

This plan is complete only when all success goals are `met` with direct evidence.
If any goal remains `at-risk`, `unmet`, or `blocked`, the task stays open.
