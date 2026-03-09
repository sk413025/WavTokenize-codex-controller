# Next Experiment

Start in `AGENTS.md`.

This file records the current preferred official next step. It is a Markdown decision note for Codex, not a second controller.

## Current Choice
Run:
- `experiments/manifests/exp0304_material_generalization_smoke.json`

Family:
- `material-generalization`

## Why This Is Next
This is the only official family that has already passed preflight far enough to justify another run.

Current evidence:
- `controller_runs/exp0304_preflight_live/preflight_report.json`
- `controller_runs/exp0304_preflight_objective_live/preflight_report.json`

Both show:
- assets are available
- fallback to pretrained `WavTokenizer` is acceptable
- `cuda:1` currently has enough headroom for `smoke`
- `cuda:1` does not currently justify a `short` or `full` run

## Problem This Experiment Solves
Right now, the repo has preflight evidence for `material-generalization`, but not real execution evidence for the official ladder.

This smoke run answers these concrete questions:
- can the current worktree actually start the official `exp_0304` family
- do the cache bindings and checkpoint fallback work in practice
- can the training stage build the model and load data
- does the run produce logs and basic run-local artifacts

This experiment does **not** try to prove final research quality. It only proves that the first official family can run safely enough to justify a later `short` run.

If this smoke run succeeds, the next Codex-side decision is not automatically another operational fix.
The default post-smoke review should:
- check whether the new evidence is only about startup and artifact generation or whether any preservation, denoise, or generalization evidence was added
- use `result-comparison` plus `followup-generation` before deciding whether the line deserves a `short` run, a bounded fix, or a new hypothesis worktree
- escalate to a new hypothesis if the main remaining problem is still `train-good/test-bad` rather than run readiness

## Why Not The Other Official Families
`anchor-then-material`
- `controller_runs/exp0305c_chain_preflight_live/preflight_report.json` is blocked by GPU headroom
- both sequential GPUs need more free memory before the smoke ladder is justified

`hubert-then-distalign`
- `controller_runs/exp0228_preflight_live/preflight_report.json` is blocked by GPU headroom
- there is no current evidence that justifies moving beyond preflight

## How Codex Should Run It
1. Read `docs/project_goal.md`, `docs/experiment_policy.md`, and `docs/research_loop.md`.
2. Use `experiment-decomposition` to confirm that the bounded next step is the `smoke` rung.
3. Use `official-run-ladder` for the canonical run procedure.
4. Execute `exp0304_material_generalization_smoke.json`.
5. Use `stage-monitoring` while the run is active.
6. Use `run-diagnosis` if it fails or stalls.
7. Use `result-comparison` only after the run finishes and evidence exists.

## Expected Successful Evidence
- `preflight_material_gen` completes
- `train_material_gen_smoke` completes
- stage logs exist and show model/data startup
- run-local outputs are created under the run directory
- the result is good enough to justify the later `short` rung

## Stop Rule
Do not jump to `short` unless the smoke run actually succeeds.

If smoke fails:
- diagnose the family
- prefer a bounded manifest, adapter, or experiment-code fix
- do not add new controller infrastructure
