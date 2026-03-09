# Official Run Playbook

Start in `AGENTS.md`.

This is the canonical real-run path for the first official family: `material-generalization`.
Use it for the current official next step: `preflight -> smoke`, with `short` only after smoke succeeds and later evidence justifies it.

## Before You Run Anything
1. Read `AGENTS.md`.
2. Read `docs/project_goal.md`.
3. Read `docs/experiment_policy.md`.
4. Read `docs/research_loop.md`.
5. Read `docs/next_experiment.md`.
6. Use `experiment-decomposition` to confirm the next bounded step is `smoke`.
7. Open the relevant manifests:
   - `experiments/manifests/exp0304_material_generalization_preflight.json`
   - `experiments/manifests/exp0304_material_generalization_smoke.json`

## Step 1: Run Preflight
Run:
```bash
python -m codex_controller run experiments/manifests/exp0304_material_generalization_preflight.json --run-id exp0304_preflight_live
```

Then inspect:
```bash
python -m codex_controller status controller_runs/exp0304_preflight_live
python -m codex_controller monitor-run controller_runs/exp0304_preflight_live --print-report
```

Read these artifacts:
- `controller_runs/exp0304_preflight_live/preflight_report.json`
- `controller_runs/exp0304_preflight_live/analysis.json`
- `controller_runs/exp0304_preflight_live/monitor_report.json`

## Step 2: Decide The Immediate Ladder Tier
Use the preflight evidence only.

If `preflight_report.json` says:
- `ready_for_smoke`: continue to the smoke run
- `ready_for_real_run`: this still does not override the current plan; run the smoke ladder first unless Codex has newer contrary evidence
- `blocked`: stop and use `run-diagnosis`

## Step 3: Run The Smoke Ladder
Run:
```bash
python -m codex_controller run experiments/manifests/exp0304_material_generalization_smoke.json --run-id exp0304_smoke_live
```

While it runs, use:
```bash
python -m codex_controller monitor-run controller_runs/exp0304_smoke_live --print-report
python -m codex_controller status controller_runs/exp0304_smoke_live
```

## Step 4: Decide Whether Short Is Justified
Only after the smoke run succeeds, review:
- `controller_runs/exp0304_smoke_live/state.json`
- `controller_runs/exp0304_smoke_live/analysis.json`
- `controller_runs/exp0304_smoke_live/monitor_report.json`
- stage logs

If smoke shows only startup viability, stop there and record the result.
If smoke shows stable execution and enough headroom for a longer run, then `Codex(default)` may propose the short ladder as the next experiment.

## Step 5: Run The Short Ladder
Only if Codex explicitly accepts that next step after smoke review, run:
```bash
python -m codex_controller run experiments/manifests/exp0304_material_generalization_short.json --run-id exp0304_short_live
```

While it runs, use:
```bash
python -m codex_controller monitor-run controller_runs/exp0304_short_live --print-report
python -m codex_controller status controller_runs/exp0304_short_live
```

## Step 6: Review The Result In The Same Codex Session
After the smoke run or short run:
1. Use `stage-monitoring` to summarize stage evidence.
2. Use `run-diagnosis` if the run failed, stalled, or missed artifacts.
3. Use `result-comparison` if the run completed and produced usable evidence.
4. Use `followup-generation` only after the evidence review is complete.

## Stop Conditions
Stop and do not continue to another tier when:
- preflight is blocked
- the smoke run fails or stalls
- the smoke run only proves startup but does not justify a longer run yet
- the short run fails or stalls
- the short run completes but evidence is too weak to justify a bigger run

This playbook keeps the first real official path fully driven by `AGENTS.md + docs + skills`, with `codex_controller` used only to execute and record the run.
