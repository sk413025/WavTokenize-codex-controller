# Official Run Playbook

Start in `AGENTS.md`.

This is the canonical real-run path for the first official family: `material-generalization`.
Use it for a real `preflight -> short-run` decision flow.

## Before You Run Anything
1. Read `AGENTS.md`.
2. Read `docs/project_goal.md`.
3. Read `docs/experiment_policy.md`.
4. Read `docs/research_loop.md`.
5. Use `experiment-decomposition` to confirm the next bounded step is `preflight`.
6. Open the relevant manifest:
   - `experiments/manifests/exp0304_material_generalization_preflight.json`

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

## Step 2: Decide The Next Ladder Tier
Use the preflight evidence only.

If `preflight_report.json` says:
- `ready_for_real_run`: continue to the short run
- `ready_for_smoke`: stop and switch to the smoke ladder instead of forcing a short run
- `blocked`: stop and use `run-diagnosis`

## Step 3: Run The Short Ladder
Only if preflight returned `ready_for_real_run`, run:
```bash
python -m codex_controller run experiments/manifests/exp0304_material_generalization_short.json --run-id exp0304_short_live
```

While it runs, use:
```bash
python -m codex_controller monitor-run controller_runs/exp0304_short_live --print-report
python -m codex_controller status controller_runs/exp0304_short_live
```

## Step 4: Review The Result In The Same Codex Session
After the short run:
1. Use `stage-monitoring` to summarize stage evidence.
2. Use `run-diagnosis` if the run failed, stalled, or missed artifacts.
3. Use `result-comparison` if the run completed and produced usable evidence.
4. Use `followup-generation` only after the evidence review is complete.

## Stop Conditions
Stop and do not continue to another tier when:
- preflight is blocked
- preflight says `ready_for_smoke` rather than `ready_for_real_run`
- the short run fails or stalls
- the short run completes but evidence is too weak to justify a bigger run

This playbook keeps the first real official path fully driven by `AGENTS.md + docs + skills`, with `codex_controller` used only to execute and record the run.
