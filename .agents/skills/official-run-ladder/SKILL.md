---
name: official-run-ladder
description: Run the canonical official experiment ladder through AGENTS-first docs and native Codex collaboration.
---

Use this skill when launching a real official run, especially the first `exp_0304` path.

Workflow:
1. Start in `AGENTS.md`.
2. Read `docs/project_goal.md`, `docs/experiment_policy.md`, and `docs/research_loop.md`.
3. Read `docs/next_experiment.md`.
4. Read `docs/official_run_playbook.md`.
5. Use `experiment-decomposition` to confirm the next bounded step.
6. For `material-generalization`, run preflight first and then the smoke ladder if the preflight evidence still supports it.
7. Inspect `preflight_report.json`, `analysis.json`, and `monitor_report.json` before deciding whether the smoke result justifies a short run.
8. Use `stage-monitoring`, `run-diagnosis`, and `result-comparison` to close the loop in the same Codex session.

Checks:
- Do not treat `codex_controller` docs as the primary run guide.
- Do not skip directly to `short` just because preflight says `ready_for_real_run`; the current official next step is still `smoke` unless newer evidence justifies otherwise.
