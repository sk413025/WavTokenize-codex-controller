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
6. At launch time, pair the official run with a fresh `monitor` handoff that names the run id and the one allowed next step, if any.
7. If the run needs durable stage handoff or bounded auto-continue, prefer the existing official manifest and `codex_controller` run surface over ad hoc shell sequencing.
8. For `material-generalization`, run preflight first and then the smoke ladder if the preflight evidence still supports it.
9. Inspect `preflight_report.json`, `analysis.json`, and `monitor_report.json` before deciding whether the smoke result justifies a short run.
10. On `clean_success`, `Codex(default)` may execute one declared next step. On any other terminal class, notify and stop.
11. Use `stage-monitoring`, `run-diagnosis`, and `result-comparison` to close the loop in the same Codex session.

Checks:
- Do not treat `codex_controller` docs as the primary run guide.
- Do not use shell queues or watcher scripts when an official manifest already expresses the same bounded sequence.
- Do not skip directly to `short` just because preflight says `ready_for_real_run`; the current official next step is still `smoke` unless newer evidence justifies otherwise.
