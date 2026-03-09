---
name: official-run-ladder
description: Run the canonical official experiment ladder through AGENTS-first docs and native Codex collaboration.
---

Use this skill when launching a real official run from the stable line.

Workflow:
1. Start in `AGENTS.md`.
2. Read `docs/project_goal.md`, `docs/experiment_policy.md`, and `docs/research_loop.md`.
3. Read `docs/next_experiment.md`.
4. Use `experiment-decomposition` to confirm the next bounded step.
5. Follow the bounded monitor handoff and next-step guidance in `docs/research_loop.md` when starting the official run.
6. If the run needs durable stage handoff or bounded auto-continue, prefer the existing official manifest and `codex_controller` run surface over ad hoc shell sequencing.
7. Follow the current official family and rung declared in `docs/next_experiment.md`; keep family-specific command examples in the reference playbook, not in this skill.
8. Inspect `preflight_report.json`, `analysis.json`, and `monitor_report.json` before deciding whether the current result justifies the next rung.
9. On `clean_success`, `Codex(default)` may execute one declared next step. On any other terminal class, notify and stop.
10. Use `stage-monitoring`, `run-diagnosis`, and `result-comparison` to close the loop in the same Codex session.
11. Consult `docs/reference/official_run_playbook.md` only when you need concrete command examples for the current official family.

Checks:
- Do not treat `codex_controller` docs as the primary run guide.
- Do not use shell queues or watcher scripts when an official manifest already expresses the same bounded sequence.
- Do not hard-code a family-specific rung here; let `docs/next_experiment.md` and the selected manifest define that bounded step.
