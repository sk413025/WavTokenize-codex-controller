# Single Golden Session

This is the official Codex-started path.

1. Start in `AGENTS.md`.
2. Read `docs/project_goal.md`, `docs/experiment_policy.md`, `docs/research_loop.md`, `docs/worktree_policy.md`, and `docs/next_experiment.md`.
3. Read `families/README.md` for the stable family layout when code-side navigation matters.
4. Use repo-local skills as Markdown playbooks, with `official-run-ladder` as the default entry for a real official run.
5. Use native multi-agent collaboration to decompose, inspect, monitor, and review the work.
6. Use `python -m codex_controller ...` only when a manifest needs validation, execution, monitoring, resume, or status inspection.
7. Review results in the same Codex session using native review and the repo checklist.
8. Keep final interpretation, promotion decisions, and follow-up choices with `Codex(default)`.

Suggested lifecycle-to-skill mapping:
- `plan` / `prepare`: `experiment-decomposition`
- `execute` / active `monitor`: `official-run-ladder` or bounded shell execution plus `stage-monitoring`
- `analyze` / `diagnose`: `result-comparison` and `run-diagnosis`
- `propose_next`: `followup-generation`

In short: `AGENTS.md` starts the work, docs and skills explain the workflow, native Codex handles orchestration, and `codex_controller` only records project-specific run facts.
