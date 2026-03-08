# Knowledge Layer

Start in `AGENTS.md`.

This directory stores durable project-specific memory.

Keep here only facts that are useful across runs:
- `experiments/index.json`: family summaries and latest run references
- `failures/index.json`: recurring operational failures and suggested fixes
- `best_runs.json`: manually reviewed best-known runs by family
- `policies/controller_defaults.json`: minimal runtime defaults only

Do not treat `knowledge/` as a second controller policy engine.
High-level research goals and workflow rules belong in Markdown docs.
