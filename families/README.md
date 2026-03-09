# Families Surface

Start in `AGENTS.md`.

This directory is the code-side family layout for the stable worktree. It replaces the old
root-level `exp_xxxx` sprawl with a smaller set of explicit roles.

## Layout
- `official/`
  - official experiment families that may be launched directly through manifests and adapters
- `deps/`
  - active dependency families used by official training, checkpoints, imports, or shared model code
- `eval/`
  - evaluation and comparison families that remain active because official or promoted lines still use their library modules, scripts, or artifacts
- `compat_legacy/`
  - contains only 5 core module files still imported by active pipelines (models, losses, data utilities)
  - historical training scripts and analysis tools were removed from the active tree and remain recoverable via git history at tag `pre-compat-legacy-cleanup`
  - not a research home — these are dependency residue pending extraction to `deps/` (Phase 3)

## Python Roles
- `managed entrypoint`
  - directly runnable family scripts that are explicitly owned by official manifests or adapters
- `library-only module`
  - model, data, helper, or runtime code that may be imported by active surfaces but is not itself a launch target

The stable worktree should not keep unmanaged standalone Python scripts under `families/`.
If a family-local script is not manifest-backed and is not a library module, move it to
`quarantine/python/` with the original relative path preserved.

## Operating Rules
- The stable worktree should not gain new root-level `exp_xxxx` directories.
- New hypotheses belong in a new branch and a new worktree, not under `families/` in the stable line.
- If a hypothesis is promoted, merge it back into an existing `families/official/*`, `families/deps/*`, or `families/eval/*` path when possible.
- Create a new official family under `families/official/*` only when the work has been promoted into the official control surface.
- Do not keep ad hoc `train_*`, `eval_*`, `analyze_*`, `plot_*`, `generate_*`, `compare_*`, or `run_*` scripts in the active family tree unless they are managed entrypoints.

## Navigation
- For official launches, start from `docs/next_experiment.md`, `official-run-ladder`, and `experiments/manifests/*.json`.
- For implementation details, official manifests and adapters may point into `families/official/*`.
- For active imports and checkpoint dependencies, inspect `families/deps/*`.
- For active evaluation library modules and remaining comparison surfaces, inspect `families/eval/*`.
- Treat `families/compat_legacy/*` as dependency residue (5 core files only), not as a place to start new work.
- Removed compat_legacy scripts remain recoverable in git history at tag `pre-compat-legacy-cleanup`.
- Historical standalone scripts that are not part of the active surface live under `quarantine/python/families/*`.
