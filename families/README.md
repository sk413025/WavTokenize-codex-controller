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
  - evaluation and comparison families that remain active because official or promoted lines still use their scripts or artifacts
- `compat_legacy/`
  - still-imported technical legacy families that are not part of the official start path and should not be treated as new research homes

## Operating Rules
- The stable worktree should not gain new root-level `exp_xxxx` directories.
- New hypotheses belong in a new branch and a new worktree, not under `families/` in the stable line.
- If a hypothesis is promoted, merge it back into an existing `families/official/*`, `families/deps/*`, or `families/eval/*` path when possible.
- Create a new official family under `families/official/*` only when the work has been promoted into the official control surface.

## Navigation
- For official launches, start from `docs/next_experiment.md`, `official-run-ladder`, and `experiments/manifests/*.json`.
- For implementation details, official manifests and adapters may point into `families/official/*`.
- For active imports and checkpoint dependencies, inspect `families/deps/*`.
- For baseline comparisons and report-generation scripts, inspect `families/eval/*`.
- Treat `families/compat_legacy/*` as dependency residue, not as a recommended place to start new work.
