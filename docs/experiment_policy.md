# Experiment Policy

Start in `AGENTS.md`.

This file defines the operating gates for official experiment families and the common execution gates that also apply to bounded hypothesis runs. Keep the policy here, not in Python controller logic.

## Ladder
Official execution ladders use four tiers:
- `preflight`
- `smoke`
- `short`
- `full`

## Gate Rules
`preflight`
- verifies assets, cache bindings, checkpoint fallbacks, output paths, and GPU headroom
- may classify a family as blocked, ready for smoke, or ready for a larger run
- should not make promotion or follow-up decisions

`smoke`
- proves the command can start, load data, build the model, and emit basic artifacts
- should be the first real launch when preflight passes but risk is still high

`short`
- proves the family can sustain a bounded real run and produce meaningful artifacts
- is the normal gate before a full run

`full`
- is allowed only after the family has passed the earlier gates or Codex explicitly overrides that sequence

## Common Execution Gates
Use these gates for any real launch, including isolated hypothesis worktrees, unless a family-specific document adds stricter checks.

`run-ready minimum gate`
- confirm the command line and required flags are settled enough to launch without further design decisions
- verify required assets, cache bindings, checkpoint paths, and output paths resolve in the current worktree
- run the smallest useful static or bounded startup checks for the family before spending scarce GPU time
- treat missing prerequisites as a preflight failure, not as a reason to improvise during smoke

`GPU contention and headroom gate`
- gather current GPU/process facts before launching a new run
- if headroom is unclear or another long-running job is occupying the needed device, prefer serialization or a smaller ladder tier over risky parallel launches
- keep the decision evidence-based; do not hard-code a single memory threshold that ignores model or family context
- if a launch is deferred because of contention, record that fact in Markdown or run notes rather than inventing a queueing feature

`stalled run and intervention gate`
- treat a run as needing intervention when logs stop advancing, required artifacts do not appear on schedule, or the process state and artifacts disagree
- send stalled or ambiguous cases through `stage-monitoring` first for fact collection
- send the resulting evidence to `run-diagnosis` before deciding whether to stop, rerun, or patch
- keep the final intervention decision with `Codex(default)`

`post-run transition gate`
- only a clean success may auto-progress to the next ladder step inside an already-declared bounded autonomy window
- interrupted-but-usable runs may proceed only after a user-visible event and a fresh `Codex(default)` decision
- ambiguous or failed runs must not auto-progress; they go through `stage-monitoring` and `run-diagnosis` first

## Failure Handling
- execution failures go to `run-diagnosis`
- missing artifacts or stalled logs go to `stage-monitoring` plus `run-diagnosis`
- a failed run does not imply a new runtime feature is needed
- first prefer a documentation fix, a skill update, a manifest change, or an experiment-code fix

## Promotion Handling
- promotion is a Codex review decision
- manifests and runtime may record facts, but they do not auto-promote baselines
- best-run updates should follow explicit evidence review, not automatic controller heuristics
