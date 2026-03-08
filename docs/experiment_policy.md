# Experiment Policy

Start in `AGENTS.md`.

This file defines the operating gates for official experiment families. Keep the policy here, not in Python controller logic.

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

## Failure Handling
- execution failures go to `run-diagnosis`
- missing artifacts or stalled logs go to `stage-monitoring` plus `run-diagnosis`
- a failed run does not imply a new runtime feature is needed
- first prefer a documentation fix, a skill update, a manifest change, or an experiment-code fix

## Promotion Handling
- promotion is a Codex review decision
- manifests and runtime may record facts, but they do not auto-promote baselines
- best-run updates should follow explicit evidence review, not automatic controller heuristics
