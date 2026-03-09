# Codex Controller Implementation Notes

Start in `AGENTS.md`.

This document is an appendix for the thin runtime.
Read it only when you need to validate a manifest, execute a run, monitor a run, resume a run, or inspect persisted run state.

## What `codex_controller` Is
- a small execution helper for official manifests and adapters
- a run ledger for logs, stage state, monitor output, and basic analysis
- a persistence layer after the Codex session has already decided what to do
- the preferred place for bounded stage-to-stage autonomy in official, repeatable run sequences

## What `codex_controller` Is Not
- not the primary control surface
- not the place to learn the research loop
- not a planner, promotion engine, or review system
- not a generic hypothesis queue or shell-watcher replacement for one-off experiments
- not the place to encode session-only autonomy windows

## When To Use It
Use `codex_controller` only after `AGENTS.md`, repo docs, and the relevant skills have already established the next step.
Typical cases:
- validate a manifest
- execute `preflight`, `smoke`, `short`, or `full`
- monitor an active run
- resume a failed or interrupted run
- inspect persisted run state

Use it when a sequence is both:
- official or promoted enough to deserve a manifest
- repeatable enough that stage handoff should survive beyond a single terminal session

Treat this as the official surface only. One-off hypothesis runs may write thin run-local facts for auditability, but they should stay session-owned and agent-native unless the sequence has been explicitly promoted.

Do not use it just because a one-off hypothesis would be more convenient with a watcher. Keep one-off hypothesis sequencing in the Codex session unless the flow is ready to become an official manifest.

## Runtime Contract
Each run should keep a minimal durable record:
- `manifest.snapshot.json`
- `state.json`
- per-stage logs
- `events.jsonl`
- `monitor_report.json`
- `metrics.json`
- `analysis.json`
- `diagnosis.json`

These persisted facts are the durable event surface for official runs. If a workflow needs reliable stage-to-stage continuity, resumability, and inspectable completion state, prefer this surface over ad hoc shell sequencing.

## Command Reference
```bash
python -m codex_controller validate <manifest>
python -m codex_controller describe <manifest>
python -m codex_controller run <manifest>
python -m codex_controller run <manifest> --dry-run
python -m codex_controller resume controller_runs/<run_id>
python -m codex_controller monitor-run controller_runs/<run_id> --print-report
python -m codex_controller status controller_runs/<run_id>
```
