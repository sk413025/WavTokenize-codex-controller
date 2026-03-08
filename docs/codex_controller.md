# Codex Controller Implementation Notes

Start in `AGENTS.md`.

This document is an appendix for the thin runtime.
Read it only when you need to validate a manifest, execute a run, monitor a run, resume a run, or inspect persisted run state.

## What `codex_controller` Is
- a small execution helper for official manifests and adapters
- a run ledger for logs, stage state, monitor output, and basic analysis
- a persistence layer after the Codex session has already decided what to do

## What `codex_controller` Is Not
- not the primary control surface
- not the place to learn the research loop
- not a planner, promotion engine, or review system

## When To Use It
Use `codex_controller` only after `AGENTS.md`, repo docs, and the relevant skills have already established the next step.
Typical cases:
- validate a manifest
- execute `preflight`, `smoke`, `short`, or `full`
- monitor an active run
- resume a failed or interrupted run
- inspect persisted run state

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
