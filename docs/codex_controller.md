# Codex Controller Implementation Notes

Start in `AGENTS.md`.

This document is implementation detail for `codex_controller`. It is not the repo policy source of truth.

## What `codex_controller` Is
- The lifecycle layer for manifests, adapters, run state, and packet-oriented orchestration
- A bridge from Codex-native planning into durable controller artifacts
- Not a replacement for the Codex-started session, native roles, or repo-local skills

## Official Runtime Shape
Official runs still follow:
- `plan`
- `prepare`
- `execute`
- `monitor`
- `analyze`
- `diagnose`
- `patch`
- `propose_next`
- `queue_next`

Packet-oriented controller commands:
- `prepare-run`
- `emit-packets`
- `ingest-agent-result`
- `advance-run`
- `finalize-run`

Current CLI bridge:
```bash
python -m codex_controller validate <manifest>
python -m codex_controller describe <manifest>
python -m codex_controller run <manifest>
python -m codex_controller run <manifest> --dry-run
python -m codex_controller resume controller_runs/<run_id>
python -m codex_controller status controller_runs/<run_id>
```

## Role And State Notes
- `default` remains the only official queue owner
- `agents/registry.json` is controller metadata and must map onto native Codex roles
- `controller_runs/<run_id>/` is the durable record for run state, events, handoffs, analysis, diagnosis, and follow-up artifacts

## Thin Runtime Contract
The public run contract is intentionally small:
- `state.json`
- `analysis.json`
- `diagnosis.json`
- `patch_request.json`
- `next_manifest.json`
- `controller_actions.jsonl`
- `agent_packets/`
- `agent_results/`

Supporting records remain available for debugging and lineage:
- `events.jsonl`
- `decision_log.jsonl`
- `agent_handoffs.jsonl`
- `manifest.snapshot.json`
- per-stage logs

`dispatch_plan.json` may exist as internal debug output. It is not a starting point for repo operation.

## Skills
Use repo-local skills first.

Core five:
- `controller-decomposition`
- `dispatch-handoff`
- `stage-monitoring`
- `run-diagnosis`
- `followup-generation`

Secondary:
- `manifest-authoring`
- `legacy-adapter-onboarding`
- `experiment-promotion`

For the official Codex-started flow, see `docs/golden_session.md`.
