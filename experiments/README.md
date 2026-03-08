# Experiments Surface

Start in `AGENTS.md`.

This directory is the machine-readable execution surface for official experiment families.
It is not the primary operating guide.

## What Lives Here
- `registry.json`: official manifests and adapter index
- `manifest.schema.json`: manifest contract
- `manifests/*.json`: execution contracts for official experiment ladders
- `adapters/*.json`: adapter contracts for official and legacy pipelines

## What Manifests Should Do
A manifest should describe:
- which family is being run
- what the run is trying to verify
- which stages exist
- how stage dependencies work
- what resources and artifacts are required for execution

A manifest should not become a second policy engine for planning, promotion, or autonomous behavior.
The official operating order still comes from `AGENTS.md`, repo docs, and repo skills.

## Official Families
- `material-generalization`
- `anchor-then-material`
- `hubert-then-distalign`

## Recommended First Launch
Use the `material-generalization` ladder first:
- `exp0304_material_generalization_preflight`
- `exp0304_material_generalization_smoke`
- `exp0304_material_generalization_short`
- `exp0304_material_generalization`

For the canonical real-run flow, use `official-run-ladder` and `docs/official_run_playbook.md` before opening `codex_controller` docs.
