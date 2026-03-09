# Experiments Surface

Start in `AGENTS.md`.

This directory is the machine-readable execution surface for official experiment families.
It is not the primary operating guide.

## What Lives Here
- `registry.json`: official manifests and adapter index
- `manifest.schema.json`: manifest contract
- `manifests/*.json`: execution contracts for official experiment ladders
- `adapters/*.json`: adapter contracts for official and legacy pipelines

## Registry Semantics
- `registry.json` is a thin lookup surface. It is not a planner, router, or promotion policy layer.
- `dependency_families` in `registry.json` lists active non-official families that official manifests, training code, imports, or checkpoint bindings still depend on.
- `eval_families` lists active evaluation-oriented families that remain part of the comparison surface.
- `compat_legacy_families` lists still-imported technical legacy families that are not part of the official start path.
- These are classification fields, not archive directives. Layout should follow actual dependency usage inside `families/*`.

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
- `hubert-then-distalign`

## Current Preferred Launch
Do not treat this file as the place to decide the current first ladder.

For the current preferred official first launch:
- read `docs/next_experiment.md`
- use `official-run-ladder`

Consult `docs/reference/official_run_playbook.md` only when you need concrete command examples for the current official family before opening `codex_controller` docs.
