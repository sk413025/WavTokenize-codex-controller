# Repo Skills

These are repo-local Codex skills.

Use them as Markdown playbooks that help Codex apply native multi-agent collaboration to this repo.
They should not recreate queueing, routing, packet exchange, or another controller protocol.

## Core Run-Loop Skills
- `experiment-decomposition`
- `official-run-ladder`
- `stage-monitoring`
- `run-diagnosis`
- `result-comparison`
- `followup-generation`
- `codex-native-review`

## Supporting Skills
- `manifest-authoring`
- `legacy-adapter-onboarding`
- `experiment-promotion`
- `native-handoff`

## Compatibility Skills
These may remain on disk for continuity with older notes, but they are not the preferred names:
- `controller-decomposition`
- `dispatch-handoff`

Persisted run artifacts or older notes may still mention these compatibility names. Treat them as legacy aliases, not as the recommended trigger surface.

Repo skills should remain Markdown-first.
If a skill needs helper code, that code must be small, project-specific, and clearly justified by the skill's `SKILL.md`.

## Skill Quality Gate
Treat a core skill as healthy only when it is clear on:
- when to trigger it
- what evidence or files it expects
- what output it owes back to `Codex(default)`
- what ownership stays with `Codex(default)`
- what it must not do

A skill should fail review if it:
- duplicates policy that belongs in `AGENTS.md` or the main docs without adding execution value
- reads like a reusable controller protocol instead of a Markdown playbook
- takes over stop/go, promotion, or follow-up ownership from `Codex(default)`
- requires helper code that behaves like routing, queueing, monitoring, or approval infrastructure

For a governance refinement pass, treat the skill audit as incomplete unless it checks:
- trigger clarity
- expected evidence or inputs
- output owed back to `Codex(default)`
- ownership boundaries
- whether any helper code has turned the skill into a control surface
