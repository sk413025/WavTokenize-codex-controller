# Experiments Surface

This directory defines the official Codex-controlled research surface.

## Core Files
- `registry.json`: official manifests and adapter index
- `manifest.schema.json`: research-manifest schema
- `manifests/*.json`: official closed-loop experiment manifests
- `adapters/*.json`: adapter contracts for official and legacy pipelines

## Concepts
- A manifest is a research spec, not only an execution recipe.
- An adapter formalizes how Codex can safely call a legacy or official pipeline.
- Official workflows must run through `python -m codex_controller ...`.
- Planning should be decomposed into native Codex multi-agent tasks before broad implementation.
- Internal controller agents in `agents/registry.json` are controller metadata; native Codex roles remain the execution harness.

## Official Families
- `anchor-then-material`: exp_0305c -> exp_0304 sequential controller path
- `material-generalization`: standalone exp_0304 official path
- `hubert-then-distalign`: exp_0228 base HuBERT feature-loss stage followed by distalign, replacing the legacy watcher-heavy shell flow

## Recommended First Launch
Use `material-generalization` as the first real execution ladder:
- `exp0304_material_generalization_preflight`
- `exp0304_material_generalization_smoke`
- `exp0304_material_generalization_short`
- `exp0304_material_generalization`

This keeps the first official launch Codex-started while avoiding a direct jump into a 300-epoch run.

## Native Harness Expectations
For official families, prefer this stack:
- `AGENTS.md` for repo policy
- `.codex/config.toml` for native multi-agent runtime behavior
- `.agents/skills/` for repeated workflows
- `experiments/manifests/*.json` and `experiments/adapters/*.json` for machine-readable controller contracts

Before changing controller architecture or adding agent-native machinery, run the `codex-native-review` skill first and prefer policy/skill/manifest changes over new runtime code.

## Watcher-Heavy Migration Pattern
For legacy chains that used `wait_and_launch*.sh` or `monitor*.sh`:
- treat the shell watcher as a historical reference, not the official runtime
- move sequencing into manifest stage dependencies
- record completion and failure rules in adapter contracts
- use controller-owned run state for diagnosis and follow-up generation

## Example Commands
```bash
python -m codex_controller run experiments/manifests/exp0304_material_generalization_preflight.json
python -m codex_controller run experiments/manifests/exp0304_material_generalization_smoke.json
python -m codex_controller run experiments/manifests/exp0304_material_generalization_short.json
python -m codex_controller validate experiments/manifests/exp0228_hubert_then_distalign.json
python -m codex_controller describe experiments/manifests/exp0228_hubert_then_distalign.json
python -m codex_controller run experiments/manifests/exp0228_hubert_then_distalign.json --dry-run
python -m codex_controller status controller_runs/<run_id>
```
