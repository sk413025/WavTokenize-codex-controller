# Architecture Review

Start in `AGENTS.md`.

This document tracks the remaining places where the repo still leans too hard toward runtime or CLI framing.

## Current Assessment
Most public-facing guidance is now aligned with `Codex-native`, `Markdown-first` operation.
The repo is close to `Codex-native`. The earlier start-path leak between a core skill and the appendix playbook has been closed. The main residual problems are now:
- controller-flavored naming that still leaks onto official surfaces
- a few Markdown rules that still read more like a lightweight control protocol than a minimal operating guide
- the need to keep proving that skills and code-as-tool stay thin instead of quietly growing into a second control surface

## Remaining Text-Level Issues
### Still Worth Demoting Later
- `codex_controller` still carries older naming for compatibility
- `experiments/registry.json` still reads slightly heavier than the role it should play; keep it as an index, not a control surface
- compatibility skill names such as `controller-decomposition` and `dispatch-handoff` remain on disk even though the preferred names are now `experiment-decomposition` and `native-handoff`
- command examples still exist in `docs/reference/codex_controller.md` and `docs/reference/official_run_playbook.md`, which is acceptable, but they should stay in reference sections rather than drive the repo narrative

### Acceptable Implementation-Note Usage
- `docs/reference/codex_controller.md` as an appendix
- a run-local ledger directory as the persisted run state path when a concrete execution emits one
- `controller_defaults.json` as a minimal runtime default file
- manifest fields such as `run_root` and `baseline_refs` as execution facts

### Naming Debt Intentionally Deferred
- `codex_controller` package name stays for compatibility
- `run_status_detail` is now the active neutral runtime field; legacy `controller_status` is read only for backward compatibility
- compatibility skills stay on disk so older notes still resolve

## Next Good Cleanup Targets
- eventually retire the compatibility skill names from the recommended surface
- keep `codex_controller` command usage confined to appendix-style docs and playbooks
- continue demoting controller/protocol phrasing when the same guidance can stay simpler in Markdown
- keep validating skill and tooling changes against explicit pass/fail gates instead of treating wording cleanup as enough on its own

## Sign-Off Reminder
Do not treat a governance pass as signed off merely because the language is cleaner.
The pass should stay open when start-path, research-decision, skill-boundary, or tooling-boundary checks remain partial, or when the strongest counterargument has not been answered with concrete evidence.
