# Architecture Review

Start in `AGENTS.md`.

This document tracks the remaining places where the repo still leans too hard toward runtime or CLI framing.

## Current Assessment
Most public-facing guidance is now aligned with `Codex-native`, `Markdown-first` operation.
The main remaining issues are compatibility names and a few internal runtime terms, not large architectural drift.

## Remaining Text-Level Issues
### Still Worth Demoting Later
- `codex_controller` and `controller_runs/` still carry older naming for compatibility
- compatibility skill names such as `controller-decomposition` and `dispatch-handoff` remain on disk even though the preferred names are now `experiment-decomposition` and `native-handoff`
- command examples still exist in `docs/reference/codex_controller.md` and `docs/reference/official_run_playbook.md`, which is acceptable, but they should stay in reference sections rather than drive the repo narrative

### Acceptable Implementation-Note Usage
- `docs/reference/codex_controller.md` as an appendix
- `controller_runs/<run_id>/` as the persisted run ledger path
- `controller_defaults.json` as a minimal runtime default file
- manifest fields such as `run_root` and `baseline_refs` as execution facts

### Naming Debt Intentionally Deferred
- `codex_controller` package name stays for compatibility
- `controller_runs/` directory name stays for compatibility
- `run_status_detail` is now the active neutral runtime field; legacy `controller_status` is read only for backward compatibility
- compatibility skills stay on disk so older notes still resolve

## Next Good Cleanup Targets
- eventually retire the compatibility skill names from the recommended surface
- keep `codex_controller` command usage confined to appendix-style docs and playbooks
