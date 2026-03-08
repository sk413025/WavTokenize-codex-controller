# Single Golden Session

This is the official Codex-started path.

1. Start in `AGENTS.md`.
2. Keep one Codex-started session as the golden session for planning, delegation, integration, and final queue decisions.
3. Use the core five repo-local skills first:
   `controller-decomposition`, `dispatch-handoff`, `stage-monitoring`, `run-diagnosis`, `followup-generation`.
4. Before expanding controller or agent-native architecture, run `codex-native-review`.
5. Use `python -m codex_controller ...` and the packet commands to persist controller state, not to replace the golden session.
6. Let bounded roles return results through packets and handoffs.
7. Let `default` finalize the run and any official queue mutation.

In short: `AGENTS.md` starts the work, the Codex session stays authoritative, and `codex_controller` records and advances the lifecycle.
