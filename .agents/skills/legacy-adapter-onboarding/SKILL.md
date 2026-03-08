---
name: legacy-adapter-onboarding
description: Wrap a legacy experiment family with an adapter contract so Codex can control it safely.
---

Use this skill when bringing a legacy `exp_*`, `done/`, or shell-driven workflow under official controller management.

Workflow:
1. Inventory the true entrypoint, environment assumptions, outputs, and completion signal.
2. Create an adapter file under `experiments/adapters/` with:
   - canonical entrypoint
   - adapter type
   - expected artifacts
   - completion rule
   - known failure signatures
3. If orchestration currently lives in shell watchers, move the decision logic into manifest policy and keep shell only as a thin adapter.
4. Add or update the corresponding manifest under `experiments/manifests/`.
5. Validate the manifest and perform a dry-run controller execution.
6. Update docs so the family is clearly marked official or legacy.

Checks:
- No new official workflow depends on `wait_and_launch*.sh` for decision logic.
- Adapter outputs are machine-readable enough for monitor and analyst roles.
- The migration does not assume implicit human memory.
