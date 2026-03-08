---
name: legacy-adapter-onboarding
description: Wrap a legacy experiment family with an adapter contract and a minimal official manifest.
---

Use this skill when bringing a legacy `exp_*` family under official control.

Workflow:
1. Inventory the real entrypoint, environment assumptions, outputs, and completion signal.
2. Create or revise the adapter under `experiments/adapters/`.
3. Create or revise the official manifest under `experiments/manifests/`.
4. Move orchestration rules into Markdown docs and explicit stage dependencies.
5. Validate and dry-run the result.

Checks:
- Do not preserve watcher-shell decision logic as official control.
- Do not add generic controller code when Markdown plus a manifest is enough.
