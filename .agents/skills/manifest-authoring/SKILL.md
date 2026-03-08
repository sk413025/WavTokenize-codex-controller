---
name: manifest-authoring
description: Create or revise official experiment manifests as minimal execution contracts.
---

Use this skill when editing `experiments/manifests/*.json`.

Workflow:
1. Start from `experiments/manifest.schema.json` and an existing manifest from the same family.
2. Keep the manifest focused on execution facts:
   - objective summary
   - hypothesis
   - stages and dependencies
   - resources
   - acceptance markers
   - artifacts
3. Put planning, promotion, and research-loop policy in Markdown docs, not in manifest fields.
4. Validate with `python -m codex_controller validate <manifest>`.

Checks:
- The manifest should help the runtime run a family safely.
- It should not become a second policy engine.
