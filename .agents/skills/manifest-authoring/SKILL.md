---
name: manifest-authoring
description: Create or revise official experiment manifests that Codex can validate, run, diagnose, and compare.
---

Use this skill when adding or editing files under `experiments/manifests/`.

Workflow:
1. Start from `experiments/manifest.schema.json` and an existing manifest from the same family if one exists.
2. Fill research-level fields before command details:
   - `objective`
   - `hypothesis`
   - `baseline_refs`
   - `acceptance_criteria`
   - `failure_policy`
   - `diagnosis_policy`
   - `patch_policy`
   - `next_step_policy`
3. Resolve each stage through an adapter contract when possible.
4. Ensure stages define completion and artifact expectations clearly enough for controller analysis.
5. Validate with `python -m codex_controller validate <manifest>`.
6. If the manifest changes official workflow behavior, update `experiments/README.md` or `docs/codex_controller.md`.

Checks:
- The manifest is a research spec, not a loose shell recipe.
- Baseline comparison and next-step policy are explicit.
- The manifest can be used by planner, executor, monitor, and analyst roles without hidden tribal knowledge.
