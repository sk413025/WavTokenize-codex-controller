---
name: experiment-promotion
description: Promote successful runs into durable knowledge and best-known baselines.
---

Use this skill when a run should influence `knowledge/` or the official family baseline.

Workflow:
1. Confirm the run met the manifest's acceptance criteria and baseline comparison rules.
2. Record why the run matters:
   - promoted baseline
   - reusable failure lesson
   - architecture or adapter improvement
3. Update `knowledge/best_runs.json` and any family-specific index only when the evidence is clear.
4. Keep the promotion linked to a concrete `run_id`, `experiment_id`, and objective family.
5. If promotion changes planning defaults, update `knowledge/policies/controller_defaults.json` or docs.

Checks:
- Promotions are evidence-based, not optimistic placeholders.
- The repo retains both the promoted run and the comparison context.
