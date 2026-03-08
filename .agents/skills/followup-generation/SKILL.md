---
name: followup-generation
description: Turn run outcomes into a concrete next manifest or queue recommendation while keeping queue ownership with default.
---

Use this skill when a run has finished or failed and the repo needs a bounded next step instead of open-ended advice.

Workflow:
1. Read the run outcome from `analysis.json`, `diagnosis.json`, `patch_request.json`, and `next_step_policy`.
2. Decide whether the best next action is:
   - patch and rerun
   - new manifest variant
   - promotion with no follow-up
   - stop and record the lesson
3. Draft the next manifest delta or promotion note with explicit acceptance and failure criteria.
4. Record the proposal through `emit-packets` or `ingest-agent-result` so it is attached to the run.
5. Reserve the actual queue decision for `default` during `advance-run` or `finalize-run`.
6. If the follow-up changes durable planning defaults, update knowledge or docs in the same pass.

Checks:
- The proposal is executable, not a vague recommendation.
- Follow-up generation is separate from queue mutation.
- The next step stays tied to the concrete run and experiment family.
