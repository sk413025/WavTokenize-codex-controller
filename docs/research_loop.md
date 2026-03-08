# Research Loop

Start in `AGENTS.md`.

This is the project-specific loop that Codex should follow using native multi-agent collaboration.

## Default Loop
1. Read `docs/project_goal.md` and the relevant official manifest.
2. Read `docs/worktree_policy.md`.
3. Use `experiment-decomposition` to classify the task and split the work.
4. Use `official-run-ladder` when the next step is a real official ladder run in the stable line.
5. Use native `explorer` work to inspect prior runs, artifacts, and family notes.
6. Decide whether the next step is `preflight`, `smoke`, `short`, or `full`.
7. Execute the selected manifest through `codex_controller`.
8. Use `stage-monitoring` if the run is active.
9. Use `run-diagnosis` and `result-comparison` after the run.
10. If a follow-up is needed, use `followup-generation` to draft a bounded next step.
11. Final decisions stay with `Codex(default)`.

## Allowed Follow-Up Scope
Follow-up proposals should stay within project-specific changes such as:
- manifest parameter changes
- adapter corrections
- cache or checkpoint binding fixes
- bounded experiment-code changes in onboarded families
- evaluation coverage improvements tied to official families

Avoid proposing:
- a new agent runtime
- generic controller infrastructure
- a new queue or approval model
- broad refactors unrelated to the research target

## Stop Rule
Stop a line of work when:
- repeated runs fail for the same operational reason and the fix is already known
- the proposed next step only adds architecture complexity without improving the research goal
- a new iteration would require unsupported assets or infrastructure outside the repo's narrow scope

## Outputs
This loop should produce Markdown-first conclusions:
- what happened
- what the evidence says
- what to try next
- what not to build

If machine-readable run facts are useful, they belong in `controller_runs/<run_id>/` as thin execution records, not as a second controller layer.
