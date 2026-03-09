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

## Long-Running Hypothesis Work
- For a new hypothesis, keep experiment edits in the hypothesis worktree and keep control ownership with the same `Codex(default)` session.
- Do not add new root-level `exp_xxxx` directories to the stable worktree for hypothesis code. Use a new worktree first, then promote the result back into `families/*` only after review.
- When launching a long-running hypothesis run, create a paired `monitor` handoff at launch time instead of relying on a later manual check.
- Treat the launch contract as session policy, not family runtime state: declare at most one allowed next step and keep broader autonomy out of experiment code.
- Use native roles to split observation from decision-making:
  - `monitor` or `stage-monitoring` for active run facts
  - `run-diagnosis` for stalled, failed, or weak runs
  - `result-comparison` for cross-run evidence review
  - `native-handoff` when a bounded role needs a clean evidence or edit handoff
- Bounded shell sequencing is acceptable for a single task when the next steps are already decided and no reusable control surface is created.
- If active monitoring, sequencing, or diagnosis starts to feel reusable across tasks, capture it in docs or skills rather than in runtime code.

## Promoting A Sequence To The Official Surface
- If a sequence needs bounded auto-continue, resumability, and durable stage events across more than one session, prefer promoting it to an official manifest operated through `codex_controller`.
- Promotion is appropriate only when the sequence is repeatable, has stable stage boundaries, and already belongs to an official or promoted family.
- Promotion should merge back into `families/official/*`, `families/deps/*`, or `families/eval/*` where practical rather than reintroducing root-level `exp_xxxx` directories in the stable worktree.
- Do not promote a sequence just to save manual work on a one-off hypothesis. Keep those flows agent-native and session-owned.
- When in doubt, keep the hypothesis event-driven first and promote the sequence only after the handoff rules have stabilized.

## Autonomy Window
- At the start of a task, `Codex(default)` may declare a bounded launch contract for auto-progression.
- If no broader contract is declared, default to `single-step`.
- For routine paired long-running launches, the daily operating default is `one-next-step`: one declared next step, or none.
- Broader windows are exceptional, should stay in Markdown guidance, and should not become family runtime fields or reusable queue protocols.
- Work must not auto-progress beyond the declared launch contract.

## Post-Run Transition
- `clean success`
  - may continue to the next already-decided step if it remains inside the active autonomy window
  - for long-running launches, prefer a single declared next step over a multi-step chain
- `interrupted but usable`
  - must emit a user-visible event, then return to `Codex(default)` for a decision on evaluation, resume, rerun, or stop
- `ambiguous or failed`
  - must not auto-progress; route it through `stage-monitoring` and `run-diagnosis`

## Sequencing Loss
- If a bounded sequence was expected to continue but no live sequencing mechanism remains, emit a user-visible `sequencing no longer active` event.
- Treat sequencing loss as a control-surface fact, not as a model failure.
- After sequencing loss, return control to `Codex(default)` instead of silently waiting or silently advancing.

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
