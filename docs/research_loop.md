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
- When launching a long-running hypothesis run, create a paired `monitor` handoff at launch time instead of relying on a later manual check.
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
- Do not promote a sequence just to save manual work on a one-off hypothesis. Keep those flows agent-native and session-owned.
- When in doubt, keep the hypothesis event-driven first and promote the sequence only after the handoff rules have stabilized.

## Autonomy Window
- At the start of a task, `Codex(default)` may declare a bounded autonomy window for auto-progression:
  - `single-step`
  - `through-evaluation`
  - `through-next-smoke`
  - `through-bounded-sequence`
- If no autonomy window is declared, default to `single-step`.
- For paired long-running launches, prefer a launch contract that declares at most one bounded next step.
- Work must not auto-progress beyond the declared window.
- The autonomy window belongs to the current task only. Do not turn it into a reusable queue, scheduler, or runtime protocol.

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
