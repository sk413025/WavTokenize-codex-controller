# Worktree Policy

Start in `AGENTS.md`.

This repo uses worktrees to isolate new hypotheses from the stable official control surface.

## Stable Baseline
`codex-first-controller`
- this is the stable official control-surface worktree
- keep official docs, official manifests, official adapters, and official run guidance here
- keep stable family code under `families/official/*`, `families/deps/*`, `families/eval/*`, and `families/compat_legacy/*`
- do not add new root-level `exp_xxxx` directories here
- use it for planning, review, official ladder execution, and bounded fixes in the current line

## When A New Worktree Is Required
Open a new branch and a new worktree when the task introduces a new hypothesis or changes:
- experiment code for a new idea
- official manifests or adapters to test a new idea
- evaluation behavior for an official family
- family-specific training or preflight behavior for a new direction
- any work expected to span more than one bounded fix in the stable line

Treat these as hypothesis work:
- trying a new loss, ablation, conditioning strategy, or fallback strategy
- testing a new family-specific curriculum or initialization idea
- changing a family in a way that could fail and should stay isolated from the official baseline
- changing data strategy or augmentation policy to address a persistent generalization gap
- changing evaluation protocol or held-out coverage to decide whether `train-good/test-bad` is real or only an artifact of the current evidence surface

Treat these as immediate hypothesis triggers:
- two consecutive iterations where training or in-distribution evidence improves but held-out generalization does not improve clearly
- any proposed fix that targets OOD weakness by changing loss design, curriculum, conditioning, augmentation, or evaluation design

## When A New Worktree Is Not Required
Stay in the current worktree only when the work is clearly part of the same line:
- rerunning an existing official manifest
- changing run ids, output paths, or GPU assignment
- reviewing logs, artifacts, and metrics
- Markdown-only analysis or planning
- a bounded fix that clearly belongs to the current official line and does not create a new hypothesis
- a documentation or manifest clarification that records an already-decided next step without changing the research direction

If there is doubt, isolate the work in a new worktree.

## Naming Contract
Branch:
- `exp/YYYYMMDD-<short-hypothesis>`
- example: `exp/20260308-material-loss-ablation`

Worktree path:
- `/home/sbplab/ruizi/exp-YYYYMMDD-<short-hypothesis>`

Each hypothesis worktree should start with `docs/hypothesis.md` containing:
- hypothesis statement
- target family
- problem being solved
- success evidence
- falsification condition
- what not to change

## Merge-Back Contract
Do not merge a hypothesis back into `codex-first-controller` unless:
- the hypothesis produced concrete evidence
- the result was reviewed with repo docs and skills
- the promoted change clearly improves an official family or the official control surface

Do not merge back:
- failed hypotheses with no reusable bounded fix
- inconclusive work that only increases complexity
- generic controller or runtime ideas that should stay in Markdown or skills instead

## Codex Workflow
For any new task, classify it first:
1. execution of an existing official ladder
2. bounded fix in the current line
3. new hypothesis

If it is a new hypothesis:
1. plan a new branch and worktree
2. create `docs/hypothesis.md` in that worktree
3. keep experiment edits isolated there
4. run `preflight -> smoke -> short -> full` there as needed
5. review the evidence before proposing anything for merge-back

If a hypothesis earns promotion, merge it back into the stable `families/*` layout instead of recreating a new root-level `exp_xxxx` directory.

This policy is Markdown-first on purpose. Do not add runtime code to enforce it.
