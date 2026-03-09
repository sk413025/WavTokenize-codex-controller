# Generalization Evidence Checklist

Start in `AGENTS.md`.

Use this checklist when a run appears to improve training or in-distribution behavior but the team is unsure whether that progress matters for the research goal.

## Minimum Evidence Record
- `preservation`: what evidence shows the pretrained `WavTokenizer` reconstruction behavior was preserved closely enough
- `denoise_quality`: what evidence shows noisy-to-clean reconstruction improved or at least did not regress obviously
- `generalization`: what evidence shows improvement across held-out speakers, materials, recording conditions, or other intended OOD conditions

If any dimension is missing:
- do not describe the run as a research breakthrough
- prefer `operational progress`, `candidate`, or `incomplete evidence`
- treat the missing dimension as an explicit follow-up target

## Escalate To New Hypothesis
- Escalate when two consecutive iterations improve training or in-distribution evidence without clear held-out generalization improvement.
- Escalate when the next reasonable move changes loss design, data strategy, augmentation policy, curriculum, conditioning, or evaluation design.
- Escalate when the team keeps debating whether the evidence is real only because the current held-out coverage is too narrow.

## Review Pack
- `explorer`: inspect the latest runs, evaluation artifacts, and family notes
- `analyst` for the strongest case that the current line is still healthy
- `analyst` for the strongest case that the current line is locally optimizing and missing the real gap
- one red-team pass that challenges the most comfortable conclusion
- `monitor` to decide whether the review is complete

## What Not To Build
- do not add a new controller layer just to enforce this checklist
- do not turn this into an automatic promotion engine
- do not replace native Codex review with repo-side routing or approval logic
