# Project Goal

Start in `AGENTS.md`.

## Research Goal
This repo exists to push three official experiment families built on top of `WavTokenizer`:
- `material-generalization`
- `anchor-then-material`
- `hubert-then-distalign`

The long-term research target is:
- preserve the original reconstruction behavior of the pretrained `WavTokenizer`
- improve noisy-to-clean reconstruction quality
- improve high-frequency recovery when denoising or reconstructing from degraded inputs
- improve generalization across speakers, materials, and recording conditions

## Success Definition
A run is only interesting if it improves the research target without breaking the preserved reconstruction baseline.

Use this rule in review:
- execution success alone is not enough
- a run that regresses core reconstruction is not a win
- a run that improves one narrow metric but clearly harms stability or preservation is not a win
- family baselines should be promoted only after Codex reviews concrete logs, artifacts, and evaluation outputs

## Official Families
`material-generalization`
- the simplest official path and the default first ladder for real launches

`anchor-then-material`
- a sequential experiment family where an anchor stage feeds a material generalization stage

`hubert-then-distalign`
- a sequential experiment family that replaces legacy watcher-heavy orchestration with explicit stage dependencies

## What This Repo Is Not
- not a new agent platform
- not a generic scheduler or queueing system
- not a generic monitoring framework
- not a replacement for Codex native planning, review, sub-agents, or skills

The repo should keep only project-specific experiment code, manifests, adapters, monitor facts, and durable run records.
