# Next Experiment Recommendation (Commit 5e859b0 Analysis)

Date: 2026-02-19
Scope: `families/deps/encoder_aug/runs/augmented_long_20260216`

## Decision
- Current stage decision: **Go (enter minimal setting-change design stage; do not execute yet)**
- Reason: analysis deliverables are completed and Trigger A/B are satisfied by N=100 evaluation plus statistical tests.

## Why Go (traceable)
1. SPEC required output files are complete (10/10).
   - Source: `families/deps/t453_weighted_baseline/analysis_commit_5e859b0/spec_coverage_status_20260219.json`
2. Train/val post-VQ quality deltas remain negative across evaluated epochs with N=100 per split.
   - Train ΔPESQ: `-0.5464 ~ -0.4782`; Train ΔSTOI: `-0.0407 ~ -0.0145`
   - Val ΔPESQ: `-0.3032 ~ -0.2846`; Val ΔSTOI: `-0.0674 ~ -0.0571`
   - Source: `families/deps/t453_weighted_baseline/analysis_commit_5e859b0/audio_quality_by_epoch.md`
3. Objective mismatch is consistently supported by epoch-sequence evidence (MSE improves but PESQ/STOI do not).
   - Source: `families/deps/t453_weighted_baseline/analysis_commit_5e859b0/hypothesis_scoring.json`

## Remaining limitations (must be explicit)
1. `epoch_222` checkpoint is missing (`checkpoint_epoch222.pt`), so that node is marked unavailable.
   - Source: `families/deps/t453_weighted_baseline/analysis_commit_5e859b0/audio_quality_by_epoch.md`
2. 已提供 `epoch220->222 replay` 補件（N=100 音質已補算），但它是補件替代證據，不是原 run 原生 artifact。
   - Source: `families/deps/t453_weighted_baseline/analysis_commit_5e859b0/audio_quality_epoch222_replay.md`

## Go trigger status
- Trigger A: `ΔPESQ` and `ΔSTOI` remain non-positive on val after N>=100 evaluation.
  - **Status: met**
- Trigger B: objective mismatch remains significant after statistical test (`p < 0.05`).
  - **Status: met**
- Trigger C: failure concentration appears in specific strata (e.g., high-T453 or low-SNR bins).
  - **Status: partially met** (`T453 [0.2,0.3)` has lower ΔPESQ; `T453 [0.3,0.5]` has lower ΔSTOI)

## Minimal-change candidate (single factor only)
- Candidate M1: adjust **one** sampling factor only (`t453_min_weight: 0.2 -> 0.3`), all other settings fixed.
- Acceptance threshold:
  - Val mean `ΔPESQ` improves by `>= +0.03` over current baseline.
  - Val mean `ΔSTOI` improves by `>= +0.01` over current baseline.
  - `best_val_mse` degradation `<= 1%`.
  - P2 keeps pass; P3 remains monitoring-only (not hard gate).
