# Next Experiment

Start in `AGENTS.md`.

This file records the current preferred official next step. It is a Markdown decision note for Codex, not a second controller.

## Current Choice

**PAUSED** — Moved to hypothesis worktree after multi-agent review.

Worktree: `/home/sbplab/ruizi/exp-20260309-calibrated-alignment`
Branch: `exp/20260309-calibrated-alignment`
Hypothesis: `docs/hypothesis.md` in that worktree

## Why The Short Run Is Paused

A structured multi-agent review (9 agents, 4 phases) identified critical issues blocking the planned `exp0304_material_generalization_short` run:

### 1. Loss Magnitude Crisis (Agent 1B)
- `lambda_inter_feat=0.5` causes inter_feat to contribute **93.2% of total loss**
- Raw inter_feat MSE ~1517 vs wav_mse ~0.027 (56,900x ratio)
- The smoke run's "42.5% improvement" was measured on 20 training samples with this imbalanced loss
- Alignment was never actually tested — the reconstruction gradient was suppressed

### 2. Anchor Is Dead (Agent 1A)
- All 4 anchor variants (A, B, D, E in exp_0305b) degrade after epoch ~18
- D/E crashed after only 3 epochs
- Anchor loss 3-10x larger than reconstruction loss → irreconcilable gradient conflict
- Conclusion: constraining encoder to match teacher is fundamentally flawed for denoising

### 3. Evaluation Is Insufficient (Agent 1C)
- Only 3 evaluation samples — cannot detect OOD improvement
- No per-material breakdown (box, papercup, plastic, mac not evaluated separately)
- Mac hold-out status unverified
- **Generalization stall trigger is ACTIVE** (per `research_loop.md`)

### 4. Stall Trigger Active
Per governance rules, two consecutive iterations improved training/in-distribution metrics without held-out generalization proof. The next step must be classified as **new hypothesis**.

## Hypothesis: Calibrated Intermediate Feature Alignment

The winning hypothesis (scored 49/65 by Critic, confirmed by Judge) proposes:

1. **Prerequisites (before training):**
   - Expand evaluation to 20+ samples with per-material breakdown
   - Run decoder perturbation probe (30 min GPU)
   - Verify mac hold-out status

2. **Training changes:**
   - Reduce `lambda_inter_feat` from 0.5 to 0.005 (100x reduction)
   - Add L2-normalization to intermediate features before MSE
   - Falsification gate at epoch 30

3. **Falsification condition:**
   - val_wav_mse > 0.027 at epoch 30 → alignment is harmful
   - mac PESQ <= 1.81 → alignment doesn't help OOD
   - If falsified → kill alignment, switch to Hypothesis B (decoder adapters, pure wav_mse)

## Fallback: Hypothesis B (Decoder Adapter)

If calibrated alignment fails:
- Add 1x1 Conv adapters at decoder input (identity-initialized)
- Remove ALL alignment losses
- Train with pure wav_mse from exp_0226_best (PESQ 1.8163, best ever)
- Addresses decoder fragility as the actual bottleneck

## Prior State (for reference)

The smoke rung (`exp0304_material_generalization_smoke`) completed on 2026-03-09:
- run ledger was later removed during aggressive repo slimming; recorded result was 5 epochs
- val_wav_mse: 0.0267, 42.5% better than noisy baseline (but measured on training samples)
- inter_feat: ~1517 (val), contributing 93.2% of loss at lambda=0.5

## Why Not The Other Official Families

`hubert-then-distalign`
- Blocked by GPU headroom
- No current evidence justifies moving beyond preflight

## Stop Rule

Do not return to the short run at lambda=0.5. The loss imbalance must be fixed first.

If the calibrated alignment hypothesis (worktree `exp/20260309-calibrated-alignment`) is falsified at epoch 30, do not retry with different lambda — proceed to Hypothesis B (decoder adapter).
