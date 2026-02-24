# M2 Acceptance Evaluation (2026-02-23)

Scope:
- Run: `exp_0217/runs/t453_m2_interw002_epoch100_20260223`
- Single-factor change (vs exp_0216 baseline): `intermediate_weight 0.03 -> 0.02`
- Baseline reference:
  - Quality baseline: `exp_0217/analysis_commit_5e859b0/audio_quality_by_epoch.json` (val, epoch300, N=100)
  - MSE baseline: `exp_0216/runs/augmented_long_20260216/summary.json`

## 1) Execution completeness
- Training finished to 100 epochs.
- Checkpoints present at `epoch010,020,...,100`.
- Artifacts present:
  - `metrics_history.json`
  - `summary.json`
  - `best_model.pt`, `final_model.pt`

Evidence:
- `exp_0217/runs/t453_m2_interw002_epoch100_20260223/metrics_history.json`
- `exp_0217/runs/t453_m2_interw002_epoch100_20260223/checkpoints/checkpoint_epoch100.pt`
- `exp_0217/runs/t453_m2_interw002_epoch100_20260223/summary.json`

## 2) Baseline values (for threshold comparison)
- Baseline val (epoch300, N=100):
  - mean `ΔPESQ = -0.295611`
  - mean `ΔSTOI = -0.060379`
  - mean `feature_mse = 0.035095`
- Baseline `best_val_mse = 0.038064`

Evidence:
- `exp_0217/analysis_commit_5e859b0/audio_quality_by_epoch.json`
- `exp_0216/runs/augmented_long_20260216/summary.json`

## 3) M2 measured values (N=100, fixed indices)
- M2 epoch050 val:
  - `ΔPESQ = -0.291325` (gain vs baseline `+0.004286`)
  - `ΔSTOI = -0.067137` (gain vs baseline `-0.006757`)
  - `feature_mse = 0.035346`
- M2 epoch100 val:
  - `ΔPESQ = -0.301803` (gain vs baseline `-0.006192`)
  - `ΔSTOI = -0.065780` (gain vs baseline `-0.005400`)
  - `feature_mse = 0.035045`

Evidence:
- `exp_0217/analysis_commit_5e859b0/audio_quality_m2_epoch050.json`
- `exp_0217/analysis_commit_5e859b0/audio_quality_m2_epoch100.json`

## 4) Threshold checklist
Threshold source:
- `exp_0217/analysis_commit_5e859b0/next_experiment_recommendation.md`

Checklist:
1. Val mean `ΔPESQ` improves by `>= +0.03` over baseline.
   - Result: **Fail** (best observed gain `+0.004286`).
2. Val mean `ΔSTOI` improves by `>= +0.01` over baseline.
   - Result: **Fail** (observed gains are negative).
3. `best_val_mse` degradation `<= 1%`.
   - Baseline: `0.038064`
   - M2 best: `0.039052`
   - Degradation: `+2.59%`
   - Result: **Fail**
4. P2 keeps pass.
   - Result: **Pass**
5. P3 monitoring-only (not hard gate).
   - Result: `fail` at final metrics, but non-gating by spec.

## 5) Decision
- Acceptance status for M2: **No-Go (thresholds not met)**.
- Progress status:
  - Mid-progress milestones (`epoch10/20/30/50 checkpoints`) are met.
  - Full acceptance (quality thresholds + MSE guardrail) is **not** met.
