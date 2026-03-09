# M1 Acceptance Evaluation (2026-02-22)

Scope:
- Run: `families/deps/t453_weighted_baseline/runs/t453_m1_minw03_epoch100_debug`
- Change: `t453_min_weight 0.2 -> 0.3` (single-factor)
- Baseline reference:
  - Quality baseline: `families/deps/t453_weighted_baseline/analysis_commit_5e859b0/audio_quality_by_epoch.json` (val, epoch300, N=100)
  - MSE baseline: `families/deps/encoder_aug/runs/augmented_long_20260216/summary.json`

## 1) Execution completeness
- Training finished to 100 epochs.
- Checkpoints present at `epoch010~epoch100` (every 10 epochs).
- Artifacts present:
  - `metrics_history.json`
  - `summary.json`
  - `best_model.pt`, `final_model.pt`

Evidence:
- `families/deps/t453_weighted_baseline/runs/t453_m1_minw03_epoch100_debug/metrics_history.json`
- `families/deps/t453_weighted_baseline/runs/t453_m1_minw03_epoch100_debug/checkpoints/checkpoint_epoch100.pt`
- `families/deps/t453_weighted_baseline/runs/t453_m1_minw03_epoch100_debug/summary.json`

## 2) Baseline values (for threshold comparison)
- Baseline val (epoch300, N=100):
  - mean `ΔPESQ = -0.295611`
  - mean `ΔSTOI = -0.060379`
  - mean `feature_mse = 0.035095`
- Baseline `best_val_mse = 0.038064` (exp_0216 summary)

Evidence:
- `families/deps/t453_weighted_baseline/analysis_commit_5e859b0/audio_quality_by_epoch.json`
- `families/deps/encoder_aug/runs/augmented_long_20260216/summary.json`

## 3) M1 measured values (N=100, fixed indices)
- M1 epoch050 val:
  - `ΔPESQ = -0.303331` (gain vs baseline `-0.007720`)
  - `ΔSTOI = -0.071985` (gain vs baseline `-0.011605`)
  - `feature_mse = 0.034670`
- M1 epoch060 val:
  - `ΔPESQ = -0.301419` (gain `-0.005808`)
  - `ΔSTOI = -0.068016` (gain `-0.007636`)
  - `feature_mse = 0.035031`
- M1 epoch100 val:
  - `ΔPESQ = -0.299367` (gain `-0.003756`)
  - `ΔSTOI = -0.067341` (gain `-0.006961`)
  - `feature_mse = 0.034677`

Evidence:
- `families/deps/t453_weighted_baseline/analysis_commit_5e859b0/audio_quality_m1_epoch050.json`
- `families/deps/t453_weighted_baseline/analysis_commit_5e859b0/audio_quality_m1_epoch060.json`
- `families/deps/t453_weighted_baseline/analysis_commit_5e859b0/audio_quality_m1_epoch100.json`

## 4) Threshold checklist
Threshold source:
- `families/deps/t453_weighted_baseline/analysis_commit_5e859b0/next_experiment_recommendation.md`

Checklist:
1. Val mean `ΔPESQ` improves by `>= +0.03` over baseline.
   - Result: **Fail** (best observed gain is negative).
2. Val mean `ΔSTOI` improves by `>= +0.01` over baseline.
   - Result: **Fail** (best observed gain is negative).
3. `best_val_mse` degradation `<= 1%`.
   - Baseline: `0.038064`
   - M1 best: `0.038994`
   - Degradation: `+2.44%`
   - Result: **Fail**
4. P2 keeps pass.
   - Result: **Pass**
5. P3 monitoring-only (not hard gate).
   - Result: `fail` at observed epochs, but non-gating by spec.

## 5) Decision
- Acceptance status for M1: **No-Go (thresholds not met)**.
- Progress status:
  - Mid-progress milestone (`epoch10 checkpoint`) is met.
  - Full acceptance (quality thresholds) is **not** met.
