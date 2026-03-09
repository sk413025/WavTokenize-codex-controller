# exp_0305b A/B Detailed Result Summary

## 1) Experiment Goal

Keep WavTokenizer original reconstruction behavior as stable as possible, while improving:

- Noise robustness
- High-frequency recovery

Strategy: add anchor regularization on selected encoder conv layers (teacher = frozen pretrained WavTokenizer).

## 2) Settings

- Compared variants:
- `exp_0305b_A_tail_lock`: anchor on `L16/L17` (tail only)
- `exp_0305b_B_front_tail_lock`: anchor on `L0/L1/L16/L17` (front+tail)
- Shared evaluation batch: same `families/eval/decoder_lora_eval/test` `sample01~03` (Val indices `[51, 54, 61]`)
- Metrics:
- PESQ (nb, 8kHz-resampled)
- STOI (24kHz)

## 3) Quantitative Results

| Model | PESQ | STOI | Delta PESQ vs exp_0224a | Delta STOI vs exp_0224a |
|---|---:|---:|---:|---:|
| `exp_0224a` (baseline) | 1.5856 | 0.6275 | 0.0000 | 0.0000 |
| `exp_0305b_A_tail_lock` | 1.5517 | 0.5816 | -0.0339 | -0.0459 |
| `exp_0305b_B_front_tail_lock` | 1.4601 | 0.5722 | -0.1255 | -0.0553 |

Reference baseline (`noisy_through_teacher`):

- PESQ = 1.6765
- STOI = 0.5266

Relative to `noisy_through_teacher`:

- A: PESQ `-0.1248`, STOI `+0.0550`
- B: PESQ `-0.2164`, STOI `+0.0456`

## 4) Training Dynamics (from exp_0305b history)

- A (`tail_lock`) best epoch: 12
- best `val_wav_mse` = 0.025902
- final `val_wav_mse` = 0.029915 (`+15.49%` vs best)
- B (`front_tail_lock`) best epoch: 10
- best `val_wav_mse` = 0.025738
- final `val_wav_mse` = 0.030211 (`+17.38%` vs best)

Interpretation: both runs reached best point early and then degraded, suggesting later-stage over-optimization/regularization conflict.

## 5) Conclusion

- Between A and B, **A is better** (higher PESQ and STOI).
- Compared to `exp_0224a`, **both A and B are worse** on both PESQ and STOI.
- Current result indicates partial intelligibility gain vs teacher baseline (STOI up), but not a full quality improvement (PESQ down), so the target "preserve original capability + improve denoise + recover high-frequency" is not fully achieved yet.

## 6) Suggested Next Iteration

- Use A as base (not B).
- Earlier stopping near epoch 10-12.
- Add explicit high-frequency-aware objective (or perceptual constraint) to align optimization with PESQ-like quality improvements.
