# Anti-collapse ablation summary

- timestamp: 2026-01-23T05:37:38
- run_dir: exp_0112_intermediate/runs/exp_k_v5_20260120_003843_20260120_003848
- max_steps: 400
- num_epochs: 2
- max_train_samples: 2000
- max_val_samples: 500

## Results
| lambda | val_strict_fw | val_entropy | val_topk_mass | val_KL(stu||tea) |
|---:|---:|---:|---:|---:|
| 0.020 | 0.007987 | 6.151266 | 0.209987 | 1.241666 |
| 0.050 | 0.006733 | 6.192107 | 0.164849 | 1.948029 |
| 0.100 | 0.008225 | 6.321342 | 0.152471 | 1.200900 |
