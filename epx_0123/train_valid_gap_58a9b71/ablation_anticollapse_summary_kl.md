# Anti-collapse ablation summary

- timestamp: 2026-01-23T06:22:43
- run_dir: exp_0112_intermediate/runs/exp_k_v5_20260120_003843_20260120_003848
- reg_type: kl
- reg_temperature: 1.0
- max_steps: 400
- num_epochs: 2
- max_train_samples: 2000
- max_val_samples: 500

## Results
| lambda | val_strict_fw | val_entropy | val_topk_mass | val_KL(stu||tea) |
|---:|---:|---:|---:|---:|
| 0.020 | 0.006954 | 6.342970 | 0.112946 | 1.189401 |
| 0.050 | 0.007922 | 6.328810 | 0.141652 | 0.863285 |
| 0.100 | 0.009021 | 6.278591 | 0.180513 | 0.784318 |
