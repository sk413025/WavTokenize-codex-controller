# Anti-collapse ablation summary

- timestamp: 2026-01-23T05:26:49
- run_dir: exp_0112_intermediate/runs/exp_k_v5_20260120_003843_20260120_003848
- max_steps: 200
- num_epochs: 2
- max_train_samples: 2000
- max_val_samples: 500

## Results
| lambda | val_strict_fw | val_entropy | val_topk_mass | val_KL(stu||tea) |
|---:|---:|---:|---:|---:|
| 0.000 | 0.008693 | 6.168676 | 0.194691 | 1.324630 |
| 0.005 | 0.008217 | 6.051651 | 0.248081 | 1.185493 |
| 0.010 | 0.008684 | 6.092482 | 0.233816 | 1.184284 |
