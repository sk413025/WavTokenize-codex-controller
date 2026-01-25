# Probe noise alignment

- timestamp: 2026-01-25T02:45:55
- checkpoint: exp_0112_intermediate/runs/exp_k_v5_20260120_003843_20260120_003848/best_model.pt
- train_samples: 2000
- val_samples: 500

- noise_type_accuracy: 0.6320
- corr(true_prob, acc): pearson -0.1290, spearman -0.1158
- corr(max_prob, acc): pearson -0.0178, spearman -0.0242
- acc_high_conf (top25%): 0.0080
- acc_low_conf (bottom25%): 0.0092

## Confusion matrix (val)
[[137, 13, 89], [1, 174, 3], [31, 47, 5]]

