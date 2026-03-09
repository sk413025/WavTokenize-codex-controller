# Audio Quality by Epoch (exp_0216)

- run_dir: `/home/sbplab/ruizi/WavTokenize-feature-analysis/families/deps/encoder_aug/runs/augmented_long_20260216`
- target_sr_model: 24000
- target_sr_metric: 16000
- n_target_per_split_epoch: 100

## Notes
- Epoch `222` has no checkpoint file (`checkpoint_epoch222.pt`), so post-VQ audio quality is marked unavailable.
- `feature_mse_mean` here is sample-level post-VQ MSE from evaluated subset (not full-dataset epoch metric).

## train
| epoch | status | n | feature_mse | ΔPESQ | ΔSTOI |
|---|---:|---:|---:|---:|---:|
| 050 | ok | 100 | 0.03489 | -0.5464 | -0.0407 |
| 100 | ok | 100 | 0.03410 | -0.5132 | -0.0322 |
| 150 | ok | 100 | 0.03206 | -0.5221 | -0.0303 |
| 200 | ok | 100 | 0.03132 | -0.4956 | -0.0220 |
| 220 | ok | 100 | 0.03075 | -0.4782 | -0.0145 |
| 222 | missing_checkpoint | 0 | NA | NA | NA |
| 250 | ok | 100 | 0.03069 | -0.4927 | -0.0158 |
| 300 | ok | 100 | 0.02983 | -0.4824 | -0.0143 |

## val
| epoch | status | n | feature_mse | ΔPESQ | ΔSTOI |
|---|---:|---:|---:|---:|---:|
| 050 | ok | 100 | 0.03596 | -0.2846 | -0.0674 |
| 100 | ok | 100 | 0.03645 | -0.2886 | -0.0645 |
| 150 | ok | 100 | 0.03452 | -0.2968 | -0.0637 |
| 200 | ok | 100 | 0.03488 | -0.2966 | -0.0659 |
| 220 | ok | 100 | 0.03527 | -0.2956 | -0.0612 |
| 222 | missing_checkpoint | 0 | NA | NA | NA |
| 250 | ok | 100 | 0.03484 | -0.3032 | -0.0571 |
| 300 | ok | 100 | 0.03509 | -0.2956 | -0.0604 |

## Correlations (Spearman, subset-level)
- train: corr(feature_mse, ΔPESQ)=-0.8571, corr(feature_mse, ΔSTOI)=-0.9643
- val: corr(feature_mse, ΔPESQ)=0.8929, corr(feature_mse, ΔSTOI)=-0.4286

## Statistical Tests Reference
- Detailed tests: `families/deps/t453_weighted_baseline/analysis_commit_5e859b0/statistical_tests_summary.json`
- Human summary: `families/deps/t453_weighted_baseline/analysis_commit_5e859b0/statistical_tests_summary.md`
