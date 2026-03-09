# Statistical Tests Summary (Commit 5e859b0)

- source: `families/deps/t453_weighted_baseline/analysis_commit_5e859b0/audio_quality_by_epoch.json`
- n(train_all_epochs): 700
- n(val_all_epochs): 700
- n(val_epoch300): 100

## Spearman Correlations
### train_all_epochs
- feature_mse_vs_delta_pesq: rho=-0.1870, p=0.000001, n=700
- feature_mse_vs_delta_stoi: rho=-0.5135, p=0.000000, n=700
- t453_ratio_vs_delta_pesq: rho=-0.0415, p=0.273343, n=700
- t453_ratio_vs_delta_stoi: rho=0.0414, p=0.273584, n=700

### val_all_epochs
- feature_mse_vs_delta_pesq: rho=0.2739, p=0.000000, n=700
- feature_mse_vs_delta_stoi: rho=-0.3827, p=0.000000, n=700
- t453_ratio_vs_delta_pesq: rho=-0.2018, p=0.000000, n=700
- t453_ratio_vs_delta_stoi: rho=-0.2741, p=0.000000, n=700

### val_epoch300
- feature_mse_vs_delta_pesq: rho=0.2828, p=0.004357, n=100
- feature_mse_vs_delta_stoi: rho=-0.4094, p=0.000023, n=100
- t453_ratio_vs_delta_pesq: rho=-0.1957, p=0.051011, n=100
- t453_ratio_vs_delta_stoi: rho=-0.2920, p=0.003205, n=100

## Val Epoch Bootstrap (ΔPESQ / ΔSTOI)
| epoch | n | ΔPESQ mean | 95% CI | ΔSTOI mean | 95% CI |
|---|---:|---:|---:|---:|---:|
| epoch_050 | 100 | -0.2846 | [-0.3267, -0.2436] | -0.0674 | [-0.0771, -0.0575] |
| epoch_100 | 100 | -0.2886 | [-0.3321, -0.2476] | -0.0645 | [-0.0748, -0.0540] |
| epoch_150 | 100 | -0.2968 | [-0.3374, -0.2556] | -0.0637 | [-0.0738, -0.0536] |
| epoch_200 | 100 | -0.2966 | [-0.3392, -0.2540] | -0.0659 | [-0.0760, -0.0562] |
| epoch_220 | 100 | -0.2956 | [-0.3377, -0.2552] | -0.0612 | [-0.0708, -0.0510] |
| epoch_250 | 100 | -0.3032 | [-0.3444, -0.2621] | -0.0571 | [-0.0666, -0.0471] |
| epoch_300 | 100 | -0.2956 | [-0.3379, -0.2547] | -0.0604 | [-0.0702, -0.0503] |

## Pairwise Mann-Whitney + FDR (val@epoch300)
| dimension | metric | bin_a | bin_b | n_a | n_b | p | q_fdr | sig(q<0.05) |
|---|---|---|---|---:|---:|---:|---:|---|
| t453_ratio | delta_pesq | [0,0.1) | [0.1,0.2) | 22 | 33 | 0.232469 | 0.271213 | False |
| t453_ratio | delta_pesq | [0,0.1) | [0.2,0.3) | 22 | 21 | 0.005403 | 0.012608 | True |
| t453_ratio | delta_pesq | [0,0.1) | [0.3,0.5] | 22 | 24 | 0.183389 | 0.246462 | False |
| t453_ratio | delta_pesq | [0.1,0.2) | [0.2,0.3) | 33 | 21 | 0.025372 | 0.050743 | False |
| t453_ratio | delta_pesq | [0.1,0.2) | [0.3,0.5] | 33 | 24 | 0.692115 | 0.745355 | False |
| t453_ratio | delta_pesq | [0.2,0.3) | [0.3,0.5] | 21 | 24 | 0.142260 | 0.221294 | False |
| t453_ratio | delta_stoi | [0,0.1) | [0.1,0.2) | 22 | 33 | 0.108197 | 0.189345 | False |
| t453_ratio | delta_stoi | [0,0.1) | [0.2,0.3) | 22 | 21 | 0.193648 | 0.246462 | False |
| t453_ratio | delta_stoi | [0,0.1) | [0.3,0.5] | 22 | 24 | 0.000864 | 0.002419 | True |
| t453_ratio | delta_stoi | [0.1,0.2) | [0.2,0.3) | 33 | 21 | 0.749434 | 0.749434 | False |
| t453_ratio | delta_stoi | [0.1,0.2) | [0.3,0.5] | 33 | 24 | 0.000034 | 0.000474 | True |
| t453_ratio | delta_stoi | [0.2,0.3) | [0.3,0.5] | 21 | 24 | 0.000199 | 0.000931 | True |
| snr_db | delta_pesq | <0dB | 0~10dB | 80 | 20 | 0.000446 | 0.001559 | True |
| snr_db | delta_stoi | <0dB | 0~10dB | 80 | 20 | 0.000169 | 0.000931 | True |
