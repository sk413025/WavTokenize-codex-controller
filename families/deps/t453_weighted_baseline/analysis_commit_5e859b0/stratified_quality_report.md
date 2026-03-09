# Stratified Quality Report (exp_0216)

- split: `val`
- epoch: `300`
- num_samples: 100
- top10_mass_proxy: 0.1398

## T453 ratio bins
| bin | n | feature_mse | ΔPESQ | ΔSTOI |
|---|---:|---:|---:|---:|
| [0,0.1) | 22 | 0.03452 | -0.2185 | -0.0560 |
| [0.1,0.2) | 33 | 0.03494 | -0.2766 | -0.0419 |
| [0.2,0.3) | 21 | 0.03293 | -0.4005 | -0.0447 |
| [0.3,0.5] | 24 | 0.03773 | -0.3007 | -0.1035 |

## SNR bins
| bin | n | feature_mse | ΔPESQ | ΔSTOI |
|---|---:|---:|---:|---:|
| <0dB | 80 | 0.03795 | -0.2613 | -0.0692 |
| 0~10dB | 20 | 0.02366 | -0.4329 | -0.0249 |
| 10~20dB | 0 | NA | NA | NA |
| >20dB | 0 | NA | NA | NA |

## Duration bins
| bin | n | feature_mse | ΔPESQ | ΔSTOI |
|---|---:|---:|---:|---:|
| <2s | 0 | NA | NA | NA |
| 2~5s | 100 | 0.03509 | -0.2956 | -0.0604 |
| >5s | 0 | NA | NA | NA |

## Worst 10% Val Samples (by ΔPESQ)
| rank | cache_index | ΔPESQ | ΔSTOI | t453 | snr_db | dur_s | noisy_path |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1360 | -0.8803 | -0.0226 | 0.1284 | -5.22 | 4.00 | `/home/sbplab/ruizi/WavTokenize/data/raw/papercup/nor_boy7_papercup_LDV_208.wav` |
| 2 | 185 | -0.7530 | -0.0754 | 0.0800 | -5.88 | 3.00 | `/home/sbplab/ruizi/WavTokenize/data/raw/box/nor_girl9_box_LDV_185.wav` |
| 3 | 278 | -0.7495 | -0.0510 | 0.2535 | 0.21 | 2.78 | `/home/sbplab/ruizi/WavTokenize/data/raw/box/nor_girl9_box_LDV_278.wav` |
| 4 | 95 | -0.7199 | -0.0843 | 0.2630 | -5.50 | 3.00 | `/home/sbplab/ruizi/WavTokenize/data/raw/box/nor_girl9_box_LDV_005.wav` |
| 5 | 120 | -0.7109 | -0.0463 | 0.0372 | 0.59 | 3.00 | `/home/sbplab/ruizi/WavTokenize/data/raw/box/nor_girl9_box_LDV_120.wav` |
| 6 | 17 | -0.6992 | -0.1419 | 0.3655 | -2.49 | 2.84 | `/home/sbplab/ruizi/WavTokenize/data/raw/box/nor_girl9_box_LDV_083.wav` |
| 7 | 62 | -0.6589 | -0.1645 | 0.3363 | -4.84 | 2.92 | `/home/sbplab/ruizi/WavTokenize/data/raw/box/nor_girl9_box_LDV_038.wav` |
| 8 | 1110 | -0.5907 | -0.0981 | 0.3560 | 0.34 | 2.58 | `/home/sbplab/ruizi/WavTokenize/data/raw/papercup/nor_girl9_papercup_LDV_246.wav` |
| 9 | 372 | -0.5802 | 0.0232 | 0.2055 | -1.61 | 3.40 | `/home/sbplab/ruizi/WavTokenize/data/raw/box/nor_boy7_box_LDV_016.wav` |
| 10 | 419 | -0.5751 | -0.0420 | 0.2117 | 0.68 | 4.00 | `/home/sbplab/ruizi/WavTokenize/data/raw/box/nor_boy7_box_LDV_131.wav` |

## Statistical Tests Reference
- Pairwise Mann-Whitney + FDR: `families/deps/t453_weighted_baseline/analysis_commit_5e859b0/statistical_tests_summary.md`
