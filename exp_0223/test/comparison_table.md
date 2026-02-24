# PESQ/STOI 比較表

**Val indices**: [51, 54, 61]
**評估條件**: PESQ nb (8kHz resample), STOI (24kHz)

## 各組平均結果

> `noisy_through_teacher` = noisy → Teacher Encoder+VQ → Frozen Decoder（公平比較基準）
> 其餘實驗的 ΔPESQ / ΔSTOI 皆相對此基準計算

| 實驗 | PESQ (recon) | STOI (recon) | ΔPESQ vs teacher_baseline | ΔSTOI vs teacher_baseline |
|------|-------------|-------------|--------------------------|--------------------------|
| **noisy_through_teacher** (baseline) | 1.6765 | 0.5266 | — | — |
| noisy_through_teacher_no_vq | 1.7075 | 0.5312 | +0.0310 | +0.0046 |
| V2 | 1.2312 | 0.4791 | -0.4453 | -0.0475 |
| Plan_Ori | 1.2514 | 0.4833 | -0.4251 | -0.0433 |
| exp_0216 | 1.1996 | 0.4601 | -0.4769 | -0.0665 |
| exp_0217 | 1.2032 | 0.4620 | -0.4733 | -0.0646 |
| exp_0223_v2 | 1.2053 | 0.5216 | -0.4712 | -0.0050 |
| exp_0224a (No-VQ, ep190) | 1.5856 | 0.6275 | -0.0909 | +0.1009 |
| exp_0224b_ep16 (No-VQ+DecLoRA, ep31†) | **1.8090** | **0.6559** | **+0.1325** | **+0.1293** |
| exp_0224b_ep20 (No-VQ+DecLoRA, ep20) | 1.7120 | 0.6526 | +0.0355 | +0.1260 |

> † best_model.pt 實際為 epoch 31（val_mse 準則，訓練仍在進行中）
> exp_0224b 訓練中（目前 ~30/300 epochs），上述為中期結果

## Per-Sample 明細

### noisy_through_teacher (baseline)
| Sample | PESQ recon | PESQ noisy | STOI recon | STOI noisy |
|--------|-----------|-----------|-----------|-----------|
| 1 | 1.6786 | 2.3639 | 0.5915 | 0.6654 |
| 2 | 1.8778 | 2.5907 | 0.4647 | 0.5984 |
| 3 | 1.4731 | 2.4520 | 0.5237 | 0.6314 |

### noisy_through_teacher_no_vq (Teacher Encoder, 跳過VQ)
| Sample | PESQ recon | PESQ noisy | STOI recon | STOI noisy |
|--------|-----------|-----------|-----------|-----------|
| 1 | 1.7052 | 2.3639 | 0.5942 | 0.6654 |
| 2 | 1.8343 | 2.5907 | 0.4739 | 0.5984 |
| 3 | 1.5831 | 2.4520 | 0.5256 | 0.6314 |

### V2
| Sample | PESQ recon | PESQ noisy | STOI recon | STOI noisy |
|--------|-----------|-----------|-----------|-----------|
| 1 | 1.2912 | 2.3639 | 0.5195 | 0.6654 |
| 2 | 1.1326 | 2.5907 | 0.4444 | 0.5984 |
| 3 | 1.2696 | 2.4520 | 0.4733 | 0.6314 |

### Plan_Ori
| Sample | PESQ recon | PESQ noisy | STOI recon | STOI noisy |
|--------|-----------|-----------|-----------|-----------|
| 1 | 1.2494 | 2.3639 | 0.5172 | 0.6654 |
| 2 | 1.1180 | 2.5907 | 0.4438 | 0.5984 |
| 3 | 1.3868 | 2.4520 | 0.4890 | 0.6314 |

### exp_0216
| Sample | PESQ recon | PESQ noisy | STOI recon | STOI noisy |
|--------|-----------|-----------|-----------|-----------|
| 1 | 1.2319 | 2.3639 | 0.5005 | 0.6654 |
| 2 | 1.1058 | 2.5907 | 0.4261 | 0.5984 |
| 3 | 1.2610 | 2.4520 | 0.4537 | 0.6314 |

### exp_0217
| Sample | PESQ recon | PESQ noisy | STOI recon | STOI noisy |
|--------|-----------|-----------|-----------|-----------|
| 1 | 1.2562 | 2.3639 | 0.4902 | 0.6654 |
| 2 | 1.1149 | 2.5907 | 0.4414 | 0.5984 |
| 3 | 1.2384 | 2.4520 | 0.4543 | 0.6314 |

### exp_0223_v2
| Sample | PESQ recon | PESQ noisy | STOI recon | STOI noisy |
|--------|-----------|-----------|-----------|-----------|
| 1 | 1.2650 | 2.3639 | 0.5967 | 0.6654 |
| 2 | 1.1219 | 2.5907 | 0.4699 | 0.5984 |
| 3 | 1.2290 | 2.4520 | 0.4982 | 0.6314 |

### exp_0224a (No-VQ, ep190)
| Sample | PESQ recon | PESQ noisy | STOI recon | STOI noisy |
|--------|-----------|-----------|-----------|-----------|
| 1 | 1.5962 | 2.3639 | 0.6607 | 0.6654 |
| 2 | 1.6398 | 2.5907 | 0.5928 | 0.5984 |
| 3 | 1.5208 | 2.4520 | 0.6289 | 0.6314 |

### exp_0224b_ep16 (No-VQ+DecLoRA, best_model.pt = ep31†)
| Sample | PESQ recon | PESQ noisy | STOI recon | STOI noisy |
|--------|-----------|-----------|-----------|-----------|
| 1 | 1.9359 | 2.3639 | 0.7031 | 0.6654 |
| 2 | 1.7923 | 2.5907 | 0.6121 | 0.5984 |
| 3 | 1.6987 | 2.4520 | 0.6526 | 0.6314 |

### exp_0224b_ep20 (No-VQ+DecLoRA, epoch 20)
| Sample | PESQ recon | PESQ noisy | STOI recon | STOI noisy |
|--------|-----------|-----------|-----------|-----------|
| 1 | 1.7947 | 2.3639 | 0.7006 | 0.6654 |
| 2 | 1.7287 | 2.5907 | 0.6062 | 0.5984 |
| 3 | 1.6125 | 2.4520 | 0.6511 | 0.6314 |

## 音檔說明

各子目錄包含 3 組音檔（sampleNN_noisy / clean / recon）：
- `noisy`: 原始帶噪輸入（LDV 感測器音訊）
- `clean`: 對應乾淨音訊（ground truth）
- `recon`: 各實驗重建輸出

## 解讀說明

- **PESQ(noisy) >> PESQ(recon)** 屬正常現象：LDV noisy 與 clean 時序高度對齊，PESQ 對時序對齊敏感
- **公平基準** (`noisy_through_teacher`) = noisy 直接經過相同的 Encoder+VQ+Decoder pipeline
- **noisy_through_teacher_no_vq**：跳過 VQ 後 PESQ 僅微升 +0.031（1.677→1.708），說明 VQ 本身損失有限
- **exp_0224b_ep16** 突破 teacher baseline！PESQ=1.809 > 1.677（+0.132），STOI=0.656 > 0.527（+0.129）
- exp_0224b 超越 no_vq baseline（1.809 > 1.708），說明 Decoder LoRA 有實質貢獻（不只是跳過 VQ）
- best_model.pt 顯示的 epoch 31 而非 16，表示訓練期間有更新覆蓋（val_mse 更低的 checkpoint）
