# exp_1201: Soft Distance Loss 實驗

## 實驗目的

解決 exp_1128 發現的 **Distance Loss 不可微問題**。

### 背景問題

在 exp_1128 中，我們發現：
```
Distance Loss (當前實現):
  requires_grad: False
  grad_fn: None
  ❌ Backward 失敗
```

原因：`distance_loss = distance_matrix[argmax(student), argmax(teacher)]`
- `argmax` 操作不可微
- `indexing` 操作切斷梯度圖

導致 `distance_loss_weight` 參數完全無效，實際只有 `feature_loss` 在訓練。

### 解決方案

實現 **Soft Distance Loss**：用 `softmax` 取代 `argmax`

```
舊方法 (不可微):
  student_features → argmax → student_codes → distance_matrix[codes] → loss
                       ↑
                    梯度斷裂

新方法 (可微):
  student_features → softmax → soft_codes → weighted_distance → loss
                       ↑
                    梯度保持
```

---

## 架構圖

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Teacher-Student Distillation                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐                         ┌─────────────┐                │
│  │ Noisy Audio │                         │ Clean Audio │                │
│  └──────┬──────┘                         └──────┬──────┘                │
│         │                                       │                       │
│         ▼                                       ▼                       │
│  ┌──────────────┐                        ┌──────────────┐               │
│  │   Student    │                        │   Teacher    │               │
│  │  Encoder     │                        │  Encoder     │               │
│  │  (+ LoRA)    │                        │  (Frozen)    │               │
│  └──────┬───────┘                        └──────┬───────┘               │
│         │                                       │                       │
│         ▼                                       ▼                       │
│  ┌──────────────┐                        ┌──────────────┐               │
│  │   Student    │                        │   Teacher    │               │
│  │  Features    │───────┬────────────────│  Features    │               │
│  │  (B,512,T)   │       │                │  (B,512,T)   │               │
│  └──────┬───────┘       │                └──────┬───────┘               │
│         │               │                       │                       │
│         │        ┌──────┴──────┐                │                       │
│         │        │ Feature MSE │                │                       │
│         │        │    Loss     │                │                       │
│         │        └─────────────┘                │                       │
│         │                                       │                       │
│         ▼                                       ▼                       │
│  ┌──────────────┐                        ┌──────────────┐               │
│  │   Softmax    │                        │   VQ Layer   │               │
│  │  (可微!)     │                        │  (argmax)    │               │
│  │  -dist/τ    │                        │              │               │
│  └──────┬───────┘                        └──────┬───────┘               │
│         │                                       │                       │
│         ▼                                       ▼                       │
│  ┌──────────────┐                        ┌──────────────┐               │
│  │  Soft Codes  │                        │Teacher Codes │               │
│  │  (B*T,4096)  │───────┬────────────────│  (B*T,)      │               │
│  │  機率分布    │       │                │  離散 index  │               │
│  └──────────────┘       │                └──────────────┘               │
│                         │                                               │
│                  ┌──────┴──────┐                                        │
│                  │ Soft Dist   │                                        │
│                  │    Loss     │ ← 可微! 梯度可傳回 LoRA                │
│                  └─────────────┘                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Loss 組成:
  total_loss = feature_loss_weight * feature_loss
             + soft_dist_loss_weight * soft_distance_loss  ← 新增，可微!
             + vq_loss_weight * vq_loss
```

---

## Soft Distance Loss 原理

### 數學公式

```
1. 計算 student features 到所有 codes 的距離:
   distances[i,j] = ||student_features[i] - codebook[j]||₂

2. 轉換為機率分布 (softmax):
   soft_codes[i,j] = softmax(-distances[i,:] / τ)[j]

   其中 τ (temperature) 控制分布的「軟硬程度」:
   - τ → 0: 接近 one-hot (hard)
   - τ → ∞: 接近 uniform (soft)

3. 計算期望距離:
   expected_distance[i] = Σⱼ soft_codes[i,j] * dist_matrix[teacher_code[i], j]

4. Loss:
   soft_distance_loss = mean(expected_distance)
```

### 為什麼可微？

- `softmax` 是可微操作
- `weighted sum` 是可微操作
- 梯度可以從 loss → soft_codes → distances → student_features → LoRA

---

## 實驗設置

### 超參數

| 參數 | 值 | 說明 |
|------|-------|------|
| LoRA Rank | 64 | 沿用 exp_1128 最佳配置 |
| LoRA Alpha | 128 | |
| Learning Rate | 5e-5 | |
| Batch Size | 16 | |
| Epochs | 50 | |
| Feature Loss Weight | 1.0 | |
| **Soft Dist Loss Weight** | **0.1** | 新參數 |
| **Temperature (τ)** | **1.0** | 新參數 |
| VQ Loss Weight | 0.0 | Codebook 凍結 |

### Distance Loss 模式

本實驗支持五種可微的 Distance Loss 模式：

| 模式 | Forward | Backward | 特點 |
|------|---------|----------|------|
| `soft` | 期望距離 | softmax 梯度 | 所有 codes 的加權平均距離 |
| `gumbel` | hard codes + Gumbel noise | Gumbel-softmax 梯度 | 隨機性幫助探索 codebook |
| `ste` | hard codes (argmax) | softmax 梯度 | 確定性，訓練穩定 |
| `ce` | Cross-Entropy | 分類梯度 | 直接優化 Token Accuracy |
| `margin` | max(0, d_correct - d_wrong + m) | margin 梯度 | 優化決策邊界 |

### 實驗矩陣

| 實驗 | Mode | Temperature | 其他參數 | 目標 |
|------|------|-------------|----------|------|
| **exp1** | **gumbel** | 1.0 | dist_weight=0.1 | Gumbel-Softmax (隨機探索) |
| **exp2** | **ste** | 1.0 | dist_weight=0.1 | STE (確定性) |
| **exp3** | **ce** | 0.5 | dist_weight=0.5, label_smoothing=0.05 | Cross-Entropy 直接監督 |
| **exp4** | **margin** | - | margin=0.5, dist_weight=0.3 | Margin Loss 決策邊界優化 |

比較目的：
- **Gumbel**: 引入隨機性，可能幫助逃離局部最優
- **STE**: 確定性選擇，訓練更穩定，但可能陷入局部最優
- **CE**: 直接優化 Token Accuracy，將問題視為 4096 類分類
- **Margin**: 優化決策邊界，確保正確 token 與最近錯誤 token 有足夠距離差

---

## 預期改善

| Metric | exp_1128 (無梯度) | exp_1201 (有梯度) |
|--------|------------------|------------------|
| Feature Loss | ✅ 下降 | ✅ 下降 |
| Distance Loss | ↔️ 不變 | ✅ 應該下降 |
| Token Accuracy | ⬇️ 下降 | ✅ 應該改善或持平 |
| 音質 | ❓ 待評估 | ✅ 應該改善 |

---

## 輸出檔案

每 10 epoch 儲存：

```
experiments/<exp_name>/
├── checkpoints/
│   ├── latest.pt
│   ├── best.pt
│   └── epoch_010_loss_X.XX.pt
├── plots/
│   └── training_curves_epoch_010_YYYYMMDD_HHMMSS.png
├── audio_samples/
│   └── epoch_010/
│       ├── train_1_noisy.wav
│       ├── train_1_clean.wav
│       ├── train_1_student_pred.wav
│       ├── train_1_teacher_recon.wav
│       ├── val_1_noisy.wav
│       ├── val_1_clean.wav
│       ├── val_1_student_pred.wav
│       └── val_1_teacher_recon.wav
├── spectrograms/
│   └── epoch_010/
│       ├── train_1_comparison.png
│       └── val_1_comparison.png
├── logs/
│   └── events.out.tfevents.*
├── config.json
└── training_history.json
```

---

## 執行方式

```bash
# 單一實驗
bash run_exp1_soft_dist.sh

# 監控
tail -f experiments/soft_dist_baseline/lora_soft_dist.log

# TensorBoard
tensorboard --logdir experiments/soft_dist_baseline/logs
```

---

## 實驗結果

### exp1 (Gumbel) & exp2 (STE) 結果 (2025-12-01)

| Metric | Gumbel (50 epochs) | STE (50 epochs) | 優勝者 |
|--------|-------------------|-----------------|--------|
| Train Loss | 0.4397 | 0.3527 | **STE** |
| Val Loss | 0.4548 | 0.3765 | **STE** |
| Train Feature Loss | 0.0319 | 0.0311 | **STE** |
| Val Feature Loss | 0.0332 | 0.0328 | **STE** |
| Train Token Acc | 2.19% | 3.83% | **STE** |
| Val Token Acc | 1.75% | 2.17% | **STE** |

**關鍵發現**：
1. ✅ Distance Loss 現在可微，梯度成功傳遞
2. ✅ STE 比 Gumbel 更穩定，所有指標都更好
3. ❌ Token Accuracy 仍然很低 (~2%)，需要更直接的優化方法

### 問題分析

Token Accuracy 低的原因：
- Distance-based loss 是間接優化（最小化期望距離 ≠ 最大化 Token Accuracy）
- noisy→clean 任務本身困難，4096 類分類需要更強的監督信號

### exp3 (CE) & exp4 (Margin) 設計理念

基於 exp1/2 的結果，設計新實驗直接優化 Token Accuracy：

**exp3 (Cross-Entropy)**：
- 將 token 選擇視為 4096 類分類問題
- 配置：`temperature=0.5`, `dist_weight=0.5`, `label_smoothing=0.05`
- 歷史經驗（Exp3-2 分支）：Standard CE 達到 48.16% Val Acc

**exp4 (Margin)**：
- 優化決策邊界：`loss = max(0, d_correct - d_wrong + margin)`
- 配置：`margin=0.5`（< 最近鄰距離 1.42）, `dist_weight=0.3`
- 對 codebook 幾何結構更友好

**Codebook 距離統計**：
```
最近鄰平均距離: 1.42
中位距離: 5.32
標準差: 4.56
```

---

## 相關文件

- [exp_1128/ANALYSIS_REPORT.md](../exp_1128/ANALYSIS_REPORT.md) - 問題分析報告
- [exp_1128/verify_gradient.py](../exp_1128/verify_gradient.py) - 梯度驗證腳本
