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

### 實驗矩陣

| 實驗 | Soft Dist Weight | Temperature | 目標 |
|------|-----------------|-------------|------|
| exp1 | 0.1 | 1.0 | Baseline |
| exp2 | 0.5 | 1.0 | 更強的 code 對齊壓力 |
| exp3 | 0.1 | 0.5 | 更 sharp 的分布 |
| exp4 | 0.1 | 2.0 | 更 soft 的分布 |

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

## 相關文件

- [exp_1128/ANALYSIS_REPORT.md](../exp_1128/ANALYSIS_REPORT.md) - 問題分析報告
- [exp_1128/verify_gradient.py](../exp_1128/verify_gradient.py) - 梯度驗證腳本
