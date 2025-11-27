# 實驗結果報告：LoRA Encoder Denoising (1126-1)

**日期**：2025-11-26  
**實驗目錄**：`exp_1126/1126-1`  
**實驗類型**：Teacher-Student Knowledge Distillation + LoRA Fine-tuning

---

## 📋 實驗概述

### 架構說明

```
┌─────────────────────────────────────────────────────────────┐
│                    單一 Forward Pass                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🧊 Teacher (凍結)              🔥 Student (LoRA 可訓練)     │
│  ┌─────────────────┐           ┌─────────────────┐          │
│  │ WavTokenizer    │           │ WavTokenizer    │          │
│  │ (原始權重)       │           │ + LoRA 層       │          │
│  └────────┬────────┘           └────────┬────────┘          │
│           │                             │                   │
│     clean_audio                   noisy_audio               │
│           ↓                             ↓                   │
│    teacher_features              student_features           │
│    teacher_codes                 student_codes              │
│                                                             │
│              Loss = Feature MSE + Distance Loss             │
└─────────────────────────────────────────────────────────────┘
```

### 模型參數

| 參數 | 值 |
|------|-----|
| 總參數量 | 161,143,352 |
| 可訓練參數 | 38,512 (0.024%) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Target | Encoder Conv1d layers |

### Loss 函數

```
Total Loss = 1.0 × Feature Loss + 0.1 × Distance Loss + 0.01 × VQ Loss
```

- **Feature Loss**: MSE(student_features, teacher_features)
- **Distance Loss**: mean(distance_matrix[student_codes, teacher_codes])
- **VQ Loss**: VQ commitment loss (實際為 0，因 VQ 層凍結)

---

## 📊 實驗 1：有 Distance Loss (lora_encoder_1126_1)

### 訓練配置

| 參數 | 值 |
|------|-----|
| Epochs | 20 (因錯誤中斷) |   # 音檔保存相關 可先無視
| Batch Size | 16 |
| Learning Rate | 5e-5 |
| Feature Loss Weight | 1.0 |
| Distance Loss Weight | 0.1 |
| VQ Loss Weight | 0.01 |

### 訓練結果

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Feature Loss | Distance Loss |
|-------|------------|----------|-----------|---------|--------------|---------------|
| 1 | 0.370 | 0.391 | **26.6%** | **18.0%** | 0.0423 | 3.28 |
| 2 | 0.382 | 0.382 | 11.6% | 12.6% | 0.0381 | 3.44 |
| 3 | 0.379 | 0.388 | 9.0% | 3.9% | 0.0362 | 3.43 |
| 4 | 0.379 | 0.390 | 7.3% | 2.7% | 0.0353 | 3.44 |
| 5 | 0.385 | 0.388 | 5.2% | 4.4% | 0.0346 | 3.50 |
| 10 | 0.385 | 0.399 | 4.9% | 1.8% | 0.0327 | 3.52 |
| 15 | 0.385 | 0.391 | 4.1% | 3.2% | 0.0317 | 3.54 |
| 20 | 0.382 | **0.382** | 4.2% | **7.1%** | **0.0310** | 3.51 |

### 訓練曲線趨勢

```
Feature Loss:  0.0423 → 0.0310  (↓27%)  ✅ 持續下降
Distance Loss: 3.28 → 3.51      (↑7%)   ❌ 無法優化
Token Acc:     26.6% → 4.2%     (↓84%)  ❌ 大幅下降
```

---

## 🔍 問題分析

### 1. Loss 權重失衡

```
實際 Loss 組成：
- Feature Loss 貢獻：0.031 × 1.0 = 0.031 (~8%)
- Distance Loss 貢獻：3.5 × 0.1 = 0.35 (~92%)
```

**問題**：Distance Loss 主導了總 Loss，但它是 discrete 操作，梯度無法回傳。

### 2. Token Accuracy 持續下降

| 現象 | 原因分析 |
|------|----------|
| Epoch 1 高準確率 (26.6%) | 模型尚未改變，接近原始 WavTokenizer |
| 之後快速下降 | LoRA 修改特徵分布，VQ 映射到不同 codebook entry |
| Feature Loss ↓ 但 Acc ↓ | 特徵 MSE 小不代表 VQ 結果相同 |

### 3. Distance Matrix 問題（已修復）

**發現**：原始 distance matrix 是負數 [-979.7, 0.0]

**修復**：重新計算正確的 L2 距離 [0.0, 31.3]

```python
# 修正後的計算
distance_matrix = torch.cdist(codebook, codebook, p=2)  # L2 距離
```

---

## 💡 關鍵洞察

### Feature Loss ↓ 但 Token Accuracy ↓ 的原因

```
student_features ≈ teacher_features (MSE 小)
        ↓ VQ 量化 (非線性、不可微分)
student_codes ≠ teacher_codes (不一定相同)
```

**結論**：小的特徵變化可能導致大的 token 變化，因為 VQ 是 nearest neighbor 搜索。

---

## 🎯 改進策略

### 策略 B：純 Feature Distillation (正在實驗)

```python
Feature Loss Weight = 1.0
Distance Loss Weight = 0.0  # 關閉
VQ Loss Weight = 0.0        # 關閉
```

**理論依據**：
1. 移除無法優化的 Distance Loss
2. 專注於 Feature MSE 優化
3. 如果 feature 真正一致，經過相同 VQ codebook 後 token 應該也會一致

### 實驗檔案

- `run_train.sh`：原始實驗（有 Distance Loss）
- `run_train_FD.sh`：純 Feature Distillation 實驗

---

## 📁 檔案結構

```
exp_1126/1126-1/
├── experiments/
│   ├── lora_encoder_1126_1/           # 有 Distance Loss
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   ├── plots/
│   │   ├── audio_samples/
│   │   ├── config.json
│   │   └── training_history.json
│   └── lora_encoder_1126_1_FD/        # 純 Feature Distillation
│       └── ...
├── model.py                           # Teacher-Student 模型
├── losses.py                          # Loss 函數
├── data.py                            # 數據載入
├── train.py                           # 訓練腳本
├── config.py                          # 配置
├── wavtok_distance_mat_corrected.pt   # 修正後的距離矩陣
├── run_train.sh                       # 原始實驗腳本
├── run_train_FD.sh                    # FD 實驗腳本
└── EXPERIMENT_RESULTS.md              # 本報告
```

---

## 📈 下一步計畫

1. **等待 FD 實驗完成**：比較有/無 Distance Loss 的差異
2. **如果 FD 也不行**：考慮增加 LoRA rank 或調整 target modules
3. **監控指標**：
   - Feature Loss 是否持續下降
   - Token Accuracy 是否穩定或上升

---

## 🔧 技術修復記錄

| 問題 | 修復 | 日期 |
|------|------|------|
| Distance Matrix 為負數 | 重新計算正確 L2 距離 | 2025-11-26 |
| JSON 無法序列化 Tensor | 將 learning rate 轉為 float | 2025-11-26 |
| train.py best_path 未定義 | 添加 best_path 定義 | 2025-11-26 |

---

*報告生成時間：2025-11-26*
