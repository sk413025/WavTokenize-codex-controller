# exp_1201: Soft Distance Loss 實驗

## 實驗目的

解決 exp_1128 發現的 **Distance Loss 不可微問題**。

---

## 為什麼使用 Teacher-Student 架構？

### 任務目標

我們想讓 WavTokenizer 對噪音有魯棒性：**有噪音的音訊應該產生和乾淨音訊相同的 tokens**。

### 原始架構的問題：LoRA 無法收到梯度

我們想用 LoRA 微調 Encoder，但原始 WavTokenizer 的架構是：

```
audio → Encoder → features → VQ(argmax) → codes → Decoder → reconstructed_audio
                               ↑
                          離散選擇，梯度斷裂！
```

如果直接用 `codes` 計算 loss：

```python
student_codes = vq_layer(student_features)  # argmax，不可微
teacher_codes = vq_layer(teacher_features)  # argmax，不可微
loss = some_loss(student_codes, teacher_codes)
loss.backward()  # ❌ 梯度無法傳回 Encoder，LoRA 學不到東西
```

**問題：`argmax` 是離散操作，梯度在這裡斷掉，LoRA 完全收不到訓練信號。**

### Teacher-Student 解決方案

繞過 `codes`，改在 **VQ 之前的連續 features 空間** 做蒸餾：

```
Teacher (凍結):  clean_audio  → Encoder → teacher_features ──┬── VQ → teacher_codes (用於評估)
                                                             │
Student (LoRA):  noisy_audio  → Encoder → student_features ──┴── Feature Loss (可微!)
                                              ↑
                                         梯度可以傳回 LoRA
```

| 設計選擇 | 原因 |
|---------|------|
| 在 features 空間計算 loss | 繞過 argmax，保持可微 |
| Teacher 凍結 | 提供穩定的監督信號 |
| 只訓練 Encoder 的 LoRA | 輕量微調，保留原始能力 |
| Codebook 凍結 | 保留原始 WavTokenizer 的重建品質 |

### 核心挑戰

Feature Loss 可以讓梯度流回 LoRA，但 **Feature 接近 ≠ Token 相同**（見下方「問題 1」）。

我們需要一個能**直接優化 token 匹配**且**保持可微**的 loss —— 這就是本實驗的目標。

---

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

### 為什麼原架構失敗？

exp_1128 的訓練存在 **兩個獨立但同時發生的問題**：

#### 問題 1：Feature Loss 優化方向可能錯誤

```
目標：讓 Student 和 Teacher 選到相同的 code
實際優化：讓 Student 的 feature 數值接近 Teacher
```

這兩件事 **不等價**！

想像 codebook 是一張被分成多個區域的地圖：
- Teacher 站在 Code 2 區域的某個點
- Student 原本也在 Code 2 區域
- Feature Loss 告訴 Student：「往 Teacher 的方向走」
- Student 走著走著，**不小心跨過邊界，走到 Code 1 區域了**

| 階段 | Teacher Feature | Student Feature | Code Match |
|------|-----------------|-----------------|------------|
| 初始 | [0.48, 0.52] → Code 2 | [0.45, 0.55] → Code 2 | ✅ |
| 優化後 | [0.48, 0.52] → Code 2 | [0.53, 0.47] → Code 1 | ❌ |

**Feature 更接近了，但 code 不同了！** Loss 只看「數值距離」，不看「是否在同一區域」。

#### 問題 2：Distance Loss 完全沒有梯度

就算加了 Distance Loss 想補救：

```python
student_code = argmax(student_features)  # ← 這行沒有梯度！
teacher_code = argmax(teacher_features)
distance_loss = distance_matrix[student_code, teacher_code]  # 無法反向傳播
```

`argmax`（選最大值）是離散操作，沒有導數。就像問「1.9 四捨五入是 2，請問 2 對 1.9 的導數是多少？」—— 沒有意義。

#### 問題疊加的結果

| Loss | 有梯度？ | 優化方向對嗎？ | 實際效果 |
|------|---------|---------------|---------|
| Feature Loss | ✅ | ❌ 可能導致越界 | 在訓練，但方向可能錯 |
| Distance Loss | ❌ | ✅ | **完全無效** |

整個訓練只有 Feature Loss 在作用，而它可能把 Student 推到錯誤的 code 區域。

### 解決方案

使用 **Straight-Through Estimator (STE)** 或 **Gumbel-Softmax** 讓梯度穿過離散選擇：

```
舊方法 (不可微):
  student_features → argmax → student_codes → distance_matrix[codes] → loss
                       ↑
                    梯度斷裂

新方法 (STE / Gumbel-Softmax):
  Forward:  student_features → argmax → hard_codes → distance → loss
  Backward: student_features ← softmax ← soft_probs ← distance ← loss
                                 ↑
                           梯度繞道傳回！
```

**核心技巧**：Forward 時使用 hard codes（離散），Backward 時假裝用的是 soft codes（連續），讓梯度能流回 Encoder。

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
│  │ STE/Gumbel   │                        │   VQ Layer   │               │
│  │  (可微!)     │                        │  (argmax)    │               │
│  │ Fwd: argmax  │                        │              │               │
│  │ Bwd: softmax │                        │              │               │
│  └──────┬───────┘                        └──────┬───────┘               │
│         │                                       │                       │
│         ▼                                       ▼                       │
│  ┌──────────────┐                        ┌──────────────┐               │
│  │ Student Codes│                        │Teacher Codes │               │
│  │  (B*T,)      │───────┬────────────────│  (B*T,)      │               │
│  │  離散 index  │       │                │  離散 index  │               │
│  └──────────────┘       │                └──────────────┘               │
│                         │                                               │
│                  ┌──────┴──────┐                                        │
│                  │  Distance   │                                        │
│                  │    Loss     │ ← 可微! 梯度可傳回 LoRA                │
│                  └─────────────┘                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Loss 組成:
  total_loss = feature_loss_weight * feature_loss
             + distance_loss_weight * distance_loss  ← STE/Gumbel，可微!
             + vq_loss_weight * vq_loss
```

---

## STE 與 Gumbel-Softmax 原理

### 核心問題

`argmax` 沒有梯度，因為它是階梯函數：

```
argmax([0.1, 0.3, 0.6]) = 2

對 input 微小擾動，output 不變（或跳躍）→ 導數為 0（或未定義）
```

### 解決方案：Straight-Through Estimator (STE)

**Forward**: 用 `argmax` 得到 hard codes（正常的離散選擇）
**Backward**: 假裝用的是 `softmax`，讓梯度流過

```python
# STE 實現
soft_probs = softmax(-distances / τ)           # (B*T, 4096) 連續機率
hard_indices = distances.argmin(dim=-1)        # (B*T,) 離散 index
hard_one_hot = one_hot(hard_indices)           # (B*T, 4096) one-hot

# 魔法：forward 用 hard，backward 用 soft
ste_codes = hard_one_hot - soft_probs.detach() + soft_probs
#           ↑ forward 值    ↑ 不參與梯度       ↑ 梯度來源
```

### Gumbel-Softmax

在 STE 基礎上加入 **Gumbel 噪聲**，讓採樣有隨機性：

```python
# Gumbel noise
gumbels = -log(-log(uniform(0, 1)))

# Gumbel-Softmax
logits = -distances + gumbels
soft_probs = softmax(logits / τ)

# 同樣用 STE 技巧
gumbel_codes = hard_one_hot - soft_probs.detach() + soft_probs
```

### STE vs Gumbel-Softmax

| 特性 | STE | Gumbel-Softmax |
|------|-----|----------------|
| Forward | 確定性 (argmax) | 隨機性 (Gumbel + argmax) |
| Backward | softmax 梯度 | Gumbel-softmax 梯度 |
| 優點 | 訓練穩定 | 幫助探索，避免局部最優 |
| 缺點 | 可能陷入局部最優 | 訓練較不穩定 |

### 為什麼這樣可以訓練？

雖然 forward 用的是 hard codes，但 backward 時：
- 梯度來自 `soft_probs`，這是連續且可微的
- 梯度告訴 Encoder：「如果你把 feature 往這個方向調，soft_probs 會改變，進而影響 loss」
- 長期來看，這會讓 Encoder 學會產生更接近正確 code 的 features

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

### exp3 (CE) & exp4 (Margin) 結果 (2025-12-02)

| Metric | CE (50 epochs) | Margin (30 epochs) | 優勝者 |
|--------|----------------|-------------------|--------|
| Train Loss | 3.6848 | 0.9004 | Margin |
| Val Loss | 4.1776 | 0.9894 | Margin |
| Train Feature Loss | 0.0294 | 0.0295 | 相當 |
| Val Feature Loss | 0.0312 | 0.0307 | **Margin** |
| Train Token Acc | 8.95% | 11.12% | **Margin** |
| Val Token Acc | 5.26% | 7.22% | **Margin** |
| Best Val Token Acc | 21.81% (epoch 1) | 10.17% (epoch 21) | CE |

**關鍵發現**：
1. ✅ **Margin Loss 訓練更穩定**：Val Acc 從 5.7% 逐步提升到 10.17%
2. ❌ **CE Loss 嚴重過擬合**：Train Acc 8.95% vs Val Acc 5.26%，且 Best Val Acc 出現在 epoch 1
3. ⚠️ **所有方法的 Token Accuracy 都遠低於預期**：最高只有 ~10%

### 完整實驗比較

| 實驗 | 方法 | Val Token Acc (final) | Best Val Token Acc | 特點 |
|------|------|----------------------|-------------------|------|
| exp1 | Gumbel | 1.75% | 6.42% (epoch 1) | 隨機性太高，不穩定 |
| exp2 | STE | 2.17% | 7.44% (epoch 1) | 間接優化，效果有限 |
| exp3 | CE | 5.26% | 21.81% (epoch 1) | 嚴重過擬合 |
| exp4 | Margin | **7.22%** | **10.17%** (epoch 21) | **最穩定** |

**Codebook 距離統計**：
```
最近鄰平均距離: 1.42
中位距離: 5.32
標準差: 4.56
```

---

## 結論與改進策略

### 實驗結論

| 實驗 | 方法 | 梯度可微？ | Val Token Acc | Best Val Acc | 結論 |
|------|------|-----------|--------------|--------------|------|
| exp_1128 | Hard Distance | ❌ | ~2% | - | 完全無效 |
| exp1 | Gumbel-Softmax | ✅ | 1.75% | 6.42% | 隨機性太高 |
| exp2 | STE | ✅ | 2.17% | 7.44% | 間接優化效果有限 |
| exp3 | Cross-Entropy | ✅ | 5.26% | 21.81% | **嚴重過擬合** |
| exp4 | Margin Loss | ✅ | **7.22%** | **10.17%** | **最穩定** |

### 關鍵發現

1. **STE/Gumbel 成功讓梯度流通**：Distance Loss 從完全無效變成可訓練
2. **Margin Loss 是最穩定的方法**：Val Acc 穩步提升，沒有過擬合
3. **CE Loss 過擬合嚴重**：Best Val Acc 出現在 epoch 1，之後持續下降
4. **所有方法的 Token Accuracy 都很低**：最高只有 ~10%，遠低於預期

### 問題分析

#### 關鍵發現：Epoch 1 的 Token Accuracy 最高

| 實驗 | Epoch 1 Train Acc | Final Train Acc | 下降幅度 |
|------|------------------|-----------------|---------|
| Gumbel | 24.45% | 2.19% | -91% |
| STE | 22.46% | 3.83% | -83% |
| CE | 32.49% | 8.95% | -72% |
| Margin | 27.77% | 11.12% | -60% |

**這是一個非常重要的信號：訓練反而讓 Token Accuracy 下降！**

#### Feature Loss 與 Token Accuracy 的相關性

| 實驗 | Correlation | 含義 |
|------|-------------|------|
| Gumbel | +0.60 | Feature Loss ↓ → Token Acc ↓ |
| STE | +0.69 | Feature Loss ↓ → Token Acc ↓ |
| CE | +0.81 | Feature Loss ↓ → Token Acc ↓ |
| **Margin** | **-0.13** | **Feature Loss 與 Token Acc 無關** |

#### 根本原因：優化方向衝突

```
問題不是「過擬合」，而是「Feature Loss 與 Token Match 目標衝突」
```

1. **Epoch 1 高 Token Accuracy 是「假象」**：
   - 剛初始化的 LoRA 權重接近 0
   - Student 輸出 ≈ 原始 Encoder 輸出
   - 原始 WavTokenizer 對噪音本身就有 ~20-30% 的魯棒性

2. **訓練過程「破壞」了原始魯棒性**：
   - Feature Loss 優化讓 features 在「數值上」更接近 Teacher
   - 但這個「接近」可能跨越 codebook 的決策邊界
   - 結果：Feature Loss ↓ 但 Token Accuracy ↓

3. **Margin Loss 為什麼最穩定**：
   - 它直接約束「到正確 code 的距離 < 到錯誤 code 的距離」
   - 不關心 feature 數值是否接近
   - 所以 Feature Loss 可能上升，但 Token Accuracy 保持

#### 驗證實驗建議

```bash
# 用原始 WavTokenizer（無 LoRA）處理 noisy/clean pairs
# 計算 Token Match Rate
# 如果接近 20-30%，則證實上述假設
```

### 解決方案

#### 方案 1：移除或大幅降低 Feature Loss（最推薦）

```python
# 目前（有問題）
total = 1.0 * feature_loss + 0.1 * distance_loss

# 建議（方案 A）：完全移除 Feature Loss
total = 1.0 * margin_loss  # 只用 Margin Loss

# 建議（方案 B）：降低 Feature Loss 權重
total = 0.1 * feature_loss + 1.0 * margin_loss
```

**理由**：Feature Loss 是導致 Token Accuracy 下降的主因

#### 方案 2：Early Stopping（基於 Token Accuracy）

```python
# 監控 Val Token Accuracy
# 在連續 N epochs 下降時停止
# 最佳停止點可能在 Epoch 1-5
```

**理由**：訓練越久，Token Accuracy 越低

#### 方案 3：驗證原始模型魯棒性

在做任何訓練之前，先測試：
```bash
# 不加 LoRA，直接用原始 WavTokenizer
# 處理 noisy/clean pairs，計算 Token Match Rate
```

如果原始模型已有 20-30% match rate，可能：
- 不需要訓練
- 或需要完全不同的方法（如 test-time adaptation）

#### 方案 4：純 Margin Loss 實驗

```python
# Margin Loss 是唯一與 Token Accuracy 負相關的方法
# 嘗試只用 Margin Loss，觀察是否能提升 Token Accuracy

class PureMarginLoss:
    def forward(self, ...):
        # 只計算 Margin Loss
        # 完全不用 Feature Loss
        return margin_loss
```

#### 方案 5：調整 LoRA 訓練範圍

```python
# 可能 LoRA 改變太多，破壞了原始魯棒性
# 嘗試只訓練最後一層
target_modules = ["model.9"]  # 只訓練 256→512 那層

# 或降低 LoRA rank
lora_rank = 16  # 從 64 降到 16
```

### 實驗優先級

| 優先級 | 方案 | 預期效果 | 工作量 |
|-------|------|---------|-------|
| 1 | 驗證原始模型魯棒性 | 了解 baseline | 低 |
| 2 | 純 Margin Loss | 可能提升 Token Acc | 低 |
| 3 | 移除 Feature Loss | 可能提升 Token Acc | 低 |
| 4 | Early Stopping | 保留初始魯棒性 | 低 |
| 5 | 調整 LoRA 範圍 | 減少對原始模型的破壞 | 中 |

---

## 相關文件

- [exp_1128/ANALYSIS_REPORT.md](../exp_1128/ANALYSIS_REPORT.md) - 問題分析報告
- [exp_1128/verify_gradient.py](../exp_1128/verify_gradient.py) - 梯度驗證腳本
