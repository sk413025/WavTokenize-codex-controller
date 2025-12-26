# exp_1226: 診斷與改進實驗

## 背景

基於 Exp55 (目前最佳 baseline，Val Masked Acc = 0.91%)，進行問題診斷和改進實驗。

**更新日期**：2024-12-26

---

## 用戶原始問題與驗證

### 問題一：Token Accuracy 計算邏輯是否有誤？

**原始疑問**：
> Token Accuracy 只有 ~0.9%，但聽起來音檔品質還不錯，計算邏輯是否有問題？

**驗證方式**：`verify_codes_v2.py`

```python
# 比較兩種計算方法
# 方法 1: model.forward() 內部計算
student_encoder_out = model.student.feature_extractor.encodec.encoder(noisy)
student_vq = model.student.feature_extractor.encodec.quantizer(student_encoder_out, ...)
forward_student_codes = student_vq.codes

# 方法 2: feature_extractor() 封裝計算
student_feat, fe_student_codes, _ = model.student.feature_extractor(noisy, bandwidth_id=0)

# 比較結果
assert (forward_student_codes == fe_student_codes).all()  # ✅ 100% 一致
```

**驗證結果**：
```
[Student] 兩種方法的 codes 匹配率: 100.00%
[Teacher] 兩種方法的 codes 匹配率: 100.00%

forward() 方式計算的 Token Accuracy:          5.15%
feature_extractor() 方式計算的 Token Accuracy: 5.15%
```

**結論**：✅ **計算邏輯正確**，沒有 bug。Token Accuracy 確實很低 (~5%)。

---

### 問題二：是否要試試看 Loss 計算時看看前後一個 Frame 哪個分數比較低選哪個？

**原始疑問**：
> 考慮到 noisy 和 clean 可能有時間偏移，是否在計算 Loss 時比較 t-1, t, t+1 三個 frame，選最小的 loss？

**診斷**：`diagnose_alignment.py`

```
[1] Best Lag Statistics (samples @ 24kHz)
    Mean: -136.0 (-5.7 ms)
    Std:  941.0 (39.2 ms)
    Min:  -4640 (-193.3 ms)
    Max:  1760 (73.3 ms)

[2] Analysis
    Encoder stride: 13.3 ms/frame
    Mean lag: 5.7 ms
    Samples with lag > half frame: 22/30 (73.3%)
```

**驗證**：`test_tolerant_acc.py`

```python
# Frame-Tolerant Accuracy 實作
# 對於每個 student frame，比較 teacher 的 t-1, t, t+1
# 選擇匹配的最佳 offset

class FrameTolerantAccuracy:
    def __init__(self, tolerance: int = 1):
        self.tolerance = tolerance  # ±1 frame

    def __call__(self, s_codes, t_codes, lengths):
        # 檢查每個 frame 是否與 t-1, t, t+1 任一匹配
        for offset in range(-self.tolerance, self.tolerance + 1):
            # 計算偏移後的匹配...
```

**測試結果**：
```
=== Tolerant vs Strict Accuracy ===

Average Strict Accuracy:   0.61%
Average Tolerant Accuracy: 1.31%
Average Improvement:       +0.70%
```

**結論**：❌ **改善有限**，Frame-Tolerant 只提升 +0.70%。

**根本原因**：
- 時間偏移不是主要問題
- 主要問題是 **Mode Collapse**（Student codes 集中在少數幾個值）

**但仍然實作此功能**：
- `losses_tolerant.py`：Frame-Tolerant Feature Loss
- `train_exp65_anti_collapse.py`：整合 `--use_frame_tolerant` 參數

---

## 關鍵問題診斷

### 問題一：Token Accuracy 計算邏輯與疑慮

#### 計算流程

```
┌─────────────────────────────────────────────────────────────┐
│                    Token Accuracy 計算                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Student Path:                                                │
│  noisy_audio → Student Encoder (LoRA) → features → VQ → student_codes
│                        ↓                                      │
│                  (可訓練)                                     │
│                                                               │
│  Teacher Path:                                                │
│  clean_audio → Teacher Encoder (凍結) → features → VQ → teacher_codes
│                        ↓                                      │
│                    (凍結)                                     │
│                                                               │
│  Accuracy = (student_codes == teacher_codes) / total          │
│                    ↑                                          │
│              嚴格相等比較                                     │
└─────────────────────────────────────────────────────────────┘
```

#### 關鍵發現

**觀察現象**：
- Train 音檔聽起來與 Teacher 相似
- Feature Loss 持續下降
- 但 Token Accuracy 只有 ~0.9%

**原因分析**：

Token Accuracy 計算的是 **VQ codes 的嚴格相等**，而非 features 的相似度：

```python
# exp_1219/losses.py:202-216
def compute_masked_accuracy(predictions, targets, lengths, encoder_stride=320):
    correct = (predictions == targets).float()  # ← 嚴格相等！
    masked_correct = correct * mask
    accuracy = num_correct / (num_total + 1e-8)
    return accuracy
```

**這意味著**：
1. 即使 Student features 與 Teacher features 的 MSE 很低（方向和大小都接近）
2. 經過 VQ 量化時，可能選到**不同的 codebook entry**
3. 因為 VQ 是離散選擇（argmin distance），微小的差異可能導致完全不同的 code

#### 驗證證據

**診斷工具**：`quick_token_acc_check.py`

```
=== 診斷結果 ===

Sample 1:
  Valid tokens: 225
  Correct: 2
  Accuracy: 0.89%

  First 20 tokens:
    Student: [95, 1141, 388, 110, 1326, 774, ...]
    Teacher: [789, 747, 152, 479, 128, 966, ...]
    Match:   ['✗', '✗', '✗', '✗', '✗', '✗', ...]

  Student top-5 codes: [(1760, 34), (1623, 22), (1751, 8), ...]
  Teacher top-5 codes: [(1145, 6), (1474, 4), (1284, 4), ...]

  Student unique codes: 120
  Teacher unique codes: 179
```

**關鍵發現：Mode Collapse**
- Student codes 集中在少數幾個值：`1760, 1834, 1623`（每個出現 20-34 次）
- Teacher codes 分散多樣（每個只出現 3-6 次）
- Student unique codes < Teacher unique codes

---

### 問題二：Noisy-Clean 時間偏移

#### 診斷工具：`diagnose_alignment.py`

```
=== 診斷結果 ===

[1] Best Lag Statistics (samples @ 24kHz)
    Mean: -136.0 (-5.7 ms)
    Std:  941.0 (39.2 ms)
    Min:  -4640 (-193.3 ms)
    Max:  1760 (73.3 ms)

[2] Analysis
    Encoder stride: 13.3 ms/frame
    Mean lag: 5.7 ms

    Samples with lag > half frame: 22/30 (73.3%)
```

**結論**：73.3% 的樣本有超過半個 frame 的時間偏移

#### Frame-Tolerant 測試

**診斷工具**：`test_tolerant_acc.py`

```
=== Tolerant vs Strict Accuracy ===

Average Strict Accuracy:   0.61%
Average Tolerant Accuracy: 1.31%
Average Improvement:       +0.70%
```

**結論**：
- 即使允許 ±1 frame 容忍度，Accuracy 仍然極低（1.31%）
- **時間偏移不是主要問題**，主要問題是 Mode Collapse

---

## 實驗列表

### Exp62: Capacity Expansion

**目的**：測試增加模型容量是否能提升表現

**配置**：
- LoRA rank: 256 → 384
- Alpha: 768
- Epochs: 200 → 500
- Early stopping patience: 100

**假設**：如果模型容量不足，增加 rank 應該能提升 val accuracy

**驗證方式**：比較 Val Masked Acc 是否有顯著提升

**實驗狀態**：🟡 運行中（Epoch 35/500）

**目前結果**：
| 指標 | 數值 |
|------|------|
| Best Val Acc | 0.71% @ Epoch 28 |
| Train Acc | ~1.55% |
| Feature Distance | ~3.65 |

**初步結論**：❌ 增加容量未能顯著提升（0.71% < baseline 0.91%）

---

### Exp63: VQ-Aware Loss

**目的**：讓 Student features 更適合 VQ 量化

**新增 Loss**：
1. **VQ Commitment Loss** (λ=0.1)：
   - 公式：`||student_feature - codebook[teacher_code]||^2`
   - 目的：讓 student feature 更接近正確的 codebook centroid

2. **VQ Distortion Loss** (λ=0.1)：
   - 使用 soft assignment 計算加權距離
   - 目的：最小化 VQ 量化誤差

**假設**：如果問題是 VQ bottleneck，這些 loss 應該改善 decode 音質

**實驗狀態**：🟡 運行中（Epoch 39/300）

**目前結果**：
| 指標 | 數值 |
|------|------|
| Best Val Acc | **0.82%** @ Epoch 32 |
| Train Acc | ~2.0% |
| VQ Commit Loss | ~0.27 |
| VQ Distort Loss | ~0.064 |

**初步結論**：📈 有改善（0.82% vs 0.71% of Exp62），但仍低於 baseline

---

### Exp64: Curriculum Learning

**目的**：從簡單樣本開始訓練，逐步增加難度

**配置**：
- 難度定義：SNR (高 SNR = 簡單)
- 初始使用 30% 最簡單樣本
- 每 30 epochs 增加 10%

**假設**：先學會簡單的 denoising 可能幫助泛化

**實驗狀態**：⚪ 待執行

---

### Exp65: Anti-Collapse + Frame-Tolerant (重點實驗)

**目的**：解決 Student encoder mode collapse 問題 + 處理時間偏移

**問題證據**（來自 `quick_token_acc_check.py`）：

| 指標 | Student | Teacher |
|------|---------|---------|
| Top code 出現次數 | 24-34 次 | 3-6 次 |
| 最常見 codes | 1760, 1834, 1623 | 分散多樣 |
| Unique codes | 113-128 | 169-200 |

**新增 Loss**：

1. **Code Entropy Loss** (λ=0.1)：
   ```python
   # 計算 batch 內 code distribution 的 entropy
   # Loss = 1 - (entropy / max_entropy)
   # 高 entropy = 更多樣 = 好
   ```

2. **Feature Diversity Loss** (λ=0.1)：
   ```python
   # 計算 batch 內不同樣本的 feature cosine similarity
   # Loss = ReLU(similarity - margin)
   # 懲罰太相似的 features
   ```

3. **Batch Contrastive Loss** (λ=0.1)：
   ```python
   # InfoNCE: student 應該與對應的 teacher 最相似
   # 與其他 teacher features 不相似
   ```

4. **Frame-Tolerant Loss** (整合)：
   ```python
   # 對於每個 student frame，比較 teacher 的 t-1, t, t+1
   # 選擇最小的 loss，處理時間偏移問題
   # 預設：use_frame_tolerant=True, frame_tolerance=1
   ```

**驗證方式**：
1. 觀察 code entropy 是否增加
2. 觀察 unique codes 數量是否接近 teacher
3. 觀察 Token Accuracy 是否提升
4. 觀察 `offset_zero_ratio` 了解偏移分布

**實驗狀態**：⚪ 待執行

---

### Exp66: Post-VQ Feature Loss (新增)

**目的**：直接優化 VQ 量化後的特徵，改善解碼品質

**背景**（來自 `verify_codes_v2.py` 診斷）：

| 階段 | Cosine Sim | 說明 |
|------|------------|------|
| Pre-VQ | 0.495 | Encoder 輸出（目前優化目標） |
| Post-VQ | 0.9325 | 量化後（解碼使用） |

**核心概念**：
- 目前訓練只優化 Pre-VQ encoder output
- 但解碼使用的是 Post-VQ quantized features
- 直接優化 Post-VQ 應該更有效改善音質

**新增 Loss**：
1. **Post-VQ Feature Loss** (λ=0.5)：
   ```python
   # MSE between student Post-VQ and teacher Post-VQ
   # 使用 Straight-Through Estimator 讓梯度穿過 VQ
   loss = MSE(VQ(student_encoder_out), VQ(teacher_encoder_out))
   ```

2. **Post-VQ Cosine Loss** (λ=0.5)：
   ```python
   # 最大化 Post-VQ features 的 cosine similarity
   loss = 1 - cosine_similarity(VQ(student), VQ(teacher))
   ```

**實作檔案**：
- `losses_post_vq.py`：Post-VQ Loss 函數
- `train_exp66_post_vq.py`：訓練腳本
- `run_exp66_post_vq.sh`：執行腳本

**驗證方式**：
1. 觀察 Post-VQ Cosine Sim 是否提升
2. 聽 validation 音檔品質是否改善
3. 比較 Token Accuracy 變化

**實驗狀態**：⚪ 待執行

---

## 輔助工具

### 診斷工具

| 工具 | 用途 |
|------|------|
| `quick_token_acc_check.py` | 詳細檢查 token 預測情況，診斷 mode collapse |
| `diagnose_alignment.py` | 檢查 noisy-clean 時間偏移 |
| `diagnose_vq_quality.py` | 完整的 VQ 品質診斷 |
| `test_tolerant_acc.py` | 測試 frame-tolerant accuracy |
| `verify_codes_calculation.py` | 驗證 codes 計算一致性 (V1) |
| `verify_codes_v2.py` | 驗證 codes 計算一致性 (V2, 更完整) |

### Loss 實作

| 檔案 | 內容 |
|------|------|
| `losses.py` | VQ-Aware Loss (V3) |
| `losses_diversity.py` | Anti-Collapse + Frame-Tolerant Loss (V4) |
| `losses_tolerant.py` | Frame-Tolerant Loss (獨立模組) |
| `losses_post_vq.py` | Post-VQ Feature Loss (V5) |
| `data_curriculum.py` | Curriculum Learning Dataset |

---

## 執行方式

```bash
# Exp62: Capacity Expansion
bash exp_1226/run_exp62_capacity.sh

# Exp63: VQ-Aware Loss
bash exp_1226/run_exp63_vq_aware.sh

# Exp64: Curriculum Learning
bash exp_1226/run_exp64_curriculum.sh

# Exp65: Anti-Collapse
bash exp_1226/run_exp65_anti_collapse.sh

# Exp66: Post-VQ Feature Loss (推薦)
bash exp_1226/run_exp66_post_vq.sh
```

---

## 實驗結果總結

| 實驗 | 方法 | Best Val Acc | 狀態 | 結論 |
|------|------|-------------|------|------|
| Baseline (Exp55) | Feature + Triplet | **0.91%** | ✅ 完成 | 最佳 baseline |
| Exp62 | Capacity (rank=384) | 0.71% | 🟡 運行中 | ❌ 無改善 |
| Exp63 | VQ-Aware Loss | 0.82% | 🟡 運行中 | 📈 有改善但不足 |
| Exp64 | Curriculum Learning | - | ⚪ 待執行 | - |
| Exp65 | Anti-Collapse | - | ⚪ 待執行 | - |
| Exp66 | Post-VQ Loss | - | ⚪ 待執行 | - |

---

## 結論與下一步

### 核心問題

1. **Token Accuracy 低的根本原因**：不是 Feature Loss 沒下降，而是：
   - VQ 是離散選擇，微小 feature 差異 → 不同 code
   - Student encoder mode collapse，輸出多樣性不足

2. **時間偏移影響有限**：Frame-Tolerant 測試顯示只有 +0.7% 改善

3. **計算邏輯確認無誤**：`verify_codes_v2.py` 驗證兩種方法 100% 一致

### 建議優先順序

1. **Exp65 Anti-Collapse**：最直接解決 mode collapse
2. **Exp66 Post-VQ**：直接優化解碼使用的特徵
3. **Exp63 VQ-Aware**：改善 VQ 量化品質（已在運行）
4. **Exp64 Curriculum**：探索性實驗

---

## 附錄一：用戶問題解答總結

### Q1: Token Accuracy 計算邏輯是否有誤？

**A: ✅ 計算正確，無誤。**

驗證工具 `verify_codes_v2.py` 確認：
- `model.forward()` 與 `feature_extractor()` 計算結果 **100% 一致**
- Token Accuracy ~5% 是真實的
- 音檔聽起來相似是因為 **Post-VQ Cosine Sim = 0.93**（VQ codebook 魯棒性）

### Q2: 是否要試試看前後 Frame 選最低分數？

**A: ⚠️ 有實作，但效果有限。**

- 已實作 `losses_tolerant.py`：Frame-Tolerant Loss
- 測試結果：+0.70% 改善（0.61% → 1.31%）
- **根本原因不是時間偏移**，而是 Mode Collapse
- 仍保留此功能在 Exp65 中使用

---

## 附錄二：Token Accuracy 計算疑慮總結

### 為什麼 Train 音檔聽起來相似但 Accuracy 低？

```
┌────────────────────────────────────────────────────────────────┐
│                        完整解釋                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│ 1. 訓練優化的是 Feature Loss (MSE)：                            │
│    L = ||student_feature - teacher_feature||^2                  │
│    → 這個 loss 確實在下降                                       │
│    → features 越來越相似                                        │
│                                                                  │
│ 2. 但 Token Accuracy 計算的是 VQ codes 相等：                   │
│    student_feature → argmin(distance to codebook) → code        │
│    teacher_feature → argmin(distance to codebook) → code        │
│    accuracy = (student_code == teacher_code)                    │
│                                                                  │
│ 3. 問題：VQ 是離散選擇                                          │
│    即使兩個 features 很接近，如果它們在 codebook centroid 的    │
│    邊界上，可能選到不同的 code                                  │
│                                                                  │
│    Example:                                                      │
│    codebook = [c1, c2, c3, ...]                                 │
│    student_feature 距離 c1=0.49, c2=0.51 → 選 c1                │
│    teacher_feature 距離 c1=0.51, c2=0.49 → 選 c2                │
│    → 即使 features 相似，codes 完全不同！                       │
│                                                                  │
│ 4. 更嚴重：Mode Collapse                                        │
│    Student encoder 不論輸入什麼，都輸出相似的 features          │
│    → 所有 features 都映射到少數幾個 codes                       │
│    → 這就是為什麼 student codes 集中在 1760, 1834, 1623         │
│                                                                  │
│ 5. 為什麼音檔聽起來相似？                                       │
│    decode(student_codes) 的結果可能仍然可辨識，因為：           │
│    - VQ decoder 有一定的魯棒性                                  │
│    - 人耳對某些失真不敏感                                       │
│    - 但精確度和音質確實受影響（"破破的"）                       │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

### 可能的解決方向

1. **VQ-Aware Training**：直接優化 code 預測 (Exp63)
2. **Anti-Collapse**：增加 output diversity (Exp65)
3. **Post-VQ Loss**：直接優化量化後特徵 (Exp66)
4. **More Data**：增加訓練數據多樣性

---

## 附錄三：診斷方法驗證 (2024-12-26 更新)

### 驗證工具

**`verify_codes_v2.py`** - 確認 `model.forward()` 和 `feature_extractor()` 的計算一致性

### 驗證結果

```
=== Student Codes 比較 ===
forward() codes shape: torch.Size([1, 1, 194])
feature_extractor() codes shape: torch.Size([1, 1, 194])

[Student] 兩種方法的 codes 匹配率: 100.00%
[Teacher] 兩種方法的 codes 匹配率: 100.00%

=== Token Accuracy ===
forward() 方式計算的 Token Accuracy:          5.15%
feature_extractor() 方式計算的 Token Accuracy: 5.15%

=== Cosine Similarity 比較 ===
Pre-VQ (encoder output) Cosine Similarity:  0.4950
Post-VQ (quantized) Cosine Similarity:      0.9325
```

### 關鍵發現

| 指標 | 數值 | 說明 |
|------|------|------|
| **Pre-VQ Cosine Sim** | 0.495 | Encoder 輸出的相似度（訓練優化目標） |
| **Post-VQ Cosine Sim** | 0.9325 | VQ 量化後的相似度 |
| **Token Accuracy** | 5.15% | VQ codes 的嚴格匹配 |

### 為什麼音頻聽起來相似？

```
┌────────────────────────────────────────────────────────────────┐
│                    Pre-VQ vs Post-VQ 解釋                       │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  訓練優化的是 Pre-VQ features (encoder output)                  │
│  → Pre-VQ Cosine Sim = 0.495 (不高)                             │
│                                                                  │
│  但解碼使用的是 Post-VQ features (quantized)                    │
│  → Post-VQ Cosine Sim = 0.9325 (很高！)                         │
│                                                                  │
│  為什麼 Post-VQ 更高？                                          │
│  VQ codebook 設計良好，相鄰的 codes 解碼後音質相似              │
│  即使 Student 和 Teacher codes 不同，                           │
│  只要它們映射到"附近"的 codebook entries，                      │
│  解碼後的音頻仍然相似                                           │
│                                                                  │
│  結論：                                                          │
│  Token Accuracy 低 (~5%) 是真實的                               │
│  音頻聽起來還可以是因為 VQ 的魯棒性                             │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

### 診斷方法確認

✅ **`model.forward()` 計算的 codes 與 `feature_extractor()` 100% 一致**

之前誤解的原因：
- 錯誤地比較了 Pre-VQ encoder output 的 cosine similarity
- 以為 cosine sim 低代表 features 不相似
- 實際上 Post-VQ features 才是解碼使用的，其 cosine sim = 0.93

正確的理解：
- **Token Accuracy ~5%** 是正確的（Student 和 Teacher 選到不同的 VQ codes）
- **音頻聽起來相似** 是因為 VQ codebook 設計使得相鄰 codes 解碼相似
- **核心問題仍然是 Mode Collapse** - Student codes 集中在少數幾個
