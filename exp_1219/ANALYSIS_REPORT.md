# Exp48 配置分析報告

基於 exp48 最佳配置進行的技術分析。

**更新日期**: 2025-12-19

---

## 🔑 關鍵發現

1. **Triplet Margin 0.2 可能太小** - 55% 的 code 對距離 < 0.5，建議增加到 0.5
2. **LoRA 佔 Encoder 21%** - 不是 11%，計算正確後是合理的
3. **方向相似度問題** - 實際驗證發現 cos_sim = 0.21，這才是真正問題！

---

## 1. Triplet Loss Margin (0.2) 分析

### Codebook 距離統計

| 指標 | 數值 |
|------|------|
| Codebook 大小 | 4096 codes × 512 dim |
| **所有 code 對之間的 L2 距離** | |
| Min | 0.0000 |
| Max | 31.3005 |
| Mean | 5.3154 |
| Median | 5.3198 |
| Std | 4.5576 |

### Distance Percentiles

| Percentile | Distance |
|------------|----------|
| 1% | 0.0000 |
| 5% | 0.0000 |
| 10% | 0.0000 |
| 25% | 0.0000 |
| 50% | 5.3198 |
| 75% | 8.4490 |
| 90% | 11.4262 |
| 95% | 13.2452 |
| 99% | 16.8124 |

### Nearest Neighbor 距離

| 指標 | 數值 |
|------|------|
| Min NN distance | 0.0000 |
| Max NN distance | 2.8367 |
| **Mean NN distance** | **0.5683** |
| Median NN distance | 0.0000 |

### Margin 分析

當前設置: `triplet_margin = 0.2`

| 比較 | 比值 |
|------|------|
| margin / mean_dist | 0.0376 (3.76%) |
| margin / median_dist | 0.0376 (3.76%) |
| margin / mean_NN_dist | 0.3519 (35.19%) |

### 重要發現: 55% code 有重複鄰居

| 類型 | 數量 | 比例 |
|------|------|------|
| NN 距離 = 0 (重複 code) | 2262 | 55.2% |
| NN 距離 > 0 (有效 code) | 1834 | 44.8% |

**有效 code 的 NN 距離統計**:
- Mean: **1.27**
- Median: 1.18
- Min: 0.69
- Max: 2.84

### 結論與建議 (修正版)

**問題**: 之前用"全部 code 的 NN mean = 0.57"來分析是錯誤的！

因為 55% 的 code 有完全相同的鄰居 (NN=0)：
- 這些重複 code 的 triplet loss 無效 (正確/錯誤沒區別)
- 應該基於"有效 code"的 NN 距離 (mean = 1.27) 來設置 margin

**Margin 分析**:
| Margin | 佔有效 NN mean (1.27) | 評估 |
|--------|----------------------|------|
| 0.2 | 16% | ⚠️ 太小，約束太弱 |
| 0.5 | 39% | ✓ 合適 |
| 1.0 | 79% | 可能太激進 |

**建議: margin=0.5**

原因：
1. 0.5 約為有效 NN mean 的 39%，是合理的區分度
2. 當前 triplet loss 仍高 (~0.76)，說明 margin=0.2 並未滿足
3. Token 過度集中問題需要更強的約束

---

## 2. LoRA Rank 128 參數比例

### WavTokenizer 原始參數

| 組件 | 參數量 | 佔比 |
|------|--------|------|
| **Total** | **80,552,420** | 100% |
| Encoder | 8,802,816 | 10.93% |
| Decoder | 8,802,274 | 10.93% |
| Quantizer | Frozen | - |

### LoRA 配置 (all_18 layers, rank=128)

| 指標 | 數值 |
|------|------|
| Target layers 原始參數 | 4,600,320 (5.71%) |
| **LoRA 新增參數** | **1,852,288** |
| 佔 WavTokenizer 全模型 | **2.25%** |
| 佔 Encoder | 11.08% |

### 結論

- LoRA rank=128 在 all_18 層上新增約 **185 萬參數**
- 這佔整個 WavTokenizer 的 **2.25%**
- 佔 Encoder 的 **11.08%**

這是一個相對較大的 LoRA 配置，提供了足夠的表達能力。如果需要減少參數，可以：
- 降低 rank 到 64 (參數量約減半)
- 使用 critical_8 層配置 (只訓練 8 層而非 18 層)

---

## 3. MSE Loss 數學分析與特徵尺度問題

### 你的問題

> 如果 Zstu 區間在 [-0.01, 0.02]、Ztea 區間在 [0.3, 0.8]，MSE 展開為：
> ||Zstu - Ztea||² = ||Zstu||² + ||Ztea||² - 2<Zstu, Ztea>
> 是否會因為 ||Ztea||² 沒有梯度、<Zstu, Ztea> 趨近於 0，導致特徵不夠接近？

### 數學分析

**你的公式是正確的**，但結論需要澄清：

```
MSE = ||Zstu||² + ||Ztea||² - 2<Zstu, Ztea>
∂MSE/∂Zstu = 2*Zstu - 2*Ztea = 2*(Zstu - Ztea)
```

### 關鍵點

1. **||Ztea||² 沒有梯度** ✓
   - 正確，Ztea 是 Teacher 輸出，被 detach
   - 但這不是問題，因為我們要優化 Zstu

2. **<Zstu, Ztea> 趨近於 0？** ✗
   - 內積取決於"方向"而非只是"大小"
   - 如果 Zstu 雖小但與 Ztea 方向相似，內積仍為正值
   - 實驗顯示：即使極端尺度差異，cosine similarity 仍有 0.48

3. **梯度 = 2*(Zstu - Ztea)**
   - 這個梯度**直接指向 Ztea**
   - 不會"只能抓平均中間距離"
   - 方向永遠正確，只是步長可能不合適

### 真正的潛在問題

| 問題 | 描述 | 影響 |
|------|------|------|
| **尺度不匹配** | Zstu 範圍遠小於 Ztea | 梯度過大，可能不穩定 |
| **優化效率** | 需要大幅改變輸出範圍 | 收斂緩慢 |
| **方向 vs 大小** | MSE 同時優化方向和大小 | 可能不夠靈活 |

### 實驗驗證

極端情況模擬 (Zstu ∈ [-0.01, 0.02], Ztea ∈ [0.3, 0.8]):

| 指標 | 無 Normalize | L2 Normalize |
|------|-------------|--------------|
| MSE | 162.68 | 1.03 |
| Cosine Sim | 0.48 | 0.48 |

**說明**: 主要問題是"尺度"而非"方向"

### 解決方案建議

#### 方案 1: Feature Normalization (推薦)

```python
# 在比較前 normalize
def feature_loss(z_stu, z_tea):
    z_stu_norm = F.layer_norm(z_stu, z_stu.shape[-1:])
    z_tea_norm = F.layer_norm(z_tea, z_tea.shape[-1:])
    return F.mse_loss(z_stu_norm, z_tea_norm)
```

#### 方案 2: Cosine Similarity Loss

```python
# 只關心方向，忽略大小
def cosine_loss(z_stu, z_tea):
    return 1 - F.cosine_similarity(z_stu, z_tea, dim=-1).mean()
```

#### 方案 3: L2 Normalize + MSE

```python
def normalized_mse(z_stu, z_tea):
    z_stu_n = F.normalize(z_stu, dim=-1)
    z_tea_n = F.normalize(z_tea, dim=-1)
    return F.mse_loss(z_stu_n, z_tea_n)
```

#### 方案 4: 組合 Loss

```python
def combined_loss(z_stu, z_tea, alpha=0.5):
    mse = F.mse_loss(z_stu, z_tea)
    cos = 1 - F.cosine_similarity(z_stu, z_tea, dim=-1).mean()
    return alpha * mse + (1-alpha) * cos
```

### 結論

**你的擔憂部分正確**，但問題不在於"只能抓平均中間距離"，而是：

1. 尺度差異導致 MSE 值過大，梯度可能過大
2. 優化效率低，需要很多步才能收斂

**建議**: 在現有 Feature Loss 中加入 normalization，或者增加 Cosine Similarity Loss 作為補充。

---

## 4. 實際特徵驗證結果 (新增)

使用 exp48 checkpoint 載入模型，實際測試特徵分布：

### 測試結果

| 指標 | Student | Teacher | 比較 |
|------|---------|---------|------|
| 特徵範圍 | [-3.85, 5.10] | [-0.99, 0.98] | Student 範圍更大 |
| Mean | -0.024 | -0.027 | 相近 |
| Std | 0.83 | 0.58 | Student 稍大 |
| L2 Norm (平均) | 18.72 | 13.07 | **ratio = 1.43** |
| **Cosine Similarity** | - | - | **0.21 ± 0.09** |

### MSE 分解

```
||Zstu||² = 350.87
||Ztea||² = 173.87
<Zstu, Ztea> = 51.41
MSE = 421.93 ✓ (展開驗證正確)
```

### 診斷結果

⚠️ **發現問題：方向相似度低 (cos_sim = 0.21)**

這意味著：
1. **尺度問題不嚴重** - norm_ratio = 1.43 還可以接受
2. **方向問題才是關鍵** - Student 和 Teacher 特徵的"方向"不夠對齊
3. 你擔心的 "Zstu 範圍小導致內積趨近 0" 的問題 **不存在**
   - 實際上 Student 範圍比 Teacher 還大
   - 問題是方向不對齊，而非尺度

### 建議解決方案

**優先嘗試加入 Cosine Similarity Loss**：

```python
# 在 MaskedCombinedLoss 中加入
def cosine_loss(self, student_features, teacher_features, lengths):
    # student_features: (B, D, T)
    # Reshape to (B*T, D)
    stu = student_features.permute(0, 2, 1).reshape(-1, D)
    tea = teacher_features.permute(0, 2, 1).reshape(-1, D)

    # Cosine similarity loss
    cos_sim = F.cosine_similarity(stu, tea, dim=1)

    # Apply mask
    mask = create_length_mask(lengths, ...)
    mask_flat = mask.reshape(-1)

    # Loss = 1 - cosine_similarity (越接近越好)
    loss = (1 - cos_sim) * mask_flat
    return loss.sum() / (mask_flat.sum() + 1e-8)
```

---

## 總結

| 問題 | 當前配置 | 評估 | 建議 |
|------|----------|------|------|
| Triplet Margin | 0.2 | ⚠️ 可增大 | **增到 0.5** (55% code NN < 0.5) |
| LoRA Rank | 128 | ✓ 足夠 | 21% Encoder，合理 |
| Feature Loss | MSE | ⚠️ 方向問題 | **加入 Cosine Loss** |

---

## 下一步實驗建議

### 高優先級

1. **Exp49**: 加入 Cosine Similarity Loss
   ```
   --feature_weight 1.0 --cosine_weight 0.5 --triplet_weight 1.0
   ```

2. **Exp50**: 增加 Triplet Margin
   ```
   --triplet_margin 0.5
   ```

### 中優先級

3. **Exp51**: 組合改進
   ```
   --triplet_margin 0.5 --cosine_weight 0.5
   ```

4. **Exp52**: Layer Normalized MSE
