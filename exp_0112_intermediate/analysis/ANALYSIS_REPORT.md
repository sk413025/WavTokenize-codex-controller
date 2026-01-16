# 中間層監督分析報告

**日期**: 2026-01-15
**分析目標**: 找出最佳的中間層監督策略

---

## 一、噪音敏感度分析（原始模型）

### 1.1 測量方法

**重要**: 此分析測量的是**原始模型（無 LoRA）**對噪音的敏感度：
```
噪音敏感度 = 1 - cos_sim(feature(clean_audio), feature(noisy_audio))
```

這代表「同一層對 Clean 和 Noisy 輸入的特徵差異」。

### 1.2 各層噪音敏感度

| Layer | Cos Sim | Sensitivity | 層組 | 備註 |
|-------|---------|-------------|------|------|
| L0 | 0.041 | **0.959** | input | ★ 最敏感 |
| L1 | 0.057 | **0.943** | low_level | ★ 極敏感 |
| L2 | 0.172 | 0.828 | low_level | |
| L3 | 0.105 | **0.895** | low_level (當前監督) | ★ 很敏感 |
| L4 | 0.121 | 0.879 | low_level | |
| L5 | 0.351 | 0.649 | mid_level | |
| **L6** | 0.081 | **0.919** | mid_level (當前監督) | ★ 中層最敏感 |
| L7 | 0.165 | 0.835 | mid_level | |
| L8 | 0.303 | 0.697 | mid_level | |
| L9 | 0.479 | 0.521 | semantic | |
| **L10** | **0.946** | **0.054** | semantic | ★ 最穩定（特異點）|
| L11 | 0.566 | 0.434 | semantic | |
| L12 | 0.593 | 0.407 | semantic | |
| L13 | 0.589 | 0.411 | abstract | |
| L14 | 0.556 | 0.444 | abstract | |
| L15 | 0.379 | 0.621 | abstract | |

### 1.3 層組摘要

```
Layer Group      Avg CosSim  Avg Sensitivity  特性
──────────────────────────────────────────────────────
input (L0)       0.041       0.959            ★ 最敏感
low_level (L1-4) 0.114       0.886            ★ 非常敏感
mid_level (L5-8) 0.225       0.775            中等敏感
semantic (L9-12) 0.646       0.354            較穩定
abstract (L13+)  0.508       0.492            中等
```

### 1.4 與 exp_1231_feature 的關係

**釐清**: 兩個實驗測量的是不同的東西：

| 實驗 | 測量內容 | 結論 |
|------|----------|------|
| exp_1231_feature | Clean vs Noisy 特徵差異 | mid_level (L5-L8) 變化最大 |
| 本分析 | 同上，但更精細的層級 | L0, L1, L6 最敏感 |

**實際上兩者一致**：
- L6 是 mid_level 中最敏感的層 (cos_sim=0.081)
- 本分析發現淺層 (L0-L1) 也很敏感，之前可能被歸類為「input」而忽略

---

## 二、訓練後 Student-Teacher 距離

### 2.1 測量方法

此分析測量的是**訓練後的模型**：
```
距離 = 1 - cos_sim(student_feature(noisy), teacher_feature(clean))
```

這代表「LoRA 訓練後，Student 學會了多少」。

### 2.2 各層距離

| Layer | Cos Sim | Cos Loss | 層組 |
|-------|---------|----------|------|
| L0 | 0.040 | **0.960** | input |
| L1 | 0.037 | **0.963** | low_level |
| L2 | 0.164 | 0.836 | low_level |
| L3 | 0.242 | 0.758 | low_level (★ 當前監督) |
| L4 | 0.126 | **0.874** | low_level |
| L5 | 0.323 | 0.677 | mid_level |
| L6 | 0.233 | 0.767 | mid_level (★ 當前監督) |
| L7 | 0.165 | 0.836 | mid_level |
| L8 | 0.242 | 0.759 | mid_level |
| L9 | 0.284 | 0.716 | semantic |
| **L10** | **0.746** | **0.254** | semantic (最相似) |
| L11 | 0.306 | 0.694 | semantic |
| L12 | 0.130 | 0.870 | semantic |
| L13 | 0.129 | 0.871 | abstract |
| L14 | 0.109 | 0.891 | abstract |
| L15 | 0.246 | 0.754 | abstract |

### 2.3 關鍵發現

1. **L10 在訓練前後都是最穩定的層**
   - 噪音敏感度: cos_sim=0.946 (最穩定)
   - 訓練後距離: cos_sim=0.746 (最相似)
   - 這可能是語義聚合層，天然對噪音魯棒

2. **淺層 (L0-L4) 學習最困難**
   - 噪音敏感度高 (sensitivity ≈ 0.88-0.96)
   - 訓練後距離仍大 (cos_loss ≈ 0.76-0.96)

3. **當前監督位置 (L3, L6) 選擇合理**
   - L3: 淺層中的代表
   - L6: 中層中最敏感的層

---

## 三、訓練收斂性分析

### 3.1 Loss 趨勢 (265 epochs)

```
Intermediate Loss:
  Train: 772.1 → 670.2 (↓13.2%)
  Val:   766.6 → 708.9 (↓7.5%)
  Gap:   5.5 → 38.7 (擴大)

Feature Loss:
  Train: 83.4 → 45.3 (↓45.7%)
  Val:   86.3 → 55.9 (↓35.3%)
  Gap:   2.9 → 10.6 (擴大)
```

### 3.2 過擬合指標

| 指標 | Train | Val | Gap |
|------|-------|-----|-----|
| Total Loss | 371.6 | 385.9 | 14.3 |
| Match Acc | 3.44% | 0.89% | **2.55%** |
| Intermediate | 670.2 | 708.9 | 38.7 |
| Feature | 45.3 | 55.9 | 10.6 |

**觀察**: Train-Val Gap 存在但不嚴重，主要問題是 Val 性能整體偏低。

---

## 四、當前 Loss 設計分析

### 4.1 Loss 組成

```
Total Loss = Feature Loss + Triplet Loss + Intermediate Loss

其中:
- Feature Loss (L17): MSE(student, teacher) 權重=1.0
- Triplet Loss (L17): 讓 student 靠近正確 codebook 權重=1.0
- Intermediate L3: Cosine Loss 權重=0.5
- Intermediate L6: Cosine Loss 權重=0.5
```

### 4.2 問題分析

| 問題 | 說明 | 影響 |
|------|------|------|
| 錯過最敏感層 | L0-L2 沒有監督 | 最敏感的層學習不足 |
| L10 沒利用 | 最穩定層沒用於對比 | 錯過錨點機會 |
| 權重可能不平衡 | Intermediate 只佔 1/3 | 中間層監督太弱 |

---

## 五、建議的改進方案

### 5.1 方案 A: 增加淺層監督

```python
# 原本
intermediate_indices = [3, 6]  # L3, L6

# 建議: 加入最敏感的層
intermediate_indices = [1, 3, 6]  # L1, L3, L6
intermediate_weights = {
    1: 1.0,   # 最敏感，強監督
    3: 0.8,
    6: 0.6,
}
```

### 5.2 方案 B: 利用 L10 作為錨點

```python
# L10 是最穩定的層 (cos_sim=0.946)
# 用它來做 consistency regularization

l10_student = student_intermediates[10]
l10_teacher = teacher_intermediates[10]

# 這兩者應該非常接近，可以用更嚴格的 loss
l10_loss = F.mse_loss(l10_student, l10_teacher) * 2.0
```

### 5.3 方案 C: 漸進式權重

```python
# 根據噪音敏感度調整權重
sensitivity = {0: 0.96, 1: 0.94, 3: 0.90, 6: 0.92, 10: 0.05, ...}

def compute_adaptive_weight(layer_idx):
    return sensitivity[layer_idx]  # 敏感度越高，權重越大
```

---

## 六、圖表

### 6.1 各層距離分布
![Layer Distances](layer_distances.png)

### 6.2 噪音敏感度比較
![Noise Sensitivity](noise_sensitivity_comparison.png)

### 6.3 監督位置推薦
![Supervision Recommendation](supervision_recommendation.png)

### 6.4 訓練收斂性
![Training Convergence](training_convergence.png)

### 6.5 Loss 設計分析
![Loss Design](loss_design_analysis.png)

---

*報告更新時間: 2026-01-15*
