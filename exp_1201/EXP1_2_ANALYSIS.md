# exp_1201 Gumbel/STE 實驗分析：為什麼 Token Accuracy 仍然很低？

**分析日期**: 2025-12-02
**實驗**: exp1 (Gumbel-Softmax) & exp2 (STE)
**結論**: Token Accuracy ~2% 的根本原因是 **Distance-based Loss 的間接優化本質**

---

## 1. 實驗結果總覽

### 最終指標 (50 epochs)

| Metric | Gumbel | STE | 變化 |
|--------|--------|-----|------|
| Train Loss | 0.4397 | 0.3527 | STE -20% |
| Val Loss | 0.4548 | 0.3765 | STE -17% |
| Train Feature Loss | 0.0319 | 0.0311 | STE -3% |
| Val Feature Loss | 0.0332 | 0.0328 | STE -1% |
| **Train Token Acc** | **2.19%** | **3.83%** | STE +75% |
| **Val Token Acc** | **1.75%** | **2.17%** | STE +24% |
| Train Distance Loss | 3.52 | 3.41 | STE -3% |
| Val Distance Loss | 3.65 | 3.64 | 相近 |

### 訓練曲線特徵

```
Token Accuracy 變化:
  Epoch 1:  ~24% (初始隨機)
  Epoch 2:  ~4-6% (快速下降)
  Epoch 5+: ~2-4% (穩定在低位)

Feature Loss 變化:
  Epoch 1:  0.042
  Epoch 50: 0.032 (持續下降 ✅)

Distance Loss 變化:
  Epoch 1:  3.33 (Gumbel) / 3.37 (STE)
  Epoch 50: 3.52 (Gumbel) / 3.41 (STE) (略微上升或持平)
```

---

## 2. Token Accuracy 低的根本原因

### 2.1 Distance-based Loss 的間接優化本質

**問題核心**：最小化期望距離 ≠ 最大化 Token Accuracy

```
Distance Loss 優化的是:
  E[dist(soft_codes, teacher_codes)]
  = Σᵢ softmax(-d/τ)ᵢ × distance_matrix[teacher, i]

Token Accuracy 需要的是:
  argmin_i dist(student_features, codebook[i]) == teacher_token
```

**為什麼會脫節？**

假設 student features 到 codebook 的距離分布：
```
Token 100 (正確): distance = 1.5
Token 101 (錯誤): distance = 1.6
Token 102 (錯誤): distance = 1.7
...
Token 500 (錯誤): distance = 5.0
```

Distance Loss 會：
- 懲罰所有 tokens 的加權距離
- 但 **梯度被平均分散** 到所有 tokens
- Token 100 和 101 的距離差 (0.1) 太小，不足以產生足夠的梯度差異

結果：模型可能讓所有距離都減少一點，但 **決策邊界** 沒有變得更清晰。

### 2.2 Softmax 的平滑效應

```python
# Temperature = 1.0 時的 softmax
distances = [1.5, 1.6, 1.7, 2.0, 5.0]
logits = [-1.5, -1.6, -1.7, -2.0, -5.0]
softmax(logits) = [0.32, 0.29, 0.26, 0.20, 0.01]
                     ↑      ↑
                  差異很小，梯度相似
```

問題：
- Softmax 把相近的距離映射到相近的機率
- 正確 token (0.32) 和最近錯誤 token (0.29) 的機率差只有 0.03
- 這個差異不足以產生強烈的學習信號

### 2.3 Token Accuracy 快速下降的現象

**Epoch 1 的高 Token Accuracy (~24%) 是假象**

原因分析：
1. 初始 LoRA 權重接近零，student ≈ teacher
2. 隨機初始化時，某些 tokens 剛好對齊
3. 一旦開始優化 Feature Loss，Token Accuracy 反而下降

這說明：**Feature-level 對齊 ≠ Token-level 對齊**

---

## 3. Feature Loss 和 Token Accuracy 的衝突

### 3.1 兩個目標的幾何解釋

```
Feature Space:
                    ○ Codebook[100] (Teacher Token)
                   /
                  /  ← Token Accuracy 需要: 最近鄰 = 100
                 /
    ○ --------- ● --------- ○
  Codebook[99]   Student    Codebook[101]
                Features
                    ↑
                    Feature Loss 需要: 接近 Teacher Features
```

**問題**：
- Teacher Features 不一定恰好在 Codebook[100] 上
- Feature Loss 可能把 Student Features 拉向 Teacher Features
- 但這個位置可能讓 Student 更接近錯誤的 Codebook entry

### 3.2 實驗證據

從數據可以看到：
```
Feature Loss: 0.042 → 0.032 (下降 24%)  ✅ 優化成功
Token Accuracy: 24% → 2% (下降 92%)     ❌ 反而惡化
```

這證明 Feature Loss 和 Token Accuracy **可能是負相關的**。

---

## 4. Distance Matrix 的使用問題

### 4.1 預計算 Distance Matrix 的假設

```python
distance_matrix[i, j] = ||codebook[i] - codebook[j]||
```

這假設：
- Codebook 是固定的 ✅ (我們凍結了 VQ)
- Token 之間的「語義距離」 ≈ 歐氏距離 ❓

### 4.2 歐氏距離 ≠ 語義距離

Codebook 的幾何結構可能很複雜：
```
Codebook 距離統計:
  最近鄰平均距離: 1.42
  中位距離: 5.32
  最大距離: 31.30
  標準差: 4.56
```

問題：
- 很多 tokens 的最近鄰距離只有 1.42
- 這些 tokens 可能在語義上完全不同
- Distance Loss 無法區分「接近正確 token」和「接近類似錯誤 token」

---

## 5. STE vs Gumbel：為什麼 STE 更好？

### 5.1 理論分析

| 特性 | Gumbel | STE |
|------|--------|-----|
| Forward | Hard codes + Gumbel noise | Hard codes (argmax) |
| Backward | Gumbel-softmax 梯度 | Softmax 梯度 |
| 隨機性 | 有 (每次不同) | 無 (確定性) |

### 5.2 Gumbel 的問題

```python
# Gumbel-Softmax 採樣
gumbel_noise = -log(-log(uniform(0, 1)))
logits_with_noise = logits + gumbel_noise
codes = softmax(logits_with_noise / τ)
```

問題：
- Gumbel noise 引入額外隨機性
- 可能導致「探索」錯誤的 tokens
- 在這個任務中，codebook 太大 (4096)，隨機探索效率低

### 5.3 STE 的優勢

```python
# STE
hard_codes = one_hot(argmax(-distances))
soft_codes = softmax(-distances / τ)
codes = hard_codes - soft_codes.detach() + soft_codes  # 梯度走 soft
```

優勢：
- Forward 是確定性的（選最近的 code）
- 訓練更穩定
- 在這個困難任務中，穩定性 > 探索性

---

## 6. 為什麼需要 CE / Margin Loss？

### 6.1 CE Loss 的優勢

```python
# CE Loss
logits = -distances / temperature
loss = CrossEntropy(logits, teacher_tokens)
```

CE Loss 直接優化：
- P(correct_token | student_features) 最大化
- 每個錯誤 token 都被明確懲罰
- 梯度信號更強、更直接

### 6.2 Margin Loss 的優勢

```python
# Margin Loss
correct_dist = distances[teacher_token]
wrong_dist = min(distances[other_tokens])
loss = max(0, correct_dist - wrong_dist + margin)
```

Margin Loss 專注於：
- 決策邊界的優化
- 確保正確 token 比最近的錯誤 token 更近
- 不需要關心距離的絕對值，只關心相對順序

---

## 7. 結論與建議

### 7.1 Token Accuracy 低的根本原因

1. **Distance-based Loss 是間接優化**：最小化期望距離 ≠ 最大化正確率
2. **Softmax 平滑效應**：相近距離產生相近機率，梯度差異不足
3. **Feature Loss 和 Token Accuracy 衝突**：優化 features 可能惡化 token 選擇
4. **Codebook 幾何複雜**：歐氏距離 ≠ 語義距離

### 7.2 解決方案

| 方法 | 原理 | 預期效果 |
|------|------|----------|
| **CE Loss** | 直接分類監督 | Token Acc 提升到 5-15% |
| **Margin Loss** | 優化決策邊界 | 更穩定的提升 |
| **更低的 Temperature** | 讓 softmax 更尖銳 | 可能幫助，但風險梯度爆炸 |
| **去掉 Feature Loss** | 只優化 token | 可能過擬合 |

### 7.3 下一步

1. **exp3 (CE)**: 用分類損失直接監督 token 選擇
2. **exp4 (Margin)**: 用 margin loss 優化決策邊界
3. 如果仍然不行，考慮：
   - 添加 Projection Head（讓模型有專門的 layer 適應 VQ 空間）
   - 使用 Codebook Embedding 作為 target（直接對齊到正確 code）

---

## 8. 附錄：詳細數據

### 8.1 Token Accuracy 變化趨勢 (Gumbel)

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 24.45% | 6.42% |
| 2 | 4.22% | 4.10% |
| 5 | 2.43% | 2.22% |
| 10 | 2.37% | 2.14% |
| 20 | 2.20% | 1.81% |
| 50 | 2.19% | 1.75% |

### 8.2 Distance Loss 變化趨勢

| Epoch | Gumbel Train | STE Train |
|-------|--------------|-----------|
| 1 | 3.33 | 3.37 |
| 10 | 3.59 | 3.39 |
| 20 | 3.55 | 3.40 |
| 50 | 3.52 | 3.41 |

注意：Distance Loss 沒有明顯下降，這說明 **即使梯度可微，模型也難以優化 token 選擇**。

---

**分析完成**: 2025-12-02
**結論**: Distance-based Loss 本質上是間接優化，需要 CE/Margin 等直接監督方法來提升 Token Accuracy。
