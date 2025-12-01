# exp_1128 訓練行為分析報告

## 實驗概述

**目標**：LoRA Encoder Denoising - 讓 Student (noisy → encoder+LoRA) 學習 Teacher (clean → encoder) 的輸出

**實驗矩陣**：
| 實驗 | LoRA Rank | Distance Loss Weight | 狀態 | 備註 |
|------|-----------|---------------------|------|------|
| exp1 | 32 | 0.05 | ✅ 完成 | |
| exp2 | 32 | 0.10 | ⏹️ 提前終止 | 與 exp1 行為相同 |
| exp3 | 64 | 0.05 | ✅ 完成 | |
| exp4 | 64 | 0.10 | ⏹️ 提前終止 | 與 exp3 行為相同 |

**提前終止原因**：由於 distance_loss 不可微（見下方分析），不同的 distance_loss_weight 不會產生不同的訓練效果。exp2 和 exp4 的訓練曲線與 exp1/exp3 幾乎相同，因此提前終止節省計算資源。

---

## 觀察到的現象 (以 exp3 為例)

| Metric | 初始值 | 最終值 (Epoch 40) | 趨勢 |
|--------|--------|-------------------|------|
| Train Token Acc | 34.4% | ~15% | ⬇️ **下降** |
| Val Token Acc | 17.3% | ~7% | ⬇️ **下降** |
| Feature Loss | 0.042 | 0.028 | ✅ 改善 33% |
| Distance Loss | 3.15 | ~3.0 | ↔️ 幾乎不變 |
| VQ Loss | 0.0006 | 0.0009 | ↔️ 幾乎不變 |

**核心問題**：Feature Loss 改善，但 Token Accuracy 反而下降！

---

## 問題分析

### 問題 1：為什麼 Token Accuracy 持續下降？

#### 根本原因：VQ 的非可微性 (Non-differentiability)

VQ 使用 **Straight-Through Estimator (STE)**：

```python
# core_vq.py:302
if self.training:
    quantize = x + (quantize - x).detach()  # STE: 梯度穿透，但 argmax 不變
```

#### 問題所在

```
Student Pipeline:
noisy_audio → Encoder(+LoRA) → features_student → VQ(argmax) → codes_student
                  ↑ 可訓練                           ↑ 不可微（argmax）
```

1. **Loss 優化目標**：`feature_loss = MSE(student_features, teacher_features)`
2. **LoRA 學習的是**：讓 features 在 MSE 空間接近
3. **但**：MSE 空間的「接近」≠ VQ 空間的「相同 code」

#### 具體例子

假設 codebook 有 3 個 codes 在 2D 空間：
```
Code 0: [0.0, 1.0]
Code 1: [1.0, 0.0]
Code 2: [0.5, 0.5]
```

| 階段 | Teacher | Student | Code Match |
|------|---------|---------|------------|
| 初始 | [0.48, 0.52] → Code 2 | [0.45, 0.55] → Code 2 | ✅ |
| 優化後 | [0.48, 0.52] → Code 2 | [0.53, 0.47] → Code 1 | ❌ |

Student features 更接近 Teacher (MSE 降低)，但跨越了 decision boundary，導致 code 不同！

**結論**：Feature Loss 降低 ≠ Code Match Rate 提高

---

### 問題 2：為什麼 VQ Loss 幾乎不變？

#### VQ Loss 的定義

```python
# core_vq.py:310
commit_loss = F.mse_loss(quantize.detach(), x)  # ||z_q - z_e||²
```

VQ Loss = **Commitment Loss** = features 到最近 code 的距離

#### 為什麼不變？

1. **Codebook 凍結** — 所有實驗 `vq_loss_weight=0.0`，codebook 不更新
2. **EMA 更新凍結** — codebook 的 EMA 更新需要 training mode + 非零 weight
3. **分布變化小** — LoRA 只改變少量參數，features 整體分布變化不大

#### 這是預期行為嗎？

✅ **是的**，因為：
- Codebook 凍結 → code entries 不會移動
- `vq_loss_weight = 0.0` → 這個 loss 只是監控，不參與優化

---

### 問題 3：為什麼 Distance Loss 幾乎不改善？

#### 關鍵發現：Distance Loss 不可微！

```python
# losses.py:44-50
student_codes = model_output['student_codes']  # argmax 的結果！
student_flat = student_codes[:, 0, :].reshape(-1).long()  # 轉成 index
distances = distance_matrix[student_flat, teacher_flat]   # 純 indexing
distance_loss = distances.mean()
```

**梯度流分析**：
- `student_codes` 來自 `argmax`（不可微）
- `distance_matrix[i, j]` 是純 indexing（不可微）
- **梯度無法從 `distance_loss` 傳回到 LoRA 參數！**

#### 實際有梯度的 Loss

只有 `feature_loss = F.mse_loss(student_features, teacher_features)` 有梯度！

```
實際訓練的 Loss:
total_loss = 1.0 * feature_loss + 0.05 * distance_loss(無梯度) + 0.0 * vq_loss
           = feature_loss (只有這個有梯度)
```

---

## 驗證方法

### PDB 梯度檢查

創建 `pdb_gradient_check.txt`：

```bash
# 設置斷點在 backward 之後
b exp_1128/train.py:BACKWARD_LINE

run --exp_name debug --batch_size 2 --num_epochs 1

c
# 檢查各 loss 的 requires_grad
p feature_loss.requires_grad  # 應該是 True
p distance_loss.requires_grad  # 應該是 False 或無 grad_fn
p distance_loss.grad_fn  # 應該是 None

q
```

---

## 解決方案

### 方案 1：Soft Distance Loss（推薦）

用 softmax over distances 取代 argmax：

```python
def soft_distance_loss(student_features, teacher_codes, codebook, distance_matrix, temperature=1.0):
    """
    可微的 distance loss

    Args:
        student_features: (B, 512, T) - 連續的 features
        teacher_codes: (B, 1, T) - Teacher 的離散 codes
        codebook: (4096, 512) - VQ codebook
        distance_matrix: (4096, 4096) - 預計算的距離矩陣
        temperature: softmax temperature
    """
    B, C, T = student_features.shape

    # Reshape features: (B, 512, T) -> (B*T, 512)
    features_flat = student_features.permute(0, 2, 1).reshape(-1, C)

    # 計算到所有 codes 的距離: (B*T, 4096)
    logits = -torch.cdist(features_flat.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)
    # 或用內積: logits = features_flat @ codebook.T

    # Soft assignment: (B*T, 4096)
    soft_codes = F.softmax(logits / temperature, dim=-1)

    # Teacher codes: (B*T,)
    teacher_flat = teacher_codes[:, 0, :].reshape(-1).long()

    # 計算 weighted distance
    # distance_matrix[teacher_flat] 取得 teacher code 到所有 codes 的距離: (B*T, 4096)
    teacher_distances = distance_matrix[teacher_flat]  # (B*T, 4096)

    # Weighted sum: 期望距離
    expected_distance = (soft_codes * teacher_distances).sum(dim=-1)  # (B*T,)

    return expected_distance.mean()
```

**優點**：
- 完全可微
- 保留 distance-based 的概念
- 可以用 temperature 控制 soft vs hard

### 方案 2：Gumbel-Softmax VQ

讓 VQ 的 argmax 變成可微：

```python
def gumbel_softmax_vq(features, codebook, temperature=1.0):
    """
    用 Gumbel-Softmax 取代 argmax
    """
    logits = -torch.cdist(features, codebook)
    soft_codes = F.gumbel_softmax(logits, tau=temperature, hard=True)
    # hard=True: forward 是 one-hot，backward 是 soft
    quantized = soft_codes @ codebook
    return quantized, soft_codes
```

### 方案 3：重新定義目標

**考慮**：也許 feature-level 對齊才是正確的目標？

| 目標 | 優點 | 缺點 |
|------|------|------|
| Feature 對齊 | 可微、連續優化 | 不保證 code 相同 |
| Code 對齊 | 離散 token 一致 | 不可微、難優化 |

**建議**：如果最終目標是音質，feature 對齊可能就夠了。Token accuracy 只是一個中間指標。

---

## 建議的下一步

### 短期 (驗證當前結論)

1. **用 PDB 驗證梯度流** — 確認 distance_loss 確實沒有梯度
2. **聽音頻樣本** — Feature loss 改善是否帶來音質改善？

### 中期 (實現改進)

1. **實現 Soft Distance Loss** — 上述方案 1
2. **對比實驗**：
   - exp_1128 (原版，distance_loss 無梯度)
   - exp_1129 (新版，soft_distance_loss 有梯度)

### 長期 (架構改進)

1. 考慮 Gumbel-Softmax VQ
2. 考慮是否需要 code-level 對齊，還是 feature-level 就夠

---

## 附錄：關鍵代碼位置

| 檔案 | 行數 | 內容 |
|------|------|------|
| `model.py` | 185-187 | Student forward (VQ) |
| `losses.py` | 44-50 | Distance loss 計算 |
| `core_vq.py` | 302 | STE 實現 |
| `core_vq.py` | 309-311 | Commitment loss |

---

## 結論

1. **Token Accuracy 下降是預期行為** — MSE 優化不保證 code 對齊
2. **VQ Loss 不變是正確的** — Codebook 凍結 + weight=0
3. **Distance Loss 無效** — 因為完全不可微，只是監控指標
4. **實際有效的 Loss 只有 Feature MSE**

**關鍵洞見**：目前的訓練本質上只在優化 feature-level MSE，distance_loss 雖然有 weight，但沒有梯度，不影響訓練。

---

## 補充說明

### Q1: 為什麼 Student features 更接近 Teacher 卻跨越 decision boundary？

這是因為 VQ 的 decision boundary 是由 **codebook 中所有 codes 的相對位置** 決定的（Voronoi diagram），而非由單一 Teacher feature 決定。

```
例子：2D 空間中的 Voronoi cells

        ┌─────────────────────────┐
        │                         │
        │    Code 0 ●             │  ← Code 0 的 Voronoi cell
        │      │                  │
        │      │                  │
        ├──────┼──────────────────┤  ← Decision boundary
        │      │                  │
        │  Teacher ★              │
        │      ↑                  │
        │  Student 初始位置       │  ← 在同一個 cell (Code 2)
        │      ↓                  │
        │  Student 優化後 ●       │  ← MSE 降低，但跨越到 Code 1 的 cell！
        │      │                  │
        │      │    Code 1 ●      │
        │                         │
        └─────────────────────────┘
```

**重點**：MSE 優化方向可能與 Voronoi cell 邊界垂直，導致 features 跨越邊界。

### Q2: 為什麼 Val VQ Loss 接近零？

Validation 時 `model.eval()` 會：
1. 關閉 Dropout
2. **關閉 VQ 的 training mode** → `core_vq.py:301-302` 的 STE 和 commitment loss 計算被跳過

```python
# core_vq.py:301-311
if self.training:  # ← eval() 時為 False
    quantize = x + (quantize - x).detach()  # STE
    ...
    if self.commitment_weight > 0:
        commit_loss = F.mse_loss(quantize.detach(), x)  # commitment loss
        loss = loss + commit_loss * self.commitment_weight
```

因此 validation 時返回的 `vq_loss` 是初始值 `torch.tensor([0.0])`，接近零是預期行為。

### Q3: 音質較差的可能原因

1. **Token mismatch** — Token accuracy 下降意味著解碼時用了「錯誤」的 codes
2. **Feature distribution shift** — LoRA 改變了 feature 分布，但 decoder 是為原始分布訓練的
3. **Overfitting to MSE** — MSE 優化可能引入 artifacts（如過度平滑）

**建議**：
- 比較 `student_pred.wav` vs `baseline_noisy.wav` 聽音質差異
- 如果 student_pred 比 baseline 差，說明 LoRA 反而損害了音質
