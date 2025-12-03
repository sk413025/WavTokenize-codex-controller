# VQ Selection Diagnosis Results - 2025-12-03

## 實驗對比

| 指標 | 未訓練模型 | 訓練後 (exp5 strong_feature_ce) | 變化 |
|------|-----------|--------------------------------|------|
| **Feature L2 Distance** | 5.18 | 4.39 | ↓ 15% ✅ |
| **Token Accuracy** | 4.99% | 0.66% | ↓ 87% 🔴 |
| **Code Distance** | 5.18 | 4.53 | ↓ 13% |
| **Avg Rank of Correct Code** | 1971 | 1480 | ↓ 25% |
| **Top-5 Accuracy** | 6.43% | 6.74% | ≈ 持平 |

## 關鍵發現

### 1. Feature Loss 有效，但 Token Accuracy 反而下降

- Feature L2 Distance 從 5.18 降到 4.39 (↓15%) → **Feature Loss 有效**
- 但 Token Accuracy 從 4.99% 降到 0.66% (↓87%) → **訓練反而破壞了 token 選擇**

### 2. Avg Rank 改善但仍然很差

- 正確 code 的平均排名從 1971 降到 1480
- 在 4096 個 codes 中，rank 1480 ≈ 中間位置
- **說明模型完全沒有學到如何選擇正確的 code**

### 3. Top-5 Accuracy 幾乎沒變

- 訓練前後都約 6-7%
- **說明訓練沒有讓正確 code 進入前幾名**

---

## VQ Loss 梯度分析

### VQ Loss 的計算方式 (WavTokenizer core_vq.py:309-311)

```python
if self.training:
    if self.commitment_weight > 0:
        commit_loss = F.mse_loss(quantize.detach(), x)
        loss = loss + commit_loss * self.commitment_weight
```

### 梯度流向

```
commit_loss = MSE(quantize.detach(), x)
                    ↑                ↑
              無梯度（凍結）      有梯度（student features）
```

### 優化方向

**VQ Loss 的目標**：讓 `x`（student features）接近 `quantize`（已選中的 codebook embedding）

**問題**：這不是我們想要的！
- 我們想要：讓 student 選擇的 code == teacher 選擇的 code
- VQ Loss 做的是：讓 features 更靠近「它自己選的 code」（不管對不對）

### 結論

**VQ Loss 有梯度回流，但優化方向是「錯的」！**

它只會讓 features 更「穩定」地選擇某個 code，但不保證這個 code 是正確的。

---

## 問題根源分析

### 為什麼訓練讓 Token Accuracy 下降？

1. **Feature Loss 和 VQ 選擇的目標不一致**
   - Feature Loss：讓 student features 接近 teacher features
   - VQ 選擇：argmin(distance to codebook)
   - 這兩個目標沒有直接關聯！

2. **LoRA 可能破壞了原有的 feature 結構**
   - 原始 WavTokenizer 的 encoder 產生的 features 被設計成能被 VQ 正確量化
   - LoRA 修改了 encoder，features 的分布可能偏離了 codebook 的「最佳區域」

3. **Noisy audio 的 features 本來就和 clean audio 不同**
   - Student 處理 noisy audio，Teacher 處理 clean audio
   - 即使 features 在 MSE 意義上接近，VQ 的 argmin 選擇可能完全不同

---

## 建議的解決方案

### 方案 1：直接監督 token 選擇（推薦）

不用 Feature Loss，直接用 Cross-Entropy 監督 token 選擇：

```python
# Student features → 到 codebook 的距離 → logits
distances = cdist(student_features, codebook)
logits = -distances / temperature

# 直接監督選擇 teacher 的 code
loss = CrossEntropy(logits, teacher_codes)
```

### 方案 2：更強的 temperature

降低 temperature 讓 softmax 分布更銳利：
- 目前 temperature = 0.5
- 可以嘗試 temperature = 0.1 或更低

### 方案 3：Contrastive Loss

使用 InfoNCE 或類似的 contrastive loss：
- 正樣本：student features 和對應的 teacher code
- 負樣本：其他 codes

### 方案 4：重新思考架構

- 不用 LoRA 修改 encoder
- 改為在 encoder 之後加一個「校正網絡」
- 或者在 VQ 之後加一個「token 校正器」

---

## 下一步實驗建議

1. **exp7 (VQ only)** - 仍然值得嘗試，但預期效果有限（VQ Loss 優化方向不對）

2. **exp8 (純 CE Loss，無 Feature Loss)** - 直接監督 token 選擇

3. **exp9 (更低 temperature)** - 讓 softmax 更銳利

4. **診斷 noisy vs clean** - 比較 noisy audio 和 clean audio 在**同一個 encoder** 下的 VQ 選擇差異
