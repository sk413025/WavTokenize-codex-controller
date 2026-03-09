# Exp 4: Codebook Refresh Results

## 實驗目的
通過定期刷新未使用的 codebook entries，強制模型使用多樣化的 tokens，防止 token collapse。

## 方法
```python
if step % refresh_interval == 0:
    usage_count = count_codebook_usage(sliding_window)
    unused_mask = (usage_count < threshold)
    with torch.no_grad():
        codebook[unused_mask] = torch.randn_like(codebook[unused_mask])
```

## Baseline (exp_k v6 @ epoch 300)
- Entropy: 6.07
- Top-10 Mass: 19.7%
- Strict Accuracy: 0.91%

## 成功判準
- Val entropy > 6.07
- Val top-10 mass < 19.7%
- Val strict acc >= 0.82% (90% of baseline)

---

## 實驗設計

### 測試維度
- **Refresh Interval**: 50 / 100 / 200 steps (刷新頻率)
- **Usage Threshold**: 5 / 10 / 20 次 (使用門檻)

### 實驗矩陣

| 實驗 | Interval | Threshold | 刷新次數 | 策略 |
|------|----------|-----------|----------|------|
| 4a   | 100      | 10        | 10次     | 中等 (基準) |
| 4b   | 100      | 5         | 10次     | 中頻+嚴格 |
| 4b-2 | 50       | 5         | 20次     | 高頻+嚴格 (最激進) |
| 4c   | 50       | 10        | 20次     | 高頻+中等 |
| 4c-2 | 200      | 20        | 5次      | 低頻+寬鬆 (最保守) |
| 4d   | 100      | 5         | 10次     | 重複驗證 |

---

## 實驗結果

### ❌ 所有實驗均失敗

| 實驗 | Entropy | Top-10 Mass | Strict Acc | Δ Entropy | Δ Top-10 | Success |
|------|---------|-------------|------------|-----------|----------|---------|
| **Baseline** | **6.07** | **19.7%** | **0.91%** | - | - | - |
| 4a   | 5.58    | 25.4%       | 1.01%      | **-0.49** | **+5.7%** | ❌ |
| 4b   | 5.70    | 26.4%       | 0.51%      | **-0.37** | **+6.7%** | ❌ |
| 4b-2 | 5.53    | 26.8%       | 0.56%      | **-0.54** | **+7.1%** | ❌ |
| 4c   | 5.27    | 32.5%       | 0.59%      | **-0.80** | **+12.8%** | ❌ |
| 4c-2 | 5.56    | 29.1%       | 0.57%      | **-0.51** | **+9.4%** | ❌ |
| 4d   | 5.36    | 29.0%       | 0.43%      | **-0.71** | **+9.3%** | ❌ |

### 關鍵觀察

1. **所有實驗的 Entropy 都下降**
   - 範圍: -0.37 ~ -0.80
   - 最差: Exp 4c (高頻+中等) -0.80

2. **所有實驗的 Top-10 Mass 都上升**
   - 範圍: +5.7% ~ +12.8%
   - 最差: Exp 4c (高頻+中等) +12.8%

3. **高頻刷新反而更差**
   - Exp 4c (50 steps): Entropy 5.27, Top-10 32.5% ❌❌
   - Exp 4b-2 (50 steps): Entropy 5.53, Top-10 26.8% ❌
   - 比中頻/低頻刷新結果更差

4. **Codebook refresh 機制正常但無效**
   - 平均每次刷新 ~3300-3400 codes (80%+)
   - 但刷新後的 codes 仍然不被使用
   - 模型固執地使用少數固定 codes

---

## 深入分析

### 為什麼 Codebook Refresh 失敗？

#### 問題 1: 刷新後立即被忽略

觀察刷新記錄：
```
Step 100: 刷新 2784 codes → Used codes: 1312
Step 200: 刷新 3351 codes → Used codes: 745  
Step 300: 刷新 3354 codes → Used codes: 742
...
Step 800: 刷新 3356 codes → Used codes: 740
```

**發現**:
- 第一次刷新後，used codes 從 1312 → 745 (驟降)
- 之後穩定在 ~740 codes
- 刷新的 codes 在下一個 window 就被拋棄

**原因**: 模型的 encoder 已經學會將所有輸入映射到少數 codes 的區域，新刷新的 codes 無法被 encoder 選中。

#### 問題 2: Encoder-Codebook 不匹配

```
Encoder Output → Quantizer → Codebook
    ↓
已經學會輸出
在 [c1, c2, ..., c740] 附近的向量
    ↓
刷新後的 codes 在不同區域
    ↓
Encoder 輸出永遠選不到新 codes
```

**根本原因**: 只刷新 codebook 而不更新 encoder，導致：
- Encoder 仍然輸出在舊 codes 附近的向量
- 新刷新的 codes 在向量空間的其他位置
- Quantization 永遠選不到新 codes

#### 問題 3: 訓練動力學問題

即使新 codes 偶爾被選中：
- Encoder gradient 會推動它遠離新 codes
- 因為新 codes 是隨機初始化，reconstruction loss 高
- 模型學會避免使用新 codes

### 高頻刷新為什麼更差？

**Exp 4c (interval=50) 最差的原因**:

1. **過度干擾訓練**
   - 每 50 步就刷新一次
   - 模型沒有時間穩定學習
   - 訓練變得不穩定

2. **頻繁重置導致退化**
   - 剛開始使用的 codes 還沒穩定就被刷新
   - 導致可用 codes 越來越少
   - 最終只剩極少數"安全" codes

3. **Threshold=10 的副作用**
   - 保留使用 ≥10 次的 codes
   - 高頻刷新 + 中等門檻 = 很少 codes 能達標
   - 結果: 最嚴重的 collapse

---

## 結論

### ❌ Codebook Refresh 方法失敗

**原因總結**:

1. **治標不治本**
   - 只刷新 codebook，不更新 encoder
   - Encoder 已經學會輸出在少數區域
   - 新 codes 永遠不會被選中

2. **違反訓練動力學**
   - 刷新的 codes reconstruction loss 高
   - Gradient 推動模型遠離新 codes
   - 與訓練目標衝突

3. **高頻刷新有害**
   - 過度干擾訓練穩定性
   - 導致更嚴重的 collapse

### 深層問題

Token collapse 的根本原因不在於 codebook：
- **Encoder 輸出空間坍縮**: Encoder 學會將所有輸入映射到少數區域
- **VQ-VAE 的固有問題**: Codebook learning 與 encoder learning 不平衡
- **需要架構層面的改變**: 而非簡單的後處理

---

## Phase 2 總體結論

### 嘗試的方法

1. **Exp 3: Entropy Regularization** ❌
   - 正則化目標錯誤
   - 忽略時序結構
   - 訓練不穩定

2. **Exp 4: Codebook Refresh** ❌
   - 治標不治本
   - Encoder-Codebook 不匹配
   - 高頻刷新有害

### 關鍵發現

✅ **初始模型優於訓練後模型**
- Step 0: Entropy 6.26, Top-10 11.8%
- Step 1000: Entropy 5.27-5.70, Top-10 25-33%
- **證實**: 問題出在訓練過程

❌ **Loss-level 和 Codebook-level 修復均無效**
- 需要更深層次的架構改變

### 建議方向

**Phase 3 應該考慮**:

1. **Encoder 架構改進**
   - 增加 encoder 輸出的多樣性
   - 例如: Multiple codebook heads, Residual VQ

2. **訓練策略改變**
   - Curriculum on codebook usage
   - Adversarial training for diversity

3. **根本重新思考**
   - 是否需要 VQ？
   - 考慮 continuous latent space

---

## 實驗資訊

- 訓練步數: 1000 steps
- Batch size: 8 (grad_accum=2, effective=16)
- Learning rate: 1e-4
- LoRA rank: 256, alpha: 512
- Evaluation interval: 200 steps
- Codebook size: 4096
- 完成時間: 2026-02-03

