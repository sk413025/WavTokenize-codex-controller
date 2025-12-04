# EmbDistillation 失敗診斷報告

**日期**: 2025-12-03
**實驗**: exp_1203 (exp7: Feature+VQ, exp8: EmbDistillation)

---

## 問題描述

**Loss 下降但 Token Accuracy 也下降**

| 實驗 | Loss 趨勢 | Token Acc 趨勢 |
|------|----------|---------------|
| exp7 (Feature+VQ) | 0.41 → 0.29 ↓ | 30% → 5% ↓ |
| exp8 (EmbDistillation) | 0.041 → 0.031 ↓ | 26% → 9% ↓ |

---

## 診斷結果

### 1. 初始狀態分析

使用 `diagnose_emb_distillation_failure.py` 對未訓練模型進行分析：

| 指標 | 數值 | 解讀 |
|------|------|------|
| **Token Accuracy** | **4.00%** | 幾乎是隨機 (1/4096 ≈ 0.02%) |
| **Teacher code 平均排名** | **2020 / 4096** | 排在中間，非常遠 |
| **到 Teacher code 距離** | **5.32** | 很大的距離 |
| **Codebook 內部平均距離** | **6.09** | 作為參照 |
| **比值** | **87.27%** | 距離幾乎等於 codebook 直徑 |
| **Rank ≤ 100** | **19.83%** | 只有 20% 在正確附近 |
| **Margin (邊界距離)** | **0.29** | 相對較小 |

### 2. 關鍵發現

```
┌─────────────────────────────────────────────────────────────┐
│  Student embedding 到 Teacher code 的距離：5.32             │
│  Codebook 內部平均距離：6.09                                │
│                                                             │
│  ⚠️ 比值：87.27%                                            │
│                                                             │
│  Noisy audio 的 embedding 離 Clean audio 選的 code 很遠！  │
│  幾乎等於「隨機選一個 code」的距離                           │
└─────────────────────────────────────────────────────────────┘
```

### 3. 問題根源

**Noise 對 encoder 的影響非常大**

1. Clean audio → Teacher embedding → Teacher code (正確的 code)
2. Noisy audio → Student embedding → 距離 Teacher code 很遠
3. MSE Loss 只優化「距離」，不保證「選對」
4. 4096 個 codebook 在 512 維空間中非常擁擠
5. LoRA 只有 154,048 參數 (0.19%)，容量可能不足

### 4. 為何 MSE Loss 失敗

```
MSE Loss = ||student_emb - codebook[teacher_code]||²
```

- MSE Loss 下降 = 平均距離減少
- 但 **距離減少 ≠ 選對 code**
- 因為 codebook 空間中：
  - 4096 個 code
  - 每個 code 的 Voronoi 區域很小
  - Student embedding 可能「距離減少」但仍在錯誤區域

```
          Teacher Code                     Teacher Code
               ●                                ●
              / \                              /|\
             /   \                            / | \
            /     \                          /  |  \
           ●       ●   Wrong code           ●  |   ●
          / \     / \                      / \ |  / \
         /   \   /   \                    /   \| /   \
        ●-----●-●-----●                  ●-----●-----●
             ↑                                 ↑
      Student emb 初始                 Student emb 訓練後

      MSE: 5.32                        MSE: 3.0 (↓)
      但仍在錯誤的 Voronoi 區域！
```

---

## 解決方案

### 方案 A: CE Loss (分類目標) ⭐ 推薦

```python
# CE Loss = -log(softmax(-distances)[teacher_code])
logits = -distances / temperature
ce_loss = CrossEntropy(logits, teacher_codes)
```

**優點**:
- 直接優化「選對 code」的概率
- 不需要 embedding 完全對齊
- 只需排名第一
- 梯度永遠指向正確方向

**exp9 配置**:
```bash
--ce_token_weight 1.0          # 主要 Loss
--emb_to_codebook_weight 0.1   # 輔助
--temperature 0.1               # 更尖銳的 softmax
--learning_rate 1e-4            # 較大學習率
```

### 方案 B: 增加 LoRA rank

```bash
--lora_rank 256   # 從 64 增加到 256
--lora_alpha 512  # 相應調整
```

**優點**: 更多可訓練參數
**缺點**: 可能仍不夠，且破壞原始能力

### 方案 C: 訓練整個 encoder

**優點**: 容量充足
**缺點**: 完全破壞原始能力

### 方案 D: 換架構 (Transformer Denoiser)

不修改 encoder，使用額外的 Transformer 在 token 空間做去噪：

```
Noisy audio → Encoder (frozen) → Noisy tokens → Transformer → Denoised tokens
```

**優點**: 不改變 encoder
**缺點**: 需要額外訓練

---

## Baseline 參考

原始 WavTokenizer 在 noisy vs clean 的 Token Match Rate:

| 數據集 | Match Rate |
|--------|-----------|
| Train (seen speakers) | 36.22% |
| Val (unseen speakers) | 5.39% |

這說明：
1. 即使原始模型，在已知說話者上也只有 36% 匹配率
2. 在未知說話者上幾乎沒有匹配 (5%)
3. 這是一個**非常困難的任務**

---

## 下一步行動

1. ✅ 創建 exp9: CE Token Loss 實驗
2. ⏳ 運行 exp9 並觀察 Token Accuracy 趨勢
3. ⏳ 如果 CE Loss 也失敗，考慮換架構

---

## 相關文件

- 診斷腳本: `diagnose_emb_distillation_failure.py`
- 診斷結果: `emb_distillation_diagnosis.json`
- exp9 腳本: `run_exp9_ce_token_loss.sh`
- Baseline 結果: `experiments/baseline_robustness/baseline_robustness_results.json`
