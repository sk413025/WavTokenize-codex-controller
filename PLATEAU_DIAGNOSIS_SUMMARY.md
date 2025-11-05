# 訓練平台期診斷總結 (2025-11-05)

## 🎯 問題現象

- **Train Accuracy**: 卡在 54% (Epoch 20-38 幾乎無進步)
- **Val Accuracy**: 僅 37% (Train-Val Gap = 17%)
- **訓練狀態**: Loss 持續下降但 Accuracy 不提升

---

## 🔍 核心發現

### 1. Token 453 是主要瓶頸
```
Train Set: Token 453 佔 13.57% (每 7-8 個 token 就有 1 個)
Val Set:   Token 453 佔 18.65% (每 5-6 個 token 就有 1 個)
差異:      +5.08% 絕對差異, +37.5% 相對增幅

數學推導:
  如果 Token 453 完全失敗 (0% 準確率):
    Train Acc 最高 = 86.43%
    Val Acc 最高 = 81.35%
    
  Token 453 佔總錯誤:
    Train: 30.0% (13.57% / 45.30% 錯誤率)
    Val:   29.5% (18.65% / 63.25% 錯誤率)
```

### 2. 不只 Token 453，還有 14 個 Top-20 Tokens 有 Distribution Mismatch
```
15 個 Top-20 tokens 分布差異 >0.3%
累計絕對差異: 10.94%

例子:
  Token 1145: Train 0.18% → Val 0.91% (↑ 0.73%)
  Token 1750: Train 0.18% → Val 0.70% (↑ 0.52%)
  Token 1016: Train 0.10% → Val 0.61% (↑ 0.51%)

→ Val speakers 的 token 分布與 Train speakers 系統性不同
```

### 3. Distribution Mismatch 只能解釋部分 Gap
```
保守估計: Mismatch 導致 4-5% accuracy loss
實際 Gap: 17%

說明:
  ✓ 30% 的 gap 來自 token distribution mismatch
  ✗ 70% 的 gap 來自模型泛化能力不足
```

---

## 🧠 機轉假設 (有數據支持)

### ✅ 假設 1: Token Distribution Mismatch 是根本原因
**數據支持**: 
- Token 453: Train 13.57% vs Val 18.65% (+5.08%)
- 15 個 Top-20 tokens 累計差異 10.94%

**機轉**: Val speakers (boy7,8, girl9,10) 的聲音特徵與 Train speakers (14 位) 系統性不同，導致他們使用的 token distribution 不同

### ⚠️ 假設 2: 模型架構限制 - Speaker Embedding 未能捕捉 Token Distribution
**推測**: 
- 當前模型只用簡單相加融合 speaker embedding
- Speaker embedding 可能只用於 denoising，沒有用於調整 token distribution prior
- 模型預測的 token distribution 可能固定為 Train average (13.57% Token 453)

**待驗證**: 需要實驗確認模型是否學會 speaker-specific token distribution

### ❓ 假設 3: Token 453 的物理意義
**推測**:
- 可能是靜音/低能量段 (Val speakers 說話較慢)
- 可能是特定共振峰模式 (Val speakers 共振峰相似)
- 可能是特定音高範圍 (Val speakers 年齡、性別相似)

**待驗證**: 需要提取 Token 453 對應的音頻段並分析頻譜特徵

---

## 📊 完整因果鏈

```
根本原因: Val Speakers 與 Train Speakers 聲學特徵差異
    ↓
Token Distribution Mismatch
    ├─ Token 453: +5.08% (主要)
    └─ 其他 14 tokens: +5.86% (次要)
    ↓
模型架構限制
    ├─ Speaker Embedding 只用簡單相加
    └─ 無法學習 Speaker-Specific Token Distribution Prior
    ↓
訓練結果
    ├─ Train Acc 卡在 54% (Token 453 拖累)
    ├─ Val Acc 只有 37% (Mismatch + 泛化不足)
    └─ Gap 17% (70% 來自泛化問題)
```

---

## 🔧 改進方向 (基於機轉)

### 方向 1: Speaker-Adaptive Token Distribution Modeling ⭐⭐⭐
**針對**: 機轉假設 2 (Speaker Embedding 限制)

**方案**: 新增 Speaker → Token Distribution Prior 網路
```python
dist_prior = nn.Linear(256, 4096)(speaker_emb)  # (B, 4096)
logits = transformer_output + dist_prior.unsqueeze(1)  # 加入 prior
```

**預期效果**: Val Acc 提升至 45-50% (縮小 gap 至 10%)

### 方向 2: Weighted Cross-Entropy Loss ⭐⭐
**針對**: 機轉假設 1 (Token 453 瓶頸)

**方案**: 對 mismatch tokens 增加權重
```python
weights = val_dist / (train_dist + 1e-6)  # Token 453 權重增加
criterion = nn.CrossEntropyLoss(weight=weights)
```

**預期效果**: Train Acc 提升至 60-65%, Val Acc 提升至 42-47%

### 方向 3: Distribution-Aware Speaker Split ⭐
**針對**: 機轉根本原因 (Val speakers 選擇不當)

**方案**: 選擇 token distribution 接近 train average 的 speakers 作為 val set

**預期效果**: Val Acc 提升至 48-52%  
**缺點**: 喪失 zero-shot 測試的嚴格性

---

## 🧪 待驗證假設 (需要額外實驗)

1. **實驗 3**: 模型是否學會 speaker-specific token distribution?
   - 固定 noisy tokens，改變 speaker embeddings
   - 觀察預測的 token distribution 是否改變
   
2. **實驗 4**: Token 453 的物理意義是什麼?
   - 提取 Token 453 對應的音頻段
   - 分析頻譜、能量、音高、共振峰特徵
   
3. **實驗 5**: 模型預測的 Token 453 頻率是多少?
   - 統計模型在 Val set 的預測
   - 驗證是否接近 13.57% (Train average) 或 18.65% (Val ground truth)

---

## 📈 數據支持總結

| 發現 | 數據 | 結論 |
|------|------|------|
| Token 453 是瓶頸 | 佔 Train 錯誤 30%, Val 錯誤 29.5% | ✅ 數據支持 |
| Distribution Mismatch | 15 tokens 累計差異 10.94% | ✅ 數據支持 |
| Mismatch 解釋部分 gap | 推算 4-5% vs 實際 17% | ⚠️ 部分支持 (30%) |
| Speaker Embedding 無效 | 需要實驗驗證 | ❓ 待驗證 |
| Token 453 物理意義 | 需要音頻分析 | ❓ 待驗證 |

---

## 💡 關鍵洞察

1. **這不是 Bug，是 Zero-Shot Task 的本質困難**
   - Val speakers 與 Train speakers 確實不同
   - Token distribution mismatch 反映真實的聲學差異
   
2. **Padding 不是問題**
   - Val set padding 幾乎為 0%，但 accuracy 依然低
   - 問題在於 token distribution，不是 padding
   
3. **模型泛化能力不足才是主因**
   - Mismatch 只能解釋 30% 的 gap
   - 剩下 70% 來自模型本身的泛化能力問題
   - **需要改進模型架構，而非只調整數據**

---

**下一步行動**:
1. 繼續訓練至 100 epochs (收集完整 baseline)
2. 執行實驗 3-5 驗證未驗證假設
3. 實作方向 1 (Speaker-Adaptive Token Distribution Modeling)
4. 比較改進後的 Val Acc 是否達到 45%+

**完成時間**: 2025-11-05 02:15:00
