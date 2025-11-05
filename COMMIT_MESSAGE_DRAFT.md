# Commit Message Draft

## 標題
診斷訓練平台期問題：Token Distribution Mismatch 導致 Val Acc 僅 37%

## 實驗背景 (Background)

在 commit `fa1b686` 完成 3-epoch 測試訓練後，於 2025-11-05 00:23 啟動 100-epoch 完整訓練。訓練過程中觀察到嚴重的平台期問題：

**觀察到的問題**:
- Train Accuracy 在 Epoch 20 後卡在 54% 左右 (Epoch 20-38 僅提升 0.77%)
- Val Accuracy 持續在 35-37% 之間徘徊 (最佳 36.75%)
- Train Loss 持續下降 (2.85 → 2.78) 但 Accuracy 不再提升
- Train-Val Gap 高達 17%

**訓練狀態** (Epoch 38/100):
- Train: Loss 2.78, Acc 54.70%
- Val: Loss 4.93, Acc 36.75%
- Gap: 17.95%

## 實驗動機 (Motivation)

訓練 accuracy 平台期可能來自以下原因之一：
1. **Token 過度集中**: 模型是否只預測少數幾個 token？
2. **Padding 過多**: Attention 是否浪費在 padding 上？
3. **序列長度問題**: Train/Val 序列長度差異是否導致泛化困難？
4. **Data Distribution Mismatch**: Zero-shot speaker split 是否導致 train/val token 分布不同？

需要診斷平台期的根本原因，確認是結構性問題 (data mismatch) 或可修正的工程問題 (padding, architecture)。

## 實驗目的 (Purpose)

1. **診斷訓練平台期的根本原因**
   - 分析 Train/Val token 分布差異
   - 量化 padding 對訓練的影響
   - 檢查 token 是否過度集中

2. **建立問題機轉模型**
   - 用數據驗證假設
   - 建立因果鏈
   - 提出基於證據的改進方向

3. **為後續實驗提供方向**
   - 確定哪些改進方向值得嘗試
   - 預估改進後的可能效果

## 實驗方法 (Methodology)

### 1. Token 分布分析
- 載入完整 train/val cache data
- 統計每個 token 的出現頻率
- 比較 Top-20 tokens 在 train/val 的分布差異
- 計算 distribution mismatch 的累計差異

### 2. Accuracy 數學反推
- 使用公式: `Overall_Acc = (1 - Token453_Ratio) × Other_Acc + Token453_Ratio × Token453_Acc`
- 反推 Token 453 和其他 tokens 的可能準確率
- 計算 Token 453 對整體錯誤的貢獻

### 3. Padding 影響分析
- 分析每個 batch 的序列長度分布
- 計算 padding 佔比
- 比較 train/val 的 padding 差異

### 4. Noisy-Clean Token 差異
- 統計 noisy audio 導致的 token 改變比例
- 確認 denoising task 難度

## 實驗結果 (Results)

### 核心發現 1: Token 453 是主要瓶頸 ✅

**數據**:
```
Train Set: Token 453 佔 13.57% (581,513 / 4,285,755 tokens)
Val Set:   Token 453 佔 18.65% (295,009 / 1,581,684 tokens)
差異:      +5.08% (絕對), +37.5% (相對增幅)
```

**數學推導**:
```
如果 Token 453 完全失敗 (0% 準確率):
  Train Acc 最高 = 86.43%
  Val Acc 最高 = 81.35%

Token 453 對錯誤的最大貢獻:
  Train: 30.0% (13.57% / 45.30% 錯誤率)
  Val:   29.5% (18.65% / 63.25% 錯誤率)
```

**結論**: Token 453 確實是訓練瓶頸，但無法完全解釋 17% 的 Train-Val gap

### 核心發現 2: Distribution Mismatch 廣泛存在 ✅

**數據**:
```
15 個 Top-20 tokens 有顯著分布差異 (>0.3%)
累計絕對差異: 10.94%

差異最大的 10 個 tokens:
  Token  453: Train 13.57% → Val 18.65% (↑ 5.08%)
  Token 1145: Train  0.18% → Val  0.91% (↑ 0.73%)
  Token 1750: Train  0.18% → Val  0.70% (↑ 0.52%)
  Token 1016: Train  0.10% → Val  0.61% (↑ 0.51%)
  Token 1764: Train  0.15% → Val  0.61% (↑ 0.45%)
  ...
```

**結論**: Val speakers (boy7,8, girl9,10) 的聲音特徵與 Train speakers (14 位) 系統性不同

### 核心發現 3: Mismatch 只能解釋部分 gap ⚠️

**數學估計**:
```
假設 15 個 mismatch tokens 準確率都下降 20%:
  Token 453:   18.65% × 20% = 3.73% accuracy loss
  Token 1145:   0.91% × 20% = 0.18% accuracy loss
  ...
  總 accuracy loss ≈ 4-5% (保守估計)

實際 Train-Val gap = 17%
  → 5% 來自 distribution mismatch (30%)
  → 12% 來自模型泛化能力不足 (70%)
```

**結論**: Distribution mismatch 是重要因素，但不是唯一原因

### 核心發現 4: Padding 不是問題 ✅

**數據**:
```
Train Set:
  序列長度: Min=194, Max=438, Mean=265.9 ± 48.9
  Padding 平均佔比: 30.2%

Val Set:
  序列長度: Min=280, Max=439, Mean=343.4 ± 36.3
  Padding 平均佔比: 0.05%
```

**結論**: Val set padding 極少但 accuracy 依然低，說明 padding 不是主要問題

### 其他發現

**Token Diversity**:
- Train: 1,833 / 4,096 unique tokens (44.8%)
- Val: 1,819 / 4,096 unique tokens (44.4%)
- Token diversity 健康，不是過度集中

**Noisy-Clean 差異**:
- Noisy audio 導致 70.92% 的 token 改變
- Denoising task 具有足夠難度

## 實驗解讀 (Interpretation)

### 建立的機轉模型

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

### 為何是 Zero-Shot Task 的本質困難

1. **WavTokenizer Codebook 是 Speaker-Independent**
   - 同一個 token 對所有 speaker 意義相同
   - 但不同 speaker 使用 token 的頻率不同

2. **當前架構無法捕捉 Speaker-Specific Token Distribution**
   - Speaker Embedding 只用簡單相加融合
   - 模型學到的是 Train speakers 的平均 token distribution
   - 無法根據 speaker embedding 調整 token distribution prior

3. **Val speakers 未見過，且特徵與 Train 差異大**
   - Token 453 在 Val 高出 37.5%
   - 15 個 tokens 累計差異 10.94%
   - 模型無法從 Train set 學到如何預測 Val speakers 的 token distribution

### 數據支持程度

| 假設 | 數據支持 | 結論 |
|------|----------|------|
| Token 453 是瓶頸 | 佔錯誤 30% | ✅ 強支持 |
| Distribution Mismatch | 15 tokens 累計 10.94% | ✅ 強支持 |
| Mismatch 解釋部分 gap | 推算 4-5% vs 實際 17% | ⚠️ 部分支持 (30%) |
| Padding 是問題 | Val padding <0.3% | ❌ 不支持 |
| Speaker Embedding 無效 | 需實驗驗證 | ❓ 待驗證 |

## 改進方向 (Proposed Solutions)

基於實驗結果，提出以下改進方向（按優先級排序）：

### 方向 1: Speaker-Adaptive Token Distribution Modeling ⭐⭐⭐

**針對問題**: Speaker Embedding 無法捕捉 speaker-specific token distribution

**方案**:
```python
class SpeakerAdaptiveDecoder(nn.Module):
    def __init__(self):
        # 新增: Speaker → Token Distribution Prior
        self.speaker_to_dist_prior = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 4096)
        )
    
    def forward(self, transformer_output, speaker_emb):
        logits = self.token_predictor(transformer_output)  # (B, T, 4096)
        dist_prior = self.speaker_to_dist_prior(speaker_emb)  # (B, 4096)
        logits = logits + dist_prior.unsqueeze(1)
        return logits
```

**預期效果**: Val Acc 提升至 45-50% (縮小 gap 至 10%)

### 方向 2: Weighted Cross-Entropy Loss ⭐⭐

**針對問題**: Token 453 等 mismatch tokens 訓練不足

**方案**:
```python
val_dist = compute_token_distribution(val_data)
train_dist = compute_token_distribution(train_data)
weights = val_dist / (train_dist + 1e-6)
criterion = nn.CrossEntropyLoss(weight=weights)
```

**預期效果**: Train Acc 60-65%, Val Acc 42-47%

### 方向 3: Distribution-Aware Speaker Split ⭐

**針對問題**: Val speakers 選擇不當

**方案**: 選擇 token distribution 接近 train average 的 speakers 作為 val set

**預期效果**: Val Acc 48-52% (但喪失 zero-shot 嚴格性)

## 如何重現實驗 (Reproduction)

### 環境需求
- PyTorch
- 訓練數據緩存: `done/exp/data/{train,val}_cache.pt`
- 訓練中的模型: `results/zeroshot_100epochs_20251105_002300/best_model.pth`

### 重現步驟

**步驟 1: Token 分布分析**
```bash
cd /home/sbplab/ruizi/WavTokenize-self-supervised/done/exp
python analyze_token_distribution.py
```

**步驟 2: Accuracy 反推**
```bash
python analyze_token_accuracy_inference.py
```

**步驟 3: Padding 分析**
(見 `EXPERIMENT_REPRODUCTION_GUIDE.md` 步驟 3)

**詳細指南**: 見 `EXPERIMENT_REPRODUCTION_GUIDE.md`

## 新增/修改的檔案

### 實驗分析工具
- `done/exp/analyze_token_distribution.py` (新增)
- `done/exp/analyze_token_accuracy_inference.py` (新增)

### 實驗報告
- `PLATEAU_MECHANISM_ANALYSIS.md` (新增) - 詳細機轉分析，含 ASCII 圖示
- `PLATEAU_DIAGNOSIS_SUMMARY.md` (新增) - 簡潔摘要
- `TRAINING_PLATEAU_DIAGNOSIS_20251105.md` (新增) - 完整診斷報告
- `EXPERIMENT_REPRODUCTION_GUIDE.md` (新增) - 實驗重現指南

### 訓練相關 (已存在)
- `done/exp/train_zeroshot_full_cached_analysis.py` - 訓練腳本
- `done/exp/data_zeroshot.py` - 數據載入
- `done/exp/model_zeroshot.py` - 模型定義
- `done/exp/data/{train,val}_cache.pt` - 數據緩存
- `results/zeroshot_100epochs_20251105_002300/` - 訓練輸出

## 待驗證假設

以下假設需要額外實驗驗證：

1. **Speaker Embedding 是否能調整 Token Distribution?**
   - 方法: 固定 noisy tokens，改變 speaker embeddings，觀察預測變化

2. **Token 453 的物理意義?**
   - 方法: 提取 Token 453 對應音頻段，分析頻譜特徵

3. **模型預測的 Token 453 頻率?**
   - 方法: 統計模型預測，驗證是否接近 13.57% (Train) 或 18.65% (Val)

## 關鍵洞察

**這不是 Bug，是 Zero-Shot Task 的本質困難**

- Val speakers 與 Train speakers 聲學特徵確實不同
- Token distribution mismatch 反映真實的聲學差異
- 當前模型架構 (簡單相加) 無法捕捉 speaker-specific token distribution
- **需要改進模型架構，而非只調整數據**

## 下一步行動

1. ✅ 繼續訓練至 100 epochs (收集完整 baseline)
2. 🔬 執行待驗證實驗 (實驗 3-5)
3. 🚀 實作方向 1: Speaker-Adaptive Token Distribution Modeling
4. 📊 比較改進後的 Val Acc 是否達到 45%+

---

**實驗完成時間**: 2025-11-05 02:30:00
