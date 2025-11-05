# 訓練機轉診斷進度報告

**時間**: 2025-11-05 02:30

## 問題陳述

**核心疑問**: 為何 Training Loss 無法下降？Training Accuracy 無法上升？

## 訓練狀態（已停止）

- **Epoch**: 77/100 (已手動停止)
- **Train**: Loss 2.65, Acc 56.25%
- **Val**: Loss 4.91, Acc 38.57%
- **Learning Rate**: 1e-06 (幾乎為零)

**關鍵觀察**:
- Epoch 1-20: 快速進步（39% → 53%）
- Epoch 20-77: **嚴重平台期**（54% → 56%，僅提升 2%）
- Val Accuracy 持續在 36-38% 徘徊

## 已完成的診斷

### ✅ 診斷 1: 梯度流動分析

**工具**: `diagnose_training_mechanism.py`

**結果**:
```
梯度統計（52 層）:
  - 所有層梯度 norm 在 0.007 ~ 4.0 範圍內
  - ✅ 無梯度消失
  - ✅ 無梯度爆炸
  - Speaker projection 梯度: 4.0
  - Transformer layers 梯度: 0.13 ~ 0.27
  - Output projection 梯度: 0.37
```

**結論**: 梯度流動正常，不是梯度問題

### ✅ 診斷 2: 權重更新分析

**結果**:
```
權重變化（10 步後）:
  - Speaker projection: 相對變化 1.62%
  - Transformer layers: 相對變化 0.44% ~ 1.74%
  - ✅ 所有層都在更新
```

**結論**: 參數確實在更新，不是優化器問題

## 正在進行的診斷

### 🔄 診斷 3: 預測行為分析

**工具**: `diagnose_prediction_behavior.py` (運行中)

**檢查項目**:
1. 模型預測的 token 分布 vs Ground Truth
2. Top-20 tokens 的預測頻率
3. Token 0 和 Token 453 的詳細分析
4. 預測信心度（max probability, entropy）
5. Speaker Embedding 影響力測試
6. Per-Token 準確率統計

**狀態**: 
- ✅ 已修正記憶體問題（從 73GB 降到 1GB）
- 🔄 正在遍歷 Train set (2016 batches)
- ⏳ 預計完成時間: ~10-15 分鐘

**查看進度**:
```bash
# 方法 1: 附加到 tmux
tmux attach -t diagnosis

# 方法 2: 查看日誌
tail -f /home/sbplab/ruizi/WavTokenize-self-supervised/done/exp/diagnose_pred_full.log
```

## 待執行的診斷

### 📋 診斷 4: Loss 組成分析

**目的**: 分析哪些 token 貢獻了大部分 loss

**檢查項目**:
- Per-token loss 貢獻
- Token 453 loss vs 其他 tokens
- Information entropy vs Cross entropy

### 📋 診斷 5: Codebook 質量分析

**目的**: 檢查 frozen codebook 是否合適

**檢查項目**:
- Codebook embedding 聚類
- Noisy token vs Clean token 在 embedding space 的距離
- Token 453 的特殊性

## 機轉假設（待驗證）

### 假設 1: 模型學會了"預測眾數"策略 ⭐⭐⭐

**假設內容**:
- 模型發現"always predict Token 0" 可以得到 ~32% accuracy
- 這比隨機猜測 (1/4096 ≈ 0.02%) 好很多
- 所以模型收斂到這個"safe strategy"

**如何驗證**:
- 檢查模型預測的 Token 0 比例是否接近 32%
- 檢查其他 tokens 的預測準確率

**預期結果**:
- 如果成立: Train 預測 Token 0 佔 ~32%, Val 預測 Token 453 佔 ~24%
- 如果不成立: 預測分布應該更均勻

### 假設 2: Speaker Embedding 影響力太弱 ⭐⭐

**假設內容**:
- Speaker embedding 只是"裝飾品"
- 模型主要依賴 noisy tokens 本身
- 無法利用 speaker information

**如何驗證**:
- Zero speaker embedding 測試
- Random speaker embedding 測試
- 計算預測改變百分比

**預期結果**:
- 如果成立: <5% tokens 改變
- 如果不成立: >20% tokens 改變

### 假設 3: Frozen Codebook 限制表達能力 ⭐⭐

**假設內容**:
- WavTokenizer codebook 不適合 denoising task
- Frozen embeddings 無法調整

**如何驗證**:
- 分析 noisy/clean token pairs 的 embedding 距離
- 檢查是否有 tokens 過於相似

### 假設 4: Task 已接近理論上限 ⭐

**假設內容**:
- 70% tokens 被 noise 改變
- 56% accuracy 可能已經很接近上限

**如何驗證**:
- 計算 Oracle accuracy
- 分析人類在這個任務上的表現

## 改進方向（基於假設）

如果假設 1 成立:
- 使用 Focal Loss 降低眾數 tokens 權重
- 使用 Label Smoothing

如果假設 2 成立:
- 改用 Cross-Attention 而非簡單相加
- 使用 FiLM (Feature-wise Linear Modulation)
- 增加 Speaker Embedding 維度

如果假設 3 成立:
- Fine-tune codebook（允許微調）
- 添加 learnable projection layer

如果假設 4 成立:
- 降低 noise level
- 使用 Curriculum Learning

## 下一步行動

1. ⏳ **等待診斷 3 完成** (10-15 分鐘)
2. 📊 **分析預測行為報告**
3. 🔬 **根據結果決定是否執行診斷 4, 5**
4. 📝 **撰寫完整診斷報告**
5. 🚀 **實作最有希望的改進方向**

## 相關文檔

- `TRAINING_MECHANISM_HYPOTHESIS.md` - 機轉假設詳細說明
- `PLATEAU_MECHANISM_ANALYSIS.md` - Token Distribution Mismatch 分析
- `diagnose_training_mechanism.py` - 梯度與權重診斷工具
- `diagnose_prediction_behavior.py` - 預測行為診斷工具（運行中）

---

**更新時間**: 2025-11-05 02:30  
**診斷狀態**: 診斷 3 運行中  
**預計完成**: 2025-11-05 02:45
