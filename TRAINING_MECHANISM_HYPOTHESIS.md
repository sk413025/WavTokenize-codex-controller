# 訓練無法改善的深層機轉分析

## 問題陳述

**核心疑問**: 為何 Training Loss 和 Accuracy 都無法下降？

**觀察現象**:
- Epoch 1-20: Train Loss 4.59 → 2.85, Acc 39% → 53% (快速進步)  
- Epoch 20-59: Train Loss 2.85 → 2.66, Acc 53% → 56% (**嚴重平台期**)
- Val Accuracy: 持續在 35-38% 徘徊
- Learning Rate: 已降至 3.13e-06 (從 1e-4 降到幾乎為零)

## 初步診斷結果（基於 diagnose_training_mechanism.py）

### ✅ 診斷 1: 梯度流動 - 正常

```
梯度統計（52 層）:
  - 所有層梯度 norm 在 0.007 ~ 4.0 範圍內
  - 無梯度消失 (所有 norm > 1e-6)
  - 無梯度爆炸 (所有 norm < 100)
  - Speaker projection 梯度最大: 4.0 (合理)
  - Transformer layers 梯度: 0.13 ~ 0.27 (正常)
  - Output projection 梯度: 0.37 (正常)
```

**結論**: ✅ 梯度流動健康，不是梯度消失/爆炸問題

### ✅ 診斷 2: 權重更新 - 正常

```
權重變化統計（10 步後）:
  - Speaker projection: 相對變化 1.62%
  - Transformer layers: 相對變化 0.44% ~ 1.74%
  - 所有層都在更新（無 frozen 層）
```

**結論**: ✅ 參數確實在更新，不是優化器問題

## 缺少的診斷數據

基於目前分析，我們還需要以下數據來建立完整機轉：

### 🔍 缺少的診斷 1: **模型預測行為分析**

**需要的數據**:
1. **預測 token 分布 vs Ground Truth 分布**
   - 模型是否學會了預測 "正確的分布"？
   - 還是只是學會了預測 "最常見的 token"？

2. **Per-token 準確率**
   - 哪些 token 完全學不會？
   - Token 453 的準確率是多少？

3. **預測信心度**
   - Logits 的最大值有多大？
   - Softmax 後的最大機率是多少？
   - 預測是否過於自信（overconfident）？

**為何重要**: 可以判斷模型是否陷入 "預測眾數" 的策略

### 🔍 缺少的診斷 2: **Loss 組成分析**

**需要的數據**:
1. **Per-token Loss 貢獻**
   - Token 453 貢獻多少 loss？
   - Top-20 tokens 貢獻多少 loss？
   - Rare tokens 貢獻多少 loss？

2. **Cross-Entropy 分解**
   - Information Entropy: -Σ p_true * log(p_true)
   - Cross Entropy: -Σ p_true * log(p_pred)  
   - KL Divergence: H(p_true, p_pred) - H(p_true)
   
**為何重要**: 可以判斷模型是 "完全不會預測" 還是 "分布不匹配"

### 🔍 缺少的診斷 3: **Speaker Embedding 影響力**

**需要的數據**:
1. **Speaker embedding 對預測的實際影響**
   - Zero speaker embedding vs Normal: 預測改變多少？
   - Random speaker embedding vs Normal: 預測改變多少？
   - Swapped speaker embeddings: 預測改變多少？

2. **Speaker embedding 與 Token embedding 的相對強度**
   - Token embedding norm: ~多少？
   - Speaker projection 輸出 norm: ~多少？
   - 相加後誰主導？

**為何重要**: 可以判斷 speaker conditioning 是否真的在起作用

### 🔍 缺少的診斷 4: **Frozen Codebook 的影響**

**需要的數據**:
1. **Codebook embedding 質量**
   - Codebook embeddings 的離散程度
   - Clean tokens vs Noisy tokens 在 embedding space 的距離
   - 是否存在 "相似 token" 導致難以區分？

2. **Embedding space 分析**
   - Token embeddings 聚類結構
   - 是否有 token 過於相似？
   - Token 453 與其他 tokens 的距離

**為何重要**: 可以判斷 frozen codebook 是否限制了學習能力

### 🔍 缺少的診斷 5: **Learning Rate 與 Optimization**

**需要的數據**:
1. **Loss landscape 平坦度**
   - 小擾動 (1e-4) 對 loss 的影響
   - Sharp minima vs Flat minima

2. **Learning rate 敏感度**
   - 當前 LR (3e-6) 是否太小？
   - 測試 10x, 100x LR 是否能降低 loss？

3. **Optimizer 狀態**
   - AdamW 的 momentum 統計
   - 梯度的 moving average

**為何重要**: 可以判斷是否陷入局部最優

### 🔍 缺少的診斷 6: **數據質量**

**需要的數據**:
1. **Noisy vs Clean token 差異程度**
   - 平均多少 % token 被 noise 改變？
   - 改變的 token 中，有多少是 Token 453？

2. **Task 難度**
   - 如果人類看 noisy tokens，能預測 clean tokens 嗎？
   - Baseline (隨機猜測) 的 accuracy 是多少？
   - Upper bound (Oracle) 的 accuracy 是多少？

**為何重要**: 可以判斷 56% accuracy 是否已接近任務上限

## 機轉假設（基於現有數據）

### 假設 1: **模型已學會 "Safe Strategy" - 預測眾數** ⭐⭐⭐

**證據**:
- Train set 最常見 token: **Token 0** (32%)
- Val set 最常見 token: **Token 453** (24%)
- Train accuracy 卡在 54% = (1 - 0.32) × random + 0.32 × 100%？

**驗證方法**:
```python
# 檢查模型預測的 token 分布
pred_dist = Counter(predictions.flatten().tolist())
true_dist = Counter(clean_tokens.flatten().tolist())

# 如果 pred_dist 過度集中在 Token 0/453，說明模型在"賭眾數"
```

**如果成立**:
- 模型學會了 "預測訓練集的眾數" 作為安全策略
- 這解釋了為何 train acc ~54% 但 val acc 只有 37%
- 因為 val 的眾數是 Token 453 (24%)，而 train 是 Token 0 (32%)

### 假設 2: **Speaker Embedding 影響力太弱** ⭐⭐

**證據**:
- Speaker projection 權重相對變化: 1.62%
- Transformer layers 權重相對變化: 0.44% ~ 1.74%
- 兩者量級相近，說明 speaker 沒有主導

**驗證方法**:
```python
# 比較有/無 speaker embedding 的預測差異
diff_pct = (pred_normal != pred_zero_speaker).float().mean() * 100
# 如果 < 5%，說明 speaker embedding 幾乎無用
```

**如果成立**:
- Speaker embedding 只是"裝飾品"
- 模型主要依賴 noisy tokens 本身
- 無法利用 speaker information 來調整預測

### 假設 3: **Frozen Codebook 限制了表達能力** ⭐⭐

**證據**:
- Codebook 來自 WavTokenizer (預訓練)
- 該 codebook 可能不適合 denoising task
- Token embeddings 是 frozen 的，無法針對 denoising 調整

**驗證方法**:
```python
# 分析 noisy token embedding 與 clean token embedding 的距離
noisy_emb = codebook[noisy_tokens]
clean_emb = codebook[clean_tokens]
distance = (noisy_emb - clean_emb).norm(dim=-1).mean()

# 如果距離很大，說明 noisy/clean 在 embedding space 很遠
# 但 Transformer 必須學習這個複雜的映射
```

**如果成立**:
- Frozen codebook 強迫模型學習複雜的非線性映射
- 如果允許 fine-tune codebook，可能學得更好

### 假設 4: **Task 本質上很困難（接近上限）** ⭐

**證據**:
- Noisy audio 導致 70.92% tokens 改變
- 即使人類也難以從 noisy tokens 恢復 clean tokens
- 56% accuracy 可能已經很接近任務理論上限

**驗證方法**:
```python
# 計算 Oracle accuracy (最佳可能)
# 如果 70% token 改變，且這些改變是隨機的
# Oracle accuracy = 30% (unchanged) + 70% × (1/4096) ≈ 30%

# 但實際 train acc 56%，說明模型確實學到了一些 pattern
```

**如果成立**:
- 56% 已經是很好的結果
- 進一步提升需要改變 task 設定（降低 noise level）

## 下一步診斷計劃

### 優先級 1: 預測行為分析
```python
# 實作 diagnose_prediction_behavior.py
1. 統計模型預測的 token 分布
2. 計算 per-token accuracy
3. 分析預測信心度 (max prob)
4. 比較 pred_dist vs true_dist
```

### 優先級 2: Speaker Embedding 影響力
```python
# 實作 diagnose_speaker_influence.py
1. Zero speaker embedding 測試
2. Random speaker embedding 測試  
3. Swapped speaker embedding 測試
4. 計算預測改變百分比
```

### 優先級 3: Loss 組成分析
```python
# 實作 diagnose_loss_composition.py
1. Per-token loss 貢獻
2. KL divergence 分解
3. Information entropy 分析
```

### 優先級 4: Frozen Codebook 分析
```python
# 實作 diagnose_codebook_quality.py
1. Codebook embedding 聚類
2. Noisy vs Clean token 距離
3. Token 453 特性分析
```

## 預期發現

基於目前數據，我預期會發現：

1. **模型確實在預測眾數** (Token 0 在 train, Token 453 在 val)
2. **Speaker embedding 影響 <5% tokens** (幾乎無用)
3. **Token 453 準確率接近 0%** (完全學不會)
4. **Logits 過於自信** (max prob > 0.9)
5. **Frozen codebook 導致某些 token pair 難以區分**

如果這些預期成立，改進方向應該是：

1. **增強 Speaker Conditioning**
   - 使用 cross-attention 而非簡單相加
   - 使用 FiLM (Feature-wise Linear Modulation)

2. **Fine-tune Codebook**
   - 允許微調 codebook embeddings
   - 或訓練額外的 projection layer

3. **改變 Loss Function**
   - 使用 Focal Loss 降低眾數 tokens 的權重
   - 使用 Label Smoothing 降低過度自信

4. **降低 Task 難度**
   - 降低 noise level
   - 或使用 curriculum learning (從簡單到困難)

---

**診斷腳本狀態**: 
- ✅ `diagnose_training_mechanism.py` (梯度、權重更新) - 運行中
- 🔲 `diagnose_prediction_behavior.py` - 待實作
- 🔲 `diagnose_speaker_influence.py` - 待實作  
- 🔲 `diagnose_loss_composition.py` - 待實作
- 🔲 `diagnose_codebook_quality.py` - 待實作
