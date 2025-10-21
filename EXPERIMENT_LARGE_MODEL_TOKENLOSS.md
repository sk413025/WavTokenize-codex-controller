# 大模型 + 混合 Token Loss 實驗說明

**日期**: 2025-10-19  
**實驗目標**: 解決小模型頻譜重建「拼不起來」的問題

---

## 🔍 問題診斷

### 前一個實驗的問題 (codebook_emb_202510181441)

1. **模型容量嚴重不足** ⭐⭐⭐⭐⭐
   - 可訓練參數: 只有 **1.26M**
   - d_model=128, 2層 Transformer 太小
   - 無法學習複雜的 token 間關係
   - **訓練 Loss**: 8.0 → 3.8 (有學習，但不夠)

2. **Cross Entropy Loss 的缺陷** ⭐⭐⭐⭐
   - 只優化離散 token ID 的分類準確率
   - 不考慮聲學連續性
   - Token 0 和 Token 1 在 embedding 空間可能很近，但 CE 把它們當完全不同的類別
   - **導致頻譜破碎**

3. **驗證完全失敗** ⭐⭐⭐⭐
   - Val Loss: 1000000.0000 (所有 epoch)
   - Val Acc: 0.0000
   - 驗證函數有 bug，無法評估真實性能

---

## ✅ 解決方案: 大模型 + 混合 Token Loss

### 1️⃣ 模型容量提升 (4-5倍)

| 參數 | 舊值 | 新值 | 提升倍數 |
|------|------|------|----------|
| d_model | 128 | **256** | 2x |
| Encoder Layers | 2 | **4** | 2x |
| Decoder Layers | 2 | **4** | 2x |
| Feedforward Dim | 256 | **1024** | 4x |
| Attention Heads | 2 | **8** | 4x |
| 可訓練參數 | 1.26M | **~5-6M** | 4-5x |

**預期效果**:
- 更強的長時依賴建模
- 更好的 token 間關係學習
- 更豐富的特徵表示

---

### 2️⃣ 混合 Token Loss 設計

#### 為什麼需要混合損失？

**問題**: 單一 L2 Loss 可能導致模型「躺平」，輸出所有 embedding 的平均值。

**解決**: 混合三種互補的損失函數

```python
Total_Loss = 0.2 * CE_Loss + 1.0 * L2_Loss + 0.5 * Consistency_Loss
```

#### 各損失函數的作用

##### A. Cross Entropy Loss (權重 0.2) - 強監督信號

```python
# 計算方式
CE_Loss = CrossEntropy(predicted_logits, target_token_ids)

# 作用
- 直接優化 token ID 分類準確率
- 提供強監督信號，防止模型躺平
- 確保預測的 token 大致正確
```

**為什麼權重較小 (0.2)?**
- CE Loss 很強，容易主導訓練
- 我們更重視聲學連續性，而非離散分類

##### B. L2 Loss (權重 1.0) - 聲學連續性

```python
# 計算方式 (關鍵: 在 Embedding 空間！)
pred_embed = embedding_layer(predicted_tokens)    # [B, L, 512]
target_embed = embedding_layer(target_tokens)     # [B, L, 512]
L2_Loss = ||pred_embed - target_embed||²

# 作用
- 直接優化 embedding 向量的距離
- Token 0 和 Token 1 如果語義接近，embedding 也會接近
- 保證聲學平滑性和連續性
```

**為什麼權重最大 (1.0)?**
- 這是我們的主要優化目標
- 直接對應頻譜重建質量

##### C. Consistency Loss (權重 0.5) - 時間平滑性

```python
# 計算方式 (相鄰時間步的 embedding 差異)
pred_diff = pred_embed[:, 1:] - pred_embed[:, :-1]      # 相鄰差
target_diff = target_embed[:, 1:] - target_embed[:, :-1]
Consistency_Loss = ||pred_diff - target_diff||²

# 作用
- 確保相鄰 token 的平滑過渡
- 防止頻譜出現突變或破碎
- 保持時間連續性
```

**為什麼需要 (0.5)?**
- 補充 L2 Loss，專注於相鄰關係
- 避免頻譜「跳躍」

---

### 3️⃣ 技術改進細節

#### Python Unbuffered 輸出 (-u)

```bash
python -u wavtokenizer_transformer_denoising.py ...
```

**作用**:
- 強制立即輸出到日誌，無緩存
- `tail -f` 實時監控時無延遲
- 程式崩潰時確保最後的錯誤信息被記錄

#### 調整學習率

```bash
--learning_rate 5e-5  # 從 1e-4 降低
```

**原因**: 大模型需要更小的學習率，避免訓練不穩定

#### 調整 Batch Size

```bash
--batch_size 4  # 從 8 降低
--gradient_accumulation_steps 2  # 保持有效 batch size = 8
```

**原因**: 大模型需要更多記憶體

---

## 📊 預期效果對比

| 指標 | 小模型 (舊) | 大模型+混合Loss (新) |
|------|------------|---------------------|
| 模型參數 | 1.26M | ~5-6M |
| 訓練 Loss | 3.8 | < 2.0 (預估) |
| 頻譜連續性 | ❌ 破碎 | ✅ 平滑 |
| 收斂速度 | 慢 | 更快 |
| 聽覺質量 | 差 | 明顯提升 |
| 驗證性能 | 無效 (bug) | 正常 (已修正) |

---

## 🚀 執行方式

### 啟動訓練

```bash
cd /home/sbplab/ruizi/c_code
nohup bash run_transformer_large_tokenloss.sh > logs/large_tokenloss_bg_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 監控訓練

```bash
# 實時查看日誌
tail -f logs/transformer_large_tokenloss_*.log

# 查看 GPU 使用
nvidia-smi

# 查看訓練進程
ps aux | grep wavtokenizer_transformer_denoising
```

### 檢查結果

```bash
# 查看訓練曲線
results/transformer_large_tokenloss_*/training_history_epoch_100.png

# 聽音頻樣本
results/transformer_large_tokenloss_*/audio_samples/epoch_100/

# 查看頻譜圖
results/transformer_large_tokenloss_*/audio_samples/epoch_100/*_spec.png
```

---

## 📝 關鍵檢查點

### Epoch 50-100

- [ ] 訓練 Loss 是否下降到 < 5.0？
- [ ] 驗證 Loss 是否正常（不是 1000000）？
- [ ] 頻譜圖是否比之前更平滑？

### Epoch 200-300

- [ ] 訓練 Loss 是否 < 3.0？
- [ ] Enhanced 音頻是否更接近 Target？
- [ ] 是否還有頻譜破碎現象？

### Epoch 500+

- [ ] 是否達到收斂？
- [ ] 是否需要調整學習率？
- [ ] 是否需要調整 loss 權重？

---

## 🔧 如果效果不佳

### 如果 Loss 不下降

1. **降低學習率**: `5e-5 → 3e-5 → 1e-5`
2. **檢查梯度**: 可能梯度爆炸或消失
3. **減少模型複雜度**: 先試 `d_model=192, layers=3`

### 如果頻譜仍然破碎

1. **增加 Consistency 權重**: `0.5 → 0.8 → 1.0`
2. **降低 CE 權重**: `0.2 → 0.1 → 0.05`
3. **檢查 token_loss_system.py 的實現**

### 如果模型躺平 (Loss 不變)

1. **增加 CE 權重**: `0.2 → 0.4 → 0.6`
2. **檢查是否有梯度流**
3. **檢查 embedding_layer 是否被正確使用**

---

## 📚 理論背景

### 為什麼在 Embedding 空間計算 L2？

**離散 Token ID 的問題**:
- Token 0 和 Token 4095 的數值差很大（4095）
- 但它們的語義可能很接近（都表示低頻成分）
- 直接用 ID 計算距離會誤導模型

**Embedding 空間的優勢**:
- WavTokenizer 的 codebook embedding 已經學習了聲學相似性
- 語義相近的 token 在 embedding 空間也接近
- L2 距離反映真實的聲學差異

### 為什麼需要 Consistency Loss？

**L2 Loss 的局限**:
- 只約束每個時間點的 embedding
- 不約束相鄰時間點的關係
- 可能導致「每個點都對，但整體不連續」

**Consistency Loss 的作用**:
- 約束相鄰時間步的變化率
- 確保 `enhanced[t] - enhanced[t-1] ≈ target[t] - target[t-1]`
- 保證整體的時間連續性

---

## ✅ 總結

這個實驗結合了：
1. **大模型**: 提供足夠的學習容量
2. **混合損失**: CE (分類) + L2 (連續) + Consistency (平滑)
3. **技術優化**: unbuffered 輸出、學習率調整、記憶體管理

**預期**: 解決頻譜破碎問題，獲得更好的音頻重建質量！ 🎯
