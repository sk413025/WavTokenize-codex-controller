# 離散 Token 訓練混合損失設計

**實驗日期**: 2025-01-23  
**實驗目的**: 解決 Frozen Codebook 實驗的問題，設計適合離散 token 訓練的混合損失函數

---

## 問題背景

### Frozen Codebook 實驗的問題

**實驗結果** (182 epochs, 43.5 小時):
- Train Accuracy: **66.78%** ✓
- Val Accuracy: **14.92%** ✗ (沒有改善)
- 嚴重過擬合: Train/Val gap = **51.9%**

**根本原因**:
1. **Token 準確度 ≠ 音頻質量**: 預測正確的 token ID 不代表重建的音頻好
2. **離散性問題**: CrossEntropyLoss 只看 token ID，忽略 embedding 空間的連續性
3. **數據限制**: 只用了 100 句/語者，而非全部 288 句

---

## 解決方案：混合損失函數

### 設計理念

借鑑 `ttt2.py` 的成功經驗（連續特徵訓練）：

**ttt2.py 的策略**:
```
階段 1 (Early): 內容一致性損失 (Content Consistency)
  → 學習：相同 content_id 的樣本應有相似的表示
  → 目的：不同語者/材質說同一句話 → 相似的語義表示

階段 2 (Later): L2 特徵損失 + 內容一致性
  → 學習：去噪重建，同時保持語者身份
  → 目的：在去噪的同時保留語者特徵
```

**適配到離散 Token**:
```
1. Token CrossEntropy Loss
   → 確保預測正確的 token ID
   
2. Content Consistency Loss (動態權重)
   → 相同 content_id 的 token embeddings 應該相似
   → 早期高權重，後期衰減
   
3. Embedding L2 Loss
   → 在 embedding 空間接近 clean token
   → 利用 Frozen Codebook 的連續表示
```

---

## 混合損失函數設計

### ASCII 架構圖：Loss 計算位置

```
訓練流程 (每個 Batch):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                      輸入資料
                         ↓
    ┌────────────────────────────────────────────────┐
    │  Noisy Audio → WavTokenizer.encode()           │
    │  Clean Audio → WavTokenizer.encode()           │
    │  → Noisy Token IDs (B, T)                      │
    │  → Clean Token IDs (B, T)                      │
    │  → Content IDs (B,)                            │
    └────────────────────────────────────────────────┘
                         ↓
    ┌────────────────────────────────────────────────┐
    │  Token Denoising Transformer                   │
    │                                                 │
    │  Noisy Tokens → Frozen Codebook Lookup         │
    │               → Positional Encoding            │
    │               → Transformer Encoder            │
    │               → Output Projection              │
    │               → Logits (B, T, 4096)            │
    └────────────────────────────────────────────────┘
                         ↓
    ┌────────────────────────────────────────────────┐
    │  DiscreteHybridLoss.forward()                  │
    │  ════════════════════════════════              │
    │                                                 │
    │  📍 Loss 計算位置 1: CE Loss                   │
    │  ├─ Input: logits (B,T,4096), target_tokens   │
    │  ├─ Process: CrossEntropyLoss(logits, target) │
    │  └─ Output: ce_loss (scalar)                   │
    │                                                 │
    │  📍 Loss 計算位置 2: Content Consistency       │
    │  ├─ Input: logits, content_ids                 │
    │  ├─ Process:                                   │
    │  │   1. pred_tokens = logits.argmax(-1)       │
    │  │   2. embeddings = codebook[pred_tokens]    │
    │  │   3. sentence_emb = embeddings.mean(dim=1) │
    │  │   4. 計算相同 content_id 的中心             │
    │  │   5. cosine_similarity(emb, center)        │
    │  └─ Output: content_loss (scalar)              │
    │                                                 │
    │  📍 Loss 計算位置 3: Embedding L2              │
    │  ├─ Input: logits, target_tokens, codebook    │
    │  ├─ Process:                                   │
    │  │   1. pred_embeddings = codebook[pred_tok]  │
    │  │   2. target_embeddings = codebook[target]  │
    │  │   3. MSE(pred_emb, target_emb)             │
    │  └─ Output: embed_loss (scalar)                │
    │                                                 │
    │  🎯 總損失計算                                 │
    │  total_loss = 1.0 * ce_loss                    │
    │             + w(epoch) * content_loss          │
    │             + 0.3 * embed_loss                 │
    │                                                 │
    │  其中 w(epoch) 是動態權重:                     │
    │  - Epoch 0-50:  0.5 → 0.25 (warmup)           │
    │  - Epoch 50+:   0.25 → ~0.01 (decay)          │
    └────────────────────────────────────────────────┘
                         ↓
                  total_loss.backward()
                  optimizer.step()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 1. Token CrossEntropy Loss

**公式**:
$$
\mathcal{L}_{\text{CE}} = -\frac{1}{B \times T} \sum_{b=1}^{B} \sum_{t=1}^{T} \log P(y_{b,t} | \hat{y}_{b,t})
$$

**作用**: 
- 確保模型預測正確的 token ID
- 標準的分類損失

**實現**:
```python
logits_flat = pred_logits.reshape(B * T, vocab_size)
targets_flat = target_tokens.reshape(B * T)
ce_loss = nn.CrossEntropyLoss()(logits_flat, targets_flat)
```

---

### 2. Content Consistency Loss

**核心理念**:
相同 `content_id` 的樣本，即使語者、材質不同，也應該有相似的語義表示。

**計算步驟**:

1. **獲取 sentence embeddings**:
   ```python
   pred_tokens = logits.argmax(dim=-1)  # (B, T)
   pred_embeddings = codebook[pred_tokens]  # (B, T, 512)
   sentence_embeddings = pred_embeddings.mean(dim=1)  # (B, 512)
   ```

2. **計算每個 content_id 的中心**:
   ```python
   for content_id in unique_contents:
       mask = (content_ids == content_id)
       content_embeddings = sentence_embeddings[mask]  # (N, 512)
       center = content_embeddings.mean(dim=0)  # (512,)
   ```

3. **計算損失 (餘弦相似度)**:
   ```python
   similarities = F.cosine_similarity(content_embeddings, center, dim=1)
   content_loss = 1.0 - similarities.mean()
   ```

**動態權重策略**:
$$
w_{\text{content}}(e) = \begin{cases}
w_{\max} \cdot (1 - 0.5 \cdot \frac{e}{E_{\text{warmup}}}) & e < E_{\text{warmup}} \\
w_{\max} \cdot 0.5 \cdot \exp(-3 \cdot \frac{e - E_{\text{warmup}}}{E_{\text{total}} - E_{\text{warmup}}}) & e \geq E_{\text{warmup}}
\end{cases}
$$

其中:
- $e$: 當前 epoch
- $E_{\text{warmup}} = 50$: Warmup epochs
- $E_{\text{total}} = 600$: 總 epochs
- $w_{\max} = 0.5$: 最大權重

**權重變化曲線**:
```
Epoch 0-50:   權重從 0.5 線性下降到 0.25
Epoch 50-600: 權重從 0.25 指數衰減到 ~0.01
```

**理由**:
- **早期 (Epoch 0-50)**: 高權重，讓模型學習"相同內容 → 相似表示"
- **中期 (Epoch 50-200)**: 逐漸降低，開始專注於去噪
- **後期 (Epoch 200+)**: 接近 0，完全專注於 token 預測和 embedding 重建

---

### 3. Embedding L2 Loss

**公式**:
$$
\mathcal{L}_{\text{embed}} = \frac{1}{B \times T} \sum_{b=1}^{B} \sum_{t=1}^{T} \| \mathbf{e}^{\text{pred}}_{b,t} - \mathbf{e}^{\text{target}}_{b,t} \|_2^2
$$

**作用**:
- 在 **embedding 空間** 約束模型輸出
- 利用 Frozen Codebook 的連續表示
- 比純 token ID 更平滑的梯度

**實現**:
```python
pred_tokens = pred_logits.argmax(dim=-1)  # (B, T)
pred_embeddings = codebook[pred_tokens]    # (B, T, 512)
target_embeddings = codebook[target_tokens]  # (B, T, 512)

embed_loss = F.mse_loss(pred_embeddings, target_embeddings)
```

**優勢**:
- Token ID 是離散的（0, 1, 2, ...），embedding 是連續的
- 即使預測錯誤的 token，如果 embedding 接近，音頻質量仍可能較好

---

### 4. 總損失函數

$$
\mathcal{L}_{\text{total}} = w_{\text{CE}} \cdot \mathcal{L}_{\text{CE}} + w_{\text{content}}(e) \cdot \mathcal{L}_{\text{content}} + w_{\text{embed}} \cdot \mathcal{L}_{\text{embed}}
$$

**默認權重**:
- $w_{\text{CE}} = 1.0$ (固定)
- $w_{\text{content}} = 0.5$ (最大值，動態衰減)
- $w_{\text{embed}} = 0.3$ (固定)

---

## 實驗設計

### 數據改進

**之前** (Frozen Codebook):
```python
max_sentences_per_speaker = 100  # AudioDataset 默認值
訓練樣本數 = 14 speakers × 100 = 1400
```

**現在** (Hybrid Loss):
```python
max_sentences_per_speaker = None  # 使用全部
訓練樣本數 = 14 speakers × 288 = 4032  (↑188%)
```

---

### 訓練配置

| 參數 | 值 | 說明 |
|-----|-----|------|
| **模型** | | |
| d_model | 512 | Transformer 維度 |
| nhead | 8 | Attention heads |
| num_layers | 6 | Transformer 層數 |
| dim_feedforward | 2048 | FFN 維度 |
| dropout | 0.1 | Dropout rate |
| **訓練** | | |
| batch_size | 8 | Batch size |
| num_epochs | 600 | 總 epochs |
| learning_rate | 1e-4 | 初始學習率 |
| optimizer | AdamW | 優化器 |
| scheduler | CosineAnnealing | LR 調度器 |
| **損失權重** | | |
| ce_weight | 1.0 | CrossEntropy 固定 |
| content_weight | 0.5 | Content 最大值 |
| embed_weight | 0.3 | Embedding 固定 |
| warmup_epochs | 50 | Content warmup |

---

### 預期改善

**與 Frozen Codebook 比較**:

| 指標 | Frozen Codebook | Hybrid Loss (預期) |
|------|----------------|-------------------|
| Train Acc | 66.78% | ~70% (更多數據) |
| Val Acc | 14.92% | **>30%** (混合損失) |
| 過擬合 | 嚴重 (51.9% gap) | **減輕** (<30% gap) |
| 音頻質量 | 未知 (無評估) | **改善** (embedding 約束) |
| 訓練樣本 | 1400 | 4032 (↑188%) |

**改善原因**:
1. **Content Consistency**: 學習語義表示，減少對語者特徵的過擬合
2. **Embedding L2**: 連續空間約束，更平滑的優化
3. **更多數據**: 4032 vs 1400 樣本，減少過擬合
4. **動態權重**: 早期學內容，後期學去噪

---

## 使用方式

### 1. 訓練腳本

```bash
cd /home/sbplab/ruizi/c_code/try

# 執行訓練
bash run_token_denoising_hybrid.sh
```

### 2. Python API

```python
from discrete_hybrid_loss import DiscreteHybridLoss

# 創建損失函數
criterion = DiscreteHybridLoss(
    codebook=codebook,      # (4096, 512)
    wavtokenizer=None,      # 不使用 spectral loss
    device='cuda',
    ce_weight=1.0,
    content_weight=0.5,
    embed_weight=0.3,
    warmup_epochs=50
)

# 計算損失
loss_dict = criterion(
    pred_logits=logits,          # (B, T, 4096)
    target_tokens=clean_tokens,  # (B, T)
    noisy_tokens=noisy_tokens,   # (B, T)
    content_ids=content_ids,     # (B,)
    current_epoch=epoch,
    total_epochs=600
)

# 使用總損失
total_loss = loss_dict['total_loss']
total_loss.backward()

# 查看各組件
print(f"CE Loss: {loss_dict['ce_loss']:.4f}")
print(f"Content Loss: {loss_dict['content_loss']:.4f}")
print(f"Embed Loss: {loss_dict['embed_loss']:.4f}")
print(f"Content Weight: {loss_dict['content_weight']:.3f}")
```

---

## 監控指標

### 訓練過程

**每個 batch 輸出**:
```
loss: 總損失
ce:   CrossEntropy 損失
cont: Content Consistency 損失
emb:  Embedding L2 損失
acc:  Token 準確率
cw:   Content weight (動態)
```

**每個 epoch 輸出**:
```
Epoch 10/600
  Train - Total Loss: 2.5134, CE: 2.1234, Content: 0.2345, Embed: 0.1555, Acc: 45.23%
  Val   - Total Loss: 3.2145, CE: 2.8123, Content: 0.2456, Embed: 0.1566, Acc: 28.45%
  Learning Rate: 9.85e-05
```

### 關鍵指標觀察

1. **Content Weight 衰減**:
   - Epoch 0: ~0.50
   - Epoch 50: ~0.25
   - Epoch 200: ~0.05
   - Epoch 600: ~0.01

2. **Validation Accuracy**:
   - 目標: 持續上升，不要停滯在 15%
   - 預期: Epoch 100 時達到 >25%

3. **Train/Val Gap**:
   - 目標: <30% (vs Frozen Codebook 的 51.9%)
   - 監控: Val Loss 不應該持續上升

4. **Embedding Loss**:
   - 應該持續下降
   - 代表模型在 embedding 空間學得更好

---

## 實驗假設與驗證

### 假設 1: Content Consistency 改善泛化
**假設**: 相同內容的 embeddings 相似 → 模型學到語義而非記憶
**驗證**: Val Acc 應該 >30% (vs Frozen Codebook 的 15%)

### 假設 2: Embedding L2 改善音頻質量
**假設**: Embedding 空間約束 → 即使 token 不完全正確，音頻仍可接受
**驗證**: 需要解碼音頻進行主觀評估

### 假設 3: 更多數據減少過擬合
**假設**: 4032 樣本 vs 1400 → Train/Val gap 縮小
**驗證**: Train/Val gap 應該 <30%

### 假設 4: 動態權重平衡學習
**假設**: 早期學內容，後期學去噪 → 更好的收斂
**驗證**: 觀察各損失組件的變化趨勢

---

## 檔案清單

### 核心檔案

1. **`discrete_hybrid_loss.py`** (425 行)
   - `DiscreteHybridLoss` 類別
   - 實現三種損失的計算
   - 動態權重調整

2. **`train_token_denoising_hybrid.py`** (471 行)
   - 完整訓練流程
   - 支持混合損失
   - 返回 content_id 的 collate_fn

3. **`run_token_denoising_hybrid.sh`**
   - 執行腳本
   - 設置 `max_sentences_per_speaker=None`
   - 配置混合損失權重

4. **`HYBRID_LOSS_DESIGN.md`** (本文件)
   - 設計理念說明
   - 使用方式文檔

---

## 後續計劃

### 短期 (實驗中)
- [ ] 運行完整 600 epochs
- [ ] 監控 Val Acc 是否突破 30%
- [ ] 觀察 Content Weight 衰減曲線
- [ ] 記錄各損失組件變化

### 中期 (實驗後)
- [ ] 解碼 tokens 為音頻，進行質量評估
- [ ] 比較 Frozen Codebook vs Hybrid Loss 的音頻質量
- [ ] 可視化 embedding 空間 (t-SNE)
- [ ] 分析 content_id 的聚類情況

### 長期 (優化)
- [ ] 嘗試不同的權重配置
- [ ] 可能加入 Spectral Loss (頻譜約束)
- [ ] 探索其他 Content Consistency 計算方式
- [ ] 考慮 Multi-stage Training (分階段訓練)

---

## 參考文獻

### 相關實驗

1. **Frozen Codebook** (2025-01-23)
   - 文檔: `FROZEN_CODEBOOK_ANALYSIS_20251023.md`
   - 結果: 嚴重過擬合 (Train 67%, Val 15%)
   - 問題: Token 準確度 ≠ 音頻質量

2. **ttt2.py** (連續特徵訓練)
   - 策略: 內容一致性 + L2 特徵損失
   - 成功經驗: 早期學內容，後期學去噪
   - 啟發: 相同句子應有相似表示

### 相關程式碼

- `ttdata.py`: AudioDataset，支持 `return_content_id=True`
- `models/wavtokenizer_transformer.py`: TokenDenoisingTransformer
- `decoder/pretrained.py`: WavTokenizer

---

## 總結

**問題**: Frozen Codebook 過擬合嚴重 (Train 67%, Val 15%)

**解決方案**: 混合損失 = CrossEntropy + Content Consistency + Embedding L2

**核心創新**:
1. 借鑑 ttt2.py 的內容一致性理念
2. 適配到離散 token 訓練
3. 動態權重調整 (早期學內容，後期學去噪)
4. 使用全部 288 句數據 (vs 之前 100 句)

**預期改善**:
- Val Acc: 15% → **>30%**
- Train/Val Gap: 52% → **<30%**
- 音頻質量: 未知 → **改善** (embedding 約束)

**實驗編號**: EXP20250123_HYBRID_LOSS  
**實驗狀態**: ⏳ 待執行  
**預計時長**: ~48 小時 (600 epochs)
