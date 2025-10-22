# 模型架構對比分析
**生成日期**: 2025-10-22  
**對比對象**: 現有模型 vs Frozen Codebook 模型

---

## 📊 核心差異總覽

| 特性 | 現有模型 | Frozen Codebook 模型 |
|------|----------|---------------------|
| **Codebook Embedding** | ✅ 可訓練 (2M 參數) | ❌ 完全凍結 (查表) |
| **Transformer 架構** | Encoder-Decoder | Encoder Only |
| **可訓練參數** | ~5-6M | ~3-4M |
| **損失函數** | CE + L2 + Coherence + Manifold | Cross-Entropy Only |
| **訓練複雜度** | 高 (多個損失項) | 低 (單一損失) |
| **記憶體佔用** | 較高 | 較低 |
| **預訓練知識** | 可能被破壞 | 完全保留 |

---

## 🔍 詳細對比

### 1. Embedding 層設計

#### 現有模型
```python
# 重新創建 Codebook Embedding (可訓練)
self.codebook_embedding = nn.Embedding(4096, 512)

# 從 WavTokenizer 初始化
with torch.no_grad():
    self.codebook_embedding.weight.copy_(wavtokenizer_codebook)

# ✅ 可以根據降噪任務微調
# ⚠️ 可能偏離原始聲學知識
```

#### Frozen Codebook 模型
```python
# 直接註冊為 buffer (不可訓練)
self.register_buffer('codebook', wavtokenizer_codebook)

# 查表操作 (無梯度)
embeddings = self.codebook[token_ids]

# ✅ 完全保留 WavTokenizer 知識
# ⚠️ 無法針對特定任務微調
```

---

### 2. Transformer 架構

#### 現有模型: Encoder-Decoder
```
┌─────────────────────────────────────────────────────────────┐
│           Encoder-Decoder 架構 (現有模型)                    │
└─────────────────────────────────────────────────────────────┘

Noisy Tokens              Target Tokens (Teacher Forcing)
  [2347, 891, ...]          [2351, 893, ...]
       │                           │
       │ Embedding (trainable)     │ Embedding (trainable)
       ▼                           ▼
  Noisy Embed                 Target Embed
  (B, T, 512)                 (B, T, 512)
       │                           │
       │                           │
       ▼                           │
┌──────────────────┐               │
│ Transformer      │               │
│ Encoder          │               │
│                  │               │
│ - Self Attention │               │
│ - Feed Forward   │               │
│ - Layer Norm     │               │
└──────────────────┘               │
       │                           │
       │ Encoder Output            │
       │ (B, T, 512)               │
       │                           │
       └───────────┐               │
                   ▼               │
            ┌──────────────────────┴──────┐
            │ Transformer Decoder         │
            │                             │
            │ - Masked Self Attention     │
            │ - Cross Attention           │
            │   (attend to encoder)       │
            │ - Feed Forward              │
            └─────────────────────────────┘
                       │
                       ▼
                Output Projection
                  (B, T, 4096)
                       │
                       ▼
                  Clean Tokens

⏱️ 推論: 需要 autoregressive decoding (逐個生成 token)
```

**特點**:
- ✅ 使用 target tokens 作為 decoder 輸入 (teacher forcing)
- ✅ 可以學習更複雜的序列依賴
- ⚠️ 推論時需要 autoregressive decoding (較慢)
- ⚠️ 更多參數 (encoder + decoder)

#### Frozen Codebook 模型: Encoder Only
```
┌─────────────────────────────────────────────────────────────┐
│              Encoder-Only 架構 (本模型)                      │
└─────────────────────────────────────────────────────────────┘

Noisy Tokens
  [2347, 891, 3102, 1456, ...]
       │
       │ Frozen Codebook Lookup ❄️
       │ (無梯度, 無參數)
       ▼
  Embeddings
  (B, T, 512)
       │
       │ + Positional Encoding
       ▼
┌──────────────────────────────┐
│ Transformer Encoder (6 層)   │
│                              │
│ ┌──────────────────────────┐ │
│ │ Layer 1:                 │ │
│ │ - Multi-Head Attention   │ │
│ │   (8 heads, parallel)    │ │
│ │ - Feed Forward (2048)    │ │
│ │ - Residual + LayerNorm   │ │
│ └──────────────────────────┘ │
│           ...                │
│ ┌──────────────────────────┐ │
│ │ Layer 6:                 │ │
│ │ - Multi-Head Attention   │ │
│ │ - Feed Forward           │ │
│ │ - Residual + LayerNorm   │ │
│ └──────────────────────────┘ │
└──────────────────────────────┘
       │
       │ Hidden States (B, T, 512)
       ▼
  Linear Projection
  (512 → 4096)
       │
       ▼
  Output Logits (B, T, 4096)
       │
       │ Argmax
       ▼
  Clean Tokens
  [2351, 893, 3105, 1459, ...]

⚡ 推論: 並行處理整個序列 (一次完成)
```

**特點**:
- ✅ 更簡單的架構
- ✅ 推論速度快 (並行處理整個序列)
- ✅ 類比 BERT 的 masked language modeling
- ⚠️ 無法利用 target 信息

---

### 3. 損失函數

#### 現有模型: 混合損失
```python
total_loss = (
    ce_weight * ce_loss +              # 15.0 * CE
    l2_embed_weight * l2_loss +        # 1.5 * L2
    coherence_weight * coherence +     # 0.2 * Coherence
    manifold_weight * manifold         # 0.1 * Manifold
)
```

**優點**:
- ✅ CE Loss: 確保 token 預測準確
- ✅ L2 Loss: 保持聲學相似性
- ✅ Coherence: 時間平滑
- ✅ Manifold: 正則化

**缺點**:
- ⚠️ 需要精心調整權重
- ⚠️ 訓練不穩定 (多個損失項競爭)
- ⚠️ 計算成本高

#### Frozen Codebook 模型: 純 CE Loss
```python
loss = F.cross_entropy(logits, target_tokens)
```

**優點**:
- ✅ 簡單直接
- ✅ 穩定訓練
- ✅ 無需權重調整
- ✅ 計算高效

**缺點**:
- ⚠️ 無聲學約束 (依賴 frozen codebook)
- ⚠️ 無時間平滑約束

---

### 4. 訓練流程

#### 現有模型
```python
# Step 1: Encode to tokens
noisy_tokens = wavtokenizer.encode(noisy_audio)
clean_tokens = wavtokenizer.encode(clean_audio)

# Step 2: Get embeddings (可訓練)
noisy_embed = codebook_embedding(noisy_tokens)
clean_embed = codebook_embedding(clean_tokens)

# Step 3: Transformer
logits = transformer_encoder_decoder(noisy_embed, clean_embed)

# Step 4: Multiple losses
ce_loss = cross_entropy(logits, clean_tokens)
l2_loss = mse(predicted_embed, clean_embed)
coherence_loss = temporal_smoothness(predicted_embed)
manifold_loss = distance_to_input(predicted_embed, noisy_embed)

total_loss = combine_losses(...)
```

#### Frozen Codebook 模型
```python
# Step 1: Encode to tokens
noisy_tokens = wavtokenizer.encode(noisy_audio)
clean_tokens = wavtokenizer.encode(clean_audio)

# Step 2: Frozen lookup (無梯度)
embeddings = frozen_codebook[noisy_tokens]

# Step 3: Transformer Encoder
logits = transformer_encoder(embeddings)

# Step 4: Single loss
loss = cross_entropy(logits, clean_tokens)
```

---

## 🎯 設計哲學對比

### 現有模型: **微調 Embedding**

**核心假設**:
> WavTokenizer 的 Codebook 雖然好，但不是為降噪設計的。
> 我們需要針對降噪任務重新學習 token → embedding 的映射。

**策略**:
1. 從 WavTokenizer Codebook 初始化
2. 允許梯度更新
3. 使用多個損失函數引導學習方向
4. 期望學到更適合降噪的 embedding

**風險**:
- Catastrophic Forgetting (遺忘原始聲學知識)
- Overfitting (過度擬合降噪數據)
- Training Instability (多損失競爭)

---

### Frozen Codebook 模型: **保留預訓練知識**

**核心假設**:
> WavTokenizer 的 Codebook 已經是最佳的音訊表示。
> 降噪可以純粹在 token 空間進行，無需修改 embedding。

**策略**:
1. 完全凍結 WavTokenizer Codebook
2. 只學習 token → token 的映射
3. 類比機器翻譯的 frozen pretrained embedding
4. 讓 Transformer 學習序列級的降噪模式

**優勢**:
- Preserve Knowledge (保留聲學知識)
- Better Generalization (更好泛化)
- Faster Training (更少參數)
- Simpler Training (單一損失)

---

## 🔬 實驗問題

### 關鍵問題 1: Frozen vs Trainable Embedding
- **Frozen 是否足夠?** Codebook 是否已經涵蓋降噪所需的所有聲學模式?
- **Trainable 是否必要?** 微調 embedding 是否真的能提升降噪效果?
- **如何測試?** 比較兩者在相同數據集上的 token 準確率、音訊質量

### 關鍵問題 2: 損失函數設計
- **混合損失是否必要?** L2、Coherence、Manifold 是否顯著改善效果?
- **純 CE 是否足夠?** 如果 codebook frozen，CE 能否保證聲學質量?
- **如何測試?** 比較訓練速度、穩定性、最終效果

### 關鍵問題 3: Encoder vs Encoder-Decoder
- **Decoder 的價值?** Teacher forcing 是否顯著幫助?
- **Encoder-only 的限制?** 無法利用 target 信息是否是致命缺陷?
- **如何測試?** 比較模型容量、推論速度、降噪質量

---

## 📈 預期結果分析

### 情境 1: Frozen Codebook 成功 (Token Acc > 60%)
**結論**:
- ✅ WavTokenizer Codebook 確實是最佳表示
- ✅ Token-level 降噪有效
- ✅ 無需重新訓練 embedding

**影響**:
- 未來模型可以更輕量
- 訓練成本大幅降低
- 部署更容易

### 情境 2: Frozen Codebook 失敗 (Token Acc < 30%)
**可能原因**:
- ⚠️ Codebook 不適配降噪任務
- ⚠️ 需要 embedding 微調以學習噪音模式
- ⚠️ Encoder-only 架構不足

**後續策略**:
- 嘗試部分微調 (fine-tune last few codebook entries)
- 引入 adapter layers
- 回歸 trainable embedding

### 情境 3: 兩者效果相當
**結論**:
- ✅ Frozen Codebook 足夠 + 更高效
- ⚠️ Trainable Embedding 更靈活但不必要

**建議**:
- 優先使用 Frozen Codebook (更簡單)
- 只在特殊情況下微調

---

## 🚀 實驗計劃

### Phase 1: 基礎訓練 (Epochs 1-100)
**目標**: 驗證基本可行性
- Token Accuracy > 10%
- Loss 下降趨勢明確
- 無明顯過擬合

### Phase 2: 深度訓練 (Epochs 100-500)
**目標**: 達到實用水平
- Token Accuracy > 40%
- 音訊可辨識
- 頻譜連續性改善

### Phase 3: 對比實驗 (Epochs 500-1000)
**目標**: 與現有模型比較
- Token Accuracy 差異
- 音訊質量差異 (PESQ, STOI)
- 訓練效率差異

---

## 📝 評估指標

### 定量指標
| 指標 | 現有模型 | Frozen Codebook | 勝者 |
|------|----------|-----------------|------|
| Token Accuracy | ? | ? | TBD |
| PESQ | ? | ? | TBD |
| STOI | ? | ? | TBD |
| 訓練時間/epoch | ? | ? | TBD |
| 收斂速度 (epochs) | ? | ? | TBD |
| 可訓練參數 | ~5-6M | ~3-4M | Frozen ✅ |

### 定性指標
- 聽覺質量
- 頻譜連續性
- 語者特徵保留
- 噪音抑制效果

---

## 💡 總結

### Frozen Codebook 模型的創新點
1. **完全凍結預訓練 Codebook** (保留聲學知識)
2. **Token-to-Token 映射** (類比機器翻譯)
3. **更輕量的架構** (Encoder-only)
4. **更簡單的訓練** (單一損失函數)

### 需要驗證的假設
1. WavTokenizer Codebook 是否足以支持降噪?
2. Token-level 映射是否保留足夠的聲學信息?
3. Frozen embedding 是否比 trainable 更好?

### 預期貢獻
- 提供更高效的降噪方案
- 驗證 frozen pretrained embedding 的有效性
- 探索 token-level 音訊處理的可能性

---

**生成函式**: MODEL_COMPARISON_ANALYSIS.md  
**相關實驗**: frozen_codebook_* vs large_tokenloss_*  
**建議閱讀順序**: 
1. README_FROZEN_CODEBOOK.md (了解新模型)
2. 本文檔 (理解差異)
3. TOKEN_DENOISING_TRANSFORMER_EXPLANATION.md (深入細節)
