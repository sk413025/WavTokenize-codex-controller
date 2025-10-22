# Token Denoising Transformer with Frozen Codebook

**生成日期**: 2025-10-22  
**實驗類型**: 音訊降噪 (Token-level)  
**核心創新**: 完全凍結 WavTokenizer Codebook

---

## 📊 與現有模型的關鍵差異

### **現有模型** (`wavtokenizer_transformer_denoising.py`)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    現有模型架構 (Trainable Embedding)                │
└─────────────────────────────────────────────────────────────────────┘

    Noisy Audio (1, 24000)
         │
         │ [WavTokenizer Encoder] ❄️ FROZEN
         ▼
    Noisy Tokens (B, T)  例: (4, 75)
    [2347, 891, 3102, 1456, ...]
         │
         │ [Codebook Embedding Layer] 🔥 TRAINABLE (2M 參數)
         │  - nn.Embedding(4096, 512)
         │  - 從 WavTokenizer 初始化
         │  - 訓練時會更新 ⚠️
         ▼
    Embeddings (B, T, 512)
         │
         │ [Transformer Encoder-Decoder] 🔥 TRAINABLE
         │  - Encoder: 處理 noisy embeddings
         │  - Decoder: 使用 target tokens (teacher forcing)
         │  - Multi-head Attention
         ▼
    Output Logits (B, T, 4096)
         │
         │ [Argmax]
         ▼
    Clean Tokens (B, T)
    [2351, 893, 3105, 1459, ...]
         │
         │ [Codebook Lookup] 🔥 TRAINABLE embedding
         ▼
    Clean Embeddings (B, T, 512)
         │
         │ [WavTokenizer Decoder] ❄️ FROZEN
         ▼
    Denoised Audio (1, 24000)

Loss = 15.0*CE + 1.5*L2 + 0.2*Coherence + 0.1*Manifold
```

**特點**:
- ✅ 可訓練 Codebook Embedding (4096 × 512 = 2M 參數)
- ✅ 使用 Encoder-Decoder 架構
- ✅ 混合損失函數 (CE + L2 + Coherence + Manifold)
- ⚠️ 可能破壞 WavTokenizer 預訓練的聲學知識
- ⚠️ 較多參數 (~5-6M)

---

### **本模型** (`token_denoising_transformer.py`)

```
┌─────────────────────────────────────────────────────────────────────┐
│                   本模型架構 (Frozen Codebook)                       │
└─────────────────────────────────────────────────────────────────────┘

    Noisy Audio (1, 24000)
         │
         │ [WavTokenizer Encoder] ❄️ FROZEN
         ▼
    Noisy Tokens (B, T)  例: (8, 75)
    [2347, 891, 3102, 1456, ...]
         │
         │ [Frozen Codebook Lookup] ❄️ FROZEN (0 參數)
         │  - register_buffer('codebook', ...)
         │  - 直接查表: embeddings = codebook[token_ids]
         │  - 無梯度流動 ✅
         ▼
    Embeddings (B, T, 512)
         │
         │ + [Positional Encoding]
         ▼
    Embeddings (B, T, 512)
         │
         │ [Transformer Encoder ONLY] 🔥 TRAINABLE (~3M 參數)
         │  - 6 層 TransformerEncoderLayer
         │  - 8 個注意力頭
         │  - FFN dim: 2048
         │  - 並行處理整個序列 ⚡
         ▼
    Hidden States (B, T, 512)
         │
         │ [Linear Projection] 🔥 TRAINABLE
         │  - nn.Linear(512, 4096)
         ▼
    Output Logits (B, T, 4096)
         │
         │ [Argmax]
         ▼
    Clean Tokens (B, T)
    [2351, 893, 3105, 1459, ...]
         │
         │ [Frozen Codebook Lookup] ❄️ FROZEN
         ▼
    Clean Embeddings (B, T, 512)
         │
         │ [WavTokenizer Decoder] ❄️ FROZEN
         ▼
    Denoised Audio (1, 24000)

Loss = CrossEntropy ONLY (簡單！)
```

**特點**:
- ✅ **完全凍結 Codebook** (保留 WavTokenizer 聲學知識)
- ✅ 只訓練 Transformer Encoder (~3-4M 參數)
- ✅ 類比機器翻譯的 Frozen Pretrained Embedding
- ✅ 更輕量、更快收斂
- ✅ Token-to-Token 映射 (無需重建音訊特徵)

---

## 🎯 核心設計理念

### 1. **Codebook 是最佳表示，無需重新學習**

WavTokenizer 的 Codebook 經過大規模音訊數據訓練：
- 4096 個 512-D 向量
- 涵蓋各種聲學模式
- 已經是音訊的最佳離散表示

**為什麼要凍結?**
- 重新訓練可能破壞這些知識
- 降噪任務可以直接在 token 空間進行

### 2. **類比機器翻譯**

```
機器翻譯:
  英文詞 IDs → [Frozen Pretrained Embedding] → Transformer → 中文詞 IDs

Token Denoising:
  Noisy Token IDs → [Frozen Codebook] → Transformer → Clean Token IDs
```

### 3. **Token-level 學習**

不需要重建連續的音訊特徵，只需要學習：
- Noisy Token → Clean Token 的映射關係
- Token 序列的時間依賴性
- 上下文信息的利用

---

## 🚀 使用方式

### 完整訓練流程視覺化

```
┌─────────────────────────────────────────────────────────────────────┐
│                     訓練流程 (Epoch 循環)                            │
└─────────────────────────────────────────────────────────────────────┘

For each Epoch:

    ┌───────────────────────────────────────────┐
    │        訓練階段 (Training)                 │
    └───────────────────────────────────────────┘
    
    For each Batch:
    
        1️⃣ 載入音訊對
        ──────────────
        noisy_audio (B, 24000)
        clean_audio (B, 24000)
             │
             │ [WavTokenizer Encoder] ❄️ FROZEN
             ▼
        noisy_tokens (B, T)  例: [2347, 891, 3102, ...]
        clean_tokens (B, T)  例: [2351, 893, 3105, ...]
        
        2️⃣ Frozen Codebook Lookup
        ──────────────────────────
        embeddings = codebook[noisy_tokens]  # (B, T, 512)
                     │
                     │ ❄️ 無梯度！
                     │ ✅ Codebook 保持不變
                     ▼
        
        3️⃣ Transformer Forward
        ───────────────────────
        embeddings → [Positional Encoding]
                  → [Transformer Encoder] 🔥
                  → [Linear Projection] 🔥
                  → logits (B, T, 4096)
        
        4️⃣ 計算損失
        ──────────
        loss = CrossEntropyLoss(logits, clean_tokens)
        
        5️⃣ 反向傳播
        ──────────
        loss.backward()
        optimizer.step()
        
        ✅ 只有 Transformer + Projection 被更新
        ❄️ Codebook 保持凍結
    
    ┌───────────────────────────────────────────┐
    │        驗證階段 (Validation)               │
    └───────────────────────────────────────────┘
    
    For each Val Batch:
    
        noisy_tokens → [Model] → predicted_tokens
        
        Token Accuracy = (predicted == clean).mean()
        Val Loss = CrossEntropyLoss(predicted, clean)
    
    ┌───────────────────────────────────────────┐
    │        保存檢查點                          │
    └───────────────────────────────────────────┘
    
    If val_loss < best_val_loss:
        保存 best_model.pt
    
    If epoch % 100 == 0:
        保存 checkpoint_epoch_XXX.pt
    
    If epoch % 10 == 0:
        繪製 training_history.png

End For
```

### 訓練

```bash
cd /home/sbplab/ruizi/c_code/try
bash run_token_denoising_frozen_codebook.sh
```

### 監控訓練

```bash
tail -f ../logs/token_denoising_frozen_codebook_*.log
```

### 推論 (降噪單個音訊)

```python
from token_denoising_transformer import WavTokenizerTransformerDenoiser

# 創建 denoiser
denoiser = WavTokenizerTransformerDenoiser(
    wavtokenizer_config_path="config/wavtokenizer_...",
    transformer_model_path="results/.../best_model.pt",
    device='cuda'
)

# 降噪
denoiser.denoise('noisy.wav', 'denoised.wav')

# 分析 Token 變化
denoiser.compare_tokens('noisy.wav', 'clean_ground_truth.wav')
```

### 推論流程視覺化

```
┌─────────────────────────────────────────────────────────────────────┐
│                     推論流程 (Inference)                             │
└─────────────────────────────────────────────────────────────────────┘

noisy.wav (24000 samples)
     │
     │ [WavTokenizer Encoder] ❄️
     ▼
noisy_tokens (1, T)  例: [2347, 891, 3102, 1456, ...]
     │
     │ [Frozen Codebook Lookup] ❄️
     ▼
embeddings (1, T, 512)
     │
     │ [Trained Transformer] 🎓
     ▼
predicted_logits (1, T, 4096)
     │
     │ [Argmax]
     ▼
clean_tokens (1, T)  例: [2351, 893, 3105, 1459, ...]
     │
     │ [Frozen Codebook Lookup] ❄️
     ▼
clean_embeddings (1, T, 512)
     │
     │ [WavTokenizer Decoder] ❄️
     ▼
denoised.wav (24000 samples)

Token 變化率: (noisy_tokens != clean_tokens).mean()
例: 15% 的 tokens 被修正
```

---

## 📈 預期效果

### **優勢**

1. **更快收斂**
   - 參數更少 (~3-4M vs ~5-6M)
   - 不需要重新學習 embedding
   - 只專注於 token 映射

2. **更穩定訓練**
   - Codebook 凍結 → embedding 不漂移
   - 減少過擬合風險
   - 更好的泛化能力

3. **更好的泛化**
   - 保留 WavTokenizer 的預訓練知識
   - 對未見過的語者/噪音更魯棒

### **預期指標** (200 epochs)

| 指標 | 目標值 |
|------|--------|
| Token 準確率 | > 60% |
| Token 變化率 | 10-20% |
| 訓練時間/epoch | < 5 分鐘 |
| 收斂速度 | < 100 epochs |

---

## 🔬 實驗假設

### **假設 1**: Codebook 已經是最佳表示
- ✅ WavTokenizer 在大規模數據上預訓練
- ✅ Codebook 涵蓋豐富的聲學模式
- ✅ 不需要針對降噪任務重新調整

### **假設 2**: Token 映射足以實現降噪
- ✅ 噪音和乾淨音訊在 token 空間有明確對應關係
- ✅ Transformer 可以學習這種映射
- ✅ 不需要在連續特徵空間操作

### **假設 3**: 凍結 > 微調
- ✅ 凍結保留預訓練知識
- ✅ 微調可能導致知識遺忘 (catastrophic forgetting)
- ✅ 降噪任務與 WavTokenizer 預訓練任務相似

---

## 📚 相關文檔

- [`TOKEN_DENOISING_TRANSFORMER_EXPLANATION.md`](./TOKEN_DENOISING_TRANSFORMER_EXPLANATION.md): 詳細架構說明
- [`TOKEN_RELATIONSHIP_EXPLANATION (1).md`](./TOKEN_RELATIONSHIP_EXPLANATION%20(1).md): Token-Codebook 因果關係
- [`token_denoising_transformer.py`](./token_denoising_transformer.py): 模型實現
- [`train_token_denoising.py`](./train_token_denoising.py): 訓練腳本

---

## 🔧 技術細節

### 模型架構

```python
TokenDenoisingTransformer(
    codebook=wavtokenizer.codebook,  # (4096, 512) - 凍結
    d_model=512,                      # 與 codebook 維度一致
    nhead=8,                          # 8 個注意力頭
    num_layers=6,                     # 6 層 Transformer Encoder
    dim_feedforward=2048,             # 前饋網絡維度
    dropout=0.1
)
```

### 前向傳播

```python
# Step 1: Frozen Codebook Lookup (不計算梯度)
embeddings = codebook[noisy_token_ids]  # (B, T, 512)

# Step 2: Positional Encoding
embeddings = pos_encoding(embeddings)

# Step 3: Transformer Encoding
hidden = transformer_encoder(embeddings)  # (B, T, 512)

# Step 4: Project to Vocabulary
logits = output_proj(hidden)  # (B, T, 4096)

# Step 5: Predict Clean Tokens
clean_token_ids = logits.argmax(dim=-1)  # (B, T)
```

### 損失函數

```python
# Cross-Entropy Loss (Token-level)
loss = F.cross_entropy(
    logits.reshape(B*T, 4096),
    clean_token_ids.reshape(B*T)
)
```

---

## 📊 訓練配置

```bash
# 模型參數
--d_model 512              # 與 Codebook 維度一致
--nhead 8                  # 8 個注意力頭
--num_layers 6             # 6 層 Encoder
--dim_feedforward 2048     # 前饋網絡維度
--dropout 0.1              # Dropout 率

# 訓練參數
--batch_size 8             # 批次大小
--num_epochs 1000          # 訓練輪數
--learning_rate 1e-4       # 學習率
--weight_decay 0.01        # 權重衰減
--save_every 100           # 每 100 epochs 保存

# 數據分割
--val_speakers girl9 girl10 boy7 boy8
--train_speakers boy1 boy3 boy4 boy5 boy6 boy9 boy10 girl2 girl3 girl4 girl6 girl7 girl8 girl11
```

---

## ⚠️ 注意事項

1. **Codebook 必須完全凍結**
   - 檢查: `assert not model.codebook.requires_grad`
   - 避免意外更新

2. **d_model 必須等於 Codebook 維度**
   - WavTokenizer Codebook: 512-D
   - 否則需要額外的 projection layer

3. **Token IDs 範圍**
   - 有效範圍: [0, 4095]
   - 檢查數據是否超出範圍

4. **記憶體優化**
   - Frozen Codebook 不佔用梯度記憶體
   - 可以使用較大的 batch size

---

## 🎉 預期貢獻

1. **驗證 Frozen Codebook 的有效性**
   - 是否足以支持降噪任務?
   - 與可訓練 embedding 的差異?

2. **提供更輕量的降噪方案**
   - 更少參數
   - 更快訓練
   - 更好部署

3. **探索 Token-level 降噪**
   - 離散化是否損失重要信息?
   - Token 映射的可解釋性?

---

**生成函式**: README_FROZEN_CODEBOOK.md  
**實驗編號**: frozen_codebook_YYYYMMDD_HHMMSS  
**相關文件**: try/ 資料夾內所有文件
