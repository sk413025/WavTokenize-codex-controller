# WavTokenizer-Transformer Denoising 模型架構詳解

## 📋 目錄
1. [整體架構概覽](#整體架構概覽)
2. [模型組件詳解](#模型組件詳解)
3. [數據流程](#數據流程)
4. [Loss Function 作用位置](#loss-function-作用位置)
5. [訓練與推理流程](#訓練與推理流程)
6. [關鍵創新點](#關鍵創新點)

---

## 🏗️ 整體架構概覽

### 完整端到端降噪流程（訓練模式）

```
┌═══════════════════════════════════════════════════════════════════════════════════════┐
║                         WavTokenizer-Transformer 降噪系統                              ║
║                              完整訓練流程架構圖                                         ║
└═══════════════════════════════════════════════════════════════════════════════════════┘

                    ┌─────────────────┐         ┌─────────────────┐
                    │  Noisy Audio    │         │  Clean Audio    │
                    │  [B, 1, 81920]  │         │  [B, 1, 81920]  │
                    │  (~3.4s@24kHz)  │         │  (~3.4s@24kHz)  │
                    └────────┬────────┘         └────────┬────────┘
                             │                           │
                             │                           │
          ╔══════════════════▼═══════════════╗          │
          ║  WavTokenizer Encoder (凍結)     ║          │
          ║  ✗ 參數不訓練                     ║          │
          ║  ✓ VQ-VAE 預訓練模型              ║          │
          ║  ✓ 連續→離散轉換                  ║          │
          ╚══════════════════╤═══════════════╝          │
                             │                           │
                             │                           │
                    ┌────────▼────────┐                 │
                    │  Noisy Tokens   │                 │
                    │  [B, 75]        │                 │
                    │  IDs: 0-4095    │                 │
                    └────────┬────────┘                 │
                             │                           │
                             │         ╔═════════════════▼════════════════╗
                             │         ║  WavTokenizer Encoder (凍結)     ║
                             │         ║  同上，處理 clean audio            ║
                             │         ╚═════════════════╤════════════════╝
                             │                           │
                             │                  ┌────────▼────────┐
                             │                  │  Clean Tokens   │
                             │                  │  [B, 75]        │
                             │                  │  IDs: 0-4095    │
                             │                  └────────┬────────┘
                             │                           │
        ┌────────────────────┴────────────┐             │
        │  構建 Encoder 輸入序列           │             │
        │  input_tokens =                 │             │
        │    [noisy_tokens, EOS]          │             │
        │  [B, 76]                        │             │
        └────────────┬────────────────────┘             │
                     │                                  │
                     │            ┌─────────────────────┴──────────────┐
                     │            │  構建 Decoder 輸入序列              │
                     │            │  decoder_input = [SOS, clean_tokens]│
                     │            │  [B, 76]                            │
                     │            └────────────┬────────────────────────┘
                     │                         │
                     │            ┌────────────┴─────────────────────┐
                     │            │  構建目標序列                     │
                     │            │  target_tokens =                 │
                     │            │    [clean_tokens, EOS]           │
                     │            │  [B, 76]                         │
                     │            └────────────┬─────────────────────┘
                     │                         │
    ╔════════════════▼═════════════╗          │
    ║  Token Embedding (混合策略)   ║          │
    ║  ┌────────────────────────┐  ║          │
    ║  │ Codebook Tokens        │  ║          │
    ║  │ (0-4095)               │  ║          │
    ║  │ ↓                      │  ║          │
    ║  │ nn.Embedding           │  ║          │
    ║  │   .from_pretrained()   │  ║          │
    ║  │ [4096, 512] ❄️ 凍結   │  ║          │
    ║  └────────────────────────┘  ║          │
    ║  ┌────────────────────────┐  ║          │
    ║  │ Special Tokens         │  ║          │
    ║  │ (4096-4098)            │  ║          │
    ║  │ ↓                      │  ║          │
    ║  │ nn.Embedding(3, 512)   │  ║          │
    ║  │ 🔥 可訓練              │  ║          │
    ║  └────────────────────────┘  ║          │
    ╚════════════════╤═════════════╝          │
                     │                         │
            ┌────────▼────────┐       ╔════════▼════════╗
            │  src_emb        │       ║  tgt_emb        ║
            │  [B, 76, 512]   │       ║  [B, 76, 512]   ║
            └────────┬────────┘       ╚════════╤════════╝
                     │                         │
    ╔════════════════▼═════════════╗  ╔════════▼════════╗
    ║  Embedding Projection        ║  ║  Embedding      ║
    ║  Linear(512 → 256)           ║  ║  Projection     ║
    ║  🔥 可訓練                   ║  ║  (同左)         ║
    ╚════════════════╤═════════════╝  ╚════════╤════════╝
                     │                         │
            ┌────────▼────────┐       ┌────────▼────────┐
            │  [B, 76, 256]   │       │  [B, 76, 256]   │
            │  × √256         │       │  × √256         │
            │  + pos_encoding │       │  + pos_encoding │
            └────────┬────────┘       └────────┬────────┘
                     │                         │
                     │                         │
    ╔════════════════▼═════════════════════════▼═════════════╗
    ║           Transformer (核心可訓練模型)                  ║
    ║  ┌───────────────────────────────────────────────────┐ ║
    ║  │  Encoder (處理 noisy tokens)                      │ ║
    ║  │  ┌─────────────────────────────────────────────┐  │ ║
    ║  │  │ Layer 1: Multi-Head Self-Attention (8 heads)│  │ ║
    ║  │  │          + Feed Forward (FFN dim=1024)      │  │ ║
    ║  │  ├─────────────────────────────────────────────┤  │ ║
    ║  │  │ Layer 2: Multi-Head Self-Attention (8 heads)│  │ ║
    ║  │  │          + Feed Forward (FFN dim=1024)      │  │ ║
    ║  │  ├─────────────────────────────────────────────┤  │ ║
    ║  │  │ Layer 3: Multi-Head Self-Attention (8 heads)│  │ ║
    ║  │  │          + Feed Forward (FFN dim=1024)      │  │ ║
    ║  │  ├─────────────────────────────────────────────┤  │ ║
    ║  │  │ Layer 4: Multi-Head Self-Attention (8 heads)│  │ ║
    ║  │  │          + Feed Forward (FFN dim=1024)      │  │ ║
    ║  │  └─────────────────────────────────────────────┘  │ ║
    ║  │  Output: Encoder Memory [B, 76, 256]             │ ║
    ║  └──────────────────────┬────────────────────────────┘ ║
    ║                         │                              ║
    ║  ┌──────────────────────▼───────────────────────────┐  ║
    ║  │  Decoder (生成 clean tokens)                     │  ║
    ║  │  ┌─────────────────────────────────────────────┐ │  ║
    ║  │  │ Layer 1:                                    │ │  ║
    ║  │  │   - Masked Self-Attention (Causal, 8 heads) │ │  ║
    ║  │  │   - Cross-Attention (attend to Encoder)     │ │  ║
    ║  │  │   - Feed Forward (FFN dim=1024)             │ │  ║
    ║  │  ├─────────────────────────────────────────────┤ │  ║
    ║  │  │ Layer 2:                                    │ │  ║
    ║  │  │   - Masked Self-Attention (Causal, 8 heads) │ │  ║
    ║  │  │   - Cross-Attention (attend to Encoder)     │ │  ║
    ║  │  │   - Feed Forward (FFN dim=1024)             │ │  ║
    ║  │  ├─────────────────────────────────────────────┤ │  ║
    ║  │  │ Layer 3:                                    │ │  ║
    ║  │  │   - Masked Self-Attention (Causal, 8 heads) │ │  ║
    ║  │  │   - Cross-Attention (attend to Encoder)     │ │  ║
    ║  │  │   - Feed Forward (FFN dim=1024)             │ │  ║
    ║  │  ├─────────────────────────────────────────────┤ │  ║
    ║  │  │ Layer 4:                                    │ │  ║
    ║  │  │   - Masked Self-Attention (Causal, 8 heads) │ │  ║
    ║  │  │   - Cross-Attention (attend to Encoder)     │ │  ║
    ║  │  │   - Feed Forward (FFN dim=1024)             │ │  ║
    ║  │  └─────────────────────────────────────────────┘ │  ║
    ║  │  Output: Decoder Output [B, 76, 256]            │  ║
    ║  └─────────────────────┬────────────────────────────┘  ║
    ║                        │                               ║
    ╚════════════════════════╤═══════════════════════════════╝
                             │
                    ┌────────▼────────┐
                    │  Decoder Output │
                    │  [B, 76, 256]   │
                    └────────┬────────┘
                             │
            ╔════════════════▼═════════════╗
            ║  Output Projection           ║
            ║  Linear(256 → 4096)          ║
            ║  🔥 可訓練                   ║
            ╚════════════════╤═════════════╝
                             │
                    ┌────────▼────────┐
                    │     Logits      │
                    │  [B, 76, 4096]  │
                    │  (每個位置預測   │
                    │   4096個token    │
                    │   的機率分佈)     │
                    └────────┬────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
        │                                         │
╔═══════▼════════╗                    ╔═══════════▼═══════════╗
║  訓練模式       ║                    ║    推理模式            ║
║                ║                    ║                       ║
║  Loss 計算     ║                    ║  argmax(logits)       ║
╚═══════╤════════╝                    ╚═══════════╤═══════════╝
        │                                         │
        │                                ┌────────▼────────┐
        │                                │ Predicted Tokens│
        │                                │  [B, 76]        │
        │                                └────────┬────────┘
        │                                         │
        │                         ╔═══════════════▼═══════════════╗
        │                         ║  WavTokenizer Decoder (凍結)  ║
        │                         ║  離散→連續轉換                 ║
        │                         ╚═══════════════╤═══════════════╝
        │                                         │
        │                                ┌────────▼────────┐
        │                                │ Denoised Audio  │
        │                                │ [B, 1, 81920]   │
        │                                └─────────────────┘
        │
        │  ┌─────────────────────────────────────────────────┐
        │  │  Loss Function 選項                             │
        │  ├─────────────────────────────────────────────────┤
        │  │                                                 │
        │  │  ▶ 模式 1: CrossEntropy Loss (預設)             │
        │  │    ┌────────────────────────────────────────┐   │
        │  │    │ loss = CrossEntropy(                   │   │
        │  │    │   logits.view(-1, 4096),               │   │
        │  │    │   target_tokens.view(-1)               │   │
        │  │    │ )                                      │   │
        │  │    │                                        │   │
        │  │    │ 特點:                                  │   │
        │  │    │ ✓ 簡單快速                             │   │
        │  │    │ ✓ 直接優化 token 預測準確度            │   │
        │  │    │ ✓ 標準做法                             │   │
        │  │    └────────────────────────────────────────┘   │
        │  │                                                 │
        │  │  ▶ 模式 2: Token Loss System                    │
        │  │    (--use_token_loss)                           │
        │  │    ┌────────────────────────────────────────┐   │
        │  │    │ pred_tokens = argmax(logits)           │   │
        │  │    │ pred_emb = Embedding(pred_tokens)      │   │
        │  │    │ target_emb = Embedding(target_tokens)  │   │
        │  │    │                                        │   │
        │  │    │ loss = 0.3×L2_loss                     │   │
        │  │    │      + 0.4×consistency_loss            │   │
        │  │    │      + 0.1×manifold_loss               │   │
        │  │    │      + 0.1×norm_loss                   │   │
        │  │    │      + 0.1×coherence_loss              │   │
        │  │    │                                        │   │
        │  │    │ 特點:                                  │   │
        │  │    │ ✓ 考慮 embedding 空間幾何結構          │   │
        │  │    │ ✓ 多組件損失                           │   │
        │  │    │ ✓ 更符合音頻降噪本質                   │   │
        │  │    └────────────────────────────────────────┘   │
        │  │                                                 │
        │  └─────────────────────────────────────────────────┘
        │
        ▼
  ┌──────────────┐
  │ 反向傳播      │
  │              │
  │ loss.backward()
  │ optimizer.step()
  └──────────────┘


┌═══════════════════════════════════════════════════════════════════════════┐
║                          參數訓練狀態總覽 (Large 配置)                     ║
├═══════════════════════════════════════════════════════════════════════════┤
║  組件                              │ 參數量      │ 訓練狀態                ║
├───────────────────────────────────┼────────────┼────────────────────────║
║  WavTokenizer Encoder/Decoder     │ ~50M       │ ❄️  凍結 (freeze=True) ║
║  Codebook Embedding (4096×512)    │ ~2.1M      │ ❄️  凍結               ║
║  Special Token Embedding (3×512)  │ ~1.5K      │ 🔥 可訓練              ║
║  Embedding Projection (512→256)   │ ~131K      │ 🔥 可訓練              ║
║  Positional Encoding              │ 400×256    │ ❄️  固定 (不訓練)      ║
║  Transformer (4+4 layers, 8 heads)│ ~3-4M      │ 🔥 可訓練              ║
║  Output Projection (256→4096)     │ ~1.05M     │ 🔥 可訓練              ║
├───────────────────────────────────┼────────────┼────────────────────────║
║  總參數                            │ ~56M       │                        ║
║  可訓練參數                        │ ~4-5M      │ ← 主要優化這部分       ║
└═══════════════════════════════════════════════════════════════════════════┘
```

### 推理模式簡化流程

```
┌═══════════════════════════════════════════════════════════════════════════┐
║                        推理模式 (Inference Mode)                          ║
└═══════════════════════════════════════════════════════════════════════════┘

                    ┌─────────────────┐
                    │  Noisy Audio    │
                    │  [B, 1, 81920]  │
                    └────────┬────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ WavTokenizer Encoder │ ❄️
                  └──────────┬───────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ Noisy Tokens   │
                    │ [B, 75]        │
                    └────────┬───────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Token Embedding      │
                  │ + Projection         │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Transformer Encoder  │ 🔥
                  │ (只用 Encoder)        │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Output Projection    │ 🔥
                  └──────────┬───────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ Logits         │
                    │ [B, L, 4096]   │
                    └────────┬───────┘
                             │
                             ▼
                      argmax(logits)
                             │
                             ▼
                  ┌────────────────────┐
                  │ Predicted Tokens   │
                  │ [B, L]             │
                  └────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ WavTokenizer Decoder │ ❄️
                  └──────────┬───────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ Denoised Audio │
                    │ [B, 1, 81920]  │
                    └────────────────┘

註: ❄️ = 凍結 (不訓練), 🔥 = 可訓練
```

---

## 🧩 模型組件詳解

### 1. **WavTokenizer (凍結組件)**

#### 位置：預訓練模型，不參與訓練
```python
self.wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
for param in self.wavtokenizer.parameters():
    param.requires_grad = False
```

#### 功能：
- **Encoder**: 將連續音頻波形 → 離散 token 序列
  - 輸入: `[batch, 1, time_samples]` (例: [2, 1, 81920])
  - 輸出: `[batch, seq_len]` (例: [2, 75])
  - Token 範圍: 0-4095 (codebook size)
  
- **Decoder**: 將離散 token 序列 → 連續音頻波形
  - 輸入: `[batch, seq_len]`
  - 輸出: `[batch, 1, time_samples]`

#### 核心技術：
- VQ-VAE (Vector Quantized Variational AutoEncoder)
- Codebook: 4096 個 512 維的向量
- 訓練於大規模音頻數據，已學會音頻的壓縮表示

---

### 2. **Codebook Embedding Layer (創新點！)**

#### 位置：模型初始化
```python
# 1. 從 WavTokenizer 提取預訓練 codebook
pretrained_embeddings = self._extract_codebook_embeddings()  # [4096, 512]

# 2. 創建凍結的 codebook embedding
self.codebook_embedding = nn.Embedding.from_pretrained(
    pretrained_embeddings, 
    freeze=True  # 凍結，不訓練
)

# 3. 創建可學習的 special token embedding
self.special_token_embedding = nn.Embedding(3, 512)  # PAD, SOS, EOS
```

#### Token 映射策略：
```
Token ID 範圍     │  Embedding 來源                │ 可訓練?
─────────────────┼───────────────────────────────┼────────
0 ~ 4095         │ codebook_embedding            │ ❌ 凍結
4096 (PAD)       │ special_token_embedding[0]    │ ✅ 可學習
4097 (SOS)       │ special_token_embedding[1]    │ ✅ 可學習
4098 (EOS)       │ special_token_embedding[2]    │ ✅ 可學習
```

#### 優雅的實現 (偏移計算法)：
```python
def get_token_embeddings(self, tokens):
    # 初始化為 codebook embeddings
    raw_embeddings = torch.zeros(..., 512)  # codebook dim
    
    # 處理 codebook tokens (0-4095)
    codebook_mask = tokens < self.codebook_size
    if codebook_mask.any():
        raw_embeddings[codebook_mask] = self.codebook_embedding(tokens[codebook_mask])
    
    # 處理 special tokens (4096-4098)
    special_mask = tokens >= self.codebook_size
    if special_mask.any():
        # 關鍵：使用偏移計算 4096→0, 4097→1, 4098→2
        special_indices = tokens[special_mask] - self.codebook_size
        raw_embeddings[special_mask] = self.special_token_embedding(special_indices)
    
    return self.embedding_projection(raw_embeddings)  # 512 → 256
```

---

### 3. **Embedding Projection Layer**

```python
self.embedding_projection = nn.Linear(512, 256)
```

#### 功能：
- 將 codebook 的高維表示 (512) 投影到 Transformer 的工作維度 (256)
- 在保持性能的同時平衡計算效率
- 可訓練，學習最佳的維度映射

---

### 4. **Positional Encoding**

```python
self.pos_encoding = self._create_positional_encoding(d_model=256, max_length=400)
```

#### 功能：
- 為 token 序列添加位置信息
- 使用正弦/餘弦函數，固定不訓練
- 維度: `[1, 400, 256]` (支持更長序列)

#### 計算公式：
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

---

### 5. **Transformer (核心可訓練組件)**

```python
self.transformer = nn.Transformer(
    d_model=256,           # 模型維度 (Large 配置)
    nhead=8,               # 注意力頭數 (更多頭數，更強表達能力)
    num_encoder_layers=4,  # 編碼器層數 (更深網絡)
    num_decoder_layers=4,  # 解碼器層數 (更深網絡)
    dim_feedforward=1024,  # 前饋網絡維度 (4×d_model)
    dropout=0.1,
    batch_first=True
)
```

#### 架構：
```
Encoder (4 layers, 8 heads per layer):
  Input: Noisy Token Embeddings + Positional Encoding [B, L_src, 256]
  ↓
  Self-Attention Layer 1 (8 heads, d_model=256)
  ↓ 
  Feed Forward Network (dim=1024)
  ↓
  Self-Attention Layer 2 (8 heads, d_model=256)
  ↓
  Feed Forward Network (dim=1024)
  ↓
  Self-Attention Layer 3 (8 heads, d_model=256)
  ↓
  Feed Forward Network (dim=1024)
  ↓
  Self-Attention Layer 4 (8 heads, d_model=256)
  ↓
  Feed Forward Network (dim=1024)
  ↓
  Encoder Output [B, L_src, 256]

Decoder (4 layers, 8 heads per layer):
  Input: Clean Token Embeddings (Teacher Forcing) + Positional Encoding [B, L_tgt, 256]
  ↓
  Masked Self-Attention Layer 1 (Causal Mask, 8 heads)
  ↓
  Cross-Attention (attend to Encoder Output, 8 heads)
  ↓
  Feed Forward Network (dim=1024)
  ↓
  Masked Self-Attention Layer 2 (Causal Mask, 8 heads)
  ↓
  Cross-Attention (attend to Encoder Output, 8 heads)
  ↓
  Feed Forward Network (dim=1024)
  ↓
  Masked Self-Attention Layer 3 (Causal Mask, 8 heads)
  ↓
  Cross-Attention (attend to Encoder Output, 8 heads)
  ↓
  Feed Forward Network (dim=1024)
  ↓
  Masked Self-Attention Layer 4 (Causal Mask, 8 heads)
  ↓
  Cross-Attention (attend to Encoder Output, 8 heads)
  ↓
  Feed Forward Network (dim=1024)
  ↓
  Decoder Output [B, L_tgt, 256]
```

#### Mask 機制：
```python
# 1. Causal Mask (防止看到未來信息)
tgt_mask = self.generate_square_subsequent_mask(seq_len)
# 形狀: [seq_len, seq_len]
# 上三角為 -inf，下三角為 0

# 2. Padding Mask (忽略填充位置)
src_padding_mask = (src_tokens == self.pad_token)
tgt_padding_mask = (tgt_tokens == self.pad_token)
```

---

### 6. **Output Projection Layer**

```python
self.output_projection = nn.Linear(256, 4096)
```

#### 功能：
- 將 Transformer 輸出投影回 codebook 空間
- 輸入: `[B, L, 256]`
- 輸出: `[B, L, 4096]` (logits for each token position)
- 可訓練，學習最佳的逆映射

---

## 🔄 數據流程

### **訓練模式完整流程**

#### Step 1: Audio → Tokens (使用凍結的 WavTokenizer)
```python
# 輸入
noisy_audio: [2, 1, 81920]   # 批次大小2, 單聲道, ~3.4秒@24kHz
clean_audio: [2, 1, 81920]

# WavTokenizer Encoder (凍結)
noisy_tokens = self.wavtokenizer.encode(noisy_audio)  # [2, 75]
clean_tokens = self.wavtokenizer.encode(clean_audio)  # [2, 75]
```

#### Step 2: 準備 Transformer 輸入序列
```python
# 編碼器輸入: noisy_tokens + EOS
input_tokens = torch.cat([noisy_tokens, EOS_token], dim=1)  # [2, 76]

# 解碼器輸入: SOS + clean_tokens (Teacher Forcing)
decoder_input = torch.cat([SOS_token, clean_tokens], dim=1)  # [2, 76]

# 目標序列: clean_tokens + EOS
target_tokens = torch.cat([clean_tokens, EOS_token], dim=1)  # [2, 76]
```

#### Step 3: Embedding
```python
# 獲取 embeddings (混合策略)
src_emb = self.get_token_embeddings(input_tokens)      # [2, 76, 256]
tgt_emb = self.get_token_embeddings(decoder_input)     # [2, 76, 256]

# 添加位置編碼 + 縮放
src_emb = src_emb * sqrt(256) + pos_encoding[:, :76, :]
tgt_emb = tgt_emb * sqrt(256) + pos_encoding[:, :76, :]
```

#### Step 4: Transformer 前向傳播
```python
# Transformer 處理
output = self.transformer(
    src=src_emb,                    # [2, 76, 256]
    tgt=tgt_emb,                    # [2, 76, 256]
    tgt_mask=causal_mask,           # [76, 76] 下三角
    src_key_padding_mask=src_pad,   # [2, 76]
    tgt_key_padding_mask=tgt_pad    # [2, 76]
)
# 輸出: [2, 76, 256]
```

#### Step 5: 投影到 Token 空間
```python
logits = self.output_projection(output)  # [2, 76, 4096]
```

#### Step 6: Loss 計算 (兩種模式)

**模式 1: CrossEntropy Loss (預設)**
```python
# Reshape for loss calculation
logits_flat = logits.view(-1, 4096)      # [152, 4096]
target_flat = target_tokens.view(-1)     # [152]

# 只計算 codebook tokens 的損失 (忽略 special tokens)
mask = target_flat < 4096                 # [152]
loss = CrossEntropyLoss(logits_flat[mask], target_flat[mask])
```

**模式 2: Token Loss System (ttt2.py 邏輯)**
```python
# 使用多個損失組件
predicted_tokens = torch.argmax(logits, dim=-1)

loss_dict = compute_combined_token_loss(
    enhanced_tokens=predicted_tokens,
    target_tokens=target_tokens,
    noisy_tokens=noisy_tokens,
    embedding_layer=self.codebook_embedding,
    weights={
        'l2': 0.3,           # L2 距離損失
        'consistency': 0.4,   # 內容一致性
        'manifold': 0.1,      # Manifold 正則化
        'normalization': 0.1, # 正則化損失
        'coherence': 0.1      # 連貫性損失
    }
)

total_loss = loss_dict['total']
```

---

## 📍 Loss Function 作用位置

### **位置示意圖**

```
                    Transformer 輸出
                         │
                         ▼
                   Output Projection
                    Linear(128→4096)
                         │
                         ▼
        ┌────────────────────────────────┐
        │    Logits [B, L, 4096]         │
        └────────────────────────────────┘
                         │
                         │
        ┌────────────────┴─────────────────┐
        │                                  │
        ▼                                  ▼
┌──────────────────┐            ┌──────────────────┐
│ 訓練模式          │            │ 推理模式          │
│                  │            │                  │
│  Loss 計算位置    │            │  argmax()        │
│  ↓               │            │  ↓               │
│  與 Target       │            │  Predicted       │
│  Tokens 比較     │            │  Tokens          │
└──────────────────┘            └──────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│         Loss Function 選項               │
├─────────────────────────────────────────┤
│                                         │
│ 1. CrossEntropy Loss (預設)             │
│    - 簡單、快速                          │
│    - 直接優化 token 預測準確度           │
│    - 位置: logits vs target_tokens      │
│                                         │
│ 2. Token Loss System (--use_token_loss) │
│    - 多組件損失                          │
│    - 考慮 embedding 空間的幾何結構       │
│    - 位置: 在 embedding 空間計算         │
│                                         │
└─────────────────────────────────────────┘
```

### **詳細 Loss 計算流程**

#### **CrossEntropy Loss 流程**
```python
# 在 train_epoch() 函數中
def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    for batch in dataloader:
        # 1. 前向傳播
        output = model(noisy_audio, clean_audio)
        logits = output['logits']              # [B, L, 4096]
        target_tokens = output['target_tokens'] # [B, L]
        
        # 2. Reshape
        logits_flat = logits.reshape(-1, 4096)   # [B*L, 4096]
        target_flat = target_tokens.reshape(-1)  # [B*L]
        
        # 3. Clamp (防止 CUDA 錯誤)
        target_flat = torch.clamp(target_flat, 0, 4095)
        
        # 4. 創建 mask (只計算 codebook tokens)
        mask = target_flat < 4096
        
        # 5. 計算損失
        if mask.sum() > 0:
            loss = criterion(logits_flat[mask], target_flat[mask])
        
        # 6. 反向傳播
        loss.backward()
        optimizer.step()
```

#### **Token Loss System 流程**
```python
# 在 train_epoch_with_token_loss() 函數中
def train_epoch_with_token_loss(model, dataloader, optimizer, device, epoch, loss_weights):
    for batch in dataloader:
        # 1. 前向傳播
        output = model(noisy_audio, clean_audio)
        logits = output['logits']
        target_tokens = output['target_tokens']
        noisy_tokens = output['noisy_tokens']
        
        # 2. 獲取預測 tokens
        predicted_tokens = torch.argmax(logits, dim=-1)
        
        # 3. Clamp all tokens
        predicted_tokens = torch.clamp(predicted_tokens, 0, 4095)
        target_tokens = torch.clamp(target_tokens, 0, 4095)
        noisy_tokens = torch.clamp(noisy_tokens, 0, 4095)
        
        # 4. 計算組合損失
        loss_dict = compute_combined_token_loss(
            enhanced_tokens=predicted_tokens,
            target_tokens=target_tokens,
            noisy_tokens=noisy_tokens,
            embedding_layer=model.codebook_embedding,
            weights=loss_weights
        )
        
        total_loss = loss_dict['total']
        
        # 5. 反向傳播
        total_loss.backward()
        optimizer.step()
```

---

## 🎯 訓練與推理流程對比

### **訓練流程 (Training Mode)**

```
1. 數據準備
   ├─ Noisy Audio [B, 1, T]
   └─ Clean Audio [B, 1, T]

2. Token 化 (凍結 WavTokenizer)
   ├─ noisy_tokens = encode(noisy_audio)
   └─ clean_tokens = encode(clean_audio)

3. 序列構造
   ├─ input_tokens = [noisy_tokens, EOS]
   ├─ decoder_input = [SOS, clean_tokens]  ← Teacher Forcing
   └─ target_tokens = [clean_tokens, EOS]

4. Embedding (混合策略)
   ├─ Codebook tokens: 使用預訓練 embedding (凍結)
   └─ Special tokens: 使用可學習 embedding

5. Transformer 前向
   ├─ Encoder: 處理 input_tokens (4 layers, 8 heads)
   ├─ Decoder: 處理 decoder_input (with causal mask, 4 layers, 8 heads)
   └─ Output: [B, L, 256]

6. 投影到 Token 空間
   └─ logits = Linear(256→4096)  # [B, L, 4096]

7. Loss 計算
   ├─ CrossEntropy: logits vs target_tokens
   └─ 或 Token Loss: 多組件損失

8. 反向傳播
   ├─ loss.backward()
   ├─ 梯度裁剪 (防止爆炸)
   └─ optimizer.step()

9. 每 N epochs 保存
   ├─ 音頻樣本 (對比 noisy/enhanced/clean)
   ├─ 頻譜圖 (視覺化降噪效果)
   └─ 模型檢查點
```

### **推理流程 (Inference Mode)**

```
1. 數據準備
   └─ Noisy Audio [B, 1, T]

2. Token 化
   └─ noisy_tokens = encode(noisy_audio)

3. Transformer 前向 (Encoder-only)
   ├─ 只使用 Encoder 進行 self-attention (4 layers, 8 heads)
   ├─ 沒有 Teacher Forcing
   └─ Output: [B, L, 256]

4. 投影到 Token 空間
   └─ logits = Linear(256→4096)

5. 預測 Tokens
   └─ predicted_tokens = argmax(logits, dim=-1)

6. Token 重建為音頻
   └─ denoised_audio = decode(predicted_tokens)

7. 輸出
   └─ Denoised Audio [B, 1, T]
```

---

## 💡 關鍵創新點

### 1. **預訓練 Codebook Embedding**
```
優勢:
✅ 保留 WavTokenizer 的音頻語義知識
✅ 加速收斂 (不需從零學習 token 表示)
✅ 提升性能 (利用大規模預訓練)
✅ 減少訓練參數 (4096×512 凍結)

對比:
❌ 傳統方法: 隨機初始化 nn.Embedding(4099, 128)
   - 丟失預訓練知識
   - 需要更多訓練時間
   - 性能可能較差
```

### 2. **混合 Embedding 策略**
```
Token 類型       │ Embedding 來源          │ 訓練策略
────────────────┼────────────────────────┼─────────
Codebook (4096) │ 預訓練 WavTokenizer     │ 凍結
Special (3)     │ 隨機初始化              │ 可學習

理由:
- Codebook tokens: 音頻內容，應保留預訓練語義
- Special tokens: 任務特定，需要學習新含義
```

### 3. **優雅的偏移計算**
```python
# 不使用循環，利用 token ID 連續性
special_mask = tokens >= 4096
special_indices = tokens[special_mask] - 4096  # 4096→0, 4097→1, 4098→2
raw_embeddings[special_mask] = self.special_token_embedding(special_indices)

優勢:
✅ 向量化操作，GPU 高效
✅ 代碼簡潔，易維護
✅ 自動處理混合 token 類型
```

### 4. **Target Token 填充修正**
```python
# 正確做法: 先填充 token IDs，再 embedding
if tgt_seq_len < max_pos_len:
    pad = torch.full((B, pad_size), self.pad_token)  # 使用 pad_token ID
    tgt_tokens_padded = torch.cat([tgt_tokens, pad], dim=1)

tgt_emb = self.get_token_embeddings(tgt_tokens_padded)  # 所有位置都有正確 embedding

# 錯誤做法: 先 embedding，再用零向量填充
tgt_emb = self.get_token_embeddings(tgt_tokens)
pad = torch.zeros((B, pad_size, 128))  # ❌ 零向量，非 pad_token embedding
tgt_emb = torch.cat([tgt_emb, pad], dim=1)
```

### 5. **Token Loss System (可選)**
```
組件                  │ 權重  │ 作用
─────────────────────┼──────┼────────────────────────────
L2 Distance          │ 0.3  │ 預測與目標的直接距離
Consistency          │ 0.4  │ 保持與 noisy 的內容一致性
Manifold Reg         │ 0.1  │ 確保在 codebook manifold 上
Normalization        │ 0.1  │ embedding 範數正則化
Coherence            │ 0.1  │ 序列連貫性

優勢:
✅ 考慮 embedding 空間的幾何結構
✅ 不只優化離散預測，也優化連續表示
✅ 更符合音頻降噪的本質
```

---

## 📊 模型參數統計

```python
# Large 配置 (d_model=256, nhead=8, layers=4×4)

組件                          │ 參數量        │ 可訓練?
─────────────────────────────┼──────────────┼────────
WavTokenizer (Encoder+Decoder)│ ~50M         │ ❌ 凍結
Codebook Embedding (4096×512) │ ~2.1M        │ ❌ 凍結
Special Token Embedding (3×512)│ ~1.5K       │ ✅ 訓練
Embedding Projection (512→256)│ ~131K        │ ✅ 訓練
Positional Encoding           │ 400×256      │ ❌ 固定
Transformer (4+4 layers)      │ ~3-4M        │ ✅ 訓練
  ├─ Multi-Head Attention     │ ~2M          │
  └─ Feed Forward Networks    │ ~1-2M        │
Output Projection (256→4096)  │ ~1.05M       │ ✅ 訓練
─────────────────────────────┼──────────────┼────────
總參數                        │ ~56M         │
可訓練參數                    │ ~4-5M        │ ← 主要訓練這部分
```

---

## 🔍 Loss Function 數學表達

### CrossEntropy Loss
```
給定:
- logits: [B, L, 4096] 模型輸出
- target: [B, L] 目標 token IDs

計算:
L_CE = -∑ᵢ ∑ₜ log(softmax(logits[i,t])[target[i,t]])

其中:
- i: batch index
- t: sequence position
- 只計算 target < 4096 的位置 (忽略 special tokens)
```

### Token Loss System
```
給定:
- pred_tokens: [B, L] 預測的 token IDs
- target_tokens: [B, L] 目標 token IDs  
- noisy_tokens: [B, L] 噪聲 token IDs
- E: embedding layer (4096×512)

計算:
pred_emb = E[pred_tokens]      # [B, L, 512]
target_emb = E[target_tokens]  # [B, L, 512]
noisy_emb = E[noisy_tokens]    # [B, L, 512]

L_total = w₁·L_L2 + w₂·L_consistency + w₃·L_manifold + w₄·L_norm + w₅·L_coherence

其中:
L_L2 = ||pred_emb - target_emb||²
L_consistency = ||pred_emb - noisy_emb||²
L_manifold = distance_to_manifold(pred_emb)
L_norm = ||pred_emb||²
L_coherence = ||pred_emb[t+1] - pred_emb[t]||²
```

---

## 🎬 訓練命令示例

```bash
# Large 配置 - CrossEntropy Loss (當前運行)
python wavtokenizer_transformer_denoising.py \
    --d_model 256 \
    --nhead 8 \
    --num_encoder_layers 4 \
    --num_decoder_layers 4 \
    --dim_feedforward 1024 \
    --max_length 400 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_epochs 1000 \
    --learning_rate 1e-4 \
    --ce_weight 15.0 \
    --l2_embed_weight 1.5 \
    --disable_scheduler \
    --output_dir results/transformer_large

# Small 配置 - 用於快速實驗
python wavtokenizer_transformer_denoising.py \
    --d_model 128 \
    --nhead 2 \
    --num_encoder_layers 2 \
    --num_decoder_layers 2 \
    --dim_feedforward 256 \
    --batch_size 8 \
    --num_epochs 300 \
    --learning_rate 1e-4 \
    --output_dir results/transformer_small

# Token Loss System (實驗性)
python wavtokenizer_transformer_denoising.py \
    --d_model 256 \
    --nhead 8 \
    --num_encoder_layers 4 \
    --num_decoder_layers 4 \
    --batch_size 4 \
    --num_epochs 1000 \
    --learning_rate 1e-4 \
    --use_token_loss \
    --l2_weight 0.3 \
    --consistency_weight 0.4 \
    --manifold_weight 0.1 \
    --normalization_weight 0.1 \
    --coherence_weight 0.1 \
    --output_dir results/token_loss_exp
```

---

## 📚 總結

這個模型的核心設計哲學是：

1. **分離關注點**: WavTokenizer 負責音頻↔Token 轉換，Transformer 負責降噪
2. **知識復用**: 利用預訓練 codebook 的音頻語義知識
3. **高效訓練**: 凍結大部分參數，只訓練降噪 Transformer
4. **靈活損失**: 支持簡單的 CrossEntropy 和複雜的 Token Loss System
5. **端到端**: 輸入音頻，輸出降噪音頻，中間自動處理 tokenization

這種架構在保持高效的同時，充分利用了預訓練模型的能力，是一個設計優雅、實用性強的降噪系統！
