# Token Denoising Transformer: 基於凍結 Codebook 的音訊降噪

**生成日期**: 2025-10-22  
**生成函式**: TOKEN_DENOISING_TRANSFORMER_EXPLANATION.md  
**相關文件**: [TOKEN_RELATIONSHIP_EXPLANATION.md](./TOKEN_RELATIONSHIP_EXPLANATION.md)

---

## 🎯 核心概念

本方法利用 [TOKEN_RELATIONSHIP_EXPLANATION.md](./TOKEN_RELATIONSHIP_EXPLANATION.md) 中解釋的 **Token-Codebook 因果關係**，設計一個 **完全凍結 WavTokenizer Codebook** 的降噪 Transformer。

### 關鍵洞察

根據 TOKEN_RELATIONSHIP_EXPLANATION.md 的核心公式：

```
量化特徵[t] = Codebook[Token[t]]
```

我們知道：
1. **Token 是 Codebook 的索引** (整數 ID)
2. **Codebook 已經學到音訊的最佳表示** (4096 個 512-D 向量)
3. **不需要重新訓練 embedding**，直接用 WavTokenizer 的 Codebook

---

## 📊 架構設計：類比機器翻譯

### 機器翻譯 vs Token Denoising

```
┌────────────────────────────────────────────────────────────────────┐
│              機器翻譯 (Machine Translation)                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  英文 Token IDs: [23, 456, 89, ...]                                │
│         ↓                                                           │
│  Frozen English Embedding (預訓練的)                               │
│         ↓                                                           │
│  Transformer Encoder-Decoder                                        │
│         ↓                                                           │
│  Output Projection to Chinese Vocab                                │
│         ↓                                                           │
│  中文 Token IDs: [67, 234, 12, ...]                                │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│              Token Denoising (我們的方法)                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Noisy Token IDs: [2347, 891, 3102, ...]                          │
│         ↓                                                           │
│  Frozen WavTokenizer Codebook (已訓練好的)                         │
│         ↓                                                           │
│  Transformer Encoder                                                │
│         ↓                                                           │
│  Output Projection to Same Vocab                                   │
│         ↓                                                           │
│  Clean Token IDs: [2351, 893, 3105, ...]                          │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

**核心類比**：
- **英文 → 中文** ≈ **Noisy → Clean**
- **預訓練的詞嵌入** ≈ **WavTokenizer 的 Codebook**
- **翻譯模型** ≈ **Denoising Transformer**

---

## 🔗 與 TOKEN_RELATIONSHIP_EXPLANATION.md 的直接關聯

### 1. 重用 Vector Quantization 機制

TOKEN_RELATIONSHIP_EXPLANATION.md 中的 VQ 公式：

```
給定：
- 連續特徵: z ∈ ℝ^512
- Codebook: C = {c₀, c₁, ..., c₄₀₉₅}  (4096 個 512-D 向量)

量化過程:
1. 找最近鄰：
   k* = argmin_{k} ||z - cₖ||₂²

2. 輸出：
   - 離散 token: k* ∈ {0, 1, ..., 4095}
   - 量化特徵: q = c_{k*} ∈ ℝ^512
```

**我們的使用方式**：
```python
# Step 1: 直接使用 WavTokenizer 的 Codebook (凍結)
self.register_buffer('codebook', wavtokenizer_codebook)  # (4096, 512)

# Step 2: Token → Embedding (完全不訓練)
embeddings = self.codebook[noisy_token_ids]  # 查表操作

# Step 3: Transformer 處理
clean_logits = self.transformer(embeddings)

# Step 4: 輸出 Clean Token IDs
clean_token_ids = clean_logits.argmax(dim=-1)
```

### 2. 完整流程對應

根據 TOKEN_RELATIONSHIP_EXPLANATION.md 的編碼-解碼流程：

```
原始流程 (TOKEN_RELATIONSHIP_EXPLANATION.md):
══════════════════════════════════════════════

音訊 → Encoder → 連續特徵 → VQ → Token
                                  ↓
                            Codebook Lookup
                                  ↓
                            量化特徵 → Decoder → 音訊


我們的降噪流程:
══════════════════════════════════════════════

Noisy 音訊 → WavTokenizer Encode → Noisy Token IDs
                                         ↓
                                 Frozen Codebook Lookup
                                         ↓
                                   Embeddings (512-D)
                                         ↓
                              Transformer Denoising
                                         ↓
                                Output Logits (4096-D)
                                         ↓
                                  Clean Token IDs
                                         ↓
                                 Codebook Lookup
                                         ↓
                              Clean 量化特徵 (512-D)
                                         ↓
                            WavTokenizer Decode → Clean 音訊
```

### 3. Codebook 的角色

TOKEN_RELATIONSHIP_EXPLANATION.md 說明：

```
Codebook 是一個「字典」：

    索引 (Token)  →  向量 (512-D)
    ──────────────────────────────
    0            →  [0.21, 0.88, -0.51, ...]
    1            →  [0.15, 0.73, -0.22, ...]
    2347         →  [0.23, -0.45, 0.89, ...]  ← 訓練學到的最佳表示
    ...
    4095         →  [-0.45, 0.23, 0.11, ...]
```

**我們的使用**：
- ✅ **完全凍結這個字典**（不修改任何 entry）
- ✅ **只學習 Noisy Token → Clean Token 的映射**
- ✅ **輸入輸出都使用同一個 Codebook**

---

## 🏗️ 詳細架構

### 完整數據流

```
┌──────────────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                                     │
└──────────────────────────────────────────────────────────────────────┘

輸入: Paired (Noisy Audio, Clean Audio)

Step 1: WavTokenizer Encoding (凍結)
─────────────────────────────────────
    Noisy Audio (1, 24000)
         │ WavTokenizer.encode_infer()
         ▼
    Noisy Token IDs (1, 75)
    例: [2347, 891, 3102, 1456, ...]
         │
         │ 同時
         │
    Clean Audio (1, 24000)
         │ WavTokenizer.encode_infer()
         ▼
    Clean Token IDs (1, 75)  ← Ground Truth
    例: [2351, 893, 3105, 1458, ...]


Step 2: Token → Embedding (凍結 Codebook)
──────────────────────────────────────────
    Noisy Token IDs (1, 75)
         │
         │ self.codebook[noisy_token_ids]
         │ ↓ 查表，不反向傳播到 codebook
         ▼
    Noisy Embeddings (1, 75, 512)
    
    每個 token 被替換成對應的 512-D 向量:
    ┌─────────────────────────────────────────┐
    │ t=0: codebook[2347] = [0.23, -0.45, ...]│
    │ t=1: codebook[891]  = [0.15, 0.73,  ...]│
    │ t=2: codebook[3102] = [0.19, 0.92,  ...]│
    │ ...                                      │
    └─────────────────────────────────────────┘


Step 3: Positional Encoding
────────────────────────────
    Noisy Embeddings (1, 75, 512)
         │ + Sinusoidal Position Encoding
         ▼
    Positioned Embeddings (1, 75, 512)


Step 4: Transformer Encoder (可訓練)
─────────────────────────────────────
    Positioned Embeddings (1, 75, 512)
         │
         │ Multi-Head Self-Attention
         │ Feed-Forward Network
         │ Layer Norm
         │ × 6 layers
         ▼
    Hidden States (1, 75, 512)


Step 5: Output Projection (可訓練)
───────────────────────────────────
    Hidden States (1, 75, 512)
         │ Linear(512 → 4096)
         ▼
    Logits (1, 75, 4096)
    
    每個時間步的 logits 表示 4096 個 token 的機率分佈


Step 6: Loss Computation
─────────────────────────
    Logits (1, 75, 4096)  vs  Clean Token IDs (1, 75)
         │
         │ Cross-Entropy Loss
         ▼
    Loss = -log P(Clean Token | Noisy Embedding)


┌──────────────────────────────────────────────────────────────────────┐
│                    INFERENCE PHASE                                    │
└──────────────────────────────────────────────────────────────────────┘

輸入: Noisy Audio

Step 1: Encode
──────────────
    Noisy Audio (1, 24000)
         │ WavTokenizer.encode_infer()
         ▼
    Noisy Token IDs (1, 75)
    例: [2347, 891, 3102, ...]


Step 2-5: 同 Training (凍結的 Codebook + 訓練好的 Transformer)
────────────────────────────────────────────────────────────
    Noisy Token IDs → Embeddings → Transformer → Logits


Step 6: Greedy Decoding
────────────────────────
    Logits (1, 75, 4096)
         │ argmax(dim=-1)
         ▼
    Predicted Clean Token IDs (1, 75)
    例: [2351, 893, 3105, ...]


Step 7: Token → Embedding → Audio
──────────────────────────────────
    Predicted Clean Token IDs (1, 75)
         │ codebook[predicted_tokens]
         ▼
    Clean Embeddings (1, 512, 75)
         │ WavTokenizer.decode()
         ▼
    Denoised Audio (1, 24000)
```

---

## 🔍 關鍵技術細節

### 1. Frozen Codebook 的實現

```python
class TokenDenoisingTransformer(nn.Module):
    def __init__(self, codebook, ...):
        super().__init__()
        
        # ✅ 正確做法: register_buffer (不參與梯度更新)
        self.register_buffer('codebook', codebook)
        
        # ❌ 錯誤做法: nn.Embedding (會訓練新的 embedding)
        # self.embedding = nn.Embedding(4096, 512)
    
    def forward(self, noisy_token_ids):
        # 直接查表，梯度不會回傳到 codebook
        embeddings = self.codebook[noisy_token_ids]  # (B, T, 512)
        
        # 只有 Transformer 和 output_proj 會被訓練
        hidden = self.transformer(embeddings)
        logits = self.output_proj(hidden)
        
        return logits
```

**為什麼使用 `register_buffer`？**
- ✅ 不會被加入 `model.parameters()`
- ✅ 會隨模型一起移動到 GPU/CPU
- ✅ 會被保存在 `state_dict` 中
- ✅ 梯度不會回傳（`requires_grad=False`）

### 2. Token ID 到 Embedding 的映射

根據 TOKEN_RELATIONSHIP_EXPLANATION.md 的查表機制：

```python
# TOKEN_RELATIONSHIP_EXPLANATION.md 中的驗證代碼:
token_idx = tokens[0, 0, 0].item()  # 例: 2347
codebook_vector = codebook[token_idx]  # 從 codebook 查找
quantized_vector = features[0, :, 0]  # 量化後的特徵

# 驗證: codebook_vector == quantized_vector ✓

# 我們的使用 (批次化):
noisy_tokens = torch.tensor([[2347, 891, 3102]])  # (1, 3)
embeddings = self.codebook[noisy_tokens]  # (1, 3, 512)

# 等價於:
# embeddings[0, 0, :] = codebook[2347]
# embeddings[0, 1, :] = codebook[891]
# embeddings[0, 2, :] = codebook[3102]
```

### 3. 損失函數設計

```python
def compute_loss(self, logits, target_token_ids):
    """
    Cross-Entropy Loss for Token Classification
    
    Args:
        logits: (B, T, 4096) - 每個時間步對 4096 個 token 的預測
        target_token_ids: (B, T) - Ground Truth Clean Token IDs
    
    Returns:
        loss: scalar
        accuracy: scalar
    """
    B, T, vocab_size = logits.shape
    
    # Reshape for cross-entropy
    loss = F.cross_entropy(
        logits.reshape(B * T, vocab_size),  # (B*T, 4096)
        target_token_ids.reshape(B * T)      # (B*T,)
    )
    
    # Token 準確率
    pred_tokens = logits.argmax(dim=-1)  # (B, T)
    accuracy = (pred_tokens == target_token_ids).float().mean()
    
    return loss, accuracy
```

**為什麼用 Cross-Entropy？**
- Token ID 是離散分類問題（4096 個類別）
- Cross-Entropy 是標準的分類損失
- 梯度只回傳到 Transformer 和 output_proj，不影響 Codebook

---

## 📐 數學推導

### 問題定義

根據 TOKEN_RELATIONSHIP_EXPLANATION.md 的量化公式，我們擴展到降噪場景：

```
給定:
- Noisy Audio: x_noisy
- Clean Audio: x_clean (Ground Truth)
- WavTokenizer Encoder: E(·)
- WavTokenizer Quantizer: Q(·)
- Frozen Codebook: C = {c₀, c₁, ..., c₄₀₉₅}

編碼過程 (TOKEN_RELATIONSHIP_EXPLANATION.md):
    z = E(x)              # 連續特徵 ∈ ℝ^{512×T}
    k = Q(z, C)           # Token IDs ∈ {0,...,4095}^T
    q = C[k]              # 量化特徵 = codebook[token]

降噪目標:
    找到映射 f: K_noisy → K_clean
    使得 f(Q(E(x_noisy))) ≈ Q(E(x_clean))
```

### Transformer 建模

```
模型: f_θ (參數 θ 包含 Transformer + Output Projection)

前向傳播:
1. Token → Embedding (凍結):
   e_t = C[k_t^noisy]  ∈ ℝ^512

2. Positional Encoding:
   ẽ_t = e_t + PE(t)

3. Transformer:
   h_t = Transformer_θ(ẽ_0, ..., ẽ_T)  ∈ ℝ^512

4. Output Projection:
   logits_t = W_out · h_t + b_out  ∈ ℝ^4096

5. Prediction:
   k_t^pred = argmax(logits_t)

損失函數:
   L(θ) = -∑_{t=0}^{T-1} log P(k_t^clean | k_0^noisy, ..., k_{T-1}^noisy; θ)
        = -∑_{t=0}^{T-1} log softmax(logits_t)[k_t^clean]
```

### 訓練目標

```
最小化: L(θ) = E_{(x_noisy, x_clean) ~ D} [CrossEntropy(f_θ(Q(E(x_noisy))), Q(E(x_clean)))]

約束條件:
1. Codebook C 保持凍結: ∇_C L = 0
2. WavTokenizer (E, Q, Decoder) 保持凍結
3. 只訓練 θ (Transformer + Output Projection)
```

---

## 🆚 與其他方法的比較

### 方法 1: Embedding Space Translation (之前的方案)

```
Noisy Audio → Encoder → Noisy Features (512-D 連續)
                              ↓
                          U-Net 降噪
                              ↓
                        Clean Features (512-D 連續)
                              ↓
                        Re-quantization
                              ↓
                        Clean Tokens
```

**問題**：
- ❌ 沒有完全利用 Token 的離散性
- ❌ Re-quantization 步驟不可微分（需要 Gumbel-Softmax 等技巧）
- ❌ 在連續空間操作，難以保證輸出在 Codebook 範圍內

### 方法 2: Token Mapping (我們的方案) ✅

```
Noisy Audio → Encoder → Noisy Tokens (離散 ID)
                              ↓
                      Frozen Codebook Lookup
                              ↓
                      Transformer 降噪
                              ↓
                      Output Logits (4096-D)
                              ↓
                        Clean Tokens (離散 ID)
```

**優勢**：
- ✅ **完全離散化**：輸入輸出都是 Token ID
- ✅ **凍結 Codebook**：重用 WavTokenizer 的知識
- ✅ **端到端可微**：Cross-Entropy 損失直接優化
- ✅ **類比翻譯模型**：成熟的 Seq2Seq 框架

---

## 🧪 與 TOKEN_RELATIONSHIP_EXPLANATION.md 的實驗對應

TOKEN_RELATIONSHIP_EXPLANATION.md 的驗證代碼：

```python
# 1. 獲取 Token 和 Codebook
features, tokens = model.encode_infer(audio, bandwidth_id=torch.tensor([0]))
codebook = model.feature_extractor.encodec.quantizer.vq.layers[0].codebook

# 2. 驗證 Token → Codebook Lookup
token_idx = tokens[0, 0, 0].item()
codebook_vector = codebook[token_idx]
quantized_vector = features[0, :, 0]

# 3. 確認相等
assert torch.allclose(codebook_vector, quantized_vector, atol=1e-5)
```

**我們的使用**（完全相同的機制）：

```python
# 1. 訓練時: 獲取 Noisy 和 Clean Tokens
_, noisy_tokens = wavtokenizer.encode_infer(noisy_audio, ...)
_, clean_tokens = wavtokenizer.encode_infer(clean_audio, ...)

# 2. Frozen Codebook Lookup (跟 TOKEN_RELATIONSHIP_EXPLANATION.md 一樣)
noisy_embeddings = codebook[noisy_tokens]  # (B, T, 512)

# 3. Transformer 預測 Clean Tokens
logits = transformer(noisy_embeddings)
pred_clean_tokens = logits.argmax(dim=-1)

# 4. 解碼時: Token → Codebook → Audio (跟 TOKEN_RELATIONSHIP_EXPLANATION.md 一樣)
clean_embeddings = codebook[pred_clean_tokens]
denoised_audio = wavtokenizer.decode(clean_embeddings, ...)
```

---

## 📊 實驗設計

### 資料集需求

```python
from torch.utils.data import Dataset

class NoisyCleanPairDataset(Dataset):
    """
    配對的噪音-乾淨音檔資料集
    
    回傳:
        {
            'noisy': Tensor (1, 24000),  # 噪音音訊
            'clean': Tensor (1, 24000)   # 乾淨音訊 (Ground Truth)
        }
    """
    
    def __init__(self, noisy_dir, clean_dir):
        self.noisy_files = sorted(Path(noisy_dir).glob('*.wav'))
        self.clean_files = sorted(Path(clean_dir).glob('*.wav'))
        
        assert len(self.noisy_files) == len(self.clean_files)
    
    def __getitem__(self, idx):
        noisy, _ = torchaudio.load(self.noisy_files[idx])
        clean, _ = torchaudio.load(self.clean_files[idx])
        
        return {'noisy': noisy, 'clean': clean}
```

### 訓練流程

```python
# 1. 載入 WavTokenizer (凍結)
wavtokenizer = WavTokenizer.from_hparams0802(config_path)
wavtokenizer.eval()
for param in wavtokenizer.parameters():
    param.requires_grad = False

# 2. 獲取 Codebook
codebook = wavtokenizer.feature_extractor.encodec.quantizer.vq.layers[0].codebook

# 3. 創建 Transformer (只訓練這個)
transformer = TokenDenoisingTransformer(
    codebook=codebook,  # 凍結的
    d_model=512,
    nhead=8,
    num_layers=6
)

# 4. 訓練
trainer = TokenDenoisingTrainer(
    transformer=transformer,
    wavtokenizer=wavtokenizer,
    train_loader=train_loader
)

for epoch in range(100):
    train_loss, train_acc = trainer.train_epoch()
    val_loss, val_acc = trainer.validate()
    
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2%}")
    print(f"           Val Loss={val_loss:.4f}, Acc={val_acc:.2%}")
```

### 評估指標

```python
# 1. Token-level 指標
token_accuracy = (pred_tokens == clean_tokens).float().mean()
token_change_rate = (noisy_tokens != pred_tokens).float().mean()

# 2. Audio-level 指標
from pesq import pesq
from pystoi import stoi

pesq_score = pesq(24000, clean_audio.numpy(), denoised_audio.numpy(), 'wb')
stoi_score = stoi(clean_audio.numpy(), denoised_audio.numpy(), 24000)

# 3. Spectral 指標
def spectral_convergence(x, y):
    return torch.norm(x - y, p='fro') / torch.norm(x, p='fro')

spec_conv = spectral_convergence(clean_spec, denoised_spec)
```

---

## 💡 關鍵優勢總結

### 1. 完全重用 WavTokenizer 的 Codebook

```
TOKEN_RELATIONSHIP_EXPLANATION.md 說:
    "Codebook 是訓練學到的最佳表示"
    
我們:
    ✅ 凍結 Codebook，不修改任何 entry
    ✅ 相信 WavTokenizer 已經學到的音訊表示
    ✅ 只學習 Noisy → Clean 的 Token 映射
```

### 2. 離散化降噪

```
TOKEN_RELATIONSHIP_EXPLANATION.md 說:
    "Token 是 Codebook 的索引 (整數)"
    
我們:
    ✅ 輸入: Noisy Token IDs (離散)
    ✅ 輸出: Clean Token IDs (離散)
    ✅ 中間: Frozen Codebook Embeddings
    ✅ 損失: Cross-Entropy (標準分類)
```

### 3. 類比機器翻譯

```
成熟的 NLP 模型:
    - BERT: Masked Token Prediction
    - GPT: Next Token Prediction
    - Translation: Source Token → Target Token
    
我們的模型:
    - Token Denoising: Noisy Token → Clean Token
    - 使用相同的架構 (Transformer)
    - 使用相同的損失 (Cross-Entropy)
    - 使用相同的技巧 (Frozen Embedding)
```

---

## 🔧 實作檢查清單

使用本方法時，請確保：

- [ ] **Codebook 完全凍結**
  ```python
  assert not transformer.codebook.requires_grad
  ```

- [ ] **WavTokenizer 凍結**
  ```python
  for param in wavtokenizer.parameters():
      assert not param.requires_grad
  ```

- [ ] **只訓練 Transformer + Output Projection**
  ```python
  trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
  print(f"Trainable parameters: {trainable_params:,}")
  ```

- [ ] **Token ID 查表正確**
  ```python
  # 驗證 (參考 TOKEN_RELATIONSHIP_EXPLANATION.md)
  embeddings = codebook[token_ids]
  assert embeddings.shape == (B, T, 512)
  ```

- [ ] **輸出 Token 在合法範圍**
  ```python
  assert pred_tokens.min() >= 0
  assert pred_tokens.max() < 4096
  ```

---

## 📚 參考文件

1. **TOKEN_RELATIONSHIP_EXPLANATION.md**
   - Vector Quantization 機制
   - Codebook 的作用
   - Token-Feature 的因果關係

2. **token_denoising_transformer.py**
   - 完整實作代碼
   - 訓練和推理流程

3. **相關論文**
   - VQ-VAE: "Neural Discrete Representation Learning"
   - Transformer: "Attention is All You Need"
   - Audio Codecs: "SoundStream", "EnCodec"

---

## 🎯 下一步

1. **準備配對資料集**
   - 噪音音檔 + 對應的乾淨音檔
   - 建議數量: 10k+ pairs

2. **訓練 Transformer**
   ```bash
   python token_denoising_transformer.py --train \
       --noisy-dir data/noisy \
       --clean-dir data/clean \
       --epochs 100
   ```

3. **評估降噪效果**
   - Token Accuracy
   - PESQ/STOI 分數
   - 主觀聽感測試

4. **可能的改進**
   - 使用 Decoder-only Transformer (GPT-style)
   - 加入 Noise Conditioning (不同噪音類型)
   - Multi-task Learning (同時預測噪音類型)

---

**實驗執行者**: GitHub Copilot  
**理論基礎**: TOKEN_RELATIONSHIP_EXPLANATION.md  
**實作文件**: token_denoising_transformer.py  
**最後更新**: 2025-10-22
