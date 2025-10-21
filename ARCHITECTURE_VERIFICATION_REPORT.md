# 訓練架構核心驗證報告

## 🎯 您提出的三個關鍵問題

### 問題 1：之前訓練不好是因為沒有正確回傳嗎？

**答案**：**是的，這是主要原因之一！**

#### 證據分析

**驗證損失失效的影響**：
- ❌ **無法判斷過擬合**：訓練損失持續下降到 3.61，但驗證損失恆為 1000000
- ❌ **Best Model 選擇錯誤**：`best_model.pth` 實際上是 Epoch 1 的模型（因為驗證損失永遠是 1000000）
- ❌ **無法調整訓練策略**：不知道模型是否已經過擬合，應該繼續訓練還是提前停止

**但這不是唯一問題**：
```
問題層級：
1. 🔴 驗證函數失效 → 無法評估模型泛化能力
2. 🟡 模型容量/損失函數 → 影響訓練效果本身
3. 🟢 其他細節 → 次要影響
```

**修復後的預期**：
- ✅ 能看到真實的 Val Loss（預計 3.5-4.5 之間）
- ✅ 能判斷是否過擬合（Val Loss vs Train Loss）
- ✅ Best Model 能根據最低 Val Loss 正確選擇

---

### 問題 2：確認完全是使用預訓練 codebook 凍結的嗎？

**答案**：**是的，100% 確認使用預訓練 codebook 並完全凍結！**

#### 代碼證據

**1. Codebook 提取（第 287-310 行）**
```python
def _extract_codebook_embeddings(self):
    """從 WavTokenizer 提取預訓練的 codebook embeddings"""
    try:
        # 從預訓練的 WavTokenizer 提取 VQ codebook
        vq_layers = self.wavtokenizer.feature_extractor.encodec.quantizer.vq.layers
        
        # 提取第一個 VQ 層的 codebook（4096 個 codes）
        if len(vq_layers) == 1:
            codebook_embeddings = vq_layers[0].codebook
        else:
            # 如果有多層，拼接所有層
            codebook_embeddings = torch.cat([vq.codebook for vq in vq_layers], dim=0)
        
        logging.info(f"成功提取 codebook embeddings: shape={codebook_embeddings.shape}")
        
        # detach 以避免梯度計算
        return codebook_embeddings.detach()  # ✅ 斷開梯度
```

**2. Embedding 層創建（第 248-252 行）**
```python
# 使用預訓練權重並凍結
self.codebook_embedding = nn.Embedding.from_pretrained(
    pretrained_embeddings,   # ✅ 使用從 WavTokenizer 提取的預訓練權重
    freeze=True              # ✅ 完全凍結，不可訓練
)
logging.info(f"成功創建 codebook embedding: shape={pretrained_embeddings.shape}, freeze=True")
```

**3. WavTokenizer 本身也凍結（第 224-228 行）**
```python
# 載入預訓練的 WavTokenizer
self.wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)

# 凍結 WavTokenizer 的所有參數
for param in self.wavtokenizer.parameters():
    param.requires_grad = False  # ✅ 整個 WavTokenizer 凍結
```

#### 驗證方式

您可以運行以下代碼確認：

```python
import torch
from wavtokenizer_transformer_denoising import WavTokenizerTransformerDenoiser

# 載入模型
checkpoint = torch.load('results/transformer_large_tokenloss_large_tokenloss_202510190523/checkpoint_epoch_300.pth')
model = WavTokenizerTransformerDenoiser(...)
model.load_state_dict(checkpoint['model_state_dict'])

# 檢查 codebook_embedding 是否凍結
print(f"Codebook Embedding 是否凍結: {not model.codebook_embedding.weight.requires_grad}")
# 預期輸出: True

# 檢查 WavTokenizer 是否凍結
wavtokenizer_frozen = all(not p.requires_grad for p in model.wavtokenizer.parameters())
print(f"WavTokenizer 是否完全凍結: {wavtokenizer_frozen}")
# 預期輸出: True

# 檢查可訓練參數
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"可訓練參數: {trainable_params:,} / {total_params:,}")
# 預期: 可訓練參數遠小於總參數（WavTokenizer 被排除）
```

---

### 問題 3：Token 應該是 WavTokenizer Decoder 看得懂的，接近 WavTokenizer Encoder 提取的對應 target 的 token 對吧？

**答案**：**是的，這是核心設計理念！但有重要的細節需要釐清。**

#### 架構流程圖

```
訓練時：

Input (Noisy Audio) ─────────────────┐
                                      │
                    ┌─────────────────▼──────────────────┐
                    │   WavTokenizer Encoder (凍結)      │
                    │   - 提取音頻的 discrete tokens      │
                    └─────────────────┬──────────────────┘
                                      │
                            Noisy Tokens [B, L]
                                      │
                    ┌─────────────────▼──────────────────┐
                    │   Transformer Denoiser (可訓練)    │
                    │   - Input: noisy_tokens            │
                    │   - Teacher Forcing: clean_tokens   │
                    │   - Output: predicted_logits       │
                    └─────────────────┬──────────────────┘
                                      │
                          Predicted Logits [B, L, 4096]
                                      │
                         argmax ──────┴────── Predicted Tokens [B, L]
                                      │
                    ┌─────────────────▼──────────────────┐
                    │   WavTokenizer Decoder (凍結)      │
                    │   - 將 tokens 轉回音頻              │
                    └─────────────────┬──────────────────┘
                                      │
                           Denoised Audio (輸出)

Target (Clean Audio) ────────────────┐
                                      │
                    ┌─────────────────▼──────────────────┐
                    │   WavTokenizer Encoder (凍結)      │
                    │   - 提取乾淨音頻的 tokens           │
                    └─────────────────┬──────────────────┘
                                      │
                            Clean Tokens [B, L]
                                      │
                                  (作為監督信號)
```

#### 關鍵設計原則

**1. Token 空間的一致性**
```python
# ✅ 正確：所有 tokens 都來自同一個 WavTokenizer 的 codebook
noisy_tokens = wavtokenizer.encode(noisy_audio)   # 範圍 [0, 4095]
clean_tokens = wavtokenizer.encode(clean_audio)   # 範圍 [0, 4095]
predicted_tokens = transformer(noisy_tokens)       # 範圍 [0, 4095]

# 三者都在同一個 token 空間，WavTokenizer Decoder 可以理解
```

**2. 訓練目標**
```python
# Transformer 的學習目標：
# 給定 noisy_tokens，預測出接近 clean_tokens 的 tokens

Loss = CrossEntropy(predicted_logits, clean_tokens)  # 主要損失
     + L2_Embed(predicted_embed, clean_embed)        # 聲學相似性
     + Coherence(predicted_tokens)                   # 時間平滑
     + Manifold(predicted_tokens, noisy_tokens)      # 正則化
```

**3. Decoder 的兼容性**

**✅ 設計保證**：
- Transformer 輸出的 tokens 範圍：[0, 4095]
- WavTokenizer Encoder 產生的 tokens 範圍：[0, 4095]
- WavTokenizer Decoder 能理解的 tokens 範圍：[0, 4095]
- **完全一致！**

**代碼證據（第 348-356 行）**：
```python
def encode_audio_to_tokens(self, audio):
    """使用 WavTokenizer Encoder 將音頻轉換為 tokens"""
    with torch.no_grad():
        features, discrete_code = self.wavtokenizer.encode_infer(
            audio, bandwidth_id=torch.tensor([0])
        )
        tokens = discrete_code[0]  # 提取第一層的 codes
        
        tokens = tokens.long()
        # 確保 tokens 在詞彙範圍內 [0, 4095]
        tokens = torch.clamp(tokens, 0, self.codebook_size - 1)
        
        return tokens  # ✅ 返回 [0, 4095] 範圍的 tokens
```

**代碼證據（第 358-377 行）**：
```python
def decode_tokens_to_audio(self, tokens):
    """使用 WavTokenizer Decoder 將 tokens 轉換為音頻"""
    with torch.no_grad():
        # tokens 形狀：[batch_size, seq_len]，值範圍 [0, 4095]
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)  # [1, batch_size, seq_len]
        
        # ✅ WavTokenizer 的 codes_to_features 接受 [0, 4095] 的 tokens
        features = self.wavtokenizer.codes_to_features(tokens)
        
        # ✅ WavTokenizer 的 decoder 將 features 轉回音頻
        audio = self.wavtokenizer.decoder(features)
        
        return normalize_audio_dimensions(audio)
```

#### 潛在問題檢查

**⚠️ 需要確認的點**：

**1. Transformer 輸出是否會產生越界的 tokens？**

檢查代碼（第 283 行）：
```python
# Output projection
self.output_projection = nn.Linear(d_model, self.codebook_size)
# ✅ 輸出維度 = 4096（codebook_size），對應 [0, 4095]
```

取 argmax 後（第 724 行）：
```python
predicted_tokens = torch.argmax(logits, dim=-1)
# logits shape: [B, L, 4096]
# argmax 結果: [B, L]，值範圍 [0, 4095] ✅
```

**2. 是否有額外的 token 處理導致不兼容？**

檢查特殊 tokens（第 232-235 行）：
```python
self.pad_token = self.codebook_size      # 4096
self.sos_token = self.codebook_size + 1  # 4097
self.eos_token = self.codebook_size + 2  # 4098
```

**⚠️ 關鍵：這些特殊 tokens 只在訓練時使用（teacher forcing），不會傳給 Decoder！**

檢查推理流程（第 594-604 行）：
```python
else:  # 推理模式
    # Step 2: Transformer 降噪
    denoised_tokens = self.forward_transformer(noisy_tokens)
    
    # Step 3: 將 denoised tokens 轉換回音頻
    denoised_audio = self.decode_tokens_to_audio(denoised_tokens)
    
    return {
        'denoised_audio': denoised_audio,
        'denoised_tokens': denoised_tokens,  # ✅ 只包含 [0, 4095] 的 tokens
        'noisy_tokens': noisy_tokens
    }
```

**✅ 確認：特殊 tokens (4096-4098) 不會傳給 WavTokenizer Decoder**

---

## 📊 總結表格

| 項目 | 狀態 | 說明 |
|------|------|------|
| **驗證損失回傳** | ❌ → ✅ | 已修復，添加 `return_logits=True` 參數 |
| **Codebook 凍結** | ✅ | 100% 使用預訓練 codebook，完全凍結 |
| **Token 兼容性** | ✅ | Transformer 輸出 [0-4095]，WavTokenizer 完全兼容 |
| **WavTokenizer 凍結** | ✅ | Encoder + Decoder 完全凍結 |
| **特殊 Token 處理** | ✅ | PAD/SOS/EOS 只在訓練時用，不傳給 Decoder |

---

## 🎯 核心問題回答

### Q1: 之前訓練不好是因為沒有正確回傳嗎？

**A1**: **部分正確**。驗證損失失效導致：
- ❌ 無法判斷過擬合
- ❌ Best Model 選擇錯誤
- ❌ 無法調整訓練策略

但訓練本身（Train Loss 3.61）可能已經學到了降噪能力，需要聽音頻樣本確認。

### Q2: 確認完全是使用預訓練 codebook 凍結的嗎？

**A2**: **100% 確認**！
- ✅ Codebook 從 WavTokenizer 提取
- ✅ `freeze=True` 完全凍結
- ✅ WavTokenizer 整個凍結
- ✅ 只有 Transformer 可訓練

### Q3: Token 應該是 WavTokenizer Decoder 看得懂的對吧？

**A3**: **完全正確**！
- ✅ Transformer 輸出 [0-4095]
- ✅ WavTokenizer Encoder/Decoder 使用同樣範圍
- ✅ 特殊 tokens 不傳給 Decoder
- ✅ Token 空間完全兼容

---

## 🚀 下一步建議

1. **立即執行**：聽取 epoch 100/200/300 的音頻樣本
   - 評估降噪效果
   - 檢查頻譜連續性

2. **根據音頻質量決定**：
   - ✅ **若質量好** → 修復驗證後繼續訓練到 500-600 epochs
   - ⚠️ **若質量一般** → 重新訓練，從頭記錄驗證損失
   - ❌ **若質量差** → 檢討模型設計或損失函數

3. **確認機制**：運行測試腳本驗證 codebook 確實凍結

**您想要我幫您：**
1. 創建 codebook 凍結驗證腳本？
2. 創建音頻質量評估腳本？
3. 直接重新啟動訓練？
