# Codebook Embedding 架構設計文檔

## 📅 實驗日期
2025-10-17

## 🎯 核心洞察

**問題**：原始設計使用隨機初始化的 `nn.Embedding` 層來學習 token 表示

**洞察**：WavTokenizer 已經提供了一個意義豐富、結構良好的 token 空間（它的 codebook），我們應該直接利用這個預訓練的表示，而不是重新學習。

## 🏗️ 架構對比

### ❌ 原始架構（錯誤）

```python
class TTT2TokenModel(nn.Module):
    def __init__(self, ...):
        # 創建隨機初始化的 embedding
        self.token_embedding = nn.Embedding(4096, 512)  # 丟失預訓練語義！
    
    def forward(self, audio):
        tokens = self.encode_audio(audio)
        # 使用隨機 embedding
        features = self.token_embedding(tokens)
        ...
```

**問題**：
1. ❌ 丟失了 WavTokenizer 從大量數據中學到的語義結構
2. ❌ Token embedding 需要從零開始學習
3. ❌ 增加了訓練難度和時間
4. ❌ 可能無法充分利用 WavTokenizer 的能力

### ✅ 改進架構（正確）

```python
class TTT2TokenModel(nn.Module):
    def __init__(self, ...):
        # 提取 WavTokenizer 的預訓練 codebook
        self.codebook_weights = self._extract_codebook_weights()  # [4096, 512]
        # 凍結 codebook（保留預訓練語義）
        self.codebook_weights = self.codebook_weights.detach()
        
        # 可選：維度投影（如果需要不同的 embed_dim）
        self.codebook_projection = nn.Linear(512, embed_dim) if embed_dim != 512 else nn.Identity()
    
    def _extract_codebook_weights(self):
        """從 WavTokenizer 提取預訓練的 codebook"""
        vq_layers = self.wavtokenizer.feature_extractor.encodec.quantizer.vq.layers
        codebook_weights = torch.cat([vq.codebook for vq in vq_layers], dim=0)
        return codebook_weights
    
    def forward(self, audio):
        tokens = self.encode_audio(audio)
        
        # 使用預訓練的 codebook embedding
        features = F.embedding(tokens, self.codebook_weights)  # 保留語義！
        features = self.codebook_projection(features)
        
        # Feature enhancement
        enhanced_features = self.feature_enhancer(features)
        
        # 量化回 token 空間（通過最近鄰搜索）
        enhanced_tokens = self.quantize_features_to_tokens(enhanced_features)
        ...
```

**優點**：
1. ✅ 保留了 WavTokenizer 的預訓練語義結構
2. ✅ Token embedding 無需學習，直接可用
3. ✅ 降低訓練難度
4. ✅ 充分利用預訓練模型的能力
5. ✅ 更好的初始化，加快收斂

## 🔬 技術細節

### 1. Codebook 提取

```python
def _extract_codebook_weights(self):
    """
    從 WavTokenizer 提取預訓練的 codebook weights
    
    參考：decoder/pretrained.py line 239
    tmp = torch.cat([vq.codebook for vq in self.feature_extractor.encodec.quantizer.vq.layers], dim=0)
    """
    vq_layers = self.wavtokenizer.feature_extractor.encodec.quantizer.vq.layers
    codebook_weights = torch.cat([vq.codebook for vq in vq_layers], dim=0)
    
    # 輸出：[codebook_size, codebook_dim]
    # 例如：[4096, 512]
    return codebook_weights.detach()  # 凍結權重
```

**關鍵點**：
- `vq.codebook` 是預訓練的 vector quantization codebook
- 每個 entry 是一個 512 維的向量，代表一個 token 的語義
- 這些向量是從大量音頻數據中學習得到的

### 2. Token → Feature Embedding

```python
def forward(self, audio):
    # 編碼為 tokens
    tokens = self.encode_audio(audio)  # [B, L]
    
    # 使用預訓練 codebook 作為 embedding
    features = F.embedding(tokens, self.codebook_weights)  # [B, L, 512]
    
    # 可選：投影到不同維度
    features = self.codebook_projection(features)  # [B, L, embed_dim]
```

**為什麼使用 `F.embedding` 而不是 `nn.Embedding`**：
- `nn.Embedding` 會創建可訓練的參數
- `F.embedding` 是功能性 API，使用提供的權重進行查表
- 我們要保持 codebook 凍結，所以使用 `F.embedding`

### 3. Feature → Token 量化

```python
def quantize_features_to_tokens(self, features):
    """
    將增強的特徵量化回 discrete tokens
    通過在 codebook 中尋找最近鄰
    """
    # features: [B, L, codebook_dim]
    # codebook_weights: [codebook_size, codebook_dim]
    
    # 計算歐氏距離
    distances = torch.cdist(features, self.codebook_weights)  # [B*L, codebook_size]
    
    # 找最近的 codebook entry
    tokens = torch.argmin(distances, dim=-1)  # [B, L]
    
    return tokens
```

**為什麼這樣設計**：
- 增強後的特徵可能不再精確對應某個 codebook entry
- 通過最近鄰搜索找到最接近的 token
- 確保輸出的 tokens 是有效的（在 codebook 中存在）

## 📊 實驗驗證

### 測試結果

```bash
測試 TTT2TokenModel 初始化...
成功提取 codebook weights: shape=torch.Size([4096, 512])
- 來自 1 個 VQ 層
- 總共 4096 個 codes
- 每個 code 維度: 512
Codebook 維度匹配: 512

測試前向傳播...
Enhanced audio shape: torch.Size([2, 1, 24000])
Enhanced tokens shape: torch.Size([2, 75])
Noisy tokens shape: torch.Size([2, 75])

✅ 所有測試通過！
```

### 參數統計

```
總參數量: 87,910,885
可訓練參數量: 7,358,465 (8.4%)
```

**關鍵觀察**：
- WavTokenizer 參數（凍結）：80,552,420 (91.6%)
- Feature Enhancer（可訓練）：7,358,465 (8.4%)
- Codebook weights 不在可訓練參數中（已凍結）

## 🎨 完整數據流

```
Input Audio [B, 1, T]
    ↓
[WavTokenizer Encoder - 凍結]
    ↓
Noisy Tokens [B, L]
    ↓
[F.embedding with pretrained codebook]  ← 關鍵：使用預訓練表示
    ↓
Noisy Features [B, L, 512]
    ↓
[Codebook Projection]
    ↓
Noisy Features [B, L, embed_dim]
    ↓
[+ Positional Encoding]
    ↓
[Feature Enhancer - 可訓練]  ← 唯一可訓練的部分
    ↓
Enhanced Features [B, L, embed_dim]
    ↓
[Feature to Codebook Projection]
    ↓
Enhanced Codebook Features [B, L, 512]
    ↓
[Quantize via Nearest Neighbor Search]  ← 關鍵：量化回 token 空間
    ↓
Enhanced Tokens [B, L]
    ↓
[WavTokenizer Decoder - 凍結]
    ↓
Enhanced Audio [B, 1, T]
```

## 💡 設計理念

### 1. 利用預訓練知識

**原則**：不要重新發明輪子
- WavTokenizer 已經從大量數據中學習了良好的音頻表示
- 我們的任務是增強這些表示，而不是重新學習它們

### 2. 最小化可訓練參數

**原則**：只訓練必要的部分
- 凍結 WavTokenizer encoder/decoder（已經很好了）
- 凍結 codebook weights（預訓練的語義）
- 只訓練 Feature Enhancer（我們要學習的部分）

### 3. 保持語義一致性

**原則**：輸入和輸出使用相同的語義空間
- Noisy tokens → Codebook embedding（預訓練空間）
- Enhanced features → Quantize to codebook（回到預訓練空間）
- 確保整個系統在一致的語義框架下運作

## 🔄 與離散化訓練的對比

### 離散化訓練的問題

```python
# 離散化方法
tokens = torch.argmax(logits, dim=-1)  # 不可微分
loss = criterion(tokens, target_tokens)  # 無法反向傳播
```

### Codebook Embedding 的優勢

```python
# Codebook embedding 方法
# 前向：連續特徵 → 量化 tokens
enhanced_features = enhancer(noisy_features)  # 可微分
enhanced_tokens = quantize(enhanced_features)  # 離散化在最後

# 反向：直接在連續特徵空間計算損失
loss = MSE(enhanced_features, target_features)  # 完全可微分
```

**關鍵區別**：
- 離散化：在中間步驟引入不可微分操作
- Codebook：在連續空間訓練，只在最後量化

## 📈 預期改進

基於這個架構改進，我們預期：

1. **更快收斂**
   - 使用預訓練 embedding 作為初始化
   - 減少需要學習的內容

2. **更好性能**
   - 保留 WavTokenizer 的語義結構
   - 增強而不是重建表示

3. **更穩定訓練**
   - 連續特徵空間的損失計算
   - 避免離散化的梯度問題

4. **更小的模型**
   - 只訓練 Feature Enhancer（8.4% 參數）
   - Codebook 凍結（無需存儲梯度）

## 🚀 下一步

1. ✅ 完成架構修改
2. ⏭️ 運行完整訓練實驗
3. ⏭️ 對比原始架構 vs 改進架構的性能
4. ⏭️ 分析 token 分佈和語義保留情況

## 📚 參考

- WavTokenizer codebook extraction: `decoder/pretrained.py` line 239
- Vector Quantization: `encoder/quantization/core_vq.py`
- Original issue discussion: User feedback on random embedding initialization

## 🏆 核心貢獻

**這個改進的核心價值**：
> 通過直接使用 WavTokenizer 的預訓練 codebook 作為 token embedding，我們避免了重新學習已知的語義結構，讓模型專注於學習如何增強特徵（從 noisy 到 clean），而不是學習什麼是有意義的音頻表示。

---

**實驗編號**: TTT2-Token-Codebook-001  
**日期**: 2025-10-17  
**狀態**: ✅ 架構實現完成，待訓練驗證
# Codebook Embedding 架構改進總結

## 📅 日期
2025-10-17

## 🎯 核心問題

**用戶洞察**：
> WavTokenizer 已經提供了一個意義豐富、結構良好的 token 空間（它的 codebook）。你不應該重新創建一個隨機初始化的 nn.Embedding 層來學習一套全新的 token 表示，而應該直接利用 WavTokenizer 已經訓練好的 codebook 作為你的 Embedding 層。

## ✅ 解決方案

### 改進前
```python
self.token_embedding = nn.Embedding(4096, 512)  # ❌ 隨機初始化
```

### 改進後
```python
self.codebook_weights = self._extract_codebook_weights()  # ✅ 使用預訓練
noisy_features = F.embedding(noisy_tokens, self.codebook_weights)  # ✅ 凍結
```

## 💡 關鍵優勢

1. **保留語義**：使用 WavTokenizer 學到的音頻表示
2. **更快收斂**：預訓練 embedding 作為初始化
3. **更少參數**：只訓練 Feature Enhancer（8.4%）
4. **更穩定**：在一致的語義空間內操作

## 📊 實驗結果

```
✅ Codebook 提取成功：[4096, 512]
✅ 前向傳播測試通過
✅ 參數統計：7.36M 可訓練 / 87.9M 總參數
```

## 📚 文檔

- `CODEBOOK_EMBEDDING_ARCHITECTURE.md`：完整技術文檔
- `ttt2_token.py`：實現代碼
- Git Commit: `2281c8f`

## 🚀 下一步

運行完整訓練實驗，驗證性能改進。

---

**這個改進體現了深度學習的核心原則**：
> 不要重新發明輪子。利用已有的預訓練知識，專注於學習你真正需要的部分。
