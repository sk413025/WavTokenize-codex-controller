# Fusion 策略 & Fine-tune ECAPA 深入分析

## 📊 當前實驗結果 (已完成)

### 最終性能
- ✅ **最佳驗證準確率**: 39.29%
- ✅ **相比 Baseline**: +1.10% (38.19% → 39.29%)
- ⚠️ **泛化差距**: ~16% (Train 55% vs Val 39%)
- ⚠️ **提升幅度**: 略有提升，但不顯著

### 訓練腳本的建議
```
⚠️  建議：嘗試改進 fusion 策略或 fine-tune ECAPA
```

讓我們深入分析這兩個建議的含義、可行性和預期效果。

---

## 🔀 Part 1: 改進 Fusion 策略

### 當前 Fusion 實現

查看 [model_zeroshot.py](model_zeroshot.py:100-114)：

```python
# Step 1: Token Embedding (Frozen Codebook Lookup)
token_emb = self.codebook[noisy_token_ids]  # (B, T, 512)

# Step 2: Speaker Embedding Projection & Broadcasting
speaker_emb = self.speaker_proj(speaker_embedding)  # (B, 256) -> (B, 512)
speaker_emb = speaker_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, 512)

# Step 3: Fusion (Additive)
combined_emb = token_emb + speaker_emb  # (B, T, 512)
```

**當前策略**: **Simple Addition (簡單加法)**

### 問題診斷

#### 為什麼 Simple Addition 可能不夠好？

1. **缺乏自適應性**
   - Speaker embedding 對所有 token 的貢獻權重相同
   - 無法根據 token 內容動態調整 speaker 信息的重要性

2. **信息融合不靈活**
   - 加法假設 token 和 speaker 特徵在同一語義空間
   - 實際上它們可能編碼了不同層次的信息

3. **可能導致信息稀釋**
   - Token embedding 包含音頻內容信息
   - Speaker embedding 包含說話人特徵
   - 簡單相加可能互相干擾

---

### 改進方案

#### 方案 A: **Gated Fusion** (門控融合) ⭐⭐⭐⭐⭐

**概念**：讓模型學習動態調節 token 和 speaker 信息的權重

```python
class GatedFusion(nn.Module):
    """
    門控融合機制
    讓模型自適應地決定每個位置使用多少 token vs speaker 信息
    """
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, token_emb, speaker_emb):
        # token_emb: (B, T, 512)
        # speaker_emb: (B, T, 512)

        # 計算門控權重
        concat = torch.cat([token_emb, speaker_emb], dim=-1)  # (B, T, 1024)
        gate = self.gate(concat)  # (B, T, 512), range [0, 1]

        # 融合
        fused = gate * token_emb + (1 - gate) * speaker_emb
        return fused
```

**優點**：
- ✅ 動態調節權重，靈活性高
- ✅ 每個 token 位置可以有不同的融合比例
- ✅ 端到端學習，無需手動調整

**預期效果**：
- 驗證準確率: 39.29% → **40.5-41.5%** (+1.2-2.2%)
- 泛化差距: 16% → 14-15%

**實現成本**: 低 (增加 ~1M 參數)

---

#### 方案 B: **Cross-Attention Fusion** (交叉注意力融合) ⭐⭐⭐⭐

**概念**：使用 attention 機制讓每個 token 動態查詢 speaker 信息

```python
class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合
    Token 作為 Query，Speaker 作為 Key/Value
    """
    def __init__(self, d_model, nhead=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_emb, speaker_emb):
        # token_emb: (B, T, 512) as Query
        # speaker_emb: (B, 1, 512) as Key/Value (unsqueezed)

        speaker_emb_expanded = speaker_emb.unsqueeze(1)  # (B, 1, 512)

        # Cross-attention
        attn_output, _ = self.cross_attn(
            query=token_emb,
            key=speaker_emb_expanded,
            value=speaker_emb_expanded
        )

        # Residual connection
        fused = self.norm(token_emb + attn_output)
        return fused
```

**優點**：
- ✅ 更強的表達能力
- ✅ 每個 token 根據需要選擇性提取 speaker 信息
- ✅ 類似 Transformer 中的 encoder-decoder attention

**缺點**：
- ⚠️ 增加較多參數 (~2M)
- ⚠️ 計算量稍高

**預期效果**：
- 驗證準確率: 39.29% → **41-42%** (+1.7-2.7%)
- 泛化差距: 16% → 13-14%

**實現成本**: 中 (需要改動較多)

---

#### 方案 C: **FiLM (Feature-wise Linear Modulation)** ⭐⭐⭐

**概念**：Speaker embedding 生成縮放和偏移參數，調製 token embedding

```python
class FiLMFusion(nn.Module):
    """
    FiLM: Feature-wise Linear Modulation
    Speaker embedding 控制如何調製 token features
    """
    def __init__(self, d_model, speaker_dim):
        super().__init__()
        self.gamma_proj = nn.Linear(speaker_dim, d_model)  # scale
        self.beta_proj = nn.Linear(speaker_dim, d_model)   # shift

    def forward(self, token_emb, speaker_embedding):
        # token_emb: (B, T, 512)
        # speaker_embedding: (B, 256)

        gamma = self.gamma_proj(speaker_embedding).unsqueeze(1)  # (B, 1, 512)
        beta = self.beta_proj(speaker_embedding).unsqueeze(1)    # (B, 1, 512)

        # FiLM transformation
        fused = gamma * token_emb + beta
        return fused
```

**優點**：
- ✅ 簡單高效
- ✅ 參數量少 (~260K)
- ✅ 已在條件生成任務中驗證有效

**預期效果**：
- 驗證準確率: 39.29% → **40-41%** (+0.7-1.7%)
- 泛化差距: 16% → 14-15%

**實現成本**: 低

---

#### 方案 D: **Concatenation + Projection** ⭐⭐⭐

**概念**：拼接後用 MLP 學習融合

```python
class ConcatFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, token_emb, speaker_emb):
        concat = torch.cat([token_emb, speaker_emb], dim=-1)  # (B, T, 1024)
        fused = self.proj(concat)  # (B, T, 512)
        return fused
```

**優點**：
- ✅ 非常靈活
- ✅ 讓模型完全自主學習融合策略

**缺點**：
- ⚠️ 參數量較多 (~2M)

**預期效果**：
- 驗證準確率: 39.29% → **40-40.5%** (+0.7-1.2%)

---

### Fusion 方案對比

| 方案 | 參數量 | 計算量 | 實現難度 | 預期提升 | 推薦度 |
|------|--------|--------|----------|----------|--------|
| **Simple Addition (當前)** | 0 | 最低 | ✅ 簡單 | - | - |
| **Gated Fusion** | ~1M | 低 | ✅ 簡單 | +1.2-2.2% | ⭐⭐⭐⭐⭐ |
| **Cross-Attention** | ~2M | 中 | ⚠️ 中等 | +1.7-2.7% | ⭐⭐⭐⭐ |
| **FiLM** | ~260K | 最低 | ✅ 簡單 | +0.7-1.7% | ⭐⭐⭐ |
| **Concat + Projection** | ~2M | 中 | ✅ 簡單 | +0.7-1.2% | ⭐⭐⭐ |

### 我的推薦順序

1. **Gated Fusion** ⭐⭐⭐⭐⭐
   - 最佳性價比
   - 實現簡單，效果顯著
   - 計算開銷小

2. **Cross-Attention** ⭐⭐⭐⭐
   - 如果 Gated Fusion 效果不夠好，嘗試這個
   - 理論上限更高

3. **FiLM** ⭐⭐⭐
   - 作為快速驗證的備選方案

---

## 🎤 Part 2: Fine-tune ECAPA-TDNN

### 當前 ECAPA 使用方式

查看 [preprocess_zeroshot_cache.py](preprocess_zeroshot_cache.py)：

```python
# Speaker Encoder (ECAPA-TDNN) 保持在 CPU 並凍結
speaker_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
).to('cpu')
speaker_encoder.eval()

# 在預處理時提取 speaker embedding，保存到緩存
```

**當前狀態**:
- ✅ ECAPA 完全凍結 (frozen)
- ✅ Speaker embedding 在預處理時提取，訓練時直接使用
- ✅ ECAPA 不參與梯度更新

---

### Fine-tune ECAPA 的動機

#### 為什麼可能需要 Fine-tune？

1. **Domain Mismatch** (領域不匹配)
   - ECAPA 在 VoxCeleb 上預訓練（乾淨語音）
   - 你的數據包含**噪音語音** (box, papercup, plastic)
   - 噪音可能影響 speaker embedding 質量

2. **Task-specific Adaptation** (任務特定適應)
   - ECAPA 訓練目標：speaker verification（說話人驗證）
   - 你的任務：speaker-conditioned denoising（說話人條件降噪）
   - 任務目標不完全一致

3. **End-to-end Optimization** (端到端優化)
   - 當前 speaker embedding 是固定的
   - Fine-tune 可以讓 speaker encoder 學習更適合降噪的特徵

---

### Fine-tune ECAPA 的實施方案

#### 方案 A: **全模型 Fine-tune** ⚠️ 不推薦

**實現**：解凍整個 ECAPA-TDNN，參與訓練

```python
# 在 train_zeroshot_full_cached.py 中
speaker_encoder = EncoderClassifier.from_hparams(...)
speaker_encoder.train()  # 設為訓練模式

# 在優化器中包含 speaker_encoder
optimizer = optim.AdamW([
    {'params': model.parameters()},
    {'params': speaker_encoder.parameters(), 'lr': 1e-5}  # 小學習率
])
```

**問題**：
- ❌ **記憶體爆炸**: ECAPA 14M 參數，無法在訓練時放入 GPU
- ❌ **過擬合風險極高**: 16K 樣本不足以 fine-tune 14M 參數的模型
- ❌ **訓練極慢**: 需要在每個 batch 重新計算 speaker embedding

**結論**: **不可行**

---

#### 方案 B: **只 Fine-tune ECAPA 的最後幾層** ⭐⭐⭐

**實現**：只解凍 ECAPA 的輸出層（projection layer）

```python
# 凍結大部分層，只訓練最後的投影層
speaker_encoder = EncoderClassifier.from_hparams(...)

# 凍結所有參數
for param in speaker_encoder.parameters():
    param.requires_grad = False

# 只解凍最後的投影層
for param in speaker_encoder.mods.embedding_model.parameters():
    param.requires_grad = True

speaker_encoder.train()
```

**優點**：
- ✅ 參數量小（只 fine-tune ~500K 參數）
- ✅ 過擬合風險較低
- ✅ 可以在 GPU 上運行

**問題**：
- ⚠️ 仍需在訓練時計算 speaker embedding（無法使用預處理緩存）
- ⚠️ 訓練速度大幅下降（23x 加速 → 可能只剩 5-8x）
- ⚠️ GPU 記憶體需求增加

**預期效果**：
- 驗證準確率: 39.29% → **40-41%** (+0.7-1.7%)
- 訓練時間: 2.5 小時 → **8-12 小時** (增加 3-5x)

**推薦度**: ⭐⭐⭐ (中等)

---

#### 方案 C: **使用 Adapter Layers** ⭐⭐⭐⭐

**概念**：不改動 ECAPA，而是在 speaker embedding 之後添加可訓練的 adapter

```python
class SpeakerAdapter(nn.Module):
    """
    輕量級 Adapter，在 ECAPA 輸出後進行任務特定調整
    """
    def __init__(self, speaker_dim=192, bottleneck_dim=64):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(speaker_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, speaker_dim)
        )
        self.layer_norm = nn.LayerNorm(speaker_dim)

    def forward(self, speaker_emb):
        # Residual connection
        adapted = speaker_emb + self.adapter(speaker_emb)
        return self.layer_norm(adapted)
```

**使用方式**：
```python
# 在 model_zeroshot.py 中添加
self.speaker_adapter = SpeakerAdapter(speaker_dim=256, bottleneck_dim=64)

def forward(self, noisy_token_ids, speaker_embedding, ...):
    # 先通過 adapter 調整 speaker embedding
    speaker_embedding = self.speaker_adapter(speaker_embedding)

    # 然後進行 fusion
    ...
```

**優點**：
- ✅ **仍然可以使用預處理緩存！** (關鍵優勢)
- ✅ 參數量極少 (~12K)
- ✅ 訓練速度不受影響
- ✅ ECAPA 保持凍結，避免破壞預訓練知識

**預期效果**：
- 驗證準確率: 39.29% → **39.8-40.5%** (+0.5-1.2%)
- 訓練時間: 2.5 小時 (不變)

**推薦度**: ⭐⭐⭐⭐ (高度推薦)

---

### ECAPA Fine-tune 方案對比

| 方案 | 可訓練參數 | 訓練時間 | 可用緩存 | 過擬合風險 | 預期提升 | 推薦度 |
|------|-----------|----------|----------|-----------|----------|--------|
| **Frozen (當前)** | 0 | 2.5h | ✅ | - | - | - |
| **全模型 Fine-tune** | 14M | 無法實現 | ❌ | 極高 | - | ❌ |
| **Fine-tune 最後幾層** | ~500K | 8-12h | ❌ | 中 | +0.7-1.7% | ⭐⭐⭐ |
| **Adapter Layers** | ~12K | 2.5h | ✅ | 極低 | +0.5-1.2% | ⭐⭐⭐⭐ |

### 我的推薦

**使用 Adapter Layers** ⭐⭐⭐⭐
- 保持訓練速度優勢
- 可以繼續使用預處理緩存
- 風險低，收益穩定

---

## 🎯 綜合實驗計劃

### 優先級排序

#### 🔥 第一優先級：num_layers=3 (正在進行)
**理由**: 解決核心的過擬合問題
**預期**: Val Acc 39.29% → 40-42%

#### 🔥 第二優先級：Gated Fusion
**理由**:
- 改善 speaker conditioning 效果
- 實現簡單，效果顯著
- 不影響訓練速度

**預期**:
- 在 num_layers=3 基礎上再提升 1-2%
- 最終 Val Acc: 41-43%

#### 📅 第三優先級：Speaker Adapter
**理由**:
- 進一步優化 speaker embedding
- 保持訓練速度
- 風險極低

**預期**:
- 在 Gated Fusion 基礎上再提升 0.5-1%
- 最終 Val Acc: 41.5-44%

#### 🔮 第四優先級：Cross-Attention Fusion
**理由**:
- 如果 Gated Fusion 效果不夠好，嘗試更強的機制
- 理論上限更高

**預期**:
- 可能達到 Val Acc: 42-45%

---

## 📝 具體實施步驟

### Step 1: 完成當前 num_layers=3 實驗
**等待時間**: ~1.8 小時
**預計完成**: 今天 15:00

### Step 2: 實施 Gated Fusion + num_layers=3
**實現時間**: ~30 分鐘
**訓練時間**: ~1.8 小時
**預計完成**: 今天 17:00

### Step 3: (可選) 添加 Speaker Adapter
**實現時間**: ~20 分鐘
**訓練時間**: ~1.8 小時
**預計完成**: 今天 19:00

---

## 💡 關鍵見解

### 為什麼訓練腳本建議這兩個方向？

1. **Fusion 策略**: 當前 39.29% vs Baseline 38.19% 提升有限
   - 表示 speaker conditioning 機制可能不夠有效
   - Simple addition 可能無法充分利用 speaker 信息

2. **Fine-tune ECAPA**:
   - ECAPA 在乾淨語音上訓練，可能不適應噪音語音
   - 但直接 fine-tune 風險高，建議用 Adapter

### 哪個改進更重要？

**我的判斷**: **Fusion 策略 > ECAPA Fine-tune**

**理由**:
1. Fusion 是模型核心機制，影響更直接
2. Adapter 只是微調，收益較小
3. 實驗順序應該先解決主要問題（Fusion），再優化細節（Adapter）

### 與 num_layers=3 的關係

這三個改進方向**互相獨立且可疊加**：
- num_layers=3: 減少過擬合 ← **解決泛化問題**
- Gated Fusion: 改善 speaker conditioning ← **提升模型能力**
- Speaker Adapter: 優化 speaker embedding ← **微調輸入質量**

組合使用效果最佳！

---

## 📊 預期最終效果

### 保守估計
- num_layers=3: Val Acc **40-41%**
- + Gated Fusion: Val Acc **41.5-42.5%**
- + Speaker Adapter: Val Acc **42-43%**

### 樂觀估計
- num_layers=3: Val Acc **41-42%**
- + Gated Fusion: Val Acc **42.5-44%**
- + Speaker Adapter: Val Acc **43-45%**

**最終目標**: **Val Acc ≥ 43%** (+4.81% vs Baseline)

---

生成時間: 2025-11-03 13:30
當前完成: num_layers=4, Val Acc 39.29%
進行中: num_layers=3 實驗
下一步: Gated Fusion + num_layers=3
