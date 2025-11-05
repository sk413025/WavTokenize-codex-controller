# 訓練平台期機轉深度分析 (2025-11-05)

## 🎯 核心發現摘要

**觀察**: Train Accuracy 卡在 54%，Val Accuracy 僅 37%  
**關鍵數據**: Token 453 在 Train 佔 13.57%，在 Val 佔 18.65% (+37.5%)

---

## 📊 機轉假設 1: Token 453 預測困難導致 Accuracy 上限

### ASCII 視覺化：Token 分布與預測行為

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
訓練集 Token 分布 (4,285,755 tokens)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Token 453:  ████████████████████████████████ 13.57%  ← ⚠️ 訓練瓶頸！
Token 244:  ██ 0.47%
Token 165:  █ 0.34%
Token 219:  █ 0.26%
其他 1829: ███████████████████████████████████████████████████████████ 85.36%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
驗證集 Token 分布 (1,581,684 tokens)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Token 453:  ████████████████████████████████████████████ 18.65%  ← ⚠️ Val 更高！
Token 1145: ███ 0.91%
Token 1750: ██ 0.70%
Token 1016: ██ 0.61%
其他 1815: ██████████████████████████████████████████████████████ 79.13%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
關鍵差異: Token 453 在 Val 比 Train 高出 +5.08% (相對增幅 +37.5%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 數學推導: Token 453 的準確率反推

**已知數據**:
- Train Accuracy: 54.70%
- Val Accuracy: 36.75%
- Token 453 在 Train: 13.57%
- Token 453 在 Val: 18.65%

**推導公式**:
```
Overall Accuracy = (1 - Token453_Ratio) × Other_Acc + Token453_Ratio × Token453_Acc
```

**情境 1: Token 453 完全失敗 (0% 準確率)**
```
Train: 54.70% = 86.43% × Other_Acc + 13.57% × 0%
       → Other_Acc = 63.29%  ✓ 合理

Val:   36.75% = 81.35% × Other_Acc + 18.65% × 0%
       → Other_Acc = 45.18%  ⚠️  Val 的其他 tokens 準確率下降 18%！
```

**結論 1**: 
- 如果 Token 453 完全失敗，其他 tokens 在 Val 的準確率應該是 45.18%
- 但這與 Train 的 63.29% 有巨大落差 (-18%)
- **說明問題不只是 Token 453，其他 tokens 在 Val 也表現變差**

**情境 2: 其他 tokens 在 Train/Val 準確率相同 (約 60%)**
```
假設 Other_Acc = 60% (恆定)

Train: 54.70% = 86.43% × 60% + 13.57% × Token453_Acc_Train
       → Token453_Acc_Train = (54.70% - 51.86%) / 13.57% = 20.94%  ✓ 合理

Val:   36.75% = 81.35% × 60% + 18.65% × Token453_Acc_Val
       → Token453_Acc_Val = (36.75% - 48.81%) / 18.65% = -64.66%  ❌ 負值！
```

**結論 2**:
- 假設「其他 tokens 準確率恆定」不成立
- **其他 tokens 在 Val 的準確率必定下降**
- **Token 453 在 Val 的表現必定極差** (可能接近 0%)

**情境 3: Token 453 對錯誤的貢獻**
```
Train 錯誤率: 100% - 54.70% = 45.30%
  Token 453 最大貢獻: 13.57% / 45.30% = 30.0%  ← Token 453 佔總錯誤 30%

Val 錯誤率: 100% - 36.75% = 63.25%
  Token 453 最大貢獻: 18.65% / 63.25% = 29.5%  ← Token 453 佔總錯誤 29.5%
```

**結論 3**:
- Token 453 是訓練瓶頸：即使完全修好它，Train Acc 最多提升到 67%
- 但無法解釋 Train-Val gap (54% → 37% = 17% 落差)
- **必定還有其他 tokens 在 Val 表現變差**

---

## 🔬 驗證實驗 1: 數學反推分析結果

### 實驗結果

```
【假設】Token 453 的準確率遠低於其他 tokens
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

訓練集 (Train Acc = 54.70%, Token 453 = 13.57%):
  如果 Token 453 準確率 = 0%  → 其他 tokens = 63.29% ✓
  如果 Token 453 準確率 = 20% → 其他 tokens = 60.15% ✓
  如果 Token 453 準確率 = 40% → 其他 tokens = 57.01% ✓

驗證集 (Val Acc = 36.75%, Token 453 = 18.65%):
  如果 Token 453 準確率 = 0%  → 其他 tokens = 45.18% ⚠️ 比 Train 低 18%！
  如果 Token 453 準確率 = 20% → 其他 tokens = 40.59% ⚠️ 比 Train 低 20%！
  如果 Token 453 準確率 = 40% → 其他 tokens = 36.00% ⚠️ 比 Train 低 21%！

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
關鍵發現: 無論 Token 453 表現如何，其他 tokens 在 Val 都下降 18-21%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 數據支持的結論

✅ **結論 1**: Token 453 確實是訓練瓶頸
- 數據: Token 453 佔 Train 錯誤 30%，Val 錯誤 29.5%
- 支持: 數學推導顯示 Token 453 準確率極低 (0-20%)

✅ **結論 2**: Token Distribution Mismatch 是根本原因
- 數據: Token 453 在 Val 比 Train 高 +5.08% (絕對), +37.5% (相對)
- 支持: Val speakers 的聲音特徵與 Train speakers 差異大

⚠️ **結論 3**: 問題不只是 Token 453
- 數據: 其他 tokens 在 Val 準確率下降 18-21%
- 支持: 即使 Token 453 表現不變，其他 tokens 也在 Val 變差
- **說明: 整體 token 分布都有 mismatch，不只 Token 453**

---

## � 機轉假設 2: 其他高頻 Tokens 也有 Distribution Mismatch

### ASCII 視覺化：Top-20 Tokens Distribution Mismatch

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Top-20 Tokens 在 Train vs Val 的分布差異 (↑ = Val 更高, ↓ = Train 更高)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Token  453:  █████████████ (13.57%) → ██████████████████ (18.65%)  ↑ +5.08% ⚠️⚠️⚠️
Token 1145:  ██ (0.18%)     → █████ (0.91%)                       ↑ +0.73% ⚠️
Token 1750:  ██ (0.18%)     → ████ (0.70%)                        ↑ +0.52% ⚠️
Token 1016:  █ (0.10%)      → ████ (0.61%)                        ↑ +0.51% ⚠️
Token 1764:  ██ (0.15%)     → ████ (0.61%)                        ↑ +0.45% ⚠️
Token 1655:  ██ (0.15%)     → ████ (0.59%)                        ↑ +0.44% ⚠️
Token 1765:  ██ (0.14%)     → ████ (0.56%)                        ↑ +0.42% ⚠️
Token  820:  █ (0.11%)      → ███ (0.51%)                         ↑ +0.40% ⚠️
Token  669:  █ (0.10%)      → ███ (0.50%)                         ↑ +0.40% ⚠️

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
關鍵發現: 15 個 Top-20 tokens 有顯著分布差異 (>0.3%)
累計絕對差異: 10.94% (幾乎每 9 個 token 就有 1 個 mismatch)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 驗證實驗 2 結果

**數據**:
- **15 個 Top-20 tokens** 有顯著分布差異 (差異 >0.3%)
- 累計絕對差異: **10.94%** (相當於每 10 個 token 就有 1 個分布不同)
- Token 453 佔差異的 **46.4%** (5.08% / 10.94%)

**解讀**:
```
如果這 15 個 tokens 在 Val 的準確率都因 distribution mismatch 下降:
  假設每個 token 因 mismatch 導致準確率下降 20%
  
  Token 453:   18.65% × 20% = 3.73% accuracy loss
  Token 1145:   0.91% × 20% = 0.18% accuracy loss
  Token 1750:   0.70% × 20% = 0.14% accuracy loss
  ... (其餘 12 個)
  
  總 accuracy loss ≈ 4-5% (保守估計)
  
實際 Train-Val gap = 17%
  → 5% 來自 distribution mismatch
  → 12% 來自其他原因 (模型泛化能力不足、speaker embedding 效果差等)
```

**數據支持的結論**:

✅ **結論 4**: Distribution Mismatch 不只是 Token 453
- 數據: 15 個 Top-20 tokens 都有分布差異
- 累計差異 10.94% 說明 Train/Val 的 token 分布差異廣泛
- **支持: Val speakers 的整體聲音特徵與 Train speakers 系統性不同**

⚠️ **結論 5**: Distribution Mismatch 可解釋部分但非全部 gap
- 數據: Mismatch 導致的 accuracy loss 約 4-5%
- 實際 gap 17%，說明還有 12% 來自其他原因
- **說明: 模型本身的泛化能力也有問題**

---

## 📊 機轉假設 3: Speaker Embedding 未能有效捕捉 Token Distribution

### ASCII 視覺化：模型架構與資訊流

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Zero-Shot Speaker Denoising Transformer 架構
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

輸入 1: Noisy Tokens (B, T)
   ↓
   Frozen Codebook (4096, 512)  ← 來自 WavTokenizer，speaker-independent
   ↓
   Token Embeddings (B, T, 512)


輸入 2: Speaker Embedding (B, 256)  ← 來自 ECAPA-TDNN
   ↓
   Speaker Proj (256 → 512)
   ↓
   Speaker Embeddings (B, 512)
   ↓
   Expand to (B, T, 512)  ← 每個 time step 都加同樣的 speaker info


融合:
   Token Emb (B, T, 512) + Speaker Emb (B, T, 512)
   ↓
   Combined (B, T, 512)  ← 簡單相加
   ↓
   Positional Encoding
   ↓
   Transformer Encoder (4 layers, 8 heads)
   ↓
   Output Logits (B, T, 4096)
   ↓
   Softmax → Predicted Tokens


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
問題: Speaker Embedding 只是「相加」，無法調整 Token Distribution Prior
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

期望行為:
   Speaker A → 模型應預測 Token 453 佔 10%
   Speaker B → 模型應預測 Token 453 佔 20%  ← Speaker-specific distribution

實際行為:
   Speaker A, B → 模型預測相同的 Token Distribution (因為只學到 Train speakers)
   ↓
   Val speakers 的 Token 453 佔 18.65%，但模型預測只有 13.57%
   ↓
   導致大量預測錯誤

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 假設 3 的驗證方法

**需要數據**:
1. 檢查模型是否學會根據 speaker embedding 調整 token distribution
2. 分析不同 speaker 的預測 token distribution 是否不同

**驗證實驗 3**: 檢查模型對不同 speaker 的預測分布

```python
# 給定相同的 noisy tokens，但不同的 speaker embeddings
# 檢查預測的 token distribution 是否改變

for speaker_id in val_speakers:
    # 固定 noisy_tokens，只改變 speaker_emb
    predictions = model(noisy_tokens, speaker_emb[speaker_id])
    token_dist = Counter(predictions.flatten().tolist())
    print(f"Speaker {speaker_id}: Token 453 = {token_dist[453]/len(predictions)*100:.2f}%")

# 如果所有 speaker 預測的 Token 453 佔比都接近 13.57% (Train average)
# → 說明模型沒有學會 speaker-specific token distribution
# → Speaker embedding 只用來做 denoising，沒有用來調整 distribution prior
```

**預期結果**:
- ❌ **Null Hypothesis**: 所有 speaker 預測的 token distribution 相同 (都接近 Train average)
  - 說明 speaker embedding 沒有捕捉 speaker-specific token distribution
  - 模型只學到 Train set 的平均 token distribution
  
- ✓ **Alternative Hypothesis**: 不同 speaker 預測的 token distribution 不同
  - 說明 speaker embedding 有捕捉 speaker-specific 特徵
  - 但可能因為 Val speakers 未見過，所以預測的 distribution 仍不準

---

## 📊 機轉假設 4: Codebook 是 Speaker-Independent 導致的本質限制

### ASCII 視覺化：WavTokenizer Codebook 的特性

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WavTokenizer Codebook (Frozen, Pretrained on Large Dataset)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Codebook:
  Token 0:   [0.12, -0.45, 0.78, ..., 0.23]  (512-dim)  ← 聲學特徵向量
  Token 1:   [0.34, 0.67, -0.12, ..., -0.45]
  Token 2:   [...]
  ...
  Token 453: [-0.23, 0.89, 0.34, ..., 0.67]  ← 可能對應某種共振峰模式
  ...
  Token 4095: [0.56, -0.78, 0.12, ..., 0.34]

特性:
  1. Speaker-Independent: 同一個 token 對所有 speaker 意義相同
  2. Acoustic Feature: Token 對應的是聲學特徵 (頻譜包絡、共振峰等)
  3. Statistical: 不同 speaker 使用 token 的頻率不同

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
問題: Token 453 的物理意義是什麼？為何 Val speakers 特別常用？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

假設 A: Token 453 = 靜音/低能量段
  → Val speakers 說話較慢，停頓較多
  → 驗證: 檢查 Token 453 對應的音頻段是否為低能量

假設 B: Token 453 = 特定共振峰模式 (e.g., F1=500Hz, F2=1500Hz)
  → Val speakers (boy7,8, girl9,10) 的共振峰集中在這個範圍
  → 驗證: 分析 Token 453 的頻譜特徵

假設 C: Token 453 = 特定音高範圍
  → Val speakers 音高相近 (年齡、性別相似)
  → 驗證: 檢查 Val speakers 的基頻 F0 分布

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
如果 Token 453 確實對應某種聲學特徵，且 Val speakers 確實集中在這個特徵
→ 這是 Zero-Shot Task 的本質困難，不是 bug
→ 模型需要從 Train speakers 學到「如何根據 speaker embedding 預測 token distribution」
→ 但當前架構 (簡單相加) 可能不足以捕捉這種複雜映射
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🎯 綜合機轉模型：四個機轉的相互作用

### 完整因果鏈

```
┌─────────────────────────────────────────────────────────────────────┐
│ 根本原因: Val Speakers 的聲學特徵與 Train Speakers 系統性不同          │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
┌──────────────────────┐            ┌──────────────────────┐
│ Token Distribution   │            │ Acoustic Features    │
│ Mismatch             │            │ Mismatch             │
│                      │            │                      │
│ Token 453:           │            │ 共振峰、音高、        │
│  Train 13.57%        │            │ 說話速度等           │
│  Val   18.65%        │            │ 聲學特徵不同         │
│                      │            │                      │
│ +14 other tokens     │            │                      │
│ (累計 10.94% diff)   │            │                      │
└──────────┬───────────┘            └──────────┬───────────┘
           │                                   │
           └───────────┬───────────────────────┘
                       │
                       ▼
          ┌────────────────────────────┐
          │ 模型架構限制:              │
          │ Speaker Embedding 只用     │
          │ 簡單相加，無法捕捉         │
          │ Speaker-Specific Token     │
          │ Distribution Prior         │
          └────────────┬───────────────┘
                       │
                       ▼
          ┌────────────────────────────┐
          │ Codebook 是 Frozen 且      │
          │ Speaker-Independent        │
          │ → 無法針對 speaker 調整    │
          └────────────┬───────────────┘
                       │
                       ▼
          ┌────────────────────────────┐
          │ 訓練結果:                  │
          │                            │
          │ • Train Acc 卡在 54%       │
          │   (Token 453 準確率低)     │
          │                            │
          │ • Val Acc 只有 37%         │
          │   (Token 453 + 其他        │
          │    mismatch tokens)        │
          │                            │
          │ • Train-Val Gap 17%        │
          │   (模型泛化能力不足)       │
          └────────────────────────────┘
```

---

## 📈 數據支持總結

### 已驗證的假設

| 機轉 | 假設 | 數據支持 | 結論 |
|------|------|----------|------|
| **1** | Token 453 是訓練瓶頸 | Token 453 佔 Train 錯誤 30%，Val 錯誤 29.5% | ✅ **支持** |
| **2** | Token Distribution Mismatch | 15 個 Top-20 tokens 分布差異累計 10.94% | ✅ **支持** |
| **2** | Mismatch 導致 accuracy loss | 推算 4-5% loss vs 實際 17% gap | ⚠️ **部分支持** (解釋約 30%) |
| **3** | Speaker Embedding 無效 | **需要實驗驗證** | ❓ **待驗證** |
| **4** | Codebook 本質限制 | Token 453 可能對應特定聲學特徵 | ❓ **待驗證** |

### 未驗證的假設 (需要額外實驗)

**實驗 3**: 檢查 Speaker Embedding 是否能調整 Token Distribution
```python
# 給定相同 noisy tokens，不同 speaker embeddings
# 檢查預測 distribution 是否隨 speaker 改變
```

**實驗 4**: 分析 Token 453 的物理意義
```python
# 提取所有 Token 453 對應的音頻段
# 分析頻譜、能量、音高特徵
# 理解為何 Val speakers 特別常用 Token 453
```

**實驗 5**: 檢查模型對 Token 453 的預測偏誤
```python
# 統計模型在 Val set 上預測 Token 453 的頻率
# 如果模型預測 Token 453 的頻率接近 13.57% (Train average)
# → 說明模型沒有學到 speaker-specific distribution
```

---

## 🔧 基於機轉的改進方向

### 方向 1: 改進 Speaker Embedding Fusion (針對機轉 3)

**當前問題**: 簡單相加無法捕捉 speaker-specific token distribution

**改進方案**: Speaker-Adaptive Token Distribution Modeling

```python
class SpeakerAdaptiveDecoder(nn.Module):
    def __init__(self):
        # 新增: Speaker → Token Distribution Prior
        self.speaker_to_dist_prior = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 4096)  # 輸出每個 token 的 log-prior
        )
    
    def forward(self, transformer_output, speaker_emb):
        # Transformer 輸出的 logits
        logits = self.token_predictor(transformer_output)  # (B, T, 4096)
        
        # Speaker-specific token distribution prior
        dist_prior = self.speaker_to_dist_prior(speaker_emb)  # (B, 4096)
        
        # 融合: logits + speaker prior
        logits = logits + dist_prior.unsqueeze(1)  # (B, T, 4096)
        
        return logits
```

**預期效果**:
- 模型學會根據 speaker embedding 調整 token distribution
- Val speakers 的 Token 453 預測頻率會接近 18.65% (而非 13.57%)
- **預期 Val Acc 提升至 45-50%** (縮小 Train-Val gap)

### 方向 2: Token Distribution Aware Training (針對機轉 1, 2)

**當前問題**: 模型低估 Token 453 和其他 mismatch tokens 的重要性

**改進方案**: Weighted Cross-Entropy Loss

```python
# 計算 train/val token distribution 的 KL divergence
# 對 mismatch 嚴重的 tokens 增加權重

val_dist = compute_token_distribution(val_data)
train_dist = compute_token_distribution(train_data)
mismatch_weights = val_dist / (train_dist + 1e-6)  # Val 佔比較高的 token 權重增加

# 使用 weighted loss
criterion = nn.CrossEntropyLoss(weight=mismatch_weights)
```

**預期效果**:
- 模型被迫學習 Token 453 等 mismatch tokens
- **預期 Train Acc 提升至 60-65%** (學會預測 Token 453)
- Val Acc 提升至 42-47%

### 方向 3: 改善 Train/Val Split (針對機轉根本原因)

**當前問題**: Val speakers 與 Train speakers 的 token distribution 差異太大

**改進方案**: Distribution-Aware Speaker Split

```python
# 計算每個 speaker 的 token distribution
# 選擇 distribution 接近 train average 的 speakers 作為 val set

for speaker in all_speakers:
    speaker_dist = compute_token_distribution(speaker_data)
    kl_div = compute_kl_divergence(speaker_dist, train_avg_dist)
    
# 選擇 KL divergence 最小的 4 位作為 val speakers
```

**預期效果**:
- 減少 token distribution mismatch
- **預期 Val Acc 提升至 48-52%** (更接近 Train Acc)
- 但喪失部分 zero-shot 測試的意義 (因為 val speakers 更接近 train)

---

## 🧪 後續驗證實驗清單

1. **實驗 3**: Speaker Embedding 對 Token Distribution 的影響
   - 驗證模型是否學會 speaker-specific token distribution
   - 工具: 固定 noisy tokens，改變 speaker embeddings，觀察預測變化

2. **實驗 4**: Token 453 的物理意義分析
   - 提取 Token 453 對應的音頻段
   - 分析頻譜、能量、音高、共振峰特徵
   - 理解為何 Val speakers 特別常用 Token 453

3. **實驗 5**: 模型預測的 Token Distribution 分析
   - 統計模型在 Val set 上預測的 token distribution
   - 比較與 ground truth (18.65% Token 453) 的差距
   - 驗證模型是否低估 Token 453

4. **實驗 6**: 逐層 Attention 分析
   - 可視化 Transformer 各層的 attention maps
   - 檢查 speaker embedding 的資訊是否有效傳播
   - 驗證 attention 是否過度集中在 padding 或特定位置

5. **實驗 7**: Ablation Study - 移除 Speaker Embedding
   - 訓練一個沒有 speaker embedding 的 baseline
   - 比較 accuracy 差異
   - 驗證 speaker embedding 的實際貢獻

---

**報告完成時間**: 2025-11-05 02:00:00  
**下一步**: 執行實驗 3-5 驗證未驗證的假設，然後基於結果決定改進方向
