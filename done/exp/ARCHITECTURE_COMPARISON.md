# Zero-Shot Speaker Denoising 架構對比

## ASCII 架構圖對比

### Baseline 架構（當前 train.py）

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Baseline: Token-Only Denoising                     │
└──────────────────────────────────────────────────────────────────────┘

Input: Noisy Audio (24kHz waveform)
   │
   ├──> WavTokenizer.encode() ──> Noisy Tokens [B, T]
   │                                    │
   │                                    ▼
   │                        ┌────────────────────────┐
   │                        │  Frozen Codebook       │
   │                        │  Lookup: [4096, 512]   │
   │                        └────────────────────────┘
   │                                    │
   │                                    ▼
   │                        Token Embeddings [B, T, 512]
   │                                    │
   │                                    ▼
   │                        ┌────────────────────────┐
   │                        │  Positional Encoding   │
   │                        └────────────────────────┘
   │                                    │
   │                                    ▼
   │                        ┌────────────────────────┐
   │                        │  Transformer Encoder   │
   │                        │  - 4 layers            │
   │                        │  - 8 heads             │
   │                        │  - d_model=512         │
   │                        └────────────────────────┘
   │                                    │
   │                                    ▼
   │                        ┌────────────────────────┐
   │                        │  Output Projection     │
   │                        │  Linear(512, 4096)     │
   │                        └────────────────────────┘
   │                                    │
   │                                    ▼
   │                        Logits [B, T, 4096]
   │                                    │
   │                                    ▼
   │                        Predicted Tokens [B, T]
   │
Target: Clean Tokens [B, T]
   │
   ▼
Loss: CrossEntropy(Logits, Clean Tokens)

═══════════════════════════════════════════════════════════════════════

❌ 問題分析:

1. 缺少 Speaker Identity 信息
   - 模型不知道目標 speaker 是誰
   - 無法學習 speaker-specific 的 token 映射

2. Zero-Shot 能力為零
   - 訓練集 speakers: {boy1, boy3, ..., girl11} (14 個)
   - 驗證集 speakers: {boy7, boy8, girl9, girl10} (4 個完全不同)
   - 模型無法泛化到新 speakers

3. 38% Val Acc 天花板
   - 只能預測 speaker-invariant tokens
   - 62% 的 speaker-dependent tokens 無法正確預測

═══════════════════════════════════════════════════════════════════════
```

---

### 新架構: Zero-Shot Speaker Denoising

```
┌──────────────────────────────────────────────────────────────────────┐
│              Zero-Shot Speaker Denoising (Experiment)                 │
└──────────────────────────────────────────────────────────────────────┘

Input: Noisy Audio (24kHz waveform) [B, audio_len]
   │
   ├────────────────┬─────────────────────────────────────┐
   │                │                                     │
   │                ▼                                     ▼
   │    ┌──────────────────────┐         ┌──────────────────────────┐
   │    │  WavTokenizer        │         │  Noise-Robust Speaker    │
   │    │  .encode()           │         │  Encoder (預訓練)         │
   │    └──────────────────────┘         │  ----------------------  │
   │                │                    │  - ECAPA-TDNN backbone   │
   │                │                    │  - Contrastive Learning  │
   │                ▼                    │  - Frozen 或 Fine-tune   │
   │    Noisy Tokens [B, T]              └──────────────────────────┘
   │                │                                     │
   │                │                                     ▼
   │                │                    Speaker Embedding [B, 256]
   │                │                                     │
   │                │                                     ▼
   │                │                    ┌──────────────────────────┐
   │                │                    │  Speaker Projection      │
   │                │                    │  Linear(256, 512)        │
   │                │                    └──────────────────────────┘
   │                │                                     │
   │                │                                     ▼
   │                │                    Speaker Embedding [B, 512]
   │                │                                     │
   │                ▼                                     │
   │    ┌────────────────────────┐                       │
   │    │  Frozen Codebook       │                       │
   │    │  Lookup: [4096, 512]   │                       │
   │    └────────────────────────┘                       │
   │                │                                     │
   │                ▼                                     │
   │    Token Embeddings [B, T, 512] ◄───────────────────┘
   │                │                    (broadcast & add)
   │                ▼
   │    Combined Embeddings [B, T, 512]
   │    = Token Emb + Speaker Emb (每個 time step)
   │                │
   │                ▼
   │    ┌────────────────────────┐
   │    │  Positional Encoding   │
   │    └────────────────────────┘
   │                │
   │                ▼
   │    ┌────────────────────────┐
   │    │  Transformer Encoder   │
   │    │  - 4 layers            │
   │    │  - 8 heads             │
   │    │  - d_model=512         │
   │    │  - dropout=0.1         │
   │    └────────────────────────┘
   │                │
   │                ▼
   │    ┌────────────────────────┐
   │    │  Output Projection     │
   │    │  Linear(512, 4096)     │
   │    └────────────────────────┘
   │                │
   │                ▼
   │    Logits [B, T, 4096]
   │                │
   │                ▼
   │    Predicted Tokens [B, T]
   │
Target: Clean Tokens [B, T]
   │
   ▼
Loss: CrossEntropy(Logits, Clean Tokens)

═══════════════════════════════════════════════════════════════════════

✅ 關鍵改進:

1. 引入 Speaker Embedding
   - 從 noisy audio 直接提取 speaker 信息
   - 使用 noise-robust speaker encoder
   - 不需要 clean reference

2. Speaker-Conditioned Denoising
   - Token Emb + Speaker Emb 提供完整信息
   - 模型知道"目標 speaker 是誰"
   - 可以學習 speaker-specific 的 denoising 策略

3. Zero-Shot 泛化能力
   - Speaker encoder 預訓練在大規模數據上
   - 可以處理未見過的 speakers
   - 預期 Val Acc 從 38% → 60-75%

═══════════════════════════════════════════════════════════════════════
```

---

## 詳細差異對比表

| 特性 | Baseline | Zero-Shot Experiment |
|------|----------|---------------------|
| **輸入** | Noisy Tokens | Noisy Tokens + Noisy Audio |
| **Speaker Info** | ❌ 無 | ✅ Speaker Embedding (256-dim) |
| **Speaker Encoder** | ❌ 無 | ✅ ECAPA-TDNN (noise-robust) |
| **Token Embedding** | Frozen Codebook [4096, 512] | Frozen Codebook [4096, 512] |
| **Embedding Fusion** | 僅 Token Emb | Token Emb + Speaker Emb |
| **Transformer** | 4 layers, 8 heads | 4 layers, 8 heads (相同) |
| **Dropout** | 0.0 | 0.1 (正則化) |
| **Output** | Logits [B, T, 4096] | Logits [B, T, 4096] |
| **Loss** | CrossEntropy | CrossEntropy |
| **訓練集 Speakers** | 14 個 | 14 個 (相同) |
| **驗證集 Speakers** | 4 個 (unseen) | 4 個 (unseen) |
| **Zero-Shot 能力** | ❌ 無法泛化 | ✅ 可以泛化 |
| **Val Acc (預期)** | 38% | 60-75% |
| **參數量 (新增)** | 0 | ~6M (speaker encoder) |
| **參數量 (可訓練)** | ~14.7M | ~14.7M (speaker encoder frozen) |

---

## 核心創新點

### 1. Noise-Robust Speaker Encoder

**為什麼需要？**
- 傳統 speaker encoder 在 clean audio 上訓練
- 遇到 noisy audio 時 embedding 質量下降
- 需要對噪音魯棒的 speaker encoder

**如何實現？**
```python
# 階段 1: 對比學習訓練
positive_pairs = (speaker_A_noisy, speaker_A_clean)  # 同 speaker
negative_pairs = (speaker_A_noisy, speaker_B_clean)  # 不同 speaker

# 目標: 同 speaker 的 noisy/clean 應該接近
similarity(speaker_A_noisy, speaker_A_clean) >
similarity(speaker_A_noisy, speaker_B_clean)

# 階段 2: Freeze 並用於 denoising
speaker_emb = frozen_speaker_encoder(noisy_audio)
```

**預訓練方案**:
- 選項 1: 使用公開的預訓練模型（如 resemblyzer, ECAPA-TDNN）
- 選項 2: 在你的數據集上用對比學習微調
- 選項 3: 使用 speaker verification 預訓練模型

---

### 2. Speaker-Conditioned Token Denoising

**核心思想**:
```python
# Baseline: 只有 token 信息
embedding = codebook[noisy_tokens]  # [B, T, 512]

# New: Token + Speaker 信息
token_emb = codebook[noisy_tokens]  # [B, T, 512]
speaker_emb = speaker_encoder(noisy_audio)  # [B, 256]
speaker_emb = speaker_proj(speaker_emb)  # [B, 512]
speaker_emb = speaker_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, 512]

# Fusion: 加法
combined_emb = token_emb + speaker_emb  # [B, T, 512]
```

**為什麼用加法而非拼接？**
- ✅ 保持 d_model=512 不變
- ✅ 參數量不增加
- ✅ 類似於 multi-modal fusion 的常見做法
- ❌ 拼接會變成 [B, T, 1024]，需要重新設計 Transformer

---

### 3. Zero-Shot 泛化機制

**訓練階段**:
```
訓練集: 14 speakers × 288 sentences
- Speaker encoder 看過這些 speakers
- Denoising transformer 學習:
  "給定 (noisy_tokens, speaker_emb)，預測 clean_tokens"
```

**測試階段（Zero-Shot）**:
```
驗證集: 4 unseen speakers × 288 sentences
- Speaker encoder 從未見過這些 speakers
- 但可以從 noisy audio 提取 embedding（泛化能力）
- Denoising transformer 根據新的 speaker_emb 進行 denoising
```

**為什麼能 Zero-Shot？**
1. **Speaker encoder 的泛化**:
   - 預訓練在大規模數據（如 VoxCeleb）
   - 學到通用的 speaker representation
   - 即使是新 speaker，也能提取有意義的 embedding

2. **Denoising transformer 的條件化學習**:
   - 不是記憶特定 speaker
   - 而是學習"如何根據 speaker_emb 調整 denoising 策略"
   - 類似於 style transfer 中的 conditional generation

---

## 訓練策略差異

### Baseline

```python
# 數據分割
train_speakers = 14 speakers
val_speakers = 4 unseen speakers

# 訓練循環
for epoch in range(num_epochs):
    for noisy_tokens, clean_tokens in train_loader:
        logits = model(noisy_tokens)  # 只有 token 輸入
        loss = CrossEntropy(logits, clean_tokens)
        loss.backward()

# 問題: 驗證集是完全不同的 speakers
# 模型無法泛化 → Val Acc 只有 38%
```

### Zero-Shot Experiment

```python
# 數據分割（相同）
train_speakers = 14 speakers
val_speakers = 4 unseen speakers

# 訓練循環
for epoch in range(num_epochs):
    for noisy_audio, noisy_tokens, clean_tokens in train_loader:
        # 提取 speaker embedding (從 noisy audio)
        speaker_emb = speaker_encoder(noisy_audio)  # [B, 256]

        # Denoising (conditioned on speaker)
        logits = model(noisy_tokens, speaker_emb)  # Token + Speaker
        loss = CrossEntropy(logits, clean_tokens)
        loss.backward()

# 優勢: 驗證集雖然是 unseen speakers
# 但 speaker_encoder 可以提取新 speaker 的 embedding
# Denoising model 根據新 embedding 進行 zero-shot denoising
# → Val Acc 預期 60-75%
```

---

## 預期效果對比

| 指標 | Baseline | Zero-Shot Experiment | 改進幅度 |
|------|----------|---------------------|---------|
| Train Acc | 90.30% | 85-90% | 類似 |
| Val Acc (unseen speakers) | 38.03% | 60-75% | **+60-100%** |
| Train-Val Gap | 52.27% | 15-25% | **大幅縮小** |
| Best Val Loss | 4.6754 | 3.5-4.0 | 降低 15-25% |
| Overfitting | 嚴重 | 輕微 | **顯著改善** |
| Zero-Shot 能力 | ❌ 無 | ✅ 有 | **質的飛躍** |

---

## 實驗設計

### 階段 1: Speaker Encoder 選擇

**選項 A: 使用預訓練模型（推薦）**
```python
# 使用 resemblyzer (簡單) 或 speechbrain ECAPA-TDNN (更好)
from resemblyzer import VoiceEncoder
speaker_encoder = VoiceEncoder()
# 或
from speechbrain.pretrained import EncoderClassifier
speaker_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)
```

**選項 B: 微調 Speaker Encoder（進階）**
```python
# 在你的數據集上用對比學習微調
# 使用 (noisy, clean) pairs 訓練 noise-robust 特性
```

### 階段 2: 整合訓練

1. **Freeze speaker encoder**
2. **訓練 denoising transformer** (with speaker conditioning)
3. **評估 zero-shot 能力**

### 階段 3: 消融實驗

| 實驗 | Speaker Encoder | Speaker Frozen | Val Acc (預期) |
|------|----------------|----------------|---------------|
| Baseline | ❌ 無 | N/A | 38% |
| Exp 1 | ✅ Resemblyzer | ✅ Yes | 55-65% |
| Exp 2 | ✅ ECAPA-TDNN | ✅ Yes | 60-70% |
| Exp 3 | ✅ ECAPA-TDNN | ❌ Fine-tune | 65-75% |

---

## 文件結構

```
done/exp/
├── speaker_encoder.py       # Noise-robust speaker encoder
├── model_zeroshot.py         # Zero-shot denoising transformer
├── data_zeroshot.py          # Dataset with audio waveform
├── loss_contrastive.py       # Contrastive loss (optional)
├── train_zeroshot.py         # Training script
├── run_zeroshot.sh           # Execution script
└── ARCHITECTURE_COMPARISON.md  # This file
```

---

## 總結

**Baseline 的根本問題**:
- ❌ 缺少 speaker identity 信息
- ❌ 無法泛化到 unseen speakers
- ❌ 38% 準確率天花板

**Zero-Shot Experiment 的解決方案**:
- ✅ 引入 noise-robust speaker encoder
- ✅ Speaker-conditioned token denoising
- ✅ 真正的 zero-shot 泛化能力
- ✅ 預期準確率 60-75%（提升 60-100%）

**下一步**:
1. 實現 speaker_encoder.py
2. 修改 model.py 加入 speaker conditioning
3. 修改 data.py 返回 audio waveform
4. 創建 train_zeroshot.py 和 run_zeroshot.sh
5. 開始訓練並驗證效果
