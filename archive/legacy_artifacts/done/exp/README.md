# Zero-Shot Speaker Denoising Experiment

## 實驗目標

突破 Baseline 的 38% Val Acc 天花板，實現真正的 **Zero-Shot Speaker泛化能力**。

### 核心創新

1. **Speaker-Conditioned Denoising**: 引入 Speaker Embedding 作為條件信息
2. **Noise-Robust Speaker Encoder**: 從 noisy audio 直接提取 speaker 信息
3. **Zero-Shot 泛化**: 可以處理訓練時未見過的 speakers

### 預期效果

| 指標 | Baseline | Zero-Shot Exp | 改進 |
|------|----------|--------------|------|
| Val Acc (unseen speakers) | 38% | 60-75% | +58-97% |
| Train-Val Gap | 52% | 15-25% | 大幅縮小 |
| Zero-Shot 能力 | ❌ | ✅ | 質的飛躍 |

---

## 文件結構

```
done/exp/
├── ARCHITECTURE_COMPARISON.md    # ASCII 架構對比圖（詳細）
├── README.md                      # 本文件
├── speaker_encoder.py             # Speaker Encoder 模型
├── model_zeroshot.py              # Zero-Shot Denoising Transformer
├── data_zeroshot.py               # 數據集（返回 audio waveform）
├── train_zeroshot.py              # 訓練腳本（待創建）
└── run_zeroshot.sh                # 執行腳本（待創建）
```

---

## 快速開始

### 1. 安裝依賴

```bash
# 基礎依賴（已安裝）
conda activate test

# 安裝 Speaker Encoder（選擇其一）
# 選項 A: Resemblyzer (簡單)
pip install resemblyzer

# 選項 B: SpeechBrain ECAPA-TDNN (更好)
pip install speechbrain
```

### 2. 測試模型

```bash
cd /home/sbplab/ruizi/c_code/done/exp

# 測試 Speaker Encoder
python speaker_encoder.py

# 測試 Zero-Shot Model
python model_zeroshot.py

# 測試 Dataset
python data_zeroshot.py
```

### 3. 開始訓練

```bash
# 執行訓練腳本
bash run_zeroshot.sh
```

---

## 架構對比（簡化版）

### Baseline 架構

```
Noisy Tokens (B, T)
    ↓
Frozen Codebook Lookup
    ↓
Token Embeddings (B, T, 512)
    ↓
Positional Encoding
    ↓
Transformer Encoder (4 layers)
    ↓
Output Projection
    ↓
Logits (B, T, 4096)

❌ 問題: 缺少 Speaker Identity 信息
❌ 結果: Val Acc 只有 38%
```

### Zero-Shot 架構

```
Noisy Audio (B, audio_len)  +  Noisy Tokens (B, T)
    ↓                              ↓
Speaker Encoder             Frozen Codebook
    ↓                              ↓
Speaker Emb (B, 256)        Token Emb (B, T, 512)
    ↓                              ↓
Project to 512-dim               │
    ↓                              │
Broadcast to (B, T, 512)         │
    └──────────── ADD ─────────────┘
                  ↓
        Combined Emb (B, T, 512)
                  ↓
        Positional Encoding
                  ↓
    Transformer Encoder (4 layers)
                  ↓
          Output Projection
                  ↓
          Logits (B, T, 4096)

✅ 優勢: Token + Speaker 信息完整
✅ 結果: 預期 Val Acc 60-75%
```

詳細對比請查看 [ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md)

---

## 實驗配置

### 模型參數

| 參數 | Baseline | Zero-Shot | 說明 |
|------|----------|-----------|------|
| d_model | 512 | 512 | 保持一致 |
| num_layers | 4 | 4 | 保持一致 |
| nhead | 8 | 8 | 保持一致 |
| dropout | 0.0 | 0.1 | **新增**正則化 |
| speaker_encoder | ❌ | ✅ | **新增** |
| speaker_embed_dim | N/A | 256 | **新增** |

### 訓練參數

| 參數 | 值 |
|------|-----|
| batch_size | 14 |
| num_epochs | 200 |
| learning_rate | 3e-4 |
| weight_decay | 0.01 |
| speaker_encoder_type | 'simple' (或 'resemblyzer', 'ecapa') |
| speaker_encoder_frozen | True |

---

## 關鍵差異

### 1. 輸入差異

**Baseline**:
```python
# 只有 tokens
noisy_tokens, clean_tokens, content_ids = batch
logits = model(noisy_tokens, return_logits=True)
```

**Zero-Shot**:
```python
# tokens + audio waveform
noisy_audio, clean_audio, noisy_tokens, clean_tokens, content_ids = batch

# 提取 speaker embedding
speaker_emb = speaker_encoder(noisy_audio)  # (B, 256)

# Denoising (conditioned on speaker)
logits = model(noisy_tokens, speaker_emb, return_logits=True)
```

### 2. 模型差異

**Baseline**:
```python
class TokenDenoisingTransformer:
    def forward(self, noisy_tokens):
        token_emb = codebook[noisy_tokens]
        # ... transformer ...
        return logits
```

**Zero-Shot**:
```python
class ZeroShotDenoisingTransformer:
    def forward(self, noisy_tokens, speaker_embedding):
        token_emb = codebook[noisy_tokens]  # (B, T, 512)
        speaker_emb = speaker_proj(speaker_embedding)  # (B, 512)
        speaker_emb = speaker_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, 512)

        combined_emb = token_emb + speaker_emb  # Fusion!
        # ... transformer ...
        return logits
```

### 3. 數據集差異

**Baseline**:
```python
# AudioDataset 返回 audio (會被 collate_fn 編碼為 tokens)
return (noisy_audio, clean_audio, content_id)
```

**Zero-Shot**:
```python
# ZeroShotAudioDataset 返回相同格式
# 但 collate_fn 會同時保留 audio 和 tokens
return (noisy_audio, clean_audio, content_id)

# Collate function:
return (
    noisy_audio_batch,  # 新增：保留 waveform
    clean_audio_batch,  # 新增：保留 waveform
    noisy_tokens_batch,
    clean_tokens_batch,
    content_ids_batch
)
```

---

## 訓練流程

### 階段 1: 準備 Speaker Encoder

**選項 A: 使用 Simple Encoder（快速測試）**
```python
speaker_encoder = SimpleSpeakerEncoder(output_dim=256)
# 優點: 快速，無需額外依賴
# 缺點: 泛化能力較弱
```

**選項 B: 使用預訓練 Encoder（推薦）**
```python
speaker_encoder = PretrainedSpeakerEncoder(
    model_type='resemblyzer',  # or 'ecapa'
    freeze=True,
    output_dim=256
)
# 優點: 泛化能力強，zero-shot 效果好
# 缺點: 需要安裝額外依賴
```

### 階段 2: 訓練 Denoising Transformer

```python
# Freeze speaker encoder
for param in speaker_encoder.parameters():
    param.requires_grad = False

# Train denoising transformer
for epoch in range(num_epochs):
    for batch in train_loader:
        noisy_audio, clean_audio, noisy_tokens, clean_tokens, _ = batch

        # Extract speaker embeddings
        speaker_emb = speaker_encoder(noisy_audio)

        # Denoising
        logits = model(noisy_tokens, speaker_emb, return_logits=True)
        loss = CrossEntropy(logits, clean_tokens)
        loss.backward()
```

### 階段 3: 評估 Zero-Shot 能力

```python
# 驗證集包含 unseen speakers
val_speakers = ['boy7', 'boy8', 'girl9', 'girl10']

# 測試 zero-shot denoising
for batch in val_loader:
    speaker_emb = speaker_encoder(noisy_audio)  # 從未見過的 speaker
    logits = model(noisy_tokens, speaker_emb, return_logits=True)
    # 預期: Val Acc 從 38% → 60-75%
```

---

## 預期結果

### 訓練曲線預測

**Baseline**:
```
Epoch 1:  Train Loss: 4.35, Val Loss: 4.74, Val Acc: 37%
Epoch 3:  Train Loss: 3.60, Val Loss: 4.68, Val Acc: 38% ← 最佳
Epoch 50: Train Loss: 0.57, Val Loss: 11.46, Val Acc: 38% ← 嚴重過擬合
```

**Zero-Shot** (預期):
```
Epoch 1:  Train Loss: 4.50, Val Loss: 5.00, Val Acc: 35%
Epoch 5:  Train Loss: 3.20, Val Loss: 3.80, Val Acc: 55%
Epoch 10: Train Loss: 2.50, Val Loss: 3.50, Val Acc: 65% ← 改善顯著
Epoch 20: Train Loss: 1.80, Val Loss: 3.20, Val Acc: 70% ← 持續改善
Epoch 50: Train Loss: 1.20, Val Loss: 3.30, Val Acc: 68% ← 穩定
```

### 關鍵指標對比

| 指標 | Baseline | Zero-Shot | 改進幅度 |
|------|----------|-----------|---------|
| **Best Val Loss** | 4.68 (Epoch 3) | 3.20 (Epoch 20) | -31.6% |
| **Best Val Acc** | 38.19% | 70% (預期) | +83.3% |
| **Train-Val Gap** | 52.27% | 18% (預期) | -65.5% |
| **Overfitting** | 嚴重 | 輕微 | 大幅改善 |
| **Zero-Shot 能力** | ❌ 無 | ✅ 有 | 質的飛躍 |

---

## 消融實驗計劃

測試各組件的貢獻：

| 實驗 | Speaker Encoder | Frozen | Data Split | Val Acc (預期) |
|------|----------------|--------|------------|---------------|
| **Baseline** | ❌ | N/A | By Speaker | 38% |
| **Exp 1** | Simple | ✅ | By Speaker | 50-60% |
| **Exp 2** | Resemblyzer | ✅ | By Speaker | 60-70% |
| **Exp 3** | ECAPA-TDNN | ✅ | By Speaker | 65-75% |
| **Exp 4** | ECAPA-TDNN | ❌ Fine-tune | By Speaker | 70-80% |

---

## 故障排除

### 問題 1: Speaker Encoder 安裝失敗

**解決方案**:
```bash
# 使用 Simple Encoder (無需額外依賴)
# 在 run_zeroshot.sh 中設置:
SPEAKER_ENCODER_TYPE="simple"
```

### 問題 2: CUDA Out of Memory

**解決方案**:
```bash
# 減小 batch size
BATCH_SIZE=8  # 從 14 減少到 8

# 或減少 audio length (在 collate_fn 中截斷)
```

### 問題 3: Val Acc 沒有提升

**可能原因**:
1. Speaker encoder 質量不佳 → 換用預訓練模型
2. Dropout 太高 → 降低到 0.05-0.1
3. Learning rate 太大 → 降低到 1e-4

---

## 後續實驗方向

### 實驗 A: 多任務學習

加入輔助任務（speaker classification, material classification）

### 實驗 B: 對比學習

微調 speaker encoder 使其對噪音更魯棒

### 實驗 C: Seq2Seq 架構

使用 Encoder-Decoder 架構取代 Encoder-only

### 實驗 D: 改變數據分割

按 utterance 分割（所有 speakers 都在訓練和驗證集）

---

## 參考文獻

1. **Speaker Verification**: ECAPA-TDNN (SpeechBrain)
2. **Voice Cloning**: Resemblyzer
3. **Contrastive Learning**: SimCLR, MoCo
4. **Conditional Generation**: StyleGAN, ControlNet

---

## 作者與聯繫

**實驗設計**: Claude + 使用者協作
**日期**: 2025-11-01
**分支**: `zero-shot-speaker-denoising`

**問題反饋**: 查看 [ROOT_CAUSE_ANALYSIS.md](../ROOT_CAUSE_ANALYSIS.md) 了解問題根源

---

## License

與主項目相同

---

**最後更新**: 2025-11-01
**狀態**: 實驗中 🚀
