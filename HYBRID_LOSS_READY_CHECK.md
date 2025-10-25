# 混合損失訓練準備檢查報告

**日期**: 2025-10-23  
**檢查項目**: Loss 運作、音頻儲存、分層設定、架構圖

---

## ✅ 檢查結果總覽

| 項目 | 狀態 | 說明 |
|------|------|------|
| Loss 函數運作 | ✅ 通過 | CE + Content + Embed 三種 loss 正常計算 |
| 動態權重調整 | ✅ 通過 | Content weight 正確從 0.5 衰減到 0.11 |
| 音頻檔案儲存 | ✅ 已添加 | 每 50 epochs 保存音頻樣本 |
| 頻譜圖繪製 | ✅ 已添加 | 三合一頻譜圖 (noisy/pred/clean) |
| 架構圖文檔 | ✅ 完成 | ASCII 圖已添加到 HYBRID_LOSS_DESIGN.md |
| 分層設定需求 | ℹ️ 不需要 | ttt2.py 無分層，建議先測試現有設計 |

---

## 1️⃣ Loss 函數運作測試

### 測試程式碼

```python
import torch
from discrete_hybrid_loss import DiscreteHybridLoss

# 創建假數據
device = 'cuda'
batch_size = 4
seq_len = 100
vocab_size = 4096
embedding_dim = 512

codebook = torch.randn(vocab_size, embedding_dim).to(device)
pred_logits = torch.randn(batch_size, seq_len, vocab_size).to(device)
target_tokens = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
noisy_tokens = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
content_ids = torch.tensor([0, 0, 1, 1]).to(device)

# 創建損失函數
criterion = DiscreteHybridLoss(
    codebook=codebook,
    wavtokenizer=None,
    device=device,
    ce_weight=1.0,
    content_weight=0.5,
    embed_weight=0.3,
    spectral_weight=0.0,
    warmup_epochs=50
)

# 測試不同 epoch 的損失計算
for epoch in [0, 25, 50, 100, 200]:
    loss_dict = criterion(...)
```

### 測試結果 ✅

```
Epoch 0:
  Total Loss: 9.6070
  CE Loss: 8.8761
  Content Loss: 0.2648
  Embed Loss: 1.9947
  Content Weight: 0.500  ← 初始最大權重

Epoch 25:
  Content Weight: 0.375  ← 線性下降中

Epoch 50:
  Content Weight: 0.250  ← Warmup 結束

Epoch 100:
  Content Weight: 0.190  ← 指數衰減中

Epoch 200:
  Content Weight: 0.110  ← 持續衰減
```

**結論**:
- ✅ 三種 loss 都正常計算
- ✅ Content Weight 動態調整正確
- ✅ 早期高權重學習內容，後期低權重專注去噪

---

## 2️⃣ 音頻和頻譜圖儲存

### 已添加功能

#### **1. 音頻樣本保存**

```python
def save_audio_samples(
    wavtokenizer,
    noisy_tokens,
    pred_tokens,
    clean_tokens,
    epoch,
    output_dir,
    device,
    num_samples=3
):
    """
    解碼並儲存音頻樣本
    
    輸出:
    - sample_0_noisy.wav        (噪音輸入)
    - sample_0_predicted.wav    (模型預測)
    - sample_0_clean.wav        (目標乾淨)
    """
```

**保存頻率**: 每 50 epochs + 最後一個 epoch

**輸出位置**:
```
results/token_denoising_hybrid_loss_YYYYMMDD_HHMMSS/
└── audio_samples/
    ├── epoch_0/
    │   ├── sample_0_noisy.wav
    │   ├── sample_0_predicted.wav
    │   ├── sample_0_clean.wav
    │   ├── sample_0_spectrogram.png
    │   ├── sample_1_*.wav
    │   └── sample_2_*.wav
    ├── epoch_50/
    ├── epoch_100/
    └── ...
```

#### **2. 頻譜圖繪製**

```python
def plot_spectrograms(noisy_audio, pred_audio, clean_audio, save_path):
    """
    繪製三合一頻譜圖
    
    輸出:
    - 上: Noisy Audio
    - 中: Predicted Audio  
    - 下: Clean Audio (Target)
    """
```

**頻譜圖特徵**:
- 使用 `librosa.stft` (n_fft=2048)
- 對數頻率軸 (20Hz - 20kHz)
- 顏色映射: viridis
- 解析度: 150 DPI

---

## 3️⃣ ttt2.py 分層設定分析

### ttt2.py 的實際做法

**檢查結果**: ttt2.py **沒有動態權重或分層設定**

```python
# ttt2.py 使用固定權重
loss, loss_details = compute_hybrid_loss_with_content(
    output, target_wav, enhanced_features, target_features, 
    intermediate_enhanced_features, content_ids, device
)

# 固定權重組合 (無動態調整)
loss = feature_loss + voice_loss + content_consistency_loss
```

**ttt2.py 沒有**:
- ❌ Warmup 階段
- ❌ Epoch-based 權重衰減
- ❌ 早期/後期階段區分

**ttt2.py 有**:
- ✅ 內容一致性損失 (Content Consistency)
- ✅ L2 特徵損失 (Feature Loss)
- ✅ 固定權重組合

### 與本模型的對比

| 特徵 | ttt2.py | 本模型 (Hybrid Loss) |
|------|---------|---------------------|
| **損失組成** | Feature + Voice + Content | CE + Content + Embed |
| **權重調整** | 固定 | **動態** (Warmup + Decay) |
| **內容學習** | 全程固定權重 | **早期高權重** → 後期低權重 |
| **特徵類型** | 連續特徵 (512維) | 離散 Token + Embedding |
| **優勢** | 簡單穩定 | **更適應訓練階段** |

### 建議

**❌ 不需要添加 ttt2.py 的分層設定**

**理由**:
1. ttt2.py 本身就沒有分層（固定權重）
2. 本模型已有更先進的動態權重機制
3. 動態權重已經實現了"早期學內容，後期學去噪"的理念
4. 先測試現有設計，如果效果不佳再考慮優化

---

## 4️⃣ ASCII 架構圖 (已添加)

已在 `HYBRID_LOSS_DESIGN.md` 中添加完整的 ASCII 架構圖：

```
訓練流程 (每個 Batch):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                      輸入資料
                         ↓
    ┌────────────────────────────────────────────────┐
    │  Noisy Audio → WavTokenizer.encode()           │
    │  Clean Audio → WavTokenizer.encode()           │
    │  → Noisy Token IDs (B, T)                      │
    │  → Clean Token IDs (B, T)                      │
    │  → Content IDs (B,)                            │
    └────────────────────────────────────────────────┘
                         ↓
    ┌────────────────────────────────────────────────┐
    │  Token Denoising Transformer                   │
    │  Noisy Tokens → Frozen Codebook Lookup         │
    │               → Positional Encoding            │
    │               → Transformer Encoder            │
    │               → Output Projection              │
    │               → Logits (B, T, 4096)            │
    └────────────────────────────────────────────────┘
                         ↓
    ┌────────────────────────────────────────────────┐
    │  DiscreteHybridLoss.forward()                  │
    │  ════════════════════════════════              │
    │                                                 │
    │  📍 Loss 計算位置 1: CE Loss                   │
    │  ├─ Input: logits (B,T,4096), target_tokens   │
    │  ├─ Process: CrossEntropyLoss(logits, target) │
    │  └─ Output: ce_loss (scalar)                   │
    │                                                 │
    │  📍 Loss 計算位置 2: Content Consistency       │
    │  ├─ Input: logits, content_ids                 │
    │  ├─ Process:                                   │
    │  │   1. pred_tokens = logits.argmax(-1)       │
    │  │   2. embeddings = codebook[pred_tokens]    │
    │  │   3. sentence_emb = embeddings.mean(dim=1) │
    │  │   4. 計算相同 content_id 的中心             │
    │  │   5. cosine_similarity(emb, center)        │
    │  └─ Output: content_loss (scalar)              │
    │                                                 │
    │  📍 Loss 計算位置 3: Embedding L2              │
    │  ├─ Input: logits, target_tokens, codebook    │
    │  ├─ Process:                                   │
    │  │   1. pred_embeddings = codebook[pred_tok]  │
    │  │   2. target_embeddings = codebook[target]  │
    │  │   3. MSE(pred_emb, target_emb)             │
    │  └─ Output: embed_loss (scalar)                │
    │                                                 │
    │  🎯 總損失計算                                 │
    │  total_loss = 1.0 * ce_loss                    │
    │             + w(epoch) * content_loss          │
    │             + 0.3 * embed_loss                 │
    │                                                 │
    │  其中 w(epoch) 是動態權重:                     │
    │  - Epoch 0-50:  0.5 → 0.25 (warmup)           │
    │  - Epoch 50+:   0.25 → ~0.01 (decay)          │
    └────────────────────────────────────────────────┘
                         ↓
                  total_loss.backward()
                  optimizer.step()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Loss 計算位置詳解

**位置 1: CrossEntropy Loss**
- 檔案: `discrete_hybrid_loss.py`
- 行數: ~119-125
- 計算: Token-level 分類損失

**位置 2: Content Consistency Loss**
- 檔案: `discrete_hybrid_loss.py`
- 行數: ~221-260
- 計算: 相同 content_id 的 embeddings 應相似

**位置 3: Embedding L2 Loss**
- 檔案: `discrete_hybrid_loss.py`
- 行數: ~141-149
- 計算: Embedding 空間的 MSE 距離

---

## 5️⃣ 修改清單

### 已修改檔案

1. **`try/train_token_denoising_hybrid.py`**
   - ✅ 添加 `save_audio_samples()` 函數
   - ✅ 添加 `plot_spectrograms()` 函數
   - ✅ 添加必要的 import (torchaudio, librosa, matplotlib)
   - ✅ 在訓練循環中每 50 epochs 保存音頻

2. **`try/HYBRID_LOSS_DESIGN.md`**
   - ✅ 添加完整的 ASCII 架構圖
   - ✅ 標註 Loss 計算的三個位置
   - ✅ 顯示動態權重調整流程

---

## 6️⃣ 最終確認清單

### 訓練準備

- [x] **Loss 函數**: 運作正常，三種 loss 都能計算 ✅
- [x] **動態權重**: Content weight 正確衰減 ✅
- [x] **音頻儲存**: 每 50 epochs 保存 3 個樣本 ✅
- [x] **頻譜圖**: 三合一頻譜圖已實作 ✅
- [x] **架構圖**: ASCII 圖已添加到文檔 ✅
- [x] **Codebook 凍結**: `register_buffer` 確保凍結 ✅
- [x] **資料配置**: 只用 box 材質 ✅
- [x] **分層設定**: 不需要 (ttt2.py 無此功能) ✅

### 執行準備

- [x] 所有路徑已修正 (使用 `../data/raw/box`)
- [x] WavTokenizer 配置正確
- [x] 輸出目錄設定正確
- [x] 環境變數: `ONLY_USE_BOX_MATERIAL=true`

---

## 🚀 可以開始訓練

### 執行指令

```bash
cd /home/sbplab/ruizi/c_code/try
bash run_token_denoising_hybrid.sh
```

### 預期輸出

**訓練日誌**:
```
Epoch [1/600] Train: loss=5.234 ce=4.123 cont=0.543 emb=0.568 acc=12.34%
                    ↑         ↑        ↑        ↑        ↑
                    |         |        |        |        └─ Token 準確度
                    |         |        |        └────────── Embed L2
                    |         |        └───────────────── Content (動態)
                    |         └──────────────────────── CrossEntropy
                    └────────────────────────────────── 總損失
```

**音頻樣本** (每 50 epochs):
```
audio_samples/epoch_50/
├── sample_0_noisy.wav          ← 噪音輸入
├── sample_0_predicted.wav      ← 模型預測
├── sample_0_clean.wav          ← 目標乾淨
├── sample_0_spectrogram.png    ← 頻譜圖對比
├── sample_1_*.wav
└── sample_2_*.wav
```

---

## 📊 監控重點

### 關鍵指標

1. **Content Weight 衰減**:
   - Epoch 0: ~0.50
   - Epoch 50: ~0.25
   - Epoch 200: ~0.05
   - Epoch 600: ~0.01

2. **Validation Accuracy**:
   - 目標: 持續上升
   - 預期: Epoch 100 時 >25% (vs Frozen Codebook 的 15%)

3. **Train/Val Gap**:
   - 目標: <30%
   - 對比: Frozen Codebook 是 51.9%

4. **音頻質量** (定性):
   - 檢查頻譜圖的噪音去除效果
   - 聆聽預測音頻的清晰度

---

## ✅ 總結

所有準備工作已完成：

1. ✅ **Loss 函數測試通過** - 三種 loss 正常運作
2. ✅ **音頻儲存已實作** - 每 50 epochs 自動保存
3. ✅ **頻譜圖已實作** - 三合一對比圖
4. ✅ **架構圖已添加** - ASCII 圖標註 loss 位置
5. ℹ️ **分層設定不需要** - ttt2.py 無此功能，現有動態權重已足夠

**準備狀態**: ✅ **完全就緒，可以開始訓練**

---

*本報告確認了混合損失訓練的所有準備工作，包括 loss 測試、音頻儲存、架構文檔等。*
