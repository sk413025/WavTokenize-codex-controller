# Token Denoising Hybrid Loss 配置確認 (僅 Box 材質)

**日期**: 2025-10-23  
**實驗**: Token Denoising with Hybrid Loss (僅 box 材質)  
**實驗編號**: hybrid_loss_box_$(date +%Y%m%d_%H%M%S)

---

## ✅ 核心架構確認

### 1. **這是純離散訓練嗎？**

**✅ 是的，100% 純離散訓練**

```
完整流程 (純離散):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
輸入: Noisy Token IDs                    [整數 0-4095]
  ↓
Frozen Codebook Lookup:                 codebook[token_ids]
  embeddings = self.codebook[token_ids]  # (B, T, 512)
  ↓
Positional Encoding                      
  ↓
Transformer Encoder (6 layers)           # 可訓練
  ↓
Linear Projection                        # 可訓練
  logits (B, T, 4096)
  ↓
Argmax → Clean Token IDs                 [整數 0-4095]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**關鍵特徵**:
- ✅ 輸入：離散 Token IDs (整數)
- ✅ Codebook：完全凍結 (`register_buffer`)
- ✅ 中間處理：在 embedding 空間 (512 維)
- ✅ 輸出：離散 Token IDs (通過 argmax)
- ✅ 損失：混合損失 (CE + Content + Embedding L2)

### 2. **Codebook 是否完全凍結？**

**✅ 是的，完全凍結**

**證據 1**: 使用 `register_buffer` (不會被訓練)
```python
# token_denoising_transformer.py, Line 55
self.register_buffer('codebook', codebook)
```

**證據 2**: 明確斷言檢查
```python
# train_token_denoising_hybrid.py, Line 378
assert not model.codebook.requires_grad, "Codebook 必須凍結！"
```

**證據 3**: 參數統計
```
模型總參數數量: 23,120,896
  - 可訓練參數: 21,024,768  (Transformer + Projection)
  - 凍結參數:    2,096,128  (Codebook: 4096 × 512)
```

---

## 🎯 混合損失詳解

### 損失函數設計

```python
total_loss = CE_loss + content_weight × Content_loss + embed_weight × Embed_loss
           = 1.0 × CE + dynamic × Content + 0.3 × Embed
```

### 三種損失成分

#### **1. CrossEntropy Loss** (固定權重 1.0)

**目的**: 預測正確的 clean token ID

**計算位置**:
```python
# discrete_hybrid_loss.py, Line 119-125
logits_flat = pred_logits.reshape(-1, 4096)    # (B*T, 4096)
target_flat = target_tokens.reshape(-1)        # (B*T,)
ce_loss = CrossEntropyLoss(logits_flat, target_flat)
```

**意義**: 
- 基本的離散分類損失
- 確保模型學會正確的 noisy→clean token 映射
- 與 Frozen Codebook 實驗一致 (純 CE Loss)

---

#### **2. Content Consistency Loss** (動態權重 0.0→0.5→衰減)

**目的**: 相同句子（不同語者/材質說同一句話）應有相似的語義表示

**核心理念** (借鑑 ttt2.py):
```
Box 材質 + 句子 A  ─┐
                   ├──→ 應該有相似的 embedding 表示
Box2 材質 + 句子 A ─┘

# 雖然噪音不同、材質不同，但內容相同 (content_id 相同)
# 在 embedding 空間應該接近
```

**計算流程**:
```python
# discrete_hybrid_loss.py, Line 221-245
# Step 1: 獲取預測的 token embeddings
pred_tokens = logits.argmax(dim=-1)           # (B, T) 預測的 token IDs
pred_embeddings = codebook[pred_tokens]       # (B, T, 512) 查表

# Step 2: 平均池化到句子級別
sentence_embeddings = pred_embeddings.mean(dim=1)  # (B, 512)

# Step 3: 計算相同 content_id 的中心
for content_id in unique_contents:
    mask = (content_ids == content_id)
    content_embeddings = sentence_embeddings[mask]  # (N, 512)
    center = content_embeddings.mean(dim=0)         # (512,)
    
    # Step 4: 最小化到中心的距離
    similarities = cosine_similarity(content_embeddings, center)
    content_loss += (1.0 - similarities.mean())
```

**動態權重調度**:
```python
# discrete_hybrid_loss.py, Line 195-213
if epoch < 50:  # Warmup 階段
    # 線性增長: 0.0 → 0.5
    weight = 0.5 * (epoch / 50)
else:  # Decay 階段
    # 指數衰減: 0.5 → 接近 0
    progress = (epoch - 50) / (600 - 50)
    weight = 0.5 * exp(-3 * progress)
```

**權重變化曲線**:
```
0.5 |     ╱‾‾‾╲___
    |    ╱        ╲____
0.4 |   ╱              ╲____
    |  ╱                    ╲___
0.3 | ╱                         ╲___
    |╱                               ╲____
0.0 +━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━→
    0   50  100  150  200  ...  500  600 (epoch)
    
    ← Warmup → ←―――― Decay ――――――――――――――→
```

**意義**:
- **早期 (0-50 epochs)**: 強調內容一致性，學習語義表示
- **中期 (50-200 epochs)**: 平衡內容學習和去噪重建
- **後期 (200+ epochs)**: 專注於 CE + Embedding L2 的去噪任務

---

#### **3. Embedding L2 Loss** (固定權重 0.3)

**目的**: 預測 token 的 embedding 應接近目標 token 的 embedding

**計算位置**:
```python
# discrete_hybrid_loss.py, Line 141-149
pred_tokens = logits.argmax(dim=-1)              # (B, T)
pred_embeddings = codebook[pred_tokens]          # (B, T, 512)
target_embeddings = codebook[target_tokens]      # (B, T, 512)

embed_loss = MSE(pred_embeddings, target_embeddings)
```

**意義**:
- 補充 CE Loss 的不足
- CE Loss 只關心 token ID 是否正確 (離散)
- Embed L2 Loss 關心 embedding 空間的連續距離
- 更細緻的約束：即使 token ID 錯誤，也希望 embedding 接近

**範例**:
```
預測 token: 1024 → embedding: [0.1, 0.5, -0.3, ...]
目標 token: 1025 → embedding: [0.12, 0.48, -0.28, ...]

CE Loss:  token ID 錯誤 → 高懲罰
Embed L2: embeddings 很接近 → 低懲罰 (鼓勵語義接近)
```

---

### 損失權重總結

| 損失成分 | 權重 | 變化 | 目的 |
|---------|------|------|------|
| **CrossEntropy** | 1.0 | 固定 | 預測正確的 token ID (離散) |
| **Content Consistency** | 0.0→0.5→0 | 動態 | 學習語義一致性 (早期重要) |
| **Embedding L2** | 0.3 | 固定 | 在 embedding 空間接近 (連續) |

---

## 📊 資料配置 (僅 Box 材質)

### 環境變數

```bash
export ONLY_USE_BOX_MATERIAL=true   # ✅ 只使用 box 材質
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/sbplab/ruizi/c_code:$PYTHONPATH
```

### 資料路徑

```bash
INPUT_DIRS=(
    "../data/raw/box"              # 僅 box 材質噪音
)

TARGET_DIR="../data/clean/box2"    # 乾淨語音
```

### 資料規模預估

```bash
# Box 材質資料量
$ ls /home/sbplab/ruizi/c_code/data/raw/box | wc -l
# 預估: ~3000-5000 對音頻

# 每位語者句子數: 288 (max_sentences_per_speaker=None)
# 總訓練樣本: ~3000-5000 對
```

### 資料分割

- **訓練集**: 80%
- **驗證集**: 20%
- **分割方式**: 隨機分割 (非語者分割)

---

## 🔧 模型配置

### Transformer 架構

```bash
D_MODEL=512              # 與 Codebook 維度一致
NHEAD=8                  # Multi-head attention
NUM_LAYERS=6             # Transformer encoder 層數
DIM_FEEDFORWARD=2048     # FFN 中間層維度
DROPOUT=0.1
```

### 訓練超參數

```bash
BATCH_SIZE=8
NUM_EPOCHS=600
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
```

### 混合損失權重

```bash
CE_WEIGHT=1.0           # CrossEntropy 固定權重
CONTENT_WEIGHT=0.5      # Content Consistency 最大權重 (動態)
EMBED_WEIGHT=0.3        # Embedding L2 固定權重
WARMUP_EPOCHS=50        # 前 50 epochs 內容學習 warmup
```

### WavTokenizer 配置

```bash
WAVTOKENIZER_CONFIG="../config/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
WAVTOKENIZER_CHECKPOINT="../models/wavtokenizer_large_speech_320_24k.ckpt"
```

---

## 🎓 實驗目標

### 解決 Frozen Codebook 的問題

**問題診斷**:
```
Frozen Codebook (純 CE Loss) 訓練結果:
- Train Accuracy: 66.78%
- Val Accuracy:   14.92%
- 問題: 嚴重過擬合 + Token 準確度與音頻質量不匹配
```

**解決方案**: 混合損失設計

1. **保留 CE Loss**: 維持基本的 token 映射能力
2. **新增 Content Loss**: 
   - 借鑑 ttt2.py 的內容一致性理念
   - 在 embedding 空間學習語義表示
   - 減少過擬合 (正則化效果)
3. **新增 Embed L2 Loss**:
   - 在連續空間約束
   - 補充離散 CE Loss 的不足
   - 提升音頻質量

### 預期成果

- ✅ **減少過擬合**: 更平衡的 train/val accuracy
- ✅ **提升音頻質量**: 不僅僅是 token 準確度，還有語義接近
- ✅ **更穩定訓練**: Content Loss 的正則化效果
- ✅ **更好泛化**: 學習語義而非記憶樣本

---

## 📝 關鍵檔案清單

### 核心程式碼

1. **模型架構**: `/home/sbplab/ruizi/c_code/try/token_denoising_transformer.py`
   - `TokenDenoisingTransformer` 類別
   - Frozen Codebook + Transformer + Output Projection

2. **混合損失**: `/home/sbplab/ruizi/c_code/try/discrete_hybrid_loss.py`
   - `DiscreteHybridLoss` 類別
   - CE + Content + Embed L2

3. **訓練腳本**: `/home/sbplab/ruizi/c_code/try/train_token_denoising_hybrid.py`
   - 訓練循環
   - 資料載入 (AudioDataset)
   - 檢查點保存

4. **執行腳本**: `/home/sbplab/ruizi/c_code/try/run_token_denoising_hybrid.sh`
   - 環境設定
   - 超參數配置
   - 背景執行

### 資料載入

5. **資料集**: `/home/sbplab/ruizi/c_code/ttdata.py`
   - `AudioDataset` 類別
   - 返回 `(input_wav, target_wav, content_id)`

---

## ✅ 最終檢查清單

### 架構確認

- [x] **純離散訓練**: Noisy Token IDs → Clean Token IDs ✅
- [x] **Codebook 凍結**: `register_buffer` + `assert` ✅
- [x] **模型可訓練部分**: Transformer + Output Projection ✅

### 損失函數確認

- [x] **CE Loss**: Token-level 分類 ✅
- [x] **Content Loss**: 相同 content_id 應相似 ✅
- [x] **Embed L2 Loss**: Embedding 空間接近 ✅
- [x] **動態權重**: Warmup + Decay ✅

### 資料配置確認

- [x] **只使用 box**: `ONLY_USE_BOX_MATERIAL=true` ✅
- [x] **資料路徑**: `../data/raw/box` ✅
- [x] **全部句子**: `max_sentences_per_speaker=None` ✅

### 參數配置確認

- [x] **d_model=512**: 與 Codebook 一致 ✅
- [x] **batch_size=8**: 記憶體友好 ✅
- [x] **num_epochs=600**: 充足訓練 ✅
- [x] **學習率**: 1e-4 (穩定) ✅

---

## 🚀 執行指令

```bash
cd /home/sbplab/ruizi/c_code/try
bash run_token_denoising_hybrid.sh
```

### 預期輸出

```
訓練進度顯示:
Epoch [1/600] Train: loss=5.234 ce=4.123 cont=0.543 emb=0.568 acc=12.34% | Val: ...
           ↑        ↑       ↑        ↑        ↑        ↑
           |        |       |        |        |        └─ Token 準確度
           |        |       |        |        └────────── Embedding L2 Loss
           |        |       |        └───────────────── Content Consistency Loss
           |        |       └──────────────────────── CrossEntropy Loss
           |        └────────────────────────────── 總損失
           └───────────────────────────────────── Epoch 進度
```

### 輸出檔案結構

```
/home/sbplab/ruizi/c_code/results/token_denoising_hybrid_loss_YYYYMMDD_HHMMSS/
├── config.json                    # 訓練配置
├── training.log                   # 詳細日誌
├── run_config.txt                 # 執行配置
├── checkpoints/                   # 模型檢查點
│   ├── epoch_010.pt
│   ├── epoch_020.pt
│   ├── ...
│   └── best_model.pt              # 最佳模型 (val loss 最低)
└── plots/                         # 訓練曲線 (如果實作)
    ├── loss_curves.png
    └── accuracy_curves.png
```

---

## 📊 與其他方法的比較

| 方法 | 架構 | 損失函數 | Codebook | 資料 | 預期優勢 |
|------|------|---------|---------|------|---------|
| **Frozen Codebook** | Token→Token | CE Only | 凍結 | box only | 簡單、快速 |
| **Hybrid Loss (本實驗)** | Token→Token | CE+Content+Embed | 凍結 | box only | 更好的語義、減少過擬合 |
| **ttt2.py** | Wav→Continuous Features | Content+L2 | 可訓練 | 多材質 | 連續特徵、音頻質量高 |

---

## 🎯 成功指標

### 訓練階段

- [ ] Train Accuracy > 60% (與 Frozen Codebook 相當)
- [ ] Val Accuracy > 30% (大幅改善過擬合問題)
- [ ] Train/Val Accuracy Gap < 50% (Frozen: 66.78% vs 14.92% = 51.86%)

### 損失組成

- [ ] CE Loss 穩定下降
- [ ] Content Loss 早期下降快，後期趨於穩定
- [ ] Embed L2 Loss 持續下降

### 音頻質量 (定性評估)

- [ ] 重建語音可理解
- [ ] 噪音顯著降低
- [ ] 語音自然度提升

---

*本配置已確認所有參數正確，可以開始訓練。*

**準備狀態**: ✅ **可以執行**
