# Distance Scores 詳細技術文檔

## 目錄

- [核心定義](#核心定義)
- [物理意義](#物理意義)
- [計算過程](#計算過程)
- [資料流程](#資料流程)
- [訓練使用](#訓練使用)
- [設計理念](#設計理念)
- [實作細節](#實作細節)

---

## 核心定義

### Distance Scores 是什麼？

**簡答**：`distance_scores (B, T, 4096)` 是指每個時間步的音訊特徵向量到所有 4096 個 Codebook Embeddings 的距離。

**詳細定義**：

```
distance_scores[b, t, k] = -||feature[b,t] - codebook[k]||²

其中:
- feature[b,t]:  第 b 個樣本第 t 個時間步的音訊特徵向量 (512-D)
- codebook[k]:   第 k 個 VQ token 的 embedding 向量 (512-D)
- ||·||²:        歐氏距離的平方
- 負號:          距離越小（負值絕對值越小）表示越接近
```

### 維度說明

```
資料批次中的 distance_scores:

距離矩陣:     (B, T, 4096)
             │  │   │
             │  │   └─ 到所有 4096 個 codebook embeddings 的距離
             │  └───── T 個時間步（音訊長度）
             └──────── Batch size

具體例子:
- B = 28      (batch size)
- T = 75      (約 3 秒音訊，24kHz / 320 = 75 frames/sec)
- 4096        (WavTokenizer codebook size)

總共: 28 × 75 × 4096 = 8,601,600 個距離值
```

---

## 物理意義

### VQ-VAE Quantization 過程

```
Audio Waveform (24kHz)
        ↓
WavTokenizer Encoder (CNN layers)
        ↓
Audio Features: (B, T, 512)
        ↓
        ┌─────────────────────────────────────┐
        │   VQ Quantization (關鍵步驟)        │
        │                                     │
        │   對每個 feature[t] (512-D):        │
        │                                     │
        │   1. 計算到所有 codebook 的距離:    │
        │      dist[k] = -||feat - cb[k]||²  │
        │                                     │
        │   2. 選擇最近的:                    │
        │      token = argmax(dist)           │
        │                                     │
        │   3. 🎯 保存完整距離矩陣 (4096,)    │
        └─────────────────────────────────────┘
        ↓
Token Sequence: (B, T)
Distance Matrix: (B, T, 4096)  ← 我們額外捕獲的！
```

### 距離的語義含義

**為什麼距離重要？**

在 VQ-VAE 的 codebook 空間中，距離反映了**聲學相似度**：

```
例子: 某個時間步發出 "ah" 音

距離矩陣 (4096 個值):

Token 0    (無關音素 "t"):      -3.45   很遠
Token 1    (無關音素 "k"):      -2.89   遠
...
Token 450  (類似 "ah" 變體):    -0.08   接近
Token 451  (類似 "ah" 變體):    -0.06   接近
Token 452  (類似 "ah" 變體):    -0.03   很接近
Token 453  ("ah" 標準發音):     -0.01   最接近！⭐
Token 454  ("ah" 輕微變體):     -0.04   接近
Token 455  ("ah" 鼻音化):       -0.07   接近
...
Token 2100 (完全不同 "sh"):     -4.21   很遠
...
Token 4095 (無關音素 "p"):      -3.78   很遠
```

**語義結構**：

```
距離值分布:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        -5.0        -2.0      -0.5  -0.01
         │           │         │     │
     完全不同    稍微相關   接近  最接近
     (其他音素) (同類音素) (變體) (選中)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VQ 只使用: argmax (最接近的那個)
我們保留: 完整的 4096 個距離 (包含相似度結構)
```

### 為什麼這很有價值？

**傳統 One-Hot Target 的問題**：

```
One-Hot (只知道正確答案):
    Token 453:  1.0  ← 正確
    其他:       0.0  ← 錯誤

    ✗ 不知道 Token 454 和 Token 2100 哪個"更錯"
    ✗ 丟失了語義相似度資訊
```

**Distance-based Soft Target 的優勢**：

```
Soft Target (知道相似度結構):
    Token 453:  0.60  ← 最好
    Token 454:  0.20  ← 也可接受
    Token 452:  0.15  ← 勉強可以
    Token 451:  0.04  ← 有點遠
    Token 2100: 0.0001 ← 完全錯誤

    ✓ 保留了"接近但未被選中"的候選資訊
    ✓ 符合人類聽覺的"容錯"特性
    ✓ 類似 Knowledge Distillation，但 Teacher 是 VQ-VAE
```

---

## 計算過程

### 數學公式

**歐氏距離展開**：

```
dist[i, j] = -||feature[i] - codebook[j]||²

展開為高效計算形式:
= -(||feature[i]||² - 2·feature[i]·codebook[j] + ||codebook[j]||²)

矩陣形式 (batch 計算):
dist = -(
    features.pow(2).sum(dim=1, keepdim=True)     # (N, 1)
    - 2 * features @ codebook.T                   # (N, K) 內積
    + codebook.pow(2).sum(dim=1, keepdim=True).T  # (1, K)
)

結果: (N, K) 距離矩陣
其中 N = B×T, K = 4096
```

### 程式碼實作

**Hook 捕獲 Distances**（`preprocess_zeroshot_cache_with_distances_hdf5.py:87-94`）：

```python
def hooked_quantize(features):
    """
    攔截 WavTokenizer 的 quantize 過程，捕獲 distance 計算

    Args:
        features: (B, T, D) 或 (B*T, D) 音訊特徵

    Returns:
        原始 quantize 的結果 + 副作用：捕獲 distances
    """
    # 調用原始 quantize
    result = original_quantize(features)

    # Flatten features
    if features.dim() == 3:
        B, T, D = features.shape
        features_flat = features.reshape(-1, D)  # (B*T, D)
    else:
        features_flat = features  # 已經是 (N, D)

    # 獲取 codebook embeddings
    embed = codebook.embed.t()  # (K, D) where K=4096

    # 計算距離矩陣
    dist = -(
        features_flat.pow(2).sum(1, keepdim=True)  # (N, 1)
        - 2 * features_flat @ embed                 # (N, K)
        + embed.pow(2).sum(0, keepdim=True)         # (1, K)
    )  # 結果: (N, K) = (B*T, 4096)

    # 保存到 capture 實例
    capture_instance.distances = dist.detach().cpu()

    return result
```

### 數值範圍

**典型距離值**：

```
距離範圍: 負無窮 ~ 0
         (理論上，實際上有下界)

實際觀察:
- 最接近的 token:  -0.001 ~ -0.1
- 稍微接近:        -0.1 ~ -1.0
- 較遠:            -1.0 ~ -3.0
- 很遠:            -3.0 ~ -10.0

為什麼是負數?
- 計算的是 -||·||² (負的距離平方)
- 這樣 argmax 就能選最近的 (數值最大的)
- 與 softmax 相容 (值越大機率越高)
```

---

## 資料流程

### 預處理階段

**完整流程**（`preprocess_zeroshot_cache_with_distances_hdf5.py`）：

```
Step 1: 載入音訊對
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Noisy Audio (帶雜訊)  ┐
Clean Audio (乾淨)    ┘ → DataLoader 載入


Step 2: 安裝 Distance Capture Hook
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WavTokenizer.encodec.quantizer.quantize
        ↓
    安裝 Hook (攔截 distance 計算)
        ↓
    創建兩個 Capture 實例:
    - distance_capture_noisy
    - distance_capture_clean


Step 3: 編碼 Noisy Audio
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
distance_capture_noisy.activate()
    ↓
noisy_batch → WavTokenizer.encode_infer()
    ├─ Encoder: Audio → Features (B, T, 512)
    ├─ Quantizer: Features → Tokens (B, T)
    └─ Hook 捕獲: distances (B*T, 4096) ⭐
    ↓
distance_capture_noisy.deactivate()

獲得:
- noisy_tokens:     (B, T)
- noisy_distances:  (B*T, 4096) → reshape → (B, T, 4096)


Step 4: 編碼 Clean Audio (同樣流程)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
distance_capture_clean.activate()
    ↓
clean_batch → WavTokenizer.encode_infer()
    ↓
distance_capture_clean.deactivate()

獲得:
- clean_tokens:     (B, T)
- clean_distances:  (B, T, 4096) ⭐ 這個最重要！


Step 5: 提取 Speaker Embedding
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
clean_batch → Speaker Encoder
    ↓
speaker_emb: (B, 192)


Step 6: 保存到 HDF5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
單一 HDF5 檔案: cache_with_distances.h5 (32GB)

結構:
/train
  ├─ noisy_tokens/     (7776 samples, variable length)
  ├─ clean_tokens/     (7776 samples, variable length)
  ├─ noisy_distances/  (7776 samples, T×4096)
  ├─ clean_distances/  (7776 samples, T×4096) ⭐
  └─ speaker_emb/      (7776 samples, 192-D)

/val
  ├─ noisy_tokens/     (1440 samples, variable length)
  ├─ clean_tokens/     (1440 samples, variable length)
  ├─ noisy_distances/  (1440 samples, T×4096)
  ├─ clean_distances/  (1440 samples, T×4096) ⭐
  └─ speaker_emb/      (1440 samples, 192-D)

壓縮: gzip level 4
記憶體: Memory-mapped (載入時 RAM <500MB)
```

### 訓練階段

**資料載入與使用**（`train_with_distances.py` + `data_zeroshot_hdf5_v2.py`）：

```
Step 1: HDF5 Dataset
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class HDF5ZeroShotDataset:
    def __getitem__(self, idx):
        return {
            'token_ids':        # (T,) - noisy tokens
            'distance_scores':  # (T, 4096) - clean distances ⭐
            'mel_spectrogram':  # 未使用
            'speaker_emb':      # (192,)
        }

注意:
- 'token_ids' 實際是 noisy_tokens
- 'distance_scores' 實際是 clean_distances
- 命名來自資料集的 key mapping


Step 2: DataLoader Collation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def cached_collate_fn_with_distances(batch):
    """
    處理變長序列，padding 到 batch 最大長度
    """
    # 找出 batch 中最長的序列
    max_len = max(item['token_ids'].shape[0] for item in batch)

    # Padding (PAD_TOKEN = 4096)
    padded_tokens = []
    padded_distances = []

    for item in batch:
        T = item['token_ids'].shape[0]
        pad_len = max_len - T

        # Pad tokens with 4096
        tokens = F.pad(item['token_ids'], (0, pad_len), value=4096)

        # Pad distances with zeros (4096 維向量)
        distances = F.pad(
            item['distance_scores'],
            (0, 0, 0, pad_len),
            value=0
        )

        padded_tokens.append(tokens)
        padded_distances.append(distances)

    return {
        'noisy_tokens':   torch.stack(padded_tokens),      # (B, T)
        'clean_tokens':   torch.stack(padded_targets),     # (B, T)
        'distance_scores': torch.stack(padded_distances),  # (B, T, 4096)
        'speaker_emb':    torch.stack(speaker_embs)        # (B, 192)
    }


Step 3: Training Loop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
for batch in train_loader:
    # 解包 batch
    noisy_tokens = batch['noisy_tokens'].to(device)        # (B, T)
    clean_tokens = batch['clean_tokens'].to(device)        # (B, T)
    distance_scores = batch['distance_scores'].to(device)  # (B, T, 4096)
    speaker_emb = batch['speaker_emb'].to(device)          # (B, 192)

    # Forward pass
    logits = model(noisy_tokens, speaker_emb, return_logits=True)
    # logits: (B, T, 4096)

    # 計算 Loss (根據實驗類型)
    if loss_type == 'ce':
        # Baseline: 只用 hard target
        loss = CE(logits, clean_tokens)

    elif loss_type == 'soft':
        # Soft Target: 使用 distance_scores ⭐
        soft_targets = F.softmax(distance_scores / temperature, dim=-1)
        loss = KL_div(logits, soft_targets)

    elif loss_type == 'hybrid':
        # Hybrid: 混合使用
        soft_targets = F.softmax(distance_scores / temperature, dim=-1)
        loss = α*KL_div(logits, soft_targets) + β*CE(logits, clean_tokens)

    # Backward & Update
    loss.backward()
    optimizer.step()
```

---

## 訓練使用

### 為什麼使用 Clean Distances？

**關鍵問題**：我們有 noisy_distances 和 clean_distances，訓練時用哪個？

**答案**：使用 **clean_distances** 作為 soft target。

**原因分析**：

```
訓練目標: 從 Noisy Tokens 預測 Clean Tokens

輸入:  noisy_tokens  (B, T)        ← 帶雜訊的觀察
       speaker_emb   (B, 192)      ← 說話者資訊
            ↓
    Denoising Model
            ↓
輸出:  pred_logits   (B, T, 4096)  ← 預測的 token 分布

目標:  clean_tokens  (B, T)        ← 正確答案 (hard)
       clean_distances (B, T, 4096) ← 相似度結構 (soft)


為什麼用 Clean Distances?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 監督信號應該來自"正確答案"
   - clean_distances 告訴我們"哪些 token 接近正確答案"
   - 這是我們希望模型學習的目標分布

2. Noisy distances 已經被破壞
   - noisy_distances 反映的是錯誤的相似度
   - 我們不希望模型學習雜訊的模式

3. Knowledge Distillation 類比
   - Clean Audio → Teacher Model 的輸出
   - clean_distances → Teacher 的 soft labels
   - Model → Student 要學習 Teacher 的知識
```

**具體例子**：

```
假設正確發音是 "ah" (Token 453):

Clean Distances (來自正確音訊):
    Token 452:  -0.03   → softmax → 0.15  ← 接近
    Token 453:  -0.01   → softmax → 0.60  ← 最好 ⭐
    Token 454:  -0.04   → softmax → 0.20  ← 接近
    Token 2100: -4.21   → softmax → 0.0001 ← 錯誤

    ✓ 提供正確的相似度結構
    ✓ 告訴模型：452 和 454 也是可接受的


Noisy Distances (來自雜訊音訊，假設雜訊讓它變成 "aah"):
    Token 450:  -0.02   → 可能選到錯誤的 token
    Token 453:  -0.15   → 正確 token 排名下降
    Token 780:  -0.03   → 雜訊引入的錯誤候選

    ✗ 包含雜訊引入的錯誤模式
    ✗ 不應該作為學習目標
```

### Loss Function 的使用

**Baseline (CE Loss)**：

```python
# 只使用 hard target
loss = F.cross_entropy(
    logits.view(-1, 4096),      # (B*T, 4096)
    clean_tokens.view(-1),      # (B*T,)
    ignore_index=4096           # 忽略 PAD
)

優點: 簡單、穩定
缺點: 不利用 distance 資訊
```

**Soft Target Loss (Exp1, Exp2)**：

```python
# 使用 clean_distances 作為 soft target
class SoftTargetLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5):
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, pred_logits, target_distances, target_tokens):
        """
        Args:
            pred_logits:      (B, T, 4096) - 模型預測
            target_distances: (B, T, 4096) - clean_distances ⭐
            target_tokens:    (B, T)       - clean_tokens
        """
        B, T, C = pred_logits.shape

        # Step 1: 將 distances 轉為 soft targets
        soft_targets = F.softmax(
            target_distances / self.temperature,  # Temperature scaling
            dim=-1
        )  # (B, T, 4096) 機率分布

        # Step 2: 計算 KL Divergence (soft loss)
        pred_log_probs = F.log_softmax(
            pred_logits / self.temperature,
            dim=-1
        )
        soft_loss = F.kl_div(
            pred_log_probs.reshape(-1, C),
            soft_targets.reshape(-1, C),
            reduction='batchmean'
        ) * (self.temperature ** 2)  # Temperature 補償

        # Step 3: 計算 Cross Entropy (hard loss)
        hard_loss = F.cross_entropy(
            pred_logits.reshape(-1, C),
            target_tokens.reshape(-1),
            ignore_index=4096
        )

        # Step 4: 混合
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return total_loss
```

**參數說明**：

```
temperature (T):
- 控制 soft target 的平滑度
- T=1.0: 尖銳（接近 one-hot）
- T=2.0: 平滑（推薦） ⭐
- T=5.0: 很平滑（可能太模糊）

alpha (α):
- 控制 soft/hard 的權重
- α=0.0: 純 hard target (退化為 baseline)
- α=0.5: 平衡 (Exp1) ⭐
- α=0.7: 更重視 soft target (Exp2) ⭐
- α=1.0: 純 soft target
```

**Hybrid Loss (Exp3)**：

```python
# 三重監督
class HybridDistanceLoss(nn.Module):
    def forward(self, pred_logits, target_distances, target_tokens):
        # 1. Soft Target Loss (KL Divergence)
        soft_targets = F.softmax(target_distances / T, dim=-1)
        soft_loss = KL_div(pred_logits, soft_targets)

        # 2. Hard Target Loss (Cross Entropy)
        hard_loss = CE(pred_logits, target_tokens)

        # 3. Distribution Matching (MSE)
        pred_probs = F.softmax(pred_logits, dim=-1)
        wasserstein_loss = F.mse_loss(pred_probs, soft_targets)

        # 總 Loss
        total = α*soft_loss + β*hard_loss + γ*wasserstein_loss
        # 例: 0.3*soft + 0.3*hard + 0.4*wass

        return total
```

---

## 設計理念

### 核心創新點

**1. VQ-VAE Distances 作為免費的 Knowledge Distillation**

```
傳統 Knowledge Distillation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input → Large Teacher Model → Soft Labels
                ↓
        Train Student Model

問題:
- 需要訓練/維護一個大型 Teacher Model
- 計算成本高
- 需要額外的推理時間


VQ-VAE Distance-based KD (本方法):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input → WavTokenizer (已預訓練) → Distances
                                    ↓
                            Soft Labels (免費!)

優勢:
✓ VQ-VAE 已經預訓練好（免費的 Teacher）
✓ Distances 在 quantization 過程中自然產生
✓ 無需額外計算（只需用 Hook 捕獲）
✓ 包含豐富的語義相似度資訊
```

**2. 符合人類聽覺的容錯機制**

```
人類聽覺特性:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
聽到 "ah" 音時：
- 能分辨"標準 ah" vs "稍微鼻音的 ah" vs "完全不同的 sh"
- 對"接近的變體"有一定容忍度
- 能理解"程度感"（多接近、多不同）


傳統 One-Hot Target:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Token 453 (標準 ah):       正確 ✓
Token 454 (鼻音 ah):       錯誤 ✗  ← 懲罰一樣
Token 2100 (完全不同 sh):  錯誤 ✗  ← 懲罰一樣

問題: 不符合人類直覺


Distance-based Soft Target:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Token 453 (標準 ah):       0.60  ← 最好
Token 454 (鼻音 ah):       0.20  ← 可接受
Token 455 (輕微變體):     0.15  ← 勉強
Token 2100 (sh):           0.0001 ← 完全錯

優勢: ✓ 反映真實的相似度梯度
     ✓ 符合人類聽覺特性
```

**3. 保留 VQ 量化過程中的資訊**

```
VQ Quantization 的資訊損失:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

連續特徵空間 (512-D):
    feature = [0.15, -0.30, ..., 0.52]
        ↓
    計算到所有 codebook 的距離
        ↓
    選擇最近的 → token_id = 453
        ↓
    丟棄其他 4095 個距離資訊 ← 資訊損失！ ✗


我們的方法:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

連續特徵空間 (512-D):
    feature = [0.15, -0.30, ..., 0.52]
        ↓
    計算到所有 codebook 的距離
        ↓
    選擇最近的 → token_id = 453
        ↓
    保存完整距離矩陣 (4096,) ← 保留資訊！ ✓
        ↓
    用於訓練時的 soft target
```

### 實驗假設

**為什麼這個方法可能有效？**

```
假設 1: Soft Target 提供更豐富的監督信號
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hard Target:   "答案是 453，其他都錯"
Soft Target:   "453 最好(60%)，454 也可以(20%)，452 勉強(15%)..."

預期:
- 模型學會"程度感"
- 對接近的候選有更好的理解
- 訓練更穩定（梯度更平滑）


假設 2: 減輕過擬合到離散 token
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hard Target 問題:
- 強迫模型輸出 one-hot 分布
- 可能過於自信（overconfident）
- 對訓練資料過擬合

Soft Target 優勢:
- 鼓勵模型輸出平滑分布
- 更好的不確定性估計
- 泛化能力可能更強


假設 3: 產生更自然的音訊
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hard Target → 尖銳的 token 預測 → 可能產生"數位感"的音訊
Soft Target → 平滑的分布 → 可能產生更自然、連續的音訊

預期音訊品質:
- Baseline: 清晰但可能有"顆粒感"
- Soft α=0.5: 平衡，自然度提升
- Soft α=0.7: 更平滑，但可能準確率下降
```

---

## 實作細節

### 預處理效能

**記憶體使用**：

```
原始方法 (一次載入所有 .pt 檔案):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7776 train + 1440 val samples
每個樣本:
- token_ids:        75 × 8 bytes
- distance_scores:  75 × 4096 × 4 bytes ≈ 1.2 MB
- mel_spectrogram:  80 × 646 × 2 × 4 bytes ≈ 400 KB
- speaker_emb:      192 × 4 bytes

總計: 約 79 GB RAM ← OOM Killer! ✗


HDF5 串流式預處理:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
逐檔處理，立即寫入 HDF5:
- 峰值記憶體: <500 MB ✓
- 處理時間: 61 分鐘
- 輸出檔案: 32 GB (gzip 壓縮)
- 壓縮率: 59% (32GB / 79GB)
```

**HDF5 檔案結構**：

```
cache_with_distances.h5 (32 GB)
│
├─ /train (7776 samples)
│   ├─ token_ids/
│   │   ├─ 0: (75,) int32
│   │   ├─ 1: (73,) int32
│   │   └─ ... (變長)
│   │
│   ├─ distance_scores/
│   │   ├─ 0: (75, 4096) float32
│   │   ├─ 1: (73, 4096) float32
│   │   └─ ...
│   │
│   ├─ mel_spectrogram/
│   │   └─ ...
│   │
│   └─ speaker_emb/
│       ├─ 0: (192,) float32
│       └─ ...
│
└─ /val (1440 samples)
    └─ (相同結構)

壓縮: gzip level 4
Chunks: (1, max_seq_len) 或 (1, max_seq_len, 4096)
訪問模式: Memory-mapped (SWMR mode)
```

### 訓練效能

**記憶體與速度**：

```
訓練時記憶體使用:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
模型參數:     14.8M × 4 bytes = 59 MB
優化器狀態:   2× 模型參數 = 118 MB
Batch 數據:   28 × 75 × 4096 × 4 bytes ≈ 34 MB
梯度:         同模型參數 = 59 MB
其他:         約 30 MB

總計: < 300 MB per GPU ✓


訓練速度 (batch_size=28):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseline (CE Loss):
- 速度: ~14-15 it/s
- 每個 epoch: ~19 分鐘

Soft Target (KL Div):
- 速度: ~13-14 it/s
- 每個 epoch: ~20 分鐘
- 額外開銷: softmax + KL 計算

Hybrid Loss:
- 速度: ~12-13 it/s
- 每個 epoch: ~21 分鐘
- 額外開銷: 三個 loss 計算


總訓練時間估計 (200 epochs):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseline:  200 × 19 min = 3,800 min ≈ 63 hours
Soft:      200 × 20 min = 4,000 min ≈ 67 hours
Hybrid:    200 × 21 min = 4,200 min ≈ 70 hours
```

### Padding 處理

**關鍵問題**：變長序列如何處理？

```
Batch 中的序列長度:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sample 0: 75 frames  (3.0 秒音訊)
Sample 1: 68 frames  (2.72 秒)
Sample 2: 73 frames  (2.92 秒)
Sample 3: 75 frames  (3.0 秒)

Batch max_len = 75


Padding 策略:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Token IDs Padding:
   原始: [453, 234, ..., 890]  (68,)
   Pad:  [453, 234, ..., 890, 4096, 4096, ...]  (75,)
         └─────────────┬─────────┘  └────┬────┘
              原始數據            PAD_TOKEN=4096

2. Distance Scores Padding:
   原始: (68, 4096)
   Pad:  (75, 4096)
         ├─ 前 68 個: 原始 distances
         └─ 後 7 個:  全 0 向量 (4096,)

3. Loss 計算中忽略 Padding:
   - CE Loss: ignore_index=4096
   - Soft Loss: mask 掉 padding 位置
```

**實作細節**（`data_zeroshot_hdf5_v2.py:115-140`）：

```python
def cached_collate_fn_with_distances(batch):
    """Dynamic padding for variable-length sequences"""

    # 找出 batch 最大長度
    max_len = max(item['token_ids'].shape[0] for item in batch)

    # Pad 每個樣本
    padded_tokens = []
    padded_distances = []

    for item in batch:
        T = item['token_ids'].shape[0]
        pad_len = max_len - T

        if pad_len > 0:
            # Pad token_ids with PAD_TOKEN (4096)
            tokens = torch.cat([
                item['token_ids'],
                torch.full((pad_len,), PAD_TOKEN, dtype=torch.long)
            ])

            # Pad distance_scores with zeros
            distances = torch.cat([
                item['distance_scores'],
                torch.zeros((pad_len, 4096), dtype=torch.float32)
            ])
        else:
            tokens = item['token_ids']
            distances = item['distance_scores']

        padded_tokens.append(tokens)
        padded_distances.append(distances)

    return {
        'noisy_tokens': torch.stack(padded_tokens),
        'distance_scores': torch.stack(padded_distances),
        # ...
    }
```

### 數值穩定性

**Softmax 溫度縮放**：

```python
# 問題: distances 範圍很大 ([-10, 0])
distances = tensor([[-5.2, -0.01, -3.4, ...]])  # (B, T, 4096)

# 直接 softmax 會導致數值不穩定
probs = F.softmax(distances, dim=-1)
# 可能結果: [0.0000, 0.9999, 0.0001, ...]  太尖銳！

# 解決: Temperature Scaling
T = 2.0
scaled = distances / T  # [-2.6, -0.005, -1.7, ...]
probs = F.softmax(scaled, dim=-1)
# 結果: [0.02, 0.65, 0.10, ...]  更平滑 ✓
```

**KL Divergence 穩定性**：

```python
# 問題: KL divergence 對 0 機率敏感
# KL(P||Q) = Σ P(i) log(P(i)/Q(i))
#          = Σ P(i) [log(P(i)) - log(Q(i))]

# PyTorch 實作（穩定版本）:
loss = F.kl_div(
    input=pred_log_probs,   # log_softmax(logits)  ← 對數空間
    target=soft_targets,     # softmax(distances)
    reduction='batchmean',
    log_target=False         # target 不是對數
)

# Temperature 補償
loss = loss * (temperature ** 2)

# 為什麼需要 T^2 補償?
# - Softmax 縮放後梯度被縮小了 T 倍
# - KL divergence 又縮小了 T 倍
# - 總共縮小 T^2 倍，需要補償回來
```

---

## 總結

### 關鍵要點

1. **Distance Scores 定義**：
   - 每個時間步的音訊特徵向量到所有 4096 個 codebook embeddings 的負歐氏距離平方
   - 保留了 VQ quantization 過程中的完整相似度資訊

2. **為什麼重要**：
   - 傳統 one-hot target 只知道"正確答案"
   - Distance-based soft target 還知道"哪些接近、哪些遠離"
   - 提供更豐富的監督信號

3. **如何計算**：
   - 在 WavTokenizer 的 quantize 過程中用 Hook 捕獲
   - 公式：`dist[i,j] = -||feature[i] - codebook[j]||²`
   - 高效批次計算：矩陣乘法 + 廣播

4. **如何使用**：
   - 預處理：保存 clean_distances 到 HDF5
   - 訓練：轉為 soft targets (經過 softmax)
   - Loss：KL divergence 或混合 loss

5. **核心創新**：
   - 免費的 Knowledge Distillation (VQ-VAE 作為 Teacher)
   - 符合人類聽覺的容錯機制
   - 保留量化過程中的資訊，避免損失

### 實驗設計邏輯

```
相同模型架構 (ZeroShotDenoisingTransformer, 14.8M 參數)
        ↓
只改變 Loss Function:

┌──────────┬────────────────────────────────────┐
│ Baseline │  CE Loss (hard target only)        │
│ Exp1     │  Soft Target α=0.5 (平衡)          │
│ Exp2     │  Soft Target α=0.7 (更重視 soft)   │
│ Exp3     │  Hybrid (三重監督)                 │
└──────────┴────────────────────────────────────┘

預期:
- Baseline: 準確率高，但可能音訊尖銳
- Exp1/2:   準確率略降，但音訊更自然
- Exp3:     理論最優，但訓練可能不穩定

評估:
- Token Accuracy (量化指標)
- Perplexity (不確定性)
- 音訊品質 (主觀聽感) ← 最終目標！
```

---

## 參考資料

### 相關檔案

- 預處理：`preprocess_zeroshot_cache_with_distances_hdf5.py`
- 資料載入：`data_zeroshot_hdf5_v2.py`
- Loss 函數：`losses_with_distances.py`
- 訓練腳本：`train_with_distances.py`
- 模型定義：`model_zeroshot.py`

### 相關論文概念

- **VQ-VAE**: Neural Discrete Representation Learning (van den Oord et al., 2017)
- **Knowledge Distillation**: Distilling the Knowledge in a Neural Network (Hinton et al., 2015)
- **Soft Targets**: Born Again Neural Networks (Furlanello et al., 2018)

### 技術棧

- PyTorch 2.x
- HDF5 (h5py) - Memory-mapped I/O
- WavTokenizer - Pretrained audio codec
- Transformer Encoder - Sequence modeling

---

**文檔版本**: v1.0
**最後更新**: 2025-11-22
**作者**: Claude Code
**實驗**: Zero-Shot Speaker Denoising with VQ Distance Soft Targets
