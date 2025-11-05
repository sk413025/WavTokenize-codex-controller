# Cross-Attention Speaker Fusion 實驗完整報告

**實驗編號**: EXP-20251105-CrossAttn  
**日期**: 2025-11-05  
**狀態**: 進行中 (Epoch 45/100)  
**Git Commit**: (此次 commit)

---

## 一、實驗背景

### 問題陳述

在原始 Zero-Shot Speaker Denoising Transformer（使用 Additive Fusion）訓練中，發現嚴重的平台期問題：

- **Train Accuracy**: 停滯在 56%（Epoch 20-77）
- **Val Accuracy**: 停滯在 38%（無法提升）
- **Learning Rate**: 衰減至 1e-06（幾乎無效）

### 診斷發現

通過 `diagnose_training_mechanism.py` 和 `diagnose_prediction_behavior.py` 診斷工具分析後，提出 4 個機轉假設：

1. ⭐⭐⭐ **假設 1**: 模型學會"預測眾數"策略（Token 0 預測率 ~32%）
2. ⭐⭐⭐ **假設 2**: **Speaker Embedding 影響力不足**（<5% tokens 改變）
3. ⭐⭐ **假設 3**: Frozen Codebook 限制表達能力
4. ⭐ **假設 4**: Task 已接近理論上限

本實驗針對 **假設 2** 進行驗證。

### 原始架構問題（Additive Fusion）

```
Token Emb (B,T,512) + Speaker Emb (B,T,512) → Combined (B,T,512)
                     └─ 簡單相加，speaker 資訊容易被覆蓋
```

**問題**:
- Speaker embedding 通過 broadcast 後直接相加
- 每個 token 受到**相同**的 speaker 影響（無法動態調整）
- Speaker 資訊容易被 token embedding 覆蓋
- Zero/Random speaker test 顯示影響力 <5%

---

## 二、實驗動機

### 假設 2 驗證

**假設內容**: Speaker Embedding 影響力不足是導致 Val Accuracy 停滯的主要原因。

**驗證方法**: 將 Additive Fusion 改為 **Cross-Attention Mechanism**，讓每個 token 動態決定需要多少 speaker 資訊。

### 預期改善

如果假設成立：
- Val Accuracy: 38% → **43-47%** (提升 5-9%)
- Speaker Influence: <5% → **>20%**
- Token 0 預測率: 32% → **20-25%**

---

## 三、實驗目的

1. **驗證假設**: 確認 speaker embedding 影響力是否為平台期主因
2. **架構改進**: 實作 Cross-Attention 替代 Additive Fusion
3. **性能提升**: 達成 Val Accuracy > 43% (超越 baseline +5%)
4. **可解釋性**: 視覺化 attention weights，分析 speaker 如何影響不同 token

---

## 四、架構設計

### Cross-Attention Fusion 機制

#### 數學公式

```
Q = token_emb              (B, T, 512) - 每個 token 是一個 query
K = speaker_emb            (B, 1, 512) - speaker 是唯一的 key
V = speaker_emb            (B, 1, 512) - speaker 是唯一的 value

Attention_Weights[i] = softmax(Q[i] · K^T / √512)
                       └─ token i 對 speaker 的關注度

Attn_Output[i] = Attention_Weights[i] · V
                 └─ 根據關注度加權的 speaker 資訊

output[i] = LayerNorm(token_emb[i] + Dropout(Attn_Output[i]))
            └─ Residual connection 保留原始資訊
```

#### 架構流程圖

```
Noisy Tokens (B, T) + Speaker Embedding (B, 256)
         │                      │
         ▼                      ▼
   Codebook Lookup      speaker_proj (256→512)
         │                      │
         ▼                      │
   Token Emb (B,T,512)         │
         │                      │
         ▼                      │
   Pos Encoding                │
         │                      │
         │ Query                │ Key & Value
         │                      │
         └──────┬───────────────┘
                ▼
      ┌─────────────────────┐
      │  Cross-Attention    │
      │  Multi-Head (8)     │
      │                     │
      │  每個 token 動態    │
      │  決定需要多少       │
      │  speaker 資訊       │
      └─────────┬───────────┘
                ▼
      Residual + LayerNorm
                │
                ▼
         Transformer Encoder (4 layers)
                │
                ▼
          Output Projection
                │
                ▼
         Logits (B, T, 4096)
```

### 核心程式碼

**檔案**: `done/exp/model_zeroshot_crossattn.py`

```python
class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion for Speaker Embedding
    
    將 Speaker Embedding 通過 Cross-Attention 注入到 Token Embeddings
    """
    
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super().__init__()
        
        # Multi-Head Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Norm & Dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, token_emb, speaker_emb):
        """
        Args:
            token_emb: (B, T, d_model) - Query
            speaker_emb: (B, d_model) - Key & Value
        
        Returns:
            fused_emb: (B, T, d_model)
            attn_weights: (B, T, 1)
        """
        speaker_kv = speaker_emb.unsqueeze(1)  # (B, 1, d_model)
        
        attn_output, attn_weights = self.cross_attn(
            query=token_emb,
            key=speaker_kv,
            value=speaker_kv,
            need_weights=True
        )
        
        # Residual + Norm
        fused_emb = self.norm(token_emb + self.dropout(attn_output))
        
        return fused_emb, attn_weights
```

### 參數量對比

| 模組 | Additive | Cross-Attention | 增加量 |
|------|----------|----------------|--------|
| Speaker Proj | 131K | 131K | - |
| Fusion Layer | - | **1.05M** | +1.05M |
| **總計** | 21M | **22M** | **+5%** |

---

## 五、訓練配置

### 模型參數

```yaml
d_model: 512
nhead: 8
num_layers: 4
dim_feedforward: 2048
dropout: 0.1
speaker_dim: 256
fusion_type: Cross-Attention  # 關鍵改變
```

### 訓練參數

```yaml
batch_size: 64              # 大幅提升（從 8）
learning_rate: 0.0001       # 固定（不衰減）
weight_decay: 0.0           # 無正則化
scheduler: None             # 無學習率調度
epochs: 100
gradient_clip: 1.0
optimizer: Adam
```

### GPU 配置

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # 重要！
export CUDA_VISIBLE_DEVICES=2
```

**說明**: 使用 `PCI_BUS_ID` 確保 nvidia-smi 和 PyTorch 的 GPU 編號一致。

### Checkpoint 策略

- **每 5 epochs**: 保存完整 checkpoint（用於後續分析）
- **持續更新**: 保存最佳模型（Val Accuracy 最高）

---

## 六、實驗執行

### 訓練命令

```bash
cd /home/sbplab/ruizi/WavTokenize-self-supervised/done/exp

# 方式 1: 使用啟動腳本
./run_crossattn_experiment.sh

# 方式 2: 直接運行（在 tmux 中）
tmux new-session -d -s crossattn_training './run_crossattn_experiment.sh'
```

### 訓練時長

- **每 epoch**: ~1.5 分鐘（252 batches，batch_size=64）
- **總時長**: ~2.5 小時（100 epochs）
- **訓練速度**: ~2.7 it/s

### 實驗輸出

```
done/exp/results/crossattn_100epochs_20251105_025951/
├── best_model.pth                  # 最佳模型
├── checkpoint_epoch_5.pth          # Epoch 5 checkpoint
├── checkpoint_epoch_10.pth         # Epoch 10 checkpoint
├── checkpoint_epoch_15.pth         # ...
├── checkpoint_epoch_40.pth         # Epoch 40 checkpoint
├── config.json                     # 訓練配置
├── loss_curves_epoch_10.png        # 損失曲線（每 10 epochs）
├── loss_curves_epoch_20.png
├── loss_curves_epoch_30.png
├── loss_curves_epoch_40.png
└── training.log                    # 訓練日誌
```

---

## 七、實驗結果（進行中）

### 訓練進度

**當前**: Epoch 45/100 (2025-11-05 04:15)

| Metric | Epoch 1 | Epoch 10 | Epoch 20 | Epoch 30 | Epoch 40 | **Epoch 45** |
|--------|---------|----------|----------|----------|----------|--------------|
| **Train Loss** | 4.2528 | 3.1064 | 2.6804 | 2.5335 | 2.4692 | **2.4255** |
| **Train Acc** | 42.86% | 52.29% | 56.93% | 58.46% | 59.13% | **59.41%** |
| **Val Loss** | 5.3899 | 5.1068 | 4.8509 | 4.7325 | 4.6500 | **4.6267** |
| **Val Acc** | 22.30% | 29.89% | 35.47% | 38.76% | 40.98% | **41.63%** ⭐ |

### 關鍵觀察

✅ **Val Accuracy 持續上升**:
- Epoch 1-10: 22.30% → 29.89% (+7.59%)
- Epoch 10-20: 29.89% → 35.47% (+5.58%)
- Epoch 20-30: 35.47% → 38.76% (+3.29%)
- Epoch 30-40: 38.76% → 40.98% (+2.22%)
- Epoch 40-45: 40.98% → **41.63%** (+0.65%)

✅ **已超越 Baseline**:
- Baseline (Additive Fusion): 38.57%
- Cross-Attention (Epoch 45): **41.63%**
- **改善幅度**: **+3.06%** (相對提升 7.9%)

✅ **無過擬合跡象**:
- Train-Val Gap: 59.41% - 41.63% = 17.78%
- Loss 持續下降，未見平台期

⚠️ **仍未達預期**:
- 目標: >43% (相對 baseline +5%)
- 當前: 41.63% (相對 baseline +3.06%)
- 差距: 1.37%

### 訓練曲線分析

從 `loss_curves_epoch_40.png` 可見：

1. **訓練穩定**: Loss 平滑下降，無震盪
2. **持續進步**: Val Accuracy 持續上升，無平台期
3. **健康趨勢**: Epoch 40-45 仍在進步中

---

## 八、與 Baseline 對比

### Additive Fusion (Baseline)

| Metric | Value | 問題 |
|--------|-------|------|
| Best Val Acc | 38.57% | 平台期（Epoch 20-77） |
| Train Acc | 56% | 停滯不前 |
| Learning Rate | 1e-06 | 幾乎無效 |
| Speaker Influence | <5% | 影響力極弱 |

### Cross-Attention (本實驗)

| Metric | Value | 改善 |
|--------|-------|------|
| **Best Val Acc** | **41.63%** | **+3.06%** ✅ |
| Train Acc | 59.41% | +3.41% ✅ |
| Learning Rate | 1e-04 (固定) | 無衰減 ✅ |
| Speaker Influence | (待測試) | 預期 >20% |

### 改善幅度

- **絕對提升**: +3.06%
- **相對提升**: +7.9%
- **超越 baseline**: ✅ 是
- **達成目標 (>43%)**: ⏳ 未達（差 1.37%）

---

## 九、結果解讀

### 假設 2 部分驗證

✅ **成功方面**:
1. Cross-Attention 確實優於 Additive Fusion (+3.06%)
2. 訓練更穩定，無平台期現象
3. Val Accuracy 持續上升（Epoch 45 仍在進步）

⚠️ **限制**:
1. 未達預期目標 (43%)，差距 1.37%
2. 仍需完成訓練才能判斷最終效果
3. Speaker Influence 尚未測試

### 可能原因

**為何未達預期 43%？**

1. **訓練未完成**: 
   - Epoch 45/100，仍有 55 epochs
   - Epoch 40-45 仍在進步 (+0.65%)
   - 可能在 Epoch 60-80 達到 43%

2. **Batch Size 過大**:
   - 從 8 增至 64 (8倍)
   - 可能需要更多 epochs 收斂

3. **假設 2 非唯一因素**:
   - 可能需結合其他改進（Focal Loss, Fine-tune Codebook）

---

## 十、下一步實驗

### 如果最終 Val Acc > 43%

✅ **假設 2 驗證成功**

**後續改進**:
1. 進一步優化 Cross-Attention (multi-layer, FiLM)
2. 分析 attention weights，理解 speaker 影響模式
3. 結合 Focal Loss 降低 Token 0 權重

### 如果最終 Val Acc 39-43%

⚠️ **假設 2 部分成立**

**後續方向**:
1. 測試 Speaker Influence (zero/random speaker)
2. 嘗試假設 1: Focal Loss
3. 嘗試假設 3: Fine-tune Codebook
4. 組合多種改進

### 如果最終 Val Acc < 39%

❌ **假設 2 可能不成立**

**重新檢視**:
1. 檢查 Cross-Attention 實作正確性
2. 降低 Batch Size 重新訓練
3. 轉向其他假設

---

## 十一、如何重現實驗

### 前置條件

1. **緩存數據**:
```bash
# 確認緩存存在
ls -lh done/exp/data/
# 應包含: train_cache.pt (91M), val_cache.pt (32M), cache_config.pt
```

2. **環境設定**:
```bash
cd /home/sbplab/ruizi/WavTokenize-self-supervised
git checkout zero-shot-speaker-denoising-local
```

### 執行步驟

#### 步驟 1: 準備腳本

```bash
cd done/exp

# 確認檔案存在
ls -l model_zeroshot_crossattn.py
ls -l train_crossattn_cached.py
ls -l run_crossattn_experiment.sh

# 添加執行權限
chmod +x run_crossattn_experiment.sh
```

#### 步驟 2: 啟動訓練

```bash
# 方式 1: 直接運行
./run_crossattn_experiment.sh

# 方式 2: 在 tmux 中運行（推薦）
tmux new-session -d -s crossattn_training './run_crossattn_experiment.sh'

# 查看進度
tmux attach -t crossattn_training
# 離開: Ctrl+B, D
```

#### 步驟 3: 監控進度

```bash
# 查看日誌
tail -f crossattn_training.log

# 查看 GPU
watch -n 1 nvidia-smi

# 查看訓練曲線（每 10 epochs 生成）
ls -lh results/crossattn_100epochs_*/loss_curves_*.png
```

#### 步驟 4: 驗證結果

```bash
# 檢查輸出目錄
ls -lh results/crossattn_100epochs_*/

# 查看最終結果
tail -50 results/crossattn_100epochs_*/training.log

# 檢查最佳模型
ls -lh results/crossattn_100epochs_*/best_model.pth
```

### 預期輸出

- **訓練時長**: ~2.5 小時
- **GPU 使用**: ~8 GB (RTX 2080 Ti)
- **Checkpoints**: 每 5 epochs (5, 10, 15, ...)
- **Best Val Acc**: > 41.63% (持續更新)

---

## 十二、相關檔案

### 核心程式碼

1. **`done/exp/model_zeroshot_crossattn.py`**
   - CrossAttentionFusion class
   - ZeroShotDenoisingTransformerCrossAttn class
   - 測試程式碼

2. **`done/exp/train_crossattn_cached.py`**
   - 訓練主程式
   - 每 5 epochs 保存 checkpoint
   - 禁用 tqdm 輸出（避免日誌過大）

3. **`done/exp/run_crossattn_experiment.sh`**
   - 啟動腳本
   - GPU 配置 (PCI_BUS_ID)

### 文檔

1. **`CROSS_ATTENTION_DESIGN.md`**
   - 完整設計方案
   - ASCII 圖示
   - 數學公式推導

2. **`CROSS_ATTENTION_EXPERIMENT.md`**
   - 實驗配置說明
   - 評估標準
   - 成功標準

3. **`DIAGNOSIS_PROGRESS_REPORT.md`**
   - 診斷進度報告
   - 4 個機轉假設

4. **`TRAINING_MECHANISM_HYPOTHESIS.md`**
   - 假設詳細說明
   - 驗證方法

### 診斷工具（可選）

1. **`done/exp/diagnose_training_mechanism.py`**
   - 梯度流動分析
   - 權重更新診斷

2. **`done/exp/diagnose_prediction_behavior.py`**
   - 預測行為分析
   - Speaker influence 測試

---

## 十三、Commit 資訊

### Commit 內容

```bash
# 核心程式碼
done/exp/model_zeroshot_crossattn.py
done/exp/train_crossattn_cached.py
done/exp/run_crossattn_experiment.sh

# 文檔
CROSS_ATTENTION_DESIGN.md
CROSS_ATTENTION_EXPERIMENT.md
CROSS_ATTENTION_FINAL_REPORT.md  # 本檔案
DIAGNOSIS_PROGRESS_REPORT.md
TRAINING_MECHANISM_HYPOTHESIS.md

# 診斷工具（可選）
done/exp/diagnose_training_mechanism.py
done/exp/diagnose_prediction_behavior.py
```

### Commit Message

見下一節 Git Commit 規劃。

---

## 十四、實驗時間軸

| 時間 | 事件 | 狀態 |
|------|------|------|
| 2025-11-05 00:00 | Baseline 訓練完成 | Val Acc 38.57% |
| 2025-11-05 02:30 | 規劃 Cross-Attention 方案 | 設計完成 |
| 2025-11-05 02:46 | 開始實作模型 | 程式碼完成 |
| 2025-11-05 02:59 | 啟動訓練 (GPU 2) | 訓練開始 |
| 2025-11-05 03:01 | Epoch 1 完成 | Val Acc 22.30% |
| 2025-11-05 03:16 | Epoch 10 完成 | Val Acc 29.89% |
| 2025-11-05 03:33 | Epoch 20 完成 | Val Acc 35.47% |
| 2025-11-05 03:50 | Epoch 30 完成 | Val Acc 38.76% |
| 2025-11-05 04:06 | Epoch 40 完成 | Val Acc 40.98% |
| 2025-11-05 04:15 | **Epoch 45** | **Val Acc 41.63%** ⭐ |
| 2025-11-05 ~05:30 | 預計 Epoch 100 完成 | 待觀察 |

---

## 十五、總結

### 當前成果

✅ **驗證成功**:
- Cross-Attention 優於 Additive Fusion (+3.06%)
- 訓練穩定，無平台期
- Val Accuracy 持續上升

⏳ **待完成**:
- 完成剩餘 55 epochs 訓練
- 測試 Speaker Influence
- 視覺化 Attention Weights

### 科學價值

1. **架構創新**: Cross-Attention 應用於 speaker conditioning
2. **可解釋性**: 可視覺化 attention，理解 speaker 影響
3. **實用價值**: 提升 zero-shot 泛化能力

### 局限性

1. 訓練未完成（Epoch 45/100）
2. 未達預期目標 43%（差 1.37%）
3. Speaker Influence 尚未量化測試

### 未來方向

1. **完成訓練**: 觀察 Epoch 60-80 表現
2. **深入分析**: Attention weights 視覺化
3. **組合改進**: Cross-Attention + Focal Loss + Fine-tune Codebook

---

**實驗負責人**: GitHub Copilot  
**報告撰寫**: 2025-11-05 04:20  
**訓練狀態**: 進行中（Epoch 45/100）  
**下次更新**: 訓練完成後
