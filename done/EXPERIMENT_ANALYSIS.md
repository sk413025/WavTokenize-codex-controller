# 實驗結果分析報告
**日期**: 2025-11-01
**分析對象**: Token Denoising Transformer 四組實驗比較

---

## 一、實驗配置總覽

| 實驗名稱 | 策略 | Epochs | 數據集大小 | Mask Ratio | 關鍵參數 |
|---------|------|--------|-----------|------------|---------|
| **Baseline** | CE-only | 200 | 20,736 對 | N/A | max_sentences=288 |
| **A (Dynamic)** | mask_dynamic | 600 | ~1,152 對 | 0.2 | max_sentences=1 |
| **B (Progressive)** | mask_progressive | 600 | ~1,152 對 | 0.1→0.3 | max_sentences=1 |
| **C (Weighted)** | mask_weighted | 600 | ~1,152 對 | 0.1 | weighted_scale=2.0 |

### 數據量差異
- **Baseline**: 每個 speaker 288 句 → 總共 20,736 音頻對
- **Mask 實驗 (A/B/C)**: 每個 speaker 1 句 → 總共 ~1,152 音頻對
- **數據量差距**: ~18倍

---

## 二、實驗結果對比

### 2.1 最佳驗證損失（早期最佳點）

| 實驗 | 最佳 Val Loss | 達到時間 | Val Acc |
|------|--------------|---------|---------|
| **Baseline** | **4.6754** ✅ | Epoch 3 | 38.19% |
| **B (Progressive)** | 6.4497 | 早期 | ~38% (初期) |
| **A (Dynamic)** | 6.5238 | 早期 | ~38% (初期) |
| **C (Weighted)** | 7.0747 | 早期 | ~38% (初期) |

**結論**: Baseline 的最佳驗證損失明顯優於所有 mask 策略（低 27-34%）

### 2.2 最終訓練表現

#### Baseline (Epoch 59/200)
```
Train Loss: 0.4354  |  Train Acc: 90.30%
Val Loss:   13.0980 |  Val Acc:   38.03%
```

#### A - Dynamic (Epoch 600/600)
```
Train Loss: 0.6598  |  Train Acc: 96.69%
Val Loss:   7.4766  |  Val Acc:   12.59% ⚠️
```

#### B - Progressive (Epoch 600/600)
```
Train Loss: 0.8973  |  Train Acc: 92.59%
Val Loss:   7.3722  |  Val Acc:   12.59% ⚠️
```

#### C - Weighted (Epoch 600/600)
```
Train Loss: 0.5934  |  Train Acc: 98.78% ⚠️ 嚴重過擬合
Val Loss:   8.2275  |  Val Acc:   12.76% ⚠️
```

---

## 三、關鍵問題分析

### 🔴 問題 1: 嚴重的泛化能力缺失

**現象**:
- 所有 mask 策略的驗證準確率僅 **12-13%**
- Baseline 的驗證準確率為 **38%**
- Train/Val Accuracy 差距巨大（80-86 個百分點）

**原因分析**:

1. **數據量嚴重不足**
   - Mask 實驗只有 ~1,152 對音頻（Baseline 的 5.6%）
   - 每個 speaker 只有 1 句話，無法學習 speaker 的多樣性
   - 模型只是記憶訓練集，無法泛化

2. **Mask 策略增加學習難度**
   - Mask 預測需要更多樣化的數據才能學習 token 間的依賴關係
   - 數據不足時，mask 反而變成噪音

3. **驗證集與訓練集分布不一致**
   - 訓練集和驗證集使用不同的 speakers
   - 數據量不足時，speaker 的個體差異無法被充分學習

### 🔴 問題 2: 過擬合嚴重

**過擬合指標**:

| 實驗 | Train Acc | Val Acc | 差距 | 過擬合程度 |
|------|-----------|---------|------|-----------|
| Baseline | 90.30% | 38.03% | 52.27% | 中度 |
| A (Dynamic) | 96.69% | 12.59% | 84.10% | 嚴重 ⚠️ |
| B (Progressive) | 92.59% | 12.59% | 80.00% | 嚴重 ⚠️ |
| C (Weighted) | 98.78% | 12.76% | 86.02% | 極度嚴重 ⚠️⚠️ |

**C (Weighted) 過擬合最嚴重**:
- 訓練準確率高達 98.78%（接近完美記憶）
- 驗證準確率只有 12.76%（接近隨機）
- 加權損失 (2.0x) 可能讓模型過度關注被 mask 的 token

### 🔴 問題 3: 驗證損失持續上升

所有實驗都顯示：
- 早期驗證損失達到最佳點
- 之後驗證損失持續上升（overfitting）
- Baseline 在 Epoch 3 達到最佳，之後上升至 13.0980
- Mask 實驗在早期達到最佳，之後持續惡化

---

## 四、三種 Mask 策略比較

### A - Dynamic (動態遮罩)
**配置**: mask_ratio=0.2（固定 20% 遮罩）

**表現**:
- 最終 Val Loss: 7.4766
- Train Acc: 96.69%（高，過擬合）
- Val Acc: 12.59%（差）

**分析**:
- 20% 的 mask ratio 可能過高
- 在數據不足時，過多遮罩導致學習困難
- 模型只學會記憶訓練樣本的模式

### B - Progressive (漸進式遮罩) ✅ 相對最佳
**配置**: mask_ratio 從 0.05 漸進到 0.3（100 epochs）

**表現**:
- 最終 Val Loss: 7.3722（三者中最低）✅
- Train Acc: 92.59%（相對較低，較少過擬合）
- Val Acc: 12.59%（仍然很差）

**分析**:
- 漸進式策略理論上應該有幫助（curriculum learning）
- 在數據充足的情況下可能會表現更好
- 當前數據量下，仍然無法解決泛化問題

### C - Weighted (加權損失) ⚠️ 最差
**配置**: weighted_loss_scale=2.0（被 mask token 的損失加倍）

**表現**:
- 最終 Val Loss: 8.2275（最高）⚠️
- Train Acc: 98.78%（最高，嚴重過擬合）⚠️
- Val Acc: 12.76%（最差）

**分析**:
- 加權策略讓模型過度關注 masked tokens
- 在數據不足時，加劇了過擬合問題
- **不建議在小數據集上使用高權重**

---

## 五、改進方案（按優先級排序）

### 🔥 優先級 1: 增加訓練數據量

**當前問題**: 數據量是所有問題的根源

**解決方案**:
```python
# 當前配置（失敗）
max_sentences_per_speaker = 1  # ❌ 太少

# 建議配置
max_sentences_per_speaker = 100  # ✅ 最低建議值
# 或
max_sentences_per_speaker = 288  # ✅ 與 baseline 一致
```

**預期效果**:
- 數據量增加 100-288 倍
- 模型能學習 speaker 和 content 的多樣性
- 減少過擬合，提升泛化能力

**實施優先級**: ⭐⭐⭐⭐⭐ （最高）

---

### 🔥 優先級 2: 降低 Mask Ratio

**當前問題**: Mask ratio 過高（0.2）導致學習困難

**解決方案**:
```python
# 當前配置
mask_ratio = 0.2  # Dynamic: ❌ 太高
mask_ratio = 0.1  # Progressive/Weighted: ⚠️ 仍偏高

# 建議配置
mask_ratio = 0.05  # ✅ 先從低比例開始
# Progressive 策略
progressive_start_ratio = 0.02  # ✅ 從 2% 開始
progressive_end_ratio = 0.15    # ✅ 逐步到 15%
progressive_epochs = 200        # ✅ 更長的漸進期
```

**預期效果**:
- 降低學習難度
- 減少對數據量的需求
- 模型能先學好基礎預測，再學習復雜模式

**實施優先級**: ⭐⭐⭐⭐

---

### 🔥 優先級 3: 加入正則化技術

**當前問題**: 無任何正則化（dropout=0.0）

**解決方案 A: Dropout**
```python
# 當前配置
dropout = 0.0  # ❌ 無正則化

# 建議配置
dropout = 0.1  # ✅ 輕度正則化
# 或
dropout = 0.2  # ✅ 中度正則化（數據很少時）
```

**解決方案 B: Label Smoothing**
```python
# 在 CrossEntropyLoss 中加入 label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**解決方案 C: Weight Decay 調整**
```python
# 當前配置
weight_decay = 0.01  # 已有，但可能需要調整

# 建議配置
weight_decay = 0.05  # ✅ 增加 L2 正則化
```

**預期效果**:
- 防止模型過度記憶訓練數據
- 提升泛化能力
- 減緩驗證損失上升趨勢

**實施優先級**: ⭐⭐⭐⭐

---

### 🔥 優先級 4: 實施早停策略

**當前問題**:
- 訓練持續到最後，驗證損失持續上升
- Baseline 在 Epoch 3 達到最佳，卻訓練到 Epoch 59+

**解決方案**:
```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # 停止訓練
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

# 使用方式
early_stopping = EarlyStopping(patience=20)
if early_stopping(val_loss):
    print("Early stopping triggered")
    break
```

**建議配置**:
```python
patience = 20  # 20 epochs 無改善則停止
min_delta = 0.001  # 最小改善閾值
```

**預期效果**:
- 在最佳點停止，避免過擬合
- 節省訓練時間和計算資源
- 獲得更好的最終模型

**實施優先級**: ⭐⭐⭐⭐

---

### 🔥 優先級 5: 調整學習率策略

**當前問題**: 學習率在固定 epoch 減半，未考慮驗證損失

**當前策略**:
```python
# Epoch 24: lr = 3e-4 → 1.5e-4
# Epoch 45: lr = 1.5e-4 → 7.5e-5
```

**解決方案 A: ReduceLROnPlateau**
```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10,
    verbose=True,
    min_lr=1e-6
)

# 在驗證後
scheduler.step(val_loss)
```

**解決方案 B: Cosine Annealing with Warm Restarts**
```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=50,      # 第一次重啟週期
    T_mult=2,    # 週期倍增因子
    eta_min=1e-6 # 最小學習率
)
```

**預期效果**:
- 根據驗證損失動態調整學習率
- 避免在已經過擬合時還用高學習率
- 可能發現更好的局部最優點

**實施優先級**: ⭐⭐⭐

---

### 🔥 優先級 6: 降低 Weighted Loss Scale

**針對 C (Weighted) 實驗**

**當前問題**: weighted_loss_scale=2.0 導致最嚴重的過擬合

**解決方案**:
```python
# 當前配置
weighted_loss_scale = 2.0  # ❌ 過高

# 建議配置
weighted_loss_scale = 1.2  # ✅ 輕微加權
# 或
weighted_loss_scale = 1.5  # ✅ 中度加權
# 或逐漸增加
start_scale = 1.0
end_scale = 1.5
current_scale = start_scale + (end_scale - start_scale) * (epoch / total_epochs)
```

**預期效果**:
- 減少對 masked tokens 的過度關注
- 降低過擬合風險
- 平衡 masked 和 non-masked tokens 的學習

**實施優先級**: ⭐⭐⭐

---

### 🔥 優先級 7: 數據增強

**當前問題**: 無數據增強，數據多樣性不足

**解決方案**:
```python
# 在音頻層面的增強（在 tokenization 前）
def augment_audio(waveform, sr=24000):
    # 1. 時間拉伸（±10%）
    if random.random() > 0.5:
        rate = random.uniform(0.9, 1.1)
        waveform = librosa.effects.time_stretch(waveform, rate=rate)

    # 2. 音高偏移（±2 半音）
    if random.random() > 0.5:
        n_steps = random.uniform(-2, 2)
        waveform = librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)

    # 3. 添加輕微噪音
    if random.random() > 0.5:
        noise = np.random.randn(len(waveform)) * 0.005
        waveform = waveform + noise

    return waveform

# 在 token 層面的增強（在 tokenization 後）
def augment_tokens(tokens, mask_id=4095):
    # 1. Token dropout（隨機將少量 token 替換為 mask）
    if random.random() > 0.5:
        num_dropout = int(len(tokens) * 0.05)
        dropout_indices = random.sample(range(len(tokens)), num_dropout)
        for idx in dropout_indices:
            tokens[idx] = mask_id

    return tokens
```

**注意**: WavTokenizer 是預訓練的，音頻增強可能影響 token 品質，需謹慎使用。

**預期效果**:
- 增加數據多樣性
- 提升模型魯棒性
- 部分緩解數據不足問題

**實施優先級**: ⭐⭐

---

### 🔥 優先級 8: 調整模型容量

**當前配置**:
```python
d_model = 512
nhead = 8
num_layers = 4
dim_feedforward = 2048
總參數: 19,367,936
可訓練: 14,710,784
```

**問題**: 對於小數據集，模型可能過大

**解決方案 A: 減少模型容量**
```python
# 選項 1: 減少層數
num_layers = 2  # ✅ 減半

# 選項 2: 減少 d_model
d_model = 256   # ✅ 減半

# 選項 3: 減少 feedforward
dim_feedforward = 1024  # ✅ 減半

# 預期參數量: ~3-5M（適合小數據集）
```

**解決方案 B: 增加模型容量**（在數據量充足後）
```python
# 當 max_sentences_per_speaker >= 100 時
num_layers = 6
d_model = 512
dim_feedforward = 2048
```

**預期效果**:
- 小模型在小數據集上泛化更好
- 減少過擬合風險
- 但需要與數據量匹配

**實施優先級**: ⭐⭐

---

## 六、推薦的實驗流程

### 階段 1: 數據量測試（最優先）

**目標**: 確定合適的數據量

```bash
# 實驗 1.1: 中等數據量
max_sentences_per_speaker = 50
num_epochs = 200
mask_strategy = "progressive"
mask_ratio = 0.05
dropout = 0.1

# 實驗 1.2: 大數據量
max_sentences_per_speaker = 100
num_epochs = 200
mask_strategy = "progressive"
mask_ratio = 0.05
dropout = 0.1

# 實驗 1.3: 完整數據量（與 baseline 一致）
max_sentences_per_speaker = 288
num_epochs = 200
mask_strategy = "progressive"
mask_ratio = 0.05
dropout = 0.1
```

**成功指標**: Val Acc > 30%, Train-Val Acc gap < 20%

---

### 階段 2: Mask 策略優化

**前提**: 完成階段 1，數據量充足

```bash
# 實驗 2.1: Progressive（低 mask ratio）
mask_strategy = "progressive"
progressive_start_ratio = 0.02
progressive_end_ratio = 0.15
progressive_epochs = 200

# 實驗 2.2: Dynamic（低 mask ratio）
mask_strategy = "dynamic"
mask_ratio = 0.05

# 實驗 2.3: Weighted（低權重）
mask_strategy = "weighted"
mask_ratio = 0.05
weighted_loss_scale = 1.2
```

---

### 階段 3: 正則化調優

```bash
# 實驗 3.1: Dropout 測試
dropout = [0.0, 0.1, 0.2, 0.3]

# 實驗 3.2: Weight Decay 測試
weight_decay = [0.01, 0.05, 0.1]

# 實驗 3.3: Label Smoothing
label_smoothing = [0.0, 0.05, 0.1]
```

---

### 階段 4: 模型架構優化

```bash
# 實驗 4.1: 層數測試
num_layers = [2, 4, 6]

# 實驗 4.2: 維度測試
d_model = [256, 512, 768]
```

---

## 七、立即可執行的最佳配置建議

基於當前分析，建議立即執行以下配置：

### 配置 A: Conservative（保守穩健）
```json
{
  "max_sentences_per_speaker": 100,
  "num_epochs": 200,
  "batch_size": 14,
  "learning_rate": 0.0003,
  "weight_decay": 0.05,
  "dropout": 0.2,

  "mask_strategy": "progressive",
  "mask_ratio": 0.05,
  "progressive_start_ratio": 0.02,
  "progressive_end_ratio": 0.10,
  "progressive_epochs": 150,

  "early_stopping_patience": 20,
  "use_label_smoothing": true,
  "label_smoothing": 0.1
}
```

**預期**: Val Acc 提升到 25-30%，過擬合減輕

---

### 配置 B: Aggressive（激進改進）
```json
{
  "max_sentences_per_speaker": 288,
  "num_epochs": 200,
  "batch_size": 14,
  "learning_rate": 0.0003,
  "weight_decay": 0.05,
  "dropout": 0.15,

  "mask_strategy": "progressive",
  "mask_ratio": 0.08,
  "progressive_start_ratio": 0.03,
  "progressive_end_ratio": 0.15,
  "progressive_epochs": 150,

  "early_stopping_patience": 25,
  "use_label_smoothing": true,
  "label_smoothing": 0.05,

  "lr_scheduler": "reduce_on_plateau",
  "scheduler_patience": 10,
  "scheduler_factor": 0.5
}
```

**預期**: Val Acc 提升到 35-38%，接近 baseline

---

## 八、結論

### 當前實驗的核心問題

1. **數據量不足是根本原因**
   - Mask 實驗數據量僅為 Baseline 的 5.6%
   - 導致嚴重過擬合和泛化能力缺失

2. **Mask 策略在小數據集上無效**
   - 所有 mask 策略的 Val Acc 都只有 12-13%
   - Progressive 略優，但仍遠不如 Baseline

3. **缺乏正則化機制**
   - Dropout = 0.0
   - 無 early stopping
   - 導致訓練持續到嚴重過擬合

### 改進優先級總結

| 優先級 | 改進項目 | 預期效果 | 實施難度 |
|-------|---------|---------|---------|
| ⭐⭐⭐⭐⭐ | 增加數據量 (100-288 句/speaker) | 巨大 | 容易 |
| ⭐⭐⭐⭐ | 降低 Mask Ratio (0.02-0.10) | 顯著 | 容易 |
| ⭐⭐⭐⭐ | 加入 Dropout (0.1-0.2) | 顯著 | 容易 |
| ⭐⭐⭐⭐ | 實施 Early Stopping | 顯著 | 中等 |
| ⭐⭐⭐ | 調整學習率策略 | 中等 | 中等 |
| ⭐⭐⭐ | 降低 Weighted Scale (1.2-1.5) | 中等 | 容易 |
| ⭐⭐ | 數據增強 | 小-中等 | 困難 |
| ⭐⭐ | 調整模型容量 | 小-中等 | 中等 |

### 最終建議

**立即執行**: 配置 A（Conservative）
- 增加數據量到 100 句/speaker
- 降低 mask ratio 到 0.05
- 加入 dropout=0.2 和 early stopping
- 預期 Val Acc 可提升至 25-30%

**下一步**: 配置 B（Aggressive）
- 如果配置 A 成功，使用完整數據量（288 句）
- 微調 mask ratio 和正則化參數
- 目標：Val Acc 接近或超越 Baseline（38%）

**長期目標**:
- 在數據充足的基礎上，系統性測試不同 mask 策略
- 探索 mask 策略對音質提升的實際效果
- 建立完整的實驗對比基準

---

## 九、附錄：快速對比表

### 性能排名（綜合考慮）

| 排名 | 實驗 | 最佳 Val Loss | 最終 Val Acc | 過擬合程度 | 推薦度 |
|-----|------|--------------|-------------|-----------|-------|
| 🥇 1 | **Baseline** | 4.6754 | 38.03% | 中度 | ⭐⭐⭐⭐⭐ |
| 🥈 2 | Progressive (B) | 6.4497 | 12.59% | 嚴重 | ⭐ |
| 🥉 3 | Dynamic (A) | 6.5238 | 12.59% | 嚴重 | ⭐ |
| 4 | Weighted (C) | 7.0747 | 12.76% | 極嚴重 | ❌ |

**明確結論**: 在當前條件下，**Baseline (CE-only) 是唯一可用的配置**。所有 mask 策略都因數據不足而失敗。

---

**報告完成日期**: 2025-11-01
**建議執行**: 立即按「配置 A」重新訓練，驗證數據量假設
