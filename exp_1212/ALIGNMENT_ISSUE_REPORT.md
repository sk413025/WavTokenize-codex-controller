# exp_1212: 資料對齊問題驗證報告

**日期**: 2025-12-12
**目的**: 驗證 Token Accuracy 上不去的根本原因

---

## 問題驗證結果

### 來源1: Per-Pair Mismatch (同一對 noisy/clean 長度不同)

| 指標 | TRAIN | VAL |
|------|-------|-----|
| 有不一致的 pair | 100% (±1 sample) | 40.35% |
| 顯著不一致 (>10ms) | **0%** | **40.08%** |
| 嚴重不一致 (>100ms) | 0% | **37.11%** |
| 方向性 | Noisy 長 1 sample | Clean 更長 |
| 平均差異 | 0.04 ms | **266.4 ms** |
| P95 差異 | 0.04 ms | **1164 ms** |
| Max 差異 | 0.04 ms | **2414 ms** |

**結論**:
- **TRAIN 幾乎沒問題** - 每對只差 1 sample (可忽略)
- **VAL 有嚴重問題** - 40% 的 pair 中 clean 比 noisy 長很多 (平均 266ms)

---

### 來源2: Cross-Sample Mismatch (Batch 內長度差異)

| 指標 | TRAIN | VAL |
|------|-------|-----|
| 樣本長度範圍 | 1.96s ~ 4.20s | 2.19s ~ 5.85s |
| 長度標準差 | 0.41s | 0.65s |
| Batch 內 range (avg) | 1.29s | 1.99s |
| **Frame-level Padding 比例** | **19.7%** | **25.5%** |

**結論**:
- **約 20-25% 的 frames 是 padding**
- 這些 padding frames 仍被計入 loss/accuracy，會：
  - 稀釋梯度信號
  - 讓 accuracy 看起來偏低
  - 增加訓練噪音

---

### Token 長度對齊

| 指標 | TRAIN | VAL |
|------|-------|-----|
| Token 長度不一致 | 0% | 0% |

**結論**: Cache 中的 tokens 已對齊，但訓練時讀 wav 重新計算會引入新的 mismatch

---

## 問題影響分析

```
                        Token Acc 慢 / Plateau
                                |
        +-----------------------+------------------------+
        |                                                |
  【來源1】Per-Pair Mismatch                    【來源2】Cross-Sample Padding
        |                                                |
  VAL: 40% pairs 有長度差                        20-25% frames 是 padding
  Clean 系統性更長                                       |
        |                                                |
        v                                                v
  VAL acc 被嚴重低估                            Loss/Acc 被稀釋
  (尾段錯位)                                    梯度信號減弱
```

### 為什麼 Token Accuracy 上不去？

1. **TRAIN 主要問題是來源2** (padding 19.7%)
   - Per-pair 差異可忽略 (1 sample)
   - 但 batch padding 讓約 1/5 的計算無意義

2. **VAL 兩個問題都嚴重**
   - 來源1: 40% pairs 有大幅錯位
   - 來源2: 25.5% padding
   - Val acc 嚴重失真，無法反映真實性能

3. **訓練上限估算**
   - 根據 DATASET_ALIGNMENT_REPORT.md，teacher noisy vs clean acc ~35-38%
   - 目前 train acc ~30%，已接近上限
   - 但如果修正 padding，acc 數字會更準確

---

## 解決方案

### 方案 A: Per-Sample Min-Length 截斷 (推薦)

**原理**: 在 Dataset 層將每對 noisy/clean 截到相同長度

```python
def __getitem__(self, idx):
    noisy = load_audio(self.data[idx]['noisy_path'])
    clean = load_audio(self.data[idx]['clean_path'])

    # 截到最短長度
    min_len = min(len(noisy), len(clean))
    noisy = noisy[:min_len]
    clean = clean[:min_len]

    return {'noisy_audio': noisy, 'clean_audio': clean, 'length': min_len}
```

**優點**:
- 完全消除來源1
- 實現簡單
- 不需修改 loss

**缺點**:
- 會丟棄部分資料 (VAL 損失較多)

---

### 方案 B: Masked Loss (推薦配合 A)

**原理**: 只在有效 frames 上計算 loss/acc，忽略 padding

```python
def masked_loss(student_out, teacher_out, lengths, encoder_stride=320):
    B, D, T = student_out.shape

    # 計算每個樣本有效 frame 數
    frame_lengths = lengths // encoder_stride

    # 建立 mask: (B, T)
    mask = torch.arange(T, device=student_out.device) < frame_lengths.unsqueeze(1)
    mask = mask.unsqueeze(1).float()  # (B, 1, T)

    # Masked MSE
    diff_sq = (student_out - teacher_out) ** 2
    loss = (diff_sq * mask).sum() / mask.sum()

    return loss
```

**優點**:
- 消除來源2 (padding 影響)
- loss 更純淨
- acc 更準確

**缺點**:
- 需修改 loss 和 accuracy 計算

---

### 方案 C: Bucket Sampling (進階)

**原理**: 將樣本按長度分組，同 batch 內樣本長度相近

```python
class BucketSampler:
    def __init__(self, lengths, batch_size, num_buckets=10):
        # 將樣本分到不同長度桶
        self.buckets = self._create_buckets(lengths, num_buckets)

    def __iter__(self):
        # 從同一桶內取樣
        for bucket in self.buckets:
            yield from self._sample_from_bucket(bucket)
```

**優點**:
- 最大程度減少 padding
- 訓練效率更高

**缺點**:
- 實現複雜
- 可能影響隨機性

---

## 建議實施順序

```
優先級 1 (立即實施):
├── 方案 A: Per-Sample Min-Length 截斷
│   └── 修改 Dataset.__getitem__()
│
└── 方案 B: Masked Loss
    └── 修改 loss function 和 accuracy 計算

優先級 2 (驗證後再決定):
└── 方案 C: Bucket Sampling
    └── 如果 padding 仍是瓶頸
```

---

## 預期效果

| 指標 | 修復前 | 修復後 (預期) |
|------|--------|--------------|
| VAL per-pair mismatch | 40% | 0% |
| Padding 比例 | 20-25% | 仍有，但不影響 loss |
| Train acc (顯示) | ~30% | ~35%+ (更準確) |
| Val acc (顯示) | ~18% | ~35%+ (大幅提升) |
| 訓練信號品質 | 被稀釋 | 更純淨 |

**注意**: 修復後 accuracy 數字會變化，這是因為：
1. 移除了無效 frames 的污染
2. 顯示的是更準確的真實性能
3. 不代表模型變好了，而是度量更準確了

---

## 實現狀態

### ✅ 方案 A: Per-Sample Min-Length 截斷

**檔案**: `exp_1212/data_aligned.py`

實現內容:
- `AlignedNoisyCleanPairDataset`: 每對 noisy/clean 截到相同長度
- `aligned_collate_fn`: 返回 `lengths` 供 masked loss 使用
- `create_aligned_dataloaders`: 創建修復版 dataloader

```python
# 關鍵修復
min_len = min(len(noisy_audio), len(clean_audio))
noisy_audio = noisy_audio[:min_len]
clean_audio = clean_audio[:min_len]
```

### ✅ 方案 B: Masked Loss

**檔案**: `exp_1212/losses_masked.py`

實現內容:
- `create_length_mask`: 創建 frame-level mask
- `MaskedFeatureLoss`: Masked MSE loss
- `MaskedTripletLoss`: Masked Triplet loss
- `MaskedCrossEntropyLoss`: Masked CE loss
- `compute_masked_accuracy`: Masked 準確率計算
- `MaskedCombinedLoss`: 組合版 masked loss

```python
# 關鍵原理
mask = torch.arange(T) < frame_lengths.unsqueeze(1)  # (B, T)
loss = (diff_sq * mask).sum() / mask.sum()  # 只計算有效 frames
```

### ✅ 訓練腳本

**檔案**: `exp_1212/train_aligned.py`

特點:
- 同時報告 `masked_acc` 和 `unmasked_acc` 便於比較
- 記錄 padding 比例統計
- 繪製 masked vs unmasked accuracy gap 曲線

### ✅ 實驗腳本

**檔案**: `exp_1212/run_exp34_aligned.sh`

配置:
- 與 Exp31 相同的超參數作為 baseline
- feature_weight=1.0, triplet_weight=0.5
- lora_rank=128, dropout=0.2, weight_decay=0.05

---

## 下一步

1. ~~在 exp_1212 實現方案 A + B~~ ✅ 已完成
2. 運行 `bash run_exp34_aligned.sh` 開始實驗
3. 比較修復前後的 loss/acc 變化
4. 分析 masked_acc vs unmasked_acc 差異

---

## 檔案清單

```
exp_1212/
├── ALIGNMENT_ISSUE_REPORT.md     # 本報告
├── verify_alignment_issues.py    # 驗證腳本
├── alignment_verification_results.json  # 驗證結果
├── data_aligned.py               # 方案 A: 修復版 Dataset
├── losses_masked.py              # 方案 B: Masked Loss
├── train_aligned.py              # 修復版訓練腳本
└── run_exp34_aligned.sh          # Exp34 啟動腳本
```
