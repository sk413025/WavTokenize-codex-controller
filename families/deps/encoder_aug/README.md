# exp_0216: 資料增強 + LoRA Rank 降低實驗

**日期**: 2026-02-16
**目標**: 突破 val MSE 0.037 天花板，緩解 overfitting

---

## 問題背景

Plan Ori 和 V2 兩個實驗都在 epoch ~70 開始 overfitting，best val MSE 約 0.037：

| 實驗 | Best Val MSE | Overfit Epoch | Final MSE |
|------|-------------|---------------|-----------|
| Plan Ori (LoRA-256) | 0.0374 | ~71 | 0.0418 |
| V2 (4-layer RVQ) | 0.0371 | ~78 | 0.0398 |

**根本原因**:
1. **無資料增強**: 10,368 個訓練樣本，完全沒有 augmentation
2. **LoRA rank 過大**: rank=256 在 18 層 transformer 上產生過多可訓練參數

---

## 改進策略

### 1. 資料增強 (Data Augmentation)

| 增強方式 | 機率 | 參數 | 效果 |
|---------|------|------|------|
| **SNR Remix** | 50% | SNR ∈ [-5, 25] dB | 從現有 pair 分離 noise，隨機 SNR 重混 → 無限變體 |
| **Random Gain** | 30% | ±3 dB | 讓模型不依賴特定音量 |
| **Random Crop** | 30% | 最短 70% | 增加位置多樣性 |
| **Time Stretch** | 20% | 0.95x ~ 1.05x | 增加時間尺度多樣性 |

### 2. 降低 LoRA Rank

| 參數 | Plan Ori | exp_0216 | 說明 |
|------|----------|----------|------|
| `lora_rank` | 256 | **64** | 減少 75% 可訓練參數 |
| `lora_alpha` | 512 | **128** | 保持 alpha/rank = 2 |
| `weight_decay` | 0.01 | **0.02** | 加強 L2 正則化 |

---

## 預期結果

| 指標 | Plan Ori | exp_0216 預期 |
|------|----------|--------------|
| Best Val MSE | 0.0374 | **< 0.035** |
| Overfit Epoch | ~71 | **> 120** |
| Final MSE (ep300) | 0.0418 | **< 0.038** |

---

## 檔案結構

```
families/deps/encoder_aug/
├── README.md                    # 本文件
├── data_augmented.py            # AugmentedCurriculumDataset
├── train_augmented.py           # 訓練腳本
└── runs/                        # 實驗輸出
    ├── augmented_step_*/        # Short-run 結果
    └── augmented_epoch_*/       # Long-run 結果
```

---

## 執行方式

```bash
# Smoke test (10 steps)
conda activate test
python families/deps/encoder_aug/train_augmented.py --mode step --steps 10 \
    --batch_size 4 --grad_accum 1 --eval_interval 10

# Short-run (1000 steps)
python families/deps/encoder_aug/train_augmented.py --mode step --steps 1000

# Long-run (300 epochs)
python families/deps/encoder_aug/train_augmented.py --mode epoch --epochs 300 \
    --batch_size 8 --grad_accum 2 --save_audio_interval 50
```

---

## 與 Plan Ori 的差異對比

```diff
- lora_rank=256, lora_alpha=512
+ lora_rank=64,  lora_alpha=128

- weight_decay=0.01
+ weight_decay=0.02

- CurriculumDataset (無增強)
+ AugmentedCurriculumDataset:
+   SNR Remix:    p=0.5, range=[-5, 25] dB
+   Random Gain:  p=0.3, ±3 dB
+   Random Crop:  p=0.3, min_ratio=0.7
+   Time Stretch: p=0.2, [0.95, 1.05]
```
