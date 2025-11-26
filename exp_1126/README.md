# Exp_1126: 1D Wasserstein Loss with Loss Scaling

## 實驗背景

本實驗重現 commit `0502ca619f86f1d336eba0eb23e507b46207eca5` (exp5-1-2) 的實驗設置。

### 問題分析
- **Wasserstein Loss ≈ 0.38** (僅為 CrossEntropy 的 4.36%)
- **Wasserstein gradient ≈ CE gradient 的 0.59%**
- 即使設置 alpha=0.5，CrossEntropy 仍貢獻了 95.8% 的梯度

### 核心創新
實現 **Loss Scaling** 技術，解決 Wasserstein Loss 與 CrossEntropy Loss 量級不匹配問題。

### 解決方案
- **Loss Scaling Factor: 23.0x**
- 使 Wasserstein Loss 與 CE Loss 同量級
- 現在可使用 pure Wasserstein (alpha=1.0) 正常訓練

## 目錄結構

```
exp_1126/
├── README.md                    # 本說明文件
├── train_exp_1126.py           # 主訓練腳本
├── run_exp_1126.sh             # 執行腳本
├── model_zeroshot_lora.py      # Zero-Shot Denoising Transformer 模型
├── wasserstein_loss_1d.py      # 1D Wasserstein Loss 實現
├── config.py                   # 配置常數
├── data_zeroshot.py            # Zero-Shot 數據集
└── shared/
    └── visualization_utils.py  # 可視化工具
```

## 實驗配置

### 訓練超參數
| 參數 | 值 |
|------|-----|
| Batch Size | 28 |
| Epochs | 120 |
| Learning Rate | 1e-4 |
| Weight Decay | 0.01 |
| Dropout | 0.15 |
| Patience | 10 |

### Wasserstein Loss 超參數
| 參數 | 值 | 說明 |
|------|-----|------|
| Alpha | 1.0 | 100% Wasserstein |
| Scale Factor | 23.0 | Loss 縮放因子 |
| Version | 1D | 內存友好版本 |

### 模型超參數
| 參數 | 值 |
|------|-----|
| d_model | 512 |
| nhead | 8 |
| num_layers | 4 |
| dim_feedforward | 2048 |
| fusion_method | cross_attn |
| use_learnable_gate | True |

## 如何執行

### 1. 確保數據緩存存在
```bash
# 訓練緩存
ls -la /home/sbplab/ruizi/c_code/done/exp/data3/train_cache.pt

# 驗證緩存
ls -la /home/sbplab/ruizi/c_code/done/exp/data3/val_cache.pt
```

### 2. 修改 GPU 設置（如需要）
編輯 `run_exp_1126.sh`，修改 `CUDA_VISIBLE_DEVICES` 變數。

### 3. 執行訓練
```bash
cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1126
./run_exp_1126.sh
```

### 4. 監控訓練進度
```bash
# 查看即時 log
tail -f runs/exp_1126_*/training.log
```

## 預期結果

根據原始實驗：
- **Alpha=0.5 (未縮放)**: Val Acc 45.83%, Gap +3.6% (無 overfitting)
- **Alpha=1.0 (Scaled)**: 預期 Val Acc 47-49%, token distribution 更均衡

## Loss Scaling 技術說明

### 為什麼需要 Loss Scaling？

1. **量級差異**: Wasserstein Loss 通常比 CE Loss 小很多
2. **梯度不平衡**: 即使權重相等，CE 的梯度也遠大於 Wasserstein
3. **學習被 CE 主導**: Wasserstein 的語義距離訊號被忽略

### Loss Scaling 優勢

- ✅ Wasserstein Loss 縮放至與 CE Loss 同量級
- ✅ 純 Wasserstein (α=1.0) 現在有足夠大的 gradient
- ✅ 學習速度與 CE 相當，同時保留 token 距離資訊
- ✅ Token distribution 更均衡

## 1D Wasserstein vs 2D Wasserstein

### 1D Wasserstein (本實驗使用)
- **複雜度**: O(n)
- **內存**: O(n)
- **Batch Size**: 可使用 28+
- **假設**: Token 有自然順序

### 2D Wasserstein
- **複雜度**: O(n²)
- **內存**: O(n²)
- **Batch Size**: 受限於 8
- **假設**: 任意 token 距離

## 相關文件

- 原始 commit: `0502ca619f86f1d336eba0eb23e507b46207eca5`
- 分支: `exp5-wavtokenizer-lora`
- 相關文檔: `done/exp5/exp5-1/exp5-1-2/LOSS_SCALING_EXPLANATION.md`

## 作者

重現日期: 2025-11-26
