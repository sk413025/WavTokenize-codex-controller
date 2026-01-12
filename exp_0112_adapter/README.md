# Exp 0112_adapter: Exp J - 中層 Adapter 去噪

## 實驗目標

如果 Exp I（三區差異化 LoRA）仍無法突破 Baseline，說明 **LoRA 本身的容量可能不足以學習去噪任務**。

本實驗嘗試不同的微調方法：**在噪音敏感層插入專門的 Adapter 模組**。

### 為什麼選擇 Adapter？

| 方法 | 優點 | 缺點 |
|------|------|------|
| **LoRA** | 輕量、不改變架構 | 容量受限於原始層結構 |
| **Adapter** | 可自由設計容量和結構 | 需要插入額外模組 |
| **Full Fine-tune** | 最大容量 | 可能破壞原始能力 |

**Adapter 的關鍵優勢**：
1. 可以針對去噪任務設計專門結構
2. 原始權重完全凍結，保留 WavTokenizer 能力
3. 位置可精確控制，直接處理噪音敏感區域

## 核心設計

### Adapter 架構

```
┌─────────────────────────────────────────────────────────────┐
│  Input (from L4 downsample, shape: [B, 128, T])             │
│     ↓                                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  DenoiseAdapter                                      │   │
│  │  ├─ LayerNorm(128)                                   │   │
│  │  ├─ Conv1d(128 → 32, kernel=1)  # down_proj          │   │
│  │  ├─ GELU                                             │   │
│  │  ├─ Dropout(0.1)                                     │   │
│  │  ├─ Conv1d(32 → 128, kernel=1)  # up_proj            │   │
│  │  └─ Residual: output = input + scale * adapter_out   │   │
│  └─────────────────────────────────────────────────────┘   │
│     ↓                                                       │
│  L5-L8 (原始層，權重凍結)                                   │
│     ↓                                                       │
│  L9-L17 → Quantizer → Tokens                                │
└─────────────────────────────────────────────────────────────┘
```

### 設計理由

1. **插入位置 (L4 之後)**：
   - L5-L8 是噪音破壞最嚴重的區域 (cos_sim = 0.21-0.29)
   - 在進入這些層之前先「預處理」特徵

2. **Bottleneck 結構**：
   - 128 → 32 → 128 的降維再升維
   - 限制 Adapter 容量，避免過擬合
   - 強制學習壓縮的去噪表示

3. **可學習 Scale**：
   - 初始值小 (0.01)，訓練初期 Adapter 影響小
   - 隨訓練逐漸增加，平滑過渡

4. **Residual 連接**：
   - `output = input + scale * adapter(input)`
   - 即使 Adapter 學不好，也不會破壞原始信息

### 參數統計

| 組件 | 參數量 | 可訓練 |
|------|--------|--------|
| WavTokenizer 原始 | ~84M | ❌ 凍結 |
| Adapter | ~8K-50K | ✅ 訓練 |
| **Total Trainable** | ~0.01-0.06% | - |

相比 LoRA (~3.7M, 2.25%)，Adapter 更輕量但更專注。

## 與之前實驗對比

| 實驗 | 方法 | 可訓練參數 | 目標 |
|------|------|------------|------|
| Baseline | LoRA 全層 | 3.7M (2.25%) | 基準 |
| Exp I | LoRA 三區差異化 LR | 3.7M (2.25%) | 強化中層學習 |
| **Exp J** | 中層 Adapter | **~8K-50K (0.01-0.06%)** | 專門去噪模組 |

## 預期效果

1. **如果成功**：
   - Val Acc 超過 Baseline (1.06%)
   - Train-Val Gap 減小
   - 證明專門的去噪模組比 LoRA 更有效

2. **如果失敗**：
   - 確認「從 encoder 角度去噪」這個方向本身有問題
   - 應考慮前置去噪網路或其他架構

## 執行方式

```bash
# 確保 Exp I 正在運行或已完成
# 執行 Exp J
./exp_0112_adapter/run_exp_j.sh

# 或手動執行
python exp_0112_adapter/train.py \
    --exp_name exp_j_adapter \
    --adapter_type simple \
    --adapter_init_scale 0.01 \
    --lr 1e-4
```

## 檔案結構

```
exp_0112_adapter/
├── README.md           # 本文件
├── models.py           # DenoiseAdapter, TeacherStudentAdapter
├── train.py            # 訓練腳本
├── run_exp_j.sh        # 執行腳本
└── runs/
    └── exp_j_adapter/
        ├── config.json
        ├── history.json
        ├── training_curves.png
        ├── best_model.pt
        └── audio_samples/
```

## 監控指標

- **Val Acc**: 目標 > Baseline (1.06%)
- **Train-Val Gap**: 目標 < Baseline (2.33%)
- **Adapter Scale**: 監控 Adapter 影響程度的變化

## 變體實驗 (可選)

如果 simple adapter 效果不佳，可嘗試：

1. **MultiHead Adapter**:
   ```bash
   python exp_0112_adapter/train.py \
       --exp_name exp_j_multihead \
       --adapter_type multihead
   ```

2. **更大容量**:
   ```bash
   python exp_0112_adapter/train.py \
       --exp_name exp_j_large \
       --adapter_hidden_dim 64  # 預設是 32
   ```

3. **多位置 Adapter** (需要修改 models.py):
   - 在 L4 和 L8 都插入 Adapter
   - 形成「前置處理 + 後置處理」結構

## 日期

- 創建: 2026-01-12
- 狀態: 待執行 (等 Exp I 結果)
