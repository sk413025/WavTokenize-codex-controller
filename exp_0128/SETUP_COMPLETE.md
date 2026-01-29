# Exp 0128 Setup Complete

✅ **實驗 2：Noise-Balanced Sampling 已準備完成**

---

## 檔案結構

```
exp_0128/
└── noise_balanced_sampling/
    ├── README.md                    # 實驗說明文件
    ├── sampler.py                   # NoiseBalancedSampler 實作
    ├── data_balanced.py             # Noise-Balanced DataLoader
    ├── train_short_run.py           # Short-run 訓練腳本 (800-1000 steps)
    ├── run_exp2.sh                  # 執行腳本 ✅
    ├── test_integration.py          # 整合測試
    └── (輸出目錄將在此創建)
```

---

## 驗證結果

### 1. Noise Type Distribution (Training Set)

| Noise Type | Count | Percentage |
|------------|-------|------------|
| **box**     | 4,032 | 38.9% |
| **papercup** | 3,744 | 36.1% |
| **plastic**  | 2,592 | 25.0% |

**Total**: 10,368 samples

### 2. Sampler Configuration

- Batch size: 12
- Samples per type per batch: 4
- Total batches per epoch: 648
- **每個 batch 確保 box/papercup/plastic 各 1/3**

### 3. Integration Test

✅ NoiseBalancedSampler 正常運作
✅ DataLoader 可正常載入資料
✅ 環境變數已設定

---

## 執行實驗

### 方式 1：使用腳本（推薦）

```bash
bash exp_0128/noise_balanced_sampling/run_exp2.sh
```

### 方式 2：直接執行

```bash
# 設定環境
export PYTHONPATH="/home/sbplab/ruizi/WavTokenize-feature-analysis:$PYTHONPATH"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

# 執行訓練
CUDA_VISIBLE_DEVICES=1 python exp_0128/noise_balanced_sampling/train_short_run.py \
    --steps 1000 \
    --batch_size 2 \
    --grad_accum 2 \
    --output_dir exp_0128/noise_balanced_sampling/run_exp2_test \
    --seed 42
```

---

## 預期輸出

執行完成後將產生：

```
exp_0128/noise_balanced_sampling/run_exp2_YYYYMMDD_HHMMSS/
├── config.json              # 實驗配置
├── metrics_history.json     # 訓練過程 metrics (每 200 steps)
└── summary.json             # 最終結果 + baseline 對比
```

以及：

```
exp_0128/noise_balanced_sampling/run_exp2_YYYYMMDD_HHMMSS.log  # 完整訓練 log
```

---

## 成功判準

查看 `summary.json` 中的 `"success"` 欄位：

```json
{
  "success": true/false
}
```

**Success = true** 需同時滿足：
1. ✅ Val entropy **↑** (> 6.07，baseline)
2. ✅ Val top-10 mass **↓** (< 19.7%，baseline)
3. ✅ Val strict acc **不惡化** (≥ 0.82%，允許 10% 下降)

---

## Baseline 數據（Exp K v6）

| 指標 | 數值 |
|------|------|
| Val Strict Acc | 0.91% |
| Val Entropy | 6.07 |
| Val Top-10 Mass | 19.7% |
| Val Unique Tokens | 1,665 |

來源：[exp_0125/tracin_token_collapse_589e6d/metrics_overview.json](../exp_0125/tracin_token_collapse_589e6d/metrics_overview.json)

---

## 預期資源與時間

- **GPU**: 1 張 (CUDA_VISIBLE_DEVICES=1)
- **VRAM**: ~12-16 GB
- **Time**: 2-3 hours (1000 steps @ batch_size=2, grad_accum=2)
- **Disk**: < 1 GB (checkpoints 未保存)

---

## 下一步（如果成功）

1. ✅ 執行實驗 1（Soft Reweighting）
2. ✅ 組合實驗 1 + 2
3. ✅ Full training (300 epochs)

## 下一步（如果失敗）

1. ⚠️ 檢查 noise type distribution 是否真的平衡
2. ⚠️ 調整 batch_size 或 effective_batch_size
3. ⚠️ 嘗試其他方法（VQ margin regularization 等）

---

## 參考文件

- **TracIn 診斷**: [exp_0125/tracin_token_collapse_589e6d/CONCLUSION.md](../exp_0125/tracin_token_collapse_589e6d/CONCLUSION.md)
- **Baseline 訓練**: [exp_0112_intermediate/train_v6.py](../exp_0112_intermediate/train_v6.py)
- **實驗說明**: [noise_balanced_sampling/README.md](noise_balanced_sampling/README.md)

---

✅ **所有準備工作已完成，可以開始執行實驗！**

執行指令：
```bash
bash exp_0128/noise_balanced_sampling/run_exp2.sh
```
