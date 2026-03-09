# Exp 0128: Component Check Report

✅ **所有組件已驗證完成，可以執行實驗**

執行時間：2026-01-28

---

## ✅ 驗證結果

### 1. CUDA & GPU Setup
- ✅ CUDA available: **True**
- ✅ GPU: **NVIDIA GeForce RTX 2080 Ti**
- ✅ Device: **cuda:0**
- ✅ Environment variable: `CUDA_VISIBLE_DEVICES=0`

### 2. Noise Type Extraction
- ✅ `nor_boy10_box_LDV_132.wav` → **box**
- ✅ `nor_girl3_papercup_LDV_115.wav` → **papercup**
- ✅ `nor_boy4_plastic_LDV_281.wav` → **plastic**

### 3. Module Imports
- ✅ Config imports (`WAVTOK_CONFIG`, `WAVTOK_CKPT`, `TRAIN_CACHE`, `VAL_CACHE`)
- ✅ Model imports (`TeacherStudentIntermediate`)
- ✅ Loss imports (`IntermediateSupervisionLossV6`)
- ✅ Sampler imports (`NoiseBalancedSampler`)
- ✅ DataLoader imports (`create_noise_balanced_dataloaders`)

### 4. Data Cache
- ✅ Train cache: `/home/sbplab/ruizi/c_code/done/exp/data3/train_cache.pt`
  - Samples: **10,368** (after filtering clean→clean)
  - Noise distribution:
    - box: 4,032 (38.9%)
    - papercup: 3,744 (36.1%)
    - plastic: 2,592 (25.0%)
- ✅ Val cache: `/home/sbplab/ruizi/c_code/done/exp/data3/val_cache.pt`
  - Samples: **1,728** (after filtering clean→clean)

### 5. NoiseBalancedSampler
- ✅ Batch size: 12
- ✅ Samples per type: 4 (各 1/3)
- ✅ Total batches per epoch: 648
- ✅ Sampler correctly groups samples by noise type

### 6. Training Scripts
- ✅ `train_short_run.py` exists and imports successfully
- ✅ `run_exp2.sh` exists and configured correctly
  - GPU: **0** (CUDA_VISIBLE_DEVICES=0)
  - Steps: **1000**
  - Batch size: **2**
  - Grad accum: **2**
  - Effective batch size: **4**
  - Learning rate: **1e-4**
  - Seed: **42**

---

## 📋 Configuration Summary

### Environment
```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/sbplab/ruizi/WavTokenize-feature-analysis:$PYTHONPATH"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test
```

### Training Parameters
| Parameter | Value |
|-----------|-------|
| GPU | cuda:0 (RTX 2080 Ti) |
| Steps | 1000 |
| Batch size | 2 |
| Gradient accumulation | 2 |
| Effective batch size | 4 |
| Learning rate | 1e-4 |
| Seed | 42 |
| Eval interval | 200 steps |

### Data Configuration
| Split | Samples | Batches (bs=2) |
|-------|---------|----------------|
| Train | 10,368 | 648 (with balanced sampling) |
| Val | 1,728 | 864 |

### Noise Balance (Per Batch)
- **box**: 1/3 (目標)
- **papercup**: 1/3 (目標)
- **plastic**: 1/3 (目標)

---

## 🎯 Baseline Metrics (Exp K v6)

從 TracIn 分析 (commit d0f9ecb) 獲得：

| 指標 | 數值 |
|------|------|
| Val Strict Acc | 0.91% |
| Val Entropy | 6.07 |
| Val Top-10 Mass | 19.7% |
| Val Unique Tokens | 1,665 |

---

## ✅ Success Criteria

實驗成功需同時滿足：

1. ✅ Val entropy **↑** (> 6.07)
2. ✅ Val top-10 mass **↓** (< 19.7%)
3. ✅ Val strict acc **不惡化** (≥ 0.82%，允許 10% 下降)

---

## 🚀 Ready to Execute

所有組件驗證完成，可以開始執行實驗：

```bash
bash exp_0128/noise_balanced_sampling/run_exp2.sh
```

**預期時間**: 2-3 小時 (1000 steps)

**輸出**:
- `exp_0128/noise_balanced_sampling/run_exp2_YYYYMMDD_HHMMSS/`
  - `config.json` - 實驗配置
  - `metrics_history.json` - 訓練過程 metrics
  - `summary.json` - 最終結果 + baseline 對比
- `exp_0128/noise_balanced_sampling/run_exp2_YYYYMMDD_HHMMSS.log` - 完整 log

---

## 📊 Next Steps

### If Successful (success=true in summary.json)
1. ✅ 執行實驗 1（Soft Reweighting）
2. ✅ 組合實驗 1 + 2
3. ✅ Full training (300 epochs)

### If Failed (success=false)
1. ⚠️ 檢查 metrics_history.json 分析訓練過程
2. ⚠️ 調整 batch_size 或 effective_batch_size
3. ⚠️ 嘗試其他方法（VQ margin regularization 等）

---

✅ **Component check complete! Ready to run experiment.**
