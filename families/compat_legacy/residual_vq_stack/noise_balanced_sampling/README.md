# Exp 0128: Noise-Balanced Sampling

## 目的

解決 Exp K v6 的 valid token collapse 問題（TracIn 診斷結果）。

## 問題診斷（來自 TracIn 分析）

TracIn 分析（commit d0f9ecb）發現：

| 指標 | Proponents (導致 collapse) | 全體 Train | 倍數 |
|------|---------------------------|-----------|------|
| **papercup 材質** | **57%** | 33% | **1.7x** |
| **SNR** | -2.24 dB | -1.88 dB | 較低 0.36 dB |

**結論**：papercup 材質 + 低 SNR 的訓練樣本主導了 LoRA 更新，導致模型學會 noise-dependent encoding，無法在 validation set 上泛化。

## 解決方案：實驗 2 - Noise-Balanced Sampling

### 方法

強制每個 batch 的 **papercup/plastic/box 各佔 1/3**，平衡噪音材質分佈。

### 實作

1. **NoiseBalancedSampler** ([sampler.py](sampler.py))
   - 從檔名提取 noise type: `{speaker}_{noise_type}_LDV_{id}.wav`
   - 將 dataset 按 noise type 分組
   - 每個 batch 從三組中各抽 1/3 樣本

2. **Noise-Balanced DataLoader** ([data_balanced.py](data_balanced.py))
   - 基於 `exp_1226/data_curriculum.py`
   - 使用 `NoiseBalancedSampler` 取代 `CurriculumSampler`
   - 保持其他配置不變

3. **Short-Run Training** ([train_short_run.py](train_short_run.py))
   - 基於 `exp_0112_intermediate/train_v6.py`（exp_k v6 baseline）
   - 800-1000 steps
   - 評估 val collapse metrics

## 配置與 Baseline 一致性

| 配置項 | exp_k v6 (baseline) | exp_0128-2 (此實驗) | 是否一致 |
|--------|---------------------|---------------------|---------|
| LoRA rank | 256 | 256 | ✅ |
| LoRA alpha | 512 | 512 | ✅ |
| LoRA dropout | 0.2 | 0.2 | ✅ |
| Intermediate layers | [3, 4, 6] | [3, 4, 6] | ✅ |
| Layer weights | {3: 0.3, 4: 0.5, 6: 0.5} | {3: 0.3, 4: 0.5, 6: 0.5} | ✅ |
| Intermediate weight | 0.5 (固定) | 0.5 (固定) | ✅ |
| Feature weight | 0.0 | 0.0 | ✅ |
| Triplet weight | 0.0 | 0.0 | ✅ |
| Learning rate | 1e-4 | 1e-4 | ✅ |
| Weight decay | 0.01 | 0.01 | ✅ |
| Batch size | 8 | 8 | ✅ |
| Grad accumulation | 2 | 2 | ✅ |
| Effective batch | 16 | 16 | ✅ |
| Optimizer | AdamW | AdamW | ✅ |
| AMP | GradScaler | GradScaler | ✅ |
| Dataset | CurriculumDataset | CurriculumDataset | ✅ |

**唯一差異**: 使用 `NoiseBalancedSampler` 替換預設 random sampler，強制每個 batch 的 noise material 平衡分佈 (papercup/plastic/box 各 1/3)

## Baseline（Exp K v6）

| 指標 | 數值 |
|------|------|
| Val Strict Acc | 0.91% |
| Val Entropy | 6.07 |
| Val Top-10 Mass | 19.7% |
| Val KL(student‖teacher) | 1.25 |

## 成功判準

- ✅ Val entropy **↑** (> 6.07)
- ✅ Val top-10 mass **↓** (< 19.7%)
- ✅ Val strict acc **不惡化** (≥ 0.82%)

## 執行

### 快速測試

```bash
bash exp_0128/noise_balanced_sampling/run_exp2.sh
```

### 自定義參數

```bash
source /home/sbplab/miniconda3/etc/profile.d/conda.sh
conda activate test

CUDA_VISIBLE_DEVICES=1 python exp_0128/noise_balanced_sampling/train_short_run.py \
    --steps 1000 \
    --batch_size 2 \
    --grad_accum 2 \
    --lr 1e-4 \
    --output_dir exp_0128/noise_balanced_sampling/run_exp2_test \
    --seed 42 \
    --eval_interval 200
```

## 預期資源

- **GPU**: 1 張 (CUDA_VISIBLE_DEVICES=1)
- **時間**: 2-3 小時（1000 steps）
- **記憶體**: ~12GB VRAM

## 輸出檔案

```
exp_0128/noise_balanced_sampling/run_exp2_YYYYMMDD_HHMMSS/
├── config.json              # 實驗配置
├── metrics_history.json     # 每 200 steps 的 collapse metrics
├── loss_history.json        # 每 step 的 loss 值
├── summary.json             # 最終結果 + baseline 對比
├── training_curves.png      # 訓練曲線 (loss + collapse metrics)
├── final_model.pt           # 最終模型
├── checkpoints/             # 模型 checkpoints (每 200 steps)
│   ├── checkpoint_step0200.pt
│   ├── checkpoint_step0400.pt
│   └── ...
└── audio_samples/           # 音檔樣本
    ├── train/
    │   ├── step_0000/
    │   │   ├── sample_1_noisy.wav
    │   │   ├── sample_1_clean.wav
    │   │   └── sample_1_student_recon.wav
    │   ├── step_0500/
    │   └── ...
    └── val/
        └── (同上結構)
```

### 輸出說明

1. **Checkpoints**: 每 200 steps 保存，包含 model_state_dict, optimizer_state_dict
2. **音檔樣本**:
   - 初始 (step 0), 每 500 steps, 最終 (step 1000)
   - 每次保存 2 個樣本 (train/val 各 2 個)
   - 包含 noisy, clean, student_recon 三種音檔
3. **訓練曲線**:
   - 左圖: Total Loss, Main Loss, Intermediate Loss
   - 右圖: Entropy 和 Top-10 Mass (雙 Y 軸)
4. **最終模型**: step 1000 的完整狀態

## 結果分析

查看 `summary.json`：

```json
{
  "baseline": {
    "entropy": 6.07,
    "top_10_mass": 0.197,
    "strict_acc": 0.0091
  },
  "final": {
    "entropy": ...,
    "top_10_mass": ...,
    "strict_acc": ...
  },
  "improvement": {
    "entropy": ...,
    "top_10_mass": ...,
    "strict_acc": ...
  },
  "success": true/false
}
```

## 下一步

如果實驗 2 成功：
- ✅ 進行實驗 1（Soft Reweighting）
- ✅ 組合實驗 1 + 2（full training）

如果實驗 2 失敗：
- ⚠️ 檢查 noise type distribution
- ⚠️ 調整 batch_size 或 sampling 策略
- ⚠️ 考慮其他解決方案（VQ margin regularization 等）

## 參考

- TracIn 診斷結果: [exp_0125/tracin_token_collapse_589e6d/CONCLUSION.md](../../exp_0125/tracin_token_collapse_589e6d/CONCLUSION.md)
- Baseline 配置: [exp_0112_intermediate/train_v6.py](../../exp_0112_intermediate/train_v6.py)
