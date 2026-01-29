# exp_0128: 實驗 1 - TracIn-Weighted Soft Reweighting

## 概述

基於 TracIn 診斷結果 (exp_0125/tracin_token_collapse_589e6d) 的 **軟性重加權 (Soft Reweighting)** 修復方案。

## 動機

TracIn 分析發現 token collapse 的主要proponents (高影響樣本) 特徵：
- SNR 較低 (-2.24 dB vs baseline -1.46 dB)
- papercup 材質過多 (57% vs baseline 33%)
- 噪音分佈異常 (Cohen's d = -0.107，偏向噪聲依賴編碼)

**反事實實驗** (Section 3) 顯示：
- 刪除 proponents → collapse **惡化** (entropy 6.07 → 5.61)
- 刪除 opponents → collapse 改善 (entropy 6.07 → 6.53)

結論：proponents 包含「困難但必要」的樣本，不能完全刪除。

## 方法

### 軟性重加權 (Soft Reweighting)

使用 TracIn scores 計算樣本權重：

```
weight_i = 1 / (1 + α × TracIn_score_i)
```

- **High TracIn score** (proponents) → **低權重** (down-weighted)
- **Low TracIn score** (opponents) → **高權重** (up-weighted)
- **α**: 控制重加權強度 (建議測試 0.3, 0.5, 0.7)

### 實現細節

1. **TracInWeightedSampler** ([tracin_weighted_sampler.py](tracin_weighted_sampler.py)):
   - 讀取 `tracin_scores_5ckpt.csv` (5 checkpoints × 2000 samples)
   - 對每個樣本，取 5 個 checkpoints 的平均 TracIn score
   - 使用 PyTorch `WeightedRandomSampler` 進行加權抽樣

2. **DataLoader** ([data_weighted.py](data_weighted.py)):
   - 基於 `exp_1226/data_curriculum.py` 的 `CurriculumDataset`
   - 使用 `TracInWeightedSampler` 替換預設 sampler

3. **Training Script** ([train_short_run.py](train_short_run.py)):
   - 1000 steps short-run 驗證
   - batch_size=2, grad_accum=2 (有效 batch=4，與 baseline 一致)
   - lr=1e-4, weight_decay=0.01
   - 其他配置與 exp_k v6 baseline 完全相同

## 配置與 Baseline 一致性

| 配置項 | exp_k v6 (baseline) | exp_0128-1 (此實驗) | 是否一致 |
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
| Batch size | 2 | 2 | ✅ |
| Grad accumulation | 2 | 2 | ✅ |
| Effective batch | 4 | 4 | ✅ |
| Optimizer | AdamW | AdamW | ✅ |
| AMP | GradScaler | GradScaler | ✅ |
| Dataset | CurriculumDataset | CurriculumDataset | ✅ |

**唯一差異**: 使用 `TracInWeightedSampler` 替換預設 random sampler

## 輸出內容

### 1. Checkpoints
- 位置: `{output_dir}/checkpoints/`
- 保存頻率: 每 200 steps
- 內容: model_state_dict, optimizer_state_dict, alpha, config

### 2. 音檔樣本
- 位置: `{output_dir}/audio_samples/train/` 和 `/val/`
- 保存頻率: 每 500 steps + 初始 + 最終
- 每次保存 2 個樣本，包含:
  - `sample_N_noisy.wav` (輸入噪聲音檔)
  - `sample_N_clean.wav` (目標乾淨音檔)
  - `sample_N_student_recon.wav` (學生模型重建音檔)

### 3. 訓練曲線
- 位置: `{output_dir}/training_curves.png`
- 更新頻率: 每 200 steps
- 內容:
  - 左圖: Total Loss, Main Loss, Intermediate Loss
  - 右圖: Entropy 和 Top-10 Mass (雙 Y 軸)

### 4. Metrics 與 Loss History
- `metrics_history.json`: 每 200 steps 的 collapse metrics
- `loss_history.json`: 每 step 的 loss 值
- `summary.json`: 最終結果摘要與成功判定

### 5. 最終模型
- 位置: `{output_dir}/final_model.pt`
- 內容: 最終 step 的完整狀態

## 成功判準

與 exp_k v6 baseline (val epoch 300) 比較：

| Metric | Baseline | 期望變化 |
|--------|----------|---------|
| Entropy | 6.07 | ↑ (改善) |
| Top-10 Mass | 19.7% | ↓ (改善) |
| Strict Accuracy | 0.91% | ≥ 0.82% (不惡化超過 10%) |

**成功條件**:
```python
success = (
    entropy > 6.07 AND
    top_10_mass < 19.7% AND
    strict_acc >= 0.0091 * 0.9
)
```

## 執行

### 單獨執行 (GPU 1)

```bash
bash exp_0128/soft_reweighting/run_exp1.sh
```

### 與實驗 2 平行執行

```bash
bash exp_0128/start_parallel.sh
```

## 參數調整

測試不同 alpha 值：

```bash
# Alpha = 0.3 (弱重加權)
ALPHA=0.3 bash exp_0128/soft_reweighting/run_exp1.sh

# Alpha = 0.5 (中等重加權，預設)
ALPHA=0.5 bash exp_0128/soft_reweighting/run_exp1.sh

# Alpha = 0.7 (強重加權)
ALPHA=0.7 bash exp_0128/soft_reweighting/run_exp1.sh
```

## 預期執行時間

- GPU: RTX 3090/A100
- 1000 steps ≈ 2-3 小時

## 參考

- TracIn 診斷報告: [exp_0125/tracin_token_collapse_589e6d/CONCLUSION.md](../../exp_0125/tracin_token_collapse_589e6d/CONCLUSION.md)
- Baseline 實驗: [exp_0112_intermediate/run_exp_k_v6.sh](../../exp_0112_intermediate/run_exp_k_v6.sh)
- 相關實驗: [exp_0128/noise_balanced_sampling](../noise_balanced_sampling) (實驗 2)
