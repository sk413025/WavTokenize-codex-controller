# exp_0128: Token Collapse 修復驗證實驗 (Phase 1)

## 概述

基於 TracIn 診斷結果 ([exp_0125/tracin_token_collapse_589e6d](../exp_0125/tracin_token_collapse_589e6d/CONCLUSION.md)) 的 **Phase 1 短期驗證實驗**，測試兩種修復方案對 Exp K v6 token collapse 問題的有效性。

## 背景：TracIn 診斷結果

### 問題現象

Exp K v6 在訓練 epoch 300 時出現嚴重的 validation token collapse：

| Metric | Value | Severity |
|--------|-------|----------|
| Val Entropy | 6.07 | 🔴 嚴重偏低 (應 > 8.0) |
| Val Top-10 Mass | 19.7% | 🔴 過度集中 (應 < 10%) |
| Val Strict Acc | 0.91% | 🔴 極低準確率 |
| Val KL(student‖teacher) | 1.25 | 🔴 分佈嚴重偏移 |

### TracIn 診斷發現

**Proponents (導致 collapse 的高影響樣本) 特徵**:

1. **噪音材質失衡**: papercup 57% (baseline 33%) → 1.7x 過度代表
2. **SNR 偏低**: -2.24 dB (baseline -1.46 dB) → 偏向噪聲依賴編碼
3. **分佈異常**: Cohen's d = -0.107 (偏離 clean distribution)

**反事實實驗結果**:

- 刪除 proponents → collapse **惡化** (entropy 6.07 → 5.61)
- 刪除 opponents → collapse **改善** (entropy 6.07 → 6.53)

**結論**: Proponents 包含「困難但必要」的樣本，不能完全刪除，需要軟性調整。

## Phase 1 驗證實驗

### 實驗 1: TracIn-Weighted Soft Reweighting

📁 位置: [soft_reweighting/](soft_reweighting/)

**方法**: 使用 TracIn influence scores 進行軟性重加權

```
weight_i = 1 / (1 + α × TracIn_score_i)
```

- High influence proponents → 降低權重 (down-weighted)
- Low influence opponents → 提高權重 (up-weighted)

**參數**: α ∈ {0.3, 0.5, 0.7} (預設 0.5)

**預期**: 減少 proponents 影響，同時保留樣本多樣性

---

### 實驗 2: Noise-Balanced Sampling

📁 位置: [noise_balanced_sampling/](noise_balanced_sampling/)

**方法**: 強制每個 batch 的 noise material 平衡分佈

- papercup : plastic : box = 1 : 1 : 1
- 解決 papercup 過度代表問題 (57% → 33%)

**實現**: `NoiseBalancedSampler` - 從三種材質組中各抽 1/3 樣本

**預期**: 平衡噪音分佈，降低 noise-dependent encoding 風險

---

## 配置一致性

兩個實驗與 baseline (exp_k v6) **完全一致**，唯一差異為 sampler：

| 配置項 | exp_k v6 | exp_0128-1 | exp_0128-2 |
|--------|----------|------------|------------|
| LoRA (rank/alpha/dropout) | 256/512/0.2 | 256/512/0.2 | 256/512/0.2 |
| Intermediate layers | [3, 4, 6] | [3, 4, 6] | [3, 4, 6] |
| Layer weights | {3:0.3, 4:0.5, 6:0.5} | {3:0.3, 4:0.5, 6:0.5} | {3:0.3, 4:0.5, 6:0.5} |
| Loss weights | inter=0.5, feat=0.0, tri=0.0 | 同左 | 同左 |
| Optimizer (lr/wd) | AdamW (1e-4/0.01) | AdamW (1e-4/0.01) | AdamW (1e-4/0.01) |
| Batch (size/accum) | 8/2 (eff=16) | 8/2 (eff=16) | 8/2 (eff=16) |
| Dataset | CurriculumDataset | CurriculumDataset | CurriculumDataset |
| **Sampler** | Random | TracInWeighted | NoiseBalanced |

## 成功判準

與 baseline (exp_k v6 @ epoch 300) 比較：

| Metric | Baseline | 期望變化 | 判定 |
|--------|----------|---------|------|
| Entropy | 6.07 | ↑ 增加 | ✅ entropy > 6.07 |
| Top-10 Mass | 19.7% | ↓ 減少 | ✅ top_10_mass < 19.7% |
| Strict Acc | 0.91% | 不惡化 | ✅ strict_acc ≥ 0.82% |

**成功條件**: 三者同時滿足

```python
success = (
    entropy > 6.07 AND
    top_10_mass < 0.197 AND
    strict_acc >= 0.0091 * 0.9
)
```

## 執行方式

### 方案 A: 平行執行 (推薦，節省時間)

需要 2 張 GPU：

```bash
bash exp_0128/start_parallel.sh
```

- GPU 0: 實驗 2 (2-3 小時)
- GPU 1: 實驗 1 (2-3 小時)
- 總時間: ~2-3 小時 (vs 序列 4-6 小時)

監控進度：
```bash
# 查看日誌
tail -f exp_0128/logs_parallel_*/exp1_gpu1.log
tail -f exp_0128/logs_parallel_*/exp2_gpu0.log

# 查看 GPU 使用率
watch -n 1 nvidia-smi
```

---

### 方案 B: 序列執行

只有 1 張 GPU 時：

```bash
# Step 1: 實驗 2 (2-3 小時)
bash exp_0128/noise_balanced_sampling/run_exp2.sh

# Step 2: 實驗 1 (2-3 小時)
bash exp_0128/soft_reweighting/run_exp1.sh
```

---

## 輸出結構

每個實驗的輸出目錄結構：

```
exp_0128/{experiment_name}/run_{exp_name}_{TIMESTAMP}/
├── config.json              # 實驗配置
├── metrics_history.json     # Collapse metrics (每 200 steps)
├── loss_history.json        # Loss 值 (每 step)
├── summary.json             # 最終結果 + baseline 對比
├── training_curves.png      # 訓練曲線圖
├── final_model.pt           # 最終模型
├── checkpoints/             # 每 200 steps
│   ├── checkpoint_step0200.pt
│   ├── checkpoint_step0400.pt
│   └── ...
└── audio_samples/           # 音檔樣本
    ├── train/step_XXXX/
    │   ├── sample_1_noisy.wav
    │   ├── sample_1_clean.wav
    │   └── sample_1_student_recon.wav
    └── val/step_XXXX/
        └── (同上)
```

## 結果分析

查看 `summary.json`：

```bash
cat exp_0128/*/run_*/summary.json | jq
```

關鍵字段：
```json
{
  "experiment": "exp_0128_...",
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
    "entropy": ...,      // 正值 = 改善
    "top_10_mass": ...,  // 負值 = 改善
    "strict_acc": ...    // 正值 = 改善
  },
  "success": true/false
}
```

## 下一步 (Phase 2)

### 如果兩個實驗都成功

進行 **full training (300 epochs)** 驗證：
- 實驗 3: Full Noise-Balanced Training
- 實驗 4: Full TracIn-Weighted Training
- 實驗 5: 組合兩種方法

### 如果只有一個成功

- 成功的方法 → Phase 2 full training
- 失敗的方法 → 調整參數或放棄

### 如果都失敗

考慮其他方法（Section 6.3）：
- VQ margin regularization
- Gradient clipping on proponents
- Multi-task learning

## 預期時間線

| 階段 | 時間 | 備註 |
|------|------|------|
| Phase 1 (此實驗) | 2-3 小時 | 平行執行 |
| 結果分析 | 0.5 小時 | 檢查 summary.json |
| Phase 2 (full training) | 2-3 天 | 如果 Phase 1 成功 |

## 資源需求

- **GPU**: 2 張 (平行) 或 1 張 (序列)
- **VRAM**: ~12GB per GPU
- **儲存**: ~5GB per experiment (checkpoints + audio)

## 參考文件

- TracIn 診斷報告: [exp_0125/tracin_token_collapse_589e6d/CONCLUSION.md](../exp_0125/tracin_token_collapse_589e6d/CONCLUSION.md)
- Baseline 配置: [exp_0112_intermediate/train_v6.py](../exp_0112_intermediate/train_v6.py)
- 實驗 1 詳情: [soft_reweighting/README.md](soft_reweighting/README.md)
- 實驗 2 詳情: [noise_balanced_sampling/README.md](noise_balanced_sampling/README.md)
