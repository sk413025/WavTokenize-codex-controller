# exp_0128: 準備就緒檢查清單

## ✅ 實驗 1: TracIn-Weighted Soft Reweighting

### 核心組件
- ✅ [soft_reweighting/tracin_weighted_sampler.py](soft_reweighting/tracin_weighted_sampler.py) - TracInWeightedSampler 實現
- ✅ [soft_reweighting/data_weighted.py](soft_reweighting/data_weighted.py) - DataLoader 創建
- ✅ [soft_reweighting/train_short_run.py](soft_reweighting/train_short_run.py) - 訓練腳本
- ✅ [soft_reweighting/run_exp1.sh](soft_reweighting/run_exp1.sh) - 執行腳本 (GPU 1)
- ✅ [soft_reweighting/README.md](soft_reweighting/README.md) - 文檔

### 功能完整性
- ✅ Checkpoint 保存 (每 200 steps)
- ✅ 音檔保存 (train/val, 每 500 steps)
- ✅ Loss 曲線圖 (更新於每 200 steps)
- ✅ Collapse metrics 評估 (每 200 steps)
- ✅ 最終模型保存
- ✅ 與 baseline 配置一致 (除 sampler)

### 關鍵參數
- alpha: 0.5 (可調整: 0.3, 0.5, 0.7)
- batch_size: 2
- grad_accum: 2
- steps: 1000
- GPU: 1 (CUDA_VISIBLE_DEVICES=1)

---

## ✅ 實驗 2: Noise-Balanced Sampling

### 核心組件
- ✅ [noise_balanced_sampling/sampler.py](noise_balanced_sampling/sampler.py) - NoiseBalancedSampler 實現
- ✅ [noise_balanced_sampling/data_balanced.py](noise_balanced_sampling/data_balanced.py) - DataLoader 創建
- ✅ [noise_balanced_sampling/train_short_run.py](noise_balanced_sampling/train_short_run.py) - 訓練腳本
- ✅ [noise_balanced_sampling/run_exp2.sh](noise_balanced_sampling/run_exp2.sh) - 執行腳本 (GPU 0)
- ✅ [noise_balanced_sampling/README.md](noise_balanced_sampling/README.md) - 文檔

### 功能完整性
- ✅ Checkpoint 保存 (每 200 steps)
- ✅ 音檔保存 (train/val, 每 500 steps)
- ✅ Loss 曲線圖 (更新於每 200 steps)
- ✅ Collapse metrics 評估 (每 200 steps)
- ✅ 最終模型保存
- ✅ 與 baseline 配置一致 (除 sampler)

### 關鍵參數
- Noise balance: papercup:plastic:box = 1:1:1
- batch_size: 2
- grad_accum: 2
- steps: 1000
- GPU: 0 (CUDA_VISIBLE_DEVICES=0)

---

## ✅ 執行工具

- ✅ [start_parallel.sh](start_parallel.sh) - 平行執行兩個實驗
- ✅ [README.md](README.md) - 總體說明文檔

---

## 配置驗證

### 與 Baseline 一致性檢查

| 配置項 | exp_k v6 | exp_0128-1 | exp_0128-2 | 狀態 |
|--------|----------|------------|------------|------|
| LoRA rank | 256 | 256 | 256 | ✅ |
| LoRA alpha | 512 | 512 | 512 | ✅ |
| LoRA dropout | 0.2 | 0.2 | 0.2 | ✅ |
| Intermediate layers | [3,4,6] | [3,4,6] | [3,4,6] | ✅ |
| Layer weights | {3:0.3, 4:0.5, 6:0.5} | {3:0.3, 4:0.5, 6:0.5} | {3:0.3, 4:0.5, 6:0.5} | ✅ |
| Intermediate weight | 0.5 | 0.5 | 0.5 | ✅ |
| Feature weight | 0.0 | 0.0 | 0.0 | ✅ |
| Triplet weight | 0.0 | 0.0 | 0.0 | ✅ |
| Learning rate | 1e-4 | 1e-4 | 1e-4 | ✅ |
| Weight decay | 0.01 | 0.01 | 0.01 | ✅ |
| Batch size | 2 | 2 | 2 | ✅ |
| Grad accumulation | 2 | 2 | 2 | ✅ |
| Optimizer | AdamW | AdamW | AdamW | ✅ |
| AMP | GradScaler | GradScaler | GradScaler | ✅ |
| Loss function | MaskedCombinedLossV2 | MaskedCombinedLossV2 | MaskedCombinedLossV2 | ✅ |
| Inter loss | IntermediateSupervisionLossV6 | IntermediateSupervisionLossV6 | IntermediateSupervisionLossV6 | ✅ |
| Dataset | CurriculumDataset | CurriculumDataset | CurriculumDataset | ✅ |

**差異點** (預期):
- exp_0128-1: 使用 `TracInWeightedSampler` (alpha=0.5)
- exp_0128-2: 使用 `NoiseBalancedSampler` (1:1:1 ratio)

---

## 依賴檢查

### 必要文件
- ✅ `exp_1201/config.py` (TRAIN_CACHE, VAL_CACHE, WAVTOK_CONFIG, WAVTOK_CKPT)
- ✅ `exp_0112_intermediate/models.py` (TeacherStudentIntermediate)
- ✅ `exp_0112_intermediate/train_v6.py` (IntermediateSupervisionLossV6)
- ✅ `exp_1219/losses.py` (MaskedCombinedLossV2)
- ✅ `exp_1226/data_curriculum.py` (CurriculumDataset, collate_fn_curriculum)
- ✅ `exp_0125/tracin_token_collapse_589e6d/tracin_scores_5ckpt.csv` (TracIn scores)

### 數據文件
- ✅ Train cache: 已存在 (從 config 中引用)
- ✅ Val cache: 已存在 (從 config 中引用)
- ✅ WavTokenizer config: 已存在
- ✅ WavTokenizer checkpoint: 已存在

---

## 執行前檢查

### 環境變數
```bash
export PYTHONPATH="/home/sbplab/ruizi/WavTokenize-feature-analysis:$PYTHONPATH"
```
✅ 已在執行腳本中設定

### GPU 可用性
需要檢查：
```bash
nvidia-smi
```
- GPU 0: 可用於實驗 2
- GPU 1: 可用於實驗 1

### 磁碟空間
每個實驗預估需要 ~5GB：
- Checkpoints: ~2GB (5 checkpoints × ~400MB)
- Audio samples: ~500MB
- Logs & metrics: ~100MB

建議可用空間: **> 15GB**

---

## 🚀 執行指令

### 平行執行 (推薦)

```bash
cd /home/sbplab/ruizi/WavTokenize-feature-analysis
bash exp_0128/start_parallel.sh
```

### 監控進度

```bash
# 查看日誌
tail -f exp_0128/logs_parallel_*/exp1_gpu1.log
tail -f exp_0128/logs_parallel_*/exp2_gpu0.log

# 查看 GPU 使用率
watch -n 1 nvidia-smi

# 查看進程狀態
ps aux | grep train_short_run
```

---

## 預期輸出

### 實驗 1
```
exp_0128/soft_reweighting/run_exp1_YYYYMMDD_HHMMSS/
├── config.json
├── metrics_history.json
├── loss_history.json
├── summary.json
├── training_curves.png
├── final_model.pt
├── checkpoints/
│   ├── checkpoint_step0200.pt
│   ├── checkpoint_step0400.pt
│   ├── checkpoint_step0600.pt
│   ├── checkpoint_step0800.pt
│   └── checkpoint_step1000.pt
└── audio_samples/
    ├── train/
    │   ├── step_0000/
    │   ├── step_0500/
    │   └── step_1000/
    └── val/
        ├── step_0000/
        ├── step_0500/
        └── step_1000/
```

### 實驗 2
同上結構，位於 `exp_0128/noise_balanced_sampling/run_exp2_YYYYMMDD_HHMMSS/`

---

## 成功判準

查看 `summary.json` 中的 `success` 欄位：

```json
{
  "success": true,  // ← 此欄位為 true 表示成功
  "improvement": {
    "entropy": 0.XX,      // > 0 表示改善
    "top_10_mass": -0.XX, // < 0 表示改善
    "strict_acc": 0.XX    // > -0.0009 表示不惡化
  }
}
```

成功條件：
- ✅ entropy > 6.07 (baseline)
- ✅ top_10_mass < 0.197 (baseline)
- ✅ strict_acc ≥ 0.0082 (baseline × 0.9)

---

## 預期執行時間

- 實驗 1: 2-3 小時 (1000 steps)
- 實驗 2: 2-3 小時 (1000 steps)
- **平行執行總時間**: ~2-3 小時

---

## ✅ 最終確認

- [x] 所有腳本已創建
- [x] 所有腳本可執行 (chmod +x)
- [x] 配置與 baseline 一致
- [x] 功能完整 (checkpoint/audio/plot)
- [x] 文檔完整
- [x] GPU 分配正確 (Exp1=GPU1, Exp2=GPU0)
- [x] 環境變數設定正確
- [x] 依賴文件存在

---

## 🎯 準備就緒！

執行以下指令開始實驗：

```bash
bash exp_0128/start_parallel.sh
```

祝實驗成功！🚀
