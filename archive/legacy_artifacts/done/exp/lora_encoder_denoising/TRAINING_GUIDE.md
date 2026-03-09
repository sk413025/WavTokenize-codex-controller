# LoRA Encoder Denoising 訓練指南

## 🎯 快速開始

### 1. 使用真實數據訓練

數據已準備完成！我們已找到原始音訊檔案並配置好數據載入：

```bash
cd /home/sbplab/ruizi/WavTokenize-self-supervised/done/exp/lora_encoder_denoising

# 啟動訓練（使用真實 noisy-clean audio pairs）
python train.py \
  --exp_name lora_denoising_v1 \
  --num_epochs 50 \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --lora_rank 16 \
  --lora_alpha 32
```

**數據統計**:
- Training: 7,776 samples
- Validation: 1,440 samples
- 數據來源:
  - Noisy: `/home/sbplab/ruizi/WavTokenize/data/raw/box` + `papercup` (9,792 WAV files)
  - Clean: `/home/sbplab/ruizi/WavTokenize/data/clean/box2` (5,184 WAV files)

---

## 📂 數據配置

### 自動從 file paths 載入音訊

我們的 `data.py` 已經修改為支援從 cache 中的檔案路徑自動載入音訊：

```python
# data.py 自動處理:
# 1. 優先使用 cache 中的 audio waveforms (如果有)
# 2. 否則從 noisy_path/clean_path 載入音訊檔案
# 3. Fallback 到 dummy data (for smoke test)
```

**Cache 結構**:
```
data_with_distances/
  ├── train_cache_with_distances.pt  (64GB, 7,776 samples)
  ├── val_cache_with_distances.pt    (15GB, 1,440 samples)
  └── cache_with_distances.h5        (32GB, HDF5 format)

每個 sample 包含:
  - noisy_path: str  (相對路徑到 noisy audio)
  - clean_path: str  (相對路徑到 clean audio)
  - noisy_tokens, clean_tokens: (T,) int64
  - noisy_distances, clean_distances: (T, 4096) float32
  - speaker_embedding: (192,) float32
```

---

## 🚀 訓練選項

### 基礎訓練

```bash
python train.py --exp_name my_experiment
```

### 完整參數說明

```bash
python train.py \
  --exp_name lora_denoising_full \        # 實驗名稱（必須）
  --num_epochs 50 \                       # Epoch 數量
  --batch_size 16 \                       # Batch size
  --learning_rate 5e-5 \                  # 學習率
  --weight_decay 0.01 \                   # Weight decay
  --grad_clip 1.0 \                       # Gradient clipping
  --lora_rank 16 \                        # LoRA rank
  --lora_alpha 32 \                       # LoRA alpha
  --lora_dropout 0.1 \                    # LoRA dropout
  --feature_loss_weight 1.0 \             # Feature loss 權重
  --distance_loss_weight 0.1 \            # Distance loss 權重
  --vq_loss_weight 0.01 \                 # VQ loss 權重
  --num_workers 4 \                       # DataLoader workers
  --seed 42                               # Random seed
```

### 小規模測試（推薦先執行）

```bash
# 使用 smoke test 驗證訓練流程
python smoke_test.py

# 小規模訓練測試（2-3 分鐘）
python train.py \
  --exp_name test_run \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 1e-4
```

---

## 📊 訓練監控

### Tensorboard

```bash
# 啟動 Tensorboard
tensorboard --logdir experiments/

# 瀏覽器打開 http://localhost:6006
```

**可視化指標**:
- `train/loss` - 總 loss
- `train/feature_loss` - Feature MSE loss
- `train/distance_loss` - Distance-based soft target loss
- `train/vq_loss` - VQ commit loss
- `train/lr` - Learning rate
- `val/*` - Validation metrics

### 訓練輸出

```
experiments/
  └── lora_denoising_v1/
      ├── checkpoints/
      │   ├── best.pt                    # 最佳 checkpoint
      │   ├── latest.pt                  # 最新 checkpoint
      │   └── epoch_XXX_loss_Y.YYYY.pt   # Top-K checkpoints
      ├── logs/
      │   └── events.out.tfevents.*      # Tensorboard logs
      └── config.json                    # 訓練配置
```

---

## 🔧 進階配置

### 使用不同的 LoRA 配置

```bash
# 更小的 LoRA (更快，參數更少)
python train.py \
  --exp_name lora_r8 \
  --lora_rank 8 \
  --lora_alpha 16

# 更大的 LoRA (更強表達能力)
python train.py \
  --exp_name lora_r32 \
  --lora_rank 32 \
  --lora_alpha 64
```

### 調整 Loss 權重

```bash
# 強調 feature matching
python train.py \
  --exp_name feature_heavy \
  --feature_loss_weight 2.0 \
  --distance_loss_weight 0.05

# 強調 distance-based soft targets
python train.py \
  --exp_name distance_heavy \
  --feature_loss_weight 0.5 \
  --distance_loss_weight 0.5
```

---

## 🧪 實驗建議

### 初始實驗

1. **Smoke Test** - 2-5 分鐘驗證
   ```bash
   ./run_smoke_test.sh
   ```

2. **小規模訓練** - 10-15 分鐘快速測試
   ```bash
   python train.py --exp_name quick_test --num_epochs 5 --batch_size 4
   ```

3. **完整訓練** - 3-6 小時（視硬體而定）
   ```bash
   python train.py --exp_name full_run --num_epochs 50 --batch_size 16
   ```

### 超參數搜索

建議的超參數範圍：

| 參數 | 範圍 | 預設值 |
|------|------|--------|
| `lora_rank` | 8-32 | 16 |
| `lora_alpha` | 16-64 | 32 |
| `learning_rate` | 1e-5 to 1e-4 | 5e-5 |
| `feature_loss_weight` | 0.5-2.0 | 1.0 |
| `distance_loss_weight` | 0.05-0.5 | 0.1 |

---

## ⚡ 性能優化

### GPU 記憶體優化

```bash
# 減小 batch size
python train.py --exp_name low_mem --batch_size 4

# 使用 gradient accumulation (未實現，可手動添加)
# 可以在 train.py 中添加 gradient accumulation 邏輯
```

### 訓練速度優化

```bash
# 增加 DataLoader workers
python train.py --exp_name fast --num_workers 8

# 使用更大的 batch size (如果記憶體允許)
python train.py --exp_name fast --batch_size 32
```

---

## 📝 檢查點管理

### 載入 checkpoint 繼續訓練

目前 `train.py` 不支援 resume，但可以手動添加。修改 `train.py`:

```python
# 在 Trainer.__init__() 中添加:
if args.resume_from is not None:
    self.load_checkpoint(args.resume_from)
```

### 手動載入 checkpoint

```python
import torch

checkpoint = torch.load('experiments/lora_denoising_v1/checkpoints/best.pt')

# 包含:
# - epoch: int
# - global_step: int
# - model_state_dict: OrderedDict
# - optimizer_state_dict: dict
# - scheduler_state_dict: dict
# - val_metrics: dict
# - config: dict
```

---

## 🐛 故障排除

### 常見問題

1. **FileNotFoundError: Audio file not found**
   - 檢查 `data_with_distances/*.pt` 中的路徑是否正確
   - 確認音訊檔案存在於 `../../data/raw/` 和 `../../data/clean/`

2. **CUDA Out of Memory**
   - 減小 `--batch_size`
   - 減小 `--lora_rank`

3. **Distance matrix not found**
   - 執行 `python generate_distance_matrix.py` 生成

4. **訓練 loss NaN**
   - 降低 `--learning_rate`
   - 增加 `--grad_clip`

### 檢查數據

```bash
# 使用 check_data.py 驗證數據格式
python check_data.py
```

---

## 📖 相關文件

- [REPRODUCE.md](./REPRODUCE.md) - 完整的實驗重現指南
- [EXPERIMENT_REPORT.md](./EXPERIMENT_REPORT.md) - 實驗背景與結果報告
- [README.md](./README.md) - 項目概覽

---

## 🎓 技術細節

### Teacher-Student 架構

```
Noisy Audio → Student Encoder (LoRA) → Student Features
Clean Audio → Teacher Encoder (frozen) → Teacher Features

Loss = Feature MSE + Distance-based Soft Target + VQ Commit Loss
```

### LoRA 配置

- **Target Modules**: 4個 strided convolutions
  ```
  feature_extractor.encodec.encoder.model.{0,3,6,9}.conv.conv
  ```
- **Trainable Parameters**: ~19,256 (0.024% of total 80M)
- **Parameter Efficiency**: 4,183x reduction

### Loss Function

```python
total_loss = 1.0 * feature_loss       # Feature-level MSE
           + 0.1 * distance_loss      # Distance-based soft target
           + 0.01 * vq_loss           # VQ commit loss
```

---

**準備好開始訓練了嗎？**

```bash
# 立即開始！
python train.py --exp_name my_first_experiment
```
