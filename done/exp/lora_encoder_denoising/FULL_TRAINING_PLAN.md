# LoRA Encoder Denoising - 完整訓練計劃

## 📊 系統狀態

### 可用資源
- **GPU 0**: GTX 1080 Ti (11 GB, 11.1 GB free)
- **GPU 1**: RTX 2080 Ti (11 GB, 10.8 GB free)
- **GPU 2**: RTX 2080 Ti (11 GB, 10.8 GB free)

### 數據規模
- **訓練集**: 7,776 samples
- **驗證集**: 1,440 samples
- **Batch 處理**: ~3,888 iterations/epoch (batch_size=2)

### 預估訓練時間
- **每個 iteration**: ~0.1s (測試觀察到 10-12 it/s)
- **每個 epoch**: ~6-7 分鐘 (3,888 iterations)
- **50 epochs**: ~5-6 小時

---

## 🎯 訓練策略

### Phase 1: Baseline 訓練 (最高優先級)

**目標**: 建立性能基準，驗證系統穩定性

```bash
# Experiment: baseline_r16_lr5e5
python train.py \
  --exp_name baseline_r16_lr5e5 \
  --num_epochs 50 \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --feature_loss_weight 1.0 \
  --distance_loss_weight 0.1 \
  --vq_loss_weight 0.01 \
  --num_workers 4 \
  --seed 42
```

**預期結果**:
- Loss 收斂趨勢
- Feature/Distance/VQ loss 比例
- 最佳 checkpoint validation loss

---

### Phase 2: LoRA Rank 探索 (平行執行)

測試不同 LoRA rank 對性能的影響

**Experiment 2a: Small LoRA (rank=8)**
```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  --exp_name lora_r8_lr5e5 \
  --num_epochs 50 \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --num_workers 4 \
  --seed 42
```

**Experiment 2b: Large LoRA (rank=32)**
```bash
CUDA_VISIBLE_DEVICES=2 python train.py \
  --exp_name lora_r32_lr5e5 \
  --num_epochs 50 \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --num_workers 4 \
  --seed 42
```

**比較指標**:
- Trainable parameters: 8 vs 16 vs 32
- Validation loss
- Training speed
- Overfitting 程度

---

### Phase 3: Learning Rate 探索

基於 Phase 1 結果，測試不同 learning rate

**Experiment 3a: Higher LR**
```bash
python train.py \
  --exp_name baseline_r16_lr1e4 \
  --num_epochs 50 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --num_workers 4 \
  --seed 42
```

**Experiment 3b: Lower LR**
```bash
python train.py \
  --exp_name baseline_r16_lr1e5 \
  --num_epochs 50 \
  --batch_size 8 \
  --learning_rate 1e-5 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --num_workers 4 \
  --seed 42
```

---

### Phase 4: Loss Weight 調整

探索 Feature vs Distance loss 的最佳比例

**Experiment 4a: Feature-Heavy**
```bash
python train.py \
  --exp_name feature_heavy_r16 \
  --num_epochs 50 \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --feature_loss_weight 2.0 \
  --distance_loss_weight 0.05 \
  --vq_loss_weight 0.01 \
  --num_workers 4 \
  --seed 42
```

**Experiment 4b: Distance-Heavy**
```bash
python train.py \
  --exp_name distance_heavy_r16 \
  --num_epochs 50 \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --feature_loss_weight 0.5 \
  --distance_loss_weight 0.5 \
  --vq_loss_weight 0.01 \
  --num_workers 4 \
  --seed 42
```

---

## 📈 監控與評估

### 1. Tensorboard 實時監控

```bash
# 啟動 Tensorboard (在另一個 terminal)
tensorboard --logdir experiments/ --port 6006
```

**關鍵指標**:
- `train/loss` - 總 loss 下降趨勢
- `train/feature_loss` - Feature MSE
- `train/distance_loss` - Distance-based soft target
- `train/vq_loss` - VQ commit loss
- `train/lr` - Learning rate schedule
- `val/loss` - Validation loss (early stopping 指標)

### 2. 定期檢查

每 5-10 epochs 檢查：
```bash
# 查看最新訓練狀態
tail -30 experiments/<exp_name>/logs/events.out.tfevents.*

# 檢查 checkpoint
ls -lh experiments/<exp_name>/checkpoints/
```

### 3. 性能評估

訓練完成後，使用最佳 checkpoint 評估：
- Feature distance (Teacher vs Student)
- Code match rate (VQ tokens)
- 音訊質量 (如有評估腳本)

---

## 🚀 執行策略

### 建議執行順序

**Day 1**: Phase 1 - Baseline
```bash
# GPU 0: Baseline 訓練
CUDA_VISIBLE_DEVICES=0 nohup python train.py \
  --exp_name baseline_r16_lr5e5 \
  --num_epochs 50 \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --num_workers 4 \
  --seed 42 \
  > experiments/baseline_r16_lr5e5.log 2>&1 &

echo $! > baseline_pid.txt
```

**Day 2**: Phase 2 - LoRA Rank 探索 (平行)
```bash
# GPU 1: rank=8
CUDA_VISIBLE_DEVICES=1 nohup python train.py \
  --exp_name lora_r8_lr5e5 \
  --num_epochs 50 \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --num_workers 4 \
  --seed 42 \
  > experiments/lora_r8.log 2>&1 &

# GPU 2: rank=32
CUDA_VISIBLE_DEVICES=2 nohup python train.py \
  --exp_name lora_r32_lr5e5 \
  --num_epochs 50 \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --num_workers 4 \
  --seed 42 \
  > experiments/lora_r32.log 2>&1 &
```

**Day 3+**: 根據前面結果決定 Phase 3 & 4

---

## 📋 Launch Script

創建便捷啟動腳本：

```bash
# run_baseline.sh
#!/bin/bash

cd /home/sbplab/ruizi/WavTokenize-self-supervised/done/exp/lora_encoder_denoising

# 設定實驗參數
EXP_NAME="baseline_r16_lr5e5"
GPU_ID=0
EPOCHS=50
BATCH_SIZE=8
LR=5e-5
LORA_RANK=16
LORA_ALPHA=32

# 創建 experiments 目錄
mkdir -p experiments

# 啟動訓練
CUDA_VISIBLE_DEVICES=$GPU_ID nohup python train.py \
  --exp_name $EXP_NAME \
  --num_epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --lora_rank $LORA_RANK \
  --lora_alpha $LORA_ALPHA \
  --num_workers 4 \
  --seed 42 \
  > experiments/${EXP_NAME}.log 2>&1 &

# 保存 PID
echo $! > experiments/${EXP_NAME}.pid

echo "✓ 訓練已啟動"
echo "  Experiment: $EXP_NAME"
echo "  GPU: $GPU_ID"
echo "  PID: $(cat experiments/${EXP_NAME}.pid)"
echo "  Log: experiments/${EXP_NAME}.log"
echo ""
echo "監控訓練進度:"
echo "  tail -f experiments/${EXP_NAME}.log"
echo "  tensorboard --logdir experiments/"
```

---

## 🔧 故障排除

### 訓練中斷恢復

目前 train.py 不支援 resume，如需恢復需手動實現：

```python
# 在 Trainer.__init__() 中添加
if args.resume_from is not None:
    checkpoint = torch.load(args.resume_from)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    self.current_epoch = checkpoint['epoch'] + 1
    self.global_step = checkpoint['global_step']
```

### GPU OOM

如果遇到記憶體不足：
1. 減小 `--batch_size` (8 → 4 → 2)
2. 減小 `--lora_rank` (16 → 8)
3. 設定 `--use_amp False` (關閉混合精度)

### 訓練停滯

如果 loss 不下降：
1. 檢查 learning rate (可能太小或太大)
2. 檢查 loss weights 比例
3. 檢查梯度流動 (參考 DEBUG_GUIDE.md)

---

## 📊 實驗追蹤表

| Exp Name | LoRA Rank | LR | Feature Weight | Distance Weight | Val Loss | Best Epoch | Notes |
|----------|-----------|-----|----------------|-----------------|----------|------------|-------|
| baseline_r16_lr5e5 | 16 | 5e-5 | 1.0 | 0.1 | - | - | Baseline |
| lora_r8_lr5e5 | 8 | 5e-5 | 1.0 | 0.1 | - | - | Small LoRA |
| lora_r32_lr5e5 | 32 | 5e-5 | 1.0 | 0.1 | - | - | Large LoRA |
| baseline_r16_lr1e4 | 16 | 1e-4 | 1.0 | 0.1 | - | - | Higher LR |
| baseline_r16_lr1e5 | 16 | 1e-5 | 1.0 | 0.1 | - | - | Lower LR |
| feature_heavy_r16 | 16 | 5e-5 | 2.0 | 0.05 | - | - | Feature-focused |
| distance_heavy_r16 | 16 | 5e-5 | 0.5 | 0.5 | - | - | Distance-focused |

---

## 🎓 預期學習曲線

**正常訓練應該觀察到**:
1. **Epoch 0-5**: Loss 快速下降 (warmup 階段)
2. **Epoch 5-20**: Loss 穩定下降
3. **Epoch 20-40**: Loss 緩慢收斂
4. **Epoch 40-50**: Loss 趨於穩定

**異常情況**:
- Loss = NaN → Learning rate 太大
- Loss 不下降 → Learning rate 太小 or 梯度消失
- Validation loss 上升 → Overfitting

---

## ✅ 開始訓練 Checklist

- [x] 測試訓練驗證成功 (commit f2c8efc)
- [x] GPU 狀態正常 (3 GPUs available)
- [x] 數據載入測試通過 (7,776 + 1,440 samples)
- [ ] 創建 launch scripts
- [ ] 啟動 Tensorboard
- [ ] 啟動 Phase 1 Baseline 訓練
- [ ] 定期監控訓練進度 (每 2-3 小時)
- [ ] 記錄實驗結果到追蹤表

---

**準備就緒！可以開始完整訓練實驗！** 🚀

建議從 **Phase 1 Baseline** 開始，驗證系統穩定性後再進行其他實驗。
