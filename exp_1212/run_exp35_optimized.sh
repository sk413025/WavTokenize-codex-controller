#!/bin/bash
# ============================================================
# Exp35: Aligned Dataset + Masked Loss + Optimized Training
# ============================================================
#
# 基於 Exp34 的分析，進行以下優化：
#
# Exp34 問題診斷：
# - Train m_acc 36 epochs 只從 12.7% → 13.7% (+1%)
# - Val m_acc 停滯在 ~13%
# - u_acc vs m_acc 差距大，模型在 padding 上過擬合
# - lr=2e-5 收斂太慢
#
# Exp35 優化策略：
# 1. 提高 Learning Rate: 2e-5 → 1e-4 (5x)
# 2. 使用 Cosine Scheduler 配合 Warmup
# 3. 增加 Triplet weight: 0.5 → 1.0 (加強 codebook 空間學習)
# 4. 加入 Gradient Clipping 穩定訓練
# 5. 1000 epochs 長期訓練
#
# 預期效果：
# - 初期學習更快 (higher lr)
# - 後期更穩定 (cosine decay)
# - 更好的 codebook 空間理解 (stronger triplet)
# ============================================================

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1212

echo "============================================================"
echo "Exp35: Optimized Aligned Training"
echo "============================================================"
echo "優化內容 (相比 Exp34)："
echo "  1. lr: 2e-5 → 1e-4 (5x faster)"
echo "  2. 加入 Cosine Scheduler + Warmup"
echo "  3. triplet_weight: 0.5 → 1.0"
echo "  4. 加入 Gradient Clipping (max_norm=1.0)"
echo "  5. epochs: 50 → 1000"
echo "============================================================"

python train_aligned.py \
    --exp_name exp35_optimized \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --weight_decay 0.05 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --ce_weight 0.0 \
    --lr 1e-4 \
    --use_scheduler \
    --warmup_epochs 10 \
    --grad_clip 1.0 \
    --batch_size 16 \
    --num_epochs 1000 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --check_interval 100 \
    2>&1 | tee exp35.log

echo ""
echo "============================================================"
echo "Exp35 completed at: $(date)"
echo "============================================================"
