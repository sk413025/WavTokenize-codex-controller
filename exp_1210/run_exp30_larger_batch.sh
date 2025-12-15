#!/bin/bash
# ============================================================
# Exp30: Larger Batch + Smaller LoRA + Higher LR
# ============================================================
#
# 目的: 測試更大 batch size 對訓練的影響
#
# 改動:
# - LoRA rank: 128 → 64 (減少記憶體)
# - LoRA alpha: 256 → 128
# - Batch size: 16 → 20
# - Learning rate: 2e-5 → 2e-4 (10x)
#
# 理論:
# - 更大 batch → 更穩定梯度 → 可用更大 LR
# - 較小 LoRA → 減少 overfitting 風險
# - 較大 LR → 更快跳出局部最優
#
# 預期:
# - 訓練更穩定
# - 可能減少 train-val gap
# ============================================================

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1210

echo "============================================================"
echo "Exp30: Larger Batch + Smaller LoRA + Higher LR"
echo "============================================================"
echo "Key changes:"
echo "  - LoRA rank: 64 (was 128)"
echo "  - LoRA alpha: 128 (was 256)"
echo "  - Batch size: 20 (was 16)"
echo "  - Learning rate: 2e-4 (was 2e-5, 10x)"
echo "============================================================"

python train_lora_v3.py \
    --exp_name exp30_larger_batch \
    --lora_rank 64 \
    --lora_alpha 128 \
    --feature_weight 1.0 \
    --triplet_weight 0.5 \
    --triplet_margin 0.2 \
    --soft_ce_weight 0.0 \
    --dw_weight 0.0 \
    --lr 2e-4 \
    --batch_size 20 \
    --num_epochs 50 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --check_interval 100 \
    2>&1 | tee exp30.log

echo ""
echo "============================================================"
echo "Exp30 completed at: $(date)"
echo "============================================================"
