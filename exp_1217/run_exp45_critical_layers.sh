#!/bin/bash
# ============================================================
# Exp45: 關鍵 8 層 + 高 Rank (8層, rank=256)
# ============================================================
# 假設: 選擇性 LoRA 在關鍵層可能與全層低 rank 效果相當
# ============================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1217

echo "============================================================"
echo "Exp45: 關鍵 8 層 + 高 Rank"
echo "============================================================"
echo "  - Loss: Feature=1.0, Triplet=0.5, CE=0.5 (平衡)"
echo "  - LoRA: critical_8 layers, rank=256"
echo "============================================================"

python train.py \
    --exp_name exp45_critical_layers \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lora_layers critical_8 \
    --feature_weight 1.0 \
    --triplet_weight 0.5 \
    --triplet_margin 0.2 \
    --ce_weight 0.5 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --use_scheduler \
    --warmup_epochs 10 \
    --grad_clip 1.0 \
    --batch_size 16 \
    --num_epochs 100 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --check_interval 50 \
    2>&1 | tee exp45.log

echo "============================================================"
echo "Exp45 completed at: $(date)"
echo "============================================================"
