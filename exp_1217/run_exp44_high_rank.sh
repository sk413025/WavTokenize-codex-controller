#!/bin/bash
# ============================================================
# Exp44: 更高 LoRA Rank (18層, rank=256)
# ============================================================
# 假設: 更高容量可能突破性能上限
# 使用 Exp40-43 中最佳的 Loss 配置 (待定，先用平衡配置)
# ============================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1217

echo "============================================================"
echo "Exp44: 高 Rank (18層, rank=256)"
echo "============================================================"
echo "  - Loss: Feature=1.0, Triplet=0.5, CE=0.5 (平衡)"
echo "  - LoRA: all_18 layers, rank=256"
echo "============================================================"

python train.py \
    --exp_name exp44_high_rank \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 1.0 \
    --triplet_weight 0.5 \
    --triplet_margin 0.2 \
    --ce_weight 0.5 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --use_scheduler \
    --warmup_epochs 10 \
    --grad_clip 1.0 \
    --batch_size 12 \
    --num_epochs 100 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --check_interval 50 \
    2>&1 | tee exp44.log

echo "============================================================"
echo "Exp44 completed at: $(date)"
echo "============================================================"
