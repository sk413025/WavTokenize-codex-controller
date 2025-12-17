#!/bin/bash
# ============================================================
# Exp47: 純 Triplet Loss (T=1.0)
# ============================================================
# 假設: Triplet Loss 獨自能提供有效的 codebook 對齊信號
# 對照: Exp35 (F=1.0, T=0.5) 發現 Triplet 佔 76-81% loss
# ============================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1217

echo "============================================================"
echo "Exp47: 純 Triplet Loss"
echo "============================================================"
echo "Config: Feature=0.0, Triplet=1.0, CE=0.0"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

python train.py \
    --exp_name exp47_triplet_only \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 0.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --ce_weight 0.0 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --batch_size 16 \
    --num_epochs 100 \
    --num_workers 4 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 5 \
    --grad_clip 1.0

echo "============================================================"
echo "Exp47 完成!"
echo "============================================================"
