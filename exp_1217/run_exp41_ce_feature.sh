#!/bin/bash
# ============================================================
# Exp41: CE 主導 + 輕 Feature (F=0.5, CE=1.0)
# ============================================================
# 假設: CE 主導優化 token，Feature 保持音質
# ============================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1217

echo "============================================================"
echo "Exp41: CE 主導 + 輕 Feature"
echo "============================================================"
echo "  - Loss: Feature=0.5, Triplet=0.0, CE=1.0"
echo "  - LoRA: all_18 layers, rank=128"
echo "============================================================"

python train.py \
    --exp_name exp41_ce_feature \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 0.5 \
    --triplet_weight 0.0 \
    --triplet_margin 0.2 \
    --ce_weight 1.0 \
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
    2>&1 | tee exp41.log

echo "============================================================"
echo "Exp41 completed at: $(date)"
echo "============================================================"
