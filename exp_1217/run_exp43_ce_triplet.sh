#!/bin/bash
# ============================================================
# Exp43: CE + Triplet (無 Feature) (T=0.5, CE=1.0)
# ============================================================
# 假設: 測試 Triplet 是否能替代 Feature 保持音質
# ============================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1217

echo "============================================================"
echo "Exp43: CE + Triplet (無 Feature)"
echo "============================================================"
echo "  - Loss: Feature=0.0, Triplet=0.5, CE=1.0"
echo "  - LoRA: all_18 layers, rank=128"
echo "============================================================"

python train.py \
    --exp_name exp43_ce_triplet \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 0.0 \
    --triplet_weight 0.5 \
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
    2>&1 | tee exp43.log

echo "============================================================"
echo "Exp43 completed at: $(date)"
echo "============================================================"
