#!/bin/bash
# ============================================================
# Exp40: Feature 主導 + 輕 CE (F=1.0, CE=0.5)
# ============================================================
# 假設: Feature Loss 保持音質，CE 輔助 token 預測
# ============================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

cd /home/sbplab/ruizi/WavTokenize-self-supervised/exp_1217

echo "============================================================"
echo "Exp40: Feature 主導 + 輕 CE"
echo "============================================================"
echo "  - Loss: Feature=1.0, Triplet=0.0, CE=0.5"
echo "  - LoRA: all_18 layers, rank=128"
echo "============================================================"

python train.py \
    --exp_name exp40_feature_ce \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 1.0 \
    --triplet_weight 0.0 \
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
    2>&1 | tee exp40.log

echo "============================================================"
echo "Exp40 completed at: $(date)"
echo "============================================================"
