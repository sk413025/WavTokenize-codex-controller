#!/bin/bash
# Exp75: Progressive Loss Schedule
#
# 核心改進：
# - Phase 1 (1-100): 只訓練 Feature Loss (連續空間)
# - Phase 2 (101-200): 加入 Soft Token Loss
# - Phase 3 (201-300): Soft Token 為主

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "Exp75: Progressive Loss Schedule"
echo "=============================================="
echo "Phase 1 (1-100): Feature only"
echo "Phase 2 (101-200): Feature + Soft Token (gradual)"
echo "Phase 3 (201-300): Soft Token dominant"
echo "=============================================="

python -u exp_1231/train_exp75_progressive.py \
    --exp_name exp75_progressive \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --triplet_margin 0.2 \
    --soft_token_temperature 1.0 \
    --vq_commitment_weight 0.1 \
    --vq_distortion_weight 0.1 \
    --phase1_ratio 0.33 \
    --phase2_ratio 0.33 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --batch_size 8 \
    --num_epochs 300 \
    --num_workers 4 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 10 \
    --grad_clip 1.0 \
    --gradient_accumulation_steps 2 \
    --early_stopping_patience 100 \
    --seed 42

echo "=============================================="
echo "Exp75 completed!"
echo "=============================================="
