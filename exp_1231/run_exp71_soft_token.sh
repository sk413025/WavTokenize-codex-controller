#!/bin/bash
# Exp71: Soft Token Loss (KL Divergence)
#
# 核心改進：
# - 用 KL Divergence 監督整個 logits 分布
# - 不只看 argmax，讓梯度更平滑

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "Exp71: Soft Token Loss (KL Divergence)"
echo "=============================================="
echo "Feature Weight: 1.0"
echo "Triplet Weight: 1.0"
echo "Soft Token Weight: 1.0"
echo "Soft Token Temperature: 1.0"
echo "=============================================="

python -u exp_1231/train_exp71_soft_token.py \
    --exp_name exp71_soft_token \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --soft_token_weight 1.0 \
    --soft_token_temperature 1.0 \
    --vq_commitment_weight 0.1 \
    --vq_distortion_weight 0.1 \
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
echo "Exp71 completed!"
echo "=============================================="
