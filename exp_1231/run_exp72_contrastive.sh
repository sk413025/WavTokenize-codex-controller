#!/bin/bash
# Exp72: Contrastive Token Loss (InfoNCE)
#
# 核心改進：
# - 用對比學習的方式訓練
# - 讓 student feature 靠近正確的 code，遠離錯誤的
# - Hard Negative Mining 選擇最有挑戰的負樣本

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "Exp72: Contrastive Token Loss (InfoNCE)"
echo "=============================================="
echo "Feature Weight: 1.0"
echo "Triplet Weight: 1.0"
echo "Contrastive Weight: 0.5"
echo "Num Negatives: 16"
echo "Hard Negative Mining: True"
echo "=============================================="

python -u exp_1231/train_exp72_contrastive.py \
    --exp_name exp72_contrastive \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 1.0 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
    --contrastive_weight 0.5 \
    --contrastive_temperature 0.1 \
    --num_negatives 16 \
    --hard_negative_mining \
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
echo "Exp72 completed!"
echo "=============================================="
