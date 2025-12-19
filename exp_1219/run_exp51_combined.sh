#!/bin/bash
# Exp51: 組合改進實驗
#
# 目的: 同時解決方向對齊和 token 區分度問題
# 組合:
# - cosine_weight=0.5 (解決方向問題)
# - triplet_margin=0.5 (增強區分度)
#
# 預期: 達到最佳 Token Accuracy (20%+)

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=1

python exp_1219/train.py \
    --exp_name exp51_combined \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 1.0 \
    --cosine_weight 0.5 \
    --triplet_weight 1.0 \
    --triplet_margin 0.5 \
    --ce_weight 0.0 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --batch_size 16 \
    --num_epochs 100 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 10 \
    --grad_clip 1.0 \
    2>&1 | tee exp_1219/exp51.log
