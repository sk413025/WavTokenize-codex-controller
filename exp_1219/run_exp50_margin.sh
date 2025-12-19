#!/bin/bash
# Exp50: Triplet Margin 增大實驗
#
# 目的: 解決 55% code 有 NN=0 的問題，增強 token 區分度
# 分析: 有效 code 的 NN mean = 1.27，當前 margin=0.2 只佔 16%
# 假設: margin=0.5 (佔 39%) 能更好地區分 tokens
#
# 基於 Exp48 最佳配置:
# - feature_weight=1.0
# - triplet_weight=1.0
# 修改:
# - triplet_margin: 0.2 -> 0.5

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=1

python exp_1219/train.py \
    --exp_name exp50_margin \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 1.0 \
    --cosine_weight 0.0 \
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
    2>&1 | tee exp_1219/exp50.log
