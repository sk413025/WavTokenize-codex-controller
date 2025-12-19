#!/bin/bash
# Exp49: Cosine Similarity Loss 基準實驗
#
# 目的: 解決 Exp48 發現的特徵方向不對齊問題 (cos_sim = 0.21)
# 假設: 加入 Cosine Loss 能有效提升 cos_sim 到 0.6+
#
# 基於 Exp48 最佳配置:
# - feature_weight=1.0
# - triplet_weight=1.0
# - triplet_margin=0.2
# 新增:
# - cosine_weight=0.5

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=1

python exp_1219/train.py \
    --exp_name exp49_cosine \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 1.0 \
    --cosine_weight 0.5 \
    --triplet_weight 1.0 \
    --triplet_margin 0.2 \
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
    2>&1 | tee exp_1219/exp49.log
