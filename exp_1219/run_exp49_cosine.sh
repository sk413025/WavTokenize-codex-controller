#!/bin/bash
# Exp49: Cosine Similarity Loss 基準實驗 (修改版)
#
# 目的: 解決 Exp48 發現的特徵方向不對齊問題 (cos_sim = 0.21)
#
# 基於 exp51 結果分析:
# - cosine_weight=0.5 太強，導致 cos_sim=0.498 但 accuracy=0.82% (極低)
# - 需要降低 cosine_weight 以平衡 cos_sim 和 token accuracy
#
# 修改:
# - cosine_weight: 0.5 -> 0.1 (降低以避免過度對齊)
# - triplet_margin: 0.2 (維持原配置)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python exp_1219/train.py \
    --exp_name exp49_cosine_v2 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 1.0 \
    --cosine_weight 0.1 \
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
