#!/bin/bash
# Exp51: 組合改進實驗 (V2 修改版)
#
# 目的: 同時解決方向對齊和 token 區分度問題
#
# V1 結果分析 (cosine_weight=0.5):
# - cos_sim=0.498 (很高) 但 accuracy=0.82% (極低)
# - 結論: cosine_weight=0.5 過強，犧牲了 token accuracy
#
# V2 修改:
# - cosine_weight: 0.5 -> 0.1 (降低以平衡 cos_sim 和 accuracy)
# - triplet_margin: 0.5 (維持高 margin 增強區分度)
#
# 預期: 同時達到中等 cos_sim (~0.3) 和較高 accuracy (~5%+)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python exp_1219/train.py \
    --exp_name exp51_combined_v2 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 1.0 \
    --cosine_weight 0.1 \
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
