#!/bin/bash
# Exp52: 高 LoRA 容量實驗
#
# 基於 Exp51_v2 配置，提高 LoRA 容量:
# - lora_rank: 128 -> 256
# - lora_alpha: 256 -> 512
# - batch_size: 16 -> 12 (因記憶體增加)
#
# 其餘配置與 Exp51_v2 相同:
# - cosine_weight: 0.1
# - triplet_margin: 0.5
#
# 目的: 測試更高容量是否能提升 token accuracy


source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python exp_1219/train.py \
    --exp_name exp52_high_rank \
    --lora_rank 256 \
    --lora_alpha 512 \
    --lora_dropout 0.2 \
    --lora_layers all_18 \
    --feature_weight 1.0 \
    --cosine_weight 0.1 \
    --triplet_weight 1.0 \
    --triplet_margin 0.5 \
    --ce_weight 0.0 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --batch_size 10 \
    --num_epochs 100 \
    --num_workers 4 \
    --seed 42 \
    --use_amp \
    --use_scheduler \
    --warmup_epochs 10 \
    --grad_clip 1.0 \
    2>&1 | tee exp_1219/exp52.log
