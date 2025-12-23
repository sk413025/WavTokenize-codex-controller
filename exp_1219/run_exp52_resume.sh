#!/bin/bash
# Exp52 Resume: 從 epoch 77 繼續訓練
#
# 原始 Exp52 在 epoch 77 時被中斷（GPU 被其他程序佔用）
# Best Val Acc at interruption: 0.871%
#
# 配置與原始 Exp52 相同:
# - lora_rank: 256
# - lora_alpha: 512
# - cosine_weight: 0.1
# - triplet_margin: 0.5

source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

cd /home/sbplab/ruizi/WavTokenize-self-supervised

export CUDA_VISIBLE_DEVICES=0
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
    --resume exp_1219/runs/exp52_high_rank/best_model.pt \
    --resume_epoch 77 \
    2>&1 | tee -a exp_1219/exp52_resume.log
